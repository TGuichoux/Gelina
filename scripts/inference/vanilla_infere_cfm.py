import os, sys, json, torch, numpy as np, soundfile as sf, hydra, pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import librosa
import time

from transformers import PreTrainedTokenizerFast

from packages.gelina.src.gelina.gelina_trainer import TrainGeLina
from packages.gelina.src.gelina.utils.generation_helpers import (
	tokens2wav, _prepare_prompt, _extract_prompt, words_between, process_audio_batch
)
from packages.cfm.src.cfm.cfm_trainer import TrainCFM
from packages.cfm.src.cfm.utils.generation_helpers import generate_motion_aa
from packages.common.src.common.data.groupdataset import GroupDataModule
from packages.common.src.common.data.data_utils import normalize
from common.utils.utils import save_smpl_file

from vq.decoder_wrapper import DecoderWrapper

from external.WavTokenizer.decoder.pretrained import WavTokenizer

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)




@hydra.main(version_base=None, config_path="../../configs/cfm", config_name="infer")
def main(cfg: DictConfig):
	if cfg.seed is not None:
		pl.seed_everything(cfg.seed, workers=True)
	device = torch.device(cfg.device)
	outdir = Path(cfg.output_dir)
	outdir.mkdir(parents=True, exist_ok=True)

	wavtok = WavTokenizer.from_pretrained0802(cfg.wavtok_cfg, cfg.wavtok_ckpt).to(device).eval()
	motion_dec = DecoderWrapper(cfg.motion_vq_ckpt, cfg.motion_vq_cfg, cfg.global_vq_cfg, device).to(device).eval()
	txt_tok  = PreTrainedTokenizerFast(tokenizer_file="checkpoints/bpe256.json")
	gelina   = (TrainGeLina
				.load_from_checkpoint(cfg.ckpt_path)
				.model.to(device).eval())
	
	trainer = TrainCFM.load_from_checkpoint(cfg.cfm_path)
	cfm = trainer.model.to(device)
	cfm.eval()



	example_file = np.load(cfg.example_file, allow_pickle=True)
	betas = example_file['betas']

	text = normalize(cfg.text)
	print("Text input "+"-"*30)
	print(text)
	print("-"*40)

	# ---------- prepare prompt ----------
	prompt_speech, prompt_motion = None, None
	if cfg.prompt_path:
		prompt_text, prompt_speech, orig_audio_tokens, prompt_motion = _prepare_prompt(cfg)
		betas = torch.load(cfg.prompt_path+'betas.pth').numpy(force=True).squeeze() # if saved

		wav = tokens2wav(orig_audio_tokens, wavtok, device=device).cpu().numpy()
		sf.write(outdir / f"prompt.wav", wav.reshape((-1,)), samplerate=24000)
	
		prompt_motion = prompt_motion[:gelina.n_quant_motion] # removing exra reisudals if necessary
		
		motion_dec.decode_and_save(
				prompt_motion.transpose(1,2) - 3, outdir, f"prompt",
				betas=betas, zero_trans=cfg.zero_trans, zero_hands=cfg.zero_hands, residuals=cfg.residuals
			)
		
		prompt_speech = prompt_speech.to(device)
		prompt_motion = prompt_motion.to(device)
		text = prompt_text[:-1] + "," + text
	
	ids  = torch.tensor(txt_tok.encode(f"[BOS]{text}[EOS]"), device=device)
	# ---------- generate ----------
	outs = gelina.generate_multimodal_batch(
		x=ids,
		batch_size=cfg.batch_size,
		prompt_speech_tokens=prompt_speech,
		prompt_motion_tokens=prompt_motion,
		device=device,
		max_seqlen=cfg.max_seqlen,
		k_speech=cfg.k_speech,
		k_motion=cfg.k_motion,
		first_greedy_quant_speech=cfg.first_greedy_quant_speech,
		first_greedy_quant_motion=cfg.first_greedy_quant_motion,
		temp_speech=cfg.temp_speech,
		temp_motion=cfg.temp_motion,
		init_state=None,
		mask_motion_tokens=cfg.mask_motion_tokens,
	)
	_, _, _, _, _, cuts_speech, cuts_motion, _, motion_latents, _ = outs
	
	for i in tqdm(range(cfg.batch_size)):
		latents = motion_latents[i].unsqueeze(0)
		
		speech_tok = cuts_speech[i]
		rec_aa, out_trans, rec_expressions, _ = generate_motion_aa(latents, condition=None, model=cfm, device=device, decoder_wrapper=motion_dec, cfg=cfg, temp=cfg.cfm_temp, solver=cfg.solver)
		save_smpl_file(outdir, f"res-generated-{i}.npz", betas, rec_aa, rec_expressions, out_trans, 30)
		
		wav = tokens2wav(speech_tok[0], wavtok, device=device).cpu().numpy()
		sf.write(outdir / f"res_{i:04d}.wav", wav.reshape((-1,)), samplerate=24000)






if __name__ == "__main__":
	main()
