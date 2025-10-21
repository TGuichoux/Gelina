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


# ------------------------------ Uility functions ---------------------------------

def load_text(text_file):
	with open(text_file, 'r', encoding='utf-8') as f:
		first_line = f.readline()
	return normalize(first_line.strip())

def tokenize_speech(tokenizer, wav, device='cuda'):
	wav = torch.from_numpy(wav).to(device)
	_, code = tokenizer.encode_infer(wav.unsqueeze(0), bandwidth_id=torch.tensor([0], device=wav.device))
	return code.cpu()

def forward_tokenizer(wav, tokenizer, device='cuda'):
	code = tokenize_speech(tokenizer, wav, device).to(device)
	res_wav = tokens2wav(code, tokenizer, device).cpu().numpy()
	return res_wav

def tokenize_motion(tokenizer, motion, device='cuda'):
	motion = torch.from_numpy(motion).to(device)
	codes = tokenizer.tokenize(motion).transpose(0,1) # q,n
	return codes

def compile_attentive_gla(att_rnn):
	opts = dict(backend="inductor", mode="reduce-overhead", fullgraph=False, dynamic=True)

	# Compile all GLA blocks (encoder+decoder)
	for blk in list(att_rnn.encoder) + list(att_rnn.decoder):
		gla = blk.tmix                      # GatedLinearAttention instance
		gla.forward = torch.compile(gla.forward, **opts)
		gla.enable_fused_qkv()

		gla_cmix = blk.cmix
		gla_cmix.forward = torch.compile(gla_cmix.forward, **opts)

		# (optional) also compile the MLP if it is non-trivial
		if hasattr(blk, "ff"):
			blk.ff.forward = torch.compile(blk.ff.forward, **opts)

	# Compile cross-attention (fast path must use fused SDPA and NOT return att maps)
	if getattr(att_rnn, "cross_att", None) is not None:
		att_rnn.cross_att.forward = torch.compile(att_rnn.cross_att.forward, **opts)
		print('Compiling Cross-att')
	else:
		print("No cross att to compile")
	return att_rnn

def decode_cuts(cfg, sample_name, res_dir, cuts_speech, betas, motion_latents=None, use_cfm=True, cfm=None, motion_dec=None, speech_dec=None, gt_wav=None, device='cuda'):
	for j in tqdm(range(cfg.batch_size)):
		latents = motion_latents[j].unsqueeze(0)
		speech_tok = cuts_speech[j]

		rec_aa, out_trans, rec_expressions, _ = generate_motion_aa(latents, condition=None, model=cfm, device=device, decoder_wrapper=motion_dec, cfg=cfg, temp=cfg.cfm_temp, solver=cfg.solver, hand_pose_file=cfg.hand_pose_file)
		save_smpl_file(res_dir, f"res_{sample_name}.npz", betas, rec_aa, rec_expressions, out_trans, 30)

		if gt_wav is None:
			wav = tokens2wav(speech_tok[0], speech_dec, device=device).cpu().numpy()
		else:
			wav = gt_wav
		sf.write(res_dir / f"res_{sample_name}.wav", wav.reshape((-1,)), samplerate=24000)

		


@hydra.main(version_base=None, config_path="../../configs/cfm", config_name="infer_segments")
def main(cfg: DictConfig):
	if cfg.seed is not None:
		pl.seed_everything(cfg.seed, workers=True)

	device = torch.device(cfg.device)

	# --------------------------- Prepare output folders --------------------------------
	# -----------------------------------------------------------------------------------
	outdir = Path(cfg.output_dir)
	outdir.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	res_dir = outdir / f"{cfg.mode}_{timestamp}"
	prompt_dir = outdir / "prompts"
	gt_dir = outdir / "gt_segments"
	res_dir.mkdir(parents=True, exist_ok=True)
	prompt_dir.mkdir(parents=True, exist_ok=True)
	gt_dir.mkdir(parents=True, exist_ok=True)
	print(f"Saving samples at: {res_dir}, {prompt_dir}, {gt_dir}")
	OmegaConf.save(cfg, os.path.join(res_dir, 'config.yaml'))


	# --------------------------- Model loading -----------------------------------------
	# -----------------------------------------------------------------------------------
	wavtok = WavTokenizer.from_pretrained0802(cfg.wavtok_cfg, cfg.wavtok_ckpt).to(device).eval()
	motion_dec = DecoderWrapper(cfg.motion_vq_ckpt, cfg.motion_vq_cfg, cfg.global_vq_cfg, device).to(device).eval()
	txt_tok = PreTrainedTokenizerFast(tokenizer_file="checkpoints/bpe256.json")
	gelina = TrainGeLina.load_from_checkpoint(cfg.ckpt_path).model.to(device).eval()
	cfm = TrainCFM.load_from_checkpoint(cfg.cfm_path).model.to(device).eval()

	gelina.attentive_rnn = compile_attentive_gla(gelina.attentive_rnn)

	# --------------------------- Inference loop ---------------------------------------
	# ----------------------------------------------------------------------------------


	n_skipped = 0
	for i, file in tqdm(enumerate(os.listdir(gt_dir)), total=len(os.listdir(gt_dir))):
		# --------------------------- Data loading -------------------------------------
		# ------------------------------------------------------------------------------

		if i == cfg.n_samples:
			break
		if file[-3:] != 'npz':
			continue

		file_id = file[:-4]
		prompt_id = file_id
		sample_name = file_id
		if os.path.isfile(os.path.join(res_dir, f"res_{sample_name}.npz")):
			continue

		gt_text = load_text(os.path.join(gt_dir,file_id+'.txt'))
		gt_wav,_ = librosa.load(os.path.join(gt_dir,file_id+'.wav'), sr=cfg.src_sr) 
		gt_motion = np.load(os.path.join(gt_dir,file_id+'.npz')) # smplx
		gt_rot6d = np.load(os.path.join(gt_dir,file_id+'.npy')) # rot6d
		prompt_audio,_ = librosa.load(os.path.join(prompt_dir,prompt_id+'.wav'), sr=cfg.src_sr)
		prompt_motion = np.load(os.path.join(prompt_dir,prompt_id+'.npy'))# rot6d
		prompt_text = load_text(os.path.join(prompt_dir,prompt_id+'.txt'))
		betas = gt_motion['betas']

		# --------------------------- Input processing --------------------------------
		# -----------------------------------------------------------------------------

		if cfg.src_sr != cfg.tar_sr:
			gt_wav = librosa.resample(gt_wav, orig_sr=cfg.src_sr, target_sr=cfg.tar_sr)
			prompt_audio = librosa.resample(prompt_audio, orig_sr=cfg.src_sr, target_sr=cfg.tar_sr)

		if cfg.mode == 'cloning':
			print(f"Using prompt: {prompt_id}")

		prompt_speech_tokens = tokenize_speech(wavtok, prompt_audio, device) 
		prompt_motion_tokens = tokenize_motion(motion_dec, prompt_motion, device)
		prompt_text, speech_prompt, _, motion_prompt = _prepare_prompt( # preprocess prompt (token shifting, text normalization ...)
			cfg, framerate_factor=15,
			speech_tokens=prompt_speech_tokens,
			motion_tokens=prompt_motion_tokens,
			text=prompt_text
		)
		input_speech_tokens = tokenize_speech(wavtok, gt_wav, device)
		input_speech_tokens = process_audio_batch([{'audio_token':input_speech_tokens}])['audio_token']
		motion_prompt = motion_prompt[:gelina.n_quant_motion]  # removing extra residuals if necessary. 

		if (cfg.mode == 'cloning' or cfg.mode == 'streaming' or cfg.mode == 'RVQVAE_decoder') and (speech_prompt.shape[-1] == 0 or motion_prompt.shape[-1] == 0):
			n_skipped +=1
			print("Skipped one example because prompt too short.")
			continue

		if gt_wav.shape[-1] < 48000:
			n_skipped +=1
			print("Skipped one example because gt too short (<2s)")
			continue

		print(f"Input text: \n{gt_text}")

		# --------------------------- Inference -----------------------------------
		# -------------------------------------------------------------------------


		if cfg.mode == 'vanilla':
			ids  = torch.tensor(txt_tok.encode(f"[BOS]{normalize(gt_text)}[EOS]"), device=device)

			outs = gelina.fast_generate_multimodal_batch(
				x=ids,
				batch_size=cfg.batch_size,
				prompt_speech_tokens=None,
				prompt_motion_tokens=None,
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
			_, _, _, _, _, cuts_speech, _, _, motion_latents, _ = outs

			decode_cuts(cfg=cfg, sample_name=sample_name, res_dir=res_dir, cuts_speech=cuts_speech, betas=betas, motion_latents=motion_latents, use_cfm=True, cfm=cfm, motion_dec=motion_dec, speech_dec=wavtok, device='cuda')


		if cfg.mode == 'cloning':
			text = prompt_text + "," + gt_text
			ids  = torch.tensor(txt_tok.encode(f"[BOS]{normalize(text)}[EOS]"), device=device)

			outs = gelina.generate_multimodal_batch(
				x=ids,
				batch_size=cfg.batch_size,
				prompt_speech_tokens=speech_prompt.to(device),
				prompt_motion_tokens=motion_prompt.to(device),
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
			_, _, _, _, _, cuts_speech, _, _, motion_latents, _ = outs
			if motion_latents is None:
				n_skipped +=1
				print("Skipped one example because generation failed.")
				continue

			decode_cuts(cfg=cfg, sample_name=sample_name, res_dir=res_dir, cuts_speech=cuts_speech, betas=betas, motion_latents=motion_latents, use_cfm=True, cfm=cfm, motion_dec=motion_dec, speech_dec=wavtok, device='cuda')

		if cfg.mode == 'speech2ges':
			text = prompt_text + "," + gt_text
			ids  = torch.tensor(txt_tok.encode(f"[BOS]{normalize(text)}[EOS]"), device=device)

			outs = gelina.generate_speech2ges(
				x=ids,
				batch_size=cfg.batch_size,
				prompt_speech_tokens=speech_prompt.to(device),
				prompt_motion_tokens=motion_prompt.to(device),
				input_speech_tokens=input_speech_tokens.to(device),
				device=device,
				max_seqlen=cfg.max_seqlen,
				k_motion=cfg.k_motion,
				first_greedy_quant_motion=cfg.first_greedy_quant_motion,
				temp_motion=cfg.temp_motion,
				init_state=None,
				mask_motion_tokens=cfg.mask_motion_tokens,
			)
			_, _, _, _, _, cuts_speech, _, _, motion_latents, _ = outs

			decode_cuts(cfg=cfg, sample_name=sample_name, res_dir=res_dir, cuts_speech=cuts_speech, betas=betas, motion_latents=motion_latents, use_cfm=True, cfm=cfm, motion_dec=motion_dec, speech_dec=wavtok, device='cuda')

		if cfg.mode == 'tokenizer':
			wav = forward_tokenizer(gt_wav, wavtok)
			sf.write(res_dir / f"res_{sample_name}.wav", wav.reshape((-1,)), samplerate=24000)

			tokens = motion_dec.tokenize(torch.from_numpy(gt_rot6d))
			motion_dec.decode_and_save(tokens, res_dir, f"res_{sample_name}", betas=betas, zero_trans=False, zero_hands=True, residuals=6)


	print(f"Done. | Number of skipped examples: {n_skipped}")

if __name__ == "__main__":
	main()
