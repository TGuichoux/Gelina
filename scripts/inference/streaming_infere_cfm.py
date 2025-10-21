import os, json, torch, soundfile as sf, hydra, pytorch_lightning as pl, numpy as np
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
import pytorch_lightning as pl


from packages.gelina.src.gelina.utils.generation_helpers import tokens2wav, process_audio_batch
from common.utils.tools import resample_pose_seq
from common.utils.rotation_conversions import rot6d_to_aa
from common.utils.utils import load_config, save_smpl_file


# Change when vq decoder is clean ------------------------------
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from vq.decoder_wrapper import DecoderWrapper
# --------------------------------------------------------------

from external.WavTokenizer.decoder.pretrained import WavTokenizer
from transformers import PreTrainedTokenizerFast
from packages.gelina.src.gelina.gelina_trainer import TrainGeLina
from packages.gelina.src.gelina.streaming_wrapper import GelinaStreaming
from packages.gelina.src.gelina.utils.generation_helpers import _prepare_prompt
from packages.common.src.common.data.data_utils import normalize
from packages.cfm.src.cfm.cfm_trainer import TrainCFM
from packages.cfm.src.cfm.utils.generation_helpers import generate_motion_aa, raw2aa
from packages.common.src.common.data.smpldataset import SmplMotionDataModule



@hydra.main(version_base=None, config_path="../../configs/cfm", config_name="infer_streaming")
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


	if cfg.use_dataset:
		ds = SmplMotionDataModule(
			root_dir='/data/guichoux/',
			dataset_names=['beat_latents-v3-32'],
			selected_columns=['motion', 'motion_latents'],
			hf_repos=['TeoGchx/beat_with_latents-v3'],
			rep='rot6d',
			n_frames=32,
			train_batch_size=4,
			downsample_factor=4,
			save_processed=True,
			reload_dataset=False,
			num_workers=0
		)
		ds.setup(stage=None)

		dataset = ds.dataset['val']
	
	
		for i, data in enumerate(dataset):
			latents = data['motion_latents'].to(device) # b, n, d
			betas = data['motion']['betas'].squeeze().to(device)

			gt_pose = data['motion']['motion'].unsqueeze(0).to(device)
			gt_pose = gt_pose.squeeze(0)
			if gt_pose.shape[-1] == 157:
				n_pose = torch.zeros((gt_pose.shape[0], 337), device=device)
				n_pose[..., :25*6] = gt_pose[..., :25*6]
				n_pose[..., 55*6:] = gt_pose[..., 25*6:]
				gt_pose = n_pose
			
			if gt_pose.shape[-1] != 157:
				seed_motion = trainer.mask_motion(gt_pose.unsqueeze(0))
			latents = latents[:,trainer.n_frames_seed//trainer.sample_factor:].to(device)

			seed_motion = seed_motion[:,:trainer.n_frames_seed].to(device)


			
			rec_aa, out_trans, rec_expressions, _ = generate_motion_aa(latents, condition=seed_motion, model=cfm, device=device, decoder_wrapper=motion_dec, cfg=cfg, temp=cfg.cfm_temp, solver=cfg.solver)
			

			save_smpl_file(outdir, f"res-{i}.npz", betas.cpu().numpy(), rec_aa, rec_expressions, out_trans, 30)


			gt_pose = resample_pose_seq(gt_pose, 20, 30).to(device)
			gt_out_rot6d = gt_pose[:, :330]
			gt_out_trans = gt_pose[:, 330:-4]

			gt_rec_aa = rot6d_to_aa(gt_out_rot6d)
			save_smpl_file(outdir, f"gt-{i}.npz", betas.cpu().numpy(), gt_rec_aa.cpu().numpy(), rec_expressions, gt_out_trans.cpu().numpy(), 30)

			if i == cfg.n_samples - 1:
				break

	else:

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

			try:
				betas = torch.load(cfg.prompt_path+'betas.pth').numpy(force=True).squeeze() # if saved
			except:
				pass # use default

			wav = tokens2wav(orig_audio_tokens, wavtok, device=device).cpu().numpy()
			sf.write(outdir / f"prompt.wav", wav.reshape((-1,)), samplerate=24000)
		
			prompt_motion = prompt_motion[:gelina.n_quant_motion] # removing extra residuals if necessary
			
			motion_dec.decode_and_save(
					prompt_motion.transpose(1,2) - 3, outdir, f"prompt",
					betas=betas, zero_trans=None, zero_hands=cfg.zero_hands, residuals=cfg.residuals
				)
			
			prompt_speech = prompt_speech.to(device)
			prompt_motion = prompt_motion.to(device)
			text = prompt_text[:-1] + "," + text

		if cfg.seed_pose != "use_prompt":
			seed_poses = torch.load(cfg.seed_pose)
		else:
			seed_poses = torch.load(cfg.prompt_path+'motion_rot6d.pth')

		rec_aa, out_trans, rec_expressions, _ = raw2aa(seed_poses.detach(), motion_dec, cfg=cfg, device=device)
		save_smpl_file(outdir, f"seed.npz", betas, rec_aa, rec_expressions, out_trans, 30)

		seed_poses = trainer.mask_motion(seed_poses.unsqueeze(0))
		seed_poses = seed_poses[:,-cfg.n_frames_seed:].repeat(cfg.batch_size, 1, 1) # b, n, d

		gelina_stream = GelinaStreaming(cfg, gelina, cfm, wavtok, seed_poses=seed_poses).to(device)

		ids  = torch.tensor(txt_tok.encode(f"[BOS]{text}[EOS]"), device=device)

		# ---------- generate ----------
		outs = gelina_stream.generate_multimodal_batch(
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



		cuts_speech, cuts_motion = outs
		
		for i in tqdm(range(cfg.batch_size)):
			sf.write(outdir / f"res_{i:04d}.wav", cuts_speech[i].numpy().reshape((-1,)), samplerate=cfg.speech_rate)
			pose = cuts_motion[i].squeeze(0)

			rec_aa, out_trans, rec_expressions, _ = raw2aa(pose, motion_dec, cfg=cfg, device=device)
			save_smpl_file(outdir, f"res-generated-{i}.npz", betas, rec_aa, rec_expressions, out_trans, 30)
			





if __name__ == "__main__":
	main()
