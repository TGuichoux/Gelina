import torch
import numpy as np

from common.utils.tools import resample_pose_seq
from common.utils.rotation_conversions import rot6d_to_aa


def raw2aa(pose, decoder_wrapper, cfg=None, device='cuda', hand_pose_file="checkpoints/hands_pose.npy"):
	"""
	Convert raw outputs from CFM to axis angle (b, 55*3).
	In CFM, the motion is represented as (n, 157) (seq length, 25x6 + 3trans + 4contacts).

	"""
	if pose.shape[-1] == 157:
		n_pose = torch.zeros((pose.shape[0], 337), device=device)
		n_pose[..., : 25 * 6] = pose[..., : 25 * 6]
		n_pose[..., 55 * 6 :] = pose[..., 25 * 6 :]
		pose = n_pose

	pose = resample_pose_seq(pose, 20, 30).to(device)

	out_rot6d = pose[:, :330]

	if cfg.trans == "zero":
		out_trans = torch.zeros_like(pose[:, 330:-4])
	elif cfg.trans == "estimate":
		out_trans = decoder_wrapper.get_global_trans(
			rot6d_to_aa(out_rot6d), pose[:, 330:-4], pose[:, -4:]
		)
	else:
		out_trans = pose[:, 330:-4]

	rec_aa = rot6d_to_aa(out_rot6d)
	default_hand_pose = torch.from_numpy(np.load(hand_pose_file)).to(device)
	rec_aa[..., 75:] = default_hand_pose

	rec_expressions = np.zeros((len(rec_aa), 100))

	return (
		rec_aa.cpu().numpy(),
		out_trans.numpy(force=True),
		rec_expressions,
		pose  # b, n, d
	)
	

def generate_motion_aa(
	latents: torch.Tensor,
	condition: torch.Tensor | None,
	model,
	device: torch.device,
	decoder_wrapper=None,
	cfg=None,
	temp: float = 1.0,
	solver: str = "euler",
	hand_pose_file="checkpoints/hands_pose.npy",
):
	"""
	Generate a sequence of SMPL axis-angle poses from motion latents.

	Parameters
	----------
	latents : torch.Tensor
		Tensor of shape (1, T_lat, D) containing motion latents.
	condition : torch.Tensor or None
		Optional seed poses of shape (1, N_seed, D_out) to prime the generation.
	model
		CFM model exposing `synthesise` / `synthesise_cfg`.
	device : torch.device
		Device on which tensors will be placed.
	decoder_wrapper : DecoderWrapper, optional
		Utility for recovering global translation when `cfg.trans == "estimate"`.
	cfg : omegaconf.DictConfig, optional
		Inference configuration with fields:
		`streaming, stride, cfg, n_steps, guidance_scale, n_frames_seed,
		 trans, n_frames_seed, ...`.
	temp : float, default 1.0
		Sampling temperature.
	solver : {"euler", "rk4"}, default "euler"
		ODE solver used by the flow-matching sampler.
	hand_pose_file : str, default "checkpoints/hands_pose.npy"
		Path to a numpy file containing a default hand pose to use for all frames.

	Returns
	-------
	rec_aa : np.ndarray
		Generated axis-angle poses, shape (T, 165).
	out_trans : np.ndarray
		Global translation, shape (T, 3).
	rec_expressions : np.ndarray
		Zero-filled expression array, shape (T, 100).
	generated : torch.Tensor
		Raw output of the CFM, shape (B, N, D)  # b, n, d
	"""
	with torch.no_grad():
		if cfg.streaming:
			all_gen = []
			for k in range(0, latents.shape[1], cfg.stride):
				if cfg.cfg:
					generated = model.synthesise_cfg(
						y=latents[:, k : k + cfg.stride],
						n_timesteps=cfg.n_steps,
						temperature=temp,
						guidance_scale=cfg.guidance_scale,
						solver=solver,
						cond=condition,
					)
				else:
					generated = model.synthesise(
						y=latents[:, k : k + cfg.stride],
						n_timesteps=cfg.n_steps,
						temperature=temp,
						cond=condition,
					)

				condition = (
					generated["decoder_outputs"]
					.transpose(1, 2)[:, -cfg.n_frames_seed :]  # seed poses
					.to(device)
				)
				all_gen.append(generated["decoder_outputs"].squeeze(0).transpose(0, 1))

			pose = torch.cat(all_gen, dim=0).squeeze(0)
		else:
			if cfg.cfg:
				generated = model.synthesise_cfg(
					y=latents,
					n_timesteps=cfg.n_steps,
					temperature=temp,
					guidance_scale=cfg.guidance_scale,
					solver=solver,
					cond=condition,
				)
			else:
				generated = model.synthesise(
					y=latents,
					n_timesteps=cfg.n_steps,
					temperature=temp,
					cond=condition,
				)
			pose = generated["decoder_outputs"].squeeze(0).transpose(0, 1)

	return raw2aa(pose, decoder_wrapper=decoder_wrapper, cfg=cfg, device=device, hand_pose_file=hand_pose_file)
