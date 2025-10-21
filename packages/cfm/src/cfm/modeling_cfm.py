import datetime as dt
import math
import random

import torch
from torch import nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import EinMix
from torch import Tensor

import matcha.utils.monotonic_align as monotonic_align  # pylint: disable=consider-using-from-import
from matcha import utils
from matcha.models.baselightningmodule import BaseLightningClass
from packages.cfm.src.cfm.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
import common.utils.rotation_conversions as rc
import torch.nn.functional as F
from common.utils.losses import GeodesicLoss


from matcha.utils.model import (
	denormalize,
	duration_loss,
	fix_len_compatibility,
	generate_path,
	sequence_mask,
)

log = utils.get_pylogger(__name__)


class CFMModel(nn.Module):  # 🍵
	def __init__(
		self,
		n_feats_in,
		n_feats_out,
		decoder,# config
		cfm, # config
		out_size,
		sample_factor,
		loss_fn='mse_loss',
		use_interpolation=True,
		cond_dropout=0.1,
		seed_dropout=0.0,
		rec_loss=None, 
	):
		super().__init__()


		self.n_feats_in = n_feats_in # conditioning dim
		self.n_feats_out = n_feats_out # projected conditioning dim
		
		self.out_size = out_size # generated pose dim

		self.sample_factor = sample_factor # framerate factor between conditioning (motion_latents=5 fps, speech_latents=75 fps) and output pose framerate (20 fps)
		self.use_interpolation = use_interpolation
		self.cond_dropout = cond_dropout
		self.seed_dropout = seed_dropout

		self.proj_layer = EinMix(
			"b n d -> b n q",
			weight_shape="q d",
			d=self.n_feats_in,
			q=self.n_feats_out,
		) # project condition to appropriate dim

		self.decoder = CFM(
			in_channels=self.n_feats_out+self.out_size, # [conditioning, noisy motion]
			out_channel=self.out_size, # [motion]
			cfm_params=cfm,
			decoder_params=decoder,
			n_spks=1,
			spk_emb_dim=None,
		)
   
		if loss_fn == 'mse_loss':
			self.loss_fn = lambda pred_motion, motions, reduction='none': F.mse_loss(pred_motion, motions, reduction=reduction)
		elif loss_fn == 'l1_loss':
			self.loss_fn = lambda pred_motion, motions, reduction='none': F.l1_loss(pred_motion, motions, reduction=reduction)
		self.rec_loss = rec_loss

	@torch.inference_mode()
	def synthesise(self, y, n_timesteps=100, temperature=1.0, spks=None, scale=1.0, solver='euler', cond=None):

		b,n,_ = y.shape
		n = int(n * self.sample_factor) # 4 times upsample
		d = 1024
		mu_y = self.proj_layer(y) # projection 

		if self.use_interpolation:
			if n%2==1: n-= 1 # make sure n is even
			mu_y = resample(mu_y, self.sample_factor)[:, :n]
		else:
			mu_y = repeat(mu_y, 'b n d -> b (repeat n) d', repeat=int(self.sample_factor))[:, :n]

		y_mask = torch.ones((b,1,n), device=mu_y.device)

		mu_y = rearrange(mu_y, "b n d -> b d n")

		# Generate sample tracing the probability flow
		decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, spks, n_features=self.out_size, solver=solver, cond=cond)
		decoder_outputs = decoder_outputs#[:, :, :]

		return {
			"decoder_outputs": decoder_outputs,
		}
	
	@torch.inference_mode()
	def synthesise_cfg(
		self,
		y: Tensor,
		n_timesteps: int = 100,
		temperature: float = 1.0,
		spks: Tensor | None = None,
		guidance_scale: float = 1.5,
		solver='euler',  # 'euler' or 'rk4'
		cond=None,  # Seed poses condition
   		) -> dict[str, Tensor | float]:

		b, n_c, _ = y.shape
		mu_y = self.proj_layer(y)
		if self.use_interpolation:
			n = int(n_c * self.sample_factor)
			if n % 2:
				n -= 1
			mu_y = resample(mu_y, self.sample_factor)[:, :n]
		else:
			mu_y = repeat(mu_y, "b n d -> b (r n) d", r=int(self.sample_factor))[:, : int(n_c * self.sample_factor)]

		mu_y_cond = rearrange(mu_y, "b n d -> b d n")
		mu_y_uncond = torch.zeros_like(mu_y_cond)
		y_mask = torch.ones((b, 1, mu_y_cond.shape[-1]), device=mu_y_cond.device)

		x_uncond = self.decoder(mu_y_uncond, y_mask, n_timesteps, temperature, None, n_features=self.out_size, solver=solver, cond=cond)
		x_cond = self.decoder(mu_y_cond, y_mask, n_timesteps, temperature, spks, n_features=self.out_size, solver=solver, cond=cond)

		decoder_outputs = x_uncond + guidance_scale * (x_cond - x_uncond)

		return {"decoder_outputs": decoder_outputs}

	def forward(
		self,
		x: Tensor,                      # [B, N, D_out]
		y: Tensor,                      # [B, N_c, D_cond_in]
		spks: Tensor | None = None,
		joint_mask: Tensor | float = 1.0,
		cond: Tensor | None = None,  # Seed poses condition
	) -> dict[str, Tensor]:
		b, n, _ = x.shape

		mu_y = self.proj_layer(y)
		if self.use_interpolation:
			mu_y = resample(mu_y, self.sample_factor)[:, :n]
		else:
			mu_y = repeat(mu_y, "b n d -> b (r n) d", r=int(self.sample_factor))[:, :n]

		if self.training and self.cond_dropout > 0.0:
			drop_mask = torch.rand(b, device=x.device) < self.cond_dropout
			mu_y[drop_mask] = 0.0 # drop mu for some examples in batch
			if spks is not None:
				spks = spks.clone()
				spks[drop_mask] = 0

		if self.training and self.seed_dropout > 0.0:
			if torch.rand(1, device=x.device) < self.seed_dropout: # drop the seed poses for whole batch
				cond = None

		mu_y = rearrange(mu_y, "b n d -> b d n")
		x = rearrange(x, "b n d -> b d n")
		if cond is not None:
			cond = rearrange(cond, "b n d -> b d n")
		y_mask = torch.ones((b, 1, n), device=x.device)

		diff_loss, loss_vel, loss_rec, _ = self.decoder.compute_loss(
			x1=x,
			mask=y_mask,
			mu=mu_y,
			cond=cond,
			reduction="none",
			loss_fn=self.loss_fn,
			rec_loss=self.rec_loss,
			vel_loss=F.smooth_l1_loss,
		)

		diff_loss = diff_loss.sum(dim=2)
		diff_loss = diff_loss * joint_mask.to(diff_loss.device).detach()
		return {"loss": diff_loss.mean(), "loss_vel": loss_vel, "loss_rec": loss_rec}
