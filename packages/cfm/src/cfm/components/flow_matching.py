from abc import ABC

import torch
import torch.nn.functional as F

from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger
import common.utils.rotation_conversions as rc
from einops import rearrange, reduce, repeat


log = get_pylogger(__name__)


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, n_features=None, solver='euler'):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Seed poses

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        if n_features is None:
            z = torch.randn_like(mu) * temperature
        else:
            z = torch.randn((mu.shape[0], n_features, mu.shape[2]), device=mu.device)

        
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        if solver == 'euler':
            return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)
        elif solver == 'rk4':
            return self.solve_rk4(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: seed poses.
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        sol = []

        if cond is not None:
            cond = rearrange(cond, "b n d -> b d n")
            mu = torch.cat([torch.zeros((mu.shape[0], mu.shape[1], cond.shape[2]), device=mu.device), mu], dim=2) # concat zeros for seed poses (batch_size, n_feats, seq_len+seed_poses_len)
            mask = torch.cat([torch.ones((mu.shape[0], 1, cond.shape[2]), device=mask.device), mask], dim=2) # concat zeros for seed poses (batch_size, 1, seq_len+seed_poses_len)
        
        for step in range(1, len(t_span)):
            if cond is not None: # at each step we concat the seed pose
                input = torch.cat([cond, x], dim=2) # concat seed poses condition (batch_size, n_feats, seq_len+seed_poses_len)
                dphi_dt = self.estimator(input, mask, mu, t, spks, None)
                dphi_dt = dphi_dt[..., cond.shape[2]:]
            else:
                dphi_dt = self.estimator(x, mask, mu, t, spks, None)


            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]
    
    def solve_rk4(self, x, t_span, mu, mask, spks, cond):
        t = t_span[0]
        sol = []
        for step in range(1, len(t_span)):
            dt = t_span[step] - t
            k1 = self.estimator(x,                    mask, mu, t,         spks, cond)
            k2 = self.estimator(x + 0.5*dt*k1,        mask, mu, t+0.5*dt,  spks, cond)
            k3 = self.estimator(x + 0.5*dt*k2,        mask, mu, t+0.5*dt,  spks, cond)
            k4 = self.estimator(x +     dt*k3,        mask, mu, t+dt,      spks, cond)
            x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            t = t + dt
            sol.append(x)
        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, reduction="sum", loss_fn=F.mse_loss, vel_loss=F.smooth_l1_loss, rec_loss=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, seq_len)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, seq_len)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, seq_len)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, n = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)

        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        if cond is not None:
            y = torch.cat([cond, y], dim=2) # concat seed poses condition (batch_size, n_feats, seq_len+seed_poses_len)
            mu = torch.cat([torch.zeros((b, mu.shape[1], cond.shape[2]), device=mu.device), mu], dim=2) # concat zeros for seed poses (batch_size, n_feats, seq_len+seed_poses_len)
            mask = torch.cat([torch.ones((b, 1, cond.shape[2]), device=mask.device), mask], dim=2) # concat ones for seed poses (batch_size, 1, seq_len+seed_poses_len)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks)
        if cond is not None:
            pred = pred[:, :, cond.shape[2]:]  # Remove seed poses from prediction
        loss_cfm = loss_fn(pred, u, reduction=reduction) 

        rec = (1 - self.sigma_min) * z + pred
        rec = rc._reproject_rot6d(rec[:, :x1.shape[1]-7])

        m_pred = rc.rotation_6d_to_matrix(rec.reshape(b, n, -1, 6)) # 7 is contacts and trans
        m_tgt = rc.rotation_6d_to_matrix(x1[:, :x1.shape[1]-7].reshape(b, n, -1, 6))

        if vel_loss is not None:
            loss_vel = vel_loss(m_pred[:, 1:] - m_pred[:, :-1], m_tgt[:, 1:] - m_tgt[:, :-1], reduction="mean")
        else:
            loss_vel = torch.tensor(0.0, device=mu.device)

        if rec_loss is not None:
            loss_rec = rec_loss(m_pred, m_tgt, reduction="mean")
        else:
            loss_rec = torch.tensor(0.0, device=mu.device)
        
        return loss_cfm, loss_vel, loss_rec, y


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
