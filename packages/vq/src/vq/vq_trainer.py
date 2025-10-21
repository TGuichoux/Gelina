import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import smplx

import common.utils.rotation_conversions as rc
from common.utils.losses import GeodesicLoss
from .components.residual_vq import ResidualVQ
from .modeling_vq import RVQVAE_Smpl


class TrainVQ(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        milestones: list[int] = [100, 200, 300],
        gamma: float = 0.1,
        default_file_path: str | None = None,
        trans_loss: str = "l1",
        recons_loss: str = "l1_smooth",
        joint_weights: list[float] | None = None,
        body_only: bool = False,
        rec_loss_weight: float = 1.0,
        position_loss_weight: float = 1.0,
        trans_loss_weight: float = 1.0,
        velocity_loss_weight: float = 0.1,
        acceleration_loss_weight: float = 0.1,
        velocity_trans_loss_weight: float = 0.1,
        acceleration_trans_loss_weight: float = 0.1,
        commit: float = 0.1,
        foot_loss_weight: float = 0.1,
        contact_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = model
        self.smplx = smplx.create(
            "checkpoints/smplx_models/",
            model_type="smplx",
            gender="NEUTRAL_2020",
            num_betas=300,
            num_expression_coeffs=100,
            ext="npz",
            use_pca=False,
        ).cuda().eval()
        self.register_buffer(
            "betas",
            torch.from_numpy(np.load(default_file_path)["betas"].astype(np.float32)),
            persistent=False,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma
        self.l1_criterion = nn.SmoothL1Loss()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.geodesic_loss = GeodesicLoss()
        self.trans_loss = trans_loss
        self.recons_loss = recons_loss
        self.rec_loss_weight = rec_loss_weight
        self.position_loss_weight = position_loss_weight
        self.trans_loss_weight = trans_loss_weight
        self.velocity_trans_loss_weight = velocity_trans_loss_weight
        self.acceleration_trans_loss_weight = acceleration_trans_loss_weight
        self.commit = commit
        self.foot_loss_weight = foot_loss_weight
        self.contact_loss_weight = contact_loss_weight
        self.velocity_loss_weight = velocity_loss_weight
        self.acceleration_loss_weight = acceleration_loss_weight
        self.body_only = body_only
        self.joint_weights = joint_weights if joint_weights is not None else [1.0] * 55

    def joints_fk(self, body_rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        exps = torch.zeros(body_rot.size(0), 100, device=self.device)
        betas = self.betas.expand(body_rot.size(0), -1)
        out = self.smplx(
            betas=betas,
            transl=trans.reshape(-1, 3),
            expression=exps,
            jaw_pose=body_rot[..., 66:69],
            global_orient=body_rot[..., :3],
            body_pose=body_rot[..., 3 : 22 * 3],
            left_hand_pose=body_rot[..., 25 * 3 : 40 * 3],
            right_hand_pose=body_rot[..., 40 * 3 : 55 * 3],
            leye_pose=body_rot[..., 69:72],
            reye_pose=body_rot[..., 72:75],
            return_joints=True,
        )
        return out["joints"]

    def _trans_losses(self, pred: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, ...]:
        crit = self.l1_criterion if self.trans_loss == "l1" else self.mse_loss
        loss = crit(pred, tgt)
        vel = crit(pred[:, 1:] - pred[:, :-1], tgt[:, 1:] - tgt[:, :-1])
        acc = crit(pred[:, 2:] + pred[:, :-2] - 2 * pred[:, 1:-1], tgt[:, 2:] + tgt[:, :-2] - 2 * tgt[:, 1:-1])
        return loss, vel, acc

    def _emage_losses(
        self,
        pred: torch.Tensor,
        tgt: torch.Tensor,
        pred_trans: torch.Tensor,
        trans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bs, l, _ = pred.shape

        m_pred = rc.rotation_6d_to_matrix(pred[..., :330].reshape(bs, l, -1, 6))
        m_tgt = rc.rotation_6d_to_matrix(tgt[..., :330].reshape(bs, l, -1, 6))

        rec = self.geodesic_loss(m_pred, m_tgt, reduction="none").reshape(bs * l, 55)
        rec = (rec * torch.tensor(self.joint_weights, device=self.device)).mean()

        vel = self.l1_loss(m_pred[:, 1:] - m_pred[:, :-1], m_tgt[:, 1:] - m_tgt[:, :-1])
        acc = self.l1_loss(
            m_pred[:, 2:] + m_pred[:, :-2] - 2 * m_pred[:, 1:-1],
            m_tgt[:, 2:] + m_tgt[:, :-2] - 2 * m_tgt[:, 1:-1],
        )

        aa_tgt = rc.matrix_to_axis_angle(m_tgt).reshape(bs * l, 55 * 3)
        aa_pred = rc.matrix_to_axis_angle(m_pred).reshape(bs * l, 55 * 3)
        j_tgt = self.joints_fk(aa_tgt, trans - trans).reshape(bs, l, -1, 3)
        j_pred = self.joints_fk(aa_pred, pred_trans - pred_trans).reshape(bs, l, -1, 3)
        
        if self.body_only:
            j_tgt, j_pred = j_tgt[:, :, :25], j_pred[:, :, :25]
        pos = self.mse_loss(j_pred, j_tgt)
        contact_pred, contact_tgt = pred[..., -4:], tgt[..., -4:]
        contact = self.mse_loss(contact_pred, contact_tgt)
        foot_idx = [7, 8, 10, 11]
        static = contact_pred > 0.95
        feet = j_pred[:, :, foot_idx]
        foot_v = torch.zeros_like(feet)
        foot_v[:, :-1] = feet[:, 1:] - feet[:, :-1]
        foot_v[~static] = 0
        foot = self.mse_loss(foot_v, torch.zeros_like(foot_v))
        return rec, pos, contact, foot, (vel, acc)

    def _step(self, motions: torch.Tensor) -> dict[str, torch.Tensor]:
        bs = motions.size(0)
        pred, commit, ppl = self.model(motions)
        tr_pred, tr = pred[..., -7:-4], motions[..., -7:-4]
        loss_tr, vel_tr, acc_tr = self._trans_losses(tr_pred, tr)
        if self.recons_loss == "l1_smooth":
            loss_rec = self.l1_criterion(pred, motions)
            loss_pos = loss_contact = loss_foot = torch.tensor(0.0, device=self.device)
            vel = acc = torch.tensor(0.0, device=self.device)
        else:
            loss_rec, loss_pos, loss_contact, loss_foot, (vel, acc) = self._emage_losses(pred, motions, tr_pred, tr)
        total = (
            loss_rec * self.rec_loss_weight
            + loss_pos * self.position_loss_weight
            + loss_tr * self.trans_loss_weight
            + vel_tr * self.velocity_trans_loss_weight
            + acc_tr * self.acceleration_trans_loss_weight
            + commit * self.commit
            + loss_foot * self.foot_loss_weight
            + loss_contact * self.contact_loss_weight
            + vel * self.velocity_loss_weight
            + acc * self.acceleration_loss_weight
        )
        return dict(
            loss=total,
            commit=commit,
            perplexity=ppl,
            rec=loss_rec,
            pos=loss_pos,
            trans=loss_tr,
            v_trans=vel_tr,
            a_trans=acc_tr,
            foot=loss_foot,
            contact=loss_contact,
            vel=vel,
            acc=acc,
            batch=bs,
        )

    def _log(self, out: dict[str, torch.Tensor], prefix: str, on_step: bool):
        bs = out["batch"]
        self.log(f"{prefix}_loss", out["loss"], on_step=on_step, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"{prefix}_commit_loss", out["commit"], on_step=on_step, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"{prefix}_perplexity", out["perplexity"], on_step=on_step, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"{prefix}_velocity_loss_trans", out["v_trans"], on_step=on_step, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"{prefix}_acceleration_loss_trans", out["a_trans"], on_step=on_step, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"{prefix}_loss_trans", out["trans"], on_step=on_step, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"{prefix}_loss_foot", out["foot"], on_step=on_step, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"{prefix}_loss_contact", out["contact"], on_step=on_step, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"{prefix}_loss_rec", out["rec"], on_step=on_step, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"{prefix}_loss_pos", out["pos"], on_step=on_step, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"{prefix}_loss_acc", out["acc"], on_step=on_step, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"{prefix}_loss_vel", out["vel"], on_step=on_step, on_epoch=True, logger=True, batch_size=bs)

    def training_step(self, batch, _):
        out = self._step(batch["motion"].float().to(self.device))
        self._log(out, "train", True)
        return out["loss"]

    def validation_step(self, batch, _):
        out = self._step(batch["motion"].float().to(self.device))
        self._log(out, "val", False)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.99))
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.milestones, gamma=self.gamma)
        return {"optimizer": opt, "lr_scheduler": sch}
