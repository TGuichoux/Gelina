from typing import Any, Dict, Optional

import torch
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from common.utils.losses import GeodesicLoss


from .modeling_cfm import CFMModel

class TrainCFM(LightningModule):
    def __init__(
        self,
        n_feats_in: int, # conditioning dim
        n_feats_out: int, # projected conditioning dim
        out_size: int, # generated dim
        decoder: DictConfig,
        cfm: DictConfig,
        optimizer_cfg: DictConfig,
        scheduler_cfg: Optional[DictConfig] = None,
        joints: str = "all",
        loss_fn: str = "mse_loss",
        pose_dim: int = 337,
        sample_factor: float = 4.0, 
        use_interpolation: bool = True,
        cond_dropout: float = 0.1,
        seed_dropout: float = 0.0, 
        vel_weight: float = 0.0,
        rec_weight: float = 0.0, 
        rec_loss: Optional[Any] = None,  # Optional custom reconstruction loss function
        n_frames_seed: int = 0,  # Number of seed frames to use for conditioning
    ):
        super().__init__()
        self.save_hyperparameters()

        # Instantiate the core model
        self.model = CFMModel(
            n_feats_in=n_feats_in,
            n_feats_out=n_feats_out,
            decoder=decoder,
            cfm=cfm,
            out_size=out_size,
            loss_fn=loss_fn,
            sample_factor=sample_factor,
            use_interpolation=use_interpolation,
            cond_dropout=cond_dropout,
            seed_dropout=seed_dropout,
            rec_loss=rec_loss
        )

        # Joint masking logic
        self.pose_dim = pose_dim
        self.joint_mask = torch.ones((pose_dim,), requires_grad=False)
        self.vel_weight = vel_weight
        self.rec_weight = rec_weight
        self.sample_factor = sample_factor
        self.n_frames_seed = n_frames_seed
        if joints == 'body_only':
            print("Masking hands ...")
            # zero out left+right hands
            self.joint_mask[25*6:40*6] = 0
            self.joint_mask[40*6:55*6] = 0

        
    def on_fit_start(self) -> None:
        # Log all hyperparameters to W&B once at the start of training
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)

    def on_train_epoch_start(self) -> None:
        loader = self.trainer.train_dataloader
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(self.current_epoch)

    def mask_motion(self, motion):
        if self.pose_dim == 157: # if we remove the hands
            mask = torch.ones((motion.shape[2],), dtype=torch.bool)
            mask[25*6:40*6] = 0
            mask[40*6:55*6] = 0
            motion = motion[..., mask].reshape(motion.shape[0], motion.shape[1], -1)
        return motion

    def get_losses(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        motion = batch["motion"]
        latents = batch['motion_latents'] if 'motion_latents' in batch.keys() else batch['speech_latents']
        
        seed_poses = None
        if self.n_frames_seed > 0:
            seed_poses = motion[:,:self.n_frames_seed,:]
            motion = motion[:,self.n_frames_seed:,:]
            latents = latents[:,(self.n_frames_seed//self.sample_factor):,:]

        seed_poses = self.mask_motion(seed_poses) if seed_poses is not None else None

        motion = self.mask_motion(motion)
        

        return self.model(
            x=motion,
            y=latents,
            spks=batch.get("spks", None),
            joint_mask=self.joint_mask,
            cond=seed_poses
        )

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        losses = self.get_losses(batch)
        for name, loss in losses.items():
            self.log(f"sub_loss/train_{name}", loss, on_step=True, on_epoch=True, sync_dist=True)
        total = losses["loss"] + self.vel_weight * losses.get("loss_vel", 0.0) + self.rec_weight * losses.get("loss_rec", 0.0)

        self.log("train_loss", total, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        losses = self.get_losses(batch)
        for name, loss in losses.items():
            self.log(f"sub_loss/val_{name}", loss, on_step=False, on_epoch=True, sync_dist=True)
        total = losses["loss"] + self.vel_weight * losses.get("loss_vel", 0.0) + self.rec_weight * losses.get("loss_rec", 0.0)
        self.log("val_loss", total, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return total

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        norms = grad_norm(self, norm_type=2)
        self.log_dict({f"grad_norm/{k}": v for k, v in norms.items()}, logger=True)

    def configure_optimizers(self) -> Any:
        # Now called as usual
        optimizer = self.hparams.optimizer_cfg(params=self.parameters())
        if self.hparams.scheduler_cfg is not None:
            scheduler = instantiate(self.hparams.scheduler_cfg, optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler_cfg.interval,
                    "frequency": self.hparams.scheduler_cfg.frequency,
                    "name": getattr(self.hparams.scheduler_cfg, "name", "lr"),
                },
            }
        return {"optimizer": optimizer}