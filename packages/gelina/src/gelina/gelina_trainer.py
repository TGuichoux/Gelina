from typing import List, Optional, Tuple
import fnmatch

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import get_cosine_schedule_with_warmup

from packages.common.src.common.blocks.attentive_rnn import AttentiveRNN
from packages.gelina.src.gelina.modeling_gelina import GelinaModel


class TrainGeLina(pl.LightningModule):
    def __init__(
        self,
        attentive_rnn: AttentiveRNN,
        logits_head_speech: nn.Module,
        d_model: int,
        quant_layer_speech: List[int],
        n_codebook_speech: int,
        n_special_token_in: int,
        n_special_token_out: int,
        n_txt_vocab: int,
        txt_encoder: Optional[nn.Module] = None,
        logits_head_motion: Optional[nn.Module] = None,
        quant_layer_motion: Optional[List[int]] = None,
        n_codebook_motion: Optional[int] = None,
        frame_rate_factor: int = 15,
        mode: str = "bimodal",
        mask_text_p: float = 0.0,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        n_warmup_steps: int = 500,
        n_training_steps: int = 300_000,
        load_weights: Optional[str] = None,
        loss_weight: Optional[dict] = None,
        freeze: Optional[List[str]] = None,
        pe_mode: str = "none",
        save_hparam: bool = True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        
        quant_layer_motion = quant_layer_motion or []

        self.model = GelinaModel(
            attentive_rnn=attentive_rnn,
            logits_head_speech=logits_head_speech,
            logits_head_motion=logits_head_motion,
            d_model=d_model,
            n_quant_speech=len(quant_layer_speech),
            n_quant_motion=len(quant_layer_motion),
            n_codebook_speech=n_codebook_speech,
            n_codebook_motion=n_codebook_motion,
            n_txt_vocab=n_txt_vocab,
            n_special_token_in=n_special_token_in,
            n_special_token_out=n_special_token_out,
            txt_encoder=txt_encoder,
            mask_text_p=mask_text_p,
            frame_rate_factor=frame_rate_factor,
            mode=mode,
            pe_mode=pe_mode,
        )

        if load_weights:
            sd = torch.load(load_weights, map_location="cpu", weights_only=False)["state_dict"]
            self.load_state_dict(
                {k: v for k, v in sd.items() if k in self.state_dict() and v.shape == self.state_dict()[k].shape},
                strict=False,
            )

        self.freeze_patterns = freeze or []
        self._freeze_modules()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.n_warmup_steps = n_warmup_steps
        self.n_training_steps = n_training_steps
        self.loss_weight = loss_weight or {}

    def _freeze_modules(self) -> None:
        if not self.freeze_patterns:
            return
        for name, module in self.named_modules():
            if any(fnmatch.fnmatch(name, pat) for pat in self.freeze_patterns):
                for p in module.parameters():
                    p.requires_grad = False
                module.eval()

    def on_train_epoch_start(self) -> None:
        sampler = getattr(self.trainer.train_dataloader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)

    def step(self, batch):
        text_token = batch["text_token"]
        speech_token = batch.get("audio_token")
        motion_token = batch.get("motion_token")

        logits_speech, logits_motion, losses, att = self.model(
            x=text_token,
            y_speech=speech_token,
            y_motion=motion_token,
            encoder_mask=batch["encoder_mask"],
            crossatt_mask=batch["crossatt_mask"],
            logits_mask_speech=batch.get("y_mask_speech"),
            logits_mask_motion=batch.get("y_mask_motion"),
        )

        if logits_speech is not None:
            logits_speech = logits_speech.detach()
        if logits_motion is not None:
            logits_motion = logits_motion.detach()

        return logits_speech, logits_motion, {k: v for k, v in losses.items()}, att

    def training_step(self, batch, idx):
        _, _, losses, _ = self.step(batch)

        present_losses = [losses[k] for k in ("loss_speech", "loss_motion") if k in losses]
        losses["loss_avg"] = torch.stack(present_losses).mean().detach()

        for k, v in losses.items():
            self.log(f"train_{k}", v, prog_bar=True, sync_dist=False, rank_zero_only=True, batch_size=len(batch["text_token"]))

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, rank_zero_only=True)

        total_loss = self.loss_weight.get("speech", 1.0) * losses["loss_speech"]

        if "loss_motion" in losses:
            total_loss += self.loss_weight.get("motion", 1.0) * losses["loss_motion"]

        return total_loss

    def validation_step(self, batch, idx):
        _, _, losses, _ = self.step(batch)

        present_losses = [losses[k] for k in ("loss_speech", "loss_motion") if k in losses]
        losses["loss_avg"] = torch.stack(present_losses).mean()

        for k, v in losses.items():
            self.log(f"val_{k}", v, prog_bar=True, sync_dist=False, rank_zero_only=True, batch_size=len(batch["text_token"]))

        total_loss = self.loss_weight.get("speech", 1.0) * losses["loss_speech"]

        if "loss_motion" in losses:
            total_loss += self.loss_weight.get("motion", 1.0) * losses["loss_motion"]

        return total_loss

    def on_after_backward(self) -> None:
        if self.global_step % 100 == 0 and self.global_rank == 0:
            parameters = [p.grad for p in self.model.parameters() if p.grad is not None]
            if parameters:
                grad_norm = torch.linalg.vector_norm(torch.stack([g.norm(2) for g in parameters]), 2)
                self.log("train_grad_norm", grad_norm, prog_bar=True, rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
