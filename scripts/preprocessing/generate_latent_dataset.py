"""
generate_latents.py

Efficiently extract speech/motion latents from BEAT with a pretrained GeLina,
build 🤗 Datasets for each split, and save to disk.

- Batched inference with bfloat16 autocast
- Process multiple splits in one run
- Stream directly into HF Dataset writers and save_to_disk(<out_dir>/<split>)
"""

import os
from typing import Dict, Any, Iterable

import hydra
from omegaconf import DictConfig
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset, DatasetDict

from packages.gelina.src.gelina.gelina_trainer import TrainGeLina
from common.data.groupdataset import GroupDataModule


def move_to(x, device: torch.device):
	if isinstance(x, torch.Tensor):
		return x.to(device, non_blocking=True)
	if isinstance(x, list):
		return [move_to(t, device) for t in x]
	if isinstance(x, dict):
		return {k: move_to(v, device) for k, v in x.items()}
	return x


def to_py(obj):
	if isinstance(obj, torch.Tensor):
		return obj.detach().cpu().tolist()
	if isinstance(obj, dict):
		return {k: to_py(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [to_py(v) for v in obj]
	return obj


@torch.no_grad()
def latent_record_generator(model, loader: DataLoader, device: torch.device) -> Iterable[Dict[str, Any]]:
	model.eval()
	for batch in tqdm(loader, desc="Generating latents"):
		if batch['text_token'].shape[-1] > 250:
			continue # means error in transcription
		batch = move_to(batch, device)
		with autocast(device_type=device,dtype=torch.bfloat16):
			mot_lat, speech_lat, *_ = model(
				x=batch["text_token"],
				y_speech=batch.get("audio_token"),
				y_motion=batch.get("motion_token"),
				encoder_mask=batch["encoder_mask"],
				crossatt_mask=batch["crossatt_mask"],
				logits_mask_speech=batch.get("y_mask_speech"),
				logits_mask_motion=batch.get("y_mask_motion"),
				return_latents=True,
			)
			

		B = speech_lat.shape[0]
		mot_lat = mot_lat if mot_lat is not None else None
		for b in range(B):
			rec = {
				"speech_latents": to_py(speech_lat[b]),
				"beat_motion": to_py(batch["beat_motion"][b]),
			}
			# tokens
			audio_tok = batch.get("audio_token")
			speech_tok = batch.get("speech_token")  # fallback if present
			tok = audio_tok if audio_tok is not None else speech_tok
			if tok is not None:
				rec["speech_tokens"] = to_py(tok[b])
			# motion
			if mot_lat is not None:
				rec["motion_latents"] = to_py(mot_lat[b])
			if "motion_token" in batch:
				rec["motion_tokens"] = to_py(batch["motion_token"][b])

			yield rec




def build_loader(dm: GroupDataModule, split: str, batch_size: int) -> DataLoader:
	# GroupDataModule exposes a split-aware test_dataloader(split, batch_size=...)
	return dm.test_dataloader(split, batch_size=batch_size)


@hydra.main(version_base=None, config_path="../../configs/preprocess", config_name="generate_latents")
def main(cfg: DictConfig):
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.benchmark = True
	device = "cuda"


	model = (
		TrainGeLina.load_from_checkpoint(cfg.model.checkpoint)
		.model.eval()
		.to(device)
	)
	print(f"Loaded GeLina from: {cfg.model.checkpoint}")

	dm = GroupDataModule(
		save_dir=cfg.root_dir,
		dataset_names=[cfg.dataset_name],
		hf_repos=[cfg.hf_repo],
		quant_layer_speech=[0],
		quant_layer_motion=[0],
		token_by_batch=0,
		selected_columns=[
			"text",
			"audio_token",
			"audio_duration",
			"motion_token",
			"motion_duration",
			"beat_motion",
		],
		speech_rate_hz=75,
		motion_rate_hz=5,
		num_workers=cfg.num_workers,
		n_buckets=10,
		val_batch_size=cfg.batch_size,
		tokenizer_file="checkpoints/bpe256.json",
		rvq_pattern="parallel_rvq",
		collate_fn="speech_motion",
		speaker_ids=None,
		save_to_disk=True,
	)
	dm.setup()

	ds_map = {}
	for split in cfg.splits:
		try:
			loader = build_loader(dm, split, batch_size=cfg.batch_size)
		except Exception as e:
			print(f"[Skip split={split}] {e}")
			continue

		print(f"Processing split: {split}")
		ds = Dataset.from_generator(
			latent_record_generator,
			gen_kwargs={"model": model, "loader": loader, "device": device},
			split=split,
			num_proc=1,
			writer_batch_size=cfg.writer_batch_size,
		)
		ds_map[split] = ds

	dsd = DatasetDict(ds_map)
	dsd.save_to_disk(os.path.join(cfg.root_dir, "beat_with_latents"))
	print(f"Saved DatasetDict with splits {list(dsd.keys())} to: {os.path.join(cfg.root_dir, 'beat_with_latents')}")


if __name__ == "__main__":
	main()
