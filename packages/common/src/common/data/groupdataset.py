'''
	Main datamodule class.
'''

import torch
from torch.utils.data import (
		BatchSampler,
		DataLoader,
		Dataset,
		Sampler,
		SubsetRandomSampler,
		)

import os
import numpy as np
import pyarrow as pa

from typing import Optional, Dict, Callable, List, Tuple, Union
from random import choices

import pytorch_lightning as ptl
from datasets import load_dataset, concatenate_datasets, load_from_disk, DatasetDict, DownloadMode,  Features, Sequence, Value

from common.data.dataset_builder import DatasetBuilder
from common.data.collate_builder import build
from common.data.dataset_helpers import merge_dataset_dicts, report_sizes
from transformers import PreTrainedTokenizerFast





class BucketSampler(Sampler[List[int]]):
	def __init__(
			self,
			buckets: List[List[int]],
			batch_sizes: List[int] | int,
			bucket_sampling_weights: List[Tuple[float]] = None,
			drop_last: bool = True,
			distributed: bool = True,  # TODO - implement not distributed as well
			sample_bucket: Optional[int] = None,
			seed: int = 123,
			epoch_seed: bool = True,
			):
		if type(batch_sizes) is int:
			batch_sizes = [batch_sizes] * len(buckets)
		else:
			assert len(buckets) == len(batch_sizes)

		if bucket_sampling_weights is not None:
			assert len(bucket_sampling_weights) == len(batch_sizes)
		self.bucket_sampling_weights = bucket_sampling_weights
		self.num_replicas = torch.distributed.get_world_size()
		self.rank = torch.distributed.get_rank()
		self.buckets = [b[self.rank:len(b)-len(b)%self.num_replicas:self.num_replicas] for b in buckets]
		self.num_samples = [len(b) // self.num_replicas for b in buckets]
		self.batch_sizes = batch_sizes
		self.total_sizes = [ns // bs for ns, bs in zip(self.num_samples, self.batch_sizes)]
		self.drop_last = drop_last
		self.seed = seed
		self.epoch = 0
		self.sample_bucket = sample_bucket
		self.epoch_seed = epoch_seed
		self.batch_size = batch_sizes[0]

	def set_epoch(self, epoch: int):
		self.epoch = epoch

	def __len__(self):
		return sum(self.total_sizes)

	def __iter__(self):
		generator = torch.Generator()
		generator.manual_seed(self.seed + self.epoch * self.epoch_seed + self.rank)
		pool = [
				BatchSampler(SubsetRandomSampler(b, generator=generator), bs, drop_last=self.drop_last)
				for b, bs in zip(self.buckets, self.batch_sizes)
				]
		pool = [iter(b) for b in pool]
		weights = (
				[w for w in self.bucket_sampling_weights]
				if self.bucket_sampling_weights is not None
				else None
				)
		while pool:  # sample until all buckets are done
			idx, bucket = choices(list(enumerate(pool)), weights=weights)[0]
			try:
				batch = next(bucket)
				yield batch
			except StopIteration:
				pool.pop(idx)  # if bucket is done, throw it
				if weights is not None:
					weights.pop(idx)


class GroupDataModule(ptl.LightningDataModule):
	"""
	Datamodule wiring Hugging-Face datasets, bucket sampler and a pluggable
	collate recipe.

	Parameters
	----------
	save_dir : str | Path
		Root directory where processed datasets are cached.
	dataset_names : list[str]
		Logical names (also used as recipe keys) for each dataset.
	hf_repos : list[str]
		Corresponding Hugging-Face repo IDs or local cache paths.
	quant_layer_speech : list[int]
		Speech RVQ layers to keep after pattern delay.
	quant_layer_motion : list[int]
		Motion RVQ layers to keep.
	token_by_batch : int
		Target *total* tokens per batch for dynamic bucket sizing.
	selected_columns : list[str]
		HF columns that remain after `.select_columns`.
	speech_rate_hz : int, default 75
		Speech codec frame-rate.
	motion_rate_hz : int, default 5
		Motion token frame-rate.
	num_workers : int, default 8
		Torch DataLoader workers.
	seed : int, default 123
		Global RNG seed (splits, bucket sampler).
	n_buckets : int, default 1
		Number of length buckets for bucket sampler.
	val_batch_size : int, default 8
		Fixed validation batch size.
	tokenizer_file : str | Path | None
		Path to sentencepiece / BPE tokenizer.
	rvq_pattern : str, default "delay_rvq"
		Name of the pattern fn resolved inside `data_utils`.
	collate_fn : str, default "speech"
		Registry key of the collate recipe to build.
	motion_vocab_size : int, default 512
		Motion vocab size (needed by *speech_pt* collate).
	speech_vocab_size : int, default 4096
		Speech vocab size (unused yet, kept for completeness).
	speaker_ids : list[int] | None
		If provided, filter dataset to these speakers. (BEAT only)
	save_to_disk : bool, default True
		Cache processed datasets under `save_dir`.
	"""

	def __init__(
		self,
		save_dir: str,
		dataset_names: List[str],
		hf_repos: List[str],
		mixing='concat',
		stopping_strategy='first_exhausted',
		quant_layer_speech: List[int] = [0],
		quant_layer_motion: List[int] = [],
		token_by_batch: int = 15000,
		selected_columns: List[str] = ['audio_token', 'text', 'audio_duration'],
		speech_rate_hz: int = 75,
		motion_rate_hz: int = 0,
		num_workers: int = 8,
		seed: int = 123,
		n_buckets: int = 10,
		val_batch_size: int = 8,
		train_batch_size: int = 8,
		tokenizer_file: str = None,
		rvq_pattern: str = "delay_rvq",
		collate_fn: str = "speech",
		motion_vocab_size: int = 512,
		speech_vocab_size: int = 4096,
		speaker_ids: List[int] | None = None,
		save_to_disk: bool = True,
		max_dur: float = -1,
		min_dur: float = -1,
		tag_source: bool = True,
		val_size: float = 0.01,
	):
		super().__init__()

		self.save_dir = save_dir
		self.dataset_names = dataset_names
		self.hf_repos = hf_repos
		self.mixing = mixing
		self.stopping_strategy = stopping_strategy
		self.max_dur = max_dur
		self.min_dur = min_dur

		self.collate_fn = collate_fn
		self.speaker_ids = speaker_ids
		self.selected_columns = selected_columns

		self.speech_rate_hz = speech_rate_hz
		self.motion_rate_hz = motion_rate_hz

		self.quant_layer_speech = quant_layer_speech
		self.quant_layer_motion = quant_layer_motion

		self.motion_vocab_size = motion_vocab_size
		self.speech_vocab_size = speech_vocab_size

		self.tokenizer_file = tokenizer_file

		self.pattern = rvq_pattern
		self.collate_fn = collate_fn
		if self.collate_fn == "speech":
			assert self.motion_rate_hz == 0, "In speech only, motion_rate_hz must be zero."

		# bucket sampler params:      
		self.token_by_batch = token_by_batch
		self.n_buckets = n_buckets

		# Others:
		self.num_workers = num_workers
		self.seed = seed
		self.val_batch_size = val_batch_size
		self.save_to_disk = save_to_disk
		self.tag_source = tag_source
		self.val_size = val_size
		self.train_batch_size = train_batch_size



	def setup(self, stage: str | None = None):
		# Load text tokenizer
		# Build dataset(s) from HF/disk
		dsets = []
		for name, repo in zip(self.dataset_names, self.hf_repos):
			ds = DatasetBuilder(
				dataset_name=name,
				hf_repo=repo,
				save_to_disk=self.save_to_disk,
				save_dir=self.save_dir,
				val_size=self.val_size,
				num_proc=self.num_workers,
			).get_dataset()
			print(ds)

			if self.max_dur != -1:
				def filter_batch(batch, max_dur,min_dur):
					return [min_dur<b<max_dur for b in batch['audio_duration']]
				ds = ds.filter(lambda x: filter_batch(x,self.max_dur,self.min_dur), batched=True, num_proc=max(1, min(2, self.num_workers - 1)))

			for split, d in ds.items():
				if 'motion_token' in self.selected_columns and 'motion_token' not in d.column_names:
					def add_motion(batch):
						b = len(next(iter(batch.values())))
						return {"motion_token": [[[-1]] for _ in range(b)]}

					d = d.map(
						add_motion,
						batched=True,
						num_proc=max(1, min(2, self.num_workers - 1)),
						desc="add dummy motion_token", # for merging
					).cast_column("motion_token",Sequence(Sequence(Value("int64"))))

				ds[split] = d
			print(ds)
			ds = ds.select_columns(self.selected_columns)
			dsets.append(ds)
				
		

		self.dataset = merge_dataset_dicts(
			dsets,
			names=self.dataset_names,
			mixing=self.mixing,
			stopping_strategy=self.stopping_strategy,
			seed=self.seed,
			num_workers=max(1, self.num_workers - 1),
			tag_source=self.tag_source,
		)

		# if self.speaker_ids is not None:
		# 	self.dataset = self.dataset.filter(lambda id: id in self.speaker_ids, input_columns="speaker_id")

		# Build collate function
		self.collate_fn = build(
			name=self.collate_fn,
			speech_rate_hz=self.speech_rate_hz,
			motion_rate_hz=self.motion_rate_hz,
			pattern=self.pattern, # string name of the pattern function (e.g. "parallel_rvq")
			quant_layer_speech=self.quant_layer_speech,
			quant_layer_motion=self.quant_layer_motion,
			motion_vocab_size=self.motion_vocab_size,
			tokenizer_file=self.tokenizer_file
		)
		if self.tag_source:
			report_sizes(
				dsets,
				self.dataset,
				dur_key="audio_duration",
				names=self.dataset_names
			)

		# Get buckets for BucketSampler
		def get_buckets_by_quantile(duration, n_quantile):
			idxdur = list(enumerate(duration))
			idxdur.sort(key=lambda x: x[1])
			idx, dur = zip(*idxdur)
			bucket_size = len(idx) // n_quantile
			buckets = [list(x) for x in zip(*[iter(idx)]*bucket_size)]
			return buckets
		

		if self.token_by_batch > 0:
			train_buckets = get_buckets_by_quantile(self.dataset["train"]["audio_duration"], self.n_buckets)
			max_durations = [self.dataset["train"]["audio_duration"][x[-1]] for x in train_buckets]

			batch_sizes = [int(self.token_by_batch // ((self.speech_rate_hz) * ad)) for ad in max_durations]
			self.train_batch_sampler = BucketSampler(train_buckets, batch_sizes)
		else:
			self.train_batch_sampler = None
	


	def train_dataloader(self):
		if self.train_batch_sampler is not None:
			return DataLoader(
					self.dataset["train"].with_format("torch"),
					num_workers=self.num_workers,
					collate_fn=self.collate_fn,
					batch_sampler=self.train_batch_sampler,
					)
		else:
			return DataLoader(
					self.dataset["train"].with_format("torch"),
					num_workers=self.num_workers,
					collate_fn=self.collate_fn,
					batch_size=self.train_batch_size,
					)


	def val_dataloader(self):
		return DataLoader(
				self.dataset["val"].with_format("torch"),
				batch_size=self.val_batch_size,
				num_workers=0,
				collate_fn=self.collate_fn,
				)
	
	def test_dataloader(self, split='test', batch_size=8, shuffle=False):
		return DataLoader(
				self.dataset[split].with_format("torch"),
				batch_size=batch_size,
				num_workers=0,
				collate_fn=self.collate_fn,
				shuffle=shuffle,
				)

if __name__ == "__main__":
	import torch.distributed as dist
	import os

	if dist.is_available() and not dist.is_initialized():
		dist.init_process_group("gloo", rank=0, world_size=1) 

	# dm = GroupDataModule(
	# 	save_dir="/data/guichoux",
	# 	dataset_names=["beat_motok_smpl_body_short"],
	# 	hf_repos=["TeoGchx/beat_motok_smpl_body"],  # local cache path
	# 	quant_layer_speech=[0],
	# 	quant_layer_motion=[0],
	# 	token_by_batch=20000,
	# 	selected_columns=[
	# 		"text",
	# 		"audio_token",
	# 		"audio_duration",
	# 		"motion_token",
	# 		"motion_duration",
	# 	],
	# 	speech_rate_hz=75,
	# 	motion_rate_hz=5,
	# 	num_workers=0,
	# 	n_buckets=4,
	# 	val_batch_size=8,
	# 	tokenizer_file="checkpoints/bpe256.json",
	# 	rvq_pattern="parallel_rvq",
	# 	collate_fn="speech_motion",
	# 	speaker_ids=None,
	# 	save_to_disk=True,
	# )

	# dm.setup()

	# batch = next(iter(dm.train_dataloader()))
	# print("Batch keys :", list(batch))
	# print("text_token :", batch["text_token"].shape)
	# print("speech_token :", batch["audio_token"].shape)
	# print("motion_token :", batch["motion_token"].shape)
	# print("y_mask       :", batch["y_mask"].shape)
	# assert batch['motion_token'].shape[1]*15 == (batch['audio_token'].shape[1] - 2), f"15 x motion token length: {batch['motion_token'].shape[1]*15}, speech token length: {batch['speech_token'].shape[1]}"



	# dm = GroupDataModule(
	# 	save_dir="/data/guichoux",
	# 	dataset_names=["llm_nemo"],
	# 	hf_repos=["theodorr/llm_nemo_wavtokenizer"],  # local cache path
	# 	quant_layer_speech=[0],
	# 	quant_layer_motion=[],
	# 	token_by_batch=15000,
	# 	selected_columns=[
	# 		"text",
	# 		"audio_token",
	# 		"audio_duration",
	# 	],
	# 	speech_rate_hz=75,
	# 	motion_rate_hz=0,
	# 	num_workers=0,
	# 	n_buckets=4,
	# 	val_batch_size=8,
	# 	tokenizer_file="checkpoints/bpe256.json",
	# 	rvq_pattern="parallel_rvq",
	# 	collate_fn="speech",
	# 	speaker_ids=None,
	# 	save_to_disk=True,
	# )

	# dm.setup()

	# batch = next(iter(dm.train_dataloader()))
	# print("Batch keys :", list(batch))
	# print("text_token :", batch["text_token"].shape)
	# print("speech_token :", batch["audio_token"].shape)
	# print("y_mask       :", batch["y_mask"].shape)


	# dm = GroupDataModule(
	# 	save_dir="/data/guichoux",
	# 	dataset_names=["llm_nemo"],
	# 	hf_repos=["theodorr/llm_nemo_wavtokenizer"],  # local cache path
	# 	quant_layer_speech=[0],
	# 	quant_layer_motion=[0,1,2,3,4,5], # residuals for fake motion
	# 	token_by_batch=15000,
	# 	selected_columns=[
	# 		"text",
	# 		"audio_token",
	# 		"audio_duration",
	# 	],
	# 	speech_rate_hz=75,
	# 	motion_rate_hz=5,
	# 	num_workers=0,
	# 	n_buckets=4,
	# 	val_batch_size=8,
	# 	tokenizer_file="checkpoints/bpe256.json",
	# 	rvq_pattern="parallel_rvq",
	# 	collate_fn="speech_pt",
	# 	motion_vocab_size=512,
	# 	speaker_ids=None,
	# 	save_to_disk=True,
	# )

	# dm.setup()

	# batch = next(iter(dm.train_dataloader()))
	# print("Batch keys :", list(batch))
	# print("text_token :", batch["text_token"].shape)
	# print("speech_token :", batch["audio_token"].shape)
	# print("motion_token :", batch["motion_token"].shape)
	# print("y_mask       :", batch["y_mask"].shape)
	# assert batch['motion_token'].shape[1]*15 == (batch['audio_token'].shape[1] - 2), f"15 x motion token length: {batch['motion_token'].shape[1]*15}, speech token length: {batch['speech_token'].shape[1]}"


	# dm = GroupDataModule(
	# 	save_dir="/data/guichoux",
	# 	dataset_names=["llm_nemo","beat_motok_smpl_body_short"],
	# 	hf_repos=["theodorr/llm_nemo_wavtokenizer", "TeoGchx/beat_motok_smpl_body"],  # local cache path
	# 	mixing={"train": [0.5, 0.5], "val": [0.5, 0.5], "test": None},
	# 	quant_layer_speech=[0],
	# 	quant_layer_motion=[],
	# 	token_by_batch=15000,
	# 	selected_columns=[
	# 		"text",
	# 		"audio_token",
	# 		"audio_duration",
	# 	],
	# 	speech_rate_hz=75,
	# 	motion_rate_hz=0,
	# 	num_workers=0,
	# 	n_buckets=4,
	# 	val_batch_size=8,
	# 	tokenizer_file="checkpoints/bpe256.json",
	# 	rvq_pattern="parallel_rvq",
	# 	collate_fn="speech",
	# 	motion_vocab_size=512,
	# 	speaker_ids=None,
	# 	save_to_disk=True,
	# )

	# dm.setup()

	# batch = next(iter(dm.train_dataloader()))
	# print("Batch keys :", list(batch))
	# print("text_token :", batch["text_token"].shape)
	# print("speech_token :", batch["audio_token"].shape)
	# print("y_mask       :", batch["y_mask"].shape)


	dm = GroupDataModule(
		save_dir="/data/guichoux",
		dataset_names=["mls_nemo","ltts_hf", "ggsp"],
		hf_repos=["", "theodorr/ltts_wavtokenizer"],  # local cache path
		mixing={"train": [0.5, 0.5], "val": [0.5, 0.5], "test": None},
		quant_layer_speech=[0],
		quant_layer_motion=[],
		token_by_batch=15000,
		selected_columns=[
			"text",
			"audio_token",
			"audio_duration",
		],
		speech_rate_hz=75,
		motion_rate_hz=0,
		num_workers=8,
		n_buckets=4,
		val_batch_size=8,
		tokenizer_file="checkpoints/bpe256.json",
		rvq_pattern="parallel_rvq",
		collate_fn="speech",
		motion_vocab_size=512,
		speaker_ids=None,
		save_to_disk=False,
		val_size=0.005
	)

	dm.setup()

	batch = next(iter(dm.train_dataloader()))
	print("Batch keys :", list(batch))
	print("text_token :", batch["text_token"].shape)
	print("speech_token :", batch["audio_token"].shape)
	print("y_mask       :", batch["y_mask"].shape)
	print("Sources: ", batch["source"])


# python -m torch.distributed.launch packages/common/src/common/data/groupdataset.py