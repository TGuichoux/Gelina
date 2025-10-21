'''
Utility functions for building datasets from Hugging Face.
Some datasets do not have a train/test/val split, so we need to create one.
Some datasets need extra filtering/preprocessing (rename columns, segment long sequences, etc.)

Available datasets:
"theodorr/gigaspeech_xl_wavtokenizer"
"theodorr/ltts_wavtokenizer"
"" (for mls nemo no path required)
"theodorr/tedlium_wavtokenizer"
'''

import os, sys, warnings
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
import datasets
from typing import Optional, Dict, Callable
from tqdm import tqdm
import multiprocessing as mp, itertools
import torch
import numpy as np

from common.data.data_utils import normalize, get_spk_id, segment_smplx, concat_segments

_PREPROCESS_REGISTRY: Dict[str, Callable[[DatasetDict], DatasetDict]] = {}


def register(name: str):
	'''
	Register function to handle the different preprocessing functions.
	'''
	def _wrap(func: Callable[[DatasetDict], DatasetDict]):
		if name in _PREPROCESS_REGISTRY:
			raise ValueError(f"Preprocessor '{name}' already registered.")
		_PREPROCESS_REGISTRY[name] = func
		return func
	return _wrap


class DatasetBuilder:
	'''
	Building class to create a DatasetDict object from HF repository.
	Also performs additional preprocessing depending on the dataset.
	Optionally saves processed dataset to disk.
	'''
	def __init__(
		self,
		hf_repo: str = "TeoGchx/BEAT_HML3D_whisper_watokenizer_motokenizer_smpl",
		dataset_name: str = "beat_motok_smpl",
		save_to_disk: bool = True,
		save_dir: str = "/data/guichoux/",
		split_seed: int = 123,
		val_size: float = 0.01,
		test_size: float = 0.03,
		num_proc: int = 4,
	):
		self.hf_repo = hf_repo
		self.dataset_name = dataset_name
		self.save_to_disk = save_to_disk
		self.save_dir = save_dir
		self.preprocess_key = dataset_name
		self.val_size = val_size
		self.test_size = test_size
		self.save_path = os.path.join(self.save_dir, self.dataset_name)
		self.num_proc = num_proc

		if os.path.exists(self.save_path+'_preprocessed'):
			print(f"Found preprocessed dataset at {self.save_path+'_preprocessed'}. Loading from disk ...")
			self.dataset = load_from_disk(self.save_path+'_preprocessed')
		elif not os.path.exists(self.save_path+'_preprocessed') and os.path.exists(self.save_path):
			print(f"Found non-processed data set {self.save_path}. Preprocessing ...")
			self.dataset = load_from_disk(self.save_path)
			self._preprocess_dataset(self.preprocess_key)
			if self.save_to_disk:
				print(f"Saving preprocessed dataset to disk ({self.save_path+'_preprocessed'})")
				os.makedirs(self.save_path+'_preprocessed', exist_ok=True)
				self.dataset.save_to_disk(self.save_path+'_preprocessed')
			print("Done.")
		else:
			print(f"Couldn't find cached dataset ({self.save_path}). Loading dataset from HF...")
			self.dataset = load_dataset(self.hf_repo) if self.hf_repo != "" else None
			print("Done")
			
			print("Begin preprocessing...")
			self._preprocess_dataset(self.preprocess_key)
			print("Done")
			if self.save_to_disk:
				print(f"Saving preprocessed dataset to disk ({self.save_path})")
				os.makedirs(self.save_path, exist_ok=True)
				self.dataset.save_to_disk(self.save_path)
				print("Done")

		if self.dataset is None:
			raise ValueError("Dataset could not be loaded.")

		self._ensure_splits(split_seed)

	def _ensure_splits(self, seed: int):
		ds = self.dataset
		has_val = "val" in ds
		has_test = "test" in ds
		if not has_val and not has_test:
			warnings.warn("Test and val splits missing; creating splits from train.")
			split = ds["train"].train_test_split(test_size=self.test_size, seed=seed)
			ds["train"], ds["test"] = split["train"], split["test"]
			split2 = ds["train"].train_test_split(test_size=self.val_size / (1 - self.test_size), seed=seed)
			ds["train"], ds["val"] = split2["train"], split2["test"]
		elif not has_val and has_test:
			warnings.warn("Val split missing; creating split from train.")
			split = ds["train"].train_test_split(test_size=self.val_size, seed=seed)
			ds["train"], ds["val"] = split["train"], split["test"]
		elif has_val and not has_test:
			warnings.warn("Test split missing; using val as test and creating new val from train.")
			ds["test"] = ds.pop("val")
			split = ds["train"].train_test_split(test_size=self.val_size, seed=seed)
			ds["train"], ds["val"] = split["train"], split["test"]
		self.dataset = DatasetDict(ds)

	def get_dataset(self, splits=['train', 'test', 'val']) -> DatasetDict:
		"""
		Get the dataset.
		"""
		return DatasetDict({k:self.dataset[k] for k in splits})

	def _preprocess_dataset(self, key: str) -> None:
		'''
		Run preprocessing function based on the privded key (dataset name).
		'''
		if key not in _PREPROCESS_REGISTRY:
			raise KeyError(f"Unknown preprocess '{key}'.")
		self.dataset = _PREPROCESS_REGISTRY[key](self.dataset, num_proc=self.num_proc)




# |--------------- Dataset preprocessing functions ---------------- |

def _segment_batch_V2(batch, min_dur, max_dur):
	import inspect, os
	texts, audio_tokens, audio_durations, motion_tokens, motion_durations, meta_datas, beat_motions = [], [], [], [], [], [], []
	for whisper_dict, audio_tok, motion_tok, beat_mot, meta in zip(
		batch["whisper_segments"], batch["audio_token"], batch["motion_token"], batch["beat_motion"], batch["meta_data"]
	):
		for i_seg,seg in enumerate(whisper_dict):
			st, et = seg["start"], seg["end"]
			txt = normalize(seg["text"])
			sa, ea = int(st * 75), int(et * 75)
			audio_seg = np.array(audio_tok)[:, :, sa:ea].tolist()
			sm, em = int(st * 5), int(et * 5)
			motion_seg = np.array(motion_tok)[sm:em].tolist()
			adur = (ea - sa) / 75.0
			mdur = (em - sm) / 5.0
			new_meta = {
				'file_id': meta['file_id'],
				'original_duration': meta['duration'],
				'duration': adur,
				'seg_id':i_seg,
				
			}
		
			if min_dur < adur < max_dur and len(motion_seg) > min_dur * 5:
				texts.append(txt)
				audio_tokens.append(audio_seg)
				audio_durations.append(adur)
				motion_tokens.append(motion_seg)
				motion_durations.append(mdur)
				meta_datas.append(new_meta)
				beat_motions.append(segment_smplx(beat_mot, st, et, fps=30))

	return {
		"text": texts,
		"audio_token": audio_tokens,
		"audio_duration": audio_durations,
		"motion_token": motion_tokens,
		"motion_duration": motion_durations,
		"meta_data": meta_datas,
		"beat_motion": beat_motions,
	}

@register("beat_tokenized_motion_hf")
def _process_beat_smpl_body_short(ds: DatasetDict, min_dur=2.0, max_dur=150.0, num_proc=4) -> DatasetDict:
	'''
	Process dataset dict (estimate time is <15min).
	This is the BEAT dataset with tokenized audio (wavtokenizer) and motion (RVQVAE focused on body, leaving the hands for futur work). 
	The sequences are further segmented (in BEAT the average sequence length is ~60s) using the segment from whisper.
	No segmentation is done on the additional split as it was already segmented when pushed to the HF hub (original sequences were much longer (e.g 7min)).
	This dataset is quite large, so ensure you have enough storage available. ()
	'''
	new_ds = DatasetDict()
	num_proc = max(1, num_proc)  # Ensure at least one process is used
	print(ds)
	for split_name, split_data in ds.items():
		processed = split_data.map(
			lambda x: _segment_batch_V2(x, min_dur, max_dur),
			batched=True,
			batch_size=1,
			num_proc=None,
			remove_columns=split_data.column_names,
			desc='Processing (shorten) split: ' + split_name,
			load_from_cache_file=False
		)
		new_ds[split_name] = processed

	if 'additional' in new_ds.keys():
		new_ds['train'] = concatenate_datasets([new_ds["train"], new_ds["additional"]])

	return new_ds



@register("llm_nemo")
def llm_nemo_wavtokenizer(ds: DatasetDict, min_dur=3.0, max_dur=150.1, num_proc=4):
	ds = ds["train"]
	ds = (
		ds
		.map(lambda x: {"text": normalize(concat_segments(x))}, input_columns="segments", num_proc=num_proc)
		.filter(lambda x: min_dur < x["audio_duration"] < max_dur)
	)
	return DatasetDict({"train": ds})


@register("ltts")
def ltts(ds: DatasetDict, min_dur=3.0, max_dur=150.1, num_proc=4):
	datasets_to_cat = []
	for split in ds.keys():
		dset = ds[split]
		datasets_to_cat.append(dset)
	ltts = concatenate_datasets(datasets_to_cat, axis=0)
	ds = [x.map(lambda x: {"audio_duration": len(x[0][0])/75}, input_columns="audio_token", num_proc=num_proc, desc="duration ltts").select_columns(["text_normalized", "audio_token", "audio_duration"]).rename_column("text_normalized", "text").filter(lambda dur: dur > min_dur and dur < max_dur, input_columns="audio_duration") for x in ltts]

	ds = (
		ds
		.map(lambda x: {"text": normalize(concat_segments(x))}, input_columns="segments", num_proc=num_proc, desc="duration ltts")
		.filter(lambda x: min_dur < x["audio_duration"] < max_dur)
	)
	return DatasetDict({"train": ds})


@register("mls_nemo")
def mls_nemo(ds: DatasetDict, min_dur=3.0, max_dur=150.1, num_proc=None): # when loading mls_nemo do not provide the HF dataset path
	mls10k_nemo = load_dataset(
			"theodorr/mls10k_nemo", split="train"
		).select_columns(["text", "audio_duration"])
	mls10k_wavtokenizer = load_dataset(
		"theodorr/mls10k_wavtokenizer", split="train"
	).select_columns(["audio_token"])
	mls = concatenate_datasets([mls10k_nemo, mls10k_wavtokenizer], axis=1)
	return DatasetDict({"train": mls})


@register("ggsp")
def ggsp(ds: DatasetDict, min_dur=3.0, max_dur=150.1, num_proc=None):
	max_cer=0.15
	ggsp = ds.filter(lambda cer, dur: cer < max_cer and dur > min_dur, input_columns=["cer", "audio_duration"]).select_columns(["audio_token", "nemo", "audio_duration"])
	ggsp = ggsp.rename_columns({"nemo": "text"})
	return ggsp

@register("tedlium")
def tedlium(ds: DatasetDict, min_dur=3.0, max_dur=150.1):
	new_ds = {}
	for split, ds_split in ds.items():
		new_ds[split] = ds_split.filter(lambda dur: dur > min_dur and dur < max_dur, input_columns=["audio_duration"]).select_columns(["audio_token", "audio_duration", "text"])
		new_ds[split] = new_ds[split]
	return DatasetDict(new_ds)


if __name__ == "__main__":
	'''
	Example usage:
	'''
	builder = DatasetBuilder(
		dataset_name="beat_motok_smpl_body_short",
		hf_repo="TeoGchx/BEAT_HML3D_whisper_wavtokenizer_smpl_motokenizer_body-V2",
		save_to_disk=True,
		save_dir="/data/guichoux",
	)
	dataset = builder.get_dataset()
	print(dataset)
