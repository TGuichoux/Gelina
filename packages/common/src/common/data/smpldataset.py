"""
TorchDataset
------------
Windowed random-access view over a pre-processed HuggingFace dataset that
stores SMPL-X motion sequences (plus optional latents / VQ tokens).  Given a
global index it resolves the underlying clip, crops an `n_frames` slice and
returns a dict with the selected columns, normalising continuous motion when
requested.

SmplMotionDataModule
--------------------
Lightning DataModule that
1. downloads / converts raw BEAT-style SMPL-X data to a ready-to-train format
   (resampling, foot-contact computation, statistics, caching),
2. exposes train/val/test dataloaders built on **TorchDataset** and
   supports both “continuous-only” and “latents + continuous” variants through
   the `selected_columns` and `downsample_factor` arguments.
"""

import os
import sys
import json
from typing import List, Dict

import numpy as np
import torch
import pytorch_lightning as ptl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from datasets import load_dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
import smplx

from common.data.data_utils import normalize, get_spk_id, resample_pose_seqV2
import common.utils.rotation_conversions as rc
from common.data.dataset_helpers import load_smpl_dataset


def compute_contacts(body_parms, poses, exps, smplx_model):
	'''
	Needs poses at 30 fps.
	Code from EMAGE. Liu et al. 2024.
	Computes foot contacts from SMPL-X joints.
	'''

	n,j = poses.shape
	max_length = 128
	s, r = n//max_length, n%max_length
	all_tensor = []
	for i in range(s):
		with torch.no_grad():
			joints = smplx_model(
				betas=body_parms['betas'][i*max_length:(i+1)*max_length], 
				transl=body_parms['trans'][i*max_length:(i+1)*max_length], 
				expression=exps[i*max_length:(i+1)*max_length], 
				jaw_pose=body_parms['pose_jaw'][i*max_length:(i+1)*max_length], 
				global_orient=body_parms['root_orient'][i*max_length:(i+1)*max_length], 
				body_pose=body_parms['pose_body'][i*max_length:(i+1)*max_length], 
				left_hand_pose=body_parms['left_pose_hand'][i*max_length:(i+1)*max_length], 
				right_hand_pose=body_parms['right_pose_hand'][i*max_length:(i+1)*max_length], 
				return_verts=True,
				return_joints=True,
				leye_pose=body_parms['left_pose_eye'][i*max_length:(i+1)*max_length], 
				reye_pose=body_parms['right_pose_eye'][i*max_length:(i+1)*max_length],
			)['joints'][:, (7,8,10,11), :].reshape(max_length, 4, 3).cpu()
		all_tensor.append(joints)
	if r != 0:
		with torch.no_grad():
			joints = smplx_model(
				betas=body_parms['betas'][s*max_length:s*max_length+r], 
				transl=body_parms['trans'][s*max_length:s*max_length+r], 
				expression=exps[s*max_length:s*max_length+r], 
				jaw_pose=body_parms['pose_jaw'][s*max_length:s*max_length+r], 
				global_orient=body_parms['root_orient'][s*max_length:s*max_length+r], 
				body_pose=body_parms['pose_body'][s*max_length:s*max_length+r], 
				left_hand_pose=body_parms['left_pose_hand'][s*max_length:s*max_length+r], 
				right_hand_pose=body_parms['right_pose_hand'][s*max_length:s*max_length+r], 
				return_verts=True,
				return_joints=True,
				leye_pose=body_parms['left_pose_eye'][s*max_length:s*max_length+r], 
				reye_pose=body_parms['right_pose_eye'][s*max_length:s*max_length+r],
			)['joints'][:, (7,8,10,11), :].reshape(r, 4, 3).cpu()
		all_tensor.append(joints)
	if len(all_tensor) > 0:
		joints = torch.cat(all_tensor, axis=0)
		feetv = torch.zeros(joints.shape[1], joints.shape[0])
		joints = joints.permute(1, 0, 2)
		feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
		contacts = (feetv < 0.01).numpy().astype(float).transpose(1, 0)
		contacts = torch.from_numpy(contacts).float()

	return contacts


class TorchDataset(Dataset):
	def __init__(self, dataset, n_frames, lengths, cumsum, mean, std, normalize, normalize_trans, downsample_factor=4, selected_columns=["motion", "betas"], device="cuda"):
		'''
		Takes a pre-processed HuggingFace dataset and returns a windowed
		random-access view over it. The dataset is expected to have been
		pre-processed to contain motion sequences at the target framerate
		(20 fps) and with foot contacts computed. The motion sequences are
		expected to be in the form of a dictionary with keys 'motion' and
		'betas', where 'motion' is a tensor of shape [n_frames, 337].
		'''
		super().__init__()

		self.dataset = dataset # processed dataset (at target framerate and with contacts)
		self.selected_columns = selected_columns
		
		self.dataset = self.dataset.select_columns(selected_columns)
		self.n_frames = n_frames # window size
		self.lengths = lengths # lengths of all sequences in dataset (at target frame rate)
		self.cumsum = cumsum 

		self.mean = mean
		self.std = std

		self.device = device

		self.normalize = normalize
		self.normalize_trans = normalize_trans

		self.downsample_factor = downsample_factor

	def __len__(self):
		return self.cumsum[-1]

	def __getitem__(self, item):
		'''
		Takes the sequence with smallest framerate to define window:
		motion: [3456, 337] speech_lat: [12960, 1024] (3.75 times more data) -> M_idx: [2341, 2441] (100) -> S_l_idx: [int(2341*3.75), int(2441*3.75)] (375)
		motion: [3456, 337] motion_lat: [864, 1024] (4 times less data) -> M_l_idx: [254, 382] (128) -> M_idx: [int(254*4), int(382*4)] (512)
		('cause i'm too messy 🎶)
		'''
		if item != 0:
			motion_id = np.searchsorted(self.cumsum, item) - 1
			idx = item - self.cumsum[motion_id] - 1
		else:
			motion_id = 0
			idx = 0

		motion_id = int(motion_id)
		data = self.dataset[motion_id]

		ret_dict = {}
		if 'motion_latents' in self.selected_columns:
			ret_dict['motion_latents'] = data['motion_latents'].squeeze(0)[idx:(idx + self.n_frames)] # tokens and latents are at 5 t/s while motion is at 20 fps
			m_idx_s, m_idx_e = idx*int(self.downsample_factor), (idx + self.n_frames)*int(self.downsample_factor)

			if 'motion_tokens' in self.selected_columns:
				ret_dict['motion_tokens'] = data['motion_tokens'].squeeze(0)[idx:(idx + self.n_frames)] - 3 if "motion_tokens" in data.keys() else None # forgot to do that before pushing dataset. c.f data/groupdataset.py line 947 (collate function)
			motion = data['motion']['motion'][m_idx_s:m_idx_e]

			#print("ret latent", ret_dict['motion_latents'].shape, "motion latents", data['motion_latents'].shape, "IDX", idx, "m_idx_start:", m_idx_s, "m_idx_end:", m_idx_e, "motion", data['motion']['motion'].shape)
			
		elif 'speech_latents' in self.selected_columns:
			motion = data['motion']['motion'][idx:idx+self.n_frames]
			s_idx_s, s_idx_e = int(idx*self.downsample_factor), int((idx + self.n_frames)*(self.downsample_factor))

			ret_dict['speech_latents'] = data['speech_latents'].squeeze(0)[s_idx_s:s_idx_e] # speech latents are at 75 fps while motion is at 20 fps

		else:
			motion = data['motion']['motion'][idx:idx+self.n_frames]

		ret_dict['betas'] = data['motion']['betas']
	
		if self.normalize:
			motion[...,:333] = (motion[...,:333] - self.mean) / (self.std+1e-8) # we normalize across body and translations. Not foot contacts. # as hand motion is place holder, std is zero

		elif self.normalize_trans:
			motion[...,330:333] = (motion[...,330:333] - self.mean[330:333]) / self.std[330:333]

		
		ret_dict['motion'] = motion

		#print(ret_dict['motion'].shape, ret_dict['motion_latents'].shape)

		return ret_dict



class SmplMotionDataModule(ptl.LightningDataModule):
	def __init__(
		self,
		root_dir: str,
		dataset_names: List[str],
		hf_repos: List[str],

		selected_columns: list[str] = ['motion'],
		n_frames=64,
		orig_fps=30,
		target_fps=20,
		rep='axis_angle',
		downsample_factor=4, # value used only for dataset latent, it's the sampling factor between latents and continuous motion

		train_batch_size=32,
		val_batch_size=32,
		num_workers=0,
		shuffle_train=True,
		shuffle_val=True,
		merge_train_additional=False,

		normalize=False,
		normalize_trans=False,
	
		save_processed = True,
		reload_dataset = False,
		speaker_ids = None,
	):
		super().__init__()
		self.save_dir = os.path.join(root_dir, dataset_names[0])
		self.dataset_names = dataset_names
		self.hf_repos = hf_repos

		self.speaker_ids = speaker_ids
		self.selected_columns = selected_columns
		self.n_frames = n_frames
		self.orig_fps = orig_fps
		self.target_fps = target_fps
		self.rep = rep
		self.downsample_factor = downsample_factor

		self.batch_sizes = {'train': train_batch_size, 'val': val_batch_size}
		self.shuffle_train = shuffle_train
		self.shuffle_val = shuffle_val
		self.num_workers = num_workers
		self.reload_dataset = reload_dataset
		self.save_processed = save_processed
		self.merge_train_additional = merge_train_additional

		self.normalize = normalize
		self.normalize_trans = normalize_trans
		self.global_mean = None
		self.global_std = None

		self.smplx = smplx.create(
			"checkpoints/smplx_models/", 
			model_type='smplx',
			gender='NEUTRAL_2020', 
			use_face_contour=False,
			num_betas=300 if 'beat' in self.dataset_names[0] else 16,
			num_expression_coeffs=100, 
			ext='npz',
			use_pca=False,
		).cuda().eval()

		print("SMPLX model:", self.smplx)

	def setup(self, stage=None):
		proc_path = f"{self.save_dir}/processed_data"

		# 1 load / preprocess ----------------------------------------------------------------
		if not os.path.exists(proc_path) or self.reload_dataset:
			if self.hf_repos[0] is not None:
				ds = load_smpl_dataset(self.hf_repos[0]).with_format("torch") # Fetch from hub
			else:
				ds = load_from_disk(self.save_dir).with_format("torch") # Load local save


			if not self.merge_train_additional:
				ds.pop('additional', None)
			ds = ds.rename_columns({"beat_motion": "motion_smpl"})
			#ds = ds.filter(lambda motion: len(motion['poses']) > 20, input_columns = "motion_smpl") # ensure duration is over 0.4s
			if 'motion_latents' in self.selected_columns:
				ds = ds.filter(lambda latents: latents.shape[0] > self.n_frames, input_columns = "motion_latents") 
			elif 'speech_latents' in self.selected_columns:
				ds = ds.filter(lambda latents: latents.shape[0] > self.n_frames, input_columns = "speech_latents") 		
			print(ds)	
			ds = ds.map(
				lambda x: {"motion": self.preprocess_data(x)},
				input_columns="motion_smpl",
				desc="contacts & resample",
				num_proc=None,
				writer_batch_size=100
			)
			if self.save_processed:
				ds.save_to_disk(proc_path)
		else:
			ds = load_from_disk(proc_path)

		# filter -------------------------------------------------------------------
		if self.speaker_ids is not None:
			ds = ds.filter(lambda id: id in self.speaker_ids, input_columns="speaker_id")
		self.dataset = ds

		# Optional merge ---------------------------------------------------------------------
		if self.merge_train_additional:
			ds["train"] = concatenate_datasets([ds["train"], ds["additional"]])

		# 3 lengths --------------------------------------------------------------------------
		len_path = f"{self.save_dir}/lengths.json"
		if not os.path.exists(len_path) or self.reload_dataset:
			os.makedirs(self.save_dir, exist_ok=True)
			self.compute_lengths()
		with open(len_path) as f:
			self.lengths = json.load(f)
		self.cumsum = {k: np.cumsum([0] + v) for k, v in self.lengths.items()}

		# 4 stats ----------------------------------------------------------------------------
		mean_path = f"{self.save_dir}/Mean.npy"
		if not os.path.exists(mean_path):
			os.makedirs(self.save_dir, exist_ok=True)
			self.compute_motion_statistic()
		self.global_mean = torch.from_numpy(np.load(mean_path))
		self.global_std = torch.from_numpy(np.load(f"{self.save_dir}/Std.npy"))

		# 5 log ------------------------------------------------------------------------------
		print(
			f"Total motions  Train {len(ds['train'])} / Val {len(ds['val'])} | "
			f"snippets  Train {self.cumsum['train'][-1]} / Val {self.cumsum['val'][-1]}"
		)

	def preprocess_data(self, m_data):
		body_parms, betas, poses, trans, exps = self.load_data(m_data) # frame rate is adjusted (all are set to self.orig_fps) and pose is decomped by body parts
		contacts = compute_contacts(body_parms, poses, exps, self.smplx) # compute foot contacts
		n, _ = poses.shape
		if self.rep == 'rot6d':
			mat = rc.axis_angle_to_matrix(poses.reshape(n, 55, 3))
			poses_6d = rc.matrix_to_rotation_6d(mat).reshape(n, 55*6)
			full_body = torch.cat((poses_6d, trans), dim=1)
		else:
			full_body = torch.cat((poses, trans), dim=1)
		full_body = resample_pose_seqV2(full_body, self.orig_fps, self.target_fps)
		contacts = resample_pose_seqV2(contacts, self.orig_fps, self.target_fps)
		
		full_body = torch.cat([full_body.cuda(), contacts.cuda()], dim=1)

		return {'motion':full_body, 'betas':betas[0].squeeze()}

	def compute_lengths(self):
		self.lengths = {"train":[], "val":[]}
		n_cont = 0
		for split in ["train", "val"]:
			for data in tqdm(self.dataset[split], desc="length compute"):
				motion = data['motion']['motion']
				if 'motion_latents' in data.keys():
					latents = data['motion_latents']
					if len(latents.shape) == 3:
						b,seq_len,d = latents.shape
					else:
						seq_len,d = latents.shape
					if seq_len < self.n_frames:
						n_cont+=1
						print("cont")
						continue
					self.lengths[split].append(seq_len - self.n_frames)
				else:
					if len(motion.shape) == 3:
						b,mo_length, d = motion.shape
					else:
						mo_length, d= motion.shape
					if mo_length < self.n_frames:
						continue
					self.lengths[split].append(mo_length - self.n_frames)
		print("Number of filtered sequences:", n_cont)
		with open(self.save_dir+'/lengths.json', 'w') as f:
			json.dump(self.lengths, f)
		
	def compute_motion_statistic(self):

		motion = None
		for data in tqdm(self.dataset['train'], desc='stat'):
			if motion is None:
				motion = data['motion']['motion'][...,:333]
			else:
				motion = torch.cat([motion, data['motion']['motion'][...,:333]]) # only body and translations. We do not normalize foot contacts
		train_motion_mean = motion.mean(axis=0)
		train_motion_std = motion.std(axis=0)

		np.save(self.save_dir+'/Mean.npy', train_motion_mean.numpy(force=True))
		np.save(self.save_dir+'/Std.npy', train_motion_std.numpy(force=True))

		
	def load_data(self, m_data):
		
		betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
	
		n, c = poses.shape[0], poses.shape[1]
	
		betas = betas.reshape(1, 300).numpy(force=True)

		exps = exps.reshape(n, 100).cuda().float()
	
		betas = np.tile(betas, (n, 1))
		betas = torch.from_numpy(betas).cuda().float()
		poses = poses.reshape(n, c).cuda().float()
		trans = trans.reshape(n, 3).cuda().float() 
		trans[...,1] -= trans[...,1].max()

		body_parms = {
			'root_orient': poses[:, :3].cuda(),
			'pose_body': poses[:, 3:21*3+3].cuda(),
			'pose_jaw': poses[:, 66:69].cuda(),
			'left_pose_hand': poses[:, 25*3:40*3].cuda(),
			'right_pose_hand': poses[:, 40*3:55*3].cuda(),
			'left_pose_eye': poses[:,69:72].cuda(),
			'right_pose_eye': poses[:, 72:75].cuda(),
			'trans': trans.cuda(),
			'betas': betas.cuda(),
		}

		return body_parms, betas, poses, trans, exps

	def collate(self, batch):
		return default_collate(batch)

	def unormalize(self, motion):
		motion[...,:333] = motion[...,:333] * (self.global_std.to(motion.device)+1e-8) + self.global_mean.to(motion.device) # we don't normalize on foot contacts
		return motion

	def _dl(self, split="train"):
		ds = TorchDataset(self.dataset[split].with_format("torch"), 
								self.n_frames,
								self.lengths[split],
								self.cumsum[split],
								self.global_mean,
								self.global_std,
								normalize = self.normalize,
								normalize_trans=self.normalize_trans,
								selected_columns=self.selected_columns,
								downsample_factor=self.downsample_factor
								)
		shuffle = self.shuffle_train if split == 'train' else self.shuffle_val

		return DataLoader(
			ds,
			batch_size=self.batch_sizes[split],
			shuffle=shuffle,
			num_workers=self.num_workers,
			collate_fn=self.collate
		)

	def train_dataloader(self):
		return self._dl('train')

	def val_dataloader(self):
		return self._dl('val')
	
	def test_dataloader(self, split="test"):
		return self._dl(split)

if __name__ == '__main__':
	import os
	os.environ["DATASETS_MP_START_METHOD"] = "spawn"   # before you import datasets
	
	# ds = SmplMotionDataModule(
	# 	root_dir='/data/guichoux/',
	# 	dataset_names=['beatV2'],
	# 	hf_repos=['TeoGchx/BEAT_HML3D_whisper_wavtokenizer'],
	# 	rep='rot6d',
	# 	train_batch_size=4,
	# 	save_processed=True,
	# 	reload_dataset=False
	# )
	# ds.setup(stage=None)

	# loader = ds.train_dataloader()
	# batch = next(iter(loader))
	# print(batch.keys())
	# print(batch['motion'].shape)


	ds = SmplMotionDataModule(
		root_dir='/data/guichoux/',
		dataset_names=['beat_latents-v3-8_seed'],
		selected_columns=['motion', 'motion_latents', 'motion_tokens'],
		hf_repos=['TeoGchx/beat_with_latents-v3'],
		rep='rot6d',
		n_frames=10,
		train_batch_size=32,
		downsample_factor=4,
		save_processed=True,
		reload_dataset=False,
		num_workers=8
	)
	ds.setup(stage=None)

	loader = ds.train_dataloader()
	batch = next(iter(loader))
	print(batch.keys())
	print(batch['motion'].shape)

	for batch in tqdm(loader):
		continue

	# ds = SmplMotionDataModule(
	# 	root_dir='/data/guichoux/',
	# 	dataset_names=['beat_latents_speech'],
	# 	selected_columns=['motion', 'speech_latents'],
	# 	hf_repos=['TeoGchx/beat_with_speech_only_latents'],
	# 	rep='rot6d',
	# 	n_frames=100,
	# 	train_batch_size=4,
	# 	downsample_factor=3.75,  # speech latents are at 75 fps while motion is at 20 fps
	# 	save_processed=True,
	# 	reload_dataset=False
	# )
	# ds.setup(stage=None)

	# loader = ds.train_dataloader()
	# batch = next(iter(loader))
	# print(batch.keys())
	# print(batch['motion'].shape)
# python packages/common/src/common/data/smpldataset.py