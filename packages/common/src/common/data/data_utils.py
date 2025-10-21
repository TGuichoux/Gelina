'''
Utility function for data processing. (Text normalization, id parsing etc...)
'''


from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk, DownloadMode
import numpy as np
import os
import copy
import re
import torch
from scipy.spatial.transform import Rotation as R
import num2words
import unicodedata


def get_spk_id(x):
	'''
	Parse speaker id from file name.
    x: Dict with key ['name'] containing BEAT file name (e.g 1_zhao_12X)
	'''
	name = x['name'].split('_')

	if name[0] != 'mirrored':
		return int(name[0])
	else:
		return int(name[1])


def strip_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def normalize(x: str) -> str:
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
    x = x.replace('%', 'percent').replace('♪', '').replace("'s","s").replace("…", ".")
    currency_symbols = {
        '$': 'dollars',
        '€': 'euros',
        '£': 'pounds',
        '¥': 'yen',
        '₹': 'rupees',
        '₩': 'won'
    }
    def replace_currency(match: re.Match, currency: str) -> str:
        num = match.group(1)
        return f"{num2words.num2words(num)} {currency}"
    for symbol, currency in currency_symbols.items():
        pattern = re.escape(symbol) + r'(\d+)'
        x = re.sub(pattern, lambda m, curr=currency: replace_currency(m, curr), x)
    x = re.sub(r'(\d+)', lambda m: num2words.num2words(m.group()), x)
    x = re.sub(r'[^ -~]+', '', x)
    return x



def concat_segments(segments):
	'''
	Concatenate whisper segments.
	'''
	txt = ""
	for s in segments:
		if isinstance(s['text'], str):
			txt+=s['text']
		else:
			txt+=s['text'][0]

	return txt


def pad_2d_sequence(seq, padding_value=0):
    max_x, max_y = map(max, zip(*map(lambda x: x.shape, seq)))
    pad = lambda x: torch.nn.functional.pad(
        x,
        (0, max_y - x.shape[1], 0, max_x - x.shape[0]),
        value=padding_value,
    )
    return torch.stack([pad(x) for x in seq])

def sequence_mask(lengths, max_len=None, device="cuda"):
    lengths = lengths.to(device)
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)

    return mask

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape
    
def delay_rvq(
        code,
        head_token: int = -2,
        tail_token: int = -3,
        ):
    q, _ = code.shape
    extension = torch.ones((q, q + 1)).tril() * head_token
    extension += torch.ones((q + 1, q)).tril(diagonal=-1).T * tail_token
    extension = torch.flip(extension, (1,))
    extended_code = torch.cat((code, extension), axis=1)
    for i in range(q):
        extended_code[i, :] = torch.roll(extended_code[i, :], i + 1)

    return extended_code.long()


def undelay_rvq(extended_code):
    q, _, n = extended_code.shape
    out = []
    for i in range(q):
        out.append(torch.roll(extended_code[i], -(i + 1), dims=1))
    out = torch.stack(out, dim=0)
    return out[:, :, :-(q+1)]

def parallel_rvq(
          code,
          head_token: int = -2,
          tail_token: int = -3,
            ):
    q, _ = code.shape 
    head = torch.ones((q,1)) * head_token if head_token is not None else None
    tail = torch.ones((q,1)) * tail_token if tail_token is not None else None
    extended_code = code
    if head is not None:
        extended_code = torch.cat((head, extended_code), axis=1)
    if tail is not None:
        extended_code = torch.cat((extended_code, tail), axis=1)
    return extended_code.long()

def flatten_rvq(
        code,
        head_token: int = -2,
        tail_token: int = -3,
        ):

    q, n = code.shape 
    flat = code.T.flatten()
    extended = torch.cat((torch.tensor([head_token]), flat, torch.tensor([tail_token]))).unsqueeze(0)

    return extended

def unflatten_rvq(extended_code, n_quant):
    return extended_code[:,1:-1].reshape(-1,n_quant).T



def fix_amass_poses(poses, trans):
	'''
	Just rotating body poses to match smplx standard rotations.
	'''
	poses = poses.copy()
	r=R.from_rotvec(poses[...,:3])
	f=R.from_euler('xyz',[-90,180,0],degrees=True).inv() # 90 0 0
	poses[...,:3]=(f*r).as_rotvec()

	trans = trans.copy() #fix_amass_trans(amass_pose['trans']) # x y z
	ret = trans[:,1].copy()
	trans[:,0] *=-1
	trans[:,1] = trans[:,2].copy() - trans[:,2].max()
	trans[:,2] = ret
	return poses, trans

def resample_pose_seqV2(poses, orig_fps, target_fps):
	if len(poses.shape) == 2:
		poses = poses.unsqueeze(0)
		return torch.nn.functional.interpolate(poses.permute(0, 2, 1), scale_factor=target_fps/orig_fps, mode='linear').permute(0,2,1).squeeze(0)
	elif len(poses.shape) == 3:
		return torch.nn.functional.interpolate(poses.permute(0, 2, 1), scale_factor=target_fps/orig_fps, mode='linear').permute(0,2,1)


def segment_smplx(data_dict, start, end, fps=30):
    """
    Segment SMPLX data dictionary from start to end time.
    Args:
        data_dict (dict): Dictionary containing SMPLX data.
        start (float): Start time in seconds.
        end (float): End time in seconds.
        fps (int): Frames per second for resampling.
    Returns:
        dict: Segmented SMPLX data dictionary.
    """
    start_idx = int(start * fps)
    end_idx = int(end * fps)

    segmented_data = {}
    for key, value in data_dict.items():
        if key in ['expressions', 'poses', 'trans']:
            if isinstance(value, list):
                segmented_data[key] = value[start_idx:end_idx]
            if isinstance(value, np.ndarray):
                segmented_data[key] = value[start_idx:end_idx]
            if isinstance(value, torch.Tensor):
                segmented_data[key] = value[start_idx:end_idx]
        else:
            segmented_data[key] = value

    return segmented_data