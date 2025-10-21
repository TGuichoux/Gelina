from typing import Callable, List, Optional

import torch

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def topk_sampling(seq, k=1, temp=1.):
    topk = torch.topk(seq, k, dim=-1)
    logits = seq / temp
    mask = logits < topk.values[:, [-1]]
    logits[mask]  = -float('Inf')
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def topk_samplingV2(seq, k=1, temp=1.):
    topk = torch.topk(seq, k, dim=-1)
    cutoff = topk.values[:, [-1]] 

    mask = seq < cutoff
    logits = seq.clone()
    logits[mask] = -float('Inf')

    logits = logits / temp

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def topk_topp_sampling(logits, k=50, p=0.95, temp=1.0):
    logits = logits.clone()

    if k is not None and k < logits.size(-1):
        kth_vals = torch.topk(logits, k, dim=-1).values[..., -1, None]  # shape (..., 1)
        mask = logits < kth_vals
        logits[mask] = float('-inf')

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    above_p = cumulative_probs > p
    above_p[..., 1:] = above_p[..., :-1].clone()  # Shift mask to the right
    above_p[..., 0] = False  # Keep the first logit always unmasked
    sorted_logits[above_p] = float('-inf')
    logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    logits = logits / temp

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
  

def to_vocos(x):
    x = undelay_rvq(x.T)
    x = (x - 3).clamp_min(0)
    return x

def txt_to_phon(x):
    return espeak.phonemize([x], strip=True, njobs=1)[0]
    
def phon_to_code(x):
    y = [ds.vocab_x_code[t] for t in x]
    y = [ds.vocab_x_code["BOS"]] + y + [ds.vocab_x_code["EOS"]]
    return torch.tensor(y)


def align_mask(src_mask, mel_mask):
    align_mask = torch.zeros(
        src_mask.shape[0], src_mask.shape[-1], mel_mask.shape[-1], dtype=torch.bool
    )
    for i, (src, mel) in enumerate(zip(src_mask, mel_mask)):
        w = torch.max(src.sum(-1))
        l = torch.max(mel.sum(-1))
        align_mask[i, :w, :l] = torch.ones(w, l, dtype=torch.bool)
    return align_mask


def last_that_fullfil(cond: Callable, x: torch.Tensor, strict: bool = True):
    res = cond(x).nonzero()
    if strict:
        assert len(res), f"no one fullfill {cond}"
    return res[-1]


def first_that_fullfil(cond: Callable, x: torch.Tensor, strict: bool = True):
    res = cond(x).nonzero()
    if strict:
        assert len(res), f"no one fullfill {cond}"
    return res[0]

def resample_pose_seq(poses, orig_fps, target_fps):
	if len(poses.shape) == 2:
		poses = poses.unsqueeze(0)
		return torch.nn.functional.interpolate(poses.permute(0, 2, 1), scale_factor=target_fps/orig_fps, mode='linear').permute(0,2,1).squeeze(0)
	elif len(poses.shape) == 3:
		return torch.nn.functional.interpolate(poses.permute(0, 2, 1), scale_factor=target_fps/orig_fps, mode='linear').permute(0,2,1)

