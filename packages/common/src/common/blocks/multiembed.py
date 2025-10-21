import os
import sys
from functools import partial

import torch
from torch import nn


class MultiEmbedding(nn.Module):
	"""
	Stacks multiple homogeneous embedding (same size, same embedding dim.) into one single weight.
	"""

	def __init__(self, n_level, n_emb, d_emb, padding_idx=None):
		super().__init__()
		self.weight = nn.Parameter(
			torch.empty(n_level, n_emb, d_emb, requires_grad=True)
		)
		self.n_level = n_level
		self.padding_idx = padding_idx
		nn.init.normal_(self.weight)

	def forward(self, idx):
		emb_fn = partial(nn.functional.embedding, padding_idx=self.padding_idx)
		return torch.vmap(emb_fn)(idx, self.weight)

class MotionVQEmbedding(nn.Module):
	def __init__(self, vq_model, padding_idx=None, token_shift=3, freeze_vq=True):
		super().__init__()
		self.vq_model = vq_model
		if freeze_vq:
			for param in self.vq_model.parameters():
				param.requires_grad = False
		self.vq_model.eval()

		self.padding_idx = padding_idx
		self.token_shift = token_shift
		self.padding_embedding = nn.Parameter(
			torch.empty(1, 1, vq_model.code_dim, requires_grad=True)
		)
		nn.init.normal_(self.padding_embedding, mean=0.0, std=0.02)

	def forward(self, idx: torch.LongTensor) -> torch.Tensor:
		'''
		Get codes from VQ codebook. Tokens are shifted by token_shift in collate_fn
		Special embedding for padding_idx.  
		idx: (b, n, q)  
		'''
		if self.padding_idx is None:
			return self.vq_model.quantizer.get_codebook_entry(idx - self.token_shift)
		
		pad_mask = idx == self.padding_idx
		if pad_mask.all():
			return self.padding_embedding.expand(idx.size(0), idx.size(1), -1)
		shifted_idx = idx.clone()
		shifted_idx[~pad_mask] = shifted_idx[~pad_mask] - self.token_shift
		codes = self.vq_model.quantizer.get_codebook_entry(shifted_idx)
		if pad_mask.any():
			codes[pad_mask] = self.padding_embedding
		return codes


if __name__ == "__main__":

	from utils.parser.eval_parser import EvalParser
	from utils.load_model import load_vq
	print('ok')
	parser = EvalParser()
	args = parser.parse_args()

	print('ok2')
	# b, n, q, l, d = 32, 1218, 1, 1027, 1024
	# e = torch.compile(MultiEmbedding(q, l, d))
	# i = torch.randint(l, (q, b, n))
	# print(i.shape)
	# print(e(i).shape)
	# print(e(i).shape)  

	a = 4*torch.ones(32,128, 6, dtype=torch.int64).cuda() # b, n, q
	print(a.shape)

	motion_vq = load_vq(args.vq_checkpoint)

	embeddings = MotionVQEmbedding(motion_vq, padding_idx=0, token_shift=3)
	codes = embeddings(a)
	print(codes.shape)
