"""
Collate registry + three recipes

* **speech**     text + speech tokens  
* **speech_motion**  text + speech tokens + **true** motion tokens interleaved
					 at a fixed temporal ratio  
* **speech_pt**    text + speech tokens + **random** motion tokens
					 (pre-training stage)
"""

from __future__ import annotations
from typing import Callable, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence

from common.data.data_utils import sequence_mask
from transformers import PreTrainedTokenizerFast


_COLLATE: Dict[str, type] = {}


def register_collate(name: str):
	def _wrap(cls):
		_COLLATE[name] = cls
		return cls
	return _wrap


def _resolve_pattern(pattern_name: str) -> Callable:
	"""
	Look up `pattern_name` in common.data.data_utils
	and return the callable (raises AttributeError if missing).
	"""
	from common.data import data_utils 
	return getattr(data_utils, pattern_name)


def build(name: str, **kwargs):
	"""
	Instantiate a collate recipe.

	If the caller supplied `pattern="<string>"` we turn it into the actual
	function defined in `common.data.data_utils` before calling the class
	constructor.
	"""
	if isinstance(kwargs.get("pattern"), str):
		kwargs["pattern"] = _resolve_pattern(kwargs["pattern"])
	return _COLLATE[name](**kwargs)



@register_collate("speech")
class SpeechCollate:
	"""Batch of text and **speech tokens only**."""

	def __init__(
		self,
		tokenizer_file,
		speech_rate_hz,
		pattern: Callable,
		quant_layer_speech,
		pad_text: int | None = None,
		**kwargs,
	):
		self.tokenizer_file = tokenizer_file    
		self._tok = None  
		self.pattern = pattern
		self.quant_layer_speech = quant_layer_speech
		self.pad_text = pad_text

	@property
	def tok(self):
		if self._tok is None:                   
			self._tok = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_file)
			self._bos = self._tok.convert_tokens_to_ids("[BOS]")
			self._eos = self._tok.convert_tokens_to_ids("[EOS]")
		return self._tok
	
	def __call__(self, batch: List[dict]):
		audio_token, text = zip(*[(b["audio_token"], b["text"]) for b in batch])
		try:
			text_token = [
				torch.LongTensor(self.tok.encode(f"[BOS]{t}[EOS]")) for t in text
			]
		except:
			print("error when tokenizing text: ", text)

		audio_delayed = []
		for a in audio_token:
			a = a.squeeze()
			if a.ndim == 1:
				a = a.unsqueeze(0)
			a = self.pattern(a + 3, head_token=1, tail_token=2).transpose(-1, -2)[
				..., self.quant_layer_speech
			]
			audio_delayed.append(a)

		xlen = torch.tensor([t.size(0) for t in text_token])
		ylen = torch.tensor([a.size(-2) for a in audio_delayed])

		x_mask, y_mask = map(
			lambda l, m: sequence_mask(l, device="cpu", max_len=m),
			(xlen, ylen),
			(self.pad_text, None),
		)
		audio_pad, text_pad = map(
			lambda seq: pad_sequence(seq, batch_first=True, padding_value=0),
			(audio_delayed, text_token),
		)

		enc_mask = x_mask.unsqueeze(1) & x_mask.unsqueeze(2)
		cross_mask = x_mask.unsqueeze(1) & y_mask.unsqueeze(2)
		cross_mask[:, :, 0] = True

		sources = [b.get("source", None) for b in batch]
		return {
			"text_token": text_pad,
			"audio_token": audio_pad,
			"crossatt_mask": cross_mask,
			"encoder_mask": enc_mask,
			"y_mask_speech": y_mask,
			"y_mask": y_mask,
			"source": sources,
		}


# ----------------------------------------- text + speech + **real** motion
@register_collate("speech_motion")
class SpeechMotionCollate:
	"""
	Speech-and-Motion batching

	Ensures one motion frame every `frame_rate_factor` speech frames**.
	The collate keeps speech and motion sequences separate but builds
	`y_mask` that 'tells' the model where motion frames sit in the notional
	interleaved stream.
	Speech and motion interleaving is performed during forward, after token embeddings.

	Steps
	-----
	1. _interleave_apply_pattern – trims to aligned lengths and applies the
	   quantisation pattern (here parallel pattern as different number of quantizers for each modality. c.f MusicGen for more details: https://arxiv.org/abs/2306.05284).  
	2. Pads each stream, builds individual masks.  
	3. Combines them into the interleaving mask `y_mask`.

	
		Example:
		speech: [head, s, s, s, s, s, s, tail, pad, pad, pad] -> mask : [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
		motion: [m, m, pad] -> mask : [1, 1, 0]
		interleaved: [head, s, s, s, m, s, s, s, m, tail, 0,0,0,0] (assuming speech rate is 3 and motion rate is 1)
		interleaved mask (y_mask): [1 (head), 1, 1, 1, 1 (motion), 1, 1, 1, 1 (motion), 1 (tail), 0,0,0,0]
	
	"""

	def __init__(
		self,
		speech_rate_hz,
		motion_rate_hz,
		pattern: Callable,
		quant_layer_speech,
		quant_layer_motion,
		tokenizer_file,
		**kwargs,
	):
		
		self.tokenizer_file = tokenizer_file    
		self._tok = None                         
		self.frame_rate_factor = speech_rate_hz // motion_rate_hz
		self.pattern = pattern
		self.quant_layer_speech = quant_layer_speech
		self.quant_layer_motion = quant_layer_motion
		

	@property
	def tok(self):
		if self._tok is None:         

			self._tok = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_file)
			self._bos = self._tok.convert_tokens_to_ids("[BOS]")
			self._eos = self._tok.convert_tokens_to_ids("[EOS]")
		return self._tok

	def _encode_txt(self, txt: str) -> torch.Tensor:
		ids = self.tok.encode(txt) 
		try:
			tokens = torch.tensor([self._bos, *ids, self._eos])
		except:
			print(f"Warning: text '{txt}' could not be tokenized. Using empty sequence.")
			raise ValueError(f"Text '{txt}' could not be tokenized.")
		return tokens

	def _fake_motion(self, n_frames: int) -> torch.Tensor:
		m = -3*torch.ones((n_frames, len(self.quant_layer_motion)), dtype=torch.long) # pad tokens is 0 and we shift tokens (+3) in main collate.
		return m

	def _interleave_apply_pattern(self, audio, motion):
		if motion[0][0]==-1: # means dummy token
			n_frames = audio.shape[-1] // self.frame_rate_factor
			motion = self._fake_motion(n_frames)  
	
		audio = audio.squeeze()

		motion = motion.transpose(0, 1)
		if audio.ndim == 1:
			audio = audio.unsqueeze(0)
		if motion.ndim == 1:
			motion = motion.unsqueeze(0)

		max_m = min(audio.shape[1] // self.frame_rate_factor, motion.shape[1])
		audio = audio[..., : max_m * self.frame_rate_factor]
		motion = motion[..., : max_m]

		audio = self.pattern(audio + 3, head_token=1, tail_token=2).transpose(-1, -2)[
			..., self.quant_layer_speech
		]
		motion = self.pattern(motion + 3, head_token=None, tail_token=None).transpose(
			-1, -2
		)[..., self.quant_layer_motion]
		return audio, motion


	def __call__(self, batch: List[dict]):
		audio_raw, motion_raw, text = zip(
			*[(b["audio_token"], b["motion_token"], b["text"]) for b in batch]
		)
		try:
			text_token = [self._encode_txt(t) for t in text]
		except:
			print("error when tokenizing text: ", text)
		aud_proc, mot_proc = zip(
			*[
				self._interleave_apply_pattern(a, m)
				for a, m in zip(audio_raw, motion_raw)
			]
		)

		xlen = torch.tensor([t.size(0) for t in text_token])
		ylen_s = torch.tensor([a.size(-2) for a in aud_proc])
		ylen_m = torch.tensor([m.size(-2) for m in mot_proc])

		x_mask, y_mask_s, y_mask_m = map(lambda x, y: sequence_mask(x, device="cpu", max_len=y), (xlen, ylen_s, ylen_m), (None, None, None))
		aud_pad, mot_pad, txt_pad = map(lambda x: pad_sequence(x, batch_first=True, padding_value=0), (aud_proc, mot_proc, text_token))


		b, n_s, _ = aud_pad.shape
		_, n_m, _ = mot_pad.shape
		idx = torch.arange(n_s + n_m)
		motion_idx = idx % (self.frame_rate_factor + 1) == self.frame_rate_factor
		speech_idx = ~motion_idx

		y_mask = torch.zeros((b, n_s + n_m), dtype=torch.bool).to(x_mask.device)
		y_mask[:, speech_idx] = y_mask_s
		y_mask[:, motion_idx] = y_mask_m

		enc_mask = x_mask.unsqueeze(1) & x_mask.unsqueeze(2)
		cross_mask = x_mask.unsqueeze(1) & y_mask.unsqueeze(2)
		cross_mask[:, :, 0] = True

		sources = [b.get("source", None) for b in batch]
		return {
			"text_token": txt_pad,
			"audio_token": aud_pad,
			"motion_token": mot_pad,
			"crossatt_mask": cross_mask,
			"encoder_mask": enc_mask,
			"y_mask_motion": y_mask_m,
			"y_mask_speech": y_mask_s,
			"y_mask": y_mask,
			"source": sources,
			"beat_motion": [b.get("beat_motion", None) for b in batch],
		}


# ------------------------------- speech + **pad** motion (pre-training)
@register_collate("speech_pt")
class SpeechPTCollate(SpeechMotionCollate):
	"""Same mask layout as `speech_motion`; motion tokens are pad."""

	def __init__(self, *args, motion_vocab_size: int, **kwargs):
		super().__init__(*args, **kwargs)
		self.motion_vocab_size = motion_vocab_size

	def __call__(self, batch: List[dict]):
		new_batch = []
		for b in batch:
			n_frames = b["audio_token"].shape[-1] // self.frame_rate_factor
			b_fake = dict(b)
			b_fake["motion_token"] = self._fake_motion(n_frames)
			new_batch.append(b_fake)
		return super().__call__(new_batch)

# ------------------------------- speech + **random** motion (pre-training)
@register_collate("speech_pt_random")
class SpeechPTRandomCollate(SpeechPTCollate):
	"""Same mask layout as `speech_pt`; motion tokens are random."""

	def _fake_motion(self, n_frames: int) -> torch.Tensor:
		m = torch.randint(
			0,
			self.motion_vocab_size,
			(n_frames, len(self.quant_layer_motion)),
			dtype=torch.long,
		)
		return m

if __name__ == "__main__":
	'''
		Example usage and sanity check.
	'''
	
	#from common.data.data_utils import delay_rvq, parallel_rvq
	from transformers import PreTrainedTokenizerFast

	delay_rvq = "delay_rvq"
	parallel_rvq = "parallel_rvq"
	
	txt_tok = PreTrainedTokenizerFast(tokenizer_file="checkpoints/bpe256.json")

	collate_fn = build(
	"speech",                       
	tokenizer=txt_tok,
	speech_rate_hz=75,
	pattern=delay_rvq,
	quant_layer_speech=[0],
	pad_text=None,
	)

	collate_fn = build(
		"speech_motion",                
		tokenizer=txt_tok,
		speech_rate_hz=75,
		motion_rate_hz=30,
		pattern=parallel_rvq,
		quant_layer_speech=[0, 1],
		quant_layer_motion=[0],
	)

	collate_fn = build(
		"speech_pt",                 
		tokenizer=txt_tok,
		speech_rate_hz=75,
		motion_rate_hz=30,
		pattern=parallel_rvq,
		quant_layer_speech=[0, 1],
		quant_layer_motion=[0],
		motion_vocab_size=512,
	)

	print("Everything went ok (yey 🙌)")