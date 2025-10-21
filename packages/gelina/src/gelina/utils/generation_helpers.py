import torch
from common.data.data_utils import delay_rvq, sequence_mask
from torch.nn.utils.rnn import pad_sequence
import os
from common.data.data_utils import normalize
import whisper
from torchaudio.transforms import Resample
import numpy as np


def tokens2wav(tokens, tokenizer, device='cuda'):
	# tokens 1, b, n
	features = tokenizer.codes_to_features(tokens[..., :].to(device))
	bandwidth_id = torch.tensor([0]).to(device)
	audio_out = tokenizer.decode(features, bandwidth_id=bandwidth_id)
	return audio_out

def process_audio_batch(batch):
	audio_token = [x["audio_token"] for x in batch]
	audio_token_processed = []
	for x in audio_token:
		x = x.squeeze()
		if len(x.shape) == 1:
			x = x.unsqueeze(0)
		x = delay_rvq(x+3, head_token=1, tail_token=2).transpose(-1, -2)
		audio_token_processed.append(x)
	audio_token_padded = pad_sequence(audio_token_processed, batch_first=True, padding_value=0)
	y_len = [x.shape[0] for x in audio_token_processed]
	y_mask = sequence_mask(torch.tensor(y_len), device="cpu")
	return {"audio_token": audio_token_padded,"orig_token": audio_token,"y_mask": y_mask,"y_len": y_len}


def _prepare_prompt(cfg, framerate_factor=15, speech_tokens=None, motion_tokens=None, text=None):
	'''
		Load speech and motion tokens from folder (.pth).
		Load text prompt.
		Preprocess the tokens before feeding them to the generator.
	'''
	# load text prompt
	if text is None:
		text_path = os.path.join(cfg.prompt_path, 'text.txt')
		with open(text_path, 'r', encoding='utf-8') as f:
			text = f.read()

	# load speech and motion tokens
	if speech_tokens is None:
		speech_tokens = torch.load(os.path.join(cfg.prompt_path, 'speech_tokens.pth'))

	if motion_tokens is None:
		motion_tokens = torch.load(os.path.join(cfg.prompt_path, 'motion_tokens.pth')).transpose(0,1) + 3  # shift motion IDs
	else:
		motion_tokens +=3 # shift motion IDs

	motion_tokens = motion_tokens[:,:speech_tokens.shape[-1]//15]
	speech_tokens = speech_tokens[:,:,:motion_tokens.shape[1]*framerate_factor]
	ret = process_audio_batch([{'audio_token':speech_tokens[0]}]) # (pad), head, tail, shift
	speech_tokens, orig_speech_tokens = ret['audio_token'].transpose(1,2), ret['orig_token'][0]

	return normalize(text), speech_tokens, orig_speech_tokens, motion_tokens.unsqueeze(1) # q, 1, n


def _extract_prompt(start, end,
					speech_tokens=None,
					motion_tokens=None):
	"""
	Slice a multimodal sequence between `cfg.start` s and `cfg.end` s.

	Parameters
	----------
	cfg.start, cfg.end : float
		Segment boundaries in seconds (attributes of *cfg*).
	framerate_factor : int, default 15
		Not used in the slicing itself but kept for backward compatibility.
	speech_tokens : torch.Tensor | np.ndarray | None
		Shape (..., T_s) sampled at 75 tokens / s.
	motion_tokens : torch.Tensor | np.ndarray | None
		Shape (..., T_m) sampled at 5 tokens / s.

	Returns
	-------
	speech_seg : same type as *speech_tokens* or None
	motion_seg : same type as *motion_tokens* or None
	"""
	start_s, end_s = float(start), float(end)
	if start_s >= end_s:
		raise ValueError("start must be < end")

	# ------------------------------------------------------------------ speech
	speech_seg = None
	if speech_tokens is not None:
		s0 = max(0, int(start_s * 75))
		s1 = min(speech_tokens.shape[-1], int(end_s * 75))
		speech_seg = speech_tokens[..., s0:s1]

	# ------------------------------------------------------------------ motion
	motion_seg = None
	if motion_tokens is not None:
		motion_tokens = motion_tokens.transpose(0,1)
		m0 = max(0, int(start_s * 5))
		m1 = min(motion_tokens.shape[-1], int(end_s * 5))
		motion_seg = motion_tokens[..., m0:m1]

	return speech_seg, motion_seg


def regroup(segments, max_len: float = 5.0):
    """
    Split a list of Whisper segments into consecutive chunks whose duration
    does not exceed *max_len* seconds.

    Yields
    ------
    (start, end, text) : tuple[float, float, str]
        start – timestamp of first word in chunk
        end   – timestamp of last word in chunk
        text  – whitespace-normalised sentence
    """
    cur_start, cur_end, cur_words = None, None, []
    for seg in segments:
        for w in seg["words"]:
            if cur_start is None:
                cur_start = w["start"]
            cur_end = w["end"]
            cur_words.append(w["word"].lstrip())
            if cur_end - cur_start >= max_len:
                yield cur_start, cur_end, " ".join(cur_words).strip()
                cur_start, cur_end, cur_words = None, None, []
    if cur_words:
        yield cur_start, cur_end, " ".join(cur_words).strip()
		

def mean_logprob(segments):
    return np.mean([s["avg_logprob"] for s in segments])

def wav2text(wav, asr):
	resample = Resample(24000, 16000)
	return asr.transcribe(resample(wav.cpu()).numpy(force=True).flatten().astype(np.float32), word_timestamps=True, language="en", temperature=0.0)['segments']

def words_between(wav, text):
	asr = whisper.load_model("turbo")
	rec_text = wav2text(wav, asr)
	  
	new_segs = list(regroup(rec_text))

	return new_segs, mean_logprob(rec_text)

				  