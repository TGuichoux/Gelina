from __future__ import annotations

from typing import Optional, Tuple, Dict, List

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import EinMix
from torch import nn, Tensor
import math
from tqdm import tqdm

from packages.common.src.common.utils.tools import topk_sampling
from packages.gelina.src.gelina.utils.generation_helpers import tokens2wav
from .modeling_gelina import GelinaModel

__all__: List[str] = ["GelinaModel"]  # export name now aligned


class GelinaStreaming(nn.Module):
    r"""
    Wrapper around Gelina to allow streaming.
    """

    def __init__(self, cfg, gelina, motion_decoder, speech_decoder, seed_poses=None):
        super().__init__()

        self.cfg = cfg 
        self.gelina = gelina
        self.motion_dec = motion_decoder
        self.speech_dec = speech_decoder
        self.seed_poses = seed_poses
        self.last_tokens = None
        self.generated_motion = None
        self.generated_speech = None

    def _decode_motion(self, buffer):
        batched_mu = torch.stack(buffer).transpose(0,1)

        generated = self.motion_dec.synthesise_cfg(
                    y=batched_mu,
                    n_timesteps=self.cfg.n_steps,
                    temperature=self.cfg.cfm_temp,
                    guidance_scale=self.cfg.guidance_scale,
                    solver=self.cfg.solver,
                    cond=self.seed_poses,
                )
        pose = generated['decoder_outputs'].transpose(1,2)
        self.seed_poses = pose[:,-self.cfg.n_frames_seed:]

        self.generated_motion = torch.cat([self.generated_motion, pose], axis=1) if self.generated_motion is not None else pose

    def _decode_speech(self, buffer):
        batched_speech = (torch.stack(buffer, dim=2).squeeze(-1) - self.gelina.n_special_token_in).clamp_min(0).to(torch.int64)
        
        if self.last_tokens is not None:
            batched_speech = torch.cat([self.last_tokens, batched_speech], dim=2) # to ensure continuity

        speech = tokens2wav(batched_speech, self.speech_dec, device=batched_speech.device).cpu()
        self.last_tokens = batched_speech[:,:,-75:]
       
        self.generated_speech = torch.cat([self.generated_speech, speech[:,self.cfg.speech_up*75:]], axis=1) if self.generated_speech is not None else speech

    @torch.inference_mode()
    def generate_multimodal_batch(
        self,
        x: Tensor,
        batch_size: int=3,
        prompt_speech_tokens: Optional[Tensor] = None,
        prompt_motion_tokens: Optional[Tensor] = None, # q, b, n
        device: str = "cpu",
        max_seqlen: int = 1000,
        k_speech: int = 100,
        k_motion: int = 100,
        first_greedy_quant_motion: int = 1,
        first_greedy_quant_speech: int = 1,
        temp_speech: list = [1.0],
        temp_motion: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        init_state: Optional[dict] = None,
        mask_motion_tokens: bool | str = False,
    ):
        prompt, p_len, y_embd, x_enc, stop_token, all_stop_token = \
            self.gelina._prepare_synthesis(x, batch_size, prompt_speech_tokens, prompt_motion_tokens, device)
       
        # Initialize state ---------------------------------------------------------------|
        state = init_state
        if state is None:
            state = self.gelina.attentive_rnn.init_state(max_seqlen=max_seqlen, batch_size=batch_size)

        qs, qs_speech, qs_motion, atts, stop_tokens = [], [], [], [], []    
        logits_motion = []
        logits_speech = []
        y_embd_motion_list = []
        y_embd_speech_list = []

        motion_buffer = []
        speech_buffer = []
        for t in tqdm(range(max_seqlen+p_len)):

            # AR backbone step -----------------------------------------------------------|
            y_embd, att, state = self.gelina.attentive_rnn.step(y_embd, x_enc, t, state)
            atts.append(att)

            # Token sampling -------------------------------------------------------------|
            if ((t-self.gelina.frame_rate_factor)%(self.gelina.frame_rate_factor+1))==0: # if motion step
                y_embd_motion_list.append(y_embd.squeeze(1))
                logits, q_sampled = self.gelina._sample_logits(y_embd, self.gelina.logits_head_motion, first_greedy_quant_motion, k=k_motion, temp=temp_motion)
                logits_motion.append(logits)
                if mask_motion_tokens == 'zero':
                    q_sampled = torch.zeros((self.gelina.n_quant_motion, batch_size, 1), dtype=torch.int32).to(device)
                elif mask_motion_tokens == 'random':
                    q_sampled = torch.randint(self.gelina.n_special_token_in, self.gelina.n_codebook_motion+self.gelina.n_special_token_in, (self.gelina.n_quant_motion, batch_size, 1)).to(device)
                if t >= p_len:
                    qs_motion.append(q_sampled)

                    motion_buffer.append(y_embd.squeeze(1))
                    if len(motion_buffer) >= self.cfg.motion_buffer_size:
                        motion = self._decode_motion(motion_buffer)
                        motion_buffer = []
                
            else: # if speech step
                y_embd_speech_list.append(y_embd.squeeze(1))
                logits, q_sampled = self.gelina._sample_logits(y_embd, self.gelina.logits_head_speech, first_greedy_quant_speech, k=k_speech, temp=temp_speech)
                logits_speech.append(logits)
                if t >= p_len:
                    qs_speech.append(q_sampled)  

                    speech_buffer.append(q_sampled)
                    if len(speech_buffer) >= self.cfg.speech_buffer_size:
                        speech = self._decode_speech(speech_buffer)
                        speech_buffer = []  
                
                qs.append(q_sampled)    

                # Check if stop token --------------------------------------------------------|
                is_stop_token = (q_sampled == stop_token).prod(dim=0)
                stop_tokens.append(is_stop_token)
                all_stop_token.logical_or_(is_stop_token)
                if all_stop_token.prod():
                    print("Stopped at step", t)
                    break
                
            # Embed next input -----------------------------------------------------------|
            if prompt is not None and t < p_len:
                y_embd = prompt[:,[t]]
            else:
                if ((t-self.gelina.frame_rate_factor)%(self.gelina.frame_rate_factor+1))==0:
                    y_embd = self.gelina.rvq_embed_motion(q_sampled)
                else:
                    y_embd = self.gelina.rvq_embed_speech(q_sampled)
                y_embd = reduce(y_embd, "q b n d -> b n d", "sum")

        if len(motion_buffer) > 0:
            motion = self._decode_motion(motion_buffer)
            motion_buffer = []

        if len(speech_buffer) > 0:
            speech = self._decode_speech(speech_buffer)
            speech_buffer = []

        y_embd_motion_list = torch.stack(y_embd_motion_list).transpose(0,1)

        stop_tokens.append(torch.ones(batch_size, 1, device=device))
        stop_tokens = torch.stack(stop_tokens, dim=1).squeeze(-1)
        stop_idx = (stop_tokens * rearrange(torch.arange(stop_tokens.shape[1], device=stop_tokens.device), "n -> 1 n")).long()

        speech_cuts = []
        motion_cuts = []
        for i in range(stop_idx.shape[0]):
            idx = torch.unique(stop_idx[i])[1] 
            idx_motion = idx//(self.gelina.frame_rate_factor+1)
            idx_speech = idx - idx_motion

            speech_cuts.append(self.generated_speech[[i], :idx_speech*self.cfg.speech_up])
            motion_cuts.append(self.generated_motion[[i], :idx_motion*self.cfg.motion_up])

        return speech_cuts, motion_cuts