from __future__ import annotations

from typing import Optional, Tuple, Dict, List

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import EinMix
from torch import nn, Tensor
import math
from tqdm import tqdm

from packages.common.src.common.blocks.attentive_rnn import AttentiveRNN
from packages.common.src.common.blocks.multiembed import MultiEmbedding
from packages.common.src.common.utils.tools import topk_sampling
from packages.common.src.common.data.data_utils import sequence_mask, undelay_rvq

__all__: List[str] = ["GelinaModel"]  # export name now aligned

torch.set_float32_matmul_precision('high')


class GelinaModel(nn.Module):
    r"""Versatile autoregressive model that jointly predicts *speech* and/or
    *motion* RVQ (residual-vector-quantised) token streams, optionally
    conditioned on text and speaker embeddings.

    Three operational modes
    -----------------------
    * **"bimodal"** – interleaved speech & motion targets
    * **"speech"**  – speech-only generation
    * **"motion"**  – motion-only generation
    """

    def __init__(
        self,
        attentive_rnn: AttentiveRNN,
        logits_head_speech: nn.Module,
        logits_head_motion: nn.Module,
        d_model: int,
        n_quant_speech: int,
        n_quant_motion: int,
        n_codebook_speech: int,
        n_codebook_motion: int,
        n_txt_vocab: int,
        n_special_token_in: int = 0,
        n_special_token_out: int = 0,
        *,
        frame_rate_factor: int = 15,
        txt_encoder: Optional[nn.Module] = None,
        mask_text_p: float = 0.0,
        mode: str = "bimodal",  # 'bimodal' | 'speech' | 'motion'
        pe_mode: str = 'modality_agnostic',  # 'modality_agnostic' | 'modality_aware' | 'none',
        ignore_index=0, # pad token
    ) -> None:
        """Instantiate a :class:`GelinaModel`.

        Parameters
        ----------
        attentive_rnn
            The core AttentiveRNN decoder that autoregressively predicts the
            next latent given previous latents and an encoded text context.
        d_model
            Dimensionality of all latent embeddings / hidden states.
        n_quant_speech
            Number of quantisers (stages) in the RVQ stack for speech.
        n_quant_motion
            Number of quantisers (stages) in the RVQ stack for motion.
        n_codebook_speech
            Size of *each* code-book in the speech RVQ stack (excluding any
            special tokens).
        n_codebook_motion
            Size of *each* code-book in the motion RVQ stack (excluding any
            special tokens).
        n_txt_vocab
            Size of the text vocabulary (including padding & mask tokens).
        n_special_token_in
            Additional special tokens *input* to the model (e.g. SOS/EOS).
        n_special_token_out
            Additional special tokens *output* by the model (e.g. EOS only).
        frame_rate_factor
            Ratio ``(audio-fps // motion-fps)`` – controls interleaving stride
            when operating in ``mode="bimodal"``.
        txt_encoder
            Optional text-encoding module (e.g. Transformer encoder).
        spk_encoder
            Optional speaker-embedding module whose output is prepended to the
            latent sequence as a *speaker token*.
        mask_text_p
            Probability of replacing an entire text sequence with `<mask>`
            during training (text-in-noise augmentation).
        mode
            Generation mode: one of ``{"bimodal", "speech", "motion"}``.
        pe_mode
            Positional encoding mode: one of ``{"modality_agnostic", "modality_aware", "none"}``.
            - ``"modality_agnostic"`` applies a single positional encoding to the entire sequence.
            - ``"modality_aware"`` applies separate positional encodings to speech and motion tokens.
            - ``"none"`` disables positional encoding.
        """
        super().__init__()

        if mode not in {"bimodal", "speech", "motion"}:
            raise ValueError(f"Unsupported mode '{mode}'.")
        self.mode = mode
        self.pe_mode = pe_mode
        if self.pe_mode not in {"modality_agnostic", "modality_aware", "none"}:
            raise ValueError(f"Unsupported positional encoding mode '{self.pe_mode}'.")
        self.frame_rate_factor = frame_rate_factor
        self.mask_text_p = mask_text_p
        self.n_quant_speech = n_quant_speech
        self.n_quant_motion = n_quant_motion
        self.n_codebook_motion = n_codebook_motion
        self.n_codebook_speech = n_codebook_speech
        self.n_special_token_in = n_special_token_in
        self.ignore_index = ignore_index

        self.attentive_rnn = attentive_rnn
        self.txt_encoder = txt_encoder

        self.n_txt_vocab = n_txt_vocab  
        self.txt_embed = nn.Embedding(
            n_txt_vocab,
            d_model,
            padding_idx=0,
        )

        self.rvq_embed_speech = MultiEmbedding(
            n_quant_speech,
            n_codebook_speech + n_special_token_in,
            d_model,
            padding_idx=0,
        )
        self.rvq_embed_motion = MultiEmbedding(
            n_quant_motion,
            n_codebook_motion + n_special_token_in,
            d_model,
            padding_idx=0,
        )

        self.logits_head_speech = logits_head_speech
        self.logits_head_motion = logits_head_motion


    def _mask_text(self, x: Tensor) -> Tensor:
        """Randomly replace *entire* text sequences with a `<mask>` token
        (id = ``self.n_txt_vocab - 1``) with probability ``mask_text_p``."""
        if self.mask_text_p == 0.0:
            return x

        device = x.device
        drop_batch: Tensor = torch.rand(x.size(0), device=device) < self.mask_text_p
        if drop_batch.any():
            x[drop_batch] = self.n_txt_vocab - 1  # inplace masked token
        return x
        
    @staticmethod
    def _apply_pe(seq: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if seq is None:
            return None

        b, n, d = seq.shape
        device = seq.device
        position = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)     
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d)
        )                                                                                     
        pe = torch.zeros(n, d, dtype=torch.float32, device=device)                           
        pe[:, 0::2] = torch.sin(position * div_term)                                          
        pe[:, 1::2] = torch.cos(position * div_term)                                          

        return seq + pe.unsqueeze(0).expand(b, -1, -1)   

    @staticmethod
    def _embed_sequence(seq: Optional[Tensor], embed: nn.Module) -> Optional[Tensor]:
        """Quantised token sequence (b n q) ➔ summed codebook embeddings (b n d)."""
        if seq is None:
            return None
        embd = embed(rearrange(seq, "b n q -> q b n"))
        q, b, n, d = embd.shape
        return reduce(embd, "q b n d -> b n d", "sum", q=q)

    def _interleave_embeddings(
        self,
        y_embd_speech: Optional[Tensor],
        y_embd_motion: Optional[Tensor],
        *,
        batch_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Interleave speech and motion embeddings into a single sequence.

        Parameters
        ----------
        y_embd_speech
            Speech RVQ embeddings of shape ``(b, n_speech, d)`` *or* ``None``.
        y_embd_motion
            Motion RVQ embeddings of shape ``(b, n_motion, d)`` *or* ``None``.
        batch_size
            Batch size (used to allocate the interleaved buffer).

        Returns
        -------
        y_embd
            Interleaved embedding sequence suitable for the decoder input.
        motion_idx
            Boolean index selecting motion-time-steps in ``y_embd``.
        speech_idx
            Boolean index selecting speech-time-steps in ``y_embd``.
        """
        if self.mode == "bimodal":
            assert y_embd_speech.size(1) - 2 == (self.frame_rate_factor)*y_embd_motion.size(1), \
                f"Interleaving requires that the number of speech tokens is equal to (frame_rate_factor) * number \
                of motion tokens, but got {y_embd_speech.size(1)} and {y_embd_motion.size(1)} and fr factor {self.frame_rate_factor} \
                (motion_len * factor = {(self.frame_rate_factor+1)*y_embd_motion.size(1)})."

            # Construct boolean masks indicating modality positions
            motion_idx = torch.zeros(
                (y_embd_speech.shape[1] + y_embd_motion.shape[1]), dtype=torch.bool, device=y_embd_speech.device
            )
            motion_idx[self.frame_rate_factor + 1 :: self.frame_rate_factor + 1] = True
            speech_idx = ~motion_idx

            # Interleave embeddings according to masks
            y_embd = torch.empty(                       
                (batch_size, motion_idx.size(0), y_embd_motion.shape[-1]), device=y_embd_motion.device,
            )

            if self.pe_mode == 'modality_aware':
                y_embd_motion = self._apply_pe(y_embd_motion)
                y_embd_speech = self._apply_pe(y_embd_speech)

            y_embd[:, motion_idx] = y_embd_motion
            y_embd[:, speech_idx] = y_embd_speech
        elif self.mode == "speech":
            y_embd = y_embd_speech
            speech_idx = torch.ones(y_embd.shape[1], dtype=torch.bool, device=y_embd.device)
            motion_idx = ~speech_idx
        else:  # "motion"
            y_embd = y_embd_motion
            motion_idx = torch.ones(y_embd.shape[1], dtype=torch.bool, device=y_embd.device)
            speech_idx = ~motion_idx

        if self.pe_mode == 'modality_agnostic':
            y_embd = self._apply_pe(y_embd)

        return y_embd, motion_idx, speech_idx

    def _pred_modality_logits(
        self,
        y_hat: Tensor,
        motion_idx: Tensor,
        speech_idx: Tensor,
        motion_tokens: Optional[Tensor] = None,
        speech_tokens: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Tensor, Tensor]:
        """Apply modality-specific projection heads to decoder states.

        Parameters
        ----------
        y_hat
            Decoder hidden states (b, L-1, d) excluding the final step.
        motion_idx
            Boolean index selecting motion-time-steps (length ``L``).
        speech_idx
            Boolean index selecting speech-time-steps (length ``L``).

        Returns
        -------
        logits_speech
            Logits over speech code-books, or ``None`` if speech is disabled.
        logits_motion
            Logits over motion code-books, or ``None`` if motion is disabled.
        motion_head_idx
            Boolean index that aligns ``y_hat`` with ``logits_motion``.
        speech_head_idx
            Boolean index that aligns ``y_hat`` with ``logits_speech``.
        """
        logits_speech = logits_motion = None

        if self.mode == "bimodal":
            # Shift indices so that prediction at step *t* corresponds to code *t+1*
            motion_head_idx = torch.cat((motion_idx[1:], motion_idx[:1]))[:-1]
            speech_head_idx = ~motion_head_idx

            logits_motion = self.logits_head_motion(y_hat[:, motion_head_idx], target=motion_tokens)

            if speech_tokens is not None:
                speech_tokens = speech_tokens[:, 1:]
            logits_speech = self.logits_head_speech(y_hat[:, speech_head_idx], target=speech_tokens)

        elif self.mode == "motion":
            motion_head_idx = torch.ones(y_hat.shape[1], dtype=torch.bool, device=y_hat.device)
            speech_head_idx = ~motion_head_idx
            if motion_tokens is not None:
                motion_tokens = motion_tokens[:, 1:]  # skip SOS token
            logits_motion = self.logits_head_motion(y_hat)
        else:  # "speech"
            speech_head_idx = torch.ones(y_hat.shape[1], dtype=torch.bool, device=y_hat.device)
            motion_head_idx = ~speech_head_idx
            logits_speech = self.logits_head_speech(y_hat)

        return logits_speech, logits_motion, motion_head_idx, speech_head_idx

    def _mask_and_reshape(
        self,
        logits: Optional[Tensor],
        targets: Optional[Tensor],
        mask: Optional[Tensor],
        *,
        slice_idx: int = 0,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Apply an optional boolean mask and flatten logits / targets.

        Parameters
        ----------
        logits
            Logits tensor of shape ``(b, n, q, l)`` *or* ``None``.
        targets
            Target indices of shape ``(b, n, q)`` *or* ``None``.
        mask
            Boolean mask selecting a subset of time-steps (`True` = keep).
        slice_idx
            Starting offset along the temporal dimension (for SOS token).

        Returns
        -------
        masked_logits
            Masked (or original) logits with leading ``b`` removed if masked.
        masked_targets
            Masked (or original) integer targets.
        flat_logits
            Logits reshaped to ``(b·n·q, l)`` suitable for CE loss.
        flat_targets
            Targets reshaped to ``(b·n·q)`` suitable for CE loss.
        """
        if logits is None:
            return None, None, None, None

        if mask is not None:
            masked_logits = logits[mask[:,slice_idx:], :, :]
            masked_target = targets[:,slice_idx:][mask[:,slice_idx:], :]
            flat_logits = rearrange(masked_logits, "n q l -> (n q) l")
            flat_target = rearrange(masked_target, "n q   -> (n q)")
        else:
            masked_logits = logits
            masked_target = targets[:, slice_idx:]
            flat_logits = rearrange(masked_logits, "b n q l -> (b n q) l")
            flat_target = rearrange(masked_target, "b n q   -> (b n q)")

        return masked_logits, masked_target, flat_logits, flat_target

    def _get_masked_logits(
        self,
        logits_speech: Optional[Tensor],
        y_speech: Optional[Tensor],
        logits_mask_speech: Optional[Tensor],
        logits_motion: Optional[Tensor],
        y_motion: Optional[Tensor],
        logits_mask_motion: Optional[Tensor],
    ):
        """Wrapper around `_mask_and_reshape` for both modalities."""
        if self.mode == "bimodal":
            # In bimodal mode, speech sequence has head and tail tokens but motion does not.
            ml_s, mt_s, fl_s, ft_s = self._mask_and_reshape(
                logits_speech, y_speech, logits_mask_speech, slice_idx=1
            )
            ml_m, mt_m, fl_m, ft_m = self._mask_and_reshape(
                logits_motion, y_motion, logits_mask_motion
            )
        else:
            ml_s, mt_s, fl_s, ft_s = self._mask_and_reshape(
                logits_speech, y_speech, logits_mask_speech, slice_idx=1
            )
            ml_m, mt_m, fl_m, ft_m = self._mask_and_reshape(
                logits_motion, y_motion, logits_mask_motion, slice_idx=1
            )

        return ml_s, mt_s, fl_s, ft_s, ml_m, mt_m, fl_m, ft_m


    def _compute_losses(
        self,
        flat_logits_speech: Optional[Tensor],
        flat_target_speech: Optional[Tensor],
        flat_logits_motion: Optional[Tensor],
        flat_target_motion: Optional[Tensor],
        reduction: str = "mean",
    ) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        if flat_logits_speech is not None:
            losses["loss_speech"] = F.cross_entropy(
                flat_logits_speech,
                flat_target_speech,
                ignore_index=self.ignore_index,
                reduction=reduction,
            )
        if flat_logits_motion is not None:
            has_valid_motion = (flat_target_motion != self.ignore_index).any()
            if has_valid_motion:
                losses["loss_motion"] = F.cross_entropy(
                    flat_logits_motion,
                    flat_target_motion,
                    ignore_index=self.ignore_index,
                    reduction=reduction,
                )

        return losses

    def forward(
        self,
        x: Tensor,
        y_speech: Optional[Tensor],
        y_motion: Optional[Tensor],
        encoder_mask: Tensor,
        crossatt_mask: Tensor,
        *,
        logits_mask_speech: Optional[Tensor] = None,
        logits_mask_motion: Optional[Tensor] = None,
        attention_only: bool = False,
        forced_attention: Optional[Tensor] = None,
        init_state: Optional[Tuple[Tensor, Tensor]] = None,
        return_latents: bool = False,
        reduction: str = "mean",
    ):
        """
        Parameters
        ----------
        x : (b, t_txt)
            Text token ids.
        y_speech : (b, n, q_speech) or ``None``
        y_motion : (b, m, q_motion) or ``None``
        encoder_mask : (b, t_txt)
        crossatt_mask : (b, L) where L = n + m
        forced_attention : optional pre-computed weights
        """
        x = self._mask_text(x)
        x_embd = self.txt_embed(x)
        x_enc = self.txt_encoder(x_embd, mask=encoder_mask)

        y_embd_speech = self._embed_sequence(y_speech, self.rvq_embed_speech)
        y_embd_motion = self._embed_sequence(y_motion, self.rvq_embed_motion)

        y_embd, motion_idx, speech_idx = self._interleave_embeddings(
            y_embd_speech, y_embd_motion, batch_size=x.size(0)
        )

        y_hat, att = self.attentive_rnn(
            y_embd[:, :-1, :],
            x_enc,
            mask=crossatt_mask[:, :-1],
            forced_attention=(
                forced_attention[:, :, : y_embd.size(1) - 1]  # align lengths
                if forced_attention is not None
                else None
            ),
            attention_only=attention_only,
            init_state=init_state,
        )
        if attention_only:
            return att

        # 4. Modality-specific projection heads ---------------------------- #
        logits_speech, logits_motion, motion_head_idx, speech_head_idx = self._pred_modality_logits(
            y_hat, motion_idx, speech_idx, motion_tokens=y_motion, speech_tokens=y_speech
        )

        (
            masked_logits_speech,
            masked_target_speech,
            flat_logits_speech,
            flat_target_speech,
            masked_logits_motion,
            masked_target_motion,
            flat_logits_motion,
            flat_target_motion,
        ) = self._get_masked_logits(
            logits_speech,
            y_speech,
            logits_mask_speech,
            logits_motion,
            y_motion,
            logits_mask_motion,
        )

      
        losses = self._compute_losses(
            flat_logits_speech,
            flat_target_speech,
            flat_logits_motion,
            flat_target_motion,
            reduction=reduction,
        )

        if return_latents:
            return (
                y_hat[:, motion_head_idx],
                y_hat[:, speech_head_idx],
                logits_speech,
                logits_motion,
                losses,
                att,
                masked_logits_speech,
                masked_target_speech,
                masked_logits_motion,
                masked_target_motion,
            )

        return (
            logits_speech,
            logits_motion,
            losses,
            att,
        )

    def _prepare_synthesis(self, x, batch_size, prompt_speech_tokens, prompt_motion_tokens, device):
        x = repeat(x, "n -> b n", b=batch_size).to(device) # extend text
        x_embd = self.txt_embed(x)
        x_enc = self.txt_encoder(x_embd)

        stop_token = torch.ones(self.n_quant_speech, 1, 1, device=device) * 2
        all_stop_token = torch.zeros(batch_size, 1, device=device).bool()

        y_start = torch.ones(self.n_quant_speech, batch_size, 1, device=device).long()
        y_embd = reduce(self.rvq_embed_speech(y_start), "q b n d -> b n d", "sum")

        prompt = None
        p_len = 0
        if prompt_speech_tokens is not None or prompt_motion_tokens is not None:
            if prompt_speech_tokens.shape[1] != batch_size:
                prompt_speech_tokens = repeat(prompt_speech_tokens, "q 1 n -> q b n", b=batch_size)
                prompt_motion_tokens = repeat(prompt_motion_tokens, "q 1 n -> q b n", b=batch_size)

            prompt_speech_tokens = prompt_speech_tokens.transpose(0,1).transpose(1,2) # b n q
            prompt_motion_tokens = prompt_motion_tokens.transpose(0,1).transpose(1,2) # b n q

            prompt_speech = self._embed_sequence(seq=prompt_speech_tokens, embed=self.rvq_embed_speech)
            prompt_motion = self._embed_sequence(seq=prompt_motion_tokens, embed=self.rvq_embed_motion)

            p_len = prompt_speech.shape[1] + prompt_motion.shape[1]
            prompt, *_ = self._interleave_embeddings(y_embd_speech=prompt_speech, y_embd_motion=prompt_motion, batch_size=batch_size)

            assert p_len == prompt.shape[1], f'p_len: {p_len}, prompt {prompt.shape}'
    
        return prompt, p_len, y_embd, x_enc, stop_token, all_stop_token

    @staticmethod
    def _sample_logits(y_embd, logit_head, first_greedy_quant, k, temp: list = []):
        logits = logit_head(y_embd)
        logits = rearrange(logits, "b 1 q l -> q b l")
        q_sampled = []
        for i, q in enumerate(logits):
            q_sampled.append(
                topk_sampling(q, k=k, temp=temp[i])
                if i < first_greedy_quant
                else topk_sampling(q, k=1)
            )
        return logits, torch.stack(q_sampled)

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
        p=0.95,
        init_state: Optional[dict] = None,
        mask_motion_tokens: bool | str = False,
    ):

        prompt, p_len, y_embd, x_enc, stop_token, all_stop_token = \
            self._prepare_synthesis(x, batch_size, prompt_speech_tokens, prompt_motion_tokens, device)
       
        # Initialize state ---------------------------------------------------------------|
        state = init_state
        if state is None:
            state = self.attentive_rnn.init_state(max_seqlen=max_seqlen, batch_size=batch_size)

        qs, qs_speech, qs_motion, atts, stop_tokens = [], [], [], [], []    
        logits_motion = []
        logits_speech = []
        y_embd_motion_list = []
        y_embd_speech_list = []

        for t in tqdm(range(max_seqlen+p_len), leave=False):

            # AR backbone step -----------------------------------------------------------|
            y_embd, att, state = self.attentive_rnn.step(y_embd, x_enc, t, state)
            atts.append(att)

            # Token sampling -------------------------------------------------------------|
            if ((t-self.frame_rate_factor)%(self.frame_rate_factor+1))==0: # if motion step
                y_embd_motion_list.append(y_embd.squeeze(1))
                logits, q_sampled = self._sample_logits(y_embd, self.logits_head_motion, first_greedy_quant_motion, k=k_motion, temp=temp_motion)
                logits_motion.append(logits)
                if mask_motion_tokens == 'zero':
                    q_sampled = torch.zeros((self.n_quant_motion, batch_size, 1), dtype=torch.int32).to(device)
                elif mask_motion_tokens == 'random':
                    q_sampled = torch.randint(self.n_special_token_in, self.n_codebook_motion+self.n_special_token_in, (self.n_quant_motion, batch_size, 1)).to(device)
                if t >= p_len:
                    qs_motion.append(q_sampled)    

            else: # if speech step
                y_embd_speech_list.append(y_embd.squeeze(1))
                logits, q_sampled = self._sample_logits(y_embd, self.logits_head_speech, first_greedy_quant_speech, k=k_speech, temp=temp_speech)
                logits_speech.append(logits)
                if t >= p_len:
                    qs_speech.append(q_sampled)    
                
                qs.append(q_sampled)
                # Check if stop token --------------------------------------------------------|
                is_stop_token = (q_sampled == stop_token).prod(dim=0)
                stop_tokens.append(is_stop_token)
                all_stop_token.logical_or_(is_stop_token)
                if all_stop_token.prod() and t>p_len:
                    print("Stopped at step", t)
                    break

            # Embed next input -----------------------------------------------------------|
            if prompt is not None and t < p_len:
                y_embd = prompt[:,[t]]
            else:
                if ((t-self.frame_rate_factor)%(self.frame_rate_factor+1))==0:
                    y_embd = self.rvq_embed_motion(q_sampled)
                else:
                    y_embd = self.rvq_embed_speech(q_sampled)
                y_embd = reduce(y_embd, "q b n d -> b n d", "sum")
  
        if len(qs_motion) == 0:
            return None, None, None, None, None, None, None, None, None, None, 

        atts =  torch.cat(atts, dim=2) if atts[0] is not None else None

        qs_speech = torch.stack(qs_speech, dim=2).squeeze(-1)
        qs_motion = torch.stack(qs_motion, dim=2).squeeze(-1)
        speech_rvq = (qs_speech - self.n_special_token_in).clamp_min(0)
        motion_rvq = (qs_motion - self.n_special_token_in).clamp_min(0) 

        y_embd_motion_list = torch.stack(y_embd_motion_list).transpose(0,1)

        stop_tokens.append(torch.ones(batch_size, 1, device=device))
        stop_tokens = torch.stack(stop_tokens, dim=1).squeeze(-1)
        stop_idx = (stop_tokens * rearrange(torch.arange(stop_tokens.shape[1], device=stop_tokens.device), "n -> 1 n")).long()

        speech_cuts = []
        motion_cuts = []
        motion_latent_cuts = []
        for i in range(stop_idx.shape[0]):
            idx = torch.unique(stop_idx[i])[1] - p_len
            idx_motion = idx//(self.frame_rate_factor+1) + 10 # little margin
            idx_speech = idx + 150 - idx_motion
            if atts is not None:
                speech_cuts.append((speech_rvq[:,[i], :idx_speech], atts[i, :, :idx]))
                motion_cuts.append((motion_rvq[:,[i], :idx_motion], atts[i, :, :idx]))
            else:
                speech_cuts.append((speech_rvq[:,[i], :idx_speech],))
                motion_cuts.append((motion_rvq[:,[i], :idx_motion],))
            motion_latent_cuts.append(y_embd_motion_list[i][:idx_motion])
                
        return qs, qs_speech, qs_motion, atts, stop_tokens, speech_cuts, motion_cuts, torch.stack(logits_motion), motion_latent_cuts, torch.stack(y_embd_speech_list).transpose(0,1)
        

    @torch.inference_mode()
    def generate_speech2ges(
        self,
        x: Tensor,
        batch_size: int=3,
        prompt_speech_tokens: Optional[Tensor] = None,
        prompt_motion_tokens: Optional[Tensor] = None, # q, b, n
        input_speech_tokens: Optional[Tensor] = None, # q, b, n
        device: str = "cpu",
        max_seqlen: int = 1000,
        k_motion: int = 100,
        first_greedy_quant_motion: int = 1,
        temp_motion: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        init_state: Optional[dict] = None,
        mask_motion_tokens: bool | str = False,
    ):

        prompt, p_len, y_embd, x_enc, stop_token, all_stop_token = \
            self._prepare_synthesis(x, batch_size, prompt_speech_tokens, prompt_motion_tokens, device)
       
        # Initialize state ---------------------------------------------------------------|
        state = init_state
        if state is None:
            state = self.attentive_rnn.init_state(max_seqlen=max_seqlen, batch_size=batch_size)

        qs, qs_speech, qs_motion, atts, stop_tokens = [], [], [], [], []    
        logits_motion = []
        y_embd_motion_list = []
        speech_idx = 0
        input_speech_tokens = input_speech_tokens.squeeze(1)

        for t in tqdm(range(max_seqlen+p_len), leave=False):

            # AR backbone step -----------------------------------------------------------|
            y_embd, att, state = self.attentive_rnn.step(y_embd, x_enc, t, state)
            atts.append(att)

            # Token sampling -------------------------------------------------------------|
            if ((t-self.frame_rate_factor)%(self.frame_rate_factor+1))==0: # if motion step
                y_embd_motion_list.append(y_embd.squeeze(1))
                logits, q_sampled = self._sample_logits(y_embd, self.logits_head_motion, first_greedy_quant_motion, k=k_motion, temp=temp_motion)
                logits_motion.append(logits)
                if mask_motion_tokens == 'zero':
                    q_sampled = torch.zeros((self.n_quant_motion, batch_size, 1), dtype=torch.int32).to(device)
                elif mask_motion_tokens == 'random':
                    q_sampled = torch.randint(self.n_special_token_in, self.n_codebook_motion+self.n_special_token_in, (self.n_quant_motion, batch_size, 1)).to(device)
                if t >= p_len:
                    qs_motion.append(q_sampled)    

            else: # if speech step
                if t >= p_len:
                    qs_speech.append(input_speech_tokens[:,speech_idx].unsqueeze(1))
                    speech_idx+=1

                    
                # Check if stop token --------------------------------------------------------|
                if speech_idx == input_speech_tokens.shape[1]:
                    is_stop_token = torch.ones((batch_size,1), device=device)
                else:
                    is_stop_token = torch.zeros((batch_size,1), device=device)

                stop_tokens.append(is_stop_token)
                all_stop_token.logical_or_(is_stop_token)
                if all_stop_token.prod():
                    print("Stopped at step", t)
                    break

            # Embed next input -----------------------------------------------------------|
            if prompt is not None and t < p_len:
                y_embd = prompt[:,[t]]
            else:
                if ((t-self.frame_rate_factor)%(self.frame_rate_factor+1))==0:
                    y_embd = self.rvq_embed_motion(q_sampled)
                else:
                    y_embd = self.rvq_embed_speech(q_sampled)
                y_embd = reduce(y_embd, "q b n d -> b n d", "sum")

            

        atts =  torch.cat(atts, dim=2) if atts[0] is not None else None

        qs_speech = torch.stack(qs_speech, dim=2).squeeze(-1)
        qs_motion = torch.stack(qs_motion, dim=2).squeeze(-1)
        speech_rvq = (qs_speech - self.n_special_token_in).clamp_min(0)
        motion_rvq = (qs_motion - self.n_special_token_in).clamp_min(0) 

        y_embd_motion_list = torch.stack(y_embd_motion_list).transpose(0,1)

        stop_tokens.append(torch.ones(batch_size, 1, device=device))
        stop_tokens = torch.stack(stop_tokens, dim=1).squeeze(-1)
        stop_idx = (stop_tokens * rearrange(torch.arange(stop_tokens.shape[1], device=stop_tokens.device), "n -> 1 n")).long()

        speech_cuts = []
        motion_cuts = []
        motion_latent_cuts = []
        for i in range(stop_idx.shape[0]):
            idx = torch.unique(stop_idx[i])[1] - p_len
            idx_motion = idx//(self.frame_rate_factor+1) + 10 # little margin
            idx_speech = idx + 150 - idx_motion
            if atts is not None:
                speech_cuts.append((speech_rvq[:,[i], :idx_speech], atts[i, :, :idx]))
                motion_cuts.append((motion_rvq[:,[i], :idx_motion], atts[i, :, :idx]))
            else:
                speech_cuts.append((speech_rvq[:,[i], :idx_speech],))
                motion_cuts.append((motion_rvq[:,[i], :idx_motion],))
            motion_latent_cuts.append(y_embd_motion_list[i][:idx_motion])
                
        return qs, qs_speech, qs_motion, atts, stop_tokens, speech_cuts, motion_cuts, torch.stack(logits_motion), motion_latent_cuts, None
        

    @torch.inference_mode()
    def fast_generate_multimodal_batch(
        self,
        x: Tensor,
        batch_size: int = 3,
        prompt_speech_tokens: Optional[Tensor] = None,
        prompt_motion_tokens: Optional[Tensor] = None,  # q, b, n
        device: str = "cpu",
        max_seqlen: int = 1000,
        k_speech: int = 100,
        k_motion: int = 100,
        first_greedy_quant_motion: int = 1,
        first_greedy_quant_speech: int = 1,
        temp_speech: list = [1.0],
        temp_motion: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        p: float = 0.95,
        init_state: Optional[dict] = None,
        mask_motion_tokens: bool | str = False,
    ):
        """
        Optimized version:
        - Precomputes interleave schedule and preallocates outputs
        - Reduces per-step Python and allocation overhead
        - Uses AMP autocast for backbone step
        """
        
        logits_head_motion = self.logits_head_motion
        logits_head_speech = self.logits_head_speech
        rvq_embed_motion = self.rvq_embed_motion
        rvq_embed_speech = self.rvq_embed_speech
        attentive_rnn = self.attentive_rnn
        frame_rate_factor = self.frame_rate_factor  # e.g., 14 if speech:motion = 15:1
        n_q_motion = self.n_quant_motion
        n_q_speech = self.n_quant_speech if hasattr(self, "n_quant_speech") else 1
        n_special_in = self.n_special_token_in
        n_codebook_motion = self.n_codebook_motion

        prompt, p_len, y_embd, x_enc, stop_token, all_stop_token = self._prepare_synthesis(
            x, batch_size, prompt_speech_tokens, prompt_motion_tokens, device
        )

        state = init_state or attentive_rnn.init_state(max_seqlen=max_seqlen, batch_size=batch_size)

        # --- Build schedule (True = motion step, False = speech step)
        total_T = max_seqlen + p_len
        # pattern: after every 'frame_rate_factor' speech steps comes 1 motion step
        # We use a rolling counter to avoid expensive modulo in the loop.
        schedule = torch.empty(total_T, dtype=torch.bool, device=device)
        ctr = 0
        for t in range(total_T):
            if ctr == frame_rate_factor:  # motion step
                schedule[t] = True
                ctr = 0
            else:  # speech step
                schedule[t] = False
                ctr += 1
 
        num_motion_steps = int(schedule.sum().item())
        num_speech_steps = total_T - num_motion_steps

        logits_motion_list = []
        logits_speech_list = []

        qs_motion = torch.empty((n_q_motion, batch_size, num_motion_steps), dtype=torch.int32, device=device)
        qs_speech = torch.empty((n_q_speech, batch_size, num_speech_steps), dtype=torch.int32, device=device)

        y_embd_motion = torch.empty((batch_size, num_motion_steps, y_embd.shape[-1]), dtype=y_embd.dtype, device=device)
        y_embd_speech = torch.empty((batch_size, num_speech_steps, y_embd.shape[-1]), dtype=y_embd.dtype, device=device)


        atts_list = []

        stop_tokens = torch.zeros((batch_size, total_T), dtype=torch.bool, device=device)

    
        if mask_motion_tokens == "zero":
            masked_motion_tokens = torch.zeros((n_q_motion, batch_size, 1), dtype=torch.int32, device=device)
        elif mask_motion_tokens == "random":
            masked_motion_tokens = torch.randint(
                n_special_in, n_codebook_motion + n_special_in, (n_q_motion, batch_size, 1), device=device, dtype=torch.int32
            )
        else:
            masked_motion_tokens = None 

        m_idx = 0
        s_idx = 0

        import contextlib
        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=torch.bfloat16)
            if (device == "cuda") else contextlib.nullcontext()
        )
        self.attentive_rnn.to_mode("fused_recurrent")  # fast per-token kernel
        kv_cache = self.attentive_rnn.cross_att.precompute_kv(k_input=x_enc, v_input=None)

        with autocast_ctx:
            for t in range(total_T):
                # Backbone step
                y_embd, _, state = attentive_rnn.step(y_embd, x_enc, t, state, kv_cache)

                # Motion or speech branch
                if schedule[t]:  # MOTION STEP
                    y_embd_motion[:, m_idx, :] = y_embd.squeeze(1)
                    logits, q_sampled = self._sample_logits(
                        y_embd, logits_head_motion, first_greedy_quant_motion, k=k_motion, temp=temp_motion
                    )
                    logits_motion_list.append(logits)

                    if masked_motion_tokens is not None:
                        q_sampled = masked_motion_tokens

                    if t >= p_len:
                        qs_motion[:, :, m_idx] = q_sampled.squeeze(-1)
                    if prompt is not None and t < p_len:
                        y_embd = prompt[:, [t]]
                    else:
                        emb = rvq_embed_motion(q_sampled)
                        y_embd = emb.sum(dim=0)  # reduce "q b n d -> b n d"

                    m_idx += 1

                else:  # SPEECH STEP
                    y_embd_speech[:, s_idx, :] = y_embd.squeeze(1)
                    logits, q_sampled = self._sample_logits(
                        y_embd, logits_head_speech, first_greedy_quant_speech, k=k_speech, temp=temp_speech
                    )
                    logits_speech_list.append(logits)

                    if t >= p_len:
                        qs_speech[:, :, s_idx] = q_sampled.squeeze(-1)

                    is_stop = (q_sampled == stop_token).prod(dim=0)  # (b,1) -> (b,1) bool
                    stop_tokens[:, t] = is_stop.squeeze(-1)
                    all_stop_token.logical_or_(is_stop)
                    if bool(all_stop_token.prod()) and t > p_len:
                        total_T = t + 1
                        break

                    if prompt is not None and t < p_len:
                        y_embd = prompt[:, [t]]
                    else:
                        emb = rvq_embed_speech(q_sampled)
                        y_embd = emb.sum(dim=0)

                    s_idx += 1

        # --- Handle case where no motion was produced
        if m_idx == 0:
            return (None, None, None, None, None, None, None, None, None, None)


        qs_motion = qs_motion[:, :, :m_idx]
        qs_speech = qs_speech[:, :, :s_idx]
        y_embd_motion = y_embd_motion[:, :m_idx, :]
        y_embd_speech = y_embd_speech[:, :s_idx, :]
        atts = torch.cat(atts_list, dim=2) if (len(atts_list) > 0 and atts_list[0] is not None) else None

        # Convert to rvq ids (clamp specials away)
        speech_rvq = (qs_speech - n_special_in).clamp_min(0)
        motion_rvq = (qs_motion - n_special_in).clamp_min(0)

        if stop_tokens.shape[1] < total_T:
            pad = torch.zeros((batch_size, total_T - stop_tokens.shape[1]), dtype=torch.bool, device=device)
            stop_tokens = torch.cat([stop_tokens, pad], dim=1)
        tail = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        st_all = torch.cat([stop_tokens[:, :total_T], tail], dim=1)
        first_true = torch.argmax(st_all.int(), dim=1)  # (b,)
        idx = (first_true - p_len).clamp_min(0)
        idx_motion = (idx // (frame_rate_factor + 1)) + 10  # margin
        idx_speech = idx + 150 - idx_motion


        speech_cuts, motion_cuts, motion_latent_cuts = [], [], []
        for i in range(batch_size):
            i_s = int(idx_speech[i].item())
            i_m = int(idx_motion[i].item())
            i_t = int(first_true[i].item())

            speech_cuts.append((speech_rvq[:, [i], :i_s],))
            motion_cuts.append((motion_rvq[:, [i], :i_m],))

            motion_latent_cuts.append(y_embd_motion[i, :i_m, :])


        logits_motion = torch.stack(logits_motion_list) if len(logits_motion_list) else None
        logits_speech = torch.stack(logits_speech_list) if len(logits_speech_list) else None

        qs_placeholder = None

        return (
            qs_placeholder,
            qs_speech,                        # [n_q_speech, B, T_s]
            qs_motion,                        # [n_q_motion, B, T_m]
            None,                             # or None
            stop_tokens[:, :total_T],         # [B, T]
            speech_cuts,                      # list per batch
            motion_cuts,                      # list per batch
            logits_motion if logits_motion is not None else torch.empty(0),
            motion_latent_cuts,               # list per batch (embeddings)
            y_embd_speech,                    # [B, T_s, D]
        )