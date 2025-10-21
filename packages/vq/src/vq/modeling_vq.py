import random
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .components.encdec import Encoder, Decoder
from .components.residual_vq import ResidualVQ
from common.utils.losses import GeodesicLoss
import common.utils.rotation_conversions as rc
import numpy as np
import smplx


class RVQVAE_Smpl(nn.Module):
    def __init__(self, input_width=263, nb_code=1024, code_dim=512, output_emb_width=512,
                 down_t=3, stride_t=2, width=512, depth=3, dilation_growth_rate=3, activation='relu', norm=None, num_quantizers=6,
                 quantize_dropout_prob=0.5, quantize_dropout_cutoff_index=0, shared_codebook=False, mu=0.99):
        super().__init__()
        assert output_emb_width == code_dim

        self.code_dim = code_dim
        self.num_code = nb_code

        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

        self.quantizer = ResidualVQ(
            num_quantizers=num_quantizers,
            quantize_dropout_prob=quantize_dropout_prob,
            quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
            shared_codebook=shared_codebook,
            nb_code=nb_code,
            code_dim=code_dim,
            mu=mu,
        )

    def preprocess(self, x):
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        return x.permute(0, 2, 1)

    def encode(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        return code_idx, all_codes

    def forward(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_out = self.decoder(x_quantized)
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        x = x_d.sum(dim=0).permute(0, 2, 1)
        x_out = self.decoder(x)
        return x_out



