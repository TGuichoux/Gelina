
from einops.layers.torch import EinMix
import torch 
from torch import nn
from typing import List, Optional, Tuple


class SimpleLogitHead(nn.Module):
    def __init__(self, n_quant: int, n_codebook: int, n_special_token_out: int, d_model: int):
        super().__init__()
        self.n_quant = n_quant
        self.n_codebook = n_codebook
        self.n_special_token_out = n_special_token_out
        self.d_model = d_model

        self.model = EinMix(
                "b n d -> b n q l",
                weight_shape="q l d",
                q=n_quant,
                d=d_model,
                l=n_codebook + n_special_token_out,
            )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        b,n,d = x.shape
        assert d == self.d_model, f"Expected input of shape (b, n, {self.d_model}), got {x.shape}"

        return self.model(x)

