# depth_transformer.py
from typing import Callable, Optional, List
import torch
from torch import nn
from gelina.components.logit_heads import SimpleLogitHead
from gelina.utils.tools import _diag_bool_mask, _topk_sampler


class DepthTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_codebook: int,
        n_special_token_out: int,
        n_quant: int = 6,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
        sampler: Optional[Callable] = _topk_sampler,
    ) -> None:
        super().__init__()

        self.k = n_quant  # number of residual levels
        self.v = n_codebook + n_special_token_out

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)
        self.res_emb = nn.Embedding(self.v, d_model)
        self.heads = nn.ModuleList(
            [
                SimpleLogitHead(
                    n_quant=1,
                    n_codebook=n_codebook,
                    n_special_token_out=n_special_token_out,
                    d_model=d_model,
                )
                for _ in range(n_quant)
            ]
        )

        self.sampler = sampler

    def forward(
        self,
        x: torch.Tensor,                                         # (b,n,d)
        target: Optional[torch.Tensor] = None,                  # (b,n,k) or None
        sampler: Optional[Callable] = None,
        topk: int = 8,
        temperature: float = 1.0,
    ) -> torch.Tensor:                                          # (b,n,k,v)

        if target is None and self.training:
            raise ValueError("Target must be provided during training. (teacher forcing, avoid sampling)")
        if not self.training:
            target = None  # disable target if not training
        b, n, _ = x.shape
        attn_mask = _diag_bool_mask(n, x.device)                # (n,n)

        logits_all = []
        h = x
        for i in range(self.k): # for each residual code
            h = self.enc(h, mask=attn_mask) # embed the sequence of embeddings (b,n,d)
            logit = self.heads[i](h).squeeze(2) # predict logits for this residual level (b,n,v)
            logits_all.append(logit)

            tok = target[:, :, i] if target is not None else self.sampler(logit, k=topk, t=temperature) # sample tokens (b,n) (or use target)
            h = h + self.res_emb(tok) # (b,n,d) embed sampled tokens and add to the input

        return torch.stack(logits_all, 2) # (b,n,k,v)


if __name__ == "__main__":
    b, n, d = 32, 227, 1024
    net = DepthTransformer(d_model=d, n_codebook=512, n_special_token_out=3, n_quant=6, ).cuda()
    inp = torch.randn(b, n, d, device="cuda")

    target = torch.randint(0, 512 + 3, (b, n, 6), device="cuda")  # (b,n,k) with k=6
    out = net(inp, target=target)
    print(out.shape)  # (4,10,6,257)
