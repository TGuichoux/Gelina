import torch
from typing import Callable, Optional

def _diag_bool_mask(n: int, device) -> torch.Tensor:          # (n, n)
    m = torch.ones(n, n, dtype=torch.bool, device=device)
    m.fill_diagonal_(False)
    return m


def _topk_sampler(logits: torch.Tensor, k: int, t: float) -> torch.Tensor:
    logits = logits / t
    v, idx = logits.topk(k, -1) # (b, n, k)
    p = torch.softmax(v, -1)
    s = torch.multinomial(p.view(-1, k), 1).view(*logits.shape[:2])
    return idx.gather(-1, s.unsqueeze(-1)).squeeze(-1)        # (b, n)