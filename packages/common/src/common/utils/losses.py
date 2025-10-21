


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class GeodesicLossEmage(nn.Module):
    '''
        Copyright (c) HuaWei, Inc. and its affiliates.
        liu.haiyang@huawei.com
    '''
    def __init__(self):
        super(GeodesicLossEmage, self).__init__()

    def compute_geodesic_distance(self, m1, m2):
        """ Compute the geodesic distance between two rotation matrices.

        Args:
            m1, m2: Two rotation matrices with the shape (batch x 3 x 3).

        Returns:
            The minimal angular difference between two rotation matrices in radian form [0, pi].
        """
        m1 = m1.reshape(-1, 3, 3)
        m2 = m2.reshape(-1, 3, 3)
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.clamp(cos, min=-1 + 1E-6, max=1-1E-6)

        theta = torch.acos(cos)

        return theta

    def __call__(self, m1, m2, reduction='mean'):
        loss = self.compute_geodesic_distance(m1, m2)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            raise RuntimeError(f'unsupported reduction: {reduction}')

# -------------- New version, can support bf16 mixed precision training --------------

# ---------- utilities from the block you pasted -----------------------------

def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    mask = x > 0
    ret[mask] = torch.sqrt(x[mask])
    return ret

def matrix_to_quaternion(M):
    m00, m11, m22 = M[..., 0, 0], M[..., 1, 1], M[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x  = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y  = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z  = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, M[..., 2, 1] - M[..., 1, 2])
    o2 = _copysign(y, M[..., 0, 2] - M[..., 2, 0])
    o3 = _copysign(z, M[..., 1, 0] - M[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

def standardize_quaternion(q):
    return torch.where(q[..., :1] < 0, -q, q)

def quaternion_raw_multiply(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_invert(q):
    return q * q.new_tensor([1, -1, -1, -1])

# ---------- NaN / Inf checker ------------------------------------------------

def check_nan(x, tag):
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise FloatingPointError(f"{tag} contains NaN/Inf")

# ---------- bf16-friendly geodesic loss -------------------------------------

class GeodesicLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def compute_geodesic_distance(self, R1: torch.Tensor, R2: torch.Tensor):
        R1 = R1.reshape(-1, 3, 3)
        R2 = R2.reshape(-1, 3, 3)

        q1 = standardize_quaternion(matrix_to_quaternion(R1))  # (...,4)
        q2 = standardize_quaternion(matrix_to_quaternion(R2))

        check_nan(q1, "q1");  check_nan(q2, "q2")

        q_rel = quaternion_raw_multiply(q1, quaternion_invert(q2))
        w = torch.clamp(q_rel[..., 0], -1 + self.eps, 1 - self.eps)

        theta = 2.0 * torch.acos(w)
        check_nan(theta, "theta")

        return theta  # (...,)

    def forward(self, R1, R2, reduction="mean", dtype=torch.bfloat16):
        # keep caller’s dtype (bf16) except for the acos argument
        theta = self.compute_geodesic_distance(R1.to(torch.float32),
                                               R2.to(torch.float32)).to(dtype)
        if reduction == "mean":
            return theta.mean()
        if reduction == "none":
            return theta
        raise RuntimeError(f"unsupported reduction: {reduction}")