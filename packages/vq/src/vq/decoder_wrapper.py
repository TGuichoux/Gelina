from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple
from dotmap import DotMap

import numpy as np
import torch
from torch import Tensor, nn

from common.utils.joint_list import joints_list
from common.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rot6d_to_aa,
)
from common.utils.tools import resample_pose_seq
from common.utils.utils import save_smpl_file
from common.utils.utils import load_vq_checkpoint, load_config
from external.PantoMatrix.scripts.EMAGE_2024.models.motion_representation import VAEConvZero

from external.PantoMatrix.scripts.EMAGE_2024.utils.other_tools import (
    velocity2position, load_checkpoints
)

ori_joints = "beat_smplx_joints"

ori_joint_list = joints_list[ori_joints]
tar_joint_list_lower = joints_list["beat_smplx_lower"]
joint_mask_lower = np.zeros(len(list(ori_joint_list.keys())) * 3)
for joint_name in tar_joint_list_lower:
    joint_mask_lower[
        ori_joint_list[joint_name][1] - ori_joint_list[joint_name][0] :
        ori_joint_list[joint_name][1]
    ] = 1
    
class DecoderWrapper(nn.Module):
    """Decode RVQ tokens into SMPL‑X pose and optional global motion."""

    def __init__(
        self,
        vq_ckpt: str | os.PathLike,
        vq_config='configs/vq/model/default.yaml',
        global_vq_config='configs/others/global_vq.yml',
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        self.vq, self.vq_config = load_vq_checkpoint('vq.modeling_vq.RVQVAE_Smpl', vq_ckpt, vq_config, self.device)


        global_args =  DotMap(load_config(global_vq_config))
        global_args.vae_test_dim = 61
        global_args.vae_length = 256
        global_args.vae_layer = 4
        global_args.strides = [1, 1, 1, 1]
        self.global_motion_vq = VAEConvZero(global_args).to(self.device)
        load_checkpoints(self.global_motion_vq, 'checkpoints/global_vq/last_1700_foot.bin', global_args.e_name)
        self.global_motion_vq.eval()
    # --------------------------------------------------------------------- #

    def forward(self, motion, residuals=6):
        if len(motion.shape) == 2:
            motion = motion.unsqueeze(0)
        tokens, *_ = self.vq.encode(motion)

        out = self.decode(tokens[...,:residuals])
        return out

    def tokenize(self, motion):
        code_idx, _ = self.vq.encode(motion.unsqueeze(0).to(self.device))
        code_idx = code_idx.squeeze(0)
        return code_idx

    def resample_axis_angle(self,pose, in_fps=20, out_fps=30):
        if len(pose.shape) == 3:
            assert pose.shape[0] == 1, f"Batched input not suported, got {pose.shape}"
            pose = pose.squeeze(0)
        pose = resample_pose_seq(pose, in_fps, out_fps).to(self.device)
        trans = pose[:,330:333]
        pose = rot6d_to_aa(pose[:,:330])
        return pose, trans


    def forward_and_save(
        self,
        motion,
        out_dir: str | os.PathLike,
        name: str,
        betas: Optional[np.ndarray] = None,
        zero_trans: bool = True,
        zero_hands: bool = False,
        in_fps: int = 20,
        out_fps: int = 30,
        residuals=6,):

        tokens,codes = self.vq.encode(motion)
        self.decode_and_save(tokens, out_dir, name, betas, zero_trans, zero_hands, in_fps, out_fps, residuals)

    @torch.no_grad()
    def decode(self, motion_tokens: Tensor, residuals: int = 6) -> Tensor:
        """
            Return decoder pose for a 1‑D or 2‑D sequence of tokens.
            motion_tokens: Tensor: [batch size (optional), sequence length, n residuals]
        """
        if motion_tokens.dim() == 2:
            motion_tokens = motion_tokens.unsqueeze(0)
        motion_tokens = motion_tokens.to(self.device)[...,:residuals]

        embeds = self.vq.quantizer.get_codebook_entry(motion_tokens)
        pose = self.vq.decoder(embeds)  # (1, T, D)
        return pose

    def get_global_trans(self, aa, trans, contacts):
        global_input = self.pose_to_global_input(aa, trans, contacts)

        rec_global = self.global_motion_vq(global_input)
        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        init_trans = torch.zeros((1, 1, 1))

        rec_x = velocity2position(rec_trans_v_s[:, :, 0:1].to(self.device), 1/30, trans.unsqueeze(0)[:, 0, 0:1].to(self.device))
        rec_z = velocity2position(rec_trans_v_s[:, :, 2:3].to(self.device), 1/30, trans.unsqueeze(0)[:, 0, 2:3].to(self.device))
        rec_y = rec_trans_v_s[:, :, 1:2]

        rec_trans_world = torch.cat([rec_x, rec_y, rec_z], dim=-1).squeeze(0)
        out_trans = rec_trans_world

        return out_trans
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def decode_and_save(
        self,
        motion_tokens: Tensor,
        path: str,
        name: str,
        betas: Optional[np.ndarray] = None,
        zero_trans: bool = False,
        zero_hands: bool = True,
        in_fps: int = 20,
        out_fps: int = 30,
        residuals: int = 6,
    ) -> Tuple[Tensor, Tensor]:
        """
            Decode tokens, write SMPL npz, and return pose tensors.
            motion_tokens: (b,n,q)
        """

        if betas is None:
            sample = np.load(
                self.vq_config.default_file_path,
                allow_pickle=True,
            )
            betas = sample["betas"]
        os.makedirs(path, exist_ok=True)

        pose = self.decode(motion_tokens, residuals).squeeze(0)  # (T, D)
        pose = resample_pose_seq(pose, in_fps, out_fps).to(self.device)
        aa = rot6d_to_aa(pose[:, :330])
        if zero_trans:
            trans_world = torch.zeros((aa.size(0), 3), device=self.device)
        else:
            trans_world = self.get_global_trans(aa, pose[:, 330:333], pose[:, -4:])

        if zero_hands:
            default_hand_pose = torch.from_numpy(np.load('checkpoints/hands_pose.npy'))
            aa[..., 75:] = default_hand_pose

        expr = np.zeros((aa.size(0), 100), dtype=np.float32)

        save_smpl_file(
            path,
            f"{name}.npz",
            betas.squeeze(),
            aa.cpu().numpy(),
            expr,
            trans_world.cpu().numpy(),
            out_fps,
        )

        return pose, aa


    @staticmethod
    def pose_to_global_input(poses_in, trans_in, contacts, device="cuda"):
        poses = poses_in.clone() #torch.tensor(poses, dtype=torch.float32, device=device).clone()
        trans = trans_in.clone() #torch.tensor(trans, dtype=torch.float32, device=device).clone()

        lower_body_aa = poses[..., joint_mask_lower.astype(bool)]
        lower_body_mat = axis_angle_to_matrix(lower_body_aa.view(-1, 9, 3))
        lower_body_6d = matrix_to_rotation_6d(lower_body_mat).view(-1, 9*6)

        velocity = torch.zeros_like(trans, device=device)
        velocity[1:] = trans[1:] - trans[:-1]

        foot_contact = contacts.clone()

        global_motion_input = torch.cat([lower_body_6d, velocity, foot_contact], dim=1)

        global_motion_input[...,54:57] = 0.0
        return global_motion_input.unsqueeze(0)