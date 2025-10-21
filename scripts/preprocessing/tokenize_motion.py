import os, io, numpy as np, pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Audio, load_from_disk
import hydra
from omegaconf import DictConfig
import torch
import smplx
import common.utils.rotation_conversions as rc
from vq.modeling_vq import RVQVAE_Smpl
from external.PantoMatrix.scripts.EMAGE_2024.dataloaders.data_tools import joints_list
import yaml
from common.utils.tools import resample_pose_seq

ori_joints = "beat_smplx_joints"

ori_joint_list = joints_list[ori_joints]
tar_joint_list_lower = joints_list["beat_smplx_lower"]
joint_mask_lower = np.zeros(len(list(ori_joint_list.keys())) * 3)
for joint_name in tar_joint_list_lower:
    joint_mask_lower[
        ori_joint_list[joint_name][1] - ori_joint_list[joint_name][0] :
        ori_joint_list[joint_name][1]
    ] = 1


def pose_to_global_input(poses_in, trans_in, contacts, device="cuda"):
    poses = poses_in.clone() 
    trans = trans_in.clone() 

    lower_body_aa = poses[..., joint_mask_lower.astype(bool)]
    lower_body_mat = rc.axis_angle_to_matrix(lower_body_aa.view(-1, 9, 3))
    lower_body_6d = rc.matrix_to_rotation_6d(lower_body_mat).view(-1, 9*6)

    velocity = torch.zeros_like(trans, device=device)
    velocity[1:] = trans[1:] - trans[:-1]

    foot_contact = contacts.clone()

    global_motion_input = torch.cat([lower_body_6d, velocity, foot_contact], dim=1)

    global_motion_input[...,54:57] = 0.0
    return global_motion_input.unsqueeze(0)


def rot6d_to_aa(poses_6d):
    n = poses_6d.shape[0]
    mat = rc.rotation_6d_to_matrix(poses_6d.reshape(n, 55, 6))
    aa = rc.matrix_to_axis_angle(mat).reshape(n, 55*3)
    return aa

def process_beat_motion(m_data, smpl_model, device):
    '''
    Computes foot contacts (needed for tokenization)
    '''
    n = m_data["poses"].shape[0]

    betas = m_data["betas"].reshape(1, -1)
    betas = np.tile(betas, (n, 1))
    betas = torch.tensor(betas).float().to(device)

    poses = m_data["poses"]
    if not isinstance(poses, torch.Tensor):
        poses = torch.tensor(poses).float().to(device)
    else:
        poses = poses.to(device)
    trans = m_data["trans"]
    if not isinstance(trans, torch.Tensor):
        trans = torch.tensor(trans).float().to(device)
    else:
        trans = trans.to(device)
    exps = m_data["expressions"]
    if not isinstance(exps, torch.Tensor):
        exps = torch.tensor(exps).float().to(device)
    else:
        exps = exps.to(device)

    trans[...,1] -= trans[...,1].max()
    body_parms = {
                'root_orient': poses[:, :3].cuda(),
                'pose_body': poses[:, 3:21*3+3].cuda(),
                'pose_jaw': poses[:, 66:69].cuda(),
                'left_pose_hand': poses[:, 25*3:40*3].cuda(),
                'right_pose_hand': poses[:, 40*3:55*3].cuda(),
                'left_pose_eye': poses[:,69:72].cuda(),
                'right_pose_eye': poses[:, 72:75].cuda(),
                'trans': trans.cuda(),
                'betas': betas.cuda(),
            }

    n,j = poses.shape
    max_length = 128
    s, r = n//max_length, n%max_length
    #print(n, s, r)
    all_tensor = []
    for i in range(s):
        with torch.no_grad():
            joints = smpl_model(
                betas=body_parms['betas'][i*max_length:(i+1)*max_length], 
                transl=body_parms['trans'][i*max_length:(i+1)*max_length], 
                expression=exps[i*max_length:(i+1)*max_length], 
                jaw_pose=body_parms['pose_jaw'][i*max_length:(i+1)*max_length], 
                global_orient=body_parms['root_orient'][i*max_length:(i+1)*max_length], 
                body_pose=body_parms['pose_body'][i*max_length:(i+1)*max_length], 
                left_hand_pose=body_parms['left_pose_hand'][i*max_length:(i+1)*max_length], 
                right_hand_pose=body_parms['right_pose_hand'][i*max_length:(i+1)*max_length], 
                return_verts=True,
                return_joints=True,
                leye_pose=body_parms['left_pose_eye'][i*max_length:(i+1)*max_length], 
                reye_pose=body_parms['right_pose_eye'][i*max_length:(i+1)*max_length],
            )['joints'][:, (7,8,10,11), :].reshape(max_length, 4, 3).cpu()
        all_tensor.append(joints)
    if r != 0:
        with torch.no_grad():
            joints = smpl_model(
                betas=body_parms['betas'][s*max_length:s*max_length+r], 
                transl=body_parms['trans'][s*max_length:s*max_length+r], 
                expression=exps[s*max_length:s*max_length+r], 
                jaw_pose=body_parms['pose_jaw'][s*max_length:s*max_length+r], 
                global_orient=body_parms['root_orient'][s*max_length:s*max_length+r], 
                body_pose=body_parms['pose_body'][s*max_length:s*max_length+r], 
                left_hand_pose=body_parms['left_pose_hand'][s*max_length:s*max_length+r], 
                right_hand_pose=body_parms['right_pose_hand'][s*max_length:s*max_length+r], 
                return_verts=True,
                return_joints=True,
                leye_pose=body_parms['left_pose_eye'][s*max_length:s*max_length+r], 
                reye_pose=body_parms['right_pose_eye'][s*max_length:s*max_length+r],
            )['joints'][:, (7,8,10,11), :].reshape(r, 4, 3).cpu()
        all_tensor.append(joints)
    if len(all_tensor) > 0:
        joints = torch.cat(all_tensor, axis=0)
        feetv = torch.zeros(joints.shape[1], joints.shape[0])
        joints = joints.permute(1, 0, 2)
        feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).numpy().astype(float).transpose(1, 0)
        contacts = torch.from_numpy(contacts).float()

    mat = rc.axis_angle_to_matrix(poses.reshape(n, 55, 3))
    poses_6d = rc.matrix_to_rotation_6d(mat).reshape(n, 55*6)
    full_body = torch.cat((poses_6d, trans), dim=1)
    full_body = resample_pose_seq(full_body, 30, 20) # resampled to 20 fps
    contacts = resample_pose_seq(contacts, 30, 20)
    full_body = torch.cat([full_body.cuda(), contacts.cuda()], dim=1)
    return full_body

def rvqvae_map_smpl(motion, smpl_model, device, net):
    poses = process_beat_motion(motion,smpl_model, device)
    code_idx, all_codes = net.encode(poses.unsqueeze(0).to(device))
    code_idx = code_idx.squeeze(0).detach().cpu().numpy().astype(np.int16)
    return {"motion_token": code_idx.tolist()}


@hydra.main(version_base=None, config_path="../../configs/preprocess", config_name="tokenize_motion")
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_weights = cfg.vq_weights 
    with open(cfg.model_config, "r") as f:
        mc = yaml.safe_load(f)

    vq = RVQVAE_Smpl(
        input_width=mc["input_width"],
        nb_code=mc["nb_code"],
        code_dim=mc["code_dim"],
        output_emb_width=mc["output_emb_width"],
        down_t=mc["down_t"],
        stride_t=mc["stride_t"],
        width=mc["width"],
        depth=mc.get("depth", mc["width"]),
        dilation_growth_rate=mc["dilation_growth_rate"],
        activation=mc.get("activation", "relu"),
        norm=mc.get("norm", "batch"),
        num_quantizers=mc.get("num_quantizers", mc.get("num_quantizer")),
        quantize_dropout_prob=mc.get("quantize_dropout_prob", 0.0),
        quantize_dropout_cutoff_index=mc.get("quantize_dropout_cutoff_index", 0),
        shared_codebook=mc.get("shared_codebook", False),
        mu=mc.get("mu", 0.0),
    ).to(device).eval()
    ckpt = torch.load(model_weights, weights_only=False)['state_dict']
    for k in [k for k in ckpt.keys() if k.startswith("smplx")]:
        ckpt.pop(k)
    vq.load_state_dict(ckpt)

    smpl_model = smplx.create(
            "checkpoints/smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()
    print("SMPLX model:", smpl_model)

    dsets = load_from_disk(cfg.save_path).with_format('torch')
    dsets = dsets.map(lambda x: rvqvae_map_smpl(x, smpl_model, device, vq), input_columns="beat_motion")
    dsets.save_to_disk(cfg.save_path_2)


if __name__ == "__main__":
    main()



