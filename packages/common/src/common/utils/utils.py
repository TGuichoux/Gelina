import yaml
import numpy as np
import os
import torch
import importlib

def load_config(config_file_path):
    print("CONFIG:",config_file_path)
    try:
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{config_file_path}' was not found.")
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Error parsing YAML file: {exc}")


def save_smpl_file(save_path, name, betas, poses_aa, expressions, trans, fps):
    def to_numpy(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    np.savez(
        os.path.join(save_path, name),
        betas=to_numpy(betas).astype(np.float32).squeeze(),
        poses=to_numpy(poses_aa).astype(np.float32),
        expressions=to_numpy(expressions).astype(np.float32),
        trans=to_numpy(trans).astype(np.float32),
        model='smplx2020',
        gender='neutral',
        mocap_frame_rate=fps
    )

def _resolve_class(cls_path: str):
    mod, _, cls = cls_path.rpartition(".")
    if not mod or not cls:
        raise ValueError(f"Invalid class path: {cls_path}")
    module = importlib.import_module(mod)
    return getattr(module, cls)


def load_vq_checkpoint(cls_path: str, checkpoint_path: str, config, device: str = "cuda"):
    mc = load_config(config)
    VQClass = _resolve_class(cls_path)
    vq = VQClass(
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

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    for k in [k for k in list(state.keys()) if k.startswith("smplx")]:
        state.pop(k)
    vq.load_state_dict(state, strict=False)
    return vq, mc