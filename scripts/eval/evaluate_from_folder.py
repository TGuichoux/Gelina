#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, torch, numpy as np, soundfile as sf, hydra, pytorch_lightning as pl
from omegaconf import DictConfig
import os
from pathlib import Path
import librosa
import numpy as np
import torch
from tqdm import tqdm
from jiwer import wer, cer
import whisper
import pathlib
import common.utils.rotation_conversions as rc
from external.PantoMatrix.scripts.EMAGE_2024.emage_evaltools.mertic import FGD, L1div, BC
from common.data.data_utils import normalize
from common.utils.eval_tools import voice_similarity
import smplx
from gelina.utils.generation_helpers import wav2text
from whisper_normalizer.english import EnglishTextNormalizer


def top_k(a: np.ndarray, k: int):
    idx = np.argpartition(a, -k)[-k:]  
    return {i: a[i] for i in idx}

def load_smplx_model(device: torch.device):
    smplx_bm = smplx.create(
        "checkpoints/smplx_models/", 
        model_type='smplx',
        gender='NEUTRAL_2020', 
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100, 
        ext='npz',
        use_pca=False,
    ).cuda().eval()
    return smplx_bm

def mean_ci95(values):
    """Return (mean, 95% CI half-width) with normal approximation."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan")
    m = float(arr.mean())
    if n > 1:
        sd = float(arr.std(ddof=1))
        ci = 1.96 * sd / np.sqrt(n)
        return m, float(ci)
    else:
        return m, float("nan")

@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import pandas as pd

    df = pd.read_csv(os.path.join(cfg.root_dir,'BEAT2/beat_english_v2.0.0/train_test_split.csv'))
    df['n_ids'] = df['id'].map(lambda x: int(x.split('_')[0]))
    ids_train = df[df['type']=='train']['n_ids'].unique()
    ids_train.sort()



    ids_test = df[df['type']=='test']['n_ids'].unique()
    ids_test.sort()

    fgd = FGD(download_path="external/PantoMatrix/scripts/EMAGE_2024/emage_evaltools/")
    bc = BC(download_path="external/PantoMatrix/scripts/EMAGE_2024/emage_evaltools/", sigma=0.3, order=7)
    l1 = L1div()
    we_rate_list = []
    ce_rate_list = []
    v_sim_list = []
    asr = whisper.load_model("large-v3", device=device)
    english_normalizer = EnglishTextNormalizer()

    smplx_bm = load_smplx_model(device)

    gt_dir = Path(cfg.gt_folder)
    gen_dir = Path(cfg.gen_folder)

    ids = sorted([p.stem for p in gt_dir.glob("*.npz")])

    motion_pred = None  # default if no motion available
    for vid in tqdm(ids, desc="evaluating"):

        spk_id = int(vid.split('_')[0])
        if cfg.evaluated_speakers != 'all' and spk_id not in cfg.evaluated_speakers:
            continue
        gt_pose_file = gt_dir / f"{vid}.npz"
        gen_pose_file = gen_dir / f"res_{vid}.npz"
        gt_wav_file = gt_dir / f"{vid}.wav"
        gen_wav_file = gen_dir / f"res_{vid}.wav"
        text_file   = gt_dir / f"{vid}.txt"

        if not gt_wav_file.exists() or not gen_wav_file.exists():
            continue

        motion_pred = None
        if gen_pose_file.exists():
            motion_gt = np.load(gt_pose_file)["poses"]
            motion_pred = np.load(gen_pose_file)["poses"]

            # drop hands
            motion_gt[..., 75:] = 0.0
            motion_pred[..., 75:] = 0.0

            T = min(len(motion_gt), len(motion_pred))
            if T < 150:  # <5s
                continue
            motion_gt, motion_pred = motion_gt[:T], motion_pred[:T]

            l1.compute(motion_pred)

            motion_gt_aa = torch.from_numpy(motion_gt).float().to(device).unsqueeze(0)
            motion_pred_aa = torch.from_numpy(motion_pred).float().to(device).unsqueeze(0)

            B, N, D = motion_gt_aa.shape
            gt_mat = rc.axis_angle_to_matrix(motion_gt_aa.reshape(B, N, D // 3, 3))
            pred_mat = rc.axis_angle_to_matrix(motion_pred_aa.reshape(B, N, D // 3, 3))
            gt_6d = rc.matrix_to_rotation_6d(gt_mat).reshape(B, N, (D // 3) * 6)
            pred_6d = rc.matrix_to_rotation_6d(pred_mat).reshape(B, N, (D // 3) * 6)

            fgd.update(pred_6d, gt_6d)

            # --- BC: absolute positions --------------------------------------------------
            pred_pos = rc.get_position(
                motion_pred_aa.squeeze(0), betas=None, trans=None, smplx=smplx_bm
            ).reshape(T, -1)[:, : 55 * 3]

            audio_tensor = bc.load_audio(str(gen_wav_file), t_start=2 * 24000, t_end=int((T - 60) / 30 * 24000), sr_audio=24000)
            motion_tensor = bc.load_motion(
                pred_pos.cpu(), t_start=60, t_end=T - 60, pose_fps=30, without_file=True
            )
            bc.compute(audio_tensor, motion_tensor, length=T - 120, pose_fps=30)
            # -----------------------------------------------------------------------------

        speech_gt, _ = librosa.load(gt_wav_file, sr=24000)
        speech_pred, _ = librosa.load(gen_wav_file, sr=24000)

        v_sim_list.append(voice_similarity(gt_wav_file, gen_wav_file))

        gen_text_segments = wav2text(torch.from_numpy(speech_pred).reshape(-1), asr)
        gen_text = english_normalizer(" ".join(s['text'] for s in gen_text_segments).strip())

        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.readline()
        tar_text = english_normalizer(text)

        we_rate = min(wer(gen_text, tar_text), 1.0)  # Word error rate
        ce_rate = min(cer(gen_text, tar_text), 1.0)  # Character error rate

        if we_rate > 0.5 or ce_rate > 0.5:
            print(vid)
            print('\n')
            print(tar_text)
            print('\n')
            print("gen text:", gen_text)
            print('\n\n')

        we_rate_list.append(we_rate)
        ce_rate_list.append(ce_rate)

    fgd_val = fgd.compute() if motion_pred is not None else None
    l1_val  = l1.avg() if motion_pred is not None else None
    bc_val  = bc.avg() if motion_pred is not None else None

    # ---- New: means + 95% CI for WER, CER, Speaker Similarity -----------------
    wer_mean, wer_ci = mean_ci95(we_rate_list)
    cer_mean, cer_ci = mean_ci95(ce_rate_list)
    spk_mean, spk_ci = mean_ci95(v_sim_list)
    # ----------------------------------------------------------------------------

    metrics = {
        "fgd": fgd_val,
        "l1": l1_val,
        "BC": bc_val,
        "wer": wer_mean,
        "wer_ci95": wer_ci,
        "cer": cer_mean,
        "cer_ci95": cer_ci,
        "Spk Sim": spk_mean,
        "Spk Sim_ci95": spk_ci,
        "n_items": len(we_rate_list)  # number of evaluated items
    }
    print(metrics)

    # Top-20 highest WER indices (if you want worst cases, use np.argsort instead)
    print(top_k(np.array(we_rate_list), 20))


if __name__ == "__main__":
    main()

# processor = WavLMProcessor.from_pretrained("microsoft/wavlm-large")
# model = WavLMModel.from_pretrained("microsoft/wavlm-large").eval()
