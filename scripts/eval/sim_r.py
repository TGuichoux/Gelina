#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from common.utils.eval_tools import voice_similarity

def parse_speaker_id(filename: str):
    parts = Path(filename).stem.split("_")
    for p in parts:
        if p.isdigit():
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True, type=Path)
    ap.add_argument("--res_dir", required=True, type=Path)
    args = ap.parse_args()

    gt_files = sorted([f for f in args.gt_dir.glob("*.wav")])
    res_files = sorted([f for f in args.res_dir.glob("*.wav")])

    gt_by_spk, res_by_spk = defaultdict(list), defaultdict(list)
    for f in gt_files:
        spk = parse_speaker_id(f.name)
        if spk: gt_by_spk[spk].append(f)
    for f in res_files:
        spk = parse_speaker_id(f.name)
        if spk: res_by_spk[spk].append(f)

    common_spks = sorted(set(gt_by_spk) & set(res_by_spk), key=int)

    all_per_res_means = []

    for spk in tqdm(common_spks, desc="Speakers"):
        print(f"\nSpeaker {spk}:")
        gt_list = sorted(gt_by_spk[spk])
        res_list = sorted(res_by_spk[spk])

        for res_f in tqdm(res_list, leave=False, desc=f"RES files (spk {spk})"):
            sims = []
            for gt_f in gt_list:
                sims.append(voice_similarity(str(gt_f), str(res_f)))
            sims = np.array(sims, dtype=float)
            mean_sim = float(sims.mean()) if sims.size else 0.0
            all_per_res_means.append(mean_sim)
            print(f"  {res_f.name}  vs  {len(gt_list)}xGT  -> mean={mean_sim:.4f}")

    all_per_res_means = np.array(all_per_res_means, dtype=float)
    if all_per_res_means.size > 1:
        mean = all_per_res_means.mean()
        var  = all_per_res_means.var(ddof=1)
        std  = all_per_res_means.std(ddof=1)
        ci95 = 1.96 * std / np.sqrt(all_per_res_means.size)
    elif all_per_res_means.size == 1:
        mean = float(all_per_res_means[0])
        var = 0.0
        ci95 = 0.0
    else:
        mean = var = ci95 = float("nan")

    print("\n==== Global similarity summary ====")
    print(f"  N={all_per_res_means.size}")
    print(f"  Mean={mean:.4f}, Var={var:.6f}, 95% CI=±{ci95:.4f}")

if __name__ == "__main__":
    main()
