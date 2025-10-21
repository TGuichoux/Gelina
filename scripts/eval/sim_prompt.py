#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, glob, os, math, numpy as np
from typing import List, Tuple
from common.utils.eval_tools import voice_similarity
from tqdm import tqdm

def list_pairs(prompt_dir: str, gen_dir: str, ext: str = ".wav") -> List[Tuple[str, str]]:
    """Pair prompt and generated files using file_id / res_file_id convention."""
    p_map = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(prompt_dir, f"*{ext}"))}
    g_map = {os.path.splitext(os.path.basename(g))[0].replace("res_", ""): g for g in glob.glob(os.path.join(gen_dir, f"*{ext}"))}
    keys = sorted(set(p_map) & set(g_map))
    return [(p_map[k], g_map[k]) for k in keys]

def mean_ci95(vals: List[float]) -> Tuple[float, float]:
    """Return mean and 95% CI."""
    arr = np.array(vals, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    m = float(arr.mean())
    s = float(arr.std(ddof=1))
    ci = 1.96 * s / math.sqrt(len(arr))
    return m, ci

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_dir", default="out/segmented_outputs_V2/prompts/")
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--ext", default=".wav")
    args = ap.parse_args()

    pairs = list_pairs(args.prompt_dir, args.gen_dir, args.ext)
    scores = []
    for p, g in tqdm(pairs):
        s = float(voice_similarity(p, g))
        scores.append(s)
        print(f"{os.path.basename(g)}\tSS={s:.4f}")

    m, ci = mean_ci95(scores)
    print(f"\nSS Mean={m:.4f}  CI95=±{ci:.4f}  N={len(scores)}")

if __name__ == "__main__":
    main()
