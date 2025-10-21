#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path
import math
import pandas as pd

def run_nisqa(nisqa_script: Path, weights: Path, data_dir: Path, output_dir: Path) -> Path:
    """Run NISQA on data_dir; return path to NISQA_results.csv."""
    cmd = [
        sys.executable, str(nisqa_script),
        "--mode", "predict_dir",
        "--pretrained_model", str(weights),
        "--data_dir", str(data_dir),
        "--num_workers", "0",
        "--bs", "10",
        "--output_dir", str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    return output_dir / "NISQA_results.csv"

def mean_var_ci95(series: pd.Series):
    """Return mean, variance (unbiased), and 95% CI for a numeric series."""
    x = series.dropna().astype(float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        return float(x.iloc[0]), 0.0, 0.0
    mean = float(x.mean())
    var = float(x.var(ddof=1))
    ci95 = 1.96 * math.sqrt(var / n)
    return mean, var, ci95

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="out/segmented_outputs_V2", type=Path)
    ap.add_argument("--nisa_script", default="NISQA/run_predict.py", type=Path)
    ap.add_argument("--weights", default="NISQA/weights/nisqa_tts.tar", type=Path)
    ap.add_argument("--output_dir", default=Path("out"), type=Path)
    ap.add_argument("--ext", default=".wav")
    ap.add_argument("--save_csv", default="nisqa_summary.csv")
    args = ap.parse_args()

    base_dir: Path = args.base_dir
    subfolders = sorted([p for p in base_dir.iterdir() if p.is_dir()])

    rows = []
    for folder in subfolders:
        # Skip empty folders without audio files
        has_audio = any(folder.rglob(f"*{args.ext}"))
        if not has_audio:
            continue
        csv_path = run_nisqa(args.nisa_script, args.weights, folder, args.output_dir)
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found after processing {folder.name}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        if "mos_pred" not in df.columns:
            print(f"Warning: 'mos_pred' column missing in {csv_path} for {folder.name}", file=sys.stderr)
            continue
        mean, var, ci95 = mean_var_ci95(df["mos_pred"])
        rows.append({"folder": folder.name, "N": len(df), "mean_mos": mean, "var_mos": var, "ci95": ci95})
        print(f"{folder.name}\tN={len(df)}\tmean={mean:.4f}\tvar={var:.6f}\tCI95=±{ci95:.4f}")

    summary = pd.DataFrame(rows, columns=["folder", "N", "mean_mos", "var_mos", "ci95"]).sort_values("folder")
    summary.to_csv(args.save_csv, index=False)
    print(f"\nSaved summary to {args.save_csv}")

if __name__ == "__main__":
    main()
