#!/usr/bin/env python3
"""
Convert our npy dataset to CSV format required by TimeMixer official code.

TimeMixer's Dataset_Custom reads a CSV with:
  - First column: 'date'  (datetime string, e.g. 2000-01-01 00:00:00)
  - Remaining D columns: feature values

We flatten (N, T, D) → (N*T, D+1).
Missing positions are zero-filled (TimeMixer will apply its own MCAR mask).

Usage:
    python timemixer_data_adapter.py \\
        --dataset physionet2012 \\
        --out_dir /tmp/timemixer_data
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import load_dataset


def convert_to_csv(dataset_name: str, split: str, out_dir: str) -> str:
    """
    Convert one split to CSV.

    Returns path to written CSV file.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, M_orig, _ = load_dataset(dataset_name, split)
    N, T, D = X.shape

    X_zerofill = X * M_orig

    X_flat = X_zerofill.reshape(N * T, D)

    base_dt = pd.Timestamp("2000-01-01 00:00:00")
    dates = pd.date_range(start=base_dt, periods=N * T, freq="1h")
    date_col = dates.strftime("%Y-%m-%d %H:%M:%S")

    feat_cols = [f"feat_{i:03d}" for i in range(D)]
    df = pd.DataFrame(X_flat, columns=feat_cols)
    df.insert(0, "date", date_col)

    out_path = out_dir / f"{dataset_name}_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"[timemixer_converter] Wrote {out_path}  shape=({N*T}, {D+1})")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        convert_to_csv(args.dataset, split, args.out_dir)


if __name__ == "__main__":
    main()
