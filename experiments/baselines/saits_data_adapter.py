#!/usr/bin/env python3
"""
Convert our npy dataset format to HDF5 format required by SAITS/BRITS official code.

SAITS expects: <dataset_base_dir>/datasets.h5
  Groups: train, val, test
  Each group keys:
    X             : (N, T, D) float32, nan where missing (original + eval mask)
    missing_mask  : (N, T, D) float32, 1=observed (original mask minus eval mask)
    indicating_mask: (N, T, D) float32, 1=held-out eval positions

Usage:
    python saits_data_adapter.py \\
        --dataset physionet2012 \\
        --missing_rate 0.3 \\
        --missing_pattern mcar \\
        --out_dir /tmp/saits_data \\
        --seed 42
"""

import argparse
import pathlib
import sys

import h5py
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import (
    load_dataset, apply_mcar, apply_block_missing,
    make_input_mask, BLOCK_LENS,
)


def convert_split(dataset_name: str, split: str, missing_rate: float,
                  missing_pattern: str, seed: int) -> tuple:
    X, M_orig, _ = load_dataset(dataset_name, split)

    if missing_pattern == "mcar":
        eval_mask = apply_mcar(M_orig, missing_rate, seed=seed)
    else:
        block_len = BLOCK_LENS[missing_pattern]
        eval_mask = apply_block_missing(M_orig, block_len, missing_rate, seed=seed)

    input_mask = make_input_mask(M_orig, eval_mask)

    X_nan = X.copy()
    X_nan[(input_mask == 0)] = np.nan

    return X_nan.astype(np.float32), input_mask.astype(np.float32), eval_mask.astype(np.float32)


def convert_to_h5(dataset_name: str, missing_rate: float,
                  missing_pattern: str, out_dir: str, seed: int = 42) -> str:
    """
    Create datasets.h5 from our npy data.

    Returns
    -------
    str : path to the created HDF5 file
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "datasets.h5"

    with h5py.File(out_path, "w") as f:
        for split in ["train", "val", "test"]:
            X_nan, missing_mask, indicating_mask = convert_split(
                dataset_name, split, missing_rate, missing_pattern,
                seed=seed + hash(split) % 1000,
            )
            grp = f.create_group(split)
            grp.create_dataset("X", data=X_nan, compression="gzip")
            grp.create_dataset("missing_mask", data=missing_mask, compression="gzip")
            grp.create_dataset("indicating_mask", data=indicating_mask, compression="gzip")

    print(f"[saits_converter] Wrote {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--missing_rate", type=float, required=True)
    parser.add_argument("--missing_pattern", default="mcar")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert_to_h5(
        args.dataset, args.missing_rate,
        args.missing_pattern, args.out_dir, args.seed,
    )


if __name__ == "__main__":
    main()
