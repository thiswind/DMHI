#!/usr/bin/env python3
"""
SAITS and BRITS baseline runner via PyPOTS.

Uses pypots.imputation.SAITS and pypots.imputation.BRITS (>=1.0) which support
general tabular time series without requiring HDF5 or complex ini config files.

Usage (single config):
    python run_saits.py \\
        --method saits \\
        --dataset physionet2012 \\
        --missing_rate 0.3 \\
        --missing_pattern mcar \\
        --gpu 0 \\
        --out_dir runs/baselines

Usage (all configs):
    python run_saits.py --method saits --all_configs --gpu 0 --out_dir runs/baselines
    python run_saits.py --method brits --all_configs --gpu 1 --out_dir runs/baselines
"""

import argparse
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import (
    load_dataset, apply_mcar, apply_block_missing,
    make_input_mask, compute_metrics, save_results, save_imputed,
    iter_configs, metrics_filename, imputed_filename, BLOCK_LENS,
)

DATASET_INFO = {
    "physionet2012":     {"T": 48, "D": 35},
    "beijing":           {"T": 24, "D": 11},
    "electricity":       {"T": 96, "D": 321},
    "mimic":             {"T": 48, "D": 59},
    "cmapss":            {"T": 50, "D": 14},
    "tep":               {"T": 48, "D": 41},
    "hydraulic":         {"T": 60, "D": 17},
    "air_quality_italy": {"T": 24, "D": 13},
    "naval_propulsion":  {"T": 50, "D": 16},
}


def run_one(method: str, dataset: str, missing_rate: float,
            missing_pattern: str, gpu: int, out_dir: str, seed: int = 42) -> dict:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{gpu}"

    info = DATASET_INFO[dataset]
    T, D = info["T"], info["D"]

    X_train, M_train, _ = load_dataset(dataset, "train")
    X_val,   M_val,   _ = load_dataset(dataset, "val")
    X_test,  M_test,  _ = load_dataset(dataset, "test")

    def make_nan_input(X, M_orig, mask_seed):
        if missing_pattern == "mcar":
            eval_mask = apply_mcar(M_orig, missing_rate, seed=mask_seed)
        else:
            block_len = BLOCK_LENS[missing_pattern]
            eval_mask = apply_block_missing(M_orig, block_len, missing_rate, seed=mask_seed)
        input_mask = make_input_mask(M_orig, eval_mask)
        X_nan = X.copy().astype(np.float32)
        X_nan[input_mask == 0] = np.nan
        return X_nan, eval_mask

    X_train_nan, _ = make_nan_input(X_train, M_train, seed)
    X_val_nan,   _ = make_nan_input(X_val,   M_val,   seed + 1)
    X_test_nan, eval_mask_test = make_nan_input(X_test, M_test, seed + 2)

    # PyPOTS requires X_ori in val_set for validation-metric computation.
    # X_ori = original ground truth: NaN where originally missing (M_val==0),
    # true values elsewhere (including held-out eval positions).
    X_ori_val = X_val.copy().astype(np.float32)
    X_ori_val[M_val == 0] = np.nan

    saving_path = str(out_dir / method / dataset /
                      f"rate{int(missing_rate*100):02d}_{missing_pattern}")

    if method == "saits":
        from pypots.imputation import SAITS
        model = SAITS(
            n_steps=T,
            n_features=D,
            n_layers=2,
            d_model=256,
            n_heads=4,
            d_k=64,
            d_v=64,
            d_ffn=256,
            dropout=0.1,
            batch_size=64,
            epochs=300,
            patience=30,
            saving_path=saving_path,
            device=device,
        )
    else:
        from pypots.imputation import BRITS
        model = BRITS(
            n_steps=T,
            n_features=D,
            rnn_hidden_size=256,
            batch_size=64,
            epochs=300,
            patience=30,
            saving_path=saving_path,
            device=device,
        )

    t0 = time.time()
    model.fit(
        train_set={"X": X_train_nan},
        val_set={"X": X_val_nan, "X_ori": X_ori_val},
    )
    train_min = (time.time() - t0) / 60

    t_infer = time.time()
    imputed_result = model.impute(test_set={"X": X_test_nan})
    infer_ms = (time.time() - t_infer) * 1000 / len(X_test)

    if isinstance(imputed_result, dict):
        X_hat = np.array(imputed_result.get("imputation", imputed_result.get("X", X_test)))
    else:
        X_hat = np.array(imputed_result)

    if X_hat.shape != X_test.shape:
        X_hat = X_hat.reshape(X_test.shape)

    n_params = sum(p.numel() for p in model.model.parameters()) if hasattr(model, "model") else 0
    metrics = compute_metrics(X_hat, X_test, eval_mask_test)

    results = {
        "method": method,
        "dataset": dataset,
        "missing_rate": missing_rate,
        "missing_pattern": missing_pattern,
        "seed": seed,
        "train_min": round(train_min, 2),
        "infer_ms_per_sample": round(infer_ms, 4),
        "n_params_M": round(n_params / 1e6, 3),
        **metrics,
    }

    metrics_name = metrics_filename(method, dataset, missing_rate, missing_pattern)
    save_results(results, str(out_dir / metrics_name))

    imputed_name = imputed_filename(method, dataset, missing_rate, missing_pattern)
    save_imputed(X_hat, str(out_dir / imputed_name))

    print(f"[{method}] {dataset} rate={missing_rate} pat={missing_pattern} "
          f"MAE={metrics['mae']:.4f} train={train_min:.1f}min")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["saits", "brits"], default="saits")
    parser.add_argument("--dataset", default="physionet2012")
    parser.add_argument("--missing_rate", type=float, default=0.3)
    parser.add_argument("--missing_pattern", default="mcar")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", default="runs/baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_configs", action="store_true")
    args = parser.parse_args()

    if args.all_configs:
        for ds, rate, pat in iter_configs():
            run_one(args.method, ds, rate, pat, args.gpu, args.out_dir, args.seed)
    else:
        run_one(args.method, args.dataset, args.missing_rate,
                args.missing_pattern, args.gpu, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
