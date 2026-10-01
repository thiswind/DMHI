#!/usr/bin/env python3
"""
ImputeFormer baseline runner via PyPOTS.

Uses pypots.imputation.ImputeFormer (>=1.0) which supports general tabular
time series without requiring a graph structure.

Usage (single config):
    CUDA_VISIBLE_DEVICES=0 python run_imputeformer.py \\
        --dataset physionet2012 \\
        --missing_rate 0.3 \\
        --missing_pattern mcar \\
        --gpu 0 \\
        --out_dir runs/baselines

Usage (all configs):
    CUDA_VISIBLE_DEVICES=1 python run_imputeformer.py --all_configs --gpu 1 --out_dir runs/baselines
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
    DATASET_SHAPES,
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


def run_one(dataset: str, missing_rate: float, missing_pattern: str,
            gpu: int, out_dir: str, seed: int = 42,
            smoke: bool = False) -> dict:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from pypots.imputation import ImputeFormer

    info = DATASET_INFO[dataset]
    T, D = info["T"], info["D"]

    X_train, M_train, _ = load_dataset(dataset, "train")
    X_val,   M_val,   _ = load_dataset(dataset, "val")
    X_test,  M_test,  _ = load_dataset(dataset, "test")

    def make_nan_input(X, M_orig, eval_mask):
        input_mask = make_input_mask(M_orig, eval_mask)
        X_nan = X.copy().astype(np.float32)
        X_nan[input_mask == 0] = np.nan
        return X_nan

    if missing_pattern == "mcar":
        eval_mask_train = apply_mcar(M_train, missing_rate, seed=seed)
        eval_mask_val   = apply_mcar(M_val,   missing_rate, seed=seed + 1)
        eval_mask_test  = apply_mcar(M_test,  missing_rate, seed=seed + 2)
    else:
        block_len = BLOCK_LENS[missing_pattern]
        eval_mask_train = apply_block_missing(M_train, block_len, missing_rate, seed=seed)
        eval_mask_val   = apply_block_missing(M_val,   block_len, missing_rate, seed=seed + 1)
        eval_mask_test  = apply_block_missing(M_test,  block_len, missing_rate, seed=seed + 2)

    X_train_nan = make_nan_input(X_train, M_train, eval_mask_train)
    X_val_nan   = make_nan_input(X_val,   M_val,   eval_mask_val)
    X_test_nan  = make_nan_input(X_test,  M_test,  eval_mask_test)

    # PyPOTS requires X_ori in val_set for validation-metric computation.
    X_ori_val = X_val.copy().astype(np.float32)
    X_ori_val[M_val == 0] = np.nan

    saving_path = str(out_dir / "imputeformer" / dataset /
                      f"rate{int(missing_rate*100):02d}_{missing_pattern}")

    train_epochs = 2 if smoke else 100
    train_patience = 2 if smoke else 20
    device = "cpu" if smoke else f"cuda:{gpu}"

    model = ImputeFormer(
        n_steps=T,
        n_features=D,
        n_layers=2,
        d_input_embed=32,
        d_learnable_embed=8,
        d_proj=16,
        d_ffn=64,
        n_temporal_heads=4,
        dropout=0.1,
        batch_size=64,
        epochs=train_epochs,
        patience=train_patience,
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
        "method": "imputeformer",
        "dataset": dataset,
        "missing_rate": missing_rate,
        "missing_pattern": missing_pattern,
        "seed": seed,
        "train_min": round(train_min, 2),
        "infer_ms_per_sample": round(infer_ms, 4),
        "n_params_M": round(n_params / 1e6, 3),
        **metrics,
    }

    metrics_name = metrics_filename("imputeformer", dataset, missing_rate, missing_pattern)
    save_results(results, str(out_dir / metrics_name))

    imputed_name = imputed_filename("imputeformer", dataset, missing_rate, missing_pattern)
    save_imputed(X_hat, str(out_dir / imputed_name))

    print(f"[imputeformer] {dataset} rate={missing_rate} pat={missing_pattern} "
          f"MAE={metrics['mae']:.4f} train={train_min:.1f}min")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="physionet2012")
    parser.add_argument("--missing_rate", type=float, default=0.3)
    parser.add_argument("--missing_pattern", default="mcar")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", default="runs/baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_configs", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 2 epochs on CPU, single config only")
    args = parser.parse_args()

    if args.all_configs:
        for ds, rate, pat in iter_configs():
            run_one(ds, rate, pat, args.gpu, args.out_dir, args.seed, smoke=args.smoke)
    else:
        run_one(args.dataset, args.missing_rate, args.missing_pattern,
                args.gpu, args.out_dir, args.seed, smoke=args.smoke)


if __name__ == "__main__":
    main()
