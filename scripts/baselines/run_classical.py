#!/usr/bin/env python3
"""
Classical baseline imputation methods: Mean fill, Linear interpolation, LOCF.
Pure numpy — no GPU required.

Usage (single config):
    python run_classical.py \\
        --method mean \\
        --dataset physionet2012 \\
        --missing_rate 0.3 \\
        --missing_pattern mcar \\
        --out_dir runs/baselines \\
        --seed 42

Usage (all configs):
    python run_classical.py --all_configs --out_dir runs/baselines
"""

import argparse
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


# ---------------------------------------------------------------------------
# Imputation methods
# ---------------------------------------------------------------------------

def impute_mean(X_train: np.ndarray, M_train: np.ndarray,
                X_test: np.ndarray, input_mask_test: np.ndarray) -> np.ndarray:
    """Fill missing positions with per-feature mean from training set."""
    D = X_train.shape[2]
    feat_mean = np.zeros(D, dtype=np.float32)
    for d in range(D):
        obs = X_train[:, :, d][M_train[:, :, d] == 1]
        feat_mean[d] = float(obs.mean()) if len(obs) > 0 else 0.0

    X_hat = X_test.copy()
    for d in range(D):
        missing = input_mask_test[:, :, d] == 0
        X_hat[:, :, d][missing] = feat_mean[d]
    return X_hat


def impute_linear(X_test: np.ndarray, input_mask_test: np.ndarray) -> np.ndarray:
    """Linear interpolation along time axis per feature per sample."""
    N, T, D = X_test.shape
    X_hat = X_test.copy()
    t = np.arange(T, dtype=np.float64)

    for n in range(N):
        for d in range(D):
            obs_idx = np.where(input_mask_test[n, :, d] == 1)[0]
            if len(obs_idx) == 0:
                continue
            obs_vals = X_test[n, obs_idx, d].astype(np.float64)
            X_hat[n, :, d] = np.interp(t, obs_idx.astype(np.float64),
                                        obs_vals).astype(np.float32)
    return X_hat


def impute_locf(X_test: np.ndarray, input_mask_test: np.ndarray) -> np.ndarray:
    """Last Observation Carried Forward (forward-fill, then 0 for leading NaN)."""
    N, T, D = X_test.shape
    X_hat = np.zeros_like(X_test)

    for n in range(N):
        for d in range(D):
            last_val = 0.0
            for t in range(T):
                if input_mask_test[n, t, d] == 1:
                    last_val = float(X_test[n, t, d])
                X_hat[n, t, d] = last_val
    return X_hat


_METHOD_FN = {
    "mean": impute_mean,
    "linear": impute_linear,
    "locf": impute_locf,
}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(method: str, dataset: str, missing_rate: float,
            missing_pattern: str, out_dir: str, seed: int = 42) -> dict:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, M_train, _ = load_dataset(dataset, "train")
    X_test,  M_test,  _ = load_dataset(dataset, "test")

    if missing_pattern == "mcar":
        eval_mask = apply_mcar(M_test, missing_rate, seed=seed)
    else:
        block_len = BLOCK_LENS[missing_pattern]
        eval_mask = apply_block_missing(M_test, block_len, missing_rate, seed=seed)

    input_mask_test = make_input_mask(M_test, eval_mask)

    t0 = time.perf_counter()
    if method == "mean":
        X_hat = impute_mean(X_train, M_train, X_test, input_mask_test)
    elif method == "linear":
        X_hat = impute_linear(X_test, input_mask_test)
    elif method == "locf":
        X_hat = impute_locf(X_test, input_mask_test)
    else:
        raise ValueError(f"Unknown method: {method}")
    infer_ms = (time.perf_counter() - t0) * 1000 / len(X_test)

    metrics = compute_metrics(X_hat, X_test, eval_mask)

    results = {
        "method": method,
        "dataset": dataset,
        "missing_rate": missing_rate,
        "missing_pattern": missing_pattern,
        "seed": seed,
        "train_min": 0.0,
        "infer_ms_per_sample": round(infer_ms, 4),
        "n_params_M": 0.0,
        **metrics,
    }

    metrics_name = metrics_filename(method, dataset, missing_rate, missing_pattern)
    save_results(results, str(out_dir / metrics_name))

    imputed_name = imputed_filename(method, dataset, missing_rate, missing_pattern)
    save_imputed(X_hat, str(out_dir / imputed_name))

    print(f"[{method}] {dataset} rate={missing_rate} pat={missing_pattern} "
          f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classical baseline imputation")
    parser.add_argument("--method", choices=["mean", "linear", "locf"],
                        help="Method to run (ignored when --all_configs)")
    parser.add_argument("--dataset", default="physionet2012")
    parser.add_argument("--missing_rate", type=float, default=0.3)
    parser.add_argument("--missing_pattern", default="mcar")
    parser.add_argument("--out_dir", default="runs/baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_configs", action="store_true",
                        help="Run all methods × datasets × rates × patterns")
    args = parser.parse_args()

    methods = list(_METHOD_FN.keys())

    if args.all_configs:
        for meth in methods:
            for ds, rate, pat in iter_configs():
                run_one(meth, ds, rate, pat, args.out_dir, seed=args.seed)
    else:
        if args.method is None:
            for meth in methods:
                run_one(meth, args.dataset, args.missing_rate,
                        args.missing_pattern, args.out_dir, seed=args.seed)
        else:
            run_one(args.method, args.dataset, args.missing_rate,
                    args.missing_pattern, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
