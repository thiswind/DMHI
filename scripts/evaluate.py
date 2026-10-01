#!/usr/bin/env python3
"""Evaluate a saved RiemannianImputer checkpoint on a test split.

Usage:
    python scripts/evaluate.py --run_dir checkpoints/dmhi/<domain>_seed<seed>/
                               [--block_size 20] [--rate 0.9] [--seed 7]

Outputs metrics.json in the run directory.
"""
import argparse
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from dmhi.method.pipeline import RiemannianImputer
from dmhi.utils.eval_protocol import apply_block_missing as _apply_block_missing
from dmhi.utils.eval_protocol import (manifold_deviation as _manifold_deviation,
                                      observed_dims as _observed_dims)


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _apply_mcar(X: np.ndarray, M_orig: np.ndarray, rate: float, seed: int):
    """Apply additional synthetic MCAR missingness on top of M_orig."""
    rng = np.random.default_rng(seed)
    synth = (rng.random(X.shape) > rate).astype(np.int8)
    return (M_orig & synth).astype(np.int8)


def mae(X_hat, X_true, eval_mask):
    return float(np.abs(X_hat[eval_mask] - X_true[eval_mask]).mean())


def rmse(X_hat, X_true, eval_mask):
    return float(np.sqrt(np.mean((X_hat[eval_mask] - X_true[eval_mask]) ** 2)))


def mre(X_hat, X_true, eval_mask):
    denom = float(np.abs(X_true[eval_mask]).mean()) + 1e-8
    return float(np.abs(X_hat[eval_mask] - X_true[eval_mask]).mean()) / denom


def trajectory_error(X_hat: np.ndarray, X_true: np.ndarray, eval_mask: np.ndarray) -> float:
    """TE: mean norm of consecutive prediction differences vs ground truth (eq 8)."""
    N, T, D = X_hat.shape
    te_list = []
    for n in range(N):
        for t in range(T - 1):
            if eval_mask[n, t].any() and eval_mask[n, t + 1].any():
                diff_hat = X_hat[n, t + 1] - X_hat[n, t]
                diff_true = X_true[n, t + 1] - X_true[n, t]
                te_list.append(float(np.linalg.norm(diff_hat - diff_true)))
    return float(np.mean(te_list)) if te_list else float("nan")


def manifold_deviation(X_hat: np.ndarray, X_true: np.ndarray,
                       eval_mask: np.ndarray, k: int = 5):
    """MD: mean k-NN distance of the imputed time steps to the manifold (eq 9).

    See :func:`dmhi.utils.eval_protocol.manifold_deviation`; the query set is
    restricted to the time steps that actually contain held-out entries, and
    the distance is taken over channels observed in the split.
    """
    return _manifold_deviation(X_hat, X_true, eval_mask, k=k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Path to run directory with config.json + checkpoint.pkl")
    p.add_argument("--block_size", type=int, default=None,
                   help="Block length for block-missing pattern; defaults to "
                        "config.json's block_size. Paper Tables II/III use the "
                        "per-domain block length recorded in the checkpoint config.")
    p.add_argument("--rate", type=float, default=None,
                   help="Missing rate; defaults to config.json's missing_rate. "
                        "Paper Tables II/III use rho=0.9.")
    p.add_argument("--seed", type=int, default=42,
                   help="Reporting seed: selects the paper cell (checkpoint is "
                        "loaded from run_dir regardless).")
    p.add_argument("--mask_seed", type=int, default=None,
                   help="Seed for the eval-mask RNG. Paper convention: "
                        "mask_seed = seed + 2 (shared with baselines). "
                        "Default: seed + 2.")
    p.add_argument("--deterministic", action="store_true",
                   help="Enable strict determinism for inference (mirrors train.py).")
    args = p.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    run_dir = pathlib.Path(args.run_dir)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    dataset = cfg["dataset"]
    missing_rate = args.rate if args.rate is not None else cfg["missing_rate"]
    pattern = cfg["pattern"]
    train_min = cfg.get("train_min", None)
    run_seed = int(cfg.get("seed", args.seed))
    block_size = args.block_size if args.block_size is not None else cfg.get("block_size", 20)
    mask_seed = args.mask_seed if args.mask_seed is not None else run_seed + 2

    data_dir = pathlib.Path("data/processed") / dataset
    X_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    X_test = np.load(data_dir / "X_test.npy").astype(np.float32)
    M_test = np.load(data_dir / "M_test.npy").astype(np.int8)

    # Apply synthetic missingness to produce input mask and eval mask
    if pattern == "mcar":
        M_eval = _apply_mcar(X_test, M_test, missing_rate, seed=mask_seed)
        eval_mask = (M_test == 1) & (M_eval == 0)
    elif pattern == "block":
        block_held_out = _apply_block_missing(
            M_test, block_size, missing_rate, seed=mask_seed)
        eval_mask = block_held_out.astype(bool)
        M_eval = (M_test & ~block_held_out).astype(np.int8)
    else:
        M_eval = M_test.copy()
        eval_mask = np.zeros_like(M_test, dtype=bool)

    print(f"Eval entries: {eval_mask.sum()} / {M_test.sum()}")

    # Load model
    imputer = RiemannianImputer.load(str(run_dir / "checkpoint.pkl"))

    t0 = time.time()
    X_hat = imputer.impute(X_test, M_eval)
    elapsed_ms = (time.time() - t0) / len(X_test) * 1000

    N_test, T, D = X_test.shape
    md_mean, md_p90 = manifold_deviation(X_hat, X_test, eval_mask, k=5)
    metrics = {
        "dataset": dataset,
        "missing_rate": missing_rate,
        "pattern": pattern,
        "block_size": block_size if pattern == "block" else None,
        "seed": run_seed,
        "n_train": int(X_train.shape[0]),
        "n_test": int(N_test),
        "T": int(T),
        "D": int(D),
        "n_eval_entries": int(eval_mask.sum()),
        "mean_fill_mae": round(float(np.abs(X_test[eval_mask]).mean()), 4),
        "MAE": round(mae(X_hat, X_test, eval_mask), 6),
        "MSE": round(float(((X_hat[eval_mask] - X_test[eval_mask]) ** 2).mean()), 6),
        "RMSE": round(rmse(X_hat, X_test, eval_mask), 6),
        "MRE": round(mre(X_hat, X_test, eval_mask), 6),
        "TE": round(trajectory_error(X_hat, X_test, eval_mask), 6),
        "md_mean": round(md_mean, 6),
        "md_p90": round(md_p90, 6),
        "MD": round(md_mean, 6),
        "n_obs_dims": int(len(_observed_dims(M_test))),
        "mask_seed": int(mask_seed),
        "train_min": round(train_min, 2) if train_min is not None else None,
        "infer_ms_per_sample": round(elapsed_ms, 3),
        "deterministic_train": bool(cfg.get("deterministic", False)),
        "deterministic_eval": bool(args.deterministic),
    }

    out_path = run_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
