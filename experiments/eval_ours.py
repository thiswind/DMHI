#!/usr/bin/env python3
"""Evaluate a saved RiemannianImputer checkpoint on a test split.

Usage:
    conda activate base
    python experiments/eval_ours.py --run_dir experiments/logs/<run_id>/

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

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from dmhi.method.pipeline import RiemannianImputer


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


def _apply_block_missing(M_orig: np.ndarray, block_size: int,
                         rate: float, seed: int) -> np.ndarray:
    """Apply synthetic block missingness on top of M_orig. Returns eval mask (1=held-out)."""
    rng = np.random.default_rng(seed)
    N, T, D = M_orig.shape
    eval_mask = np.zeros_like(M_orig)
    blocks_per_sample = max(1, int(round(T * rate / block_size)))
    for n in range(N):
        for d in range(D):
            obs_times = np.where(M_orig[n, :, d] == 1)[0]
            if len(obs_times) < block_size:
                continue
            placed = 0
            attempts = 0
            while placed < blocks_per_sample and attempts < 50:
                start = rng.integers(0, T - block_size + 1)
                block_idx = np.arange(start, start + block_size)
                if np.all(M_orig[n, block_idx, d] == 1):
                    eval_mask[n, block_idx, d] = 1
                    placed += 1
                attempts += 1
    return eval_mask


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


def manifold_deviation(X_hat: np.ndarray, X_true: np.ndarray, k: int = 5) -> float:
    """MD: mean distance of imputed points from the k-NN manifold of observed (eq 9)."""
    from sklearn.neighbors import NearestNeighbors
    N, T, D = X_hat.shape
    X_obs_flat = X_true.reshape(-1, D)
    X_hat_flat = X_hat.reshape(-1, D)
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(X_obs_flat)
    dists, _ = nn.kneighbors(X_hat_flat)
    return float(dists.mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Path to run directory with config.json + checkpoint.pkl")
    p.add_argument("--block_size", type=int, default=10,
                   help="Block length for block-missing pattern")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true",
                   help="Enable strict determinism for inference (mirrors train_ours.py).")
    args = p.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    run_dir = pathlib.Path(args.run_dir)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    dataset = cfg["dataset"]
    missing_rate = cfg["missing_rate"]
    pattern = cfg["pattern"]
    train_min = cfg.get("train_min", None)

    data_dir = pathlib.Path("data/processed") / dataset
    X_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    X_test = np.load(data_dir / "X_test.npy").astype(np.float32)
    M_test = np.load(data_dir / "M_test.npy").astype(np.int8)

    # Apply synthetic missingness to produce input mask and eval mask
    if pattern == "mcar":
        M_eval = _apply_mcar(X_test, M_test, missing_rate, seed=args.seed)
        eval_mask = (M_test == 1) & (M_eval == 0)
    elif pattern == "block":
        block_held_out = _apply_block_missing(
            M_test, args.block_size, missing_rate, seed=args.seed)
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
    metrics = {
        "dataset": dataset,
        "missing_rate": missing_rate,
        "pattern": pattern,
        "block_size": args.block_size if pattern == "block" else None,
        "seed": args.seed,
        "n_train": int(X_train.shape[0]),
        "n_test": int(N_test),
        "T": int(T),
        "D": int(D),
        "n_eval_entries": int(eval_mask.sum()),
        "mean_fill_mae": round(float(np.abs(X_test[eval_mask]).mean()), 4),
        "MAE": round(mae(X_hat, X_test, eval_mask), 6),
        "RMSE": round(rmse(X_hat, X_test, eval_mask), 6),
        "MRE": round(mre(X_hat, X_test, eval_mask), 6),
        "TE": round(trajectory_error(X_hat, X_test, eval_mask), 6),
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
