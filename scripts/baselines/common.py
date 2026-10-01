#!/usr/bin/env python3
"""
Shared utilities for baseline comparison experiments.
Provides data loading, masking, metric computation, and result saving.
"""

import json
import pathlib
from datetime import datetime, timezone

import numpy as np

from dmhi.utils.eval_protocol import apply_block_missing  # canonical mask protocol


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_ROOT = pathlib.Path(__file__).parent.parent.parent / "data" / "processed"

DATASET_SHAPES = {
    "physionet2012": {"T": 48, "D": 35},
    "beijing":       {"T": 24, "D": 11},
    "electricity":   {"T": 96, "D": 321},
    "pems_bay":      {"T": 288, "D": 325},
    "mimic":         {"T": 48, "D": 59},
    "cmapss":        {"T": 50, "D": 14},
    "tep":           {"T": 48, "D": 41},
}


def load_dataset(dataset_name: str, split: str):
    """
    Load processed dataset arrays.

    Parameters
    ----------
    dataset_name : str
        One of: physionet2012, beijing, electricity, pems_bay, mimic, cmapss, tep
    split : str
        One of: train, val, test

    Returns
    -------
    X : np.ndarray, shape (N, T, D)
        Observed values; 0 where missing.
    M : np.ndarray, shape (N, T, D)
        Binary observation mask (1=observed, 0=missing).
    y : np.ndarray or None, shape (N,)
        Labels for physionet2012 / mimic. None otherwise.
    """
    d = DATA_ROOT / dataset_name
    X = np.load(d / f"X_{split}.npy")
    M = np.load(d / f"M_{split}.npy")
    y_path = d / f"y_{split}.npy"
    y = np.load(y_path) if y_path.exists() else None
    return X.astype(np.float32), M.astype(np.float32), y


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def apply_mcar(M_orig: np.ndarray, rate: float, seed: int = 42) -> np.ndarray:
    """
    Apply additional MCAR mask on top of original observation mask.

    Randomly selects `rate` fraction of currently observed positions and marks
    them as held-out (eval mask). The union of original missing + newly hidden
    positions forms the input mask; the newly hidden positions are the eval set.

    Parameters
    ----------
    M_orig : (N, T, D) float32
    rate : float  — fraction of observed positions to hold out
    seed : int

    Returns
    -------
    eval_mask : (N, T, D) float32  — 1 at newly hidden positions, 0 elsewhere
    """
    rng = np.random.default_rng(seed)
    rand = rng.random(M_orig.shape).astype(np.float32)
    eval_mask = ((rand < rate) & (M_orig == 1)).astype(np.float32)
    return eval_mask


def make_input_mask(M_orig: np.ndarray, eval_mask: np.ndarray) -> np.ndarray:
    """
    Return input mask: original observed positions minus eval positions.
    input_mask[i,t,d] = 1 iff observed AND not held-out.
    """
    return (M_orig * (1 - eval_mask)).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(X_hat: np.ndarray, X_true: np.ndarray,
                    eval_mask: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, MRE on eval_mask positions.

    Parameters
    ----------
    X_hat : (N, T, D) imputed values
    X_true : (N, T, D) ground truth (original X before masking)
    eval_mask : (N, T, D) 1 at positions to evaluate

    Returns
    -------
    dict with keys: mae, rmse, mre
    """
    mask = eval_mask.astype(bool)
    y_hat = X_hat[mask]
    y_true = X_true[mask]

    mae = float(np.mean(np.abs(y_hat - y_true)))
    rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
    denom = float(np.mean(np.abs(y_true)))
    mre = mae / denom if denom > 1e-8 else float("nan")
    return {"mae": mae, "rmse": rmse, "mre": mre}


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(results: dict, out_path: str) -> None:
    """Write results dict as JSON, creating parent directories as needed."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def save_imputed(X_imputed: np.ndarray, out_path: str) -> None:
    """Save imputed numpy array."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, X_imputed.astype(np.float32))


# ---------------------------------------------------------------------------
# Config iteration helpers
# ---------------------------------------------------------------------------

ALL_DATASETS = ["physionet2012", "beijing"]

MISSING_RATES = [0.1, 0.3, 0.5, 0.7, 0.9]

MISSING_PATTERNS = {
    "physionet2012": ["mcar", "block5", "block10", "block20"],
    "beijing":       ["mcar"],
    "electricity":   ["block10", "block20"],   # block only; MCAR handled separately
    "mimic":         ["block20"],
    "cmapss":        ["mcar", "block10", "block20"],
    "tep":           ["mcar", "block10", "block20"],
}

BLOCK_LENS = {"block5": 5, "block10": 10, "block20": 20,
              "block25": 25, "block30": 30, "block40": 40}


def iter_configs(datasets=None, rates=None):
    """
    Yield (dataset, missing_rate, missing_pattern) tuples for all experiments.
    beijing only uses mcar pattern.
    """
    if datasets is None:
        datasets = ALL_DATASETS
    if rates is None:
        rates = MISSING_RATES
    for ds in datasets:
        patterns = MISSING_PATTERNS.get(ds, ["mcar"])
        for rate in rates:
            for pat in patterns:
                yield ds, rate, pat


def metrics_filename(method: str, dataset: str, rate: float,
                     pattern: str) -> str:
    rate_str = f"{int(rate * 100):02d}"
    return f"{method}_{dataset}_{rate_str}_{pattern}_metrics.json"


def imputed_filename(method: str, dataset: str, rate: float,
                     pattern: str) -> str:
    rate_str = f"{int(rate * 100):02d}"
    return f"{method}_{dataset}_{rate_str}_{pattern}_imputed.npy"
