#!/usr/bin/env python3
"""
PSW-I baseline runner.

PSW-I = Proximal Spectrum Wasserstein Imputation (ICLR 2025).
Uses OTImputationIni class extracted from psw_i_official/benchmark.py.

The core script benchmark_sinkhornfft_val.py is missing from the repo; we
bypass it entirely and call OTImputationIni.fit_transform() directly.

Usage (single config):
    CUDA_VISIBLE_DEVICES=1 python run_psw_i.py \\
        --dataset physionet2012 \\
        --missing_rate 0.3 \\
        --missing_pattern mcar \\
        --gpu 1 \\
        --out_dir runs/baselines

Usage (all configs):
    CUDA_VISIBLE_DEVICES=1 python run_psw_i.py --all_configs --gpu 1 --out_dir runs/baselines
"""

import argparse
import os
import pathlib
import sys
import time

import numpy as np

# MUST set CUDA_VISIBLE_DEVICES before importing torch/hyperimpute
# to ensure hyperimpute picks up the correct device
_gpu_placeholder = None  # set in main() before imports


def _setup_gpu(gpu_id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import (
    load_dataset, apply_mcar, apply_block_missing,
    make_input_mask, compute_metrics, save_results, save_imputed,
    iter_configs, metrics_filename, imputed_filename, BLOCK_LENS,
)

PSW_REPO = str(pathlib.Path(__file__).parent.parent.parent /
               "reference_repos" / "baselines" / "psw_i_official")


def _load_ot_model():
    """
    Import OTImputation from psw_i_official/model.py.
    OTImputationIni is defined inside __main__ block and cannot be imported;
    OTImputation from model.py is the equivalent standalone class.
    """
    sys.path.insert(0, PSW_REPO)
    try:
        from model import OTImputation
        return OTImputation
    except ImportError as e:
        raise ImportError(
            f"Cannot import OTImputation from {PSW_REPO}/model.py: {e}. "
            "Check that psw_i_official repo is present and dependencies are installed."
        )


def run_one(dataset: str, missing_rate: float, missing_pattern: str,
            gpu: int, out_dir: str, seed: int = 42) -> dict:
    _setup_gpu(gpu)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test, M_test, _ = load_dataset(dataset, "test")
    N, T, D = X_test.shape

    if missing_pattern == "mcar":
        eval_mask_test = apply_mcar(M_test, missing_rate, seed=seed + 2)
    else:
        block_len = BLOCK_LENS[missing_pattern]
        eval_mask_test = apply_block_missing(M_test, block_len, missing_rate, seed=seed + 2)

    X_flat = X_test.copy().astype(np.float64)
    X_flat[M_test == 0] = np.nan
    X_flat[eval_mask_test == 1] = np.nan
    X_2d = X_flat.reshape(-1, D)

    OTImputation = _load_ot_model()

    # Use normalize=1 and smaller reg_sk for numerical stability.
    # Large cost matrices (sqeuclidean on high-dim data) cause Sinkhorn to
    # overflow at iteration 0 with reg_sk=1.0; normalizing M and using
    # reg_sk=0.1 avoids NaN losses.
    model = OTImputation(
        lr=0.01,
        n_epochs=100,
        batch_size=512,
        n_pairs=1,
        reg_sk=0.1,
        normalize=1,
    )

    t0 = time.time()
    X_hat_flat = model.fit_transform(X_2d)  # returns numpy (N*T, D)
    train_min = (time.time() - t0) / 60

    X_hat = np.array(X_hat_flat).reshape(N, T, D).astype(np.float32)

    metrics = compute_metrics(X_hat, X_test, eval_mask_test)

    n_params = 0

    results = {
        "method": "psw_i",
        "dataset": dataset,
        "missing_rate": missing_rate,
        "missing_pattern": missing_pattern,
        "seed": seed,
        "train_min": round(train_min, 2),
        "infer_ms_per_sample": 0.0,
        "n_params_M": float(n_params),
        **metrics,
    }

    metrics_name = metrics_filename("psw_i", dataset, missing_rate, missing_pattern)
    save_results(results, str(out_dir / metrics_name))

    imputed_name = imputed_filename("psw_i", dataset, missing_rate, missing_pattern)
    save_imputed(X_hat, str(out_dir / imputed_name))

    print(f"[psw_i] {dataset} rate={missing_rate} pat={missing_pattern} "
          f"MAE={metrics['mae']:.4f} train={train_min:.1f}min")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="physionet2012")
    parser.add_argument("--missing_rate", type=float, default=0.3)
    parser.add_argument("--missing_pattern", default="mcar")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--out_dir", default="runs/baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_configs", action="store_true")
    args = parser.parse_args()

    _setup_gpu(args.gpu)

    if args.all_configs:
        for ds, rate, pat in iter_configs():
            run_one(ds, rate, pat, args.gpu, args.out_dir, args.seed)
    else:
        run_one(args.dataset, args.missing_rate, args.missing_pattern,
                args.gpu, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
