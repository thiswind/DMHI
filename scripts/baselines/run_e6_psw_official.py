#!/usr/bin/env python3
"""E6: PSW-I official re-run under the paper protocol (IoTJ-R1, R3-2).

Protocol (frozen in MASTER-PLAN):
- 12 paper datasets, block-20 masks at rho=0.9, mask_seed = seed+2,
  3 seeds {7, 42, 123}, evaluation identical to DMHI (MAE/MSE/RMSE on
  eval_mask).
- Model: OFFICIAL OTImputationIni (benchmark.py full version: FFT spectral
  distance + MVA exponential initializer + uot_mm), vendored in
  psw_i_ini.py from github.com/FMLYD/PSW-I.
- Hyperparameters: official benchmark.py instantiation
  (noise=1e-4, numItermax=1000, stopThr=1e-6) with the official scripts/
  ECL.sh values for time-series (lr=0.05, n_epochs=200, seq_length=48,
  mva_kernel=24, distance=fft, batch_size=256, reg_sk=0.005, n_pairs=2,
  normalize=1, ot_type=uot_mm, reg_m=1, dropout=0).
- StandardScaler fitted on observed entries (official benchmark.py flow).
- Saves per-epoch validation/test MAE/MSE curves (R3-2b deep-baseline
  training-curve evidence) and the final imputed array (E3 MD input).

Runs on CPU (shared GPUs occupied); OTImputationIni is given
device='cpu'.

Usage (repo root):
    python scripts/baselines/run_e6_psw_official.py [--datasets ...] [--seeds ...]
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).parent))          # common.py, psw_i_ini.py
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))  # dmhi package
from common import (  # noqa: E402
    load_dataset, compute_metrics, BLOCK_LENS, save_imputed,
)
from psw_i_ini import OTImputationIni, MVAImputation  # noqa: E402

PAPER_DATASETS = [
    "physionet2012", "mimic", "air_quality_italy", "gait_uci",
    "mocap_karate", "gas_home", "delta_robot", "kuka", "nanodrone",
    "mhealth", "breizh_rs", "solo12",
]
RATE = 0.9
PATTERN = "block20"
SEEDS = [7, 42, 123]

OUT_DIR = pathlib.Path("runs/e6_psw_official")


def official_model():
    """Official configuration (benchmark.py instantiation + ECL.sh values)."""
    initializer = MVAImputation(window_length=24, window_type='exponential')
    return OTImputationIni(
        batch_size=256, lr=0.05, n_epochs=200, n_pairs=2, reg_sk=0.005,
        noise=1e-4, numItermax=1000, stopThr=1e-6, initializer=initializer,
        normalize=1, seq_length=48, distance='fft', ot_type='uot_mm',
        reg_m=1, dropout=0, device='cpu',
    )


def run_one(dataset: str, seed: int) -> dict:
    from dmhi.utils.eval_protocol import apply_block_missing

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_test, M_test, _ = load_dataset(dataset, "test")
    N, T, D = X_test.shape
    eval_mask = apply_block_missing(M_test, BLOCK_LENS[PATTERN], RATE,
                                    seed=seed + 2)

    # Flatten to the (n, d) matrix view the official code expects; hide
    # both originally-missing and eval entries from the model.
    X_flat = X_test.copy().astype(np.float64)
    X_flat[M_test == 0] = np.nan
    X_flat[eval_mask == 1] = np.nan
    X_2d = X_flat.reshape(-1, D)
    X_all_2d = X_test.reshape(-1, D).astype(np.float64)

    # Official flow: StandardScaler on observed entries.
    obs = np.isfinite(X_2d)
    mu = np.nanmean(X_2d, axis=0)
    sd = np.nanstd(X_2d, axis=0)
    sd[sd < 1e-12] = 1.0
    X_2d_scaled = (X_2d - mu) / sd  # NaN-safe arithmetic; observed entries exact
    X_all_scaled = (X_all_2d - mu) / sd

    # Some datasets have channels with NO observed entry left after
    # (original missingness + block mask at rho=0.9). The official MVA
    # initializer loops forever on all-NaN columns (seen: physionet2012
    # stuck 4h). Drop such columns before fitting; fill them back with
    # the training-split mean afterwards (they are unlearnable by ANY
    # method, and the eval_mask never requires them: apply_block_missing
    # only hides observed positions).
    keep = np.flatnonzero(np.isfinite(X_2d_scaled).any(axis=0))
    dropped = D - len(keep)
    X_train_full, M_train_full, _ = load_dataset(dataset, "train")
    col_fill = np.array([X_train_full[:, :, d][M_train_full[:, :, d] == 1].mean()
                         if (M_train_full[:, :, d] == 1).any() else 0.0
                         for d in range(D)], dtype=np.float64)
    X_fit = X_2d_scaled[:, keep]
    X_all_fit = X_all_scaled[:, keep]

    model = official_model()
    t0 = time.time()
    X_hat_fit, mae_off, mse_off, mre_off = model.fit_transform(
        X_fit.copy(), X_all_fit)
    train_min = (time.time() - t0) / 60

    # Inverse-transform; reassemble dropped columns with the train-mean fill.
    X_hat_2d = np.empty((N * T, D), dtype=np.float64)
    X_hat_2d[:, keep] = X_hat_fit * sd[keep] + mu[keep]
    if keep.size < D:
        gone = np.setdiff1d(np.arange(D), keep)
        X_hat_2d[:, gone] = col_fill[gone]
    X_hat = X_hat_2d.reshape(N, T, D).astype(np.float32)
    metrics = compute_metrics(X_hat, X_test, eval_mask)
    metrics["mse"] = metrics["rmse"] ** 2  # common.py omits mse; derive from rmse

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"psw_i_ini_{dataset}_seed{seed}"
    np.save(OUT_DIR / f"{run_name}_imputed.npy", X_hat)
    np.save(OUT_DIR / f"{run_name}_evalmask.npy", eval_mask)

    result = {
        "method": "psw_i_ini_official", "dataset": dataset,
        "missing_rate": RATE, "missing_pattern": PATTERN, "seed": seed,
        "train_min": round(train_min, 2),
        "dropped_allnan_cols": int(dropped),
        "paper_mae": metrics["mae"], "paper_mse": metrics["mse"],
        "paper_rmse": metrics["rmse"], "paper_mre": metrics["mre"],
        "official_scaled_mae": float(mae_off),
        "official_scaled_mse": float(mse_off),
        "official_scaled_mre": float(mre_off),
        "curve_epochs": len(model.history_),
        "curve": model.history_,
    }
    with open(OUT_DIR / f"{run_name}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[e6] {dataset} seed{seed} MAE={metrics['mae']:.4f} "
          f"MSE={metrics['mse']:.4f} train={train_min:.1f}min")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=PAPER_DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = parser.parse_args()

    for ds in args.datasets:
        for seed in args.seeds:
            out = OUT_DIR / f"psw_i_ini_{ds}_seed{seed}_metrics.json"
            if out.exists():
                print(f"[skip] {ds} seed{seed}")
                continue
            try:
                run_one(ds, seed)
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {ds} seed{seed}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
