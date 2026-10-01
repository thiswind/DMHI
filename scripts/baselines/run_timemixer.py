#!/usr/bin/env python3
"""
TimeMixer baseline runner (PyPOTS library version, v3.2 wrapper).

Replaces the previous subprocess wrapper (which invoked
reference_repos/baselines/timemixer_official/run.py) and inherited that
pipeline's `--task_name imputation` quirk: official TimeMixer evaluates only
on an internally generated MCAR mask, so block-missing cells reported the
MAE on the wrong target (mre=1.0 for all PhysioNet block-* cells). See
the TimeMixer-official rewrite notes
for the full diagnosis.

This rewrite uses PyPOTS' own TimeMixer implementation (`pypots.imputation.TimeMixer`),
which follows the same NaN-encoded-input / standard imputation protocol as our
existing SAITS / BRITS / ImputeFormer runners. The evaluation mask is the
protocol-prescribed block-missing mask, matching every other baseline.

Hyperparameters are mapped one-to-one from the previous TimeMixer-official call,
to keep MCAR cells (already-valid in v3.0.0) numerically comparable across
v3.1.0 → v3.2; the only behavioural change is correct masking for block-* cells.

Usage (single config):
    python run_timemixer.py \\
        --dataset physionet2012 \\
        --missing_rate 0.5 \\
        --missing_pattern block20 \\
        --gpu 0 \\
        --out_dir runs/baselines

Usage (all configs):
    python run_timemixer.py --all_configs --gpu 0 --out_dir runs/baselines
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

# HPs mirror the previous run_timemixer.py call to timemixer_official/run.py
# (e_layers=2, d_model=64, d_ff=128, top_k=3, down_sampling_layers=2,
# down_sampling_window=2, batch_size=64, epochs=100, patience=20). PyPOTS API
# names are slightly different but the underlying TimeMixer model is the same.
HPS = dict(
    n_layers=2,
    d_model=64,
    d_ffn=128,
    top_k=3,
    dropout=0.1,
    channel_independence=False,
    decomp_method="moving_avg",
    moving_avg=5,
    downsampling_layers=2,
    downsampling_window=2,
    apply_nonstationary_norm=False,
    batch_size=64,
    epochs=100,
    patience=20,
    num_workers=0,
)


def run_one(dataset: str, missing_rate: float, missing_pattern: str,
            gpu: int, out_dir: str, seed: int = 42,
            smoke: bool = False) -> dict:
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

    X_ori_val = X_val.copy().astype(np.float32)
    X_ori_val[M_val == 0] = np.nan

    saving_path = str(out_dir / "timemixer" / dataset /
                      f"rate{int(missing_rate*100):02d}_{missing_pattern}")

    from pypots.imputation import TimeMixer
    hps = dict(HPS)
    if smoke:
        hps["epochs"] = 2
        hps["patience"] = 2

    model = TimeMixer(
        n_steps=T,
        n_features=D,
        **hps,
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
        "method": "timemixer",
        "dataset": dataset,
        "missing_rate": missing_rate,
        "missing_pattern": missing_pattern,
        "seed": seed,
        "train_min": round(train_min, 2),
        "infer_ms_per_sample": round(infer_ms, 4),
        "n_params_M": round(n_params / 1e6, 3),
        **metrics,
        "note": "PyPOTS TimeMixer; eval mask is protocol-prescribed (block_* or mcar)",
        "wrapper_version": "v3.2_pypots_library",
    }

    metrics_name = metrics_filename("timemixer", dataset, missing_rate, missing_pattern)
    save_results(results, str(out_dir / metrics_name))

    imputed_name = imputed_filename("timemixer", dataset, missing_rate, missing_pattern)
    save_imputed(X_hat, str(out_dir / imputed_name))

    print(f"[timemixer] {dataset} rate={missing_rate} pat={missing_pattern} "
          f"MAE={metrics['mae']:.4f} MRE={metrics['mre']:.4f} "
          f"train={train_min:.1f}min")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="physionet2012")
    parser.add_argument("--missing_rate", type=float, default=0.5)
    parser.add_argument("--missing_pattern", default="block20")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", default="runs/baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_configs", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 2 epochs, single config only")
    args = parser.parse_args()

    if args.all_configs:
        for ds, rate, pat in iter_configs():
            run_one(ds, rate, pat, args.gpu, args.out_dir, args.seed, smoke=args.smoke)
    else:
        run_one(args.dataset, args.missing_rate, args.missing_pattern,
                args.gpu, args.out_dir, args.seed, smoke=args.smoke)


if __name__ == "__main__":
    main()
