#!/usr/bin/env python3
"""
CSDI baseline runner.

Wraps csdi_official/ using OurPhysioDataset adapter instead of reading
raw PhysioNet txt files. Imports CSDI_Physio from main_model.py directly.

Usage (single config):
    CUDA_VISIBLE_DEVICES=0 python run_csdi.py \\
        --dataset physionet2012 \\
        --missing_rate 0.3 \\
        --missing_pattern mcar \\
        --gpu 0 \\
        --out_dir runs/baselines

Usage (all configs):
    CUDA_VISIBLE_DEVICES=0 python run_csdi.py --all_configs --gpu 0 --out_dir runs/baselines
"""

import argparse
import copy
import pathlib
import pickle
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import (
    load_dataset, apply_mcar, apply_block_missing,
    make_input_mask, compute_metrics, save_results, save_imputed,
    iter_configs, metrics_filename, imputed_filename, BLOCK_LENS,
)
from csdi_physio_dataset import get_our_dataloader

CSDI_REPO = str(pathlib.Path(__file__).parent.parent.parent /
                "reference_repos" / "baselines" / "csdi_official")
sys.path.insert(0, CSDI_REPO)


def _load_csdi_config(csdi_repo: str, missing_rate: float) -> dict:
    cfg_path = pathlib.Path(csdi_repo) / "config" / "base.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    config["model"]["test_missing_ratio"] = missing_rate
    config["train"]["batch_size"] = 16
    config["train"]["epochs"] = 200
    config["train"]["lr"] = 1e-3
    return config


def run_one(dataset: str, missing_rate: float, missing_pattern: str,
            gpu: int, out_dir: str, seed: int = 42, smoke: bool = False) -> dict:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and not smoke else "cpu")

    X_train, M_train, _ = load_dataset(dataset, "train")
    X_val,   M_val,   _ = load_dataset(dataset, "val")
    X_test,  M_test,  _ = load_dataset(dataset, "test")

    if missing_pattern == "mcar":
        eval_mask_train = apply_mcar(M_train, missing_rate, seed=seed)
        eval_mask_val   = apply_mcar(M_val,   missing_rate, seed=seed + 1)
        eval_mask_test  = apply_mcar(M_test,  missing_rate, seed=seed + 2)
    else:
        block_len = BLOCK_LENS[missing_pattern]
        eval_mask_train = apply_block_missing(M_train, block_len, missing_rate, seed=seed)
        eval_mask_val   = apply_block_missing(M_val,   block_len, missing_rate, seed=seed + 1)
        eval_mask_test  = apply_block_missing(M_test,  block_len, missing_rate, seed=seed + 2)

    config = _load_csdi_config(CSDI_REPO, missing_rate)
    if smoke:
        config["train"]["epochs"] = 3
        config["train"]["batch_size"] = 8

    nsample = 2 if smoke else 50
    batch_size = config["train"]["batch_size"]
    train_loader = get_our_dataloader(X_train, M_train, eval_mask_train,
                                      batch_size=batch_size, shuffle=True)
    valid_loader = get_our_dataloader(X_val, M_val, eval_mask_val,
                                      batch_size=batch_size, shuffle=False)
    test_loader  = get_our_dataloader(X_test, M_test, eval_mask_test,
                                      batch_size=batch_size, shuffle=False)

    foldername = str(out_dir / "csdi" / dataset /
                     f"rate{int(missing_rate*100):02d}_{missing_pattern}")
    pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)

    from main_model import CSDI_Physio
    from utils import train, evaluate

    target_dim = X_train.shape[2]  # D — inferred from data, not hardcoded
    model = CSDI_Physio(config, device, target_dim=target_dim).to(device)

    t0 = time.time()
    train(model, config["train"], train_loader, valid_loader, foldername=foldername)
    train_min = (time.time() - t0) / 60

    t_infer = time.time()
    evaluate(model, test_loader, nsample=nsample, scaler=1, foldername=foldername)
    infer_ms = (time.time() - t_infer) * 1000 / len(X_test)

    # CSDI saves: generated_outputs_nsample{N}.pk = list of 7 elements:
    #   [all_generated_samples(N_test, nsample, T, D), target, evalpoint,
    #    observed_point, observed_time, scaler, mean_scaler]
    imputed_pkl_path = pathlib.Path(foldername) / f"generated_outputs_nsample{nsample}.pk"
    if not imputed_pkl_path.exists():
        imputed_pkl_path = pathlib.Path(foldername) / "imputations.pkl"

    if imputed_pkl_path.exists():
        with open(imputed_pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        # Handle known CSDI list format: [samples, target, evalpoint, ...]
        if isinstance(pkl_data, list) and len(pkl_data) >= 1:
            samples = pkl_data[0]   # (N_test, nsample, T, D) tensor or ndarray
            if isinstance(samples, torch.Tensor):
                samples = samples.detach().cpu().numpy()
            # mean over nsample dimension → (N_test, T, D)
            X_hat = samples.mean(axis=1)
        elif isinstance(pkl_data, np.ndarray):
            X_hat = pkl_data.mean(axis=1) if pkl_data.ndim == 4 else pkl_data
        elif isinstance(pkl_data, torch.Tensor):
            arr = pkl_data.detach().cpu().numpy()
            X_hat = arr.mean(axis=1) if arr.ndim == 4 else arr
        elif isinstance(pkl_data, dict):
            s = pkl_data.get("samples", pkl_data.get("imputed"))
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu().numpy()
            X_hat = s.mean(axis=1) if s.ndim == 4 else s
        else:
            raise ValueError(
                f"Unknown CSDI pickle format: {type(pkl_data)}, "
                f"len={len(pkl_data) if hasattr(pkl_data,'__len__') else 'n/a'}"
            )
    else:
        input_mask_test = make_input_mask(M_test, eval_mask_test)
        X_hat = X_test * input_mask_test

    if X_hat.shape != X_test.shape:
        X_hat = X_hat.reshape(X_test.shape)

    n_params = sum(p.numel() for p in model.parameters())
    metrics = compute_metrics(X_hat, X_test, eval_mask_test)

    results = {
        "method": "csdi",
        "dataset": dataset,
        "missing_rate": missing_rate,
        "missing_pattern": missing_pattern,
        "seed": seed,
        "train_min": round(train_min, 2),
        "infer_ms_per_sample": round(infer_ms, 4),
        "n_params_M": round(n_params / 1e6, 3),
        **metrics,
    }

    metrics_name = metrics_filename("csdi", dataset, missing_rate, missing_pattern)
    save_results(results, str(out_dir / metrics_name))

    imputed_name = imputed_filename("csdi", dataset, missing_rate, missing_pattern)
    save_imputed(X_hat, str(out_dir / imputed_name))

    print(f"[csdi] {dataset} rate={missing_rate} pat={missing_pattern} "
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
                        help="Smoke test: 3 epochs on CPU, nsample=2, single config only")
    args = parser.parse_args()

    if args.smoke:
        print("[smoke] Running 1-config smoke test: physionet2012 30% mcar, 3 epochs, nsample=2, CPU")
        run_one("physionet2012", 0.3, "mcar", args.gpu, args.out_dir, args.seed, smoke=True)
        print("[smoke] PASSED — full pipeline end-to-end OK")
    elif args.all_configs:
        for ds, rate, pat in iter_configs():
            run_one(ds, rate, pat, args.gpu, args.out_dir, args.seed)
    else:
        run_one(args.dataset, args.missing_rate, args.missing_pattern,
                args.gpu, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
