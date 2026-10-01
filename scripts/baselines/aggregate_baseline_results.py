#!/usr/bin/env python3
"""
Aggregate all per-method metrics.json files into baseline_summary.json.

Also merges downstream_results.json (AUC-ROC / F1) if it exists.

Writes: runs/baselines/baseline_summary.json

Usage:
    conda activate base
    python scripts/baselines/aggregate_baseline_results.py \\
        --out_dir runs/baselines
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import iter_configs, metrics_filename

CLASSICAL_METHODS = ["mean", "linear", "locf"]
GPU_METHODS = ["saits", "brits", "csdi", "imputeformer", "timemixer", "psw_i"]
ALL_METHODS = CLASSICAL_METHODS + GPU_METHODS


def load_downstream(out_dir: pathlib.Path) -> dict:
    dp = out_dir / "downstream_results.json"
    if dp.exists():
        with open(dp) as f:
            return json.load(f)
    return {}


def aggregate(out_dir: str) -> dict:
    out_dir = pathlib.Path(out_dir)
    downstream = load_downstream(out_dir)

    summary = {}

    for method in ALL_METHODS:
        method_data = {}
        for ds, rate, pat in iter_configs():
            fname = metrics_filename(method, ds, rate, pat)
            fpath = out_dir / fname
            if not fpath.exists():
                continue
            try:
                with open(fpath) as f:
                    entry = json.load(f)
            except Exception:
                continue

            key = f"{ds}|rate={rate:.1f}|pat={pat}"
            method_data[key] = {
                "mae": entry.get("mae"),
                "rmse": entry.get("rmse"),
                "mre": entry.get("mre"),
                "train_min": entry.get("train_min"),
                "infer_ms_per_sample": entry.get("infer_ms_per_sample"),
                "n_params_M": entry.get("n_params_M"),
            }

        for ds_key, scores in downstream.items():
            m, ds = ds_key.split("/", 1) if "/" in ds_key else (ds_key, "unknown")
            if m == method:
                method_data[f"downstream|{ds}"] = scores

        if method_data:
            summary[method] = method_data

    return summary


def print_table(summary: dict):
    print("\n=== Baseline Results Summary ===")
    print(f"{'Method':<15} {'Config':<35} {'MAE':>8} {'RMSE':>8} {'MRE':>8}")
    print("-" * 76)
    for method, data in summary.items():
        for config_key, vals in data.items():
            if "downstream" in config_key:
                auc = vals.get("auc_roc", "N/A")
                f1 = vals.get("f1", "N/A")
                print(f"{method:<15} {config_key:<35} AUC={auc}  F1={f1}")
            else:
                mae = f"{vals['mae']:.4f}" if vals.get("mae") is not None else "N/A"
                rmse = f"{vals['rmse']:.4f}" if vals.get("rmse") is not None else "N/A"
                mre = f"{vals['mre']:.4f}" if vals.get("mre") is not None else "N/A"
                print(f"{method:<15} {config_key:<35} {mae:>8} {rmse:>8} {mre:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="runs/baselines")
    args = parser.parse_args()

    summary = aggregate(args.out_dir)

    out_path = pathlib.Path(args.out_dir) / "baseline_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[aggregate] Wrote {out_path}")

    print_table(summary)


if __name__ == "__main__":
    main()
