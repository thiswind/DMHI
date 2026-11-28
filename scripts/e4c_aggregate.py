#!/usr/bin/env python3
"""E4c landmark-selection scheme aggregation.

Reads runs/e4c_landmark_scheme/<ds>/<scheme>_seed<7|42|123>/metrics.json,
pairs them with the deployed stratified baseline from E4
(runs/e4_stage2_fixed/<ds>/{rL0.30,k10,gamma0.02}_seed<s> — bit-identical
to each other and to Table II), and prints the sensitivity table plus a
JSON blob for the reply letter.

Usage: python scripts/e4c_aggregate.py [--base runs]
"""
import argparse
import json
import os
import sys

DS_ORDER = ["delta_robot", "physionet2012", "mocap_karate"]
NICE = {"delta_robot": "Delta Robot", "physionet2012": "PhysioNet",
        "mocap_karate": "Mocap"}
SEEDS = (7, 42, 123)


def read_metrics(path):
    with open(path) as fh:
        return json.load(fh)


def collect(base):
    table = {}
    for ds in DS_ORDER:
        row = {"stratified": {}, "uniform": {}, "random": {}}
        # stratified baseline: E4 deployed-config runs (k10 == gamma0.02 == rL0.30)
        for s in SEEDS:
            p = os.path.join(base, "e4_stage2_fixed", ds, f"k10_seed{s}", "metrics.json")
            if os.path.exists(p):
                row["stratified"][s] = read_metrics(p)
        for sch in ("uniform", "random"):
            for s in SEEDS:
                p = os.path.join(base, "e4c_landmark_scheme", ds, f"{sch}_seed{s}", "metrics.json")
                if os.path.exists(p):
                    row[sch][s] = read_metrics(p)
        table[ds] = row
    return table


def mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, var ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    table = collect(args.base)
    print(f"{'domain':13} {'scheme':10} {'n':>2} {'MAE':>8} {'std':>7} {'MSE':>8} {'RMSE':>8}")
    blob = {}
    for ds, row in table.items():
        blob[ds] = {}
        for sch in ("stratified", "uniform", "random"):
            seeds = row[sch]
            if not seeds:
                print(f"{NICE[ds]:13} {sch:10}  0 {'—':>8}")
                continue
            mae_m, mae_s = mean_std([v["MAE"] for v in seeds.values()])
            mse_m, _ = mean_std([v["MSE"] for v in seeds.values()])
            rmse_m, _ = mean_std([v["RMSE"] for v in seeds.values()])
            print(f"{NICE[ds]:13} {sch:10} {len(seeds):2} {mae_m:8.4f} {mae_s:7.4f} {mse_m:8.4f} {rmse_m:8.4f}")
            blob[ds][sch] = {"n": len(seeds),
                             "MAE_mean": round(mae_m, 4), "MAE_std": round(mae_s, 4),
                             "MSE_mean": round(mse_m, 4), "RMSE_mean": round(rmse_m, 4),
                             "seeds": {str(s): {k: round(v, 4) for k, v in m.items()
                                        if k in ("MAE", "MSE", "RMSE", "MD")}
                                       for s, m in seeds.items()}}

    # sensitivity summary: worst-case |ΔMAE| / stratified per domain
    print("\nsensitivity (uniform & random vs stratified, MAE):")
    for ds, row in blob.items():
        base = row.get("stratified", {}).get("MAE_mean")
        for sch in ("uniform", "random"):
            v = row.get(sch, {}).get("MAE_mean")
            if base and v:
                print(f"  {NICE[ds]:13} {sch:10} strat={base:.4f} {sch}={v:.4f} "
                      f"Δ={v - base:+.4f} ({100 * (v - base) / base:+.1f}%)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(blob, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
