#!/usr/bin/env python3
"""One-command reproduction of the paper's Table II/III DMHI row.

Runs scripts/evaluate.py for every released checkpoint (12 domains x 3
seeds) at the paper protocol (block missingness, L=20, rho=0.9, mask seed
= run seed + 2 — all read from each run's config.json), averages the three
seeds per domain, and prints the DMHI row in table order with per-run
values so every cell is traceable.

Usage (from the repository root, after unpacking the release zip):
    python scripts/reproduce_table.py [--ckpt_root checkpoints/dmhi] \
        [--domains ...] [--rate 0.9] [--block_size 20]

DUA note: `mimic` and `physionet2012` arrays are not in git but ship as a
release asset (dmhi-dua-arrays-v0.2.0.zip) — unpack under data/processed/
and this script evaluates all 12 domains with zero flags. Without it, the
two gated domains are auto-skipped with a note (data/DATA.md has the
credentialed acquisition route).
"""
import argparse
import glob
import json
import pathlib
import subprocess
import sys

import numpy as np

DOMAINS = [
    "physionet2012", "mimic", "air_quality_italy", "gait_uci", "mocap_karate",
    "gas_home", "delta_robot", "kuka", "nanodrone", "mhealth",
    "breizh_rs", "solo12",
]
DUA = {"mimic", "physionet2012"}
METRICS = ("MAE", "MSE", "RMSE", "MD")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt_root", default="checkpoints/dmhi")
    ckpt_help = "domains to evaluate (default: all 12 in table order)"
    ap.add_argument("--domains", nargs="*", default=None, help=ckpt_help)
    ap.add_argument("--rate", type=float, default=0.9)
    ap.add_argument("--block_size", type=int, default=20)
    args = ap.parse_args()

    domains = args.domains or DOMAINS
    root = pathlib.Path(args.ckpt_root)
    repo = pathlib.Path(__file__).resolve().parents[1]
    rows, skipped = [], []
    for ds in domains:
        run_dirs = sorted(glob.glob(str(root / f"{ds}_seed*")))
        if not run_dirs:
            skipped.append((ds, "no checkpoints found"))
            continue
        if ds in DUA and not (repo / "data/processed" / ds).exists():
            skipped.append((ds, "DUA-gated arrays not in repo (data/DATA.md)"))
            continue
        vals = {m: [] for m in METRICS}
        for rd in sorted(run_dirs, key=lambda p: int(p.rsplit("_seed", 1)[1])):
            cmd = [sys.executable, str(repo / "scripts/evaluate.py"),
                   "--run_dir", rd, "--rate", str(args.rate),
                   "--block_size", str(args.block_size)]
            subprocess.run(cmd, check=True, cwd=repo,
                           stdout=subprocess.DEVNULL)
            m = json.load(open(pathlib.Path(rd) / "metrics.json"))
            for k in METRICS:
                vals[k].append(m[k])
            print(f"  {ds} seed {m['seed']:>3}: " +
                  " ".join(f"{k}={m[k]:.6f}" for k in METRICS), flush=True)
        rows.append((ds, vals))

    print("\n=== Table II/III DMHI row (mean over seeds, paper order) ===")
    for ds, vals in rows:
        mean = {k: float(np.mean(vals[k])) for k in METRICS}
        print(f"{ds:>20}: " + " ".join(f"{k}={mean[k]:.3f}" for k in METRICS))
    if skipped:
        print("\nSkipped:")
        for ds, why in skipped:
            print(f"  {ds}: {why}")
    print("\nCompare against results/README.md (per-run reference) and the "
          "manuscript table (3-decimal cells).")


if __name__ == "__main__":
    main()
