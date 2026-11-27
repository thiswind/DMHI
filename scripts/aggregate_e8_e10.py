#!/usr/bin/env python3
"""E8/E10 aggregator: turn runs/e8_csdi + runs/e10_e7_deep JSONs into the
markdown tables that backfill the reply material (R1-2 Table C CSDI column,
R1-1 Table A/B deep-baseline columns, R1-4 deep-mem VRAM column).

Rule: ALL material numbers come from this script's stdout — never hand-copy
(the hand-copy ban). Reads only archived JSONs; safe to rerun any time.

Usage (repo root):
    python scripts/aggregate_e8_e10.py            # print all tables
    python scripts/aggregate_e8_e10.py --check    # exit 1 if incomplete
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "baselines"))
sys.path.insert(0, str(HERE.parent / "src"))

from dmhi.utils.eval_protocol import manifold_deviation  # noqa: E402

E8_DIR = pathlib.Path("runs/e8_csdi")
E10_DIR = pathlib.Path("runs/e10_e7_deep")

PAPER_DATASETS = [
    "physionet2012", "mimic", "air_quality_italy", "gait_uci",
    "mocap_karate", "gas_home", "delta_robot", "kuka", "nanodrone",
    "mhealth", "breizh_rs", "solo12",
]
E7_DATASETS = ["air_quality_italy", "physionet2012"]
SEEDS = [7, 42, 123]
DEEP_METHODS = ["imputeformer", "saits", "timemixer", "csdi"]
PROTOCOLS = ["native_pattern", "group2x12", "group2x24",
             "group4x12", "group4x24", "group8x12"]


def table_c_csdi_column():
    """CSDI MD (unified k=5 recomputation) per domain, 3-seed mean."""
    from common import load_dataset
    rows, missing = [], []
    for ds in PAPER_DATASETS:
        vals = []
        for seed in SEEDS:
            tag = f"csdi_{ds}_seed{seed}"
            p = E8_DIR / f"{tag}_imputed.npy"
            m = E8_DIR / f"{tag}_evalmask.npy"
            if not (p.exists() and m.exists()):
                missing.append(tag)
                continue
            X_hat = np.load(p)
            hide = np.load(m)
            X_test, _, _ = load_dataset(ds, "test")
            md_mean, _ = manifold_deviation(X_hat, X_test, hide, k=5)
            vals.append(md_mean)
        if vals:
            rows.append((ds, sum(vals) / len(vals)))
    return rows, missing


def table_ab_deep_columns():
    """Deep-baseline MAE per (dataset, protocol), 3-seed mean."""
    out = {}
    missing = []
    for ds in E7_DATASETS:
        for method in DEEP_METHODS:
            per_proto = {p: [] for p in PROTOCOLS}
            for seed in SEEDS:
                f = E10_DIR / f"e10_{ds}_seed{seed}_{method}.json"
                if not f.exists():
                    missing.append(f.name)
                    continue
                res = json.load(open(f))
                for p in PROTOCOLS:
                    v = res.get("extra_masks", {}).get(p, {}).get("mae")
                    if v is not None:
                        per_proto[p].append(v)
            out[(ds, method)] = {
                p: (sum(v) / len(v) if v else None) for p, v in per_proto.items()}
    return out, missing


def vram_column():
    """Peak VRAM per domain from e8 metrics (torch allocator + nvidia-smi)."""
    rows, missing = [], []
    for ds in PAPER_DATASETS:
        vals_t, vals_s = [], []
        for seed in SEEDS:
            f = E8_DIR / f"csdi_{ds}_seed{seed}_metrics.json"
            if not f.exists():
                missing.append(f.name)
                continue
            m = json.load(open(f))
            vals_t.append(m.get("vram_torch_mb"))
            vals_s.append(m.get("vram_smi_mb"))
        if vals_t:
            rows.append((ds, max(x for x in vals_t if x is not None),
                         max(x for x in vals_s if x is not None)))
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    all_missing = []

    rows, miss = table_c_csdi_column()
    all_missing += miss
    print("== Table C CSDI column (unified MD, k=5, 3-seed mean) ==")
    print("| Domain | CSDI MD |")
    print("|---|---|")
    for ds, v in rows:
        print(f"| {ds} | {v:.3f} |")

    ab, miss = table_ab_deep_columns()
    all_missing += miss
    print("\n== Table A/B deep-baseline columns (MAE, 3-seed mean) ==")
    for ds in E7_DATASETS:
        print(f"\n### {ds}")
        print("| Method | " + " | ".join(PROTOCOLS) + " |")
        print("|---" * (len(PROTOCOLS) + 1) + "|")
        for method in DEEP_METHODS:
            cells = [f"{ab[(ds, method)][p]:.4f}" if ab[(ds, method)][p] is not None
                     else "-" for p in PROTOCOLS]
            print(f"| {method} | " + " | ".join(cells) + " |")

    vr, miss = vram_column()
    all_missing += miss
    print("\n== CSDI peak VRAM (max over 3 seeds) ==")
    print("| Domain | torch alloc MB | nvidia-smi MB |")
    print("|---|---|---|")
    for ds, t, s in vr:
        print(f"| {ds} | {t} | {s} |")

    print(f"\ncomplete: {len(all_missing) == 0} (missing {len(all_missing)})")
    for m_ in all_missing[:20]:
        print("  missing:", m_)
    if args.check and all_missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
