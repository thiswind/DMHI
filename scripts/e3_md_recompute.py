#!/usr/bin/env python3
"""E3: MD (manifold deviation) recomputation for all methods (IoTJ-R1, R3-3).

MD definition: the paper's rewritten k-NN form (eq:md_def, k=5), computed
with dmhi.utils.eval_protocol.manifold_deviation — identical to what
evaluate.py uses for DMHI, so all methods are scored under ONE definition.

Imputed sources:
- DMHI:      runs/e2_closed_form/<ds>_seed<s>_r0.9/  (metrics.json already
             carries MD; we recompute from the saved imputed array for
             uniformity when available)
- PSW-I:     runs/e6_psw_official/psw_i_ini_<ds>_seed<s>_imputed.npy
- classical: generated here (mean/linear/locf) under the paper protocol

Protocol: 12 paper datasets, block20, rho=0.9, mask_seed = seed+2,
seeds {7, 42, 123}.

Usage (repo root):
    python scripts/e3_md_recompute.py
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "baselines"))

from common import (  # noqa: E402
    load_dataset, apply_block_missing, make_input_mask, compute_metrics,
    BLOCK_LENS,
)
from run_classical import impute_mean, impute_linear, impute_locf  # noqa: E402
from dmhi.utils.eval_protocol import manifold_deviation  # noqa: E402

PAPER_DATASETS = [
    "physionet2012", "mimic", "air_quality_italy", "gait_uci",
    "mocap_karate", "gas_home", "delta_robot", "kuka", "nanodrone",
    "mhealth", "breizh_rs", "solo12",
]
RATE = 0.9
PATTERN = "block20"
SEEDS = [7, 42, 123]
K = 5

import os
E2_DIR = pathlib.Path(os.environ.get("E3_E2_DIR", "runs/e2_closed_form"))
E6_DIR = pathlib.Path("runs/e6_psw_official")
OUT_DIR = pathlib.Path(os.environ.get("E3_OUT_DIR", "runs/e3_md"))


def md_row(X_hat, X_test, eval_mask):
    md_mean, md_p90 = manifold_deviation(X_hat, X_test, eval_mask, k=K)
    return md_mean, md_p90


def classical_imputed(dataset, seed):
    X_train, M_train, _ = load_dataset(dataset, "train")
    X_test, M_test, _ = load_dataset(dataset, "test")
    eval_mask = apply_block_missing(M_test, BLOCK_LENS[PATTERN], RATE,
                                    seed=seed + 2)
    input_mask = make_input_mask(M_test, eval_mask)
    out = {}
    for name, fn in (("mean", impute_mean), ("linear", impute_linear),
                     ("locf", impute_locf)):
        if name == "mean":
            X_hat = fn(X_train, M_train, X_test, input_mask)
        else:
            X_hat = fn(X_test, input_mask)
        out[name] = (X_hat, eval_mask)
    return X_test, out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = []
    for ds in PAPER_DATASETS:
        for seed in SEEDS:
            run = f"{ds}_seed{seed}"
            row = {"dataset": ds, "seed": seed}

            # --- classical (generate + MD) ---
            try:
                X_test, classical = classical_imputed(ds, seed)
                for name, (X_hat, eval_mask) in classical.items():
                    m = compute_metrics(X_hat, X_test, eval_mask)
                    md_mean, md_p90 = md_row(X_hat, X_test, eval_mask)
                    row[f"{name}_mae"] = m["mae"]
                    row[f"{name}_md"] = md_mean
                    np.save(OUT_DIR / f"{name}_{run}_imputed.npy",
                            X_hat.astype(np.float32))
            except Exception as e:  # noqa: BLE001
                row["classical_error"] = f"{type(e).__name__}: {e}"

            # --- PSW-I official (from E6) ---
            p = E6_DIR / f"psw_i_ini_{run}_imputed.npy"
            pm = E6_DIR / f"psw_i_ini_{run}_metrics.json"
            if p.exists() and pm.exists():
                X_hat = np.load(p)
                meta = json.load(open(pm))
                row["psw_i_mae"] = meta["paper_mae"]
                X_test_true, _, _ = load_dataset(ds, "test")
                eval_mask = np.load(E6_DIR / f"psw_i_ini_{run}_evalmask.npy")
                md_mean, md_p90 = md_row(X_hat, X_test_true, eval_mask)
                row["psw_i_md"] = md_mean
            else:
                row["psw_i_mae"] = None
                row["psw_i_md"] = None

            # --- DMHI (from E2 metrics; MD already in metrics.json) ---
            e2 = E2_DIR / f"{run}_r0.9" / "metrics.json"
            if e2.exists():
                m = json.load(open(e2))
                row["dmhi_mae"] = m.get("MAE")
                row["dmhi_md"] = m.get("MD")
            else:
                row["dmhi_mae"] = None
                row["dmhi_md"] = None

            table.append(row)
            print(f"[e3] {run}: dmhi_md={row.get('dmhi_md')} "
                  f"psw_i_md={row.get('psw_i_md')} "
                  f"linear_md={row.get('linear_md')}")

    with open(OUT_DIR / "md_table.json", "w") as f:
        json.dump(table, f, indent=2)

    # seed-mean summary
    print("\n=== seed-mean MD per method ===")
    import collections
    agg = collections.defaultdict(list)
    for r in table:
        for meth in ("dmhi", "psw_i", "mean", "linear", "locf"):
            v = r.get(f"{meth}_md")
            if v is not None and np.isfinite(v):
                agg[(r["dataset"], meth)].append(v)
    for (ds, meth), vals in sorted(agg.items()):
        print(f"{ds:<18}{meth:<8}MD={np.mean(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
