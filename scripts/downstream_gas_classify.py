#!/usr/bin/env python3
"""
downstream_gas_classify.py
=====================
E4f: Correlate Manifold Deviation with downstream home-activity AUROC.

Dataset : gas_home  (X_test shape: 20 × 100 × 10)
Task    : classify home activity (banana / wine / ...) → macro OvR AUROC
Seeds   : 7, 42, 123

Run from the repository root (dependencies installed):
  python scripts/downstream_gas_classify.py

Outputs:
  runs/downstream_gas/downstream_gas.jsonl
  runs/downstream_gas/downstream_gas_summary.csv
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Original server-side analysis runner; adapted to the published layout.


REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from dmhi.utils.eval_protocol import apply_block_missing as _apply_block_missing

DATA_DIR    = REPO / "data/processed/gas_home"
OUT_DIR     = REPO / "runs/downstream_gas"
NEWDS_DIR   = REPO / "runs/newds_v2/gas_home"
OURS_CKPT   = REPO / "runs/recompute033/ours"

SEEDS        = [7, 42, 123]
RATE         = 0.9
RATEPCT      = 90
BLOCK_L      = 20
METHODS_BASELINE = ["brits", "saits", "csdi", "imputeformer", "timemixer", "psw_i"]
METHODS_LOCAL    = ["ours", "linear", "mean"]
METHODS          = METHODS_LOCAL + METHODS_BASELINE


def manifold_deviation_rows(X_hat, X_true, row_mask, k=5, obs_dims=None):
    N, T, D = X_true.shape
    if obs_dims is None:
        ref = X_true.reshape(-1, D)
        q = X_hat[row_mask].reshape(-1, D)
    else:
        ref = X_true[..., obs_dims].reshape(-1, len(obs_dims))
        q = X_hat[row_mask][..., obs_dims]
    rng = np.random.default_rng(0)
    if ref.shape[0] > 30000:
        idx = rng.choice(ref.shape[0], 30000, replace=False)
        ref = ref[idx]
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(ref)
    if q.size == 0:
        return float("nan"), float("nan")
    dists, _ = nn.kneighbors(q)
    per_row = dists.mean(axis=1)
    return float(np.mean(per_row)), float(np.percentile(per_row, 90))

def feature_map(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([X.mean(axis=1), X[:, -1, :]], axis=1)


def impute_linear(X, M):
    N, T, D = X.shape
    X_hat = np.array(X, copy=True)
    t = np.arange(T, dtype=np.float64)
    for n in range(N):
        for d in range(D):
            obs = M[n, :, d].astype(bool)
            if not obs.any():
                X_hat[n, :, d] = 0.0
                continue
            if (~obs).sum() == 0:
                continue
            vals = X[n, obs, d]
            if obs.sum() == 1:
                X_hat[n, ~obs, d] = vals[0]
            else:
                X_hat[n, ~obs, d] = np.interp(t[~obs], t[obs], vals)
    return X_hat


def impute_mean(X, M):
    return np.where(M.astype(bool), X, 0.0).astype(np.float32)


def baseline_path(seed: int, method: str) -> Path:
    return (
        NEWDS_DIR
        / f"seed{seed}"
        / f"{method}_gas_home_{RATEPCT}_block{BLOCK_L}_imputed.npy"
    )


def ours_checkpoint(seed: int) -> Path:
    return OURS_CKPT / f"gas_home_seed{seed}" / "checkpoint.pkl"


def run_method(name, seed, X_test, M_eval, row_mask, obs_dims, clf, scaler, y_test):
    if name == "oracle":
        X_hat = X_test.copy()
    elif name == "ours":
        from dmhi.method.pipeline import RiemannianImputer
        ckpt = ours_checkpoint(seed)
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing ours checkpoint: {ckpt}")
        imputer = RiemannianImputer.load(str(ckpt))
        X_hat = imputer.impute(X_test, M_eval)
    elif name == "linear":
        X_hat = impute_linear(X_test, M_eval)
    elif name == "mean":
        X_hat = impute_mean(X_test, M_eval)
    else:
        p = baseline_path(seed, name)
        if not p.is_file():
            raise FileNotFoundError(f"missing baseline: {p}")
        X_hat = np.load(p).astype(np.float32)
        if X_hat.shape != X_test.shape:
            raise ValueError(f"shape mismatch: {X_hat.shape} vs {X_test.shape}")

    X_hat = np.nan_to_num(X_hat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if name == "oracle":
        md_mean = 0.0
    else:
        finite = np.isfinite(X_hat)
        rmrows = row_mask & finite[..., obs_dims].all(axis=2)
        md_mean, _ = manifold_deviation_rows(X_hat, X_test, rmrows, k=5,
                                             obs_dims=obs_dims)

    feats  = scaler.transform(feature_map(X_hat))
    n_cls  = len(np.unique(y_test))
    proba  = clf.predict_proba(feats)
    if n_cls == 2:
        auroc = float(roc_auc_score(y_test, proba[:, 1]))
    else:
        auroc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
    return md_mean, auroc


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path   = OUT_DIR / "downstream_gas.jsonl"
    summary_path = OUT_DIR / "downstream_gas_summary.csv"

    X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_test  = np.load(DATA_DIR / "X_test.npy").astype(np.float32)
    y_test  = np.load(DATA_DIR / "y_test.npy")
    M_test  = np.load(DATA_DIR / "M_test.npy").astype(np.int8)
    obs_dims = np.flatnonzero(M_test.sum(axis=(0, 1)) > 0)

    print(f"[E4f] X_train={X_train.shape} y_train counts={dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"[E4f] X_test={X_test.shape}  y_test counts={dict(zip(*np.unique(y_test, return_counts=True)))}")

    scaler = StandardScaler()
    X_tr_feat = scaler.fit_transform(feature_map(X_train))
    clf = LogisticRegression(max_iter=2000, multi_class="ovr")
    clf.fit(X_tr_feat, y_train)

    all_methods = METHODS + ["oracle"]
    rows:    list[dict] = []
    skipped: list[str]  = []

    for seed in SEEDS:
        held     = _apply_block_missing(M_test, BLOCK_L, RATE, seed=seed + 2)
        M_eval   = (M_test & ~held).astype(np.int8)
        row_mask = held.astype(bool).any(axis=2)

        for method in all_methods:
            key = f"seed={seed} method={method}"
            try:
                md_mean, auroc = run_method(
                    method, seed, X_test, M_eval, row_mask, clf, scaler, y_test
                )
                rows.append({"seed": seed, "method": method, "md_mean": md_mean, "auroc": auroc})
                print(f"OK  {key:<35}  md={md_mean:.4f}  auroc={auroc:.4f}")
            except Exception as e:
                skipped.append(f"{key}: {e}")
                print(f"SKIP {key}: {e}", file=sys.stderr)

    with jsonl_path.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")

    agg_md:  dict[str, list[float]] = defaultdict(list)
    agg_auc: dict[str, list[float]] = defaultdict(list)
    for rec in rows:
        agg_md[rec["method"]].append(rec["md_mean"])
        agg_auc[rec["method"]].append(rec["auroc"])

    summary_rows = []
    for method in sorted(agg_md, key=lambda m: np.mean(agg_md[m])):
        summary_rows.append({
            "method":      method,
            "md_mean_avg": float(np.mean(agg_md[method])),
            "auroc_avg":   float(np.mean(agg_auc[method])),
            "n_seed":      len(agg_md[method]),
        })

    with summary_path.open("w") as f:
        f.write("method,md_mean_avg,auroc_avg,n_seed\n")
        for sr in summary_rows:
            f.write(f"{sr['method']},{sr['md_mean_avg']:.8f},{sr['auroc_avg']:.8f},{sr['n_seed']}\n")

    md_vals  = np.array([sr["md_mean_avg"] for sr in summary_rows])
    auc_vals = np.array([sr["auroc_avg"] for sr in summary_rows])
    rx = np.argsort(np.argsort(md_vals))
    ry = np.argsort(np.argsort(auc_vals))
    xm, ym = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    sp = float((xm * ym).sum() / denom) if denom else float("nan")

    print(f"\n=== downstream_gas_summary (sorted by md_mean_avg) ===")
    print(f"{'method':<14} {'md_mean_avg':>12} {'auroc_avg':>10} {'n_seed':>6}")
    for sr in summary_rows:
        print(f"{sr['method']:<14} {sr['md_mean_avg']:12.6f} {sr['auroc_avg']:10.4f} {sr['n_seed']:6d}")
    print(f"\nSpearman(MD, AUROC) = {sp:.4f}  (n={len(summary_rows)} methods)")

    if skipped:
        print("\n=== Skipped ===", file=sys.stderr)
        for s in skipped:
            print(s, file=sys.stderr)


if __name__ == "__main__":
    main()
