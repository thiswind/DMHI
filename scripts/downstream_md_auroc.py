#!/usr/bin/env python3
"""E4b: correlate Manifold Deviation with downstream mortality AUROC (PhysioNet2012)."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

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

DATA_DIR = REPO / "data/processed/physionet2012"
OUT_DIR = REPO / "runs/downstream_md"
BASELINE_DIR = REPO / "runs/bench_v2/physionet2012"
OURS_DIR = REPO / "runs/recompute033/ours"

SEEDS = [7, 42, 123]
RATE = 0.9
RATEPCT = 90
BLOCK_L = 20
METHODS_BASELINE = ["brits", "saits", "csdi", "imputeformer", "timemixer", "psw_i"]
METHODS_LOCAL = ["ours", "linear", "mean"]
METHODS = METHODS_LOCAL + METHODS_BASELINE


def load_labels(path: Path) -> np.ndarray:
    y = np.load(path)
    if y.ndim == 2 and y.shape[1] > 0:
        y = y[:, -1]
    return y.astype(int).reshape(-1)

try:
    import temp_script.plan047_geom_ablation as P47

    def manifold_deviation_rows(X_hat, X_true, row_mask, k=5, obs_dims=None):
        return P47.manifold_deviation_rows(X_hat, X_true, row_mask, k=k,
                                           obs_dims=obs_dims)
except Exception:
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
    """Per sample: [mean over time per channel; last timestep per channel] -> (N, 2D)."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean_t = X.mean(axis=1)
    last_t = X[:, -1, :]
    return np.concatenate([mean_t, last_t], axis=1)


def impute_linear(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """M: 1=observed, 0=missing."""
    N, T, D = X.shape
    X_hat = np.array(X, copy=True)
    t = np.arange(T, dtype=np.float64)
    for n in range(N):
        for d in range(D):
            obs = M[n, :, d].astype(bool)
            if not obs.any():
                X_hat[n, :, d] = 0.0
                continue
            vals = X[n, obs, d]
            if (~obs).sum() == 0:
                continue
            if obs.sum() == 1:
                X_hat[n, ~obs, d] = vals[0]
            else:
                X_hat[n, ~obs, d] = np.interp(t[~obs], t[obs], vals)
    return X_hat


def impute_mean(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    X_hat = np.where(M.astype(bool), X, 0.0)
    return X_hat.astype(np.float32)


def baseline_path(seed: int, method: str) -> Path:
    """Locate realized-mask baseline output for (seed, method)."""
    return (
        BASELINE_DIR
        / f"seed{seed}"
        / f"{method}_physionet2012_{RATEPCT}_block{BLOCK_L}_imputed.npy"
    )


def ours_checkpoint(seed: int) -> Path:
    return OURS_DIR / f"physionet2012_seed{seed}" / "checkpoint.pkl"


def run_method(
    name: str,
    seed: int,
    X_test: np.ndarray,
    M_eval: np.ndarray,
    row_mask: np.ndarray,
    obs_dims: np.ndarray,
    clf: LogisticRegression,
    scaler: StandardScaler,
    y_test: np.ndarray,
) -> tuple[float, float]:
    if name == "oracle":
        X_hat = X_test
    elif name == "ours":
        from dmhi.method.pipeline import RiemannianImputer

        ckpt = ours_checkpoint(seed)
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing {ckpt}")
        imputer = RiemannianImputer.load(str(ckpt))
        X_hat = imputer.impute(X_test, M_eval)
    elif name == "linear":
        X_hat = impute_linear(X_test, M_eval)
    elif name == "mean":
        X_hat = impute_mean(X_test, M_eval)
    else:
        p = baseline_path(seed, name)
        if not p.is_file():
            raise FileNotFoundError(f"missing {p}")
        X_hat = np.load(p).astype(np.float32)
        if X_hat.shape != X_test.shape:
            raise ValueError(f"shape {X_hat.shape} != {X_test.shape}")

    X_md = X_hat.astype(np.float32, copy=False)
    X_safe = np.nan_to_num(X_md, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not np.isfinite(X_safe).all():
        raise ValueError("non-finite values after nan_to_num")

    if name == "oracle":
        md_mean = 0.0
    else:
        finite = np.isfinite(X_md)
        rmrows = row_mask & finite[..., obs_dims].all(axis=2)
        md_mean, _ = manifold_deviation_rows(X_safe, X_test, rmrows, k=5,
                                             obs_dims=obs_dims)

    feats = scaler.transform(feature_map(X_safe))
    proba = clf.predict_proba(feats)[:, 1]
    auroc = float(roc_auc_score(y_test, proba))
    return md_mean, auroc


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "downstream_md.jsonl"
    summary_path = OUT_DIR / "downstream_md_summary.csv"

    X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
    y_train = load_labels(DATA_DIR / "y_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy").astype(np.float32)
    y_test = load_labels(DATA_DIR / "y_test.npy")
    M_test = np.load(DATA_DIR / "M_test.npy").astype(np.int8)
    obs_dims = np.flatnonzero(M_test.sum(axis=(0, 1)) > 0)
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    X_tr_feat = scaler.fit_transform(feature_map(X_train))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr_feat, y_train)

    all_methods = METHODS + ["oracle"]
    rows: list[dict] = []
    skipped: list[str] = []

    for seed in SEEDS:
        held = _apply_block_missing(M_test, BLOCK_L, RATE, seed=seed + 2)
        M_eval = (M_test & ~held).astype(np.int8)
        row_mask = held.astype(bool).any(axis=2)

        for method in all_methods:
            key = f"seed={seed} method={method}"
            try:
                md_mean, auroc = run_method(
                    method, seed, X_test, M_eval, row_mask, obs_dims, clf, scaler, y_test
                )
                rec = {
                    "seed": seed,
                    "method": method,
                    "md_mean": md_mean,
                    "auroc": auroc,
                    "n_test": n_test,
                }
                rows.append(rec)
                print(f"OK {key} md_mean={md_mean:.6f} auroc={auroc:.4f}")
            except Exception as e:
                skipped.append(f"{key}: {e}")
                print(f"SKIP {key}: {e}", file=sys.stderr)

    with jsonl_path.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")

    # aggregate per method
    from collections import defaultdict

    agg_md: dict[str, list[float]] = defaultdict(list)
    agg_auc: dict[str, list[float]] = defaultdict(list)
    for rec in rows:
        agg_md[rec["method"]].append(rec["md_mean"])
        agg_auc[rec["method"]].append(rec["auroc"])

    summary_rows = []
    for method in sorted(agg_md.keys(), key=lambda m: np.nanmean(agg_md[m])):
        md_arr = np.asarray(agg_md[method], dtype=float)
        auc_arr = np.asarray(agg_auc[method], dtype=float)
        finite_md = np.isfinite(md_arr)
        summary_rows.append(
            {
                "method": method,
                "md_mean_avg": float(np.mean(md_arr[finite_md])) if finite_md.any() else float("nan"),
                "auroc_avg": float(np.mean(auc_arr[np.isfinite(auc_arr)])),
                "n_seed": len(agg_md[method]),
                "n_md_finite": int(finite_md.sum()),
            }
        )

    def fmt_float(v: float) -> str:
        return f"{v:.8f}" if np.isfinite(v) else "NA"

    with summary_path.open("w") as f:
        f.write("method,md_mean_avg,auroc_avg,n_seed,n_md_finite\n")
        for sr in summary_rows:
            f.write(
                f"{sr['method']},{fmt_float(sr['md_mean_avg'])},{fmt_float(sr['auroc_avg'])},{sr['n_seed']},{sr['n_md_finite']}\n"
            )

    print("\n=== downstream_md_summary (sorted by md_mean_avg) ===")
    print(f"{'method':<14} {'md_mean_avg':>12} {'auroc_avg':>10} {'n_seed':>6}")
    for sr in summary_rows:
        print(
            f"{sr['method']:<14} {fmt_float(sr['md_mean_avg']):>12} {fmt_float(sr['auroc_avg']):>10} {sr['n_seed']:6d}"
        )

    # correlation across methods (one point per method = seed-averaged)
    corr_rows = [
        sr for sr in summary_rows
        if np.isfinite(sr["md_mean_avg"]) and np.isfinite(sr["auroc_avg"])
    ]
    methods_for_corr = [sr["method"] for sr in corr_rows]
    md_vals = np.array([sr["md_mean_avg"] for sr in corr_rows])
    auc_vals = np.array([sr["auroc_avg"] for sr in corr_rows])
    M = len(methods_for_corr)

    def pearson(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 2:
            return float("nan")
        xm = x - x.mean()
        ym = y - y.mean()
        denom = np.sqrt((xm**2).sum() * (ym**2).sum())
        if denom == 0:
            return float("nan")
        return float((xm * ym).sum() / denom)

    def spearman(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return pearson(rx, ry)

    sp = spearman(md_vals, auc_vals)
    pe = pearson(md_vals, auc_vals)
    print(
        f"\nSpearman(MD,AUROC)={sp:.4f} Pearson(MD,AUROC)={pe:.4f} (across {M} methods)"
    )

    if skipped:
        print("\n=== Skipped runs ===", file=sys.stderr)
        for s in skipped:
            print(s, file=sys.stderr)


if __name__ == "__main__":
    main()
