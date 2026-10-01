#!/usr/bin/env python3
"""E4a: MD k-sensitivity. Recompute manifold deviation at k in {3,5,10,20} on the
SAME saved imputed arrays + SAME canonical masks as plan048, and test whether the
method RANKING by MD is stable across k (Spearman vs the k=5 ranking).

Run from the repository root with the full run tree in place (see
scripts/ANALYSIS_RUNNERS.md):
    python scripts/md_k_sensitivity.py --out runs/md_k_sensitivity/md_ksens.jsonl

Reuses: dmhi.utils.eval_protocol.apply_block_missing, the shared
manifold-deviation helper. No retraining; ours re-imputed deterministically
from released checkpoints.
Focus cells: rate in {0.5,0.9}, L in {10,20,30,40}, seeds {7,42,123},
datasets {physionet2012, air_quality_italy, hydraulic, mimic}.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from dmhi.utils.eval_protocol import apply_block_missing as _apply_block_missing  # noqa: E402
from dmhi.method.pipeline import RiemannianImputer  # noqa: E402

KS = [3, 5, 10, 20]
DATASETS = ["physionet2012", "air_quality_italy", "hydraulic", "mimic"]
SEEDS = [7, 42, 123]
ARRAY_METHODS = ["brits", "saits", "csdi", "imputeformer", "timemixer", "psw_i"]
GRID_L = [10, 20, 30, 40]
GRID_RATE = [0.5, 0.9]
OURS_ROOT = REPO / "runs/recompute033/ours"
BASE_ROOT = REPO / "runs/paper_v3/baselines"


def md_rows_multi_k(X_hat, X_true, row_mask, ks, obs_dims=None):
    """Mean k-NN distance of imputed rows to the empirical manifold, for each k."""
    N, T, D = X_hat.shape
    if obs_dims is None:
        ref = X_true.reshape(-1, D)
    else:
        ref = X_true[..., obs_dims].reshape(-1, len(obs_dims))
    if ref.shape[0] > 30000:
        idx = np.random.default_rng(0).choice(ref.shape[0], 30000, replace=False)
        ref = ref[idx]
    if obs_dims is None:
        finite_rows = row_mask & np.isfinite(X_hat).all(axis=2)
        q = X_hat[finite_rows]
    else:
        finite_rows = row_mask & np.isfinite(X_hat)[..., obs_dims].all(axis=2)
        q = X_hat[finite_rows][..., obs_dims]
    out = {}
    if q.shape[0] == 0:
        return {k: float("nan") for k in ks}
    nn = NearestNeighbors(n_neighbors=max(ks), algorithm="auto").fit(ref)
    dists, _ = nn.kneighbors(q)  # (Nq, max_k)
    for k in ks:
        out[k] = float(dists[:, :k].mean())
    return out


def spearman(rank_a, rank_b):
    """Spearman rho between two equal-length rank dicts over the same keys."""
    keys = [k for k in rank_a if k in rank_b]
    n = len(keys)
    if n < 2:
        return float("nan")
    d2 = sum((rank_a[k] - rank_b[k]) ** 2 for k in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/md_k_sensitivity/md_ksens.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w")

    for ds in DATASETS:
        ddir = REPO / "data/processed" / ds
        if not (ddir / "X_test.npy").exists():
            print(f"[skip] {ds}: no data", flush=True)
            continue
        X = np.load(ddir / "X_test.npy").astype(np.float32)
        mtp = ddir / "M_test.npy"
        M = (np.load(mtp).astype(np.int8) if mtp.exists()
             else np.ones_like(X, dtype=np.int8))
        obs_dims = np.flatnonzero(M.sum(axis=(0, 1)) > 0)
        if args.limit:
            X, M = X[:args.limit], M[:args.limit]
        assert not np.isnan(X).any(), f"{ds} X_test NaN"
        for s in SEEDS:
            ck = OURS_ROOT / f"{ds}_seed{s}" / "checkpoint.pkl"
            imputer = RiemannianImputer.load(str(ck)) if ck.exists() else None
            for L in GRID_L:
                for rate in GRID_RATE:
                    rp = int(round(rate * 100))
                    held = _apply_block_missing(M, L, rate, seed=s + 2)
                    em = held.astype(bool)
                    if int(em.sum()) == 0:
                        continue
                    row_mask = em.any(axis=2)
                    M_eval = (M & ~held).astype(np.int8)
                    per_method = {}
                    if imputer is not None:
                        Xh = imputer.impute(X, M_eval)
                        per_method["ours"] = md_rows_multi_k(Xh, X, row_mask, KS, obs_dims)
                    for m in ARRAY_METHODS:
                        ap_ = (BASE_ROOT / f"seed{s}" /
                               f"{m}_{ds}_{rp}_block{L}_imputed.npy")
                        if not ap_.exists():
                            continue
                        arr = np.load(ap_).astype(np.float32)
                        if args.limit:
                            arr = arr[:args.limit]
                        if arr.shape != X.shape:
                            continue
                        per_method[m] = md_rows_multi_k(arr, X, row_mask, KS, obs_dims)
                    if len(per_method) < 2:
                        continue
                    # rankings per k (1 = lowest MD = best)
                    ranks = {}
                    for k in KS:
                        order = sorted(per_method,
                                       key=lambda mm: per_method[mm][k]
                                       if np.isfinite(per_method[mm][k]) else 1e18)
                        ranks[k] = {mm: i + 1 for i, mm in enumerate(order)}
                    rho = {k: spearman(ranks[5], ranks[k]) for k in KS}
                    rec = {"dataset": ds, "seed": s, "L": L, "rate": rate,
                           "md_by_method_k": per_method,
                           "rank_by_k": ranks,
                           "spearman_vs_k5": rho}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"  {ds} s{s} L{L} r{rate} methods={len(per_method)} "
                          f"spearman(k3,k10,k20 vs k5)="
                          f"{rho[3]:.3f},{rho[10]:.3f},{rho[20]:.3f}", flush=True)
    f.close()
    print(f"\nWROTE {out}", flush=True)


if __name__ == "__main__":
    main()
