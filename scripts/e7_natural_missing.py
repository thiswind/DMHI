#!/usr/bin/env python3
"""E7: natural-missingness robustness (R1-1 "correlated multichannel failures").

Part A — native-pattern masking (air_quality_italy):
    The dataset's NATURAL NaN pattern (9.2% overall, per-channel rates) is
    used as the masking generator: each observed cell is hidden with its
    channel's native missing rate (per-channel Bernoulli, calibrated so the
    overall hidden fraction matches the native one). Evaluates imputation
    when fresh data arrives with the SAME missingness statistics as the
    sensor fleet's history.

Part B — synchronized multichannel failures:
    Groups of c in {2, 4, 8} channels fail TOGETHER for a contiguous run of
    len in {6, 12, 24} steps (whole sensor-group outage), applied to observed
    cells only. This is the correlated-failure regime MCAR/block protocols
    miss.

Methods: DMHI (E2 checkpoint, closed-form) vs Linear / LOCF / Mean.
Output: runs/e7_natural/<ds>_seed<n>.json with per-protocol MAE/MSE/RMSE.

Usage: python scripts/e7_natural_missing.py --dataset air_quality_italy --seed 7
"""
import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT_DIR = pathlib.Path("runs/e7_natural")


# ---------------------------------------------------------------- baselines
def linear_impute(X, M):
    """Interpolate; knots = observed cells only (M==1), queries = missing."""
    Xh = X.copy()
    N, T, D = X.shape
    t = np.arange(T, dtype=np.float64)
    for n in range(N):
        for c in range(D):
            obs = M[n, :, c].astype(bool)
            miss = ~obs
            if obs.sum() == 0:
                Xh[n, miss, c] = 0.0
            elif miss.any():
                Xh[n, miss, c] = np.interp(t[miss], t[obs], X[n, obs, c])
    return Xh


def locf_impute(X, M):
    """Last observation carried forward over observed cells only."""
    Xh = X.copy()
    N, T, D = X.shape
    for n in range(N):
        for c in range(D):
            obs = np.where(M[n, :, c] == 1)[0]
            if len(obs) == 0:
                Xh[n, :, c] = 0.0
                continue
            for tt in range(T):
                if M[n, tt, c] == 0:
                    prev = obs[obs < tt]
                    nxt = obs[obs > tt]
                    if len(prev):
                        Xh[n, tt, c] = X[n, prev[-1], c]
                    else:
                        Xh[n, tt, c] = X[n, nxt[0], c]
    return Xh


def mean_impute(X, M):
    """Per-sample mean over observed cells only."""
    Xh = X.copy()
    for n in range(X.shape[0]):
        m = M[n].astype(bool)
        mu = X[n][m].mean() if m.any() else 0.0
        Xh[n][~m] = mu
    return Xh


# ---------------------------------------------------------------- masks
def native_pattern_mask(M_obs: np.ndarray, M_full: np.ndarray,
                        seed: int) -> np.ndarray:
    """Hide observed cells with per-channel rates taken from the dataset's
    natural missing pattern (channels whose native rate is 0 use the global
    native rate)."""
    N, T, D = M_full.shape
    native_missing = (M_full == 0)
    rates = native_missing.sum(axis=(0, 1)) / (N * T)     # per-channel
    glob = native_missing.mean()
    rng = np.random.default_rng(seed)
    hide = np.zeros_like(M_obs, dtype=bool)
    for c in range(D):
        r = rates[c] if rates[c] > 0 else glob
        u = rng.random((N, T)) < r
        hide[:, :, c] = u & (M_obs[:, :, c] == 1)
    return hide


def group_outage_mask(M_obs: np.ndarray, n_ch: int, run_len: int,
                      seed: int, n_events: int = 3) -> np.ndarray:
    """Synchronized multichannel failure: n_ch channels die together for
    run_len contiguous steps, at n_events random (start, group) locations.
    Only observed cells count toward the eval set."""
    N, T, D = M_obs.shape
    rng = np.random.default_rng(seed)
    hide = np.zeros_like(M_obs, dtype=bool)
    for _ in range(n_events):
        n = rng.integers(N)
        t0 = rng.integers(0, max(1, T - run_len))
        grp = rng.choice(D, size=min(n_ch, D), replace=False)
        hide[n, t0:t0 + run_len, grp] = True
    hide &= (M_obs == 1)
    return hide


# ---------------------------------------------------------------- metrics
def scores(Xh, X, hide):
    m = hide.astype(bool)
    if m.sum() == 0:
        return None
    err = Xh[m] - X[m]
    return {"n": int(m.sum()),
            "MAE": round(float(np.abs(err).mean()), 4),
            "MSE": round(float((err ** 2).mean()), 4),
            "RMSE": round(float(np.sqrt((err ** 2).mean())), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = args.dataset
    X = np.load(f"data/processed/{ds}/X_test.npy").astype(np.float32)
    M_full = np.load(f"data/processed/{ds}/M_test.npy").astype(np.int8)
    N, T, D = X.shape

    ck = args.ckpt
    if not ck:
        for cand in (f"checkpoints/dmhi/{ds}_seed{args.seed}/checkpoint.pkl",
                     f"checkpoints/e2/{ds}_seed{args.seed}/checkpoint.pkl"):
            if pathlib.Path(cand).exists():
                ck = cand
                break
    if not ck:
        sys.exit(f"checkpoint for {ds} seed {args.seed} not found — unpack "
                 "the release zip under checkpoints/dmhi/ or pass --ckpt")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from dmhi.method.pipeline import RiemannianImputer
    imp = RiemannianImputer.load(ck)

    protocols = {}
    # — A: native-pattern masking —
    hide_a = native_pattern_mask(M_full, M_full, seed=args.seed)
    # — B: group outages —
    for n_ch, run_len in [(2, 12), (4, 12), (8, 12), (2, 24), (4, 24)]:
        key = f"group{n_ch}x{run_len}"
        protocols[key] = group_outage_mask(
            M_full, n_ch, run_len, seed=args.seed + 17 * n_ch + run_len)
    protocols["native_pattern"] = hide_a

    results = {"dataset": ds, "seed": args.seed, "T": T, "D": D,
               "native_missing_frac": round(float((M_full == 0).mean()), 4),
               "methods": {}}
    for pname, hide in protocols.items():
        if hide.sum() == 0:
            continue
        M_in = (M_full & ~hide).astype(np.int8)
        row = {}
        dmhi_hat = imp.impute(X, M_in)
        row["DMHI"] = scores(dmhi_hat, X, hide)
        row["Linear"] = scores(linear_impute(X, M_in), X, hide)
        row["LOCF"] = scores(locf_impute(X, M_in), X, hide)
        row["Mean"] = scores(mean_impute(X, M_in), X, hide)
        results["methods"][pname] = row
        dm = row["DMHI"]["MAE"] if row["DMHI"] else float("nan")
        best_base = min(v["MAE"] for v in row.values() if v)
        print(f"[e7] {ds} {pname:16s} DMHI MAE={dm:.4f} "
              f"(best baseline {best_base:.4f})", flush=True)

    path = OUT_DIR / f"{ds}_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[e7] saved {path}")


if __name__ == "__main__":
    main()
