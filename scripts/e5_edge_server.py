#!/usr/bin/env python3
"""E5 (server side): peak-RSS + inference latency for Table VI (R1-4/R2-7/8).

Measures per-method peak memory (getrusage maxrss of the worker process)
and per-sample inference time on CPU, matching the Table VI protocol
(rho=0.9, block=20). Methods: DMHI (closed-form), linear interp, mean.
Deep baselines (CSDI/ImputeFormer) are measured on the Raspberry Pi with
inference-only checkpoints; on the shared server their training-free
inference cost is dominated by torch runtime and not reported here.

Usage (server): python scripts/e5_edge_server.py --domain physionet2012
"""
import argparse
import json
import pathlib
import resource
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT_DIR = pathlib.Path("runs/e5_edge")


def peak_rss_mb() -> float:
    """Peak resident set size of THIS process in MiB (maxrss = KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="physionet2012")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rate", type=float, default=0.9)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import pickle
    ds = args.domain
    X_test = np.load(f"data/processed/{ds}/X_test.npy").astype(np.float32)[:args.n]
    M_test = np.load(f"data/processed/{ds}/M_test.npy").astype(np.int8)[:args.n]
    N, T, D = X_test.shape

    from dmhi.method.synth_missing import sample_block_mask
    rng = np.random.default_rng(args.seed)
    masks = []
    for i in range(N):
        ev = sample_block_mask(D, T, seed=args.seed + i, block_size=20,
                               rate=args.rate).astype(np.int8)
        if ev.shape != (T, D):
            ev = ev.T
        masks.append(ev[None] * M_test[i:i + 1])
    M_eval = np.concatenate(masks, axis=0)

    results = {}

    # — DMHI (checkpoint from E2, paper configuration) —
    ck = f"checkpoints/e2/{ds}_seed{args.seed}/checkpoint.pkl"
    rss_before = peak_rss_mb()
    with open(ck, "rb") as f:
        imp = pickle.load(f)
    rss_loaded = peak_rss_mb()
    for i in range(3):  # warmup
        imp.impute(X_test[i:i + 1], M_eval[i:i + 1])
    t0 = time.perf_counter()
    for i in range(N):
        imp.impute(X_test[i:i + 1], M_eval[i:i + 1])
    dmhi_ms = (time.perf_counter() - t0) / N * 1000.0
    results["DMHI"] = {
        "per_sample_ms": round(dmhi_ms, 2),
        "rss_base_mb": round(rss_before, 1),
        "rss_with_ckpt_mb": round(rss_loaded, 1),
        "rss_peak_mb": round(peak_rss_mb(), 1),
    }

    # — linear interpolation (numpy) —
    def np_interp(X, M):
        Xh = X.copy()
        t = np.arange(X.shape[1], dtype=np.float64)
        for c in range(X.shape[2]):
            m = M[0, :, c].astype(bool)
            if 0 < m.sum() < len(t):
                Xh[0, m, c] = np.interp(t[m], t[~m], X[0, ~m, c])
        return Xh

    for i in range(3):
        np_interp(X_test[i:i + 1], M_eval[i:i + 1])
    t0 = time.perf_counter()
    for i in range(N):
        np_interp(X_test[i:i + 1], M_eval[i:i + 1])
    results["Linear"] = {"per_sample_ms": round(
        (time.perf_counter() - t0) / N * 1000.0, 3),
        "rss_peak_mb": round(peak_rss_mb(), 1)}

    # — mean fill —
    t0 = time.perf_counter()
    for i in range(N):
        m = M_eval[i].astype(bool)
        x = X_test[i].copy()
        x[~m] = x[m].mean() if m.any() else 0.0
    results["Mean"] = {"per_sample_ms": round(
        (time.perf_counter() - t0) / N * 1000.0, 3),
        "rss_peak_mb": round(peak_rss_mb(), 1)}

    out = {"domain": ds, "seed": args.seed, "T": int(T), "D": int(D),
           "N": int(N), "rate": args.rate, "methods": results}
    path = OUT_DIR / f"server_{ds}_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
