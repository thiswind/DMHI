#!/usr/bin/env python3
"""CPU inference-speed benchmark reproducing Table IV (Exp-4).

Times DMHI per-sample inference on released checkpoints against classical
baselines on identical data and masks, and prints the paper's reference
numbers side by side. Domains are auto-discovered from
``checkpoints/dmhi/<domain>_seed<*>`` intersected with ``data/processed``.

Usage:
    python scripts/benchmark_speed.py                      # all domains
    python scripts/benchmark_speed.py --domain gas_home    # one domain
    python scripts/benchmark_speed.py --n 30               # sample count

Paper Table IV (Apple M4, CPU): DMHI 5.8 ms vs CSDI 4900.6 ms (~840x).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

os.environ.setdefault("DMHI_SILENT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TQDM_DISABLE", "1")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from dmhi.method.pipeline import RiemannianImputer  # noqa: E402

PAPER = {"DMHI_ms": 5.8, "CSDI_cpu_ms": 4900.6, "CSDI_gpu_ms": 823.9,
         "SAITS_gpu_ms": 0.11, "BRITS_gpu_ms": 2.8}
RUN_DIR_RE = re.compile(r"^(?P<domain>.+)_seed(?P<seed>\d+)$")


def _apply_block_missing(M, L, rate, seed):
    from dmhi.utils.eval_protocol import apply_block_missing
    return apply_block_missing(M, L, rate, seed=seed)


def time_call(fn, X, M, warmup=3):
    for i in range(min(warmup, len(X))):
        fn(X[i:i + 1], M[i:i + 1])
    elapsed, reps = 0.0, 0
    for i in range(len(X)):
        t0 = time.perf_counter()
        fn(X[i:i + 1], M[i:i + 1])
        elapsed += time.perf_counter() - t0
        reps += 1
    return elapsed / max(reps, 1) * 1000.0


def np_interp_impute(X, M):
    Xh = X.copy()
    for n in range(Xh.shape[0]):
        for c in range(Xh.shape[2]):
            t = np.arange(Xh.shape[1], dtype=np.float64)
            m = M[n, :, c].astype(bool)
            if 0 < m.sum() < len(t):
                Xh[n, m, c] = np.interp(t[m], t[~m], X[n, ~m, c])
    return Xh


def mean_impute(X, M):
    Xh = X.copy()
    Mb = M.astype(bool)
    for n in range(Xh.shape[0]):
        for c in range(Xh.shape[2]):
            m = Mb[n, :, c]
            Xh[n, ~m, c] = X[n, m, c].mean() if m.any() else 0.0
    return Xh


def discover_runs(domain=None):
    runs = []
    for d in sorted((REPO / "checkpoints" / "dmhi").iterdir()):
        if not d.is_dir():
            continue
        m = RUN_DIR_RE.match(d.name)
        if not m or not (d / "checkpoint.pkl").exists():
            continue
        if domain and m["domain"] != domain:
            continue
        if (REPO / "data" / "processed" / m["domain"]).exists():
            runs.append((m["domain"], int(m["seed"]), d))
    return runs


def main():
    ap = argparse.ArgumentParser(
        description="CPU inference-speed benchmark (paper Table IV)")
    ap.add_argument("--domain", default=None, help="restrict to one domain")
    ap.add_argument("--n", type=int, default=30, help="number of test samples")
    ap.add_argument("--rate", type=float, default=0.9)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    runs = discover_runs(args.domain)
    if not runs:
        sys.exit("no runnable <domain>_seed<*> checkpoints with local data found; "
                 "unpack the release archive under checkpoints/dmhi/ first")

    print("CPU inference-speed benchmark (paper Table IV, Exp-4)")
    print(f"config: rho={args.rate}  L={args.block}  mask_seed=seed+2  "
          f"n_samples={args.n}  device=CPU  threads={os.environ['OMP_NUM_THREADS']}")
    print(f"domains: {len(runs)} (auto-discovered)\n")

    results, worst = {}, 0.0
    print(f"{'domain':22s} {'seed':>4s} {'T':>4s} {'D':>4s} {'DMHI ms':>9s} {'interp':>8s} {'mean':>7s}")
    for dom, seed, rdir in runs:
        d = REPO / "data" / "processed" / dom
        X = np.load(d / "X_test.npy", mmap_mode="r")
        M = np.load(d / "M_test.npy", mmap_mode="r")
        n = min(args.n, len(X))
        Xs = np.asarray(X[:n]).astype(np.float32)
        Ms = np.asarray(M[:n]).astype(np.int8)
        held = _apply_block_missing(np.asarray(M).astype(np.int8),
                                    args.block, args.rate, seed=seed + 2)[:n]
        M_eval = (Ms & ~held).astype(np.int8)
        imp = RiemannianImputer.load(str(rdir / "checkpoint.pkl"))
        ms_dmhi = time_call(lambda x, m: imp.impute(x, m), Xs, M_eval)
        ms_lin = time_call(np_interp_impute, Xs, M_eval)
        ms_mean = time_call(mean_impute, Xs, M_eval)
        worst = max(worst, ms_dmhi)
        print(f"{dom:22s} {seed:4d} {Xs.shape[1]:4d} {Xs.shape[2]:4d} "
              f"{ms_dmhi:9.2f} {ms_lin:8.2f} {ms_mean:7.2f}")
        results[f"{dom}_seed{seed}"] = {
            "T": int(Xs.shape[1]), "D": int(Xs.shape[2]),
            "dmhi_ms": round(ms_dmhi, 3), "interp_ms": round(ms_lin, 3),
            "mean_ms": round(ms_mean, 3)}

    print(f"\nworst-domain DMHI latency: {worst:.2f} ms/sample on CPU")
    print("\n--- paper reference (Table IV, Apple M4) ---")
    print(f"  DMHI (deployed form)     {PAPER['DMHI_ms']:9.1f} ms/sample   CPU")
    print(f"  CSDI                     {PAPER['CSDI_cpu_ms']:9.1f} ms/sample   CPU  "
          f"({PAPER['CSDI_cpu_ms'] / PAPER['DMHI_ms']:.0f}x DMHI)")
    print(f"  CSDI                     {PAPER['CSDI_gpu_ms']:9.1f} ms/sample   GPU  "
          f"({PAPER['CSDI_gpu_ms'] / PAPER['DMHI_ms']:.0f}x DMHI)")
    print(f"  BRITS                    {PAPER['BRITS_gpu_ms']:9.1f} ms/sample   GPU")
    print(f"  SAITS                    {PAPER['SAITS_gpu_ms']:9.1f} ms/sample   GPU")
    print("\nnote: deep baselines are not re-timed here (weights are GPU-trained); "
          "the paper's measured numbers are shown for reference.")

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()
