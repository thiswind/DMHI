#!/usr/bin/env python3
"""E9: incremental-update A/B experiment (R2-6/R2-9).

Protocol (matches the paper's long-sequence setting, fig:longseq):
- Stream = test samples concatenated along time (T grows 200 -> ...).
- Windows of length W = TRAIN sequence length (the imputer's reference
  structures are indexed at that length; see stitch.py), stride S = W//4,
  tail-anchored tiling.
- A (incremental): IncrementalStitcher caches windows; each arrival of S
  steps adds exactly ONE new window, all older windows are reused.
- B (from-scratch): recompute the whole tiling each arrival.
Records per-arrival timing, imputer-call counts, fresh-segment MAE, and
max|A - B| (must be 0: identical tiling and math, cache-only difference).

Usage (server, repo root):
    python scripts/e9_incremental_ab.py --dataset mocap_karate --seed 7
"""
import argparse
import json
import pathlib
import pickle
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "baselines"))

from common import load_dataset, apply_block_missing, BLOCK_LENS  # noqa: E402
from dmhi.method.incremental import IncrementalStitcher  # noqa: E402

OUT_DIR = pathlib.Path("runs/e9_incremental")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rate", type=float, default=0.9)
    ap.add_argument("--stream_samples", type=int, default=8,
                    help="test samples concatenated into the stream")
    ap.add_argument("--batch", type=int, default=64,
                    help="samples per impute call (N dimension)")
    ap.add_argument("--ckpt", default=None,
                    help="E2 checkpoint pickle (default: checkpoints/e2/<ds>_seed<n>)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, M_train, _ = load_dataset(args.dataset, "train")
    X_test, M_test, _ = load_dataset(args.dataset, "test")
    X_test = X_test.astype(np.float32)
    N, T_sample, D = X_test.shape
    W = T_train = X_train.shape[1]           # window = train length
    S = max(1, W // 4)

    eval_mask = apply_block_missing(M_test.astype(np.int8),
                                    BLOCK_LENS["block20"], args.rate,
                                    seed=args.seed + 2)
    input_mask = (M_test.astype(np.int8) * eval_mask).astype(np.int8)

    # stream: concatenate `stream_samples` test samples along time
    reps = args.stream_samples
    sel = np.arange(min(args.batch, N))
    X_stream = np.concatenate([X_test[sel]] * reps, axis=1)      # (B, reps*T, D)
    M_stream = np.concatenate([input_mask[sel]] * reps, axis=1)
    X_true = np.concatenate([X_test[sel]] * reps, axis=1)
    M_full = np.concatenate([M_test[sel].astype(np.int8)] * reps, axis=1)
    T_total = X_stream.shape[1]

    ckpt_path = args.ckpt
    if not ckpt_path:
        for cand in (f"checkpoints/dmhi/{args.dataset}_seed{args.seed}/checkpoint.pkl",
                     f"checkpoints/e2/{args.dataset}_seed{args.seed}/checkpoint.pkl"):
            if pathlib.Path(cand).exists():
                ckpt_path = cand
                break
    if not ckpt_path:
        sys.exit(f"checkpoint for {args.dataset} seed {args.seed} not found — "
                 "unpack the release zip under checkpoints/dmhi/ or pass --ckpt")
    from dmhi.method.pipeline import RiemannianImputer
    imp = RiemannianImputer.load(ckpt_path)
    print(f"[e9] {args.dataset}: W={W} S={S} stream T={T_total} "
          f"(N={len(sel)}) imputer={ckpt_path}")

    inc = IncrementalStitcher(imp, W=W, S=S)
    steps = []
    t0 = W                                        # first full window
    out = inc.append(X_stream[:, :t0, :], M_stream[:, :t0, :])

    def seg_mae(pred, lo, hi):
        m = (M_stream[:, lo:hi, :] == 0) & (M_full[:, lo:hi, :] == 1)
        if m.sum() == 0:
            return None
        return float(np.abs(pred[:, lo:hi, :][m] - X_true[:, lo:hi, :][m]).mean())

    for t1 in range(t0 + S, T_total + 1, S):
        ts = time.perf_counter()
        out = inc.append(X_stream[:, t1 - S:t1, :], M_stream[:, t1 - S:t1, :])
        t_inc = time.perf_counter() - ts

        ts = time.perf_counter()
        ref = inc.from_scratch()
        t_scr = time.perf_counter() - ts

        dev = float(np.max(np.abs(out - ref))) if out.shape == ref.shape else None
        steps.append({
            "T": t1,
            "t_incremental_s": round(t_inc, 4),
            "t_scratch_s": round(t_scr, 4),
            "imputer_calls_total": inc.stats["imputer_calls"],
            "windows_now": len(inc._tail_anchored_starts(t1)),
            "new_seg_mae": seg_mae(out, t1 - S, t1),
            "max_abs_dev_vs_scratch": dev,
        })
        print(f"[e9] T={t1:5d} inc={t_inc:6.3f}s scr={t_scr:7.3f}s "
              f"calls={inc.stats['imputer_calls']:3d} "
              f"win={steps[-1]['windows_now']:3d} dev={dev:.1e}", flush=True)

    n_calls = inc.stats["imputer_calls"]
    n_scr = sum(s["windows_now"] for s in steps)
    summary = {
        "dataset": args.dataset, "seed": args.seed, "W": W, "S": S,
        "final_T": T_total, "imputer_calls_incremental": n_calls,
        "imputer_calls_scratch_equiv": n_scr,
        "speedup_calls": round(n_scr / max(n_calls, 1), 2),
        "mean_t_incremental_s": round(
            float(np.mean([s["t_incremental_s"] for s in steps])), 4),
        "mean_t_scratch_s": round(
            float(np.mean([s["t_scratch_s"] for s in steps])), 4),
        "mean_new_seg_mae": round(
            float(np.nanmean([s["new_seg_mae"] for s in steps])), 4),
        "max_dev": max(s["max_abs_dev_vs_scratch"] for s in steps),
        "steps": steps,
    }
    out_path = OUT_DIR / f"{args.dataset}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "steps"},
                     indent=1))


if __name__ == "__main__":
    main()
