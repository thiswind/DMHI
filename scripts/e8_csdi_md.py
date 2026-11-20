#!/usr/bin/env python3
"""E8: CSDI imputed-array backfill for the unified MD table (IoTJ-R1, R1-2 residual risk).

Why: Table C (unified MD recomputation) lacks a CSDI column because no CSDI
imputed array was ever saved. This script reruns CSDI under the EXACT e3
protocol (12 paper datasets, block20, rho=0.9, seeds {7,42,123}, mask seed =
seed+2 via run_csdi.py's convention) and saves, per run:
  - runs/e8_csdi/csdi_<ds>_seed<s>_imputed.npy   (X_hat, float32)
  - runs/e8_csdi/csdi_<ds>_seed<s>_evalmask.npy
  - runs/e8_csdi/csdi_<ds>_seed<s>_metrics.json  (mae/mse/... + vram_mb)

VRAM (#5 residual risk): a background thread samples torch.cuda.max_memory_allocated
plus nvidia-smi per-process used memory every 2 s during training+eval; peak stored
in metrics.json as vram_mb (torch allocator) and vram_smi_mb (process-level).

GPU note: server GPUs are shared; pick the freer one via --gpu and keep the
official batch size (CSDIPhysio default 16 is fine under ~4 GiB free).

Usage (server, riemann_baselines env, repo root):
    python scripts/e8_csdi_md.py --gpu 1                 # all 12x3
    python scripts/e8_csdi_md.py --gpu 1 --datasets air_quality_italy --seeds 7
"""
import argparse
import json
import pathlib
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "baselines"))

PAPER_DATASETS = [
    "physionet2012", "mimic", "air_quality_italy", "gait_uci",
    "mocap_karate", "gas_home", "delta_robot", "kuka", "nanodrone",
    "mhealth", "breizh_rs", "solo12",
]
RATE = 0.9
PATTERN = "block20"
OUT_DIR = pathlib.Path("runs/e8_csdi")


class VramSampler(threading.Thread):
    """Sample torch allocator + nvidia-smi process memory until stopped."""

    def __init__(self, gpu: int, interval: float = 2.0):
        super().__init__(daemon=True)
        self.gpu = gpu
        self.interval = interval
        self.stop_flag = threading.Event()
        self.torch_peak_mb = 0.0
        self.smi_peak_mb = 0.0

    def run(self):
        import torch
        while not self.stop_flag.is_set():
            try:
                self.torch_peak_mb = max(
                    self.torch_peak_mb,
                    torch.cuda.max_memory_allocated(self.gpu) / 1048576.0)
            except Exception:
                pass
            try:
                pid = str(__import__("os").getpid())
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                for line in out.strip().splitlines():
                    p, mem = [x.strip() for x in line.split(",")]
                    if p == pid:
                        self.smi_peak_mb = max(self.smi_peak_mb, float(mem))
            except Exception:
                pass
            self.stop_flag.wait(self.interval)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--datasets", nargs="*", default=PAPER_DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 42, 123])
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    import numpy as np
    import run_csdi
    from common import (load_dataset, apply_block_missing, make_input_mask,
                        compute_metrics, BLOCK_LENS)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        for seed in args.seeds:
            tag = f"csdi_{ds}_seed{seed}"
            done_marker = out_dir / f"{tag}_metrics.json"
            if done_marker.exists():
                print(f"[e8] skip {tag} (done)")
                continue

            sampler = VramSampler(args.gpu)
            sampler.start()
            t0 = time.time()
            try:
                # run_csdi.run_one trains+evals+saves imputed under its own
                # naming; we call it with a per-run out_dir, then lift files.
                tmp_dir = out_dir / f"_work_{tag}"
                res = run_csdi.run_one(ds, RATE, PATTERN, args.gpu,
                                       str(tmp_dir), seed=seed)
                # lift: metrics + imputed + evalmask (recompute eval mask with
                # the same convention: test seed = seed+2)
                X_test, M_test, _ = load_dataset(ds, "test")
                eval_mask = apply_block_missing(
                    M_test, BLOCK_LENS[PATTERN], RATE, seed=seed + 2)
                np.save(out_dir / f"{tag}_evalmask.npy", eval_mask)
                # sanity: hide-set size must be plausible for rho=0.9 blocks
                frac = float(eval_mask.sum()) / max(1, eval_mask.size)
                assert 0.5 < frac <= 1.0, f"implausible hide frac {frac}"
                src_imputed = tmp_dir / "csdi_09_block20_imputed.npy"
                if src_imputed.exists():
                    import shutil
                    shutil.move(str(src_imputed),
                                str(out_dir / f"{tag}_imputed.npy"))
                row = {
                    "dataset": ds, "seed": seed,
                    "mae": res["mae"], "mse": res["mse"],
                    "rmse": res.get("rmse"),
                    "train_min": res["train_min"],
                    "wall_min": round((time.time() - t0) / 60, 1),
                }
            finally:
                sampler.stop()
                row["vram_torch_mb"] = round(sampler.torch_peak_mb, 1)
                row["vram_smi_mb"] = round(sampler.smi_peak_mb, 1)

            with open(out_dir / f"{tag}_metrics.json", "w") as f:
                json.dump(row, f, indent=1)
            print(f"[e8] {tag} MAE={row['mae']:.4f} "
                  f"vram={row['vram_torch_mb']}MB(train alloc) "
                  f"wall={row['wall_min']}min")


if __name__ == "__main__":
    main()
