#!/usr/bin/env python3
"""Fit DMHI (RiemannianImputer) on a processed dataset and save a checkpoint.

Note: the paper's deployed form is training-free (zero trained parameters);
this entry point regenerates the checkpoint artifacts (Stage I embedding +
Stage II metric refinement) for research reproducibility. This is the
fitting entry point behind the paper's "Ours" checkpoints. The
deployed configuration (see configs/ and REPRODUCE.md) corresponds to the
defaults used here: d = min(12, D-2) unless overridden, k=10, k_clle=9,
r_L=0.30, lambda_F=1e-5, epochs=200 (Stage I and Stage II), patience=20,
Stage II operator = soft graph-spline (mhb).

Data convention (produced by scripts/process_newds.py or fetched per
data/DATA.md):
    data/processed/<dataset>/X_train.npy  shape (N, T, D) float32
    data/processed/<dataset>/M_train.npy  shape (N, T, D) int8   1=observed
    (same for X_val / M_val)

Datasets without native masks (e.g. cmapss) are treated as fully observed,
which matches how their baselines were generated.

Usage:
    python scripts/train.py --dataset gas_home --seed 7 \
        --out_dir checkpoints/dmhi

    python scripts/evaluate.py --run_dir checkpoints/dmhi/gas_home_seed7 \
        --block_size 20 --seed 7
"""
import argparse
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from dmhi.method.pipeline import RiemannianImputer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True,
                   help="name of a directory under data/processed/ (see data/DATA.md)")
    p.add_argument("--missing_rate", type=float, default=0.7,
                   help="evaluation missing rate recorded in config.json "
                        "(the block mask itself is applied at eval time)")
    p.add_argument("--pattern", default="block", choices=["mcar", "block"])
    p.add_argument("--block_size", type=int, default=20,
                   help="Block length (used only when --pattern block)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out_dir", default="checkpoints/dmhi")
    # Model hyperparameters — defaults follow the deployed configuration.
    p.add_argument("--d", type=int, default=0,
                   help="latent chart dimension; 0 -> min(12, D-2)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--k_clle", type=int, default=9)
    p.add_argument("--r_L", type=float, default=0.30)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--embed_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_F", type=float, default=1e-5)
    p.add_argument("--lambda_smooth", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--stage2_method", default="mhb", choices=["mhb", "karcher"],
                   help="'mhb' = paper default (soft graph-spline); "
                        "'karcher' = ablation operator")
    p.add_argument("--deterministic", action="store_true",
                   help="strict determinism: torch deterministic algorithms, "
                        "single thread, deterministic cudnn. Slower (2-4x).")
    return p.parse_args()


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load(ddir: pathlib.Path, name: str, like=None):
    p = ddir / name
    if p.exists():
        return np.load(p)
    assert like is not None, f"missing {p} and no fallback shape"
    return np.ones_like(like, dtype=np.int8)


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    ddir = pathlib.Path("data/processed") / args.dataset
    X_train = np.load(ddir / "X_train.npy").astype(np.float32)
    X_val = np.load(ddir / "X_val.npy").astype(np.float32)
    M_train = _load(ddir, "M_train.npy", like=X_train).astype(np.int8)
    M_val = _load(ddir, "M_val.npy", like=X_val).astype(np.int8)

    N, T, D = X_train.shape
    d = args.d if args.d > 0 else min(12, D - 2)
    print(f"[train] {args.dataset} train={X_train.shape} val={X_val.shape} "
          f"d={d} obs_frac={M_train.mean():.3f}", flush=True)

    imputer = RiemannianImputer(
        d=d,
        k=args.k,
        k_clle=args.k_clle,
        r_L=args.r_L,
        embed_epochs=args.embed_epochs,
        epochs=args.epochs,
        lr=args.lr,
        lambda_F=args.lambda_F,
        lambda_smooth=args.lambda_smooth,
        batch_size=args.batch_size,
        patience=args.patience,
        stage2_method=args.stage2_method,
    )

    t0 = time.time()
    imputer.fit(X_train, M_train, X_val, M_val)
    train_min = (time.time() - t0) / 60.0

    run_id = f"{args.dataset}_seed{args.seed}"
    out_dir = pathlib.Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    imputer.save(str(out_dir / "checkpoint.pkl"))
    print(f"[train] checkpoint saved: {out_dir / 'checkpoint.pkl'}")

    config = {
        "dataset": args.dataset,
        "missing_rate": args.missing_rate,
        "pattern": args.pattern,
        "block_size": args.block_size if args.pattern == "block" else None,
        "seed": args.seed,
        "d": d,
        "k": args.k,
        "k_clle": args.k_clle,
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_F": args.lambda_F,
        "lambda_smooth": args.lambda_smooth,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "train_min": round(train_min, 2),
        "run_id": run_id,
        "deterministic": bool(args.deterministic),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[train] done. run_id={run_id}  train_min={train_min:.1f}")
    print(str(out_dir))


if __name__ == "__main__":
    main()
