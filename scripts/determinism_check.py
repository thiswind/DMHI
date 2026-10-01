#!/usr/bin/env python3
"""
determinism_check.py
=====================
N4 Level-2: fixed mask, multiple inference repetitions.

Quantifies whether imputation outputs (and downstream AUROC) are stable
when the SAME block-missing mask is held fixed and inference is repeated.

Stochastic-inference methods (CSDI diffusion sampling) are expected to
produce varying imputations and downstream decisions.
Deterministic methods (DMHI, classical, and learned models at eval mode)
should produce bit-identical outputs across repetitions.

Run from the repository root (dependencies installed):
  python scripts/determinism_check.py
  python scripts/determinism_check.py --domains gait_uci gas_home physionet2012

Outputs:
  runs/determinism/n4_determinism.jsonl
  runs/determinism/n4_determinism_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ALLOW pull: unify root resolution with sibling e4_* runners.


REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from dmhi.utils.eval_protocol import apply_block_missing as _apply_block_missing
from baselines.common import load_dataset, make_input_mask

OUT_DIR = REPO / "runs/determinism"
OURS_CKPT = REPO / "runs/recompute033/ours"
LEGACY_BASELINE = REPO / "runs/baselines"

MASK_SEED = 7
RATE = 0.9
RATEPCT = 90
BLOCK_L = 20
N_CSDI_SAMPLES = 10
N_REPEATS = 5

# Methods grouped by inference-time randomness
METHODS_DETERMINISTIC = ["ours", "linear", "mean", "psw_i"]
METHODS_LEARNED_EVAL = ["brits", "saits", "imputeformer", "timemixer"]
METHODS_STOCHASTIC = ["csdi"]
ALL_METHODS = METHODS_DETERMINISTIC + METHODS_LEARNED_EVAL + METHODS_STOCHASTIC


@dataclass
class DomainCfg:
    name: str
    data_dir: Path
    baseline_dir: Path
    ours_prefix: str
    task: str  # "binary" or "multiclass"
    y_train_file: str = "y_train.npy"
    y_test_file: str = "y_test.npy"
    csdi_pk: Path | None = None
    csdi_model_dir: Path | None = None


def domain_configs() -> dict[str, DomainCfg]:
    return {
        "gait_uci": DomainCfg(
            name="gait_uci",
            data_dir=REPO / "data/processed/gait_uci",
            baseline_dir=REPO / "runs/newds_v2/gait_uci/seed7",
            ours_prefix="gait_uci",
            task="multiclass",
            csdi_pk=REPO
            / "runs/newds_v2/gait_uci/seed7/csdi/gait_uci/rate90_block20/generated_outputs_nsample50.pk",
        ),
        "gas_home": DomainCfg(
            name="gas_home",
            data_dir=REPO / "data/processed/gas_home",
            baseline_dir=REPO / "runs/newds_v2/gas_home/seed7",
            ours_prefix="gas_home",
            task="multiclass",
            csdi_pk=REPO
            / "runs/newds_v2/gas_home/seed7/csdi/gas_home/rate90_block20/generated_outputs_nsample50.pk",
        ),
        "physionet2012": DomainCfg(
            name="physionet2012",
            data_dir=REPO / "data/processed/physionet2012",
            baseline_dir=REPO / "runs/paper_v3/baselines/seed7",
            ours_prefix="physionet2012",
            task="binary",
            csdi_pk=REPO
            / "runs/paper_v3/baselines/seed7/csdi/physionet2012/rate90_block20/generated_outputs_nsample50.pk",
        ),
    }


def feature_map(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([X.mean(axis=1), X[:, -1, :]], axis=1)


def impute_linear(X: np.ndarray, M: np.ndarray) -> np.ndarray:
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
    return X_hat.astype(np.float32)


def impute_mean(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    return np.where(M.astype(bool), X, 0.0).astype(np.float32)


def baseline_npy(cfg: DomainCfg, method: str) -> Path:
    p = cfg.baseline_dir / f"{method}_{cfg.name}_{RATEPCT}_block{BLOCK_L}_imputed.npy"
    if p.is_file():
        return p
    if cfg.name == "physionet2012":
        legacy = LEGACY_BASELINE / f"{method}_{cfg.name}_{RATEPCT}_block{BLOCK_L}_imputed.npy"
        if legacy.is_file():
            return legacy
    return p


def find_pypots_ckpt(base: Path, pattern: str) -> Path:
    hits = sorted(glob.glob(str(base / "**" / pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"no checkpoint matching {pattern} under {base}")
    return Path(hits[-1])


def auroc_from_proba(y_test: np.ndarray, proba: np.ndarray) -> float:
    n_cls = len(np.unique(y_test))
    if n_cls == 2:
        return float(roc_auc_score(y_test, proba[:, 1]))
    return float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))


def eval_auroc(
    X_hat: np.ndarray, clf: LogisticRegression, scaler: StandardScaler, y_test: np.ndarray
) -> float:
    feats = scaler.transform(feature_map(X_hat))
    proba = clf.predict_proba(feats)
    return auroc_from_proba(y_test, proba)


def impute_ours(seed: int, cfg: DomainCfg, X_test: np.ndarray, M_eval: np.ndarray) -> np.ndarray:
    from dmhi.method.pipeline import RiemannianImputer

    ckpt = OURS_CKPT / f"{cfg.ours_prefix}_seed{seed}" / "checkpoint.pkl"
    imputer = RiemannianImputer.load(str(ckpt))
    return imputer.impute(X_test, M_eval)


def impute_psw_i(cfg: DomainCfg) -> np.ndarray:
    return np.load(baseline_npy(cfg, "psw_i")).astype(np.float32)


def load_csdi_samples(cfg: DomainCfg) -> np.ndarray:
    """Return (N_test, nsample, T, D) diffusion draws."""
    if cfg.csdi_pk is None or not cfg.csdi_pk.is_file():
        raise FileNotFoundError(cfg.csdi_pk)
    with open(cfg.csdi_pk, "rb") as f:
        data = pickle.load(f)
    samples = data[0]
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    return samples.astype(np.float32)


def build_nan_test(X_test: np.ndarray, M_test: np.ndarray, M_eval: np.ndarray) -> np.ndarray:
    held = (M_test.astype(bool) & ~M_eval.astype(bool))
    X_nan = X_test.copy().astype(np.float32)
    X_nan[~M_eval.astype(bool)] = np.nan
    return X_nan


def impute_learned_repeat(
    method: str,
    cfg: DomainCfg,
    X_test: np.ndarray,
    M_test: np.ndarray,
    M_eval: np.ndarray,
    rep: int,
) -> np.ndarray:
    """Reload checkpoint and impute once (rep sets torch seed for reproducibility check)."""
    torch.manual_seed(1000 + rep)
    np.random.seed(1000 + rep)

    X_nan = build_nan_test(X_test, M_test, M_eval)
    T, D = X_test.shape[1], X_test.shape[2]
    base = cfg.baseline_dir / method / cfg.name / f"rate{RATEPCT}_block{BLOCK_L}"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if method in ("saits", "brits"):
        if method == "saits":
            from pypots.imputation import SAITS

            ckpt = find_pypots_ckpt(base, "SAITS.pypots")
            model = SAITS(
                n_steps=T,
                n_features=D,
                n_layers=2,
                d_model=256,
                n_heads=4,
                d_k=64,
                d_v=64,
                d_ffn=256,
                dropout=0.1,
                device=device,
            )
        else:
            from pypots.imputation import BRITS

            ckpt = find_pypots_ckpt(base, "BRITS.pypots")
            model = BRITS(
                n_steps=T,
                n_features=D,
                rnn_hidden_size=256,
                device=device,
            )
        model.load(str(ckpt))
        out = model.impute({"X": X_nan})
        X_hat = np.array(out["imputation"] if isinstance(out, dict) else out)
    elif method == "imputeformer":
        ckpt = find_pypots_ckpt(base, "ImputeFormer.pypots")
        from pypots.imputation import ImputeFormer

        model = ImputeFormer(
            n_steps=T,
            n_features=D,
            n_layers=2,
            d_input_embed=32,
            d_learnable_embed=8,
            d_proj=16,
            d_ffn=64,
            n_temporal_heads=4,
            dropout=0.1,
            device=device,
        )
        model.load(str(ckpt))
        out = model.impute({"X": X_nan})
        X_hat = np.array(out["imputation"] if isinstance(out, dict) else out)
    elif method == "timemixer":
        ckpt = find_pypots_ckpt(base, "TimeMixer.pypots")
        from pypots.imputation import TimeMixer

        model = TimeMixer(
            n_steps=T,
            n_features=D,
            n_layers=2,
            d_model=64,
            d_ffn=128,
            top_k=3,
            dropout=0.1,
            channel_independence=False,
            decomp_method="moving_avg",
            moving_avg=5,
            downsampling_layers=2,
            downsampling_window=2,
            apply_nonstationary_norm=False,
            device=device,
        )
        model.load(str(ckpt))
        out = model.impute({"X": X_nan})
        X_hat = np.array(out["imputation"] if isinstance(out, dict) else out)
    else:
        raise ValueError(method)

    return np.nan_to_num(X_hat, nan=0.0).astype(np.float32)


def run_domain(cfg: DomainCfg, seed: int = MASK_SEED) -> list[dict]:
    X_train = np.load(cfg.data_dir / "X_train.npy").astype(np.float32)
    y_train = np.ravel(np.load(cfg.data_dir / cfg.y_train_file)).astype(int)
    X_test = np.load(cfg.data_dir / "X_test.npy").astype(np.float32)
    y_test = np.ravel(np.load(cfg.data_dir / cfg.y_test_file)).astype(int)
    M_test = np.load(cfg.data_dir / "M_test.npy").astype(np.int8)

    held = _apply_block_missing(M_test, BLOCK_L, RATE, seed=seed + 2)
    M_eval = (M_test & ~held).astype(np.int8)

    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=2000, multi_class="ovr")
    clf.fit(scaler.fit_transform(feature_map(X_train)), y_train)

    rows: list[dict] = []
    imputation_cache: dict[str, list[np.ndarray]] = {}

    def record(method: str, rep: int, X_hat: np.ndarray, stochastic_class: str):
        X_hat = np.nan_to_num(X_hat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        imputation_cache.setdefault(method, []).append(X_hat)
        auc = eval_auroc(X_hat, clf, scaler, y_test)
        rows.append(
            {
                "domain": cfg.name,
                "method": method,
                "stochastic_class": stochastic_class,
                "mask_seed": seed,
                "rep": rep,
                "auroc": auc,
            }
        )
        print(f"  {method:14s} rep={rep}  auroc={auc:.4f}")

    print(f"\n[{cfg.name}] fixed mask seed={seed}, n_test={len(y_test)}")

    # ── deterministic methods ──
    for method in METHODS_DETERMINISTIC:
        for rep in range(N_REPEATS):
            if method == "ours":
                X_hat = impute_ours(seed, cfg, X_test, M_eval)
            elif method == "linear":
                X_hat = impute_linear(X_test, M_eval)
            elif method == "mean":
                X_hat = impute_mean(X_test, M_eval)
            elif method == "psw_i":
                X_hat = impute_psw_i(cfg)
            else:
                continue
            record(method, rep, X_hat, "deterministic")

    # ── learned eval-mode (training stochastic, inference should be deterministic) ──
    for method in METHODS_LEARNED_EVAL:
        try:
            for rep in range(N_REPEATS):
                X_hat = impute_learned_repeat(method, cfg, X_test, M_test, M_eval, rep)
                record(method, rep, X_hat, "learned_eval_deterministic")
        except Exception as e:
            print(f"  SKIP {method}: {e}", file=sys.stderr)
            for rep in range(N_REPEATS):
                X_hat = np.load(baseline_npy(cfg, method)).astype(np.float32)
                record(method, rep, X_hat, "learned_eval_deterministic_fallback")

    # ── CSDI: each diffusion sample is one stochastic inference draw ──
    try:
        samples = load_csdi_samples(cfg)  # (N, nsample, T, D)
        n_use = min(N_CSDI_SAMPLES, samples.shape[1])
        for rep in range(n_use):
            X_hat = samples[:, rep, :, :]
            record("csdi", rep, X_hat, "inference_stochastic")
    except Exception as e:
        print(f"  SKIP csdi: {e}", file=sys.stderr)

    # ── imputation stability metrics ──
    summary_extra: list[dict] = []
    for method, arrs in imputation_cache.items():
        if len(arrs) < 2:
            continue
        diffs = [float(np.max(np.abs(arrs[i] - arrs[j]))) for i in range(len(arrs)) for j in range(i + 1, len(arrs))]
        aucs = [r["auroc"] for r in rows if r["method"] == method]
        summary_extra.append(
            {
                "domain": cfg.name,
                "method": method,
                "stochastic_class": rows[[r["method"] for r in rows].index(method)]["stochastic_class"]
                if method in [r["method"] for r in rows]
                else "",
                "n_rep": len(aucs),
                "auroc_min": float(min(aucs)),
                "auroc_max": float(max(aucs)),
                "auroc_range": float(max(aucs) - min(aucs)),
                "auroc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "impute_max_diff": float(max(diffs)),
                "impute_mean_diff": float(np.mean(diffs)),
            }
        )

    return rows, summary_extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["gait_uci", "gas_home", "physionet2012"],
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "n4_determinism.jsonl"
    summary_path = OUT_DIR / "n4_determinism_summary.csv"

    cfgs = domain_configs()
    all_rows: list[dict] = []
    all_summary: list[dict] = []

    for dom in args.domains:
        if dom not in cfgs:
            print(f"unknown domain {dom}", file=sys.stderr)
            continue
        rows, summ = run_domain(cfgs[dom])
        all_rows.extend(rows)
        all_summary.extend(summ)

    with jsonl_path.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    fields = [
        "domain",
        "method",
        "stochastic_class",
        "n_rep",
        "auroc_min",
        "auroc_max",
        "auroc_range",
        "auroc_std",
        "impute_max_diff",
        "impute_mean_diff",
    ]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sorted(all_summary, key=lambda x: (-x["auroc_range"], x["domain"], x["method"])):
            w.writerow({k: s.get(k, "") for k in fields})

    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    print("\n=== Top unstable methods (by AUROC range) ===")
    for s in sorted(all_summary, key=lambda x: -x["auroc_range"])[:12]:
        print(
            f"{s['domain']:16s} {s['method']:14s}  range={s['auroc_range']:.4f}  "
            f"impute_diff={s['impute_max_diff']:.2e}  class={s['stochastic_class']}"
        )


if __name__ == "__main__":
    main()
