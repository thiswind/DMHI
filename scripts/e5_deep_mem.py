#!/usr/bin/env python3
"""R1 gap-filler (R2-7/R2-8): peak memory + per-sample latency for the deep
baselines (ImputeFormer, CSDI) on the x86-64 server, plus a torch-free DMHI
reference row — gas_home, seed 7, block-20, rate 0.9, CPU only.

Why this script exists: runs/e5_edge (Pi + server) measured DMHI/Linear/Mean
only; the reviewers' Table VI asks for deep-baseline peak memory alongside
DMHI. Deep baselines are trained here with the exact Table II protocol
(architecture, epochs, batch size, lr, nsample) via the same adapters the
paper runners use (run_imputeformer.py / run_csdi.py + csdi_physio_dataset).

Mask protocol (canonical, shared by all three workers):
    dmhi.utils.eval_protocol.apply_block_missing(M, block_len=20, rate=0.9,
    seed) — train seed=7, val seed=8, test seed=9 (seed family 7, test=+2,
    matching the baseline runners' convention).

Measurement: each method runs in an isolated subprocess; peak RSS =
getrusage(RUSAGE_SELF).ru_maxrss at exit (covers import + training + model
load + inference). Latency: 3 warm-up calls, then N timed single-sample
inference calls (mean + median).

Usage (server, riemann_baselines env, from repo root):
    python scripts/e5_deep_mem.py --method dmhi
    python scripts/e5_deep_mem.py --method imputeformer
    python scripts/e5_deep_mem.py --method csdi

Output: runs/e5_edge/deep_mem_<method>_gas_home.json
"""
import argparse
import json
import os
import pathlib
import resource
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent                    # repo root (~/DMHI on the server)
DATA = REPO / "data" / "processed" / "gas_home"
OUT = pathlib.Path("runs/e5_edge")
PY = sys.executable


def _rss_mb() -> float:
    """Peak RSS of this process in MiB (Linux maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _syspath():
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts" / "baselines"))


def _masks():
    """Load gas_home splits + canonical block-20/rate-0.9 eval masks."""
    import numpy as np
    from dmhi.utils.eval_protocol import apply_block_missing
    X = np.load(DATA / "X_test.npy")
    M = np.load(DATA / "M_test.npy")
    Xtr = np.load(DATA / "X_train.npy")
    Mtr = np.load(DATA / "M_train.npy")
    Xv = np.load(DATA / "X_val.npy")
    Mv = np.load(DATA / "M_val.npy")
    E = apply_block_missing(M, 20, 0.9, seed=9)
    Etr = apply_block_missing(Mtr, 20, 0.9, seed=7)
    Ev = apply_block_missing(Mv, 20, 0.9, seed=8)
    return X, M, E, Xtr, Mtr, Etr, Xv, Mv, Ev


def _report(lat_s, extra):
    lat_ms = sorted(1000.0 * x for x in lat_s)
    rec = {
        "per_sample_ms": round(sum(lat_ms) / len(lat_ms), 2),
        "latency_ms_median": round(lat_ms[len(lat_ms) // 2], 2),
        "n_timed": len(lat_ms),
        "rss_peak_mb": round(_rss_mb(), 1),
    }
    rec.update(extra)
    print(json.dumps(rec))


def worker_dmhi():
    """Released e2 checkpoint, torch-free numpy path — reference row."""
    import pickle

    import numpy as np

    _syspath()
    X, M, E, *_ = _masks()
    ck_path = REPO / "checkpoints" / "e2" / "gas_home_seed7" / "checkpoint.pkl"
    with open(ck_path, "rb") as f:
        imp = pickle.load(f)
    Min = (M.astype(bool) & ~E.astype(bool)).astype(np.int8)
    N = len(X)
    for i in range(3):
        imp.impute(X[i:i + 1].astype(np.float32), Min[i:i + 1])
    lat = []
    for i in range(N):
        t0 = time.perf_counter()
        imp.impute(X[i:i + 1].astype(np.float32), Min[i:i + 1])
        lat.append(time.perf_counter() - t0)
    _report(lat, {"versions": "torch-free (numpy only)"})


def worker_imputeformer():
    """PyPOTS ImputeFormer — Table II architecture + training protocol."""
    import numpy as np
    import torch
    from pypots.imputation import ImputeFormer

    torch.manual_seed(7)
    np.random.seed(7)
    _syspath()
    X, M, E, Xtr, Mtr, Etr, Xv, Mv, Ev = _masks()

    def nan_in(Xa, Mo, Eo):
        keep = Mo.astype(bool) & ~Eo.astype(bool)
        Xn = Xa.copy().astype(np.float32)
        Xn[~keep] = np.nan
        return Xn

    Xtr_n = nan_in(Xtr, Mtr, Etr)
    Xv_n = nan_in(Xv, Mv, Ev)
    Xte_n = nan_in(X, M, E)
    Xv_ori = Xv.copy().astype(np.float32)
    Xv_ori[Mv == 0] = np.nan

    t0 = time.time()
    model = ImputeFormer(
        n_steps=Xtr.shape[1],
        n_features=Xtr.shape[2],
        n_layers=2,
        d_input_embed=32,
        d_learnable_embed=8,
        d_proj=16,
        d_ffn=64,
        n_temporal_heads=4,
        dropout=0.1,
        batch_size=64,
        epochs=100,
        patience=20,
        device="cpu",
    )
    model.fit(train_set={"X": Xtr_n}, val_set={"X": Xv_n, "X_ori": Xv_ori})
    train_min = (time.time() - t0) / 60

    N = len(X)
    for i in range(3):
        model.impute({"X": Xte_n[i:i + 1].copy()})
    lat = []
    for i in range(N):
        t0 = time.perf_counter()
        model.impute({"X": Xte_n[i:i + 1].copy()})
        lat.append(time.perf_counter() - t0)
    _report(lat, {
        "train_min": round(train_min, 2),
        "versions": f"torch {torch.__version__}, pypots (ImputeFormer)",
    })


def worker_csdi():
    """csdi_official CSDI_Physio via OurPhysioDataset — Table II config."""
    import numpy as np
    import torch
    import yaml

    torch.manual_seed(7)
    np.random.seed(7)
    _syspath()
    csdi_repo = REPO / "reference_repos" / "baselines" / "csdi_official"
    sys.path.insert(0, str(csdi_repo))
    from csdi_physio_dataset import get_our_dataloader
    from main_model import CSDI_Physio
    from utils import train

    X, M, E, Xtr, Mtr, Etr, Xv, Mv, Ev = _masks()
    with open(csdi_repo / "config" / "base.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["test_missing_ratio"] = 0.9
    cfg["train"].update({"batch_size": 16, "epochs": 200, "lr": 1e-3})

    device = torch.device("cpu")
    model = CSDI_Physio(cfg, device, target_dim=Xtr.shape[2]).to(device)
    train_loader = get_our_dataloader(Xtr, Mtr, Etr, batch_size=16, shuffle=True)
    valid_loader = get_our_dataloader(Xv, Mv, Ev, batch_size=16, shuffle=False)

    t0 = time.time()
    train(model, cfg["train"], train_loader, valid_loader, foldername="")
    train_min = (time.time() - t0) / 60

    test_loader = get_our_dataloader(X, M, E, batch_size=1, shuffle=False)
    nsample = 50
    lat = []
    for bi, batch in enumerate(test_loader):
        if bi < 3:
            model.evaluate(batch, n_samples=nsample)  # warm-up
            continue
        t0 = time.perf_counter()
        model.evaluate(batch, n_samples=nsample)
        lat.append(time.perf_counter() - t0)
    _report(lat, {
        "train_min": round(train_min, 2),
        "nsample": nsample,
        "versions": f"torch {torch.__version__}, csdi_official (ermongroup/CSDI)",
    })


WORKERS = {
    "dmhi": worker_dmhi,
    "imputeformer": worker_imputeformer,
    "csdi": worker_csdi,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(WORKERS))
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()

    if args.worker:
        WORKERS[args.method]()
        return

    OUT.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [PY, __file__, "--method", args.method, "--worker"],
        capture_output=True, text=True,
    )
    print(f"== deep-mem {args.method} (gas_home seed7 block20 rate0.9 CPU) ==")
    print(r.stdout.strip() or r.stderr.strip()[-600:])
    try:
        rec = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        rec = {"error": r.stderr.strip()[-600:], "returncode": r.returncode}
    rec.update({
        "method": args.method,
        "dataset": "gas_home",
        "seed": 7,
        "rate": 0.9,
        "block": 20,
        "protocol": (
            "peak RSS = getrusage maxrss at worker exit (import+train+"
            "inference); latency = 3 warm-up + N timed single-sample calls; "
            "mask = apply_block_missing(block=20, rate=0.9; train seed=7, "
            "val seed=8, test seed=9); CPU only, x86-64 server"
        ),
    })
    out = OUT / f"deep_mem_{args.method}_gas_home.json"
    out.write_text(json.dumps(rec, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
