#!/usr/bin/env python3
"""E5 (Raspberry Pi side): latency / peak RSS / energy estimate for Table VI.

Device: Raspberry Pi 3 Model B+ (4x Cortex-A53 @1.4GHz, 905MB RAM, Debian 13
trixie aarch64). Companions to `e5_edge_server.py` (same Table VI protocol:
rho=0.9, block=20, per-sample CPU inference).

Methods: DMHI (closed-form, torch-free build+impute via T4 lazy import),
Linear interpolation, Mean fill. Deep baselines: separate torch time-box.

Energy (Method A, power-model estimate — NO external power meter):
  E_inf = (P_full - P_idle) * (cpu_seconds / n_cores)
with published board-input anchors for Pi 3B+ (matched stress pair):
  P_idle = 1.9 W, P_full = 5.1 W  (pidramble.com, `stress --cpu 4`;
  cross-checked element14 review 4.89 W full / 1.90 W headless idle)
cpu_seconds/n_cores is the exact time-integral of mean core utilization
(linear power-vs-utilization model), from getrusage (no sampling error).
All energy numbers must be reported as "power-model estimate".

Mask convention (matches e5_edge_server.py): sample_block_mask returns
1 = held-out. eval cells = held_out & natively observed; DMHI input mask
M_input = M_test & ~held_out (1 = observed, as the pipeline expects).

Isolation: each method runs in a fresh subprocess (`--worker`) so peak RSS
is uncontaminated; the parent aggregates into runs/e5_edge/pi_<ds>_seed<s>.json.

Usage (Pi): python3 scripts/e5_edge_pi.py --domain mocap_karate --seed 7
"""
import argparse
import json
import pathlib
import resource
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = pathlib.Path("runs/e5_edge")

# Pi 3B+ board-input power anchors (see module docstring for sources).
P_IDLE_W = 1.9
P_FULL_W = 5.1
N_CORES = 4


def cpu_seconds() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def worker(args):
    """Run one method end-to-end; print JSON to stdout (parent collects)."""
    import numpy as np

    sys.path.insert(0, str(REPO / "src"))
    from dmhi.method.synth_missing import sample_block_mask

    ds, seed, rate, n_cap, method, reps = (
        args.domain, args.seed, args.rate, args.n, args.method, args.reps)

    X_test = np.load(f"data/processed/{ds}/X_test.npy").astype(np.float32)[:n_cap]
    M_test = np.load(f"data/processed/{ds}/M_test.npy").astype(np.int8)[:n_cap]
    N, T, D = X_test.shape

    held = []
    for i in range(N):
        ev = sample_block_mask(D, T, seed=seed + i, block_size=20,
                               rate=rate).astype(np.int8)
        if ev.shape != (T, D):
            ev = ev.T
        held.append(ev[None] * M_test[i:i + 1])   # 1 = to-impute (kept & native)
    held = np.concatenate(held, axis=0)
    M_input = (M_test & ~held.astype(bool)).astype(np.int8)  # 1 = observed

    res = {"method": method, "N": int(N), "T": int(T), "D": int(D)}

    if method == "DMHI":
        from dmhi.method.pipeline import RiemannianImputer
        X_train = np.load(f"data/processed/{ds}/X_train.npy").astype(np.float32)
        M_train = np.load(f"data/processed/{ds}/M_train.npy").astype(np.int8)
        imp = RiemannianImputer(d=12)
        t0 = time.perf_counter()
        imp.build(X_train, M_train)
        res["build_s"] = round(time.perf_counter() - t0, 2)
        res["torch_in_worker"] = "torch" in sys.modules
        for i in range(3):
            imp.impute(X_test[i:i + 1], M_input[i:i + 1])
        lat = []
        for r in range(reps):
            for i in range(N):
                t0 = time.perf_counter()
                imp.impute(X_test[i:i + 1], M_input[i:i + 1])
                lat.append(time.perf_counter() - t0)
    elif method == "Linear":
        def lin(X, M):
            Xh = X.copy()
            t = np.arange(X.shape[1], dtype=np.float64)
            for c in range(X.shape[2]):
                obs = M[0, :, c].astype(bool)      # knots = observed only
                miss = ~obs
                if miss.any() and obs.any():
                    Xh[0, miss, c] = np.interp(t[miss], t[obs], X[0, obs, c])
            return Xh
        for i in range(3):
            lin(X_test[i:i + 1], M_input[i:i + 1])
        lat = []
        for r in range(reps):
            for i in range(N):
                t0 = time.perf_counter()
                lin(X_test[i:i + 1], M_input[i:i + 1])
                lat.append(time.perf_counter() - t0)
    elif method == "Mean":
        for i in range(3):
            _ = mean_fill(X_test[i], M_input[i])
        lat = []
        for r in range(reps):
            for i in range(N):
                t0 = time.perf_counter()
                _ = mean_fill(X_test[i], M_input[i])
                lat.append(time.perf_counter() - t0)
    else:
        raise SystemExit(f"unknown method {method}")

    lat_ms = sorted(x * 1000.0 for x in lat)
    k = len(lat_ms)
    med = lat_ms[k // 2] if k % 2 else 0.5 * (lat_ms[k // 2 - 1] + lat_ms[k // 2])
    res.update({
        "n_latency_samples": k,
        "latency_ms_median": round(med, 2),
        "latency_ms_p90": round(lat_ms[int(0.9 * (k - 1))], 2),
        "latency_ms_mean": round(sum(lat_ms) / k, 2),
        "cpu_s": round(cpu_seconds(), 2),
        "wall_s": round(sum(lat), 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
    })
    # E_inf per sample (mJ): ΔP × utilization-integral / N  (model estimate)
    e_total_j = (P_FULL_W - P_IDLE_W) * (cpu_seconds() / N_CORES)
    res["energy_mj_per_sample_est"] = round(e_total_j / k * 1000.0, 1)
    res["energy_model"] = (
        f"E=(P_full-P_idle)*cpu_s/{N_CORES}; P_idle={P_IDLE_W}W, "
        f"P_full={P_FULL_W}W (Pi 3B+ board-input, pidramble stress pair; "
        "power-model estimate, no external meter)")
    print("JSON:" + json.dumps(res))


def mean_fill(x, m):
    obs = m.astype(bool)
    out = x.copy()
    if (~obs).any():
        fill = x[obs].mean() if obs.any() else 0.0
        out[~obs] = fill
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="mocap_karate")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rate", type=float, default=0.9)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--reps", type=int, default=None,
                    help="passes over the test slice; default: enough for "
                         ">=100 latency samples")
    ap.add_argument("--methods", default="DMHI,Linear,Mean")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker_method")
    a = ap.parse_args()

    if a.worker:
        a.method = a.worker_method
        a.reps = a.reps or 4
        worker(a)
        return

    X_test = np.load(f"data/processed/{a.domain}/X_test.npy", mmap_mode="r")
    N_avail = min(a.n, X_test.shape[0])
    reps = a.reps or max(1, -(-100 // N_avail))  # ceil → ≥100 samples

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"domain": a.domain, "seed": a.seed, "rate": a.rate,
           "n_test": N_avail, "reps": reps, "device": "Raspberry Pi 3 Model B+",
           "os": "Debian 13 trixie aarch64", "threads": 4,
           "methods": {}}
    for meth in a.methods.split(","):
        env_python = sys.executable
        cmd = [env_python, str(REPO / "scripts" / "e5_edge_pi.py"),
               "--domain", a.domain, "--seed", str(a.seed),
               "--rate", str(a.rate), "--n", str(a.n),
               "--reps", str(reps), "--worker",
               "--worker_method", meth]
        print(f"[e5-pi] {a.domain}/{meth} ...", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("JSON:")), None)
        if line is None:
            out["methods"][meth] = {"error": p.stderr[-500:]}
            continue
        out["methods"][meth] = json.loads(line[5:])

    path = OUT_DIR / f"pi_{a.domain}_seed{a.seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    import numpy as np  # noqa: F401  (mmap shape probe in main)
    main()
