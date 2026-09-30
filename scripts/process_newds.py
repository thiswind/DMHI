#!/usr/bin/env python3
"""Process raw multi-domain datasets into the (N,T,D) processed format used by
the imputation harness (data/processed/<name>/{X,M}_{train,val,test}.npy).

Datasets share these traits relevant to the method's applicability domain:
strong cross-channel coupling on a curved low-dimensional manifold, with whole
samples being short trajectories. Missingness is injected at eval time by the
harness (block20), so the masks written here are all-ones (data are complete).

Splits are subject-disjoint where a subject id exists, to avoid leakage.
Per-channel z-score uses train statistics only.

Usage (run from the repository root, with dependencies installed):
  python scripts/process_newds.py --dataset gait_uci
  python scripts/process_newds.py --dataset mocap_karate
  python scripts/process_newds.py --dataset mhealth
"""
import argparse
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
RAW = REPO / "data/raw_newds"
OUT = REPO / "data/processed"


def _save(name, Xtr, Xva, Xte):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0)
    sd[sd < 1e-6] = 1.0
    for split, X in [("train", Xtr), ("val", Xva), ("test", Xte)]:
        Xz = ((X - mu) / sd).astype(np.float32)
        np.save(d / f"X_{split}.npy", Xz)
        np.save(d / f"M_{split}.npy", np.ones_like(Xz, dtype=np.int8))
        print(f"  {name}/{split}: X={Xz.shape}", flush=True)
    print(f"[OK] {name} -> {d}", flush=True)


def _split_by_subject(samples, subjects, val_frac=0.15, test_frac=0.2):
    uniq = sorted(set(subjects))
    rng = np.random.default_rng(0)
    uniq = list(rng.permutation(uniq))
    n = len(uniq)
    n_te = max(1, int(round(n * test_frac)))
    n_va = max(1, int(round(n * val_frac)))
    te = set(uniq[:n_te])
    va = set(uniq[n_te:n_te + n_va])
    tr = set(uniq[n_te + n_va:])
    idx = {"train": [], "val": [], "test": []}
    for i, s in enumerate(subjects):
        if s in te:
            idx["test"].append(i)
        elif s in va:
            idx["val"].append(i)
        else:
            idx["train"].append(i)
    arr = np.asarray(samples, dtype=np.float32)
    return (arr[idx["train"]], arr[idx["val"]], arr[idx["test"]])


def proc_gait():
    import pandas as pd
    df = pd.read_csv(RAW / "uci_gait/gait.csv")
    # D index = (leg-1)*3 + (joint-1): 6 bilateral joint angles
    subs, samples, subj_of = [], [], []
    for (sub, cond, rep), g in df.groupby(["subject", "condition", "replication"]):
        mat = np.full((101, 6), np.nan, dtype=np.float32)
        for _, row in g.iterrows():
            d = int((row["leg"] - 1) * 3 + (row["joint"] - 1))
            t = int(row["time"])
            mat[t, d] = row["angle"]
        if np.isnan(mat).any():
            continue
        samples.append(mat)
        subj_of.append(int(sub))
    print(f"gait: {len(samples)} samples", flush=True)
    return _split_by_subject(samples, subj_of)


def proc_mocap():
    base = RAW / "mocap_karate/data_impute/data"
    samples, subj_of = [], []
    for sk in sorted(base.glob("skill_*")):
        for f in sorted(sk.glob("*.csv")):
            import pandas as pd
            m = pd.read_csv(f).iloc[:, 1:].to_numpy(dtype=np.float32)  # drop idx col
            if m.shape[0] < 100:
                continue
            m = m[:100]
            if np.isnan(m).any():
                continue
            samples.append(m)
            subj_of.append(int(f.stem.split("_")[1]))
    print(f"mocap: {len(samples)} samples, D={samples[0].shape[1]}", flush=True)
    return _split_by_subject(samples, subj_of)


def proc_mhealth(T=100, stride=250, cap_per_subj=80):
    base = RAW / "mhealth/MHEALTHDATASET"
    samples, subj_of = [], []
    for f in sorted(base.glob("mHealth_subject*.log")):
        sid = int("".join(c for c in f.stem if c.isdigit()))
        raw = np.loadtxt(f, dtype=np.float32)
        sig = raw[:, :23]  # drop activity label (col 23)
        n_win = 0
        for s in range(0, sig.shape[0] - T, stride):
            w = sig[s:s + T]
            if np.isnan(w).any():
                continue
            samples.append(w)
            subj_of.append(sid)
            n_win += 1
            if n_win >= cap_per_subj:
                break
    print(f"mhealth: {len(samples)} windows, D=23", flush=True)
    return _split_by_subject(samples, subj_of)


def proc_gas(T=100):
    """HT Sensor home-activity gas dataset: each induction id is one trajectory
    of 8 MOX gas resistances + temperature + humidity (D=10). Chemical cross-
    sensitivities give a strongly nonlinear, coupled low-dimensional manifold;
    we resample each induction to T points on a normalised time axis."""
    import pandas as pd
    base = RAW / "gas_home"
    df = pd.read_csv(base / "HT_Sensor_dataset.dat", sep=r"\s+")
    cols = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "Temp.", "Humidity"]
    samples, sid = [], []
    grid = np.linspace(0.0, 1.0, T)
    for gid, g in df.groupby("id"):
        g = g.sort_values("time")
        tt = g["time"].to_numpy(dtype=np.float64)
        if len(tt) < T or tt.max() <= tt.min():
            continue
        tn = (tt - tt.min()) / (tt.max() - tt.min())
        mat = np.empty((T, len(cols)), dtype=np.float32)
        ok = True
        for j, c in enumerate(cols):
            v = g[c].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(v)):
                ok = False
                break
            mat[:, j] = np.interp(grid, tn, v)
        if not ok:
            continue
        samples.append(mat)
        sid.append(int(gid))
    print(f"gas_home: {len(samples)} inductions, D={len(cols)}", flush=True)
    return _split_by_subject(samples, sid)


def proc_breizh(n_fields=600, T=45):
    """BreizhCrops Sentinel-2 field time series (region frh01, L1C): each field
    is a (T=45, 13-band) multispectral reflectance trajectory. Surface
    reflectance across bands lies on a low-dimensional vegetation/soil manifold;
    cloud contamination is the real-world block-missingness driver. We sample a
    small subset of fields to keep CSDI fast."""
    import breizhcrops as bzh
    ds = bzh.BreizhCrops(region="frh01", level="L1C",
                         root=str(RAW / "breizh"))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(n_fields, len(ds)), replace=False)
    samples, sid = [], []
    for i in idx:
        x = np.asarray(ds[int(i)][0], dtype=np.float32)
        if x.shape[0] < T:
            continue
        x = x[:T, :13]
        if not np.all(np.isfinite(x)):
            continue
        samples.append(x)
        sid.append(int(i))
    print(f"breizh_rs: {len(samples)} fields, D=13", flush=True)
    return _split_by_subject(samples, sid)


def proc_delta(T=100):
    """Delta-robot force/torque assembly traces (Zenodo 13641620): each of 524
    recordings is one wheel-assembly cycle of 6 coupled channels (3 forces, 3
    torques). The end-effector follows a repeated mechanical trajectory, so the
    6-D signal lies on a strongly coupled, curved cyclic manifold (the industrial
    analogue of the kinematic MoCap win). High imputation value under sensor
    dropout in process monitoring."""
    import pandas as pd
    df = pd.read_csv(RAW / "delta_robot.csv")
    cols = ["Force_x", "Force_y", "Force_z", "Torque_x", "Torque_y", "Torque_z"]
    grid = np.linspace(0.0, 1.0, T)
    samples, sid = [], []
    for rid, g in df.groupby("idx"):
        v = g[cols].to_numpy(dtype=np.float64)
        if v.shape[0] < T or not np.all(np.isfinite(v)):
            continue
        tn = np.linspace(0.0, 1.0, v.shape[0])
        mat = np.stack([np.interp(grid, tn, v[:, j])
                        for j in range(len(cols))], axis=1).astype(np.float32)
        samples.append(mat)
        sid.append(int(rid))
    print(f"delta_robot: {len(samples)} recordings, D={len(cols)}", flush=True)
    return _split_by_subject(samples, sid)


def proc_breizh_pheno(pool=4000, n_fields=600, T=45):
    """Remote-sensing variant emphasising curvature: from BreizhCrops frh01 we
    keep the fields with the strongest seasonal dynamics (highest mean per-band
    temporal std). A strong green-up/senescence cycle traces a markedly curved
    trajectory in 13-band reflectance space, where block (cloud) gaps make a
    linear chord pierce the manifold, unlike the near-linear generic mix."""
    import breizhcrops as bzh
    ds = bzh.BreizhCrops(region="frh01", level="L1C", root=str(RAW / "breizh"))
    rng = np.random.default_rng(0)
    cand = rng.choice(len(ds), size=min(pool, len(ds)), replace=False)
    scored = []
    for i in cand:
        x = np.asarray(ds[int(i)][0], dtype=np.float32)
        if x.shape[0] < T:
            continue
        x = x[:T, :13]
        if not np.all(np.isfinite(x)):
            continue
        amp = float(x.std(axis=0).mean())  # seasonal dynamics strength
        scored.append((amp, int(i), x))
    scored.sort(key=lambda r: -r[0])
    sel = scored[:n_fields]
    samples = [s[2] for s in sel]
    sid = [s[1] for s in sel]
    print(f"breizh_pheno: {len(samples)} high-amplitude fields, D=13", flush=True)
    return _split_by_subject(samples, sid)


def proc_hydraulic(T=100):
    """UCI/ZeMA hydraulic test-rig condition monitoring. The rig repeats a 60 s
    constant-load cycle while 17 sensors (6 pressures, motor power, 2 flows, 4
    temperatures, vibration, 3 virtual efficiency channels) observe one shared
    hydraulic circuit, so the per-cycle 17-D signal lies on a low-dimensional,
    strongly coupled, curved manifold. Each sensor is resampled from its native
    rate to a common T. High imputation value under sensor dropout in
    predictive maintenance."""
    base = RAW / "hydraulic"
    files = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1",
             "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]
    grid = np.linspace(0.0, 1.0, T)
    chans = []
    for nm in files:
        arr = np.loadtxt(base / f"{nm}.txt")  # (2205, cols_at_rate)
        n, c = arr.shape
        src = np.linspace(0.0, 1.0, c)
        rs = np.stack([np.interp(grid, src, arr[i]) for i in range(n)], axis=0)
        chans.append(rs.astype(np.float32))
    X = np.stack(chans, axis=-1)  # (2205, T, 17)
    rng = np.random.default_rng(0)
    cap = 600  # keep CSDI tractable and on par with the other showcase datasets
    if X.shape[0] > cap:
        X = X[rng.choice(X.shape[0], size=cap, replace=False)]
    print(f"hydraulic: N={X.shape[0]} T={X.shape[1]} D={X.shape[2]}", flush=True)
    idx = rng.permutation(X.shape[0])
    n = X.shape[0]
    ntr, nva = int(n * 0.65), int(n * 0.15)
    tr, va, te = idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]
    return X[tr], X[va], X[te]


def proc_mc_maze(bin_ms=10, n_neurons=50):
    """Neural population activity from the Neural Latents Benchmark MC_Maze_Small
    (DANDI:000140), monkey M1/PMd during delayed center-out reaches with barriers.
    Population spiking is the canonical 'neural manifold': activity lives on a
    low-dimensional, curved trajectory set by the movement. We bin to bin_ms and
    keep the most active neurons. Inferring masked bins is the imputation analogue
    of the benchmark's co-smoothing task, an emerging, high-value application."""
    from nlb_tools.nwb_interface import NWBDataset
    from nlb_tools.make_tensors import make_train_input_tensors
    ds = NWBDataset(str(RAW / "mc_maze" / "000140" / "sub-Jenkins"))
    ds.resample(bin_ms)
    d = make_train_input_tensors(ds, dataset_name="mc_maze_small",
                                 trial_split="train", save_file=False)
    hi = d["train_spikes_heldin"]
    ho = d.get("train_spikes_heldout")
    X = np.concatenate([hi, ho], axis=-1) if ho is not None else hi
    X = X.astype(np.float32)  # (n_trials, T, n_all_neurons)
    order = np.argsort(-X.reshape(-1, X.shape[-1]).sum(0))
    X = X[:, :, order[:n_neurons]]
    print(f"mc_maze: N={X.shape[0]} T={X.shape[1]} D={X.shape[2]}", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.permutation(X.shape[0])
    n = X.shape[0]
    ntr, nva = int(n * 0.65), int(n * 0.15)
    tr, va, te = idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]
    return X[tr], X[va], X[te]


def proc_pulsedb():
    """Cuff-less BP waveforms fetched from VitalDB (see fetch_pulsedb.py). Each
    window holds PPG, ECG and ABP over a few cardiac cycles; the three signals
    are coupled through cardiovascular dynamics, tracing a curved closed loop.
    Subject (case)-disjoint split avoids leakage."""
    z = np.load(RAW / "pulsedb" / "segments.npz")
    X, sid = z["X"], z["sid"]
    return _split_by_subject([X[i] for i in range(len(X))],
                             [int(s) for s in sid])


def proc_solo12(L=300, stride=50, T=100):
    """Solo12 open-source quadruped (robot dog) walking recordings
    (paLeziart/solo12-recordings). Each of two trotting experiments logs the 12
    measured actuator joint angles (3 per leg: HAA/HFE/Knee) at high rate. The
    joints are tightly coupled by the gait pattern and leg kinematics, so a short
    window traces a curved, strongly coupled trajectory on the locomotion
    manifold; a block gap forces reconstructing a coordinated whole-body posture
    rather than interpolating each joint independently. We slice sliding windows
    of L raw samples and resample each to T. Experiment 1 -> train/val,
    experiment 2 -> test (recording-disjoint). High value under joint-encoder
    dropout in legged robots."""
    base = RAW / "solo12" / "Solo12_Walk_Recordings"
    grid = np.linspace(0.0, 1.0, T)
    src = np.linspace(0.0, 1.0, L)

    def windows(arr):
        out = []
        for s in range(0, arr.shape[0] - L + 1, stride):
            w = arr[s:s + L]
            m = np.stack([np.interp(grid, src, w[:, j])
                          for j in range(w.shape[1])], axis=1)
            out.append(m.astype(np.float32))
        return out

    q1 = np.load(base / "experiment_walk_1.npz")["q_mes"].astype(np.float64)
    q2 = np.load(base / "experiment_walk_2.npz")["q_mes"].astype(np.float64)
    w1, w2 = windows(q1), windows(q2)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(w1))
    nva = max(1, int(len(w1) * 0.18))
    va = [w1[i] for i in idx[:nva]]
    tr = [w1[i] for i in idx[nva:]]
    te = w2
    print(f"solo12: train={len(tr)} val={len(va)} test={len(te)} D=12",
          flush=True)
    return (np.asarray(tr, dtype=np.float32),
            np.asarray(va, dtype=np.float32),
            np.asarray(te, dtype=np.float32))



def proc_kuka(L=100, stride_tr=70, stride_te=40):
    """KUKA KR300 R2500 ultra SE industrial-robot identification benchmark
    (Weigand et al., 2022; DOI 10.26204/data/5), 10 Hz filtered. The forward
    model couples 6 motor torques (u, Nm) to the 6 measured axis positions
    (y, deg) through strongly nonlinear dynamics: backlash, pose-dependent
    inertia and gravity, Coriolis and temperature-dependent friction. We stack
    [y, u] into a 12-D signal whose short windows live on a curved, tightly
    coupled torque-pose manifold, so a block gap cannot be linearly interpolated
    without violating the dynamics. Windows from the benchmark train stream ->
    train/val, from the independent test stream -> test (run-disjoint). High
    value under joint-sensor / torque dropout in industrial manipulators."""
    from scipy.io import loadmat
    m = loadmat(str(RAW / "kuka" /
                    "forward_identification_without_raw_data.mat"))

    def stream(y, u):
        return np.concatenate([y.T, u.T], axis=1).astype(np.float64)

    def windows(arr, stride):
        return [arr[s:s + L].astype(np.float32)
                for s in range(0, arr.shape[0] - L + 1, stride)]

    wtr = windows(stream(m["y_train"], m["u_train"]), stride_tr)
    wte = windows(stream(m["y_test"], m["u_test"]), stride_te)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(wtr))
    nva = max(1, int(len(wtr) * 0.15))
    va = [wtr[i] for i in idx[:nva]]
    tr = [wtr[i] for i in idx[nva:]]
    print(f"kuka: train={len(tr)} val={len(va)} test={len(wte)} D=12",
          flush=True)
    return (np.asarray(tr, dtype=np.float32),
            np.asarray(va, dtype=np.float32),
            np.asarray(wte, dtype=np.float32))



def proc_nanodrone(L=100, stride_tr=80, stride_te=120):
    """NanoBench nano-quadrotor system-identification benchmark
    (idsia-robotics/nanodrone-sysid-benchmark): real Crazyflie 2.1 Brushless
    flights in a Vicon arena at 100 Hz. We keep the dynamics state plus the
    actuation: unit quaternion attitude (S^3, curved), body-frame linear and
    angular velocity, and the four motor angular speeds (D=14). Coreless-DC
    actuation and severe aerodynamic nonlinearities couple these channels on a
    strongly curved manifold, so a block gap cannot be linearly interpolated
    without breaking the rigid-body/quaternion constraints. Square/Random/Chirp
    trajectories -> train/val, the held-out Melon trajectory -> test
    (trajectory-disjoint). High value for low-altitude UAV state estimation
    under telemetry/sensor dropout."""
    import pandas as pd
    cols = ["qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz",
            "m1_rads", "m2_rads", "m3_rads", "m4_rads"]
    base = RAW / "nanodrone"

    def windows(files, stride):
        out = []
        for f in files:
            v = pd.read_csv(f)[cols].to_numpy(dtype=np.float64)
            for s in range(0, v.shape[0] - L + 1, stride):
                w = v[s:s + L]
                if np.all(np.isfinite(w)):
                    out.append(w.astype(np.float32))
        return out

    tr_files = sorted((base / "train").glob("*.csv"))
    te_files = sorted((base / "test").glob("*.csv"))
    wtr = windows(tr_files, stride_tr)
    wte = windows(te_files, stride_te)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(wtr))
    nva = max(1, int(len(wtr) * 0.15))
    va = [wtr[i] for i in idx[:nva]]
    tr = [wtr[i] for i in idx[nva:]]
    print(f"nanodrone: train={len(tr)} val={len(va)} test={len(wte)} D=14",
          flush=True)
    return (np.asarray(tr, dtype=np.float32),
            np.asarray(va, dtype=np.float32),
            np.asarray(wte, dtype=np.float32))



def proc_nanodrone_dec(L=100, stride_tr=40, stride_te=60, dec=4):
    """Temporally decimated NanoBench (every dec-th sample -> ~25 Hz). At native
    100 Hz consecutive states are so close that a straight chord stays near the
    manifold, so linear interpolation is geometrically cheap. Decimating widens
    the per-step motion so a block gap genuinely spans the curved attitude /
    rigid-body trajectory, isolating the manifold-fidelity (MD) advantage."""
    import pandas as pd
    cols = ["qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz",
            "m1_rads", "m2_rads", "m3_rads", "m4_rads"]
    base = RAW / "nanodrone"

    def windows(files, stride):
        out = []
        for f in files:
            v = pd.read_csv(f)[cols].to_numpy(dtype=np.float64)[::dec]
            for s in range(0, v.shape[0] - L + 1, stride):
                w = v[s:s + L]
                if np.all(np.isfinite(w)):
                    out.append(w.astype(np.float32))
        return out

    wtr = windows(sorted((base / "train").glob("*.csv")), stride_tr)
    wte = windows(sorted((base / "test").glob("*.csv")), stride_te)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(wtr))
    nva = max(1, int(len(wtr) * 0.15))
    va = [wtr[i] for i in idx[:nva]]
    tr = [wtr[i] for i in idx[nva:]]
    print(f"nanodrone_dec: train={len(tr)} val={len(va)} test={len(wte)} D=14",
          flush=True)
    return (np.asarray(tr, dtype=np.float32),
            np.asarray(va, dtype=np.float32),
            np.asarray(wte, dtype=np.float32))



PROC = {"gait_uci": proc_gait, "mocap_karate": proc_mocap,
        "mhealth": proc_mhealth, "gas_home": proc_gas,
        "breizh_rs": proc_breizh, "delta_robot": proc_delta,
        "breizh_pheno": proc_breizh_pheno, "hydraulic": proc_hydraulic,
        "mc_maze": proc_mc_maze, "pulsedb": proc_pulsedb, "solo12": proc_solo12, "kuka": proc_kuka, "nanodrone": proc_nanodrone, "nanodrone_dec": proc_nanodrone_dec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(PROC))
    args = ap.parse_args()
    Xtr, Xva, Xte = PROC[args.dataset]()
    _save(args.dataset, Xtr, Xva, Xte)


if __name__ == "__main__":
    main()
