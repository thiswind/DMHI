#!/usr/bin/env python3
"""Fetch a small cuff-less-BP subset from VitalDB (open API) instead of the large
PulseDB Box archive. For each case we pull PPG, ECG-II and arterial BP (ABP) at
100 Hz, cut clean 10 s windows and resample each to a common T. The three
waveforms are driven by one cardiovascular cycle, so each window is a curved,
strongly coupled closed trajectory. Output is raw segments for proc_pulsedb."""
import argparse
import pathlib

import numpy as np

RAW = pathlib.Path.cwd() / "data/raw_newds"
TRACKS = ["SNUADC/PLETH", "SNUADC/ECG_II", "SNUADC/ART"]


def main(max_cases=200, target_n=700, T=100, win_s=10, per_case=4):
    import vitaldb
    cids = vitaldb.find_cases(TRACKS)
    grid = np.linspace(0.0, 1.0, T)
    segs, sid = [], []
    fs = 100
    w = win_s * fs
    for ci, cid in enumerate(cids[:max_cases]):
        if len(segs) >= target_n:
            break
        try:
            v = vitaldb.load_case(cid, TRACKS, 1.0 / fs)
        except Exception:
            continue
        if v is None or v.shape[0] < w:
            continue
        kept = 0
        nwin = v.shape[0] // w
        for k in range(nwin):
            if kept >= per_case:
                break
            seg = v[k * w:(k + 1) * w]
            if not np.all(np.isfinite(seg)):
                continue
            if np.any(seg.std(0) < 1e-6):
                continue
            abp = seg[:, 2]
            if abp.min() < 20 or abp.max() > 300:  # drop nonphysiological ABP
                continue
            src = np.linspace(0.0, 1.0, w)
            rs = np.stack([np.interp(grid, src, seg[:, j]) for j in range(3)], 1)
            segs.append(rs.astype(np.float32))
            sid.append(int(cid))
            kept += 1
        if ci % 20 == 0:
            print(f"  scanned {ci} cases, kept {len(segs)} segs", flush=True)
    X = np.stack(segs, 0)
    out = RAW / "pulsedb"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "segments.npz", X=X, sid=np.array(sid))
    print(f"[OK] pulsedb segments X={X.shape} from {len(set(sid))} cases "
          f"-> {out/'segments.npz'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cases", type=int, default=200)
    ap.add_argument("--target_n", type=int, default=700)
    main(**vars(ap.parse_args()))
