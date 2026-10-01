"""UCI Air Quality (Italy, id 360).

9358 hourly rows × 15 cols. Native -200 sentinel marks missingness.
13 numeric chemical+ambient features (drop Date, Time text cols).

Strategy A: sliding window T=24 (1 day), stride=12.
Strategy B (fallback): T=12 stride=6.

70/10/20 split by time order (no shuffling — time series).
"""
from __future__ import annotations

import json
import pathlib
import zipfile
from typing import Any

import numpy as np


FEATURES = [
    "CO_GT", "PT08_S1_CO", "NMHC_GT", "C6H6_GT", "PT08_S2_NMHC",
    "NOx_GT", "PT08_S3_NOx", "NO2_GT", "PT08_S4_NO2", "PT08_S5_O3",
    "T", "RH", "AH",
]


def _unzip(raw_dir: pathlib.Path) -> pathlib.Path:
    raw_dir = pathlib.Path(raw_dir)
    csvs = list(raw_dir.rglob("AirQualityUCI.csv"))
    if csvs:
        return csvs[0]
    zips = list(raw_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip and no AirQualityUCI.csv in {raw_dir}")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(raw_dir)
    csvs = list(raw_dir.rglob("AirQualityUCI.csv"))
    if not csvs:
        raise FileNotFoundError("AirQualityUCI.csv not found after extract")
    return csvs[0]


def _parse_csv(csv_path: pathlib.Path) -> np.ndarray:
    """Parse semicolon-delimited Italian-locale CSV. Returns (N, 13) np.float32 with NaN at -200."""
    rows = []
    with open(csv_path, encoding="latin-1") as f:
        header = f.readline().strip().split(";")
        idx_map = {}
        # UCI uses these column names exactly:
        name_map = {
            "CO_GT": "CO(GT)", "PT08_S1_CO": "PT08.S1(CO)", "NMHC_GT": "NMHC(GT)",
            "C6H6_GT": "C6H6(GT)", "PT08_S2_NMHC": "PT08.S2(NMHC)",
            "NOx_GT": "NOx(GT)", "PT08_S3_NOx": "PT08.S3(NOx)",
            "NO2_GT": "NO2(GT)", "PT08_S4_NO2": "PT08.S4(NO2)",
            "PT08_S5_O3": "PT08.S5(O3)", "T": "T", "RH": "RH", "AH": "AH",
        }
        for our_name, csv_name in name_map.items():
            if csv_name not in header:
                raise KeyError(f"Column {csv_name} not in {header}")
            idx_map[our_name] = header.index(csv_name)
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < max(idx_map.values()) + 1:
                continue
            row = []
            for fname in FEATURES:
                tok = parts[idx_map[fname]].replace(",", ".").strip()
                if not tok:
                    row.append(np.nan)
                else:
                    try:
                        v = float(tok)
                        row.append(np.nan if v == -200 else v)
                    except ValueError:
                        row.append(np.nan)
            # Skip rows that are entirely empty / "blank trailing rows"
            if not all(np.isnan(v) for v in row):
                rows.append(row)
    return np.array(rows, dtype=np.float32)


def preprocess(raw_dir: str | pathlib.Path, out_dir: str | pathlib.Path,
               strategy: str = "A") -> dict[str, Any]:
    raw_dir = pathlib.Path(raw_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = _unzip(raw_dir)
    arr = _parse_csv(csv_path)  # (N_total, 13)

    if strategy == "A":
        T, stride = 24, 12
    elif strategy == "B":
        T, stride = 12, 6
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    # Standardize per-feature using non-NaN values
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, keepdims=True) + 1e-8
    arr_norm = (arr - mu) / sd

    M_native = (~np.isnan(arr_norm)).astype(np.int8)
    arr_filled = np.where(np.isnan(arr_norm), 0.0, arr_norm).astype(np.float32)

    # Sliding windows in time order
    N_total = arr_filled.shape[0]
    X_list, M_list = [], []
    for start in range(0, N_total - T + 1, stride):
        X_list.append(arr_filled[start:start + T])
        M_list.append(M_native[start:start + T])
    X = np.stack(X_list, axis=0).astype(np.float32)
    M = np.stack(M_list, axis=0).astype(np.int8)
    N, T_, D = X.shape
    assert T_ == T and D == len(FEATURES)

    # Drop windows that are >50% missing globally
    keep = M.mean(axis=(1, 2)) >= 0.5
    X = X[keep]
    M = M[keep]
    N = X.shape[0]

    # 70/10/20 split by time order (no shuffle)
    n_train, n_val = int(0.7 * N), int(0.1 * N)
    tr = np.arange(n_train)
    va = np.arange(n_train, n_train + n_val)
    te = np.arange(n_train + n_val, N)

    np.save(out_dir / "X_train.npy", X[tr])
    np.save(out_dir / "M_train.npy", M[tr])
    np.save(out_dir / "X_val.npy", X[va])
    np.save(out_dir / "M_val.npy", M[va])
    np.save(out_dir / "X_test.npy", X[te])
    np.save(out_dir / "M_test.npy", M[te])

    manifest = {
        "dataset": "air_quality_italy",
        "strategy": strategy,
        "shape": {"N_train": int(tr.size), "N_val": int(va.size),
                  "N_test": int(te.size), "T": int(T_), "D": int(D)},
        "features": FEATURES,
        "window_T": T,
        "stride": stride,
        "standardized": True,
        "mean": mu.flatten().tolist(),
        "std": sd.flatten().tolist(),
        "native_missing_ratio": float(1 - M.mean()),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strategy", default="A")
    args = ap.parse_args()
    m = preprocess(args.raw, args.out, args.strategy)
    print(json.dumps(m, indent=2))
