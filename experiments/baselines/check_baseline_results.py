#!/usr/bin/env python3
"""
Verify completeness of baseline experiment results.

For each (method, dataset, rate, pattern) combination, checks whether the
corresponding metrics.json exists in the output directory.

Reports missing results so failed experiments can be re-run.

Usage:
    python check_baseline_results.py --out_dir experiments/logs/baselines
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import iter_configs, metrics_filename

GPU_METHODS = ["saits", "brits", "csdi", "imputeformer", "timemixer", "psw_i"]
CLASSICAL_METHODS = ["mean", "linear", "locf"]
ALL_METHODS = CLASSICAL_METHODS + GPU_METHODS


def check(out_dir: str) -> tuple:
    out_dir = pathlib.Path(out_dir)
    missing = []
    present = []

    for method in ALL_METHODS:
        for ds, rate, pat in iter_configs():
            fname = metrics_filename(method, ds, rate, pat)
            fpath = out_dir / fname
            if fpath.exists():
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    if "mae" in data and data["mae"] > 0:
                        present.append((method, ds, rate, pat))
                    else:
                        missing.append((method, ds, rate, pat, "invalid_json"))
                except Exception as e:
                    missing.append((method, ds, rate, pat, f"parse_error: {e}"))
            else:
                missing.append((method, ds, rate, pat, "missing"))

    return present, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="experiments/logs/baselines")
    args = parser.parse_args()

    present, missing = check(args.out_dir)

    total = len(present) + len(missing)
    print(f"\nResults completeness: {len(present)}/{total}")
    print(f"Present : {len(present)}")
    print(f"Missing : {len(missing)}")

    if missing:
        print("\n=== MISSING / INVALID ===")
        for item in missing:
            method, ds, rate, pat, reason = item
            print(f"  {method:15s} {ds:20s} rate={rate:.1f} pat={pat:10s} [{reason}]")
        sys.exit(1)
    else:
        print("\n✓ All results present and valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()
