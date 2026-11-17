#!/usr/bin/env python3
"""Aggregate E2 metrics.json files into seed-mean table (MAE/MSE/RMSE/MD/latency).

Run on the server: python scripts/e2_aggregate.py
Reads runs/e2_closed_form/<ds>_seed<s>_r<rate>/metrics.json (108 files).
"""
import json
import glob
import collections

import numpy as np

rows = collections.defaultdict(list)
for f in sorted(glob.glob("runs/e2_closed_form/*/metrics.json")):
    m = json.load(open(f))
    rows[(m["dataset"], m["missing_rate"])].append(
        (m["MAE"], m["MSE"], m["RMSE"], m.get("MD", float("nan")),
         m.get("infer_ms_per_sample", float("nan")))
    )

print(f"{'dataset':<18}{'rate':<6}{'MAE':>10}{'MSE':>10}{'RMSE':>10}{'MD':>10}{'ms/smp':>8}  n")
for (ds, rate) in sorted(rows.keys()):
    a = np.array(rows[(ds, rate)], dtype=float)
    mean = a.mean(axis=0)
    print(f"{ds:<18}{rate:<6}{mean[0]:>10.4f}{mean[1]:>10.4f}{mean[2]:>10.4f}"
          f"{mean[3]:>10.4f}{mean[4]:>8.2f}  {len(a)}")
