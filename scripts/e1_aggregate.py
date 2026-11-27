#!/usr/bin/env python3
"""Aggregate E1 A/B results: nometa (pure closed-form) vs default.

Historical arm logs written before the learned-metric branch was removed
(the two arms were bit-identical in 36/36 runs; see REPRODUCE.md). This
script only reads the archived runs/e1_* metrics.json files.
"""
import json, glob, collections
import numpy as np

rows = collections.defaultdict(lambda: collections.defaultdict(list))
for arm in ("nometa", "default"):
    for f in glob.glob(f"runs/e1_{arm}/*/metrics.json"):
        d = json.load(open(f))
        rows[d["dataset"]][arm].append(d)

paper = {  # manuscript Table II (MAE) / Table III (RMSE) at rho=0.9, for reference
    "physionet2012": 0.393, "mimic": 0.095, "air_quality_italy": 0.377,
    "gait_uci": 0.150, "mocap_karate": 0.075, "gas_home": 0.051,
    "delta_robot": 0.198, "kuka": 0.159, "nanodrone": 0.192,
    "mhealth": 0.197, "breizh_rs": 0.310, "solo12": 0.183,
}
paper_rmse = {"physionet2012": 0.682, "mimic": 0.341, "air_quality_italy": 0.637,
    "gait_uci": 0.213, "mocap_karate": 0.162, "gas_home": 0.143,
    "delta_robot": 0.534, "kuka": 0.288, "nanodrone": 0.380,
    "mhealth": 0.604, "breizh_rs": 0.555, "solo12": 0.260}

print(f"{'dataset':<18}{'arm':<9}{'n':<4}{'MAE':>8}{'RMSE':>8}{'fit_s':>8}{'inf_s':>8}  paper_MAE/RMSE")
for ds in sorted(rows):
    for arm in ("nometa", "default"):
        lst = rows[ds].get(arm, [])
        if not lst:
            continue
        mae = np.mean([d["mae"] for d in lst]); rmse = np.mean([d["rmse"] for d in lst])
        fit = np.mean([d["fit_s"] for d in lst]); inf = np.mean([d["infer_s"] for d in lst])
        pm = paper.get(ds, float("nan")); pr = paper_rmse.get(ds, float("nan"))
        print(f"{ds:<18}{arm:<9}{len(lst):<4}{mae:>8.4f}{rmse:>8.4f}{fit:>8.1f}{inf:>8.2f}  {pm:.3f}/{pr:.3f}")
