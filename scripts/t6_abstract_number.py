#!/usr/bin/env python3
"""T6: compute the abstract's headline number from measured data only.

Metric: per-domain maximal MSE reduction of DMHI vs the strongest LEARNING
baseline (BRITS/SAITS/CSDI/ImputeFormer/TimeMixer), at rho=0.9, 3-seed mean.

DMHI numbers: runs/e2_stage2_fixed/<ds>_seed{s}_r0.9/metrics.json (post
Stage-II-gate fix re-evaluation, 108/108 complete). Baseline numbers: paper
Table II (tab:main results, MAE/MSE @ rho=0.9) transcribed from the
submitted manuscript (read-only source) — these are the published baseline
values, not re-measured here.

Usage: python scripts/t6_abstract_number.py
"""
import json
import pathlib
from statistics import mean

RUNS = pathlib.Path("runs/e2_stage2_fixed")
SEEDS = (7, 42, 123)

# paper Table II MSE @ rho=0.9: {domain_key: {method: MSE}}
# learning baselines only (Linear/Ffill/Mean excluded by definition)
PAPER_MSE = {
    #              BRITS   SAITS   CSDI   ImputeFormer TimeMixer
    "physionet2012":  {"BRITS": 0.753, "SAITS": 0.718, "CSDI": 0.669, "ImputeFormer": 0.710, "TimeMixer": 0.915},
    "mimic":          {"BRITS": 0.247, "SAITS": 0.248, "CSDI": 0.517, "ImputeFormer": 0.286, "TimeMixer": 0.291},
    "air_quality_italy": {"BRITS": 1.029, "SAITS": 0.701, "CSDI": 0.674, "ImputeFormer": 0.747, "TimeMixer": 1.152},
    "gait_uci":       {"BRITS": 0.286, "SAITS": 0.133, "CSDI": 0.060, "ImputeFormer": 0.495, "TimeMixer": 0.450},
    "mocap_karate":   {"BRITS": 0.410, "SAITS": 0.312, "CSDI": 0.307, "ImputeFormer": 0.817, "TimeMixer": 0.947},
    "gas_home":       {"BRITS": 0.340, "SAITS": 0.225, "CSDI": 0.131, "ImputeFormer": 0.371, "TimeMixer": 0.553},
    "delta_robot":    {"BRITS": 1.304, "SAITS": 1.052, "CSDI": 1.081, "ImputeFormer": 1.293, "TimeMixer": 1.146},
    "kuka":           {"BRITS": 0.544, "SAITS": 0.209, "CSDI": 0.109, "ImputeFormer": 0.681, "TimeMixer": 0.639},
    "nanodrone":      {"BRITS": 0.733, "SAITS": 0.423, "CSDI": 0.210, "ImputeFormer": 0.896, "TimeMixer": 0.736},
    "mhealth":        {"BRITS": 0.498, "SAITS": 0.451, "CSDI": 0.346, "ImputeFormer": 0.446, "TimeMixer": 0.613},
    "breizh_rs":      {"BRITS": 0.928, "SAITS": 0.889, "CSDI": 0.834, "ImputeFormer": 0.893, "TimeMixer": 1.025},
    "solo12":         {"BRITS": 0.300, "SAITS": 0.225, "CSDI": 0.423, "ImputeFormer": 0.914, "TimeMixer": 0.841},
}


def main():
    rows = []
    for ds, bl in PAPER_MSE.items():
        vals = []
        for s in SEEDS:
            p = RUNS / f"{ds}_seed{s}_r0.9" / "metrics.json"
            d = json.load(open(p))
            vals.append(d["MSE"])
        dmhi = mean(vals)
        best_bl_name = min(bl, key=bl.get)
        best_bl = bl[best_bl_name]
        red = (best_bl - dmhi) / best_bl * 100.0
        rows.append((ds, dmhi, best_bl_name, best_bl, red))
        print(f"{ds:18s} DMHI(3seed)={dmhi:.4f}  best-bl={best_bl_name:12s} "
              f"{best_bl:.3f}  reduction={red:+6.1f}%")

    best = max(rows, key=lambda r: r[4])
    print("\n=== Headline (max per-domain MSE reduction vs strongest learning baseline) ===")
    print(f"{best[0]}: {best[4]:.1f}%  (DMHI {best[1]:.4f} vs {best[2]} {best[3]:.3f})")

    wins = sum(1 for r in rows if r[4] > 0)
    print(f"domains where DMHI beats ALL learning baselines: {wins}/12")

    out = {"rows": [
        {"domain": r[0], "dmhi_mse_3seed": round(r[1], 4),
         "best_baseline": r[2], "baseline_mse": r[3],
         "reduction_pct": round(r[4], 1)} for r in rows],
        "headline": {"domain": best[0], "reduction_pct": round(best[4], 1)}}
    pathlib.Path("results").mkdir(exist_ok=True)
    with open("results/t6_abstract_number.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/t6_abstract_number.json")


if __name__ == "__main__":
    main()
