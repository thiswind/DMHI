#!/usr/bin/env python3
"""E4/E4b aggregator: hyperparameter sensitivity curves (R1-3/R1-5, R2-2/R2-9).

Reads runs/e4_sensitivity/<ds>/<tag>_seed<n>/metrics.json, computes seed-mean
+/- std for MAE / MSE / RMSE / MD per sweep point, and emits one figure per
hyperparameter (gamma_s, r_L, k, d), each with one subplot per dataset.
Paper-default value is marked with a vertical dashed line.

Outputs:
  results/e4_sensitivity_curves.json   (all aggregated numbers, traceable)
  results/e4_sensitivity_<param>.pdf   (4 figures, camera-ready)
"""
import json
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.environ.get("E4_RUNS", "runs/e4_stage2_fixed")  # post-gate-fix re-eval
OUT_DIR = "results"
DATASETS = ["mocap_karate", "delta_robot", "physionet2012"]
DS_LABEL = {
    "mocap_karate": "Mocap (high rate)",
    "delta_robot": "Delta Robot (mid rate)",
    "physionet2012": "PhysioNet (low rate)",
}
METRICS = ["MAE", "MSE", "RMSE", "MD"]

# tag patterns per parameter; defaults follow the deployed configuration
PARAMS = {
    "gamma_s": dict(pattern=r"^gamma([0-9.e-]+)$", parse=float,
                    default=0.02, xlabel=r"smoothing coefficient $\gamma_s$",
                    logx=True, order=[0.001, 0.003, 0.01, 0.02, 0.1, 0.3, 1.0]),
    "r_L": dict(pattern=r"^rL([0-9.]+)$", parse=float,
                default=0.30, xlabel=r"landmark fraction $r_L$",
                logx=False, order=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50]),
    "k": dict(pattern=r"^k([0-9]+)$", parse=int,
              default=10, xlabel=r"graph neighborhood size $k$",
              logx=False, order=[3, 6, 10, 16, 24, 30]),
    "d": dict(pattern=r"^d([0-9]+)$", parse=int,
              default=None, xlabel=r"latent chart dimension $d$",
              logx=False, order=None),  # per-dataset defaults below
}
DS_DEFAULT_D = {"mocap_karate": 12, "delta_robot": 4, "physionet2012": 8}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
})


def collect(ds: str, param: str, spec: dict):
    """Return {sweep_value: {metric: (mean, std, n, paths)}} for one dataset."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RUNS, ds, "*_seed*", "metrics.json"))):
        tag = os.path.basename(os.path.dirname(path))
        m = re.match(r"^(.+)_seed\d+$", tag)
        if not m:
            continue
        base = m.group(1)
        g = re.match(spec["pattern"], base)
        if not g:
            continue
        val = spec["parse"](g.group(1))
        rec = json.load(open(path))
        entry = out.setdefault(val, {k: [] for k in METRICS})
        for k in METRICS:
            entry[k].append(float(rec[k]))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_data = {}
    for param, spec in PARAMS.items():
        per_ds = {ds: collect(ds, param, spec) for ds in DATASETS}
        all_data[param] = {
            ds: {
                str(v): {k: [float(np.mean(xs)), float(np.std(xs)), len(xs)]
                         for k, xs in d.items()}
                for v, d in sorted(per_ds_ds.items())
            }
            for ds, per_ds_ds in per_ds.items()
        }

        fig, axes = plt.subplots(1, len(DATASETS), figsize=(7.0, 2.1), sharey=False)
        colors = {"MAE": "#1f77b4", "MSE": "#d62728", "RMSE": "#2ca02c", "MD": "#9467bd"}
        for ax, ds in zip(axes, DATASETS):
            data = per_ds[ds]
            vals = sorted(data.keys())
            default = DS_DEFAULT_D[ds] if param == "d" else spec["default"]
            for k in METRICS:
                means = np.array([np.mean(data[v][k]) for v in vals])
                stds = np.array([np.std(data[v][k]) for v in vals])
                ax.plot(vals, means, "-o", ms=2.5, lw=1.0, color=colors[k],
                        label=k, markeredgewidth=0.4)
                ax.fill_between(vals, means - stds, means + stds,
                                color=colors[k], alpha=0.15, lw=0)
            if default in vals:
                ax.axvline(default, color="0.35", ls="--", lw=0.8)
            if spec["logx"]:
                ax.set_xscale("log")
            ax.set_xlabel(spec["xlabel"])
            ax.set_title(DS_LABEL[ds], fontsize=8)
            ax.grid(True, lw=0.3, alpha=0.4)
            ax.tick_params(length=2.5, width=0.5)
        axes[0].set_ylabel("error (lower is better)")
        # Legend OUTSIDE the data area: dedicated row beneath the subplots,
        # horizontally centered, small font, single-row (4 entries fit).
        handles, labels_ = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="lower center", ncol=len(METRICS),
                   frameon=False, fontsize=7, columnspacing=1.6,
                   handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(pad=0.4)
        fig.subplots_adjust(bottom=0.30)  # reserve room for the outside legend
        fig_path = os.path.join(OUT_DIR, f"e4_sensitivity_{param}.pdf")
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"[fig] {fig_path}  ({param}: "
              + ", ".join(f"{ds} {len(per_ds[ds])}pts" for ds in DATASETS) + ")")

    json_path = os.path.join(OUT_DIR, f"e4_sensitivity_curves_{os.path.basename(RUNS.rstrip('/'))}.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=1)
    print(f"[json] {json_path}")


if __name__ == "__main__":
    sys.exit(main())
