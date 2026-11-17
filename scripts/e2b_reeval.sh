#!/usr/bin/env bash
# E2b: re-EVALUATE all E2 checkpoints after the Stage-II gate fix (501c59c).
# Checkpoints unchanged (build() is deterministic); only impute() changed,
# so we re-run evaluate.py on every (ds, seed, rho) and archive into
# runs/e2_stage2_fixed/. Includes per-run timing for Table VI cross-check.
set -uo pipefail
export OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
PY=~/miniconda3/envs/riemann_baselines/bin/python
OUTBASE="runs/e2_stage2_fixed"
mkdir -p "$OUTBASE"

DATASETS=(physionet2012 mimic air_quality_italy gait_uci mocap_karate gas_home delta_robot kuka nanodrone mhealth breizh_rs solo12)

for DS in "${DATASETS[@]}"; do
  for SEED in 7 42 123; do
    RUN="${DS}_seed${SEED}"
    CKPT="checkpoints/e2/${RUN}"
    [ -f "$CKPT/config.json" ] || { echo "[MISS ckpt] $RUN"; continue; }
    for RATE in 0.9 0.7 0.5; do
      DEST="$OUTBASE/${RUN}_r${RATE}"
      if [ -f "${DEST}/metrics.json" ]; then continue; fi
      echo "[eval] $RUN rho=$RATE"
      if $PY scripts/evaluate.py --run_dir "$CKPT" --rate "$RATE" --block_size 20 --seed "$SEED"; then
        mkdir -p "$DEST"
        cp "$CKPT/metrics.json" "$DEST/metrics.json"
      else
        echo "[FAIL eval] $RUN r$RATE"
      fi
    done
  done
done
echo "E2b complete."
