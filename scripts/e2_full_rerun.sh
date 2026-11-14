#!/usr/bin/env bash
# E2: full re-run, pure closed-form (route A), paper protocol (train -> checkpoint -> evaluate).
# 12 datasets x 3 seeds; fit once per (ds,seed), evaluate at rho in {0.5,0.7,0.9}.
# evaluate.py always writes run_dir/metrics.json (overwrite), so we archive it
# after each evaluation into runs/e2_closed_form/<run>_r<rate>/.
set -uo pipefail
PY=~/miniconda3/envs/riemann_baselines/bin/python
OUTBASE="runs/e2_closed_form"
mkdir -p "$OUTBASE"

DATASETS=(physionet2012 mimic air_quality_italy gait_uci mocap_karate gas_home delta_robot kuka nanodrone mhealth breizh_rs solo12)

for DS in "${DATASETS[@]}"; do
  for SEED in 7 42 123; do
    RUN="${DS}_seed${SEED}"
    CKPT="checkpoints/e2/${RUN}"
    if [ -f "$OUTBASE/${RUN}_r0.9/metrics.json" ]; then
      echo "[skip] $RUN done"; continue
    fi
    if [ ! -f "$CKPT/config.json" ]; then
      echo "[fit ] $RUN"
      $PY scripts/train.py --dataset "$DS" --seed "$SEED" --out_dir checkpoints/e2 || { echo "[FAIL fit] $RUN"; continue; }
    fi
    for RATE in 0.5 0.7 0.9; do
      DEST="$OUTBASE/${RUN}_r${RATE}"
      if [ -f "$DEST/metrics.json" ]; then continue; fi
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
echo "E2 complete."
