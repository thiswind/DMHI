#!/usr/bin/env bash
# E4: wide-range sensitivity sweep (MASTER-PLAN D4, R1-5 / R2-2 / R2-9).
# Usage: bash scripts/e4_sensitivity.sh <dataset>
# One-variable-at-a-time around paper defaults, rho=0.9 block20, 3 seeds.
# gamma grid is log-spaced 0.001->1 with 7 points and INCLUDES the paper
# default 0.02; landmark fraction 5%-50% (6 pts, incl. default 0.30);
# graph k in {3,6,10,16,24,30} (incl. default 10).
set -uo pipefail
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=~/miniconda3/envs/riemann_baselines/bin/python

DS="${1:?dataset required}"
OUT="runs/e4_sensitivity/${DS}"
GAMMAS=(0.001 0.003 0.01 0.02 0.1 0.3 1.0)
RLS=(0.05 0.10 0.20 0.30 0.40 0.50)
KS=(3 6 10 16 24 30)

run_one () {  # $1=tag $2...=extra train args
  local TAG="$1"; shift
  for SEED in 7 42 123; do
    local DEST="${OUT}/${TAG}_seed${SEED}"
    if [ -f "${DEST}/metrics.json" ]; then continue; fi
    # build_checkpoint.py run_id format: {dataset}_seed{seed}_{suffix}
    local CKPT="checkpoints/e4/${DS}_seed${SEED}_${TAG}_seed${SEED}"
    if [ ! -f "${CKPT}/config.json" ]; then
      $PY scripts/build_checkpoint.py --dataset "$DS" --seed "$SEED" --out_dir "checkpoints/e4" \
        --run_suffix "${TAG}_seed${SEED}" "$@" \
        || { echo "[FAIL train] ${DS} ${TAG} seed${SEED}"; continue; }
    fi
    # evaluate.py has no --out_dir; it writes CKPT/metrics.json (overwrite).
    if $PY scripts/evaluate.py --run_dir "$CKPT" --rate 0.9 --block_size 20 --seed "$SEED"; then
      mkdir -p "${DEST}"; cp "${CKPT}/metrics.json" "${DEST}/metrics.json"
      echo "[done] ${DS} ${TAG} seed${SEED}"
    else
      echo "[FAIL eval] ${DS} ${TAG} seed${SEED}"
    fi
  done
}

for G in "${GAMMAS[@]}"; do run_one "gamma${G}" --spline_gamma "$G"; done
for R in "${RLS[@]}";    do run_one "rL${R}"     --r_L "$R"; done
for K in "${KS[@]}";     do run_one "k${K}"      --k "$K"; done
echo "[e4] ${DS} complete."
