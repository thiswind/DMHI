#!/usr/bin/env bash
# E4b: latent-dimension sensitivity sweep (4th hyperparameter, meeting 2026-11-15).
# Companion to e4_sensitivity.sh (gamma_s / r_L / k already done there).
# Grid per dataset, clipped to d <= D-2 (train.py auto rule: min(12, D-2)),
# default point always included:
#   delta_robot    D=6  -> {2,3,4}    (default 4)
#   physionet2012  D=10 -> {2,4,8}    (default 8)
#   mocap_karate   D=39 -> {2,4,8,12,16,24} (default 12)
# Same protocol as E4: rho=0.9, block=20, 3 seeds, idempotent.
set -uo pipefail
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=~/miniconda3/envs/riemann_baselines/bin/python

DS="${1:?dataset required}"
OUT="runs/e4_sensitivity/${DS}"
case "$DS" in
  delta_robot)   DVALS=(2 3 4) ;;
  physionet2012) DVALS=(2 4 8) ;;
  mocap_karate)  DVALS=(2 4 8 12 16 24) ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac

run_one () {  # $1=tag $2...=extra train args
  local TAG="$1"; shift
  for SEED in 7 42 123; do
    local DEST="${OUT}/${TAG}_seed${SEED}"
    if [ -f "${DEST}/metrics.json" ]; then continue; fi
    local CKPT="checkpoints/e4/${DS}_seed${SEED}_${TAG}_seed${SEED}"
    if [ ! -f "${CKPT}/config.json" ]; then
      $PY scripts/train.py --dataset "$DS" --seed "$SEED" --out_dir "checkpoints/e4" \
        --run_suffix "${TAG}_seed${SEED}" "$@" \
        || { echo "[FAIL train] ${DS} ${TAG} seed${SEED}"; continue; }
    fi
    if $PY scripts/evaluate.py --run_dir "$CKPT" --rate 0.9 --block_size 20 --seed "$SEED"; then
      mkdir -p "${DEST}"; cp "${CKPT}/metrics.json" "${DEST}/metrics.json"
      echo "[done] ${DS} ${TAG} seed${SEED}"
    else
      echo "[FAIL eval] ${DS} ${TAG} seed${SEED}"
    fi
  done
}

for DV in "${DVALS[@]}"; do run_one "d${DV}" --d "$DV"; done
echo "[e4b] ${DS} complete."
