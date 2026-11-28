#!/usr/bin/env bash
# E4c: landmark-selection scheme sweep (reviewer R1-3 "sensitivity to different
# landmark selections", advisor R1-3b).
# Compares uniform / random landmark picks against the deployed stratified
# scheme at the SAME r_L = 0.30, k=10, per-domain d, gamma_s=0.02.
#   domains: mocap_karate, delta_robot, physionet2012 (same 3 as E4)
#   seeds:   7 42 123
#   runs:    2 schemes x 3 domains x 3 seeds = 18 (stratified baseline is
#            already covered by the k10/gamma0.02/rL0.30 E4 runs, which are
#            bit-identical to Table II)
# The random scheme uses landmark_seed = the run seed (7/42/123), so each
# replicate gets an independent pick; reproducible via the recorded seed.
# Protocol identical to E4: rho=0.9, block=20, idempotent.
set -uo pipefail
export OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
PY=~/miniconda3/envs/riemann_baselines/bin/python

DS="${1:?dataset required}"
OUT="runs/e4c_landmark_scheme/${DS}"
SCHEMES=(uniform random)

run_one () {  # $1=tag $2...=extra build args
  local TAG="$1"; shift
  for SEED in 7 42 123; do
    local DEST="${OUT}/${TAG}_seed${SEED}"
    if [ -f "${DEST}/metrics.json" ]; then continue; fi
    local CKPT="checkpoints/e4c/${DS}_seed${SEED}_${TAG}_seed${SEED}"
    if [ ! -f "${CKPT}/config.json" ]; then
      $PY scripts/build_checkpoint.py --dataset "$DS" --seed "$SEED" --out_dir "checkpoints/e4c" \
        --run_suffix "${TAG}_seed${SEED}" --landmark_seed "$SEED" "$@" \
        || { echo "[FAIL build] ${DS} ${TAG} seed${SEED}"; continue; }
    fi
    if $PY scripts/evaluate.py --run_dir "$CKPT" --rate 0.9 --block_size 20 --seed "$SEED"; then
      mkdir -p "${DEST}"; cp "${CKPT}/metrics.json" "${DEST}/metrics.json"
      echo "[done] ${DS} ${TAG} seed${SEED}"
    else
      echo "[FAIL eval] ${DS} ${TAG} seed${SEED}"
    fi
  done
}

for S in "${SCHEMES[@]}"; do run_one "${S}" --landmark_scheme "$S"; done
echo "[e4c] ${DS} complete."
