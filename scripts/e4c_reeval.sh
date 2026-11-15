#!/usr/bin/env bash
# E4c: re-EVALUATE all E4/E4b sweep checkpoints after the Stage-II gate fix.
# gamma_s was dead before the fix (bit-identical curves); r_L/k/d were run
# with dead Stage II too, so the whole sweep is re-evaluated for consistency.
# Idempotent: skips configs whose runs/e4_stage2_fixed/<ds>/<tag>_seed<n>/
# metrics.json already exists.
set -uo pipefail
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=~/miniconda3/envs/riemann_baselines/bin/python
DS="${1:?dataset required}"
SRC="checkpoints/e4"
OUT="runs/e4_stage2_fixed/${DS}"
mkdir -p "$OUT"

shopt -s nullglob
for CFG in "${SRC}"/${DS}_seed*_*_seed*/; do
  NAME="$(basename "$CFG")"                       # e.g. mocap_karate_seed7_gamma0.02_seed7
  SEED="${NAME##*_seed}"                          # 7
  BASE="${NAME%_seed*}"                           # mocap_karate_seed7_gamma0.02
  TAG="${BASE#${DS}_seed${SEED}_}"                # gamma0.02
  [ -f "${CFG}/config.json" ] || continue
  DEST="${OUT}/${TAG}_seed${SEED}"
  if [ -f "${DEST}/metrics.json" ]; then continue; fi
  SEEDNO="${SEED}"
  # pass the training seed so evaluate picks the same paper cell
  if $PY scripts/evaluate.py --run_dir "$CFG" --rate 0.9 --block_size 20 --seed "$SEEDNO"; then
    mkdir -p "$DEST"; cp "${CFG}/metrics.json" "${DEST}/metrics.json"
    echo "[done] ${DS} ${TAG} seed${SEED}"
  else
    echo "[FAIL] ${DS} ${TAG} seed${SEED}"
  fi
done
echo "[e4c] ${DS} complete."
