#!/bin/bash
# IoTJ-R1 gap-filler launcher: e8 (CSDI unified-MD backfill, 12x3) +
# e10 (deep baselines under e7 protocols, 2x3x4) on the shared GPU server.
#
# Run on DKE-LLM-04-SOFIA from ~/DMHI/code:
#     nohup bash scripts/run_e8_e10.sh <gpu_id> > runs/e8_e10.log 2>&1 &
#
# Safety: refuses to start if the GPU has < 6 GiB free (wxq's VLLM eats
# ~45 GiB per card; the deep models need ~2-4 GiB each). Re-checks before
# each stage. Interruptible: both e8/e10 skip already-done runs, so rerun
# the same command to resume.
set -u
GPU="${1:-1}"
cd "$(dirname "$0")/.."

free_mib () {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -1
}

guard () {
    free=$(free_mib)
    if [ "$free" -lt 6000 ]; then
        echo "[guard] GPU $GPU has only ${free} MiB free (< 6000) — aborting stage."
        exit 3
    fi
    echo "[guard] GPU $GPU free: ${free} MiB — OK"
}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate riemann_baselines

echo "=== stage 1/2: e8 CSDI backfill (12 domains x 3 seeds) ==="
guard
python scripts/e8_csdi_md.py --gpu "$GPU"

echo "=== stage 2/2: e10 deep baselines on e7 protocols (2 ds x 3 seeds x 4 methods) ==="
guard
python scripts/e10_e7_deep.py --gpu "$GPU"

echo "=== aggregation (CPU) ==="
python scripts/aggregate_e8_e10.py --check && echo "ALL COMPLETE" || echo "INCOMPLETE — rerun to resume"
