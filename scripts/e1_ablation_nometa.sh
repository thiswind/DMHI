#!/usr/bin/env bash
# E1: no-metric A/B 对照 — 12 datasets x 3 seeds, rho=0.9 block.
# Arm A (default): metric net trained (legacy released path)
# Arm B (nometa):  ablation_no_metric=True (pure closed-form)
# Usage: bash scripts/e1_ablation_nometa.sh [arm]   # arm in {default, nometa}
set -euo pipefail
ARM="${1:-default}"
PY=~/miniconda3/envs/riemann_baselines/bin/python
OUT="runs/e1_${ARM}"
mkdir -p "$OUT"

DATASETS=(physionet2012 mimic air_quality_italy gait_uci mocap_karate gas_home delta_robot kuka nanodrone mhealth breizh_rs solo12)

for DS in "${DATASETS[@]}"; do
  for SEED in 7 42 123; do
    RUN="${DS}_seed${SEED}"
    if [ -f "$OUT/$RUN/metrics.json" ]; then
      echo "[skip] $RUN done"; continue
    fi
    echo "[run ] $RUN (${ARM})"
    $PY - <<PYEOF
import sys, pathlib, json, time
import numpy as np
sys.path.insert(0, "src")
sys.argv = ["train.py"]
from dmhi.method.pipeline import RiemannianImputer
from dmhi.utils.eval_protocol import apply_block_missing, manifold_deviation

ARM = "${ARM}"; DS = "${DS}"; SEED = ${SEED}; OUTDIR = "runs/e1_${ARM}"
np.random.seed(SEED)
ddir = pathlib.Path("data/processed") / DS
X_train = np.load(ddir / "X_train.npy").astype(np.float32)
X_val   = np.load(ddir / "X_val.npy").astype(np.float32)
X_test  = np.load(ddir / "X_test.npy").astype(np.float32)
M_train = np.load(ddir / "M_train.npy").astype(np.int8)
M_val   = np.load(ddir / "M_val.npy").astype(np.int8)
M_test  = np.load(ddir / "M_test.npy").astype(np.int8)
N, T, D = X_train.shape
d = min(12, D - 2)

imp = RiemannianImputer(d=d, epochs=200, patience=20)
if ARM == "nometa":
    imp.ablation_no_metric = True
elif ARM == "default":
    imp.use_learned_metric = True; imp.ablation_no_metric = False
t0 = time.time(); imp.build(X_train, M_train, X_val, M_val); fit_s = time.time() - t0

# block-missing eval at rho=0.9 (paper protocol, block_size=20)
M_eval_test = apply_block_missing(M_test, 20, 0.9, seed=SEED)
t0 = time.time()
X_hat = imp.impute(X_test, M_eval_test)
infer_s = time.time() - t0
eval_mask = (M_test.astype(bool) & ~M_eval_test.astype(bool))
od = np.zeros(D, dtype=bool)
for t in range(T):
    if M_test[:, t].all(1).any(): od |= M_test[:, t].all(0)
err = X_hat[eval_mask] - X_test[eval_mask]
mae = float(np.abs(err).mean()); mse = float((err**2).mean()); rmse = float(np.sqrt(mse))
md = manifold_deviation(X_hat, X_test, eval_mask, obs_dims=od)
out = dict(arm=ARM, dataset=DS, seed=SEED, mae=mae, mse=mse, rmse=rmse, md=md,
           fit_s=round(fit_s,2), infer_s=round(infer_s,4))
OUTDIR="runs/e1_${ARM}"
p = pathlib.Path(OUTDIR) / f"{DS}_seed{SEED}" / "metrics.json"
p.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(p, "w"), indent=2)
print(json.dumps(out))
PYEOF
  done
done
echo "E1 ${ARM} complete."
