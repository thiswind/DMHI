# Reproduction Guide

Three levels, increasing in data requirements.

## Level 0 — smoke test (no data, seconds)

```bash
pip install -e .            # or: pip install -r requirements.txt
python tests/test_smoke.py
```

Fits a `RiemannianImputer` on the synthetic `src/dmhi/utils/sample/` set and checks
the pipeline end to end: finite errors, observed entries preserved.

## Level 1 — evaluate a paper domain end to end (minutes, no downloads)

**Ten of the paper's twelve reported domains ship inside this repository**
under `data/processed/` (see `data/DATA.md`) — nothing to download. DMHI
needs **no training phase to impute**; to keep this level download-free, the
command below first regenerates the checkpoint's fitting stages locally
(Stage I embedding + Stage II metric refinement — minutes on a laptop CPU),
then evaluates exactly as the paper does (block missingness, length 20,
deterministic):

```bash
python scripts/train.py --dataset gas_home --seed 7 \
    --out_dir checkpoints/dmhi

python scripts/evaluate.py --run_dir checkpoints/dmhi/gas_home_seed7 \
    --block_size 20 --rate 0.9 --seed 7
```

Paper protocol reminder: Tables II/III are **rho=0.9, L=20, mask seed =
seed+2, mean over the per-domain seed set** (see `results/README.md`).
`--rate` selects the missing rate, `--mask_seed` overrides the mask RNG seed
if you need to probe mask sensitivity. Reference value for the released
gas_home seed-7 checkpoint: `MAE=0.051499  MSE=0.020727  RMSE=0.143970
MD=0.198470` — reproducibly to six decimals within one environment, matching
the paper's reported gas_home cell (0.051 / 0.020 / 0.143 / 0.197) at the
table's three decimals.

Verified reference run (Apple M4, CPU only, Python 3.12): training 0.4 min;
`MAE=0.150197  RMSE=0.327957  MRE=0.251315  TE=0.259683` against a
mean-fill MAE of 0.5976.

**What reproduces exactly, and what does not.** DMHI's *inference* is
deterministic: the same checkpoint + the same mask reproduce metrics to six
decimals within one environment. Weights are byte-identical regardless of
platform (verified: the released `.pkl` leaves deserialize with zero
difference, and MPS vs CPU give the same output), but the linear algebra
below them is not: different BLAS backends (Apple Accelerate vs
Linux/OpenBLAS) shift the third decimal by about `1e-3`. No such shift
changes any bold/underline verdict in the paper — see `results/README.md`
for the full 12-domain comparison. *Training*, however, is
hardware-sensitive: the same seed on GPU vs CPU yields different Stage II
optimization trajectories (e.g. a CPU re-train of gas_home lands at 0.1502
while the released checkpoint evaluates at 0.0515 — both beat mean-fill
0.5976 by a wide margin). Same-hardware re-runs are stable (the paper's N=4
determinism table). For numbers that line up with the paper's tables, use
the released checkpoints (Level 2).

To target a specific reported cell, take its hyperparameters from
`configs/<domain>_seed<N>.json` (see `configs/README.md`) and pass them to
`train.py` (they map 1:1 to CLI flags). The three DUA-restricted /
oversized domains (physionet2012, mimic, electricity) are handled in
`data/DATA.md` § "Domains not in the repository".

## Level 2 — released checkpoints + baselines

- **Checkpoints.** The trained "Ours" checkpoints for all released runs
  (407 MB uncompressed, 50 runs, 207 MB zip) are distributed as a release
  archive (`dmhi-checkpoints-<version>.zip`, see GitHub *Releases*). The zip
  stores the run directories flat (`<domain>_seed<N>/`), so unpack it into
  `checkpoints/dmhi/`:

  ```bash
  mkdir -p checkpoints/dmhi
  unzip dmhi-checkpoints-v0.1.0.zip -d checkpoints/dmhi
  ```

  Then evaluate without retraining:

  ```bash
  python scripts/evaluate.py --run_dir checkpoints/dmhi/hydraulic_seed7 \
      --block_size 20 --seed 7 --deterministic
  ```

  Each run directory carries the `config.json` written at training time
  (hyperparameters exact; see `configs/README.md` for the reconstruction
  caveat on non-primary domains).

- **Baselines.** `scripts/baselines/` runs classical and PyPOTS
  baselines (`run_classical.py`, `run_saits.py`, `run_timemixer.py`,
  `run_imputeformer.py`, …) on the same shipped arrays with byte-identical
  masks (`baselines/common.py`); PyPOTS runners need
  `pip install -e ".[baselines]"`. The two runners backed by external author
  repositories expect local checkouts at `csdi_official/` and
  `psw_i_official/` (not redistributable here; provenance in
  `baselines/README.md`).

- **e4 analyses.** MD k-sensitivity, downstream transfer, N=4 determinism:
  `scripts/md_k_sensitivity.py`, `scripts/downstream_*.py`, `scripts/determinism_check.py`, consuming the same artifacts.

## Notes for reviewers

- **Same-mask discipline:** for each cell every method derives the block mask
  from the same `(domain, block_size, seed)` arithmetic, so masks are
  byte-identical across methods. Mask-draw variability is about `1e-3` MAE,
  well below method gaps.
- **Determinism:** given checkpoint and mask, imputation is deterministic on
  CPU — no GPU required at inference; six-decimal reproduction within one
  environment, `1e-3` agreement across different BLAS backends.
- **Data honesty:** ten domains are verifiable with zero external access
  (the paper's twelve minus the two DUA-gated ICU streams); for those two we
  explain in `data/DATA.md` exactly why we cannot hand you the arrays and how
  to obtain them yourself; the oversized `electricity` stream (scale study,
  not a paper table) ships as a release archive instead of bloating git.
