# Reproduction Guide

Three levels, increasing in data requirements.

## Level 0 — smoke test (no data, seconds)

```bash
pip install -e .            # or: pip install -r requirements.txt
python tests/test_smoke.py
```

Builds a `RiemannianImputer` on the synthetic `src/dmhi/utils/sample/` set and checks
the pipeline end to end: finite errors, observed entries preserved. Torch is
**not** required for this level (or for Level 1): the deployed path is pure
NumPy/SciPy closed form.

## Level 1 — evaluate a paper domain end to end (minutes, no downloads)

**Ten of the paper's twelve reported domains ship inside this repository**
under `data/processed/` (see `data/DATA.md`) — nothing to download. DMHI is
**training-free**: despite the historical script name, `train.py` runs no
gradient training in the default configuration — it deterministically
regenerates the checkpoint's closed-form structures (Stage I manifold chart +
Stage II graph-Tikhonov operator) in seconds on a laptop CPU, then
`evaluate.py` applies the paper protocol (block missingness, length 20,
deterministic):

```bash
python scripts/train.py --dataset gas_home --seed 7 \
    --out_dir checkpoints/local

python scripts/evaluate.py --run_dir checkpoints/local/gas_home_seed7 \
    --block_size 20 --rate 0.9 --seed 7
```

Paper protocol reminder: Tables II/III are **rho=0.9, L=20, mask seed =
seed+2, mean over the per-domain seed set** (see `results/README.md`).
`--rate` selects the missing rate, `--mask_seed` overrides the mask RNG seed
if you need to probe mask sensitivity.

**Reference values for this exact run** (gas_home, seed 7, `n_eval=18000`,
mean-fill MAE 0.5816):

| where you rebuild the checkpoint | MAE | MSE | RMSE | MD |
|---|---|---|---|---|
| Linux / OpenBLAS (the paper's stack) | 0.051789 | 0.020789 | 0.144185 | 0.198525 |
| Apple Accelerate (macOS NumPy) | 0.146519 | 0.094725 | 0.307775 | 0.560508 |

Both beat mean-fill by 4x; see the platform note below for why they differ
and which one matches the paper's printed cell (0.051 — the OpenBLAS row).

**What is deterministic, and what is platform-sensitive.** Two different
claims must not be conflated:

1. *Inference is deterministic and portable.* Given a checkpoint and a mask,
   imputation is deterministic; re-evaluating the same released checkpoint on
   Apple M4 vs Linux/OpenBLAS agrees to ~3e-4 in every metric (six-decimal
   reproduction within one environment). No GPU is needed at inference.
2. *Checkpoint construction is deterministic per platform but sensitive to
   the LAPACK backend.* Re-running `train.py` twice on the same machine
   reproduces all six printed decimals (verified on both an Apple M4 and a
   Linux server). Across machines, Stage I's spectral decomposition (PCA +
   Laplacian eigenproblem) picks different bases inside near-degenerate
   subspaces under Apple Accelerate vs OpenBLAS; the downstream manifold
   graph differs, and metrics shift — on gas_home, seed 7, from 0.0518
   (OpenBLAS, matching the paper cell) to 0.1465 (Accelerate). This is a
   property of the closed-form construction, not of any training procedure.

For numbers that line up with the paper's tables on any machine, use the
released checkpoints (Level 2), or rebuild on Linux/OpenBLAS.

To target a specific reported cell, take its hyperparameters from
`configs/<domain>_seed<N>.json` (see `configs/README.md`) and pass them to
`train.py` (they map 1:1 to CLI flags). The three DUA-restricted /
oversized domains (physionet2012, mimic, electricity) are handled in
`data/DATA.md` § "Domains not in the repository".

## Level 2 — released checkpoints + baselines

- **Checkpoints.** The released "Ours" checkpoints for all published runs
  (407 MB uncompressed, 50 runs, 207 MB zip) are distributed as a release
  archive (`dmhi-checkpoints-<version>.zip`, see GitHub *Releases*). The zip
  stores the run directories flat (`<domain>_seed<N>/`), so unpack it into
  `checkpoints/dmhi/`:

  ```bash
  mkdir -p checkpoints/dmhi
  unzip dmhi-checkpoints-v0.1.0.zip -d checkpoints/dmhi
  ```

  Then evaluate directly (no rebuilding):

  ```bash
  python scripts/evaluate.py --run_dir checkpoints/dmhi/hydraulic_seed7 \
      --block_size 20 --seed 7 --deterministic
  ```

  Each run directory carries the `config.json` written when the checkpoint
  was constructed (hyperparameters exact; see `configs/README.md` for the
  reconstruction caveat on non-primary domains). Reference value for the
  released gas_home seed-7 checkpoint: `MAE=0.051499  MSE=0.020727
  RMSE=0.143970  MD=0.198470` — reproducibly to six decimals within one
  environment, matching the paper's reported gas_home cell
  (0.051 / 0.020 / 0.143 / 0.197) at the table's three decimals.
  **DUA note:** checkpoints pickle their anchor arrays, so any run built on
  the DUA-restricted domains (`mimic`, `physionet2012`) must not be
  redistributed — release archives must exclude those run directories;
  rebuild them locally after obtaining the credentialed sources
  (`data/DATA.md`).

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

- **Training-free by construction:** the default `build()` computes
  closed-form structures only (no optimizer, no epochs of gradient descent,
  no learned weights); `use_learned_metric=True` exists solely as an opt-in
  ablation and requires torch. `RiemannianImputer.fit()` does not exist —
  the API is `build()` / `impute()`.
- **Same-mask discipline:** for each cell every method derives the block mask
  from the same `(domain, block_size, seed)` arithmetic, so masks are
  byte-identical across methods. Mask-draw variability is about `1e-3` MAE,
  well below method gaps.
- **Determinism:** given checkpoint and mask, imputation is deterministic on
  CPU — no GPU required at inference; six-decimal reproduction within one
  environment, `1e-3`-scale agreement across different BLAS backends.
- **Data honesty:** ten domains are verifiable with zero external access
  (the paper's twelve minus the two DUA-gated ICU streams); for those two we
  explain in `data/DATA.md` exactly why we cannot hand you the arrays and how
  to obtain them yourself; the oversized `electricity` stream (scale study,
  not a paper table) ships as a release archive instead of bloating git.
