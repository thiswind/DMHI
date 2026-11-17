# Reproduction Guide

Three levels, increasing in data requirements.

> **Which level reproduces the paper's table numbers?** Level 2 (released
> checkpoints + evaluate). Level 1 rebuilding is a sanity check whose
> numbers can shift with your CPU architecture — see the platform note
> below before comparing anything against the tables.

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
**training-free**: `scripts/build_checkpoint.py` runs no model training
in the default configuration — it deterministically constructs the
checkpoint artifacts (Stage I manifold chart via spectral decomposition plus
a fixed-schedule landmark refinement; Stage II graph-Tikhonov operator in
closed form; no learned weights, no validation early-stopping) in seconds
on a laptop CPU, then `evaluate.py`
applies the paper protocol (block missingness, length 20, deterministic):

```bash
python scripts/build_checkpoint.py --dataset gas_home --seed 7 \
    --out_dir checkpoints/local

python scripts/evaluate.py --run_dir checkpoints/local/gas_home_seed7 \
    --block_size 20 --rate 0.9 --seed 7
```

Paper protocol reminder: Tables II/III are **rho=0.9, L=20, mask seed =
seed+2, mean over the per-domain seed set** (see `results/README.md`).
`--rate` selects the missing rate, `--mask_seed` overrides the mask RNG seed
if you need to probe mask sensitivity.

**Reference values for this exact run** (gas_home, seed 7, `n_eval=18000`,
mean-fill MAE 0.5816; paper protocol is mask seed = seed+2):

| where you rebuild the checkpoint | MAE | MSE | RMSE | MD |
|---|---|---|---|---|
| x86-64 Linux (the paper's stack) | 0.051789 | 0.020789 | 0.144185 | 0.198525 |
| arm64 macOS (Accelerate *or* OpenBLAS numpy) | 0.146519 | 0.094725 | 0.307775 | 0.560508 |

Both rows beat mean-fill by 4x; see the platform note below for why they
differ and which one matches the paper's printed cell (0.051 — the x86-64
row). Do not compare a rebuilt-on-arm64 number against the paper's tables.

**What is deterministic, and what is platform-sensitive.** Two different
claims must not be conflated:

1. *Inference is deterministic and portable.* Given a checkpoint and a mask,
   imputation is deterministic; re-evaluating the same released checkpoint
   reproduces the paper's table numbers bit-for-bit on x86-64 (all 36 runs;
   plus a mocap spot-check under the newest scipy 1.16.3 / scikit-learn
   1.9.0). On arm64 (Apple Silicon) 40 of 48 printed cells match exactly;
   the rest can differ by ±1 in the third decimal (absolute difference
   ≤ 0.003, BLAS SIMD reduction order) with no change to any ranking or
   conclusion. No GPU is needed at inference.
2. *Checkpoint construction is deterministic per machine but not portable
   across CPU architectures.* Re-running `build_checkpoint.py` twice on the
   same machine reproduces all six printed decimals (verified on an Apple
   M4 and on a Linux/x86-64 server, and bit-identically across numpy 2.x
   and numpy 1.26 on the same M4). Across architectures (arm64 vs x86-64),
   floating-point SIMD reduction orders differ; Stage I's spectral
   decompositions pick different bases inside near-degenerate subspaces,
   and the iterative landmark refinement amplifies the bit-level
   differences, so metrics can shift visibly — on gas_home, seed 7, from
   0.0518 (x86-64 Linux, matching the paper cell) to 0.1465 (arm64 macOS,
   under both Accelerate and OpenBLAS numpy builds). Rebuilding on a
   non-x86-64 machine prints a NOTE pointing here. This is a property of
   the closed-form construction, not of any training procedure.

For numbers that line up with the paper's tables on any machine, use the
released checkpoints (Level 2), or rebuild on x86-64 Linux.

To target a specific reported cell, take its hyperparameters from
`configs/<domain>_seed<N>.json` (see `configs/README.md`) and pass them to
`build_checkpoint.py` (they map 1:1 to CLI flags). The two DUA-gated
domains (physionet2012, mimic) are covered in `data/DATA.md`.

## Level 2 — released checkpoints + baselines

- **Checkpoints.** The released "Ours" checkpoints for the 12 paper-table
  domains (213.6 MB zip, 36 runs: 12 domains x seeds {7, 42, 123}) are
  distributed as a release archive (`dmhi-checkpoints-v0.2.0.zip`, see
  GitHub *Releases*). The zip stores the run directories flat
  (`<domain>_seed<N>/`) together with a `README.md`, so unpack it into
  `checkpoints/dmhi/`:

  ```bash
  mkdir -p checkpoints/dmhi
  unzip dmhi-checkpoints-v0.2.0.zip -d checkpoints/dmhi
  ```

  Then evaluate directly (no rebuilding):

  ```bash
  python scripts/evaluate.py --run_dir checkpoints/dmhi/gas_home_seed7 \
      --rate 0.9 --block_size 20 --seed 7
  ```

  (Drop `--deterministic` unless torch is installed — it is an optional
  strictness flag, not needed for exact reproduction.) Each run directory
  carries the `config.json` written when the checkpoint was constructed
  (hyperparameters exact; see `configs/README.md` for the reconstruction
  caveat on non-primary domains) and a `metrics.json` sidecar computed at
  the paper protocol (rho=0.9) on x86-64 — the reference value for
  gas_home seed 7 is `MAE=0.051789  MSE=0.020789  RMSE=0.144185
  MD=0.198525`, matching the paper's reported gas_home cell
  (0.051 / 0.020 / 0.143 / 0.197) at the table's three decimals.
  **DUA domains.** The raw arrays for `mimic` / `physionet2012` are not in
  git; during peer review their *processed* arrays ship as a second
  release asset `dmhi-dua-arrays-v0.2.0.zip` (14 MB) — unpack it under
  `data/processed/` and every domain in the table becomes reproducible
  with zero flags:

  ```bash
  unzip dmhi-dua-arrays-v0.2.0.zip -d data/processed/
  python scripts/reproduce_table.py
  ```

  The script evaluates all 12 domains x 3 seeds at the paper protocol and
  prints the complete DMHI row (per-run values + 3-seed means). Without
  the DUA zip it still runs, auto-skipping the two gated domains with a
  note. See `data/DATA.md` for the credentialed acquisition route.

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

- **Training-free by construction:** the default `build()` involves no
  learned weights and no data-driven stopping (no optimizer over model
  parameters, no validation early-stopping): Stage I combines closed-form
  spectral decomposition with a fixed-schedule deterministic landmark
  refinement, and Stage II is solved in closed form.
  `use_learned_metric=True` exists solely as an opt-in
  ablation and requires torch. The public API is `build()` / `impute()` —
  there is no `fit()` and no training entry point.
- **Same-mask discipline:** for each cell every method derives the block mask
  from the same `(domain, block_size, seed)` arithmetic, so masks are
  byte-identical across methods. Mask-draw variability is about `1e-3` MAE,
  well below method gaps.
- **Determinism:** given checkpoint and mask, imputation is deterministic on
  CPU — no GPU required at inference; six-decimal reproduction within one
  environment, `1e-3`-scale agreement across different BLAS backends.
- **Data honesty:** ten domains are verifiable with zero external access;
  the two DUA-gated ICU streams ship as a review-stage array zip (see
  Level 2) with the credentialed acquisition route in `data/DATA.md`;
  the oversized `electricity` stream (scale study, not a paper table)
  stays out of git by size policy.
