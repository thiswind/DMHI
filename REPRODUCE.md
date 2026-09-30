# Reproduction Guide

Three levels, increasing in data requirements.

## Level 0 — smoke test (no data, seconds)

```bash
pip install -e .            # or: pip install -r requirements.txt
python smoke_test.py
```

Fits a `RiemannianImputer` on the synthetic `data/sample/` set and checks
the pipeline end to end: finite errors, observed entries preserved.

## Level 1 — train + evaluate a real paper domain (minutes, no downloads)

**17 of the paper's 20 domains ship inside this repository** under
`data/processed/` (see `data/DATA.md`) — nothing to download. Train the
deployed configuration from scratch, then evaluate it exactly as the paper
does (block missingness, length 20, deterministic):

```bash
python experiments/train_ours.py --dataset gas_home --seed 7 \
    --out_dir checkpoints/ours

python experiments/eval_ours.py --run_dir checkpoints/ours/gas_home_seed7 \
    --block_size 20 --seed 7
```

Verified reference run (Apple M4, CPU only, Python 3.12): training 0.4 min;
`MAE=0.150197  RMSE=0.327957  MRE=0.251315  TE=0.259683` against a
mean-fill MAE of 0.5976. Because DMHI's inference path is deterministic,
**re-running the evaluation on the same checkpoint and mask reproduces these
numbers to six decimals**; re-running training with the same seed reproduces
them up to floating-point reduction order (paper Table on N=4 determinism).

To target a specific reported cell, take its hyperparameters from
`configs/<domain>_seed<N>.json` (see `configs/README.md`) and pass them to
`train_ours.py` (they map 1:1 to CLI flags). The three DUA-restricted /
oversized domains (physionet2012, mimic, electricity) are handled in
`data/DATA.md` § "Domains not in the repository".

## Level 2 — released checkpoints + baselines

- **Checkpoints.** The trained "Ours" checkpoints for all reported cells
  (≈575 MB uncompressed (51 runs, 252 MB zip)) are distributed as a release archive
  (`dmhi-checkpoints-<version>.zip`, see GitHub *Releases*). Unpack under
  `checkpoints/ours/` and evaluate without retraining:

  ```bash
  python experiments/eval_ours.py --run_dir checkpoints/ours/hydraulic_seed7 \
      --block_size 20 --seed 7 --deterministic
  ```

  Each run directory carries the `config.json` written at training time
  (hyperparameters exact; see `configs/README.md` for the reconstruction
  caveat on non-primary domains).

- **Baselines.** `experiments/baselines/` runs classical and PyPOTS
  baselines (`run_classical.py`, `run_saits.py`, `run_timemixer.py`,
  `run_imputeformer.py`, …) on the same shipped arrays with byte-identical
  masks (`baselines/common.py`); PyPOTS runners need
  `pip install -e ".[baselines]"`. The two runners backed by external author
  repositories expect local checkouts at `csdi_official/` and
  `psw_i_official/` (not redistributable here; provenance in
  `baselines/README.md`).

- **e4 analyses.** MD k-sensitivity, downstream transfer, N=4 determinism:
  `experiments/e4*.py`, consuming the same artifacts.

## Notes for reviewers

- **Same-mask discipline:** for each cell every method derives the block mask
  from the same `(domain, block_size, seed)` arithmetic, so masks are
  byte-identical across methods. Mask-draw variability is about `1e-3` MAE,
  well below method gaps.
- **Determinism:** given checkpoint and mask, imputation is deterministic on
  CPU — no GPU required at inference; six-decimal reproduction across
  platforms.
- **Data honesty:** 17 domains are verifiable with zero external access; for
  the two DUA-restricted domains we explain in `data/DATA.md` exactly why we
  cannot hand you the arrays and how to obtain them yourself; the oversized
  `electricity` stream ships as a release archive instead of bloating git.
