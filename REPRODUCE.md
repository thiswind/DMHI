# Reproduction Guide

Three levels, increasing in data requirements.

## Level 0 — smoke test (no data, seconds)

```bash
pip install -e .            # or: pip install -r requirements.txt
python smoke_test.py
```

This fits a `RiemannianImputer` on the synthetic `data/sample/` set, checks
that the pipeline runs end to end, returns finite errors, and preserves
observed entries.

## Level 1 — re-run model inference on real data

The full processed splits live in `data/processed/<domain>/` and the trained
"Ours" checkpoints in `checkpoints/ours/<domain>_seed<N>/`. These binary
files are not tracked in git (see `data/DATA.md`); copy them from the
training server:

```bash
# requires SSH access to the training server
scp -r DKE-LLM-04-SOFIA:~/workspaces/riemannian-ts-imputation/data/processed/ \
    data/

scp -r \
    "DKE-LLM-04-SOFIA:~/workspaces/riemannian-ts-imputation/experiments/logs/recompute033/ours/" \
    checkpoints/
```

Once in place, evaluate a single reported cell:

```bash
# hydraulic seed 7 at the deployed block size
python experiments/eval_ours.py \
    --run_dir checkpoints/ours/hydraulic_seed7 \
    --block_size 20 --seed 7 --deterministic
```

`eval_ours.py` reads `config.json` from the run directory for `dataset`,
`missing_rate`, and `pattern`; draws the block-missing eval mask; loads the
checkpoint; and prints MAE / RMSE / MRE / TE.

Every checkpoint directory in `checkpoints/ours/` ships a `config.json` with
the exact hyperparameters used for the corresponding paper cell. The matching
per-cell configs are mirrored in `configs/` for browsing (see
`configs/README.md`).

### Baselines

`experiments/baselines/` runs the classical and PyPOTS baselines
(`run_classical.py`, `run_saits.py`, `run_timemixer.py`, `run_csdi.py`, …).
They read data from `data/processed/` via repository-relative paths and
evaluate on byte-identical masks (`baselines/common.py`). The PyPOTS-backed
runners need the optional extra (`pip install -e ".[baselines]"`); the two
runners backed by external author repositories expect local checkouts at
`csdi_official/` and `psw_i_official/` (not included — see
`baselines/README.md`). Per-runner provenance:
`experiments/baselines/README.md`.

The paper's e4 analyses (MD k-sensitivity, downstream transfer, N=4
determinism) re-run from the same artifacts via `experiments/e4*.py`.

## Notes for reviewers

- **Same-mask discipline:** for each `(domain, block length)` cell every
  method sees the same generator and seed arithmetic, so masks are
  byte-identical across methods. Mask-draw variability is about `1e-3` MAE,
  well below method gaps.
- **Determinism:** given the checkpoint and mask, imputation reproduces to
  six decimals across platforms; no GPU is required at inference.
- **config.json provenance:** configs for the three primary monitoring
  domains (`hydraulic`, `physionet2012`, `air_quality_italy`) were written at
  training time by `train_ours.py`. Configs for the remaining domains were
  reconstructed from the checkpoint objects and are marked
  `"reconstructed": true`; the hyperparameters are read directly from the
  fitted imputer and are exact.
