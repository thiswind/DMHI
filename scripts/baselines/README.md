# Baselines

Every baseline in the paper is launched by one runner under this directory.
All runners consume the same processed splits (`data/processed/<domain>/`)
and evaluate on byte-identical masks: `common.py` derives the evaluation mask
purely from the cell's `(domain, missing_rate, block_size, seed)`, so every
method sees identical holes.

| runner | backend | provenance |
|---|---|---|
| `run_classical.py` | built-in (NumPy only) | Mean fill, linear interpolation, LOCF — implemented in-repo |
| `run_saits.py` | PyPOTS (>=1.0) | calls `pypots.imputation.SAITS` / `pypots.imputation.BRITS`; data interface via `saits_data_adapter.py` |
| `run_timemixer.py` | PyPOTS | library wrapper (v3.2); data interface via `timemixer_data_adapter.py` |
| `run_imputeformer.py` | PyPOTS | calls the PyPOTS ImputeFormer implementation |
| `run_csdi.py` | authors' repo, local checkout | wraps `csdi_official/` (official CSDI code) via the `OurPhysioDataset` adapter; the runner resolves the checkout as a sibling of `scripts/` — not included in this repository, fetch upstream into `csdi_official/` |
| `run_psw_i.py` | authors' repo, extracted class | PSW-I (ICLR 2025); `OTImputationIni` extracted from `psw_i_official/benchmark.py` and adapted to our data interface — not included, fetch upstream into `psw_i_official/` |
| `run_downstream.py` | built-in | downstream evaluation for baseline imputations |
| `aggregate_baseline_results.py` | built-in | result aggregation across runs |

Install the PyPOTS-backed runners with the optional extra:
`pip install -e ".[baselines]"`.

## Running a classical baseline cell

`--missing_pattern` is a **key name**, not a raw length: use `mcar`,
`block5`, `block10`, `block20`, `block25`, `block30`, or `block40`
(see `BLOCK_LENS` in `common.py`). The `--seed` value is used directly as
the mask RNG seed; the paper's "seed s" convention corresponds to
`--seed $((s+2))`:

```bash
python scripts/baselines/run_classical.py --method mean \
    --dataset gas_home --missing_rate 0.9 --missing_pattern block20 --seed 9
# [mean] gas_home rate=0.9 pat=block20 MAE=0.5938 RMSE=0.8472
```
