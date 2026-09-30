# Expected results

Reference metrics for reproducing the paper's reported numbers with this
repository. All numbers are deterministic given the same checkpoint and the
same mask (mask seed = run seed + 2, see `src/dmhi/utils/eval_protocol.py`).

## Table IV — inference speed (Apple M4, CPU)

| method | ms / sample | device | source |
|---|---|---|---/|
| DMHI (deployed form) | 5.8 | CPU (Apple M4) | paper, measured |
| CSDI | 4900.6 | CPU (Apple M4) | paper, measured |
| CSDI | 823.9 | GPU | training-time ledger |
| BRITS | 2.8 | GPU | training-time ledger |

`python scripts/benchmark_speed.py` re-measures the DMHI column locally:
verified 1.7–8.0 ms/sample across all 44 released runs on Apple M4.

## Level-2 spot checks (released checkpoints, seed 7)

| domain | MAE | MSE | RMSE | MD |
|---|---|---|---|---|
| gas_home | 0.055369 | 0.026157 | 0.161731 | 0.187710 |
| hydraulic | (run `scripts/evaluate.py` — deterministic, compare against this ledger) |
| physionet2012 | (DUA-gated data; see data/DATA.md) |
