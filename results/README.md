# Expected results

Reference metrics for reproducing the paper's reported numbers with this
repository. Every number is deterministic given the same checkpoint and the
same mask (mask seed = run seed + 2, see `src/dmhi/utils/eval_protocol.py`).

## Table II — DMHI row (revision R1 numbers), reproduced from the released checkpoints

Protocol exactly as in the paper: block missingness of length `L=20` at
`rho=0.9`, mask seed = seed + 2, metric = the mean over the three reported
seeds {7, 42, 123}. Each row is produced by

```bash
python scripts/evaluate.py --run_dir checkpoints/dmhi/<domain>_seed<seed> --rate 0.9 --block_size 20
```

over the three seeds, then averaged. These are the DMHI-row numbers reported
in the revised manuscript; they come from the checkpoints in
`dmhi-checkpoints-v0.2.0.zip` (Stage-II gate consistency fix relative to the
original submission).

| domain | MAE | MSE | RMSE | MD | seeds |
|---|---|---|---|---|---|
| gas_home | 0.051 | 0.020 | 0.143 | 0.197 | 7, 42, 123 |
| mocap_karate | 0.075 | 0.026 | 0.162 | 0.866 | 7, 42, 123 |
| delta_robot | 0.198 | 0.284 | 0.533 | 0.267 | 7, 42, 123 |
| gait_uci | 0.153 | 0.054 | 0.232 | 0.393 | 7, 42, 123 |
| breizh_rs | 0.310 | 0.308 | 0.555 | 0.630 | 7, 42, 123 |
| solo12 | 0.174 | 0.064 | 0.253 | 0.605 | 7, 42, 123 |
| kuka | 0.159 | 0.083 | 0.288 | 0.732 | 7, 42, 123 |
| nanodrone | 0.192 | 0.144 | 0.379 | 1.092 | 7, 42, 123 |
| mhealth | 0.197 | 0.365 | 0.604 | 1.075 | 7, 42, 123 |
| air_quality_italy | 0.141 | 0.054 | 0.232 | 0.509 | 7, 42, 123 |
| mimic | 0.095 | 0.101 | 0.318 | 1.202 | 7, 42, 123 |
| physionet2012 | 0.342 | 0.410 | 0.640 | 0.927 | 7, 42, 123 |

Notes:

- `mimic` and `physionet2012` are DUA-gated domains: their arrays are **not**
  in this repository. Re-process them from the credentialed sources with the
  `newds_v2` protocol if you need the raw data; the released checkpoints
  evaluate them once the processed arrays are in place under
  `data/processed/`.
- All 12 domains were verified to reproduce this table on x86-64
  (OpenBLAS/numpy 2.2.6) and arm64 (Accelerate/numpy 2.2.6), and a
  representative subset on numpy 1.26.4 without torch installed, agreeing to
  the four decimals shown (cross-platform residuals <= 3e-4).

### Why the third decimal can differ across machines

Inference is deterministic and the released weights are reproducible
bit-for-bit; what is *not* bit-identical across platforms is the
floating-point arithmetic underneath the linear algebra (CPU-architecture
SIMD reduction order; see REPRODUCE.md). Verified on this repository:

- the deployed path is device-independent (MPS vs CPU give the same result);
- re-evaluating under a pinned environment matching the paper's stack
  (`numpy 2.2.6 / scipy 1.15.3 / scikit-learn 1.7.2`) reproduces the
  repository value exactly;
- the residuals across x86-64 / arm64 / numpy-1.26 are of order 1e-4 and do
  not change any table cell at three decimals.

## Table IV — inference speed

| method | ms / sample | device | source |
|---|---|---|---|
| DMHI (deployed form) | 5.8 | CPU (Apple M4) | paper, measured |
| CSDI | 4900.6 | CPU (Apple M4) | paper, measured |
| CSDI | 823.9 | GPU | training-time ledger |
| BRITS | 2.8 | GPU | training-time ledger |

`python scripts/benchmark_speed.py` re-measures the DMHI column locally:
verified 1.8-8.2 ms/sample across all released runs on Apple M4, i.e.
around the paper's reported 5.8 ms CPU latency and ~840x faster than CSDI
on the same CPU.
