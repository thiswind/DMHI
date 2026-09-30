# Datasets

## What is in this release

`data/processed/` contains the full processed splits for every domain used in the
paper.  Each domain directory holds the six arrays the harness expects:

```
X_train.npy   float32  (N_train, T, D)
X_val.npy     float32  (N_val,   T, D)
X_test.npy    float32  (N_test,  T, D)
M_train.npy   int8     (N_train, T, D)   1 = observed
M_val.npy     int8     (N_val,   T, D)
M_test.npy    int8     (N_test,  T, D)
```

Domains with a downstream label also have `y_train.npy`, `y_val.npy`, `y_test.npy`.

The stored test mask `M_test` is near-fully observed.  The benchmark protocol
carves synthetic block-missing gaps at eval time via
`eval_ours.py::_apply_block_missing`.

`data/sample/` is a small synthetic smoke set (`(40, 24, 5)`) used only by
`smoke_test.py`.  It is not related to any reported result.

## Domain table

| Domain | `X_test` shape | Role |
|---|---|---|
| `physionet2012` | (4000, 48, 35) | ICU monitoring — primary curved stream |
| `mimic` | (11, 48, 59) | ICU monitoring — secondary (n_test=11) |
| `air_quality_italy` | (151, 24, 13) | Monitoring — boundary / low-curvature |
| `hydraulic` | (120, 100, 17) | Monitoring — boundary / flat manifold |
| `electricity` | (293, 96, 321) | Long-sequence scaling (Exp-8 only) |
| `gait_uci` | (60, 101, 6) | Curved array, has downstream label y |
| `gas_home` | (20, 100, 10) | Curved array, has downstream label y |
| `mocap_karate` | (11, 100, 39) | Curved array |
| `delta_robot` | (105, 100, 6) | Curved array |
| `kuka` | (89, 100, 12) | Curved array |
| `nanodrone` | (162, 100, 14) | Curved array |
| `nanodrone_dec` | (78, 100, 14) | Nanodrone decimated (25 Hz) |
| `mhealth` | (160, 100, 23) | Near-linear boundary array |
| `breizh_pheno` | (120, 45, 13) | Near-linear boundary array |
| `breizh_rs` | (120, 45, 13) | Near-linear boundary array |
| `solo12` | (135, 100, 12) | Near-linear boundary array |

## Checkpoints

Trained "Ours" checkpoints live in `../checkpoints/ours/<domain>_seed<N>/`.
Each checkpoint directory contains:

- `checkpoint.pkl` — the fitted `RiemannianImputer` object (not tracked in git;
  see note below)
- `config.json` — dataset, seed, deployed missing rate, and model
  hyperparameters consumed by `eval_ours.py`

The `.pkl` files are excluded from git to avoid committing large binary blobs.
See `REPRODUCE.md` for how to obtain or re-generate them.

## What is NOT committed to git

`.pkl` (checkpoint) and `.npy` (data array) files are large binary files and are
excluded by `.gitignore`.  This repository tracks only text and JSON.  Readers
can reconstruct the full release directory by:

1. Running `python smoke_test.py` to verify the pipeline works on the synthetic
   sample without any downloads.
2. Fetching the processed splits and checkpoints as described in `REPRODUCE.md`
   (Level 2), then placing them under `data/processed/<domain>/` and
   `checkpoints/ours/<domain>_seed<N>/`.

## Source provenance

Every domain is built from a public source set.  The naming cross-reference in
`../paper/body/appendix.tex` (section "Dataset Naming Cross-Reference") maps
each paper name to its original repository URL.  Processing recipes are in
`temp_script/process_newds.py` on the training server.
