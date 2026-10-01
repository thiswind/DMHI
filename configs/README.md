# Configurations

One JSON file per reported paper cell, named `<domain>_seed<N>.json`. These
are the exact hyperparameters behind every "Ours" number in the paper; each
file mirrors the `config.json` written next to the corresponding checkpoint
(`checkpoints/dmhi/<domain>_seed<N>/config.json` — checkpoints themselves are
not redistributed, see `../REPRODUCE.md`).

## Schema

| key | meaning |
|---|---|
| `dataset` | domain identifier (see `data/DATA.md`) |
| `missing_rate` | evaluation missing rate (e.g. `0.7`) |
| `pattern` | missing pattern (`block`) |
| `block_size` | contiguous block length of the evaluation mask |
| `seed` | global seed for split / mask / initialization |
| `d` | latent chart dimension (Stage I output) |
| `k` | graph k-NN size |
| `k_clle` | number of landmarks for the CLLE embedding |
| `epochs` / `lr` / `batch_size` / `patience` | Stage II fitting |
| `lambda_F` / `lambda_smooth` | loss weights |
| `train_min` | train-split normalization floor (optional, domain-specific) |
| `run_id` | canonical run identifier (`<domain>_<pattern>bs<block>_<rate>_seed<N>`) |
| `deterministic` | deterministic inference flag |
| `reconstructed` | `true` when the config was recovered from the fitted checkpoint instead of written at training time (optional; hyperparameters are read from the fitted imputer and are exact — see `../REPRODUCE.md`) |

## Index

20 domains, 55 files. The ten domains that ship inside the repository cover
the twelve domains reported in the paper's Tables II and III; the remaining
domains come from the broader evaluation campaign (ablations and scale
studies) and are listed in `data/DATA.md`:

- `hydraulic`: 4 files
- 16 domains with 3 files each: `air_quality_italy`, `breizh_pheno`,
  `breizh_rs`, `delta_robot`, `electricity`, `gait_uci`, `gas_home`, `kuka`,
  `mhealth`, `mimic`, `mocap_karate`, `nanodrone`, `nanodrone_dec`,
  `physionet2012`, `solo12`, `swiss_trajectory`
- `pulsedb`, `mc_maze`, `cmapss`: 1 file each

## Validation

`tests/test_configs.py` asserts that every file parses, carries the required
keys, and that the seed in the filename matches the `seed` field.
