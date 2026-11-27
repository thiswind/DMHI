# Configurations

One JSON file per run directory, named `<domain>_seed<N>.json`. Each file
mirrors the `config.json` written next to the corresponding checkpoint
under `checkpoints/dmhi/<domain>_seed<N>/`. The 36 checkpoints behind the
12-domain tables of the paper (seeds 7/42/123) ship in the release zip and
their per-run `config.json` is inside the zip; the files in this directory
cover the release checkpoints and the broader evaluation campaign (ablations
and scale studies, see `data/DATA.md`). A run's `config.json` is the
authoritative record of its build parameters.

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
| `epochs` / `lr` / `batch_size` / `patience` | deprecated, ignored: configured the retired Stage II training loop; kept so the per-cell schema stays stable (no learned parameters are fitted) |
| `lambda_F` / `lambda_smooth` | deprecated, ignored: formerly Stage II loss weights |
| `build_min` (`train_min` in older files) | wall-clock minutes of the deterministic checkpoint construction (no training) |
| `run_id` | canonical run identifier (`<domain>_<pattern>bs<block>_<rate>_seed<N>`) |
| `deterministic` | deterministic inference flag |
| `reconstructed` | `true` when the config was recovered from the built checkpoint instead of written at build time (optional; hyperparameters are read from the built imputer and are exact; see `../REPRODUCE.md`) |

## Index

20 domains, 55 files. The ten domains that ship inside the repository cover
the twelve domains reported in the paper's Tables II and III; the remaining
domains come from the broader evaluation campaign (ablations and scale
studies) and are listed in `data/DATA.md`:

- `hydraulic`: 4 files (ablation-only domain; not in the 12-domain tables
  and not in the release checkpoint zip)
- 16 domains with 3 files each: `air_quality_italy`, `breizh_pheno`,
  `breizh_rs`, `delta_robot`, `electricity`, `gait_uci`, `gas_home`, `kuka`,
  `mhealth`, `mimic`, `mocap_karate`, `nanodrone`, `nanodrone_dec`,
  `physionet2012`, `solo12`, `swiss_trajectory`
- `pulsedb`, `mc_maze`, `cmapss`: 1 file each

## Validation

`tests/test_configs.py` asserts that every file parses, carries the required
keys, and that the seed in the filename matches the `seed` field.
