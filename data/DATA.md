# Datasets

The paper reports Tables II/III on 12 multivariate time series domains.
**Ten of the twelve ship directly in this repository** under `data/processed/`
as ready-to-use `(N, T, D)` arrays — for these, reproduction needs no
downloads at all. The two exceptions are DUA-restricted (see the last
section).

On top of the paper's twelve, this repository also ships arrays and
checkpoints from the broader evaluation campaign (ablation and scale-study
domains such as `hydraulic`, `breizh_pheno`, `nanodrone_dec`, `mc_maze`,
`pulsedb`, `swiss_trajectory`, `cmapss`). Those are development artifacts and
are **not** part of the paper's reported table; they are kept so the
protocols behind the ablation and efficiency studies can be inspected.

## Processed format (all domains)

```
data/processed/<domain>/
  X_{train,val,test}.npy   float32 (N, T, D)  z-scored with train statistics
  M_{train,val,test}.npy   int8   (N, T, D)   1 = observed, 0 = missing
```

Splits are subject-disjoint where a subject id exists (no leakage).
Evaluation missingness (block, length 20) is applied at run time by
`scripts/evaluate.py` / `scripts/baselines/common.py` from the
cell's `(domain, missing_rate, block_size, seed)` — every method sees
byte-identical masks.

## Domain index

`paper` marks the twelve domains reported in Tables II/III (Table I of the
paper lists the same twelve; the Hydraulic row in the submitted Table I is
commented out). The remaining rows belong to the ablation and scale studies.

| domain | N/T/D (train) | paper | in repo | source | rebuild script |
|---|---|---|---|---|---|
| `physionet2012` | 4000/48/35 | ✅ | ❌ see below | PhysioNet Challenge 2012 | see below |
| `mimic` | 37/48/59 | ✅ | ❌ see below | MIMIC-III | see below |
| `air_quality_italy` | 527/24/13 | ✅ | ✅ | Italian air quality (De Vito et al.) | `scripts/preprocess_air_italy.py` |
| `gait_uci` | 180/101/6 | ✅ | ✅ | UCI gait dynamics | `scripts/process_newds.py` |
| `gas_home` | 64/100/10 | ✅ | ✅ | UCI HT Sensor gas | `scripts/process_newds.py` |
| `mocap_karate` | 34/100/39 | ✅ | ✅ | MoCap-Impute | `scripts/process_newds.py` |
| `delta_robot` | 340/100/6 | ✅ | ✅ | Zenodo 13641620 | `scripts/process_newds.py` |
| `kuka` | 485/100/12 | ✅ | ✅ | KUKA KR300 R2500 identification benchmark | `scripts/process_newds.py` |
| `nanodrone` | 582/100/14 | ✅ | ✅ | NanoBench nano-quadrotor | `scripts/process_newds.py` |
| `mhealth` | 480/100/23 | ✅ | ✅ | UCI MHEALTH | `scripts/process_newds.py` |
| `breizh_rs` | 390/45/13 | ✅ | ✅ | BreizhCrops Sentinel-2 (frh01, L1C) | `scripts/process_newds.py` |
| `solo12` | 111/100/12 | ✅ | ✅ | Solo12 quadruped walk recordings | `scripts/process_newds.py` |
| `hydraulic` | 390/100/17 | — | ✅ | UCI / ZeMA hydraulic test rig | `scripts/process_newds.py` |
| `breizh_pheno` | 390/45/13 | — | ✅ | BreizhCrops (phenology variant) | `scripts/process_newds.py` |
| `nanodrone_dec` | 279/100/14 | — | ✅ | NanoBench, decimated ~25 Hz | `scripts/process_newds.py` |
| `mc_maze` | 48/70/50 | — | ✅ | Neural Latents Benchmark MC_Maze_Small (DANDI:000140) | `scripts/process_newds.py` |
| `pulsedb` | 456/100/3 | — | ✅ | VitalDB cuff-less BP waveforms | `scripts/fetch_pulsedb.py` + `scripts/process_newds.py` |
| `swiss_trajectory` | 1600/48/12 | — | ✅ | synthetic (swiss-roll + OU dynamics, pre-committed seed 20260520) | `scripts/build_swiss_trajectory.py` |
| `cmapss` | 254/50/14 | — | ✅ | NASA C-MAPSS | arrays included; data are complete (all-ones masks), no rebuild script needed for reported cells |
| `electricity` | 1022/96/321 | — | ❌ see below | long-stream scale study only | see below |

Array integrity is asserted by `tests/test_data.py` (shapes, masks, shipped
domains).

## Rebuilding a domain from raw sources

`scripts/process_newds.py` regenerates the in-repo domains from
`data/raw_newds/<name>/` (raw downloads are not redistributed; each proc
function documents its upstream). Typical flow:

```bash
# pulsedb: fetch raw windows from the VitalDB open API, then process
python scripts/fetch_pulsedb.py
python scripts/process_newds.py --dataset pulsedb

# synthetic domain: regenerate bit-for-bit from the pre-committed seed
python scripts/build_swiss_trajectory.py
```

The in-repo arrays and the rebuild scripts produce the same splits; when in
doubt, trust the shipped arrays (they are the ones behind the paper tables).

## Domains not in the repository, and how to get them

The three below are **not redistributed here on purpose**. For each we state
the reason and the exact reader-side route:

| domain | why not in git | how to obtain | then what |
|---|---|---|---|
| `physionet2012` | PhysioNet Health Data License / DUA **forbids redistribution** of derived record data | create a free credentialed account at [physionet.org](https://physionet.org/content/challenge-2012/), accept the DUA, download Set A/B | process to the (N,T,D) format with train-statistics z-scoring (protocol as in the paper); or request our processed copy for verification under the same DUA terms |
| `mimic` | MIMIC-III access requires **credentialed** training and its DUA **forbids redistribution** | apply at [mimic.mit.edu](https://mimic.mit.edu/ivwww/mimic-iii/) (e.gCredentialing), download MIMIC-III Clinical Database | same processing protocol as above; we cannot hand out copies, DUA forbids it |
| `electricity` | pure size: 215 MB would dominate the repository | shipped as a **release archive** alongside the trained checkpoints (see [REPRODUCE.md](../REPRODUCE.md)); used only by the long-stream scale study | unpack under `data/processed/electricity/` |

We deliberately do not ship even "derived" versions of the DUA datasets:
publishing model outputs that leak record-level values would violate the
agreements. If you need verification on these domains and hold the
credentials yourself, open an issue and we will help you run the pipeline on
your side.

## Sample data

`src/dmhi/utils/sample/` contains a tiny synthetic set used by
`tests/test_smoke.py` (regenerable via `src/dmhi/utils/make_sample.py`) — it
is not a paper domain.
