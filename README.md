# DMHI — Deterministic Manifold Harmonic Imputation

Reference implementation of the method in *"DMHI: Deterministic Manifold
Harmonic Imputation for Edge Deployable Multivariate Time Series"*
(under review at the IEEE Internet of Things Journal, 2026).

DMHI imputes block-missing multivariate time series with a three-stage
manifold pipeline — **Embed → intrinsic Compute → inverse Project**:

- **Stage I** — `MaskedManifoldEmbedder` (`src/dmhi/method/embed.py`): PCA
  (>=95% variance) → mask-weighted, time-constrained k-NN graph → landmark
  CLLE into a `d`-dim latent chart.
- **Stage II** — `mhb.mhb_spline_extend` (`src/dmhi/method/mhb.py`):
  closed-form **soft graph-spline** extension on the manifold harmonic basis,
  with spectral filter `h(λ) = w / (w + γλ)`, `γ = 0.02`.
- **Stage III** — symmetric CLLE inverse (`src/dmhi/method/clle.py`) back to
  the ambient series.

The deployed inference path carries **no neural-network weights**: the
learned state is the Stage I PCA basis and landmark coordinates — which is
what makes the method edge-deployable. End-to-end entry point:
`dmhi.method.pipeline.RiemannianImputer`.

## Installation

```bash
pip install -e .          # or: pip install -r requirements.txt
```

Python >= 3.10; CPU is sufficient for inference and for the smoke test.

## Quick start (no data required)

```bash
python tests/test_smoke.py
```

Fits a deliberately tiny `RiemannianImputer` on the synthetic sample in
`src/dmhi/utils/sample/`, carves a block hold-out, imputes it, and checks finiteness
and observed-entry preservation. Runs in seconds on CPU.

## The 30-second speed demo (Table IV)

DMHI's headline claim is CPU-only, millisecond inference — ~840× faster than
the diffusion baseline CSDI on the same CPU. See it yourself after
unpacking the checkpoint release under `checkpoints/dmhi/`:

```bash
python scripts/benchmark_speed.py
```

It times DMHI per-sample inference on every released checkpoint (50 runs,
all on plain CPU) against linear interpolation and mean fill on identical
data and masks, then prints the paper's reference numbers side by side.
Verified on an Apple M4: DMHI lands at 1.8–8.2 ms/sample across all 50
runs — around the paper's reported 5.8 ms — while CSDI needs 4900.6 ms on
the same CPU, i.e. **~840×** the paper's 5.8 ms DMHI latency. One flag,
`--domain gas_home`, benchmarks a single domain.

## Repository layout

```
src/dmhi/method/        The deployed pipeline (importable package)
  pipeline.py             RiemannianImputer — end-to-end fit / impute
  embed.py                Stage I: mask-aware manifold embedding
  mhb.py                  Stage II: manifold harmonic basis + soft graph-spline
  clle.py                 Stage III: symmetric CLLE inverse projection
  metric.py, utils.py     graph construction and helpers
  + 8 supporting modules  (karcher, losses, stitch, window_filter, …)
scripts/
  train.py               train the deployed configuration on a paper domain
  evaluate.py            evaluate a saved checkpoint on a test split
  benchmark_speed.py     the 30-second CPU speed demo (Table IV)
  process_newds.py       rebuild a shipped domain from data/raw_newds/
  fetch_pulsedb.py, build_swiss_trajectory.py, preprocess_air_italy.py
                         per-domain data acquisition / preprocessing
  downstream_md_auroc.py, downstream_{gas,gait}_classify.py,
  md_k_sensitivity.py, determinism_check.py
                         paper analyses (see scripts/ANALYSIS_RUNNERS.md)
  baselines/              classical / SAITS / BRITS / TimeMixer / ImputeFormer /
                          CSDI / PSW-I runners (see baselines/README.md)
configs/                 one JSON per reported paper cell (see configs/README.md)
data/
  processed/              10 of the paper's twelve reported domains, ready to use
  DATA.md                 provenance + restricted-domain acquisition routes
tests/                   smoke + config + shipped-data sanity tests
results/                 expected metrics for reproduction spot checks
REPRODUCE.md             reproduction guide (checkpoints, masks, determinism)
```

## Imputing a paper domain — no training required

The paper reports Tables II/III on 12 domains. Ten of them ship inside the
repository (`data/processed/`); the two exceptions (DUA-restricted
physionet2012 and mimic) have documented acquisition routes in
`data/DATA.md`. The repository additionally ships the remaining arrays and
checkpoints from the broader evaluation campaign — see `data/DATA.md` for
the full inventory and `results/README.md` for the domain-to-metric ledger.

The paper's headline property carries over to the code: **DMHI's deployed
form uses zero trained parameters and requires no training phase**. Impute
directly with the released checkpoint — no `fit()`, no GPU:

```bash
python scripts/evaluate.py \
    --run_dir checkpoints/dmhi/gas_home_seed7 \
    --block_size 20 --rate 0.9 --seed 7
```

The command above returns `MAE=0.051499 / MSE=0.020727 /
RMSE=0.143970 / MD=0.198470`, matching the paper's gas_home cell at the three
decimals used in Table II/III (0.051 / 0.020 / 0.143 / 0.197; mask seed =
seed+2, the convention shared with baselines). Tables II/III report the mean
over seeds, listed per domain in `results/README.md`.

Deterministic inference means the same checkpoint + mask reproduces metrics
to six decimals **within one environment**. Across platforms the linear
algebra differs by BLAS backend (Apple Accelerate vs Linux/OpenBLAS), which
shifts the third decimal by about `1e-3`; no such shift changes any
bold/underline verdict in the paper. Pretrained checkpoints for all reported
cells (~207 MB zip, 50 runs) are distributed as a GitHub release archive;
per-cell hyperparameters live in `configs/`. Full walkthrough:
`REPRODUCE.md` (three levels).

Researchers who want to regenerate a checkpoint from scratch (Stage I
embedding + Stage II metric refinement) can use `scripts/train.py` — see
`REPRODUCE.md` Level 1 for the training-vs-inference story.

## Citation

If this code or the method is useful to you, please cite the paper:

```bibtex
@article{hu2026dmhi,
  title   = {DMHI: Deterministic Manifold Harmonic Imputation for Edge Deployable Multivariate Time Series},
  author  = {Hu, Kuang and Wang, Ruihang and Yue, Kun and Liu, Xilong and Wang, Jiahui},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {Under review}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
