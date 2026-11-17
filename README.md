# DMHI — Deterministic Manifold Harmonic Imputation

Reference implementation of *"DMHI: Deterministic Manifold Harmonic
Imputation for Edge Deployable Multivariate Time Series"* (IEEE Internet of
Things Journal, 2026, under review).

DMHI imputes block-missing multivariate time series with a three-stage
manifold pipeline: embed → intrinsic compute → inverse project. There is
no training phase and no learned weights. `build()` constructs the Stage I
manifold chart by deterministic spectral decomposition plus a
fixed-schedule landmark refinement (fixed iteration count, no validation
early-stopping); Stage II solves the graph-Tikhonov system in closed form;
inference runs on NumPy/SciPy/scikit-learn alone (torch is an optional
dependency used only by the ablation-only learned-metric paths). Entry
point: `dmhi.method.pipeline.RiemannianImputer`.

## Requirements

Python >= 3.10. CPU is sufficient for inference and the smoke test.
Runtime dependencies for the default path: numpy, scipy, scikit-learn
(torch only for the optional learned-metric ablations).

## Install

    pip install -e .

or `pip install -r requirements.txt`.

## Smoke test

    python tests/test_smoke.py

Builds a small `RiemannianImputer` on the bundled synthetic sample and checks
finiteness and observed-entry preservation. Runs in seconds.

## Reproduce the paper tables

From a fresh clone:

    # from the GitHub release page, download two archives, then:
    unzip dmhi-checkpoints-v0.2.0.zip -d checkpoints/dmhi
    unzip dmhi-dua-arrays-v0.2.0.zip -d data/processed
    python scripts/reproduce_table.py

This evaluates all 12 domains x 3 seeds at the paper protocol
(L=20, rho=0.9, mask seed = run seed + 2), averages the seeds, and prints
the complete DMHI row of Table II/III. Each run's value is printed, so
every cell is traceable.

The same commands run on all three platforms we tested; only the unpack
step differs on Windows (no `unzip` there; `tar` ships with Windows 10+):

| Platform | Unpack with | Result |
|---|---|---|
| x86-64 Linux | `unzip` | all 48 cells bit-for-bit |
| x86-64 native Windows (PowerShell) | `tar -xf <zip> -C <dir>` | all 48 cells bit-for-bit (tested: Win 11, Python 3.14, numpy 2.4.3) |
| Apple Silicon macOS (arm64) | `unzip` | up to 15 of 48 printed cells differ by <=0.003 (3rd decimal); no ranking or conclusion changes |

Dependencies are unpinned on purpose: a fresh `pip install -r
requirements.txt` works on all three platforms; the validated version set
for this release is recorded in the release notes. Details on what is
platform-sensitive: `REPRODUCE.md`.

Single run instead of the whole table:

    python scripts/evaluate.py \
        --run_dir checkpoints/dmhi/gas_home_seed7 \
        --block_size 20 --rate 0.9 --seed 7

Expected on x86-64: `MAE=0.051789 / MSE=0.020789 / RMSE=0.144185 /
MD=0.198525`. Note `--rate` defaults to the checkpoint's build protocol
(e.g. 0.7 for some domains), not the paper's 0.9. Per-domain, per-seed
expected values: `results/README.md`. Per-cell hyperparameters:
`configs/`.

## Benchmark

    python scripts/benchmark_speed.py

Times CPU inference for every released checkpoint. Use `--domain <name>` for
a single domain.

## Layout

    src/dmhi/method/   the pipeline (importable package)
    scripts/           build_checkpoint / evaluate / reproduce_table / benchmark
    configs/           one JSON per reported paper cell
    data/processed/    10 of 12 paper domains in git; the other 2 ship in the
                       DUA release zip (see above)
    tests/             smoke, config, and data sanity tests
    results/           expected metrics for spot checks

## Data

10 of the 12 paper domains ship under `data/processed/`. The remaining
two (`physionet2012`, `mimic`) are DUA-gated: their processed arrays ship
in `dmhi-dua-arrays-v0.2.0.zip` on the release page for reviewer
convenience (withdrawable on request); the credentialed acquisition route
is in `data/DATA.md`.

## Citation

    @article{hu2026dmhi,
      title   = {DMHI: Deterministic Manifold Harmonic Imputation for Edge Deployable Multivariate Time Series},
      author  = {Hu, Kuang and Wang, Ruihang and Yue, Kun and Liu, Xilong and Wang, Jiahui},
      journal = {IEEE Internet of Things Journal},
      year    = {2026},
      note    = {Under review}
    }

## License

MIT. See `LICENSE`.
