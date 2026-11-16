# DMHI — Deterministic Manifold Harmonic Imputation

Reference implementation of *"DMHI: Deterministic Manifold Harmonic
Imputation for Edge Deployable Multivariate Time Series"* (IEEE Internet of
Things Journal, 2026, under review).

DMHI imputes block-missing multivariate time series with a three-stage
manifold pipeline (embed → intrinsic compute → inverse project). There is
no training phase; `build()` only stores/computes closed-form structures
and inference uses no neural-network weights, so deployment needs only
NumPy/SciPy/scikit-learn (torch is an optional dependency used solely by
the ablation-only learned-metric paths). Entry point:
`dmhi.method.pipeline.RiemannianImputer`.

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

## Evaluate a checkpoint

Download the checkpoint archive from the GitHub release and unpack it under
`checkpoints/dmhi/` (see `REPRODUCE.md`). Then:

    python scripts/evaluate.py \
        --run_dir checkpoints/dmhi/gas_home_seed7 \
        --block_size 20 --rate 0.9 --seed 7

Expected output: `MAE=0.051499 / MSE=0.020727 / RMSE=0.143970 / MD=0.198470`
(mask seed = seed + 2). Per-domain, per-seed expected values:
`results/README.md`. Per-cell hyperparameters: `configs/`.

## Benchmark

    python scripts/benchmark_speed.py

Times CPU inference for every released checkpoint. Use `--domain <name>` for
a single domain.

## Layout

    src/dmhi/method/   the pipeline (importable package)
    scripts/           train / evaluate / benchmark / data / analysis scripts
    configs/           one JSON per reported paper cell
    data/processed/    10 of the 12 paper domains (see data/DATA.md)
    tests/             smoke, config, and data sanity tests
    results/           expected metrics for spot checks

## Data

10 of the 12 paper domains ship under `data/processed/`. physionet2012 and
mimic are access-restricted; acquisition routes are documented in
`data/DATA.md`.

## Reproduce

Full walkthrough (checkpoints, masks, determinism, retraining):
`REPRODUCE.md`.

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
