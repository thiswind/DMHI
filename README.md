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
python smoke_test.py
```

Fits a deliberately tiny `RiemannianImputer` on the synthetic sample in
`data/sample/`, carves a block hold-out, imputes it, and checks finiteness
and observed-entry preservation. Runs in seconds on CPU.

## Repository layout

```
src/dmhi/method/        The deployed pipeline (importable package)
  pipeline.py             RiemannianImputer — end-to-end fit / impute
  embed.py                Stage I: mask-aware manifold embedding
  mhb.py                  Stage II: manifold harmonic basis + soft graph-spline
  clle.py                 Stage III: symmetric CLLE inverse projection
  metric.py, utils.py     graph construction and helpers
experiments/
  eval_ours.py            evaluate a saved checkpoint on a test split
  e4*.py                  paper analyses (MD k-sensitivity, downstream, N=4 determinism)
  baselines/              classical / SAITS / BRITS / TimeMixer / ImputeFormer /
                          CSDI / PSW-I runners (see baselines/README.md)
configs/                 one JSON per reported paper cell (see configs/README.md)
data/
  DATA.md                 dataset provenance, file semantics, fetch instructions
  sample/                 tiny synthetic smoke set (make_sample.py regenerates it)
tests/                   config sanity tests
smoke_test.py            clone-and-run end-to-end check
REPRODUCE.md             reproduction guide (checkpoints, masks, determinism)
```

Heavy artifacts (processed datasets, trained checkpoints, run logs) are
**not** part of the repository — see `data/DATA.md` and `REPRODUCE.md`.

## Evaluating a reported cell

After placing the processed splits under `data/processed/` and checkpoints
under `checkpoints/ours/` (fetch instructions in `REPRODUCE.md` /
`data/DATA.md`):

```bash
python experiments/eval_ours.py \
    --run_dir checkpoints/ours/hydraulic_seed7 \
    --block_size 20 --seed 7 --deterministic
```

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
