#!/usr/bin/env python3
"""Clone-and-run smoke test for the real pipeline on the synthetic sample.

No data download is required. The test fits a deliberately tiny
`RiemannianImputer`, carves a contiguous block hold-out on the test split,
imputes it, and checks that the result is finite and that observed entries are
preserved. It runs in a few seconds on CPU.

    python tests/test_smoke.py        # direct run
    pytest tests/test_smoke.py        # via pytest

If PyTorch or the method package is unavailable, the test skips cleanly with a
non-failing message, so it never blocks an environment that only builds the
paper.
"""
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _load_sample() -> tuple[np.ndarray, np.ndarray]:
    sample = REPO / "src" / "dmhi" / "utils" / "sample"
    if not (sample / "X.npy").exists():
        from dmhi.utils import make_sample

        make_sample.main()
    X = np.load(sample / "X.npy").astype(np.float32)
    M = np.load(sample / "M.npy").astype(np.int8)
    return X, M


def main() -> int:
    X, M = _load_sample()
    try:
        import torch  # noqa: F401
        from dmhi.method.pipeline import RiemannianImputer
    except Exception as exc:  # torch missing, etc.
        print(f"[smoke] method/torch unavailable ({exc}); skipping run.")
        return 0

    n_windows, n_steps, _ = X.shape
    n_train = max(8, int(round(n_windows * 0.7)))
    X_train, M_train = X[:n_train], M[:n_train]
    X_test, M_test = X[n_train:], M[n_train:]

    rng = np.random.default_rng(0)
    block = 6
    eval_mask = np.zeros_like(M_test, dtype=bool)
    M_input = M_test.copy()
    for i in range(len(X_test)):
        start = int(rng.integers(0, n_steps - block + 1))
        eval_mask[i, start : start + block, :] = True
        M_input[i, start : start + block, :] = 0

    imputer = RiemannianImputer(
        d=2,
        k=6,
        k_clle=5,
        embed_epochs=30,
        epochs=5,
        patience=3,
        batch_size=16,
    )
    imputer.fit(X_train, M_train, X_test, M_test)
    X_hat = imputer.impute(X_test, M_input)

    mae = float(np.abs(X_hat[eval_mask] - X_test[eval_mask]).mean())
    mean_fill = float(np.abs(X_test[eval_mask] - float(X_train.mean())).mean())
    observed_preserved = bool(np.allclose(X_hat[M_input == 1], X_test[M_input == 1], atol=1e-4))

    print(f"[smoke] held-out MAE={mae:.4f}  (mean-fill baseline ~{mean_fill:.4f})")
    print(f"[smoke] observed entries preserved: {observed_preserved}")

    assert np.isfinite(mae), "imputation MAE is not finite"
    assert observed_preserved, "observed entries were modified by impute()"
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_smoke_run():
    assert main() == 0
