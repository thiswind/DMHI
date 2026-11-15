"""Tests for IncrementalStitcher (E9): cache reuse gives bit-identical
output to from-scratch tiling, and append() actually reuses windows."""

import numpy as np
import pytest

from dmhi.method.incremental import IncrementalStitcher


class _DummyImputer:
    """Deterministic stand-in: impute() returns input masked-mean fill."""

    def __init__(self):
        self.calls = 0

    def impute(self, X, M):
        self.calls += 1
        X = X.astype(np.float32)
        M = M.astype(np.int8)
        mu = (X * M).sum(axis=1, keepdims=True) / np.clip(
            M.sum(axis=1, keepdims=True), 1, None)
        return np.where(M.astype(bool), X, np.broadcast_to(mu, X.shape).copy())


def test_incremental_equals_from_scratch():
    rng = np.random.default_rng(0)
    N, T, D, W, S = 3, 120, 4, 40, 10
    X = rng.normal(size=(N, T, D)).astype(np.float32)
    M = (rng.random((N, T, D)) > 0.3).astype(np.int8)

    imp = _DummyImputer()
    inc = IncrementalStitcher(imp, W=W, S=S)
    out = inc.append(X[:, :50, :], M[:, :50, :])
    for t0 in range(50, T, S):
        out = inc.append(X[:, t0:t0 + S, :], M[:, t0:t0 + S, :])

    ref = inc.from_scratch()
    np.testing.assert_array_equal(out, ref)
    # Streaming did exactly one imputer call per FINAL window (9 at T=120),
    # vs 2+3+...+9 = 44 if every arrival recomputed all windows from scratch.
    n_windows = len(inc._tail_anchored_starts(T))
    assert inc.stats["imputer_calls"] == n_windows
    assert inc.stats["cache_hits"] > 0
    scratch_calls = sum(len(inc._tail_anchored_starts(t))
                        for t in range(50, T + 1, S))
    assert inc.stats["imputer_calls"] < scratch_calls


def test_append_stride_enforced():
    inc = IncrementalStitcher(_DummyImputer(), W=10, S=5)
    X = np.zeros((1, 12, 2), np.float32)
    M = np.ones((1, 12, 2), np.int8)
    inc.append(X[:, :12, :], M[:, :12, :])
    with pytest.raises(ValueError):
        inc.append(X[:, :3, :], M[:, :3, :])


def test_observed_entries_frozen():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2, 60, 3)).astype(np.float32)
    M = (rng.random((2, 60, 3)) > 0.4).astype(np.int8)
    inc = IncrementalStitcher(_DummyImputer(), W=20, S=5)
    out = inc.append(X, M)
    assert np.array_equal(out[M.astype(bool)], X[M.astype(bool)])
