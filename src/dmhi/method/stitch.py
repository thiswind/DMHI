"""Overlapping-window stitching for long-sequence imputation.

The deployed closed-form solve diagonalizes a ``T x T`` graph Laplacian
(:func:`dmhi.method.mhb.mhb_basis`, dense ``np.linalg.eigh``), which is
``O(T^3)`` in the sequence length and becomes the Stage-II bottleneck for long
IoT streams. Stitching keeps the per-window cost fixed by imputing a long
series in overlapping windows of a fixed length ``W`` (the training window)
with stride ``S < W``, then deterministically blending the overlaps. Total
cost is ``O((T / S) * W^3)``, i.e. linear in ``T`` at fixed ``W``, while
observed entries are always frozen back to their input values.

The blend is a fixed triangular (Bartlett) weight over each window's local
index, so the result is byte-for-byte reproducible: there is no randomness and
no data-dependent window count beyond the deterministic tiling. Observed
entries are always frozen back to their input values.

This module is the long-sequence (Section VI-G) estimator: the released
``data_derived/m4_long_real_stream.json`` ``dmhi_stitch`` row and the
``e3_longseq`` records are produced by this routine. The batch driver that
sweeps stream lengths and writes those records is an upstream experiment runner
retained outside this public tree (see ``paper/docs/REPRODUCE.md``, Layer B),
so this file has no in-tree caller; it is imported by that external driver.
"""
from __future__ import annotations

from typing import List

import numpy as np


def _triangular_weights(W: int) -> np.ndarray:
    """Bartlett window, strictly positive so every covered step has weight."""
    if W <= 1:
        return np.ones(max(1, W), dtype=np.float64)
    w = 1.0 - np.abs((np.arange(W) - (W - 1) / 2.0) / ((W - 1) / 2.0))
    return np.clip(w, 1e-3, None)


def window_starts(T: int, W: int, S: int) -> List[int]:
    """Deterministic tiling of ``[0, T)`` by length-``W`` windows with stride
    ``S``; the last window is right-aligned so the tail is always covered."""
    if T <= W:
        return [0]
    if S < 1:
        raise ValueError("stride S must be >= 1")
    starts = list(range(0, T - W + 1, S))
    if starts[-1] != T - W:
        starts.append(T - W)
    return starts


def stitch_impute(imputer, X_long: np.ndarray, M_long: np.ndarray,
                  W: int = 96, S: int = 48) -> np.ndarray:
    """Impute a long ``(N, T, D)`` batch via overlapping windows of length ``W``.

    Args:
        imputer: a fitted ``RiemannianImputer`` (trained at window length ``W``),
            exposing ``impute(X, M) -> hat_X``.
        X_long:  ``(N, T, D)`` float32, typically ``T >> W``.
        M_long:  ``(N, T, D)`` int8/bool mask, 1 = observed.
        W, S:    window length and stride (``S < W`` gives overlap).

    Returns:
        ``(N, T, D)`` float32 imputation; observed entries equal ``X_long``.
    """
    X_long = np.asarray(X_long)
    M_long = np.asarray(M_long)
    N, T, D = X_long.shape
    if T <= W:
        return imputer.impute(X_long, M_long)
    if not (0 < S <= W):
        raise ValueError("require 0 < S <= W for a valid overlapping tiling")
    starts = window_starts(T, W, S)
    wgt = _triangular_weights(W)                       # (W,)
    acc = np.zeros((N, T, D), dtype=np.float64)
    wsum = np.zeros((N, T, 1), dtype=np.float64)
    for s in starts:
        xw = X_long[:, s:s + W, :]
        mw = M_long[:, s:s + W, :]
        hw = imputer.impute(xw, mw).astype(np.float64)  # (N, W, D)
        acc[:, s:s + W, :] += hw * wgt[None, :, None]
        wsum[:, s:s + W, :] += wgt[None, :, None]
    out = (acc / np.clip(wsum, 1e-12, None)).astype(np.float32)
    mb = M_long.astype(bool)
    out[mb] = X_long[mb].astype(np.float32)             # freeze observed
    return out
