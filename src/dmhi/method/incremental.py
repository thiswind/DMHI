"""Incremental (streaming) imputation on sliding windows (E9, R2-6/R2-9).

Goal: prove that when new data segments arrive (window slides by stride S),
only the NEW windows need imputation — the graph/Laplacian/eigendecomp of
untouched windows are reused via a per-window result cache. This gives
O(changed windows) per step instead of O(all windows), while producing
bit-identical output to the from-scratch tail-anchored tiling.

API:
    inc = IncrementalStitcher(imputer, W=W, S=S)
    out_k = inc.append(X_new, M_new)   # streaming; X_new has S new steps
    # out_k: (N, T_k, D) full imputed series after the k-th arrival

Fallback: no learned state is required — the imputer is training-free
(closed-form), so each new window is imputed from scratch in ~ms while old
window results are cached.
"""

from __future__ import annotations

import numpy as np

from .stitch import _triangular_weights


class IncrementalStitcher:
    """Sliding-window stitcher with per-window result caching.

    Parameters mirror ``stitch_impute_long`` (W window length, S stride).
    Windows are anchored from the series TAIL: starts = multiples of S
    measured backwards from (T - W), so each ``append()`` of S new steps
    shifts the whole tiling by exactly S — a window start s (multiple of S
    from the tail) remains a valid start after arrival, its coverage
    [s, s+W) is unchanged in absolute terms, hence its cached imputation is
    reusable verbatim. Only the newly created start (the previous tail
    position) requires an imputer call.

    ``from_scratch()`` runs the same tail-anchored tiling WITHOUT cache for
    the A/B comparison; per-step equality with the incremental path is
    asserted in tests.
    """

    def __init__(self, imputer, W: int, S: int):
        if not (0 < S <= W):
            raise ValueError("require 0 < S <= W")
        self.imputer = imputer
        self.W = int(W)
        self.S = int(S)
        self._cache: dict[int, np.ndarray] = {}   # abs start -> (N, W, D)
        self._weights = _triangular_weights(W)
        self._X: np.ndarray | None = None
        self._M: np.ndarray | None = None
        self._T = 0
        self._n_imputer_calls = 0
        self._n_cache_hits = 0

    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict:
        return {"imputer_calls": self._n_imputer_calls,
                "cache_hits": self._n_cache_hits,
                "windows_total": self._n_imputer_calls + self._n_cache_hits}

    # ------------------------------------------------------------------
    def _tail_anchored_starts(self, T: int) -> list[int]:
        """Starts of full windows, spaced by S, anchored at the tail:
        {last - k*S : k >= 0, last - k*S >= 0} with last = T - W."""
        if T < self.W:
            return []
        last = T - self.W
        return sorted(last - k * self.S for k in range(last // self.S + 1))

    def append(self, X_new: np.ndarray, M_new: np.ndarray) -> np.ndarray:
        """Ingest the next S steps (first call may be longer) and return the
        full imputed series. Cached windows are reused verbatim; only new
        starts invoke the imputer."""
        X_new = np.asarray(X_new, dtype=np.float32)
        M_new = np.asarray(M_new, dtype=np.int8)
        if X_new.ndim != 3:
            raise ValueError("X_new must be (N, T_new, D)")
        if self._X is None:
            self._X, self._M = X_new.copy(), M_new.copy()
        else:
            if X_new.shape[1] != self.S:
                raise ValueError(
                    f"append expects exactly S={self.S} new steps, "
                    f"got {X_new.shape[1]}")
            self._X = np.concatenate([self._X, X_new], axis=1)
            self._M = np.concatenate([self._M, M_new], axis=1)
        self._T = self._X.shape[1]

        starts = self._tail_anchored_starts(self._T)
        for s in starts:
            if s in self._cache:
                self._n_cache_hits += 1
                continue
            xw = self._X[:, s:s + self.W, :]
            mw = self._M[:, s:s + self.W, :]
            self._cache[s] = self.imputer.impute(xw, mw).astype(np.float64)
            self._n_imputer_calls += 1
        # drop cache entries that fell out of the tiling
        live = set(starts)
        for s in [s for s in self._cache if s not in live]:
            del self._cache[s]
        return self._stitch(starts)

    def _stitch(self, starts) -> np.ndarray:
        N, T, D = self._X.shape
        acc = np.zeros((N, T, D), dtype=np.float64)
        wsum = np.zeros((N, T, 1), dtype=np.float64)
        wgt = self._weights
        for s in starts:
            acc[:, s:s + self.W, :] += self._cache[s] * wgt[None, :, None]
            wsum[:, s:s + self.W, :] += wgt[None, :, None]
        out = (acc / np.clip(wsum, 1e-12, None)).astype(np.float32)
        mb = self._M.astype(bool)
        out[mb] = self._X[mb].astype(np.float32)
        return out

    def from_scratch(self) -> np.ndarray:
        """Same tail-anchored tiling, computed WITHOUT cache (A/B baseline).
        Does not disturb the incremental cache or counters."""
        starts = self._tail_anchored_starts(self._T)
        acc = np.zeros(self._X.shape, dtype=np.float64)
        wsum = np.zeros((*self._X.shape[:2], 1), dtype=np.float64)
        wgt = self._weights
        for s in starts:
            hw = self.imputer.impute(
                self._X[:, s:s + self.W, :], self._M[:, s:s + self.W, :]
            ).astype(np.float64)
            acc[:, s:s + self.W, :] += hw * wgt[None, :, None]
            wsum[:, s:s + self.W, :] += wgt[None, :, None]
        out = (acc / np.clip(wsum, 1e-12, None)).astype(np.float32)
        mb = self._M.astype(bool)
        out[mb] = self._X[mb].astype(np.float32)
        return out
