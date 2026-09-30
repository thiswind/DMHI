"""Canonical evaluation protocol shared by DMHI and every baseline runner.

Every reported number in the paper carves evaluation masks from the same
implementation of block missingness with the same seeds, so that each method
imputes byte-identical holes:

* protocol: rho = 0.9 held-out blocks of length L = 20 (nine domains) or the
  L grid {10, 20, 30, 40} (three primary domains);
* mask seed convention: mask_seed = run_seed + 2;
* seeds {7, 42, 123}; metrics MAE / MSE / RMSE / MD on held-out entries only.

`apply_block_missing` is line-identical to the former
``experiments/evaluate.py::_apply_block_missing`` and
``experiments/baselines/common.py::apply_block_missing``, which were unified
here when the repository was restructured to the published layout.
"""
from __future__ import annotations

import numpy as np


def apply_block_missing(M_orig: np.ndarray, block_len: int, rate: float,
                        seed: int) -> np.ndarray:
    """Apply synthetic block missingness on top of M_orig.

    For each sample and channel, contiguous blocks of ``block_len``
    timesteps are held out from observed positions, targeting ``rate``
    of the observed fraction. Returns the eval mask (1 = held-out).
    """
    rng = np.random.default_rng(seed)
    N, T, D = M_orig.shape
    eval_mask = np.zeros_like(M_orig)
    blocks_per_sample = max(1, int(round(T * rate / block_len)))
    for n in range(N):
        for d in range(D):
            obs_times = np.where(M_orig[n, :, d] == 1)[0]
            if len(obs_times) < block_len:
                continue
            placed = 0
            attempts = 0
            while placed < blocks_per_sample and attempts < 50:
                start = rng.integers(0, T - block_len + 1)
                block_idx = np.arange(start, start + block_len)
                if np.all(M_orig[n, block_idx, d] == 1):
                    eval_mask[n, block_idx, d] = 1
                    placed += 1
                attempts += 1
    return eval_mask
