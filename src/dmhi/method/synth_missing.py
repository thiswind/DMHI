"""Synthetic block missingness for Stage II v2 training (REVISE-004)."""
from __future__ import annotations

import numpy as np


def sample_block_mask(
    B: int,
    T: int,
    seed: int,
    block_size: int | None = None,
    rate: float | None = None,
) -> np.ndarray:
    """Sample a block-missing mask over time.

    Args:
        B: batch size (number of trajectories).
        T: sequence length.
        seed: RNG seed for this batch.
        block_size: block length; None → uniform choice from {20, 30}.
        rate: target fraction of timesteps missing; None → Uniform(0.5, 0.9).

    Returns:
        miss: (B, T) bool, True = synthetic missing timestep.
    """
    rng = np.random.default_rng(seed)
    if block_size is None:
        block_size = int(rng.choice([20, 30]))
    if rate is None:
        rate = float(rng.uniform(0.5, 0.9))
    block_size = max(1, min(block_size, T))
    bps = max(1, int(round(T * rate / block_size)))

    miss = np.zeros((B, T), dtype=bool)
    for b in range(B):
        placed = 0
        attempts = 0
        while placed < bps and attempts < 50:
            if T <= block_size:
                miss[b, :] = True
                break
            start = int(rng.integers(0, T - block_size + 1))
            miss[b, start : start + block_size] = True
            placed += 1
            attempts += 1
    return miss
