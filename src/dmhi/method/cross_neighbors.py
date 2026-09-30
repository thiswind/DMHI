"""Cross-subject latent neighbor bank for Karcher imputation."""
from __future__ import annotations

import numpy as np


def build_train_latent_bank(
    Z_train: np.ndarray,
    M_train: np.ndarray,
    bank_max_points: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """Collect observed (n,t) latent vectors into a flat bank (K, d).

    Args:
        Z_train: (N, T, d) training embeddings.
        M_train: (N, T, D) mask, 1=observed.
        bank_max_points: subsample cap when K is larger.
        seed: RNG seed for subsampling.

    Returns:
        Z_flat: (K, d) float32.
    """
    obs_ts = M_train.any(axis=2)
    rows = []
    for n in range(Z_train.shape[0]):
        idx_t = np.where(obs_ts[n])[0]
        if len(idx_t) == 0:
            continue
        rows.append(Z_train[n, idx_t])
    if not rows:
        raise ValueError("build_train_latent_bank: no observed timesteps in training set")
    Z_flat = np.concatenate(rows, axis=0).astype(np.float32)
    if len(Z_flat) > bank_max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(Z_flat), size=bank_max_points, replace=False)
        Z_flat = Z_flat[sel]
    return Z_flat


def _mahalanobis_sq_batch(z: np.ndarray, Z_flat: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distances from z to each row of Z_flat under G."""
    diff = (Z_flat - z[None, :]).astype(np.float64)
    G64 = G.astype(np.float64)
    try:
        L = np.linalg.cholesky(G64 + 1e-8 * np.eye(G64.shape[0]))
        y = np.linalg.solve(L, diff.T)
        return np.sum(y * y, axis=0)
    except np.linalg.LinAlgError:
        return np.sum(diff * diff, axis=1)


def query_cross_neighbors(
    z_t: np.ndarray,
    Z_flat: np.ndarray,
    G_at_t: np.ndarray,
    k_cross: int,
    cross_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cross-subject neighbors by Mahalanobis distance under G_at_t.

    Returns:
        Z_nbrs: (k_eff, d) neighbor latent vectors.
        w: (k_eff,) normalized Gaussian weights exp(-d^2 / (2*sigma^2)).
    """
    if k_cross <= 0 or len(Z_flat) == 0:
        return np.zeros((0, z_t.shape[0]), dtype=np.float32), np.zeros(0, dtype=np.float64)

    d2 = _mahalanobis_sq_batch(z_t.astype(np.float64), Z_flat, G_at_t)
    k_eff = min(k_cross, len(Z_flat))
    idx = np.argpartition(d2, k_eff - 1)[:k_eff]
    Z_nbrs = Z_flat[idx].astype(np.float32)
    d_sel = d2[idx]
    sigma2 = max(float(cross_sigma) ** 2, 1e-8)
    w = np.exp(-0.5 * d_sel / sigma2)
    w_sum = float(w.sum())
    if w_sum < 1e-12:
        w = np.ones(k_eff, dtype=np.float64) / k_eff
    else:
        w = w / w_sum
    return Z_nbrs, w
