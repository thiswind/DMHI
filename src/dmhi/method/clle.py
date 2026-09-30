"""Stage I lifting and Stage III inverse via Constrained Locally Linear Embedding.

Both functions reuse the same simplex_weights formula (adapted from SUDE clle.py).
Stage III maps updated latent codes Z' back to ambient data space via the inverse
PCA + observed-entry freeze.
"""
import numpy as np
from typing import Dict, Optional, Tuple

from .utils import simplex_weights, simplex_weights_metric


def lift_non_landmark(X_samp: np.ndarray, x_i: np.ndarray) -> np.ndarray:
    """Stage I: compute CLLE simplex weights for x_i w.r.t. landmark samples.

    Args:
        X_samp: (n, d) landmark embeddings (nearest k_clle landmarks).
        x_i: (d,) query embedding coordinate (unused in weight computation —
             weights are in terms of the embedding space gram matrix).

    Returns:
        w: (n,) float32 weights summing to 1.
    """
    return simplex_weights(X_samp, x_i)


def clle_inverse(
    Z_prime: np.ndarray,
    id_L: np.ndarray,
    clle_weights: Dict[int, Tuple[np.ndarray, np.ndarray]],
    X_pca: np.ndarray,
    V: np.ndarray,
    mu: np.ndarray,
    X_obs: np.ndarray,
    M: np.ndarray,
    G_field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Stage III: map updated latent codes Z' back to ambient space hat_X.

    For each non-landmark t:
        Re-solve simplex weights of Z'[t] w.r.t. nearest landmarks in Z'.
        tilde_x'[t] = w @ X_pca[landmarks]

    For each landmark j:
        tilde_x'[j] = X_pca[j]   (keep original PCA coordinates)

    Then:
        hat_X = tilde_x' @ V^T + mu          (inverse PCA, all T)
        hat_X = M * X_obs + (1-M) * hat_X    (freeze observed entries)

    Args:
        Z_prime: (T, d) updated latent codes after Karcher imputation.
        id_L: (n_L,) landmark time indices.
        clle_weights: {t: (global_nn_ids, weights)} from embedder.fit().
            nn_ids are global time indices into id_L.
        X_pca: (T, d_pca) original PCA-projected data from fit().
        V: (D, d_pca) PCA loading matrix.
        mu: (D,) PCA mean.
        X_obs: (T, D) original observed data (may have NaN for missing).
        M: (T, D) binary mask, 1=observed 0=missing.
        G_field: optional (T, d, d) field of SPD metric tensors evaluated at
            ``Z_prime``. When providedtage D fix for
            break ③), non-landmark simplex weights are solved in the
            G-Mahalanobis inner product so that the learned anisotropy
            propagates into ``hat_X``. When ``None`` (e.g. ablation_no_metric
            or G_net is None), falls back to the original Euclidean
            ``simplex_weights``.

    Returns:
        hat_X: (T, D) float32 imputed data with observed entries frozen.
    """
    T, d = Z_prime.shape
    d_pca = X_pca.shape[1]
    id_L_set = set(id_L.tolist())
    Z_L_prime = Z_prime[id_L]          # (n_L, d) updated landmark embeddings
    X_pca_L = X_pca[id_L]              # (n_L, d_pca) original landmark PCA coords

    tilde_x = np.zeros((T, d_pca), dtype=np.float32)

    # Landmark t: keep original PCA coordinates
    tilde_x[id_L] = X_pca_L

    # Non-landmark t: re-solve weights in Z' space
    for t, (nn_ids_global, _) in clle_weights.items():
        nn_local = np.array([
            np.searchsorted(id_L, gid) for gid in nn_ids_global
        ], dtype=int)
        nn_local = np.clip(nn_local, 0, len(id_L) - 1)
        Z_nn = Z_L_prime[nn_local]              # (k, d) updated landmark latents
        #tage D (fix break ③):
        # When a per-timestep metric G_field[t] is available, solve the
        # simplex-weight problem in the G-Mahalanobis inner product so the
        # learned anisotropy actually steers reconstruction. Without G_field
        # (e.g. ablation_no_metric), fall back to the Euclidean formula.
        if G_field is not None:
            w = simplex_weights_metric(Z_nn, Z_prime[t], G_field[t])
        else:
            w = simplex_weights(Z_nn, Z_prime[t])
        tilde_x[t] = w @ X_pca_L[nn_local]     # reconstruct in original PCA space

    # Inverse PCA: (T, d_pca) @ (d_pca, D) + (D,)
    hat_X = (tilde_x.astype(np.float64) @ V.T.astype(np.float64) + mu.astype(np.float64)).astype(np.float32)

    # Freeze observed entries
    M_float = M.astype(np.float32)
    X_obs_safe = np.where(np.isnan(X_obs), 0.0, X_obs).astype(np.float32)
    hat_X = M_float * X_obs_safe + (1.0 - M_float) * hat_X

    return hat_X
