"""Utility functions for Stage I manifold embedding.

All functions are pure NumPy (no torch) for CPU-efficient graph/PCA ops.
"""
import numpy as np
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Landmark selection
# ---------------------------------------------------------------------------

def time_stratified_landmarks(T: int, r_L: float = 0.12) -> np.ndarray:
    """Return sorted landmark time indices with time-stratified uniform sampling.

    Guarantees coverage: divides [0, T) into ceil(r_L * T) equal-width bins
    and draws one index uniformly per bin. Endpoints 0 and T-1 are always
    included to anchor the chart.

    Args:
        T: number of time steps.
        r_L: fraction of time steps to use as landmarks (default 0.12).

    Returns:
        Sorted 1-D int array of length n_L = max(2, ceil(r_L * T)).
    """
    n_L = max(2, int(np.ceil(r_L * T)))
    bins = np.array_split(np.arange(T), n_L)
    landmarks = np.array([rng[len(rng) // 2] for rng in bins])
    landmarks[0] = 0
    landmarks[-1] = T - 1
    return np.unique(landmarks).astype(int)


# ---------------------------------------------------------------------------
# Gaussian kernel weights for Karcher neighbourhood
# ---------------------------------------------------------------------------

def gaussian_weights(neighbor_indices: np.ndarray, t: int, sigma: float = 3.0) -> np.ndarray:
    """Normalized Gaussian weights centered at time t.

    Args:
        neighbor_indices: 1-D array of time indices (observed).
        t: query time index.
        sigma: bandwidth in time steps.

    Returns:
        1-D float32 array summing to 1, length == len(neighbor_indices).
    """
    d = np.abs(neighbor_indices.astype(float) - t)
    w = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
    total = w.sum()
    if total < 1e-12:
        w = np.ones_like(w) / len(w)
    else:
        w /= total
    return w


# ---------------------------------------------------------------------------
# Mask-overlap distance
# ---------------------------------------------------------------------------

def mask_overlap_distance(
    X_pca: np.ndarray,
    M: np.ndarray,
    i: int,
    j: int,
    alpha: float = 0.5,
    tau_min: int = 2,
) -> float:
    """Missingness-aware distance between time steps i and j in PCA space.

    Uses only dimensions jointly observed in both i and j. Falls back to
    Euclidean distance on mean-filled data when overlap is too small.

    Args:
        X_pca: (T, D_pca) PCA-projected data (NaN-free, mean-filled).
        M: (T, D) original binary mask (1=observed).
        i, j: time indices.
        alpha: penalty weight for missingness overlap (0=pure Euclidean).
        tau_min: minimum overlap count before fallback.

    Returns:
        Non-negative scalar distance.
    """
    m_i = M[i].astype(bool)
    m_j = M[j].astype(bool)
    overlap = np.logical_and(m_i, m_j).sum()

    if overlap < tau_min:
        # fallback: Euclidean on PCA (mean-fill already applied)
        return float(np.linalg.norm(X_pca[i] - X_pca[j]))

    # Project to PCA dims; since X_pca already encodes all D dims, use as is
    diff = X_pca[i] - X_pca[j]
    base_dist = float(np.linalg.norm(diff))
    # penalise by fraction of non-overlapping dimensions
    miss_penalty = 1.0 - overlap / max(m_i.sum(), m_j.sum(), 1)
    return base_dist * (1.0 + alpha * miss_penalty)


# ---------------------------------------------------------------------------
# Time-window k-NN affinity graph
# ---------------------------------------------------------------------------

def build_time_window_graph(
    X_pca: np.ndarray,
    M: np.ndarray,
    k: int = 10,
    W_search: int = 5,
    alpha: float = 0.5,
    tau_min: int = 2,
) -> csr_matrix:
    """Build sparse affinity matrix on all T time steps.

    Only considers neighbours within a ±W_search time window. Always adds
    chain edges (t, t+1) with weight 1 to ensure graph connectivity.

    Args:
        X_pca: (T, d_pca) PCA coords (NaN-free).
        M: (T, D) binary mask.
        k: number of nearest neighbours within window.
        W_search: half-width of search window in time steps.
        alpha: missingness penalty weight.
        tau_min: minimum overlap for mask-aware distance.

    Returns:
        Symmetric sparse (T, T) affinity matrix (float32).
    """
    T = X_pca.shape[0]
    rows, cols, data = [], [], []

    for t in range(T):
        lo = max(0, t - W_search)
        hi = min(T, t + W_search + 1)
        candidates = [s for s in range(lo, hi) if s != t]
        if not candidates:
            continue
        dists = np.array([
            mask_overlap_distance(X_pca, M, t, s, alpha, tau_min)
            for s in candidates
        ], dtype=np.float32)
        nn = min(k, len(candidates))
        idx = np.argsort(dists)[:nn]
        for i in idx:
            s = candidates[i]
            w = float(np.exp(-dists[i]))
            rows += [t, s]
            cols += [s, t]
            data += [w, w]

    # chain edges
    for t in range(T - 1):
        rows += [t, t + 1]
        cols += [t + 1, t]
        data += [1.0, 1.0]

    W = csr_matrix((data, (rows, cols)), shape=(T, T), dtype=np.float32)
    W.data = np.minimum(W.data, 1.0)  # cap at 1 for symmetric entries
    return W


# ---------------------------------------------------------------------------
# Simplex weights (CLLE) — adapted from sude_py/clle.py lines 17-21
# ---------------------------------------------------------------------------

def simplex_weights(X_samp: np.ndarray, x_i: np.ndarray) -> np.ndarray:
    """Compute barycentric (simplex) weights for x_i w.r.t. rows of X_samp.

    Solves  min ||x_i - W X_samp||^2  s.t. sum(W)=1  via closed-form
    Lagrangian: W = (S^{-1} 1) / (1^T S^{-1} 1)  where S = (X_samp-x_i)(X_samp-x_i)^T.

    Adapted from ZPGuiGroupWhu/sude sude_py/clle.py (lines 17-21).

    Args:
        X_samp: (n, d) neighbour embeddings.
        x_i: (d,) query point.

    Returns:
        (n,) float32 weights summing to 1.
    """
    n = X_samp.shape[0]
    Z = X_samp - x_i                         # (n, d)
    S = Z @ Z.T                              # (n, n)
    # Tikhonov regularization to avoid singular S
    S += (0.1 ** 2 / max(n, 1)) * np.trace(S) * np.eye(n)
    ones = np.ones(n, dtype=np.float64)
    try:
        w = np.linalg.solve(S.astype(np.float64), ones)
    except np.linalg.LinAlgError:
        w = ones
    denom = w.sum()
    if abs(denom) < 1e-12:
        w = np.ones(n) / n
    else:
        w /= denom
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# G-Mahalanobis simplex weightstage D)
# ---------------------------------------------------------------------------

def simplex_weights_metric(
    X_samp: np.ndarray, x_i: np.ndarray, G: np.ndarray
) -> np.ndarray:
    """G-Mahalanobis variant of :func:`simplex_weights`.

    Solves the same constrained least-squares problem as ``simplex_weights``
    but in the local Riemannian inner product ``<u, v>_G = u^T G v``:

        min_w ||x_i - W X_samp||_G^2   s.t.  sum(w) = 1

    Closed-form is identical to the Euclidean case except the Gram matrix is
    replaced by  S = (X_samp - x_i) G (X_samp - x_i)^T .  When G = I this
    reduces exactly to :func:`simplex_weights`.

    This is the Stage-III mechanism through which the learned metric G(z)
    influences the inverse mapping ``hat_X``.  Without it (the Euclidean
    formula), Stage II Karcher updates affect ``hat_X`` only via the
    coordinates of ``Z_prime[t]``; the **direction-dependent** anisotropy
    learned by ``G_net`` is lost in Stage III.

    Args:
        X_samp: (n, d) neighbour latent embeddings.
        x_i:    (d,)   query latent point.
        G:      (d, d) SPD metric tensor at ``x_i``.

    Returns:
        (n,) float32 weights summing to 1.
    """
    n = X_samp.shape[0]
    Z = (X_samp - x_i).astype(np.float64)           # (n, d)
    G64 = G.astype(np.float64)
    S = Z @ G64 @ Z.T                               # (n, n) Mahalanobis Gram
    S += (0.1 ** 2 / max(n, 1)) * np.trace(S) * np.eye(n)
    ones = np.ones(n, dtype=np.float64)
    try:
        w = np.linalg.solve(S, ones)
    except np.linalg.LinAlgError:
        w = ones
    denom = w.sum()
    if abs(denom) < 1e-12:
        w = np.ones(n) / n
    else:
        w /= denom
    return w.astype(np.float32)
