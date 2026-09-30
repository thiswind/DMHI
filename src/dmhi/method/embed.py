"""Stage I: Missingness-aware manifold embedding.

MaskedManifoldEmbedder fits a d-dimensional chart on a (T, D) time series
with observed/missing mask M (1=observed). The chart is built via:

1. PCA pre-projection (variance-threshold, NaN filled by column mean).
2. Time-stratified landmark selection.
3. Time-window mask-aware k-NN graph on landmarks.
4. Laplacian Eigenmaps (LE) initialisation for landmark embeddings.
5. Nesterov gradient descent with cosine-annealing LR (from SUDE learning_s.py).
6. CLLE lifting for non-landmark time steps.

The optimizer and LR schedule are adapted from ZPGuiGroupWhu/sude sude_py/learning_s.py.
"""
import numpy as np
from scipy.sparse import eye as speye
from scipy.sparse.linalg import eigsh
from typing import Dict, Optional, Tuple

from .utils import (
    time_stratified_landmarks,
    build_time_window_graph,
    simplex_weights,
)


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

def _pca_fit(X_filled: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA, keep top-d components.

    Args:
        X_filled: (T, D) NaN-free float32 data.
        d: number of PCA dimensions to keep.

    Returns:
        V: (D, d) loading matrix (columns = eigenvectors)
        mu: (D,) column mean
        X_pca: (T, d) projected data
    """
    mu = X_filled.mean(axis=0)
    Xc = (X_filled - mu).astype(np.float64)
    if Xc.shape[0] < Xc.shape[1]:
        # sample-side SVD (faster when T < D)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt[:d].T        # (D, d)
        X_pca = Xc @ V      # (T, d)
    else:
        cov = Xc.T @ Xc / max(Xc.shape[0] - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:d]
        V = eigenvectors[:, idx]
        X_pca = Xc @ V
    return V.astype(np.float32), mu.astype(np.float32), X_pca.astype(np.float32)


def _mean_fill_nan(X: np.ndarray) -> np.ndarray:
    """Fill NaN entries with column mean (in-place copy)."""
    Xf = X.copy()
    col_means = np.nanmean(Xf, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(Xf))
    Xf[inds] = np.take(col_means, inds[1])
    return Xf


# ---------------------------------------------------------------------------
# Laplacian Eigenmaps initialisation
# ---------------------------------------------------------------------------

def _le_init(W_L: np.ndarray, d: int) -> np.ndarray:
    """Compute LE initialisation for |I_L| x d landmark embeddings.

    Uses the 2nd .. (d+1)-th smallest eigenvectors of the normalised Laplacian.

    Args:
        W_L: (n_L, n_L) symmetric affinity matrix (dense, small).
        d: embedding dimension.

    Returns:
        Z_L: (n_L, d) float32 embedding.
    """
    n_L = W_L.shape[0]
    D_diag = W_L.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        D_inv_sqrt = np.where(D_diag > 1e-12, 1.0 / np.sqrt(np.maximum(D_diag, 1e-12)), 0.0)
    # Normalised Laplacian: L = I - D^{-1/2} W D^{-1/2}
    Lnorm = np.eye(n_L) - (D_inv_sqrt[:, None] * W_L * D_inv_sqrt[None, :])
    # eigsh needs sparse; use eigh for small dense matrix
    eigenvalues, eigenvectors = np.linalg.eigh(Lnorm)
    # skip constant eigenvector (index 0); take indices 1 .. d
    idx = np.argsort(eigenvalues)[1: d + 1]
    Z_L = eigenvectors[:, idx].astype(np.float32)
    # Normalise to unit variance per dimension
    std = Z_L.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Z_L /= std
    return Z_L


# ---------------------------------------------------------------------------
# Optimiser: Nesterov + cosine-annealing LR  (adapted from sude learning_s.py)
# ---------------------------------------------------------------------------

def _cosine_lr(epoch: int, max_epochs: int, alpha_max: float, alpha_min: float, warmup: int) -> float:
    if epoch < warmup:
        return alpha_min + (alpha_max - alpha_min) * epoch / max(warmup, 1)
    t = epoch - warmup
    T_cos = max_epochs - warmup
    return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + np.cos(np.pi * t / max(T_cos, 1)))


# ---------------------------------------------------------------------------
# Loss functions for landmark optimisation
# ---------------------------------------------------------------------------

def _loss_laplacian(Z_L: np.ndarray, L_mat: np.ndarray) -> float:
    """Laplacian quadratic form tr(Z^T L Z)."""
    return float(np.trace(Z_L.T @ L_mat @ Z_L))


def _grad_laplacian(Z_L: np.ndarray, L_mat: np.ndarray) -> np.ndarray:
    return 2.0 * (L_mat @ Z_L)


def _loss_time(Z_L: np.ndarray, id_L: np.ndarray, tau: float, lam: float) -> float:
    if len(id_L) < 2 or lam == 0:
        return 0.0
    steps = np.diff(Z_L, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    hinge = np.maximum(0.0, norms - tau)
    return lam * float(hinge.sum())


def _grad_time(Z_L: np.ndarray, tau: float, lam: float) -> np.ndarray:
    if lam == 0:
        return np.zeros_like(Z_L)
    n_L = Z_L.shape[0]
    g = np.zeros_like(Z_L)
    steps = np.diff(Z_L, axis=0)           # (n_L-1, d)
    norms = np.linalg.norm(steps, axis=1)  # (n_L-1,)
    for i, (s, n) in enumerate(zip(steps, norms)):
        if n > tau + 1e-9:
            direction = s / n
            g[i] -= lam * direction
            g[i + 1] += lam * direction
    return g


def _loss_norm(Z_L: np.ndarray, lam: float) -> float:
    mu = Z_L.mean(axis=0)
    cov = np.cov(Z_L.T) if Z_L.shape[0] > 1 else np.eye(Z_L.shape[1])
    if Z_L.shape[1] == 1:
        cov = np.array([[float(np.var(Z_L))]])
    return lam * (float((mu ** 2).sum()) + float(np.linalg.norm(cov - np.eye(cov.shape[0])) ** 2))


def _grad_norm(Z_L: np.ndarray, lam: float) -> np.ndarray:
    n, d = Z_L.shape
    Z64 = Z_L.astype(np.float64)
    mu = Z64.mean(axis=0)                  # (d,)
    g_mean = 2.0 * lam * mu / n            # broadcast
    Zc = Z64 - mu
    if n > 1:
        # Clip Zc to avoid overflow in matmul
        Zc = np.clip(Zc, -50.0, 50.0)
        cov = Zc.T @ Zc / (n - 1)
        cov = np.clip(cov, -1e6, 1e6)
    else:
        cov = np.eye(d)
    diff_cov = cov - np.eye(d)             # (d, d)
    g_cov = lam * 4.0 * Zc @ diff_cov / (n - 1) if n > 1 else np.zeros_like(Z64)
    return (g_mean + g_cov).astype(Z_L.dtype)


# ---------------------------------------------------------------------------
# Main embedding class
# ---------------------------------------------------------------------------

class MaskedManifoldEmbedder:
    """Fit a d-dim manifold chart on (T, D) time series with mask M.

    After fit(), stores:
        V, mu         — PCA loading and mean
        id_L          — landmark time indices
        Z             — (T, d) full embedding (landmarks + CLLE lifts)
        clle_weights  — {t: (neighbor_ids_in_id_L, weights)} for non-landmarks
    """

    def __init__(
        self,
        d: int = 8,
        k: int = 10,
        W_search: int = 5,
        k_clle: int = 9,
        r_L: float = 0.12,
        lambda_I: float = 0.01,
        lambda_norm: float = 0.1,
        max_epochs: int = 200,
        alpha_max_factor: float = 2.5,
        alpha_min_factor: float = 0.5,
        warmup: int = 10,
        d_pca: Optional[int] = None,
        hold_blend_alpha: float = 0.0,
        graph_alpha: float = 0.5,
    ):
        self.d = d
        # d_pca: PCA components for CLLE inverse quality.
        # If None, defaults to D-1 (see fit()).
        # Setting d_pca > d improves CLLE inverse quality (less info loss).
        self.d_pca = d_pca
        self.hold_blend_alpha = float(hold_blend_alpha)
        self.k = k
        self.W_search = W_search
        self.k_clle = k_clle
        self.r_L = r_L
        self.lambda_I = lambda_I
        self.lambda_norm = lambda_norm
        self.max_epochs = max_epochs
        self.alpha_max_factor = alpha_max_factor
        self.alpha_min_factor = alpha_min_factor
        self.warmup = warmup
        self.graph_alpha = graph_alpha

        # Fitted attributes (set by fit)
        self.V: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.id_L: Optional[np.ndarray] = None
        self.Z: Optional[np.ndarray] = None
        self.clle_weights: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None
        self.X_pca: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, M: np.ndarray) -> "MaskedManifoldEmbedder":
        """Fit the manifold chart.

        Args:
            X: (T, D) float32 time series (may contain NaN for missing).
            M: (T, D) int8 mask, 1=observed 0=missing.

        Returns:
            self (for chaining)
        """
        T, D = X.shape

        # --- Step 2: landmarks (must compute n_L before d to enforce d < n_L) ---
        id_L = time_stratified_landmarks(T, self.r_L)
        n_L = len(id_L)

        # Need n_L ≥ d + 2 for LE to give d non-trivial eigenvectors:
        # eigenvectors 1..d require n_L > d.  Also d must fit PCA and T.
        d = min(self.d, D, T - 1, n_L - 1)
        d = max(d, 1)   # always at least 1 dimension

        # d_pca: number of PCA components kept for CLLE inverse reconstruction.
        # Using more PCA components than d improves inverse quality (less info
        # loss from ambient→PCA). Default: keep as many as safely possible.
        if self.d_pca is not None:
            d_pca = min(self.d_pca, D - 1, T - 1)
        else:
            d_pca = min(D - 1, T - 1)   # keep almost all variance by default
        d_pca = max(d_pca, d)   # never fewer than latent dim

        # --- Step 1: PCA with NaN fill ---
        X_filled = _mean_fill_nan(X.astype(np.float32))
        V, mu, X_pca_full = _pca_fit(X_filled, d_pca)
        # X_pca_low is used for graph + manifold (d dims); X_pca is for CLLE inverse (d_pca dims)
        X_pca_low = X_pca_full[:, :d]
        self.V, self.mu, self.X_pca = V, mu, X_pca_full
        self.id_L = id_L
        self.d = d   # update to actual d used (may be < self.d if n_L was small)

        # --- Step 3: affinity graph on landmarks ---
        # Graph and CLLE weights use X_pca_low (d-dim) for manifold structure.
        X_pca_L = X_pca_low[id_L]
        M_L = M[id_L]
        W_full = build_time_window_graph(
            X_pca_low, M, k=self.k, W_search=self.W_search, alpha=self.graph_alpha
        )
        # extract sub-graph for landmarks
        W_L_sp = W_full[id_L][:, id_L]
        W_L = np.array(W_L_sp.todense(), dtype=np.float32)

        # --- Step 4: LE initialisation ---
        D_deg = W_L.sum(axis=1)
        D_mat = np.diag(D_deg)
        L_mat = (D_mat - W_L).astype(np.float32)  # unnormalised Laplacian
        Z_L = _le_init(W_L, d)

        # --- Step 5: Nesterov gradient descent ---
        N = n_L
        # LR scaled by 1/N for stable convergence regardless of graph size
        alpha_max = self.alpha_max_factor / max(N, 1)
        alpha_min = self.alpha_min_factor / max(N, 1)

        # compute tau = median consecutive step norm at init
        steps_init = np.linalg.norm(np.diff(Z_L, axis=0), axis=1)
        tau = float(np.median(steps_init)) if len(steps_init) > 0 else 1.0
        tau = max(tau, 1e-3)

        Z_L = Z_L.astype(np.float64)
        L_mat_64 = L_mat.astype(np.float64)
        velocity = np.zeros_like(Z_L)
        max_grad_norm = 10.0

        for epoch in range(self.max_epochs):
            lr = _cosine_lr(epoch, self.max_epochs, alpha_max, alpha_min, self.warmup)
            # Nesterov look-ahead
            Z_look = Z_L + 0.9 * velocity
            # Clip Z_look to prevent overflow in gradient computations
            Z_look = np.clip(Z_look, -100.0, 100.0)
            g = (
                _grad_laplacian(Z_look, L_mat_64) / max(n_L, 1)
                + _grad_time(Z_look, tau, self.lambda_I)
                + _grad_norm(Z_look, self.lambda_norm)
            )
            # Gradient clipping
            g_norm = np.linalg.norm(g)
            if g_norm > max_grad_norm:
                g = g * (max_grad_norm / g_norm)
            velocity = 0.9 * velocity - lr * g
            Z_L = Z_L + velocity
            # Safety: clip after update
            Z_L = np.clip(Z_L, -100.0, 100.0)
            if not np.isfinite(Z_L).all():
                Z_L = np.nan_to_num(Z_L, nan=0.0, posinf=1.0, neginf=-1.0)
                velocity = np.zeros_like(Z_L)

        Z_L = Z_L.astype(np.float32)

        # --- Step 6: CLLE lift for non-landmarks ---
        id_L_set = set(id_L.tolist())
        non_landmarks = [t for t in range(T) if t not in id_L_set]

        clle_weights: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for t in non_landmarks:
            # find k_clle nearest landmarks in PCA space (low-dim for graph quality)
            dists_to_L = np.linalg.norm(X_pca_L - X_pca_low[t], axis=1)
            nn_k = min(self.k_clle, n_L)
            nn_idx_local = np.argsort(dists_to_L)[:nn_k]
            nn_ids = id_L[nn_idx_local]             # global time indices
            # CLLE: compute barycentric weights in PCA space (x_t ≈ Σ w_j x_L[nn])
            # then apply to embedded landmarks. Previously this passed Z_L and zeros,
            # which produced weights forcing Σ w_j Z_L[nn] ≈ 0 (collapse to origin).
            w = simplex_weights(X_pca_L[nn_idx_local], X_pca_low[t])
            clle_weights[t] = (nn_ids, w)

        self.clle_weights = clle_weights

        # --- Assemble full Z ---
        Z = np.zeros((T, d), dtype=np.float32)
        Z[id_L] = Z_L
        for t, (nn_ids, w) in clle_weights.items():
            nn_local = np.searchsorted(id_L, nn_ids)
            Z[t] = w @ Z_L[nn_local]

        # --- Final scale stabilisation ---
        # Laplacian GD's trivial minimiser is Z=0; lambda_norm regularisation alone
        # is not always strong enough to keep variance from collapsing across
        # max_epochs iterations. Force unit std per dim to guarantee a usable scale
        # for downstream Stage II metric learning and Stage III Karcher imputation.
        z_std = Z.std(axis=0)
        z_std = np.where(z_std < 1e-6, 1.0, z_std)
        Z = (Z / z_std).astype(np.float32)
        # Also rescale stored landmark codes so transform() stays consistent.
        Z_L_scaled = Z[id_L]
        # Keep clle_weights as-is: they are simplex weights on Z_L, scale-invariant
        # by linearity (w @ (Z_L / s) = (w @ Z_L) / s).

        self.Z = Z
        # Also re-export normalised landmark Z so external callers reading
        # embedder.Z[id_L] get the scaled values.
        return self

    # ------------------------------------------------------------------

    def pca_project(self, X: np.ndarray) -> np.ndarray:
        """Project (T, D) sample to PCA space (T, d_pca) using fitted V, mu.

        NaN entries are filled by column mean before projection.
        """
        assert self.V is not None, "Call fit() first."
        X_filled = _mean_fill_nan(X.astype(np.float32))
        Xc = (X_filled - self.mu).astype(np.float64)
        return (Xc @ self.V.astype(np.float64)).astype(np.float32)

    def transform(
        self,
        X: np.ndarray,
        M: np.ndarray,
        hold_t: np.ndarray | None = None,
    ) -> tuple:
        """Lift a new (T, D) sample onto the fitted chart via CLLE.

        Reuses V, mu from fit(). Non-landmarks are lifted to nearest fitted
        landmarks in X_pca space using simplex weights.

        Args:
            X: (T, D) float32 (may have NaN).
            M: (T, D) int8 mask.

        Returns:
            Z_new: (T, d) float32 embedding.
            X_pca: (T, d_pca) float32 PCA projection of this sample.
        """
        assert self.V is not None, "Call fit() first."
        T = X.shape[0]
        X_pca = self.pca_project(X)          # (T, d_pca) — full PCA for CLLE inverse
        X_pca_low = X_pca[:, :self.d]        # (T, d) — low-dim for neighbor search

        Z_L = self.Z[self.id_L]
        # Reference landmark PCA (low-dim only) for neighbor distance computation
        X_pca_L = self.X_pca[self.id_L, :self.d]  # (n_L, d)

        Z_new = np.zeros((T, self.d), dtype=np.float32)
        for t in range(T):
            # Blend PCA distance with temporal distance.
            # alpha = fraction of missing features at time t; when most features
            # are missing the zero-filled PCA is uninformative, so we lean more
            # on temporal distance.
            obs_frac_t = float(M[t].mean()) if M is not None else 1.0
            alpha = 1.0 - obs_frac_t   # 0 = fully observed → pure PCA; 1 = all missing → pure temporal
            if hold_t is not None and bool(hold_t[t]):
                alpha = float(self.hold_blend_alpha)

            pca_dist = np.linalg.norm(X_pca_L - X_pca_low[t], axis=1)
            temp_dist = np.abs(self.id_L.astype(float) - t)
            pca_dist_n  = pca_dist  / (pca_dist.max()  + 1e-8)
            temp_dist_n = temp_dist / (temp_dist.max() + 1e-8)
            combined = (1.0 - alpha) * pca_dist_n + alpha * temp_dist_n

            nn_k = min(self.k_clle, len(self.id_L))
            nn_idx_local = np.argsort(combined)[:nn_k]
            # CLLE: barycentric weights in PCA space, applied in embedding space.
            # Previously: simplex_weights(Z_L[nn], zeros) → reconstructed Z_new → 0.
            w = simplex_weights(X_pca_L[nn_idx_local], X_pca_low[t])
            Z_new[t] = w @ Z_L[nn_idx_local]

        return Z_new, X_pca
