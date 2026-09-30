"""Stage II — Manifold Harmonic Basis (MHB) imputation.: faithful implementation of Algorithm 1
(`00_paper/cn_materials/sections/sec4_method.tex`, eqs. (24)(25)(27)(28)(29)).

Pipeline contract (mirrors `karcher_impute`):

    Z_prime = mhb_impute(Z_n, G_net, I_miss, A, K=5, gamma=1e-3)

where
    Z_n    : (T, d) float32 latent trajectory from Stage I
    G_net  : fitted SPDMetricNet (eval mode) — supplies r_t = lambda_min(G(z_t))
    I_miss : 1-D int array of missing time indices (M^* in the paper)
    A      : (T, T) symmetric non-negative affinity over the FULL trajectory
             (the Stage I mask-weighted time-window k-NN graph, eqs. (5)-(7);
             built by `utils.build_time_window_graph` — includes missing rows
             and chain edges, unlike the ablation harness which graphs only
             the observed subsequence)

Implementation notes (registered in the 036A memo):
  * Eigenpairs are computed with dense `numpy.linalg.eigh` instead of Lanczos.
    For the paper's regime (T <= 60) this is exact, faster than ARPACK and —
    crucially for the fixed-seed inference claim (paper §4.3.6) — fully
    deterministic (no random start vector).
  * B_K takes the K smallest eigenpairs INCLUDING the lambda_1 = 0 constant
    component, exactly as Algorithm 1 line 2 states ("前 K 个最小特征对").
    The gamma*Lambda Tikhonov term then leaves the constant component
    unpenalised (lambda_1 = 0), which is the intended low-frequency bias.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def mhb_basis(
    A: np.ndarray,
    K: int,
    *,
    solver: str = "dense",
    auto_threshold: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Eqs. (24)-(25): symmetric normalised Laplacian -> first K eigenpairs.

    Args:
        A: (T, T) symmetric non-negative affinity (dense or scipy sparse).
        K: basis bandwidth (paper default 5).
        solver: ``"dense"`` (default) uses ``numpy.linalg.eigh`` over the full
            spectrum, exact and fully deterministic, and is the deployed path
            for the paper's short-window regime. ``"truncated"`` computes only
            the K smallest eigenpairs with a shift-invert Lanczos solver
            (``scipy.sparse.linalg.eigsh``), turning the ``O(T^3)`` dense solve
            into ``O(T^2 K)`` / ``O(nnz K)`` for long sequences while returning
            the identical low-frequency subspace; a fixed start vector keeps it
            deterministic, and it falls back to dense on any failure.
            ``"auto"`` uses truncated only when ``T > auto_threshold`` and
            ``K < T - 1`` (where it is actually faster), else dense.
        auto_threshold: length above which ``solver="auto"`` switches to the
            truncated path.

    Returns:
        lam: (K_eff,) ascending eigenvalues  (Lambda = diag(lam), eq. 25)
        B:   (T, K_eff) column-orthonormal eigenvectors (B_K, eq. 25)
        with K_eff = min(K, T).
    """
    if hasattr(A, "todense"):
        A = np.asarray(A.todense())
    A = np.asarray(A, dtype=np.float64)
    T = A.shape[0]
    # Degree matrix D = diag(A 1); chain edges in the production graph
    # guarantee deg > 0, but guard anyway (isolated row -> identity row in L).
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    # Eq. (24): L = I - D^{-1/2} A D^{-1/2}
    L_sym = np.eye(T) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    L_sym = 0.5 * (L_sym + L_sym.T)  # numerical symmetrisation
    K_eff = max(1, min(int(K), T))

    use_truncated = solver == "truncated" or (
        solver == "auto" and T > int(auto_threshold) and K_eff < T - 1
    )
    if use_truncated:
        trunc = _mhb_basis_truncated(L_sym, K_eff)
        if trunc is not None:
            return trunc
        # else: fall through to the dense path (deterministic, always works)

    eigvals, eigvecs = np.linalg.eigh(L_sym)
    lam = np.maximum(eigvals[:K_eff], 0.0)  # clip -1e-16 noise: Lambda >= 0
    B = eigvecs[:, :K_eff]
    return lam, B


def _mhb_basis_truncated(L_sym: np.ndarray, K_eff: int):
    """Smallest ``K_eff`` eigenpairs of a PSD normalised Laplacian via
    shift-invert Lanczos. Returns ``(lam, B)`` ascending, or ``None`` if the
    sparse solver is unavailable or fails (caller falls back to dense).

    The normalised Laplacian has a zero eigenvalue (constant component), so a
    small negative shift ``sigma`` avoids the singular factorisation that an
    exact ``sigma=0`` would hit; a fixed all-ones start vector makes the Lanczos
    iteration deterministic, preserving the fixed-seed replay guarantee.
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
    except Exception:
        return None
    T = L_sym.shape[0]
    if K_eff >= T - 1:
        return None
    try:
        v0 = np.ones(T, dtype=np.float64) / np.sqrt(T)
        lam, B = eigsh(csr_matrix(L_sym), k=K_eff, sigma=-1e-6, which="LM",
                       v0=v0)
        order = np.argsort(lam)
        lam = np.maximum(lam[order], 0.0)
        B = B[:, order]
        return lam, B
    except Exception:
        return None


def reliability_weights(Z: np.ndarray, G_net) -> np.ndarray:
    """Algorithm 1 lines 3-5: r_t = lambda_min(G(z_t; phi)) for every t.

    Args:
        Z: (T, d) float32 latent trajectory.
        G_net: fitted SPDMetricNet; forward (1, T, d) -> (1, T, d, d).

    Returns:
        r: (T,) float64, strictly positive (G = LL^T + lambda_geo*I is SPD).
           Non-finite entries (defensive) are replaced by 1.0.
    """
    import torch

    G_net.eval()
    device = next(G_net.parameters()).device
    with torch.no_grad():
        Z_t = torch.from_numpy(Z.astype(np.float32)).unsqueeze(0).to(device)
        G_field = G_net(Z_t)[0].cpu().numpy().astype(np.float64)  # (T, d, d)
    G_field = 0.5 * (G_field + np.transpose(G_field, (0, 2, 1)))
    r = np.linalg.eigvalsh(G_field)[:, 0]  # lambda_min per time step
    r = np.where(np.isfinite(r) & (r > 0), r, 1.0)
    return r


def mhb_solve_coeffs(
    B_O: np.ndarray,
    lam: np.ndarray,
    Z_O: np.ndarray,
    r: Optional[np.ndarray] = None,
    gamma: float = 0.0,
) -> np.ndarray:
    """Eqs. (27)-(28): closed-form weighted LS with manifold Tikhonov.

    Solves, jointly for all latent dims d' (they share the same K x K system):

        C = (B_O^T R B_O + gamma * Lambda)^{-1} B_O^T R Z_O

    Args:
        B_O: (n_obs, K) basis rows at observed time steps (B_K[O, :]).
        lam: (K,) eigenvalues (Lambda = diag(lam)).
        Z_O: (n_obs, d) observed latent coordinates (Z[O, :]).
        r:   (n_obs,) reliability weights; None -> R = I.
        gamma: Tikhonov coefficient (paper fixes 1e-3; 0 disables).

    Returns:
        C: (K, d) basis coefficients.
    """
    B_O = np.asarray(B_O, dtype=np.float64)
    Z_O = np.asarray(Z_O, dtype=np.float64)
    n_obs, K = B_O.shape
    if r is None:
        r = np.ones(n_obs, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64).reshape(n_obs)
    RB = r[:, None] * B_O                                  # R B_O
    Mat = B_O.T @ RB + float(gamma) * np.diag(np.asarray(lam, dtype=np.float64))
    rhs = RB.T @ Z_O                                       # B_O^T R Z_O
    try:
        C = np.linalg.solve(Mat, rhs)
    except np.linalg.LinAlgError:
        # Defensive jitter — Mat is PSD + gamma*Lambda, singular only in
        # degenerate cases (e.g. gamma=0 with rank-deficient B_O).
        C = np.linalg.solve(Mat + 1e-10 * np.eye(K), rhs)
    return C


def mhb_build_latent_graph(Z: np.ndarray, k: int = 10) -> np.ndarray:
    """kNN graph in latent space plus a temporal chain rescue).

    Dead rows connect to state-similar moments anywhere in the series;
    chain edges preserve temporal continuity.  Used by the soft spline
    operator instead of the mask-weighted PCA-distance graph, which is
    ill-defined for fully missing rows.
    """
    T = Z.shape[0]
    Zf = np.asarray(Z, dtype=np.float64)
    k_eff = min(int(k), max(1, T - 1))
    D2 = ((Zf[:, None, :] - Zf[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D2, np.inf)
    A_new = np.zeros((T, T), dtype=np.float64)
    nn = np.argsort(D2, axis=1)[:, :k_eff]
    rows = np.repeat(np.arange(T), k_eff)
    A_new[rows, nn.ravel()] = 1.0
    A_new = np.maximum(A_new, A_new.T)
    idx = np.arange(T - 1)
    A_new[idx, idx + 1] = 1.0
    A_new[idx + 1, idx] = 1.0
    return A_new


def mhb_spline_extend(
    Z: np.ndarray,
    I_miss: np.ndarray,
    row_frac: np.ndarray,
    *,
    spline_gamma: float = 0.02,
    k_latent: int = 10,
    w_floor: float = 0.05,
) -> np.ndarray:
    """Soft manifold harmonic extension (graph spline).

    Solves  min_Z' sum_t w_t ||z'_t - z_t||^2 + g tr(Z'^T L Z')
    closed-form: Z' = (diag(w) + g L)^{-1} diag(w) Z.

    Spectral filter h(lambda) = w / (w + g lambda) replaces the hard
    K-band projection.  As g -> 0 the operator reduces to leaving Stage I
    coordinates untouched on missing rows.
    """
    T, d = Z.shape
    Z_prime = Z.copy()
    I_miss = np.asarray(I_miss, dtype=int)
    if I_miss.size == 0:
        return Z_prime
    A_lat = mhb_build_latent_graph(Z, k=k_latent)
    deg = A_lat.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    L_sym = np.eye(T) - (d_inv_sqrt[:, None] * A_lat * d_inv_sqrt[None, :])
    L_sym = 0.5 * (L_sym + L_sym.T)
    w = np.asarray(row_frac, dtype=np.float64).reshape(T)
    w = np.maximum(w, float(w_floor))
    g = float(spline_gamma)
    Mat = np.diag(w) + g * L_sym + 1e-10 * np.eye(T)
    rhs = w[:, None] * Z.astype(np.float64)
    Z_rec = np.linalg.solve(Mat, rhs).astype(np.float32)
    miss_set = set(I_miss.tolist())
    obs_idx = np.array([t for t in range(T) if t not in miss_set], dtype=int)
    Z_prime[I_miss] = Z_rec[I_miss]
    Z_prime[obs_idx] = Z[obs_idx]
    return Z_prime


def mhb_impute_hard(
    Z: np.ndarray,
    G_net,
    I_miss: np.ndarray,
    A: np.ndarray,
    K: int = 5,
    gamma: float = 1e-3,
) -> np.ndarray:
    """Algorithm 1 (MHB Imputation), end to end.

    Args:
        Z: (T, d) float32 latent trajectory from Stage I.
        G_net: fitted SPDMetricNet (reliability weights); None -> R = I.
        I_miss: 1-D int array of missing time indices.
        A: (T, T) full-trajectory affinity (Stage I graph, eqs. (5)-(7)).
        K: basis bandwidth (paper default 5).
        gamma: manifold Tikhonov coefficient (paper default 1e-3).

    Returns:
        Z_prime: (T, d) float32; observed rows identical to Z (exact
        write-back, Algorithm 1 last line).
    """
    T, d = Z.shape
    Z_prime = Z.copy()
    I_miss = np.asarray(I_miss, dtype=int)
    if I_miss.size == 0:
        return Z_prime
    miss_set = set(I_miss.tolist())
    obs_idx = np.array([t for t in range(T) if t not in miss_set], dtype=int)
    if obs_idx.size == 0:
        return Z_prime  # nothing observed — leave Stage I estimate untouched

    # Lines 1-2: Laplacian + basis
    lam, B = mhb_basis(A, K)

    # Lines 3-5: reliability weights on observed rows
    if G_net is not None:
        r_full = reliability_weights(Z, G_net)
        r_obs = r_full[obs_idx]
    else:
        r_obs = None

    # Lines 6-8: closed-form coefficients (shared K x K system over dims)
    B_O = B[obs_idx, :]
    Z_O = Z[obs_idx, :].astype(np.float64)
    C = mhb_solve_coeffs(B_O, lam, Z_O, r=r_obs, gamma=gamma)

    # Line 9: reconstruct + exact observed write-back
    Z_rec = (B @ C).astype(np.float32)
    Z_prime[I_miss] = Z_rec[I_miss]
    Z_prime[obs_idx] = Z[obs_idx]
    return Z_prime


def mhb_impute(
    Z: np.ndarray,
    G_net,
    I_miss: np.ndarray,
    A: np.ndarray,
    K: int = 5,
    gamma: float = 1e-3,
    *,
    mode: str = "soft",
    row_frac: Optional[np.ndarray] = None,
    spline_gamma: float = 0.02,
    k_latent: int = 10,
) -> np.ndarray:
    """Stage II latent repair — default soft graph spline.

    mode='soft' (default): full-spectrum Laplacian-regularized extension on a
    latent kNN graph; mode='hard': legacy K-band MHB projection (Algorithm 1).
    """
    if str(mode).lower() == "hard":
        return mhb_impute_hard(Z, G_net, I_miss, A, K=K, gamma=gamma)
    T = Z.shape[0]
    if row_frac is None:
        row_frac = np.ones(T, dtype=np.float64)
    return mhb_spline_extend(
        Z,
        I_miss,
        row_frac,
        spline_gamma=spline_gamma,
        k_latent=k_latent,
    )
