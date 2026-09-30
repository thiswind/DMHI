"""Stage II: Karcher mean imputation in the learnable Riemannian metric.

For each missing time step t, initialise z_t as a Gaussian-weighted average
of observed neighbours, then iterate natural-gradient steps under G(z_t)
until convergence (Karcher mean fixed-point).

All computation is NumPy; the metric net is called via get_metric() which
handles the torch/MPS backend internally and returns numpy.
"""
import numpy as np
from typing import TYPE_CHECKING, Callable, Optional

from .utils import gaussian_weights

if TYPE_CHECKING:
    from .metric import SPDMetricNet


# ---------------------------------------------------------------------------
# Karcher diagnosticstage A)
# Tracks call rate, skip rate (observed empty), and actual iteration steps.
# Resetable per impute() call from pipeline.py.
# ---------------------------------------------------------------------------

_KARCHER_STATS = {"n_calls": 0, "n_skipped": 0, "n_steps_taken": 0}
_LAST_KARCHER_DETAIL: dict = {"step_norms": [], "hold_ts": []}


def reset_karcher_detail() -> None:
    _LAST_KARCHER_DETAIL["step_norms"] = []
    _LAST_KARCHER_DETAIL["hold_ts"] = []


def get_last_karcher_detail() -> dict:
    return {
        "step_norms": list(_LAST_KARCHER_DETAIL.get("step_norms", [])),
        "hold_ts": list(_LAST_KARCHER_DETAIL.get("hold_ts", [])),
    }


def reset_karcher_stats() -> None:
    """Reset Karcher activation counters before a new impute() pass."""
    _KARCHER_STATS["n_calls"] = 0
    _KARCHER_STATS["n_skipped"] = 0
    _KARCHER_STATS["n_steps_taken"] = 0


def get_karcher_stats() -> dict:
    """Return current Karcher activation counters as a plain dict."""
    return dict(_KARCHER_STATS)


def _normalize_metric_g(G: np.ndarray, d: int) -> np.ndarray:
    """Trace-normalize SPD metric (REVISE-003 K3)."""
    tr = float(np.trace(G))
    scale = tr / max(d, 1)
    if scale < 1e-12:
        return G
    return G / scale


def _postprocess_metric_g(
    G: np.ndarray,
    d: int,
    normalize_g: bool = False,
    g_scale: float = 1.0,
    eigen_clip: float | None = None,
) -> np.ndarray:
    """Optional trace-normalize, eigenvalue clip, and scale for Karcher."""
    G = G.astype(np.float64, copy=True)
    if eigen_clip is not None and eigen_clip > 0:
        w, v = np.linalg.eigh(G)
        w = np.clip(w, 1e-8, float(eigen_clip))
        G = (v * w) @ v.T
    if normalize_g:
        G = _normalize_metric_g(G, d)
    if g_scale != 1.0:
        G = G * float(g_scale)
    return G


def _adaptive_eta(eta: float, G: np.ndarray) -> float:
    """Cap step size by condition number (REVISE-003 K1 variant)."""
    try:
        cond = float(np.linalg.cond(G))
    except np.linalg.LinAlgError:
        return eta
    if not np.isfinite(cond) or cond < 1.0:
        cond = 1.0
    return min(eta, 1.0 / np.sqrt(cond))


def _resolve_metric_g(
    G_net: "SPDMetricNet",
    z: np.ndarray,
    t: int,
    d: int,
    normalize_g: bool,
    g_scale: float,
    eigen_clip: float | None,
    g_active_mask: Optional[np.ndarray],
    pullback_blend_epsilon: float,
    pullback_at_t: Optional[Callable[[int], np.ndarray]],
    karcher_metric_mode: str = "learned",
) -> np.ndarray:
    """Metric for Karcher step, optional pullback blend (Phase 2.8)."""
    if g_active_mask is not None and not bool(g_active_mask[t]):
        return np.eye(d, dtype=np.float64)
    mode = str(karcher_metric_mode).lower()
    if mode == "identity":
        return _postprocess_metric_g(
            np.eye(d, dtype=np.float64), d, normalize_g, g_scale, eigen_clip
        )
    if mode == "pullback":
        if pullback_at_t is None:
            return _postprocess_metric_g(
                np.eye(d, dtype=np.float64), d, normalize_g, g_scale, eigen_clip
            )
        G_pb = pullback_at_t(int(t)).astype(np.float64)
        return _postprocess_metric_g(G_pb, d, normalize_g, g_scale, eigen_clip)
    G_learned = G_net.get_metric(z.astype(np.float32)).astype(np.float64)
    eps = float(pullback_blend_epsilon)
    if eps > 0.0 and pullback_at_t is not None:
        G_pb = pullback_at_t(int(t)).astype(np.float64)
        G_learned = (1.0 - eps) * G_learned + eps * G_pb
    return _postprocess_metric_g(G_learned, d, normalize_g, g_scale, eigen_clip)


def karcher_impute(
    Z: np.ndarray,
    G_net: "SPDMetricNet",
    I_miss: np.ndarray,
    k_time: int = 10,
    sigma: float = 3.0,
    N_max: int = 5,
    eta: float = 1.0,
    eps: float = 1e-5,
    use_clle_init: bool = True,
    # REVISE-003 consumer-path fixes (default off for backward compat)
    normalize_g: bool = False,
    clip_step_norm: Optional[float] = None,
    adaptive_eta: bool = False,
    g_scale: float = 1.0,
    eigen_clip: Optional[float] = None,
    g_active_mask: Optional[np.ndarray] = None,
    #: cross-subject latent bank
    Z_bank: Optional[np.ndarray] = None,
    cross_k: int = 0,
    cross_sigma: float = 3.0,
    # Phase 2.8: G_use = (1-ε) G_net + ε G_pullback in Karcher only
    pullback_blend_epsilon: float = 0.0,
    pullback_at_t: Optional[Callable[[int], np.ndarray]] = None,
    karcher_metric_mode: str = "learned",
    # Phase 2.12: oracle blend Z_k toward Z_gt at hold
    z_hold_oracle_alpha: float = 0.0,
    z_oracle_Z_gt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Impute missing latent codes via iterated Karcher mean.

    For each t in I_miss:
        1. Find k_time temporally nearest *observed* time steps N_t.
        2. Optionally add top-k_cross points from Z_bank (Mahalanobis under G).
        3. Initialise z_t from CLLE lift Z[t].
        4. Iterate natural-gradient steps with merged neighbor weights.

    Args:
        Z: (T, d) float32 latent array.
        G_net: fitted SPDMetricNet (in eval mode, on device).
        I_miss: 1-D int array of time indices to impute.
        Z_bank: (K, d) training latent bank for cross-subject neighbors.
        cross_k: number of cross-subject neighbors (0 = disabled).
        cross_sigma: Gaussian bandwidth for cross-subject weights.

    Returns:
        Z_prime: (T, d) float32 with I_miss entries filled.
    """
    from .cross_neighbors import query_cross_neighbors

    _KARCHER_STATS["n_calls"] += 1
    reset_karcher_detail()
    step_norms: list[float] = []

    T, d = Z.shape
    Z_prime = Z.copy()
    I_miss_set = set(I_miss.tolist())
    observed = np.array([t for t in range(T) if t not in I_miss_set], dtype=int)

    if len(observed) == 0:
        _KARCHER_STATS["n_skipped"] += 1
        return Z_prime

    for t in I_miss:
        t = int(t)
        time_dists = np.abs(observed - t)
        nn_k = min(k_time, len(observed))
        nn_idx = np.argsort(time_dists)[:nn_k]
        N_t = observed[nn_idx]

        w_time = gaussian_weights(N_t, t, sigma)
        Z_time = Z_prime[N_t].astype(np.float64)

        z_anchor = Z[t].astype(np.float64)
        G_anchor = _resolve_metric_g(
            G_net,
            z_anchor,
            t,
            d,
            normalize_g,
            g_scale,
            eigen_clip,
            g_active_mask,
            pullback_blend_epsilon,
            pullback_at_t,
            karcher_metric_mode,
        )

        Z_parts = [Z_time]
        w_parts = [w_time.astype(np.float64)]

        if cross_k > 0 and Z_bank is not None and len(Z_bank) > 0:
            Z_cross, w_cross = query_cross_neighbors(
                z_anchor,
                Z_bank,
                G_anchor,
                cross_k,
                cross_sigma,
            )
            if len(Z_cross) > 0:
                Z_parts.append(Z_cross.astype(np.float64))
                w_parts.append(w_cross)

        Z_nbrs = np.vstack(Z_parts)
        w = np.concatenate(w_parts)
        w_sum = float(w.sum())
        if w_sum < 1e-12:
            w = np.ones(len(w), dtype=np.float64) / max(len(w), 1)
        else:
            w = w / w_sum
        w_64 = w

        if use_clle_init:
            z_t = z_anchor.copy()
        else:
            z_t = (w[:, None] * Z_nbrs).sum(axis=0)

        for _ in range(N_max):
            G_t = _resolve_metric_g(
                G_net,
                z_t,
                t,
                d,
                normalize_g,
                g_scale,
                eigen_clip,
                g_active_mask,
                pullback_blend_epsilon,
                pullback_at_t,
                karcher_metric_mode,
            )
            u = (w_64[:, None] * (Z_nbrs - z_t)).sum(axis=0)
            try:
                step = np.linalg.solve(G_t, u)
            except np.linalg.LinAlgError:
                step = u
            eta_eff = _adaptive_eta(eta, G_t) if adaptive_eta else eta
            if clip_step_norm is not None:
                sn = float(np.linalg.norm(step))
                if sn > clip_step_norm and sn > 1e-12:
                    step = step * (clip_step_norm / sn)
            sn = float(np.linalg.norm(step))
            step_norms.append(sn)
            z_t = z_t + eta_eff * step
            _KARCHER_STATS["n_steps_taken"] += 1
            if sn < eps:
                break

        z_out = z_t.astype(np.float32)
        alpha = float(z_hold_oracle_alpha)
        if alpha > 0.0 and z_oracle_Z_gt is not None:
            z_gt_t = z_oracle_Z_gt[t].astype(np.float32)
            z_out = ((1.0 - alpha) * z_out + alpha * z_gt_t).astype(np.float32)
        Z_prime[t] = z_out
        _LAST_KARCHER_DETAIL["hold_ts"].append(t)

    _LAST_KARCHER_DETAIL["step_norms"] = step_norms
    return Z_prime
