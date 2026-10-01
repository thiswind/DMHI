"""Window filtering for Stage II G training."""
from __future__ import annotations

import numpy as np


def window_scalar_obs(M_n: np.ndarray) -> float:
    """Fraction of observed entries in window (N,T,D) or (T,D)."""
    return float(M_n.mean())


def window_joint_obs(M_n: np.ndarray) -> float:
    """Mean per-timestep observed-dimension ratio."""
    if M_n.ndim == 3:
        M_n = M_n[0]
    per_t = M_n.mean(axis=1)
    return float(per_t.mean())


def window_dynamics(Z_n: np.ndarray, M_n: np.ndarray) -> float:
    """Mean latent step norm on timesteps with sufficient observations."""
    if M_n.ndim == 3:
        M_n = M_n[0]
    T = Z_n.shape[0]
    if T < 2:
        return 0.0
    dz = np.linalg.norm(np.diff(Z_n.astype(np.float64), axis=0), axis=1)
    pair_obs = (M_n[:-1].mean(axis=1) > 0.1) & (M_n[1:].mean(axis=1) > 0.1)
    if not pair_obs.any():
        return 0.0
    return float(dz[pair_obs].mean())


def filter_windows(
    M_train: np.ndarray,
    Z_train: np.ndarray,
    rho_win: float = 0.25,
    rho_joint: float = 0.15,
    tau_dyn: float | None = None,
    min_run: int = 5,
    min_scalar_obs: float | None = None,
) -> np.ndarray:
    """Return indices of training windows suitable for G training.

    Uses native ``M_train`` only (no eval block mask). ``tau_dyn`` defaults to
    the 25th percentile of dynamics among windows passing obs filters.
    """
    N = M_train.shape[0]
    candidates: list[int] = []
    dyn_vals: list[float] = []

    obs_threshold = float(min_scalar_obs) if min_scalar_obs is not None else rho_win

    for n in range(N):
        m_n = M_train[n]
        if window_scalar_obs(m_n) < obs_threshold:
            continue
        if window_joint_obs(m_n) < rho_joint:
            continue
        dyn = window_dynamics(Z_train[n], m_n)
        candidates.append(n)
        dyn_vals.append(dyn)

    if not candidates:
        return np.arange(min(N, 64), dtype=np.int64)

    dyn_arr = np.asarray(dyn_vals, dtype=np.float64)
    if tau_dyn is None:
        tau_dyn = float(np.quantile(dyn_arr, 0.25))

    kept = [
        n
        for n, dyn in zip(candidates, dyn_vals)
        if dyn >= tau_dyn or dyn >= tau_dyn * 0.5
    ]
    if len(kept) < min_run:
        kept = candidates[: max(min_run, len(candidates))]
    return np.asarray(kept, dtype=np.int64)


def filter_summary(
    M_train: np.ndarray,
    Z_train: np.ndarray,
    indices: np.ndarray,
    rho_win: float,
    rho_joint: float,
    tau_dyn: float | None,
) -> dict:
    """Diagnostics for filtered window set."""
    if tau_dyn is None and len(indices):
        tau_dyn = float(
            np.quantile(
                [window_dynamics(Z_train[i], M_train[i]) for i in indices], 0.25
            )
        )
    return {
        "n_total": int(M_train.shape[0]),
        "n_kept": int(len(indices)),
        "rho_win": rho_win,
        "rho_joint": rho_joint,
        "tau_dyn": tau_dyn,
        "mean_obs_kept": float(np.mean([window_scalar_obs(M_train[i]) for i in indices]))
        if len(indices)
        else 0.0,
    }
