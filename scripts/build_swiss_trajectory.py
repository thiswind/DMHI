#!/usr/bin/env python3
"""tage A: build SwissTrajectory dataset.

Each trajectory is a smooth random walk on a 2D swiss-roll surface,
embedded in R^3 via the standard swiss-roll parametrisation, then
lifted to R^12 via a fixed random orthogonal matrix + Gaussian noise.

Output (under data/processed/swiss_trajectory/):
  X_{train,val,test}.npy : (N, T, D) float32, standardised
  M_{train,val,test}.npy : (N, T, D) int8, all 1 (input fully observed;
                            test-time missing applied in run scripts)
  z_{train,val,test}.npy : (N, T, 2) ground-truth intrinsic coords
                            (for D2/D3 diagnostics only, not training)
  dataset_card.json      : reproducibility metadata
"""
import json
import pathlib

import numpy as np

OUT = pathlib.Path("data/processed/swiss_trajectory")
OUT.mkdir(parents=True, exist_ok=True)

# --- PRE-COMMITTED hyperparameters (do NOT tune after seeing results) ---
N_TOTAL = 2400
T = 48
D = 12
SIGMA_NOISE = 0.05
OU_TAU = 10.0
SEED = 20260520

# swiss-roll parametric domain
U_MIN, U_MAX = 1.5 * np.pi, 4.5 * np.pi
V_MIN, V_MAX = 0.0, 21.0

rng = np.random.default_rng(SEED)


def ou_walk(N, T, mu, lo, hi, tau, sigma_step, rng):
    """Reflected Ornstein-Uhlenbeck process on [lo, hi]."""
    X = np.zeros((N, T))
    X[:, 0] = rng.uniform(lo, hi, size=N)
    dt = 1.0
    for t in range(1, T):
        drift = -(X[:, t - 1] - mu) / tau * dt
        noise = sigma_step * np.sqrt(dt) * rng.standard_normal(N)
        X[:, t] = X[:, t - 1] + drift + noise
        X[:, t] = np.clip(X[:, t], lo, hi)
    return X


u_traj = ou_walk(N_TOTAL, T, (U_MIN + U_MAX) / 2, U_MIN, U_MAX, OU_TAU, 0.40, rng)
v_traj = ou_walk(N_TOTAL, T, (V_MIN + V_MAX) / 2, V_MIN, V_MAX, OU_TAU, 0.60, rng)

# standard swiss-roll embedding in R^3
x1 = u_traj * np.cos(u_traj)
x2 = v_traj
x3 = u_traj * np.sin(u_traj)
X_r3 = np.stack([x1, x2, x3], axis=-1)               # (N, T, 3)

# random orthogonal lift R^3 -> R^12 (fixed seed)
Q_full = rng.standard_normal((D, D))
Q_full, _ = np.linalg.qr(Q_full)
Q = Q_full[:, :3]                                    # (D, 3) embedding
X_r12 = X_r3 @ Q.T                                   # (N, T, D)

# additive ambient noise
X_r12 = X_r12 + SIGMA_NOISE * rng.standard_normal(X_r12.shape)
X_r12 = X_r12.astype(np.float32)

# per-dim standardisation
flat = X_r12.reshape(-1, D)
mu_d = flat.mean(axis=0)
sigma_d = flat.std(axis=0) + 1e-6
X_r12 = (X_r12 - mu_d) / sigma_d

# splits
n_train, n_val = 1600, 400
idx = rng.permutation(N_TOTAL)
i_tr, i_va, i_te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

M_all = np.ones_like(X_r12, dtype=np.int8)
z_all = np.stack([u_traj, v_traj], axis=-1).astype(np.float32)   # (N, T, 2)

for name, ids in [("train", i_tr), ("val", i_va), ("test", i_te)]:
    np.save(OUT / f"X_{name}.npy", X_r12[ids])
    np.save(OUT / f"M_{name}.npy", M_all[ids])
    np.save(OUT / f"z_{name}.npy", z_all[ids])

(OUT / "dataset_card.json").write_text(json.dumps({
    "name": "swiss_trajectory",
    "purpose": " — controlled synthetic manifold for "
               "Riemannian metric mechanism validation",
    "N_total": N_TOTAL, "T": T, "D": D,
    "intrinsic_dim": 2,
    "manifold": ("swiss-roll surface, parametric (u cos u, v, u sin u), "
                 "u in [1.5pi, 4.5pi], v in [0, 21]"),
    "ambient_lift": "fixed orthogonal R^3 -> R^12",
    "noise_sigma": SIGMA_NOISE,
    "trajectory_dynamics": (
        "OU process: drift -(X-mu)/tau dt + sigma_step sqrt(dt) eta, "
        "tau=10, sigma_step_u=0.4, sigma_step_v=0.6, reflected on bounds"
    ),
    "splits": {"train": n_train, "val": n_val, "test": N_TOTAL - n_train - n_val},
    "seed": SEED,
    "input_mask_policy": "all-observed; missing applied at run-time",
    "Q": Q.tolist(),
    "mu_d": mu_d.tolist(),
    "sigma_d": sigma_d.tolist(),
}, indent=2, ensure_ascii=False))

print(f"[swiss_trajectory] saved {N_TOTAL} trajectories to {OUT}")
print(f"  shape  : {X_r12.shape}, range [{X_r12.min():.3f}, {X_r12.max():.3f}]")
print(f"  splits : train={n_train}, val={n_val}, test={N_TOTAL - n_train - n_val}")
