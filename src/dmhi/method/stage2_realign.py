"""Stage II realign training: G learns embedder pullback on filtered windows (S2-B/S2-R)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn

from .window_filter import filter_summary, filter_windows
from .karcher import karcher_impute
from .losses import loss_aniso, loss_identity, loss_pullback, loss_reg, loss_smooth
from .metric import SPDMetricNet
from .pipeline import get_device
from .synth_missing import sample_block_mask

if TYPE_CHECKING:
    from .pipeline import RiemannianImputer

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


@dataclass
class RealignConfig:
    rho_win: float = 0.25
    rho_joint: float = 0.15
    tau_dyn: float | None = None
    lambda_pullback: float = 1.0
    lambda_aniso: float = 0.05
    lambda_F: float = 1e-5
    lambda_smooth: float = 1e-4
    n_pullback_cache: int = 512
    timesteps_per_window: int = 3
    min_joint_at_t: float = 0.20
    # P3b: hold-segment supervision aligned with eval block protocol
    lambda_hold_mae: float = 0.0
    use_eval_block_train: bool = False
    block_len: int = 20
    block_rate: float = 0.7
    block_seed: int = 44
    # Phase 2.6 I-B: train G on high-observation (clean) windows only
    train_on_clean_windows: bool = True
    min_window_obs_rate: float = 0.70
    pullback_primary: bool = True
    # Phase 2.11 IA tracks
    pullback_mse_only: bool = False
    lambda_identity: float = 0.0
    identity_teacher: bool = False


def _sample_pullback_cache(
    embedder,
    X_train: np.ndarray,
    M_train: np.ndarray,
    Z_train: np.ndarray,
    indices: np.ndarray,
    n_cache: int,
    timesteps_per_window: int,
    min_joint_at_t: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute (Z, G_pullback) pairs for distillation."""
    try:
        from experiments.embedder_pullback import embedder_pullback_at
    except ImportError as exc:
        raise NotImplementedError(
            "Stage-II realign training needs the embedder-pullback ablation "
            "module, which is not shipped in this release. The deployed "
            "imputer (stage2_method='mhb') does not use it."
        ) from exc

    rng = np.random.default_rng(seed)
    pairs_z: list[np.ndarray] = []
    pairs_g: list[np.ndarray] = []

    for n in indices:
        m_n = M_train[n]
        x_n = X_train[n]
        z_n = Z_train[n]
        T = x_n.shape[0]
        good_t = np.where(m_n.mean(axis=1) >= min_joint_at_t)[0]
        if len(good_t) == 0:
            continue
        pick = rng.choice(
            good_t, size=min(timesteps_per_window, len(good_t)), replace=False
        )
        for t in pick:
            try:
                G_t = embedder_pullback_at(embedder, x_n, m_n, int(t))
            except Exception:
                continue
            pairs_z.append(z_n[t].astype(np.float32))
            pairs_g.append(G_t.astype(np.float32))
            if len(pairs_z) >= n_cache:
                break
        if len(pairs_z) >= n_cache:
            break

    if not pairs_z:
        raise RuntimeError("pullback cache empty — relax filter thresholds")

    Z_pts = np.stack(pairs_z, axis=0)
    G_pts = np.stack(pairs_g, axis=0)
    if len(Z_pts) > n_cache:
        sel = rng.choice(len(Z_pts), n_cache, replace=False)
        Z_pts = Z_pts[sel]
        G_pts = G_pts[sel]
    return Z_pts, G_pts


def retrain_stage2_realign(
    imputer: "RiemannianImputer",
    X_train: np.ndarray,
    M_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    M_val: Optional[np.ndarray] = None,
    cfg: Optional[RealignConfig] = None,
) -> dict:
    """Train G on native M with pullback supervision; no eval block mask."""
    if imputer._embedder is None:
        raise RuntimeError("retrain_stage2_realign requires fitted embedder")
    if imputer.ablation_no_metric:
        raise RuntimeError("ablation_no_metric must be False")

    cfg = cfg or RealignConfig()
    device = get_device()
    imputer._device = device
    imputer.stage2_realign = True
    imputer.stage2_v2 = False
    imputer.consumer_karcher_fix = True

    N, T, _D = X_train.shape
    d_actual = imputer._embedder.d
    Z_train = imputer._embed_batch(X_train, M_train, desc="Embed train")
    Z_val = None
    if X_val is not None and M_val is not None:
        Z_val = imputer._embed_batch(X_val, M_val, desc="Embed val")

    indices = filter_windows(
        M_train,
        Z_train,
        rho_win=cfg.rho_win,
        rho_joint=cfg.rho_joint,
        tau_dyn=cfg.tau_dyn,
        min_scalar_obs=cfg.min_window_obs_rate if cfg.train_on_clean_windows else None,
    )
    filt_info = filter_summary(
        M_train, Z_train, indices, cfg.rho_win, cfg.rho_joint, cfg.tau_dyn
    )
    logger.info("Window filter: %s", filt_info)

    Z_pts, G_pts = _sample_pullback_cache(
        imputer._embedder,
        X_train,
        M_train,
        Z_train,
        indices,
        n_cache=cfg.n_pullback_cache,
        timesteps_per_window=cfg.timesteps_per_window,
        min_joint_at_t=cfg.min_joint_at_t,
        seed=42,
    )
    logger.info("Pullback cache: %d (Z, G) pairs", len(Z_pts))

    lambda_geo_eff = 0.0 if imputer.ablation_no_geo_reg else imputer.lambda_geo
    metric_net = SPDMetricNet(
        d=d_actual,
        n_head=imputer.n_head,
        n_layers=imputer.L_enc,
        d_ff=imputer.d_ff,
        lambda_geo=lambda_geo_eff,
    ).to(device)
    imputer._metric_net = metric_net

    optimiser = torch.optim.Adam(metric_net.parameters(), lr=imputer.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=imputer.epochs, eta_min=imputer.lr * 0.1
    )

    Z_cache_t = torch.from_numpy(Z_pts).to(device)
    G_cache_t = torch.from_numpy(G_pts).to(device)
    cache_n = Z_cache_t.shape[0]

    best_val = float("inf")
    no_improve = 0
    imputer._best_state = None

    epoch_iter = (
        _tqdm(range(imputer.epochs), desc="Stage II realign", unit="ep", dynamic_ncols=True)
        if _HAS_TQDM
        else range(imputer.epochs)
    )

    for epoch in epoch_iter:
        metric_net.train()
        perm_idx = np.random.permutation(len(indices))
        epoch_loss = 0.0
        n_batches = 0

        for b_start in range(0, len(indices), imputer.batch_size):
            batch_idx = indices[perm_idx[b_start : b_start + imputer.batch_size]]
            Z_batch = torch.from_numpy(Z_train[batch_idx]).to(device)

            G_full = metric_net(Z_batch)
            l_a = loss_aniso(G_full)
            l_r = loss_reg(G_full)
            l_s = loss_smooth(G_full)

            pb_sel = torch.randint(0, cache_n, (min(64, cache_n),), device=device)
            Z_pb = Z_cache_t[pb_sel].unsqueeze(0)
            G_tgt = G_cache_t[pb_sel]
            G_pred = metric_net(Z_pb)[0]
            l_pb = loss_pullback(G_pred, G_tgt)

            if cfg.pullback_mse_only:
                total = cfg.lambda_pullback * l_pb
            else:
                total = (
                    cfg.lambda_pullback * l_pb
                    + cfg.lambda_aniso * l_a
                    + cfg.lambda_F * l_r
                    + cfg.lambda_smooth * l_s
                )
                if cfg.pullback_primary:
                    total = cfg.lambda_pullback * l_pb + 0.1 * (
                        cfg.lambda_aniso * l_a
                        + cfg.lambda_F * l_r
                        + cfg.lambda_smooth * l_s
                    )

            if cfg.identity_teacher and cfg.lambda_identity > 0.0:
                l_id = loss_identity(G_full)
                total = total + cfg.lambda_identity * l_id

            if cfg.lambda_hold_mae > 0.0 or cfg.use_eval_block_train:
                l_hold = Z_batch.new_tensor(0.0)
                kw = imputer._karcher_train_kwargs()
                metric_net.eval()
                for bi, b_idx in enumerate(batch_idx):
                    z_np = Z_train[b_idx].astype(np.float32)
                    if cfg.use_eval_block_train:
                        miss_1d = sample_block_mask(
                            1, T, cfg.block_seed + int(b_idx) + epoch * 997
                        )[0]
                    else:
                        miss_1d = np.zeros(T, dtype=bool)
                    I_miss = np.where(miss_1d)[0]
                    if len(I_miss) == 0:
                        continue
                    z_obs = z_np.copy()
                    z_hat = karcher_impute(
                        z_obs,
                        metric_net,
                        I_miss,
                        **kw,
                    )
                    diff = z_hat[miss_1d] - z_np[miss_1d]
                    hold_val = float((diff ** 2).mean())
                    l_hold = l_hold + torch.tensor(
                        hold_val, dtype=torch.float32, device=device
                    )
                metric_net.train()
                if cfg.lambda_hold_mae > 0.0:
                    total = total + cfg.lambda_hold_mae * l_hold / max(
                        len(batch_idx), 1
                    )

            optimiser.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(metric_net.parameters(), 1.0)
            optimiser.step()

            epoch_loss += total.item()
            n_batches += 1

        scheduler.step()

        if Z_val is not None:
            metric_net.eval()
            with torch.no_grad():
                Z_v = torch.from_numpy(Z_val[: min(64, len(Z_val))]).to(device)
                G_v = metric_net(Z_v)
                val_loss = (
                    cfg.lambda_aniso * loss_aniso(G_v).item()
                    + cfg.lambda_F * loss_reg(G_v).item()
                )
            metric_net.train()
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                no_improve = 0
                imputer._best_state = {
                    k: v.cpu().clone() for k, v in metric_net.state_dict().items()
                }
            else:
                no_improve += 1
            if _HAS_TQDM and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(
                    train=f"{epoch_loss / max(n_batches, 1):.3f}",
                    val=f"{val_loss:.3f}",
                    patience=f"{no_improve}/{imputer.patience}",
                )
            if no_improve >= imputer.patience and epoch >= max(10, imputer.patience):
                logger.info("Early stop epoch %d val=%.4f", epoch + 1, val_loss)
                break
        else:
            imputer._best_state = {
                k: v.cpu().clone() for k, v in metric_net.state_dict().items()
            }

    if imputer._best_state is not None:
        metric_net.load_state_dict(
            {k: v.to(device) for k, v in imputer._best_state.items()}
        )
    else:
        imputer._best_state = {
            k: v.cpu().clone() for k, v in metric_net.state_dict().items()
        }
    metric_net.eval()

    imputer._Z_train = Z_train
    imputer._X_train = X_train.astype(np.float32)
    imputer._M_train = M_train.astype(np.int8)
    imputer._obs_rate = float(M_train.mean())
    imputer._latent_bank = None
    imputer._is_fitted = True

    return {
        "filter": filt_info,
        "pullback_cache_size": int(len(Z_pts)),
        "epochs_ran": epoch + 1,
    }
