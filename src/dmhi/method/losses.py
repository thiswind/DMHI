"""Loss functions for training the SPDMetricNet (Stage II).

All functions operate on torch tensors (float32, on MPS/CUDA/CPU).
They are called inside the training loop in pipeline.py.
"""
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metric import SPDMetricNet


def loss_geo(
    Z_hat: torch.Tensor,
    Z_target: torch.Tensor,
    G: torch.Tensor,
    beta: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """MLE Gaussian metric loss (Riemannian metric learning standard form).

    L_geo = 0.5 * mean (z_hat - z_target)^T G (z_hat - z_target)
          - 0.5 * beta * mean log det Gtage E (fix break ④): the original loss

        L_geo = mean (z_hat - z_target)^T G (z_hat - z_target)

    is monotonically minimised by G → 0. Combined with the SPD floor
    G_net = ... + lambda_geo * I, the global optimum is G ≡ lambda_geo·I —
    a *constant isotropic* metric, regardless of the data — and the
    learned metric carries no useful anisotropy.

    The MLE form adds an antagonist:
      - Term 1 (Mahalanobis residual) pushes G to be small in directions
        where the residual is large.
      - Term 2 (-log det G) is entropy-like and pushes G to be large
        overall.
    Their equilibrium gives G(z) ~ inverse of the local residual
    covariance — naturally anisotropic, naturally non-trivial. This is
    the standard form used in Hauberg & Freifeld (2014) and Lebanon
    (2006) for Riemannian metric learning.

    Numerical stability: log det is computed via Cholesky on (G + eps*I)
    to avoid log(0). If G becomes near-singular the Cholesky may fail —
    if training diverges, lowering ``beta`` (e.g. 0.5 → 0.2) re-balances
    the two terms.

    Args:
        Z_hat: (B, T, d) imputed latent codes (zeros at masked positions).
        Z_target: (B, T, d) ground-truth latent codes.
        G: (B, T, d, d) SPD metric field.
        beta: weight on the log-det regulariser. Higher → larger,
            more isotropic G. Default 0.5.
        eps: floor added to G before Cholesky for numerical safety.

    Returns:
        Scalar loss.
    """
    diff = (Z_hat - Z_target).unsqueeze(-1)                        # (B, T, d, 1)
    d_G_sq = (diff.transpose(-1, -2) @ G @ diff).squeeze(-1).squeeze(-1)  # (B, T)

    d = G.shape[-1]
    I = torch.eye(d, device=G.device, dtype=G.dtype) * eps
    L = torch.linalg.cholesky(G + I)                               # (B, T, d, d)
    log_det_G = 2.0 * torch.log(
        torch.diagonal(L, dim1=-2, dim2=-1)
    ).sum(dim=-1)                                                  # (B, T)

    return 0.5 * d_G_sq.mean() - 0.5 * beta * log_det_G.mean()


def loss_geo_masked(
    Z_hat: torch.Tensor,
    Z_target: torch.Tensor,
    G: torch.Tensor,
    miss_mask: torch.Tensor,
    beta: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """MLE geo loss averaged only over missing timesteps (REVISE-004 Stage II v2).

    Args:
        miss_mask: (B, T) bool or float, True/1 = missing position.
    """
    diff = (Z_hat - Z_target).unsqueeze(-1)
    d_G_sq = (diff.transpose(-1, -2) @ G @ diff).squeeze(-1).squeeze(-1)

    d = G.shape[-1]
    I = torch.eye(d, device=G.device, dtype=G.dtype) * eps
    L = torch.linalg.cholesky(G + I)
    log_det_G = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

    per_t = 0.5 * d_G_sq - 0.5 * beta * log_det_G
    m = miss_mask.to(dtype=per_t.dtype)
    denom = m.sum().clamp(min=1.0)
    return (per_t * m).sum() / denom


def loss_aniso(G: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Penalise near-constant G(t) across time (REVISE-004, anti D2 collapse).

    Uses temporal Frobenius variation (MPS-safe, differentiable).
    """
    if G.shape[1] < 2:
        return G.new_tensor(0.0)
    diff = G[:, 1:] - G[:, :-1]
    per_b = diff.pow(2).sum(dim=(-1, -2)).mean(dim=1).sqrt()
    return (-torch.log(per_b.mean() + eps))


def loss_reg(G: torch.Tensor) -> torch.Tensor:
    """Frobenius regularisation on the Cholesky factor.

    Penalises ||G||_F^2 to prevent metric collapse or explosion.

    Args:
        G: (B, T, d, d) SPD metric field.

    Returns:
        Scalar loss.
    """
    return G.pow(2).sum(dim=(-1, -2)).mean()


def loss_smooth(G: torch.Tensor) -> torch.Tensor:
    """Temporal smoothness of the metric field.

    L_smooth = mean_{b,t} ||G_{t+1} - G_t||_F^2

    Args:
        G: (B, T, d, d) SPD metric field.

    Returns:
        Scalar loss (0 if T < 2).
    """
    if G.shape[1] < 2:
        return G.new_tensor(0.0)
    diff = G[:, 1:] - G[:, :-1]         # (B, T-1, d, d)
    return diff.pow(2).sum(dim=(-1, -2)).mean()


def loss_time(Z: torch.Tensor, tau: float) -> torch.Tensor:
    """Hinge loss on consecutive latent step norms.

    L_time = mean_t max(0, ||z_{t+1} - z_t|| - tau)

    Args:
        Z: (B, T, d) latent sequence.
        tau: target maximum step size.

    Returns:
        Scalar loss.
    """
    if Z.shape[1] < 2:
        return Z.new_tensor(0.0)
    steps = Z[:, 1:] - Z[:, :-1]        # (B, T-1, d)
    norms = steps.norm(dim=-1)           # (B, T-1)
    hinge = torch.clamp(norms - tau, min=0.0)
    return hinge.mean()


def loss_identity(G: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Frobenius distance from G to identity (IA2 identity-teacher)."""
    if G.dim() == 4:
        d = G.shape[-1]
        eye = torch.eye(d, device=G.device, dtype=G.dtype).expand_as(G)
    else:
        d = G.shape[-1]
        eye = torch.eye(d, device=G.device, dtype=G.dtype).expand_as(G)
    return (G - eye).pow(2).sum(dim=(-1, -2)).mean()


def loss_pullback(G_pred: torch.Tensor, G_tgt: torch.Tensor) -> torch.Tensor:
    """Frobenius loss matching predicted G to embedder pullback targets (S2-R).

    Args:
        G_pred: (N, d, d) or (B, T, d, d)
        G_tgt: same shape as G_pred
    """
    diff = G_pred - G_tgt
    return diff.pow(2).sum(dim=(-1, -2)).mean()


def loss_norm(Z: torch.Tensor) -> torch.Tensor:
    """Gauge-fixing loss: zero mean + unit covariance.

    L_norm = ||mean(Z)||^2 + ||Cov(Z) - I||_F^2
    Computed over the (B*T, d) flattened population.

    Args:
        Z: (B, T, d) latent sequence.

    Returns:
        Scalar loss.
    """
    BT = Z.shape[0] * Z.shape[1]
    Z_flat = Z.reshape(BT, -1)           # (B*T, d)
    mu = Z_flat.mean(dim=0)              # (d,)
    Zc = Z_flat - mu
    if BT > 1:
        cov = (Zc.T @ Zc) / (BT - 1)    # (d, d)
    else:
        cov = torch.eye(Z.shape[-1], device=Z.device, dtype=Z.dtype)
    I = torch.eye(Z.shape[-1], device=Z.device, dtype=Z.dtype)
    return mu.pow(2).sum() + (cov - I).pow(2).sum()
