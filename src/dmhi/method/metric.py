"""Stage II: Learnable position-dependent SPD metric field.

SPDMetricNet maps a latent sequence Z ∈ R^{T×d} to a field of SPD matrices
G(z_t) ∈ R^{d×d} via:

  TransformerEncoder → linear head → lower-triangular L (softplus diagonal)
  G = L L^T + lambda_geo * I_d

This guarantees G is SPD for any input. The network runs on MPS/CUDA/CPU
but always returns numpy arrays for downstream Karcher iterations.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SPDMetricNet(nn.Module):
    """Transformer-based learnable SPD metric field.

    Args:
        d: latent dimension (must match embedder d).
        n_head: number of attention heads (must divide d).
        n_layers: number of TransformerEncoder layers.
        d_ff: feedforward hidden size.
        lambda_geo: minimum eigenvalue regularisation (additive λI).
    """

    def __init__(
        self,
        d: int = 8,
        n_head: int = 4,
        n_layers: int = 4,
        d_ff: int = 128,
        lambda_geo: float = 1e-3,
    ):
        super().__init__()
        self.d = d
        self.lambda_geo = lambda_geo
        self.n_tril = d * (d + 1) // 2   # lower-triangular entries

        # Input projection (if d < minimum for nhead)
        self.d_model = max(d, n_head * 4)
        self.input_proj = nn.Linear(d, self.d_model) if self.d_model != d else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(self.d_model, self.n_tril)

    def _cholesky_from_flat(self, flat: torch.Tensor) -> torch.Tensor:
        """Build lower-triangular L from flat vector of n_tril entries.

        Args:
            flat: (..., n_tril) float32.

        Returns:
            L: (..., d, d) lower-triangular with softplus diagonal.
        """
        d = self.d
        *batch, _ = flat.shape
        L = torch.zeros(*batch, d, d, device=flat.device, dtype=flat.dtype)
        tril_idx = torch.tril_indices(d, d, device=flat.device)
        L[..., tril_idx[0], tril_idx[1]] = flat
        # Ensure positive diagonal via softplus
        diag_idx = torch.arange(d, device=flat.device)
        L[..., diag_idx, diag_idx] = torch.nn.functional.softplus(
            L[..., diag_idx, diag_idx]
        ) + 1e-4
        return L

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute G field for a batch of latent sequences.

        Args:
            Z: (B, T, d) float32 on self.device.

        Returns:
            G: (B, T, d, d) SPD matrices.
        """
        h = self.input_proj(Z)                    # (B, T, d_model)
        h = self.transformer(h)                   # (B, T, d_model)
        flat = self.head(h)                       # (B, T, n_tril)
        L = self._cholesky_from_flat(flat)        # (B, T, d, d)
        G = L @ L.transpose(-1, -2)              # (B, T, d, d)
        G = G + self.lambda_geo * torch.eye(
            self.d, device=Z.device, dtype=Z.dtype
        )
        return G

    def get_metric(self, z: np.ndarray) -> np.ndarray:
        """Return G(z) as a numpy (d, d) SPD matrix for a single latent vector.

        Args:
            z: (d,) float32 numpy array.

        Returns:
            G: (d, d) float32 numpy array.
        """
        device = next(self.parameters()).device
        z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            G = self.forward(z_t)   # (1, 1, d, d)
        return G[0, 0].cpu().numpy()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class MLPMetricNet(nn.Module):
    """MLP-based SPD metric field (ablation: replaces Transformer backbone).

    Args:
        d: latent dimension.
        hidden: hidden layer width.
        n_layers: number of hidden layers.
        lambda_geo: minimum eigenvalue regularisation (additive λI).
    """

    def __init__(
        self,
        d: int = 8,
        hidden: int = 256,
        n_layers: int = 3,
        lambda_geo: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.d = d
        self.lambda_geo = lambda_geo
        self.n_tril = d * (d + 1) // 2

        layers: list = [nn.Linear(d, hidden), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, self.n_tril))
        self.net = nn.Sequential(*layers)

    def _cholesky_from_flat(self, flat: torch.Tensor) -> torch.Tensor:
        d = self.d
        *batch, _ = flat.shape
        L = torch.zeros(*batch, d, d, device=flat.device, dtype=flat.dtype)
        tril_idx = torch.tril_indices(d, d, device=flat.device)
        L[..., tril_idx[0], tril_idx[1]] = flat
        diag_idx = torch.arange(d, device=flat.device)
        L[..., diag_idx, diag_idx] = torch.nn.functional.softplus(
            L[..., diag_idx, diag_idx]
        ) + 1e-4
        return L

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        flat = self.net(Z)                            # (B, T, n_tril)
        L = self._cholesky_from_flat(flat)            # (B, T, d, d)
        G = L @ L.transpose(-1, -2)
        G = G + self.lambda_geo * torch.eye(
            self.d, device=Z.device, dtype=Z.dtype
        )
        return G

    def get_metric(self, z: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            G = self.forward(z_t)
        return G[0, 0].cpu().numpy()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def diagnose_G_field(G_net, Z_sample: torch.Tensor, device: torch.device) -> dict:
    """Diagnostic statistics of the learned G(z) field on a sample of latents.

    Used bytage A to verify whether the metric
    network has learned a non-trivial (anisotropic) SPD field, or has
    collapsed to a constant multiple of identity.

    Args:
        G_net: a trained MetricNet (SPDMetricNet / MLPMetricNet / LSTMMetricNet)
        Z_sample: (B, T, d) latent codes to evaluate G on (typically a small
                  batch of training latents)
        device: target device for the forward pass

    Returns:
        dict with keys:
          eig_min, eig_max:         scalar min/max eigenvalue across batch
          eig_ratio_log10:          log10(max/min) — anisotropy span
          frob_norm_mean:           mean Frobenius norm of G across batch
    """
    G_net.eval()
    with torch.no_grad():
        G = G_net(Z_sample.to(device))            # (B, T, d, d)
    G_np = G.cpu().numpy().reshape(-1, G.shape[-1], G.shape[-1])
    eigs = np.linalg.eigvalsh(G_np)               # (B*T, d), real symmetric SPD
    eig_min = float(eigs.min())
    eig_max = float(eigs.max())
    return {
        "eig_min": eig_min,
        "eig_max": eig_max,
        "eig_ratio_log10": float(
            np.log10(max(eig_max, 1e-12) / max(eig_min, 1e-12))
        ),
        "frob_norm_mean": float(np.linalg.norm(G_np, axis=(-1, -2)).mean()),
    }


class LSTMMetricNet(nn.Module):
    """Bi-LSTM-based SPD metric field (ablation: replaces Transformer backbone).

    Args:
        d: latent dimension.
        hidden: LSTM hidden size (per direction).
        n_layers: number of LSTM layers.
        lambda_geo: minimum eigenvalue regularisation (additive λI).
    """

    def __init__(
        self,
        d: int = 8,
        hidden: int = 128,
        n_layers: int = 2,
        lambda_geo: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.d = d
        self.lambda_geo = lambda_geo
        self.n_tril = d * (d + 1) // 2

        self.lstm = nn.LSTM(
            d, hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
        )
        self.head = nn.Linear(hidden * 2, self.n_tril)

    def _cholesky_from_flat(self, flat: torch.Tensor) -> torch.Tensor:
        d = self.d
        *batch, _ = flat.shape
        L = torch.zeros(*batch, d, d, device=flat.device, dtype=flat.dtype)
        tril_idx = torch.tril_indices(d, d, device=flat.device)
        L[..., tril_idx[0], tril_idx[1]] = flat
        diag_idx = torch.arange(d, device=flat.device)
        L[..., diag_idx, diag_idx] = torch.nn.functional.softplus(
            L[..., diag_idx, diag_idx]
        ) + 1e-4
        return L

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(Z)                           # (B, T, 2*hidden)
        flat = self.head(h)                           # (B, T, n_tril)
        L = self._cholesky_from_flat(flat)            # (B, T, d, d)
        G = L @ L.transpose(-1, -2)
        G = G + self.lambda_geo * torch.eye(
            self.d, device=Z.device, dtype=Z.dtype
        )
        return G

    def get_metric(self, z: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            G = self.forward(z_t)
        return G[0, 0].cpu().numpy()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
