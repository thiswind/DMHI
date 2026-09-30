"""End-to-end Riemannian Geodesic Imputation pipeline.

RiemannianImputer.fit(X_train, M_train, X_val, M_val) → trains Stage I + II.
RiemannianImputer.impute(X, M) → runs Stage I + II + III and returns hat_X.

Device priority: MPS → CUDA → CPU.
All data must be float32. M must be int8 or bool (1=observed, 0=missing).
Call `conda activate base` before running any Python script using this module.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional
import logging
import os
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from .embed import MaskedManifoldEmbedder
from .metric import SPDMetricNet, diagnose_G_field
from .karcher import (
    karcher_impute,
    reset_karcher_stats,
    get_karcher_stats,
    get_last_karcher_detail,
    reset_karcher_detail,
)
from .clle import clle_inverse
from .mhb import mhb_impute
from .utils import build_time_window_graph
from .losses import loss_aniso, loss_geo, loss_geo_masked, loss_reg, loss_smooth, loss_time
from .synth_missing import sample_block_mask
from .trace import (
    PipelineTracer,
    TraceRecord,
    collect_hold_simplex_stats,
    hold_timesteps,
    per_dim_mae,
)

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def get_device() -> torch.device:
    """Return MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DecoderMLP(nn.Module):
    """Small MLP that maps latent code z (d-dim) → ambient x (D-dim).

    Trained on observed (z, x) pairs from training set after Stage II.
    Replaces CLLE inverse in Stage III for better reconstruction on
    heavily missing datasets like PhysioNet.
    """

    def __init__(self, d_in: int, d_out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class RiemannianImputer:
    """Three-stage deterministic manifold-harmonic imputation (DMHI).

    Stage I:   MaskedManifoldEmbedder  (numpy, CPU) -- mask-aware latent embedding
    Stage II:  soft graph-spline repair on the manifold-harmonic basis
               (``mhb_impute``, the default ``stage2_method='mhb'``)
    Stage III: CLLE symmetric inverse (``clle_inverse``) that freezes observed
               entries; no neural network on the inference path.

    The retired Karcher-mean Stage II and the learned DecoderMLP Stage III remain
    in the codebase as ablation/training paths only; they are off by default.
    """

    def __init__(
        self,
        # Stage I
        d: int = 8,
        k: int = 10,
        W_search: int = 5,
        k_clle: int = 9,
        d_pca: int | None = None,
        hold_blend_alpha: float = 0.0,
        r_L: float = 0.12,
        lambda_I: float = 0.01,
        lambda_norm: float = 0.1,
        embed_epochs: int = 200,
        # Stage II — metric net
        n_head: int = 4,
        L_enc: int = 4,
        d_ff: int = 128,
        lambda_geo: float = 1e-3,
        # Stage II — training
        #tage E: lambda_F decreased from 1e-4 to 1e-5
        # to give the new MLE-style loss_geo more room. The Frobenius penalty
        # on G is now soft-floor (anti-explosion) rather than the dominant
        # regulariser.
        lambda_F: float = 1e-5,
        lambda_smooth: float = 1e-4,
        lr: float = 1e-4,
        epochs: int = 200,
        patience: int = 20,
        batch_size: int = 64,
        synth_miss_rate: float = 0.3,
        # Stage II — Karcher
        N_max: int = 5,
        k_time: int = 10,
        sigma: float = 3.0,
        eta: float = 0.02,
        # Stage III — DecoderMLP
        decoder_hidden: int = 256,
        decoder_epochs: int = 200,
        decoder_lr: float = 1e-3,
        decoder_patience: int = 30,
        # --- Ablation flags (do not set in production) ---
        ablation_no_embed: bool = False,
        ablation_no_metric: bool = False,
        ablation_use_decoder: bool = False,
        ablation_no_mask_dist: bool = False,
        ablation_no_window_knn: bool = False,
        ablation_no_hinge: bool = False,
        ablation_metric_backbone: str = "transformer",
        ablation_no_geo_reg: bool = False,
        # REVISE-003: orthogonal consumer-path switches (inference only)
        ablation_no_karcher_g: bool = False,
        ablation_no_simplex_g: bool = False,
        # P1: G only in Karcher, not Stage III simplex
        scoped_g_simplex: bool = True,
        # P2: hold-block timesteps use spatial-only CLLE lift
        hold_aware_stage1: bool = False,
        # P6/P7: Karcher metric post-processing at inference
        force_normalize_g: bool = False,
        g_karcher_scale: float = 0.5,
        g_eigen_clip: float | None = None,
        g_apply_mode: str = "global",
        g_neighborhood_pad: int = 10,
        karcher_use_clle_init: bool = True,
        # Phase 2.8: blend G_net with embedder pullback in Karcher (eval knob)
        pullback_blend_epsilon: float = 0.0,
        # Phase 2.10: post-Karcher Z blend + metric mode oracle
        karcher_blend_lambda: float = 1.0,
        karcher_metric_mode: str = "learned",
        # REVISE-003 Stage C: Karcher consumer fixes (K1–K3)
        consumer_karcher_fix: bool = True,
        # REVISE-004 Stage II v2: block synth missing + in-loop Karcher
        stage2_v2: bool = False,
        # CONTRAST-REALIGN Stage II: pullback on filtered native windows
        stage2_realign: bool = False,
        lambda_aniso: float = 0.01,
        #: cross-subject Karcher neighbors (ablation only)
        ablation_cross_subject_knn: bool = False,
        cross_k: int = 20,
        cross_sigma: float = 3.0,
        bank_max_points: int = 50_000,
        # Phase 2.12 LOG→PRESET oracle (inference only)
        oracle_x_hold_fill: bool = False,
        oracle_x_hold_alpha: float = 1.0,
        z_n_oracle_alpha: float = 0.0,
        z_prime_oracle_alpha: float = 0.0,
        z_hold_oracle_alpha: float = 0.0,
        # /:
        # Stage II operator selection. 'mhb' (paper Algorithm 1) is the
        # canonical default since; 'karcher' is kept as the ablation
        # operator. Inference-time choice — checkpoints trained with either
        # value are interchangeable.
        stage2_method: str = "mhb",
        mhb_params: dict | None = None,
    ):
        self.d = d
        self.k = k
        self.W_search = W_search
        self.k_clle = k_clle
        self.d_pca = d_pca
        self.hold_blend_alpha = float(hold_blend_alpha)
        self.r_L = r_L
        self.lambda_I = lambda_I
        self.lambda_norm = lambda_norm
        self.embed_epochs = embed_epochs

        self.n_head = n_head
        self.L_enc = L_enc
        self.d_ff = d_ff
        self.lambda_geo = lambda_geo

        self.lambda_F = lambda_F
        self.lambda_smooth = lambda_smooth
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.synth_miss_rate = synth_miss_rate

        self.N_max = N_max
        self.k_time = k_time
        self.sigma = sigma
        self.eta = eta

        self.decoder_hidden = decoder_hidden
        self.decoder_epochs = decoder_epochs
        self.decoder_lr = decoder_lr
        self.decoder_patience = decoder_patience

        self.ablation_no_embed = ablation_no_embed
        self.ablation_no_metric = ablation_no_metric
        self.ablation_use_decoder = ablation_use_decoder
        self.ablation_no_mask_dist = ablation_no_mask_dist
        self.ablation_no_window_knn = ablation_no_window_knn
        self.ablation_no_hinge = ablation_no_hinge
        self.ablation_metric_backbone = ablation_metric_backbone
        self.ablation_no_geo_reg = ablation_no_geo_reg
        self.ablation_no_karcher_g = ablation_no_karcher_g
        self.ablation_no_simplex_g = ablation_no_simplex_g
        self.scoped_g_simplex = scoped_g_simplex
        self.hold_aware_stage1 = hold_aware_stage1
        self.force_normalize_g = force_normalize_g
        self.g_karcher_scale = g_karcher_scale
        self.g_eigen_clip = g_eigen_clip
        self.g_apply_mode = g_apply_mode
        self.g_neighborhood_pad = g_neighborhood_pad
        self.karcher_use_clle_init = karcher_use_clle_init
        self.pullback_blend_epsilon = float(pullback_blend_epsilon)
        self.karcher_blend_lambda = float(karcher_blend_lambda)
        self.karcher_metric_mode = str(karcher_metric_mode)
        self.consumer_karcher_fix = consumer_karcher_fix
        self.stage2_v2 = stage2_v2
        self.stage2_realign = stage2_realign
        self.lambda_aniso = lambda_aniso
        self.ablation_cross_subject_knn = ablation_cross_subject_knn
        self.cross_k = cross_k
        self.cross_sigma = cross_sigma if cross_sigma > 0 else sigma
        self.bank_max_points = bank_max_points
        self.oracle_x_hold_fill = bool(oracle_x_hold_fill)
        self.oracle_x_hold_alpha = float(oracle_x_hold_alpha)
        self.z_n_oracle_alpha = float(z_n_oracle_alpha)
        self.z_prime_oracle_alpha = float(z_prime_oracle_alpha)
        self.z_hold_oracle_alpha = float(z_hold_oracle_alpha)
        self.stage2_method = str(stage2_method)
        # Paper defaults: soft spline; hard band-limit via mode='hard'
        self.mhb_params = dict(mhb_params) if mhb_params else {
            "mode": "soft",
            "spline_gamma": 0.02,
            "k_latent": 10,
            "K": 5,
            "gamma": 1e-3,
        }
        if (consumer_karcher_fix or stage2_v2 or stage2_realign) and eta >= 1.0:
            self.eta = 0.1  # K1: smaller step when fix / v2 profile enabled

        self._val_history: list = []
        self._ablation_pca = None

        self._embedder: Optional[MaskedManifoldEmbedder] = None
        self._metric_net: Optional[SPDMetricNet] = None
        self._decoder: Optional[DecoderMLP] = None
        self._device: Optional[torch.device] = None
        self._tau: float = 1.0  # temporal hinge threshold from Stage I
        self._is_fitted: bool = False
        # K-NN ambient anchoring (stored after fit)
        self._Z_train: Optional[np.ndarray] = None   # (N, T, d)
        self._X_train: Optional[np.ndarray] = None   # (N, T, D)
        self._M_train: Optional[np.ndarray] = None   # (N, T, D)
        self._obs_rate: float = 1.0                   # training observation rate
        self._latent_bank: Optional[np.ndarray] = None
        self._tracer: Optional[PipelineTracer] = None
        self._trace_held_block: Optional[np.ndarray] = None
        self._trace_x_gt: Optional[np.ndarray] = None
        self._trace_m_native: Optional[np.ndarray] = None

    def _effective_cross_k(self) -> int:
        """Cross-subject neighbors disabled when no metric or flag off."""
        if self.ablation_no_metric or not self.ablation_cross_subject_knn:
            return 0
        return int(self.cross_k)

    def _ensure_latent_bank(self) -> None:
        """Build training latent bank for cross-subject Karcher (once per impute pass)."""
        if self._latent_bank is not None:
            return
        if self._Z_train is None or self._M_train is None:
            raise RuntimeError(
                "_ensure_latent_bank requires _Z_train/_M_train; call fit() or load a fitted imputer."
            )
        from .cross_neighbors import build_train_latent_bank

        self._latent_bank = build_train_latent_bank(
            self._Z_train,
            self._M_train,
            bank_max_points=self.bank_max_points,
            seed=42,
        )
        print(
            f"[crossNN] latent bank size={self._latent_bank.shape[0]} "
            f"(max_points={self.bank_max_points})",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        M_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        M_val: Optional[np.ndarray] = None,
    ) -> "RiemannianImputer":
        """Train Stage I and Stage II on a dataset.

        Args:
            X_train: (N, T, D) float32 training samples.
            M_train: (N, T, D) int8/bool mask (1=observed).
            X_val:   (N_v, T, D) float32 validation samples (for early stop).
            M_val:   (N_v, T, D) mask.

        Returns:
            self
        """
        N, T, D = X_train.shape
        device = get_device()
        self._device = device
        logger.info(f"Device: {device}")

        # --- Stage I: embed a representative reference sample ---
        # Synthesise reference by averaging the top-K most-observed samples.
        # This gives a smoother, more representative temporal pattern for the
        # manifold chart, especially important for heavily missing datasets.
        obs_frac = M_train.astype(float).mean(axis=(1, 2))
        top_k_ref = min(5, N)
        top_k_idx = np.argsort(obs_frac)[-top_k_ref:]
        # Average the top-K samples; use the combined mask (any-observed)
        X_ref = X_train[top_k_idx].mean(axis=0)      # (T, D) average
        M_ref = (M_train[top_k_idx].sum(axis=0) > 0).astype(np.int8)  # union mask

        logger.info("Fitting Stage I (manifold embedder)...")
        if self.ablation_no_embed:
            # Ablation: skip manifold embedding, use raw sklearn PCA only
            from sklearn.decomposition import PCA as _PCA
            logger.info("  [ablation_no_embed] Using plain PCA for Stage I.")
            pca = _PCA(n_components=min(self.d, D - 1, T - 1))
            X_flat = X_ref.copy()
            X_flat[np.isnan(X_flat)] = 0.0
            pca.fit(X_flat)
            self._ablation_pca = pca
            self._embedder = None
            d_actual = pca.n_components_
        else:
            lambda_I_eff = 0.0 if self.ablation_no_hinge else self.lambda_I
            W_search_eff = T if self.ablation_no_window_knn else self.W_search
            graph_alpha_eff = 0.0 if self.ablation_no_mask_dist else 0.5
            embedder = MaskedManifoldEmbedder(
                d=self.d,
                k=self.k,
                W_search=W_search_eff,
                k_clle=self.k_clle,
                r_L=self.r_L,
                lambda_I=lambda_I_eff,
                lambda_norm=self.lambda_norm,
                max_epochs=self.embed_epochs,
                graph_alpha=graph_alpha_eff,
                d_pca=self.d_pca,
                hold_blend_alpha=self.hold_blend_alpha,
            )
            embedder.fit(X_ref, M_ref)
            self._embedder = embedder
        if not self.ablation_no_embed:
            # Sync actual d (embedder may reduce d if n_L was too small)
            d_actual = embedder.d
        logger.info(f"Actual embedding dim d={d_actual}")

        # Estimate tau from Stage I step norms
        if self.ablation_no_embed:
            X_ref_clean = X_ref.copy(); X_ref_clean[np.isnan(X_ref_clean)] = 0.0
            Z_ref = self._ablation_pca.transform(X_ref_clean)
        else:
            Z_ref = embedder.Z
        steps = np.linalg.norm(np.diff(Z_ref, axis=0), axis=1)
        self._tau = float(np.median(steps)) if len(steps) > 0 else 1.0
        self._tau = max(self._tau, 1e-3)

        # --- Stage II: train metric net (or skip if ablation_no_metric) ---
        lambda_geo_eff = 0.0 if self.ablation_no_geo_reg else self.lambda_geo

        if self.ablation_metric_backbone == "mlp":
            from .metric import MLPMetricNet
            metric_net = MLPMetricNet(
                d=d_actual, hidden=self.d_ff, n_layers=3, lambda_geo=lambda_geo_eff,
            ).to(device)
        elif self.ablation_metric_backbone == "lstm":
            from .metric import LSTMMetricNet
            metric_net = LSTMMetricNet(
                d=d_actual, hidden=self.d_ff // 2, n_layers=2, lambda_geo=lambda_geo_eff,
            ).to(device)
        else:
            metric_net = SPDMetricNet(
                d=d_actual,
                n_head=self.n_head,
                n_layers=self.L_enc,
                d_ff=self.d_ff,
                lambda_geo=lambda_geo_eff,
            ).to(device)
        self._metric_net = metric_net

        self._val_history = []

        if self.ablation_no_metric:
            logger.info("  [ablation_no_metric] Skipping Stage II training.")
            self._metric_net = None
        else:
            optimiser = torch.optim.Adam(metric_net.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=self.epochs, eta_min=self.lr * 0.1
            )

        # Pre-embed all training samples (needed for Stage III decoder even if no_metric)
        logger.info("Embedding training samples for Stage II...")
        Z_train = self._embed_batch(X_train, M_train, desc="Embed train")   # (N, T, d)

        Z_val = None
        if X_val is not None and M_val is not None:
            Z_val = self._embed_batch(X_val, M_val, desc="Embed val")

        if not self.ablation_no_metric:
            best_val_loss = float("inf")
            no_improve = 0
            tau_t = torch.tensor(self._tau, dtype=torch.float32)

        logger.info(f"Training Stage II for up to {self.epochs} epochs...")
        epoch_iter = (
            _tqdm(range(self.epochs), desc="Stage II", unit="ep", dynamic_ncols=True)
            if _HAS_TQDM else range(self.epochs)
        ) if not self.ablation_no_metric else []
        for epoch in epoch_iter:
            metric_net.train()
            perm = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches = 0

            for b_start in range(0, N, self.batch_size):
                idx = perm[b_start: b_start + self.batch_size]
                Z_batch = torch.from_numpy(Z_train[idx]).to(device)  # (B, T, d)

                if self.stage2_v2:
                    batch_seed = int(epoch * 1_000_003 + b_start)
                    Z_hat, miss_t = self._stage2_v2_forward_batch(
                        metric_net, Z_batch, batch_seed
                    )
                    G = metric_net(Z_hat.detach())
                    l_g = loss_geo_masked(Z_hat, Z_batch, G, miss_t)
                    l_a = loss_aniso(G)
                else:
                    synth_mask = (
                        torch.rand_like(Z_batch[:, :, 0]) > self.synth_miss_rate
                    ).float().unsqueeze(-1)
                    Z_obs = Z_batch * synth_mask
                    Z_hat = Z_obs
                    G = metric_net(Z_hat)
                    l_g = loss_geo(Z_hat, Z_batch, G)
                    l_a = Z_hat.new_tensor(0.0)

                l_r = loss_reg(G)
                l_s = loss_smooth(G)
                tau_t_device = tau_t.to(Z_hat.device)
                l_t = loss_time(Z_hat, tau_t_device.item())
                total = (
                    l_g
                    + self.lambda_aniso * l_a
                    + self.lambda_F * l_r
                    + self.lambda_smooth * l_s
                    + self.lambda_I * l_t
                )

                optimiser.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(metric_net.parameters(), 1.0)
                optimiser.step()

                epoch_loss += total.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation
            if Z_val is not None:
                val_loss = self._val_loss(metric_net, Z_val, device)
                self._val_history.append(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    no_improve = 0
                    self._best_state = {
                        k: v.cpu().clone() for k, v in metric_net.state_dict().items()
                    }
                else:
                    no_improve += 1
                if _HAS_TQDM and hasattr(epoch_iter, 'set_postfix'):
                    epoch_iter.set_postfix(
                        train=f"{avg_loss:.3f}", val=f"{val_loss:.3f}",
                        best=f"{best_val_loss:.3f}", patience=f"{no_improve}/{self.patience}"
                    )
                if no_improve >= self.patience:
                    logger.info(f"Early stop at epoch {epoch+1} (val={val_loss:.4f} best={best_val_loss:.4f})")
                    if _HAS_TQDM and hasattr(epoch_iter, 'close'):
                        epoch_iter.close()
                    break
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Ep{epoch+1}: train={avg_loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f} patience={no_improve}")
            else:
                if _HAS_TQDM and hasattr(epoch_iter, 'set_postfix'):
                    epoch_iter.set_postfix(train=f"{avg_loss:.3f}")
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Ep{epoch+1}: train={avg_loss:.4f}")

        # Restore best weights if val was used
        if not self.ablation_no_metric:
            if Z_val is not None and hasattr(self, "_best_state"):
                metric_net.load_state_dict({
                    k: v.to(device) for k, v in self._best_state.items()
                })
            metric_net.eval()

        # Store training data for Stage III k-NN anchoring
        self._Z_train = Z_train           # (N, T, d)
        self._X_train = X_train.astype(np.float32)
        self._M_train = M_train.astype(np.int8)
        # Observation rate determines Stage III strategy:
        # Dense (obs > 0.4): CLLE inverse (accurate per-sample PCA)
        # Sparse (obs ≤ 0.4): global k-NN (training anchors more reliable)
        self._obs_rate = float(M_train.mean())

        # --- Stage III (optional): train DecoderMLP ---
        if self.ablation_use_decoder:
            logger.info("  [ablation_use_decoder] Fitting DecoderMLP for Stage III.")
            d_in = Z_train.shape[-1]
            d_out = X_train.shape[-1]
            self._fit_decoder(Z_train, X_train, M_train, d_in, d_out, device)
        else:
            self._decoder = None

        # ---tage A: G-field diagnostic ---
        if self._metric_net is not None:
            sample_n = min(8, Z_train.shape[0])
            Z_sample_diag = torch.from_numpy(Z_train[:sample_n].astype(np.float32))
            try:
                G_diag = diagnose_G_field(self._metric_net, Z_sample_diag, device)
                # Use print so Gate A verification is visible regardless of
                # the caller's logging configuration.
                print(f"[ABL-004 diagnostic] G diagnostic: {G_diag}", flush=True)
                logger.info(f"G diagnostic: {G_diag}")
            except Exception as e:
                print(f"[ABL-004 diagnostic] G diagnostic failed: {e}", flush=True)
                logger.warning(f"G diagnostic failed: {e}")

        self._is_fitted = True
        logger.info("Fitting complete.")
        return self

    def retrain_stage2_v2(
        self,
        X_train: np.ndarray,
        M_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        M_val: Optional[np.ndarray] = None,
    ) -> "RiemannianImputer":
        """Re-train Stage II only (REVISE-004): frozen embedder, fresh metric, stage2_v2 loop."""
        if self.ablation_no_embed:
            raise RuntimeError("retrain_stage2_v2 requires manifold embedder (not ablation_no_embed).")
        if self._embedder is None:
            raise RuntimeError("retrain_stage2_v2: load a fitted imputer with embedder first.")
        if self.ablation_no_metric:
            raise RuntimeError("retrain_stage2_v2: ablation_no_metric must be False.")

        self.stage2_v2 = True
        N, T, D = X_train.shape
        device = get_device()
        self._device = device
        d_actual = self._embedder.d
        logger.info(
            f"retrain_stage2_v2: N={N} d={d_actual} epochs={self.epochs} "
            f"lambda_aniso={self.lambda_aniso}"
        )

        lambda_geo_eff = 0.0 if self.ablation_no_geo_reg else self.lambda_geo
        if self.ablation_metric_backbone == "mlp":
            from .metric import MLPMetricNet

            metric_net = MLPMetricNet(
                d=d_actual, hidden=self.d_ff, n_layers=3, lambda_geo=lambda_geo_eff,
            ).to(device)
        elif self.ablation_metric_backbone == "lstm":
            from .metric import LSTMMetricNet

            metric_net = LSTMMetricNet(
                d=d_actual, hidden=self.d_ff // 2, n_layers=2, lambda_geo=lambda_geo_eff,
            ).to(device)
        else:
            metric_net = SPDMetricNet(
                d=d_actual,
                n_head=self.n_head,
                n_layers=self.L_enc,
                d_ff=self.d_ff,
                lambda_geo=lambda_geo_eff,
            ).to(device)
        self._metric_net = metric_net

        optimiser = torch.optim.Adam(metric_net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.epochs, eta_min=self.lr * 0.1
        )

        logger.info("Embedding training samples (frozen Stage I)...")
        Z_train = self._embed_batch(X_train, M_train, desc="Embed train")
        Z_val = None
        if X_val is not None and M_val is not None:
            Z_val = self._embed_batch(X_val, M_val, desc="Embed val")

        best_val_loss = float("inf")
        no_improve = 0
        tau_t = torch.tensor(self._tau, dtype=torch.float32)
        self._val_history = []

        logger.info(f"Training Stage II v2 for up to {self.epochs} epochs...")
        epoch_iter = (
            _tqdm(range(self.epochs), desc="Stage II v2", unit="ep", dynamic_ncols=True)
            if _HAS_TQDM
            else range(self.epochs)
        )
        for epoch in epoch_iter:
            metric_net.train()
            perm = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches = 0

            for b_start in range(0, N, self.batch_size):
                idx = perm[b_start : b_start + self.batch_size]
                Z_batch = torch.from_numpy(Z_train[idx]).to(device)

                batch_seed = int(epoch * 1_000_003 + b_start)
                Z_hat, miss_t = self._stage2_v2_forward_batch(
                    metric_net, Z_batch, batch_seed
                )
                G = metric_net(Z_hat.detach())
                l_g = loss_geo_masked(Z_hat, Z_batch, G, miss_t)
                l_a = loss_aniso(G)
                l_r = loss_reg(G)
                l_s = loss_smooth(G)
                tau_t_device = tau_t.to(Z_hat.device)
                l_t = loss_time(Z_hat, tau_t_device.item())
                total = (
                    l_g
                    + self.lambda_aniso * l_a
                    + self.lambda_F * l_r
                    + self.lambda_smooth * l_s
                    + self.lambda_I * l_t
                )

                optimiser.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(metric_net.parameters(), 1.0)
                optimiser.step()

                epoch_loss += total.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            if Z_val is not None:
                val_loss = self._val_loss(metric_net, Z_val, device)
                self._val_history.append(val_loss)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    no_improve = 0
                    self._best_state = {
                        k: v.cpu().clone() for k, v in metric_net.state_dict().items()
                    }
                else:
                    no_improve += 1
                if _HAS_TQDM and hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix(
                        train=f"{avg_loss:.3f}",
                        val=f"{val_loss:.3f}",
                        best=f"{best_val_loss:.3f}",
                        patience=f"{no_improve}/{self.patience}",
                    )
                if no_improve >= self.patience:
                    logger.info(
                        f"Early stop at epoch {epoch+1} "
                        f"(val={val_loss:.4f} best={best_val_loss:.4f})"
                    )
                    if _HAS_TQDM and hasattr(epoch_iter, "close"):
                        epoch_iter.close()
                    break
                if (epoch + 1) % 20 == 0:
                    logger.info(
                        f"Ep{epoch+1}: train={avg_loss:.4f} val={val_loss:.4f} "
                        f"best={best_val_loss:.4f} patience={no_improve}"
                    )
            else:
                if _HAS_TQDM and hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix(train=f"{avg_loss:.3f}")
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Ep{epoch+1}: train={avg_loss:.4f}")

        if Z_val is not None and hasattr(self, "_best_state"):
            metric_net.load_state_dict(
                {k: v.to(device) for k, v in self._best_state.items()}
            )
        metric_net.eval()

        self._Z_train = Z_train
        self._X_train = X_train.astype(np.float32)
        self._M_train = M_train.astype(np.int8)
        self._obs_rate = float(M_train.mean())
        self._latent_bank = None
        if self._effective_cross_k() > 0:
            self._ensure_latent_bank()

        sample_n = min(8, Z_train.shape[0])
        Z_sample_diag = torch.from_numpy(Z_train[:sample_n].astype(np.float32))
        try:
            G_diag = diagnose_G_field(self._metric_net, Z_sample_diag, device)
            print(f"[stage2_v2 retrain] G diagnostic: {G_diag}", flush=True)
        except Exception as e:
            print(f"[stage2_v2 retrain] G diagnostic failed: {e}", flush=True)

        self._is_fitted = True
        logger.info("retrain_stage2_v2 complete.")
        return self

    def retrain_stage2_realign(
        self,
        X_train: np.ndarray,
        M_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        M_val: Optional[np.ndarray] = None,
        realign_cfg: Optional[object] = None,
    ) -> "RiemannianImputer":
        """Re-train Stage II with embedder-pullback supervision (CONTRAST-REALIGN)."""
        from .stage2_realign import RealignConfig, retrain_stage2_realign

        cfg = realign_cfg if realign_cfg is not None else RealignConfig(
            lambda_aniso=max(self.lambda_aniso, 0.05)
        )
        info = retrain_stage2_realign(
            self, X_train, M_train, X_val, M_val, cfg=cfg
        )
        self._realign_info = info
        logger.info("retrain_stage2_realign complete: %s", info)
        return self

    def _fit_decoder(
        self,
        Z_train: np.ndarray,
        X_train: np.ndarray,
        M_train: np.ndarray,
        d_in: int,
        d_out: int,
        device: torch.device,
    ) -> None:
        """Train DecoderMLP on observed (z_t, x_t) pairs from training set."""
        N, T, D = X_train.shape

        # Collect pairs where any feature is observed at that timestep
        obs_ts = M_train.any(axis=2)  # (N, T) bool: at least one feature observed
        obs_idx = np.argwhere(obs_ts)  # (K, 2) each row is (n, t)

        Z_pairs = Z_train[obs_idx[:, 0], obs_idx[:, 1], :]   # (K, d)
        X_pairs = X_train[obs_idx[:, 0], obs_idx[:, 1], :]   # (K, D)
        M_pairs = M_train[obs_idx[:, 0], obs_idx[:, 1], :]   # (K, D) per-feature obs

        logger.info(f"DecoderMLP: {len(Z_pairs):,} observed (z,x) pairs for training.")

        decoder = DecoderMLP(d_in, d_out, hidden=self.decoder_hidden).to(device)
        self._decoder = decoder
        opt_dec = torch.optim.Adam(decoder.parameters(), lr=self.decoder_lr)
        sched_dec = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_dec, T_max=self.decoder_epochs, eta_min=self.decoder_lr * 0.05
        )

        K = len(Z_pairs)
        bs = min(512, K)
        split = int(0.9 * K)
        perm = np.random.permutation(K)
        tr_idx, vl_idx = perm[:split], perm[split:]

        best_vl, best_state, no_imp = float("inf"), None, 0
        dec_iter = (
            _tqdm(range(self.decoder_epochs), desc="Decoder", unit="ep", dynamic_ncols=True)
            if _HAS_TQDM else range(self.decoder_epochs)
        )
        for ep in dec_iter:
            decoder.train()
            p2 = np.random.permutation(split)
            ep_loss = 0.0; nb = 0
            for b0 in range(0, split, bs):
                bi = tr_idx[p2[b0: b0 + bs]]
                z_b = torch.from_numpy(Z_pairs[bi]).to(device)
                x_b = torch.from_numpy(X_pairs[bi]).to(device)
                m_b = torch.from_numpy(M_pairs[bi].astype(np.float32)).to(device)
                x_hat = decoder(z_b)
                loss = ((x_hat - x_b) ** 2 * m_b).sum() / m_b.sum().clamp(min=1)
                opt_dec.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                opt_dec.step()
                ep_loss += loss.item(); nb += 1
            sched_dec.step()
            ep_loss /= max(nb, 1)

            # Validation
            decoder.eval()
            with torch.no_grad():
                z_v = torch.from_numpy(Z_pairs[vl_idx]).to(device)
                x_v = torch.from_numpy(X_pairs[vl_idx]).to(device)
                m_v = torch.from_numpy(M_pairs[vl_idx].astype(np.float32)).to(device)
                x_vhat = decoder(z_v)
                vl_loss = (((x_vhat - x_v) ** 2 * m_v).sum() / m_v.sum().clamp(min=1)).item()

            if _HAS_TQDM and hasattr(dec_iter, 'set_postfix'):
                dec_iter.set_postfix(
                    train=f"{ep_loss:.4f}", val=f"{vl_loss:.4f}",
                    best=f"{best_vl:.4f}", pat=f"{no_imp}/{self.decoder_patience}"
                )

            if vl_loss < best_vl - 1e-6:
                best_vl = vl_loss
                best_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= self.decoder_patience:
                logger.info(f"Decoder early stop at ep {ep+1} (val={vl_loss:.5f})")
                if _HAS_TQDM and hasattr(dec_iter, 'close'):
                    dec_iter.close()
                break

        if best_state is not None:
            decoder.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        decoder.eval()
        logger.info(f"Decoder fitted (best val MSE={best_vl:.5f})")

    def _karcher_train_kwargs(self) -> dict:
        """Karcher kwargs aligned with inference consumer-path fixes."""
        use_fix = self.consumer_karcher_fix or self.stage2_v2
        norm_g = use_fix or self.force_normalize_g
        return dict(
            k_time=self.k_time,
            sigma=self.sigma,
            N_max=self.N_max,
            eta=self.eta,
            use_clle_init=self.karcher_use_clle_init,
            normalize_g=norm_g,
            clip_step_norm=1.0 if use_fix else None,
            adaptive_eta=use_fix,
            g_scale=float(getattr(self, "g_karcher_scale", 1.0)),
            eigen_clip=getattr(self, "g_eigen_clip", None),
        )

    @staticmethod
    def _g_active_mask_from_hold(
        miss_mask_T: np.ndarray, T: int, pad: int
    ) -> np.ndarray:
        """True on hold block timesteps ± pad (Phase 2.6 I-A3)."""
        active = np.zeros(T, dtype=bool)
        hold_idx = np.where(miss_mask_T)[0]
        if len(hold_idx) == 0:
            return active
        lo = max(0, int(hold_idx.min()) - pad)
        hi = min(T - 1, int(hold_idx.max()) + pad)
        active[lo : hi + 1] = True
        return active

    def _stage2_v2_forward_batch(
        self,
        metric_net: SPDMetricNet,
        Z_batch: torch.Tensor,
        batch_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Block-mask → Karcher impute → metric field (REVISE-004)."""
        B, T, _d = Z_batch.shape
        miss_np = sample_block_mask(B, T, batch_seed)
        Z_np = Z_batch.detach().cpu().numpy()
        Z_hat_np = Z_np.copy()
        kw = self._karcher_train_kwargs()

        was_training = metric_net.training
        metric_net.eval()
        for b in range(B):
            I_miss = np.where(miss_np[b])[0]
            if len(I_miss) == 0:
                continue
            Z_obs = Z_np[b].copy()
            bank = self._latent_bank if self._effective_cross_k() > 0 else None
            Z_hat_np[b] = karcher_impute(
                Z_obs,
                metric_net,
                I_miss,
                Z_bank=bank,
                cross_k=self._effective_cross_k(),
                cross_sigma=self.cross_sigma,
                **kw,
            )
        if was_training:
            metric_net.train()

        Z_hat = torch.from_numpy(Z_hat_np).to(Z_batch.device)
        miss_t = torch.from_numpy(miss_np).to(Z_batch.device)
        return Z_hat, miss_t

    def _val_loss(
        self,
        metric_net: SPDMetricNet,
        Z_val: np.ndarray,
        device: torch.device,
    ) -> float:
        metric_net.eval()
        total = 0.0
        n_batches = 0
        with torch.no_grad():
            for b_start in range(0, len(Z_val), self.batch_size):
                Z_b = torch.from_numpy(Z_val[b_start: b_start + self.batch_size]).to(device)
                if self.stage2_v2:
                    Z_hat, miss_t = self._stage2_v2_forward_batch(
                        metric_net, Z_b, batch_seed=b_start
                    )
                    G = metric_net(Z_hat)
                    l_g = loss_geo_masked(Z_hat, Z_b, G, miss_t)
                    l_a = loss_aniso(G)
                else:
                    synth_mask = (
                        torch.rand_like(Z_b[:, :, 0]) > self.synth_miss_rate
                    ).float().unsqueeze(-1)
                    Z_hat = Z_b * synth_mask
                    G = metric_net(Z_hat)
                    l_g = loss_geo(Z_hat, Z_b, G)
                    l_a = Z_b.new_tensor(0.0)
                l_r = loss_reg(G)
                l_s = loss_smooth(G)
                l_t = loss_time(Z_hat, self._tau)
                total += (
                    l_g
                    + self.lambda_aniso * l_a
                    + self.lambda_F * l_r
                    + self.lambda_smooth * l_s
                    + self.lambda_I * l_t
                ).item()
                n_batches += 1
        metric_net.train()
        return total / max(n_batches, 1)

    def _embed_batch(self, X: np.ndarray, M: np.ndarray, desc: str = "Embedding") -> np.ndarray:
        """Embed a batch (N, T, D) → (N, T, d) using the fitted embedder."""
        N, T, D = X.shape
        if self.ablation_no_embed and self._ablation_pca is not None:
            d = self._ablation_pca.n_components_
            Z_all = np.zeros((N, T, d), dtype=np.float32)
            for n in range(N):
                x_clean = X[n].copy(); x_clean[np.isnan(x_clean)] = 0.0
                Z_all[n] = self._ablation_pca.transform(x_clean)
            return Z_all
        d = self._embedder.d
        Z_all = np.zeros((N, T, d), dtype=np.float32)
        it = (
            _tqdm(range(N), desc=desc, unit="sample", dynamic_ncols=True, leave=False)
            if _HAS_TQDM else range(N)
        )
        for n in it:
            Z_all[n], _ = self._embedder.transform(X[n], M[n])
        return Z_all

    def _knn_ambient_inverse(
        self,
        Z_prime: np.ndarray,   # (T, d) Karcher-imputed latent codes
        x_n: np.ndarray,       # (T, D) current sample
        m_n: np.ndarray,       # (T, D) current mask
        k_anchor: int = 15,
    ) -> np.ndarray:
        """Stage III via global k-NN ambient reconstruction.

        Finds the k globally nearest training samples based on Frobenius
        distance of full latent trajectory Z_prime vs Z_train[j].
        For each timestep t, reconstructs ambient via distance-weighted average
        of training observed values.

        Advantages over CLLE inverse for sparse data (e.g. PhysioNet 80% miss):
        - No per-sample PCA noise (doesn't rely on mean-filled landmark coords)
        - Global trajectory similarity captures patient-level patterns
        - Directly uses observed training ambient values
        """
        T, d = Z_prime.shape
        D = self._X_train.shape[2]
        N_tr = self._Z_train.shape[0]

        # Distance only over OBSERVED timesteps (exclude Karcher-imputed ones)
        # obs_mask_T: (T,) bool — which timesteps the current sample has any obs
        obs_mask_T = m_n.any(axis=1)  # (T,) bool
        if not obs_mask_T.any():
            obs_mask_T = np.ones(T, dtype=bool)  # fallback: all timesteps

        Z_obs = Z_prime[obs_mask_T, :]                       # (T_obs, d)
        Z_tr_obs = self._Z_train[:, obs_mask_T, :]           # (N, T_obs, d)
        diff = Z_tr_obs - Z_obs[None, :, :]                  # (N, T_obs, d)
        dists = (diff ** 2).sum(axis=(1, 2))                 # (N,)

        # k nearest training samples
        k = min(k_anchor, N_tr)
        nn_idx = np.argsort(dists)[:k]
        w = 1.0 / (dists[nn_idx] + 1e-8)           # (k,)
        w /= w.sum()

        # For each timestep t, reconstruct from observed neighbors
        hat_X = np.zeros((T, D), dtype=np.float32)
        for t in range(T):
            x_nn = self._X_train[nn_idx, t, :]      # (k, D)
            m_nn = self._M_train[nn_idx, t, :]      # (k, D)
            for feat_j in range(D):
                obs = m_nn[:, feat_j].astype(bool)
                if obs.any():
                    w_obs = w[obs] / w[obs].sum()
                    hat_X[t, feat_j] = float((w_obs * x_nn[obs, feat_j]).sum())
                # else: leave 0 (will be overridden if observed in x_n)

        # Freeze observed entries
        M_f = m_n.astype(np.float32)
        X_safe = np.where(np.isnan(x_n), 0.0, x_n).astype(np.float32)
        hat_X = M_f * X_safe + (1.0 - M_f) * hat_X
        return hat_X

    def set_trace(
        self,
        tracer: Optional[PipelineTracer],
        held_block: Optional[np.ndarray] = None,
        x_gt: Optional[np.ndarray] = None,
        m_native: Optional[np.ndarray] = None,
    ) -> None:
        """Enable per-sample JSONL trace (inference only)."""
        self._tracer = tracer
        self._trace_held_block = held_block
        self._trace_x_gt = x_gt
        self._trace_m_native = m_native

    def clear_trace(self) -> None:
        self._tracer = None
        self._trace_held_block = None
        self._trace_x_gt = None
        self._trace_m_native = None

    def _need_z_gt(self) -> bool:
        return (
            self._tracer is not None
            or self.z_n_oracle_alpha > 0.0
            or self.z_prime_oracle_alpha > 0.0
            or self.z_hold_oracle_alpha > 0.0
        )

    def _embed_z_gt(
        self,
        n: int,
        x_n: np.ndarray,
        m_n: np.ndarray,
        miss_mask_T: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self._embedder is None or self.ablation_no_embed:
            return None
        x_gt_n = x_n
        m_gt_n = m_n
        if self._trace_x_gt is not None:
            x_gt_n = self._trace_x_gt[n]
        if self._trace_m_native is not None:
            m_gt_n = self._trace_m_native[n]
        hold_t = miss_mask_T if getattr(self, "hold_aware_stage1", False) else None
        Z_gt, _ = self._embedder.transform(x_gt_n, m_gt_n, hold_t=hold_t)
        return Z_gt

    def _hold_metric_fro_stats(
        self,
        Z_seq: np.ndarray,
        hold_ts: np.ndarray,
        eps_pb: float,
        pullback_at_t: Optional[Callable[[int], np.ndarray]],
    ) -> tuple[float, float, float]:
        """Mean ||G_net-I||_F, ||G_pb-G_net||_F, ||G_eff-I||_F @ hold."""
        if (
            self._metric_net is None
            or len(hold_ts) == 0
            or self.ablation_no_metric
        ):
            return 0.0, 0.0, 0.0
        d = Z_seq.shape[1]
        I_d = np.eye(d, dtype=np.float32)
        self._metric_net.eval()
        fro_net: list[float] = []
        fro_pb: list[float] = []
        fro_eff: list[float] = []
        with torch.no_grad():
            Z_pt = (
                torch.from_numpy(Z_seq.astype(np.float32))
                .unsqueeze(0)
                .to(self._device)
            )
            G_all = self._metric_net(Z_pt)[0].cpu().numpy()
        for t in hold_ts:
            ti = int(t)
            G_net = G_all[ti]
            fro_net.append(float(np.linalg.norm(G_net - I_d, ord="fro")))
            G_pb = None
            if eps_pb > 0.0 and pullback_at_t is not None:
                G_pb = pullback_at_t(ti).astype(np.float32)
                fro_pb.append(float(np.linalg.norm(G_pb - G_net, ord="fro")))
            mode = str(getattr(self, "karcher_metric_mode", "learned"))
            if mode == "identity":
                G_eff = I_d
            elif mode == "pullback" and G_pb is not None:
                G_eff = G_pb
            elif G_pb is not None and eps_pb > 0.0:
                G_eff = (1.0 - eps_pb) * G_net + eps_pb * G_pb
            else:
                G_eff = G_net
            fro_eff.append(float(np.linalg.norm(G_eff - I_d, ord="fro")))
        return (
            float(np.mean(fro_net)) if fro_net else 0.0,
            float(np.mean(fro_pb)) if fro_pb else 0.0,
            float(np.mean(fro_eff)) if fro_eff else 0.0,
        )

    # ------------------------------------------------------------------
    # Impute
    # ------------------------------------------------------------------

    def impute(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Impute missing values for a batch (N, T, D).

        Args:
            X: (N, T, D) float32 (NaN for missing or any value — M takes precedence).
            M: (N, T, D) int8/bool mask.

        Returns:
            hat_X: (N, T, D) float32 with observed entries identical to X.
        """
        assert self._is_fitted, "Call fit() first."
        N, T, D = X.shape
        hat_X = np.zeros_like(X, dtype=np.float32)

        # ---tage A: reset Karcher counters ---
        reset_karcher_stats()
        self._latent_bank = None
        cross_k_eff = self._effective_cross_k()
        if cross_k_eff > 0:
            self._ensure_latent_bank()
        Z_bank = self._latent_bank

        it = (
            _tqdm(range(N), desc="Imputing", unit="sample", dynamic_ncols=True)
            if _HAS_TQDM else range(N)
        )
        for n in it:
            x_n = X[n]         # (T, D)
            m_n = M[n]         # (T, D)

            tau_obs = max(1, D // 10)
            miss_mask_T = (m_n.sum(axis=1) < tau_obs).astype(bool)

            # Stage I — returns embedding + per-sample PCA projection
            if self.ablation_no_embed and self._ablation_pca is not None:
                x_clean = x_n.copy(); x_clean[np.isnan(x_clean)] = 0.0
                Z_n = self._ablation_pca.transform(x_clean)   # (T, d)
                X_pca_n = Z_n
            else:
                hold_t = miss_mask_T if getattr(self, "hold_aware_stage1", False) else None
                Z_n, X_pca_n = self._embedder.transform(x_n, m_n, hold_t=hold_t)

            # Stage II: find time steps that need Karcher reconstruction.
            I_miss = np.where(miss_mask_T)[0]

            k_time_eff = max(self.k_time, 20) if getattr(
                self, "stage2_realign", False
            ) else self.k_time

            kw = self._karcher_train_kwargs()
            kw["k_time"] = k_time_eff
            if getattr(self, "g_apply_mode", "global") == "hold_neighborhood":
                kw["g_active_mask"] = self._g_active_mask_from_hold(
                    miss_mask_T, T, int(getattr(self, "g_neighborhood_pad", 10))
                )

            eps_pb = float(getattr(self, "pullback_blend_epsilon", 0.0))
            _pullback_at_t: Optional[Callable[[int], np.ndarray]] = None
            if (
                eps_pb > 0.0
                and self._embedder is not None
                and not self.ablation_no_embed
            ):
                try:
                    from experiments.embedder_pullback import embedder_pullback_at
                except ImportError as exc:
                    raise NotImplementedError(
                        "pullback_blend_epsilon>0 needs the embedder-pullback ablation "
                        "module, which is not part of this release. "
                        "The deployed path uses the default pullback_blend_epsilon=0.0."
                    ) from exc

                _pb_cache: dict[int, np.ndarray] = {}

                def _pullback_at_t(t_idx: int) -> np.ndarray:
                    ti = int(t_idx)
                    if ti not in _pb_cache:
                        _pb_cache[ti] = embedder_pullback_at(
                            self._embedder, x_n, m_n, ti
                        )
                    return _pb_cache[ti]

                kw["pullback_blend_epsilon"] = eps_pb
                kw["pullback_at_t"] = _pullback_at_t

            kw["karcher_metric_mode"] = str(
                getattr(self, "karcher_metric_mode", "learned")
            )

            Z_gt: Optional[np.ndarray] = None
            if self._need_z_gt():
                Z_gt = self._embed_z_gt(n, x_n, m_n, miss_mask_T)

            I_hold = I_miss.astype(int)
            if Z_gt is not None and self.z_n_oracle_alpha > 0.0 and len(I_hold) > 0:
                alpha_n = float(self.z_n_oracle_alpha)
                for t_idx in I_hold:
                    ti = int(t_idx)
                    Z_n[ti] = (
                        (1.0 - alpha_n) * Z_n[ti] + alpha_n * Z_gt[ti]
                    ).astype(np.float32)

            kw["z_hold_oracle_alpha"] = float(getattr(self, "z_hold_oracle_alpha", 0.0))
            if kw["z_hold_oracle_alpha"] > 0.0 and Z_gt is not None:
                kw["z_oracle_Z_gt"] = Z_gt

            Z_k_raw: Optional[np.ndarray] = None
            if (
                len(I_miss) > 0
                and self._metric_net is not None
                and not self.ablation_no_metric
                and not self.ablation_no_karcher_g
            ):
                self._metric_net.eval()
                reset_karcher_detail()
                if str(getattr(self, "stage2_method", "mhb")) == "mhb":
                    # — paper Algorithm 1 path.
                    # Graph A = Stage I mask-weighted time-window k-NN graph
                    # (eqs. (5)-(7)) built on THIS sample's low-d PCA coords,
                    # mirroring MaskedManifoldEmbedder.fit() exactly.
                    mp = dict(getattr(self, "mhb_params", None) or {})
                    if self._embedder is not None and not self.ablation_no_embed:
                        X_graph = X_pca_n[:, : self._embedder.d]
                        g_k = int(mp.get("k_graph", self._embedder.k))
                        g_W = int(self._embedder.W_search)
                        g_alpha = float(self._embedder.graph_alpha)
                    else:
                        X_graph = X_pca_n
                        g_k = int(mp.get("k_graph", self.k))
                        g_W = int(self.W_search)
                        g_alpha = 0.5
                    A_n = build_time_window_graph(
                        X_graph, m_n, k=g_k, W_search=g_W, alpha=g_alpha
                    )
                    Z_k_raw = mhb_impute(
                        Z_n,
                        self._metric_net,
                        I_miss,
                        A_n,
                        K=int(mp.get("K", 5)),
                        gamma=float(mp.get("gamma", 1e-3)),
                        mode=str(mp.get("mode", "soft")),
                        # Soft-spline per-step weights w_t (paper eq. graph-spline).
                        # Canonical configuration is w_t == 1; we pass the per-row
                        # observation fraction (floored at 0.05 in mhb_spline_extend)
                        # as a benign reliability prior. The two are numerically
                        # equivalent on the deployed grid (|dMAE| < 1e-4, measured on
                        # hydraulic via evaluate.py), so reported numbers match the
                        # w_t==1 claim in the paper/supplement.
                        row_frac=m_n.astype(np.float64).mean(axis=1),
                        spline_gamma=float(mp.get("spline_gamma", 0.02)),
                        k_latent=int(mp.get("k_latent", 10)),
                    )
                else:
                    Z_k_raw = karcher_impute(
                        Z_n,
                        self._metric_net,
                        I_miss,
                        Z_bank=Z_bank,
                        cross_k=cross_k_eff,
                        cross_sigma=self.cross_sigma,
                        **kw,
                    )
                Z_k = Z_k_raw
                lam = float(getattr(self, "karcher_blend_lambda", 1.0))
                if lam < 1.0 - 1e-12:
                    Z_prime = Z_n.copy()
                    for t_idx in I_miss:
                        ti = int(t_idx)
                        Z_prime[ti] = (1.0 - lam) * Z_n[ti] + lam * Z_k[ti]
                else:
                    Z_prime = Z_k
            else:
                Z_prime = Z_n  # ablation_no_metric or no missing entries
                Z_k_raw = None

            if Z_gt is not None and self.z_prime_oracle_alpha > 0.0 and len(I_hold) > 0:
                alpha_p = float(self.z_prime_oracle_alpha)
                for t_idx in I_hold:
                    ti = int(t_idx)
                    Z_prime[ti] = (
                        (1.0 - alpha_p) * Z_prime[ti] + alpha_p * Z_gt[ti]
                    ).astype(np.float32)

            # Stage III: inverse mapping
            if self._decoder is not None:
                # DecoderMLP path (ablation_use_decoder or default decoder)
                with torch.no_grad():
                    z_t = torch.from_numpy(Z_prime.astype(np.float32)).to(self._device)
                    x_hat_t = self._decoder(z_t).cpu().numpy()
                hat_X[n] = x_hat_t
                hat_X[n][m_n == 1] = x_n[m_n == 1]
            elif self.ablation_no_embed and self._ablation_pca is not None:
                # ablation_no_embed: reconstruct via sklearn PCA.inverse_transform.
                # This is the proper "no manifold embedding" comparison — plain PCA
                # instead of the masked-aware manifold chart, not column-mean fallback.
                hat_X_rec = self._ablation_pca.inverse_transform(Z_prime).astype(np.float32)
                hat_X[n] = hat_X_rec
                obs_bool = m_n.astype(bool)
                x_safe = np.where(np.isnan(x_n), 0.0, x_n).astype(np.float32)
                hat_X[n][obs_bool] = x_safe[obs_bool]
            elif self._embedder is None:
                # Fallback: column means (should not reach here under normal ablation)
                hat_X[n] = x_n.copy()
                col_means = np.nanmean(x_n, axis=0)
                col_means = np.where(np.isnan(col_means), 0.0, col_means)
                miss_pos = (m_n == 0)
                hat_X[n][miss_pos] = np.broadcast_to(col_means, x_n.shape)[miss_pos]
                hat_X[n] = np.nan_to_num(hat_X[n], nan=0.0)
            else:
                # Pure CLLE inverse fallback.
                #
                #tage D (fix break ③):
                # Evaluate the learned metric G(z) along Z_prime so that the
                # downstream simplex-weight solve in clle_inverse uses
                # G-Mahalanobis geometry. When ablation_no_metric or the
                # metric net is unavailable, G_field_n stays None and
                # clle_inverse silently falls back to Euclidean weights —
                # critical for keeping the `no_metric` ablation faithful.
                G_field_n = None
                use_metric_simplex = (
                    self._metric_net is not None
                    and not self.ablation_no_metric
                    and not self.ablation_no_simplex_g
                    and not self.scoped_g_simplex
                )
                if use_metric_simplex:
                    self._metric_net.eval()
                    with torch.no_grad():
                        Z_pt = (
                            torch.from_numpy(Z_prime.astype(np.float32))
                            .unsqueeze(0)
                            .to(self._device)
                        )
                        G_t = self._metric_net(Z_pt)            # (1, T, d, d)
                        G_field_n = G_t[0].cpu().numpy()        # (T, d, d)
                if self._tracer is not None:
                    hat_euclid = clle_inverse(
                        Z_prime,
                        self._embedder.id_L,
                        self._embedder.clle_weights,
                        X_pca_n,
                        self._embedder.V,
                        self._embedder.mu,
                        x_n,
                        m_n,
                        G_field=None,
                    )
                    hat_metric = clle_inverse(
                        Z_prime,
                        self._embedder.id_L,
                        self._embedder.clle_weights,
                        X_pca_n,
                        self._embedder.V,
                        self._embedder.mu,
                        x_n,
                        m_n,
                        G_field=G_field_n,
                    )
                    hat_X[n] = hat_metric if use_metric_simplex else hat_euclid
                else:
                    hat_X[n] = clle_inverse(
                        Z_prime,
                        self._embedder.id_L,
                        self._embedder.clle_weights,
                        X_pca_n,
                        self._embedder.V,
                        self._embedder.mu,
                        x_n,
                        m_n,
                        G_field=G_field_n,
                    )

                if self._tracer is not None:
                    hold_eval = None
                    if self._trace_held_block is not None:
                        hold_eval = self._trace_held_block[n].astype(bool)
                    else:
                        hold_eval = (m_n == 0)
                    obs_eval = m_n.astype(bool) & ~hold_eval
                    I_hold_t = hold_timesteps(m_n, tau_obs)
                    z_delta = np.linalg.norm(
                        (Z_prime - Z_n)[I_hold_t], axis=1
                    ) if len(I_hold_t) else np.array([0.0])
                    block_t0 = int(I_hold_t[0]) if len(I_hold_t) else -1
                    ent, mx_w, g_er = collect_hold_simplex_stats(
                        Z_prime,
                        self._embedder.id_L,
                        self._embedder.clle_weights,
                        G_field_n if use_metric_simplex else None,
                        I_hold_t,
                    )
                    z_gt_err = 0.0
                    z_k_pre = 0.0
                    z_k_post = 0.0
                    z_prime_err = 0.0
                    z_k_delta = 0.0
                    x_recon_zgt = 0.0
                    if Z_gt is not None and len(I_hold_t) > 0:
                        z_gt_err = float(
                            np.linalg.norm(Z_n[I_hold_t] - Z_gt[I_hold_t], axis=1).mean()
                        )
                        z_k_pre = z_gt_err
                        if Z_k_raw is not None:
                            z_k_post = float(
                                np.linalg.norm(
                                    Z_k_raw[I_hold_t] - Z_gt[I_hold_t], axis=1
                                ).mean()
                            )
                            z_k_delta = float(
                                np.linalg.norm(
                                    Z_k_raw[I_hold_t] - Z_n[I_hold_t], axis=1
                                ).mean()
                            )
                        z_prime_err = float(
                            np.linalg.norm(
                                Z_prime[I_hold_t] - Z_gt[I_hold_t], axis=1
                            ).mean()
                        )
                        Z_gt_inv = Z_prime.copy()
                        Z_gt_inv[I_hold_t] = Z_gt[I_hold_t]
                        hat_zgt = clle_inverse(
                            Z_gt_inv,
                            self._embedder.id_L,
                            self._embedder.clle_weights,
                            X_pca_n,
                            self._embedder.V,
                            self._embedder.mu,
                            x_n,
                            m_n,
                            G_field=None,
                        )
                        x_gt_ref = (
                            self._trace_x_gt[n]
                            if self._trace_x_gt is not None
                            else x_n
                        )
                        x_recon_zgt = per_dim_mae(hat_zgt, x_gt_ref, hold_eval)
                    pb_fn = _pullback_at_t if eps_pb > 0.0 else None
                    g_net_fro, g_pb_fro, g_i_fro = self._hold_metric_fro_stats(
                        Z_prime, I_hold_t, eps_pb, pb_fn
                    )
                    kd = get_last_karcher_detail()
                    step_norms = kd.get("step_norms", [])
                    k_step_mean = float(np.mean(step_norms)) if step_norms else 0.0
                    stage3_gain = per_dim_mae(hat_euclid, x_n, hold_eval) - per_dim_mae(
                        hat_metric, x_n, hold_eval
                    )
                    if getattr(self, "oracle_x_hold_fill", False):
                        x_fill = (
                            self._trace_x_gt[n]
                            if self._trace_x_gt is not None
                            else x_n
                        )
                        alpha_x = float(getattr(self, "oracle_x_hold_alpha", 1.0))
                        hat_X[n][hold_eval] = (
                            (1.0 - alpha_x) * hat_X[n][hold_eval]
                            + alpha_x * x_fill[hold_eval]
                        )
                    rec = TraceRecord(
                        episode_id=int(n),
                        variant=self._tracer.variant,
                        block_t0=block_t0,
                        n_hold_T=int(len(I_hold_t)),
                        mae_hold=per_dim_mae(hat_X[n], x_n, hold_eval),
                        mae_obs=per_dim_mae(hat_X[n], x_n, obs_eval),
                        s1_z_delta_hold_mean=0.0,
                        s2_z_delta_hold_mean=float(z_delta.mean()) if z_delta.size else 0.0,
                        s3_mae_hold_euclid=per_dim_mae(hat_euclid, x_n, hold_eval),
                        s3_mae_hold_metric=per_dim_mae(hat_metric, x_n, hold_eval),
                        g_eig_ratio_hold_mean=g_er,
                        simplex_entropy_hold_mean=ent,
                        simplex_max_weight_hold_mean=mx_w,
                        karcher_steps=get_karcher_stats().get("n_steps_taken", 0),
                        z_gt_err_hold_mean=z_gt_err,
                        z_k_pre_err=z_k_pre,
                        z_k_post_err=z_k_post,
                        z_prime_err=z_prime_err,
                        z_k_delta=z_k_delta,
                        g_net_fro_hold=g_net_fro,
                        g_pb_fro_hold=g_pb_fro,
                        g_i_fro_hold=g_i_fro,
                        karcher_step_norm_mean=k_step_mean,
                        stage3_gain=stage3_gain,
                        x_recon_from_zgt_mae=x_recon_zgt,
                    )
                    self._tracer.write(rec)
                elif getattr(self, "oracle_x_hold_fill", False):
                    hold_eval = (
                        self._trace_held_block[n].astype(bool)
                        if self._trace_held_block is not None
                        else (m_n == 0)
                    )
                    x_fill = (
                        self._trace_x_gt[n]
                        if self._trace_x_gt is not None
                        else x_n
                    )
                    alpha_x = float(getattr(self, "oracle_x_hold_alpha", 1.0))
                    hat_X[n][hold_eval] = (
                        (1.0 - alpha_x) * hat_X[n][hold_eval]
                        + alpha_x * x_fill[hold_eval]
                    )

        # ---tage A: report Karcher stats ---
        _karcher_stats = get_karcher_stats()
        if not _env_flag("DMHI_SILENT"):
            print(f"[ABL-004 diagnostic] karcher_stats: {_karcher_stats}", flush=True)
        logger.info(f"karcher_stats: {_karcher_stats}")

        return hat_X

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import pickle, pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "RiemannianImputer":
        import io
        import pickle

        import torch

        class _CPUUnpickler(pickle.Unpickler):
            """Load pickles saved on MPS/CUDA into CPU tensors (Linux server compat)."""

            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda b: torch.load(
                        io.BytesIO(b), map_location="cpu", weights_only=False
                    )
                return super().find_class(module, name)

        with open(path, "rb") as f:
            try:
                obj = pickle.load(f)
            except (RuntimeError, pickle.UnpicklingError):
                f.seek(0)
                obj = _CPUUnpickler(f).load()
        device = get_device()
        obj._device = device
        if obj._metric_net is not None:
            obj._metric_net = obj._metric_net.to(device)
        # Backward compat for pickles saved before REVISE-003 flags
        if not hasattr(obj, "ablation_no_karcher_g"):
            obj.ablation_no_karcher_g = False
        if not hasattr(obj, "ablation_no_simplex_g"):
            obj.ablation_no_simplex_g = False
        if not hasattr(obj, "scoped_g_simplex"):
            obj.scoped_g_simplex = True
        if not hasattr(obj, "hold_aware_stage1"):
            obj.hold_aware_stage1 = False
        if not hasattr(obj, "force_normalize_g"):
            obj.force_normalize_g = False
        if not hasattr(obj, "g_karcher_scale"):
            obj.g_karcher_scale = 0.5
        if not hasattr(obj, "g_eigen_clip"):
            obj.g_eigen_clip = None
        if not hasattr(obj, "g_apply_mode"):
            obj.g_apply_mode = "global"
        if not hasattr(obj, "g_neighborhood_pad"):
            obj.g_neighborhood_pad = 10
        if not hasattr(obj, "karcher_use_clle_init"):
            obj.karcher_use_clle_init = True
        if not hasattr(obj, "pullback_blend_epsilon"):
            obj.pullback_blend_epsilon = 0.0
        if not hasattr(obj, "karcher_blend_lambda"):
            obj.karcher_blend_lambda = 1.0
        if not hasattr(obj, "karcher_metric_mode"):
            obj.karcher_metric_mode = "learned"
        if not hasattr(obj, "consumer_karcher_fix"):
            obj.consumer_karcher_fix = False
        if not hasattr(obj, "stage2_v2"):
            obj.stage2_v2 = False
        if not hasattr(obj, "stage2_realign"):
            obj.stage2_realign = False
        if not hasattr(obj, "lambda_aniso"):
            obj.lambda_aniso = 0.01
        if not hasattr(obj, "ablation_cross_subject_knn"):
            obj.ablation_cross_subject_knn = False
        if not hasattr(obj, "cross_k"):
            obj.cross_k = 20
        if not hasattr(obj, "cross_sigma"):
            obj.cross_sigma = getattr(obj, "sigma", 3.0)
        if not hasattr(obj, "bank_max_points"):
            obj.bank_max_points = 50_000
        if not hasattr(obj, "_latent_bank"):
            obj._latent_bank = None
        if not hasattr(obj, "_tracer"):
            obj._tracer = None
        if not hasattr(obj, "_trace_held_block"):
            obj._trace_held_block = None
        if not hasattr(obj, "_trace_x_gt"):
            obj._trace_x_gt = None
        if not hasattr(obj, "_trace_m_native"):
            obj._trace_m_native = None
        if not hasattr(obj, "oracle_x_hold_fill"):
            obj.oracle_x_hold_fill = False
        if not hasattr(obj, "z_n_oracle_alpha"):
            obj.z_n_oracle_alpha = 0.0
        if not hasattr(obj, "z_prime_oracle_alpha"):
            obj.z_prime_oracle_alpha = 0.0
        if not hasattr(obj, "z_hold_oracle_alpha"):
            obj.z_hold_oracle_alpha = 0.0
        if not hasattr(obj, "stage2_method"):
            #: canonical protocol = MHB inference on any checkpoint;
            # pre-036A checkpoints therefore backfill to 'mhb'.
            obj.stage2_method = "mhb"
        if not hasattr(obj, "mhb_params"):
            obj.mhb_params = {
                "mode": "soft",
                "spline_gamma": 0.02,
                "k_latent": 10,
                "K": 5,
                "gamma": 1e-3,
            }
        if not hasattr(obj, "d_pca"):
            obj.d_pca = None
        if not hasattr(obj, "hold_blend_alpha"):
            obj.hold_blend_alpha = 0.0
        if not hasattr(obj, "oracle_x_hold_alpha"):
            obj.oracle_x_hold_alpha = 1.0
        return obj
