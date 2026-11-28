"""End-to-end Riemannian Geodesic Imputation pipeline.

RiemannianImputer.build(X_train, M_train, X_val, M_val) → constructs Stage I
+ II (closed-form structures; deterministic, no learned parameters).
RiemannianImputer.impute(X, M) → runs Stage I + II + III and returns hat_X.

All data must be float32. M must be int8 or bool (1=observed, 0=missing).
The package is torch-free: NumPy/SciPy only, runs on CPU (e.g. Raspberry Pi
edge nodes).
Call `conda activate base` before running any Python script using this module.
"""
from __future__ import annotations

import numpy as np
from typing import Optional
import logging
import os
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from .embed import MaskedManifoldEmbedder
from .karcher import (
    reset_karcher_stats,
    get_karcher_stats,
    get_last_karcher_detail,
    reset_karcher_detail,
)
from .clle import clle_inverse
from .mhb import mhb_impute
from .utils import build_time_window_graph
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


class RiemannianImputer:
    """Three-stage deterministic manifold-harmonic imputation (DMHI).

    Stage I:   MaskedManifoldEmbedder  (numpy, CPU) -- mask-aware latent embedding
    Stage II:  soft graph-spline repair on the manifold-harmonic basis
               (``mhb_impute``, the default ``stage2_method='mhb'``)
    Stage III: CLLE symmetric inverse (``clle_inverse``) that freezes observed
               entries; no neural network on the inference path.

    The method is training-free: no learned parameters are fitted or stored,
    and the package does not depend on torch. Constructor parameters that
    used to configure the retired learned-metric / decoder branches are kept
    for backward compatibility with old ``config.json`` files but ignored.
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
        landmark_scheme: str = "stratified",
        landmark_seed: int = 0,
        lambda_I: float = 0.01,
        lambda_norm: float = 0.1,
        embed_epochs: int = 200,
        # Stage II — metric net (DEPRECATED, ignored: retired learned-metric branch)
        n_head: int = 4,
        L_enc: int = 4,
        d_ff: int = 128,
        lambda_geo: float = 1e-3,
        # Stage II — learned-metric ablation only (DEPRECATED, ignored)
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
        # Stage III — learned decoder (DEPRECATED, ignored: retired branch)
        decoder_hidden: int = 256,
        decoder_epochs: int = 200,
        decoder_lr: float = 1e-3,
        decoder_patience: int = 30,
        # --- Ablation flags (do not set in production) ---
        ablation_no_embed: bool = False,
        # DEPRECATED, ignored. IoTJ-R1 (Reviewer 3, comment 1): the deployed
        # pipeline is pure closed-form; the learned SPD metric field and the
        # learned decoder were removed from this release, so these flags no
        # longer select any code path. Kept only so that old ``config.json``
        # files and old scripts keep loading without error.
        use_learned_metric: bool = False,
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
        # Phase 2.10: post-Karcher Z blend + metric mode ('identity' | 'pullback';
        # the legacy 'learned' mode raises — the learned metric was removed)
        karcher_blend_lambda: float = 1.0,
        karcher_metric_mode: str = "identity",
        # REVISE-003 Stage C: Karcher consumer fixes (K1–K3)
        consumer_karcher_fix: bool = True,
        # REVISE-004 Stage II v2: block synth missing + in-loop Karcher
        # (DEPRECATED, ignored: v2 was a training-loop profile)
        stage2_v2: bool = False,
        # CONTRAST-REALIGN Stage II: pullback on filtered native windows
        # (DEPRECATED, ignored: realign was a training-loop profile)
        stage2_realign: bool = False,
        lambda_aniso: float = 0.01,
        #: cross-subject Karcher neighbors (DEPRECATED, ignored: the bank was
        #: queried under the removed learned metric; kept for config compat)
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
        # operator. Inference-time choice — checkpoints built with either
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
        # E4c landmark-selection study (reviewer R1-3); fail fast on typos.
        if str(landmark_scheme).lower() not in ("stratified", "uniform", "random"):
            raise NotImplementedError(
                f"landmark_scheme must be stratified|uniform|random, "
                f"got {landmark_scheme!r}")
        self.landmark_scheme = landmark_scheme
        self.landmark_seed = landmark_seed
        self.lambda_I = lambda_I
        self.lambda_norm = lambda_norm
        self.embed_epochs = embed_epochs

        self.n_head = n_head
        self.L_enc = L_enc
        self.d_ff = d_ff
        self.lambda_geo = lambda_geo

        # The training hyper-parameters of the retired branches (``lr``,
        # ``epochs``, ``patience``, ``batch_size``, ``lambda_F``,
        # ``lambda_smooth``, ``decoder_*``) are accepted by the constructor
        # signature for compatibility with older scripts, but intentionally
        # NOT stored: nothing in this release reads them.

        self.synth_miss_rate = synth_miss_rate

        self.N_max = N_max
        self.k_time = k_time
        self.sigma = sigma
        self.eta = eta

        self.ablation_no_embed = ablation_no_embed
        # IoTJ-R1: the deployed pipeline is training-free (Reviewer 3,
        # comment 1). The learned-metric branch was removed from this
        # release, so ``ablation_no_metric`` is now a constant: the
        # Euclidean / G_net=None / G_field=None path is the only path.
        # Kept as an attribute because karcher/mhb/clle consumers and old
        # configs branch on it.
        self.ablation_no_metric = True
        # Deprecated (ignored): kept as attributes so that pickled
        # checkpoints and old configs written by previous releases still load.
        self.use_learned_metric = False
        self.ablation_use_decoder = False
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
        self.karcher_metric_mode = str(karcher_metric_mode).lower()
        if self.karcher_metric_mode not in ("identity", "pullback"):
            raise NotImplementedError(
                f"karcher_metric_mode={karcher_metric_mode!r} is not available; "
                "the learned metric network was removed. Use 'identity' or "
                "'pullback'.")
        self.consumer_karcher_fix = consumer_karcher_fix
        # Deprecated (ignored): the v2 / realign Stage-II training profiles
        # were removed; kept as plain attributes for old-checkpoint compat.
        self.stage2_v2 = False
        self.stage2_realign = False
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
        if consumer_karcher_fix and eta >= 1.0:
            self.eta = 0.1  # K1: smaller step when fix profile enabled

        self._ablation_pca = None

        self._embedder: Optional[MaskedManifoldEmbedder] = None
        # No learned components exist in this release. The attributes are
        # kept (always None) so that external ``is not None`` checks on old
        # checkpoints keep working.
        self._metric_net = None
        self._decoder = None
        self._tau: float = 1.0  # temporal hinge threshold from Stage I
        self._is_fitted: bool = False
        # K-NN ambient anchoring (stored after fit)
        self._Z_train: Optional[np.ndarray] = None   # (N, T, d)
        self._X_train: Optional[np.ndarray] = None   # (N, T, D)
        self._M_train: Optional[np.ndarray] = None   # (N, T, D)
        self._obs_rate: float = 1.0                   # observation rate of the build data
        self._latent_bank: Optional[np.ndarray] = None
        self._tracer: Optional[PipelineTracer] = None
        self._trace_held_block: Optional[np.ndarray] = None
        self._trace_x_gt: Optional[np.ndarray] = None
        self._trace_m_native: Optional[np.ndarray] = None

    def _needs_torch(self) -> bool:
        """Whether this configuration touches torch. Always False: the
        package is torch-free (kept for backward compatibility)."""
        return False

    # ------------------------------------------------------------------
    # Build (closed-form construction; no learned parameters)
    # ------------------------------------------------------------------

    def build(
        self,
        X_train: np.ndarray,
        M_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        M_val: Optional[np.ndarray] = None,
    ) -> "RiemannianImputer":
        """Construct Stage I and Stage II on a dataset.

        This computes deterministic structures only — no neural network and
        no learned weights. Stage I chart construction combines closed-form
        spectral decomposition with a fixed-schedule deterministic landmark
        refinement (no validation early-stopping, no data-driven stopping);
        Stage II solves the soft graph-Tikhonov system in closed form.

        Args:
            X_train: (N, T, D) float32 training samples.
            M_train: (N, T, D) int8/bool mask (1=observed).
            X_val:   (N_v, T, D) float32 validation samples (unused; kept
                     for signature compatibility).
            M_val:   (N_v, T, D) mask (unused).

        Returns:
            self
        """
        N, T, D = X_train.shape
        # Torch-free deployment path (edge devices): no device selection.
        logger.info("Device: cpu (torch-free closed-form path)")

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

        logger.info("Building Stage I (manifold embedder)...")
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
                landmark_scheme=self.landmark_scheme,
                landmark_seed=self.landmark_seed,
                lambda_I=lambda_I_eff,
                lambda_norm=self.lambda_norm,
                max_epochs=self.embed_epochs,
                graph_alpha=graph_alpha_eff,
                d_pca=self.d_pca,
                hold_blend_alpha=self.hold_blend_alpha,
            )
            embedder.build(X_ref, M_ref)
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

        # --- Stage II: closed-form graph-Tikhonov operator (no metric net) ---
        # No learned metric, no torch import: Stage II is the closed-form
        # soft graph-spline / MHB refinement solved at inference time by
        # mhb_impute(); nothing here needs a neural network or a set of
        # learned weights. Only the build latents/ambient/mask are stored for
        # the Stage III k-NN anchor fallback.
        # Stage II is a closed-form operator; there is no learned metric
        # network and nothing to train.
        self._metric_net = None
        logger.info("  Stage II: closed-form operator only; nothing to train.")

        # Pre-embed all training samples (stored for Stage III anchoring)
        logger.info("Embedding training samples for Stage II...")
        Z_train = self._embed_batch(X_train, M_train, desc="Embed train")   # (N, T, d)

        logger.info("Stage II closed-form graph-Tikhonov operator ready "
                    "(no optimizer in this path).")

        # Store training data for Stage III k-NN anchoring
        self._Z_train = Z_train           # (N, T, d)
        self._X_train = X_train.astype(np.float32)
        self._M_train = M_train.astype(np.int8)
        # Observation rate determines Stage III strategy:
        # Dense (obs > 0.4): CLLE inverse (accurate per-sample PCA)
        # Sparse (obs ≤ 0.4): global k-NN (training anchors more reliable)
        self._obs_rate = float(M_train.mean())

        # No decoder: Stage III is always the CLLE inverse.
        self._decoder = None

        self._is_fitted = True
        logger.info("Fitting complete.")
        return self

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
        assert self._is_fitted, "Call build() first."
        N, T, D = X.shape
        hat_X = np.zeros_like(X, dtype=np.float32)

        # --- Stage A: reset Karcher counters ---
        reset_karcher_stats()
        self._latent_bank = None
        if float(getattr(self, "pullback_blend_epsilon", 0.0)) > 0.0:
            raise NotImplementedError(
                "pullback_blend_epsilon>0 needs the embedder-pullback ablation "
                "module, which is not part of this release. "
                "The deployed path uses the default pullback_blend_epsilon=0.0."
            )

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

            # Stage II: time steps that need reconstruction.
            I_miss = np.where(miss_mask_T)[0]

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

            Z_k_raw: Optional[np.ndarray] = None
            _stage2 = str(getattr(self, "stage2_method", "mhb"))
            if len(I_miss) > 0 and _stage2 == "mhb":
                # — paper Algorithm 1 path (closed-form Stage II).
                # The soft graph-Tikhonov refinement never consumes a learned
                # metric: G_net is always None, so the training-free
                # deployment includes Stage II as-is.
                # Graph A = Stage I mask-weighted time-window k-NN graph
                # (eqs. (5)-(7)) built on THIS sample's low-d PCA coords,
                # mirroring MaskedManifoldEmbedder.build() exactly.
                reset_karcher_detail()
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
                    None,
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
                # No missing entries, or stage2_method='karcher' (the legacy
                # Karcher operator required the removed learned metric and is
                # no longer available; Stage I estimate is left untouched).
                Z_prime = Z_n
                Z_k_raw = None

            if Z_gt is not None and self.z_prime_oracle_alpha > 0.0 and len(I_hold) > 0:
                alpha_p = float(self.z_prime_oracle_alpha)
                for t_idx in I_hold:
                    ti = int(t_idx)
                    Z_prime[ti] = (
                        (1.0 - alpha_p) * Z_prime[ti] + alpha_p * Z_gt[ti]
                    ).astype(np.float32)

            # Stage III: inverse mapping
            if self.ablation_no_embed and self._ablation_pca is not None:
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
                # CLLE inverse with Euclidean simplex weights.
                #
                # No learned metric exists in this release, so the simplex-
                # weight problem is always solved with the Euclidean Gram
                # matrix (G_field=None: the only path).
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
                    hat_metric = hat_euclid
                    hat_X[n] = hat_euclid
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
                        G_field=None,
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
                        None,
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
                    # No learned metric: the G-Frobenius trace stats are
                    # identically zero (same values the retired diagnostic
                    # returned when no metric net was fitted).
                    g_net_fro, g_pb_fro, g_i_fro = 0.0, 0.0, 0.0
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

    def __getstate__(self):
        """Return the picklable state. No device / learned components exist
        in this release; legacy keys are dropped so checkpoints stay pure
        NumPy and load on machines without torch."""
        state = self.__dict__.copy()
        state.pop("_device", None)
        state["_metric_net"] = None
        state["_decoder"] = None
        return state

    def __setstate__(self, state):
        state = dict(state)
        state.pop("_device", None)
        self.__dict__.update(state)
        # Normalize legacy checkpoints: there is no learned component in
        # this release, so these are always None.
        self._metric_net = None
        self._decoder = None

    def save(self, path: str) -> None:
        import pickle, pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "RiemannianImputer":
        import pickle

        class _LegacyUnpickler(pickle.Unpickler):
            """Load closed-form checkpoints saved by earlier releases.

            Earlier releases stored a ``torch.device`` string in ``_device``
            (always ``None`` on the training-free path). The shim maps that
            class to ``None`` so such pickles load without torch. Any
            checkpoint that embeds actual learned weights (a pickled metric
            network or decoder) is rejected: it contradicts the training-free
            method and cannot be reproduced by this release.
            """

            _RETIRED = ("dmhi.method.metric", "dmhi.method.losses",
                        "dmhi.method.stage2_realign")

            def find_class(self, module, name):
                if module == "torch" and name == "device":
                    return lambda *a, **k: None
                if module.startswith("torch"):
                    raise RuntimeError(
                        "This checkpoint embeds torch objects (learned "
                        "weights); it was produced by a retired training "
                        "branch and is not loadable by the training-free "
                        "release. Rebuild the checkpoint with "
                        "scripts/build_checkpoint.py."
                    )
                if module in self._RETIRED or module.startswith("torch."):
                    raise RuntimeError(
                        f"This checkpoint embeds a retired module "
                        f"({module}.{name}); it contains learned weights and "
                        "is not loadable by the training-free release. "
                        "Rebuild the checkpoint with scripts/build_checkpoint.py."
                    )
                return super().find_class(module, name)

        with open(path, "rb") as f:
            try:
                obj = pickle.load(f)
            except ModuleNotFoundError:
                # Legacy pickle referencing a removed module: retry through
                # the shim so the error message names the retired component.
                f.seek(0)
                obj = _LegacyUnpickler(f).load()
        # Old checkpoints may carry a learned-metric flag or device; both
        # are meaningless now and are normalized away by __setstate__.
        obj.use_learned_metric = False
        obj.ablation_use_decoder = False
        obj.ablation_no_metric = True
        obj._metric_net = None
        obj._decoder = None
        if hasattr(obj, "_device"):
            del obj._device
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
            obj.karcher_metric_mode = "identity"
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
