"""Pipeline numerical trace for ablation debugging."""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


def _entropy(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    w = w[w > 1e-12]
    if w.size == 0:
        return 0.0
    w = w / w.sum()
    return float(-(w * np.log(w)).sum())


def hold_timesteps(m_n: np.ndarray, tau_obs: int) -> np.ndarray:
    """Time indices where obs count < tau_obs (Karcher hold region)."""
    miss_mask_T = (m_n.sum(axis=1) < tau_obs).astype(bool)
    return np.where(miss_mask_T)[0]


def per_dim_mae(hat: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(np.abs(hat[mask] - ref[mask]).mean())


@dataclass
class TraceRecord:
    episode_id: int
    variant: str
    block_t0: int = -1
    n_hold_T: int = 0
    mae_hold: float = 0.0
    mae_obs: float = 0.0
    s1_z_delta_hold_mean: float = 0.0
    s2_z_delta_hold_mean: float = 0.0
    s3_mae_hold_euclid: float = 0.0
    s3_mae_hold_metric: float = 0.0
    g_eig_ratio_hold_mean: float = 0.0
    simplex_entropy_hold_mean: float = 0.0
    simplex_max_weight_hold_mean: float = 0.0
    karcher_steps: int = 0
    # Phase 2.12 LOG→PRESET (T1–T11)
    z_gt_err_hold_mean: float = 0.0
    z_k_pre_err: float = 0.0
    z_k_post_err: float = 0.0
    z_prime_err: float = 0.0
    z_k_delta: float = 0.0
    g_net_fro_hold: float = 0.0
    g_pb_fro_hold: float = 0.0
    g_i_fro_hold: float = 0.0
    karcher_step_norm_mean: float = 0.0
    stage3_gain: float = 0.0
    x_recon_from_zgt_mae: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class PipelineTracer:
    """Append-only JSONL trace writer."""

    def __init__(self, path: pathlib.Path | str, variant: str):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.variant = variant
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, record: TraceRecord) -> None:
        record.variant = self.variant
        self._fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "PipelineTracer":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def collect_hold_simplex_stats(
    Z_prime: np.ndarray,
    id_L: np.ndarray,
    clle_weights: dict,
    G_field_n: np.ndarray | None,
    hold_ts: np.ndarray,
) -> tuple[float, float, float]:
    """Mean entropy, max weight, G eig ratio on hold timesteps."""
    from .utils import simplex_weights, simplex_weights_metric

    entropies: list[float] = []
    max_ws: list[float] = []
    eig_ratios: list[float] = []
    Z_L = Z_prime[id_L]
    for t in hold_ts:
        if t not in clle_weights:
            continue
        nn_ids_global, _ = clle_weights[t]
        nn_local = np.array(
            [np.searchsorted(id_L, gid) for gid in nn_ids_global], dtype=int
        )
        nn_local = np.clip(nn_local, 0, len(id_L) - 1)
        Z_nn = Z_L[nn_local]
        if G_field_n is not None:
            G_t = G_field_n[t]
            w = simplex_weights_metric(Z_nn, Z_prime[t], G_t)
            eigs = np.linalg.eigvalsh(G_t)
            eig_ratios.append(
                float(
                    np.log10(max(eigs.max(), 1e-12) / max(eigs.min(), 1e-12))
                )
            )
        else:
            w = simplex_weights(Z_nn, Z_prime[t])
        entropies.append(_entropy(w))
        max_ws.append(float(w.max()))
    er = float(np.mean(eig_ratios)) if eig_ratios else 0.0
    ent = float(np.mean(entropies)) if entropies else 0.0
    mx = float(np.mean(max_ws)) if max_ws else 0.0
    return ent, mx, er


def summarize_traces(jsonl_paths: list[pathlib.Path]) -> list[dict]:
    rows: list[dict] = []
    for p in jsonl_paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows
