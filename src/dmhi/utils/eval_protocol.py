"""Canonical evaluation protocol shared by DMHI and every baseline runner.

Every reported number in the paper carves evaluation masks from the same
implementation of block missingness with the same seeds, so that each method
imputes byte-identical holes:

* protocol: rho = 0.9 held-out blocks of length L = 20 (nine domains) or the
  L grid {10, 20, 30, 40} (three primary domains);
* mask seed convention: mask_seed = run_seed + 2;
* seeds {7, 42, 123}; metrics MAE / MSE / RMSE / MD on held-out entries only.

The mask is realized as non-overlapping runs, so the held-out fraction equals
the nominal ``rate`` exactly: exactly ``round(n_obs * rate)`` observed entries
are hidden per (sample, channel) and ``n_eval`` does not drift with the seed.

Manifold deviation (eq. 9) is implemented here as
:func:`manifold_deviation_rows`: the mean distance from each *imputed* time step
to its k nearest neighbours on the reference manifold, where the query set is
restricted to the time steps that were actually held out. Reported MD in the
paper is the mean over those rows (``md_mean``); the 90th percentile
(``md_p90``) is reported alongside for tail sensitivity.
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency, required for MD only
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    NearestNeighbors = None

#: Reference-manifold points are capped at this many rows for the k-NN fit.
MD_REF_CAP = 30000
#: Fixed seed for the subsampling draw, so MD is reproducible across runs.
MD_REF_SEED = 0


def apply_block_missing(M_orig: np.ndarray, block_len: int, rate: float,
                        seed: int = 42) -> np.ndarray:
    """Apply synthetic block missingness on top of M_orig.

    For each sample and channel, exactly ``round(n_obs * rate)`` observed
    entries are held out as non-overlapping runs of up to ``block_len``
    timesteps (the final run is truncated), separated by random gaps drawn
    from a uniform multinomial split of the remaining ``n_obs - target``
    observed positions.

    Returns the eval mask (1 = held-out).
    """
    rng = np.random.default_rng(seed)
    N, T, D = M_orig.shape
    eval_mask = np.zeros_like(M_orig)
    for n in range(N):
        for d in range(D):
            obs = np.where(M_orig[n, :, d] == 1)[0]
            n_obs = len(obs)
            if n_obs == 0:
                continue
            target = int(round(n_obs * rate))
            target = max(0, min(target, n_obs))
            if target == 0:
                continue
            sizes = []
            rem = target
            while rem > 0:
                b = min(block_len, rem)
                sizes.append(b)
                rem -= b
            k = len(sizes)
            free = n_obs - target
            if free > 0:
                gaps = rng.multinomial(free, [1.0 / (k + 1)] * (k + 1))
            else:
                gaps = np.zeros(k + 1, dtype=int)
            order = rng.permutation(k)
            sizes = [sizes[i] for i in order]
            cursor = 0
            for i, b in enumerate(sizes):
                cursor += int(gaps[i])
                pos = obs[cursor:cursor + b]
                eval_mask[n, pos, d] = 1
                cursor += b
    return eval_mask.astype(M_orig.dtype)


def observed_dims(M_orig: np.ndarray) -> np.ndarray:
    """Channel indices that carry at least one observation anywhere in the split.

    Channels that are entirely missing in a dataset are excluded from the
    manifold, since their imputed values are not comparable to any reference
    point.
    """
    return np.flatnonzero(np.asarray(M_orig).sum(axis=(0, 1)) > 0)


def manifold_deviation_rows(X_hat: np.ndarray, X_true: np.ndarray,
                            row_mask: np.ndarray, k: int = 5,
                            obs_dims: np.ndarray | None = None):
    """MD (eq. 9): nearest-neighbour distance of imputed rows to the manifold.

    ``row_mask`` selects the query time steps -- the paper restricts this to
    the steps that contain held-out entries (``eval_mask.any(axis=2)``).
    ``obs_dims`` restricts the distance to channels observed in the split.

    The reference manifold is capped at ``MD_REF_CAP`` points drawn with a
    fixed seed, which keeps the k-NN fit tractable on the largest dataset.

    Returns ``(md_mean, md_p90)``.
    """
    if NearestNeighbors is None:
        raise ImportError("scikit-learn is required for manifold deviation")
    D = X_true.shape[-1]
    if obs_dims is None:
        ref = np.asarray(X_true).reshape(-1, D)
        q = np.asarray(X_hat)[row_mask].reshape(-1, D)
    else:
        ref = np.asarray(X_true)[..., obs_dims].reshape(-1, len(obs_dims))
        q = np.asarray(X_hat)[row_mask][..., obs_dims]
    if ref.shape[0] > MD_REF_CAP:
        idx = np.random.default_rng(MD_REF_SEED).choice(
            ref.shape[0], MD_REF_CAP, replace=False)
        ref = ref[idx]
    if q.size == 0:
        return float("nan"), float("nan")
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(ref)
    dists, _ = nn.kneighbors(q)
    per_row = dists.mean(axis=1)
    return float(np.mean(per_row)), float(np.percentile(per_row, 90))


def manifold_deviation(X_hat: np.ndarray, X_true: np.ndarray,
                       eval_mask: np.ndarray, k: int = 5,
                       obs_dims: np.ndarray | None = None):
    """MD with the query set derived from ``eval_mask`` (paper eq. 9).

    Thin wrapper around :func:`manifold_deviation_rows` that also drops query
    rows whose imputed values are not finite.
    """
    eval_mask = np.asarray(eval_mask).astype(bool)
    row_mask = eval_mask.any(axis=2)
    if obs_dims is None:
        obs_dims = np.flatnonzero(np.isfinite(X_true).sum(axis=(0, 1)) > 0)
    finite = np.isfinite(X_hat)
    row_mask = row_mask & finite[..., obs_dims].all(axis=2)
    return manifold_deviation_rows(X_hat, X_true, row_mask, k=k,
                                   obs_dims=obs_dims)
