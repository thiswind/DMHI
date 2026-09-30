#!/usr/bin/env python3
"""Generate the tiny synthetic smoke-test sample shipped under src/dmhi/utils/sample/.

The sample is purely synthetic: 40 short multivariate windows whose channels are
smooth nonlinear functions of a 2-D latent trajectory, matching the `(N, T, D)`
convention of the real processed splits. Masks are fully observed; the smoke
test carves a contiguous block at run time, exactly as the benchmark protocol
does.

Deterministic (fixed seed) so the shipped arrays are reproducible:

    python src/dmhi/utils/make_sample.py
"""
import os

import numpy as np

N, T, D = 40, 24, 5


def main() -> None:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 2.0 * np.pi, T)
    X = np.zeros((N, T, D), dtype=np.float32)
    for n in range(N):
        a, b = rng.uniform(0.5, 1.5, size=2)
        ph = rng.uniform(0.0, 2.0 * np.pi, size=2)
        z1 = a * np.sin(t + ph[0])
        z2 = b * np.cos(0.5 * t + ph[1])
        channels = [z1, z2, 0.5 * z1 * z2, np.sin(z1) + z2, z1 - 0.3 * z2]
        X[n] = np.stack(channels, axis=1) + rng.normal(0.0, 0.02, size=(T, D))
    M = np.ones((N, T, D), dtype=np.int8)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "X.npy"), X)
    np.save(os.path.join(out, "M.npy"), M)
    print(f"wrote X{X.shape} M{M.shape} -> {out}")


if __name__ == "__main__":
    main()
