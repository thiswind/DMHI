"""DMHI: Deterministic Manifold Harmonic Imputation for multivariate time series.

Reference implementation of the method described in:

    DMHI: Deterministic Manifold Harmonic Imputation for Edge Deployable
    Multivariate Time Series (IEEE Internet of Things Journal, under review,
    2026).

The deployed inference path is a three-stage manifold pipeline
(Embed -> intrinsic Compute -> inverse Project) with no neural-network
weights at inference time. Entry point:

    from dmhi.method.pipeline import RiemannianImputer
"""

__version__ = "0.1.0"
