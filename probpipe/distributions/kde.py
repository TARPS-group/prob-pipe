"""Kernel density estimation distribution backed by TFP.

Provides :class:`KDEDistribution`, a Gaussian KDE that satisfies both
:class:`~probpipe.core.protocols.SupportsLogProb` and
:class:`~probpipe.core.protocols.SupportsSampling`.  Useful for
converting a sampling-only distribution (e.g., MCMC output) into one
that supports density evaluation.
"""

from __future__ import annotations

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ._tfp_base import TFPDistribution
from ..core.constraints import Constraint, real
from ..custom_types import ArrayLike
from .._weights import Weights

__all__ = ["KDEDistribution"]


class KDEDistribution(TFPDistribution):
    """Gaussian kernel density estimate as a ProbPipe distribution.

    Wraps a TFP ``MixtureSameFamily(Categorical, MultivariateNormalDiag)``
    to provide a smooth density approximation from a set of weighted
    samples.  Inherits all protocol implementations from
    :class:`TFPDistribution`.

    Parameters
    ----------
    samples : array-like
        Sample matrix of shape ``(n,)`` or ``(n, d)``.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Non-negative weights.  A pre-built :class:`~probpipe.Weights`
        object is also accepted.  Mutually exclusive with
        *log_weights*.  When neither is given, uniform weights are used.
    log_weights : array-like, :class:`~probpipe.Weights`, or None
        Log-unnormalized weights.  A pre-built :class:`~probpipe.Weights`
        object is also accepted.  Mutually exclusive with *weights*.
    bandwidth : array-like or None
        Per-dimension bandwidth (standard deviation of each Gaussian
        kernel), shape ``(d,)`` or scalar.  If ``None``, Silverman's
        rule is used: ``n^{-1/(d+4)} * std_j`` for each dimension *j*.
    name : str or None
        Distribution name for provenance.
    """

    def __init__(
        self,
        samples: ArrayLike,
        weights: ArrayLike | Weights | None = None,
        *,
        log_weights: ArrayLike | Weights | None = None,
        bandwidth: ArrayLike | None = None,
        name: str | None = None,
    ):
        samples = jnp.asarray(samples, dtype=jnp.float32)
        if samples.ndim == 0:
            raise ValueError("samples must have at least 1 dimension.")
        if samples.ndim == 1:
            samples = samples[:, None]  # (n,) -> (n, 1)
            self._scalar = True
        else:
            self._scalar = False

        n, d = samples.shape
        self._samples = samples
        self._d = d
        self._name = name

        # Weights
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)
        w = self._w.normalized

        # Bandwidth (Silverman's rule default)
        if bandwidth is not None:
            bw = jnp.broadcast_to(jnp.asarray(bandwidth, dtype=jnp.float32), (d,))
        else:
            std = jnp.sqrt(self._w.variance(samples))
            # Silverman's rule: n^{-1/(d+4)} * std
            silverman_factor = n ** (-1.0 / (d + 4))
            bw = silverman_factor * jnp.maximum(std, 1e-8)
        self._bandwidth = bw

        # Build the TFP mixture distribution
        if d == 1:
            components = tfd.Normal(
                loc=samples[:, 0],
                scale=bw[0],
            )
        else:
            components = tfd.MultivariateNormalDiag(
                loc=samples,
                scale_diag=jnp.broadcast_to(bw, (n, d)),
            )
        self._tfp_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=w),
            components_distribution=components,
        )

    # -- KDE-specific properties -----------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def n(self) -> int:
        """Number of kernel centres (samples)."""
        return self._samples.shape[0]

    @property
    def support(self) -> Constraint:
        return real

    def __repr__(self) -> str:
        return (
            f"KDEDistribution(n={self.n}, "
            f"event_shape={self.event_shape})"
        )
