"""Kernel density estimation distribution backed by TFP.

Provides :class:`KDEDistribution`, a Gaussian KDE that satisfies both
:class:`~probpipe.core.protocols.SupportsLogProb` and
:class:`~probpipe.core.protocols.SupportsSampling`.  Useful for
converting a sampling-only distribution (e.g., MCMC output) into one
that supports density evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ._tfp_base import TFPDistribution
from ..core.distribution import ArrayDistribution, _mc_expectation
from ..core.constraints import Constraint, real
from ..core.protocols import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from ..custom_types import Array, ArrayLike, PRNGKey

__all__ = ["KDEDistribution"]


class KDEDistribution(
    ArrayDistribution,
    SupportsSampling,
    SupportsLogProb,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
):
    """Gaussian kernel density estimate as a ProbPipe distribution.

    Wraps a TFP ``MixtureSameFamily(Categorical, MultivariateNormalDiag)``
    to provide a smooth density approximation from a set of weighted
    samples.  Satisfies :class:`SupportsLogProb` and
    :class:`SupportsSampling`.

    Parameters
    ----------
    samples : array-like
        Sample matrix of shape ``(n,)`` or ``(n, d)``.
    weights : array-like or None
        Non-negative weights of shape ``(n,)``.  Normalised internally.
        If ``None``, uniform weights are used.
    bandwidth : array-like or None
        Per-dimension bandwidth (standard deviation of each Gaussian
        kernel), shape ``(d,)`` or scalar.  If ``None``, Silverman's
        rule is used: ``n^{-1/(d+4)} * std_j`` for each dimension *j*.
    name : str or None
        Distribution name for provenance.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        samples: ArrayLike,
        weights: ArrayLike | None = None,
        *,
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
        if weights is not None:
            w = jnp.asarray(weights, dtype=jnp.float32)
            w = w / w.sum()
        else:
            w = jnp.ones(n, dtype=jnp.float32) / n
        self._weights = w

        # Bandwidth (Silverman's rule default)
        if bandwidth is not None:
            bw = jnp.broadcast_to(jnp.asarray(bandwidth, dtype=jnp.float32), (d,))
        else:
            std = jnp.sqrt(jnp.average((samples - jnp.average(samples, weights=w, axis=0)) ** 2, weights=w, axis=0))
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

    # -- Distribution interface ------------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def n(self) -> int:
        """Number of kernel centres (samples)."""
        return self._samples.shape[0]

    # -- shape properties ------------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        if self._scalar:
            return ()
        return (self._d,)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def dtype(self) -> jnp.dtype:
        return self._samples.dtype

    @property
    def support(self) -> Constraint:
        return real

    # -- sampling & density ----------------------------------------------------

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        s = self._tfp_dist.sample(seed=key, sample_shape=sample_shape)
        if self._scalar:
            return s  # MixtureSameFamily with 1D Normal already returns scalar-shaped
        return s

    def _log_prob(self, x: ArrayLike) -> Array:
        x = jnp.asarray(x, dtype=jnp.float32)
        if self._scalar and x.ndim == 0:
            return self._tfp_dist.log_prob(x)
        if self._scalar and x.ndim >= 1 and x.shape[-1:] != (self._d,):
            return self._tfp_dist.log_prob(x)
        return self._tfp_dist.log_prob(x)

    # -- moments ---------------------------------------------------------------

    def _mean(self) -> Array:
        m = jnp.einsum("n,nd->d", self._weights, self._samples)
        if self._scalar:
            return m[0]
        return m

    def _variance(self) -> Array:
        m = self._mean()
        if self._scalar:
            diff = self._samples[:, 0] - m
            sample_var = jnp.einsum("n,n->", self._weights, diff ** 2)
            return sample_var + self._bandwidth[0] ** 2
        diff = self._samples - m
        sample_var = jnp.einsum("n,nd->d", self._weights, diff ** 2)
        return sample_var + self._bandwidth ** 2

    def _cov(self) -> Array:
        m = self._mean()
        if self._scalar:
            return self._variance()
        flat = self._samples
        diff = flat - m
        sample_cov = jnp.einsum("ni,nj,n->ij", diff, diff, self._weights)
        return sample_cov + jnp.diag(self._bandwidth ** 2)

    # -- expectation -----------------------------------------------------------

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Any:
        return _mc_expectation(
            self, f, key=key, num_evaluations=num_evaluations,
            return_dist=return_dist,
        )

    def __repr__(self) -> str:
        return (
            f"KDEDistribution(n={self.n}, "
            f"event_shape={self.event_shape})"
        )
