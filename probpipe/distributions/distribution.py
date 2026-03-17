"""
Core distribution abstractions for prob-pipe.

Provides:
  - ``Distribution``          – Abstract base class following TFP shape semantics.
  - ``TFPDistribution``       – Mixin that delegates to an internal ``tfd.*`` instance.
  - ``EmpiricalDistribution`` – Weighted set of samples.
  - ``Provenance``            – Lightweight lineage tracker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from ..custom_types import Array, ArrayLike, PRNGKey


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Provenance:
    """Tracks how a distribution was created."""

    operation: str
    parents: tuple[Distribution, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parent_names = ", ".join(
            p.name or type(p).__name__ for p in self.parents
        )
        return f"Provenance({self.operation!r}, parents=[{parent_names}])"


# ---------------------------------------------------------------------------
# Distribution ABC
# ---------------------------------------------------------------------------

class Distribution(ABC):
    """
    Abstract base for all prob-pipe distributions.

    Shape semantics follow TFP conventions:

    * ``event_shape``  – shape of a single draw (e.g. ``(d,)`` for a
      *d*-dimensional vector distribution).
    * ``batch_shape``  – shape of independent-but-not-identically-distributed
      parameter batches.
    * ``sample(key, sample_shape)`` returns an array of shape
      ``sample_shape + batch_shape + event_shape``.
    """

    # -- core abstract interface ---------------------------------------------

    @property
    @abstractmethod
    def event_shape(self) -> tuple[int, ...]:
        ...

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.float32

    @abstractmethod
    def sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        ...

    @abstractmethod
    def log_prob(self, x: ArrayLike) -> Array:
        ...

    # -- optional concrete methods ------------------------------------------

    def prob(self, x: ArrayLike) -> Array:
        return jnp.exp(self.log_prob(x))

    def mean(self) -> Array:
        raise NotImplementedError(f"{type(self).__name__}.mean()")

    def variance(self) -> Array:
        raise NotImplementedError(f"{type(self).__name__}.variance()")

    # -- naming & provenance ------------------------------------------------

    @property
    def name(self) -> str | None:
        return getattr(self, "_name", None)

    @property
    def source(self) -> Provenance | None:
        return getattr(self, "_source", None)

    def _with_source(self, source: Provenance) -> Distribution:
        self._source = source
        return self

    # -- conversion ---------------------------------------------------------

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        num_samples: int = 1024,
        **kwargs: Any,
    ) -> Distribution:
        """Convert *other* into an instance of *cls* by sampling."""
        raise NotImplementedError(
            f"{cls.__name__}.from_distribution() is not implemented."
        )

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [type(self).__name__]
        if self.name:
            parts.append(f"name={self.name!r}")
        parts.append(f"event_shape={self.event_shape}")
        if self.batch_shape:
            parts.append(f"batch_shape={self.batch_shape}")
        return f"{parts[0]}({', '.join(parts[1:])})"


# ---------------------------------------------------------------------------
# TFPDistribution mixin
# ---------------------------------------------------------------------------

class TFPDistribution(Distribution):
    """
    Base class for distributions backed by a ``tfd.Distribution`` instance.

    Subclasses set ``self._tfp_dist`` in ``__init__``.  Sampling,
    ``log_prob``, ``mean``, and ``variance`` delegate to TFP.
    """

    _tfp_dist: tfd.Distribution

    # -- shape delegation ---------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self._tfp_dist.event_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self._tfp_dist.batch_shape)

    @property
    def dtype(self) -> jnp.dtype:
        return self._tfp_dist.dtype

    # -- sampling & density -------------------------------------------------

    def sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        return self._tfp_dist.sample(seed=key, sample_shape=sample_shape)

    def log_prob(self, x: ArrayLike) -> Array:
        return self._tfp_dist.log_prob(jnp.asarray(x))

    def prob(self, x: ArrayLike) -> Array:
        return self._tfp_dist.prob(jnp.asarray(x))

    def mean(self) -> Array:
        return self._tfp_dist.mean()

    def variance(self) -> Array:
        return self._tfp_dist.variance()


# ---------------------------------------------------------------------------
# EmpiricalDistribution
# ---------------------------------------------------------------------------

class EmpiricalDistribution(Distribution):
    """
    Weighted empirical distribution over a finite set of samples.

    Parameters
    ----------
    samples : array-like, shape ``(n, *event_shape)``
        The support points.  The leading axis is the sample axis.
    weights : array-like, shape ``(n,)``, optional
        Non-negative weights (normalised internally).  Defaults to uniform.
    name : str, optional
        An optional name for provenance / JointDistribution integration.
    """

    def __init__(
        self,
        samples: ArrayLike,
        weights: ArrayLike | None = None,
        *,
        name: str | None = None,
    ):
        samples = jnp.asarray(samples, dtype=jnp.float32)
        if samples.ndim < 2:
            samples = samples.reshape(-1, 1)

        n = samples.shape[0]

        if weights is None:
            weights = jnp.ones(n) / n
        else:
            weights = jnp.asarray(weights, dtype=jnp.float32)
            if weights.shape != (n,):
                raise ValueError(
                    f"weights shape {weights.shape} does not match "
                    f"number of samples {n}."
                )
            if jnp.any(weights < 0):
                raise ValueError("weights must be non-negative.")
            total = jnp.sum(weights)
            if total <= 0:
                raise ValueError("weights must sum to a positive value.")
            weights = weights / total

        self._samples = samples
        self._weights = weights
        self._name = name

    # -- properties ---------------------------------------------------------

    @property
    def n(self) -> int:
        return self._samples.shape[0]

    @property
    def dim(self) -> int:
        return int(np.prod(self._samples.shape[1:]))

    @property
    def samples(self) -> Array:
        return self._samples

    @property
    def weights(self) -> Array:
        return self._weights

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._samples.shape[1:]

    @property
    def dtype(self) -> jnp.dtype:
        return self._samples.dtype

    # -- sampling -----------------------------------------------------------

    def sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        n_draws = int(np.prod(sample_shape)) if sample_shape else 1
        indices = jax.random.choice(
            key, self.n, shape=(n_draws,), p=self._weights, replace=True,
        )
        draws = self._samples[indices]
        if sample_shape:
            return draws.reshape(sample_shape + self.event_shape)
        return draws.squeeze(axis=0)

    # -- density (Gaussian approximation) -----------------------------------

    def log_prob(self, x: ArrayLike) -> Array:
        """Gaussian-approximation log-density."""
        x = jnp.asarray(x)
        mu = self.mean()
        var = self.variance()
        # Diagonal Gaussian approx; clamp variance to avoid log(0)
        var = jnp.maximum(var, 1e-12)
        log_norm = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * var))
        diff = x - mu
        return log_norm - 0.5 * jnp.sum(diff**2 / var, axis=-1)

    # -- moments ------------------------------------------------------------

    def mean(self) -> Array:
        return jnp.einsum("n,n...->...", self._weights, self._samples)

    def variance(self) -> Array:
        mu = self.mean()
        diff = self._samples - mu
        return jnp.einsum("n,n...->...", self._weights, diff**2)

    def cov(self) -> Array:
        """Weighted sample covariance matrix, shape ``(d, d)``."""
        mu = self.mean()
        diff = self._samples.reshape(self.n, -1) - mu.ravel()
        return jnp.einsum("ni,nj,n->ij", diff, diff, self._weights)

    # -- conversion ---------------------------------------------------------

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        num_samples: int = 1024,
        name: str | None = None,
        **kwargs: Any,
    ) -> EmpiricalDistribution:
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        ed = cls(samples, name=name or other.name)
        ed._source = Provenance("from_distribution", parents=(other,))
        return ed
