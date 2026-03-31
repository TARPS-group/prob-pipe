"""Empirical and bootstrap distribution classes.

Provides:
  - ``EmpiricalDistribution[T]``        – Generic weighted empirical distribution.
  - ``ArrayEmpiricalDistribution``       – Array specialization with moments.
  - ``BootstrapReplicateDistribution[T]``    – Bootstrap resampling over datasets.
  - ``ArrayBootstrapReplicateDistribution``  – Array specialization.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .._utils import prod
from .protocols import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)

import jax
import jax.numpy as jnp

from ..custom_types import Array, ArrayLike, PRNGKey
from .constraints import Constraint, real
from . import _distribution_base as _base
from .._utils import _auto_key
from ._distribution_base import Distribution
from ._array_distributions import (
    ArrayDistribution,
    BootstrapDistribution,
)


# ---------------------------------------------------------------------------
# EmpiricalDistribution (generic base)
# ---------------------------------------------------------------------------

class EmpiricalDistribution[T](
    Distribution[T],
    SupportsSampling,
    SupportsExpectation,
):
    """
    Weighted empirical distribution over a finite set of samples.

    This is the generic base class.  It stores samples in a numpy object
    array, supporting arbitrary sample types ``T`` (arrays, pytrees,
    distributions, callables, etc.).  Use :class:`ArrayEmpiricalDistribution`
    when TFP-style shape semantics (``batch_shape``, ``event_shape``,
    ``flatten_value``, ``support``, etc.) are required — it stores samples
    as a stacked JAX array for efficient vectorised operations.

    Parameters
    ----------
    samples : sequence of T
        The support points.  Must be a non-empty sequence (list, tuple,
        or array).
    weights : array-like, shape ``(n,)``, optional
        Non-negative weights (normalised internally).  Mutually exclusive
        with *log_weights*.  When neither is given the distribution is
        uniform.
    log_weights : array-like, shape ``(n,)``, optional
        Log-unnormalised weights.  Preferred when weights span many orders
        of magnitude (e.g. importance sampling).  Normalised internally via
        ``jax.nn.softmax``.  Mutually exclusive with *weights*.
    name : str, optional
        An optional name for provenance / JointDistribution integration.
    """

    def __init__(
        self,
        samples: Sequence[T] | ArrayLike,
        weights: ArrayLike | None = None,
        *,
        log_weights: ArrayLike | None = None,
        name: str | None = None,
    ):
        # If samples are already a JAX or numpy array, store as-is.
        # Otherwise store as a numpy object array for generic indexing.
        if isinstance(samples, (jnp.ndarray, np.ndarray)):
            self._samples = samples
        else:
            self._samples = np.asarray(samples, dtype=object)
        n = len(self._samples)
        if n == 0:
            raise ValueError("samples must be a non-empty sequence.")
        self._init_weights(n, weights, log_weights=log_weights, name=name)

    def _init_weights(
        self,
        n: int,
        weights: ArrayLike | None = None,
        *,
        log_weights: ArrayLike | None = None,
        name: str | None = None,
    ) -> None:
        """Validate and store weights, name, and flags.
        """
        if weights is not None and log_weights is not None:
            raise ValueError(
                "Provide either weights or log_weights, not both."
            )

        if weights is not None:
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
            self._log_weights = jnp.log(weights)
            self._is_uniform = False
        elif log_weights is not None:
            log_weights = jnp.asarray(log_weights, dtype=jnp.float32)
            if log_weights.shape != (n,):
                raise ValueError(
                    f"log_weights shape {log_weights.shape} does not match "
                    f"number of samples {n}."
                )
            self._log_weights = log_weights
            self._is_uniform = False
        else:
            self._log_weights = None
            self._is_uniform = True

        self._weights_cache: Array | None = None
        self._name = name
        self._approximate = True

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    # -- properties ---------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of samples."""
        return len(self._samples)

    @property
    def samples(self) -> np.ndarray:
        """The stored samples as a numpy object array."""
        return self._samples

    @property
    def is_uniform(self) -> bool:
        """True when all samples have equal weight."""
        return self._is_uniform

    @property
    def weights(self) -> Array:
        """Normalised weights, shape ``(n,)``."""
        if self._is_uniform:
            return jnp.ones(self.n, dtype=jnp.float32) / self.n
        if self._weights_cache is None:
            self._weights_cache = jax.nn.softmax(self._log_weights)
        return self._weights_cache

    @property
    def log_weights(self) -> Array | None:
        """Normalised log-weights, shape ``(n,)``.  ``None`` when uniform."""
        if self._is_uniform:
            return None
        return self._log_weights - jax.scipy.special.logsumexp(self._log_weights)

    # -- sampling -----------------------------------------------------------

    def _sample_one(self, key: PRNGKey) -> T:
        """Draw a single sample (with replacement according to weights)."""
        if self._is_uniform:
            idx = jax.random.randint(key, shape=(), minval=0, maxval=self.n)
        else:
            idx = jax.random.choice(key, self.n, p=self.weights)
        return self._samples[idx]

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Any:
        """Draw samples with replacement according to weights.

        When samples are stored as an array (JAX or numpy), returns an
        array of shape ``(*sample_shape, *event_shape)`` via fancy
        indexing.  When samples are stored as a numpy object array
        (arbitrary types), returns an object array of shape
        ``sample_shape``.
        """
        if sample_shape == ():
            return self._sample_one(key)
        n_draws = prod(sample_shape)
        if self._is_uniform:
            indices = jax.random.randint(key, shape=(n_draws,), minval=0, maxval=self.n)
        else:
            indices = jax.random.choice(
                key, self.n, shape=(n_draws,), p=self.weights, replace=True,
            )
        draws = self._samples[indices]
        return draws.reshape(sample_shape + draws.shape[1:])

    @property
    def _is_object_array(self) -> bool:
        """True when samples are stored as a numpy object array."""
        return isinstance(self._samples, np.ndarray) and self._samples.dtype == object

    def _eval_f(self, f: Callable, samples: Any) -> Array:
        """Evaluate *f* over *samples*, using vmap when possible."""
        if self._is_object_array:
            return jnp.stack([f(x) for x in samples])
        return jax.vmap(f)(samples)

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Any:
        """Compute ``E[f(X)]`` over the empirical support.

        When ``num_evaluations`` is ``None``, the expectation is computed
        exactly as a weighted sum over all support points.  When
        ``num_evaluations`` is specified and smaller than ``self.n``, a
        random subsample is used.
        """
        if num_evaluations is not None and num_evaluations < self.n:
            # Subsample — this is approximate
            if key is None:
                key = _auto_key()
            idx = jax.random.choice(key, self.n, shape=(num_evaluations,), replace=False)
            f_vals = self._eval_f(f, self._samples[idx])

            rd = return_dist if return_dist is not None else _base.RETURN_APPROX_DIST
            if rd:
                sub_w = None
                if not self._is_uniform:
                    sub_w = self.weights[idx]
                    sub_w = sub_w / jnp.sum(sub_w)
                return BootstrapDistribution(f_vals, weights=sub_w)

            if self._is_uniform:
                return jnp.mean(f_vals, axis=0)
            sub_w = self.weights[idx]
            sub_w = sub_w / jnp.sum(sub_w)
            return jnp.einsum("n,n...->...", sub_w, f_vals)

        # Exact: evaluate f on all support points
        f_vals = self._eval_f(f, self._samples)
        if self._is_uniform:
            return jnp.mean(f_vals, axis=0)
        return jnp.einsum("n,n...->...", self.weights, f_vals)


# ---------------------------------------------------------------------------
# ArrayEmpiricalDistribution
# ---------------------------------------------------------------------------

class ArrayEmpiricalDistribution(
    EmpiricalDistribution[Array],
    ArrayDistribution,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
):
    """Empirical distribution with full :class:`ArrayDistribution` shape semantics.

    Stores samples as a stacked JAX array for efficient vectorised
    operations (``jax.vmap``-based sampling and expectations).

    Inherits weight management from :class:`EmpiricalDistribution` and
    adds TFP-style shape properties (``batch_shape``, ``flatten_value``,
    ``unflatten_value``, ``support``, etc.) via :class:`ArrayDistribution`,
    plus exact weighted moments (mean, variance, covariance).

    Use this instead of :class:`EmpiricalDistribution` when the distribution
    must interoperate with :class:`JointDistribution` components or other
    code that requires :class:`ArrayDistribution` instances.
    """

    def __init__(
        self,
        samples: ArrayLike,
        weights: ArrayLike | None = None,
        *,
        log_weights: ArrayLike | None = None,
        name: str | None = None,
    ):
        # Store only the JAX array — bypass the generic base's storage.
        self._samples = jnp.asarray(samples, dtype=jnp.float32)
        if self._samples.ndim == 0:
            raise ValueError("samples must have at least 1 dimension (the sample axis).")
        self._init_weights(
            self._samples.shape[0], weights, log_weights=log_weights, name=name,
        )

    # -- array-specific properties ------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of a single sample (excluding the sample axis)."""
        return self._samples.shape[1:]

    @property
    def dim(self) -> int:
        """Flat dimensionality of each sample (product of event_shape, or 1 for scalars)."""
        return max(1, prod(self._samples.shape[1:]))

    @property
    def dtype(self) -> jnp.dtype:
        return self._samples.dtype

    @property
    def support(self) -> Constraint:
        return real

    # -- moments ------------------------------------------------------------

    def _mean(self) -> Array:
        if self._is_uniform:
            return jnp.mean(self._samples, axis=0)
        return jnp.einsum("n,n...->...", self.weights, self._samples)

    def _variance(self) -> Array:
        mu = self._mean()
        diff = self._samples - mu
        if self._is_uniform:
            return jnp.mean(diff**2, axis=0)
        return jnp.einsum("n,n...->...", self.weights, diff**2)

    def _cov(self) -> Array:
        """Weighted sample covariance matrix, shape ``(d, d)``."""
        mu = self._mean()
        # Flatten to 2D: (n, d)
        flat_samples = self._samples.reshape(self.n, -1)
        diff = flat_samples - mu.reshape(-1)
        if self._is_uniform:
            return jnp.einsum("ni,nj->ij", diff, diff) / self.n
        return jnp.einsum("ni,nj,n->ij", diff, diff, self.weights)


# ---------------------------------------------------------------------------
# BootstrapReplicateDistribution (generic base)
# ---------------------------------------------------------------------------

class BootstrapReplicateDistribution[T](
    Distribution[T],
    SupportsSampling,
    SupportsExpectation,
):
    """N-fold product of an empirical distribution (bootstrap resampling).

    This is the generic base class.  It stores observations in a numpy
    object array, supporting arbitrary observation types ``T``.  Use
    :class:`ArrayBootstrapReplicateDistribution` when TFP-style shape semantics
    (``batch_shape``, ``event_shape``, ``support``, etc.) are required.

    Each sample from this distribution is a bootstrapped dataset — ``n``
    observations drawn i.i.d. with replacement from the source data.
    This provides the sampling distribution over datasets needed for
    BayesBag (bagged posteriors).

    When the source is an :class:`EmpiricalDistribution`, ``n`` defaults
    to the number of samples in the empirical distribution.  Otherwise
    ``n`` must be specified explicitly.

    Parameters
    ----------
    source : EmpiricalDistribution or sequence
        The data to bootstrap from.  If an ``EmpiricalDistribution``,
        its samples and weights are used directly.  If a sequence,
        it is treated as an equally-weighted dataset.
    n : int or None
        Number of observations per bootstrap dataset.  Defaults to the
        number of samples in ``source``.
    name : str or None
        Distribution name for provenance.

    Examples
    --------
    >>> data = EmpiricalDistribution(observed_data)
    >>> bootstrap = BootstrapReplicateDistribution(data)
    >>> # Each sample is a bootstrapped dataset of the same size
    >>> boot_dataset = sample(bootstrap, key=jax.random.PRNGKey(0))
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        source: EmpiricalDistribution | Sequence | ArrayLike,
        *,
        n: int | None = None,
        name: str | None = None,
    ):
        if isinstance(source, EmpiricalDistribution):
            self._data = source.samples
            self._is_uniform = source.is_uniform
            self._weights = None if source.is_uniform else source.weights
            default_n = source.n
        elif isinstance(source, (jnp.ndarray, np.ndarray)):
            self._data = source
            if self._data.ndim == 0:
                raise ValueError("source must have at least 1 dimension (the observation axis).")
            if len(self._data) == 0:
                raise ValueError("source must be a non-empty sequence.")
            self._is_uniform = True
            self._weights = None
            default_n = len(self._data)
        else:
            self._data = np.asarray(source, dtype=object)
            if len(self._data) == 0:
                raise ValueError("source must be a non-empty sequence.")
            self._is_uniform = True
            self._weights = None
            default_n = len(self._data)
        self._init_bootstrap_state(default_n, n=n, name=name)

    def _init_bootstrap_state(
        self,
        default_n: int,
        *,
        n: int | None,
        name: str | None,
    ) -> None:
        """Set n, source_n, name, and flags. Shared with Array subclass."""
        if n is None:
            self._n = default_n
        else:
            if n < 1:
                raise ValueError(f"n must be positive, got {n}")
            self._n = n

        self._name = name
        self._source_n = len(self._data)
        self._approximate = True

    # -- properties ---------------------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def n(self) -> int:
        """Number of observations per bootstrap dataset."""
        return self._n

    @property
    def source_n(self) -> int:
        """Number of observations in the source data."""
        return self._source_n

    @property
    def data(self) -> np.ndarray:
        """The source data as a numpy object array."""
        return self._data

    @property
    def weights(self) -> Array:
        """Normalised weights, shape ``(source_n,)``."""
        if self._is_uniform:
            return jnp.ones(self._source_n, dtype=jnp.float32) / self._source_n
        return self._weights

    @property
    def is_uniform(self) -> bool:
        """True when all source observations have equal weight."""
        return self._is_uniform

    def _sample_one(self, key: PRNGKey) -> Any:
        """Draw a single bootstrapped dataset."""
        if self._weights is None:
            idx = jax.random.choice(key, self._source_n, shape=(self._n,), replace=True)
        else:
            idx = jax.random.choice(
                key, self._source_n, shape=(self._n,), replace=True, p=self._weights,
            )
        return self._data[idx]

    @property
    def _is_object_data(self) -> bool:
        """True when source data is stored as a numpy object array."""
        return isinstance(self._data, np.ndarray) and self._data.dtype == object

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Any:
        """Draw bootstrap datasets."""
        if sample_shape == ():
            return self._sample_one(key)

        total = prod(sample_shape)
        keys = jax.random.split(key, total)
        if self._is_object_data:
            results = np.empty(total, dtype=object)
            for i in range(total):
                results[i] = self._sample_one(keys[i])
            return results.reshape(sample_shape)
        results = jax.vmap(self._sample_one)(keys)
        return results.reshape(*sample_shape, *results.shape[1:])

    # -- expectation --------------------------------------------------------

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Array | BootstrapDistribution:
        """Compute ``E[f(dataset)]`` via Monte Carlo over bootstrap datasets.

        Parameters
        ----------
        f : callable
            Function mapping a bootstrap dataset to an array.
        key : PRNGKey, optional
            Random key.  Auto-generated if ``None``.
        num_evaluations : int, optional
            Number of bootstrap datasets to draw.  Defaults to
            :data:`DEFAULT_NUM_EVALUATIONS`.
        return_dist : bool, optional
            If ``True``, return a :class:`BootstrapDistribution` over
            the evaluations.  Defaults to :data:`RETURN_APPROX_DIST`.
        """
        if key is None:
            key = _auto_key()
        if num_evaluations is None:
            num_evaluations = _base.DEFAULT_NUM_EVALUATIONS

        datasets = self._sample(key, sample_shape=(num_evaluations,))
        if self._is_object_data:
            f_vals = jnp.stack([f(datasets[i]) for i in range(num_evaluations)])
        else:
            f_vals = jax.vmap(f)(datasets)

        rd = return_dist if return_dist is not None else _base.RETURN_APPROX_DIST
        if rd:
            return BootstrapDistribution(f_vals)
        return jnp.mean(f_vals, axis=0)

    def __repr__(self) -> str:
        return (
            f"BootstrapReplicateDistribution(n={self._n}, "
            f"source_n={self._source_n})"
        )


class ArrayBootstrapReplicateDistribution(BootstrapReplicateDistribution[Array], ArrayDistribution):
    """Joint bootstrap distribution with full :class:`ArrayDistribution` shape semantics.

    Inherits all functionality from :class:`BootstrapReplicateDistribution` and adds
    TFP-style shape properties (``batch_shape``, ``event_shape``, ``support``,
    etc.) via :class:`ArrayDistribution`.

    Use this instead of :class:`BootstrapReplicateDistribution` when the distribution
    must interoperate with code that requires :class:`ArrayDistribution` instances.
    """

    def __init__(
        self,
        source: EmpiricalDistribution | ArrayLike,
        *,
        n: int | None = None,
        name: str | None = None,
    ):
        # Coerce to JAX array, then let the generic base store it.
        if isinstance(source, EmpiricalDistribution):
            jax_data = jnp.asarray(list(source.samples), dtype=jnp.float32)
            # Reconstruct via the array path so the base stores a JAX array.
            self._data = jax_data
            self._is_uniform = source.is_uniform
            self._weights = None if source.is_uniform else source.weights
            default_n = source.n
        else:
            jax_data = jnp.asarray(source, dtype=jnp.float32)
            if jax_data.ndim == 0:
                raise ValueError("source must have at least 1 dimension (the observation axis).")
            self._data = jax_data
            self._is_uniform = True
            self._weights = None
            default_n = jax_data.shape[0]

        self._event_shape_per_obs = self._data.shape[1:]
        self._init_bootstrap_state(default_n, n=n, name=name)

    # -- array-specific properties ------------------------------------------

    @property
    def obs_shape(self) -> tuple[int, ...]:
        """Shape of a single observation (excluding the observation axis)."""
        return self._event_shape_per_obs

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of a single bootstrap dataset: ``(n, *obs_shape)``."""
        return (self._n, *self._event_shape_per_obs)

    @property
    def dim(self) -> int:
        """Flat dimensionality of each bootstrap dataset."""
        return self._n * max(1, prod(self._event_shape_per_obs))

    @property
    def dtype(self) -> jnp.dtype:
        return self._data.dtype

    @property
    def support(self) -> Constraint:
        return real

    def __repr__(self) -> str:
        return (
            f"ArrayBootstrapReplicateDistribution(n={self._n}, "
            f"source_n={self._source_n}, "
            f"obs_shape={self._event_shape_per_obs})"
        )
