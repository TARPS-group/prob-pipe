"""
Core distribution abstractions for ProbPipe.

Provides:
  - ``Distribution``                – Generic base class parameterized by value type T.
  - ``PyTreeArrayDistribution``     – Pytree-of-arrays layer with batch/event shape
                                      semantics, sampling, expectations, flatten/unflatten,
                                      and flat-view interop.
  - ``ArrayDistribution``           – Single-array specialization (T = Array) with TFP
                                      shape conventions. All standard distributions
                                      (Normal, Gamma, MVN, …) inherit from this.
  - ``TFPDistribution``             – Re-exported from ``distributions/_tfp_base``.
  - ``EmpiricalDistribution``       – Weighted set of samples.
  - ``BootstrapDistribution``       – MC error tracking via bootstrap resampling.
  - ``FlattenedView``               – Wraps any ``PyTreeArrayDistribution`` as a flat
                                      ``ArrayDistribution`` for algorithm interoperability.

See also:
  - :mod:`probpipe.core.provenance` for ``Provenance``.
  - :mod:`probpipe.core.constraints` for ``Constraint`` and support singletons.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

from .._utils import prod
from .protocols import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsNamedComponents,
    SupportsSampling,
    SupportsVariance,
)

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ..custom_types import Array, ArrayLike, PRNGKey
from .provenance import Provenance
from .constraints import (
    Constraint,
    _supports_compatible,
    real,
    positive,
    non_negative,
    non_negative_integer,
    boolean,
    unit_interval,
    simplex,
    positive_definite,
    sphere,
    interval,
    greater_than,
    integer_interval,
)

# ---------------------------------------------------------------------------
# Type variables
# ---------------------------------------------------------------------------

T = TypeVar('T')
T_out = TypeVar('T_out')

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_NUM_EVALUATIONS: int = 1024
"""Default number of function evaluations for sample-based expectations."""

RETURN_APPROX_DIST: bool = True
"""When True, approximate expectations return a BootstrapDistribution
capturing MC error instead of a plain array."""


def set_default_num_evaluations(n: int) -> None:
    """Set the global default for ``expectation()`` on infinite-support distributions."""
    global DEFAULT_NUM_EVALUATIONS
    if n < 1:
        raise ValueError("num_evaluations must be at least 1")
    DEFAULT_NUM_EVALUATIONS = n


def set_return_approx_dist(value: bool) -> None:
    """Set whether approximate expectations return error-tracking distributions."""
    global RETURN_APPROX_DIST
    RETURN_APPROX_DIST = bool(value)


# ---------------------------------------------------------------------------
# Auto-key helper (for convenience when key is omitted)
# ---------------------------------------------------------------------------

_AUTO_KEY_COUNTER: int = 0


def _auto_key() -> PRNGKey:
    """Generate a JAX PRNGKey from a global counter.

    Convenient for interactive / exploratory use.  Not reproducible
    across runs — pass an explicit key when reproducibility matters.
    """
    global _AUTO_KEY_COUNTER
    key = jax.random.PRNGKey(_AUTO_KEY_COUNTER)
    _AUTO_KEY_COUNTER += 1
    return key


# ---------------------------------------------------------------------------
# Sampling & expectation helpers (used by concrete distribution classes)
# ---------------------------------------------------------------------------

def _vmap_sample(
    dist,
    key: PRNGKey,
    sample_shape: tuple[int, ...] = (),
) -> Any:
    """Draw samples via ``jax.vmap`` over ``dist._sample_one``.

    Suitable for any distribution whose ``_sample_one(key)`` draws a
    single sample as an array or pytree of arrays.

    Parameters
    ----------
    dist
        Distribution with a ``_sample_one(key)`` method.
    key : PRNGKey
        JAX PRNG key.
    sample_shape : tuple of int
        Shape prefix for independent draws.
    """
    if sample_shape == ():
        return dist._sample_one(key)
    n = prod(sample_shape)
    keys = jax.random.split(key, n)
    flat_samples = jax.vmap(dist._sample_one)(keys)
    return jax.tree.map(
        lambda x: x.reshape(*sample_shape, *x.shape[1:]),
        flat_samples,
    )


def _mc_expectation(
    dist,
    f: Callable,
    *,
    key: PRNGKey | None = None,
    num_evaluations: int | None = None,
    return_dist: bool | None = None,
) -> Any:
    """Estimate ``E[f(X)]`` where ``X ~ dist`` via Monte Carlo.

    Parameters
    ----------
    dist
        Distribution with a ``_sample(key, sample_shape)`` method.
    f : callable
        Function mapping a single sample to an array (or pytree of arrays).
    key : PRNGKey, optional
        JAX PRNG key for sampling.  Auto-generated if ``None``.
    num_evaluations : int, optional
        Number of samples to draw.  If ``None``, uses
        ``DEFAULT_NUM_EVALUATIONS``.
    return_dist : bool, optional
        If ``True``, return a ``BootstrapDistribution`` capturing
        estimation uncertainty.  If ``False``, return a plain array.
        If ``None``, use the global ``RETURN_APPROX_DIST`` setting.
    """
    n = num_evaluations if num_evaluations is not None else DEFAULT_NUM_EVALUATIONS
    if key is None:
        key = _auto_key()
    samples = dist._sample(key, sample_shape=(n,))
    evals = jax.vmap(f)(samples)

    rd = return_dist if return_dist is not None else RETURN_APPROX_DIST
    if rd:
        return BootstrapDistribution(evals, name="E[f(X)]")
    return jax.tree.map(lambda v: jnp.mean(v, axis=0), evals)


# ---------------------------------------------------------------------------
# Distribution[T] — generic base class
# ---------------------------------------------------------------------------

class Distribution(Generic[T], ABC):
    """
    Abstract base for all ProbPipe distributions, parameterized by
    value type ``T``.

    Provides naming, provenance, conversion, and approximation tracking.
    Sampling and expectation capabilities are provided by the
    :class:`~probpipe.core.protocols.SupportsSampling` protocol.
    """

    # -- approximation tracking ---------------------------------------------

    @property
    def is_approximate(self) -> bool:
        """Whether this distribution is an approximation (e.g., from sampling or MCMC)."""
        return getattr(self, "_approximate", False)

    # -- naming & provenance ------------------------------------------------

    @property
    def name(self) -> str | None:
        return getattr(self, "_name", None)

    @property
    def source(self) -> Provenance | None:
        return getattr(self, "_source", None)

    def with_source(self, source: Provenance) -> Distribution:
        """Attach provenance to this distribution (write-once)."""
        if getattr(self, "_source", None) is not None:
            raise RuntimeError(
                f"Source already set on {self!r}. "
                "Provenance is write-once; create a new distribution instead."
            )
        self._source = source
        return self

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [type(self).__name__]
        if self.name:
            parts.append(f"name={self.name!r}")
        return f"{parts[0]}({', '.join(parts[1:])})"


# ---------------------------------------------------------------------------
# PyTreeArrayDistribution[T] — pytree of arrays with shape semantics
# ---------------------------------------------------------------------------

class PyTreeArrayDistribution(Distribution[T]):
    """
    Distribution over a pytree of arrays with TFP-style shape semantics.

    All leaves are JAX arrays. ``batch_shape`` is shared across all leaves.
    Each leaf has its own ``event_shape``. The full shape identity is:
    ``treedef + batch_shape + per-leaf event_shapes``.

    This class is purely about shape semantics and pytree structure.
    Concrete subclasses declare their capabilities via protocol inheritance
    (e.g., ``SupportsSampling``, ``SupportsExpectation``).

    This is the workhorse layer for distributions over structured values
    (e.g., dicts of arrays for joint parameter distributions). Single-array
    distributions use the ``ArrayDistribution`` subclass.
    """

    # -- pytree structure ---------------------------------------------------

    @property
    @abstractmethod
    def treedef(self) -> jax.tree_util.PyTreeDef:
        """The pytree structure of a single sample."""
        ...

    # -- shape properties ---------------------------------------------------

    @property
    @abstractmethod
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape, shared across all leaves."""
        ...

    @property
    @abstractmethod
    def event_shapes(self) -> T:
        """Per-leaf event shapes.

        Returns a pytree with the same structure as T, with
        ``tuple[int, ...]`` at each leaf position.
        """
        ...

    @property
    def flat_event_shapes(self) -> list[tuple[int, ...]]:
        """Event shapes as a flat list in canonical leaf order."""
        return self.treedef.flatten_up_to(self.event_shapes)

    @property
    def event_size(self) -> int:
        """Total number of scalar elements in one event (analogous to ``numpy.ndarray.size``).

        Equal to the sum of ``prod(s)`` over all leaf event shapes.
        """
        return sum(prod(s) for s in self.flat_event_shapes)

    # -- flatten / unflatten ------------------------------------------------

    def flatten_value(self, value: T) -> Array:
        """Flatten the event dimensions of a pytree value into a single trailing axis.

        Each leaf's event dimensions are raveled and the results are
        concatenated in canonical leaf order.  All leading dimensions
        (``sample_shape``, ``batch_shape``, or both) are preserved.

        Shape contract for each leaf::

            (*sample_shape, *batch_shape, *event_shape)
            → (*sample_shape, *batch_shape, prod(event_shape))

        After concatenating across leaves the result has shape
        ``(*sample_shape, *batch_shape, event_size)``.
        """
        leaves = jax.tree.leaves(value)
        event_shapes = self.flat_event_shapes
        flat_leaves = []
        for leaf, es in zip(leaves, event_shapes):
            leaf = jnp.asarray(leaf)
            n_event = prod(es)
            # Reshape: (*batch_dims, *event_shape) -> (*batch_dims, n_event)
            n_event_dims = len(es)
            batch_dims = leaf.shape[:leaf.ndim - n_event_dims] if n_event_dims else leaf.shape
            flat_leaves.append(leaf.reshape(*batch_dims, n_event))
        return jnp.concatenate(flat_leaves, axis=-1)

    def unflatten_value(self, flat: Array) -> T:
        """Unflatten a flat array back to the pytree structure.

        Reverses :meth:`flatten_value`.  All dimensions preceding the
        final ``event_size`` axis are preserved (whether they represent
        ``sample_shape``, ``batch_shape``, or both).

        Parameters
        ----------
        flat : Array
            Array of shape ``(*sample_shape, *batch_shape, event_size)``.

        Returns
        -------
        T
            Pytree with each leaf reshaped to
            ``(*sample_shape, *batch_shape, *event_shape)``.
        """
        event_shapes = self.flat_event_shapes
        batch_dims = flat.shape[:-1]
        leaves = []
        offset = 0
        for es in event_shapes:
            n = prod(es)
            chunk = flat[..., offset:offset + n]
            leaves.append(chunk.reshape(*batch_dims, *es))
            offset += n
        return jax.tree.unflatten(self.treedef, leaves)

    def as_flat_distribution(self) -> ArrayDistribution:
        """View this distribution as a flat ``ArrayDistribution``.

        Returns an ``ArrayDistribution`` with ``event_shape=(event_size,)``.
        Enables interoperability with algorithms that expect flat vectors
        (MCMC, optimizers, VI methods).
        """
        return FlattenedView(self)

    # -- supports (pytree of per-leaf constraints) ---------------------------

    @property
    def supports(self) -> T:
        """Per-leaf constraints. Default returns ``real`` for every leaf."""
        return jax.tree.map(lambda _: real, self.event_shapes)


# ---------------------------------------------------------------------------
# ArrayDistribution — distribution over arrays (TFP shape semantics)
# ---------------------------------------------------------------------------

class ArrayDistribution(PyTreeArrayDistribution[Array]):
    """
    Distribution over a single array with TFP-style shape semantics.

    Shape semantics follow TFP conventions:

    * ``event_shape``  -- shape of a single draw (e.g. ``(d,)`` for a
      *d*-dimensional vector distribution).
    * ``batch_shape``  -- shape of independent-but-not-identically-distributed
      parameter batches.
    * ``_sample(key, sample_shape)`` returns an array of shape
      ``sample_shape + batch_shape + event_shape``.

    Standard distributions (Normal, Gamma, Poisson, etc.) inherit from this class.
    """

    # -- shape properties ---------------------------------------------------

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

    # -- PyTreeArrayDistribution interface (trivial single-leaf) -------------

    @property
    def treedef(self) -> jax.tree_util.PyTreeDef:
        """Treedef for a single array (one-leaf pytree)."""
        return jax.tree.structure(None)

    @property
    def event_shapes(self) -> Array:
        """Single event shape, wrapped to satisfy the pytree interface."""
        return self.event_shape

    @property
    def flat_event_shapes(self) -> list[tuple[int, ...]]:
        """Single-leaf: just the one event_shape."""
        return [self.event_shape]

    @property
    def event_size(self) -> int:
        """Total flat dimensionality."""
        es = self.event_shape
        return prod(es)

    def flatten_value(self, value: ArrayLike) -> Array:
        """Flatten event dimensions into a single trailing axis.

        Leading dimensions (``sample_shape``, ``batch_shape``) are
        preserved::

            (*sample_shape, *batch_shape, *event_shape)
            → (*sample_shape, *batch_shape, prod(event_shape))
        """
        value = jnp.asarray(value)
        es = self.event_shape
        n_event = prod(es)
        if not es:
            # scalar event: just add a trailing dim
            return value[..., None]
        n_batch = value.ndim - len(es)
        batch_dims = value.shape[:n_batch]
        return value.reshape(*batch_dims, n_event)

    def unflatten_value(self, flat: ArrayLike) -> Array:
        """Unflatten a flat trailing axis back to event dimensions.

        Leading dimensions (``sample_shape``, ``batch_shape``) are
        preserved::

            (*sample_shape, *batch_shape, event_size)
            → (*sample_shape, *batch_shape, *event_shape)
        """
        flat = jnp.asarray(flat)
        es = self.event_shape
        if not es:
            return flat[..., 0]
        batch_dims = flat.shape[:-1]
        return flat.reshape(*batch_dims, *es)

    # -- support ------------------------------------------------------------

    @property
    def support(self) -> Constraint:
        """The support of this distribution (set of values with non-zero density)."""
        raise NotImplementedError(f"{type(self).__name__}.support")

    @property
    def supports(self) -> Constraint:
        """For ArrayDistribution, supports is just the singular support."""
        return self.support

    # -- support compatibility (for conversion) ------------------------------

    @classmethod
    def _check_support_compatible(cls, other: ArrayDistribution) -> None:
        """Raise ValueError if *other*'s support is incompatible with *cls*."""
        try:
            target_support = cls._default_support()
        except NotImplementedError:
            return  # can't check if target doesn't declare support
        try:
            source_support = other.support
        except NotImplementedError:
            return  # can't check if source doesn't declare support

        if not _supports_compatible(source_support, target_support):
            raise ValueError(
                f"Cannot convert {type(other).__name__} (support={source_support}) "
                f"to {cls.__name__} (support={target_support}). "
                f"Pass check_support=False to override."
            )

    @classmethod
    def _default_support(cls) -> Constraint:
        """Return the default support for this distribution class.

        Override in subclasses. Used by ``_check_support_compatible``
        when no instance is available yet.
        """
        raise NotImplementedError

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
# TFPDistribution – re-exported for backward compatibility
# ---------------------------------------------------------------------------
# The implementation lives in ``distributions/_tfp_base.py`` since no
# ``core/`` module uses it.  Placed here (after all core symbols are
# defined) to avoid circular-import issues.

from ..distributions._tfp_base import TFPDistribution  # noqa: E402


# ---------------------------------------------------------------------------
# EmpiricalDistribution (generic base)
# ---------------------------------------------------------------------------

class EmpiricalDistribution(
    Distribution[T],
    SupportsSampling,
    SupportsExpectation,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
):
    """
    Weighted empirical distribution over a finite set of samples.

    This is the generic base class.  It stores samples as JAX arrays and
    provides weighted resampling, moments, and expectations.  It does
    **not** inherit from :class:`ArrayDistribution` — use
    :class:`ArrayEmpiricalDistribution` when TFP-style shape semantics
    (``batch_shape``, ``flatten_value``, ``support``, etc.) are required.

    Parameters
    ----------
    samples : array-like, shape ``(n, *event_shape)``
        The support points.  The leading axis is the sample axis.
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
        samples: ArrayLike,
        weights: ArrayLike | None = None,
        *,
        log_weights: ArrayLike | None = None,
        name: str | None = None,
    ):
        samples = jnp.asarray(samples, dtype=jnp.float32)
        if samples.ndim == 0:
            raise ValueError("samples must have at least 1 dimension (the sample axis).")

        n = samples.shape[0]

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

        self._samples = samples
        self._weights_cache: Array | None = None
        self._name = name
        self._approximate = True

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    # -- properties ---------------------------------------------------------

    @property
    def n(self) -> int:
        return self._samples.shape[0]

    @property
    def dim(self) -> int:
        """Flat dimensionality of each sample (product of event_shape, or 1 for scalars)."""
        return max(1, prod(self._samples.shape[1:]))

    @property
    def samples(self) -> Array:
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

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._samples.shape[1:]

    @property
    def dtype(self) -> jnp.dtype:
        return self._samples.dtype

    # -- sampling -----------------------------------------------------------

    def _sample_one(self, key: PRNGKey) -> Array:
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
    ) -> Array:
        """Draw samples using efficient batched resampling."""
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
        return draws.reshape(sample_shape + self.event_shape)

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

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Array | ArrayDistribution:
        """Compute ``E[f(X)]`` exactly over the empirical support.

        When ``num_evaluations`` is ``None``, the expectation is computed
        exactly as a weighted sum over all support points (always returns
        ``Array``).  When ``num_evaluations`` is specified and smaller
        than ``self.n``, a random subsample is used and ``return_dist``
        controls whether a ``BootstrapDistribution`` is returned.
        """
        if num_evaluations is not None and num_evaluations < self.n:
            # Subsample — this is approximate
            if key is None:
                key = _auto_key()
            idx = jax.random.choice(key, self.n, shape=(num_evaluations,), replace=False)
            sub_samples = self._samples[idx]
            f_vals = jax.vmap(f)(sub_samples)

            rd = return_dist if return_dist is not None else RETURN_APPROX_DIST
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

        # Exact: evaluate f on all support points — always returns Array
        f_vals = jax.vmap(f)(self._samples)
        if self._is_uniform:
            return jnp.mean(f_vals, axis=0)
        return jnp.einsum("n,n...->...", self.weights, f_vals)


# ---------------------------------------------------------------------------
# ArrayEmpiricalDistribution
# ---------------------------------------------------------------------------

class ArrayEmpiricalDistribution(EmpiricalDistribution[Array], ArrayDistribution):
    """Empirical distribution with full :class:`ArrayDistribution` shape semantics.

    Inherits all functionality from :class:`EmpiricalDistribution` and adds
    TFP-style shape properties (``batch_shape``, ``flatten_value``,
    ``unflatten_value``, ``support``, etc.) via :class:`ArrayDistribution`.

    Use this instead of :class:`EmpiricalDistribution` when the distribution
    must interoperate with :class:`JointDistribution` components or other
    code that requires :class:`ArrayDistribution` instances.
    """

    @property
    def support(self) -> Constraint:
        return real  # empirical samples can be any real values



# ---------------------------------------------------------------------------
# BootstrapDistribution
# ---------------------------------------------------------------------------

class BootstrapDistribution(ArrayDistribution, SupportsSampling, SupportsMean, SupportsVariance):
    """Distribution over bootstrap-resampled means of a statistic.

    Given *n* evaluations ``f(x_1), ..., f(x_n)`` where ``x_i ~ P``,
    this represents the sampling distribution of the sample mean
    ``(1/n) sum f(x_i)``, capturing Monte Carlo error.

    Parameters
    ----------
    evaluations : array-like, shape ``(n, *stat_shape)``
        The individual ``f(x_i)`` values.
    weights : array-like, shape ``(n,)``, optional
        Non-negative weights (normalised internally).  When ``None``,
        uniform weights are used.
    name : str, optional
        Distribution name.
    """

    def __init__(
        self,
        evaluations: ArrayLike,
        *,
        weights: ArrayLike | None = None,
        name: str | None = None,
    ):
        self._evaluations = jnp.asarray(evaluations, dtype=jnp.float32)
        if self._evaluations.ndim == 0:
            raise ValueError("evaluations must have at least 1 dimension.")
        self._n = self._evaluations.shape[0]

        if weights is not None:
            w = jnp.asarray(weights, dtype=jnp.float32)
            self._weights = w / jnp.sum(w)
        else:
            self._weights = None

        self._name = name
        self._approximate = True

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    @property
    def n(self) -> int:
        """Number of function evaluations."""
        return self._n

    @property
    def evaluations(self) -> Array:
        return self._evaluations

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._evaluations.shape[1:]

    def _mean(self) -> Array:
        """Point estimate: (weighted) mean of evaluations."""
        if self._weights is None:
            return jnp.mean(self._evaluations, axis=0)
        return jnp.einsum("n,n...->...", self._weights, self._evaluations)

    def _variance(self) -> Array:
        """Variance of the sampling distribution (approx Var[f(X)] / n_eff)."""
        mu = self._mean()
        diff = self._evaluations - mu
        if self._weights is None:
            sample_var = jnp.mean(diff ** 2, axis=0)
            return sample_var / self._n
        # Weighted variance / effective sample size
        sample_var = jnp.einsum("n,n...->...", self._weights, diff ** 2)
        n_eff = 1.0 / jnp.sum(self._weights ** 2)
        return sample_var / n_eff

    def _sample_one(self, key: PRNGKey) -> Array:
        """Draw a single bootstrap resample of the mean."""
        if self._weights is None:
            idx = jax.random.choice(key, self._n, shape=(self._n,), replace=True)
            return jnp.mean(self._evaluations[idx], axis=0)
        else:
            idx = jax.random.choice(
                key, self._n, shape=(self._n,), replace=True, p=self._weights
            )
            return jnp.mean(self._evaluations[idx], axis=0)

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw bootstrap resamples of the mean."""
        if sample_shape == ():
            return self._sample_one(key)
        total = prod(sample_shape)
        keys = jax.random.split(key, total)

        def _one_resample(k):
            if self._weights is None:
                idx = jax.random.choice(k, self._n, shape=(self._n,), replace=True)
                return jnp.mean(self._evaluations[idx], axis=0)
            else:
                idx = jax.random.choice(
                    k, self._n, shape=(self._n,), replace=True, p=self._weights
                )
                return jnp.mean(self._evaluations[idx], axis=0)

        results = jax.vmap(_one_resample)(keys)
        return results.reshape(sample_shape + self.event_shape)

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

    @property
    def support(self) -> Constraint:
        return real

    def __repr__(self) -> str:
        return f"BootstrapDistribution(n={self._n}, event_shape={self.event_shape})"

# ---------------------------------------------------------------------------
# FlattenedView — wrap a PyTreeArrayDistribution as an ArrayDistribution
# ---------------------------------------------------------------------------

class FlattenedView(ArrayDistribution, SupportsSampling, SupportsLogProb):
    """Wraps a ``PyTreeArrayDistribution`` as a flat ``ArrayDistribution``.

    Sampling produces flat vectors of shape ``(event_size,)``, and
    ``_log_prob`` accepts flat vectors and delegates to the wrapped
    distribution after unflattening.

    This is the primary interoperability mechanism: any algorithm written
    for ``ArrayDistribution`` works with ``PyTreeArrayDistribution`` via
    ``dist.as_flat_distribution()``.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(self, base: PyTreeArrayDistribution):
        self._base = base

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._base.event_size,)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._base.batch_shape

    def _sample_one(self, key: PRNGKey) -> Array:
        pytree_sample = self._base._sample_one(key)
        return self._base.flatten_value(pytree_sample)

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        pytree_samples = self._base._sample(key, sample_shape)
        return self._base.flatten_value(pytree_samples)

    def _log_prob(self, x: ArrayLike) -> Array:
        x = jnp.asarray(x)
        value = self._base.unflatten_value(x)
        return self._base._log_prob(value)

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

    @property
    def support(self) -> Constraint:
        return real

    @property
    def base_distribution(self) -> PyTreeArrayDistribution:
        """The underlying pytree distribution."""
        return self._base

    def unflatten_sample(self, flat_sample: ArrayLike):
        """Convenience: unflatten a flat sample back to the pytree structure."""
        return self._base.unflatten_value(jnp.asarray(flat_sample))

    def __repr__(self) -> str:
        return (
            f"FlattenedView(base={type(self._base).__name__}, "
            f"event_shape={self.event_shape})"
        )


# ---------------------------------------------------------------------------
# JointBootstrapDistribution
# ---------------------------------------------------------------------------


class JointBootstrapDistribution(ArrayDistribution, SupportsSampling):
    """N-fold product of an empirical distribution (bootstrap resampling).

    Each sample from this distribution is a bootstrapped dataset — ``n``
    observations drawn i.i.d. with replacement from the source data.
    This provides the sampling distribution over datasets needed for
    BayesBag (bagged posteriors).

    When the source is an :class:`EmpiricalDistribution`, ``n`` defaults
    to the number of samples in the empirical distribution.  Otherwise
    ``n`` must be specified explicitly.

    Parameters
    ----------
    source : EmpiricalDistribution or array-like
        The data to bootstrap from.  If an ``EmpiricalDistribution``,
        its samples and weights are used directly.  If array-like,
        it is treated as an equally-weighted dataset where the leading
        axis is the observation axis.
    n : int or None
        Number of observations per bootstrap dataset.  Defaults to the
        number of samples in ``source`` when ``source`` is an
        ``EmpiricalDistribution`` or array.
    name : str or None
        Distribution name for provenance.

    Examples
    --------
    >>> data = EmpiricalDistribution(observed_data)
    >>> bootstrap = JointBootstrapDistribution(data)
    >>> # Each sample is a bootstrapped dataset of the same size
    >>> boot_dataset = sample(bootstrap, key=jax.random.PRNGKey(0))
    >>> boot_dataset.shape  # (n, *event_shape)
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        source: EmpiricalDistribution | ArrayLike,
        *,
        n: int | None = None,
        name: str | None = None,
    ):
        if isinstance(source, EmpiricalDistribution):
            self._data = source.samples
            self._weights = None if source.is_uniform else source.weights
            default_n = source.n
        else:
            self._data = jnp.asarray(source, dtype=jnp.float32)
            if self._data.ndim == 0:
                raise ValueError("source must have at least 1 dimension (the observation axis).")
            self._weights = None
            default_n = self._data.shape[0]

        if n is None:
            self._n = default_n
        else:
            if n < 1:
                raise ValueError(f"n must be positive, got {n}")
            self._n = n

        self._name = name
        self._source_n = self._data.shape[0]
        self._event_shape_per_obs = self._data.shape[1:]

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of a single bootstrap dataset: ``(n, *obs_shape)``."""
        return (self._n, *self._event_shape_per_obs)

    @property
    def n(self) -> int:
        """Number of observations per bootstrap dataset."""
        return self._n

    def _sample_one(self, key: PRNGKey) -> Array:
        """Draw a single bootstrapped dataset."""
        if self._weights is None:
            idx = jax.random.choice(key, self._source_n, shape=(self._n,), replace=True)
        else:
            idx = jax.random.choice(
                key, self._source_n, shape=(self._n,), replace=True, p=self._weights,
            )
        return self._data[idx]

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw bootstrap datasets.

        Returns shape ``(*sample_shape, n, *event_shape_per_obs)``.
        """
        if sample_shape == ():
            return self._sample_one(key)

        total = prod(sample_shape)
        keys = jax.random.split(key, total)
        results = jax.vmap(self._sample_one)(keys)
        return results.reshape(*sample_shape, self._n, *self._event_shape_per_obs)

    def __repr__(self) -> str:
        return (
            f"JointBootstrapDistribution(n={self._n}, "
            f"source_n={self._source_n}, "
            f"obs_shape={self._event_shape_per_obs})"
        )


# ---------------------------------------------------------------------------
# MarginalizedBroadcastDistribution — output marginal of a broadcast
# ---------------------------------------------------------------------------
#
# Protocol support is determined dynamically via a factory function that
# picks the right concrete subclass, so ``isinstance`` checks are truthful.
# ---------------------------------------------------------------------------


class _ArrayMarginal(ArrayEmpiricalDistribution):
    """Output marginal when broadcast outputs are stackable arrays.

    Inherits from :class:`ArrayEmpiricalDistribution` for weighted
    resampling and exact weighted moments.
    """

    def __init__(
        self,
        samples: Array,
        weights: Array | None,
        *,
        name: str | None = None,
    ):
        # _ArrayMarginal receives pre-normalised weights (or None).
        # Pass via log_weights to skip the non-negativity / sum validation
        # in EmpiricalDistribution.__init__.
        if weights is not None:
            log_weights = jnp.log(jnp.asarray(weights, dtype=jnp.float32))
            super().__init__(samples, log_weights=log_weights, name=name)
        else:
            super().__init__(samples, name=name)

    def __repr__(self):
        return f"MarginalizedBroadcastDistribution(n={self.n}, event_shape={self.event_shape})"


class _MixtureMarginal(Distribution[T]):
    """Output marginal when broadcast outputs are Distribution objects.

    Acts as a finite mixture: ``p(y) = Σ_i w_i p_i(y)``.  Protocol support
    depends on what the component distributions support.

    This base class provides no protocol methods.  The factory
    :func:`_make_mixture_marginal` dynamically constructs a subclass that
    inherits the appropriate protocol mixins.
    """

    def __init__(
        self,
        components: list,
        weights: Array | None,
        *,
        name: str | None = None,
    ):
        n = len(components)
        self._components = components
        self._w = weights if weights is not None else jnp.ones(n, dtype=jnp.float32) / n
        self._name = name
        self._approximate = True

    @property
    def n(self) -> int:
        return len(self._components)

    @property
    def components(self) -> list:
        return self._components

    @property
    def weights(self) -> Array:
        return self._w

    def __repr__(self):
        return f"MarginalizedBroadcastDistribution(mixture, n={self.n})"


# -- Mixture protocol mixins (combined dynamically) -------------------------

class _MixtureSampling:
    """SupportsSampling mixin for mixture marginals."""

    _sampling_cost: str = "medium"
    _preferred_orchestration: str | None = None

    def _sample(self, key, sample_shape=()):
        n_draws = prod(sample_shape) if sample_shape else 1
        key1, key2 = jax.random.split(key)
        indices = jax.random.choice(key1, self.n, shape=(n_draws,), p=self.weights)
        keys = jax.random.split(key2, n_draws)

        results = []
        for i in range(n_draws):
            comp = self._components[int(indices[i])]
            results.append(comp._sample(keys[i], ()))
        stacked = jnp.stack(results, axis=0)
        if sample_shape == ():
            return stacked[0]
        return stacked.reshape(sample_shape + stacked.shape[1:])


class _MixtureMean:
    """SupportsMean mixin for mixture marginals."""

    def _mean(self):
        means = jnp.stack([c._mean() for c in self._components], axis=0)
        return jnp.einsum("n,n...->...", self.weights, means)


class _MixtureVariance:
    """SupportsVariance mixin for mixture marginals (law of total variance)."""

    def _variance(self):
        means = jnp.stack([c._mean() for c in self._components], axis=0)
        variances = jnp.stack([c._variance() for c in self._components], axis=0)
        overall_mean = jnp.einsum("n,n...->...", self.weights, means)
        # Law of total variance: E[Var(Y|X)] + Var(E[Y|X])
        e_var = jnp.einsum("n,n...->...", self.weights, variances)
        diff = means - overall_mean
        var_e = jnp.einsum("n,n...->...", self.weights, diff ** 2)
        return e_var + var_e


class _MixtureLogProb:
    """SupportsLogProb mixin for mixture marginals."""

    def _log_prob(self, value):
        log_w = jnp.log(self.weights)
        component_lps = jnp.stack(
            [c._log_prob(value) for c in self._components], axis=0
        )
        return jax.scipy.special.logsumexp(log_w + component_lps)


# Map protocol → (mixin class, required component protocols)
_MIXTURE_PROTOCOL_MAP: list[tuple[type, type, tuple[type, ...]]] = [
    (SupportsSampling, _MixtureSampling, (SupportsSampling,)),
    (SupportsMean, _MixtureMean, (SupportsMean,)),
    (SupportsVariance, _MixtureVariance, (SupportsMean, SupportsVariance)),
    (SupportsLogProb, _MixtureLogProb, (SupportsLogProb,)),
]

# Cache dynamically created classes to avoid repeated type() calls
_mixture_class_cache: dict[tuple[type, ...], type] = {}


def _make_mixture_marginal(
    components: list,
    weights: Array | None,
    *,
    name: str | None = None,
) -> _MixtureMarginal:
    """Factory that builds a mixture marginal with dynamic protocol support.

    Inspects the component distributions to determine which protocols they
    all support, then creates (and caches) a concrete subclass that inherits
    the corresponding mixin classes.
    """
    # Determine which protocols all components support
    active_protocols: list[type] = []
    active_mixins: list[type] = []
    for protocol, mixin, required in _MIXTURE_PROTOCOL_MAP:
        if all(isinstance(c, req) for c in components for req in required):
            active_protocols.append(protocol)
            active_mixins.append(mixin)

    cache_key = tuple(active_protocols)
    if cache_key not in _mixture_class_cache:
        bases = tuple(active_mixins) + (_MixtureMarginal,) + tuple(active_protocols)
        cls_name = "_DynMixtureMarginal"
        _mixture_class_cache[cache_key] = type(cls_name, bases, {})

    cls = _mixture_class_cache[cache_key]
    obj = object.__new__(cls)
    _MixtureMarginal.__init__(obj, components, weights, name=name)
    return obj


class _ListMarginal(Distribution[T]):
    """Output marginal when broadcast outputs are non-stackable (e.g., strings).

    No protocol support — outputs cannot be resampled or summarised.
    """

    def __init__(
        self,
        items: list,
        weights: Array | None,
        *,
        name: str | None = None,
    ):
        self._items = items
        self._w = weights
        self._name = name

    @property
    def n(self) -> int:
        return len(self._items)

    @property
    def items(self) -> list:
        return self._items

    @property
    def weights(self) -> Array | None:
        return self._w

    def __repr__(self):
        return f"MarginalizedBroadcastDistribution(list, n={self.n})"


# Public alias for type checking / isinstance
MarginalizedBroadcastDistribution = _ArrayMarginal | _MixtureMarginal | _ListMarginal
"""Union type for the output marginal of a :class:`BroadcastDistribution`.

Concrete subtype depends on output kind:

- :class:`_ArrayMarginal` — stackable array outputs
- :class:`_MixtureMarginal` — distribution outputs (mixture)
- :class:`_ListMarginal` — non-stackable outputs
"""


def _make_marginal(
    output_samples: Any,
    weights: Array | None,
    *,
    output_distributions: list | None = None,
    name: str | None = None,
) -> MarginalizedBroadcastDistribution:
    """Factory to construct the appropriate marginal subtype."""
    if output_distributions is not None:
        return _make_mixture_marginal(output_distributions, weights, name=name)

    # Try stacking into an array
    if isinstance(output_samples, jnp.ndarray):
        return _ArrayMarginal(output_samples, weights, name=name)

    if isinstance(output_samples, list):
        try:
            stacked = jnp.stack(
                [jnp.asarray(r, dtype=jnp.float32) for r in output_samples], axis=0
            )
            return _ArrayMarginal(stacked, weights, name=name)
        except (ValueError, TypeError):
            pass
        # Check if all results are distributions
        if output_samples and all(isinstance(r, Distribution) for r in output_samples):
            return _make_mixture_marginal(output_samples, weights, name=name)
        return _ListMarginal(output_samples, weights, name=name)

    # Single array result (e.g., from vmap); ensure at least 1D for the sample axis
    arr = jnp.atleast_1d(jnp.asarray(output_samples, dtype=jnp.float32))
    return _ArrayMarginal(arr, weights, name=name)


# ---------------------------------------------------------------------------
# BroadcastDistribution — joint over broadcast inputs and function output
# ---------------------------------------------------------------------------


class BroadcastDistribution(Distribution[dict], SupportsSampling, SupportsNamedComponents):
    """Joint distribution over broadcast inputs and function output.

    Stores the paired input–output samples from a
    :class:`~probpipe.core.node.WorkflowFunction` broadcast.  Supports
    joint sampling (resampling paired input–output tuples) and named
    component access.

    Call :meth:`marginalize` to obtain the output-only marginal, which
    supports moment protocols (mean, variance, etc.) when the output
    data permits.

    .. note::

       ``BroadcastDistribution`` does **not** inherit from
       :class:`~probpipe.distributions.joint.JointDistribution`.
       ``JointDistribution`` requires all leaves to be
       ``ArrayDistribution`` instances with TFP shape semantics
       (``batch_shape``, ``event_shape``), but a broadcast output can be
       any type — arrays, distributions, strings, etc. — and input
       samples are plain arrays without distribution metadata.  The two
       hierarchies serve different roles: ``JointDistribution`` models
       structured probabilistic variables; ``BroadcastDistribution``
       records the empirical input–output mapping of a function
       evaluation.

    Parameters
    ----------
    input_samples : dict[str, Array]
        ``{arg_name: (n, *event_shape)}`` for each broadcast argument.
    output_samples : Array or list
        ``(n, *event_shape)`` for array outputs, or a list of length *n*.
    weights : Array or None
        ``(n,)`` normalised weights.  ``None`` for uniform.
    output_distributions : list of Distribution or None
        When each function evaluation returns a ``Distribution``, these
        are the *n* component distributions for the mixture marginal.
    broadcast_args : list of str
        Ordered names of the broadcast arguments.
    name : str or None
        Distribution name for provenance.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        input_samples: dict[str, Any],
        output_samples: Any,
        weights: Array | None,
        *,
        output_distributions: list | None = None,
        broadcast_args: list[str],
        name: str | None = None,
    ):
        self._input_samples = input_samples
        self._output_samples = output_samples
        self._output_distributions = output_distributions

        if weights is not None:
            weights = jnp.asarray(weights, dtype=jnp.float32)
            weights = weights / jnp.sum(weights)
        self._weights = weights
        self._broadcast_args = list(broadcast_args)
        self._name = name
        self._approximate = True
        self._marginal_cache: _ArrayMarginal | _MixtureMarginal | _ListMarginal | None = None

    # -- basic properties ---------------------------------------------------

    @property
    def n(self) -> int:
        """Number of input–output pairs."""
        first_key = self._broadcast_args[0]
        arr = self._input_samples[first_key]
        return arr.shape[0] if hasattr(arr, 'shape') else len(arr)

    @property
    def weights(self) -> Array:
        """Normalised weights, shape ``(n,)``."""
        if self._weights is None:
            return jnp.ones(self.n, dtype=jnp.float32) / self.n
        return self._weights

    @property
    def input_samples(self) -> dict[str, Any]:
        """Broadcast input samples: ``{arg_name: (n, *event_shape)}``."""
        return self._input_samples

    @property
    def samples(self) -> Any:
        """Output samples (forwarded to output marginal for backward compat)."""
        m = self.marginalize()
        return m.samples if hasattr(m, 'samples') else m.items

    # -- SupportsNamedComponents --------------------------------------------

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(self._broadcast_args) + ("_output",)

    def __getitem__(self, key: str):
        if key == "_output":
            return self.marginalize()
        if key in self._input_samples:
            arr = self._input_samples[key]
            return EmpiricalDistribution(arr, weights=self._weights)
        raise KeyError(f"Unknown component {key!r}; available: {self.component_names}")

    # -- joint sampling -----------------------------------------------------

    def _sample(self, key, sample_shape=()):
        """Resample paired input–output tuples."""
        n_draws = prod(sample_shape) if sample_shape else 1
        if self._weights is None:
            indices = jax.random.randint(key, shape=(n_draws,), minval=0, maxval=self.n)
        else:
            indices = jax.random.choice(key, self.n, shape=(n_draws,), p=self.weights, replace=True)

        result = {}
        for arg_name in self._broadcast_args:
            arr = self._input_samples[arg_name]
            result[arg_name] = arr[indices]

        # Output
        if isinstance(self._output_samples, jnp.ndarray):
            result["_output"] = self._output_samples[indices]
        elif isinstance(self._output_samples, list):
            result["_output"] = [self._output_samples[int(i)] for i in indices]
        else:
            result["_output"] = self._output_samples[indices]

        if sample_shape == ():
            return jax.tree.map(lambda x: x[0] if hasattr(x, '__getitem__') else x, result)
        return result

    # -- marginalization ----------------------------------------------------

    def marginalize(self) -> MarginalizedBroadcastDistribution:
        """Return the output marginal distribution.

        Lazy — the marginal is constructed on first call and cached.
        """
        if self._marginal_cache is None:
            self._marginal_cache = _make_marginal(
                self._output_samples,
                self._weights,
                output_distributions=self._output_distributions,
            )
        return self._marginal_cache

    @property
    def output(self) -> MarginalizedBroadcastDistribution:
        """Alias for :meth:`marginalize`."""
        return self.marginalize()

    def __repr__(self):
        return (
            f"BroadcastDistribution(n={self.n}, "
            f"broadcast_args={self._broadcast_args})"
        )

