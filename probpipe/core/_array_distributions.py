"""Array-based distribution hierarchy and helpers.

Provides:
  - ``_vmap_sample()``             – Batched sampling via ``jax.vmap``.
  - ``_mc_expectation()``          – Monte Carlo expectation helper.
  - ``PyTreeArrayDistribution[T]`` – Pytree-of-arrays layer with shape semantics.
  - ``ArrayDistribution``          – Single-array specialization (T = Array).
  - ``BootstrapDistribution``      – MC error tracking via bootstrap resampling.
  - ``FlattenedView``              – Wraps any ``PyTreeArrayDistribution`` as a flat
                                     ``ArrayDistribution``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from .._utils import prod
from .protocols import (
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)

import jax
import jax.numpy as jnp

from ..custom_types import Array, ArrayLike, PRNGKey
from .._weights import Weights, weighted_mean, weighted_variance
from .constraints import (
    Constraint,
    _supports_compatible,
    real,
)
from . import _distribution_base as _base
from .._utils import _auto_key
from ._distribution_base import Distribution


# ---------------------------------------------------------------------------
# Sampling & expectation helpers
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
    n = num_evaluations if num_evaluations is not None else _base.DEFAULT_NUM_EVALUATIONS
    if key is None:
        key = _auto_key()
    samples = dist._sample(key, sample_shape=(n,))
    evals = jax.vmap(f)(samples)

    rd = return_dist if return_dist is not None else _base.RETURN_APPROX_DIST
    if rd:
        return BootstrapDistribution(evals, name="E[f(X)]")
    return jax.tree.map(lambda v: jnp.mean(v, axis=0), evals)


# ---------------------------------------------------------------------------
# PyTreeArrayDistribution[T] — pytree of arrays with shape semantics
# ---------------------------------------------------------------------------

class PyTreeArrayDistribution[T](Distribution[T]):
    """
    Distribution over a pytree of arrays with TFP-style shape semantics.

    The type parameter ``T`` represents the **pytree structure** of a
    single sample — for example, ``dict[str, Array]`` for joint
    distributions or plain ``Array`` for single-array distributions.

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
    weights : array-like, :class:`~probpipe.Weights`, or None
        Non-negative weights (normalized internally).  A pre-built
        :class:`~probpipe.Weights` object is also accepted.  Mutually
        exclusive with *log_weights*.  When neither is given, uniform
        weights are used.
    log_weights : array-like, :class:`~probpipe.Weights`, or None
        Log-unnormalized weights.  A pre-built :class:`~probpipe.Weights`
        object is also accepted.  Mutually exclusive with *weights*.
    name : str, optional
        Distribution name.
    """

    def __init__(
        self,
        evaluations: ArrayLike,
        *,
        weights: ArrayLike | Weights | None = None,
        log_weights: ArrayLike | Weights | None = None,
        name: str | None = None,
    ):
        self._evaluations = jnp.asarray(evaluations, dtype=jnp.float32)
        if self._evaluations.ndim == 0:
            raise ValueError("evaluations must have at least 1 dimension.")
        self._n = self._evaluations.shape[0]
        self._w = Weights(n=self._n, weights=weights, log_weights=log_weights)
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
        return self._w.mean(self._evaluations)

    def _variance(self) -> Array:
        """Variance of the sampling distribution (approx Var[f(X)] / n_eff)."""
        sample_var = self._w.variance(self._evaluations)
        return sample_var / self._w.effective_sample_size

    def _sample_one(self, key: PRNGKey) -> Array:
        """Draw a single bootstrap resample of the mean."""
        idx = self._w.choice(key, shape=(self._n,))
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
            idx = self._w.choice(k, shape=(self._n,))
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
