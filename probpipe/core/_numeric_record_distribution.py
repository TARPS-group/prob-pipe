"""``NumericRecordDistribution`` and its closely-related helpers.

The primary class is :class:`NumericRecordDistribution` — a
:class:`~probpipe.core._record_distribution.RecordDistribution` that
additionally enforces numeric-leaf shape semantics (``event_shape``,
``batch_shape``, ``flat_event_shapes``, ``event_size``) and serves as
the base class for every numeric ProbPipe distribution (``Normal``,
``Beta``, ``ProductDistribution``, ...).

Provides:

  - :class:`NumericRecordDistribution` — the base class.
  - :class:`BootstrapDistribution` — MC error tracking via bootstrap
    resampling.
  - :class:`FlattenedView` — wraps any distribution as a flat-array
    distribution.
  - Private helpers ``_vmap_sample`` / ``_mc_expectation``.

Not to be confused with :mod:`_distribution_array`, which houses
:class:`DistributionArray` — a *collection of n independent scalar
distributions* along a batch axis. ``DistributionArray`` is a
``Distribution`` subclass, not a ``NumericRecordDistribution``
subclass: it represents "many random variables indexed by position",
while ``NumericRecordDistribution`` represents "one random variable
with a numeric-valued event structure".
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
from ._record_distribution import RecordDistribution


# ---------------------------------------------------------------------------
# Sampling & expectation helpers
# ---------------------------------------------------------------------------

def _vmap_sample(
    dist,
    key: PRNGKey,
    sample_shape: tuple[int, ...] = (),
) -> Any:
    """Draw samples via ``jax.vmap`` over ``dist._sample(key, ())``.

    Convenience for distributions whose ``_sample`` implementation is
    naturally a single-draw function: call this helper from ``_sample``
    and it will handle the ``sample_shape`` prefix by splitting keys
    and vmap-ing over the single-draw path.

    Parameters
    ----------
    dist
        Distribution whose ``_sample(key, ())`` draws one unbatched
        sample (array or pytree of arrays).
    key : PRNGKey
        JAX PRNG key.
    sample_shape : tuple of int
        Shape prefix for independent draws.
    """
    def _one(k: PRNGKey) -> Any:
        return dist._sample(k, ())

    if sample_shape == ():
        return _one(key)
    n = prod(sample_shape)
    keys = jax.random.split(key, n)
    flat_samples = jax.vmap(_one)(keys)
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
# NumericRecordDistribution — RecordDistribution + numeric shape semantics
# ---------------------------------------------------------------------------

class NumericRecordDistribution(RecordDistribution):
    """Distribution over numeric arrays with Record support.

    Extends :class:`RecordDistribution` with numeric-specific metadata:
    dtype, support, batch_shape, event_shape.

    Shape semantics follow TFP conventions:

    * ``event_shape``  -- shape of a single draw (e.g. ``(d,)`` for a
      *d*-dimensional vector distribution).
    * ``batch_shape``  -- shape of independent-but-not-identically-distributed
      parameter batches.

    When ``record_template`` is set (named distribution), samples are
    wrapped as :class:`~probpipe.Record`.  Otherwise, raw arrays are
    returned for backward compatibility.

    Standard distributions (Normal, Gamma, Poisson, etc.) inherit from
    this class via :class:`TFPDistribution`.
    """

    # -- per-field metadata ---------------------------------------------------

    @property
    def dtypes(self) -> dict[str, jnp.dtype]:
        """Per-field dtypes.  Default: ``float32`` for every field."""
        tpl = self.record_template
        if tpl is not None:
            return {name: jnp.float32 for name in tpl.fields}
        return {}

    @property
    def supports(self) -> dict[str, Constraint]:
        """Per-field support constraints.

        Subclasses should override to provide meaningful constraints.
        Default raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__}.supports")

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape (default: scalar, no batching)."""
        return ()

    @property
    def dtype(self) -> jnp.dtype | None:
        """Scalar dtype if all fields share one, else ``None``.

        Subclasses like ``TFPDistribution`` override with a concrete dtype.
        """
        per_field = self.dtypes
        if not per_field:
            return None
        unique = set(per_field.values())
        return unique.pop() if len(unique) == 1 else None

    @classmethod
    def _check_support_compatible(cls, other: NumericRecordDistribution) -> None:
        """Raise ValueError if *other*'s support is incompatible with *cls*."""
        try:
            target_support = cls._default_support()
        except NotImplementedError:
            return
        try:
            source_support = other.support
        except NotImplementedError:
            return
        if not _supports_compatible(source_support, target_support):
            raise ValueError(
                f"Cannot convert {type(other).__name__} (support={source_support}) "
                f"to {cls.__name__} (support={target_support}). "
                f"Pass check_support=False to override."
            )

    @classmethod
    def _default_support(cls) -> Constraint:
        """Return the default support for this distribution class.

        Override in subclasses.  Used by ``_check_support_compatible``
        when no instance is available yet.
        """
        raise NotImplementedError

    # -- event_shape (abstract) ---------------------------------------------

    @property
    @abstractmethod
    def event_shape(self) -> tuple[int, ...]:
        ...

    # -- Single-leaf pytree interface -----------------------------------------

    @property
    def treedef(self) -> jax.tree_util.PyTreeDef:
        """Treedef for a single array (one-leaf pytree)."""
        return jax.tree.structure(None)

    @property
    def flat_event_shapes(self) -> list[tuple[int, ...]]:
        """Single-leaf: just the one event_shape."""
        return [self.event_shape]

    @property
    def event_size(self) -> int:
        """Total flat dimensionality."""
        tpl = self.record_template
        if tpl is not None:
            return tpl.flat_size
        return prod(self.event_shape)

    def flatten_value(self, value) -> Array:
        """Flatten a sample (Record, NumericRecordArray, or array) to flat trailing axis.

        Delegates to ``RecordDistribution.flatten_value`` for Record-like
        inputs.  For raw arrays, flattens event dimensions preserving
        leading batch/sample dims.
        """
        from .record import Record as _Values
        from ._record_array import NumericRecordArray
        if isinstance(value, (NumericRecordArray, _Values)):
            return super().flatten_value(value)
        value = jnp.asarray(value)
        es = self.event_shape
        n_event = prod(es)
        if not es:
            return value[..., None]
        n_batch = value.ndim - len(es)
        batch_dims = value.shape[:n_batch]
        return value.reshape(*batch_dims, n_event)

    def unflatten_value(self, flat: ArrayLike):
        """Unflatten a flat trailing axis back to event dims, Record, or NumericRecordArray.

        When ``record_template`` is set with multiple fields, delegates
        to ``RecordDistribution.unflatten_value`` (returns NumericRecord
        or NumericRecordArray).  For single-field leaf distributions,
        reshapes to ``(*batch, *event_shape)`` for ``_log_prob`` compat.
        """
        tpl = self.record_template
        if tpl is not None and len(tpl.fields) > 1:
            return super().unflatten_value(flat)
        flat = jnp.asarray(flat)
        es = self.event_shape
        if not es:
            return flat[..., 0]
        batch_dims = flat.shape[:-1]
        return flat.reshape(*batch_dims, *es)

    def as_flat_distribution(self):
        """View this distribution as a flat distribution.

        Returns a :class:`FlattenedView` wrapping this distribution.
        """
        return FlattenedView(self)

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

class BootstrapDistribution(NumericRecordDistribution, SupportsSampling, SupportsMean, SupportsVariance):
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
        if name is None:
            name = "bootstrap_dist"
        super().__init__(name=name)
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

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw bootstrap resamples of the mean."""
        def _one_resample(k):
            idx = self._w.choice(k, shape=(self._n,))
            return jnp.mean(self._evaluations[idx], axis=0)

        if sample_shape == ():
            return _one_resample(key)
        total = prod(sample_shape)
        keys = jax.random.split(key, total)

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
# FlattenedView — wrap any distribution as a flat NumericRecordDistribution
# ---------------------------------------------------------------------------

_FLATTENED_VIEW_CLASS_CACHE: dict[frozenset[str], type] = {}


def _flattened_view_class_for_base(base: Distribution) -> type:
    """Return a ``FlattenedView`` subclass whose protocol bases match the
    capabilities of *base*.

    ``FlattenedView`` only delegates sampling and log-prob; those are the
    only protocols that make sense to inherit. A ``FlattenedView`` over
    a log-prob-only base should not advertise ``SupportsSampling``, and
    vice versa.
    """
    protocols: set[str] = set()
    if isinstance(base, SupportsSampling):
        protocols.add("sample")
    if isinstance(base, SupportsLogProb):
        protocols.add("log_prob")

    key = frozenset(protocols)
    if key in _FLATTENED_VIEW_CLASS_CACHE:
        return _FLATTENED_VIEW_CLASS_CACHE[key]

    extra_bases: list[type] = []
    extra_methods: dict[str, object] = {}

    if "sample" in protocols:
        extra_bases.append(SupportsSampling)

        def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
            pytree_samples = self._base._sample(key, sample_shape)
            return self._base.flatten_value(pytree_samples)

        extra_methods["_sample"] = _sample

    if "log_prob" in protocols:
        extra_bases.append(SupportsLogProb)

        def _log_prob(self, x: ArrayLike) -> Array:
            x = jnp.asarray(x)
            value = self._base.unflatten_value(x)
            return self._base._log_prob(value)

        extra_methods["_log_prob"] = _log_prob

    if not extra_bases:
        _FLATTENED_VIEW_CLASS_CACHE[key] = FlattenedView
        return FlattenedView

    new_cls = type("FlattenedView", (FlattenedView, *extra_bases), extra_methods)
    _FLATTENED_VIEW_CLASS_CACHE[key] = new_cls
    return new_cls


class FlattenedView(NumericRecordDistribution):
    """Wraps a distribution as a flat ``NumericRecordDistribution``.

    Sampling produces flat vectors of shape ``(event_size,)``, and
    ``_log_prob`` accepts flat vectors and delegates to the wrapped
    distribution after unflattening.

    This is the primary interoperability mechanism: any algorithm written
    for ``NumericRecordDistribution`` works with ``RecordDistribution`` or
    ``NumericRecordDistribution`` via ``dist.as_flat_distribution()``.

    **Dynamic protocol support:** the view's ``isinstance`` compliance
    matches the base's capabilities — a log-prob-only base produces a
    ``FlattenedView`` that is not ``SupportsSampling``, and a
    sampling-only base produces one that is not ``SupportsLogProb``.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __new__(cls, base: Distribution):
        actual_cls = _flattened_view_class_for_base(base)
        return object.__new__(actual_cls)

    def __init__(self, base: Distribution):
        self._base = base

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._base.event_size,)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return getattr(self._base, "batch_shape", ())

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
    def base_distribution(self) -> Distribution:
        """The underlying distribution."""
        return self._base

    def unflatten_sample(self, flat_sample: ArrayLike):
        """Convenience: unflatten a flat sample back to the pytree structure."""
        return self._base.unflatten_value(jnp.asarray(flat_sample))

    def __repr__(self) -> str:
        return (
            f"FlattenedView(base={type(self._base).__name__}, "
            f"event_shape={self.event_shape})"
        )
