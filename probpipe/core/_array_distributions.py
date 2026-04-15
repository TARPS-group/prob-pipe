"""Distribution hierarchy with TFP shape semantics.

Provides:
  - ``_vmap_sample()``           – Batched sampling via ``jax.vmap``.
  - ``_mc_expectation()``        – Monte Carlo expectation helper.
  - ``TFPShapeMixin``            – TFP shape conventions (dtype, support, batch_shape).
  - ``TFPRecordDistribution``    – RecordDistribution + TFP shapes (base for all TFP dists).
  - ``ArrayDistribution``        – Alias for ``TFPRecordDistribution`` (backward compat).
  - ``BootstrapDistribution``    – MC error tracking via bootstrap resampling.
  - ``FlattenedView``            – Wraps any distribution as a flat distribution.
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
# TFPShapeMixin — TFP-specific shape conventions
# ---------------------------------------------------------------------------

class TFPShapeMixin:
    """Mixin providing TFP-style shape conventions: dtype, support, batch_shape.

    Used by both ``ArrayDistribution`` (for generative TFP distributions)
    and ``TFPEmpiricalDistribution`` (for empirical distributions with
    TFP shape semantics).  Does not inherit from any base class.
    """

    @property
    def dtype(self) -> jnp.dtype:
        """Array dtype of samples (default: float32)."""
        return jnp.float32

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape (default: scalar, no batching)."""
        return ()

    @property
    def support(self) -> Constraint:
        """The support of this distribution (set of values with non-zero density)."""
        raise NotImplementedError(f"{type(self).__name__}.support")

    @property
    def supports(self) -> Constraint:
        """Singular support (alias for ``support``)."""
        return self.support

    @classmethod
    def _check_support_compatible(cls, other: TFPShapeMixin) -> None:
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


# ---------------------------------------------------------------------------
# TFPRecordDistribution — RecordDistribution + TFP shape semantics
# ---------------------------------------------------------------------------

class TFPRecordDistribution(RecordDistribution, TFPShapeMixin):
    """Distribution with TFP-style shape semantics and Record support.

    Combines :class:`RecordDistribution` (named component access,
    Record-aware flatten/unflatten) with :class:`TFPShapeMixin` (dtype,
    support, batch_shape).

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

    # -- shape properties ---------------------------------------------------

    @property
    @abstractmethod
    def event_shape(self) -> tuple[int, ...]:
        ...

    # dtype, support, _check_support_compatible, _default_support
    # inherited from TFPShapeMixin.

    # Concrete batch_shape default (no batching).
    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    # -- Single-leaf pytree interface -----------------------------------------

    @property
    def treedef(self) -> jax.tree_util.PyTreeDef:
        """Treedef for a single array (one-leaf pytree)."""
        return jax.tree.structure(None)

    @property
    def event_shapes(self):
        """Per-field event shapes, or single event shape when unnamed."""
        tpl = self.record_template
        if tpl is not None:
            return super().event_shapes  # RecordDistribution dict version
        return self.event_shape

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
        """Flatten a sample (Record or array) to a flat trailing axis.

        When *value* is a :class:`Record`, delegates to
        ``RecordDistribution.flatten_value``.  Otherwise flattens event
        dimensions of a raw array, preserving leading batch/sample dims.
        """
        from .record import Record as _Values
        if isinstance(value, _Values):
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
        """Unflatten a flat trailing axis back to event dimensions or Record.

        When ``record_template`` is set and has multiple fields, returns
        a :class:`Record`.  For single-field leaf distributions, reshapes
        to ``(*batch, *event_shape)`` to stay compatible with ``_log_prob``.
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

    # support, _check_support_compatible, _default_support from TFPShapeMixin.

    @property
    def supports(self):
        """Singular support constraint."""
        return self.support

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


# Backward compatibility alias
ArrayDistribution = TFPRecordDistribution


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
# FlattenedView — wrap any distribution as a flat ArrayDistribution
# ---------------------------------------------------------------------------

class FlattenedView(ArrayDistribution, SupportsSampling, SupportsLogProb):
    """Wraps a distribution as a flat ``ArrayDistribution``.

    Sampling produces flat vectors of shape ``(event_size,)``, and
    ``_log_prob`` accepts flat vectors and delegates to the wrapped
    distribution after unflattening.

    This is the primary interoperability mechanism: any algorithm written
    for ``ArrayDistribution`` works with ``RecordDistribution`` or
    ``ArrayDistribution`` via ``dist.as_flat_distribution()``.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(self, base: Distribution):
        self._base = base

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._base.event_size,)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return getattr(self._base, "batch_shape", ())

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
