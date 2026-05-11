"""``NumericRecordDistribution`` and its closely-related helpers.

The primary class is :class:`NumericRecordDistribution` — a
:class:`~probpipe.core._record_distribution.RecordDistribution` that
additionally enforces numeric-leaf shape semantics (``event_shape``,
``flat_event_shapes``, ``event_size``) and serves as the base class
for every numeric ProbPipe distribution (``Normal``, ``Beta``,
``ProductDistribution``, ...).

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

from .._dtype import _as_float_array, _default_float_dtype
from .._utils import prod
from .protocols import (
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

    Extends :class:`RecordDistribution` with numeric-specific metadata.
    The class is the most general numeric random variable in ProbPipe:
    samples are a pytree of ``jax.Array`` leaves named via
    :class:`RecordTemplate`. Single-leaf distributions (``Normal``,
    ``Beta``, ``MultivariateNormal``, …) are the trivial case; the
    same machinery covers future multi-leaf joint distributions.

    A ``Distribution`` represents one random variable. Collections of
    independent distributions live in
    :class:`~probpipe.DistributionArray`.

    Canonical / convenience accessor pairs
    --------------------------------------

    Per-field accessors (canonical) are the source of truth; scalar
    accessors (convenience) are derived shortcuts that raise on
    multi-leaf templates. Subclasses override the canonical side;
    convenience accessors are inherited and derived automatically.

    +------------+---------------------------------+--------------------------------------+
    | Concept    | Canonical (per-leaf)            | Convenience (single-leaf)            |
    +============+=================================+======================================+
    | Structure  | ``record_template``             | —                                    |
    +------------+---------------------------------+--------------------------------------+
    | Pytree     | ``treedef`` (from template)     | —                                    |
    +------------+---------------------------------+--------------------------------------+
    | Shapes     | ``event_shapes : dict``         | ``event_shape : tuple``              |
    |            |                                 | (raises on multi-leaf)               |
    +------------+---------------------------------+--------------------------------------+
    | Dtypes     | ``dtypes : dict`` (raises if    | ``dtype : dtype | None`` (unique or  |
    |            | not declared)                   | ``None``)                            |
    +------------+---------------------------------+--------------------------------------+
    | Supports   | ``supports : dict`` (raises if  | ``support : Constraint`` (raises on  |
    |            | not declared)                   | multi-leaf)                          |
    +------------+---------------------------------+--------------------------------------+
    | Flat dim   | ``event_size : int``            | —                                    |
    +------------+---------------------------------+--------------------------------------+

    Single-field auto-template
    --------------------------

    Any concrete subclass that declares an ``event_shape`` and is
    constructed with a ``name=`` gets an auto-built single-field
    :class:`RecordTemplate` (``RecordTemplate(**{name: event_shape})``)
    on first read of :attr:`record_template`. Subclasses that need a
    multi-field template (joint distributions) override
    ``record_template`` directly to skip the auto-build.

    ``_sample`` contract (Story A)
    ------------------------------

    - Single-leaf templates → ``_sample(key, sample_shape)`` returns
      a raw ``jax.Array`` of shape ``sample_shape + event_shape``.
    - Multi-leaf templates → ``_sample(key, sample_shape)`` returns a
      :class:`~probpipe.NumericRecord` (or
      :class:`~probpipe.NumericRecordArray` for non-empty
      ``sample_shape``) keyed by ``record_template.fields``.

    The :attr:`treedef` property locks this relationship by deriving
    from ``record_template``.

    Standard distributions (Normal, Gamma, Poisson, etc.) inherit from
    this class via :class:`TFPDistribution`.
    """

    # -- record_template auto-generation ------------------------------------

    @property
    def record_template(self):
        """Auto-build a single-field ``RecordTemplate`` from
        ``name`` + ``event_shape`` when the subclass hasn't set one.

        Cached via :meth:`object.__setattr__` on first read.
        Multi-field subclasses (joint distributions) override this
        property to skip the auto-build.
        """
        from .record import RecordTemplate
        tpl = getattr(self, "_record_template", None)
        if tpl is not None:
            return tpl
        name = getattr(self, "_name", None)
        if name is not None:
            tpl = RecordTemplate(**{name: self.event_shape})
            object.__setattr__(self, "_record_template", tpl)
            return tpl
        return None

    def _spread_to_fields(self, value):
        """Spread a single value across every field of
        :attr:`record_template`.

        Single-field auto-template subclasses (the common case)
        declare a scalar ``dtype`` / ``support`` / etc. but the
        canonical accessor is a per-field dict. This helper
        materialises ``{name: value for name in record_template.fields}``
        without each override having to spell it out.
        """
        return {name: value for name in self.record_template.fields}

    # -- per-field metadata ---------------------------------------------------

    @property
    def dtypes(self) -> dict[str, jnp.dtype]:
        """Per-field dtypes — **canonical**, subclasses must override.

        Returns a ``{field: dtype}`` dict aligned with ``record_template.fields``.
        Default raises ``NotImplementedError`` rather than returning a
        silent default-float for every field (which lied for integer-
        valued distributions like ``Bernoulli``, ``Poisson``, ``Categorical``).
        """
        raise NotImplementedError(f"{type(self).__name__}.dtypes")

    @property
    def supports(self) -> dict[str, Constraint]:
        """Per-field support constraints — **canonical**, subclasses must override.

        Subclasses should override to provide meaningful constraints.
        Default raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__}.supports")

    @property
    def dtype(self) -> jnp.dtype | None:
        """Convenience: scalar dtype if all fields share one, else ``None``.

        Derived from :attr:`dtypes`. ``dtypes`` is the canonical
        per-field accessor; subclasses override that, not this.
        """
        per_field = self.dtypes
        if not per_field:
            return None
        unique = set(per_field.values())
        return unique.pop() if len(unique) == 1 else None

    @property
    def support(self) -> Constraint:
        """Convenience: support for a single-field distribution.

        Derived from :attr:`supports`. ``supports`` is the canonical
        per-field accessor; subclasses override that (or, for
        ``TFPDistribution``-backed classes that follow the legacy
        single-field pattern, override ``support`` directly to
        short-circuit this derivation).

        Raises ``TypeError`` (via :meth:`_single_field_name`) on
        multi-field distributions; reach for :attr:`supports` then.
        """
        return self.supports[self._single_field_name()]

    @classmethod
    def _check_support_compatible(cls, other: NumericRecordDistribution) -> None:
        """Raise ``ValueError`` if *other*'s per-field supports are
        incompatible with *cls*'s default target support.

        Reads ``other.supports`` (canonical, per-leaf) so multi-leaf
        sources are checked field-by-field against the single target
        support. Single-leaf sources keep the original error
        message; multi-leaf sources include the field name.
        """
        try:
            target_support = cls._default_support()
        except NotImplementedError:
            return
        try:
            per_field = other.supports
        except NotImplementedError:
            return
        multi_leaf = len(per_field) > 1
        for field_name, source_support in per_field.items():
            if _supports_compatible(source_support, target_support):
                continue
            field_part = f" field {field_name!r}" if multi_leaf else ""
            raise ValueError(
                f"Cannot convert {type(other).__name__}{field_part} "
                f"(support={source_support}) to {cls.__name__} "
                f"(support={target_support}). "
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
        """Treedef of one sample, derived from :attr:`record_template`.

        Locks the relationship between the structural template and
        the sample's pytree shape:

        - Single-leaf template (``len(fields) <= 1``) → a leaf
          treedef (``jax.tree.structure(None)``). Matches the
          ``_sample`` contract that single-leaf distributions
          return a raw ``jax.Array``.
        - Multi-leaf template → the treedef of a ``NumericRecord``
          skeleton with the same field names. Matches the
          ``_sample`` contract that multi-leaf distributions
          return a ``NumericRecord``.

        Cached on first read; the underlying template is immutable
        post-construction so the cache is always valid.
        """
        cached = getattr(self, "_treedef", None)
        if cached is not None:
            return cached
        tpl = self.record_template
        if tpl is None or len(tpl.fields) <= 1:
            td = jax.tree.structure(None)
        else:
            from ._numeric_record import NumericRecord
            placeholder = NumericRecord(**{
                name: jnp.zeros(
                    tpl[name] if isinstance(tpl[name], tuple) else ()
                )
                for name in tpl.fields
            })
            td = jax.tree.structure(placeholder)
        object.__setattr__(self, "_treedef", td)
        return td

    @property
    def flat_event_shapes(self) -> list[tuple[int, ...]]:
        """List of per-field event shapes in template field order.

        Tree-walk over :attr:`event_shapes`: ``list(event_shapes.values())``.
        For a single-field distribution this is ``[event_shape]``;
        for a multi-leaf distribution it's one entry per leaf.
        """
        return list(self.event_shapes.values())

    def flatten_value(self, value) -> Array:
        """Flatten a sample (Record, NumericRecordArray, or array) to flat trailing axis.

        Delegates to ``RecordDistribution.flatten_value`` for Record-like
        inputs.  For raw arrays, flattens event dimensions preserving
        leading batch/sample dims.
        """
        from .record import Record
        from ._record_array import NumericRecordArray
        if isinstance(value, (NumericRecordArray, Record)):
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
        self._evaluations = _as_float_array(evaluations)
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

    @property
    def dtypes(self) -> dict[str, jnp.dtype]:
        """Per-field dtype — the evaluations' dtype spread across
        the auto-built single-field template."""
        return self._spread_to_fields(self._evaluations.dtype)

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
    def supports(self) -> dict[str, Constraint]:
        """Per-field support — bootstrap of mean values is real-valued."""
        return self._spread_to_fields(real)

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
    def supports(self) -> dict[str, Constraint]:
        """Per-field support — the flattened view is real-valued."""
        return self._spread_to_fields(real)

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
