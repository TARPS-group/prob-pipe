"""``NumericRecordDistribution`` and its closely-related helpers.

The primary class is :class:`NumericRecordDistribution` — a
:class:`~probpipe.core._record_distribution.RecordDistribution` that
additionally enforces numeric-leaf shape, dtype, and support
semantics via the canonical ``event_shapes`` / ``dtypes`` /
``supports`` accessors and their scalar convenience shortcuts
(``event_shape`` / ``dtype`` / ``support``). It is the base class
for every numeric ProbPipe distribution (``Normal``, ``Beta``,
``ProductDistribution``, ...).

Provides:

  - :class:`NumericRecordDistribution` — the base class.
  - :class:`FlatNumericRecordDistribution` — the flat-shaped subset
    (single field, ``event_shape=(N,)``), used as the input type for
    algorithms that consume a flat parameter vector and as the source
    of :meth:`~FlatNumericRecordDistribution.as_record_distribution`.
  - :class:`BootstrapDistribution` — MC error tracking via bootstrap
    resampling.
  - :class:`FlattenedDistributionView` — flat view of any distribution
    (always a ``FlatNumericRecordDistribution`` by construction).
  - :class:`NumericRecordDistributionView` — inverse, lifting a flat
    source to a Record-keyed view under a user-supplied template.
  - Private helpers ``_vmap_sample`` / ``_mc_expectation``.

Distinct from :class:`~probpipe.DistributionArray` (housed in
:mod:`_distribution_array`), which represents *n independent
distributions stacked along a batch axis* — many random variables
indexed by position, e.g. ``Normal(loc=jnp.zeros(5), scale=1.0)``
stored as a length-5 array of independent ``Normal`` instances. A
``NumericRecordDistribution`` represents *one* random variable
whose draw can itself have a numeric-valued event structure (a
scalar, a vector, or a multi-field record), and ``DistributionArray``
holds many such variables. The two compose: a ``DistributionArray``
of ``NumericRecordDistribution`` instances is the canonical way to
express a vectorized batch of structured random variables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .record import NumericRecordTemplate

from .._dtype import _as_float_array
from .._utils import prod
from .protocols import (
    SupportsCovariance,
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
    dist: "NumericRecordDistribution",
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
    dist : NumericRecordDistribution
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
    dist: "NumericRecordDistribution",
    f: Callable[[Any], Any],
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

    Extends :class:`RecordDistribution` with numeric-specific metadata
    (per-field shape, dtype, and support). The class is the most
    general numeric random variable in ProbPipe: one draw is a pytree
    of ``jax.Array`` leaves named via a :class:`RecordTemplate`.
    Single-leaf distributions (``Normal``, ``Beta``,
    ``MultivariateNormal``, ...) are the trivial case; joint
    distributions (``ProductDistribution``, ``SequentialJointDistribution``,
    ``JointGaussian``, ...) reuse the same machinery with a multi-leaf
    template.

    A ``Distribution`` represents one random variable; collections of
    independent distributions live in
    :class:`~probpipe.DistributionArray`.

    Canonical and convenience accessors
    -----------------------------------

    Per-field accessors are the canonical source of truth; scalar
    accessors are convenience shortcuts that raise on multi-leaf
    templates. Subclasses override the canonical side; the convenience
    accessors derive automatically.

    | Concept   | Canonical (per-field)            | Convenience (single-leaf)                       |
    |-----------|----------------------------------|-------------------------------------------------|
    | Structure | `record_template`                | —                                               |
    | Pytree    | `treedef` (from template)        | —                                               |
    | Shapes    | `event_shapes : dict`            | `event_shape : tuple` (raises on multi-leaf)    |
    | Dtypes    | `dtypes : dict`                  | `dtype : dtype \\| None` (unique or `None`)     |
    | Supports  | `supports : dict`                | `support : Constraint` (raises on multi-leaf)   |
    | Flat dim  | `event_size : int`               | —                                               |

    Single-field auto-template
    --------------------------

    A concrete subclass that declares ``event_shape`` and is constructed
    with a ``name=`` gets an auto-built single-field
    ``RecordTemplate(**{name: event_shape})`` on first read of
    :attr:`record_template`. Multi-field subclasses (joints) override
    ``record_template`` directly to skip the auto-build.

    ``_sample`` contract
    --------------------

    Subclasses implement ``_sample(key, sample_shape) -> draw``; the
    public :func:`~probpipe.sample` op (in
    :mod:`probpipe.core.ops`) handles key auto-generation,
    protocol dispatch, and source/provenance tracking before delegating
    to ``dist._sample(...)``. Subclass code should never call the public
    ``sample`` op on ``self`` — call ``self._sample(key, sample_shape)``
    directly to avoid the ops layer.

    The shape of one draw is fully determined by ``record_template``:

    - **Single-leaf** template → ``_sample(key, sample_shape)`` returns
      a raw ``jax.Array`` of shape ``sample_shape + event_shape``.
    - **Multi-leaf** template → ``_sample(key, sample_shape)`` returns a
      :class:`~probpipe.NumericRecord` (or
      :class:`~probpipe.NumericRecordArray` for a non-empty
      ``sample_shape``) keyed by ``record_template.fields``.

    The :attr:`treedef` property locks this invariant by deriving from
    ``record_template``.

    Standard distributions (``Normal``, ``Gamma``, ``Poisson``, ...)
    inherit from this class via :class:`TFPDistribution`.
    """

    # -- record_template auto-generation ------------------------------------

    @property
    def record_template(self):
        """Auto-build a single-field ``RecordTemplate`` from
        ``name`` + ``event_shape`` when the subclass hasn't set one.

        Cached via :meth:`object.__setattr__` on first read.
        Multi-field subclasses (joint distributions) override this
        property to skip the auto-build.

        Raises ``TypeError`` if the auto-build path can't run —
        either ``_name`` is unset, or ``event_shape`` is not
        derivable. Both error messages name the subclass and point at
        the two construction paths (set ``_record_template``
        explicitly, or declare ``event_shape`` so the auto-build can
        proceed).
        """
        from .record import RecordTemplate
        tpl = getattr(self, "_record_template", None)
        if tpl is not None:
            return tpl
        name = getattr(self, "_name", None)
        if name is None:
            raise TypeError(
                f"{type(self).__name__} has no record_template and "
                f"no name; set _record_template explicitly (multi-leaf "
                f"joints) or pass name= at construction (single-leaf)."
            )
        try:
            es = self.event_shape
        except NotImplementedError:
            raise TypeError(
                f"{type(self).__name__} must declare event_shape or "
                f"set _record_template explicitly."
            ) from None
        tpl = RecordTemplate(**{name: es})
        object.__setattr__(self, "_record_template", tpl)
        return tpl

    def renamed(self, new_name: str) -> "NumericRecordDistribution":
        """Return a renamed copy, regenerating an auto-built template.

        Extends :meth:`Distribution.renamed`. When the cached template
        is the single-field auto-build (one field keyed by the old
        name), the clone's ``_record_template`` is cleared so the next
        access rebuilds it under ``new_name``. Multi-leaf and
        user-supplied templates are left intact — their field names
        are part of the distribution's identity, not derived from
        ``name``.
        """
        clone = super().renamed(new_name)
        tpl = getattr(clone, "_record_template", None)
        if (
            tpl is not None
            and len(tpl.fields) == 1
            and tpl.fields[0] == self._name
        ):
            object.__setattr__(clone, "_record_template", None)
        return clone

    def _per_field_dict(self, value: Any) -> dict[str, Any]:
        """Build a ``{field: value}`` dict keyed by every field of
        :attr:`record_template`, with *value* repeated as the value.

        Single-field auto-template subclasses (the common case)
        declare one scalar dtype / support / etc., but the
        canonical accessor is a per-field dict. This helper saves
        each override from spelling out
        ``{name: value for name in record_template.fields}``.
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
        ``TFPDistribution``-backed classes that follow the existing
        single-field override pattern, override ``support`` directly
        to short-circuit this derivation).

        Raises ``TypeError`` (via :meth:`_single_field_name`) on
        multi-field distributions; reach for :attr:`supports` then.
        """
        return self.supports[self._single_field_name()]

    def _check_support_compatible(
        self, source: "NumericRecordDistribution",
    ) -> None:
        """Raise ``ValueError`` if *source*'s per-field supports are
        incompatible with *self*'s (the target's) per-field supports.

        Called post-construction by the converter, so both sides expose
        instance-level ``supports`` — no class-level default-support
        approximation. For a single-field target (the common case),
        every source field's support is compared against the lone
        target support. For a multi-field target, supports pair up
        field-by-field in insertion order; field-count mismatches raise
        ``ValueError`` rather than silently truncating via ``zip``.

        Sources that don't expose per-field supports (non-NRD endpoints
        like ``EmpiricalDistribution`` with object-dtype data) are
        treated as "unknown" and the check returns without complaint.
        """
        try:
            target_per_field = self.supports
            source_per_field = source.supports
        except (NotImplementedError, AttributeError):
            return

        multi_leaf_source = len(source_per_field) > 1

        if len(target_per_field) == 1:
            target_support = next(iter(target_per_field.values()))
            for field_name, source_support in source_per_field.items():
                if _supports_compatible(source_support, target_support):
                    continue
                field_part = (
                    f" field {field_name!r}" if multi_leaf_source else ""
                )
                raise ValueError(
                    f"Cannot convert {type(source).__name__}{field_part} "
                    f"(support={source_support}) to {type(self).__name__} "
                    f"(support={target_support}). "
                    f"Pass check_support=False to override."
                )
            return

        # Multi-field target — field counts must match to pair
        # positionally; ``zip`` would silently truncate, hiding bugs
        # where the converter produced a target with the wrong arity.
        if len(source_per_field) != len(target_per_field):
            raise ValueError(
                f"Cannot convert {type(source).__name__} "
                f"({len(source_per_field)} fields: "
                f"{tuple(source_per_field)}) to {type(self).__name__} "
                f"({len(target_per_field)} fields: "
                f"{tuple(target_per_field)}): field-count mismatch. "
                f"Pass check_support=False to override."
            )
        for (s_name, s_sup), (t_name, t_sup) in zip(
            source_per_field.items(), target_per_field.items(),
        ):
            if _supports_compatible(s_sup, t_sup):
                continue
            raise ValueError(
                f"Cannot convert {type(source).__name__} field "
                f"{s_name!r} (support={s_sup}) to "
                f"{type(self).__name__} field {t_name!r} "
                f"(support={t_sup}). "
                f"Pass check_support=False to override."
            )

    # -- event_shape ---------------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Single-leaf convenience shortcut for the lone field's shape.

        **Abstract** — every single-leaf subclass must override. The
        declared ``event_shape`` is the source of truth used by
        :attr:`record_template`'s auto-build path; deriving it from
        :attr:`event_shapes` here would loop back through
        ``record_template``.

        Multi-leaf subclasses don't override (single-field convenience
        doesn't apply); they set ``_record_template`` explicitly in
        ``__init__`` so the auto-build never fires, and callers reach
        for :attr:`event_shapes` (per-field dict) instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.event_shape — single-leaf "
            f"subclasses must override; multi-leaf subclasses should "
            f"use .event_shapes (per-field dict)."
        )

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
        # ``record_template`` is contractually non-``None`` on every
        # ``RecordDistribution`` (metaclass-enforced); single-field
        # templates produce a leaf treedef matching the raw-array
        # ``_sample`` contract, multi-field templates produce a
        # ``NumericRecord`` skeleton.
        tpl = self.record_template
        if len(tpl.fields) <= 1:
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

    @property
    def event_size(self) -> int:
        """Total number of scalar elements in one sample.

        For a :class:`NumericRecordTemplate` this is the cached
        ``flat_size``. For a general ``RecordTemplate``, sums the
        numeric-leaf shapes; opaque leaves contribute zero.
        """
        from .record import NumericRecordTemplate
        tpl = self.record_template
        if isinstance(tpl, NumericRecordTemplate):
            return tpl.flat_size
        return sum(
            prod(shape) if shape else 1
            for shape in tpl.leaf_shapes.values()
            if shape is not None
        )

    @staticmethod
    def flatten_value(value, *, event_shape: tuple[int, ...] = ()) -> Array:
        """Flatten a sample to a flat trailing axis.

        Accepts ``Record`` / ``NumericRecord`` / ``NumericRecordArray``
        (which already carry their template) or a raw array. Raw-array
        inputs need ``event_shape`` to disambiguate batch axes from
        event axes; without it, the input gets a trailing singleton
        axis (matching the scalar-event default).
        """
        from .record import Record
        from ._numeric_record import NumericRecord
        from ._record_array import NumericRecordArray
        if isinstance(value, (NumericRecordArray, NumericRecord)):
            return value.flatten()
        if isinstance(value, Record):
            return NumericRecord.from_record(value).flatten()
        value = jnp.asarray(value)
        if not event_shape:
            return value[..., None]
        n_event = prod(event_shape)
        n_batch = value.ndim - len(event_shape)
        return value.reshape(*value.shape[:n_batch], n_event)

    @staticmethod
    def unflatten_value(flat, *, template):
        """Unflatten a flat trailing axis back to event dims, Record, or NumericRecordArray.

        Multi-field templates → ``NumericRecord`` (single sample, i.e.
        ``flat.ndim == 1``) or ``NumericRecordArray`` (batched). Single-
        field templates → raw array reshaped to ``(*batch, *event_shape)``
        for ``_log_prob`` compatibility (preserves the original "single-
        leaf returns raw array" contract).
        """
        from ._numeric_record import NumericRecord
        from ._record_array import NumericRecordArray
        flat = jnp.asarray(flat)
        if template is not None and len(template.fields) > 1:
            if flat.ndim < 2:
                return NumericRecord.unflatten(flat, template=template)
            return NumericRecordArray.unflatten(flat, template=template)
        # Single-field path
        if template is None or not template.fields:
            return flat[..., 0]
        field_spec = template[template.fields[0]]
        es = field_spec if isinstance(field_spec, tuple) else ()
        if not es:
            return flat[..., 0]
        return flat.reshape(*flat.shape[:-1], *es)

    def as_flat_distribution(self) -> FlatNumericRecordDistribution:
        """View this distribution as a flat distribution.

        Returns a :class:`FlattenedDistributionView` wrapping this
        distribution. The view satisfies the
        :class:`FlatNumericRecordDistribution` contract regardless of
        ``self``'s structure (multi-field, multi-dim event, …) — its
        ``event_shape`` is always ``(self.event_size,)``.

        Inverse: :meth:`FlatNumericRecordDistribution.as_record_distribution`.
        """
        return FlattenedDistributionView(self)

    def as_record_distribution(
        self,
        *,
        template: NumericRecordTemplate,
        name: str | None = None,
    ) -> NumericRecordDistribution:
        """Lift this distribution to a Record-keyed view under *template*.

        **Only available on :class:`FlatNumericRecordDistribution` subclasses.**
        Calling this on a non-flat :class:`NumericRecordDistribution`
        raises :class:`TypeError` with a hint to call
        :meth:`as_flat_distribution` first.

        See :meth:`FlatNumericRecordDistribution.as_record_distribution`
        for the actual implementation and parameters.
        """
        raise TypeError(
            f"as_record_distribution is only available on "
            f"FlatNumericRecordDistribution subclasses. "
            f"{type(self).__name__} is not flat. Chain: "
            f"source.as_flat_distribution().as_record_distribution(template=...)."
        )

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts: list[str] = [type(self).__name__]
        if self.name:
            parts.append(f"name={self.name!r}")
        # Multi-field NRDs (joints) can't summarise the event with a
        # single ``event_shape`` — ``_single_field_name`` raises
        # ``TypeError`` there, and the base default raises
        # ``NotImplementedError`` for subclasses that haven't overridden
        # ``event_shape``. Either way, fall back to the per-field dict.
        try:
            parts.append(f"event_shape={self.event_shape}")
        except (TypeError, NotImplementedError):
            parts.append(f"event_shapes={self.event_shapes}")
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
        return self._per_field_dict(self._evaluations.dtype)

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
        return self._per_field_dict(real)

    def __repr__(self) -> str:
        return f"BootstrapDistribution(n={self._n}, event_shape={self.event_shape})"

# ---------------------------------------------------------------------------
# FlatNumericRecordDistribution — the flat-shaped subset of NRD
# ---------------------------------------------------------------------------


class FlatNumericRecordDistribution(NumericRecordDistribution):
    """A :class:`NumericRecordDistribution` whose samples are flat 1-D vectors.

    The flat contract:

    * exactly one field (``len(fields) == 1``)
    * ``event_shape == (N,)`` for some ``N``
    * samples shaped ``sample_shape + (N,)``

    Algorithms that operate on a flat parameter vector — MCMC kernels,
    optimisers, Hessian / curvature builders, variational families,
    Pathfinder / Laplace surrogates — should declare their input as
    :class:`FlatNumericRecordDistribution`. The natively-multivariate
    parametrics (:class:`~probpipe.MultivariateNormal`,
    :class:`~probpipe.Dirichlet`, :class:`~probpipe.Multinomial`,
    :class:`~probpipe.VonMisesFisher`) and
    :class:`FlattenedDistributionView` all satisfy this contract.

    Scalar parametrics (``Normal``, ``Beta``, …) have
    ``event_shape == ()`` and do **not** satisfy the contract directly;
    call :meth:`~NumericRecordDistribution.as_flat_distribution` to get
    a :class:`FlattenedDistributionView` (whose event_shape is ``(1,)``).

    This class is also the home of
    :meth:`as_record_distribution` — the inverse of
    :meth:`~NumericRecordDistribution.as_flat_distribution`. Receiver
    typing means non-flat callers fail at the type level rather than at
    a runtime shape check.
    """

    @property
    def flat_size(self) -> int:
        """Number of scalar elements — equal to ``event_shape[0]``.

        Validates the flat contract on access: subclasses with
        non-1-D ``event_shape`` raise ``TypeError`` here rather than
        silently truncating to the first dimension.
        """
        es = self.event_shape
        if len(es) != 1:
            raise TypeError(
                f"{type(self).__name__} declares FlatNumericRecordDistribution "
                f"but has event_shape={es}; expected 1-D (N,)."
            )
        return es[0]

    def as_record_distribution(
        self,
        *,
        template: NumericRecordTemplate,
        name: str | None = None,
    ) -> NumericRecordDistribution:
        """Lift this flat distribution to a Record-keyed view under *template*.

        Inverse of :meth:`~NumericRecordDistribution.as_flat_distribution`.
        Samples come back as :class:`NumericRecord` /
        :class:`NumericRecordArray` keyed by ``template.fields``.

        Parameters
        ----------
        template : NumericRecordTemplate
            Target structural skeleton. Must be a
            :class:`NumericRecordTemplate` — opaque (``None``) leaves
            cannot be reconstructed from a flat numeric array.
        name : str, optional
            Name for the lifted distribution. Defaults to ``self.name``.

        Returns
        -------
        NumericRecordDistribution
            A thin view over ``self``. Sampling, log-prob, moments, and
            ``expectation`` delegate to the source and reshape via the
            template. Capability protocols match the source.

        Raises
        ------
        TypeError
            If ``template`` is not a ``NumericRecordTemplate``.
        ValueError
            If ``self.flat_size`` does not match ``template.flat_size``.
        """
        from .record import NumericRecordTemplate
        if not isinstance(template, NumericRecordTemplate):
            raise TypeError(
                f"as_record_distribution requires a NumericRecordTemplate, "
                f"got {type(template).__name__}. Opaque (None) leaves "
                f"cannot be reconstructed from a flat numeric array."
            )
        if self.flat_size != template.flat_size:
            raise ValueError(
                f"flat_size mismatch: source flat_size={self.flat_size}, "
                f"template.flat_size={template.flat_size}."
            )
        cls = _numeric_record_distribution_view_class_for_base(self)
        return cls(self, template, name=name)


# ---------------------------------------------------------------------------
# FlattenedDistributionView — wrap any distribution as a flat NRD
# ---------------------------------------------------------------------------

_FLATTENED_VIEW_CLASS_CACHE: dict[frozenset[str], type] = {}


def _flattened_distribution_view_class_for_base(base: Distribution) -> type:
    """Return a ``FlattenedDistributionView`` subclass whose protocol bases
    match the capabilities of *base*.

    The view only delegates sampling and log-prob; those are the only
    protocols that make sense to inherit. A view over a log-prob-only
    base should not advertise ``SupportsSampling``, and vice versa.
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
            return self._base.flatten_value(
                pytree_samples, event_shape=self._base.event_shape,
            )

        extra_methods["_sample"] = _sample

    if "log_prob" in protocols:
        extra_bases.append(SupportsLogProb)

        def _log_prob(self, x: ArrayLike) -> Array:
            x = jnp.asarray(x)
            value = self._base.unflatten_value(
                x, template=self._base.record_template,
            )
            return self._base._log_prob(value)

        extra_methods["_log_prob"] = _log_prob

    if not extra_bases:
        _FLATTENED_VIEW_CLASS_CACHE[key] = FlattenedDistributionView
        return FlattenedDistributionView

    new_cls = type(
        "FlattenedDistributionView",
        (FlattenedDistributionView, *extra_bases),
        extra_methods,
    )
    _FLATTENED_VIEW_CLASS_CACHE[key] = new_cls
    return new_cls


class FlattenedDistributionView(FlatNumericRecordDistribution):
    """Wraps a distribution as a flat :class:`FlatNumericRecordDistribution`.

    Sampling produces flat vectors of shape ``(event_size,)``, and
    ``_log_prob`` accepts flat vectors and delegates to the wrapped
    distribution after unflattening.

    This is the primary interoperability mechanism: any algorithm written
    against :class:`FlatNumericRecordDistribution` works with an
    arbitrary :class:`RecordDistribution` /
    :class:`NumericRecordDistribution` via
    ``dist.as_flat_distribution()``.

    **Dynamic protocol support:** the view's ``isinstance`` compliance
    matches the base's capabilities — a log-prob-only base produces a
    view that is not ``SupportsSampling``, and a sampling-only base
    produces one that is not ``SupportsLogProb``.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __new__(cls, base: Distribution):
        actual_cls = _flattened_distribution_view_class_for_base(base)
        return object.__new__(actual_cls)

    def __init__(self, base: Distribution):
        self._base = base
        # Carry the base's name through; ``base.name`` is guaranteed
        # non-empty by the Distribution metaclass.
        self._name = base.name

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
        return self._per_field_dict(real)

    @property
    def base_distribution(self) -> Distribution:
        """The underlying distribution."""
        return self._base

    def unflatten_sample(self, flat_sample: ArrayLike):
        """Convenience: unflatten a flat sample back to the pytree structure."""
        return self._base.unflatten_value(
            jnp.asarray(flat_sample), template=self._base.record_template,
        )

    def __repr__(self) -> str:
        return (
            f"FlattenedDistributionView(base={type(self._base).__name__}, "
            f"event_shape={self.event_shape})"
        )


# ---------------------------------------------------------------------------
# NumericRecordDistributionView — lift a flat distribution to a Record view
# ---------------------------------------------------------------------------

_LIFTED_VIEW_CLASS_CACHE: dict[type, type] = {}


def _numeric_record_distribution_view_class_for_base(base: Distribution) -> type:
    """Return a ``NumericRecordDistributionView`` subclass advertising the
    same capability protocols as *base*.

    Mirrors :func:`_flattened_distribution_view_class_for_base` for the
    inverse direction. The protocol-bearing methods (``_sample``,
    ``_log_prob``, ``_mean``, ``_variance``, ``_cov``, ``_expectation``)
    are attached dynamically by this factory rather than living on
    :class:`NumericRecordDistributionView` itself — otherwise every
    view would appear to satisfy every protocol by virtue of method
    presence (``@runtime_checkable`` semantics).
    """
    # Cache on the source's concrete type: any two instances of the same
    # Distribution subclass advertise the same protocol set, so the
    # frozenset key would collide anyway. Type-based caching avoids
    # six runtime_checkable isinstance scans on every construction.
    cached = _LIFTED_VIEW_CLASS_CACHE.get(type(base))
    if cached is not None:
        return cached

    # Imports hoisted once for all closures below (and to avoid
    # circular-import risk at module load time).
    from ._numeric_record import NumericRecord
    from ._record_array import NumericRecordArray

    protocols: set[str] = set()
    if isinstance(base, SupportsSampling):    protocols.add("sample")
    if isinstance(base, SupportsLogProb):     protocols.add("log_prob")
    if isinstance(base, SupportsMean):        protocols.add("mean")
    if isinstance(base, SupportsVariance):    protocols.add("variance")
    if isinstance(base, SupportsCovariance):  protocols.add("cov")
    if isinstance(base, SupportsExpectation): protocols.add("expectation")

    extra_bases: list[type] = []
    extra_methods: dict[str, object] = {}

    if "sample" in protocols:
        extra_bases.append(SupportsSampling)

        def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()):
            base_sample = self._base._sample(key, sample_shape)
            flat = self._base.flatten_value(
                base_sample, event_shape=self._base.event_shape,
            )
            tpl = self.record_template
            if sample_shape == ():
                return NumericRecord.unflatten(flat, template=tpl)
            return NumericRecordArray.unflatten(
                flat, template=tpl, batch_shape=sample_shape,
            )

        extra_methods["_sample"] = _sample

    if "log_prob" in protocols:
        extra_bases.append(SupportsLogProb)

        def _log_prob(self, x) -> Array:
            if isinstance(x, (NumericRecord, NumericRecordArray)):
                flat = x.flatten()
            else:
                flat = jnp.asarray(x)
            value = self._base.unflatten_value(
                flat, template=self._base.record_template,
            )
            return self._base._log_prob(value)

        extra_methods["_log_prob"] = _log_prob

    if "mean" in protocols:
        extra_bases.append(SupportsMean)

        def _mean(self):
            flat = self._base.flatten_value(
                self._base._mean(), event_shape=self._base.event_shape,
            )
            return NumericRecord.unflatten(flat, template=self.record_template)

        extra_methods["_mean"] = _mean

    if "variance" in protocols:
        extra_bases.append(SupportsVariance)

        def _variance(self):
            flat = self._base.flatten_value(
                self._base._variance(), event_shape=self._base.event_shape,
            )
            return NumericRecord.unflatten(flat, template=self.record_template)

        extra_methods["_variance"] = _variance

    if "cov" in protocols:
        extra_bases.append(SupportsCovariance)

        def _cov(self):
            # Covariance stays flat (event_size × event_size matrix).
            # The Record / field-block structure is implicit in the
            # template's flat ordering.
            return self._base._cov()

        extra_methods["_cov"] = _cov

    if "expectation" in protocols:
        extra_bases.append(SupportsExpectation)

        def _expectation(
            self,
            f: Callable,
            *,
            key: PRNGKey | None = None,
            num_evaluations: int | None = None,
            return_dist: bool | None = None,
        ) -> Any:
            # ``f`` operates on a Record-shaped sample. We can't pass the
            # batched ``NumericRecordArray`` returned by ``self._sample``
            # through ``jax.vmap(f)`` directly — vmap strips the leading
            # axis from each leaf while preserving ``batch_shape`` aux,
            # producing an invariant violation. Instead, sample the base
            # in flat form (no aux-shape invariants) and run vmap over a
            # closure that unflattens to a Record inside the loop body.
            n = num_evaluations if num_evaluations is not None else _base.DEFAULT_NUM_EVALUATIONS
            sample_key = key if key is not None else _auto_key()
            base_samples = self._base._sample(sample_key, sample_shape=(n,))
            flat_samples = self._base.flatten_value(
                base_samples, event_shape=self._base.event_shape,
            )
            template = self.record_template

            def _f_on_flat(flat_row):
                return f(NumericRecord.unflatten(flat_row, template=template))

            evals = jax.vmap(_f_on_flat)(flat_samples)
            rd = return_dist if return_dist is not None else _base.RETURN_APPROX_DIST
            if rd:
                return BootstrapDistribution(evals, name="E[f(X)]")
            return jax.tree.map(lambda v: jnp.mean(v, axis=0), evals)

        extra_methods["_expectation"] = _expectation

    if not extra_bases:
        _LIFTED_VIEW_CLASS_CACHE[type(base)] = NumericRecordDistributionView
        return NumericRecordDistributionView

    new_cls = type(
        "NumericRecordDistributionView",
        (NumericRecordDistributionView, *extra_bases),
        extra_methods,
    )
    _LIFTED_VIEW_CLASS_CACHE[type(base)] = new_cls
    return new_cls


class NumericRecordDistributionView(NumericRecordDistribution):
    """View that lifts a flat distribution to a Record-keyed structure.

    Inverse of :class:`FlattenedDistributionView`. ``self._base`` is a
    :class:`FlatNumericRecordDistribution` (single-field, ``event_shape
    == (N,)``); ``self.record_template`` is the user-supplied
    :class:`NumericRecordTemplate` (not the source's auto-template).

    Sampling, log-prob, and moments delegate to ``self._base`` and
    reshape via the template's flatten / unflatten machinery.
    Capability protocols match the source via
    :func:`_numeric_record_distribution_view_class_for_base`.

    Constructed via
    :meth:`FlatNumericRecordDistribution.as_record_distribution`.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __new__(
        cls,
        base: Distribution,
        template: NumericRecordTemplate,
        *,
        name: str | None = None,
    ):
        actual_cls = _numeric_record_distribution_view_class_for_base(base)
        return object.__new__(actual_cls)

    def __init__(
        self,
        base: Distribution,
        template: NumericRecordTemplate,
        *,
        name: str | None = None,
    ):
        # Skip ``Distribution.__init__`` to avoid double-validation;
        # the metaclass post-init check still enforces a non-empty
        # ``_name``. ``base.name`` is guaranteed non-empty by the
        # Distribution metaclass, so the fallback is always valid.
        self._base = base
        self._name = name if name is not None else base.name
        # Pre-set the user-supplied template so the auto-build path in
        # ``NumericRecordDistribution.record_template`` is skipped.
        object.__setattr__(self, "_record_template", template)

    # ---- structural ---------------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Single-field shortcut: the lone field's shape.

        Raises ``TypeError`` via :meth:`_single_field_name` for
        multi-field templates; reach for :attr:`event_shapes` (dict)
        in that case.
        """
        return self.event_shapes[self._single_field_name()]

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-field event shapes from the user-supplied template."""
        return dict(self.record_template.numeric_leaf_shapes)

    @property
    def dtypes(self) -> dict[str, jnp.dtype]:
        """Per-field dtypes — all fields inherit the source's single dtype."""
        return self._per_field_dict(self._base.dtype)

    @property
    def supports(self) -> dict[str, Constraint]:
        """Per-field supports — all fields inherit the source's single support."""
        return self._per_field_dict(self._base.support)

    @property
    def base_distribution(self) -> Distribution:
        """The underlying single-field flat distribution."""
        return self._base

    def __repr__(self) -> str:
        return (
            f"NumericRecordDistributionView(base={type(self._base).__name__}, "
            f"template={self.record_template!r})"
        )
