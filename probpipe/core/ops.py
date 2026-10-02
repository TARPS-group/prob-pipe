"""Built-in operations for distribution computation.

Each public function (``sample``, ``mean``, ``log_prob``, …) is a
:class:`~probpipe.values._function_base.Function` created via the
``@function`` decorator.  This means every call automatically
participates in broadcasting and Prefect orchestration when a
distribution argument is passed where a concrete value is expected.

Usage::

    from probpipe import sample, mean, log_prob, condition_on

    dist = Normal("x", 0.0, 1.0)
    s = sample(dist, key=jax.random.PRNGKey(0), sample_shape=(100,))
    m = mean(dist)
    lp = log_prob(dist, jnp.array(1.5))
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from math import prod
from typing import Any

import jax
import jax.numpy as jnp

from ..custom_types import Array, PRNGKey
from ..distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsCovariance,
    SupportsExactConditioning,
    SupportsLogProb,
    SupportsMean,
    SupportsQuantile,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
)
from ..distributions._distribution import Distribution
from ..families._random_functions import RandomFunction
from ..functions import _broker, _descendants, function

__all__ = [
    "condition_on",
    "cov",
    "from_distribution",
    "log_prob",
    "mean",
    "prob",
    "quantile",
    "random_log_prob",
    "random_unnormalized_log_prob",
    "sample",
    "unnormalized_log_prob",
    "unnormalized_prob",
    "variance",
]


# ---------------------------------------------------------------------------
# Public API — each function is a Function via @function
# ---------------------------------------------------------------------------


@function
def sample(
    dist: SupportsSampling,
    *,
    key: PRNGKey | None = None,
    sample_shape: int | tuple[int, ...] = (),
) -> Any:
    """Draw samples from a distribution.

    Parameters
    ----------
    dist : SupportsSampling
        Distribution to sample from.
    key : PRNGKey, optional
        JAX PRNG key.  Auto-generated if ``None``.
    sample_shape : int or tuple of int
        Shape prefix for independent draws.  A scalar ``N`` is treated
        as sugar for ``(N,)``, matching the convention used by numpy,
        JAX, scipy, and TFP.
    """
    if not isinstance(dist, SupportsSampling):
        raise TypeError(
            f"{type(dist).__name__} does not support sampling (does not implement SupportsSampling)"
        )
    if isinstance(sample_shape, bool):
        raise TypeError("sample_shape must be an integer or tuple of integers, not bool")
    axes = sample_shape if isinstance(sample_shape, tuple) else (sample_shape,)
    if any(isinstance(axis, bool) for axis in axes):
        raise TypeError("sample_shape must be an integer or tuple of integers")
    try:
        sample_shape = tuple(operator.index(axis) for axis in axes)
    except TypeError:
        raise TypeError("sample_shape must be an integer or tuple of integers") from None
    if any(axis < 0 for axis in sample_shape):
        raise ValueError(f"sample_shape dimensions must be non-negative; got {sample_shape!r}")
    declaration = getattr(dist, "event_spec", None)
    event_spec = getattr(declaration, "spec", None)
    if key is None:
        captured = _descendants.capture_stochastic_consumer(dist)
        key = _broker._resolve_automatic_key(
            None,
            _broker._singleton_effect_plan(
                operation_kind="sample",
                execution_mode="sampled",
                sample_shape=sample_shape,
                record_path=captured.record_path,
                descendant_descriptor=captured.descendant_descriptor,
            ),
        )
        return _drawn_at_its_batch_form(
            _descendants.sample_captured_consumer(captured, key, sample_shape),
            sample_shape,
            name=getattr(dist, "name", "sample"),
            event_spec=event_spec,
        )
    return _drawn_at_its_batch_form(
        dist._sample(key, sample_shape),
        sample_shape,
        name=getattr(dist, "name", "sample"),
        event_spec=event_spec,
    )


def _drawn_at_its_batch_form(
    drawn: Any,
    sample_shape: tuple[int, ...],
    *,
    name: str,
    event_spec: Any = None,
) -> Any:
    """Wrap draws at their kind, retaining the law's naming metadata.

    A non-empty ``sample_shape`` puts those leading dimensions on one level named
    for the operation that mints them, ``sample`` (design V.2, V.9). The level is
    minted here, at the boundary, for every kind of draw: a law that assembled the
    draws itself — as a record of columns, or as an array of stored objects — did
    not name what its leading axes range over, and reading them as event shape
    says the draws were one wide value.

    A record-valued draw in raw form is the nested mapping of its raw leaves,
    stacked with the draw axes leading under a non-empty ``sample_shape``, and a
    record held inside the mapping is read as its own nested mapping. It is
    wrapped by the law's record declaration *event_spec*: one draw as a
    ``Record`` and a batch of draws as the ``RecordBatch`` or
    ``NumericRecordBatch`` that declaration calls for. Without a record
    declaration the structure is inferred from the mapping. For any other draw
    the event shape is read off the draw, which stays exact for a law whose
    event shape is symbolic until a draw binds it. The batch takes the law's own
    *name*, since the draws are that law's. A single raw draw takes the same
    name when wrapped.

    A law that built its own batch already named the level, and a term of some
    other kind is left as it is. So is a mapping whose leaves are not arrays led
    by the draw axes, since it is not the raw form of a batch of draws.
    """
    from collections.abc import Mapping

    from ..functions._result import SAMPLE_LEVEL, _make_stack
    from ._array_backend import _event_shape_of, _is_numeric_leaf, _numpy_dtype_of
    from ._batch import Batch
    from ._numeric_array_batch import NumericArrayBatch
    from ._object_batch import _is_object_array
    from ._record_batch import _batch_class_for
    from ._record_spec import RecordSpec, _reshaped_template
    from ._specs import NumericArraySpec
    from .record import Record
    from .tracked import TrackedTerm

    if isinstance(drawn, Batch):
        return drawn

    declared = event_spec if isinstance(event_spec, RecordSpec) else None
    if isinstance(drawn, Mapping) and not isinstance(drawn, TrackedTerm):
        from ..distributions._factored import _raw_record

        raw = _raw_record(drawn)
        if not sample_shape:
            return Record(name, raw, event_template=declared)
        if not _is_stacked_columns(raw, sample_shape):
            # Not the raw form of a batch of draws, so there is no batch to build.
            return drawn
        return _columns_at_a_level(
            raw, sample_shape, name=name, level=SAMPLE_LEVEL, element_spec=declared
        )

    if not sample_shape:
        if isinstance(drawn, TrackedTerm):
            return drawn
        from ..functions._result import _wrap_as_term

        return _wrap_as_term(drawn, SAMPLE_LEVEL, name=name)

    n_draw_axes = len(sample_shape)
    if isinstance(drawn, Record):
        columns = {}
        for path in drawn.event_template:
            column = drawn[path]
            if not _is_numeric_leaf(column):
                return drawn
            if tuple(_event_shape_of(column))[:n_draw_axes] != tuple(sample_shape):
                # The draws are not where the contract puts them, so there is no
                # split to make — the same conservatism the array case applies.
                return drawn
            columns[path] = column
        # Only the draw axes move; the rest of each field's declaration rides
        # through, which inferring an element template from the columns would lose.
        element_spec = _reshaped_template(drawn.event_template, lambda shape: shape[n_draw_axes:])
        return _batch_class_for(element_spec)(
            name,
            columns,
            SAMPLE_LEVEL,
            element_spec=element_spec,
            axes_per_level=(len(sample_shape),),
        )

    if _is_object_array(drawn):
        if drawn.shape[:n_draw_axes] != tuple(sample_shape):
            return drawn
        # Stored draws aggregate exactly as a sweep's rows do — each element at
        # its own kind, under the one level the operation mints.
        return _make_stack(
            list(drawn.reshape((prod(sample_shape), *drawn.shape[n_draw_axes:]))),
            batch_shape=tuple(sample_shape),
            level_names=(SAMPLE_LEVEL,),
            field_name=name,
            name=name,
        )

    if isinstance(drawn, TrackedTerm) or not _is_numeric_leaf(drawn):
        return drawn
    shape = _event_shape_of(drawn)
    if shape[: len(sample_shape)] != tuple(sample_shape):
        # The draws are not where the contract puts them, so there is no split
        # to make.
        return drawn
    return NumericArrayBatch(
        name,
        drawn,
        SAMPLE_LEVEL,
        element_spec=NumericArraySpec(
            shape=shape[len(sample_shape) :], dtype=_numpy_dtype_of(drawn)
        ),
        axes_per_level=(len(sample_shape),),
    )


def _columns_at_a_level(
    columns: Any,
    leading_shape: tuple[int, ...],
    *,
    name: str,
    level: str,
    element_spec: Any = None,
) -> Any:
    """The batch of records whose raw columns are *columns*, on the one level *level*.

    The columns are a nested mapping whose leaves are led by *leading_shape*,
    which the level holds. The element is the record declaration
    *element_spec*, or else the structure the columns imply without those
    axes, and the batch is the ``RecordBatch`` or ``NumericRecordBatch`` the
    element calls for.
    """
    from ._record_batch import _batch_class_for
    from ._record_spec import _reshaped_template
    from .record import Record

    if element_spec is None:
        stacked = Record(name, columns).event_template
        element_spec = _reshaped_template(stacked, lambda shape: shape[len(leading_shape) :])
    return _batch_class_for(element_spec)(
        name,
        columns,
        level,
        element_spec=element_spec,
        axes_per_level=(len(leading_shape),),
    )


def _is_stacked_columns(columns: Any, sample_shape: tuple[int, ...]) -> bool:
    """Whether every leaf of the nested mapping *columns* is an array led by *sample_shape*.

    An object array is the column of a field that is not an array, so it counts.
    """
    from collections.abc import Mapping

    if isinstance(columns, Mapping):
        return all(_is_stacked_columns(column, sample_shape) for column in columns.values())
    shape = getattr(columns, "shape", None)
    return shape is not None and tuple(shape[: len(sample_shape)]) == tuple(sample_shape)


def _at_the_operands_levels(computed: Any, operand: Any) -> Any:
    """Give *computed* the levels *operand* carried, when it is a batch.

    An operation whose value parameter is ``Any``-hinted receives a batch whole
    and evaluates it in one vectorized call rather than row by row. That is the
    fused implementation design V.9 allows to register above the elementwise map
    — but V.9 also says the batch axes are *preserved*, and a bare array does not
    preserve them. So the levels the operand stated are restated on the result.

    Only the axes the operand accounted for are levels; anything the operation
    added beyond them belongs to the element, which is why the split is by the
    operand's rank rather than by the result's.
    """
    from ._array_backend import _event_shape_of, _is_numeric_leaf, _numpy_dtype_of
    from ._batch import Batch, _ranks_of
    from ._numeric_array_batch import NumericArrayBatch
    from ._specs import NumericArraySpec

    if not isinstance(operand, Batch) or not _is_numeric_leaf(computed):
        return computed
    batch_shape = tuple(operand.batch_shape)
    shape = tuple(_event_shape_of(computed))
    if shape[: len(batch_shape)] != batch_shape:
        # The operation did not lay its result out over the operand's axes, so
        # there is no correspondence to restate.
        return computed
    return NumericArrayBatch(
        operand.name,
        computed,
        tuple(operand.level_names),
        element_spec=NumericArraySpec(
            shape=shape[len(batch_shape) :], dtype=_numpy_dtype_of(computed)
        ),
        axes_per_level=_ranks_of(operand.axis_groups),
    )


# -- keyword value form shared by the density ops ---------------------------
#
# Each density op accepts either a positional ``value`` or named field kwargs
# packed into one draw via ``dist._pack_value`` (single-field → the bare
# value; multi-field → a ``Record``). The ops stay plain Functions and
# resolve this in their body — exactly as ``condition_on`` resolves its named
# data kwargs from ``**kwargs``. Per-call controls use ``with_options`` (the
# Function control path).


def _resolve_value(
    op_name: str,
    dist: Any,
    value: Any,
    field_kwargs: dict[str, Any],
    *,
    allow_none: bool = False,
) -> Any:
    """Resolve a density op's value from the positional or keyword form.

    Keyword form packs ``field_kwargs`` into a single draw via
    ``dist._pack_value``; the positional form passes ``value`` through.
    Passing both is an error. With ``allow_none=False`` (default) a missing
    value also errors; ``allow_none=True`` (the ``random_*`` ops) lets
    ``value=None`` through so the bare random function is returned.

    A distribution field whose name collides with the op's own ``value`` or
    ``dist`` parameter cannot be addressed by the keyword form (it binds to the
    parameter). For a multi-field distribution, pass a positional ``Record``
    (``log_prob(d, Record("v", value=...))``); for a single-field one, pass the bare
    positional value (``log_prob(d, v)`` — a scalar ``_log_prob`` does not
    accept a ``Record``). This mirrors ``condition_on``'s ``observed``.
    """
    if field_kwargs:
        if value is not None:
            raise TypeError(f"{op_name}: pass either a positional value or field kwargs, not both.")
        return dist._pack_value(**field_kwargs)
    if value is None and not allow_none:
        raise TypeError(
            f"{op_name}: a value is required — pass it positionally or as field keyword arguments."
        )
    return value


@function
def log_prob(dist: SupportsLogProb, value: Any = None, **field_kwargs: Any) -> Array:
    """Evaluate the normalized log-density at *value*.

    Two call forms: positional ``log_prob(dist, value)`` (a single draw, or a
    batched form that broadcasts), or keyword ``log_prob(dist, field=..., ...)``
    built into one draw via :meth:`Distribution._pack_value` (single-field →
    the bare value; multi-field → a ``Record``). Use the positional form for
    batched evaluation; per-call controls use ``log_prob.with_options(...)``.
    """
    if not isinstance(dist, SupportsLogProb):
        raise TypeError(f"{type(dist).__name__} does not support log_prob")
    resolved = _resolve_value("log_prob", dist, value, field_kwargs)
    return _at_the_operands_levels(dist._log_prob(resolved), resolved)


@function
def prob(dist: SupportsLogProb, value: Any = None, **field_kwargs: Any) -> Array:
    """Evaluate the density at *value* (``exp(log_prob)``).

    See :func:`log_prob` for the positional and keyword call forms.
    """
    if not isinstance(dist, SupportsLogProb):
        raise TypeError(f"{type(dist).__name__} does not support prob (missing _log_prob method)")
    resolved = _resolve_value("prob", dist, value, field_kwargs)
    return _at_the_operands_levels(jnp.exp(dist._log_prob(resolved)), resolved)


@function
def unnormalized_log_prob(
    dist: SupportsUnnormalizedLogProb,
    value: Any = None,
    **field_kwargs: Any,
) -> Array:
    """Evaluate the unnormalized log-density at *value*.

    See :func:`log_prob` for the positional and keyword call forms.
    """
    if not isinstance(dist, SupportsUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support unnormalized_log_prob "
            f"(missing _unnormalized_log_prob method)"
        )
    resolved = _resolve_value("unnormalized_log_prob", dist, value, field_kwargs)
    return _at_the_operands_levels(dist._unnormalized_log_prob(resolved), resolved)


@function
def unnormalized_prob(
    dist: SupportsUnnormalizedLogProb,
    value: Any = None,
    **field_kwargs: Any,
) -> Array:
    """Evaluate the unnormalized density at *value* — ``exp(unnormalized_log_prob)``.

    See :func:`log_prob` for the positional and keyword call forms.
    """
    if not isinstance(dist, SupportsUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support unnormalized_prob "
            f"(missing _unnormalized_log_prob method)"
        )
    resolved = _resolve_value("unnormalized_prob", dist, value, field_kwargs)
    return _at_the_operands_levels(jnp.exp(dist._unnormalized_log_prob(resolved)), resolved)


@function
def mean(dist: SupportsMean) -> Any:
    """Compute ``E[X]`` where ``X ~ dist``.

    The result is shaped like one draw of *dist*:

    * Numeric distributions, whose draws are arrays — returns
      :class:`~probpipe.custom_types.Array`.
    * Structured distributions, whose draws are records — returns
      :class:`~probpipe.record.Record`.
    * :class:`~probpipe.RandomMeasure`, whose draws are
      distributions — returns the marginalised :class:`~probpipe.Distribution`
      with marginal ``D̄(A) = ∫ D(A) dM(D)``.

    Requires the distribution to implement :class:`SupportsMean`.
    """
    if not isinstance(dist, SupportsMean):
        raise TypeError(
            f"{type(dist).__name__} does not support mean (does not implement SupportsMean)"
        )
    return dist._mean()


@function
def variance(dist: SupportsVariance) -> Any:
    """Compute Var[X].

    Requires the distribution to implement :class:`SupportsVariance`.
    """
    if not isinstance(dist, SupportsVariance):
        raise TypeError(
            f"{type(dist).__name__} does not support variance (does not implement SupportsVariance)"
        )
    return dist._variance()


@function
def cov(dist: SupportsCovariance) -> Array:
    """Compute the covariance matrix of the flattened draw, a ``(d, d)`` array.

    The distribution's ``_cov`` returns the covariance as a linear operator,
    and the result is its dense array.

    Raises
    ------
    TypeError
        If the distribution does not implement :class:`SupportsCovariance`.
    """
    if not isinstance(dist, SupportsCovariance):
        raise TypeError(
            f"{type(dist).__name__} does not support covariance "
            f"(does not implement SupportsCovariance)"
        )
    return dist._cov().to_dense()


#: The level that the levels of an array of quantile levels are on.
_QUANTILE_LEVEL = "quantile"


@function
def quantile(dist: SupportsQuantile, q: Any) -> Any:
    """Compute quantile(s) of ``X ~ dist`` at probability level(s) ``q``.

    ``q`` is a scalar or array of probabilities in ``[0, 1]``, and the
    quantiles are computed per coordinate. The law's ``_quantile`` returns the
    event's raw form with the level axes leading in each leaf. One level is
    returned as a value of the event's kind, and an array of levels as the batch
    of those values on a level named ``quantile``. A law whose ``_quantile``
    returns a tracked term keeps that form.

    Requires the distribution to implement :class:`SupportsQuantile`. A concrete
    ``q`` outside ``[0, 1]`` raises ``ValueError`` (the check is skipped when
    ``q`` is traced, e.g. under ``jit``).
    """
    if not isinstance(dist, SupportsQuantile):
        raise TypeError(
            f"{type(dist).__name__} does not support quantile (does not implement SupportsQuantile)"
        )
    qa = jnp.asarray(q)
    if not isinstance(qa, jax.core.Tracer) and bool(jnp.any((qa < 0) | (qa > 1) | jnp.isnan(qa))):
        raise ValueError(f"quantile probabilities must lie in [0, 1]; got {q!r}")
    return _quantiles_at_their_levels(
        dist._quantile(q), tuple(qa.shape), name=getattr(dist, "name", _QUANTILE_LEVEL)
    )


def _quantiles_at_their_levels(computed: Any, level_shape: tuple[int, ...], *, name: str) -> Any:
    """Raw quantiles at *level_shape* levels, as the batch of values on the level ``quantile``.

    *computed* is the event's raw form with the level axes leading in each
    leaf. One level is returned as it is, and the function boundary wraps it at
    its kind. So is a tracked term, and so is a value whose leaves are not led by
    the level axes.
    """
    from collections.abc import Mapping

    from ..distributions._factored import _raw_record
    from ._array_backend import _event_shape_of, _is_numeric_leaf, _numpy_dtype_of
    from ._numeric_array_batch import NumericArrayBatch
    from ._specs import NumericArraySpec
    from .tracked import TrackedTerm

    if not level_shape or isinstance(computed, TrackedTerm):
        return computed
    if isinstance(computed, Mapping):
        raw = _raw_record(computed)
        if not _is_stacked_columns(raw, level_shape):
            return computed
        return _columns_at_a_level(raw, level_shape, name=name, level=_QUANTILE_LEVEL)
    if not _is_numeric_leaf(computed):
        return computed
    shape = tuple(_event_shape_of(computed))
    if shape[: len(level_shape)] != level_shape:
        return computed
    return NumericArrayBatch(
        name,
        computed,
        _QUANTILE_LEVEL,
        element_spec=NumericArraySpec(
            shape=shape[len(level_shape) :], dtype=_numpy_dtype_of(computed)
        ),
        axes_per_level=(len(level_shape),),
    )


@function
def random_log_prob(
    dist: SupportsRandomLogProb,
    value: Any = None,
    **field_kwargs: Any,
) -> RandomFunction | Distribution:
    """Return the random (normalized) log-density of a random measure.

    For a ``RandomMeasure`` ``M`` with draws ``D ~ M``, the random
    function ``x ↦ log D(x)`` is itself a callable returning a
    distribution over scalars at every input.

    When *value* is omitted, returns that callable as a
    :class:`~probpipe.RandomFunction`. When *value* is
    provided (positionally, or built from field kwargs via
    :meth:`Distribution._pack_value`), returns the array-valued distribution over
    ``log D(value)`` directly — equivalent to ``random_log_prob(dist)(value)``.
    The positional and keyword forms mirror :func:`log_prob`.

    Concrete subclasses implement a single method
    ``_random_log_prob()`` returning a ``RandomFunction``; the optional
    *value* dispatch lives entirely in this op, not on the protocol.
    """
    if not isinstance(dist, SupportsRandomLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support random_log_prob "
            f"(does not implement SupportsRandomLogProb)"
        )
    value = _resolve_value("random_log_prob", dist, value, field_kwargs, allow_none=True)
    rf = dist._random_log_prob()
    return rf if value is None else rf(value)


@function
def random_unnormalized_log_prob(
    dist: SupportsRandomUnnormalizedLogProb,
    value: Any = None,
    **field_kwargs: Any,
) -> RandomFunction | Distribution:
    """Return the random unnormalized log-density of a random measure.

    For a ``RandomMeasure`` ``M`` with draws ``D ~ M``, the random
    function ``x ↦ log D̃(x)`` (where ``D̃`` is the unnormalized density
    of ``D``) is itself a callable returning a distribution over
    scalars at every input.

    When *value* is omitted, returns that callable as a
    :class:`~probpipe.RandomFunction`. When *value* is
    provided (positionally, or built from field kwargs via
    :meth:`Distribution._pack_value`), returns the array-valued distribution over
    ``log D̃(value)`` directly — equivalent to
    ``random_unnormalized_log_prob(dist)(value)``. The positional and keyword
    forms mirror :func:`unnormalized_log_prob`.

    Concrete subclasses implement a single method
    ``_random_unnormalized_log_prob()`` returning a ``RandomFunction``;
    the optional *value* dispatch lives entirely in this op, not on
    the protocol.
    """
    if not isinstance(dist, SupportsRandomUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support random_unnormalized_log_prob "
            f"(does not implement SupportsRandomUnnormalizedLogProb)"
        )
    value = _resolve_value(
        "random_unnormalized_log_prob", dist, value, field_kwargs, allow_none=True
    )
    rf = dist._random_unnormalized_log_prob()
    return rf if value is None else rf(value)


def _split_data_kwargs(
    dist: Distribution,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate named data kwargs from inference kwargs.

    The names a given may bind are the signal: a law's ``fields`` where it
    defines them and its event components otherwise, and a kernel's given
    slots. Any kwarg whose name matches one is data (a conditioning target);
    everything else is an inference parameter.

    Guards against case-mismatched field names: a kwarg that matches a field
    only up to case (e.g. ``x=`` when the field is ``X``) is almost certainly a
    mistyped data field. Routed to ``inference_kwargs`` it would be silently
    ignored downstream (e.g. by NUTS) — a wrong result with no error — so it
    raises a :class:`TypeError` with the correct casing instead. Unknown
    kwargs that are *not* a case-variant of any field stay inference
    parameters (the inference layer validates those).

    Returns ``(data_kwargs, inference_kwargs)``.
    """
    if hasattr(dist, "fields"):
        comp_names = tuple(dist.fields)
    else:
        declaration = getattr(dist, "event_spec", None)
        comp_names = tuple(declaration.components) if declaration is not None else ()
    comp_names += tuple(getattr(dist, "given_spec", None) or ())
    comp_set = frozenset(comp_names)
    by_lower = {name.lower(): name for name in comp_names}

    data_kwargs: dict[str, Any] = {}
    inference_kwargs: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in comp_set:
            data_kwargs[k] = v
            continue
        canonical = by_lower.get(k.lower())
        if canonical is not None:
            raise TypeError(
                f"condition_on: keyword argument {k!r} does not match a field "
                f"of {type(dist).__name__}, but {canonical!r} does (case "
                f"differs) — did you mean {canonical}=...? "
                f"Fields: {comp_names}."
            )
        inference_kwargs[k] = v
    return data_kwargs, inference_kwargs


def _registry_observed(observed: Any, data_kwargs: dict[str, Any]) -> Any:
    """The observed argument in the form a registered method takes.

    Named data kwargs are bundled into one ``Record``; a positional value is
    passed through. Raises ``ValueError`` when both are given.
    """
    if not data_kwargs:
        return observed
    if observed is not None:
        raise ValueError(
            "Cannot provide both positional `observed` and named "
            f"data kwargs ({', '.join(data_kwargs)})"
        )
    from .record import Record

    return Record("observed", data_kwargs)


@function
def condition_on(
    dist: Distribution,
    observed: Any = None,
    *,
    method: str | None = None,
    exact_only: bool = False,
    **kwargs: Any,
) -> Distribution:
    """Condition a distribution on observed values.

    Observed data can be passed positionally or as named keyword
    arguments::

        # Positional (backward compatible):
        condition_on(model, y_obs)

        # Named data kwargs — bundled into a record with fields X and y:
        condition_on.with_options(n_broadcast_samples=16)(
            model, X=bootstrap["X"], y=bootstrap["y"],
        )

    When named data kwargs are distribution views from the same parent,
    the Function broadcasting machinery samples the parent once
    and distributes the fields, preserving joint correlation.

    Dispatch priority:

    1. **Explicit override** — ``method="tfp_nuts"`` (or any registered
       name) routes directly to the named inference method.
    2. **Exact conditioning** — if *dist* claims
       ``SupportsExactConditioning``, its ``_condition_on`` is called for a
       closed-form result (e.g., conjugate updates, joint marginalization).
    3. **Slice** — a factored law whose observed fields are the whole events
       of factors upstream of the rest is conditioned by the ``slice`` route
       of the ``condition_on`` operation: the result is the joint of the
       other factors at the observed values, and the inference parameters
       are its ``method_options``.
    4. **Approximate conditioning** — if *dist* claims
       ``SupportsApproximateConditioning``, its ``_condition_on`` runs, such
       as one forward pass through a pre-trained amortized posterior. An
       exact registered method outranks it, and ``exact_only=True`` skips it
       altogether.
    5. **Registry auto-select** — the inference method registry runs the
       first feasible method in selection order: exact methods before
       approximate ones, then by priority (NUTS, HMC, RWMH, etc.). A call
       with no feasible method raises ``ResolutionError``.

    Exactness is compared across routes, not only within the registry: no
    approximate route runs while an exact one applies.

    Parameters
    ----------
    dist : Distribution
        Distribution or model to condition.  Need not claim a
        conditioning capability — the registry provides inference
        methods for common model types.
    observed : Any
        Observed values to condition on.
    method : str or None
        If provided, use the named inference method from the registry
        instead of the default dispatch.
    exact_only : bool
        If ``True``, only routes that return the conditional law itself are
        considered: the approximate conditioning capability is skipped, and
        the registry excludes its approximate methods.

        ``method`` and ``exact_only`` are controls, so a field of either
        name cannot be conditioned through the named-field form. Pass it in
        the positional ``observed`` mapping instead, as in
        ``condition_on(dist, {"exact_only": value})``.
    **kwargs
        Inference parameters (e.g., ``num_results``, ``num_warmup``,
        ``random_seed``) and/or named data kwargs.  Any kwarg whose
        name matches a distribution component name is treated as
        observed data; everything else is an inference parameter.

    Returns
    -------
    Distribution
        The conditional distribution, as the selected method represents it.

    Raises
    ------
    ResolutionError
        If no registered method is feasible for *dist*; if ``method`` names a
        method that is not registered or is infeasible; or if ``exact_only``
        is set and no exact route applies.
    ValueError
        If observed values are passed both positionally and as named data
        kwargs.
    TypeError
        If a kwarg matches a component name only up to case.
    """
    from ..inference import inference_method_registry

    # Separate data kwargs (names matching fields) from
    # inference kwargs (everything else like num_results, num_warmup).
    data_kwargs, inference_kwargs = _split_data_kwargs(dist, kwargs)

    # Named data join the given, so every given value reaches the primitive in
    # one argument and the remaining keywords are the method's options.
    given = _registry_observed(observed, data_kwargs)

    # Explicit method override → always use the registry
    if method is not None:
        return inference_method_registry.execute(
            dist,
            given,
            method=method,
            exact_only=exact_only,
            **inference_kwargs,
        )

    # An exact built-in path. The given and the options pass through to
    # _condition_on, which validates the given itself; the controls stay here.
    if isinstance(dist, SupportsExactConditioning):
        return dist._condition_on(given, **inference_kwargs)

    # A factored law whose given fixes whole factors upstream of the rest is
    # sliced by the condition_on operation.
    from ..distributions._factored import SupportsFactors
    from .record import Record

    if isinstance(dist, SupportsFactors) and isinstance(given, Record | Mapping):
        from ..operations._condition import condition_on as conditioned

        sliced = conditioned.with_options(
            method="slice", exact_only=exact_only, method_options=inference_kwargs
        )
        if sliced.check(dist, given).feasible is True:
            return sliced(dist, given)

    # An approximate built-in path runs only when no exact route applies, so
    # an exact registered method outranks it and exact_only skips it.
    if not exact_only and isinstance(dist, SupportsApproximateConditioning):
        exact_candidate = inference_method_registry.check(
            dist, given, exact_only=True, **inference_kwargs
        )
        if exact_candidate.feasible is not True:
            return dist._condition_on(given, **inference_kwargs)

    # Registry auto-selects the first feasible method in selection order.
    return inference_method_registry.execute(dist, given, exact_only=exact_only, **inference_kwargs)


@function
def from_distribution(
    source: Distribution,
    target_type: type,
    *,
    key: Any | None = None,
    check_support: bool = True,
    **kwargs: Any,
) -> Any:
    """Convert *source* into an instance of *target_type*.

    Delegates to the global converter registry.

    Parameters
    ----------
    source : Distribution
        Source distribution to convert.
    target_type : type
        The target distribution class.
    key : None
        Refused when given: a conversion's draws are workflow-owned random
        events, which ``workflow_run(seed=...)`` makes reproducible.
    check_support : bool
        If ``True`` (default), verify the supports are compatible. ``False``
        reaches the selected converter only when it reads the option, as the
        moment matcher does; a sampled representation checks no support.
    **kwargs
        Additional keyword arguments passed to the converter.

    Raises
    ------
    TypeError
        If *key* is given.
    """
    from ..distributions._conversion import converter_registry

    if key is not None:
        raise TypeError(
            "from_distribution takes no key: a conversion's draws are workflow-owned random "
            "events, which workflow_run(seed=...) makes reproducible"
        )
    if not check_support:
        planned = converter_registry.check(source, target_type, **kwargs)
        if planned.method_name is not None:
            converter = converter_registry.get_method(planned.method_name)
            if "check_support" in getattr(converter, "_reads", ()):
                kwargs["check_support"] = False
    return converter_registry.convert(source, target_type, **kwargs)
