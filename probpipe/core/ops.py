"""Built-in operations for distribution computation.

Each public function (``sample``, ``mean``, ``log_prob``, …) is a
:class:`~probpipe.core.node.Function` created via the
``@function`` decorator.  This means every call automatically
participates in broadcasting and Prefect orchestration when a
distribution argument is passed where a concrete value is expected.

Usage::

    from probpipe import sample, mean, log_prob, condition_on

    dist = Normal(loc=0.0, scale=1.0)
    s = sample(dist, key=jax.random.PRNGKey(0), sample_shape=(100,))
    m = mean(dist)
    lp = log_prob(dist, jnp.array(1.5))
"""

from __future__ import annotations

import operator
from typing import Any

import jax
import jax.numpy as jnp

from ..custom_types import Array, PRNGKey
from . import _workflow_broker, _workflow_descendants
from .distribution import Distribution, RandomFunction
from .node import function
from .protocols import (
    SupportsConditioning,
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsQuantile,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
)

__all__ = [
    "condition_on",
    "cov",
    "expectation",
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
    if key is None:
        captured = _workflow_descendants.capture_stochastic_consumer(dist)
        key = _workflow_broker._resolve_automatic_key(
            None,
            _workflow_broker._singleton_effect_plan(
                operation_kind="sample",
                execution_mode="sampled",
                sample_shape=sample_shape,
                record_path=captured.record_path,
                descendant_descriptor=captured.descendant_descriptor,
            ),
        )
        return _drawn_at_its_batch_form(
            _workflow_descendants.sample_captured_consumer(captured, key, sample_shape),
            sample_shape,
            name=dist.name,
            name_is_auto=dist.name_is_auto,
        )
    return _drawn_at_its_batch_form(
        dist._sample(key, sample_shape),
        sample_shape,
        name=dist.name,
        name_is_auto=dist.name_is_auto,
    )


def _drawn_at_its_batch_form(
    drawn: Any, sample_shape: tuple[int, ...], *, name: str, name_is_auto: bool
) -> Any:
    """Give a multi-draw result the batch form of the draw's kind.

    A non-empty ``sample_shape`` puts those leading dimensions on one level named
    for the operation that mints them, ``sample`` (design V.2, V.9). The level is
    minted here, at the boundary, for every kind of draw: a law that assembled the
    draws itself — as a record of columns, or as an array of stored objects — did
    not name what its leading axes range over, and reading them as event shape
    says the draws were one wide value.

    The event shape is read off the draw, which stays exact for a law whose event
    shape is symbolic until a draw binds it. The batch takes the law's own *name*,
    since the draws are that law's, and carries the law's *name_is_auto* with it:
    the name is a caller's statement exactly when the caller's name for the law
    was one.

    A law that built its own batch already named the level, and a term of some
    other kind is left as it is.
    """
    from ._array_backend import _event_shape_of, _is_numeric_leaf, _numpy_dtype_of
    from ._batch import Batch
    from ._broadcast_distributions import SAMPLE_LEVEL, _make_stack
    from ._numeric_array_batch import NumericArrayBatch
    from ._object_batch import _is_object_array
    from ._record_batch import _batch_class_for
    from .event_template import NumericArraySpec, _reshaped_template
    from .record import Record
    from .tracked import TrackedTerm

    if not sample_shape or isinstance(drawn, Batch):
        return drawn

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
            name_is_auto=name_is_auto,
        )

    if _is_object_array(drawn):
        # Stored draws aggregate exactly as a sweep's rows do — each element at
        # its own kind, under the one level the operation mints.
        aggregate = _make_stack(
            list(drawn.reshape(-1)),
            batch_shape=tuple(sample_shape),
            level_names=(SAMPLE_LEVEL,),
            field_name=name,
            name=name,
        )
        # An aggregation names its result for the function that produced the rows
        # and marks that auto. Here the name is the law's, so whether it was a
        # caller's statement is the law's answer, not this boundary's.
        object.__setattr__(aggregate, "_name_is_auto", name_is_auto)
        return aggregate

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
        name_is_auto=name_is_auto,
    )


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
    from .event_template import NumericArraySpec

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
        name_is_auto=operand.name_is_auto,
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

    The return type is ``T``-shaped where ``T`` is *dist*'s sample type:

    * Numeric distributions (``T = Array``) — returns
      :class:`~probpipe.custom_types.Array`.
    * Structured distributions (``T = Record``) — returns
      :class:`~probpipe.record.Record`.
    * :class:`~probpipe.core._random_measures.RandomMeasure[T]` (``T``
      itself a :class:`~probpipe.core._distribution_base.Distribution[T]`)
      — returns the marginalised ``Distribution[T]`` with marginal
      ``D̄(A) = ∫ D(A) dM(D)``.

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
    """Compute the covariance matrix.

    Requires the distribution to implement :class:`SupportsCovariance`.
    """
    if not isinstance(dist, SupportsCovariance):
        raise TypeError(
            f"{type(dist).__name__} does not support covariance "
            f"(does not implement SupportsCovariance)"
        )
    return dist._cov()


@function
def quantile(dist: SupportsQuantile, q: Any) -> Any:
    """Compute quantile(s) of ``X ~ dist`` at probability level(s) ``q``.

    ``q`` is a scalar or array of probabilities in ``[0, 1]``; the return is
    ``T``-shaped per field (finite-sample distributions return the weight-aware
    empirical quantile via ``_quantile``).

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
    return dist._quantile(q)


@function
def expectation(
    dist: SupportsExpectation,
    f: Any,
    *,
    key: PRNGKey | None = None,
    num_evaluations: int | None = None,
    return_dist: bool | None = None,
) -> Any:
    """Compute E[f(X)] where X ~ dist."""
    if not isinstance(dist, SupportsExpectation):
        raise TypeError(f"{type(dist).__name__} does not support expectation")
    return dist._expectation(
        f,
        key=key,
        num_evaluations=num_evaluations,
        return_dist=return_dist,
    )


@function
def random_log_prob(
    dist: SupportsRandomLogProb,
    value: Any = None,
    **field_kwargs: Any,
) -> RandomFunction | Distribution:
    """Return the random (normalized) log-density of a random measure.

    For a ``RandomMeasure[T]`` ``M`` with draws ``D ~ M``, the random
    function ``x ↦ log D(x)`` is itself a callable returning a
    distribution over scalars at every input.

    When *value* is omitted, returns that callable as a
    :class:`~probpipe.core._random_functions.RandomFunction`. When *value* is
    provided (positionally, or built from field kwargs via
    :meth:`Distribution._pack_value`), returns the ``Distribution[Array]`` over
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

    For a ``RandomMeasure[T]`` ``M`` with draws ``D ~ M``, the random
    function ``x ↦ log D̃(x)`` (where ``D̃`` is the unnormalized density
    of ``D``) is itself a callable returning a distribution over
    scalars at every input.

    When *value* is omitted, returns that callable as a
    :class:`~probpipe.core._random_functions.RandomFunction`. When *value* is
    provided (positionally, or built from field kwargs via
    :meth:`Distribution._pack_value`), returns the ``Distribution[Array]`` over
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

    Uses the distribution's ``fields`` as the sole signal:
    any kwarg whose name matches a component name is data (conditioning
    target); everything else is an inference parameter.

    Guards against case-mismatched field names: a kwarg that matches a field
    only up to case (e.g. ``x=`` when the field is ``X``) is almost certainly a
    mistyped data field. Routed to ``inference_kwargs`` it would be silently
    ignored downstream (e.g. by NUTS) — a wrong result with no error — so it
    raises a :class:`TypeError` with the correct casing instead. Unknown
    kwargs that are *not* a case-variant of any field stay inference
    parameters (the inference layer validates those).

    Returns ``(data_kwargs, inference_kwargs)``.
    """
    comp_names = tuple(dist.fields) if hasattr(dist, "fields") else ()
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


@function
def condition_on(
    dist: Distribution,
    observed: Any = None,
    *,
    method: str | None = None,
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
    2. **Exact conditioning** — if *dist* implements
       ``SupportsConditioning``, its ``_condition_on`` is called for a
       closed-form result (e.g., conjugate updates, joint marginalization).
    3. **Registry auto-select** — the inference method registry picks
       the highest-priority feasible algorithm (NUTS, HMC, RWMH, etc.).

    Parameters
    ----------
    dist : Distribution
        Distribution or model to condition.  Need not implement
        ``SupportsConditioning`` — the registry provides inference
        methods for common model types.
    observed : Any
        Observed values to condition on.
    method : str or None
        If provided, use the named inference method from the registry
        instead of the default dispatch.
    **kwargs
        Inference parameters (e.g., ``num_results``, ``num_warmup``,
        ``random_seed``) and/or named data kwargs.  Any kwarg whose
        name matches a distribution component name is treated as
        observed data; everything else is an inference parameter.
    """
    from ..inference import inference_method_registry
    from .record import Record

    # Separate data kwargs (names matching fields) from
    # inference kwargs (everything else like num_results, num_warmup).
    data_kwargs, inference_kwargs = _split_data_kwargs(dist, kwargs)

    # Explicit method override → always use the registry
    if method is not None:
        if data_kwargs:
            if observed is not None:
                raise ValueError(
                    "Cannot provide both positional `observed` and named "
                    f"data kwargs ({', '.join(data_kwargs)})"
                )
            observed = Record("observed", data_kwargs, name_is_auto=True)
        return inference_method_registry.execute(dist, observed, method=method, **inference_kwargs)

    # Exact conditioning (conjugate updates, joint marginalization, etc.)
    # All kwargs pass through to _condition_on — it handles its own
    # validation (e.g., ProductDistribution raises KeyError on unknown names).
    if isinstance(dist, SupportsConditioning):
        return dist._condition_on(observed, **data_kwargs, **inference_kwargs)

    # Registry auto-selects the best approximate inference algorithm.
    # Data kwargs are bundled into observed as a Record object.
    if data_kwargs:
        if observed is not None:
            raise ValueError(
                "Cannot provide both positional `observed` and named "
                f"data kwargs ({', '.join(data_kwargs)})"
            )
        observed = Record("observed", data_kwargs, name_is_auto=True)
    return inference_method_registry.execute(dist, observed, **inference_kwargs)


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
    key : PRNGKey, optional
        JAX PRNG key for sampling-based conversion.
    check_support : bool
        If ``True`` (default), verify the supports are compatible.
    **kwargs
        Additional keyword arguments passed to the converter.
    """
    from ..converters import converter_registry

    return converter_registry.convert(
        source, target_type, key=key, check_support=check_support, **kwargs
    )
