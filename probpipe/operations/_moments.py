"""The distribution functionals: the moments, ``quantile``, and ``expectation``.

The moments are ``mean``, ``variance``, ``standard_deviation``, and ``cov``.
Each functional summarizes a distribution by a deterministic value. ``mean``,
``variance``, ``cov``, and ``quantile`` carry a capability route on their
matching protocol, ``closed_form``, and a Monte Carlo fallback,
``monte_carlo``, on the event kinds where the required averaging is defined. A
capability returns the law's own moment, and a numeric fallback returns that
moment of the empirical law of its draws. ``mean``, ``variance``, and
``quantile`` are computed coordinate by coordinate, so they also carry the
structural route ``by_output``: for a law whose event is a record, it splits
the event into its top-level outputs and computes each output that has an
exact route of its own by that route, on the output's view, and the outputs
without one together, as one block, so a dependent joint's root factors keep
their closed forms. ``cov`` couples the outputs, so it has no such route.

``cov(d1, d2)`` is the cross-covariance of two laws, the block of rows ``d1``
and columns ``d2`` of the covariance of the two draws taken together, with
the laws drawn as a lift draws them. Laws whose draws are coordinates of one
law's draws take ``coupled_marginals``, which runs ``cov`` on that law's
coupled marginal at those coordinates. Unrelated laws take ``independent``,
an exact zero, and refuse laws that are drawn independently but share a
factor. Other coupled laws take ``joint_lift``, which runs ``cov`` on the
law of the pair of their draws. An omitted ``d2``, or ``d1`` itself, takes
the one-law routes.

A moment of the event's kind keeps the event's outputs, in the same form, and
derives its term specs, support included, so the mean of a law over ``mu`` and
``tau`` is a record whose fields are ``mu`` and ``tau``. Selection therefore
commutes with a summary: for a law whose outputs are a record's fields, the part of
``mean(d)`` at an event path ``p`` and ``mean(d[p])`` have one estimand, and
they are equal where both take exact routes. ``cov`` is declared under the
output ``cov``.
A moment of ``d`` is labeled by the moment of a draw of ``d``, as
``𝔼[(mu, tau) ~ d]``, ``𝕍[...]``, ``√𝕍[...]``, ``ℂ[...]``, or ``ℚ[...]``, and
the law of a function lifted over laws contributes its notation in place of the
draw, as ``𝔼[f(beta ~ model; y)]``.

``expectation(d, f)`` returns ``𝔼[f(X)]`` for ``X ~ d``, labeled
``𝔼[f(mu ~ d)]``. It is the derived
operation ``mean(pushforward(d, f))``, so the integrand receives draws of
``d`` on every route: a law claiming
:class:`~probpipe.distributions._capabilities.SupportsExpectation` computes it
in closed form, and otherwise the route ``pushforward_mean`` builds the
pushforward by an evaluation rule and takes its mean. Its method names are
therefore the evaluation rules', and an integration rule registered with the
evaluation-rule registry serves both operations. The route spends the call's
``num_samples`` as the pushforward's ``max_evaluations``, since each draw is
one evaluation of the integrand.

``standard_deviation(d)`` is the derived operation ``sqrt(variance(d))``, taken
coordinate by coordinate. Its one route is the body route of its formula, so
the call takes the routes of ``variance``, ``by_output`` included, and the
derived route passes the call's ``exact_only``, ``max_evaluations``, and
``num_samples`` to ``variance``.

The draws of every Monte Carlo route are workflow-owned random events, and
their number is the ``num_samples`` control, which the call's provenance
records.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from math import prod
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..core._batch import Batch, BatchSpec, _flat_batch_spec
from ..core._call_context import (
    _collecting_call_accumulators,
    _note_num_samples_used,
    _NumSamplesUsed,
)
from ..core._dispatch import (
    Feasibility,
    MathematicalDomainError,
    ResolutionError,
)
from ..core._expression import (
    _SUMMARY_SYMBOLS,
    Applied,
    Expression,
    Summary,
    draw_of,
)
from ..core._record_batch import RecordBatch, _batch_class_for, _batch_over
from ..core._record_spec import RecordSpec
from ..core._repr import format_names
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec
from ..core._specs import OutputSpec, _has_record_outputs
from ..core.constraints import (
    _Boolean,
    _IntegerInterval,
    _NonNegativeInteger,
    _Sphere,
    interval,
    non_negative,
    unit_interval,
)
from ..core.tracked import TrackedTerm
from ..distributions._batches import _element_source
from ..distributions._capabilities import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsMean,
    SupportsQuantile,
    SupportsVariance,
    _capability_guard,
)
from ..distributions._distribution import Distribution, DistributionSpec, _value_at_declared_kind
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._factored import _raw_record
from ..distributions._merge import _coupled_part_pairs
from ..distributions._views import (
    CoupledMarginal,
    _coordinate_range,
    _copy_source,
    _leaf_paths,
)
from ..functions import function
from ..functions._call import ApplicabilityError, presented_arguments
from ..functions._descendants import _distribution_root
from ..functions._resolution import PointReport, RouteOutcome, check_point, selected, settled
from ..functions._result import SAMPLE_LEVEL
from ..functions._rules import _FORWARDED_CONTROLS
from ..values import Function, FunctionSpec
from ._operation import (
    BoundCall,
    Operation,
    RouteSource,
    _assembled,
    _call_label,
    _can_sample,
    _DelegatingRoute,
    _install_expression_rule,
    _law_samples,
    _monte_carlo_draws,
    operation,
)
from ._pushforward import (
    _as_function,
    _bound_parameter,
    _call_at,
    _EvaluationRules,
    _integrand_applicability_checks,
    _output_label,
    pushforward,
)
from ._sample import _record_batch

__all__ = [
    "cov",
    "expectation",
    "mean",
    "quantile",
    "standard_deviation",
    "variance",
]


# ---------------------------------------------------------------------------
# The output specs of the moments
# ---------------------------------------------------------------------------


def _floating(dtype: Any) -> Any:
    """*dtype* when it is floating or unset, and otherwise the default floating dtype."""
    if dtype is None or jnp.issubdtype(dtype, jnp.inexact):
        return dtype
    return np.dtype(jnp.result_type(float))


def _hull(support: Any) -> Any:
    """The declared support that holds every average of points of *support*.

    A discrete support becomes the interval it spans, and a sphere, whose
    averages fill the ball, leaves the support undeclared.
    """
    if isinstance(support, _Boolean):
        return unit_interval
    if isinstance(support, _NonNegativeInteger):
        return non_negative
    if isinstance(support, _IntegerInterval):
        return interval(support.low, support.high)
    if isinstance(support, _Sphere):
        return None
    return support


def _elementwise(spec: BatchSpec, term: Callable[[TermSpec], TermSpec]) -> BatchSpec:
    """The batch spec on *spec*'s levels whose elements' spec is *term* of *spec*'s, under the same name."""
    element_spec = spec.element_spec._with_term_spec(term(spec.element_spec.term_spec))
    return BatchSpec._from_groups(element_spec, spec.axis_groups, spec.level_names)


def _mean_term(spec: TermSpec) -> TermSpec:
    """The term spec of a mean: floating leaves on the hull of their support.

    A batch's mean is the batch of its elements' means, on the same levels.
    """
    if isinstance(spec, BatchSpec):
        return _elementwise(spec, _mean_term)
    if isinstance(spec, NumericArraySpec):
        return NumericArraySpec(spec.shape, _floating(spec.dtype), _hull(spec.support))
    if isinstance(spec, RecordSpec):
        return RecordSpec({name: _mean_term(child) for name, child in spec.children.items()})
    return spec


def _variance_term(spec: TermSpec) -> TermSpec:
    """The term spec of a variance: non-negative floating leaves.

    A batch's variance is the batch of its elements' variances, on the same levels.
    """
    if isinstance(spec, BatchSpec):
        return _elementwise(spec, _variance_term)
    if isinstance(spec, NumericArraySpec):
        return NumericArraySpec(spec.shape, _floating(spec.dtype), non_negative)
    if isinstance(spec, RecordSpec):
        return RecordSpec({name: _variance_term(child) for name, child in spec.children.items()})
    return spec


def _quantile_term(spec: TermSpec) -> TermSpec:
    """The term spec of a quantile: the event's shapes, with floating dtypes kept.

    A batch's quantile is the batch of its elements' quantiles, on the same levels.
    """
    if isinstance(spec, BatchSpec):
        return _elementwise(spec, _quantile_term)
    if isinstance(spec, NumericArraySpec):
        dtype = (
            spec.dtype if spec.dtype is not None and _floating(spec.dtype) == spec.dtype else None
        )
        return NumericArraySpec(spec.shape, dtype)
    if isinstance(spec, RecordSpec):
        return RecordSpec({name: _quantile_term(child) for name, child in spec.children.items()})
    return spec


def _array_leaves(spec: TermSpec) -> list[NumericArraySpec]:
    """The array leaves of a numeric spec, in the canonical flat order.

    A batch's leaves are its element's leaves, each led by the batch's axes.
    """
    if isinstance(spec, BatchSpec):
        batch_shape = tuple(spec.batch_shape)
        return [
            NumericArraySpec((*batch_shape, *leaf.shape), leaf.dtype)
            for leaf in _array_leaves(spec.element_spec.term_spec)
        ]
    if isinstance(spec, NumericArraySpec):
        return [spec]
    if isinstance(spec, RecordSpec):
        return [leaf for child in spec.children.values() for leaf in _array_leaves(child)]
    return []


def _summary_declaration(declaration: OutputSpec, term: TermSpec) -> OutputSpec:
    """The output spec of a summary of each output of *declaration*, whose type is *term*.

    The summary takes the outputs of *declaration*, in the same form, so the mean
    of a law over ``mu`` and ``tau`` is a record whose fields are ``mu`` and
    ``tau``, and the mean of a law over ``theta`` is declared under ``theta``.
    A summary whose type is a law, as the mean of a law over laws is, takes
    the positional form over that law, since a law's outputs are its event's.
    """
    if isinstance(term, DistributionSpec):
        return OutputSpec(term)
    return declaration._with_term_spec(term)


def _capability(method: str) -> Callable[[BoundCall, OutputSpec | None], Any]:
    """The law's capability *method*, called on the call's other arguments."""

    def call_capability(call: BoundCall, result: OutputSpec | None) -> Any:
        others = [value for name, value in call.operands.items() if name != "d"]
        return getattr(call.operands["d"], method)(*others)

    call_capability.__doc__ = f"``d.{method}()``."
    return call_capability


def _mean_result(d: DistributionSpec) -> OutputSpec:
    """A value of the event's kind, under the event's outputs and in the same form.

    Its term specs are those of an average: floating, on the hull of the
    event's support, so the mean of a Bernoulli event lies in the unit interval.
    """
    return _summary_declaration(d.event_spec, _mean_term(d.event_spec.term_spec))


def _variance_result(d: DistributionSpec) -> OutputSpec:
    """A value of the event's kind with non-negative floating leaves, under the event's outputs."""
    return _summary_declaration(d.event_spec, _variance_term(d.event_spec.term_spec))


def _flat_size(spec: TermSpec) -> int | None:
    """The number of coordinates of a flat draw of a numeric *spec*, or None while a dimension is free."""
    leaves = _array_leaves(spec)
    if any(leaf.free_dims for leaf in leaves):
        return None
    return sum(prod(leaf.shape) for leaf in leaves)


def _cov_result(d1: DistributionSpec, d2: DistributionSpec | None) -> OutputSpec | None:
    """The dense covariance of the flat draws, a ``(size of d1, size of d2)`` array under the output ``cov``.

    An omitted *d2* is read as *d1*. Each size is the number of the event's
    coordinates; an event with free dimensions leaves the shape to the
    returned value.
    """
    name = "cov"
    events = [d1.event_spec.term_spec]
    if d2 is not None:
        events.append(d2.event_spec.term_spec)
    sizes = [_flat_size(event) for event in events]
    if None in sizes:
        return OutputSpec(**{name: None})
    rows, columns = sizes[0], sizes[-1]
    return OutputSpec(**{name: NumericArraySpec((rows, columns), _cov_dtype(events))})


def _cov_dtype(events: Sequence[TermSpec]) -> Any:
    """The floating dtype every leaf of *events* shares, or None when they differ or are unset."""
    dtypes = {_floating(leaf.dtype) for event in events for leaf in _array_leaves(event)}
    return dtypes.pop() if len(dtypes) == 1 else None


def _quantile_result(d: DistributionSpec, q: TermSpec) -> OutputSpec:
    """One level returns the event's kind, and plural levels add a level named quantile.

    Parameters
    ----------
    d : DistributionSpec
        The law's spec, whose event spec the quantile's output spec
        follows.
    q : TermSpec
        The spec of the levels: a scalar for one level, or an array whose shape
        gives the axes of the ``quantile`` level.

    Returns
    -------
    OutputSpec
        The event spec, with its outputs, form, and shapes.
        Plural levels give its batch form.

    Raises
    ------
    ApplicabilityError
        If the levels are not numeric.
    """
    if not isinstance(q, NumericArraySpec):
        given = getattr(q, "type", None)
        name = given.__name__ if isinstance(given, type) else type(q).__name__
        hint = "; pass jnp.asarray(q)" if given in (list, tuple) else ""
        raise ApplicabilityError(
            f"quantile: q must be a number or an array of numbers; got {name}{hint}"
        )
    element = _summary_declaration(d.event_spec, _quantile_term(d.event_spec.term_spec))
    if not q.shape:
        return element
    return element._with_term_spec(BatchSpec(element.term_spec, quantile=q.shape))


def _expectation_result(f: TermSpec) -> OutputSpec | None:
    """The mean of the integrand's output spec, as ``mean(pushforward(d, f))`` declares it.

    It takes the integrand's outputs, in the same form, with an average's term
    specs. An integrand that declares no output leaves the output spec to the
    returned value.
    """
    if (
        isinstance(f, FunctionSpec)
        and f.output_spec is not None
        and f.output_spec.term_spec is not None
    ):
        return _summary_declaration(f.output_spec, _mean_term(f.output_spec.term_spec))
    return None


def _is_numeric(spec: TermSpec | None) -> bool:
    """Whether *spec* declares a numeric value: an array, a record of arrays, or a batch of either."""
    if isinstance(spec, BatchSpec):
        spec = _flat_batch_spec(spec).element_spec.term_spec
    return isinstance(spec, NumericSpec)


def _numeric_event(d: DistributionSpec) -> bool:
    """The law draws a numeric value: an array, a record of arrays, or a batch of either."""
    return _is_numeric(d.event_spec.term_spec)


def _numeric_events(d1: DistributionSpec, d2: DistributionSpec | None) -> bool:
    """Each law draws a numeric value: an array, a record of arrays, or a batch of either."""
    return _is_numeric(d1.event_spec.term_spec) and (
        d2 is None or _is_numeric(d2.event_spec.term_spec)
    )


def _event_typed_variance(d: DistributionSpec) -> bool:
    """The event has an event-typed variance, which a measure-valued event lacks in general."""
    return not isinstance(d.event_spec.term_spec, DistributionSpec)


# ---------------------------------------------------------------------------
# The Monte Carlo fallbacks
# ---------------------------------------------------------------------------


def _can_average(call: BoundCall, result: OutputSpec | None) -> Feasibility:
    """The law samples numeric draws, which average coordinatewise, or laws, which average to a mixture.

    The estimate assumes the mean is finite. The average of function-valued
    draws is not implemented.
    """
    event = call.operands["d"].event_spec.term_spec
    if isinstance(event, FunctionSpec):
        return Feasibility(False, "the average of function-valued draws is not implemented yet")
    if not (_is_numeric(event) or isinstance(event, DistributionSpec)):
        return Feasibility(
            False, f"cannot average draws of a non-numeric event; got {type(event).__name__}"
        )
    return _can_sample(call, result)


def _can_average_squares(call: BoundCall, result: OutputSpec | None) -> Feasibility:
    """The law samples numeric draws, whose variance is taken coordinatewise.

    The estimate assumes the variance is finite. The pointwise variance of
    function-valued draws is not implemented.
    """
    event = call.operands["d"].event_spec.term_spec
    if isinstance(event, FunctionSpec):
        return Feasibility(
            False, "the pointwise variance of function-valued draws is not implemented yet"
        )
    if not _is_numeric(event):
        return Feasibility(
            False,
            f"cannot take the variance of draws of a non-numeric event; got {type(event).__name__}",
        )
    return _can_sample(call, result)


#: The label, the output, and the level of the empirical law of a fallback's draws.
_DRAWS = "draws"


def _empirical_of(call: BoundCall, draws: Any, *, operand: str = "d") -> EmpiricalDistribution:
    """The empirical law of the draws, whose moments the fallbacks report.

    Draws of an array event are its atoms along their leading axis. Draws of a
    record event are its atoms as the batch of records the law's event
    spec calls for, whether they arrive as a nested mapping of raw
    columns or as a record of columns. A batch of records is taken as it is.
    Draws of a batch-valued event are its atoms on one level before the
    event's own levels.
    """
    # The empirical law is internal, so it takes a fixed name, which a derived
    # label could not be: a label may hold ``~``, a space, ``;``, or ``/``.
    name = _DRAWS
    event = call.operands[operand].event_spec.term_spec
    if isinstance(event, BatchSpec):
        if isinstance(draws, (RecordBatch, Mapping)):
            raw = _raw_record(draws)
        else:
            raw = jnp.asarray(draws.raw if isinstance(draws, Batch) else draws)
        level = name
        while level in _flat_batch_spec(event).level_names:
            level += "_"
        count = jax.tree.leaves(raw)[0].shape[0]
        atoms = _batch_over(raw, _flat_batch_spec(BatchSpec(event, {level: count})), name)
        return EmpiricalDistribution(atoms, label=name, event_spec=OutputSpec(**{name: event}))
    if isinstance(draws, Batch):
        records = isinstance(draws.element_spec.term_spec, RecordSpec)
        return EmpiricalDistribution(draws, output_name=None if records else name, label=name)
    if not isinstance(event, RecordSpec):
        return EmpiricalDistribution(
            jnp.asarray(draws), output_name=name, label=name, level=SAMPLE_LEVEL
        )
    atoms = _batch_class_for(event)(
        _raw_record(draws),
        SAMPLE_LEVEL,
        element=event,
        label=name,
    )
    return EmpiricalDistribution(atoms, label=name)


_mixture_factory: Callable[..., Distribution] | None = None
"""The finite mixture ``(components, weights, *, label)``, which the mixture family installs."""


def _install_mixture(factory: Callable[..., Distribution]) -> None:
    """Install the finite mixture that the Monte Carlo mean of a law over laws returns."""
    global _mixture_factory
    _mixture_factory = factory


def _mc_mean(call: BoundCall, result: OutputSpec | None) -> Any:
    """The mean of the empirical law of the draws.

    For a numeric event it is their coordinatewise average, and for an event
    whose draws are laws it is their finite mixture with equal weights, the
    Monte Carlo estimate of the mean measure.
    """
    if _is_numeric(call.operands["d"].event_spec.term_spec):
        return _empirical_of(call, _monte_carlo_draws(call, "mean"))._mean()
    if _mixture_factory is None:
        raise RuntimeError("the mixture family is not installed; import probpipe")
    draws = _monte_carlo_draws(call, "mean")
    stored = draws.raw if isinstance(draws, Batch) else draws
    laws = list(np.asarray(stored, dtype=object).reshape(-1))
    return _mixture_factory(laws, jnp.full(len(laws), 1.0 / len(laws)), label=_call_label(call))


def _mc_variance(call: BoundCall, result: OutputSpec | None) -> Any:
    """The variance of the empirical law of the draws.

    Each coordinate's variance is the mean squared deviation of the draws from
    their mean, dividing by the number of draws.
    """
    return _empirical_of(call, _monte_carlo_draws(call, "variance"))._variance()


def _dense(covariance: Any) -> Any:
    """A covariance as a dense array, whether an operator or an array holds it."""
    to_dense = getattr(covariance, "to_dense", None)
    return to_dense() if callable(to_dense) else jnp.asarray(covariance)


def _mc_cov(call: BoundCall, result: OutputSpec | None) -> Any:
    """The covariance of the empirical law of the draws of ``d1``, over their flat coordinates.

    It divides by the number of draws, as the variance does, so its diagonal
    is the variance and one draw has none.
    """
    draws = _monte_carlo_draws(call, "cov", operand="d1")
    return _dense(_empirical_of(call, draws, operand="d1")._cov())


def _check_levels(q: Any) -> Any:
    """*q* as an array of levels in ``[0, 1]``.

    A traced *q*, as under ``jit``, is not checked.

    Parameters
    ----------
    q : float or array-like
        One level, or an array of levels.

    Returns
    -------
    jax.Array
        The levels as a JAX array of *q*'s shape.

    Raises
    ------
    MathematicalDomainError
        If a concrete level lies outside ``[0, 1]`` or is NaN.
    """
    levels = jnp.asarray(q)
    if not isinstance(levels, jax.core.Tracer) and bool(
        jnp.any((levels < 0) | (levels > 1) | jnp.isnan(levels))
    ):
        raise MathematicalDomainError(f"quantile levels must lie in [0, 1]; got {q!r}")
    return levels


def _mc_quantile(call: BoundCall, result: OutputSpec | None) -> Any:
    """The quantiles of the empirical law of the draws, the level axes leading.

    Each coordinate's quantile at a level ``q`` is the generalized inverse
    ``inf{x : F(x) >= q}`` of the draws' CDF, the rule of the empirical law's
    own quantiles. A record event's quantiles at several levels are the
    declared batch of records.
    """
    levels = _check_levels(call.operands["q"])
    draws = _monte_carlo_draws(call, "quantile")
    quantiles = _empirical_of(call, draws)._quantile(levels)
    return _record_batch(quantiles, call, result)


# ---------------------------------------------------------------------------
# A call of the operation inside a call
# ---------------------------------------------------------------------------


def _inner_controls(call: BoundCall, exact_only: bool) -> dict[str, Any]:
    """The controls a call made inside *call* is resolved under: *call*'s, with ``method`` unset.

    The inner call returns its raw value, which the route that makes it assembles.
    """
    return {**call.controls, "method": None, "exact_only": exact_only, "raw": True}


def _inner_report(call: BoundCall, values: Mapping[str, Any], exact_only: bool) -> Feasibility:
    """The report of the call's operation on *values*: the route it selects and that route's exactness.

    Values the operation does not apply to, as a measure-valued output has no
    event-typed variance, are infeasible.
    """
    operation = call.operation
    controls = _inner_controls(call, exact_only)
    try:
        return check_point(operation, values, controls, operation._route_candidates(controls))
    except ApplicabilityError as error:
        return Feasibility(False, str(error))


def _inner_run(
    call: BoundCall, values: Mapping[str, Any], exact_only: bool
) -> tuple[Any, dict[str, Any]]:
    """The raw result of the call's operation on *values*, with the metadata of how it was computed.

    The inner call is resolved and run as the operation runs a call on
    *values*, inside *call*, so its draws count toward *call*'s. The metadata
    names the route that computed it, that route's exactness, and, when the
    route drew Monte Carlo samples, their number as ``num_samples``.
    """
    operation = call.operation
    controls = _inner_controls(call, exact_only)
    route_inputs, result_declaration, _ = operation._plan_route_inputs(values, controls)
    candidates = operation._route_candidates(controls)
    candidate, report = selected(
        operation.label, controls, candidates, route_inputs, result_declaration
    )
    with _collecting_call_accumulators("always") as inner_context:
        inner_value, settled_exactness = settled(
            candidate.run(route_inputs, result_declaration, report)
        )
    exactness = (
        candidate.exactness(report) if settled_exactness is None else settled_exactness
    ) is True
    metadata: dict[str, Any] = {"route": candidate.route_name, "exact": exactness}
    num_samples_used = cast(_NumSamplesUsed, inner_context.num_samples_used)
    if num_samples_used.provenance_num_samples is not None:
        metadata["num_samples"] = num_samples_used.provenance_num_samples
    return inner_value, metadata


# ---------------------------------------------------------------------------
# The summaries output by output
# ---------------------------------------------------------------------------


def _view(d: Distribution, outputs: tuple[str, ...]) -> Distribution:
    """The view of *d* at one output, or the selection of several."""
    return d[outputs[0]] if len(outputs) == 1 else d[outputs]


def _block_values(view: Distribution, value: Any) -> dict[str, Any]:
    """The raw *value* of a summary of *view*, keyed by the outputs of the view's event.

    A view whose outputs are a record's fields returns the summary keyed by
    them, and a view in the keyword form returns its one output's summary.
    """
    declaration = view.event_spec
    if _has_record_outputs(declaration):
        return {name: value[name] for name in declaration.outputs}
    ((name, _),) = declaration.outputs.items()
    return {name: value}


@dataclass(frozen=True)
class _Block:
    """Outputs of the law that one resolution of the operation computes together.

    Attributes
    ----------
    outputs : tuple of str
        The outputs, in event order.
    view : Distribution
        The law's view at the outputs: the view of one, or the selection of
        several.
    report : Feasibility
        The operation's report on the view, which names the route it selects
        and that route's exactness.
    """

    outputs: tuple[str, ...]
    view: Distribution
    report: Feasibility


@dataclass(frozen=True)
class _Partition:
    """How ``by_output`` splits a call: its blocks in event order, and the split's report."""

    blocks: tuple[_Block, ...]
    report: Feasibility


class _ByOutput(_DelegatingRoute):
    """The summary of each top-level output of a record event, by the route its view selects.

    ``mean``, ``variance``, and ``quantile`` are computed coordinate by
    coordinate, so the summary of a law ``d`` whose event is a record is
    the record of the summaries of the views ``d[B]`` over any partition of its
    outputs into blocks ``B``. The route forms the partition from the
    operation's own resolution on each output's view: an output for which
    the operation selects an exact route is a block of its own, and the outputs
    without one form one block, whose draws they share. Each block is resolved
    and run as the operation resolves a call on its view, under the call's
    controls with ``method`` unset and ``exact_only`` that of the candidate,
    inside the call itself, so its draws are the call's own.

    The route's exactness is that of its blocks: it is exact when every block's
    route is, and approximate otherwise. It applies when the event is a record
    of at least two outputs, at least one of which has an exact route, and
    the outputs without one have a route together, so a law none of whose
    outputs has an exact route keeps the Monte Carlo fallback. It splits the
    top-level outputs only, so a record-valued output is one block, whose
    route its view selects. The result's provenance names, for each block, its
    outputs, its route, that route's exactness, and the ``num_samples`` of a
    block whose route drew Monte Carlo samples.

    Parameters
    ----------
    summary : str
        The operation's name, which the route's error messages give:
        ``"mean"``, ``"variance"``, or ``"quantile"``.
    """

    source = RouteSource.STRUCTURAL

    def __init__(self, summary: str) -> None:
        super().__init__("by_output")
        self.summary = summary

    @property
    def condition(self) -> str:
        """The route's feasibility condition in words."""
        return (
            "the event is a record of several outputs, at least one of which has an exact "
            "route of its own, and the rest of which have a route together"
        )

    @staticmethod
    def _values(call: BoundCall, view: Distribution) -> dict[str, Any]:
        """The call's arguments, with the law replaced by *view*."""
        return {**call.operands, "d": view}

    def _resolve(self, call: BoundCall, view: Distribution, exact_only: bool) -> Feasibility:
        """The operation's report on *view*: the route it selects and that route's exactness.

        A view the operation does not apply to, as a measure-valued output
        has no event-typed variance, is infeasible.
        """
        return _inner_report(call, self._values(call, view), exact_only)

    def _memo(self, call: BoundCall) -> dict[Any, Any]:
        """This route's entries in the call's memo, which last as long as the call."""
        return call._memo.setdefault(self.name, {})

    def _view_at(self, call: BoundCall, outputs: tuple[str, ...]) -> Distribution:
        """The law's view at *outputs*, built once per call."""
        views = self._memo(call).setdefault("views", {})
        if outputs not in views:
            views[outputs] = _view(call.operands["d"], outputs)
        return views[outputs]

    def _classified(self, call: BoundCall) -> dict[str, Feasibility]:
        """Each output's report restricted to exact routes, resolved once per call.

        An output is exact exactly when this report is feasible. Both of the
        route's candidates read it: the operation selects an exact route on a
        view whenever one applies, so restricting the output's resolution to
        exact routes finds the outputs that are exact under either.
        """
        memo = self._memo(call)
        if "outputs" not in memo:
            memo["outputs"] = {
                c: self._resolve(call, self._view_at(call, (c,)), True)
                for c in call.operands["d"].event_spec.outputs
            }
        return memo["outputs"]

    def _partition(self, call: BoundCall, exact_only: bool) -> _Partition:
        """The blocks of the call, in event order, with the report of the split.

        The split under each restriction is computed once per call, so the run
        reuses the split its candidate's probe made.

        Parameters
        ----------
        call : BoundCall
            The call, whose law is ``call.operands["d"]``.
        exact_only : bool
            Whether the block of the outputs without an exact route is
            restricted to exact routes.

        Returns
        -------
        _Partition
            The blocks and the split's report, which is infeasible when the
            event is not a record of several outputs, when no output has
            an exact route, or when the outputs without one have no route
            together; unresolved when an output's resolution is; and
            otherwise feasible, exact exactly when every block's route is.
        """
        memo = self._memo(call)
        if ("partition", exact_only) not in memo:
            memo["partition", exact_only] = self._split(call, exact_only)
        return memo["partition", exact_only]

    def _split(self, call: BoundCall, exact_only: bool) -> _Partition:
        """The partition :meth:`_partition` returns, computed from the outputs' reports."""
        d = call.operands["d"]
        if (
            not isinstance(d, Distribution)
            or not _has_record_outputs(d.event_spec)
            or len(d.event_spec.outputs) < 2
        ):
            reason = "the event is not a record of several outputs"
            return _Partition((), Feasibility(False, reason))
        outputs = tuple(d.event_spec.outputs)
        reports = self._classified(call)
        unresolved = [report for report in reports.values() if report.feasible is None]
        if unresolved:
            pending = tuple(dict.fromkeys(item for report in unresolved for item in report.pending))
            return _Partition((), Feasibility(None, pending=pending))
        exact = tuple(c for c in outputs if reports[c].feasible is True)
        if not exact:
            return _Partition((), Feasibility(False, f"no output has an exact {self.summary}"))
        blocks = {c: _Block((c,), self._view_at(call, (c,)), reports[c]) for c in exact}
        rest = tuple(c for c in outputs if c not in blocks)
        if rest:
            view = self._view_at(call, rest)
            remaining = self._resolve(call, view, exact_only)
            if remaining.feasible is None:
                return _Partition((), Feasibility(None, pending=remaining.pending))
            if remaining.feasible is False:
                return _Partition((), self._declined(call, exact, rest, exact_only))
            blocks[rest[0]] = _Block(rest, view, remaining)
        ordered = tuple(blocks[c] for c in outputs if c in blocks)
        exactness = all(block.report.exact is True for block in ordered)
        return _Partition(ordered, PointReport(True, exact=exactness))

    def _declined(
        self, call: BoundCall, exact: tuple[str, ...], rest: tuple[str, ...], exact_only: bool
    ) -> Feasibility:
        """The report when the outputs in *rest* have no route together.

        When the call itself sets ``exact_only``, the report says how to
        compute the exact outputs, and the call's error leads with it.
        """
        if not exact_only:
            return Feasibility(
                False, f"the outputs {format_names(rest)} have no {self.summary} route"
            )
        reason = (
            f"the outputs {format_names(rest)} have no exact {self.summary}, but "
            f"{format_names(exact)} do"
        )
        if not call.controls["exact_only"]:
            return Feasibility(False, reason)
        # The suggested call passes the call's other arguments by their names,
        # as ``quantile(d['mu'], q)``.
        arguments = ", ".join(
            [f"d[{exact[0]!r}]", *(name for name in call.operands if name != "d")]
        )
        return Feasibility(
            False,
            f"{reason}; call {self.summary} on the view of each output that does, as "
            f"{self.summary}({arguments})",
            actionable=True,
        )

    def _run_block(self, call: BoundCall, block: _Block, exact_only: bool) -> tuple[Any, dict]:
        """The raw summary of *block*, with the metadata of how it was computed.

        The block is resolved and run as the operation runs a call on its view,
        inside this call, so its draws count toward this call's. The metadata
        names the block's outputs, the route that computed it, that route's
        exactness, and, when the route drew Monte Carlo samples, their number
        as ``num_samples``.
        """
        block_value, metadata = _inner_run(call, self._values(call, block.view), exact_only)
        return block_value, {"outputs": list(block.outputs), **metadata}

    def probe(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Feasibility:
        """The report of the split, restricted to exact routes under *exact_only*."""
        return self._partition(call, exact_only).report

    def run(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Any:
        """The summary of each block by its own route, assembled in the event's order.

        Parameters
        ----------
        call : BoundCall
            The call, whose law is ``call.operands["d"]``.
        result : OutputSpec or None
            The output spec of the call's result.
        method : str or None
            Unused; a ``method`` control names this route itself.
        exact_only : bool
            Whether each block is restricted to exact routes.

        Returns
        -------
        Any
            The summary keyed by the law's outputs, as a term whose
            provenance records each block's outputs, route, exactness, and
            Monte Carlo draws; under ``raw``, its raw form.

        Raises
        ------
        ResolutionError
            If the split does not realize the call, which a probe reports first.
        MathematicalDomainError
            If a block's route finds its summary undefined, as for a Cauchy
            output's mean.
        """
        d = call.operands["d"]
        partition = self._partition(call, exact_only)
        if partition.report.feasible is not True:
            raise ResolutionError(
                f"{call.operation.label}: route {self.name!r} does not apply: "
                f"{partition.report.description}"
            )
        output_values: dict[str, Any] = {}
        blocks_metadata = []
        for block in partition.blocks:
            block_value, block_metadata = self._run_block(call, block, exact_only)
            output_values.update(_block_values(block.view, block_value))
            blocks_metadata.append(block_metadata)
        assembled = {c: output_values[c] for c in d.event_spec.outputs}
        # A quantile at several levels is a batch of records, which
        # ``_record_batch`` assembles; any other value passes through it.
        summary_value = _record_batch(assembled, call, result)
        return _assembled(
            summary_value,
            call,
            result,
            parents=[d],
            metadata={"route": self.name, "blocks": blocks_metadata},
        )


# ---------------------------------------------------------------------------
# The operations
# ---------------------------------------------------------------------------


@operation(result=_mean_result)
def mean(d: Distribution):
    """The mean ``𝔼[X]`` for ``X ~ d``, a value shaped like one draw.

    A record-drawing law's mean is a ``Record`` with the law's schema, a random
    function's is its mean function, and a random measure's is the
    marginalized law. A batch-valued law's mean is elementwise: a batch on the
    event's levels whose element at each position is the mean of the draw's
    element there.

    The routes, in selection order, are the law's closed form,
    ``closed_form``; ``by_output``, which computes each output of a
    record event by the route its own view selects; and the Monte
    Carlo fallback, ``monte_carlo``. A dependent joint claims no mean, so when
    one of its outputs, such as a root factor, has an exact mean, its mean
    takes ``by_output``: that output's mean is exact, and the result as
    a whole is approximate.

    Parameters
    ----------
    d : Distribution
        The law, whose ``_mean`` gives the closed form, whose outputs' views
        give their own means otherwise, and whose draws the Monte Carlo
        fallback averages.

    Returns
    -------
    TrackedTerm
        The mean at the law's declared event kind, labeled ``𝔼[(mu, tau) ~ d]``
        for a law ``d`` over ``mu`` and ``tau``.

    Raises
    ------
    ResolutionError
        If *d* has no closed-form mean and does not sample, or, under
        ``exact_only``, if an output of a record event has no exact mean.
    MathematicalDomainError
        If the mean of *d*, or of an output whose own route is exact, is
        known to be undefined, as a Cauchy law's is.
    """


mean.capability_route(
    "closed_form",
    operand="d",
    protocol=SupportsMean,
    method="_mean",
    exact=True,
    execute=_capability("_mean"),
)
mean.register_route(_ByOutput("mean"))
mean.fallback_route("monte_carlo", check=_can_average, execute=_mc_mean, exact=False)


@operation(result=_variance_result, applicability_checks=(_event_typed_variance,))
def variance(d: Distribution):
    """The variance of ``X ~ d``, a value shaped like one draw.

    A batch-valued law's variance is elementwise, as its mean is. The routes
    are those of ``mean``: the closed form, ``closed_form``;
    ``by_output``, which computes each output of a record event by the
    route its own view selects; and the Monte Carlo fallback,
    ``monte_carlo``.

    Parameters
    ----------
    d : Distribution
        The law, whose ``_variance`` gives the closed form, whose outputs'
        views give their own variances otherwise, and whose draws the Monte
        Carlo fallback uses.

    Returns
    -------
    TrackedTerm
        The variance at the law's declared event kind, with non-negative
        leaves, labeled ``𝕍[(mu, tau) ~ d]``.

    Raises
    ------
    ApplicabilityError
        If the event is measure-valued, which has no event-typed variance in
        general.
    ResolutionError
        If *d* has no closed-form variance and does not sample, or, under
        ``exact_only``, if an output of a record event has no exact variance.
    MathematicalDomainError
        If the variance of *d*, or of an output whose own route is exact, is
        known to be undefined, as a Cauchy law's is.
    """


variance.capability_route(
    "closed_form",
    operand="d",
    protocol=SupportsVariance,
    method="_variance",
    exact=True,
    execute=_capability("_variance"),
)
variance.register_route(_ByOutput("variance"))
variance.fallback_route(
    "monte_carlo", check=_can_average_squares, execute=_mc_variance, exact=False
)


def _standard_deviation_result(d: DistributionSpec) -> OutputSpec:
    """A value of the event's kind with non-negative floating leaves, under the event's outputs."""
    return _summary_declaration(d.event_spec, _variance_term(d.event_spec.term_spec))


def _root(value: Any) -> Any:
    """The square root of each coordinate of *value*: of each array leaf, or of a function's values.

    A mapping's square root is the mapping of its fields' square roots, under
    the same keys. A function's square root takes the function's parameters,
    where its signature can be read.
    """
    if isinstance(value, Mapping):
        return {name: _root(field) for name, field in value.items()}
    if callable(value):

        def root(*args: Any, **kwargs: Any) -> Any:
            return _root(value(*args, **kwargs))

        # A callable whose signature cannot be read keeps the generic one.
        with suppress(TypeError, ValueError):
            root.__signature__ = inspect.signature(value)  # type: ignore[attr-defined]
        return root
    return jax.tree.map(jnp.sqrt, value)


def _variance_applies(d: Any) -> Any:
    """``variance`` has a route for the law."""
    return variance.check(d)


@operation(
    result=_standard_deviation_result,
    applicability_checks=(_event_typed_variance,),
    derived_route_check=_variance_applies,
)
def standard_deviation(d: Distribution):
    """The standard deviation ``sqrt(variance(d))`` of ``X ~ d``, a value shaped like one draw.

    Each coordinate's standard deviation is the square root of its variance,
    so a record-drawing law's is a ``Record`` with a field for each output,
    and a random function's is its pointwise standard deviation function.

    The call takes the routes of ``variance``: its closed form, its
    ``by_output`` route, and its Monte Carlo fallback. ``exact_only``,
    ``max_evaluations``, and ``num_samples`` set on this operation pass to
    ``variance``, but ``method`` does not, so a route of ``variance`` is chosen
    on ``variance.using(method=...)``. When ``variance`` draws Monte Carlo
    samples, the call's provenance records their number as ``num_samples``.

    Parameters
    ----------
    d : Distribution
        The law, whose variance the call takes.

    Returns
    -------
    TrackedTerm
        The standard deviation at the law's declared event kind, with
        non-negative leaves under the event's outputs, labeled
        ``√𝕍[(mu, tau) ~ d]``.

    Raises
    ------
    ApplicabilityError
        If the event is measure-valued, which has no event-typed variance in
        general.
    ResolutionError
        If ``variance`` has no route for *d*, or, under ``exact_only``, no exact
        route.
    MathematicalDomainError
        If the variance of *d* is known to be undefined, as a Cauchy law's is.
    """
    return _root(variance.using(raw=True)(d))


@operation(
    result=_cov_result,
    applicability_checks=(_numeric_events,),
    roles={"d2": (DistributionSpec,)},
)
def cov(d1: Distribution, d2: Distribution | None = None):
    """The covariance of the flattened draw of ``X1 ~ d1``, or the cross-covariance of ``X1`` and ``X2 ~ d2``.

    The flattened draw ravels each array leaf in row-major order and
    concatenates the leaves in the record's canonical order. A batch-valued
    draw is raveled in its raw form, so each leaf runs over the event's batch
    axes and then the element's own axes.

    ``cov(d1, d2)`` is ``Cov(X1, X2) = E[(X1 - E X1)(X2 - E X2)ᵀ]``, with the two
    laws drawn as a lift draws them: two laws that read one law's draws, such
    as two coupled marginals ``d["y"]`` and ``d["x"]``, are drawn together, and
    two unrelated laws are drawn independently, so their block is zero. It is
    the block of rows ``X1`` and columns ``X2`` of the covariance of the two
    draws taken together, which is the off-diagonal block of the matrix that
    ``numpy.cov`` returns for the stacked draws. An omitted *d2*,
    or ``d2`` that is *d1* itself, gives the covariance of *d1*.

    Two laws whose draws are coordinates of one law's draws, as coupled
    marginals, copies, and renames of one law are, take ``coupled_marginals``,
    which takes the covariance of that law's coupled marginal at those
    coordinates, exact when its covariance is. Two unrelated laws take
    ``independent``, an exact zero. Other laws drawn together, such as a
    transform of a coupled marginal and its parent, take ``joint_lift``,
    which takes the covariance of the law of the pair of their draws, exact
    when the lift enumerates an empirical law's atoms.

    Parameters
    ----------
    d1 : Distribution
        A law with a numeric event, whose draws give the rows.
    d2 : Distribution, optional
        A second law with a numeric event, whose draws give the columns.

    Returns
    -------
    NumericArray
        The dense ``(size of d1, size of d2)`` covariance over the events'
        coordinates in canonical order, labeled ``ℂ[(mu, tau) ~ d]`` for one
        law and ``ℂ[y ~ d, x ~ d]`` for two.

    Raises
    ------
    ApplicabilityError
        If an event is not numeric.
    ResolutionError
        If *d1* has no closed-form covariance and does not sample, if two laws
        have no route, or if *d1* and *d2* are drawn independently but share a
        law, as ``(a & b)["a"]`` and ``a`` do.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from probpipe import MultivariateNormal
    >>> d = MultivariateNormal("v", loc=jnp.zeros(2), cov=jnp.array([[2.0, 0.5], [0.5, 1.0]]))
    >>> cov(d).raw.tolist()
    [[2.0, 0.5], [0.5, 1.0]]
    """


def _two_laws(call: BoundCall) -> tuple[Distribution, Distribution] | None:
    """The call's two laws, or None when ``d2`` is omitted or is ``d1``."""
    d1, d2 = call.operands["d1"], call.operands["d2"]
    return None if d2 is None or d2 is d1 else (d1, d2)


#: The reason a route of the one-law form gives for a call on two laws.
_TWO_LAWS = "the call takes two laws, d1 and d2"

#: The reason a route of the two-law form gives for a call on one law.
_ONE_LAW = "the call takes one law"


def _closed_form_cov_check(call: BoundCall, result: OutputSpec | None) -> Feasibility:
    """``d2`` is omitted or is ``d1``, and the covariance guard of ``d1`` admits the call."""
    if _two_laws(call) is not None:
        return Feasibility(False, _TWO_LAWS)
    return _capability_guard(call.operands["d1"], "_cov")


def _closed_form_cov(call: BoundCall, result: OutputSpec | None) -> Any:
    """``d1._cov()`` as a dense array."""
    return _dense(call.operands["d1"]._cov())


def _one_law_samples(call: BoundCall, result: OutputSpec | None) -> Feasibility:
    """``d2`` is omitted or is ``d1``, and ``d1`` samples."""
    if _two_laws(call) is not None:
        return Feasibility(False, _TWO_LAWS)
    return _law_samples(call.operands["d1"])


# ---------------------------------------------------------------------------
# The cross-covariance of two laws
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CoordinatesIn:
    """Where a law's flat draw lies in the flat draw of the law whose draws it reads.

    Attributes
    ----------
    root : Distribution
        The law whose draws the law reads, which reads no other law's.
    indices : tuple of int
        The coordinate of a flat draw of *root* at each coordinate of a flat
        draw of the law, in the law's order.
    """

    root: Distribution
    indices: tuple[int, ...]


def _coordinates_in_root(law: Distribution) -> _CoordinatesIn | None:
    """Where *law*'s flat draw lies in the flat draw of the law it reads, or None.

    A coupled marginal selects coordinates of its parent's draw, a copy or a
    rename reorders those of the law it is made from, and an element of a
    batch of laws reads its stored law's. Any other law reads its own draw, so
    it is the root. A law whose coordinates cannot be read, as a coupled
    marginal of a law with a non-numeric field cannot, or whose root has a
    free dimension, gives None. For a law ``d`` over ``x`` of shape ``(2,)``
    and ``y`` of shape ``(3,)``, the coordinates of ``d["y"]`` have the root
    ``d`` and the indices ``(2, 3, 4)``.

    Parameters
    ----------
    law : Distribution
        The law whose coordinates are read.

    Returns
    -------
    _CoordinatesIn or None
        The root and the coordinates of *law* in its flat draw, or None.
    """
    try:
        return _read_coordinates(law)
    except (TypeError, ValueError):
        return None


def _read_coordinates(law: Distribution) -> _CoordinatesIn | None:
    """The coordinates :func:`_coordinates_in_root` returns, raising where a layout is unbound."""
    source = _element_source(law)
    if source is not None:
        return _coordinates_in_root(source)
    if isinstance(law, CoupledMarginal):
        parent = _coordinates_in_root(law.parent)
        if parent is None:
            return None
        return _CoordinatesIn(parent.root, tuple(parent.indices[i] for i in law._coordinates()))
    copied = _copy_source(law)
    if copied is not None:
        made_from, renames = copied
        parent = _coordinates_in_root(made_from)
        if parent is None or renames.keeps_paths:
            return parent
        if isinstance(made_from.event_spec.term_spec, BatchSpec):
            # A law over a batch has no paths below its output, so a rename
            # moves only that output's name and keeps every coordinate.
            return parent
        order = renames._coordinate_order
        if order is None:
            return parent
        return _CoordinatesIn(parent.root, tuple(parent.indices[i] for i in order))
    size = _flat_size(law.event_spec.term_spec)
    if size is None:
        return None
    return _CoordinatesIn(law, tuple(range(size)))


def _joint_selection(root: Distribution, indices: set[int]) -> Distribution:
    """The joint law of the coordinates *indices* of *root*'s flat draw.

    It is *root* itself when the leaves that hold the coordinates are all of
    *root*'s, and otherwise the coupled marginal of *root* at the paths of those
    leaves, in *root*'s order.
    """
    declaration = root.event_spec
    if not _has_record_outputs(declaration):
        return root
    leaves = _leaf_paths(declaration)
    held = [leaf for leaf in leaves if indices.intersection(_coordinate_range(declaration, leaf))]
    if len(held) == len(leaves):
        return root
    return root[held[0]] if len(held) == 1 else root[tuple(held)]


@dataclass(frozen=True)
class _JointBlock:
    """How ``coupled_marginals`` computes a call: the joint law, and its rows and columns.

    Attributes
    ----------
    joint : Distribution or None
        The law of the coordinates of both operands: their root, or its
        coupled marginal at the leaves that hold them. It is None when either
        operand has no coordinates, so the block has no entries.
    rows : tuple of int
        The coordinate of the joint law's flat draw at each coordinate of
        ``d1``'s.
    columns : tuple of int
        The coordinate of the joint law's flat draw at each coordinate of
        ``d2``'s.
    """

    joint: Distribution | None
    rows: tuple[int, ...]
    columns: tuple[int, ...]


class _CoupledMarginals(_DelegatingRoute):
    """The cross-covariance of two laws that read coordinates of one law, from that law's covariance.

    When ``d1`` and ``d2`` read coordinates of one root law ``R``, as two
    coupled marginals of ``R`` do, their joint law is ``R`` at the leaves that
    hold those coordinates. The route resolves and runs ``cov`` on that law
    inside the call, as ``by_output`` runs a block, and returns the block at
    ``d1``'s rows and ``d2``'s columns. Its exactness is that of the route it
    runs: exact for a Gaussian or an empirical ``R``, and a Monte Carlo
    estimate from draws of ``R`` for a law that only samples.
    """

    source = RouteSource.STRUCTURAL

    def __init__(self) -> None:
        super().__init__("coupled_marginals")

    @property
    def condition(self) -> str:
        """The route's feasibility condition in words."""
        return "d1 and d2 read coordinates of one law, whose covariance has a route"

    def _block(self, call: BoundCall) -> _JointBlock | Feasibility:
        """The joint law and the block's coordinates, or the reason the route does not apply."""
        memo = call._memo.setdefault(self.name, {})
        if "block" not in memo:
            memo["block"] = self._joint_block(call)
        return memo["block"]

    @staticmethod
    def _joint_block(call: BoundCall) -> _JointBlock | Feasibility:
        """The :class:`_JointBlock` of the call, computed once by :meth:`_block`."""
        laws = _two_laws(call)
        if laws is None:
            return Feasibility(False, _ONE_LAW)
        first, second = (_coordinates_in_root(law) for law in laws)
        if first is None or second is None or first.root is not second.root:
            return Feasibility(False, "d1 and d2 are not parts of one law")
        if not first.indices or not second.indices:
            return _JointBlock(None, first.indices, second.indices)
        joint = _joint_selection(first.root, {*first.indices, *second.indices})
        if isinstance(joint, CoupledMarginal):
            in_joint = {index: position for position, index in enumerate(joint._coordinates())}
        else:
            in_joint = {index: index for index in {*first.indices, *second.indices}}
        return _JointBlock(
            joint,
            tuple(in_joint[index] for index in first.indices),
            tuple(in_joint[index] for index in second.indices),
        )

    def probe(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Feasibility:
        """The report of ``cov`` on the joint law, restricted to exact routes under *exact_only*."""
        block = self._block(call)
        if isinstance(block, Feasibility):
            return block
        if block.joint is None:
            return PointReport(True, exact=True)
        report = _inner_report(call, {"d1": block.joint, "d2": None}, exact_only)
        if report.feasible is not True:
            return report
        return PointReport(True, exact=report.exact is True)

    def run(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Any:
        """The block of the joint law's covariance at ``d1``'s rows and ``d2``'s columns.

        Parameters
        ----------
        call : BoundCall
            The call, whose laws are ``call.operands["d1"]`` and ``call.operands["d2"]``.
        result : OutputSpec or None
            The output spec of the call's result.
        method : str or None
            Unused; a ``method`` control names this route itself.
        exact_only : bool
            Whether the covariance of the joint law is restricted to exact routes.

        Returns
        -------
        Any
            The block, as a term whose provenance records the joint law's
            route, exactness, and Monte Carlo draws; under ``raw``, the array.

        Raises
        ------
        ResolutionError
            If the laws do not read coordinates of one law, which a probe
            reports first.
        """
        block = self._block(call)
        if isinstance(block, Feasibility):
            raise ResolutionError(
                f"{call.operation.label}: route {self.name!r} does not apply: {block.description}"
            )
        if block.joint is None:
            value = _zero_block(call)
            metadata: dict[str, Any] = {"exact": True}
        else:
            joint_cov, metadata = _inner_run(call, {"d1": block.joint, "d2": None}, exact_only)
            rows = jnp.asarray(block.rows, dtype=jnp.int32)
            columns = jnp.asarray(block.columns, dtype=jnp.int32)
            value = jnp.asarray(_dense(joint_cov))[rows][:, columns]
        return _assembled(
            value,
            call,
            result,
            parents=[call.operands["d1"], call.operands["d2"]],
            metadata={"route": self.name, "exact": metadata["exact"], "joint": metadata},
        )


def _shared_part(d1: Distribution, d2: Distribution) -> Distribution | None:
    """A law that the root ancestors of *d1* and *d2* share, or None.

    The root ancestors share a law when a factor of one, at any depth, has
    the root ancestor of a factor of the other, as a merge reads coupling.
    The law returned is the smaller of the two coupled factors: the one that
    is not a whole root ancestor, when one of them is.
    """
    roots = (_distribution_root(d1), _distribution_root(d2))
    pairs = _coupled_part_pairs(*roots)
    if not pairs:
        return None
    first_part, second_part, _ = pairs[0]
    return second_part if first_part is roots[0] else first_part


def _independent_check(call: BoundCall, result: OutputSpec | None) -> Feasibility:
    """d1 and d2 are not coupled, their root ancestors share no law, and both sizes are bound.

    Laws whose root ancestors share a law, as ``(a & b)["a"]`` and ``a`` do,
    are refused with a reason the caller can act on, since a lift draws them
    independently although they name one law.
    """
    laws = _two_laws(call)
    if laws is None:
        return Feasibility(False, _ONE_LAW)
    d1, d2 = laws
    if _distribution_root(d1) is _distribution_root(d2):
        return Feasibility(False, "d1 and d2 are drawn together")
    shared = _shared_part(d1, d2)
    if shared is not None:
        return Feasibility(
            False,
            f"d1 and d2 both depend on the law over "
            f"{', '.join(repr(name) for name in shared.event_spec.outputs)}, but they are "
            f"drawn independently. Index both from one joint law that holds them",
            actionable=True,
        )
    for name, law in (("d1", d1), ("d2", d2)):
        if _flat_size(law.event_spec.term_spec) is None:
            return Feasibility(False, f"the size of a draw of {name} has a free dimension")
    return True


def _zero_block(call: BoundCall) -> Any:
    """The ``(size of d1, size of d2)`` block of zeros, in the floating dtype the events share."""
    d1, d2 = call.operands["d1"], call.operands["d2"]
    events = (d1.event_spec.term_spec, d2.event_spec.term_spec)
    rows, columns = (cast(int, _flat_size(event)) for event in events)
    dtype = _cov_dtype(events)
    return jnp.zeros((rows, columns), dtype=jnp.result_type(float) if dtype is None else dtype)


def _require_finite_mean(law: Distribution, name: str) -> None:
    """Raise when *law* gives its mean in closed form and that mean is undefined or infinite.

    The zero block of two independent laws holds when both means are finite.
    A law whose mean is known only through its draws is assumed to have a
    finite one, as the Monte Carlo routes assume their moments exist.

    Parameters
    ----------
    law : Distribution
        The law at the operand *name*.
    name : str
        The operand, ``"d1"`` or ``"d2"``, which the error names.

    Raises
    ------
    MathematicalDomainError
        If the law's closed-form mean does not exist, as a Cauchy law's does
        not, or is infinite.
    """
    if not isinstance(law, SupportsMean) or _capability_guard(law, "_mean").feasible is not True:
        return
    try:
        law_mean = law._mean()
    except MathematicalDomainError as error:
        raise MathematicalDomainError(f"cannot compute cov(d1, d2): {error}") from error
    leaves = jax.tree.leaves(_raw_record(law_mean))
    if not all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in leaves):
        raise MathematicalDomainError(
            f"cannot compute cov(d1, d2): the mean of {name}, {law.label!r}, is infinite"
        )


def _independent_cov(call: BoundCall, result: OutputSpec | None) -> Any:
    """The zero block of two independent laws, whose closed-form means must be finite."""
    for name in ("d1", "d2"):
        _require_finite_mean(call.operands[name], name)
    return _zero_block(call)


def _pair_of_draws(d1, d2) -> dict[str, Any]:
    """The record of the two draws, under the names ``d1`` and ``d2``.

    The parameters are unannotated, so a law passed to either is lifted and
    the map receives its draws.
    """
    return {"d1": d1, "d2": d2}


#: The map whose lift over two laws draws them as a lift draws them, which ``joint_lift`` calls.
_PAIR_OF_DRAWS = function(_pair_of_draws, traceable=True)


#: The controls a route that lifts a map forwards to it, besides the
#: ``max_evaluations`` that the call's ``num_samples`` gives.
_LIFT_FORWARDED = tuple(name for name in _FORWARDED_CONTROLS if name != "max_evaluations")


def _lift_report(rule: Feasibility, exact_only: bool, deferred: str) -> Feasibility:
    """The report of a route that lifts a map by *rule* and then runs a summary on the law, deferred to the run.

    Without *exact_only*, an exact rule leaves the route's exactness open,
    since the summary's route is chosen only once the law exists.
    """
    if rule.feasible is not True:
        return rule
    exact = rule.exact if exact_only or rule.exact is False else None
    return PointReport(True, method=rule.method_name, exact=exact, deferred=(deferred,))


def _steps_exactness(*steps: Any) -> bool | None:
    """The exactness of a result computed in *steps*: exact when each step's provenance records it."""
    recorded = tuple(_recorded_exactness(step) for step in steps)
    if False in recorded:
        return False
    return True if all(step is True for step in recorded) else None


def _noting_lifted_draws(law: Any, call: BoundCall) -> None:
    """Record the call's ``num_samples`` when the sampling lift drew *law*."""
    provenance = getattr(law, "provenance", None)
    if provenance is not None and provenance.metadata.get("route") == "sampling_lift":
        _note_num_samples_used(call.controls["num_samples"])


class _JointLift(_DelegatingRoute):
    """The cross-covariance of two coupled laws, from the law of the pair of their draws.

    The route calls a map that returns the record of its two draws, under the
    names ``d1`` and ``d2``, on the two laws. The lift draws coupled laws
    together, so the law it returns is their joint law, whose flat draw holds
    ``d1``'s coordinates and then ``d2``'s. The route returns the block of
    that law's covariance at those rows and columns. Its exactness is that of
    the evaluation rule and of the covariance's route, so the empirical
    enumeration of an empirical root gives an exact result. Each draw of the
    joint law is one evaluation of the map, so the call's ``num_samples`` is
    the map's ``max_evaluations``.
    """

    source = RouteSource.REGISTRY

    def __init__(self) -> None:
        super().__init__("joint_lift")

    @property
    def condition(self) -> str:
        """The route's feasibility condition in words."""
        return "d1 and d2 are drawn together, and an evaluation rule draws the pair of their draws"

    @staticmethod
    def _map(call: BoundCall, exact_only: bool) -> Function:
        """The pairing map under the controls the caller set, with ``num_samples`` as its ``max_evaluations``."""
        set_controls = call.operation._options
        forwarded = {name: set_controls[name] for name in _LIFT_FORWARDED if name in set_controls}
        forwarded.update(
            exact_only=exact_only,
            include_inputs=False,
            max_evaluations=call.controls["num_samples"],
        )
        return _PAIR_OF_DRAWS.using(**forwarded)

    def probe(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Feasibility:
        """The evaluation rule's report, with the covariance of the joint law deferred to the run."""
        laws = _two_laws(call)
        if laws is None:
            return Feasibility(False, _ONE_LAW)
        d1, d2 = laws
        if _distribution_root(d1) is not _distribution_root(d2):
            return Feasibility(False, "d1 and d2 are not drawn together")
        rule = self._map(call, exact_only).check(d1, d2).selected
        if rule is None:
            return Feasibility(False, "no evaluation rule draws d1 and d2 together")
        return _lift_report(
            rule, exact_only, "the covariance of the joint law is checked when it runs"
        )

    def run(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Any:
        """The block of the joint law's covariance, with the exactness of both steps.

        Parameters
        ----------
        call : BoundCall
            The call, whose laws are ``call.operands["d1"]`` and ``call.operands["d2"]``.
        result : OutputSpec or None
            The output spec of the call's result.
        method : str or None
            Unused; a ``method`` control names this route itself.
        exact_only : bool
            Whether the evaluation rule and the covariance are restricted to
            exact routes.

        Returns
        -------
        RouteOutcome
            The block, with its exactness: exact when both the rule that drew
            the joint law and the route of its covariance are.
        """
        d1, d2 = call.operands["d1"], call.operands["d2"]
        joint = self._map(call, exact_only)(d1, d2)
        _noting_lifted_draws(joint, call)
        joint_cov = cov.using(exact_only=exact_only, num_samples=call.controls["num_samples"])(
            joint
        )
        rows = cast(int, _flat_size(joint.event_spec.term_spec.children["d1"]))
        dense = jnp.asarray(joint_cov.raw)
        return RouteOutcome(dense[:rows, rows:], _steps_exactness(joint, joint_cov))


cov.capability_route(
    "closed_form",
    operand="d1",
    protocol=SupportsCovariance,
    method="_cov",
    exact=True,
    check=_closed_form_cov_check,
    execute=_closed_form_cov,
)
cov.register_route(_CoupledMarginals())
cov.structural_route("independent", check=_independent_check, execute=_independent_cov, exact=True)
cov.register_route(_JointLift())
cov.fallback_route("monte_carlo", check=_one_law_samples, execute=_mc_cov, exact=False)


@operation(result=_quantile_result, applicability_checks=(_numeric_event,))
def quantile(d: Distribution, q: Any):
    """The per-coordinate quantiles of ``X ~ d`` at the levels *q*.

    A batch-valued law's quantiles are elementwise, as its mean is. The routes
    are the law's closed form, ``closed_form``; ``by_output``,
    which computes each output of a record event by the route its own
    view selects; and the Monte Carlo fallback, ``monte_carlo``,
    which ``using(method=...)`` selects among. The fallback's quantile
    at a level ``q`` is the generalized
    inverse ``inf{x : F(x) >= q}`` of each coordinate's CDF over the draws, as
    an empirical law's is.

    Parameters
    ----------
    d : Distribution
        A law with a numeric event.
    q : float or array of float
        One level in ``[0, 1]``, or an array of them.

    Returns
    -------
    TrackedTerm
        For one level, a value of the event's kind; for several, the batch of
        those values on a level named ``quantile``, before any levels of a
        batch-valued event. It is labeled ``ℚ[mu ~ d]``.

    Raises
    ------
    ApplicabilityError
        If the event is not numeric, or *q* is not numeric.
    MathematicalDomainError
        If a concrete level lies outside ``[0, 1]``.
    ResolutionError
        If *d* has no closed-form quantiles and does not sample, or, under
        ``exact_only``, if an output of a record event has no exact
        quantiles.
    """


def _closed_form_quantile(call: BoundCall, result: OutputSpec | None) -> Any:
    """``d._quantile(q)`` at levels in ``[0, 1]``.

    A record event's quantiles at several levels are the declared batch of
    records.
    """
    quantiles = call.operands["d"]._quantile(_check_levels(call.operands["q"]))
    return _record_batch(quantiles, call, result)


quantile.capability_route(
    "closed_form",
    operand="d",
    protocol=SupportsQuantile,
    method="_quantile",
    exact=True,
    execute=_closed_form_quantile,
)
quantile.register_route(_ByOutput("quantile"))
quantile.fallback_route("monte_carlo", check=_can_sample, execute=_mc_quantile, exact=False)


class _PushforwardMean(_EvaluationRules):
    """The derived route of ``expectation``: a rule of ``pushforward`` builds the law ``f♯d``, and ``mean`` averages it.

    The route runs ``pushforward``'s route and then ``mean`` on the law it
    returns, so its method names are the evaluation rules'. The engine runs both
    steps in the scope of a derived route (VI.0). The route ranks once, as a
    derived route: among the approximate routes, and under ``exact_only`` among the
    exact ones, with the rule and the mean both restricted to exact routes. The
    mean's route is chosen only once the law exists, so without ``exact_only``
    the probe reports the rule's exactness when it is approximate and leaves it
    open otherwise, which the call's report states as approximate, and it
    defers the mean's check; the run records the exactness of both steps.

    Each draw of the law is one evaluation of the integrand, so the call's
    resolved ``num_samples`` is the map's ``max_evaluations``, whatever the map
    sets, and the call's own ``max_evaluations`` is not passed to the map. When
    the sampling lift builds the law, the call's provenance records its
    ``num_samples``.
    """

    _bind_given_first = False

    #: The controls the run forwards to the call of the map, besides the
    #: ``max_evaluations`` that the call's ``num_samples`` gives.
    _forwarded = _LIFT_FORWARDED

    def __init__(self) -> None:
        super().__init__("pushforward_mean")

    def _forwarded_controls(self, call: BoundCall) -> dict[str, Any]:
        """The controls the caller set, with the call's ``num_samples`` as the map's ``max_evaluations``."""
        return super()._forwarded_controls(call) | {"max_evaluations": call.controls["num_samples"]}

    @property
    def condition(self) -> str:
        """An evaluation rule builds the pushforward, and ``mean`` averages it."""
        return (
            "An evaluation rule builds the law of the integrand's value, as pushforward does, "
            "and a route of mean averages it."
        )

    def probe(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Feasibility:
        """The rule's report, with the mean of the law it builds deferred to the run."""
        rule = super().probe(call, result, method=method, exact_only=exact_only)
        return _lift_report(rule, exact_only, "the mean of the pushforward is checked when it runs")

    def run(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Any:
        """The mean of the pushforward law the selected rule returns, with the exactness of both steps."""
        law = super().run(call, result, method=method, exact_only=exact_only)
        _noting_lifted_draws(law, call)
        averaged = mean(law)
        return RouteOutcome(averaged.raw, _steps_exactness(law, averaged))


def _recorded_exactness(term: Any) -> bool | None:
    """The exactness *term*'s provenance records, ``None`` when it records none."""
    provenance = getattr(term, "provenance", None)
    return None if provenance is None else provenance.metadata.get("exact")


@operation(
    result=_expectation_result,
    roles={"f": (FunctionSpec,)},
    applicability_checks=_integrand_applicability_checks("expectation"),
    derived_route=_PushforwardMean(),
)
def expectation(d: Distribution, f: Any, fixed_args: Mapping[str, Any] | None = None):
    """The expectation ``𝔼[f(X)]`` for ``X ~ d``, defined as ``mean(pushforward(d, f))``.

    The integrand receives one draw of *d* at a time, whatever its parameter's
    annotation, on every route. A law claiming ``SupportsExpectation`` computes
    the expectation in closed form, and otherwise the route ``pushforward_mean``
    builds the pushforward by an evaluation rule, which
    ``using(method=...)`` names, and takes its mean. Under ``exact_only``
    both steps are restricted to exact routes. The pushforward evaluates the
    integrand at most ``num_samples`` times, the call's control, so
    ``expectation.using(num_samples=n)`` sets the number of draws of a Monte
    Carlo estimate, and the call's provenance records it.

    Parameters
    ----------
    d : Distribution
        The law to integrate against.
    f : callable or Function
        Maps one draw to a value; a ``Function``'s output spec names the
        result's kind. Exactly one of its parameters has no default and is
        absent from *fixed_args*, and each draw binds it.
    fixed_args : mapping of str to Any, optional
        The other parameters of a map with several, by name. Each is a
        constant: a law the direct call would sample, or a batch it would
        sweep, is refused, including arguments set through the map's
        construction ``bind``. ``mean(function(f)(x=d, a=A))`` integrates over the
        law ``A`` as well, and ``function(lambda a: expectation(d, f, {"a": a}))(A)``
        is the law of the expectation at a draw of ``A``.

    Returns
    -------
    TrackedTerm
        ``𝔼[f(X)]`` at the kind the integrand's output spec names,
        labeled ``𝔼[f(mu ~ d)]`` by the integrand's label: a ``Function``'s
        label, a callable's ``__name__``, and ``𝒻`` for a lambda.

    Raises
    ------
    ApplicabilityError
        If *f* is not a callable, *fixed_args* leaves other than one of its
        parameters open, the open parameter's declaration does not admit a draw
        of *d*, or a fixed argument is a law or a batch the call would lift.
    ResolutionError
        If neither the closed form nor an evaluation rule applies, or, under
        ``exact_only``, either step has no exact route.
    """
    return mean(pushforward(d, f, fixed_args))


def _integrand(call: BoundCall) -> Callable[[Any], Any]:
    """The integrand as a callable of one atom, presented as the evaluation rules present a draw.

    A record atom arrives as the nested mapping of its raw leaves and is passed
    as a ``Record`` whose fields are in the law's declared order, whatever the
    order of the mapping's keys. The fixed arguments are passed as the direct
    call presents them, with the map's construction ``bind``. An operation is
    called as itself, and any other map as its raw callable, so it traces. A
    map whose open parameter is its first and takes the atom alone is the raw
    callable itself.
    """
    f = _as_function(call.operands["f"], subject="expectation")
    fixed = dict(call.operands.get("fixed_args") or {})
    parameter = _bound_parameter(f, fixed, subject="expectation")
    operation = isinstance(f, Operation)
    body = f if operation else f.raw
    constants = {**({} if operation else f._bind), **fixed}
    if constants:
        constants = dict(presented_arguments(f._signature_info, constants))
    event = call.operands["d"].event_spec.term_spec
    record = isinstance(event, RecordSpec)
    first = next(iter(f.signature.parameters.values()), None)
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if (
        not (constants or record or operation)
        and first is not None
        and first.name == parameter
        and first.kind in positional
    ):
        return body

    def integrand(atom: Any) -> Any:
        if record and isinstance(atom, Mapping) and not isinstance(atom, TrackedTerm):
            # A law that maps the integrand with jax.vmap passes a dict whose
            # keys are sorted, so the record is laid out in the declared order.
            atom = _value_at_declared_kind(event, parameter, atom)
        return _call_at(body, f, parameter, atom, constants)

    return integrand


expectation.capability_route(
    "closed_form",
    operand="d",
    protocol=SupportsExpectation,
    method="_expectation",
    exact=True,
    execute=lambda call, _: call.operands["d"]._expectation(
        _integrand(call)
    ),  # closed form expectation
)


# ---------------------------------------------------------------------------
# The expressions of the moments
# ---------------------------------------------------------------------------


def _moment_expression(summary: str) -> Callable[[Any], Expression]:
    """The expression rule of a moment *summary*: the moment of a draw of the law, as ``𝔼[mu ~ d]``."""

    def rule(d: Any) -> Expression:
        return Summary(summary, draw_of(d))

    rule.__doc__ = f"The expression ``{_SUMMARY_SYMBOLS[summary]}[draw]`` of the moment of a draw of *d* (II.4)."
    return rule


def _cov_expression(d1: Any, d2: Any) -> Expression:
    """The expression ``ℂ[draw]`` of one law's covariance, or ``ℂ[draw, draw]`` of two laws'."""
    if d2 is None or d2 is d1:
        return Summary("Cov", draw_of(d1))
    return Summary("Cov", draw_of(d1), draw_of(d2))


_install_expression_rule(mean, _moment_expression("E"))
_install_expression_rule(variance, _moment_expression("Var"))
_install_expression_rule(standard_deviation, _moment_expression("SD"))
_install_expression_rule(cov, _cov_expression)
_install_expression_rule(quantile, _moment_expression("Q"))
_install_expression_rule(
    expectation,
    lambda d, f: Summary("E", Applied(_output_label(f, subject="expectation"), (draw_of(d),))),
)
