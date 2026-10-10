"""The distribution functionals: ``mean``, ``variance``, ``cov``, ``quantile``, and ``expectation``.

Each summarizes a distribution by a deterministic value. ``mean``, ``variance``,
``cov``, and ``quantile`` carry a capability route on their matching protocol,
``closed_form``, and a Monte Carlo fallback, ``monte_carlo``, on the event kinds
where the required averaging is defined. A capability returns the law's own
moment, and a numeric fallback returns that moment of the empirical law of its
draws. ``mean``, ``variance``, and ``quantile`` are computed coordinate by
coordinate, so they also carry the structural route ``by_component``: for a
law whose event is a record, it computes each component that has an exact
route of its own by that route, on the component's view, and the other
components together, so a dependent joint's root factors keep their closed
forms. ``cov`` couples components and has no such route.

A moment of the event's kind keeps the event's packaging and derives its
term specs, support included, and it names each component for the moment, so
the mean of a law over ``mu`` and ``tau`` holds ``mean(mu)`` and ``mean(tau)``.
A moment of ``d`` is labeled by the moment of a draw of ``d``, as
``E[(mu, tau) ~ d]``, ``Var[...]``, ``Cov[...]``, or ``Q[...]``, and the law of a
function lifted over laws contributes its notation in place of the draw, as
``E[f(beta ~ model; y)]``.

``expectation(d, f)`` returns ``E[f(X)]`` for ``X ~ d``, labeled
``E[f(mu ~ d)]``. It is the derived
operation ``mean(evaluate(f, d))``: a law claiming
:class:`~probpipe.distributions._capabilities.SupportsExpectation` computes it
in closed form, and otherwise the call takes the routes of ``evaluate``. Its
method names are therefore the evaluation rules', and an integration rule
registered with the evaluation-rule registry serves both operations.

The draws of every Monte Carlo route are workflow-owned random events, and
their number is the ``n_broadcast_samples`` control.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..core._batch import Batch, BatchSpec
from ..core._dispatch import (
    Feasibility,
    MathematicalDomainError,
    ResolutionError,
)
from ..core._expression import Applied, Expression, Summary, draw_of
from ..core._record_batch import RecordBatch, _batch_class_for
from ..core._record_spec import RecordSpec
from ..core._repr import format_names
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec
from ..core._specs import OutputSpec
from ..core.constraints import (
    _Boolean,
    _IntegerInterval,
    _NonNegativeInteger,
    _Sphere,
    interval,
    non_negative,
    unit_interval,
)
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..distributions._capabilities import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsVariance,
    _capability_guard,
)
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._factored import _raw_record
from ..functions._call import ApplicabilityError
from ..functions._resolution import PointReport, check_point, selected
from ..functions._result import SAMPLE_LEVEL
from ..values import Function, FunctionSpec
from ._evaluate import (
    _FORWARDED_CONTROLS,
    _as_function,
    _bound_parameter,
    _EvaluationRules,
    _lifts,
    evaluate,
)
from ._operation import (
    BoundCall,
    RouteSource,
    _assembled,
    _call_label,
    _DelegatingRoute,
    _install_expression_rule,
    _subject_name,
    _workflow_draws,
    operation,
)
from ._sample import _record_batch

__all__ = [
    "cov",
    "expectation",
    "mean",
    "quantile",
    "variance",
]


# ---------------------------------------------------------------------------
# The declarations of the moments
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


def _mean_term(spec: TermSpec) -> TermSpec:
    """The term spec of a mean: floating leaves on the hull of their support."""
    if isinstance(spec, NumericArraySpec):
        return NumericArraySpec(spec.shape, _floating(spec.dtype), _hull(spec.support))
    if isinstance(spec, RecordSpec):
        return RecordSpec({name: _mean_term(child) for name, child in spec.children.items()})
    return spec


def _variance_term(spec: TermSpec) -> TermSpec:
    """The term spec of a variance: non-negative floating leaves."""
    if isinstance(spec, NumericArraySpec):
        return NumericArraySpec(spec.shape, _floating(spec.dtype), non_negative)
    if isinstance(spec, RecordSpec):
        return RecordSpec({name: _variance_term(child) for name, child in spec.children.items()})
    return spec


def _quantile_term(spec: TermSpec) -> TermSpec:
    """The term spec of a quantile: the event's shapes, with floating dtypes kept."""
    if isinstance(spec, NumericArraySpec):
        dtype = (
            spec.dtype if spec.dtype is not None and _floating(spec.dtype) == spec.dtype else None
        )
        return NumericArraySpec(spec.shape, dtype)
    if isinstance(spec, RecordSpec):
        return RecordSpec({name: _quantile_term(child) for name, child in spec.children.items()})
    return spec


def _array_leaves(spec: TermSpec) -> list[NumericArraySpec]:
    """The array leaves of a numeric spec, in the canonical flat order."""
    if isinstance(spec, NumericArraySpec):
        return [spec]
    if isinstance(spec, RecordSpec):
        return [leaf for child in spec.children.values() for leaf in _array_leaves(child)]
    return []


def _summary_name(summary: str, components: Any) -> str:
    """The component a summary of *components* takes, the summary's call on them, as ``cov(mu, tau)``."""
    return f"{summary}({', '.join(components)})"


def _named_record(record: RecordSpec, summary: str) -> RecordSpec:
    """*record* with each immediate field named for *summary*, as ``mean(mu)``."""
    return record.with_path_names(
        {name: _summary_name(summary, [name]) for name in record.children}
    )


def _summary_declaration(declaration: OutputSpec, summary: str, term: TermSpec) -> OutputSpec:
    """The declaration of a summary of each component of *declaration*, whose type is *term*.

    Each component is named for the summary, as ``mean(theta)``, in the
    declaration's packaging. A summary whose type is a law, as the mean of a law
    over laws is, exposes that law's event, since a law takes its event's
    components.
    """
    if isinstance(term, DistributionSpec):
        return OutputSpec(term)
    if declaration.exposes_record:
        return OutputSpec(_named_record(cast(RecordSpec, term), summary))
    ((name, _),) = declaration.components.items()
    return OutputSpec(**{_summary_name(summary, [name]): term})


def _named_value(value: Any, summary: str, declaration: OutputSpec) -> Any:
    """*value*, a summary of each component of *declaration*, with its fields named for *summary*.

    A route returns the summary keyed by the law's components, as a record, a
    batch of records, or a mapping. An exposed record's components are the
    value's fields, which take the summary's names, and any other value is
    returned as it is.
    """
    if not declaration.exposes_record:
        return value
    if isinstance(value, (Record, RecordBatch)):
        names = value.children if isinstance(value, Record) else value.element_spec.children
        return value.with_path_names({name: _summary_name(summary, [name]) for name in names})
    if isinstance(value, Mapping):
        return {_summary_name(summary, [name]): field for name, field in value.items()}
    return value


def _summary_route(
    summary: str, execute: Callable[[BoundCall, OutputSpec | None], Any]
) -> Callable[[BoundCall, OutputSpec | None], Any]:
    """*execute*, whose value is keyed by the law's components, with its fields named for *summary*."""

    def named(call: BoundCall, result: OutputSpec | None) -> Any:
        return _named_value(execute(call, result), summary, call.operands["d"].event_spec)

    named.__doc__ = execute.__doc__
    return named


def _capability(method: str) -> Callable[[BoundCall, OutputSpec | None], Any]:
    """The law's capability *method*, called on the call's other arguments."""

    def call_capability(call: BoundCall, result: OutputSpec | None) -> Any:
        others = [value for name, value in call.operands.items() if name != "d"]
        return getattr(call.operands["d"], method)(*others)

    call_capability.__doc__ = f"``d.{method}()``."
    return call_capability


def _mean_result(d: DistributionSpec) -> OutputSpec:
    """A value of the event's kind, each component named ``mean(...)`` in the event's packaging.

    Its term specs are those of an average: floating, on the hull of the
    event's support, so the mean of a Bernoulli event lies in the unit interval.
    """
    return _summary_declaration(d.event_spec, "mean", _mean_term(d.event_spec.spec))


def _variance_result(d: DistributionSpec) -> OutputSpec:
    """A value of the event's kind with non-negative floating leaves, each component named ``variance(...)``."""
    return _summary_declaration(d.event_spec, "variance", _variance_term(d.event_spec.spec))


def _cov_result(d: DistributionSpec) -> OutputSpec | None:
    """The dense covariance of the flattened draw, a ``(size, size)`` array under ``cov(...)``.

    The component names every component of the event, as ``cov(mu, tau)``. The
    size is the number of the event's coordinates; an event with free
    dimensions leaves it to the returned value.
    """
    name = _summary_name("cov", d.event_spec.components)
    leaves = _array_leaves(d.event_spec.spec)
    if any(leaf.free_dims for leaf in leaves):
        return OutputSpec(**{name: None})
    size = sum(prod(leaf.shape) for leaf in leaves)
    dtypes = {_floating(leaf.dtype) for leaf in leaves}
    dtype = dtypes.pop() if len(dtypes) == 1 else None
    return OutputSpec(**{name: NumericArraySpec((size, size), dtype)})


def _quantile_result(d: DistributionSpec, q: TermSpec) -> OutputSpec:
    """One level returns the event's kind, and plural levels add a level named quantile.

    Parameters
    ----------
    d : DistributionSpec
        The law's spec, whose event declaration the quantile's declaration
        follows.
    q : TermSpec
        The spec of the levels: a scalar for one level, or an array whose shape
        gives the axes of the ``quantile`` level.

    Returns
    -------
    OutputSpec
        The event's declaration in its packaging, with the event's shapes and
        each component ``x`` named ``quantile(x)``. Plural levels give its batch
        form.

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
    element = _summary_declaration(d.event_spec, "quantile", _quantile_term(d.event_spec.spec))
    if not q.shape:
        return element
    return element._with_spec(BatchSpec(element.spec, quantile=q.shape))


def _expectation_result(f: TermSpec) -> OutputSpec | None:
    """The mean of the integrand's output declaration, as ``mean(evaluate(f, d))`` declares it.

    Each of the integrand's components is named ``mean(...)``, with an average's
    term specs. An integrand that declares no output leaves the declaration to
    the returned value.
    """
    if isinstance(f, FunctionSpec) and f.output_spec is not None and f.output_spec.spec is not None:
        return _summary_declaration(f.output_spec, "mean", _mean_term(f.output_spec.spec))
    return None


def _numeric_event(d: DistributionSpec) -> bool:
    """The law draws a numeric value: an array or a record of arrays."""
    return isinstance(d.event_spec.spec, NumericSpec)


def _event_typed_variance(d: DistributionSpec) -> bool:
    """The event has an event-typed variance, which a measure-valued event lacks in general."""
    return not isinstance(d.event_spec.spec, DistributionSpec)


# ---------------------------------------------------------------------------
# The Monte Carlo fallbacks
# ---------------------------------------------------------------------------


def _can_sample(call: BoundCall, result: OutputSpec | None) -> Feasibility:
    """The law samples, and the moment is assumed to exist.

    Sampling alone does not establish that a moment exists, so the estimate
    assumes the moment is finite.
    """
    d = call.operands["d"]
    if not isinstance(d, SupportsSampling):
        return Feasibility(False, f"{_subject_name(d)} does not sample")
    return _capability_guard(d, "_sample")


def _can_average(call: BoundCall, result: OutputSpec | None) -> Feasibility:
    """The law samples numeric draws, which average coordinatewise, or laws, which average to a mixture.

    The estimate assumes the mean is finite. The average of function-valued
    draws is not implemented.
    """
    event = call.operands["d"].event_spec.spec
    if isinstance(event, FunctionSpec):
        return Feasibility(False, "the average of function-valued draws is not implemented yet")
    if not isinstance(event, (NumericSpec, DistributionSpec)):
        return Feasibility(
            False, f"cannot average draws of a non-numeric event; got {type(event).__name__}"
        )
    return _can_sample(call, result)


def _can_average_squares(call: BoundCall, result: OutputSpec | None) -> Feasibility:
    """The law samples numeric draws, whose variance is taken coordinatewise.

    The estimate assumes the variance is finite. The pointwise variance of
    function-valued draws is not implemented.
    """
    event = call.operands["d"].event_spec.spec
    if isinstance(event, FunctionSpec):
        return Feasibility(
            False, "the pointwise variance of function-valued draws is not implemented yet"
        )
    if not isinstance(event, NumericSpec):
        return Feasibility(
            False,
            f"cannot take the variance of draws of a non-numeric event; got {type(event).__name__}",
        )
    return _can_sample(call, result)


def _monte_carlo_draws(call: BoundCall, operation_kind: str) -> Any:
    """``n_broadcast_samples`` workflow-owned draws of the law, the draw axis leading."""
    return _workflow_draws(
        call.operands["d"],
        (call.controls["n_broadcast_samples"],),
        operation_kind=operation_kind,
        execution_mode="monte_carlo",
    )


#: The label, the component, and the level of the empirical law of a fallback's draws.
_DRAWS = "draws"


def _empirical_of(call: BoundCall, draws: Any) -> EmpiricalDistribution:
    """The empirical law of the draws, whose moments the fallbacks report.

    Draws of an array event are its atoms along their leading axis. Draws of a
    record event are its atoms as the batch of records the law's event
    declaration calls for, whether they arrive as a nested mapping of raw
    columns or as a record of columns. A batch of records is taken as it is.
    """
    # The empirical law is internal, so it takes a fixed name, which a derived
    # label could not be: a label may hold ``~``, a space, ``;``, or ``/``.
    name = _DRAWS
    event = call.operands["d"].event_spec.spec
    if isinstance(draws, Batch):
        records = isinstance(draws.element_spec, RecordSpec)
        return EmpiricalDistribution(draws, component=None if records else name, label=name)
    if not isinstance(event, RecordSpec):
        return EmpiricalDistribution(
            jnp.asarray(draws), component=name, label=name, level=SAMPLE_LEVEL
        )
    atoms = _batch_class_for(event)(name, _raw_record(draws), SAMPLE_LEVEL, element_spec=event)
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
    if isinstance(call.operands["d"].event_spec.spec, NumericSpec):
        return _named_value(
            _empirical_of(call, _monte_carlo_draws(call, "mean"))._mean(),
            "mean",
            call.operands["d"].event_spec,
        )
    if _mixture_factory is None:
        raise RuntimeError("the mixture family is not installed; import probpipe")
    draws = _monte_carlo_draws(call, "mean")
    stored = draws.raw() if isinstance(draws, Batch) else draws
    laws = list(np.asarray(stored, dtype=object).reshape(-1))
    return _mixture_factory(laws, jnp.full(len(laws), 1.0 / len(laws)), label=_call_label(call))


def _mc_variance(call: BoundCall, result: OutputSpec | None) -> Any:
    """The variance of the empirical law of the draws.

    Each coordinate's variance is the mean squared deviation of the draws from
    their mean, dividing by the number of draws.
    """
    return _named_value(
        _empirical_of(call, _monte_carlo_draws(call, "variance"))._variance(),
        "variance",
        call.operands["d"].event_spec,
    )


def _dense(covariance: Any) -> Any:
    """A covariance as a dense array, whether an operator or an array holds it."""
    to_dense = getattr(covariance, "to_dense", None)
    return to_dense() if callable(to_dense) else jnp.asarray(covariance)


def _mc_cov(call: BoundCall, result: OutputSpec | None) -> Any:
    """The covariance of the empirical law of the draws, over their flat coordinates.

    It divides by the number of draws, as the variance does, so its diagonal
    is the variance and one draw has none.
    """
    return _dense(_empirical_of(call, _monte_carlo_draws(call, "cov"))._cov())


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
    return _record_batch(
        _named_value(quantiles, "quantile", call.operands["d"].event_spec), call, result
    )


# ---------------------------------------------------------------------------
# The summaries component by component
# ---------------------------------------------------------------------------


def _view(d: Distribution, components: tuple[str, ...]) -> Distribution:
    """The view of *d* at one component, or the selection of several."""
    return d[components[0]] if len(components) == 1 else d[components]


def _block_values(view: Distribution, value: Any, summary: str) -> dict[str, Any]:
    """The raw *value* of a summary of *view*, keyed by the components of the view's event.

    A view that exposes a record returns the summary keyed by its components'
    summary names, as ``mean(mu)``, and a view of one whole term returns that
    term's summary.
    """
    declaration = view.event_spec
    if declaration.exposes_record:
        return {name: value[_summary_name(summary, [name])] for name in declaration.components}
    ((name, _),) = declaration.components.items()
    return {name: value}


@dataclass(frozen=True)
class _Block:
    """Components of the law that one resolution of the operation computes together.

    Attributes
    ----------
    components : tuple of str
        The components, in event order.
    view : Distribution
        The law's view at the components: the view of one, or the selection of
        several.
    report : Feasibility
        The operation's report on the view, which names the route it selects
        and that route's exactness.
    """

    components: tuple[str, ...]
    view: Distribution
    report: Feasibility


@dataclass(frozen=True)
class _Partition:
    """How ``by_component`` splits a call: its blocks in event order, and the split's report."""

    blocks: tuple[_Block, ...]
    report: Feasibility


class _ByComponent(_DelegatingRoute):
    """The summary of each component of a record event, by the route that component's view selects.

    ``mean``, ``variance``, and ``quantile`` are computed coordinate by
    coordinate, so the summary of a law ``d`` whose event exposes a record is
    the record of the summaries of the views ``d[B]`` over any partition of its
    components into blocks ``B``. The route forms the partition from the
    operation's own resolution on each component's view: a component for which
    the operation selects an exact route is a block of its own, and the other
    components form one block, whose draws they share. Each block is resolved
    and run as the operation resolves a call on its view, under the call's
    controls with ``method`` unset and ``exact_only`` that of the candidate,
    inside the call itself, so its draws are the call's own.

    The route's exactness is that of its blocks: it is exact when every block's
    route is, and approximate otherwise. It applies when the event is a record
    of at least two components, at least one of which has an exact route, and
    the other components have a route together, so a law with no exact
    component keeps the Monte Carlo fallback. It splits the top-level
    components only, so a record-valued component is one block, whose route its
    view selects. The result's provenance names, for each block, its
    components, its route, and that route's exactness.

    Parameters
    ----------
    summary : str
        The summary's name, as each component of its result is named:
        ``"mean"``, ``"variance"``, or ``"quantile"``.
    """

    source = RouteSource.STRUCTURAL

    def __init__(self, summary: str) -> None:
        super().__init__("by_component")
        self.summary = summary

    @property
    def condition(self) -> str:
        """The route's feasibility condition in words."""
        return (
            "the event is a record of several components, at least one of which has an exact "
            "route of its own, and the rest of which have a route together"
        )

    @staticmethod
    def _controls(call: BoundCall, exact_only: bool) -> dict[str, Any]:
        """The controls a block is resolved under: the call's, with ``method`` unset.

        A block returns its raw value, which the route assembles.
        """
        return {**call.controls, "method": None, "exact_only": exact_only, "raw": True}

    @staticmethod
    def _values(call: BoundCall, view: Distribution) -> dict[str, Any]:
        """The call's arguments, with the law replaced by *view*."""
        return {**call.operands, "d": view}

    def _resolve(self, call: BoundCall, view: Distribution, exact_only: bool) -> Feasibility:
        """The operation's report on *view*: the route it selects and that route's exactness.

        A view the operation does not apply to, as a measure-valued component
        has no event-typed variance, is infeasible.
        """
        operation = call.operation
        controls = self._controls(call, exact_only)
        try:
            return check_point(
                operation,
                self._values(call, view),
                controls,
                operation._route_candidates(controls),
            )
        except ApplicabilityError as error:
            return Feasibility(False, str(error))

    def _partition(self, call: BoundCall, exact_only: bool) -> _Partition:
        """The blocks of the call, in event order, with the report of the split.

        Parameters
        ----------
        call : BoundCall
            The call, whose law is ``call.operands["d"]``.
        exact_only : bool
            Whether each block is restricted to exact routes.

        Returns
        -------
        _Partition
            The blocks and the split's report, which is infeasible when the
            event is not a record of several components, when no component has
            an exact route, or when the other components have no route
            together; unresolved when a component's resolution is; and
            otherwise feasible, exact exactly when every block's route is.
        """
        d = call.operands["d"]
        if (
            not isinstance(d, Distribution)
            or not d.event_spec.exposes_record
            or len(d.event_spec.components) < 2
        ):
            reason = "the event is not a record of several components"
            return _Partition((), Feasibility(False, reason))
        components = tuple(d.event_spec.components)
        views = {c: _view(d, (c,)) for c in components}
        reports = {c: self._resolve(call, views[c], exact_only) for c in components}
        unresolved = [report for report in reports.values() if report.feasible is None]
        if unresolved:
            pending = tuple(dict.fromkeys(item for report in unresolved for item in report.pending))
            return _Partition((), Feasibility(None, pending=pending))
        exact = tuple(c for c in components if reports[c].feasible is True and reports[c].exact)
        if not exact:
            return _Partition((), Feasibility(False, f"no component has an exact {self.summary}"))
        blocks = [_Block((c,), views[c], reports[c]) for c in exact]
        rest = tuple(c for c in components if c not in exact)
        if rest:
            view = _view(d, rest)
            remaining = self._resolve(call, view, exact_only)
            if remaining.feasible is None:
                return _Partition((), Feasibility(None, pending=remaining.pending))
            if remaining.feasible is False:
                return _Partition((), self._declined(call, exact, rest, exact_only))
            blocks.append(_Block(rest, view, remaining))
        blocks.sort(key=lambda block: components.index(block.components[0]))
        exactness = all(block.report.exact is True for block in blocks)
        return _Partition(tuple(blocks), PointReport(True, exact=exactness))

    def _declined(
        self, call: BoundCall, exact: tuple[str, ...], rest: tuple[str, ...], exact_only: bool
    ) -> Feasibility:
        """The report when the components in *rest* have no route together.

        When the call itself sets ``exact_only``, the report says how to
        compute the exact components, and the call's error leads with it.
        """
        if not exact_only:
            return Feasibility(
                False, f"the components {format_names(rest)} have no {self.summary} route"
            )
        reason = (
            f"the components {format_names(rest)} have no exact {self.summary}, but "
            f"{format_names(exact)} do"
        )
        if not call.controls["exact_only"]:
            return Feasibility(False, reason)
        return Feasibility(
            False,
            f"{reason}; call {self.summary} on the view of each component that does, as "
            f"{self.summary}(d[{exact[0]!r}])",
            actionable=True,
        )

    def _run_block(self, call: BoundCall, block: _Block, exact_only: bool) -> tuple[Any, str, bool]:
        """The raw summary of *block*, with the route that computed it and that route's exactness.

        The block is resolved and run as the operation runs a call on its view,
        inside this call, so its draws are this call's own.
        """
        operation = call.operation
        controls = self._controls(call, exact_only)
        point, declared, _ = operation._plan_point(self._values(call, block.view), controls)
        candidates = operation._route_candidates(controls)
        candidate, report = selected(operation.label, controls, candidates, point, declared)
        exactness = candidate.exactness(report) is True
        return candidate.run(point, declared, report), candidate.route_name, exactness

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
            The call's result declaration.
        method : str or None
            Unused; a ``method`` control names this route itself.
        exact_only : bool
            Whether each block is restricted to exact routes.

        Returns
        -------
        Any
            The summary keyed by the law's components, each named for the
            summary, as a term whose provenance records each block's
            components, route, and exactness; under ``raw``, its raw form.

        Raises
        ------
        ResolutionError
            If the split does not realize the call, which a probe reports first.
        MathematicalDomainError
            If a block's route finds its summary undefined, as for a Cauchy
            component's mean.
        """
        d = call.operands["d"]
        partition = self._partition(call, exact_only)
        if partition.report.feasible is not True:
            raise ResolutionError(
                f"{call.operation.label}: route {self.name!r} does not apply: "
                f"{partition.report.description}"
            )
        values: dict[str, Any] = {}
        records = []
        for block in partition.blocks:
            value, route, exactness = self._run_block(call, block, exact_only)
            values.update(_block_values(block.view, value, self.summary))
            records.append(
                {"components": list(block.components), "route": route, "exact": exactness}
            )
        assembled = {c: values[c] for c in d.event_spec.components}
        # A quantile at several levels is a batch of records, which
        # ``_record_batch`` assembles; any other value passes through it.
        value = _record_batch(_named_value(assembled, self.summary, d.event_spec), call, result)
        return _assembled(
            value,
            call,
            result,
            parents=[block.view for block in partition.blocks],
            metadata={"route": self.name, "blocks": records},
        )


# ---------------------------------------------------------------------------
# The operations
# ---------------------------------------------------------------------------


@operation(result=_mean_result)
def mean(d: Distribution):
    """The mean ``E[X]`` for ``X ~ d``, a value shaped like one draw.

    A record-drawing law's mean is a ``Record`` with the law's schema, a random
    function's is its mean function, and a random measure's is the
    marginalized law.

    The routes, in selection order, are the law's closed form,
    ``closed_form``; ``by_component``, which computes each component of a
    record event by the route that component's view selects; and the Monte
    Carlo fallback, ``monte_carlo``. A dependent joint claims no mean, so when
    one of its components, such as a root factor, has an exact mean, its mean
    takes ``by_component``: that component's mean is exact, and the result as
    a whole is approximate.

    Parameters
    ----------
    d : Distribution
        The law, whose ``_mean`` gives the closed form, whose components' views
        give their own means otherwise, and whose draws the Monte Carlo
        fallback averages.

    Returns
    -------
    TrackedTerm
        The mean at the law's declared event kind, labeled ``E[(mu, tau) ~ d]``
        for a law ``d`` over ``mu`` and ``tau``.

    Raises
    ------
    ResolutionError
        If *d* has no closed-form mean and does not sample, or, under
        ``exact_only``, if a component of a record event has no exact mean.
    MathematicalDomainError
        If the mean of *d*, or of a component whose own route is exact, is
        known to be undefined, as a Cauchy law's is.
    """


mean.capability_route(
    "closed_form",
    operand="d",
    protocol=SupportsMean,
    method="_mean",
    exact=True,
    execute=_summary_route("mean", _capability("_mean")),
)
mean.register_route(_ByComponent("mean"))
mean.fallback_route("monte_carlo", check=_can_average, execute=_mc_mean, exact=False)


@operation(result=_variance_result, conditions=(_event_typed_variance,))
def variance(d: Distribution):
    """The variance of ``X ~ d``, a value shaped like one draw.

    The routes are those of ``mean``: the closed form, ``closed_form``;
    ``by_component``, which computes each component of a record event by the
    route that component's view selects; and the Monte Carlo fallback,
    ``monte_carlo``.

    Parameters
    ----------
    d : Distribution
        The law, whose ``_variance`` gives the closed form, whose components'
        views give their own variances otherwise, and whose draws the Monte
        Carlo fallback uses.

    Returns
    -------
    TrackedTerm
        The variance at the law's declared event kind, with non-negative
        leaves, labeled ``Var[(mu, tau) ~ d]``.

    Raises
    ------
    ApplicabilityError
        If the event is measure-valued, which has no event-typed variance in
        general.
    ResolutionError
        If *d* has no closed-form variance and does not sample, or, under
        ``exact_only``, if a component of a record event has no exact variance.
    MathematicalDomainError
        If the variance of *d*, or of a component whose own route is exact, is
        known to be undefined, as a Cauchy law's is.
    """


variance.capability_route(
    "closed_form",
    operand="d",
    protocol=SupportsVariance,
    method="_variance",
    exact=True,
    execute=_summary_route("variance", _capability("_variance")),
)
variance.register_route(_ByComponent("variance"))
variance.fallback_route(
    "monte_carlo", check=_can_average_squares, execute=_mc_variance, exact=False
)


@operation(result=_cov_result, conditions=(_numeric_event,))
def cov(d: Distribution):
    """The covariance of the flattened draw of ``X ~ d``, a ``(size, size)`` array.

    Parameters
    ----------
    d : Distribution
        A law with a numeric event, whose ``_cov`` gives the closed form, and
        whose draws the Monte Carlo fallback uses otherwise.

    Returns
    -------
    NumericArray
        The dense covariance over the event's coordinates in canonical order,
        labeled ``Cov[(mu, tau) ~ d]``.

    Raises
    ------
    ApplicabilityError
        If the event is not numeric.
    ResolutionError
        If *d* has no closed-form covariance and does not sample.
    """


def _closed_form_cov(call: BoundCall, result: OutputSpec | None) -> Any:
    """``d._cov()`` as a dense array."""
    return _dense(call.operands["d"]._cov())


cov.capability_route(
    "closed_form",
    operand="d",
    protocol=SupportsCovariance,
    method="_cov",
    exact=True,
    execute=_closed_form_cov,
)
cov.fallback_route("monte_carlo", check=_can_sample, execute=_mc_cov, exact=False)


@operation(result=_quantile_result, conditions=(_numeric_event,))
def quantile(d: Distribution, q: Any):
    """The per-coordinate quantiles of ``X ~ d`` at the levels *q*.

    The routes are the law's closed form, ``closed_form``; ``by_component``,
    which computes each component of a record event by the route that
    component's view selects; and the Monte Carlo fallback, ``monte_carlo``,
    which ``with_options(method=...)`` selects among. The fallback's quantile
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
        those values on a level named ``quantile``. It is labeled
        ``Q[mu ~ d]``.

    Raises
    ------
    ApplicabilityError
        If the event is not numeric, or *q* is not numeric.
    MathematicalDomainError
        If a concrete level lies outside ``[0, 1]``.
    ResolutionError
        If *d* has no closed-form quantiles and does not sample, or, under
        ``exact_only``, if a component of a record event has no exact
        quantiles.
    """


def _closed_form_quantile(call: BoundCall, result: OutputSpec | None) -> Any:
    """``d._quantile(q)`` at levels in ``[0, 1]``.

    A record event's quantiles at several levels are the declared batch of
    records.
    """
    quantiles = call.operands["d"]._quantile(_check_levels(call.operands["q"]))
    return _record_batch(
        _named_value(quantiles, "quantile", call.operands["d"].event_spec), call, result
    )


quantile.capability_route(
    "closed_form",
    operand="d",
    protocol=SupportsQuantile,
    method="_quantile",
    exact=True,
    execute=_closed_form_quantile,
)
quantile.register_route(_ByComponent("quantile"))
quantile.fallback_route("monte_carlo", check=_can_sample, execute=_mc_quantile, exact=False)


def _evaluate_applies(d: Any, f: Any, fixed_args: Mapping[str, Any] | None = None) -> Any:
    """``evaluate`` has a route for the integrand over the law."""
    return evaluate.check(f, d, fixed_args)


@operation(
    result=_expectation_result,
    roles={"f": (FunctionSpec,)},
    identity_check=_evaluate_applies,
)
def expectation(d: Distribution, f: Any, fixed_args: Mapping[str, Any] | None = None):
    """The expectation ``E[f(X)]`` for ``X ~ d``, defined as ``mean(evaluate(f, d))``.

    A law claiming ``SupportsExpectation`` computes it in closed form, and
    otherwise the call takes the routes of ``evaluate``: its evaluation rules,
    which ``with_options(method=...)`` names, and its controls, such as the
    sampling lift's ``n_broadcast_samples``.

    Parameters
    ----------
    d : Distribution
        The law to integrate against.
    f : callable or Function
        Maps one draw to a value; a ``Function``'s output declaration names the
        result's kind.
    fixed_args : mapping of str to Any, optional
        The other parameters of a map with several, by name, bound as
        ``evaluate`` binds them.

    Returns
    -------
    TrackedTerm
        ``E[f(X)]`` at the kind the integrand's output declaration names,
        labeled ``E[f(mu ~ d)]`` by the integrand's label: a ``Function``'s
        label, a callable's ``__name__``, and ``f`` for a lambda.

    Raises
    ------
    ApplicabilityError
        If *f* is not a callable, or *fixed_args* leaves other than one of its
        parameters open.
    ResolutionError
        If neither the closed form nor an evaluation rule applies.
    """
    return mean(evaluate(f, d, fixed_args))


def _integrand(call: BoundCall) -> Callable[[Any], Any]:
    """The integrand as a callable of one draw, with its other parameters bound."""
    f = call.operands["f"]
    raw = f.raw() if isinstance(f, Function) else f
    fixed = dict(call.operands.get("fixed_args") or {})
    if not fixed:
        return raw
    parameter = _bound_parameter(_as_function(f), fixed)
    values = {
        name: value.raw() if isinstance(value, TrackedTerm) else value
        for name, value in fixed.items()
    }
    return lambda draw: raw(**{parameter: draw}, **values)


def _closed_form_expectation(call: BoundCall, result: OutputSpec | None) -> Any:
    """``d._expectation(f)``, each component of the integrand's output named ``mean(...)``."""
    value = call.operands["d"]._expectation(_integrand(call))
    declared = _as_function(call.operands["f"]).output_spec
    return value if declared is None else _named_value(value, "mean", declared)


expectation.capability_route(
    "closed_form",
    operand="d",
    protocol=SupportsExpectation,
    method="_expectation",
    exact=True,
    execute=_closed_form_expectation,
)


class _PushforwardMean(_EvaluationRules):
    """The mean of the pushforward ``evaluate(f, d)``, by the rule the evaluation-rule registry selects.

    The route takes ``evaluate``'s rules, rule names, and controls, apart from
    ``include_inputs``, so the mean is of the integrand's values alone. A map
    whose parameter takes the law itself, rather than its draws, has nothing
    to integrate.
    """

    _forwarded = tuple(name for name in _FORWARDED_CONTROLS if name != "include_inputs")

    def _values(self, call: BoundCall) -> tuple[Function, str, Any, dict[str, Any]]:
        """The integrand, the parameter a draw binds, the law, and the fixed arguments."""
        f = _as_function(call.operands["f"])
        fixed = dict(call.operands.get("fixed_args") or {})
        return f, _bound_parameter(f, fixed), call.operands["d"], fixed

    def probe(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Feasibility:
        """The registry's report for the law's draws."""
        f, parameter, operand, _ = self._values(call)
        if not _lifts(f, parameter, operand):
            return Feasibility(
                False,
                f"{f.label!r} takes the law itself at {parameter!r}, so it has no draws to "
                f"integrate",
            )
        return super().probe(call, result, method=method, exact_only=exact_only)

    def run(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Any:
        """The mean of the pushforward law the selected rule returns."""
        pushforward = super().run(call, result, method=method, exact_only=exact_only)
        return mean.with_options(raw=True)(pushforward)


expectation.register_route(_PushforwardMean())


# ---------------------------------------------------------------------------
# The expressions of the moments
# ---------------------------------------------------------------------------


def _moment_expression(summary: str) -> Callable[[Any], Expression]:
    """The expression rule of a moment *summary*: the moment of a draw of the law, as ``E[mu ~ d]``."""

    def rule(d: Any) -> Expression:
        return Summary(summary, draw_of(d))

    rule.__doc__ = f"The expression ``{summary}[draw]`` of the moment of a draw of *d* (II.4)."
    return rule


def _integrand_label(f: Any) -> str:
    """The label of an integrand: a Function's label, a callable's ``__name__``, and ``f`` for a lambda."""
    if isinstance(f, Function):
        return f.label
    name = getattr(f, "__name__", "f")
    return "f" if name == "<lambda>" else name


def _expectation_expression(d: Any, f: Any) -> Expression:
    """The expression of ``E[f(X)]`` for ``X ~ d``: the expectation of *f* at a draw, as ``E[f(mu ~ d)]``."""
    return Summary("E", Applied(_integrand_label(f), (draw_of(d),)))


_install_expression_rule(mean, _moment_expression("E"))
_install_expression_rule(variance, _moment_expression("Var"))
_install_expression_rule(cov, _moment_expression("Cov"))
_install_expression_rule(quantile, _moment_expression("Q"))
_install_expression_rule(expectation, _expectation_expression)
