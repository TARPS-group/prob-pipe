"""Resolution, step 6 of the call stack: the route that realizes one point of a call.

A Function realized by routes, as an operation is (VI.0), lists its candidate
routes in selection order under the call's controls, and plans each point of a
call into the object its routes read and the point's result declaration. The
engine probes the candidates in that order, from declarations alone, and
selects the first that is not infeasible: a feasible candidate runs, and an
unresolved one ranked above every feasible one leaves the selection undecided,
which a call refuses and a check reports (V.1, V.7). A Function that lists no
candidates is realized by its body.

A candidate has the members the engine reads:

1. ``label``: how a report names it, as ``route/method`` for a registry route's
   method;
2. ``route_name``: the name of the route it belongs to;
3. ``probe(call, result)`` and ``run(call, result, report)``: its feasibility
   check, which reads the point's call object and result declaration and runs
   nothing, and its execution, which returns the raw result;
4. ``exactness(report)`` and ``method_of(report)``: the exactness of the
   implementation its report selected, and the registry method, if any;
5. ``methods``: the methods of its route's registry, or ``None`` for a route
   that delegates to none.

The check of a call the engine lifts is made at each of its points: a swept
batch at each element, and a law the call broadcasts at a stand-in for its
draw, since a check draws nothing.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product
from types import MappingProxyType
from typing import Any

from ..core._batch import BatchSpec
from ..core._dispatch import Feasibility, MethodInfo, ResolutionError
from ..core._spec_base import TermSpec
from ..core._specs import OutputSpec
from ..values._binding import FunctionInputRef, input_ref_value, replace_input_refs
from ._call import CallReport
from ._plan import BroadcastPlan
from ._sweep import slice_sweep_values

__all__ = [
    "PointReport",
    "StandIn",
    "call_report",
    "check_point",
    "check_points",
    "selected",
]


@dataclass(frozen=True)
class StandIn:
    """A stand-in for one point of a lifted argument, carrying the point's declaration.

    A check draws nothing, so where a call passes a law's draw the check passes
    this stand-in, whose spec is the law's event declaration, and an empty
    sweep's point holds one whose spec is the batch's element declaration.
    """

    spec: TermSpec


@dataclass(frozen=True)
class PointReport(Feasibility):
    """What the check of one point finds, and, combined, what the check of a call finds.

    The point is feasible when a route is selected, unresolved when a route
    ranked above every feasible one still needs declarations, and infeasible
    when no route applies. A route's probe may also return one, to report a
    feasible route of a stated exactness that runs no method.

    Attributes
    ----------
    route : str or None
        The selected route, when selection can be decided and, for a call, when
        every point selects it.
    method : str or None
        The registry method a selected registry route delegates to.
    exact : bool or None
        The selected implementation's exactness; for a call, ``False`` when any
        point's is.
    result : OutputSpec or None
        The result declaration planning derived, which for a lifted call is that
        of one point.
    routes : tuple of (str, Feasibility, bool or None)
        Each probed candidate's label, report, and exactness, in selection
        order. The exactness is the implementation's that the candidate's report
        selects, its route's declaration for a route that states one.
    deferred : tuple of str
        The checks deferred to the return.
    """

    route: str | None = None
    method: str | None = None
    exact: bool | None = None
    result: OutputSpec | None = None
    routes: tuple[tuple[str, Feasibility, bool | None], ...] = ()
    deferred: tuple[str, ...] = ()


def _probe(
    candidates: Sequence[Any], call: Any, result: OutputSpec | None
) -> tuple[Any, Feasibility | None, list[tuple[Any, Feasibility]]]:
    """Probe *candidates* in selection order until one is feasible or unresolved.

    Parameters
    ----------
    candidates : sequence
        The routes, in selection order.
    call : Any
        The point's call object, which each probe reads.
    result : OutputSpec or None
        The point's result declaration.

    Returns
    -------
    tuple
        The first candidate that is not infeasible and its report, both
        ``None`` when every candidate is infeasible, and every probed candidate
        with its report.
    """
    probed: list[tuple[Any, Feasibility]] = []
    for candidate in candidates:
        report = candidate.probe(call, result)
        probed.append((candidate, report))
        if report.feasible is not False:
            return candidate, report, probed
    return None, None, probed


def _no_route(
    label: str, controls: Mapping[str, Any], probed: list[tuple[Any, Feasibility]]
) -> str:
    """The message naming each probed candidate and why it does not apply.

    The message leads with the first actionable reason, which concerns a detail
    of the call that the caller can fix rather than the kind of its arguments.
    """
    restriction = " with exact_only" if controls.get("exact_only") else ""
    if not probed:
        return f"{label}: no route applies{restriction}; none is registered"
    tried = "; ".join(
        f"{candidate.label}: {report.description or 'infeasible'}" for candidate, report in probed
    )
    lead = next((report.description for _, report in probed if report.actionable), "")
    if lead:
        return f"{label}: {lead}. Routes tried{restriction}: {tried}"
    return f"{label}: no route applies{restriction}. Tried: {tried}"


def selected(
    label: str,
    controls: Mapping[str, Any],
    candidates: Sequence[Any],
    call: Any,
    result: OutputSpec | None,
) -> tuple[Any, Feasibility]:
    """The candidate that realizes one point of a call, and its report.

    Parameters
    ----------
    label : str
        The Function's label, which the messages name.
    controls : Mapping of str to Any
        The call's resolved controls, whose ``exact_only`` the message reports.
    candidates : sequence
        The Function's routes, in selection order.
    call : Any
        The point's call object, which each probe reads.
    result : OutputSpec or None
        The point's result declaration.

    Returns
    -------
    candidate : Any
        The selected candidate.
    report : Feasibility
        The candidate's feasible report, which its ``run`` receives.

    Raises
    ------
    ResolutionError
        If no candidate is feasible, naming each one tried and why it declined,
        or if the first candidate that is not infeasible is unresolved, naming
        what it waits on.
    """
    candidate, report, probed = _probe(candidates, call, result)
    if candidate is None or report is None:
        raise ResolutionError(_no_route(label, controls, probed))
    if report.feasible is None:
        raise ResolutionError(
            f"{label}: route {candidate.label!r} is unresolved; pending: {', '.join(report.pending)}"
        )
    return candidate, report


def check_point(
    function: Any,
    values: Mapping[str, Any],
    controls: Mapping[str, Any],
    candidates: Sequence[Any],
    *,
    select: bool = True,
) -> PointReport:
    """The report of one point of a call: its planning, and its selection.

    Without *select* the point is planned and no route is selected, as for an
    empty sweep, which runs none.

    Parameters
    ----------
    function : Function
        The Function realized by routes, which plans the point.
    values : Mapping of str to Any
        The point's arguments, by parameter name.
    controls : Mapping of str to Any
        The call's resolved controls.
    candidates : sequence
        The Function's routes, in selection order.
    select : bool
        Whether to probe the candidates; ``True`` unless the point runs no route.

    Returns
    -------
    PointReport
        The point's feasibility, with its planned result declaration and its
        deferred checks.

    Raises
    ------
    ApplicabilityError
        If an applicability condition of the function fails.
    """
    call, result, deferred = function._plan_point(values, controls)
    if not select:
        return PointReport(True, result=result, deferred=deferred)
    candidate, report, probed = _probe(candidates, call, result)
    reports = tuple(
        (_label(probed_candidate, probe), probe, probed_candidate.exactness(probe))
        for probed_candidate, probe in probed
    )
    if candidate is None or report is None:
        return PointReport(
            False,
            _no_route(function.label, controls, probed),
            result=result,
            routes=reports,
            deferred=deferred,
        )
    if report.feasible is None:
        return PointReport(
            None, pending=report.pending, result=result, routes=reports, deferred=deferred
        )
    return PointReport(
        True,
        route=candidate.route_name,
        method=candidate.method_of(report),
        exact=candidate.exactness(report),
        result=result,
        routes=reports,
        deferred=deferred,
    )


def _label(candidate: Any, report: Feasibility) -> str:
    """How a report names *candidate*: ``route/method`` for the method its report names."""
    method = candidate.method_of(report)
    return candidate.label if method is None else f"{candidate.route_name}/{method}"


def _element_spec(values: Mapping[str, Any], ref: FunctionInputRef) -> TermSpec:
    """The declaration of one element of the swept argument *ref*."""
    spec = input_ref_value(values, ref).spec
    return spec.element_spec if isinstance(spec, BatchSpec) else spec


def _draws(values: Mapping[str, Any], plan: BroadcastPlan) -> dict[FunctionInputRef, StandIn]:
    """A stand-in for a draw of each law the call broadcasts over, by its reference."""
    return {ref: StandIn(input_ref_value(values, ref).event_spec.spec) for ref in plan.dist_args}


def _points(
    values: Mapping[str, Any], plan: BroadcastPlan
) -> Iterator[tuple[tuple[int, ...], dict[str, Any]]]:
    """Each point of a call the engine realizes, with its cell in the sweep.

    A swept argument contributes its element at the cell, as the engine's sweep
    reads it, and a law the call broadcasts over contributes a stand-in for its
    draw.
    """
    draws = _draws(values, plan)
    if not plan.array_args:
        yield (), replace_input_refs(values, draws)
        return
    cells = product(*(range(size) for size in plan.sweep_batch_shape))
    for index, cell in enumerate(cells):
        row = slice_sweep_values(values=values, index=index, array_groups=plan.array_groups)
        yield cell, replace_input_refs(row, draws)


def _combined(reports: list[tuple[tuple[int, ...], PointReport]]) -> PointReport:
    """The report of a call from those of its points, each with its sweep cell.

    The call is infeasible at its first infeasible point, and unresolved when a
    point is unresolved and none is infeasible. Otherwise it is feasible, with
    the route and method every point selects, if they agree, and the exactness
    its points share, approximate when any point is.
    """
    deferred = tuple(dict.fromkeys(item for _, report in reports for item in report.deferred))
    for cell, report in reports:
        if report.feasible is False:
            where = f"sweep cell {cell}: " if cell else ""
            return replace(report, description=where + report.description, deferred=deferred)
    unresolved = [report for _, report in reports if report.feasible is None]
    if unresolved:
        pending = tuple(dict.fromkeys(item for report in unresolved for item in report.pending))
        return replace(unresolved[0], pending=pending, deferred=deferred)
    first = reports[0][1]
    agree = all(
        (report.route, report.method) == (first.route, first.method) for _, report in reports
    )
    exactness = {report.exact for _, report in reports}
    exact = False if False in exactness else (first.exact if len(exactness) == 1 else None)
    return replace(
        first,
        route=first.route if agree else None,
        method=first.method if agree else None,
        exact=exact,
        deferred=deferred,
    )


def check_points(
    function: Any,
    values: Mapping[str, Any],
    controls: Mapping[str, Any],
    candidates: Sequence[Any],
    plan: BroadcastPlan,
) -> PointReport:
    """The report of a call from the checks of its points, as the engine realizes them.

    A plain call has one point. A lifted call is checked at each point until
    one is infeasible, and an empty sweep is planned at its element
    declaration and selects no route, since it runs none.
    """
    if plan.array_args and plan.n_sweep == 0:
        elements = {ref: StandIn(_element_spec(values, ref)) for ref in plan.array_args}
        point = replace_input_refs(values, {**elements, **_draws(values, plan)})
        return check_point(function, point, controls, candidates, select=False)
    reports: list[tuple[tuple[int, ...], PointReport]] = []
    for cell, point in _points(values, plan):
        report = check_point(function, point, controls, candidates)
        reports.append((cell, report))
        if report.feasible is False:
            break
    return _combined(reports)


def lifted_declaration(
    function: Any, values: Mapping[str, Any], controls: Mapping[str, Any], plan: BroadcastPlan
) -> OutputSpec | None:
    """The result declaration of one point of a call that lifts laws, as the call's check plans it.

    The point binds a stand-in for a draw of each law the call lifts, so the law
    of the evaluations declares the components that one point's result
    declares, as ``log_prob(mu)`` for a score.

    Parameters
    ----------
    function : Function
        The Function realized by routes, which plans the point.
    values : Mapping of str to Any
        The arguments of the call, or of one row of a sweep, by parameter name.
    controls : Mapping of str to Any
        The call's resolved controls.
    plan : BroadcastPlan
        The call's broadcast plan, which names the laws the call lifts.

    Returns
    -------
    OutputSpec or None
        The point's result declaration, or ``None`` when the declarations
        leave it open.

    Raises
    ------
    ApplicabilityError
        If an applicability condition of the function fails at the point.
    """
    point = replace_input_refs(values, _draws(values, plan))
    _, result, _ = function._plan_point(point, controls)
    return result


def call_report(
    point: PointReport,
    *,
    lifted: tuple[str, ...],
    conversions: Mapping[str, Any],
    candidates: Sequence[Any] = (),
) -> CallReport:
    """The CallReport of a call from the check of its points.

    Each probed candidate's report is named by its label, or by ``route/method``
    for the method its report names, and states the candidate's exactness; a
    report whose exactness is open states the call approximate. The report lists
    the methods of each registry that a route among *candidates* delegates to
    once, keyed by the names of the routes that delegate to it.
    """
    routes = tuple(
        MethodInfo(
            report.feasible,
            report.description,
            report.pending,
            method_name=label,
            exact=bool(exact),
        )
        for label, report, exact in point.routes
    )
    chosen: MethodInfo | None
    if point.feasible is None:
        chosen = None
    elif point.feasible is False:
        chosen = MethodInfo(False, point.description)
    else:
        name = point.route or ""
        if point.route is not None and point.method is not None:
            name = f"{point.route}/{point.method}"
        chosen = MethodInfo(True, method_name=name, exact=bool(point.exact))
    # Routes that delegate to one registry share its listing, as an exact and an
    # approximate candidate of one route do.
    sharing: dict[tuple[str, ...], list[str]] = {}
    for candidate in candidates:
        listing = getattr(candidate, "methods", None)
        if listing is not None and candidate.route_name not in sharing.get(listing, []):
            sharing.setdefault(listing, []).append(candidate.route_name)
    methods = {", ".join(routes): listing for listing, routes in sharing.items()}
    return CallReport(
        routes=routes,
        selected=chosen,
        deferred=point.deferred,
        result=point.result,
        lifted=lifted,
        conversions=MappingProxyType(dict(conversions)),
        methods=MappingProxyType(methods),
    )
