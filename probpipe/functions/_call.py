"""Resolve workflow controls, bind arguments, and admit them for the call engine.

Steps 2 and 3 of the call stack: the arguments bind to the wrapped function's
signature by Python's rules, and each bound argument is admitted against what
its parameter accepts: the kinds its role names where the Function declares
roles, as an operation does, and otherwise its input declaration and its
annotation. A violation of the call contract raises
:class:`ApplicabilityError`.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

from ..core._array_backend import _is_numeric_leaf
from ..core._batch import Batch, BatchSpec
from ..core._dispatch import MethodInfo
from ..core._repr import call_repr, format_names, mapping_repr, sequence_repr
from ..core._specs import (
    InputSpec,
    NumericArraySpec,
    OpaqueSpec,
    OutputSpec,
    RecordSpec,
    TermSpec,
)
from ..core.tracked import TrackedTerm
from ..distributions._capabilities import (
    SupportsConditionalCovariance,
    SupportsConditionalExpectation,
    SupportsConditionalLogProb,
    SupportsConditionalMarginals,
    SupportsConditionalMean,
    SupportsConditionalQuantile,
    SupportsConditionalRandomLogProb,
    SupportsConditionalRandomUnnormalizedLogProb,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsConditionalVariance,
)
from ..distributions._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ..distributions._distribution import Distribution, DistributionSpec
from ..values import FunctionSpec, _binding
from ..values._binding import WorkflowSignatureInfo, resolve_workflow_values
from . import _normalization

#: The conditional capabilities, each of which declares that a parameter consumes a kernel.
_KERNEL_HINT_PROTOCOLS: tuple[type, ...] = (
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsConditionalLogProb,
    SupportsConditionalRandomUnnormalizedLogProb,
    SupportsConditionalRandomLogProb,
    SupportsConditionalMean,
    SupportsConditionalVariance,
    SupportsConditionalCovariance,
    SupportsConditionalQuantile,
    SupportsConditionalExpectation,
    SupportsConditionalMarginals,
)


#: Whether the engine is probing a call for ``check``, which executes no route.
_CHECKING: ContextVar[bool] = ContextVar("probpipe_function_check", default=False)


def checking() -> bool:
    """Whether the engine is probing a call for ``check`` rather than running it.

    A route whose probe would compute part of its result to decide, as
    ``condition_on``'s exact stage does, reports the route unresolved instead
    while a check probes, since a check executes nothing.
    """
    return _CHECKING.get()


class ApplicabilityError(TypeError):
    """The arguments or declarations of a call violate its call contract.

    Raised when an argument's kind is not one its parameter accepts, when swept
    levels do not align, and when the declarations of a call conflict. The
    message names the parameter, what it accepts, and what arrived.
    """


@dataclass(frozen=True, repr=False)
class CallReport:
    """What ``check`` reports about a call, without executing it (V.1).

    A route's report is a :class:`~probpipe.core._dispatch.MethodInfo` named
    by the route, and by ``route/method`` for the method a registry route
    selects. The repr follows the repr convention (II.4), and in a notebook
    the report displays as a table with one row per route.

    Attributes
    ----------
    routes : tuple of MethodInfo
        Each probed route's report, in selection order: feasible, infeasible
        with a reason, or unresolved with the declarations it needs.
    selected : MethodInfo or None
        The selected route's report, or an infeasible report naming no route
        when none applies; ``None`` when selection cannot be decided, since a
        route ranked above every feasible one is unresolved.
    deferred : tuple of str
        The checks left to the return.
    result : OutputSpec or None
        The planned result declaration, or ``None`` when only the return
        settles it.
    lifted : tuple of str
        The parameters whose arguments lift or sweep.
    conversions : Mapping of str to ConversionInfo
        The planned conversion of each parameter that converts.
    methods : Mapping of str to tuple of str
        The methods of each registry the routes delegate to, in its selection
        order, keyed by the names of those routes joined by ``", "``. The
        routes' reports leave them out, so each registry's methods are listed
        once.
    """

    routes: tuple[MethodInfo, ...] = ()
    selected: MethodInfo | None = None
    deferred: tuple[str, ...] = ()
    result: OutputSpec | None = None
    lifted: tuple[str, ...] = ()
    conversions: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    methods: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))

    def __repr__(self) -> str:
        """The fields the report sets, as a call of its constructor (II.4)."""
        keywords: list[tuple[str, str]] = []
        if self.routes:
            keywords.append(("routes", sequence_repr(repr(info) for info in self.routes)))
        if self.selected is not None:
            keywords.append(("selected", repr(self.selected)))
        if self.deferred:
            keywords.append(("deferred", sequence_repr(repr(item) for item in self.deferred)))
        if self.result is not None:
            keywords.append(("result", repr(self.result)))
        if self.lifted:
            keywords.append(("lifted", format_names(self.lifted)))
        if self.conversions:
            converted = {name: repr(info) for name, info in self.conversions.items()}
            keywords.append(("conversions", mapping_repr(converted)))
        if self.methods:
            listed = {route: format_names(names) for route, names in self.methods.items()}
            keywords.append(("methods", mapping_repr(listed)))
        return call_repr("CallReport", keywords=keywords)

    def _repr_html_(self) -> str:
        """The report as an HTML table, for a notebook's display.

        Each probed route is a row with its route, method, exactness,
        feasibility, and reason, and the selected route's row is marked. An
        unresolved route's reason names what it waits on, and each check left
        to the return is a row of its own. The lifted arguments, the planned
        conversions, the registry methods, and the result declaration follow
        the table. Every value is escaped.
        """
        chosen = self.selected is not None and self.selected.feasible is True
        rows = []
        marked = False
        for info in self.routes:
            route, method = _route_and_method(info.method_name or "")
            mark = chosen and not marked and info.feasible is True
            marked = marked or mark
            reason = info.description
            if info.feasible is None:
                reason = "pending: " + ", ".join(info.pending)
            feasibility = _FEASIBILITY_TEXT[info.feasible]
            cells = ("selected" if mark else "", route, method, _exact_text(info.exact))
            rows.append(_html_row((*cells, feasibility, reason), selected=mark))
        for check in self.deferred:
            rows.append(_html_row(("", "", "", "", "deferred", check), selected=False))
        header = "".join(
            f"<th>{title}</th>" for title in ("", "route", "method", "exact", "feasible", "reason")
        )
        notes = []
        if self.selected is not None and self.selected.feasible is False:
            notes.append(("no route applies", self.selected.description))
        if self.selected is None:
            notes.append(("selection is undecided", "pending: " + ", ".join(self.pending)))
        if self.lifted:
            notes.append(("lifted", ", ".join(self.lifted)))
        for name, info in self.conversions.items():
            notes.append((f"conversion of {name}", repr(info)))
        for route, names in self.methods.items():
            notes.append((f"methods of {route}", ", ".join(names)))
        if self.result is not None:
            notes.append(("result", repr(self.result)))
        listed = "".join(
            f'<li><b>{html.escape(term)}</b>: <code style="white-space: pre">'
            f"{html.escape(text)}</code></li>"
            for term, text in notes
        )
        section = f"<ul>{listed}</ul>" if listed else ""
        return (
            f"<div><b>CallReport</b><table><thead><tr>{header}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>{section}</div>"
        )

    @property
    def feasible(self) -> bool | None:
        """Whether a route is selected: ``False`` when none applies, ``None`` when undecided."""
        return None if self.selected is None else self.selected.feasible

    @property
    def route(self) -> str | None:
        """The selected route's name, or ``None``."""
        if self.selected is None or self.selected.feasible is not True:
            return None
        return (self.selected.method_name or "").partition("/")[0] or None

    @property
    def method(self) -> str | None:
        """The registry method the selected route delegates to, or ``None``."""
        if self.selected is None or self.selected.feasible is not True:
            return None
        return (self.selected.method_name or "").partition("/")[2] or None

    @property
    def exact(self) -> bool | None:
        """The selected implementation's exactness, or ``None`` when none is selected."""
        return None if self.selected is None else self.selected.exact

    @property
    def pending(self) -> tuple[str, ...]:
        """What an undecided selection waits on: the needs of each unresolved route."""
        if self.selected is not None:
            return self.selected.pending
        return tuple(dict.fromkeys(item for info in self.routes for item in info.pending))

    @property
    def description(self) -> str:
        """The selected report's description, which names every route tried when none applies."""
        return "" if self.selected is None else self.selected.description


#: How a report's feasibility reads in the notebook table.
_FEASIBILITY_TEXT = {True: "feasible", False: "infeasible", None: "unresolved"}


def _route_and_method(label: str) -> tuple[str, str]:
    """A route report's label as its route and its method.

    A registry route's label is ``route/method`` for the method it selects and
    ``route (exact methods)`` for the methods it covers when it selects none.
    """
    route, _, method = label.partition("/")
    if method:
        return route, method
    route, _, covered = label.partition(" (")
    return route, covered.removesuffix(")")


def _exact_text(exact: bool | None) -> str:
    """An exactness as the notebook table shows it."""
    return "" if exact is None else ("exact" if exact else "approximate")


def _html_row(cells: tuple[str, ...], *, selected: bool) -> str:
    """One table row of escaped *cells*, bold when it is the *selected* route's."""
    style = ' style="font-weight: bold"' if selected else ""
    return f"<tr{style}>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>"


@dataclass(frozen=True)
class WorkflowCallOptions:
    """Optional call-time workflow controls outside user kwargs."""

    n_broadcast_samples: int | None = None
    include_inputs: bool | None = None


@dataclass(frozen=True)
class WorkflowCallOverrides:
    """Resolved call-time workflow settings consumed by ``Function``."""

    n_broadcast_samples: int
    include_inputs: bool


@dataclass(frozen=True)
class ResolvedWorkflowCall:
    """Fully resolved signature-shaped values plus workflow overrides."""

    values: dict[str, Any]
    overrides: WorkflowCallOverrides


def bind_call_inputs(
    info: WorkflowSignatureInfo,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    *,
    default_n_broadcast_samples: int,
    default_include_inputs: bool,
    options: WorkflowCallOptions | None = None,
) -> tuple[dict[str, Any], WorkflowCallOverrides]:
    """Bind user inputs and resolve workflow controls.

    Call inputs bind exactly like the wrapped Python function. Workflow
    controls come only from explicit ``options`` or construction defaults.
    """
    explicit_options = options if options is not None else WorkflowCallOptions()

    def resolve_option(name: str, default: Any = None) -> Any:
        explicit_value = getattr(explicit_options, name)
        if explicit_value is not None:
            return explicit_value

        return default

    overrides = WorkflowCallOverrides(
        n_broadcast_samples=resolve_option(
            "n_broadcast_samples",
            default_n_broadcast_samples,
        ),
        include_inputs=resolve_option(
            "include_inputs",
            default_include_inputs,
        ),
    )

    bound = info.signature.bind_partial(*args, **call_inputs)
    return dict(bound.arguments), overrides


def resolve_workflow_call(
    info: WorkflowSignatureInfo,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    *,
    bind: Mapping[str, Any],
    module: Any | None,
    dependency_type: type,
    workflow_name: str,
    default_n_broadcast_samples: int,
    default_include_inputs: bool,
    options: WorkflowCallOptions | None = None,
) -> ResolvedWorkflowCall:
    """Resolve one ``Function`` call into values plus overrides."""
    bound_inputs, overrides = bind_call_inputs(
        info,
        args,
        call_inputs,
        default_n_broadcast_samples=default_n_broadcast_samples,
        default_include_inputs=default_include_inputs,
        options=options,
    )
    values = resolve_workflow_values(
        info,
        bound_inputs,
        bind=bind,
        module=module,
        dependency_type=dependency_type,
        workflow_name=workflow_name,
    )
    return ResolvedWorkflowCall(values=values, overrides=overrides)


def _consumes_kernel(expected: Any) -> bool:
    """Whether an annotation declares that its parameter consumes a kernel itself."""
    if get_origin(expected) in (Union, UnionType):
        return any(_consumes_kernel(arm) for arm in get_args(expected))
    try:
        if isinstance(expected, type) and issubclass(expected, ConditionalDistribution):
            return True
    except TypeError:
        pass
    return expected in _KERNEL_HINT_PROTOCOLS


def _expects_value(expected: Any) -> bool:
    """Whether a parameter with annotation *expected* expects a value.

    A parameter expects a value when it is unannotated or annotated with a
    value type. An annotation that names a distribution, a kernel, or one of
    their capabilities declares that the parameter consumes that object, and
    ``Any`` admits every kind.
    """
    if expected is Any:
        return False
    return not (_normalization.is_distribution_hint(expected) or _consumes_kernel(expected))


def arrived_kind(value: Any) -> type[TermSpec]:
    """The kind *value* is, named by its spec class, by the kind-directed wrap of V.4.

    A tracked term is the kind of its spec; a raw mapping is a record, a numeric
    host an array, and a callable a function; any other value, a list, a tuple,
    or a set among them, is opaque.
    """
    if isinstance(value, TrackedTerm):
        spec = getattr(value, "spec", None)
        if isinstance(spec, TermSpec):
            return type(spec)
    if isinstance(value, Mapping):
        return RecordSpec
    if _is_numeric_leaf(value):
        return NumericArraySpec
    if callable(value):
        return FunctionSpec
    return OpaqueSpec


def _accepts(declared: type[TermSpec], kind: type[TermSpec]) -> bool:
    """Whether a slot declaring the kind *declared* admits a value of the kind *kind*.

    The record kinds admit each other, unification deciding their fields, and an
    opaque slot admits every kind, as ``OpaqueSpec.is_valid`` does.
    """
    if declared is OpaqueSpec or issubclass(kind, declared):
        return True
    return issubclass(declared, RecordSpec) and issubclass(kind, RecordSpec)


def _admitted_kind(value: Any, declared: type[TermSpec]) -> type[TermSpec]:
    """The kind of *value* that a slot declaring *declared* checks.

    A law at a slot of a value kind lifts, so its event kind is checked, and a
    batch at a slot of another kind is swept, so its element kind is.
    """
    lifted = (ConditionalDistributionSpec, DistributionSpec, BatchSpec)
    if isinstance(value, Distribution) and not issubclass(declared, lifted):
        return type(value.event_spec.spec)
    if isinstance(value, Batch) and not issubclass(declared, BatchSpec):
        return type(value.element_spec)
    return arrived_kind(value)


def role_admits(role: tuple[type[TermSpec], ...], kind: type[TermSpec]) -> bool:
    """Whether a parameter whose role names the kinds *role* admits a value of the kind *kind*.

    A named kind admits its subclasses, and the record kinds admit each other,
    unification deciding their fields.
    """
    return any(
        issubclass(kind, accepted)
        or (issubclass(accepted, RecordSpec) and issubclass(kind, RecordSpec))
        for accepted in role
    )


def role_lift(value: Any, role: tuple[type[TermSpec], ...]) -> Literal["sweep", "broadcast"] | None:
    """How the call lifts *value* at a parameter whose role does not admit it, or ``None``.

    A batch whose elements are of an admitted kind is swept, and a law whose
    draws are is broadcast (V.4, V.5); any other value has no lift.
    """
    if isinstance(value, Batch) and len(value.batch_shape) > 0:
        return "sweep" if role_admits(role, type(value.element_spec)) else None
    if isinstance(value, Distribution):
        event = value.event_spec.spec
        return "broadcast" if event is not None and role_admits(role, type(event)) else None
    return None


def _role_refusal(
    function_name: str | None, label: str, role: tuple[type[TermSpec], ...], value: Any
) -> ApplicabilityError:
    """The refusal of *value* at the parameter *label* names, whose role names *role*."""
    owner = f"{function_name}: " if function_name else ""
    kinds = ", ".join(kind.__name__ for kind in role)
    detail = f", whose kind is {arrived_kind(value).__name__}"
    if isinstance(value, Batch) and len(value.batch_shape) > 0:
        detail += f" with {type(value.element_spec).__name__} elements"
    elif isinstance(value, Distribution) and value.event_spec.spec is not None:
        detail += f" with {type(value.event_spec.spec).__name__} draws"
    return ApplicabilityError(
        f"{owner}{label!r} accepts {kinds}, but received a {type(value).__name__}{detail}"
    )


def _admit_by_role(
    info: WorkflowSignatureInfo,
    ref: _binding.WorkflowInputRef,
    value: Any,
    role: tuple[type[TermSpec], ...],
    *,
    function_name: str | None,
    lifts: bool,
) -> None:
    """Admit *value* at the parameter *ref* names by the kinds its role names.

    A role that names no kind marks a parameter that selects rather than
    supplies, which admits every value, and so does ``None`` at a parameter
    defaulting to it. Otherwise the value's own kind must be admitted, or, when
    the call lifts and the parameter is not annotated ``Any``, the kind of its
    elements or its draws.

    Raises
    ------
    ApplicabilityError
        If the role admits neither the value nor what it lifts to.
    """
    if not role:
        return
    if value is None and info.signature.parameters[ref.parameter_name].default is None:
        return
    if role_admits(role, arrived_kind(value)):
        return
    hint = _binding.parameter_lifting_hint(info, ref.parameter_name)
    if lifts and hint is not Any and role_lift(value, role) is not None:
        return
    raise _role_refusal(function_name, ref.label, role, value)


def admit_arguments(
    info: WorkflowSignatureInfo,
    values: Mapping[str, Any],
    *,
    input_spec: InputSpec | None = None,
    function_name: str | None = None,
    roles: Mapping[str, tuple[type[TermSpec], ...]] | None = None,
    lifts: bool = True,
) -> None:
    """Admit each bound argument against what its parameter accepts.

    A parameter with a role admits the kinds the role names, and a lifted
    argument whose elements or draws are of such a kind (VI.0). Any other
    argument is admitted against its parameter's own annotation, so an
    argument that a variadic parameter annotated ``Any`` collects is admitted
    whatever its kind. A conditional distribution at a parameter that expects a
    value is refused, since a kernel has no marginal law to lift over. Where an
    *input_spec* declares a slot, the argument's kind must be the slot's: a
    value's own kind, the event kind of a law the slot lifts, or the element
    kind of a batch it sweeps (V.4).

    Parameters
    ----------
    info : WorkflowSignatureInfo
        The wrapped function's signature and resolved annotations.
    values : Mapping of str to Any
        The bound arguments, shaped by the signature.
    input_spec : InputSpec or None
        The declared slots, whose kinds the arguments are admitted against.
    function_name : str or None
        The function's name, for the message.
    roles : Mapping of str to tuple of TermSpec subclasses, or None
        The kinds each parameter with a role accepts.
    lifts : bool
        Whether the call may lift an argument; a point evaluated with no
        lifting admits only the kinds a role names.

    Raises
    ------
    ApplicabilityError
        If an argument's kind is not one its parameter accepts, naming the
        parameter, the kinds it accepts, and what arrived.
    """
    if input_spec is not None:
        for name, spec in input_spec.items():
            if name not in values:
                continue
            value = values[name]
            kind = _admitted_kind(value, type(spec))
            if not _accepts(type(spec), kind):
                owner = f"{function_name}: " if function_name else ""
                arrived = type(value).__name__
                article = "an" if arrived[0] in "AEIOU" else "a"
                raise ApplicabilityError(
                    f"{owner}parameter {name!r} accepts {type(spec).__name__}, and got "
                    f"{article} {arrived}, whose kind is {kind.__name__}"
                )
    for ref in _binding.iter_input_refs(info, values):
        value = _binding.input_ref_value(values, ref)
        role = (roles or {}).get(ref.parameter_name)
        if role is not None:
            _admit_by_role(info, ref, value, role, function_name=function_name, lifts=lifts)
            continue
        if isinstance(value, ConditionalDistribution) and _expects_value(
            info.hints.get(ref.parameter_name)
        ):
            raise ApplicabilityError(
                f"parameter {ref.label!r} expects a value, and a value parameter accepts no "
                f"ConditionalDistribution: got {type(value).__name__} {value.name!r}, which is "
                f"a kernel with no marginal law to lift over. Condition it on a given value "
                f"first, or annotate the parameter ConditionalDistribution to consume the "
                f"kernel itself"
            )
