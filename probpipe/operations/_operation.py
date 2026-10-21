"""The operation model: roles, applicability checks, a result rule, and the routes of a call.

An **operation** is a :class:`~probpipe.values.Function` whose implementations
differ in the kinds they handle, in the declaration of their result, and in
where they come from. ``@operation`` declares one from a signature and a result
rule, and its routes are registered beside it. A call runs the engine's stack,
which reads three declarations from the operation at its steps:

1. **Admission** and lift classification read each parameter's **role**, the
   kinds it accepts, each named by its spec class: an argument the role admits
   passes whole, a batch or a law over an admitted kind is lifted, and any
   other argument is refused.
2. **Planning** checks the **applicability checks** and runs the **result
   rule**, which derives the output spec of each point's result from the
   operands' specs and from the parameters that select rather than supply,
   such as a field path; the return validates the result against it.
3. **Resolution** selects one **route** among the operation's candidates, in
   the order this module ranks them: exact routes before approximate ones, a
   fallback below every other route whatever its exactness, and registration
   order for the remaining ties. A route whose exactness is that of the
   implementations it delegates to, as a registry route's is, is ranked twice:
   restricted to its exact implementations among the exact routes, and
   otherwise among the approximate ones. A derived route ranks below the
   direct routes of its exactness and is ranked once, as the next paragraph
   states. The ``method`` and ``exact_only`` controls restrict the choice, and
   provenance records the selected route.

A derived operation is computed by its **formula** through its **derived
route**: by default the body route, which runs the declaration's body, or else
a composite route that runs a route of each constituent operation in turn. The
engine runs the derived route check and the derived route in a scope that passes
three controls of the derived call to the Function calls made within it,
operations and plain Functions alike (VI.0):

- ``exact_only`` restricts the route selection of every call in the scope, at
  any depth, whatever the call sets;
- ``max_evaluations``, when the derived call sets it, applies to each call the
  scope makes outside any other call, unless that call's view sets its own;
- ``num_samples`` passes as ``max_evaluations`` does.

A formula that estimates a summary through a lift, as ``expectation``'s does,
also spends the summary's resolved ``num_samples`` as the lift's
``max_evaluations``, since each draw is one evaluation of the map. The call of
a derived route states the Monte Carlo draws of the constituents it runs, so
its provenance records their ``num_samples``.

The scope covers the calls made in the calling thread and in the workers the
engine dispatches to. A planned conversion follows its parameter's
``conversions`` control, as in a direct call. Without ``exact_only`` a derived
route ranks among the approximate routes. Under ``exact_only`` every call in
the scope selects an exact route, and the derived route check reports under the
same restriction whether the constituents it probes have one, so a composite
route and a body route with a derived route check rank among the exact routes. A
body route without a derived route check has no such report, and ``exact_only``
excludes it.

An operation takes no key: each draw a route causes is a workflow-owned random
event, whose key :func:`_workflow_draws` derives from the workflow scope.

The operations form a registry, :data:`operation_registry`, whose ``list()`` and
``describe()`` report each operation's operands, whether it is primitive or
derived, and its routes in selection order.
"""

from __future__ import annotations

import ast
import copy
import difflib
import dis
import inspect
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, runtime_checkable

from .._messages import unknown_names
from ..core._array_backend import _event_shape_of, _is_numeric_leaf, _numpy_dtype_of
from ..core._catalog import EntrySummary, SupportsRegistryCataloging, registry_catalog
from ..core._dispatch import BaseDispatchRegistry, Feasibility, MethodInfo, ResolutionError
from ..core._expression import Expression, Named
from ..core._kinds import _KINDS
from ..core._record_spec import RecordSpec
from ..core._repr import format_names, public_class_name
from ..core._spec_base import NumericArraySpec, OpaqueSpec, TermSpec
from ..core._specs import OutputSpec
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..distributions._capabilities import _capability_guard, _guard_condition, _requirement
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._views import _drawn_form
from ..functions import _descendants, _evaluation_keys, _result
from ..functions._call import ApplicabilityError, CallReport
from ..functions._resolution import PointReport, StandIn
from ..values import Function, FunctionSpec
from ..values._binding import (
    make_signature_info_from_signature,
    resolved_hint,
    values_to_bound_arguments,
)
from ..values._function_base import _COMPLETED_AT_RETURN, _passing_controls_to_formula_calls

__all__ = [
    "BoundCall",
    "OperandSummary",
    "Operation",
    "OperationRegistry",
    "OperationRoute",
    "OperationSummary",
    "RouteSource",
    "RouteSummary",
    "operation",
    "operation_registry",
]

#: The term classes whose spec class the kind table does not record.
_UNREGISTERED_TERM_KINDS: tuple[tuple[type, type[TermSpec]], ...] = (
    (Function, FunctionSpec),
    (Record, RecordSpec),
)

#: The kinds an unannotated parameter accepts, which are the value kinds, so a
#: law there lifts and a batch is swept, as at a Function's parameter (V.5).
_VALUE_KINDS: tuple[type[TermSpec], ...] = (NumericArraySpec, RecordSpec, OpaqueSpec, FunctionSpec)


class RouteSource(Enum):
    """Where a route's implementation comes from."""

    STRUCTURAL = "structural"
    CAPABILITY = "capability"
    REGISTRY = "registry"
    FALLBACK = "fallback"


# ---------------------------------------------------------------------------
# The bound call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundCall:
    """One call of an operation, after binding and normalization.

    Attributes
    ----------
    operation : Function
        The operation, as the view the call went through.
    operands : Mapping[str, Any]
        Every bound argument by its parameter name, defaults included, in
        signature order: a term, a raw value, or a planned conversion; at a
        point of a lifted call, an element or a stand-in for a draw.
    controls : Mapping[str, Any]
        The resolved controls, ``method``, ``exact_only``, ``raw``, and
        ``method_options`` among them.

    Notes
    -----
    The engine probes a point's candidates and runs the selected one on the
    same bound call, so a route that repeats work across the probes of its two
    candidates and its run keeps it in the call's private memo, under a key of
    its own. The memo lives only as long as the call, so nothing it holds, such
    as a guard's outcome, outlives the call.
    """

    operation: Function
    operands: Mapping[str, Any]
    controls: Mapping[str, Any]
    _memo: dict[Any, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def specs(self) -> Mapping[str, TermSpec]:
        """Each supplying operand's spec: a term's own, or the kind its raw value wraps as.

        A parameter whose role accepts no kind selects rather than supplies, as a
        field path does, and has no entry. An optional argument left at ``None``
        has none either.
        """
        roles = self.operation._roles
        return MappingProxyType(
            {
                name: _spec_of(value)
                for name, value in self.operands.items()
                if roles.get(name) and value is not None
            }
        )

    @property
    def declarations(self) -> Mapping[str, Any]:
        """What the result rule and the applicability checks read, by parameter.

        A supplying operand contributes its spec, and a parameter that selects
        contributes its value.
        """
        specs = self.specs
        parameters = self.operation.signature.parameters
        empty = {inspect.Parameter.VAR_POSITIONAL: (), inspect.Parameter.VAR_KEYWORD: {}}
        return MappingProxyType(
            {
                name: specs.get(name, self.operands[name])
                if name in self.operands
                else empty.get(parameters[name].kind)
                for name in self.operation._declared_parameters
            }
        )


def _call_label(call: BoundCall) -> str:
    """The label the result of *call* takes, as its operation derives it (II.4).

    A route reads it to label a term it builds before the result boundary
    gives the result its expression. The label names the term and is never an
    output name or a level name, since a derived label may hold ``~``, a space,
    ``;``, or ``/``.
    """
    return call.operation._derived_label(call.operands)


def _assembled(
    value: Any,
    call: BoundCall,
    result: OutputSpec | None,
    *,
    parents: Iterable[TrackedTerm],
    metadata: Mapping[str, Any],
) -> Any:
    """*value*, which a route assembled from parts it computed, as the result of *call*.

    The value becomes the term *result* declares, under the call's label, with
    a provenance record of how the route assembled it: the operation's name,
    *parents*, and *metadata*, which names the route and what each part took.
    The call's own record keeps that record as a parent. Under the ``raw``
    control the value is returned as it is, since the result is detached.

    Parameters
    ----------
    value : Any
        The assembled raw result, in the form the call's routes return.
    call : BoundCall
        The call the route realizes.
    result : OutputSpec or None
        The output spec of the call's result.
    parents : iterable of TrackedTerm
        The terms the parts were computed from.
    metadata : Mapping of str to Any
        The record's metadata, whose ``"route"`` entry names the route.

    Returns
    -------
    Any
        The declared term carrying the record, or *value* under ``raw``.

    Raises
    ------
    ResultKindError
        If the kind of *value* differs from the kind *result* declares.
    ResultSchemaError
        If *value* violates the structure, dtype, or support *result* declares.
    """
    if call.controls["raw"]:
        return value
    term = _result.declared_term(value, result, _call_label(call))
    term.with_provenance(
        Provenance.create(call.operation.name, parents=list(parents), metadata=dict(metadata))
    )
    return term


def _spec_of(value: Any) -> TermSpec:
    """The spec of *value*: a term's own, and otherwise the kind the wrap table gives it.

    A stand-in for a draw or an element, which a check passes, carries its spec.
    """
    if isinstance(value, StandIn):
        return value.spec
    spec = getattr(value, "spec", None) if isinstance(value, TrackedTerm) else None
    if isinstance(spec, TermSpec):
        return spec
    if _is_numeric_leaf(value):
        return NumericArraySpec(tuple(_event_shape_of(value)), _numpy_dtype_of(value))
    return RecordSpec.infer_from({"value": value}).children["value"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@runtime_checkable
class OperationRoute(Protocol):
    """One implementation behind an operation, bound to a call rather than to argument types.

    A route has the interface of a dispatch method: a ``check`` that reads the
    call's declarations without evaluating the body, an ``execute`` that
    realizes the call over raw forms, and an ``exact`` flag. The construction
    helpers of :class:`Operation` build the four kinds; any object with these
    members registers through :meth:`Operation.register_route`.

    Attributes
    ----------
    name : str
        The route's name, unique within its operation.
    source : RouteSource
        Where the implementation comes from.
    exact : bool or None
        Whether the result denotes the requested mathematical object; ``None``
        for a route whose exactness is that of the implementation it delegates
        to, until that implementation is selected. A delegating route, which
        can restrict itself to exact implementations, is ranked twice: among
        the exact routes and among the approximate ones. A derived operation's
        derived route is ranked once, as the module docstring states.
        ``exact_only`` excludes any other route whose exactness is ``None``.
    """

    name: str
    source: RouteSource
    exact: bool | None

    def check(self, call: BoundCall, result: OutputSpec | None) -> Feasibility:
        """Whether this route realizes *call*, whose result's output spec is *result*."""
        ...

    def execute(self, call: BoundCall, result: OutputSpec | None) -> Any:
        """Realize *call* and return its raw result.

        A route may instead return the declared term with a provenance record
        of how the route produced it, which the call's own record keeps as a
        parent.
        """
        ...


def _as_feasibility(
    report: Any, source: Callable[..., Any], owner: str, subject: str = ""
) -> Feasibility:
    """*report* as a Feasibility, quoting the condition *source*'s docstring states.

    Parameters
    ----------
    report : bool, None, Feasibility, or CallReport
        What *source* returned. A CallReport is the report of another
        operation's ``check``, which a derived operation's derived route check returns.
    source : callable
        The check or condition that returned *report*, whose docstring's first
        paragraph states the condition.
    owner : str
        The checker that an unresolved report names, such as ``route 'derived'``.
    subject : str
        The prefix of an infeasible report's description, such as ``"convert: "``;
        a route's report has none, since the listing of routes names the route.

    Returns
    -------
    Feasibility
        A feasible report for ``True``, an infeasible report for ``False``, and an
        unresolved report for ``None``, the last two quoting the condition. A
        Feasibility is returned as it is, and a CallReport gives its selected
        route's report, or else an unresolved report of its pending checks.

    Raises
    ------
    TypeError
        If *report* is not a bool, ``None``, a Feasibility, or a CallReport.
    """
    if isinstance(report, Feasibility):
        return report
    if isinstance(report, CallReport):
        # Another operation's check, as a derived operation's formula reads it.
        if report.selected is not None:
            return report.selected
        return Feasibility(None, pending=report.pending)
    condition = _requirement(source)
    if report is True:
        return Feasibility(True)
    if report is False:
        reason = f"requirement not met: {condition}" if condition else "a requirement is not met"
        return Feasibility(False, f"{subject}{reason}")
    if report is None:
        suffix = f"; it requires: {condition}" if condition else ""
        return Feasibility(None, pending=(f"{owner} depends on values not yet known{suffix}",))
    raise TypeError(f"{owner} returned {report!r}; a check returns a bool, None, or a Feasibility")


def _subject_name(subject: Any) -> str:
    """*subject* as a message names it: its public class, and its label where it has one."""
    label = getattr(subject, "label", None)
    name = public_class_name(type(subject))
    return f"{name} {label!r}" if isinstance(label, str) else name


class _Route:
    """The name and exactness every constructed route declares.

    A route reads the numerical budgets of a call from its ``method_options``
    control, which the method the route runs validates.

    Parameters
    ----------
    name : str
        The route's name, by which a ``method`` control selects it.
    exact : bool or None
        Whether the result denotes the requested mathematical object; ``None``
        for a route whose exactness is that of the implementation it delegates
        to.

    Raises
    ------
    TypeError
        If *name* is not a non-empty string, or *exact* is neither a bool nor
        ``None``.
    """

    source: RouteSource
    requires: tuple[type, ...] = ()
    #: Whether the route realizes a derived operation's formula, which ranks it
    #: below the operation's direct routes of its exactness (VI.0).
    realizes_formula: bool = False

    def __init__(self, name: str, *, exact: bool | None) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError(f"a route's name must be a non-empty string; got {name!r}")
        if exact is not None and type(exact) is not bool:
            raise TypeError(f"route {name!r} must declare exact as a bool or None; got {exact!r}")
        self.name = name
        self.exact = exact

    @property
    def condition(self) -> str:
        """The feasibility condition in words."""
        return ""

    @staticmethod
    def method_options(call: BoundCall) -> dict[str, Any]:
        """The budgets *call* sets through its ``method_options`` control, by name."""
        return dict(call.controls.get("method_options", {}))

    def __repr__(self) -> str:
        """The route's class and name."""
        return f"{type(self).__name__}({self.name!r})"


class _CheckedRoute(_Route):
    """A route of any source given as a check and an execute function over the bound call.

    Parameters
    ----------
    name : str
        The route's name, by which a ``method`` control selects it.
    source : RouteSource
        Where the implementation comes from.
    check : callable
        ``check(call, result)``, which returns a bool, ``None``, or a
        ``Feasibility``; the first paragraph of its docstring is the route's
        condition.
    execute : callable
        ``execute(call, result)``, which returns the raw result.
    exact : bool or None
        Whether the result denotes the requested mathematical object; ``None``
        for a route whose exactness is that of the implementations it runs.

    Raises
    ------
    TypeError
        If *source* is not a RouteSource, or *check* or *execute* is not callable.
    """

    def __init__(
        self,
        name: str,
        *,
        source: RouteSource,
        check: Callable[[BoundCall, OutputSpec | None], Any],
        execute: Callable[[BoundCall, OutputSpec | None], Any],
        exact: bool | None,
    ) -> None:
        super().__init__(name, exact=exact)
        if not isinstance(source, RouteSource):
            raise TypeError(f"route {name!r} needs a RouteSource; got {source!r}")
        if not callable(check) or not callable(execute):
            raise TypeError(f"route {name!r} needs a callable check and a callable execute")
        self.source = source
        self._check = check
        self._execute = execute

    @property
    def condition(self) -> str:
        """The first paragraph of the check's docstring."""
        return _guard_condition(self._check)

    def check(self, call: BoundCall, result: OutputSpec | None) -> Feasibility:
        """The check function's report, a bool or ``None`` converted to a Feasibility."""
        return _as_feasibility(self._check(call, result), self._check, f"route {self.name!r}")

    def execute(self, call: BoundCall, result: OutputSpec | None) -> Any:
        """The execute function's result."""
        return self._execute(call, result)


class _CapabilityRoute(_Route):
    """A route that calls a capability of the operand it names.

    The route is feasible when the operand claims the protocol and the
    capability's guard admits the call. The guard is read by
    :func:`~probpipe.distributions._capabilities._capability_guard` with no
    arguments, or by a given *check*, which reads the call's arguments that
    the guard takes. The capability is called with the call's other arguments
    in signature order, or by a given *execute*. An approximate capability also
    receives the call's ``method_options`` as keyword options, while an exact
    one computes its object with no budget and receives none.

    Parameters
    ----------
    name : str
        The route's name, by which a ``method`` control selects it.
    operand : str
        The parameter whose argument claims the capability.
    protocol : type
        The capability protocol, such as ``SupportsMean``.
    method : str
        The capability's implementation method, such as ``"_mean"``.
    exact : bool
        Whether the capability returns the requested mathematical object.
    check : callable, optional
        ``check(call, result)``, which reads the guard with the call's arguments
        it takes, as a marginal's guard takes the path.
    execute : callable, optional
        ``execute(call, result)``, which calls the capability in place of the
        call in signature order.

    Raises
    ------
    TypeError
        If *operand* or *method* is not a string, *protocol* is not a class, or
        *check* or *execute* is given and not callable.
    """

    source = RouteSource.CAPABILITY

    def __init__(
        self,
        name: str,
        *,
        operand: str,
        protocol: type,
        method: str,
        exact: bool,
        check: Callable[[BoundCall, OutputSpec | None], Any] | None = None,
        execute: Callable[[BoundCall, OutputSpec | None], Any] | None = None,
    ) -> None:
        super().__init__(name, exact=exact)
        if not isinstance(operand, str) or not isinstance(method, str):
            raise TypeError(f"route {name!r} names its operand and method by strings")
        if not isinstance(protocol, type):
            raise TypeError(f"route {name!r} needs a protocol class; got {protocol!r}")
        for supplied in (check, execute):
            if supplied is not None and not callable(supplied):
                raise TypeError(f"route {name!r} was given a check or execute that is not callable")
        self.operand = operand
        self.protocol = protocol
        self.method = method
        self._check = check
        self._execute = execute

    @property
    def requires(self) -> tuple[type, ...]:  # type: ignore[override]
        """The protocol the operand must claim."""
        return (self.protocol,)

    @property
    def condition(self) -> str:
        """The route's condition for an operand of the protocol's own class."""
        return self.condition_for(self.protocol)

    def condition_for(self, cls: type) -> str:
        """The feasibility condition for an operand of class *cls*, in words.

        It is the given check's condition, or else the condition of the guard
        *cls* defines for the capability, or else a sentence saying that
        membership suffices.
        """
        if self._check is not None:
            return _guard_condition(self._check)
        guard = getattr(cls, f"{self.method}_guard", None)
        if guard is not None:
            return _guard_condition(guard)
        return f"membership in {self.protocol.__name__} suffices"

    def check(self, call: BoundCall, result: OutputSpec | None) -> Feasibility:
        """Protocol membership of the named operand, then the capability's guard."""
        subject = call.operands[self.operand]
        if not isinstance(subject, self.protocol):
            return Feasibility(
                False, f"{_subject_name(subject)} does not implement {self.protocol.__name__}"
            )
        if self._check is not None:
            return _as_feasibility(self._check(call, result), self._check, f"route {self.name!r}")
        return _capability_guard(subject, self.method)

    def execute(self, call: BoundCall, result: OutputSpec | None) -> Any:
        """The capability called on the call's other arguments, and on its method options
        when it is approximate.

        A given *execute* replaces the call.
        """
        if self._execute is not None:
            return self._execute(call, result)
        subject = call.operands[self.operand]
        others = [value for name, value in call.operands.items() if name != self.operand]
        capability_options = {} if self.exact else self.method_options(call)
        return getattr(subject, self.method)(*others, **capability_options)


class _DelegatingRoute(_Route, ABC):
    """A route whose exactness is that of the implementations it delegates to.

    The route is ranked twice: once restricted to exact implementations, among
    the exact routes, and once under the call's own ``exact_only``, among the
    approximate routes, so ``exact_only`` keeps only the first. A subclass
    declares its ``source`` and implements :meth:`probe` and :meth:`run`, which
    receive the restriction of the candidate being ranked.

    Parameters
    ----------
    name : str
        The route's name, by which a ``method`` control selects it.

    Raises
    ------
    TypeError
        If *name* is not a non-empty string.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name, exact=None)

    @abstractmethod
    def probe(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Feasibility:
        """The route's report for *call*, restricted to exact implementations under *exact_only*.

        A feasible report states the exactness of what it selected, as a
        ``MethodInfo`` or a ``PointReport`` does. *result* is the output spec of
        the call's result, as :meth:`check` receives it, and *method* names a method
        of a registry route's registry, ``None`` for any other route.
        """

    @abstractmethod
    def run(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Any:
        """The route's raw result for *call*, under the same restriction as its probe.

        As for :meth:`OperationRoute.execute`, the route may instead return the
        declared term with a provenance record of how it produced it.
        """

    def check(self, call: BoundCall, result: OutputSpec | None) -> Feasibility:
        """The route's report under the call's ``exact_only``."""
        return self.probe(call, result, method=None, exact_only=call.controls["exact_only"])

    def execute(self, call: BoundCall, result: OutputSpec | None) -> Any:
        """The route's result under the call's ``exact_only``."""
        return self.run(call, result, method=None, exact_only=call.controls["exact_only"])


class _RegistryRoute(_DelegatingRoute):
    """A route that delegates to a dispatch registry, whose selected method realizes the call.

    The registry receives the call's arguments in signature order, or those
    *arguments* returns, with the keyword options *options* returns, or else the
    call's ``method_options``. Its exactness is that of the method the registry
    selects, so the route is ranked twice, as every delegating route is: its
    exact methods with the exact routes and its approximate methods with the
    approximate ones.

    Parameters
    ----------
    name : str
        The route's name, by which a ``method`` control selects it.
    registry : BaseDispatchRegistry
        The registry whose selected method realizes the call.
    arguments : callable, optional
        ``arguments(call)``, which returns the registry's positional arguments.
    options : callable, optional
        ``options(call)``, which returns the keyword options of the registry's
        methods.

    Raises
    ------
    TypeError
        If *registry* is not a dispatch registry, or *arguments* or *options* is
        given and not callable.
    """

    source = RouteSource.REGISTRY

    def __init__(
        self,
        name: str,
        *,
        registry: BaseDispatchRegistry[Any],
        arguments: Callable[[BoundCall], Iterable[Any]] | None = None,
        options: Callable[[BoundCall], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(name)
        if not isinstance(registry, BaseDispatchRegistry):
            raise TypeError(f"route {name!r} needs a dispatch registry; got {registry!r}")
        for supplied in (arguments, options):
            if supplied is not None and not callable(supplied):
                raise TypeError(
                    f"route {name!r} was given arguments or options that are not callable"
                )
        self.registry = registry
        self._arguments = arguments
        self._options = options

    @property
    def condition(self) -> str:
        """The registry's methods, in its selection order."""
        methods = ", ".join(self.registry.list_methods()) or "none are registered"
        return f"the registry selects one of its methods: {methods}"

    def arguments(self, call: BoundCall) -> tuple[Any, ...]:
        """The positional arguments the registry dispatches on."""
        if self._arguments is not None:
            return tuple(self._arguments(call))
        return tuple(call.operands.values())

    def options(self, call: BoundCall) -> dict[str, Any]:
        """The keyword options passed to the registry's methods."""
        if self._options is not None:
            return dict(self._options(call))
        return self.method_options(call)

    def probe(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> MethodInfo:
        """The registry's report for *call*, restricted as the controls ask."""
        return self.registry.check(
            *self.arguments(call), method=method, exact_only=exact_only, **self.options(call)
        )

    def run(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Any:
        """The result of the registry's selected method, or of *method*."""
        return self.registry.execute(
            *self.arguments(call), method=method, exact_only=exact_only, **self.options(call)
        )


def _derived_route_scope(call: BoundCall) -> AbstractContextManager[None]:
    """The scope in which a derived route's check and run make their Function calls.

    It passes the derived call's ``exact_only``, and its ``max_evaluations`` and
    ``num_samples`` when the call's view sets them, as the module docstring
    states.
    """
    operation_view = call.operation

    def passed_count(name: str) -> int | None:
        return call.controls[name] if name in operation_view._options else None

    return _passing_controls_to_formula_calls(
        exact_only=call.controls["exact_only"],
        max_evaluations=passed_count("max_evaluations"),
        num_samples=passed_count("num_samples"),
    )


class _DerivedRoute(_Route):
    """The body route, which runs a derived operation's body as a fallback on its own domain.

    The route's check calls *derived_route_check* with the call's arguments, as the
    formula is called, so the route is feasible where the constituent
    operations have routes, and its condition is the derived route check's. Its
    exactness is that of the routes the constituents select, so it declares
    none. The engine runs the check and the body in the scope of a derived route
    and ranks the route as the module docstring states.

    Parameters
    ----------
    declaration : callable
        The derived operation's declaration, whose body states the formula.
    signature : inspect.Signature
        The operation's signature, to which the call's arguments bind.
    derived_route_check : callable or None
        The probe of the formula, called as the formula is; the first
        paragraph of its docstring is the route's condition. Without one, the
        route admits every call that planning admits.

    Notes
    -----
    Interim implementation detail: without ``exact_only`` the call records the
    exactness that the derived route check reports, which a check that probes part
    of the body can overstate. VI.0's target records the exactness of the
    routes that ran.
    """

    source = RouteSource.FALLBACK
    realizes_formula = True

    def __init__(
        self,
        declaration: Callable[..., Any],
        signature: inspect.Signature,
        derived_route_check: Callable[..., Any] | None,
    ) -> None:
        super().__init__("derived", exact=None)
        self._declaration = declaration
        self._signature = signature
        self.derived_route_check = derived_route_check

    @property
    def condition(self) -> str:
        """The first paragraph of the derived route check's docstring."""
        if self.derived_route_check is None:
            return "The constituent operations decide the call when the formula runs."
        return _guard_condition(self.derived_route_check)

    def check(self, call: BoundCall, result: OutputSpec | None) -> Feasibility:
        """The derived route check's report, or a feasible one for a route without a check."""
        if self.derived_route_check is None:
            return Feasibility(True)
        bound = values_to_bound_arguments(self._signature, call.operands)
        report = self.derived_route_check(*bound.args, **bound.kwargs)
        return _as_feasibility(report, self.derived_route_check, f"route {self.name!r}")

    def execute(self, call: BoundCall, result: OutputSpec | None) -> Any:
        """The formula's value."""
        bound = values_to_bound_arguments(self._signature, call.operands)
        return self._declaration(*bound.args, **bound.kwargs)


#: The instructions that load a module-level name, and those that load a closure's.
_GLOBAL_LOADS = frozenset({"LOAD_GLOBAL", "LOAD_NAME"})
_CLOSURE_LOADS = frozenset({"LOAD_DEREF"})


def _named_operations(declaration: Callable[..., Any]) -> tuple[Operation, ...]:
    """The operations *declaration*'s body loads by name, each once.

    A loaded name counts when it is bound to an operation, or to a view of one,
    in the declaration's module or in its closure, as ``variance`` is in
    ``jnp.sqrt(variance.using(raw=True)(d))``. The functions nested in
    the body count too. An attribute that shares an operation's name, as
    ``mean`` in ``law.mean``, is not a load of it.
    """
    code = getattr(declaration, "__code__", None)
    if code is None:
        return ()
    scope = getattr(declaration, "__globals__", {})
    closure: dict[str, Any] = {}
    for name, cell in zip(code.co_freevars, declaration.__closure__ or (), strict=False):
        try:
            closure[name] = cell.cell_contents
        except ValueError:  # a cell not yet filled
            continue
    found: dict[int, Operation] = {}
    pending = [code]
    while pending:
        current_code = pending.pop()
        for instruction in dis.get_instructions(current_code):
            if instruction.opname in _GLOBAL_LOADS:
                value = scope.get(instruction.argval)
            elif instruction.opname in _CLOSURE_LOADS:
                value = closure.get(instruction.argval)
            else:
                continue
            if isinstance(value, Operation):
                found.setdefault(id(value), value)
        pending += [const for const in current_code.co_consts if isinstance(const, type(code))]
    return tuple(found.values())


def _unchecked(route: Any) -> bool:
    """Whether *route* is a derived route without a check, which ``exact_only`` excludes."""
    return isinstance(route, _DerivedRoute) and route.derived_route_check is None


# ---------------------------------------------------------------------------
# The candidates of selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """One way a route may realize a call: the route at one exactness.

    A delegating route contributes one candidate restricted to its exact
    implementations and one for the others; for a registry route those are its
    exact and its approximate methods, and ``method`` is the registry method a
    ``method=`` control names, whose exactness the registry reports. A candidate is what the
    engine's resolution step probes and runs, through the members
    :mod:`probpipe.functions._resolution` names.
    """

    route: Any
    exact: bool | None
    index: int
    method: str | None = None

    @property
    def rank(self) -> tuple[int, int, int, int]:
        """A fallback below every other source whatever its exactness, then exact before
        approximate, then a direct route before the derived route, then registration order."""
        return (
            1 if self.route.source is RouteSource.FALLBACK else 0,
            0 if self.exact is True else 1,
            1 if self.route.realizes_formula else 0,
            self.index,
        )

    @property
    def label(self) -> str:
        """The route's name, qualified for a delegating route by the exactness the candidate covers.

        A registry route's candidate names its method, as ``route/method``, or
        its exact or approximate methods; another delegating route's names its
        exact or approximate implementations.
        """
        if not isinstance(self.route, _DelegatingRoute):
            return self.route.name
        if self.method is not None:
            return f"{self.route.name}/{self.method}"
        if self.route.realizes_formula:
            return self.route.name
        exactness = "exact" if self.exact else "approximate"
        if isinstance(self.route, _RegistryRoute):
            return f"{self.route.name} ({exactness} methods)"
        return f"{self.route.name} ({exactness})"

    @property
    def route_name(self) -> str:
        """The name of the route the candidate belongs to."""
        return self.route.name

    @property
    def collects_calls(self) -> bool:
        """Whether the calls the route makes directly report their Monte Carlo draws to the call.

        A derived route collects them, so its call states the
        draws of the constituents it runs.
        """
        return self.route.realizes_formula

    @property
    def methods(self) -> tuple[str, ...] | None:
        """The methods of a registry route's registry, in its selection order, or ``None``."""
        if not isinstance(self.route, _RegistryRoute):
            return None
        return tuple(self.route.registry.list_methods())

    def probe(self, call: BoundCall, result: OutputSpec | None) -> Feasibility:
        """The candidate's report for *call*, without executing anything.

        A derived route check runs in the scope :func:`_derived_route_scope` builds.
        Under ``exact_only`` a body route without a derived route check declines,
        and so does a route whose check reports an approximate result.
        """
        route = self.route
        exact_only = call.controls["exact_only"]
        if exact_only and _unchecked(route):
            return Feasibility(
                False, "the operation declares no derived_route_check, which exact_only requires"
            )
        with self._scope(call):
            if isinstance(route, _DelegatingRoute):
                report = route.probe(
                    call, result, method=self.method, exact_only=exact_only or self.exact is True
                )
            else:
                report = route.check(call, result)
        if exact_only and report.feasible is True and self.exactness(report) is False:
            return Feasibility(
                False, "its check reports an approximate result, but exact_only=True"
            )
        return report

    def run(self, call: BoundCall, result: OutputSpec | None, report: Feasibility) -> Any:
        """The route's raw result for *call*, by the method *report* selected for a registry route.

        A derived route runs in the scope :func:`_derived_route_scope` builds.
        """
        route = self.route
        with self._scope(call):
            if isinstance(route, _DelegatingRoute):
                selected = self.method_of(report)
                return route.run(
                    call,
                    result,
                    method=self.method if selected is None else selected,
                    exact_only=call.controls["exact_only"] or self.exact is True,
                )
            return route.execute(call, result)

    def _scope(self, call: BoundCall) -> AbstractContextManager[None]:
        """The scope that a derived route's check and run take, or no scope for any other route."""
        return _derived_route_scope(call) if self.route.realizes_formula else nullcontext()

    def exactness(self, report: Feasibility) -> bool | None:
        """The exactness of the implementation the candidate selects, as *report* gives it.

        A route that delegates its exactness reports that of what its probe
        selected: a registry route's method, the routes a composing route
        resolves, or the route a derived operation's constituent selects. Any
        other report reads at the exactness of the methods the candidate covers.
        """
        if self.route.exact is not None:
            return self.route.exact
        if isinstance(report, (MethodInfo, PointReport)) and report.exact is not None:
            return report.exact
        return self.exact

    def method_of(self, report: Feasibility) -> str | None:
        """The registry method a registry route delegates to, as *report* names it."""
        if not isinstance(self.route, _RegistryRoute):
            return None
        if isinstance(report, MethodInfo):
            return report.method_name
        if isinstance(report, PointReport):
            return report.method
        return None


def _route_candidates(route: Any, index: int, exact_only: bool) -> list[_Candidate]:
    """The candidates *route*, registered at *index*, contributes to a call under *exact_only*.

    A derived route contributes one, at the rank the module docstring states. A
    delegating route contributes two, restricted to its exact implementations
    and not. Any other route contributes one at its declared exactness.
    """
    if route.realizes_formula:
        return [_Candidate(route, True if exact_only and not _unchecked(route) else None, index)]
    if isinstance(route, _DelegatingRoute):
        return [_Candidate(route, True, index), _Candidate(route, False, index)]
    return [_Candidate(route, route.exact, index)]


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


class _RouteTable:
    """The routes of one operation, shared by every view of it, in registration order."""

    def __init__(self) -> None:
        self.routes: list[Any] = []

    def add(self, route: Any, owner: str) -> None:
        """Append *route*.

        Parameters
        ----------
        route : OperationRoute
            An object with the members of :class:`OperationRoute`, whose
            ``source`` is a RouteSource.
        owner : str
            The operation named in the error that a duplicate name raises, such
            as ``operation 'mean'``.

        Raises
        ------
        TypeError
            If *route* does not have the members of :class:`OperationRoute`.
        ValueError
            If a route of that name is registered.
        """
        if not isinstance(route, OperationRoute) or not isinstance(route.source, RouteSource):
            raise TypeError(f"{route!r} is not an OperationRoute")
        if any(existing.name == route.name for existing in self.routes):
            raise ValueError(f"{owner} already has a route named {route.name!r}")
        self.routes.append(route)


def _parameter_names(function: Callable[..., Any]) -> tuple[str, ...]:
    """The parameter names of *function*, in order."""
    return tuple(inspect.signature(function).parameters)


def _has_empty_body(function: Callable[..., Any]) -> bool:
    """Whether *function*'s body only returns ``None``, as a docstring and ``...`` do."""
    code = getattr(function, "__code__", None)
    if code is None:
        return False
    for instruction in dis.get_instructions(code):
        if instruction.opname in ("RESUME", "NOP", "CACHE", "NOT_TAKEN", "RETURN_VALUE"):
            continue
        if instruction.opname in ("LOAD_CONST", "RETURN_CONST") and instruction.argval is None:
            continue
        return False
    return True


def _formula_source(function: Callable[..., Any]) -> str | None:
    """The expression a derived declaration returns, as source, or ``None`` without source."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        return None
    returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return) and node.value]
    return ast.unparse(returns[-1].value) if returns else None


def _kinds_for(hint: Any) -> tuple[type[TermSpec], ...]:
    """The kinds a parameter annotated *hint* accepts, each named by its spec class.

    An unannotated parameter accepts the value kinds, so a law there lifts and a
    batch of values is swept; one annotated ``Any`` accepts every kind as it
    arrives; a term class accepts its own kind; and any other annotation, such
    as ``str`` for a field path, selects rather than supplies and accepts none.
    A type alias accepts the kinds of the annotation :func:`resolved_hint`
    resolves it to.
    """
    if hint is inspect.Parameter.empty:
        return _VALUE_KINDS
    hint = resolved_hint(hint)
    if hint is Any:
        return (TermSpec,)
    if not isinstance(hint, type):
        return ()
    for spec_type, (term_class, _) in _KINDS.items():
        if term_class is not None and issubclass(hint, term_class):
            return (spec_type,)
    for term_class, spec_type in _UNREGISTERED_TERM_KINDS:
        if issubclass(hint, term_class):
            return (spec_type,)
    return ()


def _validated_roles(
    signature: inspect.Signature,
    hints: Mapping[str, Any],
    roles: Mapping[str, Iterable[type[TermSpec]]] | None,
    owner: str,
) -> dict[str, tuple[type[TermSpec], ...]]:
    """Each parameter's accepted kinds: the declared role, or else the annotation's kinds.

    Parameters
    ----------
    signature : inspect.Signature
        The operation's signature, whose every parameter receives an entry.
    hints : Mapping of str to Any
        The evaluated annotations by parameter name; an unannotated parameter
        has no entry.
    roles : Mapping of str to iterable of TermSpec subclasses, or None
        The declared roles by parameter name, or ``None`` for none.
    owner : str
        The operation that an error names, such as ``operation 'mean'``.

    Returns
    -------
    dict of str to tuple of type
        The kinds by parameter name, in signature order, each kind named by its
        spec class; an empty tuple marks a parameter that selects rather than
        supplies.

    Raises
    ------
    TypeError
        If a role names a parameter the signature lacks, or lists something
        other than a TermSpec subclass.
    """
    derived = {
        name: _kinds_for(hints.get(name, inspect.Parameter.empty)) for name in signature.parameters
    }
    for name, kinds in (roles or {}).items():
        if name not in signature.parameters:
            raise TypeError(f"{owner} declares a role for {name!r}, which is not a parameter")
        accepted = tuple(kinds)
        if not all(isinstance(kind, type) and issubclass(kind, TermSpec) for kind in accepted):
            raise TypeError(
                f"{owner}: the role of parameter {name!r} must list TermSpec subclasses"
            )
        derived[name] = accepted
    return derived


def _kind_names(kinds: Iterable[type]) -> str:
    """The names of *kinds*, comma-separated."""
    return ", ".join(kind.__name__ for kind in kinds)


class _GivenCandidate:
    """The conditional route, deferred until the kernel's law is available.

    Feasibility means the inputs can be bound. Normalization, the resulting
    law's capability checks, and operation fidelity are deferred, and its exactness is unknown
    until the delegated operation executes. The kernel is never applied by a
    check. This candidate is the last fallback, after every route served by the
    kernel's declared conditional capabilities. A declaration-only given
    leaves conditioning feasibility unresolved until its values are available.
    """

    label = route_name = "conditional"
    methods = None
    #: The conditioned call runs the operation, whose Monte Carlo draws the call states.
    collects_calls = True

    def probe(self, call: BoundCall, result: OutputSpec | None) -> Feasibility:
        """Check input binding without constructing or normalizing the kernel's law."""
        from ._given import _GivenLaw

        subject = call.operands[call.operation._given_subject]
        if not isinstance(subject, _GivenLaw):
            return Feasibility(False, "the operand has no given values")
        if isinstance(subject.given, StandIn):
            return Feasibility(None, pending=("conditional delegation needs the given values",))
        from ._condition import _BIND_INPUTS, condition_on

        binding = BoundCall(
            condition_on,
            MappingProxyType({"d": subject.kernel, "given": subject.given}),
            call.controls,
        )
        report = _BIND_INPUTS.check(binding)
        if report.feasible is not True:
            return report
        exact = _BIND_INPUTS.exact(binding)
        if call.controls["exact_only"] and not exact:
            return Feasibility(
                False,
                "binding the inputs runs approximate conditioning "
                "(SupportsApproximateConditioning), which exact_only excludes",
            )
        return PointReport(
            True,
            "delegation is available; the conditioned law's operation capabilities and fidelity are checked at execution",
            exact=None if exact else False,
        )

    def run(self, call: BoundCall, result: OutputSpec | None, report: Feasibility) -> Any:
        """Condition at execution, then run the original operation under its controls."""
        from ._condition import condition_on

        subject = call.operands[call.operation._given_subject]
        law = condition_on.using(
            exact_only=call.controls["exact_only"],
            method_options=call.controls["method_options"],
        )(subject.kernel, subject.given)
        values = {**call.operands, call.operation._given_subject: law}
        bound = values_to_bound_arguments(call.operation.signature, values)
        return call.operation.using(**dict(call.controls))(*bound.args, **bound.kwargs)

    def exactness(self, report: Feasibility) -> bool | None:
        """Known approximate conditioning is approximate; otherwise fidelity is deferred."""
        return report.exact if isinstance(report, PointReport) else None

    def method_of(self, report: Feasibility) -> None:
        """The delegated operation selects its own method."""
        return None


class Operation(Function):
    """A Function realized by routes, with operand roles, applicability checks, and a result rule.

    Calling it runs the engine's stack, whose admission, planning, resolution,
    and return steps read the declarations below: each argument is admitted and
    lifted by its role, the applicability checks and the result rule derive
    the output spec of each point's result, the engine selects a route among the
    operation's candidates, and the return wraps the route's raw result at the
    declared kind. A parameter annotated ``Any`` takes its argument as it
    arrives, with no lifting, so its role only narrows what it admits. The
    routes are registered after construction with the four helpers or
    :meth:`register_route`, and a derived operation also carries its
    derived route.

    A subject annotated ``Distribution`` automatically adds keyword-only
    ``given=None`` unless the declaration already names ``given`` or a role
    overrides the subject's kind. A complete given supplies a conditional
    operand's required inputs and binds its symbolic dimensions before planning
    the result. Routes use the kernel's declared conditional twins, including
    sampling-based fallbacks and derived routes. If no such route applies,
    delegation applies the operation to ``condition_on(subject, given)`` at execution.
    A check never constructs that law. The delegation report defers the law's
    capabilities and fidelity, conservatively reporting open fidelity as
    approximate. Invalid or incomplete givens raise ``ApplicabilityError``;
    unavailable capabilities or fidelity controls raise ``ResolutionError``.
    Record-batch givens sweep at their named levels. Calls without a given
    keep their ordinary route behavior and require an unconditional subject
    unless the authored role explicitly admits kernels.

    Parameters
    ----------
    declaration : callable
        The authored signature: the operands and the parameters the result rule
        reads. An empty body declares a primitive operation, and a body that
        returns an expression states a derived operation's formula.
    result : callable
        The result rule. Its parameters are parameters of *declaration*; it
        receives each supplying operand's spec and each selecting parameter's
        value, and returns the result's ``OutputSpec``, or ``None`` when the
        operands' specs leave the result open.
    applicability_checks : iterable of callable
        Checks that the operation applies to a call at all, whatever route
        serves it. Each runs before any route is chosen and before the result
        rule, and is called like the result rule, with the specs of the
        parameters it names and never their values, except that a parameter
        whose role names ``FunctionSpec`` alone receives the map itself, whose
        annotations, roles, and ``input_spec`` are declared static information
        (V.6). Each returns ``True``, ``False``, ``None`` when the answer
        depends on values not yet known, or a ``Feasibility``; the first
        paragraph of its docstring states what it checks. A check that fails
        raises ``ApplicabilityError`` from the call and from ``check``, so no
        route runs, and one that returns ``None`` is made at the return.
    roles : mapping of str to iterable of TermSpec subclasses, optional
        The kinds a parameter accepts, overriding the kinds its annotation names.
        An empty role marks a parameter that selects rather than supplies.
    derived_route_check : callable, optional
        The derived route's own feasibility check, for a derived operation,
        whose body returns its formula. Unlike an applicability check, it is
        called with the call's argument values, as the formula is, and decides
        only whether the derived route can serve the call: when it reports the
        call infeasible, the engine tries the operation's other routes. It
        returns the report of the constituent operations' checks, such as
        ``log_prob.check(d, value)`` for ``prob``; the first paragraph of its
        docstring is the derived route's condition. A check that cannot
        probe a constituent, because
        its arguments are known only when the formula runs, returns a feasible
        ``PointReport`` whose ``deferred`` names that check. The check runs in
        the scope of the derived route (VI.0), so under ``exact_only`` it reports
        whether the constituents it probes have exact routes, and
        ``exact_only`` keeps the derived route of an operation with a check.
        A constituent's check selects the constituent's route without running
        it, so a call inside that route that the restriction leaves without a
        route raises ``ResolutionError`` only when the formula runs. Without a
        check, the derived route admits every call that planning admits.
    derived_route : OperationRoute, optional
        A derived operation's composite derived route, which replaces the
        generic route that runs the body. It runs a route of each constituent
        operation in turn and records their combined exactness, so it is ranked
        once, as a derived route, below the direct routes of its exactness.
    label : callable, optional
        The label rule, called with the call's arguments it names and
        returning the result's label. Without it, the result takes the label
        and the expression of the primary operand, the first parameter's
        argument (II.4).

    Raises
    ------
    TypeError
        If *declaration*, *result*, an applicability check,
        *derived_route_check*, or *label* is not callable, the result rule, an
        applicability check, or the label rule reads
        a name the declaration does not declare, a role is malformed, a
        primitive operation is given a derived route check or a derived route,
        a derived route is given beside a derived route check, or it is not a
        delegating route.
    """

    _options_need_named_executor: ClassVar[bool] = True

    def __init__(
        self,
        declaration: Callable[..., Any],
        *,
        result: Callable[..., OutputSpec | None],
        applicability_checks: Iterable[Callable[..., Any]] = (),
        roles: Mapping[str, Iterable[type[TermSpec]]] | None = None,
        derived_route_check: Callable[..., Any] | None = None,
        derived_route: OperationRoute | None = None,
        label: Callable[..., str] | None = None,
    ) -> None:
        if not callable(declaration):
            raise TypeError(f"an operation is declared by a function; got {declaration!r}")
        if roles is not None:
            roles = {name: tuple(kinds) for name, kinds in roles.items()}
        super().__init__(declaration, label=declaration.__name__)
        object.__setattr__(self, "_declaration_name", declaration.__name__)
        first = next(iter(self.signature.parameters), None)
        subject = (
            first if resolved_hint(self._signature_info.hints.get(first)) is Distribution else None
        )
        if "given" in self.signature.parameters or (
            roles is not None and first in roles and DistributionSpec not in roles[first]
        ):
            subject = None
        object.__setattr__(self, "_given_subject", subject)
        from ..distributions._conditional import ConditionalDistributionSpec

        object.__setattr__(
            self, "_admits_kernel", ConditionalDistributionSpec in (roles or {}).get(first, ())
        )
        if subject is not None:
            parameter = inspect.Parameter(
                "given", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Record
            )
            parameters = list(self.signature.parameters.values())
            index = next(
                (
                    i
                    for i, item in enumerate(parameters)
                    if item.kind is inspect.Parameter.VAR_KEYWORD
                ),
                len(parameters),
            )
            parameters.insert(index, parameter)
            signature = self.signature.replace(parameters=parameters)
            signature_info = make_signature_info_from_signature(
                signature, hints=self._signature_info.hints
            )
            object.__setattr__(self, "_signature_info", signature_info)
            object.__setattr__(self, "__signature__", signature)
            given_doc = (
                "    given : Record or Mapping[str, Any], optional\n"
                "        Complete values of a conditional operand's required inputs. The\n"
                "        declared conditional capabilities serve applicable routes; otherwise\n"
                "        the operation applies after condition_on at execution, and check\n"
                "        defers the resulting law's capabilities and fidelity. A RecordBatch\n"
                "        is swept at its named levels. None keeps the authored subject role.\n\n"
            )
            object.__setattr__(
                self,
                "__doc__",
                (self.__doc__ or "").replace("    Returns\n", given_doc + "    Returns\n", 1),
            )
        owner = f"operation {self.label!r}"
        if not callable(result):
            raise TypeError(f"{owner} needs a callable result rule; got {result!r}")
        applicability_checks = tuple(applicability_checks)
        if not all(callable(applicability_check) for applicability_check in applicability_checks):
            raise TypeError(f"{owner} was given an applicability check that is not callable")
        if label is not None and not callable(label):
            raise TypeError(f"{owner} needs a callable label rule; got {label!r}")
        parameters = self.signature.parameters
        read = {
            parameter
            for rule in (result, *applicability_checks)
            for parameter in _parameter_names(rule)
        }
        unknown = (read | set(_parameter_names(label) if label else ())) - set(parameters)
        if unknown:
            raise TypeError(
                f"{owner}: the result rule, an applicability check, or the label rule reads "
                f"{sorted(unknown)}, which the declaration does not declare as parameters"
            )
        derived = not _has_empty_body(declaration)
        if derived_route_check is not None:
            if not derived:
                raise TypeError(
                    f"{owner} has an empty body, so it cannot take derived_route_check; only an "
                    f"operation whose body returns its formula can"
                )
            if not callable(derived_route_check):
                raise TypeError(
                    f"{owner} needs a callable derived route check; got {derived_route_check!r}"
                )
        if derived_route is not None:
            if not derived:
                raise TypeError(
                    f"{owner} has an empty body, so it cannot take derived_route; only an "
                    f"operation whose body returns its formula can"
                )
            if derived_route_check is not None:
                raise TypeError(
                    f"{owner} cannot take both derived_route and derived_route_check; the "
                    f"derived route checks its own feasibility"
                )
            if not isinstance(derived_route, _DelegatingRoute):
                raise TypeError(
                    f"{owner} needs a derived route that delegates its exactness to the "
                    f"routes it runs; got {derived_route!r}"
                )
        set_attribute = object.__setattr__
        set_attribute(self, "_result_rule", result)
        set_attribute(self, "_rule_parameters", _parameter_names(result))
        set_attribute(self, "_applicability_checks", applicability_checks)
        set_attribute(
            self, "_declared_parameters", tuple(name for name in parameters if name in read)
        )
        set_attribute(
            self,
            "_roles",
            MappingProxyType(
                _validated_roles(self.signature, self._signature_info.hints, roles, owner)
            ),
        )
        if subject is not None:
            from ..distributions._conditional import ConditionalDistributionSpec

            adapted_roles = dict(self._roles)
            adapted_roles[subject] = (*adapted_roles[subject], ConditionalDistributionSpec)
            adapted_roles["given"] = (RecordSpec,)
            set_attribute(self, "_roles", MappingProxyType(adapted_roles))
        set_attribute(self, "_derived", derived)
        set_attribute(self, "_formula", _formula_source(declaration) if derived else None)
        set_attribute(self, "_body", declaration if derived else None)
        set_attribute(self, "_route_table", _RouteTable())
        set_attribute(self, "_label_rule", label)
        set_attribute(self, "_expression_rule", None)
        if derived_route is not None:
            # A copy carries the derived route's rank, so the caller's route keeps its own.
            derived_route = copy.copy(derived_route)
            derived_route.realizes_formula = True
            self.register_route(derived_route)
        elif derived:
            self.register_route(_DerivedRoute(declaration, self.signature, derived_route_check))

    # -- formula -----------------------------------------------------------

    @property
    def name(self) -> str:
        """The operation's key in :data:`operation_registry`, the name of its declaration."""
        return self._declaration_name

    # -- declarations ------------------------------------------------------

    def _derived_expression(self, values: Mapping[str, Any]) -> Expression | None:
        """The expression of the result of a call on *values*, the bound arguments by parameter name.

        The operation's private expression rule builds it from its operands'
        expressions where the operation has one, and
        :func:`_install_expression_rule` installs it; a rule that returns
        ``None`` leaves the result the expression its route gave it. An
        operation with a label rule gives its result that label alone.
        Otherwise the result takes the expression of the primary operand, the
        first parameter's argument, and an argument that is not a tracked term
        leaves the operation's own output label.

        Parameters
        ----------
        values : Mapping of str to Any
            The call's bound arguments, by parameter name.

        Returns
        -------
        Expression or None
            The result's expression, or ``None`` when the route's result keeps
            its own.
        """
        if self._given_subject is not None and values.get("given") is not None:
            from ._condition import _conditioned_paths

            subject = values[self._given_subject]
            view = Distribution(subject.event_spec, label=subject.label)
            view._store_expression(
                subject._embedded_expression().with_fixed(_conditioned_paths(values["given"]) or ())
            )
            values = {**values, self._given_subject: view}
        rule = self._expression_rule
        if rule is not None:
            return rule(**{name: values.get(name) for name in _parameter_names(rule)})
        rule = self._label_rule
        if rule is not None:
            return Named(rule(**{name: values.get(name) for name in _parameter_names(rule)}))
        primary = values.get(next(iter(self.signature.parameters), ""))
        return (
            primary._embedded_expression()
            if isinstance(primary, TrackedTerm)
            else Named(self.output_label)
        )

    def _derived_label(self, values: Mapping[str, Any]) -> str:
        """The label of the result of a call on *values*, the bound arguments by parameter name.

        It is the label of the expression :meth:`_derived_expression` builds,
        as :meth:`_result_label` gives it (II.4).
        """
        return self._result_label(values, self._derived_expression(values))

    def _result_label(self, values: Mapping[str, Any], expression: Expression | None) -> str:
        """The label of the result of a call on *values* whose derived expression is *expression*.

        It is the label of *expression*. A result that keeps its route's
        expression, for which *expression* is ``None``, is labeled by the label
        rule where the operation has one, and otherwise by the primary
        operand's label (II.4).

        Parameters
        ----------
        values : Mapping of str to Any
            The call's bound arguments, by parameter name.
        expression : Expression or None
            The expression :meth:`_derived_expression` builds for *values*.

        Returns
        -------
        str
            The result's label.
        """
        if expression is not None:
            return expression.render_label()
        rule = self._label_rule
        if rule is not None:
            return rule(**{name: values.get(name) for name in _parameter_names(rule)})
        primary = values.get(next(iter(self.signature.parameters), ""))
        return primary.label if isinstance(primary, TrackedTerm) else self.output_label

    @property
    def is_derived(self) -> bool:
        """Whether the operation is defined by a formula over other operations."""
        return self._derived

    @property
    def formula(self) -> str | None:
        """The formula of a derived operation, as source; ``None`` for a primitive."""
        return self._formula

    @property
    def routes(self) -> tuple[OperationRoute, ...]:
        """The registered routes, in the selection order of a call without ``exact_only``.

        A delegating route is listed where its exact implementations rank, and
        any other route whose exactness is ``None`` where approximate routes
        rank, as a derived route does. A derived route is listed after the direct
        routes that rank with it.
        """
        indexed = list(enumerate(self._route_table.routes))

        def key(entry: tuple[int, Any]) -> tuple[int, int, int, int]:
            index, route = entry
            exact = route.exact is True or (
                isinstance(route, _DelegatingRoute) and not route.realizes_formula
            )
            fallback = route.source is RouteSource.FALLBACK
            return (0 if exact else 1, 1 if fallback else 0, int(route.realizes_formula), index)

        return tuple(route for _, route in sorted(indexed, key=key))

    # -- routes ------------------------------------------------------------

    def register_route(self, route: OperationRoute) -> OperationRoute:
        """Register *route* after the routes already registered, and return it.

        Parameters
        ----------
        route : OperationRoute
            An object with the members of :class:`OperationRoute`, such as a
            route that one of the construction helpers builds.

        Returns
        -------
        OperationRoute
            *route* itself.

        Raises
        ------
        TypeError
            If *route* does not have the members of :class:`OperationRoute`.
        ValueError
            If the operation already has a route of that name.
        """
        self._route_table.add(route, f"operation {self.label!r}")
        return route

    def structural_route(
        self,
        name: str,
        *,
        check: Callable[[BoundCall, OutputSpec | None], Any],
        execute: Callable[[BoundCall, OutputSpec | None], Any],
        exact: bool,
    ) -> OperationRoute:
        """Register a route whose implementation comes from the operands' declared structure.

        Parameters
        ----------
        name : str
            The route's name.
        check : callable
            ``check(call, result)``, returning ``True``, ``False``, ``None``, or a
            ``Feasibility``; the first paragraph of its docstring is the route's
            condition.
        execute : callable
            ``execute(call, result)``, returning the raw result.
        exact : bool
            Whether the result denotes the requested mathematical object.

        Returns
        -------
        OperationRoute
            The registered route.

        Raises
        ------
        TypeError
            If *check* or *execute* is not callable, or *exact* is not a bool.
        ValueError
            If the operation already has a route named *name*.
        """
        return self.register_route(
            _CheckedRoute(
                name,
                source=RouteSource.STRUCTURAL,
                check=check,
                execute=execute,
                exact=exact,
            )
        )

    def capability_route(
        self,
        name: str,
        *,
        operand: str,
        protocol: type,
        method: str,
        exact: bool,
        check: Callable[[BoundCall, OutputSpec | None], Any] | None = None,
        execute: Callable[[BoundCall, OutputSpec | None], Any] | None = None,
    ) -> OperationRoute:
        """Register a route that calls a capability of the operand it names.

        The route requires the operand to claim *protocol* and the capability's
        guard to admit the call. Without *check*, the guard
        ``_<method>_guard()`` is read with no arguments, so membership suffices
        where the class defines no guard.

        Parameters
        ----------
        name : str
            The route's name.
        operand : str
            The parameter whose argument claims the capability.
        protocol : type
            The capability protocol.
        method : str
            The capability's implementation method, such as ``"_mean"``.
        exact : bool
            Whether the capability returns the requested mathematical object.
        check : callable, optional
            ``check(call, result)``, reading the guard with the call's arguments
            it takes, as a marginal's guard takes the path.
        execute : callable, optional
            ``execute(call, result)``, for a capability that is not called with
            the call's other arguments in signature order.

        Returns
        -------
        OperationRoute
            The registered route.

        Raises
        ------
        TypeError
            If *operand* is not a parameter, or an argument has the wrong type.
        ValueError
            If the operation already has a route named *name*.
        """
        if operand not in self.signature.parameters:
            raise TypeError(f"operation {self.label!r} has no parameter {operand!r}")
        return self.register_route(
            _CapabilityRoute(
                name,
                operand=operand,
                protocol=protocol,
                method=method,
                exact=exact,
                check=check,
                execute=execute,
            )
        )

    def registry_route(
        self,
        name: str,
        *,
        registry: BaseDispatchRegistry[Any],
        arguments: Callable[[BoundCall], Iterable[Any]] | None = None,
        options: Callable[[BoundCall], Mapping[str, Any]] | None = None,
    ) -> OperationRoute:
        """Register a route that delegates to a dispatch registry.

        The registry keeps its own priorities, and the route's exactness is that
        of the method it selects. A ``method=`` control naming one of the
        registry's methods selects this route and that method.

        Parameters
        ----------
        name : str
            The route's name.
        registry : BaseDispatchRegistry
            The registry whose selected method realizes the call.
        arguments : callable, optional
            ``arguments(call)``, the registry's positional arguments; by default
            the call's arguments in signature order.
        options : callable, optional
            ``options(call)``, the keyword options for the registry's methods; by
            default the call's ``method_options``.

        Returns
        -------
        OperationRoute
            The registered route.

        Raises
        ------
        TypeError
            If *registry* is not a dispatch registry.
        ValueError
            If the operation already has a route named *name*.
        """
        return self.register_route(
            _RegistryRoute(name, registry=registry, arguments=arguments, options=options)
        )

    def fallback_route(
        self,
        name: str,
        *,
        check: Callable[[BoundCall, OutputSpec | None], Any],
        execute: Callable[[BoundCall, OutputSpec | None], Any],
        exact: bool,
    ) -> OperationRoute:
        """Register a generic scheme applicable to a stated domain.

        A fallback ranks below every other route of the same exactness. Its
        check states the domain and the assumptions, and the first paragraph of
        its docstring is the route's condition.

        Parameters
        ----------
        name : str
            The route's name.
        check : callable
            ``check(call, result)``, returning ``True``, ``False``, ``None``, or a
            ``Feasibility``.
        execute : callable
            ``execute(call, result)``, returning the raw result.
        exact : bool
            Whether the result denotes the requested mathematical object.

        Returns
        -------
        OperationRoute
            The registered route.

        Raises
        ------
        TypeError
            If *check* or *execute* is not callable, or *exact* is not a bool.
        ValueError
            If the operation already has a route named *name*.
        """
        return self.register_route(
            _CheckedRoute(
                name,
                source=RouteSource.FALLBACK,
                check=check,
                execute=execute,
                exact=exact,
            )
        )

    # -- controls ----------------------------------------------------------

    def using(self, **controls: Any) -> Operation:
        """Return a view with revised controls.

        The controls are those of :meth:`Function.using`: ``method``
        names a route or a method of a registry route's registry,
        ``exact_only`` excludes every approximate route, ``raw`` returns the
        result detached, and ``method_options`` holds the budgets the selected
        method validates when it runs. ``None`` resets a control to its default.

        Parameters
        ----------
        **controls : Any
            The controls to revise, each passed under its name, such as
            ``exact_only=True``.

        Returns
        -------
        Operation
            The view, which shares this operation's routes, so a route
            registered later serves it as well.

        Raises
        ------
        TypeError
            If a control is unknown, ``method`` is not a string, or ``exact_only``
            or ``raw`` is not a bool; or on a framework control's own error.
        ValueError
            On a framework control's own error.
        """
        unknown = set(controls) - set(self.options)
        if unknown:
            raise TypeError(
                f"{self.label}.using(): "
                f"{unknown_names('control', sorted(unknown), sorted(self.options))}"
            )
        return super().using(**controls)

    @property
    def raw(self) -> Callable[..., Any]:
        """The evaluator that realizes one call with no lifting, tracking, or provenance.

        It is :meth:`apply`, which returns the result's raw form.
        """
        return self.apply

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The function's arguments, then the routes in registration order."""
        routes = format_names(route.name for route in self._route_table.routes)
        return [*super()._repr_arguments(), ("routes", routes)]

    # -- the declarations the engine reads ------------------------------------

    def _route_candidates(
        self, controls: Mapping[str, Any], values: Mapping[str, Any] | None = None
    ) -> tuple[_Candidate | _GivenCandidate, ...]:
        """The candidates that may realize one point under *controls*, in selection order.

        A ``method`` name resolves as :meth:`_named_candidates` states.

        Parameters
        ----------
        controls : Mapping of str to Any
            The call's resolved controls, whose ``method`` and ``exact_only``
            entries restrict the candidates.
        values : Mapping of str to Any or None
            The bound inputs. A supplied given adds the conditional delegation
            candidate; ordinary calls retain their existing candidate sequence.

        Returns
        -------
        tuple of _Candidate
            The list :meth:`_candidates` returns, as a tuple.

        Raises
        ------
        ResolutionError
            If ``method`` names no route or registry method, matches several, or
            names an approximate route while ``exact_only`` is set.
        """
        candidates = self._candidates(controls)
        if (
            self._given_subject is not None
            and candidates
            and values is not None
            and values.get("given") is not None
        ):
            return (*candidates, _GivenCandidate())
        return tuple(candidates)

    def _plan_route_inputs(
        self, argument_values: Mapping[str, Any], controls: Mapping[str, Any]
    ) -> tuple[BoundCall, OutputSpec | None, tuple[str, ...]]:
        """The bound call of one point, its result's output spec, and the checks deferred to return.

        Parameters
        ----------
        argument_values : Mapping of str to Any
            The point's arguments by parameter name, which become the bound
            call's ``operands``.
        controls : Mapping of str to Any
            The call's resolved controls, which become the bound call's
            ``controls``.

        Returns
        -------
        call : BoundCall
            The bound call over *argument_values* and *controls*.
        result_declaration : OutputSpec or None
            The result's output spec, or ``None`` when the operands' specs leave
            it open.
        deferred : tuple of str
            A description of each check that only the return can settle, such as
            an applicability check that needs values not yet known.

        Raises
        ------
        ApplicabilityError
            If an applicability check fails, or the result rule rules the call out.
        TypeError
            If the result rule returns something other than an OutputSpec or None.
        """
        operand_values = dict(argument_values)
        if self._given_subject is not None:
            from ..distributions._conditional import ConditionalDistribution
            from ._given import _GivenLaw

            given = operand_values.pop("given", None)
            subject = operand_values[self._given_subject]
            if given is not None:
                if not isinstance(subject, ConditionalDistribution):
                    raise ApplicabilityError(
                        f"{self.name}: given requires a ConditionalDistribution operand"
                    )
                operand_values[self._given_subject] = _GivenLaw(subject, given)
            elif isinstance(subject, ConditionalDistribution) and not self._admits_kernel:
                raise ApplicabilityError(
                    f"{self.name}: ConditionalDistributionSpec requires given values"
                )
        call = BoundCall(self, MappingProxyType(operand_values), controls)
        result_declaration, deferred = self._plan(call)
        if self._given_subject is not None and argument_values.get("given") is not None:
            deferred += (
                "if conditional delegation is selected, the resulting law's capabilities and fidelity are checked at execution",
            )
        return call, result_declaration, deferred

    def _plan(self, call: BoundCall) -> tuple[OutputSpec | None, tuple[str, ...]]:
        """Check the applicability checks, then derive the result's output spec.

        Parameters
        ----------
        call : BoundCall
            The bound call of one point, whose ``declarations`` the applicability
            checks and the result rule read.

        Returns
        -------
        tuple
            The output spec, or ``None`` when the operands' specs leave it open,
            and the checks deferred to the return.

        Raises
        ------
        ApplicabilityError
            If an applicability check fails, or the result rule rules the call out.
        TypeError
            If the result rule returns something other than an OutputSpec or None.
        """
        declarations = call.declarations
        deferred: list[str] = []
        for applicability_check in self._applicability_checks:
            arguments = {
                name: call.operands.get(name)
                if self._roles.get(name) == (FunctionSpec,)
                else declarations[name]
                for name in _parameter_names(applicability_check)
            }
            report = _as_feasibility(
                applicability_check(**arguments),
                applicability_check,
                f"{self.label} applicability check {applicability_check.__name__}",
                f"{self.label}: ",
            )
            if report.feasible is False:
                raise ApplicabilityError(report.description)
            deferred.extend(report.pending)
        result = self._result_rule(**{name: declarations[name] for name in self._rule_parameters})
        if result is not None and not isinstance(result, OutputSpec):
            raise TypeError(
                f"the result rule of {self.label!r} returned {result!r}; it returns an "
                f"OutputSpec or None"
            )
        if result is None or result.term_spec is None:
            deferred.append(_COMPLETED_AT_RETURN)
        return result, tuple(deferred)

    def _candidates(self, controls: Mapping[str, Any]) -> list[_Candidate]:
        """The ways the routes may realize a call under *controls*, in selection order.

        A ``method`` name resolves as :meth:`_named_candidates` states.

        Parameters
        ----------
        controls : Mapping of str to Any
            The call's resolved controls, whose ``method`` and ``exact_only``
            entries restrict the candidates.

        Returns
        -------
        list of _Candidate
            The candidates, sorted by their ``rank``. Without a ``method`` name,
            each route contributes one, and a delegating route two: one
            restricted to its exact implementations and one for the others.
            Under ``exact_only`` they are the exact candidates, the named
            registry methods, and the derived route, whose probe reports why
            ``exact_only`` excludes a body route without a derived route check.

        Raises
        ------
        ResolutionError
            If ``method`` names no route or registry method, matches several, or
            names an approximate route while ``exact_only`` is set.
        """
        method, exact_only = controls["method"], controls["exact_only"]
        if method is not None:
            candidates = self._named_candidates(method, exact_only)
        else:
            candidates = []
            for index, route in enumerate(self._route_table.routes):
                candidates += _route_candidates(route, index, exact_only)
        if exact_only:
            candidates = [
                c
                for c in candidates
                if c.exact is True or c.method is not None or c.route.realizes_formula
            ]
        return sorted(candidates, key=lambda candidate: candidate.rank)

    def _named_candidates(self, method: str, exact_only: bool) -> list[_Candidate]:
        """The candidates a ``method`` name selects.

        A plain name selects the route of that name, or the method of that name
        in the registry of the registry routes holding it; routes that share one
        registry are each a candidate with that method. The qualified form
        ``route/method`` selects the method within the named registry route.

        Parameters
        ----------
        method : str
            The ``method`` control, as a plain name or as ``route/method``.
        exact_only : bool
            The call's ``exact_only`` control, which a route that a plain name
            selects must meet.

        Returns
        -------
        list of _Candidate
            The candidates the name selects. A delegating route that a plain
            name selects contributes two: one restricted to its exact
            implementations and one for the others.

        Raises
        ------
        ResolutionError
            If the name matches nothing; if a plain name matches a route and a
            registry method, or methods of different registries, naming each
            candidate as ``route/method``; or if it names an approximate route
            while ``exact_only`` is set.
        """
        routes = list(self._route_table.routes)
        route_name, qualified, method_name = method.partition("/")
        if qualified:
            for index, route in enumerate(routes):
                if (
                    route.name == route_name
                    and isinstance(route, _RegistryRoute)
                    and method_name in route.registry.list_methods()
                ):
                    return [_Candidate(route, None, index, method_name)]
            registry_routes = {
                route.name: route for route in routes if isinstance(route, _RegistryRoute)
            }
            holder = registry_routes.get(route_name)
            if holder is None:
                detail = unknown_names("registry route", [route_name], registry_routes)
            else:
                detail = (
                    f"route {route_name!r} has no method {method_name!r}; its methods: "
                    f"{list(holder.registry.list_methods())}"
                )
            raise ResolutionError(f"{self.label}: {detail}")
        named = [(index, route) for index, route in enumerate(routes) if route.name == method]
        holders = [
            (index, route)
            for index, route in enumerate(routes)
            if isinstance(route, _RegistryRoute) and method in route.registry.list_methods()
        ]
        registries = {id(route.registry) for _, route in holders}
        if (named and holders) or len(registries) > 1:
            forms = [route.name for _, route in named]
            forms += [f"{route.name}/{method}" for _, route in holders]
            raise ResolutionError(
                f"{self.label}: method={method!r} is ambiguous; it matches "
                f"{', '.join(forms)}. Pass one of these names"
            )
        if named:
            index, route = named[0]
            candidates = _route_candidates(route, index, exact_only)
            # A derived route's probe reports why exact_only declines it.
            if exact_only and candidates[0].exact is not True and not route.realizes_formula:
                raise ResolutionError(
                    f"{self.label}: route {method!r} is not exact, but exact_only=True"
                )
            return candidates
        if holders:
            return [_Candidate(route, None, index, method) for index, route in holders]
        raise ResolutionError(f"{self.label}: {self._unknown_method(method)}")

    def _method_names(self) -> tuple[list[str], list[str]]:
        """The names a ``method`` control can take: the routes, and the registry routes' methods."""
        routes = [route.name for route in self._route_table.routes]
        methods = list(
            dict.fromkeys(
                name
                for route in self._route_table.routes
                if isinstance(route, _RegistryRoute)
                for name in route.registry.list_methods()
            )
        )
        return routes, methods

    def _names_method(self, method: str) -> bool:
        """Whether the plain name *method* names a route or a registry method of this operation."""
        routes, methods = self._method_names()
        return method in routes or method in methods

    def _unknown_method(self, method: str) -> str:
        """The message that no route or registry method is named *method*, with those that are.

        A derived operation's message adds the fix when *method* names a route
        or a method of an operation its body calls, since ``method`` is not
        passed into the formula, whichever route realizes it.
        """
        routes, methods = self._method_names()
        message = (
            f"unknown method {method!r}; available routes: {routes}, registry methods: {methods}"
        )
        constituents = _named_operations(self._body) if self._body is not None else ()
        holders = [
            constituent
            for constituent in constituents
            if constituent is not self and constituent._names_method(method)
        ]
        if holders:
            names = " or ".join(holder.name for holder in holders)
            calls = " or ".join(f"{holder.name}.using(method={method!r})" for holder in holders)
            return (
                f"{message}. {self.label} does not pass method on to {names}, so call "
                f"{calls} to choose it"
            )
        close = difflib.get_close_matches(method, routes + methods, n=1)
        return f"{message}. Did you mean {close[0]!r}?" if close else message

    # -- the summary ---------------------------------------------------------

    def summary(self) -> OperationSummary:
        """The operation's entry in the registry of operations."""
        operands = tuple(
            OperandSummary(name, self._roles[name], name in self._rule_parameters)
            for name in self.signature.parameters
        )
        routes = tuple(
            RouteSummary(
                name=route.name,
                source=route.source,
                exact=route.exact,
                requires=tuple(getattr(route, "requires", ())),
                condition=getattr(route, "condition", ""),
            )
            for route in self.routes
        )
        doc = inspect.getdoc(self) or ""
        return OperationSummary(
            name=self.label,
            priority=None,
            supported_types=operands[0].accepts if operands else (),
            description=doc.split("\n\n", 1)[0].replace("\n", " "),
            module_path=self.__module__,
            operands=operands,
            is_derived=self.is_derived,
            formula=self.formula,
            routes=routes,
        )


# ---------------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------------


def _workflow_draws(
    d: Any, sample_shape: tuple[int, ...], *, operation_kind: str, execution_mode: str
) -> Any:
    """Draws of *d* whose key is a workflow-owned random event.

    An operation takes no key: the scope in which the call runs derives the key
    from the event's structural identity, so a seeded workflow reproduces the
    draws and replay validates them.

    Parameters
    ----------
    d : Distribution
        The law to draw from.
    sample_shape : tuple of int
        The leading shape of the draws; ``()`` draws once.
    operation_kind : str
        The operation that draws, such as ``"sample"`` or ``"mean"``.
    execution_mode : str
        How the operation draws, such as ``"sampled"`` or ``"monte_carlo"``.
        With *operation_kind*, it is the event's identity within the call,
        recorded in the stochastic plan.

    Returns
    -------
    Any
        The raw draws, as ``d._sample(key, sample_shape)`` returns them. The
        draws of a law that reads another law's draws, such as a field view,
        take the raw form of *d*'s own event.
    """
    captured = _descendants.capture_stochastic_consumer(d)
    key = _evaluation_keys._resolve_automatic_key(
        None,
        _evaluation_keys._singleton_effect_plan(
            operation_kind=operation_kind,
            execution_mode=execution_mode,
            sample_shape=sample_shape,
            record_path=captured.record_path,
            descendant_descriptor=captured.descendant_descriptor,
        ),
    )
    draws = _descendants.sample_captured_consumer(captured, key, sample_shape)
    if captured.root is d:
        return draws
    return _drawn_form(draws, d.event_spec.term_spec, sample_shape)


# ---------------------------------------------------------------------------
# The registry of operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperandSummary:
    """One parameter of an operation: its role and whether the result rule reads it.

    Attributes
    ----------
    name : str
        The parameter, as the signature spells it.
    accepts : tuple of type
        The kinds it takes, each named by its spec class; empty when the
        parameter selects rather than supplies, as a field path does.
    planning : bool
        Whether the result rule reads it.
    """

    name: str
    accepts: tuple[type[TermSpec], ...]
    planning: bool


@dataclass(frozen=True)
class RouteSummary:
    """One route of an operation, as the registry of operations lists it.

    Attributes
    ----------
    name : str
        The route's name.
    source : RouteSource
        Where its implementation comes from.
    exact : bool or None
        Its declared exactness; ``None`` for a route whose exactness is that of
        the implementation it selects.
    requires : tuple of type
        The protocols a capability route needs; empty otherwise.
    condition : str
        The feasibility condition in words.
    """

    name: str
    source: RouteSource
    exact: bool | None
    requires: tuple[type, ...]
    condition: str


@dataclass(frozen=True, kw_only=True)
class OperationSummary(EntrySummary):
    """One operation, as the registry of operations lists it.

    The attributes of :class:`~probpipe.core._catalog.EntrySummary` come first,
    and for an operation they hold:

    - ``name``: the operation's name;
    - ``priority`` and ``exact``: ``None``, since operations are not ranked
      against each other and each route declares its own exactness;
    - ``supported_types``: the kinds the first operand accepts;
    - ``description``: the first paragraph of the operation's docstring;
    - ``module_path``: the module that declares the operation.

    Attributes
    ----------
    operands : tuple of OperandSummary
        Every parameter, in signature order.
    is_derived : bool
        Whether the operation is defined by a formula over other operations.
    formula : str or None
        The formula, when derived.
    routes : tuple of RouteSummary
        The routes, in selection order.
    """

    operands: tuple[OperandSummary, ...]
    is_derived: bool
    formula: str | None
    routes: tuple[RouteSummary, ...]


def _render(summary: OperationSummary) -> str:
    """The text :meth:`OperationRegistry.describe` gives for one operation."""
    signature = ", ".join(operand.name for operand in summary.operands)
    kind = f"derived: {summary.formula}" if summary.is_derived else "primitive"
    lines = [f"{summary.name}({signature}) — {kind}"]
    if summary.description:
        lines.append(f"  {summary.description}")
    lines.append("  operands:")
    for operand in summary.operands:
        accepts = _kind_names(operand.accepts) if operand.accepts else "selects, supplies no kind"
        read = "; read by the result rule" if operand.planning else ""
        lines.append(f"    {operand.name}: {accepts}{read}")
    lines.append("  routes, in selection order:")
    if not summary.routes:
        lines.append("    none registered")
    for route in summary.routes:
        exact = {True: "exact", False: "approximate", None: "exactness delegated"}[route.exact]
        requires = f"; requires {_kind_names(route.requires)}" if route.requires else ""
        condition = f": {route.condition}" if route.condition else ""
        lines.append(f"    {route.name} ({route.source.value}, {exact}{requires}){condition}")
    return "\n".join(lines)


class OperationRegistry(SupportsRegistryCataloging):
    """The registry of operations, which makes the vocabulary discoverable.

    Each entry is an operation, listed with its operands, whether it is
    primitive or derived, and its routes in selection order. The registry
    implements :class:`~probpipe.core._catalog.SupportsRegistryCataloging`
    through ``name``, ``description``, ``kind``, :meth:`entry_summaries`, and
    :meth:`describe_entry`, and it is cataloged as ``"operations"``.
    """

    @property
    def name(self) -> str:
        """``"operations"``, the registry's name in the registry catalog."""
        return "operations"

    @property
    def description(self) -> str:
        """A one-line description of the registry, which the registry catalog shows."""
        return "The operations, with their operands and their routes in selection order."

    @property
    def kind(self) -> str:
        """``"operation"``, the registry's catalog kind."""
        return "operation"

    def __init__(self) -> None:
        self._operations: dict[str, Operation] = {}

    def register(self, op: Function) -> None:
        """Register the operation *op* under its name.

        Parameters
        ----------
        op : Operation
            The operation; :func:`operation` calls this method for each operation
            it declares.

        Raises
        ------
        TypeError
            If *op* is not an operation.
        ValueError
            If an operation of that name is registered.
        """
        if not isinstance(op, Operation):
            raise TypeError(f"only an operation registers here; got {type(op).__name__}")
        if op.label in self._operations:
            raise ValueError(f"an operation named {op.label!r} is already registered")
        self._operations[op.label] = op

    def list(self) -> list[OperationSummary]:
        """One summary per operation, in registration order."""
        return [op.summary() for op in self._operations.values()]

    def describe(self, name: str | None = None) -> str:
        """The summaries as text, for the operation *name* or for all of them.

        Parameters
        ----------
        name : str or None
            An operation's name in the registry; ``None`` describes every
            operation, in registration order.

        Returns
        -------
        str
            One block per operation, the blocks separated by a blank line. A block
            opens with the signature and whether the operation is primitive or
            derived, and then lists the description, the operands, and the routes
            in selection order.

        Raises
        ------
        KeyError
            If *name* is not a registered operation.
        """
        if name is not None:
            return _render(self._get(name).summary())
        return "\n\n".join(_render(summary) for summary in self.list())

    def __getitem__(self, name: str) -> Function:
        """The operation registered as *name*.

        Parameters
        ----------
        name : str
            The name of the operation's declaration, such as ``"mean"``.

        Returns
        -------
        Operation
            The registered object itself.

        Raises
        ------
        KeyError
            If no operation is registered as *name*.
        """
        return self._get(name)

    def entry_summaries(self) -> list[OperationSummary]:
        """The summaries, as a cataloged registry reports its entries."""
        return self.list()

    def describe_entry(self, name: str) -> OperationSummary:
        """The summary of the operation *name*.

        Parameters
        ----------
        name : str
            The name of the operation's declaration, such as ``"mean"``.

        Returns
        -------
        OperationSummary
            The operation's entry as :meth:`list` reports it.

        Raises
        ------
        KeyError
            If *name* is not a registered operation.
        """
        return self._get(name).summary()

    def _get(self, name: str) -> Operation:
        """The operation registered as *name*, raising KeyError with the registered names."""
        try:
            return self._operations[name]
        except KeyError:
            raise KeyError(unknown_names("operation", [name], self._operations)) from None


operation_registry: OperationRegistry = OperationRegistry()
"""The global registry of operations."""
registry_catalog.register(operation_registry)


def operation(
    *,
    result: Callable[..., OutputSpec | None],
    applicability_checks: Iterable[Callable[..., Any]] = (),
    roles: Mapping[str, Iterable[type[TermSpec]]] | None = None,
    derived_route_check: Callable[..., Any] | None = None,
    derived_route: OperationRoute | None = None,
    label: Callable[..., str] | None = None,
    registry: OperationRegistry | None = None,
) -> Callable[[Callable[..., Any]], Operation]:
    """Declare an operation from its signature and register it.

    The decorated function's signature contains the operands and the parameters
    the result rule reads; an empty body declares a primitive operation and a
    body that returns an expression states a derived operation's formula. The
    routes are registered on the returned operation.

    Parameters
    ----------
    result : callable
        The result rule; see :class:`Operation`.
    applicability_checks : iterable of callable
        Checks that the operation applies to a call; see :class:`Operation`.
    roles : mapping of str to iterable of TermSpec subclasses, optional
        Roles that override the kinds the annotations name.
    derived_route_check : callable, optional
        The derived route's feasibility check; see :class:`Operation`.
    derived_route : OperationRoute, optional
        A derived operation's composite derived route; see :class:`Operation`.
    label : callable, optional
        The label rule; see :class:`Operation`.
    registry : OperationRegistry, optional
        The registry to register in; :data:`operation_registry` by default.

    Returns
    -------
    callable
        The decorator, which returns the registered :class:`Operation`.

    Raises
    ------
    TypeError
        As :class:`Operation` and :meth:`OperationRegistry.register` raise.
    ValueError
        If the registry already has an operation of that name.
    """

    def decorate(declaration: Callable[..., Any]) -> Operation:
        op = Operation(
            declaration,
            result=result,
            applicability_checks=applicability_checks,
            roles=roles,
            derived_route_check=derived_route_check,
            derived_route=derived_route,
            label=label,
        )
        (operation_registry if registry is None else registry).register(op)
        return op

    return decorate


def _install_expression_rule(op: Operation, rule: Callable[..., Expression | None]) -> None:
    """Install *rule* as *op*'s expression rule, which the result boundary reads (II.4).

    The rule is called with the call's arguments it names, as the label rule
    is, and returns the expression of the result, built from the operands'
    expressions, or ``None`` for a result that keeps the expression its route
    gave it. The result boundary gives each point's result the expression
    after the route returns, so every route of *op* gives its result the same
    expression, and the result's label is the expression's label.

    Parameters
    ----------
    op : Operation
        The operation whose results carry the expression.
    rule : callable
        The expression rule, whose parameters are parameters of *op*.

    Raises
    ------
    TypeError
        If *rule* is not callable, or it reads a name that *op* does not
        declare as a parameter.
    """
    owner = f"operation {op.label!r}"
    if not callable(rule):
        raise TypeError(f"{owner} needs a callable expression rule; got {rule!r}")
    unknown = set(_parameter_names(rule)) - set(op.signature.parameters)
    if unknown:
        raise TypeError(
            f"{owner}: the expression rule reads {sorted(unknown)}, which the declaration "
            f"does not declare as parameters"
        )
    object.__setattr__(op, "_expression_rule", rule)
