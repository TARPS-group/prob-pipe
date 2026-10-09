"""The operation model: roles, conditions, a result rule, and the routes that realize a call.

An **operation** is a :class:`~probpipe.values.Function` whose implementations
differ in the kinds they handle, in the declaration of their result, and in
where they come from. ``@operation`` declares one from a signature and a result
rule, and its routes are registered beside it. A call runs the engine's stack,
which reads three declarations from the operation at its steps:

1. **Admission** and lift classification read each parameter's **role**, the
   kinds it accepts, each named by its spec class: an argument the role admits
   passes whole, a batch or a law over an admitted kind is lifted, and any
   other argument is refused.
2. **Planning** checks the **applicability conditions** and runs the **result
   rule**, which derives the declaration of each point's result from the
   operands' specs and from the parameters that select rather than supply,
   such as a field path; the return validates the result against it.
3. **Resolution** selects one **route** among the operation's candidates, in
   the order this module ranks them: exact routes before approximate ones, a
   fallback below every other route whatever its exactness, and registration
   order for the remaining ties. The ``method`` and ``exact_only`` controls
   restrict the choice, and provenance records the selected route.

An operation takes no key: each draw a route causes is a workflow-owned random
event, whose key :func:`_workflow_draws` derives from the workflow scope.

The operations form a registry, :data:`operation_registry`, whose ``list()`` and
``describe()`` report each operation's operands, whether it is primitive or
derived, and its routes in selection order.
"""

from __future__ import annotations

import ast
import difflib
import dis
import inspect
import textwrap
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from .._messages import unknown_names
from ..core._array_backend import _event_shape_of, _is_numeric_leaf, _numpy_dtype_of
from ..core._dispatch import BaseDispatchRegistry, Feasibility, MethodInfo, ResolutionError
from ..core._kinds import _KINDS
from ..core._record_spec import RecordSpec
from ..core._repr import format_names, public_class_name
from ..core._spec_base import NumericArraySpec, OpaqueSpec, TermSpec
from ..core._specs import OutputSpec
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..distributions._capabilities import _capability_guard, _guard_condition, _requirement
from ..functions import _broker, _descendants
from ..functions._call import ApplicabilityError, CallReport
from ..functions._resolution import PointReport, StandIn
from ..values import Function, FunctionSpec
from ..values._binding import values_to_bound_arguments
from ..values._function_base import _COMPLETED_AT_RETURN

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
    """

    operation: Function
    operands: Mapping[str, Any]
    controls: Mapping[str, Any]

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
        """What the result rule and the applicability conditions read, by parameter.

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
    """The label the result of *call* takes, as its operation derives it (II.4)."""
    return call.operation._derived_label(call.operands)


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
        to, until that implementation is selected.
    """

    name: str
    source: RouteSource
    exact: bool | None

    def check(self, call: BoundCall, result: OutputSpec | None) -> Feasibility:
        """Whether this route realizes *call*, whose result declaration is *result*."""
        ...

    def execute(self, call: BoundCall, result: OutputSpec | None) -> Any:
        """Realize *call* and return its raw result."""
        ...


def _as_feasibility(
    report: Any, source: Callable[..., Any], owner: str, subject: str = ""
) -> Feasibility:
    """*report* as a Feasibility, quoting the condition *source*'s docstring states.

    Parameters
    ----------
    report : bool, None, Feasibility, or CallReport
        What *source* returned. A CallReport is the report of another
        operation's ``check``, which a derived operation's identity check returns.
    source : callable
        The check or condition that returned *report*, whose docstring's first
        paragraph states the condition.
    owner : str
        The checker that an unresolved report names, such as ``route 'identity'``.
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
        # Another operation's check, as a derived operation's identity reads it.
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
        options = {} if self.exact else self.method_options(call)
        return getattr(subject, self.method)(*others, **options)


class _RegistryRoute(_Route):
    """A route that delegates to a dispatch registry, whose selected method realizes the call.

    The registry receives the call's arguments in signature order, or those
    *arguments* returns, with the keyword options *options* returns, or else the
    call's ``method_options``. Its exactness is that of the
    method the registry selects, so the route is ranked twice: its exact methods
    with the exact routes and its approximate methods with the approximate ones.

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
        super().__init__(name, exact=None)
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

    def probe(self, call: BoundCall, *, method: str | None, exact_only: bool) -> MethodInfo:
        """The registry's report for *call*, restricted as the controls ask."""
        return self.registry.check(
            *self.arguments(call), method=method, exact_only=exact_only, **self.options(call)
        )

    def run(self, call: BoundCall, *, method: str | None, exact_only: bool) -> Any:
        """The result of the registry's selected method, or of *method*."""
        return self.registry.execute(
            *self.arguments(call), method=method, exact_only=exact_only, **self.options(call)
        )

    def check(self, call: BoundCall, result: OutputSpec | None) -> Feasibility:
        """The registry's report under the call's ``exact_only``."""
        return self.probe(call, method=None, exact_only=call.controls["exact_only"])

    def execute(self, call: BoundCall, result: OutputSpec | None) -> Any:
        """The result of the method the registry selects under the call's ``exact_only``."""
        return self.run(call, method=None, exact_only=call.controls["exact_only"])


def _identity_route(
    declaration: Callable[..., Any],
    signature: inspect.Signature,
    identity_check: Callable[..., Any] | None,
) -> _CheckedRoute:
    """The defining identity of a derived operation, as a fallback on its own domain.

    The route's check calls *identity_check* with the call's arguments, as the
    identity is called, so the route is feasible where the constituent
    operations have routes, and its condition is the identity check's. Its
    exactness is that of the implementations the constituents select, so it
    declares none.
    """

    def feasible(call: BoundCall, result: OutputSpec | None) -> Any:
        """The constituent operations decide the call when the identity runs."""
        if identity_check is None:
            return True
        bound = values_to_bound_arguments(signature, call.operands)
        return identity_check(*bound.args, **bound.kwargs)

    if identity_check is not None:
        feasible.__doc__ = identity_check.__doc__

    def evaluate_identity(call: BoundCall, result: OutputSpec | None) -> Any:
        bound = values_to_bound_arguments(signature, call.operands)
        return declaration(*bound.args, **bound.kwargs)

    return _CheckedRoute(
        "identity",
        source=RouteSource.FALLBACK,
        check=feasible,
        execute=evaluate_identity,
        exact=None,
    )


# ---------------------------------------------------------------------------
# The candidates of selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """One way a route may realize a call: the route at one exactness.

    A registry route contributes one candidate for its exact methods and one for
    its approximate ones; ``method`` is the registry method a ``method=`` control
    names, whose exactness the registry reports. A candidate is what the
    engine's resolution step probes and runs, through the members
    :mod:`probpipe.functions._resolution` names.
    """

    route: Any
    exact: bool | None
    index: int
    method: str | None = None

    @property
    def rank(self) -> tuple[int, int, int]:
        """A fallback below every other source whatever its exactness, then exact before
        approximate, then registration order."""
        return (
            1 if self.route.source is RouteSource.FALLBACK else 0,
            0 if self.exact is True else 1,
            self.index,
        )

    @property
    def label(self) -> str:
        """The route's name, qualified for a registry route by the methods the candidate covers."""
        if not isinstance(self.route, _RegistryRoute):
            return self.route.name
        if self.method is not None:
            return f"{self.route.name}/{self.method}"
        return f"{self.route.name} ({'exact' if self.exact else 'approximate'} methods)"

    @property
    def route_name(self) -> str:
        """The name of the route the candidate belongs to."""
        return self.route.name

    @property
    def methods(self) -> tuple[str, ...] | None:
        """The methods of a registry route's registry, in its selection order, or ``None``."""
        if not isinstance(self.route, _RegistryRoute):
            return None
        return tuple(self.route.registry.list_methods())

    def probe(self, call: BoundCall, result: OutputSpec | None) -> Feasibility:
        """The candidate's report for *call*, without executing anything."""
        route = self.route
        if isinstance(route, _RegistryRoute):
            return route.probe(
                call,
                method=self.method,
                exact_only=call.controls["exact_only"] or self.exact is True,
            )
        return route.check(call, result)

    def run(self, call: BoundCall, result: OutputSpec | None, report: Feasibility) -> Any:
        """The route's raw result for *call*, by the method *report* selected for a registry route."""
        route = self.route
        if isinstance(route, _RegistryRoute):
            return route.run(
                call,
                method=report.method_name if isinstance(report, MethodInfo) else self.method,
                exact_only=call.controls["exact_only"] or self.exact is True,
            )
        return route.execute(call, result)

    def exactness(self, report: Feasibility) -> bool | None:
        """The exactness of the implementation the candidate selects, as *report* gives it.

        A route that delegates its exactness reports that of what its probe
        selected: a registry route's method, or the route a derived operation's
        constituent selects. Any other report reads at the exactness of the
        methods the candidate covers.
        """
        if self.route.exact is not None:
            return self.route.exact
        if isinstance(report, (MethodInfo, PointReport)) and report.exact is not None:
            return report.exact
        return self.exact

    def method_of(self, report: Feasibility) -> str | None:
        """The registry method a registry route delegates to, as *report* names it."""
        if isinstance(self.route, _RegistryRoute) and isinstance(report, MethodInfo):
            return report.method_name
        return None


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


def _identity_source(function: Callable[..., Any]) -> str | None:
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
    """
    if hint is inspect.Parameter.empty:
        return _VALUE_KINDS
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


class Operation(Function):
    """A Function realized by routes, with operand roles, conditions, and a result rule.

    Calling it runs the engine's stack, whose admission, planning, resolution,
    and return steps read the declarations below: each argument is admitted and
    lifted by its role, the applicability conditions and the result rule derive
    each point's result declaration, the engine selects a route among the
    operation's candidates, and the return wraps the route's raw result at the
    declared kind. A parameter annotated ``Any`` takes its argument as it
    arrives, with no lifting, so its role only narrows what it admits. The
    routes are registered after construction with the four helpers or
    :meth:`register_route`, and a derived operation also carries its defining
    identity as a route.

    Parameters
    ----------
    declaration : callable
        The authored signature: the operands and the parameters the result rule
        reads. An empty body declares a primitive operation, and a body that
        returns an expression states a derived operation's identity.
    result : callable
        The result rule. Its parameters are parameters of *declaration*; it
        receives each supplying operand's spec and each selecting parameter's
        value, and returns the result's ``OutputSpec``, or ``None`` when the
        declarations leave the result open.
    conditions : iterable of callable
        Applicability conditions, called like the result rule and before it.
        Each returns ``True``, ``False``, ``None`` when the answer depends on
        values not yet known, or a ``Feasibility``; the first paragraph of its
        docstring states the condition.
    roles : mapping of str to iterable of TermSpec subclasses, optional
        The kinds a parameter accepts, overriding the kinds its annotation names.
        An empty role marks a parameter that selects rather than supplies.
    identity_check : callable, optional
        A derived operation's probe of its identity. It is called with the
        call's arguments, as the identity is, and returns the report of the
        constituent operations' checks, such as ``log_prob.check(d, value)``
        for ``prob``; the first paragraph of its docstring is the identity
        route's condition. Without it, the identity route admits every call
        that planning admits.
    label : callable, optional
        The label rule, called with the call's arguments it names and
        returning the result's label. Without it, the result takes the label
        of the primary operand, the first parameter's argument (II.4).

    Raises
    ------
    TypeError
        If *declaration*, *result*, a condition, *identity_check*, or *label*
        is not callable, the result rule, a condition, or the label rule reads
        a name the declaration does not declare, a role is malformed, or a
        primitive operation is given an identity check.
    """

    def __init__(
        self,
        declaration: Callable[..., Any],
        *,
        result: Callable[..., OutputSpec | None],
        conditions: Iterable[Callable[..., Any]] = (),
        roles: Mapping[str, Iterable[type[TermSpec]]] | None = None,
        identity_check: Callable[..., Any] | None = None,
        label: Callable[..., str] | None = None,
    ) -> None:
        if not callable(declaration):
            raise TypeError(f"an operation is declared by a function; got {declaration!r}")
        super().__init__(declaration.__name__, declaration)
        object.__setattr__(self, "_declaration_name", declaration.__name__)
        owner = f"operation {self.label!r}"
        if not callable(result):
            raise TypeError(f"{owner} needs a callable result rule; got {result!r}")
        conditions = tuple(conditions)
        if not all(callable(condition) for condition in conditions):
            raise TypeError(f"{owner} was given an applicability condition that is not callable")
        if label is not None and not callable(label):
            raise TypeError(f"{owner} needs a callable label rule; got {label!r}")
        parameters = self.signature.parameters
        read = {parameter for rule in (result, *conditions) for parameter in _parameter_names(rule)}
        unknown = (read | set(_parameter_names(label) if label else ())) - set(parameters)
        if unknown:
            raise TypeError(
                f"{owner}: the result rule, a condition, or the label rule reads "
                f"{sorted(unknown)}, which the declaration does not declare as parameters"
            )
        derived = not _has_empty_body(declaration)
        if identity_check is not None:
            if not derived:
                raise TypeError(
                    f"{owner} has an empty body, so it cannot take identity_check; only an "
                    f"operation whose body returns its identity can"
                )
            if not callable(identity_check):
                raise TypeError(f"{owner} needs a callable identity check; got {identity_check!r}")
        set_attribute = object.__setattr__
        set_attribute(self, "_result_rule", result)
        set_attribute(self, "_rule_parameters", _parameter_names(result))
        set_attribute(self, "_conditions", conditions)
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
        set_attribute(self, "_derived", derived)
        set_attribute(self, "_identity", _identity_source(declaration) if derived else None)
        set_attribute(self, "_route_table", _RouteTable())
        set_attribute(self, "_label_rule", label)
        set_attribute(self, "_fixed_path_rule", None)
        if derived:
            self.register_route(_identity_route(declaration, self.signature, identity_check))

    # -- identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        """The operation's key in :data:`operation_registry`, the name of its declaration."""
        return self._declaration_name

    # -- declarations ------------------------------------------------------

    def _derived_label(self, values: Mapping[str, Any]) -> str:
        """The label of the result of a call on *values*, the bound arguments by parameter name.

        The label rule derives it where the operation has one. Otherwise the
        result takes the label of the primary operand, the first parameter's
        argument, and an argument that is not a tracked term leaves the
        operation's own output label.
        """
        rule = self._label_rule
        if rule is not None:
            return rule(**{name: values.get(name) for name in _parameter_names(rule)})
        primary = values.get(next(iter(self.signature.parameters), ""))
        return primary.label if isinstance(primary, TrackedTerm) else self.output_label

    def _derived_fixed_paths(self, values: Mapping[str, Any]) -> tuple[str, ...]:
        """The paths a law or kernel result of a call on *values* holds fixed, in order (II.4).

        The operation's fixed-path rule derives them, called with the call's
        arguments it names, as the label rule is, and
        :func:`_install_fixed_path_rule` installs it. An operation without one
        derives none, so its result holds the paths its route gave it.
        """
        rule = self._fixed_path_rule
        if rule is None:
            return ()
        return tuple(rule(**{name: values.get(name) for name in _parameter_names(rule)}))

    @property
    def is_derived(self) -> bool:
        """Whether the operation is defined by an identity over other operations."""
        return self._derived

    @property
    def identity(self) -> str | None:
        """The defining identity of a derived operation, as source; ``None`` for a primitive."""
        return self._identity

    @property
    def routes(self) -> tuple[OperationRoute, ...]:
        """The registered routes, in selection order.

        A route whose exactness is delegated is listed where its exact methods
        rank.
        """
        indexed = list(enumerate(self._route_table.routes))

        def key(entry: tuple[int, Any]) -> tuple[int, int, int]:
            index, route = entry
            exact = route.exact is True or isinstance(route, _RegistryRoute)
            fallback = route.source is RouteSource.FALLBACK
            return (0 if exact else 1, 1 if fallback else 0, index)

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

    def with_options(self, **controls: Any) -> Operation:
        """Return a view with revised controls.

        The controls are those of :meth:`Function.with_options`: ``method``
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
                f"{self.label}.with_options(): "
                f"{unknown_names('control', sorted(unknown), sorted(self.options))}"
            )
        return super().with_options(**controls)

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

    def _route_candidates(self, controls: Mapping[str, Any]) -> tuple[_Candidate, ...]:
        """The candidates that may realize one point under *controls*, in selection order.

        A ``method`` name resolves as :meth:`_named_candidates` states.

        Parameters
        ----------
        controls : Mapping of str to Any
            The call's resolved controls, whose ``method`` and ``exact_only``
            entries restrict the candidates.

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
        return tuple(self._candidates(controls))

    def _plan_point(
        self, values: Mapping[str, Any], controls: Mapping[str, Any]
    ) -> tuple[BoundCall, OutputSpec | None, tuple[str, ...]]:
        """The bound call of one point, its result declaration, and the checks deferred to return.

        Parameters
        ----------
        values : Mapping of str to Any
            The point's arguments by parameter name, which become the bound
            call's ``operands``.
        controls : Mapping of str to Any
            The call's resolved controls, which become the bound call's
            ``controls``.

        Returns
        -------
        call : BoundCall
            The bound call over *values* and *controls*.
        result : OutputSpec or None
            The result's declaration, or ``None`` when the declarations leave it
            open.
        deferred : tuple of str
            A description of each check that only the return can settle, such as
            a condition that needs values not yet known.

        Raises
        ------
        ApplicabilityError
            If a condition fails, or the result rule rules the call out.
        TypeError
            If the result rule returns something other than an OutputSpec or None.
        """
        call = BoundCall(self, MappingProxyType(dict(values)), controls)
        result, deferred = self._plan(call)
        return call, result, deferred

    def _plan(self, call: BoundCall) -> tuple[OutputSpec | None, tuple[str, ...]]:
        """Check the applicability conditions, then derive the result's declaration.

        Parameters
        ----------
        call : BoundCall
            The bound call of one point, whose ``declarations`` the conditions and
            the result rule read.

        Returns
        -------
        tuple
            The declaration, or ``None`` when the declarations leave it open, and
            the checks deferred to the return.

        Raises
        ------
        ApplicabilityError
            If a condition fails, or the result rule rules the call out.
        TypeError
            If the result rule returns something other than an OutputSpec or None.
        """
        declarations = call.declarations
        deferred: list[str] = []
        for condition in self._conditions:
            arguments = {name: declarations[name] for name in _parameter_names(condition)}
            report = _as_feasibility(
                condition(**arguments),
                condition,
                f"{self.label} condition {condition.__name__}",
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
        if result is None or result.spec is None:
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
            each route contributes one, and a registry route two: one for its
            exact methods and one for its approximate ones.

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
                if isinstance(route, _RegistryRoute):
                    candidates += [_Candidate(route, True, index), _Candidate(route, False, index)]
                else:
                    candidates.append(_Candidate(route, route.exact, index))
        if exact_only:
            candidates = [c for c in candidates if c.exact is True or c.method is not None]
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
            The candidates the name selects. A registry route that a plain name
            selects as a route contributes two: one for its exact methods and one
            for its approximate ones.

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
            if isinstance(route, _RegistryRoute):
                return [_Candidate(route, True, index), _Candidate(route, False, index)]
            if exact_only and route.exact is not True:
                raise ResolutionError(
                    f"{self.label}: route {method!r} is not exact, but exact_only=True"
                )
            return [_Candidate(route, route.exact, index)]
        if holders:
            return [_Candidate(route, None, index, method) for index, route in holders]
        raise ResolutionError(f"{self.label}: {self._unknown_method(method)}")

    def _unknown_method(self, method: str) -> str:
        """The message that no route or registry method is named *method*, with those that are."""
        routes = [route.name for route in self._route_table.routes]
        methods = list(
            dict.fromkeys(
                name
                for route in self._route_table.routes
                if isinstance(route, _RegistryRoute)
                for name in route.registry.list_methods()
            )
        )
        close = difflib.get_close_matches(method, routes + methods, n=1)
        hint = f". Did you mean {close[0]!r}?" if close else ""
        return (
            f"unknown method {method!r}; available routes: {routes}, registry methods: "
            f"{methods}{hint}"
        )

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
            supported_types=operands[0].accepts if operands else (),
            description=doc.split("\n\n", 1)[0].replace("\n", " "),
            module_path=self.__module__,
            operands=operands,
            is_derived=self.is_derived,
            identity=self.identity,
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
        The raw draws, as ``d._sample(key, sample_shape)`` returns them.
    """
    captured = _descendants.capture_stochastic_consumer(d)
    key = _broker._resolve_automatic_key(
        None,
        _broker._singleton_effect_plan(
            operation_kind=operation_kind,
            execution_mode=execution_mode,
            sample_shape=sample_shape,
            record_path=captured.record_path,
            descendant_descriptor=captured.descendant_descriptor,
        ),
    )
    return _descendants.sample_captured_consumer(captured, key, sample_shape)


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
class OperationSummary:
    """One operation, as the registry of operations lists it.

    The first five attributes are those every registry entry reports.

    Attributes
    ----------
    name : str
        The operation's name.
    priority : int or None
        Always ``None``: operations are not ranked against each other.
    supported_types : tuple of type
        The kinds the first operand accepts.
    description : str
        The first paragraph of the operation's docstring.
    module_path : str
        The module that declares the operation.
    operands : tuple of OperandSummary
        Every parameter, in signature order.
    is_derived : bool
        Whether the operation is defined by an identity over other operations.
    identity : str or None
        The defining identity, when derived.
    routes : tuple of RouteSummary
        The routes, in selection order.
    """

    name: str
    priority: int | None = None
    supported_types: tuple[Any, ...] = ()
    description: str = ""
    module_path: str = ""
    operands: tuple[OperandSummary, ...]
    is_derived: bool
    identity: str | None
    routes: tuple[RouteSummary, ...]


def _render(summary: OperationSummary) -> str:
    """The text :meth:`OperationRegistry.describe` gives for one operation."""
    signature = ", ".join(operand.name for operand in summary.operands)
    kind = f"derived: {summary.identity}" if summary.is_derived else "primitive"
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


class OperationRegistry:
    """The registry of operations, which makes the vocabulary discoverable.

    Each entry is an operation, listed with its operands, whether it is
    primitive or derived, and its routes in selection order. The registry
    carries the members a cataloged registry reports: ``name``,
    ``description``, ``kind``, ``entry_summaries``, and ``describe_entry``.
    """

    name = "operations"
    description = "The operations, with their operands and their routes in selection order"
    kind = "operation"

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


def operation(
    *,
    result: Callable[..., OutputSpec | None],
    conditions: Iterable[Callable[..., Any]] = (),
    roles: Mapping[str, Iterable[type[TermSpec]]] | None = None,
    identity_check: Callable[..., Any] | None = None,
    label: Callable[..., str] | None = None,
    registry: OperationRegistry | None = None,
) -> Callable[[Callable[..., Any]], Operation]:
    """Declare an operation from its signature and register it.

    The decorated function's signature contains the operands and the parameters
    the result rule reads; an empty body declares a primitive operation and a
    body that returns an expression states a derived operation's identity. The
    routes are registered on the returned operation.

    Parameters
    ----------
    result : callable
        The result rule; see :class:`Operation`.
    conditions : iterable of callable
        The applicability conditions; see :class:`Operation`.
    roles : mapping of str to iterable of TermSpec subclasses, optional
        Roles that override the kinds the annotations name.
    identity_check : callable, optional
        A derived operation's probe of its identity; see :class:`Operation`.
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
            conditions=conditions,
            roles=roles,
            identity_check=identity_check,
            label=label,
        )
        (operation_registry if registry is None else registry).register(op)
        return op

    return decorate


def _install_fixed_path_rule(op: Operation, rule: Callable[..., Iterable[str]]) -> None:
    """Install *rule* as *op*'s fixed-path rule, which the result boundary reads (II.4).

    The rule is called with the call's arguments it names, as the label rule
    is, and returns the paths a law or kernel result holds fixed, in order.
    The result boundary records them on each point's result after the route
    returns, so every route of *op* gives its result the same fixed paths.

    Parameters
    ----------
    op : Operation
        The operation whose results hold the fixed paths.
    rule : callable
        The fixed-path rule, whose parameters are parameters of *op*.

    Raises
    ------
    TypeError
        If *rule* is not callable, or it reads a name that *op* does not
        declare as a parameter.
    """
    owner = f"operation {op.label!r}"
    if not callable(rule):
        raise TypeError(f"{owner} needs a callable fixed-path rule; got {rule!r}")
    unknown = set(_parameter_names(rule)) - set(op.signature.parameters)
    if unknown:
        raise TypeError(
            f"{owner}: the fixed-path rule reads {sorted(unknown)}, which the declaration "
            f"does not declare as parameters"
        )
    object.__setattr__(op, "_fixed_path_rule", rule)
