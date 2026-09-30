"""The operation model: roles, conditions, a result rule, and the routes that realize a call.

An **operation** is a :class:`~probpipe.values.Function` whose implementations
differ in the kinds they handle, in the declaration of their result, and in
where they come from. ``@operation`` declares one from a signature and a result
rule, and its routes are registered beside it. A call runs the engine's stack,
and this module supplies the operation's side of four of its steps:

1. **Admission** checks each argument against the kinds its **role** accepts,
   each kind named by its spec class.
2. **Planning** checks the **applicability conditions** and runs the **result
   rule**, which derives the result's declaration from the operands' specs and
   from the parameters that select rather than supply, such as a field path.
3. **Resolution** selects one **route** among the feasible ones. Exact routes
   rank before approximate ones, a fallback ranks below every other route of
   the same exactness, and registration order breaks the remaining ties. The
   ``method`` and ``exact_only`` controls restrict the choice.
4. **Return** wraps the selected route's raw result at the declared kind.

An operation takes no key: each draw a route causes is a workflow-owned random
event, whose key :func:`_workflow_draws` derives from the workflow scope.

The operations form a registry, :data:`operation_registry`, whose ``list()`` and
``describe()`` report each operation's operands, whether it is primitive or
derived, and its routes in selection order.
"""

from __future__ import annotations

import ast
import dis
import inspect
import textwrap
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import prod
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ..core._array_backend import _event_shape_of, _is_numeric_leaf, _numpy_dtype_of
from ..core._batch import Batch, BatchSpec, _ranks_of
from ..core._broadcast_distributions import _make_stack
from ..core._dispatch import BaseDispatchRegistry, Feasibility, MethodInfo, ResolutionError
from ..core._kinds import _KINDS, batch_class_for_spec
from ..core._numeric_array_batch import NumericArrayBatch
from ..core._object_batch import _is_object_array
from ..core._record_batch import _batch_class_for
from ..core._record_spec import RecordSpec, _reshaped_template
from ..core._spec_base import NumericArraySpec, TermSpec, _unify_specs
from ..core._specs import OutputSpec
from ..core.node import Node
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..distributions._capabilities import _capability_guard, _guard_condition
from ..functions import _broker, _descendants
from ..functions._result import _wrap_as_term, _wrap_declared_function_output
from ..values import Function, FunctionSpec
from ..values._binding import resolve_workflow_values, values_to_bound_arguments
from ..values._function_base import _validate_function_output

__all__ = [
    "ApplicabilityError",
    "BoundCall",
    "CallCheck",
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

#: The controls every operation adds to the framework's, with their defaults.
_OPERATION_CONTROLS: Mapping[str, Any] = MappingProxyType(
    {"method": None, "exact_only": False, "raw": False}
)

#: The term classes whose spec class the kind table does not record.
_UNREGISTERED_TERM_KINDS: tuple[tuple[type, type[TermSpec]], ...] = (
    (Function, FunctionSpec),
    (Record, RecordSpec),
)


class ApplicabilityError(TypeError):
    """The arguments or declarations of a call violate the operation's contract.

    Admission raises it for an argument whose kind the parameter's role does not
    accept, and planning raises it for a failed applicability condition or for a
    call that the result rule rules out.
    """


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
        signature order: a term, a raw value, or a planned conversion.
    controls : Mapping[str, Any]
        The resolved controls: the framework's, the operation's ``method``,
        ``exact_only``, and ``raw``, and the budgets set for its routes.
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


def _spec_of(value: Any) -> TermSpec:
    """The spec of *value*: a term's own, and otherwise the kind the wrap table gives it."""
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


def _as_feasibility(report: Any, source: Callable[..., Any], owner: str) -> Feasibility:
    """*report* as a Feasibility, quoting the condition *source*'s docstring states.

    Raises
    ------
    TypeError
        If *report* is not a bool, ``None``, or a Feasibility.
    """
    if isinstance(report, Feasibility):
        return report
    condition = _guard_condition(source)
    suffix = f": {condition}" if condition else ""
    if report is True:
        return Feasibility(True)
    if report is False:
        return Feasibility(False, f"{owner} declined{suffix}")
    if report is None:
        return Feasibility(None, pending=(f"{owner} needs values not yet known{suffix}",))
    raise TypeError(f"{owner} returned {report!r}; a check returns a bool, None, or a Feasibility")


class _Route:
    """The name, exactness, and budget controls every constructed route declares.

    Raises
    ------
    TypeError
        If *name* is not a non-empty string or *exact* is neither a bool nor
        ``None``.
    """

    source: RouteSource
    requires: tuple[type, ...] = ()

    def __init__(
        self, name: str, *, exact: bool | None, controls: Iterable[str] | None = ()
    ) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError(f"a route's name must be a non-empty string; got {name!r}")
        if exact is not None and type(exact) is not bool:
            raise TypeError(f"route {name!r} must declare exact as a bool or None; got {exact!r}")
        self.name = name
        self.exact = exact
        self.controls = None if controls is None else frozenset(controls)

    @property
    def condition(self) -> str:
        """The feasibility condition in words."""
        return ""

    def __repr__(self) -> str:
        """The route's class and name."""
        return f"{type(self).__name__}({self.name!r})"


class _CheckedRoute(_Route):
    """A route of any source given as a check and an execute function over the bound call.

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
        controls: Iterable[str] = (),
    ) -> None:
        super().__init__(name, exact=exact, controls=controls)
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
    in signature order, or by a given *execute*.

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
        controls: Iterable[str] = (),
    ) -> None:
        super().__init__(name, exact=exact, controls=controls)
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
                False, f"{type(subject).__name__} does not claim {self.protocol.__name__}"
            )
        if self._check is not None:
            return _as_feasibility(self._check(call, result), self._check, f"route {self.name!r}")
        return _capability_guard(subject, self.method)

    def execute(self, call: BoundCall, result: OutputSpec | None) -> Any:
        """The capability called on the call's other arguments, or the given execute."""
        if self._execute is not None:
            return self._execute(call, result)
        subject = call.operands[self.operand]
        others = [value for name, value in call.operands.items() if name != self.operand]
        return getattr(subject, self.method)(*others)


class _RegistryRoute(_Route):
    """A route that delegates to a dispatch registry, whose selected method realizes the call.

    The registry receives the call's arguments in signature order, or those
    *arguments* returns, with the keyword options *options* returns, or else the
    budget controls the call sets. Its exactness is that of the method the
    registry selects, so the route is ranked twice: its exact methods with the
    exact routes and its approximate methods with the approximate ones.

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
        controls: Iterable[str] | None = None,
    ) -> None:
        super().__init__(name, exact=None, controls=controls)
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
        return {
            name: value
            for name, value in call.operation._budgets().items()
            if self.controls is None or name in self.controls
        }

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


def _identity_route(declaration: Callable[..., Any], signature: inspect.Signature) -> _CheckedRoute:
    """The defining identity of a derived operation, as a fallback on its own domain.

    Its exactness is that of the implementations its constituent operations
    select, so it declares none.
    """

    def feasible(call: BoundCall, result: OutputSpec | None) -> bool:
        """The constituent operations decide the call when the identity runs."""
        return True

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
# Selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """One way a route may realize a call: the route at one exactness.

    A registry route contributes one candidate for its exact methods and one for
    its approximate ones; ``method`` is the registry method a ``method=`` control
    names, whose exactness the registry reports.
    """

    route: Any
    exact: bool | None
    index: int
    method: str | None = None

    @property
    def rank(self) -> tuple[int, int, int]:
        """Exact before approximate, a fallback below the other sources, then registration order."""
        return (
            0 if self.exact is True else 1,
            1 if self.route.source is RouteSource.FALLBACK else 0,
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


def _probe(candidate: _Candidate, call: BoundCall, result: OutputSpec | None) -> Feasibility:
    """The candidate's report for *call*, without executing anything."""
    route = candidate.route
    if isinstance(route, _RegistryRoute):
        return route.probe(
            call,
            method=candidate.method,
            exact_only=call.controls["exact_only"] or candidate.exact is True,
        )
    return route.check(call, result)


def _exactness(candidate: _Candidate, report: Feasibility) -> bool | None:
    """The exactness of the implementation *candidate* selects, as its report gives it."""
    if isinstance(candidate.route, _RegistryRoute) and isinstance(report, MethodInfo):
        return report.exact
    return candidate.route.exact


# ---------------------------------------------------------------------------
# The check report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallCheck(Feasibility):
    """What :meth:`Operation.check` reports about one call, without executing it.

    The call is feasible when a route is selected, unresolved when a route
    ranked above every feasible one still needs declarations, and infeasible
    when no route applies.

    Attributes
    ----------
    route : str or None
        The selected route, when selection can be decided.
    method : str or None
        The registry method a selected registry route delegates to.
    exact : bool or None
        The selected implementation's declared exactness.
    result : OutputSpec or None
        The result declaration planning derived.
    routes : tuple of (str, Feasibility)
        Each candidate that was probed and its report, in selection order.
    deferred : tuple of str
        The checks deferred to the return.
    """

    route: str | None = None
    method: str | None = None
    exact: bool | None = None
    result: OutputSpec | None = None
    routes: tuple[tuple[str, Feasibility], ...] = ()
    deferred: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


class _RouteTable:
    """The routes of one operation, shared by every view of it, in registration order."""

    def __init__(self) -> None:
        self.routes: list[Any] = []

    def add(self, route: Any, owner: str) -> None:
        """Append *route*.

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

    An unannotated parameter and one annotated ``Any`` accept every kind, a term
    class accepts its own kind, and any other annotation, such as ``str`` for a
    field path, selects rather than supplies and accepts none.
    """
    if hint is inspect.Parameter.empty or hint is Any:
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
            raise TypeError(f"the role of {owner}'s {name!r} must list TermSpec subclasses")
        derived[name] = accepted
    return derived


def _kind_names(kinds: Iterable[type]) -> str:
    """The names of *kinds*, comma-separated."""
    return ", ".join(kind.__name__ for kind in kinds)


class Operation(Function):
    """A Function realized by routes, with operand roles, conditions, and a result rule.

    Calling it runs the engine's stack, whose admission, planning, resolution,
    and return steps read the declarations below: each argument is admitted by
    its role, the applicability conditions and the result rule derive the
    result's declaration, a route is selected, and its raw result is wrapped at
    the declared kind. The routes are registered after construction with the
    four helpers or :meth:`register_route`, and a derived operation also carries
    its defining identity as a route.

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

    Raises
    ------
    TypeError
        If *declaration*, *result*, or a condition is not callable, the result
        rule or a condition reads a name the declaration does not declare, or a
        role is malformed.
    """

    def __init__(
        self,
        declaration: Callable[..., Any],
        *,
        result: Callable[..., OutputSpec | None],
        conditions: Iterable[Callable[..., Any]] = (),
        roles: Mapping[str, Iterable[type[TermSpec]]] | None = None,
    ) -> None:
        if not callable(declaration):
            raise TypeError(f"an operation is declared by a function; got {declaration!r}")
        super().__init__(declaration.__name__, declaration)
        owner = f"operation {self.name!r}"
        if not callable(result):
            raise TypeError(f"{owner} needs a callable result rule; got {result!r}")
        conditions = tuple(conditions)
        if not all(callable(condition) for condition in conditions):
            raise TypeError(f"{owner} was given an applicability condition that is not callable")
        parameters = self.signature.parameters
        read = {parameter for rule in (result, *conditions) for parameter in _parameter_names(rule)}
        unknown = read - set(parameters)
        if unknown:
            raise TypeError(f"{owner} has a result rule or condition reading {sorted(unknown)}")
        derived = not _has_empty_body(declaration)
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
        set_attribute(self, "_controls", MappingProxyType({}))
        if derived:
            self.register_route(_identity_route(declaration, self.signature))

    # -- declarations ------------------------------------------------------

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

        Raises
        ------
        TypeError
            If *route* does not have the members of :class:`OperationRoute`.
        ValueError
            If the operation already has a route of that name.
        """
        self._route_table.add(route, f"operation {self.name!r}")
        return route

    def structural_route(
        self,
        name: str,
        *,
        check: Callable[[BoundCall, OutputSpec | None], Any],
        execute: Callable[[BoundCall, OutputSpec | None], Any],
        exact: bool,
        controls: Iterable[str] = (),
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
        controls : iterable of str
            The budget controls the route reads.

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
                controls=controls,
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
        controls: Iterable[str] = (),
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
        controls : iterable of str
            The budget controls the route reads.

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
            raise TypeError(f"operation {self.name!r} has no parameter {operand!r}")
        return self.register_route(
            _CapabilityRoute(
                name,
                operand=operand,
                protocol=protocol,
                method=method,
                exact=exact,
                check=check,
                execute=execute,
                controls=controls,
            )
        )

    def registry_route(
        self,
        name: str,
        *,
        registry: BaseDispatchRegistry[Any],
        arguments: Callable[[BoundCall], Iterable[Any]] | None = None,
        options: Callable[[BoundCall], Mapping[str, Any]] | None = None,
        controls: Iterable[str] | None = None,
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
            default the budget controls the call sets.
        controls : iterable of str, optional
            The budget controls the registry's methods read; ``None``, the
            default, admits and forwards any control the call sets.

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
            _RegistryRoute(
                name, registry=registry, arguments=arguments, options=options, controls=controls
            )
        )

    def fallback_route(
        self,
        name: str,
        *,
        check: Callable[[BoundCall, OutputSpec | None], Any],
        execute: Callable[[BoundCall, OutputSpec | None], Any],
        exact: bool,
        controls: Iterable[str] = (),
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
        controls : iterable of str
            The budget controls the route reads.

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
                controls=controls,
            )
        )

    # -- controls ----------------------------------------------------------

    def with_options(self, **controls: Any) -> Operation:
        """Return a view with revised controls, the operation's own included.

        The framework's controls are those of :meth:`Function.with_options`. The
        operation adds ``method``, a route's name or a method of a registry
        route's registry; ``exact_only``, which excludes every approximate
        route; ``raw``, which returns the result detached; and the budgets its
        routes declare. ``None`` leaves a control unchanged.

        Raises
        ------
        TypeError
            If a control is unknown, ``method`` is not a string, or ``exact_only``
            or ``raw`` is not a bool; or on a framework control's own error.
        ValueError
            On a framework control's own error.
        """
        framework = {name: value for name, value in controls.items() if name in self.options}
        own = {name: value for name, value in controls.items() if name in _OPERATION_CONTROLS}
        budgets = {
            name: value
            for name, value in controls.items()
            if name not in framework and name not in own
        }
        unknown = self._unknown_budgets(budgets)
        if unknown:
            raise TypeError(f"Unknown controls for operation {self.name!r}: {sorted(unknown)}")
        method = own.get("method")
        if method is not None and (not isinstance(method, str) or not method):
            raise TypeError(f"method must be a route or method name; got {method!r}")
        for flag in ("exact_only", "raw"):
            if own.get(flag) is not None and type(own[flag]) is not bool:
                raise TypeError(f"{flag} must be a bool; got {own[flag]!r}")
        clone = super().with_options(**framework)
        revised = dict(self._controls)
        revised.update({name: value for name, value in own.items() if value is not None})
        revised.update({name: value for name, value in budgets.items() if value is not None})
        object.__setattr__(clone, "_controls", MappingProxyType(revised))
        return clone

    def _unknown_budgets(self, budgets: Mapping[str, Any]) -> set[str]:
        """The names in *budgets* no route declares, empty when a route admits any."""
        declared: set[str] = set()
        for route in self._route_table.routes:
            names = getattr(route, "controls", ())
            if names is None:
                return set()
            declared.update(names)
        return set(budgets) - declared

    def _budgets(self) -> dict[str, Any]:
        """The budget controls this view sets."""
        return {
            name: value for name, value in self._controls.items() if name not in _OPERATION_CONTROLS
        }

    def _resolved_controls(self) -> Mapping[str, Any]:
        """Every control's effective value for a call through this view."""
        return MappingProxyType({**self.options, **_OPERATION_CONTROLS, **self._controls})

    # -- the call ----------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the engine's stack on the call, or return the result detached under ``raw``."""
        if self._controls.get("raw", False):
            return self.apply(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def raw(self) -> Callable[..., Any]:
        """The evaluator that realizes one call with no lifting, tracking, or provenance."""
        return self.apply

    def _invoke_resolved(self, values: Mapping[str, Any], *, context: Any) -> Any:
        """Admit, plan, resolve, and execute one point of the call, then wrap its result."""
        call = BoundCall(self, MappingProxyType(dict(values)), self._resolved_controls())
        self._admit(call)
        result, _ = self._plan(call)
        candidate, report, reports = self._select(call, result)
        if candidate is None:
            raise ResolutionError(self._no_route_message(call, reports))
        if report.feasible is None:
            raise ResolutionError(
                f"{self.name}: route {candidate.label!r} is unresolved; pending: "
                f"{', '.join(report.pending)}"
            )
        route = candidate.route
        if isinstance(route, _RegistryRoute):
            value = route.run(
                call,
                method=report.method_name if isinstance(report, MethodInfo) else candidate.method,
                exact_only=call.controls["exact_only"] or candidate.exact is True,
            )
        else:
            value = route.execute(call, result)
        if call.controls["raw"]:
            return value
        return _wrap_result(value, result, self.output_name)

    def check(self, *args: Any, **kwargs: Any) -> CallCheck:
        """Report how a call would resolve, without executing any route.

        The arguments bind as the call's would, and admission and planning raise
        as the call's would, since their failures are properties of the call. The
        routes are probed in selection order until one is feasible or
        unresolved.

        Returns
        -------
        CallCheck
            The selected route, its exactness, the result declaration, each probed
            candidate's report, and the checks deferred to the return.

        Raises
        ------
        TypeError
            If the arguments do not bind to the signature.
        ApplicabilityError
            If an argument's kind is not accepted or a condition fails.
        ResolutionError
            If ``method`` names neither a route nor a registry method.
        """
        bound = self.signature.bind_partial(*args, **kwargs)
        values = resolve_workflow_values(
            self._signature_info,
            dict(bound.arguments),
            bind=self._bind,
            module=self._module,
            dependency_type=Node,
            workflow_name=self.name,
        )
        call = BoundCall(self, MappingProxyType(values), self._resolved_controls())
        self._admit(call)
        result, deferred = self._plan(call)
        candidate, report, reports = self._select(call, result)
        probed = tuple((probed_candidate.label, probe) for probed_candidate, probe in reports)
        if candidate is None:
            return CallCheck(
                False,
                self._no_route_message(call, reports),
                result=result,
                routes=probed,
                deferred=deferred,
            )
        if report.feasible is None:
            return CallCheck(
                None, pending=report.pending, result=result, routes=probed, deferred=deferred
            )
        registry = isinstance(candidate.route, _RegistryRoute)
        return CallCheck(
            True,
            route=candidate.route.name,
            method=report.method_name if registry and isinstance(report, MethodInfo) else None,
            exact=_exactness(candidate, report),
            result=result,
            routes=probed,
            deferred=deferred,
        )

    # -- the operation's side of the stack's steps ---------------------------

    def _admit(self, call: BoundCall) -> None:
        """Check each supplying argument's kind against its role.

        Raises
        ------
        ApplicabilityError
            If an argument's kind is not one its role accepts.
        """
        for name, accepts in self._roles.items():
            if not accepts or name not in call.operands:
                continue
            value = call.operands[name]
            parameter = self.signature.parameters[name]
            if value is None and parameter.default is None:
                continue
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                items = tuple(value.values())
            elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                items = tuple(value)
            else:
                items = (value,)
            for item in items:
                kind = type(_spec_of(item))
                if not issubclass(kind, accepts):
                    raise ApplicabilityError(
                        f"{self.name}: {name!r} accepts {_kind_names(accepts)}, but received a "
                        f"{type(item).__name__}, whose kind is {kind.__name__}"
                    )

    def _plan(self, call: BoundCall) -> tuple[OutputSpec | None, tuple[str, ...]]:
        """Check the applicability conditions, then derive the result's declaration.

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
                condition(**arguments), condition, f"{self.name} condition {condition.__name__}"
            )
            if report.feasible is False:
                raise ApplicabilityError(report.description)
            deferred.extend(report.pending)
        result = self._result_rule(**{name: declarations[name] for name in self._rule_parameters})
        if result is not None and not isinstance(result, OutputSpec):
            raise TypeError(
                f"the result rule of {self.name!r} returned {result!r}; it returns an "
                f"OutputSpec or None"
            )
        if result is None or result.spec is None:
            deferred.append("the result's type is completed from the returned value")
        return result, tuple(deferred)

    def _candidates(self, controls: Mapping[str, Any]) -> list[_Candidate]:
        """The ways the routes may realize a call under *controls*, in selection order.

        Raises
        ------
        ResolutionError
            If ``method`` names neither a route nor a method of a registry route's
            registry, or names an approximate route while ``exact_only`` is set.
        """
        routes = list(self._route_table.routes)
        method, exact_only = controls["method"], controls["exact_only"]
        candidates: list[_Candidate] = []
        if method is None:
            for index, route in enumerate(routes):
                if isinstance(route, _RegistryRoute):
                    candidates += [_Candidate(route, True, index), _Candidate(route, False, index)]
                else:
                    candidates.append(_Candidate(route, route.exact, index))
        else:
            named = [(index, route) for index, route in enumerate(routes) if route.name == method]
            holders = [
                (index, route)
                for index, route in enumerate(routes)
                if isinstance(route, _RegistryRoute) and method in route.registry.list_methods()
            ]
            if named:
                index, route = named[0]
                if isinstance(route, _RegistryRoute):
                    candidates = [_Candidate(route, True, index), _Candidate(route, False, index)]
                else:
                    candidates = [_Candidate(route, route.exact, index)]
                    if exact_only and route.exact is not True:
                        raise ResolutionError(
                            f"{self.name}: route {method!r} is not exact and exact_only was "
                            f"requested"
                        )
            elif holders:
                index, route = holders[0]
                candidates = [_Candidate(route, None, index, method)]
            else:
                available = ", ".join(route.name for route in routes) or "none"
                raise ResolutionError(
                    f"{self.name}: no route or registered method named {method!r}; routes: "
                    f"{available}"
                )
        if exact_only:
            candidates = [c for c in candidates if c.exact is True or c.method is not None]
        return sorted(candidates, key=lambda candidate: candidate.rank)

    def _select(
        self, call: BoundCall, result: OutputSpec | None
    ) -> tuple[_Candidate | None, Feasibility | None, list[tuple[_Candidate, Feasibility]]]:
        """Probe the candidates in selection order until one is feasible or unresolved.

        Returns
        -------
        tuple
            The first candidate that is not infeasible and its report, both
            ``None`` when every candidate is infeasible, and every probed
            candidate with its report.
        """
        reports: list[tuple[_Candidate, Feasibility]] = []
        for candidate in self._candidates(call.controls):
            report = _probe(candidate, call, result)
            reports.append((candidate, report))
            if report.feasible is not False:
                return candidate, report, reports
        return None, None, reports

    def _no_route_message(
        self, call: BoundCall, reports: list[tuple[_Candidate, Feasibility]]
    ) -> str:
        """The ResolutionError message listing each probed candidate and its reason."""
        restriction = " with exact_only" if call.controls["exact_only"] else ""
        if not reports:
            return f"{self.name}: no route applies{restriction}; none is registered"
        tried = "; ".join(
            f"{candidate.label}: {report.description or 'infeasible'}"
            for candidate, report in reports
        )
        return f"{self.name}: no route applies{restriction}. Tried: {tried}"

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
            name=self.name,
            supported_types=operands[0].accepts if operands else (),
            description=doc.split("\n\n", 1)[0].replace("\n", " "),
            module_path=self.__module__,
            operands=operands,
            is_derived=self.is_derived,
            identity=self.identity,
            routes=routes,
        )


# ---------------------------------------------------------------------------
# Randomness and the return
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
    operation_kind, execution_mode : str
        The event's identity within the call, recorded in the stochastic plan.

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


def _wrap_result(value: Any, declared: OutputSpec | None, label: str) -> Any:
    """*value* validated against *declared* and wrapped at its kind, labeled *label*.

    A type hole is completed from *value*, and an undeclared result wraps by the
    kind-directed table.

    Raises
    ------
    ValueError
        If *value* does not satisfy the declaration.
    """
    if declared is not None and isinstance(declared.spec, BatchSpec):
        return _batch_at(value, declared.spec, label)
    if declared is None:
        return _wrap_as_term(value, label)
    completed = _validate_function_output(
        function_name=label, output_spec=declared, result=value, bindings={}
    )
    return _wrap_declared_function_output(value, function_name=label, output_spec=completed)


def _batch_at(value: Any, spec: BatchSpec, label: str) -> Any:
    """*value*, whose leading axes range over *spec*'s levels, as the batch *spec* declares.

    A batch is kept as it is. Record columns and stacked arrays become the
    batch form of their element kind, whose element declaration is read from
    the value and unified with the declared one, and an object array becomes
    the batch form the kind table records for the declared element.

    Raises
    ------
    ValueError
        If the leading axes are not the declared batch shape, the element does
        not unify with the declared element, or *value* has no batch form.
    """
    batch_shape = tuple(spec.batch_shape)
    ranks = _ranks_of(spec.axis_groups)
    levels = tuple(spec.level_names)
    n_axes = len(batch_shape)

    def require_leading(shape: tuple[Any, ...]) -> None:
        if tuple(shape[:n_axes]) != batch_shape:
            raise ValueError(
                f"{label}: the result's leading axes {tuple(shape[:n_axes])} are not the "
                f"declared batch shape {batch_shape}"
            )

    if isinstance(value, Batch):
        require_leading(tuple(value.batch_shape))
        return value
    if isinstance(value, Record):
        template = value.event_template
        columns = {path: value[path] for path in template}
        for column in columns.values():
            require_leading(tuple(_event_shape_of(column)))
        element = _reshaped_template(template, lambda shape: shape[n_axes:])
        _unify_specs(spec.element_spec, element, {}, f"{label} element")
        return _batch_class_for(element)(
            label, columns, levels, element_spec=element, axes_per_level=ranks
        )
    if _is_object_array(value):
        require_leading(value.shape)
        batch_class = batch_class_for_spec(spec.element_spec)
        if batch_class is not None and batch_class is not NumericArrayBatch:
            return batch_class(
                label, value, levels, element_spec=spec.element_spec, axes_per_level=ranks
            )
        return _make_stack(
            list(value.reshape((prod(batch_shape), *value.shape[n_axes:]))),
            batch_shape=batch_shape,
            axis_groups=tuple(spec.axis_groups),
            level_names=levels,
            field_name=label,
            name=label,
        )
    if _is_numeric_leaf(value):
        shape = tuple(_event_shape_of(value))
        require_leading(shape)
        declared = spec.element_spec
        support = declared.support if isinstance(declared, NumericArraySpec) else None
        element = NumericArraySpec(shape[n_axes:], _numpy_dtype_of(value), support)
        _unify_specs(declared, element, {}, f"{label} element")
        return NumericArrayBatch(label, value, levels, element_spec=element, axes_per_level=ranks)
    raise ValueError(f"{label}: a {type(value).__name__} has no form as the declared batch")


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

        Raises
        ------
        TypeError
            If *op* is not an operation.
        ValueError
            If an operation of that name is registered.
        """
        if not isinstance(op, Operation):
            raise TypeError(f"only an operation registers here; got {type(op).__name__}")
        if op.name in self._operations:
            raise ValueError(f"an operation named {op.name!r} is already registered")
        self._operations[op.name] = op

    def list(self) -> list[OperationSummary]:
        """One summary per operation, in registration order."""
        return [op.summary() for op in self._operations.values()]

    def describe(self, name: str | None = None) -> str:
        """The summaries as text, for the operation *name* or for all of them.

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
            available = ", ".join(self._operations) or "none"
            raise KeyError(f"no operation named {name!r}; registered: {available}") from None


operation_registry: OperationRegistry = OperationRegistry()
"""The global registry of operations."""


def operation(
    *,
    result: Callable[..., OutputSpec | None],
    conditions: Iterable[Callable[..., Any]] = (),
    roles: Mapping[str, Iterable[type[TermSpec]]] | None = None,
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
        op = Operation(declaration, result=result, conditions=conditions, roles=roles)
        (operation_registry if registry is None else registry).register(op)
        return op

    return decorate
