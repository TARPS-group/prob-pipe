"""The pushforward operation: the law of a map applied to a draw of a law.

``pushforward(d, f)`` returns the law ``f♯d`` of ``f(X)`` for ``X ~ d``. The
map always receives draws of the law, one at a time, whatever its parameter's
annotation; the direct call ``f(d)`` instead passes a law whole to a parameter
annotated ``Distribution`` or ``Any`` (V.5). Its route is the evaluation-rule
registry, keyed on the map's and the law's types, from which the direct call
also selects.

The two applicability conditions of the pushforward, which ``expectation``
shares, are defined here: the map's open parameter admits a draw of the law,
and its fixed arguments are constants.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from .._messages import unknown_names
from ..core._batch import Batch, BatchSpec
from ..core._dispatch import Feasibility
from ..core._record_spec import RecordSpec
from ..core._repr import type_name
from ..core._spec_base import NumericArraySpec, TermSpec, _unify_specs
from ..core._specs import OutputSpec
from ..core.tracked import _callable_label
from ..distributions._conditional import ConditionalDistributionSpec
from ..distributions._distribution import Distribution, DistributionSpec
from ..functions import _normalization, _plan, _rules, function
from ..functions._call import ApplicabilityError, _consumes_kernel, role_admits
from ..values import Function, FunctionSpec
from ..values._binding import resolved_hint, values_to_bound_arguments
from ._operation import (
    BoundCall,
    Operation,
    _install_expression_rule,
    _kinds_for,
    _RegistryRoute,
    _spec_of,
    operation,
)

__all__ = ["pushforward"]


# ---------------------------------------------------------------------------
# The map and its open parameter
# ---------------------------------------------------------------------------


#: How the caller of an operation that takes no label labels a callable without a name.
NAMELESS_CALLABLE_FIX = "wrap it as Function(f, label=...)"


def _as_function(f: Any, *, subject: str = "pushforward") -> Function:
    """The map as a ``Function``: a plain callable is wrapped under its own name.

    Parameters
    ----------
    f : Function or callable
        The map, which a ``Function`` gives as it is.
    subject : str
        The operation the map is an argument of, which the error names.

    Returns
    -------
    Function
        *f*, or *f* wrapped under its name, ``f`` for a lambda.

    Raises
    ------
    TypeError
        If *f* is a plain callable without a name, such as a ``functools.partial``.
    """
    if isinstance(f, Function):
        return f
    return function(f, label=_callable_label(f, subject=subject, fix=NAMELESS_CALLABLE_FIX))


def _bound_parameter(
    f: Function, fixed_args: Mapping[str, Any] | None, *, subject: str = "pushforward"
) -> str:
    """The one parameter of *f* that *fixed_args* leaves open, which a draw binds.

    Parameters
    ----------
    f : Function
        The map, whose signature lists its parameters.
    fixed_args : Mapping of str to Any or None
        The arguments of the map's other parameters, by name; ``None`` fixes
        none.
    subject : str
        The operation the map is an argument of, which the error names.

    Returns
    -------
    str
        The parameter's name. An open parameter is one without a default that
        is neither fixed, bound by the map's construction ``bind``, nor
        variadic.

    Raises
    ------
    ApplicabilityError
        If *fixed_args* names a parameter *f* lacks, or leaves other than one
        parameter open.
    """
    fixed = dict(fixed_args or {})
    parameters = f.signature.parameters
    unknown = sorted(set(fixed) - set(parameters))
    if unknown:
        raise ApplicabilityError(
            f"{subject}: fixed_args of {f.label!r}: "
            f"{unknown_names('parameter', unknown, list(parameters))}"
        )
    open_parameters = [
        name
        for name, parameter in parameters.items()
        if name not in fixed
        and name not in f._bind
        and parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if not open_parameters:
        raise ApplicabilityError(
            f"{subject}: no parameter of {f.label!r} is left open for a draw to bind; "
            f"exactly one parameter must have no default and be absent from fixed_args"
        )
    if len(open_parameters) > 1:
        raise ApplicabilityError(
            f"{subject}: {f.label!r} has {len(open_parameters)} open parameters "
            f"{open_parameters}, but exactly one parameter may stay open; pass the others in "
            f"fixed_args, such as fixed_args={{{open_parameters[1]!r}: ...}}"
        )
    return open_parameters[0]


def _call_at(
    body: Callable[..., Any], f: Function, parameter: str, value: Any, constants: Mapping[str, Any]
) -> Any:
    """*body* called with *value* at *parameter* and *constants* at theirs, bound by *f*'s signature.

    Binding by the signature passes a positional-only parameter positionally,
    as the direct call does, so a map such as ``jnp.exp`` takes its argument.
    """
    bound = values_to_bound_arguments(f.signature, {**constants, parameter: value})
    return body(*bound.args, **bound.kwargs)


# ---------------------------------------------------------------------------
# The applicability conditions
# ---------------------------------------------------------------------------

#: Each kind a message names, in plain words, the more specific kinds first.
_KIND_WORDS: tuple[tuple[type[TermSpec], str], ...] = (
    (ConditionalDistributionSpec, "a conditional distribution"),
    (DistributionSpec, "a distribution"),
    (BatchSpec, "a batch"),
    (RecordSpec, "a record"),
    (NumericArraySpec, "an array"),
    (FunctionSpec, "a function"),
)


def _kind_words(kind: type[TermSpec]) -> str:
    """The kind *kind* in plain words, as a message names it."""
    for spec_class, words in _KIND_WORDS:
        if issubclass(kind, spec_class):
            return words
    return f"a value of kind {kind.__name__}"


def _arm_admits(arm: Any, event: TermSpec) -> bool:
    """Whether the annotation arm *arm* admits a draw declared by *event*.

    An arm that names a kernel class or capability admits a kernel, one that
    names a law class or capability admits a law, and one that names a
    ProbPipe term class admits a value of its kind. ``None`` admits nothing,
    and any other arm, such as ``Any``, a value class, or a class of raw
    values, admits every draw.
    """
    if arm is None or arm is type(None):
        return False
    if _consumes_kernel(arm):
        return isinstance(event, ConditionalDistributionSpec)
    if _normalization.is_distribution_hint(arm):
        return isinstance(event, DistributionSpec) and not isinstance(
            event, ConditionalDistributionSpec
        )
    kinds = _kinds_for(arm)
    return not kinds or role_admits(kinds, type(event))


def _arm_takes(arm: Any) -> str:
    """What the annotation arm *arm* takes, in plain words, for an arm that names a kind."""
    if _consumes_kernel(arm):
        return "a conditional distribution"
    if _normalization.is_distribution_hint(arm):
        return "a distribution"
    return " or ".join(_kind_words(kind) for kind in _kinds_for(arm))


def _domain_refusal(f: Function, parameter: str, event: TermSpec, fixed: Mapping[str, Any]) -> str:
    """Why the open *parameter* of *f* cannot take a draw declared by *event*, or ``""``.

    Each declaration the parameter carries must admit the draw: its slot of
    *f*'s ``input_spec``, unified with the draw together with the fixed
    arguments' slots; its role, for an operation; and otherwise its annotation,
    whose arms are read by :func:`_arm_admits`.
    """
    draw = f"the draw is {_kind_words(type(event))}"
    if f.input_spec is not None and parameter in f.input_spec:
        bindings: dict[str, int] = {}
        try:
            _unify_specs(f.input_spec[parameter], event, bindings, parameter)
        except (TypeError, ValueError) as error:
            return f"the draw does not match the input_spec of parameter {parameter!r}: {error}"
        for name, value in fixed.items():
            if name not in f.input_spec or value is None:
                continue
            try:
                _unify_specs(f.input_spec[name], _spec_of(value), bindings, name)
            except (TypeError, ValueError) as error:
                return (
                    f"fixed_args[{name!r}] does not match the input_spec of parameter "
                    f"{name!r} beside the draw: {error}"
                )
    role = f._roles.get(parameter)
    if role is not None:
        if role and not role_admits(role, type(event)):
            takes = " or ".join(_kind_words(kind) for kind in role)
            return f"its parameter {parameter!r} takes {takes}, but {draw}"
        return ""
    hints = f._signature_info.hints
    if parameter not in hints or resolved_hint(hints[parameter]) is Any:
        return ""
    arms = _normalization._arms(hints[parameter])
    if any(_arm_admits(arm, event) for arm in arms):
        return ""
    takes = " or ".join(dict.fromkeys(_arm_takes(arm) for arm in arms)) or "no value"
    return f"its parameter {parameter!r} takes {takes}, but {draw}"


def _event_of(d: Any) -> TermSpec | None:
    """The spec of one draw of the law *d* declares, or ``None`` when it declares none."""
    event = getattr(d, "event_spec", None)
    return getattr(event, "spec", None)


def _lifted_fixed_argument(f: Function, fixed: Mapping[str, Any]) -> str:
    """Why a fixed argument of *f* is not a constant, or ``""``.

    A law the direct call would sample, or a batch it would sweep, at a
    parameter whose annotation or role does not take it as it is, is not a
    constant.
    """
    for name, value in fixed.items():
        if not isinstance(value, (Distribution, Batch)):
            continue
        if _plan.lift_at(f, name, value) == "whole":
            continue
        what = "a distribution" if isinstance(value, Distribution) else "a batch"
        label = getattr(value, "label", None)
        shown = type_name(value) if label is None else f"{type_name(value)} {label!r}"
        return f"fixed_args[{name!r}] must be a fixed value, but got {shown}, which is {what}"
    return ""


def _integrand_conditions(subject: str) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """The two applicability conditions of an operation that integrates or pushes forward a map.

    Parameters
    ----------
    subject : str
        The operation's name, which each refusal names.

    Returns
    -------
    tuple of callable
        The condition that the map's open parameter admits a draw of the law,
        and the condition that the map's fixed arguments are constants, each
        called with the operation's ``d``, ``f``, and ``fixed_args``.
    """

    def draw_admitted(d: Any, f: Any, fixed_args: Any) -> Any:
        """The map's open parameter admits a draw of the law."""
        event = _event_of(d)
        if event is None:
            return True
        fn = _as_function(f, subject=subject)
        fixed = dict(fixed_args or {})
        parameter = _bound_parameter(fn, fixed, subject=subject)
        reason = _domain_refusal(fn, parameter, event, fixed)
        if not reason:
            return True
        return Feasibility(
            False, f"{subject}: cannot pass a draw of argument 'd' to {fn.label!r}, since {reason}"
        )

    def fixed_arguments_constant(f: Any, fixed_args: Any) -> Any:
        """The map's fixed arguments are constants, which the call does not lift."""
        if not fixed_args:
            return True
        reason = _lifted_fixed_argument(_as_function(f, subject=subject), dict(fixed_args))
        return Feasibility(False, f"{subject}: {reason}") if reason else True

    return draw_admitted, fixed_arguments_constant


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


def _pushforward_result(f: Any) -> OutputSpec | None:
    """The result is a law over the map's output, which the returned law declares."""
    return None


def _output_label(f: Any, *, subject: str) -> str:
    """The map's output label: a ``Function``'s ``output_label``, or the name a callable is wrapped under.

    A lambda is labeled ``𝒻``, and a callable without a name raises the
    ``TypeError`` *subject*'s caller reads.
    """
    if isinstance(f, Function):
        return f.output_label
    return _callable_label(f, subject=subject, fix=NAMELESS_CALLABLE_FIX)


def _map_output_label(f: Any) -> str:
    """The map's output label, which the pushforward takes (V.10)."""
    return _output_label(f, subject="pushforward")


def _mapped_expression() -> None:
    """None: the result carries the expression of the map's call on the law (II.4).

    The pushforward of a law displays as the map applied to a draw of the law,
    as ``f(mu ~ d)``.
    """
    return None


@operation(
    result=_pushforward_result,
    roles={"f": (FunctionSpec,)},
    conditions=_integrand_conditions("pushforward"),
    label=_map_output_label,
)
def pushforward(d: Distribution, f: Any, fixed_args: Mapping[str, Any] | None = None):
    """The law of ``f(X)`` for ``X ~ d``, the pushforward ``f♯d``.

    The map receives one draw of *d* at a time, at its one open parameter,
    whatever that parameter's annotation is, so a parameter annotated ``Any`` or
    ``Distribution`` receives draws. The draw is presented as V.5 presents one
    draw. The direct call ``f(d)`` instead passes the law whole to such a
    parameter. The evaluation rules realize the pushforward, and
    ``with_options(method=...)`` names one, such as ``empirical_enumeration`` or
    ``sampling_lift``.

    Parameters
    ----------
    d : Distribution
        The law whose draws the map receives. A ``DistributionBatch`` gives the
        batch of the pushforwards of its laws.
    f : callable or Function
        The map. Exactly one of its parameters has no default and is absent
        from *fixed_args*, and each draw binds it whatever its name.
    fixed_args : mapping of str to Any, optional
        The map's other parameters, by name. Each is a constant: a law the
        direct call would sample, or a batch it would sweep, is refused. The
        direct call ``f(x=d, a=A)`` gives the joint law of the map over both.

    Returns
    -------
    Distribution or DistributionBatch
        The law of the map's output, labeled by the map's output label and
        displayed as the map applied to a draw of *d*, as ``f(mu ~ prior)``.
        A ``DistributionBatch`` operand gives the batch of the pushforwards of
        its laws, on its levels.

    Raises
    ------
    ApplicabilityError
        If *d* is not a law, *fixed_args* leaves other than one parameter of
        *f* open, the open parameter's declaration does not admit a draw of
        *d*, or a fixed argument is a law or a batch the call would lift. A
        kernel without ``given`` raises it as well, as an interim detail: the
        design's pushforward of a kernel, ``s ↦ f♯K(s, ·)``, is not implemented
        yet.
    ResolutionError
        If no evaluation rule applies.
    """


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

#: The engine controls the pushforward forwards to the call of its map when its
#: caller set them, beside the rule the route selects and its exactness. The
#: control ``include_inputs`` is not among them, since it would make the result
#: the law of the map's inputs and outputs together.
_FORWARDED_CONTROLS = (
    "n_broadcast_samples",
    "dispatch",
    "max_workers",
    "workflow_kind",
    "method_options",
)

#: The map's own controls that the map built around it keeps.
_MAP_CONTROLS = ("n_broadcast_samples", "dispatch", "max_workers", "workflow_kind")


def _draw_map(
    f: Function, parameter: str, d: Any, fixed: Mapping[str, Any], *, exact_only: bool
) -> tuple[Function, str, dict[str, Any]]:
    """The map the evaluation rules run on, the parameter a draw binds, and its fixed arguments.

    Where the direct call samples *d*, the map is *f* itself, so a rule keyed on
    *f*'s type applies. Where the direct call would pass *d* whole, and for an
    operation, whose ``method`` names its own routes rather than evaluation
    rules, the map is a new ``Function`` whose one unannotated parameter receives each draw and
    calls *f* on it with the fixed arguments. A map whose parameter is annotated
    ``Any`` is called on the draw directly, so it traces; any other is called
    as a direct call, so the draw goes through its admission, conversions, and
    routes. The new map keeps *f*'s label, output declaration, and sample count.
    """
    operation = isinstance(f, Operation)
    if not operation and _plan.lift_at(f, parameter, d) == "broadcast":
        return f, parameter, dict(fixed)
    hint = f._signature_info.hints.get(parameter)
    direct = operation or hint is None or resolved_hint(hint) is not Any
    target = f.with_options(exact_only=True) if direct and exact_only else f
    call = target if direct else f.raw()
    # The raw callable skips the construction ``bind``, which the direct call applies.
    constants = dict(fixed) if direct else {**f._bind, **fixed}

    def each_draw(draw: Any) -> Any:
        return _call_at(call, f, parameter, draw, constants)

    each_draw.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(parameter, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    each_draw.__name__ = f.label
    mapped = function(
        each_draw,
        label=f.label,
        output_spec=f.output_spec,
        output_label=f.output_label,
        **{name: f.options[name] for name in _MAP_CONTROLS},
    )
    return mapped, parameter, {}


class _EvaluationRules(_RegistryRoute):
    """The evaluation-rule registry, from which the direct call ``f(d)`` also selects.

    The probe asks the registry for a rule for the map's and the law's types,
    and the run is the direct call of the map on the law under the controls the
    pushforward forwards, the rule the probe selected or the ``method`` control
    names, and the candidate's exactness. Where the direct call would pass the
    law whole, the rules run on the map :func:`_draw_map` builds, so the map
    receives draws.
    """

    #: The controls the run forwards to the call of the map.
    _forwarded: tuple[str, ...] = _FORWARDED_CONTROLS

    def __init__(self, name: str = "evaluation_rules") -> None:
        super().__init__(name, registry=_rules.evaluation_rule_registry)

    @property
    def condition(self) -> str:
        """The registry selects a rule for the map's and the law's types."""
        return (
            "The evaluation-rule registry selects a rule for the map's and the law's types, "
            "and the map receives draws of the law."
        )

    def _values(
        self, call: BoundCall, exact_only: bool
    ) -> tuple[Function, str, Any, dict[str, Any]]:
        """The map the rules run on, the parameter a draw binds, the law, and the fixed arguments."""
        key = ("draw map", exact_only)
        if key not in call._memo:
            f = _as_function(call.operands["f"], subject=call.operation.label)
            fixed = dict(call.operands.get("fixed_args") or {})
            parameter = _bound_parameter(f, fixed, subject=call.operation.label)
            d = call.operands["d"]
            call._memo[key] = (*_draw_map(f, parameter, d, fixed, exact_only=exact_only), d)
        mapped, parameter, fixed, d = call._memo[key]
        return mapped, parameter, d, fixed

    def _forwarded_controls(self, call: BoundCall) -> dict[str, Any]:
        """The controls the caller set on the operation, which the call of the map takes over its own.

        A control the caller left unset keeps the map's value, so a map
        constructed with its own sample count draws that many.
        """
        set_controls = call.operation._options
        return {name: set_controls[name] for name in self._forwarded if name in set_controls}

    def probe(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Feasibility:
        """The registry's report for the law, at the parameter a draw binds."""
        mapped, parameter, d, fixed = self._values(call, exact_only)
        return self.registry.check(
            mapped,
            d,
            method=method,
            exact_only=exact_only,
            parameter=parameter,
            fixed_args=fixed,
            controls={**mapped.options, **self._forwarded_controls(call), "include_inputs": False},
        )

    def run(
        self, call: BoundCall, result: OutputSpec | None, *, method: str | None, exact_only: bool
    ) -> Any:
        """The direct call of the map on the law, by the rule *method* names, if any."""
        mapped, parameter, d, fixed = self._values(call, exact_only)
        forwarded = self._forwarded_controls(call)
        forwarded.update(exact_only=exact_only, include_inputs=False)
        if method is not None:
            forwarded["method"] = method
        return _call_at(mapped.with_options(**forwarded), mapped, parameter, d, fixed)


pushforward.register_route(_EvaluationRules())
_install_expression_rule(pushforward, _mapped_expression)
