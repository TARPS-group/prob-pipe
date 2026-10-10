"""The evaluate operation: applying a map to a value, a distribution, or a batch.

``evaluate(f, v)`` is the operation form of the engine's resolution step: for a
value it returns ``f(v)``, for a distribution the pushforward law, and for a
batch the elementwise result. Its route is the evaluation-rule registry, keyed
on the map's and the operand's types, which the direct call ``f(v)`` also
takes.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from .._messages import unknown_names
from ..core._dispatch import Feasibility
from ..core._spec_base import TermSpec
from ..core._specs import OutputSpec
from ..functions import _plan, _rules, function
from ..functions._call import ApplicabilityError
from ..functions._resolution import PointReport
from ..values import Function, FunctionSpec
from ._operation import BoundCall, _install_expression_rule, _RegistryRoute, operation

__all__ = ["evaluate"]


def _evaluate_result(f: Any, v: Any, fixed_args: Any) -> OutputSpec | None:
    """The map's output declaration, under the substitution that unifying the operand gives.

    The kind depends on the operand, a law for a distribution and a batch for a
    batch, so the returned value completes it.
    """
    return None


def _map_output_label(f: Any) -> str:
    """The map's output label, which the map's own result takes (V.10)."""
    return f.output_label if isinstance(f, Function) else "evaluate"


def _mapped_expression() -> None:
    """None: the result carries the expression of the map's own call (II.4).

    A value's result is labeled by the map's output label, and the pushforward
    of a law is the map applied to a draw of the law, as ``f(mu ~ d)``.
    """
    return None


@operation(
    result=_evaluate_result,
    roles={"f": (FunctionSpec,), "v": (TermSpec,)},
    label=_map_output_label,
)
def evaluate(f: Any, v: Any, fixed_args: Mapping[str, Any] | None = None):
    """Apply the map *f* to the operand *v*.

    Parameters
    ----------
    f : Function
        The map.
    v : Any
        A value, a distribution, or a batch, bound to the one parameter
        *fixed_args* leaves open.
    fixed_args : mapping of str to Any, optional
        The map's other parameters, by name.

    Returns
    -------
    TrackedTerm
        ``f(v)`` for a value, the pushforward law for a distribution, and the
        elementwise result for a batch, labeled by the map's output label. The
        pushforward displays as the map applied to a draw of *v*, as
        ``f(mu ~ prior)``.

    Raises
    ------
    ResolutionError
        If no evaluation rule applies.
    """


def _bound_parameter(f: Function, fixed_args: Mapping[str, Any] | None) -> str:
    """The one parameter of *f* that *fixed_args* leaves open, which the operand binds.

    Parameters
    ----------
    f : Function
        The map, whose signature lists its parameters.
    fixed_args : Mapping of str to Any or None
        The arguments of the map's other parameters, by name; ``None`` fixes
        none.

    Returns
    -------
    str
        The parameter's name. An open parameter is one without a default that
        is neither fixed nor variadic.

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
            f"evaluate: fixed_args of {f.label!r}: "
            f"{unknown_names('parameter', unknown, list(parameters))}"
        )
    open_parameters = [
        name
        for name, parameter in parameters.items()
        if name not in fixed
        and parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if not open_parameters:
        raise ApplicabilityError(
            f"evaluate: no parameter of {f.label!r} is left open for the value to bind; "
            f"exactly one parameter must have no default and be absent from fixed_args"
        )
    if len(open_parameters) > 1:
        raise ApplicabilityError(
            f"evaluate: {f.label!r} has {len(open_parameters)} open parameters "
            f"{open_parameters}, but exactly one parameter may stay open; pass the others in "
            f"fixed_args, such as fixed_args={{{open_parameters[1]!r}: ...}}"
        )
    return open_parameters[0]


def _as_function(f: Any) -> Function:
    """The map as a ``Function``: a plain callable is wrapped under its own name."""
    return f if isinstance(f, Function) else function(f)


def _call_values(call: BoundCall) -> tuple[Function, str, Any, dict[str, Any]]:
    """The map, the parameter the operand binds, the operand, and the fixed arguments."""
    f, operand = _as_function(call.operands["f"]), call.operands["v"]
    fixed = dict(call.operands.get("fixed_args") or {})
    return f, _bound_parameter(f, fixed), operand, fixed


def _lifts(f: Function, parameter: str, operand: Any) -> bool:
    """Whether the direct call lifts *operand* at *parameter*: a law sampled or a batch swept."""
    return _plan.lift_at(f, parameter, operand) != "whole"


#: The engine controls evaluate forwards to the direct call of its map when its caller
#: set them, beside the rule the route selects and its exactness.
_FORWARDED_CONTROLS = (
    "n_broadcast_samples",
    "dispatch",
    "max_workers",
    "include_inputs",
    "workflow_kind",
    "method_options",
)


class _EvaluationRules(_RegistryRoute):
    """The evaluation-rule registry, from which the direct call ``f(v)`` also selects.

    The probe asks the registry for a rule for the map's and the operand's
    types, and an operand the direct call does not lift is applied by the map's
    body, which is exact. The run is the direct call of the map under the
    controls evaluate forwards, the rule the probe selected or the ``method``
    control names, and the candidate's exactness, so ``evaluate`` and the
    direct call take the same controls (VI.1).
    """

    #: The controls the run forwards to the direct call.
    _forwarded: tuple[str, ...] = _FORWARDED_CONTROLS

    def __init__(self) -> None:
        super().__init__("evaluation_rules", registry=_rules.evaluation_rule_registry)

    def _values(self, call: BoundCall) -> tuple[Function, str, Any, dict[str, Any]]:
        """The map, the parameter the operand binds, the operand, and the fixed arguments."""
        return _call_values(call)

    @property
    def condition(self) -> str:
        """The registry selects a rule, unless the direct call lifts nothing."""
        return (
            "The evaluation-rule registry selects a rule for the map's and the operand's "
            "types; an operand the direct call does not lift is applied by the map's body."
        )

    def probe(self, call: BoundCall, *, method: str | None, exact_only: bool) -> Feasibility:
        """The registry's report for the lifted operand, or the body's for one it does not lift.

        A rule named for an operand the direct call does not lift is refused, as
        the direct call refuses it.
        """
        f, parameter, operand, fixed = self._values(call)
        if not _lifts(f, parameter, operand):
            if method is not None:
                return Feasibility(
                    False,
                    f"the call lifts nothing, so the body of {f.label!r} realizes it and the "
                    f"rule {method!r} does not",
                )
            return PointReport(True, exact=True)
        return self.registry.check(
            f,
            operand,
            method=method,
            exact_only=exact_only,
            parameter=parameter,
            fixed_args=fixed,
            controls={**f.options, **self._forwarded_controls(call)},
        )

    def _forwarded_controls(self, call: BoundCall) -> dict[str, Any]:
        """The controls the caller set on the operation, which the call of the map takes over its own.

        A control the caller left unset keeps the map's value, so a map
        constructed with its own sample count draws that many.
        """
        set_controls = call.operation._options
        return {name: set_controls[name] for name in self._forwarded if name in set_controls}

    def run(self, call: BoundCall, *, method: str | None, exact_only: bool) -> Any:
        """The direct call of the map on the operand, by the rule *method* names, if any."""
        f, parameter, operand, fixed = self._values(call)
        forwarded = self._forwarded_controls(call)
        forwarded["exact_only"] = exact_only
        if method is not None:
            forwarded["method"] = method
        return f.with_options(**forwarded)(**{parameter: operand}, **fixed)


evaluate.register_route(_EvaluationRules())
_install_expression_rule(evaluate, _mapped_expression)
