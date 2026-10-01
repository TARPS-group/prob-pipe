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

from ..core._spec_base import TermSpec
from ..core._specs import OutputSpec
from ..functions import _plan, _rules
from ..functions._call import ApplicabilityError
from ..values import Function, FunctionSpec, _binding
from ._operation import BoundCall, RouteSource, _CheckedRoute, operation

__all__ = ["evaluate"]


def _evaluate_result(f: Any, v: Any, fixed_args: Any) -> OutputSpec | None:
    """The map's output declaration, under the substitution that unifying the operand gives.

    The kind depends on the operand, a law for a distribution and a batch for a
    batch, so the returned value completes it.
    """
    return None


@operation(result=_evaluate_result, roles={"f": (FunctionSpec,), "v": (TermSpec,)})
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
        elementwise result for a batch.

    Raises
    ------
    ResolutionError
        If no evaluation rule applies.
    """


def _bound_parameter(f: Function, fixed_args: Mapping[str, Any] | None) -> str:
    """The one parameter of *f* that *fixed_args* leaves open, which the operand binds.

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
            f"evaluate: fixed_args names {unknown}, which are not parameters of {f.name!r}"
        )
    open_parameters = [
        name
        for name, parameter in parameters.items()
        if name not in fixed
        and parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if len(open_parameters) != 1:
        raise ApplicabilityError(
            f"evaluate applies {f.name!r} over exactly one parameter, and fixed_args leaves "
            f"{open_parameters or 'none'} open; supply the others by name in fixed_args"
        )
    return open_parameters[0]


def _call_values(call: BoundCall) -> tuple[Function, str, Any, dict[str, Any]]:
    """The map, the parameter the operand binds, the operand, and the fixed arguments."""
    f, operand = call.operands["f"], call.operands["v"]
    fixed = dict(call.operands.get("fixed_args") or {})
    return f, _bound_parameter(f, fixed), operand, fixed


def _lifts(f: Function, parameter: str, operand: Any) -> bool:
    """Whether the direct call lifts *operand* at *parameter*: a law sampled or a batch swept."""
    hint = _binding.parameter_lifting_hint(f._signature_info, parameter)
    return _plan.is_broadcast(operand, hint) or _plan.is_swept(operand, hint)


def _evaluation_rules_check(call: BoundCall, result: OutputSpec | None) -> Any:
    """The evaluation-rule registry selects a rule for the map's and the operand's types.

    An operand the direct call does not lift is applied by the map's body, which
    is exact.
    """
    f, parameter, operand, fixed = _call_values(call)
    if not _lifts(f, parameter, operand):
        return True
    return _rules.evaluation_rule_registry.check(
        f,
        operand,
        method=None,
        exact_only=call.controls["exact_only"],
        parameter=parameter,
        fixed_args=fixed,
        controls=call.controls,
    )


#: The engine controls evaluate forwards to the direct call of its map.
_FORWARDED_CONTROLS = (
    "n_broadcast_samples",
    "dispatch",
    "max_workers",
    "include_inputs",
    "workflow_kind",
    "exact_only",
    "method_options",
)


def _evaluation_rules_execute(call: BoundCall, result: OutputSpec | None) -> Any:
    """The direct call of the map on the operand, which takes the route the registry selects."""
    f, parameter, operand, fixed = _call_values(call)
    forwarded = {name: call.controls[name] for name in _FORWARDED_CONTROLS if name in call.controls}
    return f.with_options(**forwarded)(**{parameter: operand}, **fixed)


evaluate.register_route(
    _CheckedRoute(
        "evaluation_rules",
        source=RouteSource.REGISTRY,
        check=_evaluation_rules_check,
        execute=_evaluation_rules_execute,
        exact=None,
    )
)
