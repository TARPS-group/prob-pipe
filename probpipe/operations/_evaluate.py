"""The evaluate operation: applying a map to a value, a distribution, or a batch.

``evaluate(f, v)`` is the operation form of the engine's resolution step: for a
value it returns ``f(v)``, for a distribution the pushforward law, and for a
batch the elementwise result. Its route is the evaluation-rule registry, keyed
on the map's and the operand's types, which the direct call ``f(v)`` also
takes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core._spec_base import TermSpec
from ..core._specs import OutputSpec
from ..values import FunctionSpec
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


def _evaluation_rules(call: BoundCall, result: OutputSpec | None) -> Any:
    """The evaluation-rule registry selects a rule for the map's and the operand's types."""
    raise NotImplementedError("evaluate.evaluation_rules")


evaluate.register_route(
    _CheckedRoute(
        "evaluation_rules",
        source=RouteSource.REGISTRY,
        check=_evaluation_rules,
        execute=_evaluation_rules,
        exact=None,
    )
)
