"""The evaluation-rule registry: the routes of a lifted application.

A lifted application, which is the direct call ``f(d)`` or ``f(batch)``,
resolves among **evaluation rules**: the methods of a binary dispatch registry
keyed on the map's type and the operand's type. The engine consults the registry, the
``evaluate`` operation exposes it, and the families register their rules into
it at import, so a pair with a closed form or a fused batched routine takes it
while every other pair resolves through a floor.

A **floor** is the fallback on its stated domain. The registry ranks the floors
in a tier of their own, below every other rule whatever its exactness and
priority, and orders each tier as every dispatch registry does. Two floors are
registered here:

1. the sampling lift, on a distribution operand that samples, which pushes
   draws through the map and returns an empirical law over the outputs;
2. the elementwise sweep, on a batch operand, which maps the function over
   the operand's elements.

A floor's check applies the test the planner applies to the argument at its
parameter, so the floor is feasible where the direct call takes it and nowhere else.

A rule's ``check`` and ``execute`` take the map and the operand positionally,
followed by three keywords:

1. ``parameter``: the name of the parameter the operand binds;
2. ``fixed_args``: the other arguments, by name;
3. ``controls``: the call's resolved controls.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core._batch import Batch
from ..core._dispatch import (
    BinaryDispatchMethod,
    BinaryDispatchRegistry,
    Feasibility,
    _Registration,
)
from ..distributions._capabilities import SupportsSampling
from ..distributions._distribution import Distribution
from ..values import Function, _binding
from . import _normalization, _plan

__all__ = ["FLOOR_PRIORITY", "evaluation_rule_registry"]

#: The priority each floor registers with, which orders the floors among themselves.
FLOOR_PRIORITY = -(2**31)


def _lifting_hint(f: Function, parameter: str | None) -> Any:
    """The annotation that governs lifting at *parameter* of *f*, or None without one."""
    if parameter is None:
        return None
    return _binding.parameter_lifting_hint(f._signature_info, parameter)


def _describe(operand: Any) -> str:
    """The operand's kind and name, for a report."""
    name = getattr(operand, "name", None)
    return type(operand).__name__ if name is None else f"{type(operand).__name__} {name!r}"


class _Floor(BinaryDispatchMethod):
    """A fallback on its stated domain, ranked below every other rule there."""


class _SamplingLift(_Floor):
    """The floor on distribution operands: draws from the operand pushed through the map.

    The result is an empirical law over the outputs, so the rule is approximate.
    A view samples through its parent, which must itself sample.
    """

    @property
    def name(self) -> str:
        return "sampling_lift"

    @property
    def exact(self) -> bool:
        return False

    @property
    def priority(self) -> int:
        return FLOOR_PRIORITY

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((Function,), (Distribution,))

    def check(
        self,
        f: Function,
        operand: Distribution,
        /,
        *,
        parameter: str | None = None,
        fixed_args: Mapping[str, Any] | None = None,
        controls: Mapping[str, Any] | None = None,
    ) -> Feasibility:
        """Feasible when the call samples the operand and can draw from it or from its parent.

        Returns
        -------
        Feasibility
            Infeasible when *parameter* consumes the distribution itself, when
            the call does not sample the operand, and when neither the operand
            nor its parent claims SupportsSampling.
        """
        expected = _lifting_hint(f, parameter)
        if _normalization.is_distribution_hint(expected):
            return Feasibility(False, f"parameter {parameter!r} consumes the distribution itself")
        if not _plan.is_broadcast(operand, expected):
            return Feasibility(
                False, f"the call does not sample {_describe(operand)} at parameter {parameter!r}"
            )
        parent = getattr(operand, "parent", None)
        source = parent if isinstance(parent, Distribution) else operand
        if not isinstance(source, SupportsSampling):
            return Feasibility(False, f"{_describe(source)} does not claim SupportsSampling")
        return Feasibility(True)

    def execute(
        self,
        f: Function,
        operand: Distribution,
        /,
        *,
        parameter: str | None = None,
        fixed_args: Mapping[str, Any] | None = None,
        controls: Mapping[str, Any] | None = None,
    ) -> Any:
        """The pushforward of the operand through the map, as an empirical law.

        Raises
        ------
        NotImplementedError
            Until the engine runs the lift as this rule.
        """
        raise NotImplementedError("_SamplingLift.execute")


class _ElementwiseSweep(_Floor):
    """The floor on batch operands: the function mapped over the batch's elements.

    The result is the batch of the elementwise results on the operand's
    levels, so the rule is exact.
    """

    @property
    def name(self) -> str:
        return "elementwise_sweep"

    @property
    def exact(self) -> bool:
        return True

    @property
    def priority(self) -> int:
        return FLOOR_PRIORITY

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((Function,), (Batch,))

    def check(
        self,
        f: Function,
        operand: Batch,
        /,
        *,
        parameter: str | None = None,
        fixed_args: Mapping[str, Any] | None = None,
        controls: Mapping[str, Any] | None = None,
    ) -> Feasibility:
        """Feasible when the call sweeps the operand and asks only for the outputs.

        Returns
        -------
        Feasibility
            Infeasible when the call does not sweep the operand, as when the
            parameter's annotation is ``Any`` or names a batched class the
            operand satisfies, and when the ``include_inputs`` control asks for
            the inputs, which the sweep does not return.
        """
        if not _plan.is_swept(operand, _lifting_hint(f, parameter)):
            return Feasibility(
                False, f"the call does not sweep {_describe(operand)} at parameter {parameter!r}"
            )
        if controls is not None and controls.get("include_inputs"):
            return Feasibility(
                False, "include_inputs asks for the inputs, and the sweep returns the outputs"
            )
        return Feasibility(True)

    def execute(
        self,
        f: Function,
        operand: Batch,
        /,
        *,
        parameter: str | None = None,
        fixed_args: Mapping[str, Any] | None = None,
        controls: Mapping[str, Any] | None = None,
    ) -> Any:
        """The batch of the map's results on the operand's elements.

        Raises
        ------
        NotImplementedError
            Until the engine runs the sweep as this rule.
        """
        raise NotImplementedError("_ElementwiseSweep.execute")


class _EvaluationRuleRegistry(BinaryDispatchRegistry[BinaryDispatchMethod]):
    """A binary dispatch registry whose floors rank in a tier below every other rule.

    Within each tier, selection follows the order of every dispatch registry:
    exactness, then priority, then specificity, then registration order. A
    priority override therefore moves a floor only among the floors.
    """

    def _rank(self, registration: _Registration[BinaryDispatchMethod]) -> tuple[int, int, int]:
        exactness, opt_in, priority = super()._rank(registration)
        # The floor tier follows both exactness classes of the other rules.
        tier = 2 if isinstance(registration.method, _Floor) else 0
        return (tier + exactness, opt_in, priority)


#: The routes of a lifted application, keyed on the map's and the operand's types.
evaluation_rule_registry: BinaryDispatchRegistry[BinaryDispatchMethod] = _EvaluationRuleRegistry()
evaluation_rule_registry.register(_SamplingLift())
evaluation_rule_registry.register(_ElementwiseSweep())
