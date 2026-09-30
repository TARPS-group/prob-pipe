"""The evaluation-rule registry: the routes of a lifted application.

A lifted application, which is the direct call ``f(d)`` or ``f(batch)``,
resolves among **evaluation rules**: the methods of a binary dispatch registry
keyed on the map's type and the operand's type. The engine consults the registry, the
``evaluate`` operation exposes it, and the families register their rules into
it at import, so a pair with a closed form or a fused batched routine takes it
while every other pair resolves through a floor.

A **floor** is the fallback on its stated domain and ranks below every rule
registered there. Two floors are registered here:

1. the sampling lift, on a distribution operand that samples, which pushes
   draws through the map and returns an empirical law over the outputs;
2. the elementwise sweep, on a batch operand, which maps the function over the
   batch's elements.

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
from ..core._dispatch import BinaryDispatchMethod, BinaryDispatchRegistry, Feasibility
from ..distributions._capabilities import SupportsSampling
from ..distributions._distribution import Distribution
from ..values import Function
from . import _normalization

__all__ = ["FLOOR_PRIORITY", "evaluation_rule_registry"]

#: The priority of a floor, below the priority of every rule registered above it.
FLOOR_PRIORITY = -(2**31)


def _consumes_the_operand(f: Function, parameter: str | None) -> bool:
    """Whether *parameter* of *f* declares that it consumes a distribution itself."""
    if parameter is None:
        return False
    return _normalization.is_distribution_hint(f._signature_info.hints.get(parameter))


class _SamplingLift(BinaryDispatchMethod):
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
        """Feasible when the operand, or the parent it views, samples.

        Returns
        -------
        Feasibility
            Infeasible when *parameter* consumes the distribution itself, or
            when neither the operand nor its parent claims SupportsSampling.
        """
        if _consumes_the_operand(f, parameter):
            return Feasibility(False, f"parameter {parameter!r} consumes the distribution itself")
        parent = getattr(operand, "parent", None)
        source = parent if isinstance(parent, Distribution) else operand
        if not isinstance(source, SupportsSampling):
            return Feasibility(
                False, f"{type(source).__name__} {source.name!r} does not claim SupportsSampling"
            )
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


class _ElementwiseSweep(BinaryDispatchMethod):
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
        """Feasible for every batch operand, whose elements the map receives in turn."""
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


#: The routes of a lifted application, keyed on the map's and the operand's types.
evaluation_rule_registry: BinaryDispatchRegistry[BinaryDispatchMethod] = BinaryDispatchRegistry()
evaluation_rule_registry.register(_SamplingLift())
evaluation_rule_registry.register(_ElementwiseSweep())
