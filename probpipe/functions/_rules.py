"""The evaluation-rule registry: the routes of a lifted application.

A lifted application, which is the direct call ``f(d)`` or ``f(batch)``,
resolves among **evaluation rules**: the methods of a binary dispatch registry
keyed on the map's type and the operand's type. The engine consults the
registry, the ``evaluate`` operation exposes it, and the families register
their rules into it at import, so a pair with a closed form or a fused batched
routine takes it while every other pair resolves through a floor.

Three rules are registered here:

1. the **sampling lift**, a floor on a distribution operand that samples, which
   pushes draws through the map and returns an empirical law over the outputs,
   an approximation;
2. the **elementwise sweep**, a floor on a batch operand, which maps the
   function over the operand's elements, exactly;
3. the **empirical enumeration**, an exact rule on a distribution operand whose
   lifted groups are all empirical laws, which evaluates the map once per
   combination of atoms when every lifted group of the call enumerates within
   the sample count, and so returns the exact pushforward of the empirical
   laws. A group enumerates through its root, so a view of an empirical law
   enumerates as the law does.

A **floor** is the fallback on its stated domain. The registry ranks the floors
in a tier of their own, below every other rule whatever its exactness and
priority, and orders each tier as every dispatch registry does. A floor's check
applies the test the planner applies to the argument at its parameter, its
role included, so the floor is feasible where the direct call takes it and
nowhere else.

A rule's ``check`` and ``execute`` take the map and the operand positionally,
followed by three keywords:

1. ``parameter``: the name of the parameter the operand binds;
2. ``fixed_args``: the map's other arguments, by parameter name, which hold the
   operand's own container when the operand is one argument of a variadic
   parameter;
3. ``controls``: the call's resolved controls.

The three rules here are realized by the engine itself, so their ``execute``
calls the map with ``method`` naming the rule, and the engine runs it.
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
from ..distributions._empirical import EmpiricalDistribution
from ..values import Function, _binding
from . import _descendants, _plan

__all__ = ["FLOOR_PRIORITY", "evaluation_rule_registry"]

#: The priority each floor registers with, which orders the floors among themselves.
FLOOR_PRIORITY = -(2**31)

#: The engine controls an engine-realized rule forwards when it calls the map.
_FORWARDED_CONTROLS = (
    "n_broadcast_samples",
    "dispatch",
    "max_workers",
    "include_inputs",
    "workflow_kind",
    "exact_only",
    "method_options",
)


def _describe(operand: Any) -> str:
    """The operand's kind and name, for a report."""
    name = getattr(operand, "label", None)
    return type(operand).__name__ if name is None else f"{type(operand).__name__} {name!r}"


def _call_values(
    operand: Any, parameter: str | None, fixed_args: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The call's arguments by parameter name: *fixed_args* with the operand at *parameter*.

    A variadic parameter's container already holds the operand, so it is kept as
    *fixed_args* gives it.
    """
    values = dict(fixed_args or {})
    if parameter is not None and parameter not in values:
        values[parameter] = operand
    return values


def _run_by_name(
    rule: str,
    f: Function,
    operand: Any,
    parameter: str | None,
    fixed_args: Mapping[str, Any] | None,
    controls: Mapping[str, Any] | None,
) -> Any:
    """Call *f* on the operand and the fixed arguments, with ``method`` naming *rule*.

    The engine realizes the rules this module registers, so naming one runs it
    there, under the forwarded controls.
    """
    # An unset control is None, which would reset the map's own setting.
    forwarded = {
        name: controls[name]
        for name in _FORWARDED_CONTROLS
        if controls and controls.get(name) is not None
    }
    bound = _binding.values_to_bound_arguments(
        f.signature, _call_values(operand, parameter, fixed_args)
    )
    return f.with_options(**forwarded, method=rule)(*bound.args, **bound.kwargs)


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
        lift = _plan.lift_at(f, parameter, operand)
        if lift == "whole" and isinstance(operand, Distribution):
            return Feasibility(False, f"parameter {parameter!r} consumes the distribution itself")
        if lift != "broadcast":
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
        """The pushforward of the operand through the map, as an empirical law over the outputs."""
        return _run_by_name(self.name, f, operand, parameter, fixed_args, controls)


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
        if _plan.lift_at(f, parameter, operand) != "sweep":
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
        """The batch of the map's results on the operand's elements, on its levels."""
        return _run_by_name(self.name, f, operand, parameter, fixed_args, controls)


class _EmpiricalEnumeration(BinaryDispatchMethod):
    """The exact rule on empirical laws: the map evaluated at every combination of atoms.

    Each combination of the lifted groups' atoms is evaluated once and carries
    the product of their weights, so the result is the exact pushforward of
    the empirical laws, an empirical law over the outputs. The rule admits any
    distribution operand, since a view of an empirical law lifts through its
    root, and its check decides from the groups' roots.
    """

    @property
    def name(self) -> str:
        return "empirical_enumeration"

    @property
    def exact(self) -> bool:
        return True

    @property
    def priority(self) -> int:
        return 0

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
        """Feasible when the call lifts only laws and every lifted group enumerates.

        A group enumerates when its root is an empirical law, and the product of
        the groups' atom counts is at most the ``n_broadcast_samples`` control.
        The operand's root is read as the lift reads it, through views, batch
        elements, and registered descendants, so the decision does not depend
        on the order of the arguments. A call whose operand's root is not
        empirical is declined before any plan is built.

        Returns
        -------
        Feasibility
            Infeasible when the call sweeps a batch, lifts nothing, or lifts a
            group that does not enumerate within the sample count, which the
            sampling lift then realizes.
        """
        root = _descendants.capture_stochastic_consumer(operand).root
        if not isinstance(root, EmpiricalDistribution):
            return Feasibility(False, f"{_describe(root)} is not an empirical law")
        values = _call_values(operand, parameter, fixed_args)
        plan = _plan.build_broadcast_plan(
            values=values, signature_info=f._signature_info, roles=f._roles
        )
        if plan.regime != "distribution":
            return Feasibility(
                False, f"the call does not lift {_describe(operand)} alone as a distribution"
            )
        count = (controls or {}).get("n_broadcast_samples", f.options["n_broadcast_samples"])
        stochastic = _plan.build_stochastic_plan(values, plan, count)
        if stochastic is None or stochastic.evaluation_mode != "exact":
            return Feasibility(
                False,
                f"not every lifted group is an empirical law whose atoms, with the other "
                f"groups', number at most n_broadcast_samples={count}",
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
        """The exact pushforward of the empirical laws, as an empirical law over the outputs."""
        return _run_by_name(self.name, f, operand, parameter, fixed_args, controls)


#: The rules the engine realizes itself, by name.
_ENGINE_RULES = frozenset({"sampling_lift", "elementwise_sweep", "empirical_enumeration"})


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


evaluation_rule_registry: BinaryDispatchRegistry[BinaryDispatchMethod] = _EvaluationRuleRegistry()
"""The routes of a lifted application, keyed on the map's and the operand's types."""
evaluation_rule_registry.register(_SamplingLift())
evaluation_rule_registry.register(_ElementwiseSweep())
evaluation_rule_registry.register(_EmpiricalEnumeration())
