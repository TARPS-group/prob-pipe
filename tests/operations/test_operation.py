"""Contract tests of the operation model: declaration, roles, result rule, routes, and registry."""

from __future__ import annotations

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
    BatchSpec,
    EmpiricalDistribution,
    FunctionBatch,
    NumericArrayBatch,
    NumericArraySpec,
    OpaqueSpec,
    Record,
    RecordSpec,
    ResultKindError,
    ResultSchemaError,
    TrackedTerm,
    workflow_run,
)
from probpipe.core._dispatch import (
    Feasibility,
    ResolutionError,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)
from probpipe.core._expression import Conditioned, Named, Summary, draw_of, embedded
from probpipe.core._spec_base import TermSpec
from probpipe.core._specs import OutputSpec
from probpipe.distributions._batches import DistributionBatch
from probpipe.distributions._capabilities import SupportsMean, SupportsSampling
from probpipe.distributions._distribution import Distribution, DistributionSpec, _fixed_paths
from probpipe.operations import (
    BoundCall,
    OperandSummary,
    OperationRegistry,
    OperationRoute,
    OperationSummary,
    RouteSource,
    RouteSummary,
    operation,
)
from probpipe.operations._moments import mean
from probpipe.operations._operation import _install_expression_rule, _workflow_draws
from probpipe.operations._sample import sample
from probpipe.values import Function, FunctionSpec

from ._laws import REAL, Bare, Gaussian, GuardedMean, Pair, Sampler

# ---------------------------------------------------------------------------
# A toy operation with a capability route and a Monte Carlo fallback
# ---------------------------------------------------------------------------

_REGISTRY = OperationRegistry()


def _event(d: DistributionSpec) -> OutputSpec:
    """The law's own event declaration."""
    return d.event_spec


@operation(result=_event, registry=_REGISTRY)
def center(d: Distribution):
    """The center of a law.

    A toy operation for the model's tests.
    """


def _can_sample(call: BoundCall, result: OutputSpec | None) -> bool:
    """The operand samples."""
    return isinstance(call.operands["d"], SupportsSampling)


def _sampled_center(call: BoundCall, result: OutputSpec | None) -> Any:
    draws = _workflow_draws(
        call.operands["d"],
        (call.controls["n_broadcast_samples"],),
        operation_kind="center",
        execution_mode="monte_carlo",
    )
    return jnp.mean(draws, axis=0)


_CLOSED_FORM = center.capability_route(
    "closed_form", operand="d", protocol=SupportsMean, method="_mean", exact=True
)
center.fallback_route("monte_carlo", check=_can_sample, execute=_sampled_center, exact=False)


# ---------------------------------------------------------------------------
# Helpers for operations built per test
# ---------------------------------------------------------------------------


def _open(d: Any) -> None:
    """Leaves the result to the returned value."""
    return None


def _toy(result: Any = _open, **options: Any) -> Any:
    """A fresh primitive operation ``toy(d)`` in a registry of its own."""

    @operation(result=result, registry=OperationRegistry(), **options)
    def toy(d: Distribution):
        """A toy operation."""

    return toy


def _route(answer: Any, value: Any = None, calls: list[str] | None = None, name: str = ""):
    """A check returning *answer* and an execute returning *value*, recording its runs."""

    def check(call: BoundCall, result: OutputSpec | None) -> Any:
        return answer

    def execute(call: BoundCall, result: OutputSpec | None) -> Any:
        if calls is not None:
            calls.append(name)
        return jnp.float32(value)

    return {"check": check, "execute": execute}


class _Method(UnaryDispatchMethod):
    """A registry method with a fixed feasibility and result."""

    def __init__(self, name: str, *, exact: bool, feasible: Any, value: float) -> None:
        self._name, self._exact, self._feasible, self._value = name, exact, feasible, value
        self.options: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def exact(self) -> bool:
        return self._exact

    @property
    def priority(self) -> int:
        return 1

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    def check(self, *args: Any, **options: Any) -> Feasibility:
        if self._feasible is None:
            return Feasibility(None, pending=("a declaration",))
        return Feasibility(self._feasible, "" if self._feasible else f"{self._name} declines")

    def execute(self, *args: Any, **options: Any) -> Any:
        self.options.append(options)
        return jnp.float32(self._value)


def _registry(**methods: tuple[bool, Any, float]) -> UnaryDispatchRegistry:
    registry: UnaryDispatchRegistry = UnaryDispatchRegistry()
    for name, (exact, feasible, value) in methods.items():
        registry.register(_Method(name, exact=exact, feasible=feasible, value=value))
    return registry


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


class TestDeclaration:
    def test_an_operation_is_a_registered_function(self):
        assert isinstance(center, Function)
        assert _REGISTRY["center"] is center

    def test_an_empty_body_declares_a_primitive(self):
        assert not center.is_derived
        assert center.identity is None

    def test_a_body_that_returns_states_a_derived_identity(self):
        @operation(result=_event, registry=OperationRegistry())
        def doubled(d: Distribution):
            """Twice the center."""
            return center.with_options(raw=True)(d) * 2

        assert doubled.is_derived
        assert doubled.identity == "center.with_options(raw=True)(d) * 2"
        (identity,) = doubled.routes
        assert (identity.name, identity.source, identity.exact) == (
            "identity",
            RouteSource.FALLBACK,
            None,
        )
        assert float(jnp.asarray(doubled(Gaussian("g", 1.5)))) == 3.0

    def test_the_identity_route_is_feasible_where_the_identity_check_is(self):
        def center_applies(d: Any) -> Any:
            """The center has a route for the law."""
            return center.check(d)

        @operation(result=_event, identity_check=center_applies, registry=OperationRegistry())
        def doubled(d: Distribution):
            """Twice the center."""
            return center.with_options(raw=True)(d) * 2

        declined = doubled.check(Bare("b"))
        assert declined.feasible is False
        assert "does not implement SupportsMean" in declined.description
        with pytest.raises(ResolutionError, match="does not implement SupportsMean"):
            doubled(Bare("b"))
        report = doubled.check(Gaussian("g"))
        assert (report.route, report.exact) == ("identity", True)
        (identity,) = doubled.summary().routes
        assert identity.condition == "The center has a route for the law."

    def test_a_primitive_takes_no_identity_check(self):
        with pytest.raises(TypeError, match="empty body, so it cannot take identity_check"):
            _toy(identity_check=lambda d: True)

    def test_a_result_rule_reading_an_undeclared_name_raises(self):
        def rule(d: Any, missing: Any) -> None:
            """Reads a parameter the signature lacks."""

        with pytest.raises(TypeError, match="missing"):
            _toy(result=rule)

    def test_a_result_rule_that_is_not_callable_raises(self):
        with pytest.raises(TypeError, match="result rule"):
            _toy(result="not a rule")


# ---------------------------------------------------------------------------
# Roles and admission
# ---------------------------------------------------------------------------


class TestRoles:
    def test_roles_follow_the_annotations(self):
        @operation(result=_open, registry=OperationRegistry())
        def annotated(d: Distribution, value: Any, path: str):
            """Three kinds of parameter."""

        operands = {operand.name: operand.accepts for operand in annotated.summary().operands}
        assert operands == {"d": (DistributionSpec,), "value": (TermSpec,), "path": ()}

    def test_a_declared_role_overrides_the_annotation(self):
        toy = _toy(roles={"d": (NumericArraySpec,)})
        assert toy.summary().operands[0].accepts == (NumericArraySpec,)

    def test_a_role_for_an_undeclared_parameter_raises(self):
        with pytest.raises(TypeError, match="not a parameter"):
            _toy(roles={"other": (DistributionSpec,)})

    def test_a_role_must_list_term_spec_classes(self):
        with pytest.raises(TypeError, match="TermSpec"):
            _toy(roles={"d": (int,)})

    def test_an_argument_of_a_kind_the_role_refuses_raises_applicability_error(self):
        with pytest.raises(ApplicabilityError, match="'d' accepts DistributionSpec") as caught:
            center(jnp.zeros(2))
        assert isinstance(caught.value, TypeError)
        assert "NumericArraySpec" in str(caught.value)

    def test_the_refusal_is_the_engines_applicability_error(self):
        """One class, so a caller catching the engine's refusal catches an operation's."""
        from probpipe.functions import ApplicabilityError as EngineRefusal

        with pytest.raises(EngineRefusal):
            center(jnp.zeros(2))

    def test_an_unannotated_parameter_accepts_the_value_kinds(self):
        @operation(result=_open, registry=OperationRegistry())
        def scored(d: Distribution, value):
            """An unannotated value."""

        (_, value) = scored.summary().operands
        assert value.accepts == (NumericArraySpec, RecordSpec, OpaqueSpec, FunctionSpec)

    def test_a_law_at_a_value_role_is_lifted_and_a_law_its_role_admits_passes_whole(self):
        seen: list[Any] = []

        @operation(result=_open, registry=OperationRegistry())
        def shifted(d: Distribution, value):
            """The value shifted by one."""

        def execute(call: BoundCall, result: Any) -> Any:
            seen.append((call.operands["d"], call.operands["value"]))
            return jnp.asarray(call.operands["value"]) + 1.0

        shifted.structural_route(
            "shift", check=lambda call, result: True, execute=execute, exact=True
        )
        law = Gaussian("g")
        report = shifted.check(law, Gaussian("v", 2.0))
        assert (report.route, report.lifted) == ("shift", ("value",))
        with workflow_run(seed=0):
            pushforward = shifted.with_options(n_broadcast_samples=8, dispatch="sequential")(
                law, Gaussian("v", 2.0)
            )
        assert isinstance(pushforward, EmpiricalDistribution)
        assert len(seen) == 8
        assert all(d is law and not isinstance(value, Distribution) for d, value in seen)

    def test_a_parameter_annotated_any_takes_its_argument_as_it_arrives(self):
        def rule(f: Any) -> None:
            """Leaves the result open."""

        @operation(result=rule, roles={"f": (FunctionSpec,)}, registry=OperationRegistry())
        def applied(f: Any):
            """A function, consumed as an object."""

        applied.structural_route("any", exact=True, **_route(True, 0.0))
        functions = FunctionBatch("fs", [lambda x: x, lambda x: 2 * x], "fs")
        with pytest.raises(ApplicabilityError, match=r"'f' accepts FunctionSpec.*FunctionBatch"):
            applied(functions)

    def test_apply_lifts_nothing(self):
        laws = DistributionBatch("laws", [Gaussian("g", 1.0), Gaussian("g", 2.0)], "laws")
        with pytest.raises(ApplicabilityError, match="'d' accepts DistributionSpec"):
            center.apply(laws)


# ---------------------------------------------------------------------------
# The result rule and the applicability conditions
# ---------------------------------------------------------------------------


class TestResultRule:
    def test_the_rule_reads_supplying_specs_and_selecting_values(self):
        seen: dict[str, Any] = {}

        def rule(d: Any, path: Any) -> None:
            """Records what it reads."""
            seen.update(d=d, path=path)

        @operation(result=rule, registry=OperationRegistry())
        def pick(d: Distribution, path: str, scale: Any = 1.0):
            """Reads a path."""

        pick.structural_route("any", exact=True, **_route(True, 0.0))
        law = Gaussian("g")
        pick(law, "x")
        assert seen == {"d": law.spec, "path": "x"}

    def test_a_bound_call_gives_specs_of_supplying_operands_and_values_of_selecting_ones(self):
        calls: list[BoundCall] = []

        def rule(d: Any, path: Any) -> None:
            """Leaves the result open."""

        @operation(result=rule, registry=OperationRegistry())
        def pick(d: Distribution, path: str, scale: Any = 1.0):
            """Reads a path."""

        def execute(call: BoundCall, result: Any) -> Any:
            calls.append(call)
            return jnp.float32(0.0)

        pick.structural_route("any", check=lambda call, result: True, execute=execute, exact=True)
        law = Gaussian("g")
        pick(law, "x", jnp.ones(3))
        (call,) = calls
        assert call.operation.label == "pick"
        assert set(call.specs) == {"d", "scale"}
        assert call.specs["d"] == law.spec
        assert call.specs["scale"].shape == (3,)
        assert dict(call.declarations) == {"d": law.spec, "path": "x"}
        assert call.controls["method"] is None and call.controls["exact_only"] is False

    def test_the_declaration_names_the_returned_kind(self):
        result = center(Gaussian("g", 2.0))
        assert type(result).__name__ == "NumericArray"
        assert result.spec == Gaussian("g").event_spec.spec
        assert float(jnp.asarray(result)) == 2.0

    def test_a_record_declaration_returns_a_record(self):
        def rule(d: Any) -> OutputSpec:
            """A record of one field."""
            return OutputSpec(RecordSpec(a=NumericArraySpec(())))

        toy = _toy(result=rule)
        toy.structural_route(
            "record",
            check=lambda call, result: True,
            execute=lambda call, result: {"a": jnp.float32(1.0)},
            exact=True,
        )
        assert isinstance(toy(Gaussian("g")), Record)

    def test_the_engine_refuses_a_result_that_violates_the_rule_at_return(self):
        def rule(d: Any) -> OutputSpec:
            """Three coordinates."""
            return OutputSpec(toy=NumericArraySpec((3,)))

        toy = _toy(result=rule)
        toy.structural_route(
            "short",
            check=lambda call, result: True,
            execute=lambda call, result: jnp.zeros(2),
            exact=True,
        )
        with pytest.raises(ResultSchemaError, match="shape"):
            toy(Gaussian("g"))

    def test_a_result_that_violates_the_declaration_raises_value_error(self):
        def rule(d: Any) -> OutputSpec:
            """Three coordinates."""
            return OutputSpec(toy=NumericArraySpec((3,)))

        toy = _toy(result=rule)
        toy.structural_route(
            "short",
            check=lambda call, result: True,
            execute=lambda call, result: jnp.zeros(2),
            exact=True,
        )
        with pytest.raises(ValueError, match="shape"):
            toy(Gaussian("g"))

    def test_a_type_hole_is_filled_from_the_returned_value(self):
        def rule(d: Any) -> OutputSpec:
            """A hole."""
            return OutputSpec(toy=None)

        toy = _toy(result=rule)
        toy.structural_route(
            "pair",
            check=lambda call, result: True,
            execute=lambda call, result: jnp.zeros(2),
            exact=True,
        )
        assert toy(Gaussian("g")).spec.shape == (2,)
        assert any(
            "completed from the returned value" in item
            for item in toy.check(Gaussian("g")).deferred
        )

    def test_a_rule_returning_something_else_raises_type_error(self):
        toy = _toy(result=lambda d: "not a declaration")
        toy.structural_route("any", exact=True, **_route(True, 0.0))
        with pytest.raises(TypeError, match="OutputSpec or None"):
            toy(Gaussian("g"))


class TestConditions:
    def test_a_failing_condition_raises_applicability_error_quoting_it(self):
        def positive_scale(d: Any) -> bool:
            """The law's scale is declared positive."""
            return False

        toy = _toy(conditions=(positive_scale,))
        toy.structural_route("any", exact=True, **_route(True, 0.0))
        with pytest.raises(
            ApplicabilityError, match="requirement not met: the law's scale is declared positive"
        ):
            toy(Gaussian("g"))

    def test_an_unresolved_condition_is_deferred_to_the_return(self):
        def later(d: Any) -> None:
            """A condition the values settle."""
            return None

        toy = _toy(conditions=(later,))
        toy.structural_route("any", exact=True, **_route(True, 4.0))
        assert any(
            "a condition the values settle" in item for item in toy.check(Gaussian("g")).deferred
        )
        assert float(jnp.asarray(toy(Gaussian("g")))) == 4.0


# ---------------------------------------------------------------------------
# Route selection
# ---------------------------------------------------------------------------


class TestSelection:
    def test_an_exact_route_outranks_an_approximate_one_registered_first(self):
        toy = _toy()
        toy.fallback_route("rough", exact=False, **_route(True, 1.0))
        toy.structural_route("precise", exact=True, **_route(True, 2.0))
        assert toy.check(Gaussian("g")).route == "precise"
        assert float(jnp.asarray(toy(Gaussian("g")))) == 2.0

    def test_a_fallback_ranks_below_another_route_of_the_same_exactness(self):
        toy = _toy()
        toy.fallback_route("floor", exact=False, **_route(True, 1.0))
        toy.structural_route("direct", exact=False, **_route(True, 2.0))
        assert toy.check(Gaussian("g")).route == "direct"

    def test_a_fallback_ranks_below_every_other_route_whatever_its_exactness(self):
        toy = _toy()
        toy.fallback_route("floor", exact=True, **_route(True, 1.0))
        toy.structural_route("direct", exact=False, **_route(True, 2.0))
        assert toy.check(Gaussian("g")).route == "direct"
        assert float(jnp.asarray(toy(Gaussian("g")))) == 2.0

    def test_registration_order_breaks_the_remaining_ties(self):
        toy = _toy()
        toy.structural_route("first", exact=True, **_route(True, 1.0))
        toy.structural_route("second", exact=True, **_route(True, 2.0))
        assert toy.check(Gaussian("g")).route == "first"

    def test_an_infeasible_route_passes_to_the_next(self):
        calls: list[str] = []
        toy = _toy()
        toy.structural_route("declines", exact=True, **_route(False, 1.0, calls, "declines"))
        toy.structural_route("runs", exact=True, **_route(True, 2.0, calls, "runs"))
        toy(Gaussian("g"))
        assert calls == ["runs"]

    def test_an_unresolved_route_above_a_feasible_one_blocks_selection(self):
        toy = _toy()
        toy.structural_route("waits", exact=True, **_route(None, 1.0))
        toy.fallback_route("floor", exact=False, **_route(True, 2.0))
        report = toy.check(Gaussian("g"))
        assert report.feasible is None and report.route is None
        assert report.pending
        with pytest.raises(ResolutionError, match="unresolved"):
            toy(Gaussian("g"))

    def test_no_feasible_route_raises_resolution_error_naming_each_route(self):
        toy = _toy()
        toy.structural_route("one", exact=True, **_route(False, 1.0))
        toy.fallback_route("two", exact=False, **_route(False, 2.0))
        with pytest.raises(ResolutionError, match=r"one.*two"):
            toy(Gaussian("g"))
        assert toy.check(Gaussian("g")).feasible is False

    def test_an_operation_without_routes_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="none is registered"):
            _toy()(Gaussian("g"))

    def test_a_capability_route_requires_membership(self):
        with pytest.raises(ResolutionError, match="does not implement SupportsMean"):
            center(Bare("b"))

    def test_membership_suffices_where_the_class_defines_no_guard(self):
        assert center.check(Gaussian("g")).route == "closed_form"

    def test_a_rejecting_guard_passes_to_the_fallback(self):
        report = center.check(GuardedMean("g", False))
        assert (report.route, report.exact) == ("monte_carlo", False)
        declined = {info.method_name: info for info in report.routes}["closed_form"]
        assert "the stand-in's answer admits the closed form" in declined.description

    def test_each_probed_route_reports_its_own_exactness(self):
        report = center.check(GuardedMean("g", False))
        routes = {info.method_name: info for info in report.routes}
        assert (routes["closed_form"].exact, routes["monte_carlo"].exact) == (True, False)

    def test_an_admitting_guard_selects_the_capability(self):
        assert center.check(GuardedMean("g", True)).route == "closed_form"

    def test_an_unresolved_guard_blocks_selection(self):
        assert center.check(GuardedMean("g", None)).feasible is None
        with pytest.raises(ResolutionError, match="unresolved"):
            center(GuardedMean("g", None))

    def test_provenance_records_the_selected_route(self):
        exact = center(Gaussian("g")).provenance.metadata
        assert (exact["route"], exact["exact"]) == ("closed_form", True)
        approximate = center(GuardedMean("g", False)).provenance.metadata
        assert (approximate["route"], approximate["exact"]) == ("monte_carlo", False)

    def test_the_call_takes_the_route_its_check_reports(self):
        for law in (Gaussian("g"), GuardedMean("g", False), Sampler("s")):
            assert center(law).provenance.metadata["route"] == center.check(law).route

    def test_check_executes_no_route(self):
        toy = _toy()

        def execute(call: BoundCall, result: Any) -> Any:
            raise AssertionError("check executed the route")

        toy.structural_route(
            "guarded", check=lambda call, result: True, execute=execute, exact=True
        )
        report = toy.check(Gaussian("g"))
        assert (report.feasible, report.route, report.exact) == (True, "guarded", True)
        assert [info.method_name for info in report.routes] == ["guarded"]


class TestNamingAMethod:
    @staticmethod
    def _two_registries() -> Any:
        toy = _toy()
        toy.registry_route("left", registry=_registry(nuts=(False, True, 1.0)))
        toy.registry_route(
            "right", registry=_registry(nuts=(False, True, 2.0), hmc=(False, True, 3.0))
        )
        return toy

    def test_a_plain_name_matching_one_method_runs_it(self):
        toy = self._two_registries().with_options(method="hmc")
        assert float(jnp.asarray(toy(Gaussian("g")))) == 3.0

    def test_a_name_held_by_two_registries_asks_for_route_slash_method(self):
        toy = self._two_registries().with_options(method="nuts")
        with pytest.raises(
            ResolutionError, match=r"ambiguous; it matches left/nuts, right/nuts\. Pass one"
        ):
            toy(Gaussian("g"))

    def test_the_qualified_form_selects_the_method_within_its_route(self):
        toy = self._two_registries().with_options(method="right/nuts")
        assert toy.check(Gaussian("g")).route == "right"
        assert float(jnp.asarray(toy(Gaussian("g")))) == 2.0

    def test_a_name_matching_a_route_and_a_method_is_ambiguous(self):
        toy = _toy()
        toy.structural_route("nuts", exact=True, **_route(True, 1.0))
        toy.registry_route("methods", registry=_registry(nuts=(False, True, 2.0)))
        with pytest.raises(ResolutionError, match=r"nuts, methods/nuts"):
            toy.with_options(method="nuts")(Gaussian("g"))

    def test_routes_sharing_a_registry_each_take_the_named_method(self):
        class _OnLaws(_Method):
            def check(self, *args: Any, **options: Any) -> Feasibility:
                return Feasibility(isinstance(args[0], Distribution), "the argument is no law")

        registry: UnaryDispatchRegistry = UnaryDispatchRegistry()
        registry.register(_OnLaws("nuts", exact=False, feasible=True, value=2.0))
        toy = _toy()
        toy.registry_route("first", registry=registry, arguments=lambda call: (object(),))
        toy.registry_route("second", registry=registry)
        report = toy.with_options(method="nuts").check(Gaussian("g"))
        assert (report.route, report.method) == ("second", "nuts")

    def test_a_qualified_name_no_route_holds_raises(self):
        with pytest.raises(ResolutionError, match="route 'left' has no method 'hmc'"):
            self._two_registries().with_options(method="left/hmc")(Gaussian("g"))


class TestRegistryRoutes:
    def _operation(self, registry: UnaryDispatchRegistry, stand_in: Any = True) -> Any:
        toy = _toy()
        toy.structural_route("stand_in", exact=False, **_route(stand_in, 1.0))
        toy.registry_route("methods", registry=registry)
        return toy

    def test_an_exact_registered_method_outranks_an_approximate_route(self):
        registry = _registry(precise=(True, True, 2.0), rough=(False, True, 3.0))
        toy = self._operation(registry)
        report = toy.check(Gaussian("g"))
        assert (report.route, report.method, report.exact) == ("methods", "precise", True)
        assert float(jnp.asarray(toy(Gaussian("g")))) == 2.0

    def test_an_approximate_route_outranks_the_approximate_methods_registered_after_it(self):
        registry = _registry(precise=(True, False, 2.0), rough=(False, True, 3.0))
        toy = self._operation(registry)
        assert toy.check(Gaussian("g")).route == "stand_in"
        assert float(jnp.asarray(toy(Gaussian("g")))) == 1.0

    def test_the_approximate_methods_run_when_no_route_ranks_above_them(self):
        registry = _registry(precise=(True, False, 2.0), rough=(False, True, 3.0))
        toy = self._operation(registry, stand_in=False)
        report = toy.check(Gaussian("g"))
        assert (report.route, report.method, report.exact) == ("methods", "rough", False)

    def test_method_names_a_registered_method(self):
        registry = _registry(precise=(True, True, 2.0), rough=(False, True, 3.0))
        toy = self._operation(registry).with_options(method="rough")
        assert toy.check(Gaussian("g")).method == "rough"
        assert float(jnp.asarray(toy(Gaussian("g")))) == 3.0

    def test_exact_only_excludes_the_approximate_methods_and_routes(self):
        registry = _registry(precise=(True, False, 2.0), rough=(False, True, 3.0))
        toy = self._operation(registry).with_options(exact_only=True)
        with pytest.raises(ResolutionError, match="exact_only"):
            toy(Gaussian("g"))

    def test_provenance_records_the_registry_method(self):
        registry = _registry(precise=(True, True, 2.0), rough=(False, True, 3.0))
        metadata = self._operation(registry)(Gaussian("g")).provenance.metadata
        assert (metadata["route"], metadata["method"], metadata["exact"]) == (
            "methods",
            "precise",
            True,
        )

    def test_method_options_reach_the_registered_methods(self):
        registry = _registry(precise=(True, True, 2.0))
        toy = self._operation(registry).with_options(method_options={"num_warmup": 7})
        toy(Gaussian("g"))
        assert registry.get_method("precise").options == [{"num_warmup": 7}]

    def test_a_budget_is_not_a_control_of_its_own(self):
        toy = _toy()
        toy.registry_route("methods", registry=_registry(precise=(True, True, 2.0)))
        with pytest.raises(TypeError, match="unknown control"):
            toy.with_options(num_warmup=7)

    def test_a_registry_route_is_listed_with_its_exactness_delegated(self):
        registry = _registry(precise=(True, True, 2.0), rough=(False, True, 3.0))
        routes = {route.name: route for route in self._operation(registry).summary().routes}
        assert routes["methods"].source is RouteSource.REGISTRY
        assert routes["methods"].exact is None
        assert "precise, rough" in routes["methods"].condition


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


class TestControls:
    def test_with_options_returns_a_view_and_leaves_the_operation_unchanged(self):
        view = center.with_options(method="monte_carlo")
        assert view.check(Gaussian("g")).route == "monte_carlo"
        assert center.check(Gaussian("g")).route == "closed_form"

    def test_method_runs_the_named_route_instead_of_the_selected_one(self):
        with workflow_run(seed=0):
            estimate = center.with_options(method="monte_carlo", n_broadcast_samples=4000)(
                Gaussian("g", 3.0)
            )
        assert abs(float(jnp.asarray(estimate)) - 3.0) < 0.1

    def test_method_naming_no_route_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="unknown method 'nope'"):
            center.with_options(method="nope")(Gaussian("g"))

    def test_exact_only_with_a_named_approximate_route_raises(self):
        view = center.with_options(method="monte_carlo", exact_only=True)
        with pytest.raises(ResolutionError, match="not exact"):
            view(Gaussian("g"))

    def test_exact_only_excludes_every_approximate_route(self):
        with pytest.raises(ResolutionError, match="exact_only"):
            center.with_options(exact_only=True)(Sampler("s"))

    def test_an_unknown_control_raises_type_error(self):
        with pytest.raises(TypeError, match="unknown control"):
            center.with_options(tolerance=0.1)

    @pytest.mark.parametrize("controls", [{"exact_only": "yes"}, {"raw": 1}, {"method": 3}])
    def test_a_control_of_the_wrong_type_raises_type_error(self, controls):
        with pytest.raises(TypeError):
            center.with_options(**controls)

    def test_a_route_reads_its_budgets_from_method_options(self):
        seen: list[Any] = []
        toy = _toy()

        def execute(call: BoundCall, result: Any) -> Any:
            seen.append(call.controls["method_options"]["tolerance"])
            return jnp.float32(0.0)

        toy.structural_route(
            "tolerant", check=lambda call, result: True, execute=execute, exact=True
        )
        toy.with_options(method_options={"tolerance": 0.5})(Gaussian("g"))
        assert seen == [0.5]
        with pytest.raises(TypeError, match="unknown control"):
            toy.with_options(tolerance=0.5)

    def test_a_capability_route_passes_the_method_options_to_the_capability(self):
        class Tolerant(Gaussian):
            """A normal law whose closed-form mean records the options it receives."""

            def __init__(self, label: str) -> None:
                super().__init__(label, 1.5)
                self.options: list[dict[str, Any]] = []

            def _mean(self, **options: Any) -> Any:
                self.options.append(options)
                return super()._mean()

        toy = _toy(result=_event)
        toy.capability_route(
            "estimate", operand="d", protocol=SupportsMean, method="_mean", exact=False
        )
        law = Tolerant("g")
        assert float(jnp.asarray(toy.with_options(method_options={"tolerance": 0.5})(law))) == 1.5
        toy(law)
        assert law.options == [{"tolerance": 0.5}, {}]

    def test_an_exact_capability_reads_no_budget(self):
        class Tolerant(Gaussian):
            def _mean(self, **options: Any) -> Any:
                assert not options
                return super()._mean()

        toy = _toy(result=_event)
        toy.capability_route(
            "closed_form", operand="d", protocol=SupportsMean, method="_mean", exact=True
        )
        view = toy.with_options(method_options={"tolerance": 0.5})
        assert float(jnp.asarray(view(Tolerant("g", 1.5)))) == 1.5

    def test_the_selected_capability_rejects_an_option_it_does_not_read(self):
        toy = _toy(result=_event)
        toy.capability_route(
            "estimate", operand="d", protocol=SupportsMean, method="_mean", exact=False
        )
        with pytest.raises(TypeError, match="patience"):
            toy.with_options(method_options={"patience": 3})(Gaussian("g", 1.5))

    def test_a_route_registered_after_a_view_is_seen_by_the_view(self):
        toy = _toy()
        view = toy.with_options(exact_only=True)
        toy.structural_route("late", exact=True, **_route(True, 5.0))
        assert view.check(Gaussian("g")).route == "late"

    def test_raw_returns_the_result_detached(self):
        result = center.with_options(raw=True)(Gaussian("g", 2.0))
        assert not isinstance(result, TrackedTerm)
        assert float(result) == 2.0

    def test_the_sample_count_control_sets_the_number_of_draws(self):
        law = Sampler("s")
        center.with_options(n_broadcast_samples=17)(law)
        assert law.shapes == [(17,)]


# ---------------------------------------------------------------------------
# Checks of lifted calls
# ---------------------------------------------------------------------------


def _laws(*laws: Distribution) -> DistributionBatch:
    return DistributionBatch("laws", list(laws), "laws")


class TestLiftedChecks:
    def test_a_swept_batch_is_admitted_and_planned_at_its_element_kind(self):
        batch = _laws(Gaussian("g", 1.0), Gaussian("g", 2.0))
        report = center.check(batch)
        assert (report.feasible, report.route, report.exact) == (True, "closed_form", True)
        assert report.lifted == ("d",)
        assert report.result == Gaussian("g").event_spec
        np.testing.assert_array_equal(np.asarray(center(batch).values), [1.0, 2.0])

    def test_an_element_no_route_applies_to_makes_the_check_infeasible(self):
        batch = _laws(Gaussian("g"), Bare("g"))
        report = center.check(batch)
        assert report.feasible is False
        assert "sweep cell (1,)" in report.description
        assert "does not implement SupportsMean" in report.description
        with pytest.raises(ResolutionError, match=r"Bare '\S+' does not implement SupportsMean"):
            center(batch)

    def test_elements_that_select_different_routes_leave_the_route_undecided(self):
        report = center.check(_laws(Gaussian("g", 1.0), Sampler("g", 2.0)))
        assert (report.feasible, report.route, report.exact) == (True, None, False)

    def test_an_element_kind_the_role_refuses_raises_as_the_call_does(self):
        values = NumericArrayBatch("values", jnp.zeros(3), "values", element_spec=REAL)
        with pytest.raises(ApplicabilityError, match="but got NumericArrayBatch"):
            center.check(values)
        with pytest.raises(ApplicabilityError, match="but got NumericArrayBatch"):
            center(values)

    def test_an_empty_sweep_is_planned_at_its_element_kind_and_selects_no_route(self):
        empty = DistributionBatch(
            "laws", np.empty(0, object), "laws", element_spec=Gaussian("g").spec
        )
        report = center.check(empty)
        assert (report.feasible, report.route, report.lifted) == (True, None, ("d",))

    def test_a_method_names_the_route_each_point_of_a_lifted_call_takes(self):
        laws = _laws(Gaussian("g", 1.0), Gaussian("g", 2.0))
        view = center.with_options(method="monte_carlo", n_broadcast_samples=4000)
        assert view.check(laws).route == "monte_carlo"
        with workflow_run(seed=0):
            means = view(laws)
        np.testing.assert_allclose(np.asarray(means.values), [1.0, 2.0], atol=0.1)

    def test_exact_only_excludes_the_sampling_lift_of_a_value(self):
        @operation(result=_open, registry=OperationRegistry())
        def shifted(d: Distribution, value):
            """The value shifted by one."""

        shifted.structural_route("shift", exact=True, **_route(True, 0.0))
        view = shifted.with_options(exact_only=True)
        assert view.check(Gaussian("g"), Gaussian("v")).feasible is False
        with pytest.raises(ResolutionError, match="exact_only"):
            view(Gaussian("g"), Gaussian("v"))

    def test_a_plain_call_lifts_nothing(self):
        assert center.check(Gaussian("g")).lifted == ()

    def test_a_raw_call_lifts_as_the_tracked_call_does(self):
        means = center.with_options(raw=True)(_laws(Gaussian("g", 1.0), Gaussian("g", 2.0)))
        assert not isinstance(means, TrackedTerm)
        np.testing.assert_array_equal(np.asarray(means), [1.0, 2.0])


# ---------------------------------------------------------------------------
# Raw results
# ---------------------------------------------------------------------------


def _untracked(tree: Any) -> bool:
    """Whether *tree* is a nested dict whose leaves are raw values rather than terms."""
    if type(tree) is dict:
        return all(_untracked(child) for child in tree.values())
    return not isinstance(tree, TrackedTerm)


class TestRawResults:
    @pytest.mark.parametrize("mode", ["plain", "raw", "apply"])
    @pytest.mark.parametrize("spec", [NumericArraySpec(()), BatchSpec(NumericArraySpec(()), row=2)])
    def test_a_route_returning_the_wrong_kind_is_refused(self, mode, spec):
        toy = _toy(result=lambda d: OutputSpec(toy=spec))
        toy.structural_route(
            "wrong_kind",
            check=lambda call, result: True,
            execute=lambda call, result: "text",
            exact=True,
        )
        invoke = toy.apply if mode == "apply" else toy.with_options(raw=mode == "raw")

        with pytest.raises(ValueError if mode == "apply" else ResultKindError):
            invoke(Gaussian("g"))

    @pytest.mark.parametrize("error", [TypeError("route failed"), ValueError("route failed")])
    @pytest.mark.parametrize("mode", ["plain", "apply"])
    def test_an_exception_from_the_route_is_propagated_unchanged(self, error, mode):
        def execute(call, result):
            raise error

        toy = _toy(result=_event)
        toy.structural_route(
            "failure", check=lambda call, result: True, execute=execute, exact=True
        )
        invoke = toy.apply if mode == "apply" else toy

        with pytest.raises(type(error)) as raised:
            invoke(Gaussian("g"))

        assert raised.value is error

    def test_a_raw_record_is_the_nested_mapping_of_its_raw_leaves(self):
        result = center.with_options(raw=True)(Pair("p"))
        assert type(result) is dict and set(result) == {"a", "b"}
        assert _untracked(result)
        assert float(result["a"]) == 1.0
        np.testing.assert_array_equal(np.asarray(result["b"]), [-1.0, -1.0])

    def test_a_raw_record_keeps_its_nesting(self):
        def rule(d: Any) -> OutputSpec:
            """A record with a nested group."""
            return OutputSpec(
                RecordSpec(a=NumericArraySpec(()), g=RecordSpec(b=NumericArraySpec(())))
            )

        toy = _toy(result=rule)
        toy.structural_route(
            "nested",
            check=lambda call, result: True,
            execute=lambda call, result: {"a": jnp.float32(1.0), "g": {"b": jnp.float32(2.0)}},
            exact=True,
        )
        result = toy.with_options(raw=True)(Gaussian("g"))
        assert type(result) is dict and type(result["g"]) is dict
        assert _untracked(result)
        assert float(result["g"]["b"]) == 2.0

    def test_a_raw_batch_of_records_is_the_nested_mapping_of_its_raw_columns(self):
        draws = sample.with_options(raw=True)(Pair("p"), (3,))
        assert type(draws) is dict and set(draws) == {"a", "b"}
        assert _untracked(draws)
        assert (jnp.shape(draws["a"]), jnp.shape(draws["b"])) == ((3,), (3, 2))

    def test_a_raw_moment_of_a_record_law_is_a_mapping(self):
        result = mean.with_options(raw=True)(Pair("p"))
        assert type(result) is dict and _untracked(result)

    def test_the_raw_evaluator_returns_the_raw_array(self):
        result = mean.raw()(Gaussian("g", 2.0))
        assert not isinstance(result, TrackedTerm)
        assert float(result) == 2.0

    def test_apply_returns_the_raw_form(self):
        result = center.apply(Pair("p"))
        assert type(result) is dict and _untracked(result)

    def test_a_raw_result_is_validated_against_the_declaration(self):
        def rule(d: Any) -> OutputSpec:
            """Three coordinates."""
            return OutputSpec(toy=NumericArraySpec((3,)))

        toy = _toy(result=rule)
        toy.structural_route(
            "short",
            check=lambda call, result: True,
            execute=lambda call, result: jnp.zeros(2),
            exact=True,
        )
        with pytest.raises(ValueError, match="shape"):
            toy.with_options(raw=True)(Gaussian("g"))


# ---------------------------------------------------------------------------
# The tracked result and randomness
# ---------------------------------------------------------------------------


class TestResultAndRandomness:
    def test_a_call_returns_a_tracked_term_labeled_by_its_primary_operand(self):
        result = center(Gaussian("g"))
        assert isinstance(result, TrackedTerm)
        assert result.label == "g"
        assert result.provenance is not None

    def test_a_label_rule_derives_the_label_from_the_arguments(self):
        toy = _toy(label=lambda d: f"{d.label}_toy")
        toy.structural_route("value", **_route(True, 1.0), exact=True)
        assert toy(Gaussian("g")).label == "g_toy"

    def test_a_label_rule_reads_only_the_declarations_parameters(self):
        with pytest.raises(TypeError, match="the label rule reads"):
            _toy(label=lambda law: "x")

    def test_an_expression_rule_gives_a_law_result_its_expression(self):
        returned = Gaussian("g")
        toy = _toy()
        toy.structural_route(
            "law",
            check=lambda call, result: True,
            execute=lambda call, result: returned,
            exact=True,
        )
        _install_expression_rule(toy, lambda d: Conditioned(embedded(d), ("y",)))
        result = toy(Gaussian("g"))
        assert _fixed_paths(result) == ("y",)
        assert (result.label, result.notation) == ("g", "g(g; y)")
        assert _fixed_paths(returned) == ()

    def test_an_expression_rule_labels_a_value_result(self):
        toy = _toy()
        toy.structural_route("value", **_route(True, 1.0), exact=True)
        _install_expression_rule(toy, lambda d: Summary("E", draw_of(d)))
        result = toy(Gaussian("g"))
        assert (float(result), result.label) == (1.0, "E[g ~ g]")

    def test_an_expression_rule_that_returns_none_keeps_the_routes_expression(self):
        toy = _toy()
        toy.structural_route(
            "law",
            check=lambda call, result: True,
            execute=lambda call, result: Gaussian("g").with_label("routed"),
            exact=True,
        )
        _install_expression_rule(toy, lambda: None)
        assert toy(Gaussian("g")).label == "routed"

    def test_an_operation_without_an_expression_rule_carries_its_primary_operands(self):
        law = Gaussian("g")
        assert _toy()._derived_expression({"d": law}) == embedded(law)
        labeled = _toy(label=lambda d: f"{d.label}_toy")
        assert labeled._derived_expression({"d": law}) == Named("g_toy")

    def test_an_expression_rule_reads_only_the_declarations_parameters(self):
        with pytest.raises(TypeError, match="the expression rule reads"):
            _install_expression_rule(_toy(), lambda law: Named("x"))
        with pytest.raises(TypeError, match="needs a callable expression rule"):
            _install_expression_rule(_toy(), Named("x"))

    def test_a_sweep_is_labeled_by_the_batch_it_sweeps(self):
        laws = DistributionBatch("laws", [Gaussian("g", 1.0), Gaussian("g", 2.0)], "law")
        centers = center(laws)
        assert centers.label == "laws"
        assert centers[0].label == "laws[law=0]"

    def test_an_operation_takes_no_key(self):
        assert "key" not in inspect.signature(center).parameters
        with pytest.raises(TypeError):
            center(Gaussian("g"), key=jax.random.PRNGKey(0))

    def test_a_seeded_workflow_reproduces_the_monte_carlo_route(self):
        law = Sampler("s")
        with workflow_run(seed=11):
            first = np.asarray(center(law))
        with workflow_run(seed=11):
            second = np.asarray(center(law))
        with workflow_run(seed=12):
            third = np.asarray(center(law))
        assert first == second
        assert first != third


# ---------------------------------------------------------------------------
# Route construction
# ---------------------------------------------------------------------------


class TestRouteConstruction:
    def test_every_helper_returns_an_operation_route_of_its_source(self):
        toy = _toy()
        routes = {
            toy.structural_route("s", exact=True, **_route(True)): RouteSource.STRUCTURAL,
            toy.capability_route(
                "c", operand="d", protocol=SupportsMean, method="_mean", exact=True
            ): RouteSource.CAPABILITY,
            toy.registry_route("r", registry=_registry()): RouteSource.REGISTRY,
            toy.fallback_route("f", exact=False, **_route(True)): RouteSource.FALLBACK,
        }
        for route, source in routes.items():
            assert isinstance(route, OperationRoute)
            assert route.source is source

    def test_a_duplicate_route_name_raises_value_error(self):
        toy = _toy()
        toy.structural_route("once", exact=True, **_route(True))
        with pytest.raises(ValueError, match="already has a route named 'once'"):
            toy.fallback_route("once", exact=False, **_route(True))

    def test_a_capability_route_names_a_parameter(self):
        with pytest.raises(TypeError, match="no parameter 'x'"):
            _toy().capability_route(
                "c", operand="x", protocol=SupportsMean, method="_mean", exact=True
            )

    def test_a_check_that_is_not_callable_raises_type_error(self):
        with pytest.raises(TypeError, match="callable"):
            _toy().structural_route("s", check=True, execute=lambda call, result: 0, exact=True)

    def test_an_object_without_the_route_members_does_not_register(self):
        with pytest.raises(TypeError, match="not an OperationRoute"):
            _toy().register_route(object())

    def test_a_registry_route_needs_a_dispatch_registry(self):
        with pytest.raises(TypeError, match="dispatch registry"):
            _toy().registry_route("r", registry=object())

    def test_a_capability_condition_quotes_the_guard_the_class_defines(self):
        assert (
            _CLOSED_FORM.condition_for(GuardedMean)
            == "The stand-in's answer admits the closed form."
        )
        assert _CLOSED_FORM.condition_for(Gaussian) == "membership in SupportsMean suffices"


# ---------------------------------------------------------------------------
# The registry of operations
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_a_duplicate_name_raises_value_error(self):
        registry = OperationRegistry()
        registry.register(center)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(center)

    def test_only_an_operation_registers(self):
        with pytest.raises(TypeError, match="only an operation"):
            OperationRegistry().register(Function("plain", lambda x: x))

    def test_an_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError, match="unknown operation 'absent'"):
            _REGISTRY["absent"]

    def test_list_summarizes_operands_derivation_and_routes(self):
        (summary,) = _REGISTRY.list()
        assert isinstance(summary, OperationSummary)
        assert summary.name == "center"
        assert summary.priority is None
        assert summary.description == "The center of a law."
        assert summary.module_path == __name__
        assert summary.operands == (OperandSummary("d", (DistributionSpec,), True),)
        assert (summary.is_derived, summary.identity) == (False, None)
        assert summary.routes == (
            RouteSummary(
                "closed_form",
                RouteSource.CAPABILITY,
                True,
                (SupportsMean,),
                "membership in SupportsMean suffices",
            ),
            RouteSummary("monte_carlo", RouteSource.FALLBACK, False, (), "The operand samples."),
        )

    def test_describe_renders_one_operation_or_all(self):
        text = _REGISTRY.describe("center")
        assert text.splitlines()[0] == "center(d) — primitive"
        assert "closed_form (capability, exact; requires SupportsMean)" in text
        assert "monte_carlo (fallback, approximate): The operand samples." in text
        assert _REGISTRY.describe() == text
        with pytest.raises(KeyError):
            _REGISTRY.describe("absent")

    def test_the_registry_reports_as_a_cataloged_registry(self):
        assert (_REGISTRY.name, _REGISTRY.kind) == ("operations", "operation")
        assert _REGISTRY.description
        assert _REGISTRY.entry_summaries() == _REGISTRY.list()
        assert _REGISTRY.describe_entry("center") == center.summary()
