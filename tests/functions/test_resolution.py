"""Resolution, step 6 of the stack (design V.7).

A plain call has its body as its one route. A lifted application resolves
through the evaluation-rule registry, which is a binary dispatch registry keyed
on the map's and the operand's types. Its floors are the sampling lift on a law
that samples and the elementwise sweep on a batch, and the selected route
records its fidelity in provenance.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from probpipe import (
    Distribution,
    Function,
    NumericArrayBatch,
    NumericArraySpec,
    ResolutionError,
    function,
    workflow_run,
)
from probpipe.core._dispatch import BinaryDispatchMethod, BinaryDispatchRegistry, Feasibility
from probpipe.functions import _rules
from probpipe.functions._rules import FLOOR_PRIORITY, evaluation_rule_registry

from ._design_helpers import error_of, record_law, standard_normal

SCALAR = NumericArraySpec(())


def _identity(x):
    return x


class _Unsampled(Distribution):
    """A law that declares its event and cannot sample."""


class _ClosedForm(BinaryDispatchMethod):
    """A rule that returns a marker in place of the pushforward, at a chosen rank."""

    def __init__(self, *, exact: bool, priority: int) -> None:
        self._exact = exact
        self._priority = priority

    @property
    def name(self) -> str:
        return "closed_form"

    @property
    def exact(self) -> bool:
        return self._exact

    @property
    def priority(self) -> int:
        return self._priority

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((Function,), (Distribution,))

    def check(self, f: Any, operand: Any, /, **call: Any) -> Feasibility:
        return Feasibility(True)

    def execute(self, f: Any, operand: Any, /, **call: Any) -> Any:
        return "closed form"


class TestTheRegistry:
    def test_the_registry_is_a_binary_dispatch_registry(self):
        assert isinstance(evaluation_rule_registry, BinaryDispatchRegistry)

    def test_the_sampling_lift_is_the_floor_on_a_law_that_samples(self):
        wrapped = Function("identity", _identity)

        info = evaluation_rule_registry.check(wrapped, standard_normal(), parameter="x")

        assert info.feasible is True
        assert info.method_name == "sampling_lift"
        assert info.exact is False

    def test_the_elementwise_sweep_is_the_floor_on_a_batch(self):
        wrapped = Function("identity", _identity)
        rows = NumericArrayBatch("rows", jnp.arange(3.0), "row", element_spec=SCALAR)

        info = evaluation_rule_registry.check(wrapped, rows, parameter="x")

        assert info.feasible is True
        assert info.method_name == "elementwise_sweep"
        assert info.exact is True

    def test_the_sampling_lift_declines_a_law_that_cannot_sample(self):
        wrapped = Function("identity", _identity)

        info = evaluation_rule_registry.check(wrapped, _Unsampled("bare", SCALAR), parameter="x")

        assert info.feasible is False
        assert "SupportsSampling" in info.description

    def test_the_sampling_lift_declines_a_parameter_that_consumes_the_law(self):
        def consume(x: Distribution):
            return 0.0

        info = evaluation_rule_registry.check(
            Function("consume", consume), standard_normal(), parameter="x"
        )

        assert info.feasible is False

    def test_a_view_samples_through_its_parent(self):
        wrapped = Function("identity", _identity)

        info = evaluation_rule_registry.check(wrapped, record_law()["a"], parameter="x")

        assert info.method_name == "sampling_lift"

    def test_a_floor_ranks_below_a_rule_registered_on_its_domain(self):
        registry: BinaryDispatchRegistry = BinaryDispatchRegistry()
        registry.register(_rules._SamplingLift())
        registry.register(_ClosedForm(exact=False, priority=0))

        info = registry.check(Function("identity", _identity), standard_normal(), parameter="x")

        assert info.method_name == "closed_form"
        assert FLOOR_PRIORITY < 0


class TestTheDirectCall:
    @pytest.mark.pending(
        reason="the engine resolves a lifted call through the registry", raises=AssertionError
    )
    def test_a_direct_call_takes_a_registered_exact_rule(self, monkeypatch):
        registry: BinaryDispatchRegistry = BinaryDispatchRegistry()
        registry.register(_rules._SamplingLift())
        registry.register(_ClosedForm(exact=True, priority=0))
        monkeypatch.setattr(_rules, "evaluation_rule_registry", registry)

        @function(n_broadcast_samples=6, dispatch="sequential")
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity(standard_normal())

        assert result == "closed form"

    @pytest.mark.pending(reason="route selection under exact_only")
    def test_exact_only_leaves_no_route_for_a_sampled_lift(self):
        @function
        def identity(x):
            return x

        with pytest.raises(ResolutionError):
            identity.with_options(exact_only=True)(standard_normal())

    @pytest.mark.pending(reason="route selection by the method control")
    def test_a_method_that_names_no_route_raises_resolution_error(self):
        @function
        def identity(x):
            return x

        with pytest.raises(ResolutionError):
            identity.with_options(method="quadrature")(standard_normal())

    @pytest.mark.pending(
        reason="a lift with no feasible route raises ResolutionError naming what is missing",
        raises=AssertionError,
    )
    def test_a_lift_with_no_feasible_route_names_the_missing_requirement(self):
        @function
        def identity(x):
            return x

        error = error_of(lambda: identity(_Unsampled("bare", SCALAR)))

        assert isinstance(error, ResolutionError)
        assert "SupportsSampling" in str(error)

    @pytest.mark.pending(
        reason="provenance records the selected route and its fidelity", raises=AssertionError
    )
    def test_the_selected_route_is_recorded_in_provenance(self):
        @function(n_broadcast_samples=6, dispatch="sequential")
        def identity(x):
            return x

        with workflow_run(seed=0):
            result = identity(standard_normal())

        recorded = str(dict(result.provenance.controls)) + str(dict(result.provenance.metadata))
        assert "sampling_lift" in recorded
