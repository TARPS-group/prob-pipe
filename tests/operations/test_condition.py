"""Contract tests of condition_on: currying given slots, the conditioning capabilities, and Bayes' rule."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from probpipe import Record, RecordSpec
from probpipe.core._dispatch import Feasibility, ResolutionError, UnaryDispatchRegistry
from probpipe.core._specs import InputSpec, OutputSpec
from probpipe.distributions._conditional import (
    ConditionalDistribution,
    ConditionalDistributionSpec,
)
from probpipe.distributions._distribution import Distribution, DistributionSpec
from probpipe.distributions._factored import FactoredDistribution
from probpipe.operations._condition import (
    _INFERENCE_METHOD_CONTROLS,
    InferenceMethod,
    condition_on,
    inference_method_registry,
)
from probpipe.operations._operation import ApplicabilityError

from ._laws import REAL, Amortized, Bare, ExactPosterior, Gaussian, Kernel


class _Conjugate(Distribution):
    """A model for which the suite's exact inference method applies."""

    def __init__(self, name: str) -> None:
        super().__init__(name, RecordSpec(theta=REAL, y=REAL))


class _AmortizedConjugate(Amortized):
    """An amortized model for which the suite's exact inference method also applies."""


class _AmortizedSimulator(Amortized):
    """An amortized model for which only the suite's approximate inference method applies."""


class _SuiteMethod(InferenceMethod):
    """An inference method of this suite, recording the options it receives."""

    def __init__(self, name: str, exact: bool, types: tuple[type, ...], loc: float) -> None:
        self._name, self._exact, self._types, self._loc = name, exact, types, loc
        self.options: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def exact(self) -> bool:
        return self._exact

    @property
    def priority(self) -> int:
        return 10

    def supported_types(self) -> tuple[type, ...]:
        return self._types

    def check(self, d: Any, given: Any, /, **options: Any) -> Feasibility:
        return Feasibility(True)

    def execute(self, d: Any, given: Any, /, **options: Any) -> Any:
        self.options.append(options)
        return Gaussian("theta", self._loc)


@pytest.fixture
def suite_methods(monkeypatch):
    """The bayes route, delegating to a registry of this suite's two methods for one test.

    The methods stay out of the global inference-method registry, whose own
    tests inspect every registered method.
    """
    exact = _SuiteMethod("operations_suite_exact", True, (_Conjugate, _AmortizedConjugate), 0.5)
    approximate = _SuiteMethod(
        "operations_suite_approximate",
        False,
        (_Conjugate, _AmortizedConjugate, _AmortizedSimulator),
        9.0,
    )
    registry: UnaryDispatchRegistry = UnaryDispatchRegistry()
    registry.register(exact)
    registry.register(approximate)
    (route,) = [route for route in condition_on.routes if route.name == "bayes"]
    monkeypatch.setattr(route, "registry", registry)
    return exact, approximate


@pytest.fixture
def factored_method(monkeypatch):
    """The bayes route, delegating for one test to an approximate method for factored joints."""
    method = _SuiteMethod("operations_suite_factored", False, (FactoredDistribution,), 4.0)
    registry: UnaryDispatchRegistry = UnaryDispatchRegistry()
    registry.register(method)
    (route,) = [route for route in condition_on.routes if route.name == "bayes"]
    monkeypatch.setattr(route, "registry", registry)
    return method


class _RecordingPosterior(ExactPosterior):
    """An exactly conditioning law that records the options its ``_condition_on`` receives."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.options: list[dict[str, Any]] = []

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        self.options.append(kwargs)
        return super()._condition_on(given)


class _RecordingAmortized(Amortized):
    """An amortized law that records the options its ``_condition_on`` receives."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.options: list[dict[str, Any]] = []

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        self.options.append(kwargs)
        return super()._condition_on(given)


class _StructuredKernel(ConditionalDistribution):
    """A kernel conditioning on one record-valued slot ``theta``."""

    def __init__(self, name: str = "y") -> None:
        super().__init__(name, {"theta": RecordSpec(a=REAL, b=REAL)}, REAL)

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        return Gaussian(self.name)


class TestCurry:
    def test_binding_every_given_slot_applies_the_kernel(self):
        law = condition_on(Kernel(), {"mu": 1.5})
        assert isinstance(law, Distribution)
        assert law.loc == 1.5
        report = condition_on.check(Kernel(), {"mu": 1.5})
        assert (report.route, report.exact) == ("curry", True)
        assert report.result == OutputSpec(condition_on=DistributionSpec(Kernel().event_spec))

    def test_a_record_given_binds_its_fields(self):
        assert condition_on(Kernel(), Record("given", {"mu": 2.0})).loc == 2.0

    def test_binding_some_slots_leaves_a_kernel_over_the_rest(self):
        kernel = Kernel("y", ("a", "b"))
        curried = condition_on(kernel, {"a": 1.0})
        assert isinstance(curried, ConditionalDistribution)
        assert set(curried.given_spec) == {"b"}
        assert condition_on.check(kernel, {"a": 1.0}).result == OutputSpec(
            condition_on=ConditionalDistributionSpec(InputSpec(b=REAL), kernel.event_spec)
        )
        assert condition_on(curried, {"b": 2.0}).loc == 3.0

    @pytest.mark.pending(reason="binding part of a structured slot restructures, then binds")
    def test_binding_part_of_a_structured_slot_leaves_the_rest_of_the_slot(self):
        curried = condition_on(_StructuredKernel(), {"theta/a": 1.0})
        assert set(curried.given_spec) == {"theta"}

    @pytest.mark.pending(reason="the exact bindings combine with conditioning on produced fields")
    def test_given_slots_bind_together_with_produced_fields(self):
        assert isinstance(condition_on(Kernel(), {"mu": 1.0, "y": 0.0}), Distribution)


class TestSlice:
    def test_the_slice_route_declines_a_factored_law_with_a_reason(self):
        joint = Kernel("y", ("mu",)) * Gaussian("mu")
        declined = dict(condition_on.check(joint, {"y": 0.3}).routes)["slice"]
        assert declined.feasible is False
        assert "factors" in declined.description

    def test_fixing_a_produced_field_of_a_factored_joint_reaches_bayes_rule(self, factored_method):
        joint = Kernel("y", ("mu",)) * Gaussian("mu")
        report = condition_on.check(joint, {"y": 0.3})
        assert (report.route, report.method) == ("bayes", "operations_suite_factored")
        assert condition_on(joint, {"y": 0.3}).loc == 4.0
        assert factored_method.options == [{}]


class TestConditioningCapabilities:
    def test_exact_conditioning_returns_the_conditional_law(self):
        model = ExactPosterior("model")
        law = condition_on(model, {"y": 0.3})
        assert law.loc == 1.0
        assert model.givens == [{"y": 0.3}]
        report = condition_on.check(model, {"y": 0.3})
        assert (report.route, report.exact) == ("exact_conditioning", True)

    def test_approximate_conditioning_runs_when_no_exact_route_applies(self):
        report = condition_on.check(Amortized("model"), {"y": 0.3})
        assert (report.route, report.exact) == ("approximate_conditioning", False)
        assert condition_on(Amortized("model"), {"y": 0.3}).loc == 2.0

    def test_exact_only_excludes_the_approximate_capability(self):
        with pytest.raises(ResolutionError, match="exact_only"):
            condition_on.with_options(exact_only=True)(Amortized("model"), {"y": 0.3})

    def test_the_approximate_capability_receives_the_budgets_it_declares(self):
        model = _RecordingAmortized("model")
        view = condition_on.with_options(num_results=500, random_seed=3)
        assert view(model, {"y": 0.3}).loc == 2.0
        assert model.options == [{"num_results": 500, "random_seed": 3}]

    def test_a_budget_the_approximate_capability_does_not_read_stays_out_of_its_options(self):
        model = _RecordingAmortized("model")
        condition_on.with_options(num_warmup=10)(model, {"y": 0.3})
        assert model.options == [{}]

    def test_exact_conditioning_reads_no_budget(self):
        model = _RecordingPosterior("model")
        condition_on.with_options(num_results=500)(model, {"y": 0.3})
        assert model.options == [{}]


class TestBudgets:
    def test_a_misspelled_budget_raises_type_error(self):
        with pytest.raises(TypeError, match="num_resluts"):
            condition_on.with_options(num_resluts=500)

    def test_every_registered_inference_method_declares_the_controls_it_reads(self):
        undeclared = set(inference_method_registry.list_methods()) - set(_INFERENCE_METHOD_CONTROLS)
        assert not undeclared, sorted(undeclared)

    def test_the_bayes_route_declares_every_inference_method_control(self):
        (route,) = [route for route in condition_on.routes if route.name == "bayes"]
        declared = {name for names in _INFERENCE_METHOD_CONTROLS.values() for name in names}
        assert route.controls == declared
        condition_on.with_options(num_warmup=3, step_size=0.1, init={"theta": 0.0})


class TestBayes:
    def test_an_exact_registered_method_outranks_the_approximate_capability(self, suite_methods):
        report = condition_on.check(_AmortizedConjugate("model"), {"y": 0.3})
        assert (report.route, report.method, report.exact) == (
            "bayes",
            "operations_suite_exact",
            True,
        )
        assert condition_on(_AmortizedConjugate("model"), {"y": 0.3}).loc == 0.5

    def test_the_approximate_capability_outranks_the_approximate_methods(self, suite_methods):
        report = condition_on.check(_AmortizedSimulator("model"), {"y": 0.3})
        assert report.route == "approximate_conditioning"

    def test_method_names_an_inference_method(self, suite_methods):
        view = condition_on.with_options(method="operations_suite_approximate")
        assert view.check(_AmortizedConjugate("model"), {"y": 0.3}).method == (
            "operations_suite_approximate"
        )
        assert view(_AmortizedConjugate("model"), {"y": 0.3}).loc == 9.0

    def test_exact_only_excludes_the_approximate_methods(self, suite_methods):
        view = condition_on.with_options(exact_only=True)
        assert view.check(_Conjugate("model"), {"y": 0.3}).method == "operations_suite_exact"
        with pytest.raises(ResolutionError):
            view(_AmortizedSimulator("model"), {"y": 0.3})

    def test_method_parameters_are_controls_passed_to_the_method(self, suite_methods):
        exact, _ = suite_methods
        condition_on.with_options(num_warmup=3)(_Conjugate("model"), {"y": 0.3})
        assert exact.options == [{"num_warmup": 3}]

    def test_the_route_delegates_to_the_inference_method_registry(self):
        (route,) = [route for route in condition_on.routes if route.name == "bayes"]
        assert route.registry is inference_method_registry

    def test_a_law_no_route_applies_to_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="curry"):
            condition_on(Bare("b"), {"b": 0.0})


class TestTheOperation:
    def test_the_routes_are_listed_in_selection_order(self):
        names = [route.name for route in condition_on.summary().routes]
        assert names == [
            "curry",
            "slice",
            "exact_conditioning",
            "bayes",
            "approximate_conditioning",
        ]

    def test_the_conditioned_object_is_a_distribution_or_a_kernel(self):
        with pytest.raises(
            ApplicabilityError, match="DistributionSpec, ConditionalDistributionSpec"
        ):
            condition_on(jnp.zeros(2), {"x": 1.0})

    @pytest.mark.pending(
        reason="an exact slice assembles the conditional from normalized factors",
        raises=ResolutionError,
    )
    def test_fixing_an_upstream_field_leaves_the_existing_kernel(self):
        joint = Kernel("y", ("beta",)) * Gaussian("beta")
        conditional = condition_on(joint, {"beta": 0.5})
        assert conditional.event_spec.components.keys() == {"y"}
