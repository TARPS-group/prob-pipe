"""Contract tests of condition_on: its exact stage, its normalization stage, and their routes."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from probpipe import Record, RecordSpec
from probpipe.core._dispatch import Feasibility, ResolutionError, UnaryDispatchRegistry
from probpipe.core._specs import InputSpec, OutputSpec
from probpipe.distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsConditionalLogProb,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    _is_normalized,
    _kernel_is_normalized,
)
from probpipe.distributions._conditional import (
    ConditionalDistribution,
    ConditionalDistributionSpec,
)
from probpipe.distributions._distribution import Distribution, DistributionSpec
from probpipe.operations._condition import (
    _INFERENCE_METHOD_CONTROLS,
    InferenceMethod,
    _UnnormalizedConditional,
    condition_on,
    inference_method_registry,
)
from probpipe.operations._operation import ApplicabilityError, _RegistryRoute

from ._laws import REAL, Amortized, Bare, ExactPosterior, Gaussian, Kernel, Unnormalized


class _Conjugate(Distribution, SupportsLogProb):
    """A joint over ``theta`` and ``y`` for which the suite's exact inference method applies."""

    def __init__(self, name: str) -> None:
        super().__init__(name, RecordSpec(theta=REAL, y=REAL))

    def _log_prob(self, value: Any) -> Any:
        theta, y = jnp.asarray(value["theta"]), jnp.asarray(value["y"])
        return jax.scipy.stats.norm.logpdf(theta) + jax.scipy.stats.norm.logpdf(y, theta)


class _AmortizedConjugate(Amortized):
    """An amortized model for which the suite's exact inference method also applies."""


class _AmortizedSimulator(Amortized):
    """An amortized model for which only the suite's approximate inference method applies."""


class _SuitePosterior(Distribution, SupportsSampling):
    """The suite methods' result: a point mass at ``loc`` in every field of the target's event."""

    def __init__(self, event_spec: OutputSpec, loc: float) -> None:
        super().__init__("posterior", event_spec)
        self.loc = float(loc)

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        draw = jnp.full(tuple(sample_shape), self.loc, jnp.float32)
        if self.event_spec.exposes_record:
            return Record("posterior", dict.fromkeys(self.event_spec.components, draw))
        return draw


class _SuiteMethod(InferenceMethod):
    """An inference method of this suite for the targets whose joints it names.

    A target that carries a joint is read by the joint's type, and any other
    target by its own; each call records the options it receives.
    """

    def __init__(self, name: str, exact: bool, types: tuple[type, ...], loc: float) -> None:
        self._name, self._exact, self._types, self._loc = name, exact, types, loc
        self.targets: list[Any] = []
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
        return (Distribution,)

    def check(self, target: Any, observed: Any, /, **options: Any) -> Feasibility:
        read = target.joint if isinstance(target, _UnnormalizedConditional) else target
        return Feasibility(isinstance(read, self._types), f"{self._name} does not apply")

    def execute(self, target: Any, observed: Any, /, **options: Any) -> Any:
        self.targets.append(target)
        self.options.append(options)
        return _SuitePosterior(target.event_spec, self._loc)


def _normalize_with(monkeypatch: pytest.MonkeyPatch, *methods: InferenceMethod) -> None:
    """Make every route that normalizes delegate to a registry of *methods*, for one test.

    The methods stay out of the global inference-method registry, whose own
    tests inspect every registered method.
    """
    registry: UnaryDispatchRegistry = UnaryDispatchRegistry()
    for method in methods:
        registry.register(method)
    for route in condition_on.routes:
        if isinstance(route, _RegistryRoute):
            monkeypatch.setattr(route, "registry", registry)


@pytest.fixture
def suite_methods(monkeypatch):
    """The suite's exact and approximate methods, the only ones the normalizing routes select."""
    exact = _SuiteMethod(
        "operations_suite_exact", True, (_Conjugate, _AmortizedConjugate, Unnormalized), 0.5
    )
    approximate = _SuiteMethod(
        "operations_suite_approximate",
        False,
        (_Conjugate, _AmortizedConjugate, _AmortizedSimulator, Unnormalized),
        9.0,
    )
    _normalize_with(monkeypatch, exact, approximate)
    return exact, approximate


@pytest.fixture
def approximate_method(monkeypatch):
    """An approximate method for the joints of this suite's kernels and factored laws."""
    method = _SuiteMethod(
        "operations_suite_factored", False, (Distribution, _UnnormalizedKernel), 4.0
    )
    _normalize_with(monkeypatch, method)
    return method


class _RecordingPosterior(ExactPosterior):
    """A law with exact conditioning that records the options its ``_condition_on`` receives."""

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


class _StructuredKernel(ConditionalDistribution, SupportsConditionalSampling):
    """A kernel conditioning on one record-valued slot ``theta``."""

    def __init__(self, name: str = "y") -> None:
        super().__init__(name, {"theta": RecordSpec(a=REAL, b=REAL)}, REAL)

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        return Gaussian(self.name)

    def _conditional_sample(self, given: Any, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return Gaussian(self.name)._sample(key, sample_shape)


class _NormalKernel(Kernel, SupportsConditionalSampling, SupportsConditionalLogProb):
    """The suite's normal kernel, declaring that its laws sample and have a density."""

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        law = super()._condition_on(given)
        if isinstance(law, Kernel):
            return _NormalKernel(law.name, law.slots, law.offset, law.component)
        return law

    def _conditional_sample(self, given: Any, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return self._condition_on(given)._sample(key, sample_shape)

    def _conditional_log_prob(self, given: Any, value: Any) -> Any:
        return self._condition_on(given)._log_prob(value)


class _UnnormalizedKernel(ConditionalDistribution, SupportsConditionalUnnormalizedLogProb):
    """A kernel whose laws are known only up to a constant, as a program's posterior targets are."""

    def __init__(self, name: str = "theta", slots: tuple[str, ...] = ("data",)) -> None:
        super().__init__(name, {slot: REAL for slot in slots}, REAL)
        self.slots = tuple(slots)

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        left = tuple(slot for slot in self.slots if slot not in given)
        return _UnnormalizedKernel(self.name, left) if left else Unnormalized(self.name)

    def _conditional_unnormalized_log_prob(self, given: Any, value: Any) -> Any:
        return Unnormalized(self.name)._unnormalized_log_prob(value)


class _AmortizedKernel(
    ConditionalDistribution, SupportsApproximateConditioning, SupportsConditionalSampling
):
    """A learned kernel from ``y`` to ``theta``, whose evaluation stands in for a posterior."""

    def __init__(self, name: str = "theta") -> None:
        super().__init__(name, {"y": REAL}, REAL)

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        return Gaussian(self.name, 3.0)

    def _conditional_sample(self, given: Any, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return self._condition_on(given)._sample(key, sample_shape)


class TestCurry:
    def test_binding_every_given_slot_applies_the_kernel(self):
        law = condition_on(_NormalKernel(), {"mu": 1.5})
        assert isinstance(law, Distribution)
        assert law.loc == 1.5
        report = condition_on.check(_NormalKernel(), {"mu": 1.5})
        assert (report.route, report.method, report.exact) == ("curry", None, True)
        assert report.result == OutputSpec(condition_on=DistributionSpec(Kernel().event_spec))

    def test_a_record_given_binds_its_fields(self):
        assert condition_on(_NormalKernel(), Record("given", {"mu": 2.0})).loc == 2.0

    def test_binding_some_slots_leaves_a_kernel_over_the_rest(self):
        kernel = _NormalKernel("y", ("a", "b"))
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

    def test_given_slots_bind_together_with_produced_fields(self, approximate_method):
        joint = Kernel("y", ("mu",)) * Kernel("z", ("mu",))
        law = condition_on(joint, {"mu": 1.0, "y": 0.0})
        assert isinstance(law, Distribution)
        (target,) = approximate_method.targets
        assert set(target.event_spec.components) == {"z"}
        assert set(target.joint.event_spec.components) == {"y", "z"}
        assert condition_on.check(joint, {"mu": 1.0, "y": 0.0}).route == "bayes"


class TestSlice:
    def test_the_slice_route_declines_a_factored_law_with_a_reason(self):
        joint = Kernel("y", ("mu",)) * Gaussian("mu")
        declined = dict(condition_on.check(joint, {"y": 0.3}).routes)["slice"]
        assert declined.feasible is False
        assert "factors" in declined.description

    def test_fixing_a_produced_field_of_a_factored_joint_reaches_bayes_rule(
        self, approximate_method
    ):
        joint = Kernel("y", ("mu",)) * Gaussian("mu")
        report = condition_on.check(joint, {"y": 0.3})
        assert (report.route, report.method) == ("bayes", "operations_suite_factored")
        assert condition_on(joint, {"y": 0.3}).loc == 4.0
        assert approximate_method.options == [{}]


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

    def test_the_routes_that_normalize_declare_every_inference_method_control(self):
        declared = {name for names in _INFERENCE_METHOD_CONTROLS.values() for name in names}
        normalizing = [route for route in condition_on.routes if isinstance(route, _RegistryRoute)]
        assert {route.name for route in normalizing} == {"curry", "bayes"}
        assert all(route.controls == declared for route in normalizing)
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

    def test_the_routes_delegate_to_the_inference_method_registry(self):
        normalizing = [route for route in condition_on.routes if isinstance(route, _RegistryRoute)]
        assert all(route.registry is inference_method_registry for route in normalizing)

    def test_a_law_no_route_applies_to_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="curry"):
            condition_on(Bare("b"), {"b": 0.0})


class TestTheExactStage:
    def test_the_unnormalized_conditional_carries_the_joint_and_the_given_values(self):
        model = _Conjugate("model")
        target = condition_on.with_options(method="unnormalized")(model, {"y": 0.3})
        assert isinstance(target, _UnnormalizedConditional)
        assert target.joint is model
        assert float(target.given["y"]) == pytest.approx(0.3)
        assert set(target.event_spec.components) == {"theta"}

    def test_its_density_is_the_joints_at_the_given_values(self):
        model = _Conjugate("model")
        target = condition_on.with_options(method="unnormalized")(model, {"y": 0.3})
        assert isinstance(target, SupportsUnnormalizedLogProb)
        value = Record("theta", {"theta": 0.7})
        joint = Record("value", {"theta": 0.7, "y": 0.3})
        assert float(target._unnormalized_log_prob(value)) == pytest.approx(
            float(model._log_prob(joint))
        )

    def test_it_is_unnormalized_and_claims_no_density_for_a_joint_without_one(self):
        target = condition_on.with_options(method="unnormalized")(
            _AmortizedSimulator("m"), {"y": 1}
        )
        assert not _is_normalized(target)
        assert not isinstance(target, SupportsUnnormalizedLogProb)

    def test_unnormalized_returns_a_kernels_unnormalized_law(self):
        law = condition_on.with_options(method="unnormalized")(_UnnormalizedKernel(), {"data": 1})
        assert isinstance(law, Unnormalized)

    def test_unnormalized_returns_a_normalized_result_as_it_is(self):
        law = condition_on.with_options(method="unnormalized")(_NormalKernel(), {"mu": 1.5})
        assert law.loc == 1.5
        report = condition_on.with_options(method="unnormalized").check(_NormalKernel(), {"mu": 1})
        assert (report.route, report.method, report.exact) == ("unnormalized", None, True)

    def test_unnormalized_is_selected_only_by_name(self, suite_methods):
        report = condition_on.check(_Conjugate("model"), {"y": 0.3})
        assert report.route == "bayes"
        declined = dict(report.routes).get("unnormalized")
        assert declined is None or declined.feasible is False

    def test_the_exact_stage_is_exact_before_an_approximate_stand_in(self):
        target = condition_on.with_options(method="unnormalized")(Amortized("model"), {"y": 0.3})
        assert isinstance(target, _UnnormalizedConditional)


class TestTheNormalizationStage:
    def test_a_normalized_result_is_returned_without_inference(self, suite_methods):
        exact, approximate = suite_methods
        assert condition_on(_NormalKernel(), {"mu": 1.5}).loc == 1.5
        assert exact.targets == approximate.targets == []

    def test_an_unnormalized_curried_law_is_the_target_of_a_method(self, suite_methods):
        exact, _ = suite_methods
        report = condition_on.check(_UnnormalizedKernel(), {"data": 1.0})
        assert (report.route, report.method) == ("curry", "operations_suite_exact")
        assert condition_on(_UnnormalizedKernel(), {"data": 1.0}).loc == 0.5
        (target,) = exact.targets
        assert isinstance(target, Unnormalized)

    def test_check_reports_the_exact_route_and_the_normalization_method(self, suite_methods):
        view = condition_on.with_options(method="operations_suite_approximate")
        curried = view.check(_UnnormalizedKernel(), {"data": 1.0})
        assert (curried.route, curried.method, curried.exact) == (
            "curry",
            "operations_suite_approximate",
            False,
        )
        conditioned = view.check(_Conjugate("model"), {"y": 0.3})
        assert (conditioned.route, conditioned.method) == ("bayes", "operations_suite_approximate")

    def test_a_named_method_does_not_run_on_a_normalized_result(self, suite_methods):
        view = condition_on.with_options(method="operations_suite_approximate")
        with pytest.raises(ResolutionError, match="normalized"):
            view(_NormalKernel(), {"mu": 1.0})

    def test_exact_only_raises_naming_the_unnormalized_method(self, monkeypatch):
        _normalize_with(
            monkeypatch, _SuiteMethod("operations_suite_approximate", False, (_Conjugate,), 9.0)
        )
        with pytest.raises(ResolutionError, match='method="unnormalized"'):
            condition_on.with_options(exact_only=True)(_Conjugate("model"), {"y": 0.3})

    def test_exact_only_returns_a_normalized_exact_result(self):
        view = condition_on.with_options(exact_only=True)
        assert view(_NormalKernel(), {"mu": 2.0}).loc == 2.0

    def test_a_kernel_result_is_normalized_per_value(self, approximate_method):
        kernel = condition_on(_UnnormalizedKernel("theta", ("a", "b")), {"a": 1.0})
        assert isinstance(kernel, ConditionalDistribution)
        assert set(kernel.given_spec) == {"b"}
        assert _kernel_is_normalized(kernel)
        assert isinstance(kernel, SupportsApproximateConditioning)
        assert approximate_method.targets == []
        law = condition_on(kernel, {"b": 2.0})
        assert law.loc == 4.0
        (target,) = approximate_method.targets
        assert isinstance(target, Unnormalized)

    def test_a_per_value_law_samples(self, suite_methods):
        kernel = condition_on(_UnnormalizedKernel("theta", ("a", "b")), {"a": 1.0})
        draws = kernel._conditional_sample({"b": 2.0}, jax.random.PRNGKey(0), (3,))
        assert draws.shape == (3,)

    def test_conditioning_a_produced_field_of_a_kernel_keeps_it_a_kernel(self, approximate_method):
        joint = Kernel("y", ("mu",)) * Kernel("z", ("mu",))
        kernel = condition_on(joint, {"y": 0.0})
        assert isinstance(kernel, ConditionalDistribution)
        assert set(kernel.given_spec) == {"mu"}
        assert approximate_method.targets == []
        assert condition_on(kernel, {"mu": 1.0}).loc == 4.0
        (target,) = approximate_method.targets
        assert isinstance(target, _UnnormalizedConditional)
        assert set(target.event_spec.components) == {"z"}

    def test_the_target_of_a_law_that_samples_claims_its_density(self, approximate_method):
        joint = Kernel("y", ("mu",)) * Kernel("z", ("mu",))
        law = condition_on(joint, {"mu": 1.0, "y": 0.0})
        assert law.loc == 4.0
        (target,) = approximate_method.targets
        assert isinstance(target, SupportsUnnormalizedLogProb)
        assert not isinstance(target, SupportsSampling)


class TestApproximateKernels:
    def test_evaluating_an_approximate_kernel_is_approximate(self):
        report = condition_on.check(_AmortizedKernel(), {"y": 0.3})
        assert (report.route, report.method, report.exact) == ("curry", None, False)
        assert condition_on(_AmortizedKernel(), {"y": 0.3}).loc == 3.0

    def test_exact_only_excludes_currying_an_approximate_kernel(self):
        with pytest.raises(ResolutionError, match="SupportsApproximateConditioning"):
            condition_on.with_options(exact_only=True)(_AmortizedKernel(), {"y": 0.3})


class TestTheOperation:
    def test_the_routes_are_listed_in_selection_order(self):
        names = [route.name for route in condition_on.summary().routes]
        assert names == [
            "curry",
            "slice",
            "exact_conditioning",
            "bayes",
            "approximate_conditioning",
            "unnormalized",
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
