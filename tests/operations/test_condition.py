"""Contract tests of condition_on: its exact stage, its normalization stage, and their routes."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import MultivariateNormal, Record, RecordSpec
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
from probpipe.distributions._empirical import EmpiricalDistribution
from probpipe.operations._condition import (
    _INFERENCE_METHOD_CONTROLS,
    InferenceMethod,
    _UnnormalizedConditional,
    condition_on,
    inference_method_registry,
)
from probpipe.operations._convert import convert
from probpipe.operations._operation import ApplicabilityError, _RegistryRoute
from probpipe.operations._sample import sample

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

    def check(self, target: Any, /, **options: Any) -> Feasibility:
        read = target.joint if isinstance(target, _UnnormalizedConditional) else target
        return Feasibility(isinstance(read, self._types), f"{self._name} does not apply")

    def execute(self, target: Any, /, **options: Any) -> Any:
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


class _UndeclaredKernel(ConditionalDistribution):
    """A kernel whose laws are unnormalized, which it implements without declaring a capability."""

    def __init__(self, name: str = "theta") -> None:
        super().__init__(name, {"data": REAL}, REAL)

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        return Unnormalized(self.name)


class _AmortizedKernel(
    ConditionalDistribution, SupportsApproximateConditioning, SupportsConditionalSampling
):
    """A learned kernel from ``y`` to ``theta``, whose evaluation stands in for a posterior."""

    def __init__(self, name: str = "theta") -> None:
        super().__init__(name, {"y": REAL}, REAL)

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        object.__setattr__(self, "options", kwargs)
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

    def test_the_target_of_a_curried_law_records_the_curry(self, suite_methods):
        exact, _ = suite_methods
        kernel = _UnnormalizedKernel()
        condition_on(kernel, {"data": 1.0})
        (target,) = exact.targets
        assert target.provenance.operation == "condition_on"
        assert target.provenance.metadata == {"stage": "exact", "route": "curry"}
        (parent,) = target.provenance.parents
        assert (parent.type_name, parent.name) == ("_UnnormalizedKernel", kernel.name)

    def test_the_target_of_bayes_rule_records_the_curry_of_its_slots(self, approximate_method):
        joint = Kernel("y", ("mu",)) * Kernel("z", ("mu",))
        condition_on(joint, {"mu": 1.0, "y": 0.0})
        (target,) = approximate_method.targets
        assert target.provenance.metadata == {"stage": "exact", "route": "bayes"}
        (law,) = target.provenance.parents
        assert law.provenance.metadata == {"stage": "exact", "route": "curry"}

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

    def test_a_per_value_kernel_passes_the_budgets_to_its_method_only(
        self, approximate_method, tmp_path
    ):
        from probpipe.families import StanModel

        program = tmp_path / "mean.stan"
        program.write_text(
            "data { int N; vector[N] y; } parameters { real mu; } model { y ~ normal(mu, 1); }"
        )
        kernel = condition_on(StanModel("mean", str(program)), {"N": 3})
        assert set(kernel.given_spec) == {"y"}
        view = condition_on.with_options(num_results=30, num_warmup=7)
        assert view(kernel, {"y": [1.0, 2.0, 3.0]}).loc == 4.0
        assert approximate_method.options == [{"num_results": 30, "num_warmup": 7}]

    def test_check_names_the_method_that_normalizes_the_bound_law(self, approximate_method):
        kernel = condition_on(_UnnormalizedKernel("theta", ("a", "b")), {"a": 1.0})
        report = condition_on.check(kernel, {"b": 2.0})
        assert (report.route, report.method, report.exact) == (
            "curry",
            "operations_suite_factored",
            False,
        )
        assert approximate_method.targets == []

    def test_exact_only_declines_a_per_value_kernel_no_exact_method_normalizes(self, monkeypatch):
        _normalize_with(
            monkeypatch, _SuiteMethod("operations_suite_approximate", False, (Unnormalized,), 9.0)
        )
        view = condition_on.with_options(exact_only=True)
        report = view.check(_UnnormalizedKernel("theta", ("a", "b")), {"a": 1.0})
        assert report.feasible is False
        assert 'method="unnormalized"' in report.description
        with pytest.raises(ResolutionError, match='method="unnormalized"'):
            view(_UnnormalizedKernel("theta", ("a", "b")), {"a": 1.0})

    def test_binding_the_rest_of_an_exact_per_value_kernel_reports_its_method(self, suite_methods):
        view = condition_on.with_options(exact_only=True)
        kernel = view(_UnnormalizedKernel("theta", ("a", "b")), {"a": 1.0})
        report = condition_on.check(kernel, {"b": 2.0})
        assert (report.route, report.method, report.exact) == (
            "curry",
            "operations_suite_exact",
            True,
        )
        assert condition_on(kernel, {"b": 2.0}).loc == 0.5

    def test_binding_the_rest_raises_up_front_when_no_exact_method_applies(self, monkeypatch):
        _normalize_with(
            monkeypatch, _SuiteMethod("operations_suite_exact", True, (_Conjugate,), 0.5)
        )
        kernel = condition_on.with_options(exact_only=True)(
            _UnnormalizedKernel("theta", ("a", "b")), {"a": 1.0}
        )
        report = condition_on.check(kernel, {"b": 2.0})
        assert report.feasible is False
        assert 'method="unnormalized"' in report.description
        with pytest.raises(ResolutionError, match='method="unnormalized"'):
            condition_on(kernel, {"b": 2.0})

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

    def test_check_on_mixed_keys_of_a_per_value_kernel_runs_no_method(self, approximate_method):
        joint = Kernel("y", ("mu",)) * Kernel("z", ("mu",)) * Kernel("w", ("mu",))
        kernel = condition_on(joint, {"y": 0.0})
        report = condition_on.check(kernel, {"mu": 1.0, "z": 0.5})
        assert (report.route, report.method) == ("bayes", "operations_suite_factored")
        assert approximate_method.targets == []

    def test_mixed_keys_of_a_per_value_kernel_bind_its_slots_then_condition_once(
        self, approximate_method
    ):
        joint = Kernel("y", ("mu",)) * Kernel("z", ("mu",)) * Kernel("w", ("mu",))
        kernel = condition_on(joint, {"y": 0.0})
        assert condition_on(kernel, {"mu": 1.0, "z": 0.5}).loc == 4.0
        (target,) = approximate_method.targets
        assert isinstance(target, _UnnormalizedConditional)
        assert set(target.event_spec.components) == {"w"}
        assert set(target.joint.event_spec.components) == {"y", "z", "w"}
        assert set(target.given.fields) == {"y", "z"}

    def test_conditioning_a_per_value_kernel_on_a_field_conditions_its_unnormalized_laws(
        self, approximate_method
    ):
        joint = Kernel("y", ("mu",)) * Kernel("z", ("mu",)) * Kernel("w", ("mu",))
        kernel = condition_on(condition_on(joint, {"y": 0.0}), {"z": 0.5})
        assert approximate_method.targets == []
        assert condition_on(kernel, {"mu": 1.0}).loc == 4.0
        (target,) = approximate_method.targets
        assert set(target.joint.event_spec.components) == {"y", "z", "w"}
        assert set(target.given.fields) == {"y", "z"}

    def test_the_target_of_a_law_that_samples_claims_its_density(self, approximate_method):
        joint = Kernel("y", ("mu",)) * Kernel("z", ("mu",))
        law = condition_on(joint, {"mu": 1.0, "y": 0.0})
        assert law.loc == 4.0
        (target,) = approximate_method.targets
        assert isinstance(target, SupportsUnnormalizedLogProb)
        assert not isinstance(target, SupportsSampling)


class TestAKernelThatDeclaresNothingAboutItsLaws:
    def test_check_reports_the_curry_unresolved(self, suite_methods):
        report = condition_on.check(Kernel(), {"mu": 1.0})
        assert report.feasible is None
        assert (report.route, report.method) == (None, None)
        assert any("declares no conditional capability" in entry for entry in report.pending)
        assert suite_methods[0].targets == suite_methods[1].targets == []

    def test_the_call_returns_a_normalized_law_without_inference(self, suite_methods):
        exact, approximate = suite_methods
        law = condition_on(Kernel(), {"mu": 1.0})
        assert isinstance(law, Gaussian)
        assert law.loc == 1.0
        assert exact.targets == approximate.targets == []

    def test_exact_only_returns_a_normalized_law(self):
        assert condition_on.with_options(exact_only=True)(Kernel(), {"mu": 1.0}).loc == 1.0

    def test_a_named_method_does_not_run_on_a_normalized_law(self, suite_methods):
        view = condition_on.with_options(method="operations_suite_approximate")
        assert view.check(Kernel(), {"mu": 1.0}).feasible is None
        with pytest.raises(ResolutionError, match="normalized"):
            view(Kernel(), {"mu": 1.0})
        assert suite_methods[1].targets == []

    def test_the_call_normalizes_an_unnormalized_law_by_a_method(self, suite_methods):
        exact, _ = suite_methods
        assert condition_on.check(_UndeclaredKernel(), {"data": 1.0}).feasible is None
        assert condition_on(_UndeclaredKernel(), {"data": 1.0}).loc == 0.5
        (target,) = exact.targets
        assert isinstance(target, Unnormalized)

    def test_binding_some_slots_returns_the_curried_kernel(self, suite_methods):
        curried = condition_on(Kernel("y", ("a", "b")), {"a": 1.0})
        assert type(curried) is Kernel
        assert set(curried.given_spec) == {"b"}
        assert condition_on(curried, {"b": 2.0}).loc == 3.0


class TestApproximateKernels:
    def test_evaluating_an_approximate_kernel_is_approximate(self):
        report = condition_on.check(_AmortizedKernel(), {"y": 0.3})
        assert (report.route, report.method, report.exact) == ("curry", None, False)
        assert condition_on(_AmortizedKernel(), {"y": 0.3}).loc == 3.0

    def test_exact_only_excludes_currying_an_approximate_kernel(self):
        with pytest.raises(ResolutionError, match="SupportsApproximateConditioning"):
            condition_on.with_options(exact_only=True)(_AmortizedKernel(), {"y": 0.3})

    def test_an_approximate_kernel_receives_the_budgets_it_reads(self):
        kernel = _AmortizedKernel()
        condition_on.with_options(num_results=7, random_seed=3)(kernel, {"y": 0.3})
        assert kernel.options == {"num_results": 7, "random_seed": 3}


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


# ---------------------------------------------------------------------------
# End to end: condition_on returns a normalized law for each kind of model
# ---------------------------------------------------------------------------

_MCMC = {"num_results": 60, "num_warmup": 60, "random_seed": 0}


def _logistic_joint():
    from probpipe.families import BernoulliFamily, glm_likelihood

    X = jnp.array([[1.0, 0.5], [1.0, -0.3], [1.0, 1.2], [1.0, -1.1]])
    likelihood = glm_likelihood("y", BernoulliFamily(), X=X)
    return likelihood * MultivariateNormal("beta", jnp.zeros(2), jnp.eye(2))


def _unnormalized_pair():
    """The unnormalized law of ``(a, b)`` with ``b | a ~ N(a, 1)`` and ``a ~ N(0, 1)``."""
    from probpipe import NumericArraySpec
    from probpipe.families import UnnormalizedDistribution

    return UnnormalizedDistribution(
        "pair",
        lambda v: -0.5 * (jnp.asarray(v["a"]) ** 2 + (jnp.asarray(v["b"]) - v["a"]) ** 2),
        OutputSpec(RecordSpec(a=NumericArraySpec(()), b=NumericArraySpec(()))),
    )


def _unnormalized_vector():
    from probpipe import NumericArraySpec
    from probpipe.families import UnnormalizedDistribution

    return UnnormalizedDistribution(
        "u", lambda x: -0.5 * jnp.sum((x - 1.0) ** 2), OutputSpec(x=NumericArraySpec((2,)))
    )


def _refuse_to_execute(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("check ran an inference method")


class _WholeTermKernel(ConditionalDistribution, SupportsConditionalUnnormalizedLogProb):
    """``theta | s``, whose laws are unnormalized over the whole-term event ``theta`` in R²."""

    def __init__(self) -> None:
        from probpipe import NumericArraySpec

        super().__init__("theta", {"s": REAL}, OutputSpec(theta=NumericArraySpec((2,))))

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        from probpipe.families import UnnormalizedDistribution

        s = float(given["s"])
        return UnnormalizedDistribution(
            "theta", lambda x: -0.5 * jnp.sum((jnp.asarray(x) - s) ** 2), self.event_spec
        )

    def _conditional_unnormalized_log_prob(self, given: Any, value: Any) -> Any:
        return self._condition_on(given)._unnormalized_log_prob(value)


class TestEndToEnd:
    def test_a_normal_kernel_bound_to_a_value_needs_no_inference(self):
        from probpipe.families import GaussianFamily, glm_likelihood

        X = jnp.array([[1.0, 0.5], [1.0, -0.3], [1.0, 1.2]])
        beta = jnp.array([0.2, -0.1])
        likelihood = glm_likelihood("y", GaussianFamily(), X=X, dispersion=1.0)
        report = condition_on.check(likelihood, {"beta": beta})
        assert (report.route, report.method, report.exact) == ("curry", None, True)
        law = condition_on(likelihood, {"beta": beta})
        assert _is_normalized(law)
        np.testing.assert_allclose(law._mean(), X @ beta, rtol=1e-6)

    def test_a_joint_conditioned_on_its_observation_is_normalized_by_a_method(self):
        joint, y = _logistic_joint(), {"y": jnp.array([1, 0, 1, 0])}
        view = condition_on.with_options(**_MCMC)
        report = view.check(joint, y)
        assert (report.route, report.method, report.exact) == ("bayes", "blackjax_nuts", False)
        posterior = view(joint, y)
        assert _is_normalized(posterior)
        assert tuple(posterior.event_spec.components) == ("beta",)

    def test_an_unnormalized_distribution_conditioned_on_a_field_is_normalized(self):
        view = condition_on.with_options(**_MCMC)
        posterior = view(_unnormalized_pair(), {"b": 1.0})
        assert _is_normalized(posterior)
        assert tuple(posterior.event_spec.components) == ("a",)
        assert view.check(_unnormalized_pair(), {"b": 1.0}).method == "blackjax_nuts"

    @pytest.mark.parametrize("kernel", [_UnnormalizedKernel(), _WholeTermKernel()])
    def test_a_curried_law_over_a_whole_term_is_normalized_under_that_term(self, kernel):
        view = condition_on.with_options(**_MCMC)
        given = dict.fromkeys(kernel.given_spec, 1.0)
        assert view.check(kernel, given).method == "blackjax_nuts"
        posterior = view(kernel, given)
        assert _is_normalized(posterior)
        assert not posterior.event_spec.exposes_record
        assert tuple(posterior.event_spec.components) == ("theta",)
        assert posterior.event_spec.spec.shape == kernel.event_spec.spec.shape

    def test_an_unnormalized_distribution_samples_through_a_method(self):
        view = sample.with_options(**_MCMC)
        report = view.check(_unnormalized_vector())
        assert (report.route, report.method, report.exact) == ("normalize", "blackjax_nuts", False)
        assert jnp.shape(jnp.asarray(view(_unnormalized_vector()).value)) == (2,)
        assert view(_unnormalized_vector(), sample_shape=(5,)).batch_shape == (5,)

    def test_a_law_that_samples_does_not_normalize(self):
        assert sample.check(Gaussian("g")).route == "exact"
        with pytest.raises(ResolutionError):
            sample(Bare("b"))

    def test_an_unnormalized_distribution_converts_through_a_method(self):
        view = convert.with_options(**_MCMC)
        law = _unnormalized_vector()
        assert view.check(law, EmpiricalDistribution).route == "normalize"
        empirical = view(law, EmpiricalDistribution)
        assert isinstance(empirical, EmpiricalDistribution)
        assert tuple(empirical.event_spec.components) == ("x",)
        assert isinstance(view(law, SupportsSampling), SupportsSampling)

    def test_a_pymc_model_with_a_covariate_bound_and_its_observation_conditioned(self):
        pm = pytest.importorskip("pymc")
        from probpipe.families import PyMCModel

        def regression(x=None, y=None):
            x = np.zeros(3) if x is None else np.asarray(x)
            with pm.Model() as model:
                beta = pm.Normal("beta", 0, 1)
                sigma = pm.HalfNormal("sigma", 1)
                pm.Normal("y", beta * x, sigma, observed=y)
            return model

        kernel = PyMCModel("regression", regression)
        given = {"x": np.linspace(0.0, 1.0, 6), "y": np.linspace(0.0, 1.0, 6)}
        view = condition_on.with_options(num_results=30, num_warmup=30, num_chains=1)
        report = view.check(kernel, given)
        assert report.route == "bayes"
        assert report.method in ("nutpie_nuts", "pymc_nuts")
        posterior = view(kernel, given)
        assert _is_normalized(posterior)
        assert set(posterior.event_spec.components) == {"beta", "sigma"}

    def test_a_pymc_kernel_conditioned_on_its_observation_binds_a_covariate_and_a_parameter(
        self, monkeypatch
    ):
        pm = pytest.importorskip("pymc")
        from probpipe.families import PyMCModel

        def regression(x=None, y=None):
            x = np.zeros(3) if x is None else np.asarray(x)
            with pm.Model() as model:
                beta = pm.Normal("beta", 0, 1)
                sigma = pm.HalfNormal("sigma", 1)
                pm.Normal("y", beta * x, sigma, observed=y)
            return model

        view = condition_on.with_options(num_results=30, num_warmup=30, num_chains=1)
        kernel = view(PyMCModel("regression", regression), {"y": np.linspace(0.0, 1.0, 6)})
        given = {"x": np.linspace(0.0, 1.0, 6), "beta": 0.3}
        with monkeypatch.context() as patched:
            patched.setattr(inference_method_registry, "execute", _refuse_to_execute)
            report = view.check(kernel, given)
        assert report.route == "bayes"
        assert report.method in ("nutpie_nuts", "pymc_nuts")
        posterior = view(kernel, given)
        assert _is_normalized(posterior)
        assert set(posterior.event_spec.components) == {"sigma"}

    def test_a_stan_model_bound_to_its_data_is_normalized_by_a_stan_method(self, tmp_path):
        from probpipe.families import StanModel

        program = tmp_path / "normal_mean.stan"
        program.write_text(
            "data { int N; vector[N] y; } parameters { real mu; } "
            "model { mu ~ normal(0, 1); y ~ normal(mu, 1); }"
        )
        report = condition_on.check(StanModel("mean", str(program)), {"N": 3, "y": [1.0, 2.0, 3.0]})
        assert report.route == "curry"
        stan_methods = {"nutpie_nuts", "cmdstan_nuts"} & set(
            inference_method_registry.list_methods()
        )
        if not stan_methods:
            pytest.skip("no Stan method is registered here")
        assert report.method in stan_methods | {"blackjax_rwmh"}

    @pytest.mark.usefixtures("_stan_toolchain")
    def test_a_stan_posterior_is_sampled_by_a_stan_method(self, tmp_path):
        from probpipe.families import StanModel

        program = tmp_path / "normal_mean.stan"
        program.write_text(
            "data { int N; vector[N] y; } parameters { real mu; } "
            "model { mu ~ normal(0, 1); y ~ normal(mu, 1); }"
        )
        view = condition_on.with_options(num_results=30, num_warmup=30, num_chains=1)
        posterior = view(StanModel("mean", str(program)), {"N": 3, "y": [1.0, 2.0, 3.0]})
        assert _is_normalized(posterior)

    def test_unnormalized_returns_the_exact_stage_of_a_joint(self):
        target = condition_on.with_options(method="unnormalized")(
            _logistic_joint(), {"y": jnp.array([1, 0, 1, 0])}
        )
        assert isinstance(target, _UnnormalizedConditional)
        assert not _is_normalized(target)
        assert tuple(target.event_spec.components) == ("beta",)

    def test_exact_only_raises_for_a_joint_naming_the_unnormalized_method(self):
        with pytest.raises(ResolutionError, match='method="unnormalized"'):
            condition_on.with_options(exact_only=True)(
                _logistic_joint(), {"y": jnp.array([1, 0, 1, 0])}
            )

    @pytest.mark.pending(
        reason="the engine keeps the selected route's record as a parent of its own",
        raises=AssertionError,
    )
    def test_provenance_names_both_stages(self, full_provenance_mode):
        from probpipe import provenance_ancestors

        posterior = condition_on.with_options(**_MCMC)(
            _logistic_joint(), {"y": jnp.array([1, 0, 1, 0])}
        )
        operations = {
            ancestor.parent.provenance.operation
            for ancestor in provenance_ancestors(posterior)
            if getattr(ancestor, "parent", None) is not None
            and ancestor.parent.provenance is not None
        }
        assert {"blackjax_nuts", "condition_on"} <= operations
