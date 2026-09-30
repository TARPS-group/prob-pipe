"""Contract tests of the distribution functionals: mean, variance, cov, quantile, expectation."""

from __future__ import annotations

import inspect
import math
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecordBatch,
    Record,
    RecordSpec,
    workflow_run,
)
from probpipe.core._dispatch import (
    Feasibility,
    MathematicalDomainError,
    ResolutionError,
    UnaryDispatchRegistry,
)
from probpipe.core._specs import OutputSpec
from probpipe.core.constraints import non_negative, real, unit_interval
from probpipe.distributions._capabilities import SupportsConditionalSampling, SupportsSampling
from probpipe.distributions._conditional import ConditionalDistribution
from probpipe.distributions._distribution import Distribution
from probpipe.distributions._empirical import EmpiricalDistribution
from probpipe.linalg import LinOp
from probpipe.operations import RouteSource
from probpipe.operations._moments import (
    ExpectationMethod,
    cov,
    expectation,
    expectation_method_registry,
    mean,
    quantile,
    variance,
)
from probpipe.operations._operation import ApplicabilityError
from probpipe.values import Function

from ._laws import (
    REAL,
    Bare,
    Coin,
    ExactPosterior,
    Gaussian,
    GuardedMean,
    Measure,
    Pair,
    Sampler,
    Vector,
)

_DRAWS = 4000


class _Shift(ConditionalDistribution, SupportsConditionalSampling):
    """The kernel ``y | mu``, a point mass one above its given, which only samples."""

    def __init__(self) -> None:
        super().__init__("y", {"mu": REAL}, OutputSpec(y=REAL))

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        raise NotImplementedError("the moment tests bind no given of the kernel")

    def _conditional_sample(self, given: Any, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return jnp.asarray(given["mu"], jnp.float32) + jnp.ones(sample_shape, jnp.float32)


def _dependent_joint() -> Any:
    """``y = mu + 1`` with ``mu ~ Normal(2, 1)``: a joint that samples and has no moment."""
    return _Shift() * Gaussian("mu", 2.0)


class _Ramp(Distribution, SupportsSampling):
    """A law whose i-th of n draws is ``(i, 2i)``, whatever the key.

    A draw is an array of shape (2,), or with ``record=True`` the record of the
    scalar fields ``x`` and ``y``, returned as its mapping.
    """

    def __init__(self, name: str, *, record: bool = False) -> None:
        pair = RecordSpec(x=REAL, y=REAL)
        super().__init__(name, pair if record else NumericArraySpec((2,), jnp.float32, real))
        self.record = record

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        index = jnp.arange(math.prod(sample_shape), dtype=jnp.float32).reshape(sample_shape)
        if self.record:
            return {"x": index, "y": 2.0 * index}
        return jnp.stack([index, 2.0 * index], axis=-1)


def _record_empirical() -> EmpiricalDistribution:
    """Three equally weighted record atoms, whose leaves ``b`` and ``a`` each rank the atoms alike."""
    atoms = NumericRecordBatch(
        "rows",
        {"b": jnp.array([[1.0, 2.0], [0.0, 1.0], [2.0, 3.0]]), "a": jnp.array([2.0, 1.0, 3.0])},
        "row",
        element_spec=RecordSpec(b=(2,), a=()),
    )
    return EmpiricalDistribution("post", atoms)


class _QuadratureStandIn(ExpectationMethod):
    """An opt-in method that returns a marker, reachable only by name."""

    @property
    def name(self) -> str:
        return "operations_suite_quadrature"

    @property
    def exact(self) -> bool:
        return False

    def check(self, d: Any, f: Any, /, **options: Any) -> Feasibility:
        return Feasibility(True)

    def execute(self, d: Any, f: Any, /, **options: Any) -> Any:
        return jnp.float32(7.0)


@pytest.fixture
def quadrature(monkeypatch):
    """The expectation route, delegating for one test to the shipped methods and the stand-in.

    The stand-in stays out of the global method registry, which other suites
    inspect.
    """
    registry: UnaryDispatchRegistry = UnaryDispatchRegistry()
    for name in expectation_method_registry.list_methods():
        registry.register(expectation_method_registry.get_method(name))
    registry.register(_QuadratureStandIn())
    (route,) = expectation.routes
    monkeypatch.setattr(route, "registry", registry)
    return registry


def _value(term: Any) -> float:
    return float(jnp.asarray(term))


class TestMean:
    def test_the_closed_form_mean_has_the_event_declaration(self):
        result = mean(Gaussian("g", 2.0))
        assert isinstance(result, NumericArray)
        assert result.name == "mean"
        assert result.spec == NumericArraySpec((), jnp.float32, real)
        assert _value(result) == 2.0

    def test_the_result_keeps_the_event_components_and_packaging(self):
        assert mean.check(Gaussian("g")).result == OutputSpec(
            g=NumericArraySpec((), jnp.float32, real)
        )
        assert mean.check(Pair("p")).result.exposes_record

    def test_a_record_law_has_a_record_mean_with_its_schema(self):
        result = mean(Pair("p"))
        assert isinstance(result, Record)
        assert _value(result["a"]) == 1.0
        np.testing.assert_array_equal(np.asarray(result["b"]), [-1.0, -1.0])

    def test_the_mean_of_a_bernoulli_event_is_floating_on_the_unit_interval(self):
        result = mean(Coin("c", 0.25))
        assert result.spec.support is unit_interval
        assert np.issubdtype(result.spec.dtype, np.floating)
        assert _value(result) == 0.25

    def test_a_law_without_a_closed_form_takes_the_monte_carlo_fallback(self):
        law = Sampler("s", 2.0)
        report = mean.check(law)
        assert (report.route, report.exact) == ("monte_carlo", False)
        with workflow_run(seed=1):
            estimate = mean.with_options(n_broadcast_samples=_DRAWS)(law)
        assert abs(_value(estimate) - 2.0) < 0.1

    def test_the_fallback_averages_record_draws_per_field(self):
        with workflow_run(seed=2):
            estimate = mean.with_options(n_broadcast_samples=_DRAWS)(ExactPosterior("post"))
        assert isinstance(estimate, Record)
        assert abs(_value(estimate["theta"])) < 0.1 and abs(_value(estimate["y"])) < 0.1

    def test_method_selects_the_fallback_over_the_closed_form(self):
        law = Gaussian("g", -1.0)
        assert mean.with_options(method="monte_carlo").check(law).route == "monte_carlo"

    def test_a_declining_guard_passes_to_the_fallback(self):
        assert mean.check(GuardedMean("g", False)).route == "monte_carlo"

    def test_exact_only_refuses_a_law_without_a_closed_form(self):
        with pytest.raises(ResolutionError, match="exact_only"):
            mean.with_options(exact_only=True)(Sampler("s"))

    def test_a_law_that_neither_has_a_mean_nor_samples_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="does not sample"):
            mean(Bare("b"))

    @pytest.mark.pending(reason="the Monte Carlo mean of measure-valued draws is their mixture")
    def test_the_fallback_mean_of_a_measure_is_the_mixture_of_its_draws(self):
        assert mean(Measure("m")).event_spec == Measure("m").event_spec.spec.event_spec


class TestVariance:
    def test_the_closed_form_variance_is_non_negative(self):
        result = variance(Gaussian("g", 0.0, 2.0))
        assert _value(result) == 4.0
        assert result.spec.support is non_negative

    def test_the_fallback_is_the_sample_variance(self):
        with workflow_run(seed=3):
            estimate = variance.with_options(n_broadcast_samples=_DRAWS)(Sampler("s", 0.0, 2.0))
        assert abs(_value(estimate) - 4.0) < 0.4

    def test_a_measure_valued_event_has_no_variance(self):
        with pytest.raises(ApplicabilityError, match="event-typed variance"):
            variance(Measure("m"))


class TestCov:
    def test_the_closed_form_covariance_is_the_dense_matrix_of_the_flattened_draw(self):
        result = cov(Gaussian("g", 0.0, 2.0))
        assert isinstance(result, NumericArray)
        assert result.spec.shape == (1, 1)
        np.testing.assert_array_equal(np.asarray(result), [[4.0]])

    def test_the_fallback_is_the_sample_covariance(self):
        with workflow_run(seed=4):
            estimate = cov.with_options(n_broadcast_samples=_DRAWS)(Vector("v"))
        assert estimate.spec.shape == (2, 2)
        np.testing.assert_allclose(np.asarray(estimate), np.diag([1.0, 4.0]), atol=0.3)

    def test_the_covariance_requires_a_numeric_event(self):
        with pytest.raises(ApplicabilityError, match="numeric value"):
            cov(Measure("m"))

    @pytest.mark.pending(
        reason="cov returns a LinOp once LinOp is a Function", raises=AssertionError
    )
    def test_the_covariance_is_an_operator(self):
        assert isinstance(cov(Gaussian("g")), LinOp)


class TestQuantile:
    def test_one_level_returns_the_event_kind(self):
        result = quantile(Gaussian("g", 1.0), 0.5)
        assert isinstance(result, NumericArray)
        assert _value(result) == pytest.approx(1.0)

    def test_several_levels_add_a_level_named_quantile(self):
        result = quantile(Gaussian("g"), jnp.array([0.1, 0.5, 0.9]))
        assert isinstance(result, NumericArrayBatch)
        assert (result.level_names, result.batch_shape) == (("quantile",), (3,))
        assert _value(result.values[1]) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("q", [1.5, -0.1, jnp.array([0.5, 2.0])])
    def test_a_level_outside_the_unit_interval_raises_a_domain_error(self, q):
        with pytest.raises(MathematicalDomainError, match=r"\[0, 1\]"):
            quantile(Gaussian("g"), q)

    def test_the_fallback_is_the_empirical_quantile(self):
        with workflow_run(seed=5):
            estimate = quantile.with_options(n_broadcast_samples=_DRAWS)(Sampler("s"), 0.5)
        assert abs(_value(estimate)) < 0.1

    @pytest.mark.parametrize("record", [False, True], ids=["array", "record"])
    def test_the_fallback_is_the_inverse_cdf_of_the_draws(self, record):
        # Four draws of x are 0, 1, 2, 3, whose CDF reaches 0.25 at 0.
        view = quantile.with_options(n_broadcast_samples=4)
        estimate = view(_Ramp("ramp", record=record), jnp.array([0.25, 0.5, 1.0]))
        x = estimate["x"] if record else estimate.values[:, 0]
        np.testing.assert_array_equal(np.asarray(x), [0.0, 1.0, 3.0])

    def test_method_selects_the_fallback_over_the_closed_form(self):
        view = quantile.with_options(method="monte_carlo")
        assert view.check(Gaussian("g"), 0.5).route == "monte_carlo"

    def test_one_level_of_a_record_law_is_a_record_of_its_quantiles(self):
        result = quantile(_record_empirical(), 0.5)
        assert isinstance(result, Record) and result.fields == ("b", "a")
        np.testing.assert_allclose(np.asarray(result["b"]), [1.0, 2.0])
        assert _value(result["a"]) == 2.0

    def test_several_levels_of_a_record_law_are_a_batch_of_records(self):
        result = quantile(_record_empirical(), jnp.array([0.0, 0.5, 1.0]))
        assert isinstance(result, NumericRecordBatch)
        assert (result.level_names, result.batch_shape) == (("quantile",), (3,))
        np.testing.assert_allclose(np.asarray(result["a"]), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(np.asarray(result["b"]), [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

    def test_a_raw_record_law_s_levels_are_the_mapping_of_their_columns(self):
        result = quantile.with_options(raw=True)(_record_empirical(), jnp.array([0.0, 1.0]))
        assert isinstance(result, dict) and list(result) == ["b", "a"]
        assert jnp.shape(result["b"]) == (2, 2)

    def test_quantiles_require_a_numeric_event(self):
        with pytest.raises(ApplicabilityError, match="numeric value"):
            quantile(Measure("m"), 0.5)

    def test_the_levels_are_numeric(self):
        with pytest.raises(ApplicabilityError, match="levels"):
            quantile(Gaussian("g"), "median")


class TestTheFallbacksOfAJoint:
    """A joint's draws are a mapping of columns, which each fallback reads per component."""

    def test_the_mean_is_the_average_of_each_component(self):
        with workflow_run(seed=7):
            estimate = mean.with_options(n_broadcast_samples=_DRAWS)(_dependent_joint())
        assert isinstance(estimate, Record) and estimate.fields == ("y", "mu")
        assert abs(_value(estimate["mu"]) - 2.0) < 0.1
        assert abs(_value(estimate["y"]) - 3.0) < 0.1

    def test_the_variance_is_the_sample_variance_of_each_component(self):
        with workflow_run(seed=8):
            estimate = variance.with_options(n_broadcast_samples=_DRAWS)(_dependent_joint())
        assert isinstance(estimate, Record)
        assert abs(_value(estimate["mu"]) - 1.0) < 0.15
        assert abs(_value(estimate["y"]) - 1.0) < 0.15

    def test_the_covariance_couples_the_components(self):
        with workflow_run(seed=9):
            estimate = cov.with_options(n_broadcast_samples=_DRAWS)(_dependent_joint())
        np.testing.assert_allclose(np.asarray(estimate), np.ones((2, 2)), atol=0.15)

    def test_several_quantile_levels_give_a_batch_of_records(self):
        levels = jnp.array([0.25, 0.5])
        with workflow_run(seed=10):
            estimate = quantile.with_options(n_broadcast_samples=_DRAWS)(_dependent_joint(), levels)
        assert (estimate.level_names, estimate.batch_shape) == (("quantile",), (2,))
        assert abs(float(estimate["mu"][1]) - 2.0) < 0.1
        assert abs(float(estimate["y"][1]) - 3.0) < 0.1


class TestExpectation:
    def test_a_finite_support_law_takes_the_exact_method(self):
        law = Coin("c", 0.25)
        report = expectation.check(law, lambda x: 2.0 * x)
        assert (report.route, report.method, report.exact) == ("methods", "exact", True)
        assert _value(expectation(law, lambda x: 2.0 * x)) == pytest.approx(0.5)

    def test_monte_carlo_is_the_default_approximate_method(self):
        law = Gaussian("g")
        report = expectation.check(law, lambda x: x**2)
        assert (report.method, report.exact) == ("monte_carlo", False)
        with workflow_run(seed=6):
            estimate = expectation.with_options(n_broadcast_samples=_DRAWS)(law, lambda x: x**2)
        assert abs(_value(estimate) - 1.0) < 0.1

    def test_method_selects_monte_carlo_for_a_law_with_an_exact_method(self):
        view = expectation.with_options(method="monte_carlo")
        assert view.check(Coin("c"), lambda x: x).method == "monte_carlo"

    def test_method_selects_an_opt_in_method_by_name(self, quadrature):
        view = expectation.with_options(method="operations_suite_quadrature")
        assert _value(view(Gaussian("g"), lambda x: x)) == 7.0

    def test_exact_only_refuses_a_law_without_an_exact_method(self):
        with pytest.raises(ResolutionError):
            expectation.with_options(exact_only=True)(Gaussian("g"), lambda x: x)

    def test_an_unregistered_method_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="no_such_method"):
            expectation.with_options(method="no_such_method")(Gaussian("g"), lambda x: x)

    def test_the_sample_count_control_is_the_monte_carlo_budget(self):
        law = Sampler("s")
        expectation.with_options(n_broadcast_samples=33)(law, lambda x: x)
        assert law.shapes == [(33,)]

    def test_the_result_takes_the_kind_the_integrand_declares(self):
        f = Function(
            "f",
            lambda x: {"sq": x**2},
            output_spec=OutputSpec(RecordSpec(sq=NumericArraySpec(()))),
        )
        result = expectation(Coin("c", 0.5), f)
        assert isinstance(result, Record)
        assert _value(result["sq"]) == pytest.approx(0.5)

    def test_an_undeclared_integrand_wraps_its_value_by_kind(self):
        assert isinstance(expectation(Coin("c"), lambda x: x), NumericArray)

    def test_the_integrand_is_a_callable(self):
        with pytest.raises(ApplicabilityError, match="FunctionSpec"):
            expectation(Gaussian("g"), 3.0)

    def test_the_one_route_delegates_to_the_method_registry(self):
        (route,) = expectation.summary().routes
        assert (route.name, route.source, route.exact) == ("methods", RouteSource.REGISTRY, None)
        assert "exact, monte_carlo" in route.condition
        assert expectation.routes[0].registry is expectation_method_registry

    def test_the_operation_takes_no_key(self):
        assert list(inspect.signature(expectation).parameters) == ["d", "f"]
