"""Contract tests of the distribution functionals: mean, variance, cov, quantile, expectation."""

from __future__ import annotations

import inspect
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
    InputSpec,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecordBatch,
    Record,
    RecordSpec,
    replay_run,
    workflow_run,
)
from probpipe.core._dispatch import (
    BinaryDispatchMethod,
    Feasibility,
    MathematicalDomainError,
    ResolutionError,
)
from probpipe.core._specs import OutputSpec
from probpipe.core.constraints import non_negative, real, unit_interval
from probpipe.distributions._capabilities import (
    SupportsConditionalSampling,
    SupportsCovariance,
    SupportsLogProb,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsVariance,
)
from probpipe.distributions._conditional import ConditionalDistribution
from probpipe.distributions._distribution import Distribution
from probpipe.distributions._empirical import EmpiricalDistribution
from probpipe.functions import _rules
from probpipe.linalg import LinOp
from probpipe.operations import RouteSource
from probpipe.operations._evaluate import evaluate
from probpipe.operations._moments import cov, expectation, mean, quantile, variance
from probpipe.values import Function, FunctionSpec

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
    normal_draws,
)

_DRAWS = 4000


class _Shift(ConditionalDistribution, SupportsConditionalSampling):
    """The kernel ``label | mu``, a point mass *offset* above its given, which only samples."""

    def __init__(self, label: str = "y", offset: float = 1.0) -> None:
        super().__init__(label, {"mu": REAL}, OutputSpec(**{label: REAL}))
        self.offset = offset

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        raise NotImplementedError("the moment tests bind no given of the kernel")

    def _conditional_sample(self, given: Any, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return jnp.asarray(given["mu"], jnp.float32) + jnp.full(
            sample_shape, self.offset, jnp.float32
        )


def _dependent_joint() -> Any:
    """``y = mu + 1`` with ``mu ~ Normal(2, 1)``: a joint that samples and has no moment."""
    return _Shift() * Gaussian("mu", 2.0)


class _Coupled(Distribution, SupportsSampling, SupportsMarginals):
    """``a ~ Normal(1, 1)`` and ``b = -a``: a dependent record that claims no moment.

    Each component's marginal is a normal law with every closed form, which
    ``_marginal_capabilities`` reports, so each component's view has exact
    moments.
    """

    def __init__(self) -> None:
        super().__init__("coupled", RecordSpec(a=REAL, b=REAL))

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        a = normal_draws(key, 1.0, 1.0, tuple(sample_shape))
        return {"a": a, "b": -a}

    def _marginal(self, path: Any) -> Any:
        return Gaussian(path, 1.0 if path == "a" else -1.0)

    def _marginal_capabilities(self, path: Any) -> frozenset[type]:
        return frozenset(
            {
                SupportsSampling,
                SupportsLogProb,
                SupportsMean,
                SupportsVariance,
                SupportsCovariance,
                SupportsQuantile,
            }
        )


class _Ramp(Distribution, SupportsSampling):
    """A law whose i-th of n draws is ``(i, 2i)``, whatever the key.

    A draw is an array of shape (2,), or with ``record=True`` the record of the
    scalar fields ``x`` and ``y``, returned as its mapping.
    """

    def __init__(self, label: str, *, record: bool = False) -> None:
        pair = RecordSpec(x=REAL, y=REAL)
        array = OutputSpec(**{label: NumericArraySpec((2,), jnp.float32, real)})
        super().__init__(label, pair if record else array)
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
    return EmpiricalDistribution(atoms, label="post")


class _QuadratureStandIn(BinaryDispatchMethod):
    """An opt-in evaluation rule whose pushforward is a point mass at 7, reachable only by name."""

    @property
    def name(self) -> str:
        return "operations_suite_quadrature"

    @property
    def exact(self) -> bool:
        return False

    @property
    def priority(self) -> None:
        return None

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((Function,), (Distribution,))

    def check(self, f: Any, operand: Any, /, **call: Any) -> Feasibility:
        return Feasibility(True)

    def execute(self, f: Any, operand: Any, /, **call: Any) -> Any:
        return EmpiricalDistribution(jnp.array([7.0]), component="stand_in")


@pytest.fixture
def quadrature(monkeypatch):
    """The evaluation rules, for one test, with the stand-in beside the engine's rules.

    The stand-in stays out of the global rule registry, which other suites
    inspect.
    """
    registry = type(_rules.evaluation_rule_registry)()
    for rule in (
        _rules._SamplingLift(),
        _rules._ElementwiseSweep(),
        _rules._EmpiricalEnumeration(),
        _QuadratureStandIn(),
    ):
        registry.register(rule)
    monkeypatch.setattr(_rules, "evaluation_rule_registry", registry)
    for operation in (expectation, evaluate):
        (route,) = [r for r in operation.routes if r.name == "evaluation_rules"]
        monkeypatch.setattr(route, "registry", registry)
    return registry


def _value(term: Any) -> float:
    return float(jnp.asarray(term))


class TestDerivedLabelsAreNeverComponents:
    """A derived label may hold ``~``, a space, ``;``, or ``/``, and no route names a part by it."""

    @staticmethod
    def _held() -> Distribution:
        """A law holding the nested path ``y/obs`` fixed, as conditioning on it records."""
        law = Gaussian("g", 2.0)
        return law._with_expression(law._expression.with_fixed(("y/obs",)))

    @pytest.mark.parametrize(
        ("moment", "label", "component"),
        [
            (mean, "E[g ~ g; y/obs]", "mean(g)"),
            (variance, "Var[g ~ g; y/obs]", "variance(g)"),
        ],
    )
    def test_a_monte_carlo_moment_is_labeled_over_the_nested_path(self, moment, label, component):
        held = self._held()
        with workflow_run(seed=0):
            result = moment.with_options(method="monte_carlo")(held)
        assert result.label == label
        assert tuple(moment.check(held).result.components) == (component,)

    def test_a_monte_carlo_quantile_names_its_component_for_the_law(self):
        held = self._held()
        with workflow_run(seed=0):
            result = quantile.with_options(method="monte_carlo")(held, jnp.array([0.1, 0.9]))
        assert result.label == "Q[g ~ g; y/obs]"
        assert result.level_names == ("quantile",)
        assert tuple(quantile.check(held, 0.5).result.components) == ("quantile(g)",)


class TestMean:
    def test_the_closed_form_mean_has_the_event_declaration(self):
        result = mean(Gaussian("g", 2.0))
        assert isinstance(result, NumericArray)
        assert result.label == "E[g ~ g]"
        assert result.spec == NumericArraySpec((), jnp.float32, real)
        assert _value(result) == 2.0

    def test_check_lists_the_closed_form_route_as_exact(self):
        report = mean.check(Gaussian("g", 2.0))
        routes = {info.method_name: info for info in report.routes}
        assert routes["closed_form"].exact is True
        assert (report.route, report.exact) == ("closed_form", True)

    def test_the_result_names_each_component_for_the_mean_in_the_event_packaging(self):
        assert mean.check(Gaussian("g")).result == OutputSpec(
            **{"mean(g)": NumericArraySpec((), jnp.float32, real)}
        )
        result = mean.check(Pair("p")).result
        assert result.exposes_record
        assert tuple(result.components) == ("mean(a)", "mean(b)")

    def test_a_record_law_has_a_record_mean_with_a_field_per_component(self):
        result = mean(Pair("p"))
        assert isinstance(result, Record) and result.fields == ("mean(a)", "mean(b)")
        assert _value(result["mean(a)"]) == 1.0
        np.testing.assert_array_equal(np.asarray(result["mean(b)"]), [-1.0, -1.0])

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

    def test_the_fallback_takes_a_law_whose_label_is_no_identifier(self):
        """The draws lie on a level named after the law's component, here ``my law``."""
        with workflow_run(seed=1):
            estimate = mean.with_options(n_broadcast_samples=_DRAWS)(Sampler("my law", 2.0))
        # Observed across seeds 0-5: errors of 0.001 to 0.020; the standard error is 0.016.
        np.testing.assert_allclose(_value(estimate), 2.0, atol=0.05)

    def test_the_fallback_averages_record_draws_per_field(self):
        with workflow_run(seed=2):
            estimate = mean.with_options(n_broadcast_samples=_DRAWS)(ExactPosterior("post"))
        assert isinstance(estimate, Record)
        assert abs(_value(estimate["mean(theta)"])) < 0.1
        assert abs(_value(estimate["mean(y)"])) < 0.1

    def test_method_selects_the_fallback_over_the_closed_form(self):
        law = Gaussian("g", -1.0)
        assert mean.with_options(method="monte_carlo").check(law).route == "monte_carlo"

    def test_a_rejecting_guard_passes_to_the_fallback(self):
        assert mean.check(GuardedMean("g", False)).route == "monte_carlo"

    def test_exact_only_refuses_a_law_without_a_closed_form(self):
        with pytest.raises(ResolutionError, match="exact_only"):
            mean.with_options(exact_only=True)(Sampler("s"))

    def test_a_law_that_neither_has_a_mean_nor_samples_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="does not sample"):
            mean(Bare("b"))

    def test_the_fallback_mean_of_a_measure_is_the_mixture_of_its_draws(self):
        assert mean(Measure("m")).event_spec == Measure("m").event_spec.spec.event_spec

    def test_the_mean_of_a_measure_is_a_law_that_exposes_its_event(self):
        """The mean measure is a law, so it takes its event's components."""
        result = mean.check(Measure("m")).result
        assert result == OutputSpec(Measure("m").event_spec.spec)
        assert not result.exposes_record


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

    def test_the_covariance_is_declared_under_the_call_on_every_component(self):
        assert list(cov.check(Gaussian("g")).result.components) == ["cov(g)"]
        assert list(cov.check(Pair("p")).result.components) == ["cov(a, b)"]

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
        x = estimate["quantile(x)"] if record else estimate.values[:, 0]
        np.testing.assert_array_equal(np.asarray(x), [0.0, 1.0, 3.0])

    def test_method_selects_the_fallback_over_the_closed_form(self):
        view = quantile.with_options(method="monte_carlo")
        assert view.check(Gaussian("g"), 0.5).route == "monte_carlo"

    def test_one_level_of_a_record_law_is_a_record_of_its_quantiles(self):
        result = quantile(_record_empirical(), 0.5)
        assert isinstance(result, Record)
        assert result.fields == ("quantile(b)", "quantile(a)")
        np.testing.assert_allclose(np.asarray(result["quantile(b)"]), [1.0, 2.0])
        assert _value(result["quantile(a)"]) == 2.0

    def test_several_levels_of_a_record_law_are_a_batch_of_records(self):
        result = quantile(_record_empirical(), jnp.array([0.0, 0.5, 1.0]))
        assert isinstance(result, NumericRecordBatch)
        assert (result.level_names, result.batch_shape) == (("quantile",), (3,))
        np.testing.assert_allclose(np.asarray(result["quantile(a)"]), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            np.asarray(result["quantile(b)"]), [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
        )

    def test_a_raw_record_law_s_levels_are_the_mapping_of_their_columns(self):
        result = quantile.with_options(raw=True)(_record_empirical(), jnp.array([0.0, 1.0]))
        assert isinstance(result, dict) and list(result) == ["quantile(b)", "quantile(a)"]
        assert jnp.shape(result["quantile(b)"]) == (2, 2)

    def test_quantiles_require_a_numeric_event(self):
        with pytest.raises(ApplicabilityError, match="numeric value"):
            quantile(Measure("m"), 0.5)

    def test_the_levels_are_numeric(self):
        with pytest.raises(
            ApplicabilityError, match="q must be a number or an array of numbers; got str"
        ):
            quantile(Gaussian("g"), "median")


class _RandomLine(Distribution, SupportsSampling):
    """A law over the maps ``x ↦ s x``, which only samples."""

    def __init__(self, label: str) -> None:
        super().__init__(
            label, OutputSpec(**{label: FunctionSpec(InputSpec(x=REAL), OutputSpec(y=REAL))})
        )

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        slopes = np.asarray(jax.random.normal(key, tuple(sample_shape)))
        if not sample_shape:
            return Function("line", lambda x: slopes * x)
        lines = np.empty(slopes.shape, dtype=object)
        for index, slope in np.ndenumerate(slopes):
            lines[index] = Function("line", lambda x, slope=slope: slope * x)
        return lines


class TestFunctionValuedEvents:
    """The Monte Carlo average of function draws is not implemented, so no call selects it."""

    @pytest.mark.parametrize("moment", [mean, variance], ids=["mean", "variance"])
    def test_check_reports_the_monte_carlo_route_infeasible(self, moment):
        routes = {info.method_name: info for info in moment.check(_RandomLine("f")).routes}
        assert routes["monte_carlo"].feasible is False
        assert "function-valued" in routes["monte_carlo"].description

    @pytest.mark.parametrize("moment", [mean, variance], ids=["mean", "variance"])
    def test_the_call_raises_resolution_error(self, moment):
        with pytest.raises(ResolutionError, match="function-valued"):
            moment(_RandomLine("f"))


class TestTheFallbacksOnFewDraws:
    """A fallback reports the moments of the empirical law of its draws, whatever the packaging."""

    @pytest.mark.parametrize("record", [False, True], ids=["array", "record"])
    def test_the_covariance_divides_by_the_number_of_draws(self, record):
        # The three draws (0, 0), (1, 2), and (2, 4) deviate from their mean (1, 2)
        # by (-1, -2), (0, 0), and (1, 2).
        estimate = cov.with_options(n_broadcast_samples=3)(_Ramp("ramp", record=record))
        expected = np.array([[2.0, 4.0], [4.0, 8.0]]) / 3.0
        np.testing.assert_allclose(np.asarray(estimate), expected, rtol=1e-6)

    @pytest.mark.parametrize("record", [False, True], ids=["array", "record"])
    def test_the_variance_is_the_diagonal_of_the_covariance(self, record):
        law = _Ramp("ramp", record=record)
        spread = variance.with_options(n_broadcast_samples=7)(law)
        covariance = cov.with_options(n_broadcast_samples=7)(law)
        if record:
            values = [_value(spread["variance(x)"]), _value(spread["variance(y)"])]
        else:
            values = np.asarray(spread)
        np.testing.assert_allclose(values, np.diag(np.asarray(covariance)), rtol=1e-6)

    @pytest.mark.parametrize("record", [False, True], ids=["array", "record"])
    def test_one_draw_has_no_covariance(self, record):
        estimate = cov.with_options(n_broadcast_samples=1)(_Ramp("ramp", record=record))
        np.testing.assert_array_equal(np.asarray(estimate), np.zeros((2, 2)))


class TestTheFallbacksOfAJoint:
    """A joint's draws are a mapping of columns, which each fallback reads per component."""

    def test_the_mean_is_the_average_of_each_component(self):
        with workflow_run(seed=7):
            estimate = mean.with_options(method="monte_carlo", n_broadcast_samples=_DRAWS)(
                _dependent_joint()
            )
        assert isinstance(estimate, Record) and estimate.fields == ("mean(y)", "mean(mu)")
        assert abs(_value(estimate["mean(mu)"]) - 2.0) < 0.1
        assert abs(_value(estimate["mean(y)"]) - 3.0) < 0.1

    def test_the_variance_is_the_sample_variance_of_each_component(self):
        with workflow_run(seed=8):
            estimate = variance.with_options(method="monte_carlo", n_broadcast_samples=_DRAWS)(
                _dependent_joint()
            )
        assert isinstance(estimate, Record)
        assert abs(_value(estimate["variance(mu)"]) - 1.0) < 0.15
        assert abs(_value(estimate["variance(y)"]) - 1.0) < 0.15

    def test_the_covariance_couples_the_components(self):
        with workflow_run(seed=9):
            estimate = cov.with_options(n_broadcast_samples=_DRAWS)(_dependent_joint())
        np.testing.assert_allclose(np.asarray(estimate), np.ones((2, 2)), atol=0.15)

    def test_several_quantile_levels_give_a_batch_of_records(self):
        levels = jnp.array([0.25, 0.5])
        with workflow_run(seed=10):
            estimate = quantile.with_options(method="monte_carlo", n_broadcast_samples=_DRAWS)(
                _dependent_joint(), levels
            )
        assert (estimate.level_names, estimate.batch_shape) == (("quantile",), (2,))
        assert abs(float(estimate["quantile(mu)"][1]) - 2.0) < 0.1
        assert abs(float(estimate["quantile(y)"][1]) - 3.0) < 0.1


class TestByComponent:
    """A record law's mean, variance, and quantile compute each component by its own route."""

    @staticmethod
    def _blocks(result: Any) -> list[tuple[tuple[str, ...], str, bool]]:
        """Each block's components, route, and exactness, as the result's provenance records them."""
        (assembled,) = [
            parent
            for parent in result.provenance.parents
            if parent.provenance is not None
            and parent.provenance.metadata.get("route") == "by_component"
        ]
        return [
            (tuple(block["components"]), block["route"], block["exact"])
            for block in assembled.provenance.metadata["blocks"]
        ]

    def test_a_dependent_joint_with_an_exact_component_takes_the_route(self):
        report = mean.check(_dependent_joint())
        assert (report.route, report.exact) == ("by_component", False)

    def test_the_exact_candidate_names_the_components_without_an_exact_route(self):
        routes = {info.method_name: info for info in mean.check(_dependent_joint()).routes}
        declined = routes["by_component (exact)"]
        assert declined.feasible is False
        assert declined.description == "the components ('y',) have no exact mean, but ('mu',) do"
        assert not declined.actionable
        approximate = routes["by_component (approximate)"]
        assert (approximate.feasible, approximate.exact) == (True, False)

    @pytest.mark.parametrize(
        ("moment", "closed_form", "seed"), [(mean, 2.0, 7), (variance, 1.0, 8)]
    )
    def test_the_exact_component_is_its_closed_form_and_the_rest_is_estimated(
        self, moment, closed_form, seed
    ):
        """``mu ~ Normal(2, 1)`` and ``y = mu + 1``: ``y`` has mean 3 and variance 1."""
        with workflow_run(seed=seed):
            result = moment.with_options(n_broadcast_samples=_DRAWS)(_dependent_joint())
        name = moment.name
        assert isinstance(result, Record)
        assert result.fields == (f"{name}(y)", f"{name}(mu)")
        assert _value(result[f"{name}(mu)"]) == closed_form
        # Observed across seeds 7, 8, 10, and 11: errors of the estimated
        # component up to 0.027 for the mean and 0.026 for the variance.
        assert abs(_value(result[f"{name}(y)"]) - (3.0 if moment is mean else 1.0)) < 0.08

    def test_several_quantile_levels_give_a_batch_of_records(self):
        levels = jnp.array([0.25, 0.5])
        with workflow_run(seed=10):
            result = quantile.with_options(n_broadcast_samples=_DRAWS)(_dependent_joint(), levels)
        assert (result.level_names, result.batch_shape) == (("quantile",), (2,))
        reference = jax.scipy.special.ndtri(levels)
        np.testing.assert_array_equal(
            np.asarray(result["quantile(mu)"]), np.asarray((2.0 + reference).astype(jnp.float32))
        )
        # Observed across seeds 7, 8, 10, and 11: errors up to 0.033.
        np.testing.assert_allclose(np.asarray(result["quantile(y)"]), 3.0 + reference, atol=0.1)

    def test_one_quantile_level_gives_a_record(self):
        with workflow_run(seed=11):
            result = quantile.with_options(n_broadcast_samples=_DRAWS)(_dependent_joint(), 0.5)
        assert isinstance(result, Record)
        assert _value(result["quantile(mu)"]) == 2.0

    def test_provenance_records_the_route_and_exactness_of_each_block(self):
        with workflow_run(seed=7):
            result = mean(_dependent_joint())
        assert result.provenance.metadata["route"] == "by_component"
        assert result.provenance.metadata["exact"] is False
        assert self._blocks(result) == [
            (("y",), "monte_carlo", False),
            (("mu",), "closed_form", True),
        ]

    def test_the_components_without_an_exact_route_are_one_block(self):
        """``z = mu - 1`` and ``y = mu + 1`` are computed from one set of draws."""
        joint = _Shift("z", -1.0) * _Shift("y", 1.0) * Gaussian("mu", 2.0)
        with workflow_run(seed=7):
            result = mean(joint)
        assert self._blocks(result) == [
            (("z", "y"), "monte_carlo", False),
            (("mu",), "closed_form", True),
        ]
        assert _value(result["mean(mu)"]) == 2.0
        # Shared draws differ by 2 at every draw, so their means do too.
        assert _value(result["mean(y)"]) - _value(result["mean(z)"]) == pytest.approx(2.0)

    def test_a_record_valued_component_is_one_block(self):
        """A group of fields is summarized whole, by the route its view selects."""
        joint = _Shift("y", 1.0) * Gaussian("mu", 2.0) * Gaussian("c", 5.0)
        nested = joint.with_path_names({"mu": "g/mu", "y": "g/y"})
        with workflow_run(seed=7):
            result = mean(nested)
        assert self._blocks(result) == [
            (("g",), "monte_carlo", False),
            (("c",), "closed_form", True),
        ]
        assert _value(result["mean(c)"]) == 5.0

    def test_a_law_whose_components_are_all_exact_takes_the_exact_candidate(self):
        law = _Coupled()
        report = mean.check(law)
        assert (report.route, report.exact) == ("by_component", True)
        result = mean.with_options(exact_only=True)(law)
        assert (_value(result["mean(a)"]), _value(result["mean(b)"])) == (1.0, -1.0)
        assert self._blocks(result) == [
            (("a",), "closed_form", True),
            (("b",), "closed_form", True),
        ]

    @pytest.mark.parametrize("moment", [mean, variance])
    def test_exact_only_says_how_to_compute_the_exact_components(self, moment):
        name = moment.name
        message = (
            rf"^{name}: the components \('y',\) have no exact {name}, but \('mu',\) do; call "
            rf"{name} on the view of each component that does, as {name}\(d\['mu'\]\)"
        )
        with pytest.raises(ResolutionError, match=message):
            moment.with_options(exact_only=True)(_dependent_joint())

    def test_exact_only_says_how_to_compute_the_exact_quantiles(self):
        with pytest.raises(ResolutionError, match=r"as quantile\(d\['mu'\]\)"):
            quantile.with_options(exact_only=True)(_dependent_joint(), 0.5)

    def test_naming_the_route_under_exact_only_requires_every_component_exact(self):
        view = mean.with_options(method="by_component", exact_only=True)
        assert view.check(_Coupled()).route == "by_component"
        assert view.check(_dependent_joint()).feasible is False

    def test_an_unresolved_component_leaves_the_route_unresolved(self):
        report = mean.check(_Shift() * GuardedMean("mu", None))
        assert report.feasible is None
        assert report.route is None

    def test_components_with_no_route_together_decline_the_route(self):
        routes = {info.method_name: info for info in mean.check(Gaussian("mu") * Bare("b")).routes}
        assert routes["by_component (approximate)"].description == (
            "the components ('b',) have no mean route"
        )

    def test_a_component_the_operation_does_not_apply_to_declines_the_route(self):
        """A measure-valued component has no event-typed variance, so the fallback reports."""
        report = variance.check(Measure("m") * Gaussian("g", 1.0))
        routes = {info.method_name: info for info in report.routes}
        assert report.feasible is False
        assert routes["by_component (approximate)"].description == (
            "the components ('m',) have no variance route"
        )

    def test_an_edge_free_joint_keeps_its_closed_form(self):
        report = mean.check(Gaussian("a", 1.0) * Gaussian("b", 2.0))
        assert (report.route, report.exact) == ("closed_form", True)

    def test_a_joint_with_no_exact_component_keeps_the_fallback(self):
        report = mean.check(_Shift() * Sampler("mu", 2.0))
        routes = {info.method_name: info for info in report.routes}
        assert report.route == "monte_carlo"
        assert routes["by_component (approximate)"].description == "no component has an exact mean"

    def test_a_law_of_one_component_keeps_the_fallback(self):
        routes = {info.method_name: info for info in mean.check(Sampler("s")).routes}
        assert routes["by_component (exact)"].description == (
            "the event is not a record of several components"
        )

    def test_the_covariance_has_no_route_by_component(self):
        report = cov.check(_dependent_joint())
        assert report.route == "monte_carlo"
        assert all(not info.method_name.startswith("by_component") for info in report.routes)

    def test_a_seeded_workflow_reproduces_the_result(self):
        results = []
        for _ in range(2):
            with workflow_run(seed=12):
                results.append(_value(mean(_dependent_joint())["mean(y)"]))
        assert results[0] == results[1]

    def test_replay_reproduces_the_result(self):
        with workflow_run(seed=12):
            original = mean(_dependent_joint())
        with replay_run(original.provenance):
            replayed = mean(_dependent_joint())
        assert _value(replayed["mean(y)"]) == _value(original["mean(y)"])

    def test_the_raw_result_is_the_named_mapping(self):
        with workflow_run(seed=7):
            result = mean.with_options(raw=True)(_dependent_joint())
        assert set(result) == {"mean(y)", "mean(mu)"}
        assert float(result["mean(mu)"]) == 2.0


class TestExpectation:
    def test_a_finite_support_law_takes_the_closed_form(self):
        law = Coin("c", 0.25)
        report = expectation.check(law, lambda x: 2.0 * x)
        assert (report.route, report.exact) == ("closed_form", True)
        assert _value(expectation(law, lambda x: 2.0 * x)) == pytest.approx(0.5)

    def test_the_sampling_lift_is_the_default_approximate_rule(self):
        law = Gaussian("g")
        report = expectation.check(law, lambda x: x**2)
        assert (report.route, report.method, report.exact) == (
            "evaluation_rules",
            "sampling_lift",
            False,
        )
        with workflow_run(seed=6):
            estimate = expectation.with_options(n_broadcast_samples=_DRAWS)(law, lambda x: x**2)
        assert abs(_value(estimate) - 1.0) < 0.1

    def test_method_selects_the_sampling_lift_for_a_law_with_a_closed_form(self):
        view = expectation.with_options(method="sampling_lift")
        assert view.check(Coin("c"), lambda x: x).method == "sampling_lift"

    def test_method_selects_an_opt_in_rule_by_name(self, quadrature):
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
        assert isinstance(result, Record) and result.fields == ("mean(sq)",)
        assert _value(result["mean(sq)"]) == pytest.approx(0.5)

    def test_an_undeclared_integrand_wraps_its_value_by_kind(self):
        assert isinstance(expectation(Coin("c"), lambda x: x), NumericArray)

    def test_an_undeclared_integrand_leaves_the_declaration_to_the_value(self):
        assert expectation.check(Coin("c", 0.5), lambda x: x**2).result is None

    def test_the_integrand_is_a_callable(self):
        with pytest.raises(ApplicabilityError, match="FunctionSpec"):
            expectation(Gaussian("g"), 3.0)

    def test_the_routes_are_the_closed_form_then_the_evaluation_rules(self):
        routes = [(route.name, route.source) for route in expectation.summary().routes]
        assert routes == [
            ("closed_form", RouteSource.CAPABILITY),
            ("evaluation_rules", RouteSource.REGISTRY),
            ("identity", RouteSource.FALLBACK),
        ]
        (route,) = [r for r in expectation.routes if r.name == "evaluation_rules"]
        assert route.registry is _rules.evaluation_rule_registry

    def test_the_operation_takes_no_key(self):
        assert list(inspect.signature(expectation).parameters) == ["d", "f", "fixed_args"]
