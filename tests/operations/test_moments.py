"""Contract tests of the distribution functionals: the moments, quantile, and expectation."""

from __future__ import annotations

import functools
import inspect
import math
import operator
import re
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
    Bernoulli,
    InputSpec,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecordBatch,
    Record,
    RecordSpec,
    condition_on,
    conditional_distribution,
    config,
    factor,
    log_prob,
    replay_run,
    sample,
    workflow_run,
)
from probpipe.core import _fingerprint
from probpipe.core._dispatch import (
    BinaryDispatchMethod,
    Feasibility,
    MathematicalDomainError,
    ResolutionError,
)
from probpipe.core._specs import OutputSpec, _has_record_outputs
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
from probpipe.families import GaussianProcess
from probpipe.functions import _rules
from probpipe.linalg import LinOp
from probpipe.operations import RouteSource
from probpipe.operations._moments import (
    _ByOutput,
    _root,
    cov,
    expectation,
    mean,
    quantile,
    standard_deviation,
    variance,
)
from probpipe.operations._pushforward import pushforward
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
        super().__init__({"mu": REAL}, OutputSpec(**{label: REAL}), label=label)
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

    Each output's marginal is a normal law with every closed form, which
    ``_marginal_capabilities`` reports, so each output's view has exact
    moments.
    """

    def __init__(self) -> None:
        super().__init__(RecordSpec(a=REAL, b=REAL), label="coupled")

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
        super().__init__(
            pair if record else array,
            label=label,
        )
        self.record = record

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        index = jnp.arange(math.prod(sample_shape), dtype=jnp.float32).reshape(sample_shape)
        if self.record:
            return {"x": index, "y": 2.0 * index}
        return jnp.stack([index, 2.0 * index], axis=-1)


def _record_empirical() -> EmpiricalDistribution:
    """Three equally weighted record atoms, whose leaves ``b`` and ``a`` each rank the atoms alike."""
    atoms = NumericRecordBatch(
        {"b": jnp.array([[1.0, 2.0], [0.0, 1.0], [2.0, 3.0]]), "a": jnp.array([2.0, 1.0, 3.0])},
        "row",
        element=RecordSpec(b=(2,), a=()),
        label="rows",
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
        return EmpiricalDistribution(jnp.array([7.0]), output_name="stand_in")


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
    for operation in (expectation, pushforward):
        (route,) = [
            r for r in operation.routes if r.name in ("evaluation_rules", "pushforward_mean")
        ]
        monkeypatch.setattr(route, "registry", registry)
    return registry


def _value(term: Any) -> float:
    return float(jnp.asarray(term))


class TestDerivedLabelsAreNeverOutputNames:
    """A derived label may hold ``~``, a space, ``;``, or ``/``, and no route names a part by it."""

    @staticmethod
    def _held() -> Distribution:
        """A law holding the nested path ``y/obs`` fixed, as conditioning on it records."""
        law = Gaussian("g", 2.0)
        return law._with_expression(law._expression.with_fixed(("y/obs",)))

    @pytest.mark.parametrize(
        ("moment", "label", "output_name"),
        [
            (mean, "𝔼[g ~ g; y/obs]", "g"),
            (variance, "𝕍[g ~ g; y/obs]", "g"),
        ],
    )
    def test_a_monte_carlo_moment_is_labeled_over_the_nested_path(self, moment, label, output_name):
        held = self._held()
        with workflow_run(seed=0):
            result = moment.using(method="monte_carlo")(held)
        assert result.label == label
        assert tuple(moment.check(held).result.outputs) == (output_name,)

    def test_a_monte_carlo_quantile_takes_the_laws_output(self):
        held = self._held()
        with workflow_run(seed=0):
            result = quantile.using(method="monte_carlo")(held, jnp.array([0.1, 0.9]))
        assert result.label == "ℚ[g ~ g; y/obs]"
        assert result.level_names == ("quantile",)
        assert tuple(quantile.check(held, 0.5).result.outputs) == ("g",)


class TestMean:
    def test_the_closed_form_mean_has_the_event_declaration(self):
        result = mean(Gaussian("g", 2.0))
        assert isinstance(result, NumericArray)
        assert result.label == "𝔼[g ~ g]"
        assert result.spec == NumericArraySpec((), jnp.float32, real)
        assert _value(result) == 2.0

    def test_check_lists_the_closed_form_route_as_exact(self):
        report = mean.check(Gaussian("g", 2.0))
        routes = {info.method_name: info for info in report.routes}
        assert routes["closed_form"].exact is True
        assert (report.route, report.exact) == ("closed_form", True)

    def test_the_result_takes_the_events_outputs_and_form(self):
        assert mean.check(Gaussian("g")).result == OutputSpec(
            **{"g": NumericArraySpec((), jnp.float32, real)}
        )
        result = mean.check(Pair("p")).result
        assert _has_record_outputs(result)
        assert tuple(result.outputs) == ("a", "b")

    def test_a_record_law_has_a_record_mean_with_a_field_per_output(self):
        result = mean(Pair("p"))
        assert isinstance(result, Record) and result.fields == ("a", "b")
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
            estimate = mean.using(num_samples=_DRAWS)(law)
        assert abs(_value(estimate) - 2.0) < 0.1

    def test_the_fallback_takes_a_law_whose_label_is_no_identifier(self):
        """The draws lie on a level named after the law's output name, here ``my law``."""
        with workflow_run(seed=1):
            estimate = mean.using(num_samples=_DRAWS)(Sampler("my law", 2.0))
        # Observed across seeds 0-5: errors of 0.001 to 0.020; the standard error is 0.016.
        np.testing.assert_allclose(_value(estimate), 2.0, atol=0.05)

    def test_the_fallback_averages_record_draws_per_field(self):
        with workflow_run(seed=2):
            estimate = mean.using(num_samples=_DRAWS)(ExactPosterior("post"))
        assert isinstance(estimate, Record)
        assert abs(_value(estimate["theta"])) < 0.1
        assert abs(_value(estimate["y"])) < 0.1

    def test_method_selects_the_fallback_over_the_closed_form(self):
        law = Gaussian("g", -1.0)
        assert mean.using(method="monte_carlo").check(law).route == "monte_carlo"

    def test_a_rejecting_guard_passes_to_the_fallback(self):
        assert mean.check(GuardedMean("g", False)).route == "monte_carlo"

    def test_exact_only_refuses_a_law_without_a_closed_form(self):
        with pytest.raises(ResolutionError, match="exact_only"):
            mean.using(exact_only=True)(Sampler("s"))

    def test_a_law_that_neither_has_a_mean_nor_samples_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="does not sample"):
            mean(Bare("b"))

    def test_the_fallback_mean_of_a_measure_is_the_mixture_of_its_draws(self):
        assert mean(Measure("m")).event_spec == Measure("m").event_spec.term_spec.event_spec

    def test_the_mean_of_a_measure_is_a_law_that_exposes_its_event(self):
        """The mean measure is a law, so it takes its event's outputs."""
        result = mean.check(Measure("m")).result
        assert result == OutputSpec(Measure("m").event_spec.term_spec)
        assert not _has_record_outputs(result)


class TestVariance:
    def test_the_closed_form_variance_is_non_negative(self):
        result = variance(Gaussian("g", 0.0, 2.0))
        assert _value(result) == 4.0
        assert result.spec.support is non_negative

    def test_the_fallback_is_the_sample_variance(self):
        with workflow_run(seed=3):
            estimate = variance.using(num_samples=_DRAWS)(Sampler("s", 0.0, 2.0))
        assert abs(_value(estimate) - 4.0) < 0.4

    def test_a_measure_valued_event_has_no_variance(self):
        with pytest.raises(ApplicabilityError, match="event-typed variance"):
            variance(Measure("m"))


def _scaled_rbf(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """The squared-exponential kernel with variance 4, so each point's standard deviation is 2."""
    return 4.0 * jnp.exp(-0.5 * jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1))


class TestStandardDeviation:
    """``standard_deviation`` is ``sqrt(variance(d))``, taken through the routes of ``variance``."""

    def test_the_closed_form_is_the_square_root_of_the_variance(self):
        result = standard_deviation(Gaussian("g", 0.0, 2.0))
        assert isinstance(result, NumericArray)
        assert result.label == "√𝕍[g ~ g]"
        assert result.spec == NumericArraySpec((), jnp.float32, non_negative)
        assert _value(result) == 2.0
        metadata = result.provenance.metadata
        assert (metadata["route"], metadata["exact"]) == ("derived", True)

    def test_check_takes_the_exactness_of_the_route_variance_selects(self):
        report = standard_deviation.check(Gaussian("g"))
        assert (report.route, report.exact) == ("derived", True)
        assert report.result == OutputSpec(**{"g": NumericArraySpec((), jnp.float32, non_negative)})
        report = standard_deviation.check(Sampler("s"))
        assert (report.route, report.exact) == ("derived", False)

    def test_a_record_law_has_a_record_with_a_field_per_output(self):
        result = standard_deviation(Gaussian("mu", 0.0, 2.0) * Gaussian("tau", 1.0, 3.0))
        assert isinstance(result, Record)
        assert result.label == "√𝕍[(mu, tau) ~ mu·tau]"
        assert result.fields == ("mu", "tau")
        assert _value(result["mu"]) == 2.0
        assert _value(result["tau"]) == 3.0
        assert all(result.spec.children[field].support is non_negative for field in result.fields)

    def test_exact_only_keeps_a_closed_form_variance(self):
        result = standard_deviation.using(exact_only=True)(Gaussian("g", 0.0, 3.0))
        assert _value(result) == 3.0

    def test_exact_only_refuses_a_law_whose_variance_is_only_estimated(self):
        with pytest.raises(ResolutionError, match="variance: no route applies with exact_only"):
            standard_deviation.using(exact_only=True)(Sampler("s"))

    def test_the_fallback_takes_the_calls_num_samples(self):
        law = Sampler("s", 0.0, 2.0)
        with workflow_run(seed=3):
            estimate = standard_deviation.using(num_samples=_DRAWS)(law)
        assert law.shapes == [(_DRAWS,)]
        assert estimate.provenance.metadata["exact"] is False
        # Observed across seeds 0-5: errors of 0.005 to 0.027; the standard error is 0.022.
        np.testing.assert_allclose(_value(estimate), 2.0, atol=0.07, rtol=0)

    def test_the_fallback_takes_the_square_root_of_each_coordinate(self):
        with workflow_run(seed=3):
            estimate = standard_deviation.using(num_samples=_DRAWS)(Vector("v"))
        # Observed across seeds 0-5: errors up to 0.022 for the first coordinate
        # and 0.023 for the second.
        np.testing.assert_allclose(np.asarray(estimate), [1.0, 2.0], atol=0.06, rtol=0)

    def test_a_dependent_joints_exact_output_keeps_its_closed_form(self):
        """``mu ~ Normal(2, 1)`` and ``y = mu + 1``: each has standard deviation 1."""
        with workflow_run(seed=3):
            result = standard_deviation.using(num_samples=_DRAWS)(_dependent_joint())
        assert result.fields == ("y", "mu")
        assert _value(result["mu"]) == 1.0
        assert result.provenance.metadata["exact"] is False
        # Observed across seeds 0-5: errors of 0.003 to 0.012 for ``y``.
        np.testing.assert_allclose(_value(result["y"]), 1.0, atol=0.04, rtol=0)

    def test_exact_only_takes_by_output_when_every_output_is_exact(self):
        result = standard_deviation.using(exact_only=True)(_Coupled())
        assert _value(result["a"]) == 1.0
        assert _value(result["b"]) == 1.0
        assert result.provenance.metadata["exact"] is True

    def test_a_random_function_has_a_pointwise_standard_deviation_function(self):
        law = GaussianProcess("f", lambda X: jnp.zeros(X.shape[0]), _scaled_rbf)
        result = standard_deviation(law)
        assert isinstance(result, Function)
        assert result.label == "√𝕍[f ~ GaussianProcess]"
        assert inspect.signature(result.raw).parameters.keys() == {"X"}
        points = jnp.linspace(0.0, 1.0, 3)[:, None]
        np.testing.assert_allclose(np.asarray(result(points)), [2.0, 2.0, 2.0], rtol=1e-6)

    @staticmethod
    def _grouped() -> Any:
        """``g`` groups ``mu ~ Normal(0, 2)`` and ``tau ~ Normal(1, 3)``."""
        joint = Gaussian("mu", 0.0, 2.0) * Gaussian("tau", 1.0, 3.0)
        return joint.with_path_names({"mu": "g/mu", "tau": "g/tau"})

    def test_a_record_valued_output_keeps_its_fields(self):
        result = standard_deviation(self._grouped() * Gaussian("c", 5.0, 4.0))
        assert result.label == "√𝕍[(g, c) ~ mu·tau·c]"
        group = result.at_path("g")
        assert group.fields == ("mu", "tau")
        assert (_value(group["mu"]), _value(group["tau"])) == (2.0, 3.0)
        assert _value(result["c"]) == 4.0

    def test_a_whole_term_record_event_keeps_its_fields(self):
        """A law over one record-valued term takes the term's fields, unrenamed."""
        law = factor(self._grouped(), "g")
        assert not _has_record_outputs(law.event_spec)
        result = standard_deviation(law)
        assert isinstance(result, Record) and result.fields == ("mu", "tau")
        assert (_value(result["mu"]), _value(result["tau"])) == (2.0, 3.0)
        assert tuple(standard_deviation.check(law).result.outputs) == ("g",)

    def test_a_function_whose_signature_cannot_be_read_keeps_a_generic_one(self):
        with pytest.raises(ValueError):
            inspect.signature(max)
        np.testing.assert_allclose(_root(max)(4.0, 9.0), 3.0)

    def test_a_measure_valued_event_has_no_standard_deviation(self):
        with pytest.raises(ApplicabilityError, match="event-typed variance"):
            standard_deviation(Measure("m"))

    def test_a_route_of_variance_is_chosen_on_variance(self):
        message = (
            "standard_deviation does not pass method on to variance, so call "
            "variance.using(method='monte_carlo') to choose it"
        )
        with pytest.raises(ResolutionError, match=re.escape(message)):
            standard_deviation.using(method="monte_carlo")(Gaussian("g"))


class TestCov:
    def test_the_closed_form_covariance_is_the_dense_matrix_of_the_flattened_draw(self):
        result = cov(Gaussian("g", 0.0, 2.0))
        assert isinstance(result, NumericArray)
        assert result.spec.shape == (1, 1)
        np.testing.assert_array_equal(np.asarray(result), [[4.0]])

    def test_the_fallback_is_the_sample_covariance(self):
        with workflow_run(seed=4):
            estimate = cov.using(num_samples=_DRAWS)(Vector("v"))
        assert estimate.spec.shape == (2, 2)
        np.testing.assert_allclose(np.asarray(estimate), np.diag([1.0, 4.0]), atol=0.3)

    def test_the_covariance_is_declared_under_the_operations_name(self):
        assert list(cov.check(Gaussian("g")).result.outputs) == ["cov"]
        assert list(cov.check(Pair("p")).result.outputs) == ["cov"]

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
        assert _value(result.raw[1]) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("q", [1.5, -0.1, jnp.array([0.5, 2.0])])
    def test_a_level_outside_the_unit_interval_raises_a_domain_error(self, q):
        with pytest.raises(MathematicalDomainError, match=r"\[0, 1\]"):
            quantile(Gaussian("g"), q)

    def test_the_fallback_is_the_empirical_quantile(self):
        with workflow_run(seed=5):
            estimate = quantile.using(num_samples=_DRAWS)(Sampler("s"), 0.5)
        assert abs(_value(estimate)) < 0.1

    @pytest.mark.parametrize("record", [False, True], ids=["array", "record"])
    def test_the_fallback_is_the_inverse_cdf_of_the_draws(self, record):
        # Four draws of x are 0, 1, 2, 3, whose CDF reaches 0.25 at 0.
        view = quantile.using(num_samples=4)
        estimate = view(_Ramp("ramp", record=record), jnp.array([0.25, 0.5, 1.0]))
        x = estimate["x"] if record else estimate.raw[:, 0]
        np.testing.assert_array_equal(np.asarray(x), [0.0, 1.0, 3.0])

    def test_method_selects_the_fallback_over_the_closed_form(self):
        view = quantile.using(method="monte_carlo")
        assert view.check(Gaussian("g"), 0.5).route == "monte_carlo"

    def test_one_level_of_a_record_law_is_a_record_of_its_quantiles(self):
        result = quantile(_record_empirical(), 0.5)
        assert isinstance(result, Record)
        assert result.fields == ("b", "a")
        np.testing.assert_allclose(np.asarray(result["b"]), [1.0, 2.0])
        assert _value(result["a"]) == 2.0

    def test_several_levels_of_a_record_law_are_a_batch_of_records(self):
        result = quantile(_record_empirical(), jnp.array([0.0, 0.5, 1.0]))
        assert isinstance(result, NumericRecordBatch)
        assert (result.level_names, result.batch_shape) == (("quantile",), (3,))
        np.testing.assert_allclose(np.asarray(result["a"]), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(np.asarray(result["b"]), [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

    def test_a_raw_record_law_s_levels_are_the_mapping_of_their_columns(self):
        result = quantile.using(raw=True)(_record_empirical(), jnp.array([0.0, 1.0]))
        assert isinstance(result, dict) and list(result) == ["b", "a"]
        assert jnp.shape(result["b"]) == (2, 2)

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
            OutputSpec(**{label: FunctionSpec(InputSpec(x=REAL), OutputSpec(y=REAL))}),
            label=label,
        )

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        slopes = np.asarray(jax.random.normal(key, tuple(sample_shape)))
        if not sample_shape:
            return Function(
                lambda x: slopes * x,
                label="line",
            )
        lines = np.empty(slopes.shape, dtype=object)
        for index, slope in np.ndenumerate(slopes):
            lines[index] = Function(
                lambda x, slope=slope: slope * x,
                label="line",
            )
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
    """A fallback reports the moments of the empirical law of its draws, whatever the form."""

    @pytest.mark.parametrize("record", [False, True], ids=["array", "record"])
    def test_the_covariance_divides_by_the_number_of_draws(self, record):
        # The three draws (0, 0), (1, 2), and (2, 4) deviate from their mean (1, 2)
        # by (-1, -2), (0, 0), and (1, 2).
        estimate = cov.using(num_samples=3)(_Ramp("ramp", record=record))
        expected = np.array([[2.0, 4.0], [4.0, 8.0]]) / 3.0
        np.testing.assert_allclose(np.asarray(estimate), expected, rtol=1e-6)

    @pytest.mark.parametrize("record", [False, True], ids=["array", "record"])
    def test_the_variance_is_the_diagonal_of_the_covariance(self, record):
        law = _Ramp("ramp", record=record)
        spread = variance.using(num_samples=7)(law)
        covariance = cov.using(num_samples=7)(law)
        if record:
            values = [_value(spread["x"]), _value(spread["y"])]
        else:
            values = np.asarray(spread)
        np.testing.assert_allclose(values, np.diag(np.asarray(covariance)), rtol=1e-6)

    @pytest.mark.parametrize("record", [False, True], ids=["array", "record"])
    def test_one_draw_has_no_covariance(self, record):
        estimate = cov.using(num_samples=1)(_Ramp("ramp", record=record))
        np.testing.assert_array_equal(np.asarray(estimate), np.zeros((2, 2)))


class TestTheFallbacksOfAJoint:
    """A joint's draws are a mapping of columns, which each fallback reads per output."""

    def test_the_mean_is_the_average_of_each_output(self):
        with workflow_run(seed=7):
            estimate = mean.using(method="monte_carlo", num_samples=_DRAWS)(_dependent_joint())
        assert isinstance(estimate, Record) and estimate.fields == ("y", "mu")
        assert abs(_value(estimate["mu"]) - 2.0) < 0.1
        assert abs(_value(estimate["y"]) - 3.0) < 0.1

    def test_the_variance_is_the_sample_variance_of_each_output(self):
        with workflow_run(seed=8):
            estimate = variance.using(method="monte_carlo", num_samples=_DRAWS)(_dependent_joint())
        assert isinstance(estimate, Record)
        assert abs(_value(estimate["mu"]) - 1.0) < 0.15
        assert abs(_value(estimate["y"]) - 1.0) < 0.15

    def test_the_covariance_couples_the_outputs(self):
        with workflow_run(seed=9):
            estimate = cov.using(num_samples=_DRAWS)(_dependent_joint())
        np.testing.assert_allclose(np.asarray(estimate), np.ones((2, 2)), atol=0.15)

    def test_several_quantile_levels_give_a_batch_of_records(self):
        levels = jnp.array([0.25, 0.5])
        with workflow_run(seed=10):
            estimate = quantile.using(method="monte_carlo", num_samples=_DRAWS)(
                _dependent_joint(), levels
            )
        assert (estimate.level_names, estimate.batch_shape) == (("quantile",), (2,))
        assert abs(float(estimate["mu"][1]) - 2.0) < 0.1
        assert abs(float(estimate["y"][1]) - 3.0) < 0.1


class TestByOutput:
    """A record law's mean, variance, and quantile compute each output by its own route."""

    @staticmethod
    def _blocks(result: Any) -> list[tuple[tuple[str, ...], str, bool]]:
        """Each block's outputs, route, and exactness, as the result's provenance records them."""
        (assembled,) = [
            parent
            for parent in result.provenance.parents
            if parent.provenance is not None
            and parent.provenance.metadata.get("route") == "by_output"
        ]
        return [
            (tuple(block["outputs"]), block["route"], block["exact"])
            for block in assembled.provenance.metadata["blocks"]
        ]

    def test_a_dependent_joint_with_an_exact_output_takes_the_route(self):
        report = mean.check(_dependent_joint())
        assert (report.route, report.exact) == ("by_output", False)

    def test_the_exact_candidate_names_the_outputs_without_an_exact_route(self):
        routes = {info.method_name: info for info in mean.check(_dependent_joint()).routes}
        declined = routes["by_output (exact)"]
        assert declined.feasible is False
        assert declined.description == "the outputs ('y',) have no exact mean, but ('mu',) do"
        assert not declined.actionable
        approximate = routes["by_output (approximate)"]
        assert (approximate.feasible, approximate.exact) == (True, False)

    @pytest.mark.parametrize(
        ("moment", "closed_form", "seed"), [(mean, 2.0, 7), (variance, 1.0, 8)]
    )
    def test_the_exact_output_is_its_closed_form_and_the_rest_is_estimated(
        self, moment, closed_form, seed
    ):
        """``mu ~ Normal(2, 1)`` and ``y = mu + 1``: ``y`` has mean 3 and variance 1."""
        with workflow_run(seed=seed):
            result = moment.using(num_samples=_DRAWS)(_dependent_joint())
        assert isinstance(result, Record)
        assert result.fields == ("y", "mu")
        assert _value(result["mu"]) == closed_form
        # Observed across seeds 7, 8, 10, and 11: errors of the estimated
        # output up to 0.027 for the mean and 0.026 for the variance.
        np.testing.assert_allclose(
            _value(result["y"]), 3.0 if moment is mean else 1.0, atol=0.08, rtol=0
        )

    def test_several_quantile_levels_give_a_batch_of_records(self):
        levels = jnp.array([0.25, 0.5])
        with workflow_run(seed=10):
            result = quantile.using(num_samples=_DRAWS)(_dependent_joint(), levels)
        assert (result.level_names, result.batch_shape) == (("quantile",), (2,))
        reference = jax.scipy.special.ndtri(levels)
        np.testing.assert_array_equal(
            np.asarray(result["mu"]), np.asarray((2.0 + reference).astype(jnp.float32))
        )
        # Observed across seeds 7, 8, 10, and 11: errors up to 0.033.
        np.testing.assert_allclose(np.asarray(result["y"]), 3.0 + reference, atol=0.1)

    def test_one_quantile_level_gives_a_record(self):
        with workflow_run(seed=11):
            result = quantile.using(num_samples=_DRAWS)(_dependent_joint(), 0.5)
        assert isinstance(result, Record)
        assert _value(result["mu"]) == 2.0

    def test_provenance_records_the_route_and_exactness_of_each_block(self):
        with workflow_run(seed=7):
            result = mean(_dependent_joint())
        assert result.provenance.metadata["route"] == "by_output"
        assert result.provenance.metadata["exact"] is False
        assert self._blocks(result) == [
            (("y",), "monte_carlo", False),
            (("mu",), "closed_form", True),
        ]

    def test_the_outputs_without_an_exact_route_are_one_block(self):
        """``z = mu - 1`` and ``y = mu + 1`` are computed from one set of draws."""
        joint = _Shift("z", -1.0) * _Shift("y", 1.0) * Gaussian("mu", 2.0)
        with workflow_run(seed=7):
            result = mean(joint)
        assert self._blocks(result) == [
            (("z", "y"), "monte_carlo", False),
            (("mu",), "closed_form", True),
        ]
        assert _value(result["mu"]) == 2.0
        # Shared draws differ by 2 at every draw, so their means do too.
        assert _value(result["y"]) - _value(result["z"]) == pytest.approx(2.0)

    @staticmethod
    def _nested() -> Any:
        """``g`` groups ``mu ~ Normal(2, 1)`` and ``y = mu + 1``, beside ``c ~ Normal(5, 1)``."""
        joint = _Shift("y", 1.0) * Gaussian("mu", 2.0) * Gaussian("c", 5.0)
        return joint.with_path_names({"mu": "g/mu", "y": "g/y"})

    def test_a_record_valued_output_is_one_block(self):
        """A group of fields is summarized whole, by the route its view selects."""
        with workflow_run(seed=7):
            result = mean(self._nested())
        assert self._blocks(result) == [
            (("g",), "monte_carlo", False),
            (("c",), "closed_form", True),
        ]
        assert result.fields == ("g", "c")
        group = result.at_path("g")
        assert isinstance(group, Record) and group.fields == ("y", "mu")
        assert (np.shape(group["y"]), np.shape(group["mu"])) == ((), ())
        # One set of draws gives both fields, which differ by 1 at every draw.
        np.testing.assert_allclose(_value(group["y"]) - _value(group["mu"]), 1.0, atol=1e-5)
        assert _value(result["c"]) == 5.0

    def test_a_record_valued_output_keeps_its_fields_in_the_variance(self):
        with workflow_run(seed=8):
            result = variance(self._nested())
        group = result.at_path("g")
        assert group.fields == ("y", "mu")
        # Shared draws give ``y`` and ``mu`` one variance.
        np.testing.assert_allclose(_value(group["y"]), _value(group["mu"]), rtol=1e-5)
        assert _value(result["c"]) == 1.0

    def test_a_record_valued_output_keeps_its_fields_in_several_quantiles(self):
        levels = jnp.array([0.25, 0.75])
        with workflow_run(seed=10):
            result = quantile(self._nested(), levels)
        assert (result.level_names, result.batch_shape) == (("quantile",), (2,))
        group = result["g"]
        assert np.shape(group["y"]) == np.shape(group["mu"]) == (2,)
        np.testing.assert_allclose(
            np.asarray(group["y"]) - np.asarray(group["mu"]), [1.0, 1.0], atol=1e-5
        )
        np.testing.assert_array_equal(
            np.asarray(result["c"]),
            np.asarray((5.0 + jax.scipy.special.ndtri(levels)).astype(jnp.float32)),
        )

    def test_a_law_whose_outputs_are_all_exact_takes_the_exact_candidate(self):
        law = _Coupled()
        report = mean.check(law)
        assert (report.route, report.exact) == ("by_output", True)
        result = mean.using(exact_only=True)(law)
        assert (_value(result["a"]), _value(result["b"])) == (1.0, -1.0)
        assert self._blocks(result) == [
            (("a",), "closed_form", True),
            (("b",), "closed_form", True),
        ]

    @pytest.mark.parametrize("moment", [mean, variance])
    def test_exact_only_says_how_to_compute_the_exact_outputs(self, moment):
        name = moment.name
        message = (
            rf"^{name}: the outputs \('y',\) have no exact {name}, but \('mu',\) do; call "
            rf"{name} on the view of each output that does, as {name}\(d\['mu'\]\)"
        )
        with pytest.raises(ResolutionError, match=message):
            moment.using(exact_only=True)(_dependent_joint())

    def test_exact_only_says_how_to_compute_the_exact_quantiles(self):
        joint = _dependent_joint()
        with pytest.raises(ResolutionError, match=r"as quantile\(d\['mu'\], q\)$"):
            quantile.using(exact_only=True)(joint, 0.5)
        # The suggested call, with the original level, computes the quantile exactly.
        exact = quantile.using(exact_only=True)(joint["mu"], 0.5)
        assert _value(exact) == 2.0

    def test_naming_the_route_under_exact_only_requires_every_output_exact(self):
        view = mean.using(method="by_output", exact_only=True)
        assert view.check(_Coupled()).route == "by_output"
        assert view.check(_dependent_joint()).feasible is False

    def test_an_unresolved_output_leaves_the_route_unresolved(self):
        report = mean.check(_Shift() * GuardedMean("mu", None))
        assert report.feasible is None
        assert report.route is None

    def test_outputs_with_no_route_together_decline_the_route(self):
        routes = {info.method_name: info for info in mean.check(Gaussian("mu") * Bare("b")).routes}
        assert routes["by_output (approximate)"].description == (
            "the outputs ('b',) have no mean route"
        )

    def test_an_output_the_operation_does_not_apply_to_declines_the_route(self):
        """A measure-valued output has no event-typed variance, so the fallback reports."""
        report = variance.check(Measure("m") * Gaussian("g", 1.0))
        routes = {info.method_name: info for info in report.routes}
        assert report.feasible is False
        assert routes["by_output (approximate)"].description == (
            "the outputs ('m',) have no variance route"
        )

    def test_an_edge_free_joint_keeps_its_closed_form(self):
        report = mean.check(Gaussian("a", 1.0) * Gaussian("b", 2.0))
        assert (report.route, report.exact) == ("closed_form", True)

    def test_a_joint_with_no_exact_output_keeps_the_fallback(self):
        report = mean.check(_Shift() * Sampler("mu", 2.0))
        routes = {info.method_name: info for info in report.routes}
        assert report.route == "monte_carlo"
        assert routes["by_output (approximate)"].description == "no output has an exact mean"

    def test_a_law_of_one_output_keeps_the_fallback(self):
        routes = {info.method_name: info for info in mean.check(Sampler("s")).routes}
        assert routes["by_output (exact)"].description == (
            "the event is not a record of several outputs"
        )

    def test_the_covariance_has_no_route_by_output(self):
        report = cov.check(_dependent_joint())
        assert report.route == "monte_carlo"
        assert all(not info.method_name.startswith("by_output") for info in report.routes)

    @pytest.mark.parametrize(
        ("outputs", "expected_blocks"),
        [
            (("s0", "s1", "a", "b"), [["s0", "s1"], ["a"], ["b"]]),
            (("a", "s0", "b", "s1"), [["a"], ["s0", "s1"], ["b"]]),
            (("a", "b", "s0", "s1"), [["a"], ["b"], ["s0", "s1"]]),
        ],
    )
    def test_blocks_follow_event_order_at_their_first_output(self, outputs, expected_blocks):
        factors = {
            "a": Gaussian("a", 1.0),
            "b": Gaussian("b", 2.0),
            "s0": Sampler("s0"),
            "s1": Sampler("s1"),
        }
        law = functools.reduce(operator.mul, [factors[c] for c in outputs])
        with workflow_run(seed=7):
            result = mean.using(method="by_output", num_samples=8)(law)

        assert result.fields == tuple(outputs)
        assert [list(outputs) for outputs, _, _ in self._blocks(result)] == expected_blocks
        np.testing.assert_allclose(_value(result["a"]), 1.0, atol=0, rtol=0)
        np.testing.assert_allclose(_value(result["b"]), 2.0, atol=0, rtol=0)

    def test_a_seeded_workflow_reproduces_the_result(self):
        results = []
        for _ in range(2):
            with workflow_run(seed=12):
                results.append(_value(mean(_dependent_joint())["y"]))
        assert results[0] == results[1]

    def test_replay_reproduces_the_result(self):
        with workflow_run(seed=12):
            original = mean(_dependent_joint())
        with replay_run(original.provenance):
            replayed = mean(_dependent_joint())
        assert _value(replayed["y"]) == _value(original["y"])

    def test_the_raw_result_is_the_mapping_keyed_by_the_outputs(self):
        with workflow_run(seed=7):
            result = mean.using(raw=True)(_dependent_joint())
        assert set(result) == {"y", "mu"}
        assert float(result["mu"]) == 2.0


def _samplers(n: int) -> Any:
    """The product of *n* sampling-only laws, which claims no summary and has no exact output."""
    return functools.reduce(operator.mul, [Sampler(f"s{i}") for i in range(n)])


class TestByOutputPlanning:
    """``by_output`` resolves each output once per call, and hashes the law once."""

    @pytest.fixture
    def resolutions(self, monkeypatch):
        """The restriction of each resolution the route makes, in order."""
        calls: list[bool] = []
        original = _ByOutput._resolve

        def spy(self: Any, call: Any, view: Any, exact_only: bool) -> Any:
            calls.append(exact_only)
            return original(self, call, view, exact_only)

        monkeypatch.setattr(_ByOutput, "_resolve", spy)
        return calls

    @pytest.mark.parametrize(
        ("law", "on_check", "on_run"),
        [
            # Each output once, restricted to exact routes; no exact one, so
            # neither candidate resolves the rest, and the fallback runs.
            pytest.param(lambda: _samplers(6), [True] * 6, [], id="sampling-only"),
            # Each output once, then the rest once per candidate. Resolving
            # the rest, a selection of two outputs, considers this route on
            # it too, which resolves those two outputs once and declines; the
            # run selects the rest's route again before running it.
            pytest.param(
                lambda: _Shift("z", -1.0) * _Shift("y", 1.0) * Gaussian("mu", 2.0),
                [True, True, True, True, True, True, False, True, True],
                [True, True],
                id="mixed",
            ),
            # Each output once, and no rest.
            pytest.param(_Coupled, [True, True], [], id="all-exact"),
        ],
    )
    def test_a_check_and_a_call_resolve_each_output_once(self, law, on_check, on_run, resolutions):
        """The run reuses the split its candidate's probe made within the call."""
        law = law()
        mean.check(law)
        assert resolutions == on_check
        resolutions.clear()
        with workflow_run(seed=7):
            mean(law)
        assert resolutions == [*on_check, *on_run]

    def test_the_laws_fingerprint_is_computed_once_for_all_its_views(self, monkeypatch):
        law = _samplers(8)
        hashed: list[Any] = []
        original = _fingerprint._update_distribution

        def spy(h: Any, dist: Any, *args: Any, **kwargs: Any) -> Any:
            if dist is law:
                hashed.append(dist)
            return original(h, dist, *args, **kwargs)

        monkeypatch.setattr(_fingerprint, "_update_distribution", spy)
        mean.check(law)
        mean.check(law)
        assert len(hashed) == 1


class TestExpectation:
    def test_a_route_of_mean_is_chosen_on_mean(self):
        message = (
            "expectation does not pass method on to mean, so call "
            "mean.using(method='monte_carlo') to choose it"
        )
        with pytest.raises(ResolutionError, match=re.escape(message)):
            expectation.using(method="monte_carlo")(Gaussian("g"), lambda x: x)

    def test_a_finite_support_law_takes_the_closed_form(self):
        law = Coin("c", 0.25)
        report = expectation.check(law, lambda x: 2.0 * x)
        assert (report.route, report.exact) == ("closed_form", True)
        assert _value(expectation(law, lambda x: 2.0 * x)) == pytest.approx(0.5)

    def test_the_sampling_lift_is_the_default_approximate_rule(self):
        law = Gaussian("g")
        report = expectation.check(law, lambda x: x**2)
        assert (report.route, report.method, report.exact) == (
            "pushforward_mean",
            "sampling_lift",
            False,
        )
        with workflow_run(seed=6):
            estimate = expectation.using(num_samples=_DRAWS)(law, lambda x: x**2)
        assert abs(_value(estimate) - 1.0) < 0.1

    def test_method_selects_the_sampling_lift_for_a_law_with_a_closed_form(self):
        view = expectation.using(method="sampling_lift")
        assert view.check(Coin("c"), lambda x: x).method == "sampling_lift"

    def test_method_selects_an_opt_in_rule_by_name(self, quadrature):
        view = expectation.using(method="operations_suite_quadrature")
        assert _value(view(Gaussian("g"), lambda x: x)) == 7.0

    def test_exact_only_refuses_a_law_without_an_exact_method(self):
        with pytest.raises(ResolutionError):
            expectation.using(exact_only=True)(Gaussian("g"), lambda x: x)

    def test_an_unregistered_method_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="no_such_method"):
            expectation.using(method="no_such_method")(Gaussian("g"), lambda x: x)

    def test_num_samples_is_the_number_of_evaluations_of_the_integrand(self):
        law = Sampler("s")
        estimate = expectation.using(num_samples=33, max_evaluations=2)(law, lambda x: x)
        assert law.shapes == [(33,)]
        assert estimate.provenance.metadata["num_samples"] == 33

    def test_num_samples_wins_over_the_integrands_own_max_evaluations(self):
        law = Sampler("s")
        integrand = Function(lambda x: x, label="integrand", max_evaluations=5)
        expectation.using(num_samples=33)(law, integrand)
        assert law.shapes == [(33,)]

    def test_the_session_num_samples_sets_the_evaluations_when_the_call_sets_none(self):
        config.sampling.num_samples = 29
        config.sampling.max_evaluations = 3
        law = Sampler("s")
        estimate = expectation(law, lambda x: x)
        assert law.shapes == [(29,)]
        assert estimate.provenance.metadata["num_samples"] == 29

    def test_num_samples_decides_between_enumerating_and_sampling_the_atoms(self):
        atoms = EmpiricalDistribution(jnp.arange(6.0), output_name="x")
        view = expectation.using(method="pushforward_mean", max_evaluations=100)
        enumerated = view.using(num_samples=6)(atoms, lambda x: x)
        assert enumerated.provenance.metadata["method"] == "empirical_enumeration"
        assert "num_samples" not in enumerated.provenance.metadata
        with workflow_run(seed=0):
            sampled = view.using(num_samples=5)(atoms, lambda x: x)
        assert sampled.provenance.metadata["method"] == "sampling_lift"
        assert sampled.provenance.metadata["num_samples"] == 5

    def test_the_result_takes_the_kind_the_integrand_declares(self):
        f = Function(
            lambda x: {"sq": x**2},
            output_spec=OutputSpec(RecordSpec(sq=NumericArraySpec(()))),
            label="f",
        )
        result = expectation(Coin("c", 0.5), f)
        assert isinstance(result, Record) and result.fields == ("sq",)
        assert _value(result["sq"]) == pytest.approx(0.5)

    def test_an_undeclared_integrand_wraps_its_value_by_kind(self):
        assert isinstance(expectation(Coin("c"), lambda x: x), NumericArray)

    def test_an_undeclared_integrand_leaves_the_declaration_to_the_value(self):
        assert expectation.check(Coin("c", 0.5), lambda x: x**2).result is None

    def test_an_undeclared_integrand_gives_one_result_whatever_its_names(self):
        def square(x):
            return x**2

        results = [
            expectation(Coin("c", 0.5), integrand)
            for integrand in (square, lambda x: x**2, Function(square, label="g"))
        ]
        assert [result.label for result in results] == [
            "𝔼[square(c ~ c)]",
            "𝔼[𝒻(c ~ c)]",
            "𝔼[g(c ~ c)]",
        ]
        assert len({result.spec for result in results}) == 1
        assert all(_value(result) == pytest.approx(0.5) for result in results)

    def test_a_nameless_integrand_needs_a_labeled_function(self):
        integrand = partial(lambda x, k: x**k, k=2)
        with pytest.raises(
            TypeError,
            match=r"^expectation needs an explicit label for a partial, which has no name; "
            r"wrap it as Function\(f, label=\.\.\.\)$",
        ):
            expectation(Coin("c", 0.5), integrand)
        labeled = expectation(Coin("c", 0.5), Function(integrand, label="g"))
        assert labeled.label == "𝔼[g(c ~ c)]"

    def test_the_integrand_is_a_callable(self):
        with pytest.raises(ApplicabilityError, match="FunctionSpec"):
            expectation(Gaussian("g"), 3.0)

    def test_the_routes_are_the_closed_form_then_the_pushforward_mean(self):
        summary = expectation.summary()
        assert (summary.is_derived, summary.formula) == (
            True,
            "mean(pushforward(d, f, fixed_args))",
        )
        routes = [(route.name, route.source) for route in summary.routes]
        assert routes == [
            ("closed_form", RouteSource.CAPABILITY),
            ("pushforward_mean", RouteSource.REGISTRY),
        ]
        (route,) = [r for r in expectation.routes if r.name == "pushforward_mean"]
        assert route.registry is _rules.evaluation_rule_registry

    def test_the_operation_takes_no_key(self):
        assert list(inspect.signature(expectation).parameters) == ["d", "f", "fixed_args", "given"]


# ---------------------------------------------------------------------------
# expectation integrates over draws on every route
# ---------------------------------------------------------------------------


def _the_law(law: Distribution) -> Any:
    return law


def _atoms() -> EmpiricalDistribution:
    return EmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), output_name="x")


class TestTheIntegrandReceivesDraws:
    """expectation(d, f) is mean(pushforward(d, f)): f receives draws of d, whatever its annotation."""

    @pytest.mark.parametrize("law", [Gaussian("g"), _atoms()], ids=["sampled", "closed_form"])
    @pytest.mark.parametrize("call", ["call", "check"])
    def test_an_integrand_taking_a_law_is_refused(self, law, call):
        refusal = (
            r"^expectation: cannot pass a draw of argument 'd' to '_the_law', since its "
            r"parameter 'law' takes a distribution, but the draw is an array$"
        )
        with pytest.raises(ApplicabilityError, match=refusal):
            getattr(expectation, "__call__" if call == "call" else "check")(law, _the_law)

    def test_the_refusal_runs_no_body_and_draws_nothing(self):
        calls: list[Any] = []

        def recorded(law: Distribution) -> Any:
            calls.append(law)
            return law

        class _Counting(Sampler):
            draws = 0

            def _sample(self, key, sample_shape=()):
                type(self).draws += 1
                return super()._sample(key, sample_shape)

        for options in ({}, {"exact_only": True}):
            with pytest.raises(ApplicabilityError):
                expectation.using(**options)(_Counting("s"), recorded)
        assert calls == [] and _Counting.draws == 0

    def test_a_kernel_given_values_is_refused_the_same_way(self):
        kernel = _Shift()
        with pytest.raises(ApplicabilityError, match="takes a distribution"):
            expectation(kernel, _the_law, given={"mu": 1.0})

    def test_an_integrand_annotated_any_receives_draws(self):
        received: list[Any] = []

        def square(x: Any) -> Any:
            received.append(x)
            return x * x

        assert _value(expectation(_atoms(), square)) == pytest.approx(14.0 / 3.0)
        with workflow_run(seed=3):
            estimate = expectation.using(num_samples=_DRAWS)(Gaussian("g", 1.0), square)
        assert received and not any(isinstance(x, Distribution) for x in received)
        # E[X^2] = 2 for X ~ N(1, 1); 4000 draws give a standard error of about 0.04.
        assert _value(estimate) == pytest.approx(2.0, abs=0.15)
        assert estimate.provenance.metadata["route"] == "pushforward_mean"

    def test_an_unannotated_and_a_numeric_integrand_are_admitted(self):
        def plain(x):
            return x * x

        def numeric(x: float) -> float:
            return x * x

        for integrand in (plain, numeric):
            assert _value(expectation(_atoms(), integrand)) == pytest.approx(14.0 / 3.0)

    @pytest.mark.parametrize("method", [None, "empirical_enumeration", "sampling_lift"])
    def test_a_positional_only_integrand_receives_draws_on_every_route(self, method):
        with workflow_run(seed=0):
            value = expectation.using(method=method)(_atoms(), jnp.exp)
        exact = float(jnp.mean(jnp.exp(jnp.array([1.0, 2.0, 3.0]))))
        # The sampling lift draws the three atoms by stratified resampling.
        assert _value(value) == pytest.approx(exact, rel=0.1)

    def test_a_maps_construction_bind_is_kept_on_every_route(self):
        def shift(x, a):
            return x + a

        bound = Function(shift, label="shift", bind={"a": 10.0})
        for method in (None, "empirical_enumeration"):
            value = expectation.using(method=method)(_atoms(), bound)
            assert _value(value) == pytest.approx(12.0)

    def test_an_integrand_over_a_random_measure_receives_each_draw_as_a_law(self):
        from probpipe.families import BootstrapDistribution

        measure = BootstrapDistribution("D", _atoms(), 3)

        def statistic(D: Distribution) -> Any:
            return mean(D)

        with workflow_run(seed=0):
            estimate = expectation.using(num_samples=200)(measure, statistic)
        # The replicate means of the atoms 1, 2, 3 have mean 2 and standard
        # deviation sqrt(2/9), so 200 of them give a standard error of about 0.033.
        assert _value(estimate) == pytest.approx(2.0, abs=0.2)

    def test_a_vector_event_is_integrated_coordinatewise(self):
        with workflow_run(seed=0):
            estimate = expectation.using(num_samples=_DRAWS)(Vector("v"), lambda v: v * v)
        # The coordinates have scales 1 and 2; 4000 draws give standard errors
        # of about 0.022 and 0.09 for their second moments.
        np.testing.assert_allclose(jnp.asarray(estimate), [1.0, 4.0], atol=0.3)

    def test_a_record_atom_reaches_the_integrand_alike_on_both_routes(self):
        law = _record_empirical()
        received: list[type] = []

        def total(r):
            received.append(type(r))
            return r["a"] + jnp.sum(r["b"])

        closed = expectation(law, total)
        enumerated = expectation.using(method="empirical_enumeration")(law, total)
        assert closed.provenance.metadata["route"] == "closed_form"
        assert enumerated.provenance.metadata["route"] == "pushforward_mean"
        # The atoms give 2 + 3, 1 + 1, and 3 + 5, whose mean is 5.
        assert _value(closed) == pytest.approx(_value(enumerated)) == pytest.approx(5.0)
        assert received and all(issubclass(kind, Record) for kind in received)


class TestFixedArgumentsOfExpectation:
    @staticmethod
    def _shift(x, a):
        return x + a

    @pytest.mark.parametrize("law", [Gaussian("g"), _atoms()], ids=["sampled", "closed_form"])
    def test_a_law_in_fixed_args_is_refused_on_every_route(self, law):
        with pytest.raises(
            ApplicabilityError,
            match=r"^expectation: fixed_args\['a'\] must be a fixed value, but got "
            r"Gaussian 'a', which is a distribution$",
        ):
            expectation(law, self._shift, {"a": Gaussian("a", 100.0)})

    @pytest.mark.parametrize("law", [Gaussian("g"), _atoms()], ids=["sampled", "closed_form"])
    def test_a_batch_in_fixed_args_is_refused_on_every_route(self, law):
        batch = NumericArrayBatch(jnp.array([0.0, 10.0]), "i", label="a")
        with pytest.raises(ApplicabilityError, match="got NumericArrayBatch 'a', which is a batch"):
            expectation(law, self._shift, {"a": batch})

    def test_a_constant_binds_on_every_route(self):
        assert _value(expectation(_atoms(), self._shift, {"a": 10.0})) == pytest.approx(12.0)
        enumerated = expectation.using(method="empirical_enumeration")(
            _atoms(), self._shift, {"a": 10.0}
        )
        assert _value(enumerated) == pytest.approx(12.0)

    def test_a_law_at_a_parameter_taking_a_law_is_a_constant(self):
        def offset(x, reference: Distribution):
            return x + mean(reference)

        value = expectation(_atoms(), offset, {"reference": Gaussian("r", 10.0)})
        assert _value(value) == pytest.approx(12.0)


class _ExactToSampler(BinaryDispatchMethod):
    """An exact rule whose pushforward is a law with no exact mean, so the mean is Monte Carlo."""

    name = "exact_to_sampler"
    exact = True
    priority = 90

    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        return ((Function,), (Distribution,))

    def check(self, f: Any, operand: Any, /, **call: Any) -> Feasibility:
        return Feasibility(isinstance(operand, Gaussian), "only a Gaussian")

    def __init__(self) -> None:
        self.pushed = Sampler("pushed", 3.0)

    def execute(self, f: Any, operand: Any, /, **call: Any) -> Any:
        return self.pushed


@pytest.fixture
def exact_to_sampler(monkeypatch):
    """The evaluation rules, for one test, with an exact rule whose result has no exact mean."""
    registry = type(_rules.evaluation_rule_registry)()
    exact_rule = _ExactToSampler()
    for rule in (
        _rules._SamplingLift(),
        _rules._ElementwiseSweep(),
        _rules._EmpiricalEnumeration(),
        exact_rule,
    ):
        registry.register(rule)
    monkeypatch.setattr(_rules, "evaluation_rule_registry", registry)
    for operation in (expectation, pushforward):
        (route,) = [
            r for r in operation.routes if r.name in ("evaluation_rules", "pushforward_mean")
        ]
        monkeypatch.setattr(route, "registry", registry)
    return exact_rule


class TestTheExactnessOfBothSteps:
    """pushforward_mean records the exactness of the rule and of the mean together."""

    def test_an_exact_rule_and_an_exact_mean_are_exact(self):
        result = expectation.using(method="empirical_enumeration")(_atoms(), lambda x: x)
        assert result.provenance.metadata["exact"] is True
        with_only = expectation.using(method="empirical_enumeration", exact_only=True)
        assert with_only(_atoms(), lambda x: x).provenance.metadata["exact"] is True

    def test_an_approximate_rule_is_approximate_in_the_check_and_the_record(self):
        report = expectation.check(Gaussian("g"), lambda x: x)
        assert report.exact is False
        assert "the mean of the pushforward is checked when it runs" in report.deferred
        with workflow_run(seed=0):
            result = expectation(Gaussian("g"), lambda x: x)
        assert result.provenance.metadata["exact"] is False

    def test_an_exact_rule_with_a_monte_carlo_mean_is_approximate(self, exact_to_sampler):
        with workflow_run(seed=0):
            result = expectation.using(num_samples=_DRAWS)(Gaussian("g"), lambda x: x)
        metadata = result.provenance.metadata
        assert (metadata["method"], metadata["exact"]) == ("exact_to_sampler", False)
        assert exact_to_sampler.pushed.shapes == [(_DRAWS,)]
        # 4000 draws of N(3, 1) give a standard error of about 0.016.
        assert _value(result) == pytest.approx(3.0, abs=0.1)

    def test_the_check_states_an_exact_rule_approximate_until_the_mean_runs(self):
        named = expectation.using(method="empirical_enumeration")
        report = named.check(_atoms(), lambda x: x)
        assert (report.route, report.method, report.exact) == (
            "pushforward_mean",
            "empirical_enumeration",
            False,
        )
        assert "the mean of the pushforward is checked when it runs" in report.deferred
        assert named(_atoms(), lambda x: x).provenance.metadata["exact"] is True

    def test_naming_the_route_selects_it(self):
        for options in ({}, {"exact_only": True}):
            named = expectation.using(method="pushforward_mean", **options)
            result = named(_atoms(), lambda x: x)
            assert result.provenance.metadata["route"] == "pushforward_mean"
            assert _value(result) == pytest.approx(2.0)

    def test_exact_only_refuses_monte_carlo_in_the_mean(self, exact_to_sampler):
        with pytest.raises(ResolutionError, match=r"^mean: no route applies with exact_only"):
            expectation.using(exact_only=True)(Gaussian("g"), lambda x: x)

    def test_the_mean_draws_its_own_count_when_the_call_sets_none(self, exact_to_sampler):
        with workflow_run(seed=3):
            expectation(Gaussian("g"), lambda x: x)
        assert exact_to_sampler.pushed.shapes == [(mean.options["num_samples"],)]

    def test_exact_only_refuses_monte_carlo_in_the_rule(self):
        with pytest.raises(ResolutionError, match="pushforward_mean"):
            expectation.using(exact_only=True)(Sampler("s"), lambda x: x)

    def test_an_empirical_law_takes_the_closed_form_with_and_without_exact_only(self):
        for options in ({}, {"exact_only": True}):
            report = expectation.using(**options).check(_atoms(), lambda x: x)
            assert (report.route, report.exact) == ("closed_form", True)

    def test_a_rule_name_selects_the_rule_inside_the_route(self):
        named = expectation.using(method="sampling_lift").check(_atoms(), lambda x: x)
        assert (named.route, named.method) == ("pushforward_mean", "sampling_lift")
        qualified = expectation.using(method="pushforward_mean/sampling_lift")
        assert qualified.check(_atoms(), lambda x: x).method == "sampling_lift"

    def test_the_body_and_the_route_agree_on_an_exact_case(self):
        """pushforward_mean replaces the generic route, so the body, which states the formula, never runs."""

        def scaled(x, a):
            return a * x * x

        body = expectation.apply(_atoms(), scaled, {"a": 2.0})
        route = expectation.using(method="empirical_enumeration")(_atoms(), scaled, {"a": 2.0})
        assert _value(body) == pytest.approx(_value(route)) == pytest.approx(28.0 / 3.0)


class TestTheDrawCount:
    """``num_samples`` is the number of draws of every Monte Carlo route, which provenance records."""

    @pytest.mark.parametrize("max_evaluations", [1, 5000])
    def test_a_monte_carlo_mean_follows_num_samples_and_ignores_max_evaluations(
        self, max_evaluations
    ):
        law = Sampler("s")
        estimate = mean.using(num_samples=13, max_evaluations=max_evaluations)(law)
        assert law.shapes == [(13,)]
        assert estimate.provenance.metadata["num_samples"] == 13

    @pytest.mark.parametrize(
        ("summary", "args"),
        [
            (mean, ()),
            (variance, ()),
            (standard_deviation, ()),
            (cov, ()),
            (quantile, (0.5,)),
        ],
        ids=["mean", "variance", "standard_deviation", "cov", "quantile"],
    )
    def test_each_monte_carlo_summary_records_the_draws_it_took(self, summary, args):
        law = Sampler("s")
        estimate = summary.using(num_samples=21)(law, *args)
        assert law.shapes == [(21,)]
        metadata = estimate.provenance.metadata
        assert (metadata["exact"], metadata["num_samples"]) == (False, 21)

    def test_the_standard_deviation_records_the_session_num_samples_of_its_variance(self):
        config.sampling.num_samples = 17
        estimate = standard_deviation(Sampler("s"))
        assert estimate.provenance.metadata["route"] == "derived"
        assert estimate.provenance.metadata["num_samples"] == 17

    def test_the_standard_deviation_of_a_dependent_joint_records_its_draws(self):
        with workflow_run(seed=3):
            result = standard_deviation.using(num_samples=31)(_dependent_joint())
        assert result.provenance.metadata["num_samples"] == 31

    def test_by_output_records_the_draws_of_each_monte_carlo_block(self):
        with workflow_run(seed=3):
            result = mean.using(num_samples=15)(_dependent_joint())
        assert result.provenance.metadata["route"] == "by_output"
        assert result.provenance.metadata["num_samples"] == 15
        (record,) = [
            parent.provenance
            for parent in result.provenance.parents
            if parent.provenance is not None
        ]
        blocks = {tuple(block["outputs"]): block for block in record.metadata["blocks"]}
        assert blocks[("mu",)]["exact"] is True
        assert "num_samples" not in blocks[("mu",)]
        assert blocks[("y",)]["num_samples"] == 15

    def test_an_exact_route_records_no_num_samples(self):
        estimate = mean.using(num_samples=15)(Gaussian("g", 2.0))
        assert estimate.provenance.metadata["exact"] is True
        assert "num_samples" not in estimate.provenance.metadata

    def test_a_nested_lift_spends_its_own_max_evaluations_and_the_inner_num_samples(self):
        """The outer lift evaluates its body 7 times, and each body takes 13 draws."""
        shapes: list[tuple[int, ...]] = []

        def centered(theta):
            law = Sampler("s", float(theta))
            estimate = mean.using(num_samples=13)(law)
            shapes.extend(law.shapes)
            return estimate.raw

        lifted = Function(centered, label="centered", dispatch="sequential")
        with workflow_run(seed=0):
            law = lifted.using(max_evaluations=7, num_samples=99)(Gaussian("theta", 0.0))
        assert law.num_atoms == 7
        assert shapes == [(13,)] * 7

    def test_a_lifts_num_samples_does_not_reach_a_summary_in_its_body(self):
        """A body's summary reads the session's ``num_samples``, as a call outside a derived route does."""
        config.sampling.num_samples = 17
        shapes: list[tuple[int, ...]] = []

        def centered(theta):
            law = Sampler("s", float(theta))
            estimate = mean(law)
            shapes.extend(law.shapes)
            return estimate.raw

        lifted = Function(centered, label="centered", dispatch="sequential")
        with workflow_run(seed=0):
            law = lifted.using(max_evaluations=6, num_samples=99)(Gaussian("theta", 0.0))
        assert law.num_atoms == 6
        assert shapes == [(17,)] * 6


def _nested_empirical() -> EmpiricalDistribution:
    """Three equally weighted atoms: the group ``g`` of ``mu`` and ``y``, beside ``c``."""
    atoms = NumericRecordBatch(
        {
            "g": {
                "mu": jnp.array([0.0, 1.0, 5.0]),
                "y": jnp.array([[1.0, 2.0], [0.0, 1.0], [2.0, 6.0]]),
            },
            "c": jnp.array([2.0, 1.0, 3.0]),
        },
        "row",
        element=RecordSpec(g=RecordSpec(mu=(), y=(2,)), c=()),
        label="rows",
    )
    return EmpiricalDistribution(atoms, label="post")


_SUMMARIES = [
    pytest.param(mean, id="mean"),
    pytest.param(variance, id="variance"),
    pytest.param(standard_deviation, id="standard_deviation"),
    pytest.param(lambda d: quantile(d, 0.5), id="quantile"),
    pytest.param(lambda d: quantile(d, jnp.array([0.25, 0.75])), id="quantiles"),
]


class TestASummaryTakesTheEventsNames:
    """A summary taken coordinate by coordinate takes the event's outputs and paths (VI.0)."""

    @pytest.mark.parametrize("summary", _SUMMARIES)
    def test_the_summary_declares_the_events_fields(self, summary):
        law = _nested_empirical()
        result = summary(law)
        element = (
            result.element_spec.term_spec if isinstance(result, NumericRecordBatch) else result.spec
        )
        assert tuple(element.children) == ("g", "c")
        assert tuple(element.children["g"].children) == ("mu", "y")

    @pytest.mark.parametrize("summary", _SUMMARIES)
    @pytest.mark.parametrize("path", ["c", "g", "g/mu", "g/y"])
    def test_selection_commutes_with_the_summary(self, summary, path):
        """The part of ``summary(d)`` at every event path ``p`` holds the value of ``summary(d[p])``."""
        law = _nested_empirical()
        result = summary(law)
        whole = result.at_path(path) if isinstance(result, Record) else result[path]
        part = summary(law[path])
        assert jax.tree.structure(whole.raw) == jax.tree.structure(part.raw)
        # Both are exact on the same atoms, so they differ by float32 rounding of
        # the reductions alone.
        for selected, direct in zip(
            jax.tree.leaves(whole.raw), jax.tree.leaves(part.raw), strict=True
        ):
            np.testing.assert_allclose(np.asarray(selected), np.asarray(direct), rtol=1e-6)

    def test_an_expectation_takes_the_integrands_output(self):
        f = Function(lambda x: x * x, output_spec=OutputSpec(sq=NumericArraySpec(())), label="f")
        assert tuple(expectation.check(_atoms(), f).result.outputs) == ("sq",)

    def test_a_law_of_one_whole_term_has_the_terms_summary(self):
        """The output of a whole-term law is the whole value, so paths below it index the summary."""
        atoms = NumericRecordBatch(
            {"a": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array([0.0, 0.0, 3.0])}, "row", label="rows"
        )
        law = EmpiricalDistribution(
            atoms, event_spec=OutputSpec(theta=RecordSpec(a=(), b=())), label="post"
        )
        assert mean.check(law).result.outputs == law.event_spec.outputs
        result = mean(law)
        assert isinstance(result, Record) and result.fields == ("a", "b")
        # Exact empirical averages of the same three atoms: one float32 rounding.
        np.testing.assert_allclose(_value(result["a"]), _value(mean(law["theta/a"])), rtol=1e-6)
        np.testing.assert_allclose(_value(result["b"]), 1.0, rtol=1e-6)


class TestASummaryBindsWhereADrawBinds:
    """A summary of a law plugs into the operations a draw of the law plugs into."""

    def test_a_law_scores_its_mean(self):
        law = Normal("mu", 0.5, 1.0) * Normal("tau", 1.0, 2.0)
        at_mean = log_prob(law, mean(law))
        at_values = log_prob(law, {"mu": 0.5, "tau": 1.0})
        # The closed-form means are the locations, so both calls score the same values.
        np.testing.assert_allclose(_value(at_mean), _value(at_values), rtol=0, atol=0)

    def test_an_integer_input_refuses_a_floating_mean(self):
        """The mean of an integer event is floating, so it does not conform where a draw does."""
        law = Bernoulli("z", probs=0.3)
        with workflow_run(seed=0):
            log_prob(law, sample(law))
        with pytest.raises(ApplicabilityError, match="float32"):
            log_prob(law, mean(law))

    def test_a_kernel_is_conditioned_on_a_posterior_mean(self):
        def y_given_mu(mu: jax.Array) -> Normal:
            return Normal("y", jnp.full(3, mu), 1.0)

        prior = Normal("mu", 0.0, 2.0)
        likelihood = conditional_distribution(y_given_mu, given_spec=prior.event_spec.outputs)
        posterior = EmpiricalDistribution(
            NumericRecordBatch({"mu": jnp.array([1.0, 2.0, 3.0])}, "row", label="mu"),
            label="posterior",
        )
        predictive = condition_on(likelihood * prior, mean(posterior))
        # The posterior mean of mu is 2, and the kernel's mean is mu in each coordinate.
        np.testing.assert_allclose(np.asarray(mean(predictive)), [2.0, 2.0, 2.0], rtol=1e-6)
