"""Tests for expectation(Distribution)."""

import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest

from probpipe import (
    Bernoulli,
    Beta,
    Binomial,
    BootstrapReplicateDistribution,
    Categorical,
    EmpiricalDistribution,
    Exponential,
    Function,
    Gamma,
    Normal,
    NumericArray,
    OutputSpec,
    config,
    expectation,
    mean,
    pushforward,
    variance,
    workflow_run,
)
from probpipe.core._dispatch import BinaryDispatchMethod, Feasibility, ResolutionError
from probpipe.distributions import Distribution
from probpipe.functions import _rules

# ---------------------------------------------------------------------------
# Expectation — the estimate is an array
# ---------------------------------------------------------------------------


class TestExpectationReturnsArray:
    """An expectation returns its estimate, exact or sample-based, as an array."""

    def test_return_dist_false_returns_array(self):
        d = Normal("x", loc=3.0, scale=1.0)
        result = expectation.using(num_samples=1000)(d, lambda x: x)
        assert isinstance(result, NumericArray)
        assert isinstance(jnp.asarray(result), jnp.ndarray)

    def test_bernoulli_exact_returns_array(self):
        """Finite-support exact expectations always return Array."""
        d = Bernoulli("x", probs=0.7)
        result = expectation(d, lambda x: x)
        assert isinstance(result, NumericArray)
        np.testing.assert_allclose(float(result), 0.7, atol=1e-6)

    def test_categorical_exact_returns_array(self):
        d = Categorical("x", probs=[0.1, 0.2, 0.3, 0.4])
        result = expectation(d, lambda x: x)
        assert isinstance(result, NumericArray)

    def test_empirical_exact_returns_array(self):
        """EmpiricalDistribution with num_evaluations=None is exact → Array."""
        d = EmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), output_name="x")
        result = expectation(d, lambda x: x)
        assert isinstance(result, NumericArray)


# ---------------------------------------------------------------------------
# Expectation — sample-based correctness (use return_dist=False for array comparison)
# ---------------------------------------------------------------------------


class TestExpectationSampleBased:
    """Test sample-based expectations on infinite-support distributions."""

    @pytest.fixture(autouse=True)
    def _seeded(self):
        """Each estimate draws in one seeded workflow, so its Monte Carlo error is fixed.

        The second moment of a normal has a standard error near half of its
        tolerance, so an unseeded estimate falls outside it in a few runs of a
        hundred.
        """
        with workflow_run(seed=0):
            yield

    def test_normal_mean(self):
        d = Normal("x", loc=3.0, scale=1.0)
        result = expectation.using(num_samples=10_000)(d, lambda x: x)
        np.testing.assert_allclose(float(result), 3.0, atol=0.05)

    def test_normal_second_moment(self):
        loc, scale = 2.0, 1.5
        d = Normal("x", loc=loc, scale=scale)
        result = expectation.using(num_samples=10_000)(d, lambda x: x**2)
        expected = loc**2 + scale**2
        # Second moment has higher variance than first moment (kurtosis effect)
        np.testing.assert_allclose(float(result), expected, atol=0.15)

    def test_normal_variance_from_moments(self):
        loc, scale = 1.0, 2.0
        d = Normal("x", loc=loc, scale=scale)
        ex = expectation.using(num_samples=10_000)(d, lambda x: x)
        ex2 = expectation.using(num_samples=10_000)(d, lambda x: x**2)
        var_est = float(ex2) - float(ex) ** 2
        np.testing.assert_allclose(var_est, scale**2, atol=0.15)

    def test_gamma_mean(self):
        conc, rate = 3.0, 2.0
        d = Gamma("x", concentration=conc, rate=rate)
        result = expectation.using(num_samples=10_000)(d, lambda x: x)
        np.testing.assert_allclose(float(result), conc / rate, atol=0.05)

    def test_gamma_log_sufficient_statistic(self):
        conc, rate = 3.0, 2.0
        d = Gamma("x", concentration=conc, rate=rate)
        result = expectation.using(num_samples=20_000)(d, lambda x: jnp.log(x))
        expected = float(jsp.digamma(conc)) - float(jnp.log(rate))
        np.testing.assert_allclose(float(result), expected, atol=0.05)

    def test_beta_mean(self):
        a, b = 2.0, 5.0
        d = Beta("x", alpha=a, beta=b)
        result = expectation.using(num_samples=10_000)(d, lambda x: x)
        np.testing.assert_allclose(float(result), a / (a + b), atol=0.03)

    def test_beta_log_sufficient_statistic(self):
        a, b = 2.0, 5.0
        d = Beta("x", alpha=a, beta=b)
        result = expectation.using(num_samples=20_000)(d, lambda x: jnp.log(x))
        expected = float(jsp.digamma(a)) - float(jsp.digamma(a + b))
        np.testing.assert_allclose(float(result), expected, atol=0.05)

    def test_exponential_second_moment(self):
        rate = 3.0
        d = Exponential("x", rate=rate)
        result = expectation.using(num_samples=10_000)(d, lambda x: x**2)
        np.testing.assert_allclose(float(result), 2.0 / rate**2, atol=0.03)


# ---------------------------------------------------------------------------
# Expectation — exact (finite support)
# ---------------------------------------------------------------------------


class TestExpectationExact:
    def test_bernoulli_identity(self):
        p = 0.7
        d = Bernoulli("x", probs=p)
        result = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(result), p, atol=1e-6)

    def test_bernoulli_custom_function(self):
        p = 0.4
        d = Bernoulli("x", probs=p)
        result = expectation(d, lambda x: 2 * x + 1)
        np.testing.assert_allclose(float(result), 1 + 2 * p, atol=1e-6)

    def test_categorical_identity(self):
        probs = [0.1, 0.2, 0.3, 0.4]
        d = Categorical("x", probs=probs)
        result = expectation(d, lambda x: x)
        expected = sum(i * p for i, p in enumerate(probs))
        np.testing.assert_allclose(float(result), expected, atol=1e-5)

    def test_categorical_custom_function(self):
        probs = [0.25, 0.5, 0.25]
        d = Categorical("x", probs=probs)
        result = expectation(d, lambda x: x**2)
        expected = 0 * 0.25 + 1 * 0.5 + 4 * 0.25
        np.testing.assert_allclose(float(result), expected, atol=1e-5)

    def test_binomial_mean(self):
        n, p = 10, 0.3
        d = Binomial("x", total_count=n, probs=p)
        result = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(result), n * p, atol=1e-4)

    def test_binomial_second_moment(self):
        n, p = 10, 0.3
        d = Binomial("x", total_count=n, probs=p)
        result = expectation(d, lambda x: x**2)
        expected = n * p * (1 - p) + (n * p) ** 2
        np.testing.assert_allclose(float(result), expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Expectation — EmpiricalDistribution
# ---------------------------------------------------------------------------


class TestExpectationEmpirical:
    def test_uniform_mean(self):
        samples = jnp.array([1.0, 2.0, 3.0, 4.0])
        d = EmpiricalDistribution(samples, output_name="x")
        result = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(result), 2.5, atol=1e-6)

    def test_weighted_mean(self):
        samples = jnp.array([0.0, 10.0])
        weights = jnp.array([0.3, 0.7])
        d = EmpiricalDistribution(samples, weights=weights, output_name="x")
        result = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(result), 7.0, atol=1e-5)

    def test_custom_function(self):
        samples = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.array([0.2, 0.5, 0.3])
        d = EmpiricalDistribution(samples, weights=weights, output_name="x")
        result = expectation(d, lambda x: x**2)
        expected = 0.2 * 1.0 + 0.5 * 4.0 + 0.3 * 9.0
        np.testing.assert_allclose(float(result), expected, atol=1e-5)

    def test_matches_mean_method(self):
        samples = jnp.array([1.0, 3.0, 5.0, 7.0])
        d = EmpiricalDistribution(samples, output_name="x")
        ex = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(ex), float(mean(d)), atol=1e-6)


# ---------------------------------------------------------------------------
# MC fallback mean()/variance()/cov() on base Distribution
# ---------------------------------------------------------------------------


class TestMCFallbackMethods:
    """Test that base mean(Distribution)/variance()/cov() use MC when no exact override."""

    def test_tfp_mean_still_exact(self):
        """mean(TFPDistribution) returns the exact array."""
        d = Normal("x", loc=3.0, scale=1.0)
        result = mean(d)
        assert isinstance(result, NumericArray)
        np.testing.assert_allclose(float(result), 3.0, atol=1e-6)

    def test_tfp_variance_still_exact(self):
        d = Normal("x", loc=0.0, scale=2.0)
        result = variance(d)
        assert isinstance(result, NumericArray)
        np.testing.assert_allclose(float(result), 4.0, atol=1e-6)

    def test_empirical_mean_still_exact(self):
        d = EmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), output_name="x")
        result = mean(d)
        assert isinstance(result, NumericArray)
        np.testing.assert_allclose(float(result), 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Global default tests
# ---------------------------------------------------------------------------


# Estimators that fall back to the session's counts when a call sets none.
_DEFAULT_SIZE_ESTIMATORS = [
    pytest.param(lambda: Normal("x", loc=0.0, scale=1.0), lambda x: x, id="tfp"),
    pytest.param(
        lambda: BootstrapReplicateDistribution(
            "boot", EmpiricalDistribution(jnp.arange(5.0), output_name="data")
        ),
        jnp.mean,
        id="bootstrap-replicate",
    ),
]


class TestGlobalDefaults:
    @pytest.mark.parametrize(("make_dist", "f"), _DEFAULT_SIZE_ESTIMATORS)
    def test_the_session_evaluation_count_sets_the_pushforward_size(self, make_dist, f):
        """A pushforward reads the current setting, not a copy taken at import."""
        config.sampling.max_evaluations = 7
        law = make_dist()
        assert pushforward(law, Function(f, output_spec=OutputSpec(integrand=None))).num_atoms == 7

    @pytest.mark.parametrize(("make_dist", "f"), _DEFAULT_SIZE_ESTIMATORS)
    def test_the_session_draw_count_sets_the_expectation_size(self, make_dist, f):
        """An expectation spends the current ``num_samples`` and ignores ``max_evaluations``."""
        config.sampling.num_samples = 7
        config.sampling.max_evaluations = 11
        estimate = expectation(make_dist(), f)
        assert np.isfinite(float(estimate))
        assert estimate.provenance.metadata["num_samples"] == 7


# ---------------------------------------------------------------------------
# The routes of expectation
# ---------------------------------------------------------------------------


class _StandIn(BinaryDispatchMethod):
    """An approximate rule registered opt-in, standing in for quadrature.

    Its pushforward is the point mass at -1, whatever the map and the law.
    """

    name = "quadrature_stand_in"
    exact = False
    priority = None

    def supported_types(self):
        return ((Function,), (Distribution,))

    def check(self, f, operand, /, **call):
        return Feasibility(True)

    def execute(self, f, operand, /, **call):
        return EmpiricalDistribution(jnp.array([-1.0]), output_name="stand_in")


@pytest.fixture
def rules(monkeypatch):
    """An evaluation-rule registry holding the engine's rules and the stand-in."""
    registry = type(_rules.evaluation_rule_registry)()
    for rule in (
        _rules._SamplingLift(),
        _rules._ElementwiseSweep(),
        _rules._EmpiricalEnumeration(),
        _StandIn(),
    ):
        registry.register(rule)
    monkeypatch.setattr(_rules, "evaluation_rule_registry", registry)
    for operation in (expectation, pushforward):
        (route,) = [
            r
            for r in operation._route_table.routes
            if r.name in ("evaluation_rules", "pushforward_mean")
        ]
        monkeypatch.setattr(route, "registry", registry)
    return registry


class TestExpectationRoutes:
    def test_a_finite_support_law_takes_the_closed_form(self):
        report = expectation.check(Bernoulli("b", probs=0.3), lambda x: x)
        assert report.selected.method_name == "closed_form"

    def test_a_continuous_law_takes_the_sampling_lift(self):
        report = expectation.check(Normal("n", 0.0, 1.0), lambda x: x)
        assert report.selected.method_name == "pushforward_mean/sampling_lift"

    def test_a_named_rule_runs_instead_of_the_closed_form(self):
        law = Bernoulli("b", probs=0.3)
        with workflow_run(seed=0):
            result = expectation.using(method="sampling_lift")(law, lambda x: x)
        # 256 draws give a standard error of about 0.029.
        np.testing.assert_allclose(float(result), 0.3, atol=0.1)

    def test_exact_only_refuses_a_law_without_an_exact_route(self):
        with pytest.raises(ResolutionError, match="exact_only"):
            expectation.using(exact_only=True)(Normal("n", 0.0, 1.0), lambda x: x)

    def test_an_unregistered_method_name_raises(self):
        with pytest.raises(ResolutionError, match="no_such_method"):
            expectation.using(method="no_such_method")(Normal("n", 0.0, 1.0), lambda x: x)

    def test_an_opt_in_rule_runs_only_when_named(self, rules):
        law = Normal("n", 0.0, 1.0)
        assert float(expectation.using(method="quadrature_stand_in")(law, lambda x: x)) == -1.0
        assert expectation.check(law, lambda x: x).selected.method_name == (
            "pushforward_mean/sampling_lift"
        )

    def test_a_rule_serves_pushforward_and_expectation_alike(self, rules):
        law = Normal("n", 0.0, 1.0)
        pushed = pushforward.using(method="quadrature_stand_in")(law, lambda x: x)
        assert float(mean(pushed)) == -1.0
        integrated = expectation.using(method="quadrature_stand_in")(law, lambda x: x)
        assert float(integrated) == -1.0

    def test_a_raised_priority_makes_a_rule_the_default(self, rules):
        law = Normal("n", 0.0, 1.0)
        rules.set_priorities(quadrature_stand_in=60)
        assert float(expectation(law, lambda x: x)) == -1.0
        assert expectation.check(law, lambda x: x).selected.method_name == (
            "pushforward_mean/quadrature_stand_in"
        )

    def test_the_closed_form_ranks_before_a_higher_priority_rule(self, rules):
        rules.set_priorities(quadrature_stand_in=1000)
        report = expectation.check(Bernoulli("b", probs=0.3), lambda x: x)
        assert report.selected.method_name == "closed_form"

    @pytest.mark.parametrize("bad", [0, -3])
    def test_monte_carlo_refuses_a_nonpositive_num_samples(self, bad):
        with pytest.raises(ValueError, match="positive"):
            expectation.using(num_samples=bad)(Normal("n", 0.0, 1.0), lambda x: x)

    def test_monte_carlo_refuses_a_non_integer_num_samples(self):
        with pytest.raises(TypeError, match="integer"):
            expectation.using(num_samples=2.5)(Normal("n", 0.0, 1.0), lambda x: x)
