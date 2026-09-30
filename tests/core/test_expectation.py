"""Tests for expectation(Distribution), BootstrapDistribution, and is_approximate."""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

import probpipe.distributions._distribution as dist_mod
from probpipe import (
    Bernoulli,
    Beta,
    Binomial,
    BootstrapDistribution,
    BootstrapReplicateDistribution,
    Categorical,
    EmpiricalDistribution,
    Exponential,
    Gamma,
    Normal,
    NumericArray,
    NumericRecord,
    RecordEmpiricalDistribution,
    TransformedDistribution,
    expectation,
    from_distribution,
    mean,
    sample,
    set_default_num_evaluations,
    variance,
)
from probpipe.core._dispatch import Feasibility, ResolutionError
from probpipe.operations import ExpectationMethod, expectation_method_registry

# ---------------------------------------------------------------------------
# BootstrapDistribution tests
# ---------------------------------------------------------------------------


class TestBootstrapDistribution:
    """Test BootstrapDistribution construction and properties."""

    def test_construction(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BootstrapDistribution("bd", evals)
        assert bd.num_atoms == 5
        assert bd.event_shape == ()
        assert bd.is_approximate

    def test_mean(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BootstrapDistribution("bd", evals)
        np.testing.assert_allclose(float(mean(bd)), 3.0, atol=1e-6)

    def test_variance(self):
        """Variance of bootstrap mean = Var(evals) / n."""
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BootstrapDistribution("bd", evals)
        sample_var = float(jnp.var(evals))
        expected_se_var = sample_var / 5
        np.testing.assert_allclose(float(variance(bd)), expected_se_var, atol=1e-5)

    def test_weighted(self):
        evals = jnp.array([0.0, 10.0])
        weights = jnp.array([0.3, 0.7])
        bd = BootstrapDistribution("bd", evals, weights=weights)
        np.testing.assert_allclose(float(mean(bd)), 7.0, atol=1e-5)

    def test_sample(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BootstrapDistribution("bd", evals)
        key = jax.random.PRNGKey(0)
        samples = sample(bd, key=key, sample_shape=(100,))
        assert samples.shape == (100,)
        # Bootstrap means should cluster around 3.0
        np.testing.assert_allclose(float(jnp.mean(samples)), 3.0, atol=0.5)

    def test_multidim_evals(self):
        evals = jnp.ones((10, 3))
        bd = BootstrapDistribution("bd", evals)
        assert bd.event_shape == (3,)
        assert mean(bd).shape == (3,)


# ---------------------------------------------------------------------------
# Expectation — returns BootstrapDistribution by default
# ---------------------------------------------------------------------------


class TestExpectationReturnsDist:
    """With RETURN_APPROX_DIST=True (default), sample-based expectations return BootstrapDistribution."""

    def test_return_dist_false_returns_array(self):
        d = Normal(loc=3.0, scale=1.0, name="x")
        key = jax.random.PRNGKey(0)
        result = expectation(d, lambda x: x, key=key, num_evaluations=1000)
        assert isinstance(result, NumericArray)
        assert isinstance(jnp.asarray(result), jnp.ndarray)

    def test_bernoulli_exact_returns_array(self):
        """Finite-support exact expectations always return Array."""
        d = Bernoulli(probs=0.7, name="x")
        result = expectation(d, lambda x: x)
        assert isinstance(result, NumericArray)
        np.testing.assert_allclose(float(result), 0.7, atol=1e-6)

    def test_categorical_exact_returns_array(self):
        d = Categorical(probs=[0.1, 0.2, 0.3, 0.4], name="x")
        result = expectation(d, lambda x: x)
        assert isinstance(result, NumericArray)

    def test_empirical_exact_returns_array(self):
        """EmpiricalDistribution with num_evaluations=None is exact → Array."""
        d = EmpiricalDistribution("x", jnp.array([1.0, 2.0, 3.0]))
        result = expectation(d, lambda x: x)
        assert isinstance(result, NumericArray)


# ---------------------------------------------------------------------------
# Expectation — sample-based correctness (use return_dist=False for array comparison)
# ---------------------------------------------------------------------------


class TestExpectationSampleBased:
    """Test sample-based expectations on infinite-support distributions."""

    def test_normal_mean(self):
        d = Normal(loc=3.0, scale=1.0, name="x")
        key = jax.random.PRNGKey(0)
        result = expectation(d, lambda x: x, key=key, num_evaluations=10_000)
        np.testing.assert_allclose(float(result), 3.0, atol=0.05)

    def test_normal_second_moment(self):
        loc, scale = 2.0, 1.5
        d = Normal(loc=loc, scale=scale, name="x")
        key = jax.random.PRNGKey(1)
        result = expectation(d, lambda x: x**2, key=key, num_evaluations=10_000)
        expected = loc**2 + scale**2
        # Second moment has higher variance than first moment (kurtosis effect)
        np.testing.assert_allclose(float(result), expected, atol=0.15)

    def test_normal_variance_from_moments(self):
        loc, scale = 1.0, 2.0
        d = Normal(loc=loc, scale=scale, name="x")
        key1, key2 = jax.random.split(jax.random.PRNGKey(2))
        ex = expectation(d, lambda x: x, key=key1, num_evaluations=10_000)
        ex2 = expectation(d, lambda x: x**2, key=key2, num_evaluations=10_000)
        var_est = float(ex2) - float(ex) ** 2
        np.testing.assert_allclose(var_est, scale**2, atol=0.15)

    def test_gamma_mean(self):
        conc, rate = 3.0, 2.0
        d = Gamma(concentration=conc, rate=rate, name="x")
        key = jax.random.PRNGKey(3)
        result = expectation(d, lambda x: x, key=key, num_evaluations=10_000)
        np.testing.assert_allclose(float(result), conc / rate, atol=0.05)

    def test_gamma_log_sufficient_statistic(self):
        conc, rate = 3.0, 2.0
        d = Gamma(concentration=conc, rate=rate, name="x")
        key = jax.random.PRNGKey(4)
        result = expectation(d, lambda x: jnp.log(x), key=key, num_evaluations=20_000)
        expected = float(jsp.digamma(conc)) - float(jnp.log(rate))
        np.testing.assert_allclose(float(result), expected, atol=0.05)

    def test_beta_mean(self):
        a, b = 2.0, 5.0
        d = Beta(alpha=a, beta=b, name="x")
        key = jax.random.PRNGKey(5)
        result = expectation(d, lambda x: x, key=key, num_evaluations=10_000)
        np.testing.assert_allclose(float(result), a / (a + b), atol=0.03)

    def test_beta_log_sufficient_statistic(self):
        a, b = 2.0, 5.0
        d = Beta(alpha=a, beta=b, name="x")
        key = jax.random.PRNGKey(6)
        result = expectation(d, lambda x: jnp.log(x), key=key, num_evaluations=20_000)
        expected = float(jsp.digamma(a)) - float(jsp.digamma(a + b))
        np.testing.assert_allclose(float(result), expected, atol=0.05)

    def test_exponential_second_moment(self):
        rate = 3.0
        d = Exponential(rate=rate, name="x")
        key = jax.random.PRNGKey(7)
        result = expectation(d, lambda x: x**2, key=key, num_evaluations=10_000)
        np.testing.assert_allclose(float(result), 2.0 / rate**2, atol=0.03)


# ---------------------------------------------------------------------------
# Expectation — exact (finite support)
# ---------------------------------------------------------------------------


class TestExpectationExact:
    def test_bernoulli_identity(self):
        p = 0.7
        d = Bernoulli(probs=p, name="x")
        result = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(result), p, atol=1e-6)

    def test_bernoulli_custom_function(self):
        p = 0.4
        d = Bernoulli(probs=p, name="x")
        result = expectation(d, lambda x: 2 * x + 1)
        np.testing.assert_allclose(float(result), 1 + 2 * p, atol=1e-6)

    def test_categorical_identity(self):
        probs = [0.1, 0.2, 0.3, 0.4]
        d = Categorical(probs=probs, name="x")
        result = expectation(d, lambda x: x)
        expected = sum(i * p for i, p in enumerate(probs))
        np.testing.assert_allclose(float(result), expected, atol=1e-5)

    def test_categorical_custom_function(self):
        probs = [0.25, 0.5, 0.25]
        d = Categorical(probs=probs, name="x")
        result = expectation(d, lambda x: x**2)
        expected = 0 * 0.25 + 1 * 0.5 + 4 * 0.25
        np.testing.assert_allclose(float(result), expected, atol=1e-5)

    def test_binomial_mean(self):
        n, p = 10, 0.3
        d = Binomial(total_count=n, probs=p, name="x")
        result = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(result), n * p, atol=1e-4)

    def test_binomial_second_moment(self):
        n, p = 10, 0.3
        d = Binomial(total_count=n, probs=p, name="x")
        result = expectation(d, lambda x: x**2)
        expected = n * p * (1 - p) + (n * p) ** 2
        np.testing.assert_allclose(float(result), expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Expectation — EmpiricalDistribution
# ---------------------------------------------------------------------------


class TestExpectationEmpirical:
    def test_uniform_mean(self):
        samples = jnp.array([1.0, 2.0, 3.0, 4.0])
        d = EmpiricalDistribution("x", samples)
        result = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(result), 2.5, atol=1e-6)

    def test_weighted_mean(self):
        samples = jnp.array([0.0, 10.0])
        weights = jnp.array([0.3, 0.7])
        d = EmpiricalDistribution("x", samples, weights=weights)
        result = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(result), 7.0, atol=1e-5)

    def test_custom_function(self):
        samples = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.array([0.2, 0.5, 0.3])
        d = EmpiricalDistribution("x", samples, weights=weights)
        result = expectation(d, lambda x: x**2)
        expected = 0.2 * 1.0 + 0.5 * 4.0 + 0.3 * 9.0
        np.testing.assert_allclose(float(result), expected, atol=1e-5)

    def test_matches_mean_method(self):
        samples = jnp.array([1.0, 3.0, 5.0, 7.0])
        d = RecordEmpiricalDistribution("x", samples)
        ex = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(ex), float(mean(d)), atol=1e-6)


# ---------------------------------------------------------------------------
# MC fallback mean()/variance()/cov() on base Distribution
# ---------------------------------------------------------------------------


class TestMCFallbackMethods:
    """Test that base mean(Distribution)/variance()/cov() use MC when no exact override."""

    def test_tfp_mean_still_exact(self):
        """mean(TFPDistribution) returns exact Array, not BootstrapDistribution."""
        d = Normal(loc=3.0, scale=1.0, name="x")
        result = mean(d)
        assert isinstance(result, NumericArray)
        np.testing.assert_allclose(float(result), 3.0, atol=1e-6)

    def test_tfp_variance_still_exact(self):
        d = Normal(loc=0.0, scale=2.0, name="x")
        result = variance(d)
        assert isinstance(result, NumericArray)
        np.testing.assert_allclose(float(result), 4.0, atol=1e-6)

    def test_empirical_mean_still_exact(self):
        d = RecordEmpiricalDistribution("x", jnp.array([1.0, 2.0, 3.0]))
        result = mean(d)
        assert isinstance(result, NumericRecord)
        np.testing.assert_allclose(float(result), 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# is_approximate tests
# ---------------------------------------------------------------------------


class TestIsApproximate:
    def test_tfp_distribution_exact(self):
        assert not Normal(loc=0.0, scale=1.0, name="x").is_approximate
        assert not Gamma(concentration=1.0, rate=1.0, name="x").is_approximate
        assert not Beta(alpha=1.0, beta=1.0, name="x").is_approximate
        assert not Bernoulli(probs=0.5, name="x").is_approximate

    def test_empirical_approximate_by_default(self):
        d = EmpiricalDistribution("x", jnp.array([1.0, 2.0]))
        assert d.is_approximate

    def test_bootstrap_always_approximate(self):
        bd = BootstrapDistribution("bd", jnp.array([1.0, 2.0, 3.0]))
        assert bd.is_approximate

    def test_transformed_propagates(self):
        exact_base = Normal(loc=0.0, scale=1.0, name="x")
        t_exact = TransformedDistribution("t_exact", exact_base, tfb.Exp())
        assert not t_exact.is_approximate

        approx_base = EmpiricalDistribution("x", jnp.array([1.0, 2.0]))
        t_approx = TransformedDistribution("t_approx", approx_base, tfb.Exp())
        assert t_approx.is_approximate

    def test_from_distribution_same_class_exact(self):
        d = Normal(loc=0.0, scale=1.0, name="x")
        d2 = from_distribution(d, Normal)
        assert not d2.is_approximate

    def test_from_distribution_different_class_approximate(self):
        d = Normal(loc=5.0, scale=0.1, name="x")
        d2 = from_distribution(d, Gamma, check_support=False)
        assert d2.is_approximate

    def test_from_distribution_to_empirical(self):
        d = Normal(loc=0.0, scale=1.0, name="x")
        d2 = from_distribution(d, RecordEmpiricalDistribution)
        assert d2.is_approximate


# ---------------------------------------------------------------------------
# Global default tests
# ---------------------------------------------------------------------------


# Estimators that fall back to the default sample count when a call omits
# ``num_evaluations``.
_DEFAULT_SIZE_ESTIMATORS = [
    pytest.param(lambda: Normal(loc=0.0, scale=1.0, name="x"), lambda x: x, id="tfp"),
    pytest.param(
        lambda: BootstrapReplicateDistribution(
            "boot", EmpiricalDistribution("data", jnp.arange(5.0))
        ),
        jnp.mean,
        id="bootstrap-replicate",
    ),
]


class TestGlobalDefaults:
    def test_set_default_num_evaluations(self):
        old = dist_mod.DEFAULT_NUM_EVALUATIONS
        try:
            set_default_num_evaluations(512)
            assert dist_mod.DEFAULT_NUM_EVALUATIONS == 512
        finally:
            set_default_num_evaluations(old)

    @pytest.mark.parametrize(("make_dist", "f"), _DEFAULT_SIZE_ESTIMATORS)
    def test_default_num_evaluations_sets_the_estimate_size(self, make_dist, f):
        """An estimator reads the current default sample count, not a copy taken at import."""
        old = dist_mod.DEFAULT_NUM_EVALUATIONS
        try:
            set_default_num_evaluations(7)
            law = make_dist()
            key = jax.random.PRNGKey(0)
            result = expectation(law, f, key=key)
            expected = jnp.mean(jax.vmap(f)(law._sample(key, (7,))), axis=0)
            np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-6)
        finally:
            set_default_num_evaluations(old)

    def test_set_default_invalid(self):
        with pytest.raises(ValueError):
            set_default_num_evaluations(0)


# ---------------------------------------------------------------------------
# The expectation methods
# ---------------------------------------------------------------------------


class _Stand_In(ExpectationMethod):
    """An approximate method registered opt-in, standing in for quadrature."""

    name = "quadrature_stand_in"
    exact = False
    priority = None

    def check(self, d, f, /, **options):
        return Feasibility(True)

    def execute(self, d, f, /, **options):
        return jnp.asarray(-1.0)


if "quadrature_stand_in" not in expectation_method_registry.list_methods():
    expectation_method_registry.register(_Stand_In())


class TestExpectationMethods:
    def test_a_finite_support_law_takes_the_exact_method(self):
        law = Bernoulli("b", probs=0.3)
        assert expectation_method_registry.check(law, lambda x: x).method_name == "exact"

    def test_a_continuous_law_takes_the_monte_carlo_default(self):
        law = Normal("n", 0.0, 1.0)
        assert expectation_method_registry.check(law, lambda x: x).method_name == "monte_carlo"

    def test_a_named_method_runs_instead_of_the_selected_one(self):
        law = Bernoulli("b", probs=0.3)
        result = expectation(law, lambda x: x, method="monte_carlo", key=jax.random.PRNGKey(0))
        np.testing.assert_allclose(float(result), 0.3, atol=0.1)

    def test_exact_only_refuses_a_law_without_an_exact_method(self):
        with pytest.raises(ResolutionError, match="exact_only"):
            expectation(Normal("n", 0.0, 1.0), lambda x: x, exact_only=True)

    def test_an_unregistered_method_name_raises(self):
        with pytest.raises(ResolutionError, match="No method named"):
            expectation(Normal("n", 0.0, 1.0), lambda x: x, method="no_such_method")

    def test_an_opt_in_method_runs_only_when_named(self):
        law = Normal("n", 0.0, 1.0)
        assert float(expectation(law, lambda x: x, method="quadrature_stand_in")) == -1.0
        assert expectation_method_registry.check(law, lambda x: x).method_name == "monte_carlo"

    def test_a_raised_priority_makes_a_method_the_default(self):
        law = Normal("n", 0.0, 1.0)
        try:
            expectation_method_registry.set_priorities(quadrature_stand_in=60)
            assert float(expectation(law, lambda x: x)) == -1.0
            assert expectation_method_registry.check(law, lambda x: x).method_name == (
                "quadrature_stand_in"
            )
        finally:
            expectation_method_registry.set_priorities(quadrature_stand_in=None)
        assert expectation_method_registry.check(law, lambda x: x).method_name == "monte_carlo"

    def test_the_exact_method_ranks_before_a_higher_priority_approximate_one(self):
        law = Bernoulli("b", probs=0.3)
        try:
            expectation_method_registry.set_priorities(quadrature_stand_in=1000)
            assert expectation_method_registry.check(law, lambda x: x).method_name == "exact"
        finally:
            expectation_method_registry.set_priorities(quadrature_stand_in=None)

    @pytest.mark.parametrize("bad", [0, -3])
    def test_monte_carlo_refuses_a_nonpositive_sample_count(self, bad):
        with pytest.raises(ValueError, match="positive"):
            expectation(Normal("n", 0.0, 1.0), lambda x: x, num_evaluations=bad)

    def test_monte_carlo_refuses_a_non_integer_sample_count(self):
        with pytest.raises(TypeError, match="integer"):
            expectation(Normal("n", 0.0, 1.0), lambda x: x, num_evaluations=2.5)
