"""Tests for expectation(Distribution), BootstrapDistribution, and is_approximate."""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest

import probpipe.core.distribution as dist_mod
from probpipe import (
    ArrayDistribution,
    ArrayEmpiricalDistribution,
    EmpiricalDistribution,
    BootstrapDistribution,
    Normal,
    Gamma,
    Beta,
    Exponential,
    Bernoulli,
    Categorical,
    Binomial,
    from_distribution,
    TransformedDistribution,
    DEFAULT_NUM_EVALUATIONS,
    set_default_num_evaluations,
    set_return_approx_dist,
)
import tensorflow_probability.substrates.jax.bijectors as tfb
from probpipe import expectation, log_prob, mean, sample, variance


# ---------------------------------------------------------------------------
# BootstrapDistribution tests
# ---------------------------------------------------------------------------


class TestBootstrapDistribution:
    """Test BootstrapDistribution construction and properties."""

    def test_construction(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BootstrapDistribution(evals)
        assert bd.n == 5
        assert bd.event_shape == ()
        assert bd.is_approximate

    def test_mean(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BootstrapDistribution(evals)
        np.testing.assert_allclose(float(mean(bd)), 3.0, atol=1e-6)

    def test_variance(self):
        """Variance of bootstrap mean = Var(evals) / n."""
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BootstrapDistribution(evals)
        sample_var = float(jnp.var(evals))
        expected_se_var = sample_var / 5
        np.testing.assert_allclose(float(variance(bd)), expected_se_var, atol=1e-5)

    def test_weighted(self):
        evals = jnp.array([0.0, 10.0])
        weights = jnp.array([0.3, 0.7])
        bd = BootstrapDistribution(evals, weights=weights)
        np.testing.assert_allclose(float(mean(bd)), 7.0, atol=1e-5)

    def test_sample(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BootstrapDistribution(evals)
        key = jax.random.PRNGKey(0)
        samples = sample(bd, key=key, sample_shape=(100,))
        assert samples.shape == (100,)
        # Bootstrap means should cluster around 3.0
        np.testing.assert_allclose(float(jnp.mean(samples)), 3.0, atol=0.5)

    def test_multidim_evals(self):
        evals = jnp.ones((10, 3))
        bd = BootstrapDistribution(evals)
        assert bd.event_shape == (3,)
        assert mean(bd).shape == (3,)



# ---------------------------------------------------------------------------
# Expectation — returns BootstrapDistribution by default
# ---------------------------------------------------------------------------


class TestExpectationReturnsDist:
    """With RETURN_APPROX_DIST=True (default), sample-based expectations return BootstrapDistribution."""

    def test_normal_returns_bootstrap(self):
        d = Normal(loc=3.0, scale=1.0)
        key = jax.random.PRNGKey(0)
        result = expectation(d, lambda x: x, key=key, num_evaluations=1000)
        assert isinstance(result, BootstrapDistribution)
        np.testing.assert_allclose(float(mean(result)), 3.0, atol=0.2)

    def test_return_dist_false_returns_array(self):
        d = Normal(loc=3.0, scale=1.0)
        key = jax.random.PRNGKey(0)
        result = expectation(d, lambda x: x, key=key, num_evaluations=1000, return_dist=False)
        assert isinstance(result, jnp.ndarray)

    def test_bernoulli_exact_returns_array(self):
        """Finite-support exact expectations always return Array."""
        d = Bernoulli(probs=0.7)
        result = expectation(d, lambda x: x)
        assert isinstance(result, jnp.ndarray)
        np.testing.assert_allclose(float(result), 0.7, atol=1e-6)

    def test_categorical_exact_returns_array(self):
        d = Categorical(probs=[0.1, 0.2, 0.3, 0.4])
        result = expectation(d, lambda x: x)
        assert isinstance(result, jnp.ndarray)

    def test_empirical_exact_returns_array(self):
        """EmpiricalDistribution with num_evaluations=None is exact → Array."""
        d = EmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]))
        result = expectation(d, lambda x: x)
        assert isinstance(result, jnp.ndarray)

    def test_empirical_subsample_returns_bootstrap(self):
        """EmpiricalDistribution with num_evaluations < n is approximate → Bootstrap."""
        d = EmpiricalDistribution(jnp.arange(100.0))
        key = jax.random.PRNGKey(0)
        result = expectation(d, lambda x: x, key=key, num_evaluations=10)
        assert isinstance(result, BootstrapDistribution)


# ---------------------------------------------------------------------------
# Expectation — sample-based correctness (use return_dist=False for array comparison)
# ---------------------------------------------------------------------------


class TestExpectationSampleBased:
    """Test sample-based expectations on infinite-support distributions."""

    def test_normal_mean(self):
        d = Normal(loc=3.0, scale=1.0)
        key = jax.random.PRNGKey(0)
        result = expectation(d, lambda x: x, key=key, num_evaluations=10_000, return_dist=False)
        np.testing.assert_allclose(float(result), 3.0, atol=0.1)

    def test_normal_second_moment(self):
        loc, scale = 2.0, 1.5
        d = Normal(loc=loc, scale=scale)
        key = jax.random.PRNGKey(1)
        result = expectation(d, lambda x: x ** 2, key=key, num_evaluations=10_000, return_dist=False)
        expected = loc ** 2 + scale ** 2
        np.testing.assert_allclose(float(result), expected, atol=0.15)

    def test_normal_variance_from_moments(self):
        loc, scale = 1.0, 2.0
        d = Normal(loc=loc, scale=scale)
        key1, key2 = jax.random.split(jax.random.PRNGKey(2))
        ex = expectation(d, lambda x: x, key=key1, num_evaluations=10_000, return_dist=False)
        ex2 = expectation(d, lambda x: x ** 2, key=key2, num_evaluations=10_000, return_dist=False)
        var_est = float(ex2) - float(ex) ** 2
        np.testing.assert_allclose(var_est, scale ** 2, atol=0.3)

    def test_gamma_mean(self):
        conc, rate = 3.0, 2.0
        d = Gamma(concentration=conc, rate=rate)
        key = jax.random.PRNGKey(3)
        result = expectation(d, lambda x: x, key=key, num_evaluations=10_000, return_dist=False)
        np.testing.assert_allclose(float(result), conc / rate, atol=0.1)

    def test_gamma_log_sufficient_statistic(self):
        conc, rate = 3.0, 2.0
        d = Gamma(concentration=conc, rate=rate)
        key = jax.random.PRNGKey(4)
        result = expectation(d, lambda x: jnp.log(x), key=key, num_evaluations=20_000, return_dist=False)
        expected = float(jsp.digamma(conc)) - float(jnp.log(rate))
        np.testing.assert_allclose(float(result), expected, atol=0.1)

    def test_beta_mean(self):
        a, b = 2.0, 5.0
        d = Beta(alpha=a, beta=b)
        key = jax.random.PRNGKey(5)
        result = expectation(d, lambda x: x, key=key, num_evaluations=10_000, return_dist=False)
        np.testing.assert_allclose(float(result), a / (a + b), atol=0.05)

    def test_beta_log_sufficient_statistic(self):
        a, b = 2.0, 5.0
        d = Beta(alpha=a, beta=b)
        key = jax.random.PRNGKey(6)
        result = expectation(d, lambda x: jnp.log(x), key=key, num_evaluations=20_000, return_dist=False)
        expected = float(jsp.digamma(a)) - float(jsp.digamma(a + b))
        np.testing.assert_allclose(float(result), expected, atol=0.1)

    def test_exponential_second_moment(self):
        rate = 3.0
        d = Exponential(rate=rate)
        key = jax.random.PRNGKey(7)
        result = expectation(d, lambda x: x ** 2, key=key, num_evaluations=10_000, return_dist=False)
        np.testing.assert_allclose(float(result), 2.0 / rate ** 2, atol=0.05)


# ---------------------------------------------------------------------------
# Expectation — exact (finite support)
# ---------------------------------------------------------------------------


class TestExpectationExact:

    def test_bernoulli_identity(self):
        p = 0.7
        d = Bernoulli(probs=p)
        result = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(result), p, atol=1e-6)

    def test_bernoulli_custom_function(self):
        p = 0.4
        d = Bernoulli(probs=p)
        result = expectation(d, lambda x: 2 * x + 1)
        np.testing.assert_allclose(float(result), 1 + 2 * p, atol=1e-6)

    def test_categorical_identity(self):
        probs = [0.1, 0.2, 0.3, 0.4]
        d = Categorical(probs=probs)
        result = expectation(d, lambda x: x)
        expected = sum(i * p for i, p in enumerate(probs))
        np.testing.assert_allclose(float(result), expected, atol=1e-5)

    def test_categorical_custom_function(self):
        probs = [0.25, 0.5, 0.25]
        d = Categorical(probs=probs)
        result = expectation(d, lambda x: x ** 2)
        expected = 0 * 0.25 + 1 * 0.5 + 4 * 0.25
        np.testing.assert_allclose(float(result), expected, atol=1e-5)

    def test_binomial_mean(self):
        n, p = 10, 0.3
        d = Binomial(total_count=n, probs=p)
        result = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(result), n * p, atol=1e-4)

    def test_binomial_second_moment(self):
        n, p = 10, 0.3
        d = Binomial(total_count=n, probs=p)
        result = expectation(d, lambda x: x ** 2)
        expected = n * p * (1 - p) + (n * p) ** 2
        np.testing.assert_allclose(float(result), expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Expectation — EmpiricalDistribution
# ---------------------------------------------------------------------------


class TestExpectationEmpirical:

    def test_uniform_mean(self):
        samples = jnp.array([1.0, 2.0, 3.0, 4.0])
        d = EmpiricalDistribution(samples)
        result = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(result), 2.5, atol=1e-6)

    def test_weighted_mean(self):
        samples = jnp.array([0.0, 10.0])
        weights = jnp.array([0.3, 0.7])
        d = EmpiricalDistribution(samples, weights=weights)
        result = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(result), 7.0, atol=1e-5)

    def test_custom_function(self):
        samples = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.array([0.2, 0.5, 0.3])
        d = EmpiricalDistribution(samples, weights=weights)
        result = expectation(d, lambda x: x ** 2)
        expected = 0.2 * 1.0 + 0.5 * 4.0 + 0.3 * 9.0
        np.testing.assert_allclose(float(result), expected, atol=1e-5)

    def test_subsample_returns_bootstrap(self):
        samples = jnp.arange(100.0)
        d = EmpiricalDistribution(samples)
        key = jax.random.PRNGKey(0)
        result = expectation(d, lambda x: x, key=key, num_evaluations=10)
        assert isinstance(result, BootstrapDistribution)

    def test_subsample_return_dist_false(self):
        samples = jnp.arange(100.0)
        d = EmpiricalDistribution(samples)
        key = jax.random.PRNGKey(0)
        result = expectation(d, lambda x: x, key=key, num_evaluations=10, return_dist=False)
        assert isinstance(result, jnp.ndarray)

    def test_matches_mean_method(self):
        samples = jnp.array([1.0, 3.0, 5.0, 7.0])
        d = ArrayEmpiricalDistribution(samples)
        ex = expectation(d, lambda x: x)
        np.testing.assert_allclose(float(ex), float(mean(d)), atol=1e-6)


# ---------------------------------------------------------------------------
# Bootstrap error tracking — MC error decreases with n
# ---------------------------------------------------------------------------


class TestBootstrapErrorTracking:

    def test_variance_decreases_with_n(self):
        """More evaluations → smaller MC error variance."""
        d = Normal(loc=0.0, scale=1.0)
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        bd_small = expectation(d, lambda x: x, key=key1, num_evaluations=100)
        bd_large = expectation(d, lambda x: x, key=key2, num_evaluations=10_000)
        assert isinstance(bd_small, BootstrapDistribution)
        assert isinstance(bd_large, BootstrapDistribution)
        assert float(variance(bd_large)) < float(variance(bd_small))

    def test_bootstrap_mean_matches_point_estimate(self):
        """mean(BootstrapDistribution) equals the sample mean."""
        d = Normal(loc=5.0, scale=1.0)
        key = jax.random.PRNGKey(0)
        bd = expectation(d, lambda x: x, key=key, num_evaluations=1000)
        point_est = expectation(d, lambda x: x, key=key, num_evaluations=1000, return_dist=False)
        np.testing.assert_allclose(float(mean(bd)), float(point_est), atol=1e-5)


# ---------------------------------------------------------------------------
# MC fallback mean()/variance()/cov() on base Distribution
# ---------------------------------------------------------------------------


class TestMCFallbackMethods:
    """Test that base mean(Distribution)/variance()/cov() use MC when no exact override."""

    def test_tfp_mean_still_exact(self):
        """mean(TFPDistribution) returns exact Array, not BootstrapDistribution."""
        d = Normal(loc=3.0, scale=1.0)
        result = mean(d)
        assert isinstance(result, jnp.ndarray)
        np.testing.assert_allclose(float(result), 3.0, atol=1e-6)

    def test_tfp_variance_still_exact(self):
        d = Normal(loc=0.0, scale=2.0)
        result = variance(d)
        assert isinstance(result, jnp.ndarray)
        np.testing.assert_allclose(float(result), 4.0, atol=1e-6)

    def test_empirical_mean_still_exact(self):
        d = ArrayEmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]))
        result = mean(d)
        assert isinstance(result, jnp.ndarray)
        np.testing.assert_allclose(float(result), 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# is_approximate tests
# ---------------------------------------------------------------------------


class TestIsApproximate:

    def test_tfp_distribution_exact(self):
        assert not Normal(loc=0.0, scale=1.0).is_approximate
        assert not Gamma(concentration=1.0, rate=1.0).is_approximate
        assert not Beta(alpha=1.0, beta=1.0).is_approximate
        assert not Bernoulli(probs=0.5).is_approximate

    def test_empirical_approximate_by_default(self):
        d = EmpiricalDistribution(jnp.array([1.0, 2.0]))
        assert d.is_approximate

    def test_bootstrap_always_approximate(self):
        bd = BootstrapDistribution(jnp.array([1.0, 2.0, 3.0]))
        assert bd.is_approximate

    def test_transformed_propagates(self):
        exact_base = Normal(loc=0.0, scale=1.0)
        t_exact = TransformedDistribution(exact_base, tfb.Exp())
        assert not t_exact.is_approximate

        approx_base = EmpiricalDistribution(jnp.array([1.0, 2.0]))
        t_approx = TransformedDistribution(approx_base, tfb.Exp())
        assert t_approx.is_approximate

    def test_from_distribution_same_class_exact(self):
        d = Normal(loc=0.0, scale=1.0)
        d2 = from_distribution(d, Normal)
        assert not d2.is_approximate

    def test_from_distribution_different_class_approximate(self):
        d = Normal(loc=5.0, scale=0.1)
        d2 = from_distribution(d, Gamma, check_support=False)
        assert d2.is_approximate

    def test_from_distribution_to_empirical(self):
        d = Normal(loc=0.0, scale=1.0)
        d2 = from_distribution(d, ArrayEmpiricalDistribution)
        assert d2.is_approximate


# ---------------------------------------------------------------------------
# Global default tests
# ---------------------------------------------------------------------------


class TestGlobalDefaults:

    def test_set_default_num_evaluations(self):
        old = dist_mod.DEFAULT_NUM_EVALUATIONS
        try:
            set_default_num_evaluations(512)
            assert dist_mod.DEFAULT_NUM_EVALUATIONS == 512
        finally:
            dist_mod.DEFAULT_NUM_EVALUATIONS = old

    def test_set_default_invalid(self):
        with pytest.raises(ValueError):
            set_default_num_evaluations(0)

    def test_set_return_approx_dist(self):
        old = dist_mod.RETURN_APPROX_DIST
        try:
            set_return_approx_dist(False)
            assert dist_mod.RETURN_APPROX_DIST is False
            d = Normal(loc=0.0, scale=1.0)
            result = expectation(d, lambda x: x, num_evaluations=100)
            assert isinstance(result, jnp.ndarray)
        finally:
            dist_mod.RETURN_APPROX_DIST = old
