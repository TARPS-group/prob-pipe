"""Tests for Distribution.expectation() and is_approximate."""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest

from probpipe import (
    Distribution,
    EmpiricalDistribution,
    Normal,
    Gamma,
    Beta,
    Exponential,
    Bernoulli,
    Categorical,
    Binomial,
    TransformedDistribution,
    DEFAULT_NUM_EVALUATIONS,
    set_default_num_evaluations,
)
import tensorflow_probability.substrates.jax.bijectors as tfb


# ---------------------------------------------------------------------------
# Expectation tests — sample-based (infinite support)
# ---------------------------------------------------------------------------


class TestExpectationSampleBased:
    """Test sample-based expectations on infinite-support distributions."""

    def test_normal_mean(self):
        """E[x] for Normal should approximate loc."""
        d = Normal(loc=3.0, scale=1.0)
        key = jax.random.PRNGKey(0)
        result = d.expectation(lambda x: x, key=key, num_evaluations=10_000)
        np.testing.assert_allclose(float(result), 3.0, atol=0.1)

    def test_normal_second_moment(self):
        """E[x²] for Normal(loc, scale) = loc² + scale²."""
        loc, scale = 2.0, 1.5
        d = Normal(loc=loc, scale=scale)
        key = jax.random.PRNGKey(1)
        result = d.expectation(lambda x: x ** 2, key=key, num_evaluations=10_000)
        expected = loc ** 2 + scale ** 2
        np.testing.assert_allclose(float(result), expected, atol=0.15)

    def test_normal_variance_from_moments(self):
        """Var[X] = E[X²] - E[X]² should approximate scale²."""
        loc, scale = 1.0, 2.0
        d = Normal(loc=loc, scale=scale)
        key1, key2 = jax.random.split(jax.random.PRNGKey(2))
        ex = d.expectation(lambda x: x, key=key1, num_evaluations=10_000)
        ex2 = d.expectation(lambda x: x ** 2, key=key2, num_evaluations=10_000)
        var_est = float(ex2) - float(ex) ** 2
        np.testing.assert_allclose(var_est, scale ** 2, atol=0.3)

    def test_gamma_mean(self):
        """E[x] for Gamma(concentration, rate) = concentration / rate."""
        conc, rate = 3.0, 2.0
        d = Gamma(concentration=conc, rate=rate)
        key = jax.random.PRNGKey(3)
        result = d.expectation(lambda x: x, key=key, num_evaluations=10_000)
        np.testing.assert_allclose(float(result), conc / rate, atol=0.1)

    def test_gamma_log_sufficient_statistic(self):
        """E[log x] for Gamma(a, b) = digamma(a) - log(b)."""
        conc, rate = 3.0, 2.0
        d = Gamma(concentration=conc, rate=rate)
        key = jax.random.PRNGKey(4)
        result = d.expectation(lambda x: jnp.log(x), key=key, num_evaluations=20_000)
        expected = float(jsp.digamma(conc)) - float(jnp.log(rate))
        np.testing.assert_allclose(float(result), expected, atol=0.1)

    def test_beta_mean(self):
        """E[x] for Beta(a, b) = a / (a + b)."""
        a, b = 2.0, 5.0
        d = Beta(alpha=a, beta=b)
        key = jax.random.PRNGKey(5)
        result = d.expectation(lambda x: x, key=key, num_evaluations=10_000)
        np.testing.assert_allclose(float(result), a / (a + b), atol=0.05)

    def test_beta_log_sufficient_statistic(self):
        """E[log x] for Beta(a, b) = digamma(a) - digamma(a + b)."""
        a, b = 2.0, 5.0
        d = Beta(alpha=a, beta=b)
        key = jax.random.PRNGKey(6)
        result = d.expectation(lambda x: jnp.log(x), key=key, num_evaluations=20_000)
        expected = float(jsp.digamma(a)) - float(jsp.digamma(a + b))
        np.testing.assert_allclose(float(result), expected, atol=0.1)

    def test_exponential_second_moment(self):
        """E[x²] for Exponential(rate) = 2 / rate²."""
        rate = 3.0
        d = Exponential(rate=rate)
        key = jax.random.PRNGKey(7)
        result = d.expectation(lambda x: x ** 2, key=key, num_evaluations=10_000)
        np.testing.assert_allclose(float(result), 2.0 / rate ** 2, atol=0.05)

    def test_default_num_evaluations(self):
        """Expectation with num_evaluations=None uses DEFAULT_NUM_EVALUATIONS."""
        d = Normal(loc=0.0, scale=1.0)
        # Just verify it runs without error
        result = d.expectation(lambda x: x)
        assert jnp.isfinite(result)

    def test_custom_num_evaluations(self):
        """num_evaluations parameter controls sample count."""
        d = Normal(loc=5.0, scale=0.1)
        key = jax.random.PRNGKey(10)
        # Very few evaluations — should still return a value
        result = d.expectation(lambda x: x, key=key, num_evaluations=10)
        assert jnp.isfinite(result)


# ---------------------------------------------------------------------------
# Expectation tests — exact (finite support)
# ---------------------------------------------------------------------------


class TestExpectationExact:
    """Test exact expectations on finite-support distributions."""

    def test_bernoulli_identity(self):
        """E[x] for Bernoulli(p) = p, computed exactly."""
        p = 0.7
        d = Bernoulli(probs=p)
        result = d.expectation(lambda x: x)
        np.testing.assert_allclose(float(result), p, atol=1e-6)

    def test_bernoulli_square(self):
        """E[x²] for Bernoulli(p) = p (since x ∈ {0,1})."""
        p = 0.3
        d = Bernoulli(probs=p)
        result = d.expectation(lambda x: x ** 2)
        np.testing.assert_allclose(float(result), p, atol=1e-6)

    def test_bernoulli_custom_function(self):
        """E[f(x)] for Bernoulli with a custom function."""
        p = 0.4
        d = Bernoulli(probs=p)
        # f(x) = 2x + 1 → E[f(x)] = (1-p)*1 + p*3 = 1 + 2p
        result = d.expectation(lambda x: 2 * x + 1)
        np.testing.assert_allclose(float(result), 1 + 2 * p, atol=1e-6)

    def test_categorical_identity(self):
        """E[x] for Categorical([0.1, 0.2, 0.3, 0.4]) = 0*0.1 + 1*0.2 + 2*0.3 + 3*0.4."""
        probs = [0.1, 0.2, 0.3, 0.4]
        d = Categorical(probs=probs)
        result = d.expectation(lambda x: x)
        expected = sum(i * p for i, p in enumerate(probs))
        np.testing.assert_allclose(float(result), expected, atol=1e-5)

    def test_categorical_custom_function(self):
        """E[x²] for Categorical."""
        probs = [0.25, 0.5, 0.25]
        d = Categorical(probs=probs)
        result = d.expectation(lambda x: x ** 2)
        expected = 0 * 0.25 + 1 * 0.5 + 4 * 0.25
        np.testing.assert_allclose(float(result), expected, atol=1e-5)

    def test_binomial_mean(self):
        """E[x] for Binomial(n, p) = n * p."""
        n, p = 10, 0.3
        d = Binomial(total_count=n, probs=p)
        result = d.expectation(lambda x: x)
        np.testing.assert_allclose(float(result), n * p, atol=1e-4)

    def test_binomial_second_moment(self):
        """E[x²] for Binomial(n, p) = n*p*(1-p) + (n*p)²."""
        n, p = 10, 0.3
        d = Binomial(total_count=n, probs=p)
        result = d.expectation(lambda x: x ** 2)
        expected = n * p * (1 - p) + (n * p) ** 2
        np.testing.assert_allclose(float(result), expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Expectation tests — EmpiricalDistribution (exact weighted sum)
# ---------------------------------------------------------------------------


class TestExpectationEmpirical:
    """Test exact expectations on EmpiricalDistribution."""

    def test_uniform_mean(self):
        """E[x] on uniform EmpiricalDistribution = sample mean."""
        samples = jnp.array([1.0, 2.0, 3.0, 4.0])
        d = EmpiricalDistribution(samples)
        result = d.expectation(lambda x: x)
        np.testing.assert_allclose(float(result), 2.5, atol=1e-6)

    def test_weighted_mean(self):
        """E[x] with weights."""
        samples = jnp.array([0.0, 10.0])
        weights = jnp.array([0.3, 0.7])
        d = EmpiricalDistribution(samples, weights=weights)
        result = d.expectation(lambda x: x)
        np.testing.assert_allclose(float(result), 7.0, atol=1e-5)

    def test_custom_function(self):
        """E[f(x)] = weighted sum of f over support."""
        samples = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.array([0.2, 0.5, 0.3])
        d = EmpiricalDistribution(samples, weights=weights)
        result = d.expectation(lambda x: x ** 2)
        expected = 0.2 * 1.0 + 0.5 * 4.0 + 0.3 * 9.0
        np.testing.assert_allclose(float(result), expected, atol=1e-5)

    def test_subsample(self):
        """num_evaluations < n triggers subsampling."""
        samples = jnp.arange(100.0)
        d = EmpiricalDistribution(samples)
        key = jax.random.PRNGKey(0)
        result = d.expectation(lambda x: x, key=key, num_evaluations=10)
        # Should still be a finite number, just noisy
        assert jnp.isfinite(result)

    def test_matches_mean_method(self):
        """expectation(identity) should match mean()."""
        samples = jnp.array([1.0, 3.0, 5.0, 7.0])
        d = EmpiricalDistribution(samples)
        ex = d.expectation(lambda x: x)
        np.testing.assert_allclose(float(ex), float(d.mean()), atol=1e-6)


# ---------------------------------------------------------------------------
# is_approximate tests
# ---------------------------------------------------------------------------


class TestIsApproximate:
    """Test is_approximate property and propagation."""

    def test_tfp_distribution_exact(self):
        """TFP distributions are exact by default."""
        assert not Normal(loc=0.0, scale=1.0).is_approximate
        assert not Gamma(concentration=1.0, rate=1.0).is_approximate
        assert not Beta(alpha=1.0, beta=1.0).is_approximate
        assert not Bernoulli(probs=0.5).is_approximate

    def test_empirical_approximate_by_default(self):
        """EmpiricalDistribution is approximate by default."""
        d = EmpiricalDistribution(jnp.array([[1.0], [2.0]]))
        assert d.is_approximate

    def test_transformed_propagates(self):
        """TransformedDistribution inherits is_approximate from base."""
        exact_base = Normal(loc=0.0, scale=1.0)
        t_exact = TransformedDistribution(exact_base, tfb.Exp())
        assert not t_exact.is_approximate

        approx_base = EmpiricalDistribution(jnp.array([[1.0], [2.0]]))
        t_approx = TransformedDistribution(approx_base, tfb.Exp())
        assert t_approx.is_approximate

    def test_from_distribution_same_class_exact(self):
        """Converting Normal → Normal (same class, parameter copy) stays exact."""
        d = Normal(loc=0.0, scale=1.0)
        d2 = Normal.from_distribution(d)
        assert not d2.is_approximate

    def test_from_distribution_different_class_approximate(self):
        """Converting Normal → Gamma (different class, sampling) is approximate."""
        d = Normal(loc=5.0, scale=0.1)  # positive values
        d2 = Gamma.from_distribution(d, check_support=False)
        assert d2.is_approximate

    def test_from_distribution_to_empirical(self):
        """Converting to EmpiricalDistribution is approximate."""
        d = Normal(loc=0.0, scale=1.0)
        d2 = EmpiricalDistribution.from_distribution(d)
        assert d2.is_approximate


# ---------------------------------------------------------------------------
# Global default tests
# ---------------------------------------------------------------------------


class TestGlobalDefault:

    def test_set_default(self):
        """set_default_num_evaluations changes the module-level default."""
        import probpipe.distributions.distribution as mod

        old = mod.DEFAULT_NUM_EVALUATIONS
        try:
            set_default_num_evaluations(512)
            assert mod.DEFAULT_NUM_EVALUATIONS == 512
        finally:
            mod.DEFAULT_NUM_EVALUATIONS = old

    def test_set_default_invalid(self):
        with pytest.raises(ValueError):
            set_default_num_evaluations(0)
