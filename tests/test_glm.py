"""Tests for GLMLikelihood."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import GLMLikelihood, MultivariateNormal, SimpleModel, Record, RecordTemplate, condition_on, mean
from probpipe.modeling import Likelihood, GenerativeLikelihood


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture
def poisson_lik():
    X = np.column_stack([np.ones(20), np.linspace(-1, 1, 20)]).astype(np.float32)
    return GLMLikelihood(tfp_glm.Poisson(), X)


@pytest.fixture
def bernoulli_lik():
    X = np.column_stack([np.ones(30), np.linspace(-2, 2, 30)]).astype(np.float32)
    return GLMLikelihood(tfp_glm.Bernoulli(), X)


class TestGLMLikelihood:

    def test_satisfies_likelihood_protocol(self, poisson_lik):
        assert isinstance(poisson_lik, Likelihood)

    def test_satisfies_generative_likelihood_protocol(self, poisson_lik):
        assert isinstance(poisson_lik, GenerativeLikelihood)

    def test_log_likelihood_scalar(self, poisson_lik):
        params = jnp.array([1.0, 0.5])
        data = jnp.ones(20)
        ll = poisson_lik.log_likelihood(params, data)
        assert ll.shape == ()
        assert jnp.isfinite(ll)

    def test_poisson_log_likelihood_matches_scipy(self, poisson_lik):
        """Poisson GLM log-likelihood must match sum of scipy.stats.poisson.logpmf."""
        params = jnp.array([1.0, 0.5])
        data = jnp.array([2, 1, 3, 0, 5, 1, 2, 4, 3, 1,
                          0, 2, 6, 1, 3, 2, 0, 4, 1, 3], dtype=jnp.float32)
        ll = float(poisson_lik.log_likelihood(params, data))

        # Independent baseline: Poisson log link -> rate = exp(X @ params).
        X = np.asarray(poisson_lik._x)
        eta = X @ np.asarray(params)
        rates = np.exp(eta)
        expected = scipy.stats.poisson.logpmf(np.asarray(data), rates).sum()
        np.testing.assert_allclose(ll, expected, rtol=1e-5)

    def test_bernoulli_log_likelihood_matches_scipy(self, bernoulli_lik):
        """Bernoulli GLM log-likelihood must match sum of scipy.stats.bernoulli.logpmf."""
        params = jnp.array([0.2, 1.0])
        rng = np.random.default_rng(0)
        data = rng.binomial(1, 0.5, size=30).astype(np.float32)
        ll = float(bernoulli_lik.log_likelihood(params, jnp.asarray(data)))

        # Independent baseline: Bernoulli logit link -> p = sigmoid(X @ params).
        X = np.asarray(bernoulli_lik._x)
        eta = X @ np.asarray(params)
        p = _sigmoid(eta)
        expected = scipy.stats.bernoulli.logpmf(data, p).sum()
        np.testing.assert_allclose(ll, expected, rtol=1e-5)

    def test_poisson_generate_data_moments(self):
        """Sample mean must approximate exp(intercept) for constant-rate Poisson."""
        n = 2000
        X = np.column_stack([np.ones(n), np.zeros(n)]).astype(np.float32)
        lik = GLMLikelihood(tfp_glm.Poisson(), X, seed=7)
        params = jnp.array([0.5, 0.0])  # constant rate = exp(0.5)
        expected_rate = np.exp(0.5)
        y = np.asarray(lik.generate_data(params, n))
        np.testing.assert_allclose(float(y.mean()), expected_rate, atol=0.15)

    def test_generate_data_shape(self, poisson_lik):
        params = jnp.array([1.0, 0.5])
        y = poisson_lik.generate_data(params, 20)
        assert y.shape == (20,)

    def test_generate_data_different_n(self, poisson_lik):
        params = jnp.array([1.0, 0.5])
        y = poisson_lik.generate_data(params, 10)
        assert y.shape == (10,)

    def test_bernoulli_log_likelihood(self, bernoulli_lik):
        params = jnp.array([0.0, 1.0])
        data = jnp.ones(30, dtype=jnp.float32)
        ll = bernoulli_lik.log_likelihood(params, data)
        assert jnp.isfinite(ll)

    def test_bernoulli_generate_data(self, bernoulli_lik):
        params = jnp.array([0.0, 1.0])
        y = bernoulli_lik.generate_data(params, 30)
        assert y.shape == (30,)

    def test_1d_covariate(self):
        """1-D x should be handled (transposed to column vector)."""
        x = np.linspace(-1, 1, 15).astype(np.float32)
        lik = GLMLikelihood(tfp_glm.Poisson(), x)
        assert lik._x.shape == (15, 1)
        params = jnp.array([0.5])
        ll = lik.log_likelihood(params, jnp.ones(15))
        assert jnp.isfinite(ll)

    def test_negbin(self):
        X = np.column_stack([np.ones(20), np.linspace(-1, 1, 20)]).astype(np.float32)
        lik = GLMLikelihood(tfp_glm.NegativeBinomial(), X)
        params = jnp.array([1.0, 0.3])
        ll = lik.log_likelihood(params, jnp.ones(20))
        assert jnp.isfinite(ll)
        y = lik.generate_data(params, 20)
        assert y.shape == (20,)

    def test_condition_on_with_glm(self, poisson_lik):
        """GLMLikelihood works end-to-end with SimpleModel + condition_on."""
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=5.0 * jnp.eye(2), name="beta")
        model = SimpleModel(prior, poisson_lik)
        data = jnp.array([2, 1, 3, 0, 5, 1, 2, 4, 3, 1,
                           0, 2, 6, 1, 3, 2, 0, 4, 1, 3], dtype=jnp.float32)
        posterior = condition_on(model, data, num_results=100, num_warmup=50, random_seed=0)
        m = mean(posterior)
        assert m.shape == (2,)

    def test_seed_reproducibility(self):
        X = np.column_stack([np.ones(10), np.zeros(10)]).astype(np.float32)
        params = jnp.array([1.0, 0.0])
        lik1 = GLMLikelihood(tfp_glm.Poisson(), X, seed=42)
        lik2 = GLMLikelihood(tfp_glm.Poisson(), X, seed=42)
        y1 = lik1.generate_data(params, 10)
        y2 = lik2.generate_data(params, 10)
        np.testing.assert_array_equal(y1, y2)


class TestGLMLikelihoodDataTemplate:
    """GLMLikelihood.data_template declares data field names."""

    def test_data_template_fields(self, poisson_lik):
        tpl = poisson_lik.data_template
        assert isinstance(tpl, RecordTemplate)
        assert tpl.fields == ("X", "y")

    def test_data_template_integrates_with_simple_model(self, poisson_lik):
        """SimpleModel merges GLM data_template into fields."""
        from probpipe import Normal, ProductDistribution
        prior = ProductDistribution(
            intercept=Normal(loc=0.0, scale=2.0, name="intercept"),
            slope=Normal(loc=0.0, scale=2.0, name="slope"),
        )
        model = SimpleModel(prior, poisson_lik)
        assert "X" in model.fields
        assert "y" in model.fields
        assert "intercept" in model.fields
        assert "slope" in model.fields
        assert model.parameter_names == ("intercept", "slope")


class TestGLMLikelihoodWithValues:
    """GLMLikelihood accepts Record for params and data."""

    def test_log_likelihood_with_values_params(self, poisson_lik):
        params = Record(beta=jnp.array([1.0, 0.5]))
        data = jnp.ones(20)
        ll = poisson_lik.log_likelihood(params, data)
        assert jnp.isfinite(ll)

    def test_log_likelihood_with_record_data(self, poisson_lik):
        params = jnp.array([1.0, 0.5])
        data = Record(y=jnp.ones(20))
        ll = poisson_lik.log_likelihood(params, data)
        assert jnp.isfinite(ll)

    def test_values_matches_raw(self, poisson_lik):
        params_raw = jnp.array([1.0, 0.5])
        data_raw = jnp.ones(20, dtype=jnp.float32)
        ll_raw = float(poisson_lik.log_likelihood(params_raw, data_raw))

        params_v = Record(beta=params_raw)
        data_v = Record(y=data_raw)
        ll_v = float(poisson_lik.log_likelihood(params_v, data_v))
        np.testing.assert_allclose(ll_v, ll_raw, rtol=1e-6)

    def test_coerce_multi_field_values_stacks(self, poisson_lik):
        """Multi-field Record are stacked into a flat vector (sorted field order)."""
        params_v = Record(intercept=jnp.array(1.0), slope=jnp.array(0.5))
        params_raw = jnp.array([1.0, 0.5])  # intercept, slope (sorted)
        data = jnp.ones(20)
        ll_v = float(poisson_lik.log_likelihood(params_v, data))
        ll_raw = float(poisson_lik.log_likelihood(params_raw, data))
        np.testing.assert_allclose(ll_v, ll_raw, rtol=1e-6)

    def test_condition_on_with_record_data(self, poisson_lik):
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=5.0 * jnp.eye(2), name="beta")
        model = SimpleModel(prior, poisson_lik)
        data = Record(y=jnp.array([2, 1, 3, 0, 5, 1, 2, 4, 3, 1,
                                    0, 2, 6, 1, 3, 2, 0, 4, 1, 3],
                                   dtype=jnp.float32))
        posterior = condition_on(model, data, num_results=50, num_warmup=25, random_seed=0)
        assert mean(posterior).shape == (2,)
        # Prior has no record_template, so draws are raw arrays.
        # Named draws require prior._record_template to be set.
        draws = posterior.draws()
        flat = posterior.flatten_value(draws)
        assert flat.shape == (50, 2)


class TestIncrementalConditionerAutoConvert:
    """IncrementalConditioner auto-converts non-SupportsLogProb posteriors."""

    def test_auto_convert_to_kde(self):
        """update() should work without a custom condition_fn."""
        from probpipe.modeling import IncrementalConditioner
        from probpipe.inference import ApproximateDistribution
        from probpipe.core.protocols import SupportsLogProb

        X = np.column_stack([np.ones(20), np.linspace(-1, 1, 20)]).astype(np.float32)
        lik = GLMLikelihood(tfp_glm.Poisson(), X)
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=5.0 * jnp.eye(2), name="beta")
        model = SimpleModel(prior, lik)
        data = jnp.ones(20, dtype=jnp.float32)

        # Get an ApproximateDistribution (does NOT support SupportsLogProb)
        post = condition_on(model, data, num_results=100, num_warmup=50, random_seed=0)
        assert isinstance(post, ApproximateDistribution)
        assert not isinstance(post, SupportsLogProb)

        # IncrementalConditioner should auto-convert when using post as prior
        conditioner = IncrementalConditioner(prior=post, likelihood=lik)
        post2 = conditioner.update(data=data)
        assert post2 is not None
        assert mean(post2).shape == (2,)
