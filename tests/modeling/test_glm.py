"""Tests for GLMLikelihood."""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import (
    EventTemplate,
    GLMLikelihood,
    MultivariateNormal,
    Record,
    SimpleModel,
    condition_on,
    mean,
)
from probpipe.modeling import GenerativeLikelihood, Likelihood


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture
def poisson_lik():
    X = np.asarray(np.linspace(-1, 1, 20))[:, None].astype(float)
    return GLMLikelihood(tfp_glm.Poisson(), X)


@pytest.fixture
def bernoulli_lik():
    X = np.asarray(np.linspace(-2, 2, 30))[:, None].astype(float)
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
        data = jnp.array([2, 1, 3, 0, 5, 1, 2, 4, 3, 1, 0, 2, 6, 1, 3, 2, 0, 4, 1, 3], dtype=float)
        ll = float(poisson_lik.log_likelihood(params, data))

        # Independent baseline: Poisson log link →
        #   rate = exp(intercept + X @ slopes).
        X = np.asarray(poisson_lik._x)
        params_np = np.asarray(params)
        eta = params_np[0] + X @ params_np[1:]
        rates = np.exp(eta)
        expected = scipy.stats.poisson.logpmf(np.asarray(data), rates).sum()
        np.testing.assert_allclose(ll, expected, rtol=1e-5)

    def test_bernoulli_log_likelihood_matches_scipy(self, bernoulli_lik):
        """Bernoulli GLM log-likelihood must match sum of scipy.stats.bernoulli.logpmf."""
        params = jnp.array([0.2, 1.0])
        rng = np.random.default_rng(0)
        data = rng.binomial(1, 0.5, size=30).astype(float)
        ll = float(bernoulli_lik.log_likelihood(params, jnp.asarray(data)))

        # Independent baseline: Bernoulli logit link →
        #   p = sigmoid(intercept + X @ slopes).
        X = np.asarray(bernoulli_lik._x)
        params_np = np.asarray(params)
        eta = params_np[0] + X @ params_np[1:]
        p = _sigmoid(eta)
        expected = scipy.stats.bernoulli.logpmf(data, p).sum()
        np.testing.assert_allclose(ll, expected, rtol=1e-5)

    def test_poisson_generate_data_moments(self):
        """Sample mean must approximate exp(intercept) for constant-rate Poisson.

        For 2000 i.i.d. Poisson(0.5) draws, SE of the sample mean is
        ``sqrt(0.5 / 2000) ≈ 0.016``; ``atol=0.05`` is ~3 SE — tight
        enough to catch a ~10% rate error, loose enough not to flake.
        """
        n = 2000
        X = np.asarray(np.zeros(n))[:, None].astype(float)
        lik = GLMLikelihood(tfp_glm.Poisson(), X, seed=7)
        params = jnp.array([0.5, 0.0])  # constant rate = exp(0.5)
        expected_rate = np.exp(0.5)
        y = np.asarray(lik.generate_data(params, n))
        np.testing.assert_allclose(float(y.mean()), expected_rate, atol=0.05)

    @pytest.mark.parametrize("n", [10, 20])
    def test_generate_data_shape(self, poisson_lik, n):
        params = jnp.array([1.0, 0.5])
        y = poisson_lik.generate_data(params, n)
        assert y.shape == (n,)

    def test_bernoulli_generate_data(self, bernoulli_lik):
        params = jnp.array([0.0, 1.0])
        y = bernoulli_lik.generate_data(params, 30)
        assert y.shape == (30,)

    def test_1d_covariate(self):
        """1-D x is treated as a single-column design matrix.

        Behavioral check: ``log_likelihood`` and ``per_datum_log_likelihood``
        both work on a 1-D covariate input. The reshape that would
        otherwise need a private-attribute assertion (``lik._x.shape ==
        (15, 1)``) is verified indirectly via these public-surface
        evaluations.
        """
        x = np.linspace(-1, 1, 15).astype(float)
        lik = GLMLikelihood(tfp_glm.Poisson(), x)
        params = jnp.array([0.0, 0.5])  # (intercept, slope) under fit_intercept=True
        ll = lik.log_likelihood(params, jnp.ones(15))
        assert jnp.isfinite(ll)
        # Per-datum path with 1-D covariate.
        p_lp = lik.per_datum_log_likelihood(
            params,
            Record(X=jnp.asarray(x[0]), y=jnp.array(1.0)),
        )
        assert jnp.isfinite(p_lp)

    def test_fit_intercept_false_matches_scipy(self):
        """``fit_intercept=False`` ⇒ eta = X @ params (no intercept term).

        Independent baseline: scipy.stats.poisson.logpmf with rates
        ``exp(X @ params)``. Locks in the no-intercept branch of
        ``_linear_predictor`` against a regression in the intercept-skip
        path.
        """
        X = np.asarray(np.linspace(-1, 1, 20))[:, None].astype(float)
        lik = GLMLikelihood(tfp_glm.Poisson(), X, fit_intercept=False)
        params = jnp.array([0.7])  # one slope, no intercept
        data = jnp.array([2, 1, 3, 0, 5, 1, 2, 4, 3, 1, 0, 2, 6, 1, 3, 2, 0, 4, 1, 3], dtype=float)
        ll = float(lik.log_likelihood(params, data))

        params_np = np.asarray(params)
        eta = (X @ params_np).reshape(-1)  # no intercept
        rates = np.exp(eta)
        expected = scipy.stats.poisson.logpmf(np.asarray(data), rates).sum()
        np.testing.assert_allclose(ll, expected, rtol=1e-5)

    def test_negbin(self):
        X = np.asarray(np.linspace(-1, 1, 20))[:, None].astype(float)
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
        data = jnp.array([2, 1, 3, 0, 5, 1, 2, 4, 3, 1, 0, 2, 6, 1, 3, 2, 0, 4, 1, 3], dtype=float)
        posterior = condition_on(model, data, num_results=100, num_warmup=50, random_seed=0)
        m = mean(posterior)
        assert m.shape == (2,)

    def test_seed_reproducibility(self):
        X = np.asarray(np.zeros(10))[:, None].astype(float)
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
        assert isinstance(tpl, EventTemplate)
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
        data_raw = jnp.ones(20, dtype=float)
        ll_raw = float(poisson_lik.log_likelihood(params_raw, data_raw))

        params_v = Record(beta=params_raw)
        data_v = Record(y=data_raw)
        ll_v = float(poisson_lik.log_likelihood(params_v, data_v))
        np.testing.assert_allclose(ll_v, ll_raw, rtol=1e-6)

    def test_coerce_multi_field_values_stacks(self, poisson_lik):
        """Multi-field Record are stacked into a flat vector (insertion order)."""
        params_v = Record(intercept=jnp.array(1.0), slope=jnp.array(0.5))
        params_raw = jnp.array([1.0, 0.5])  # intercept, slope (insertion order)
        data = jnp.ones(20)
        ll_v = float(poisson_lik.log_likelihood(params_v, data))
        ll_raw = float(poisson_lik.log_likelihood(params_raw, data))
        np.testing.assert_allclose(ll_v, ll_raw, rtol=1e-6)

    def test_condition_on_with_record_data(self, poisson_lik):
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=5.0 * jnp.eye(2), name="beta")
        model = SimpleModel(prior, poisson_lik)
        data = Record(
            y=jnp.array([2, 1, 3, 0, 5, 1, 2, 4, 3, 1, 0, 2, 6, 1, 3, 2, 0, 4, 1, 3], dtype=float)
        )
        posterior = condition_on(model, data, num_results=50, num_warmup=25, random_seed=0)
        assert mean(posterior).shape == (2,)
        # Prior has no event_template, so draws are raw arrays.
        # Named draws require prior._event_template to be set.
        draws = posterior.draws()
        flat = posterior.flatten_value(draws, event_shape=posterior.event_shape)
        assert flat.shape == (50, 2)


class TestIncrementalConditionerAutoConvert:
    """IncrementalConditioner auto-converts non-SupportsLogProb posteriors."""

    def test_auto_convert_to_kde(self):
        """update() should work without a custom condition_fn."""
        from probpipe.core.protocols import SupportsLogProb
        from probpipe.inference import ApproximateDistribution
        from probpipe.modeling import IncrementalConditioner

        X = np.asarray(np.linspace(-1, 1, 20))[:, None].astype(float)
        lik = GLMLikelihood(tfp_glm.Poisson(), X)
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=5.0 * jnp.eye(2), name="beta")
        model = SimpleModel(prior, lik)
        data = jnp.ones(20, dtype=float)

        # Get an ApproximateDistribution (does NOT support SupportsLogProb)
        post = condition_on(model, data, num_results=100, num_warmup=50, random_seed=0)
        assert isinstance(post, ApproximateDistribution)
        assert not isinstance(post, SupportsLogProb)

        # IncrementalConditioner should auto-convert when using post as prior
        conditioner = IncrementalConditioner(prior=post, likelihood=lik)
        post2 = conditioner.update(data=data)
        assert post2 is not None
        assert mean(post2).shape == (2,)
