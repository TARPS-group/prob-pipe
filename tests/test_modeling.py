"""Tests for probpipe.core.modeling — Likelihood, IterativeForecaster."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from probpipe import MultivariateNormal, EmpiricalDistribution, Provenance
from probpipe.core.modeling import (
    GenerativeLikelihood,
    IterativeForecaster,
    Likelihood,
)
from probpipe.core.node import AbstractModule, WorkflowFunction, wf
from probpipe import log_prob, mean, prob
from probpipe.custom_types import ArrayLike


# ---------------------------------------------------------------------------
# Simple likelihood implementations for testing
# ---------------------------------------------------------------------------


class MultivariateNormalLikelihood(Likelihood):
    """Isotropic MultivariateNormal likelihood — JAX-traceable."""

    @wf
    def log_likelihood(self, params, data):
        residuals = data - params[None, :]
        return -0.5 * jnp.sum(residuals ** 2)


class SimpleGenerativeLikelihood(GenerativeLikelihood):
    """Generate data as params + noise."""

    @wf
    def generate_data(self, params, n_samples):
        key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, shape=(n_samples, len(params)))
        return params[None, :] + noise


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


class TestLikelihood:
    def test_is_abstract_module(self):
        assert issubclass(Likelihood, AbstractModule)

    def test_log_likelihood_is_abstract(self):
        with pytest.raises(TypeError):
            Likelihood()

    def test_concrete_likelihood(self):
        lik = MultivariateNormalLikelihood()
        params = jnp.array([1.0, 2.0])
        data = jnp.array([[1.0, 2.0], [1.5, 2.5]])
        ll = lik.log_likelihood(params=params, data=data)
        assert jnp.isfinite(ll)


class TestGenerativeLikelihood:
    def test_is_abstract_module(self):
        assert issubclass(GenerativeLikelihood, AbstractModule)

    def test_concrete_generative(self):
        gen = SimpleGenerativeLikelihood()
        params = jnp.array([1.0, 2.0])
        data = gen.generate_data(params=params, n_samples=10)
        assert data.shape == (10, 2)


# ---------------------------------------------------------------------------
# IterativeForecaster
# ---------------------------------------------------------------------------


def _simple_posterior_fn(prior: MultivariateNormal, likelihood: Likelihood, data: ArrayLike):
    """A simple posterior approximation for testing."""
    # Just return an EmpiricalDistribution near the data mean
    data_mean = jnp.mean(jnp.asarray(data), axis=0)
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, shape=(50, data_mean.shape[0]))
    samples = data_mean[None, :] + noise * 0.1
    return EmpiricalDistribution(samples)


class TestIterativeForecaster:
    @pytest.fixture
    def dim(self):
        return 2

    @pytest.fixture
    def prior(self, dim):
        return MultivariateNormal(loc=jnp.zeros(dim), cov=jnp.eye(dim) * 10.0)

    @pytest.fixture
    def likelihood(self):
        return MultivariateNormalLikelihood()

    @pytest.fixture
    def gen_likelihood(self):
        return SimpleGenerativeLikelihood()

    @pytest.fixture
    def approx_post(self):
        return WorkflowFunction(func=_simple_posterior_fn, name="simple_posterior")

    def test_iterative_update(self, prior, likelihood, gen_likelihood, approx_post, dim):
        forecaster = IterativeForecaster(
            prior=prior,
            likelihood=likelihood,
            generative_likelihood=gen_likelihood,
            approx_post=approx_post,
        )
        assert forecaster.curr_posterior is prior

        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, shape=(10, dim)) + 2.0
        posterior = forecaster.update(data=data)

        assert isinstance(posterior, EmpiricalDistribution)
        assert forecaster.curr_posterior is posterior


# ---------------------------------------------------------------------------
# Distribution coverage gaps
# ---------------------------------------------------------------------------


class TestDistributionCoverageGaps:
    """Cover the few uncovered lines in distribution.py."""

    def test_batch_shape_default(self, key):
        """batch_shape default returns ()."""
        g = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        assert g.batch_shape == ()

    def test_dtype_default(self, key):
        """dtype default returns float32."""
        g = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        assert g.dtype == jnp.float32

    def test_prob_method(self, key):
        """prob() = exp(log_prob())."""
        g = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        x = jnp.ones(2)
        expected = jnp.exp(log_prob(g, x))
        actual = prob(g, x)
        assert jnp.allclose(actual, expected)

    def test_repr_with_batch_shape(self):
        """repr includes name and event_shape."""
        g = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="test")
        r = repr(g)
        assert "test" in r
        assert "event_shape" in r

    def test_empirical_dtype(self):
        """EmpiricalDistribution.dtype returns sample dtype."""
        samples = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
        ed = EmpiricalDistribution(samples)
        assert ed.dtype == jnp.float32
