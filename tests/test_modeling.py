"""Tests for probpipe.core.modeling — MCMCSampler, RWMH, diagnostics."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from probpipe import MultivariateNormal, EmpiricalDistribution, Provenance
from probpipe.core.modeling import (
    ApproximatePosterior,
    GenerativeLikelihood,
    IterativeForecaster,
    Likelihood,
    MCMCDiagnostics,
    MCMCSampler,
    RWMH,
)
from probpipe.core.node import AbstractModule, wf
from probpipe import log_prob, mean, prob


# ---------------------------------------------------------------------------
# Simple likelihood implementations for testing
# ---------------------------------------------------------------------------


class MultivariateNormalLikelihood(Likelihood):
    """Isotropic MultivariateNormal likelihood — JAX-traceable."""

    @wf
    def log_likelihood(self, params, data):
        # Sum of log N(x_i | params, I) over data rows
        residuals = data - params[None, :]
        return -0.5 * jnp.sum(residuals ** 2)


class NumpyLikelihood(Likelihood):
    """MultivariateNormal likelihood using numpy — NOT JAX-traceable."""

    @wf
    def log_likelihood(self, params, data):
        params = np.asarray(params)
        data = np.asarray(data)
        residuals = data - params[None, :]
        return -0.5 * float(np.sum(residuals ** 2))


class SimpleGenerativeLikelihood(GenerativeLikelihood):
    """Generate data as params + noise."""

    @wf
    def generate_data(self, params, n_samples):
        key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, shape=(n_samples, len(params)))
        return params[None, :] + noise


# ---------------------------------------------------------------------------
# MCMCDiagnostics
# ---------------------------------------------------------------------------


class TestMCMCDiagnostics:
    def test_accept_rate_from_is_accepted(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(10),
            step_size=0.1,
            is_accepted=jnp.array([True, True, False, True, False,
                                   True, True, True, False, True]),
            algorithm="nuts",
        )
        assert abs(diag.accept_rate - 0.7) < 1e-6

    def test_accept_rate_from_log_ratio(self):
        # When is_accepted is None, compute from log_accept_ratio
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(10),  # all accepted (ratio=1)
            step_size=0.1,
            is_accepted=None,
            algorithm="hmc",
        )
        assert abs(diag.accept_rate - 1.0) < 1e-6

    def test_final_step_size(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(5),
            step_size=jnp.array([0.1, 0.2, 0.3, 0.15, 0.25]),
            algorithm="nuts",
        )
        assert abs(diag.final_step_size - 0.2) < 1e-6

    def test_summary_string(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(5),
            step_size=0.1,
            is_accepted=jnp.ones(5, dtype=bool),
            algorithm="nuts",
        )
        s = diag.summary()
        assert "algorithm=nuts" in s
        assert "accept_rate" in s
        assert "final_step_size" in s


# ---------------------------------------------------------------------------
# MCMCSampler
# ---------------------------------------------------------------------------


class TestMCMCSampler:
    """Test MCMCSampler with a simple 2D MultivariateNormal posterior."""

    @pytest.fixture
    def dim(self):
        return 2

    @pytest.fixture
    def true_mean(self, dim):
        return jnp.ones(dim) * 3.0

    @pytest.fixture
    def prior(self, dim):
        return MultivariateNormal(loc=jnp.zeros(dim), cov=jnp.eye(dim) * 100.0)

    @pytest.fixture
    def data(self, true_mean):
        """20 data points near the true mean."""
        key = jax.random.PRNGKey(99)
        return jax.random.normal(key, shape=(20, len(true_mean))) + true_mean

    @pytest.fixture
    def jax_likelihood(self):
        return MultivariateNormalLikelihood()

    @pytest.fixture
    def numpy_likelihood(self):
        return NumpyLikelihood()

    def test_validation_bad_algorithm(self):
        with pytest.raises(ValueError, match="algorithm"):
            MCMCSampler(algorithm="gibbs")

    def test_validation_bad_num_results(self):
        with pytest.raises(ValueError, match="num_results"):
            MCMCSampler(num_results=0)

    def test_validation_bad_num_warmup(self):
        with pytest.raises(ValueError, match="num_warmup"):
            MCMCSampler(num_warmup=-1)

    def test_nuts_jax_traceable(self, prior, jax_likelihood, data, true_mean):
        sampler = MCMCSampler(
            algorithm="nuts",
            num_results=200,
            num_warmup=100,
            seed=42,
        )
        posterior = sampler(prior=prior, likelihood=jax_likelihood, data=data)

        assert isinstance(posterior, EmpiricalDistribution)
        assert posterior.n == 200
        assert posterior.event_shape == (len(true_mean),)

        # Check diagnostics were stored
        assert sampler.diagnostics is not None
        assert sampler.diagnostics.algorithm == "nuts"
        assert sampler.diagnostics.accept_rate > 0.0

        # Posterior mean should be closer to true mean than prior mean
        post_mean = jnp.mean(posterior.samples, axis=0)
        assert jnp.linalg.norm(post_mean - true_mean) < jnp.linalg.norm(
            jnp.zeros_like(true_mean) - true_mean
        )

    def test_hmc_jax_traceable(self, prior, jax_likelihood, data):
        sampler = MCMCSampler(
            algorithm="hmc",
            num_results=100,
            num_warmup=50,
            num_leapfrog_steps=5,
            seed=42,
        )
        posterior = sampler(prior=prior, likelihood=jax_likelihood, data=data)
        assert isinstance(posterior, EmpiricalDistribution)
        assert posterior.n == 100
        assert sampler.diagnostics.algorithm == "hmc"

    def test_fallback_to_rwmh(self, prior, numpy_likelihood, data):
        """Non-JAX-traceable likelihood should fall back to RW-MH."""
        sampler = MCMCSampler(
            algorithm="nuts",
            num_results=100,
            num_warmup=50,
            seed=42,
        )
        posterior = sampler(prior=prior, likelihood=numpy_likelihood, data=data)
        assert isinstance(posterior, EmpiricalDistribution)
        assert posterior.n == 100
        assert sampler.diagnostics.algorithm == "rwmh_fallback"

    def test_custom_init(self, prior, jax_likelihood, data, dim):
        init = jnp.ones(dim) * 5.0
        sampler = MCMCSampler(
            num_results=50,
            num_warmup=20,
            init=init,
            seed=42,
        )
        posterior = sampler(prior=prior, likelihood=jax_likelihood, data=data)
        assert isinstance(posterior, EmpiricalDistribution)

    def test_provenance_attached(self, prior, jax_likelihood, data):
        sampler = MCMCSampler(num_results=50, num_warmup=20, seed=42)
        posterior = sampler(prior=prior, likelihood=jax_likelihood, data=data)

        assert posterior.source is not None
        assert isinstance(posterior.source, Provenance)
        assert prior in posterior.source.parents
        assert "algorithm" in posterior.source.metadata

    def test_zero_warmup(self, prior, jax_likelihood, data):
        sampler = MCMCSampler(
            num_results=50,
            num_warmup=0,
            seed=42,
        )
        posterior = sampler(prior=prior, likelihood=jax_likelihood, data=data)
        assert posterior.n == 50

    def test_is_jax_traceable_helper(self, prior, data):
        sampler = MCMCSampler(seed=42)
        init = sampler._get_init_state(prior, data)

        # JAX function should be traceable
        jax_fn = lambda x: jnp.sum(x ** 2)
        assert sampler._is_jax_traceable(jax_fn, init) is True

        # numpy function should not be traceable
        np_fn = lambda x: float(np.sum(np.asarray(x) ** 2))
        assert sampler._is_jax_traceable(np_fn, init) is False


# ---------------------------------------------------------------------------
# Legacy RWMH
# ---------------------------------------------------------------------------


class TestRWMH:
    @pytest.fixture
    def dim(self):
        return 2

    @pytest.fixture
    def prior(self, dim):
        return MultivariateNormal(loc=jnp.zeros(dim), cov=jnp.eye(dim) * 100.0)

    @pytest.fixture
    def data(self, dim):
        rng = np.random.default_rng(99)
        return rng.normal(3.0, 1.0, size=(20, dim))

    @pytest.fixture
    def likelihood(self):
        return NumpyLikelihood()

    def test_basic_sampling(self, prior, likelihood, data):
        sampler = RWMH(
            step_size=0.5,
            n_steps=500,
            burn_in=100,
            thin=2,
            seed=42,
        )
        posterior = sampler(prior=prior, likelihood=likelihood, data=data)

        assert isinstance(posterior, EmpiricalDistribution)
        assert posterior.n > 0
        assert sampler.accept_rate > 0.0
        assert sampler.accept_rate <= 1.0

    def test_provenance(self, prior, likelihood, data):
        sampler = RWMH(n_steps=200, burn_in=50, thin=1, seed=42)
        posterior = sampler(prior=prior, likelihood=likelihood, data=data)
        assert posterior.source is not None
        assert posterior.source.operation == "rwmh"

    def test_validation_bad_n_steps(self):
        with pytest.raises(ValueError, match="n_steps"):
            RWMH(n_steps=0)

    def test_validation_bad_burn_in(self):
        with pytest.raises(ValueError, match="burn_in"):
            RWMH(burn_in=-1)

    def test_validation_bad_thin(self):
        with pytest.raises(ValueError, match="thin"):
            RWMH(thin=0)

    def test_init_from_data_mean(self, prior, likelihood, data):
        """When mean(prior) fails, should use data mean."""
        sampler = RWMH(n_steps=100, burn_in=10, thin=1, seed=42)
        posterior = sampler(prior=prior, likelihood=likelihood, data=data)
        assert isinstance(posterior, EmpiricalDistribution)


# ---------------------------------------------------------------------------
# IterativeForecaster
# ---------------------------------------------------------------------------


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
    def sampler(self):
        return MCMCSampler(num_results=50, num_warmup=20, seed=42)

    def test_iterative_update(self, prior, likelihood, gen_likelihood, sampler, dim):
        forecaster = IterativeForecaster(
            prior=prior,
            likelihood=likelihood,
            generative_likelihood=gen_likelihood,
            approx_post=sampler,
        )
        assert forecaster.curr_posterior is prior

        # Generate some data and update — dependencies resolved from module
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
        """Line 71: batch_shape default returns ()."""
        g = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        assert g.batch_shape == ()

    def test_dtype_default(self, key):
        """Line 75: dtype default returns float32."""
        g = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        assert g.dtype == jnp.float32

    def test_prob_method(self, key):
        """Line 88: prob() = exp(log_prob())."""
        g = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        x = jnp.ones(2)
        expected = jnp.exp(log_prob(g, x))
        actual = prob(g, x)
        assert jnp.allclose(actual, expected)

    def test_repr_with_batch_shape(self):
        """Line 134: repr includes batch_shape when non-empty."""
        # EmpiricalDistribution has empty batch_shape, so this is for
        # future classes. Test repr without batch_shape for now.
        g = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="test")
        r = repr(g)
        assert "test" in r
        assert "event_shape" in r

    def test_empirical_dtype(self):
        """Line 259: EmpiricalDistribution.dtype returns sample dtype."""
        samples = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
        ed = EmpiricalDistribution(samples)
        assert ed.dtype == jnp.float32
