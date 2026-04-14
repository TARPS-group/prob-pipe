"""Tests for probpipe.validation module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import Normal, MultivariateNormal, GLMLikelihood, predictive_check
from probpipe.core.distribution import EmpiricalDistribution
from probpipe.validation import predictive_check as pc_direct
from probpipe.validation._predictive_check import (
    _predictive_check_batched,
    _predictive_check_loop,
    _supports_key_arg,
)


# ---------------------------------------------------------------------------
# Helper: JAX-based generative likelihood
# ---------------------------------------------------------------------------

class PoissonLikelihood:
    """Poisson regression: y_i ~ Poisson(exp(beta_0 + beta_1 * x_i))."""

    def __init__(self, x):
        self._x = jnp.asarray(x, dtype=jnp.float32)

    def log_likelihood(self, params, data):
        log_rate = params[0] + params[1] * self._x
        rate = jnp.exp(log_rate)
        return jnp.sum(data * log_rate - rate)

    def generate_data(self, params, n_samples):
        log_rate = params[0] + params[1] * self._x[:n_samples]
        rate = jnp.exp(log_rate)
        return jax.random.poisson(jax.random.PRNGKey(0), rate)


# ---------------------------------------------------------------------------
# Helper: non-JAX generative likelihood (plain numpy/Python)
# ---------------------------------------------------------------------------

class NumpyGaussianLikelihood:
    """Gaussian likelihood using only numpy — no JAX dependency in data gen."""

    def __init__(self, rng_seed=0):
        self._rng = np.random.default_rng(rng_seed)

    def log_likelihood(self, params, data):
        mu = float(params)
        return -0.5 * np.sum((np.asarray(data) - mu) ** 2)

    def generate_data(self, params, n_samples):
        mu = float(params)
        return self._rng.normal(loc=mu, scale=1.0, size=n_samples)


# ---------------------------------------------------------------------------
# Helper: non-numeric generative likelihood (lists of strings)
# ---------------------------------------------------------------------------

class CategoricalLikelihood:
    """Generative likelihood that produces lists of category labels."""

    _categories = ["cat", "dog", "fish"]

    def __init__(self, rng_seed=0):
        self._rng = np.random.default_rng(rng_seed)

    def log_likelihood(self, params, data):
        return 0.0  # dummy

    def generate_data(self, params, n_samples):
        # params is an array of 3 probabilities
        p = np.asarray(params[:3], dtype=np.float64)
        p = np.abs(p)
        p = p / p.sum()
        return list(self._rng.choice(self._categories, size=n_samples, p=p))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prior():
    return MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="beta")


@pytest.fixture
def likelihood():
    x = jnp.linspace(-1, 1, 20)
    return PoissonLikelihood(x)


@pytest.fixture
def observed_data():
    return jnp.array([2, 1, 3, 0, 5, 1, 2, 4, 3, 1,
                       0, 2, 6, 1, 3, 2, 0, 4, 1, 3])


# ---------------------------------------------------------------------------
# Tests — JAX-based (existing)
# ---------------------------------------------------------------------------

class TestPredictiveCheck:
    """Tests for the predictive_check WorkflowFunction."""

    def test_prior_check_returns_replicated_statistics(self, prior, likelihood):
        result = predictive_check(
            prior, likelihood, test_fn=lambda d: float(jnp.mean(d)),
            n_samples=20, n_replications=50, key=jax.random.PRNGKey(0),
        )
        assert "replicated_statistics" in result
        assert result["replicated_statistics"].n == 50
        assert "observed_statistic" not in result
        assert "p_value" not in result

    def test_posterior_check_returns_p_value(
        self, prior, likelihood, observed_data
    ):
        result = predictive_check(
            prior, likelihood,
            test_fn=lambda d: float(jnp.mean(d)),
            observed_data=observed_data,
            n_replications=100,
            key=jax.random.PRNGKey(1),
        )
        assert "replicated_statistics" in result
        assert "observed_statistic" in result
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["replicated_statistics"].n == 100

    def test_n_samples_inferred_from_observed(
        self, prior, likelihood, observed_data
    ):
        """When observed_data is provided, n_samples defaults to len(observed_data)."""
        result = predictive_check(
            prior, likelihood,
            test_fn=lambda d: float(jnp.var(d)),
            observed_data=observed_data,
            n_replications=20,
            key=jax.random.PRNGKey(2),
        )
        assert result["replicated_statistics"].n == 20

    def test_n_samples_required_without_observed(self, prior, likelihood):
        with pytest.raises(ValueError, match="n_samples is required"):
            predictive_check(
                prior, likelihood,
                test_fn=lambda d: float(jnp.mean(d)),
                n_replications=10,
            )

    def test_is_workflow_function(self):
        from probpipe.core.node import WorkflowFunction
        assert isinstance(predictive_check, WorkflowFunction)

    def test_importable_from_top_level(self):
        from probpipe import predictive_check as pc
        assert callable(pc)

    def test_importable_from_subpackage(self):
        assert callable(pc_direct)

    def test_results_attached_to_distribution(self, prior, likelihood):
        """predictive_check appends results to distribution.validation_results."""
        assert len(prior.validation_results) == 0

        predictive_check(
            prior, likelihood,
            test_fn=lambda d: float(jnp.mean(d)),
            n_samples=20, n_replications=10,
            key=jax.random.PRNGKey(10),
        )
        assert len(prior.validation_results) == 1
        assert "replicated_statistics" in prior.validation_results[0]
        assert "test_fn_name" in prior.validation_results[0]

    def test_multiple_checks_accumulate(self, prior, likelihood, observed_data):
        """Multiple predictive_check calls accumulate on the distribution."""
        n_before = len(prior.validation_results)

        predictive_check(
            prior, likelihood,
            test_fn=lambda d: float(jnp.mean(d)),
            observed_data=observed_data,
            n_replications=10,
            key=jax.random.PRNGKey(20),
        )
        predictive_check(
            prior, likelihood,
            test_fn=lambda d: float(jnp.var(d)),
            observed_data=observed_data,
            n_replications=10,
            key=jax.random.PRNGKey(21),
        )
        assert len(prior.validation_results) == n_before + 2
        assert prior.validation_results[-1]["p_value"] is not None

    def test_test_fn_name_captured(self, prior, likelihood):
        def my_custom_stat(data):
            return float(jnp.max(data))

        predictive_check(
            prior, likelihood,
            test_fn=my_custom_stat,
            n_samples=20, n_replications=10,
            key=jax.random.PRNGKey(30),
        )
        assert prior.validation_results[-1]["test_fn_name"] == "my_custom_stat"


# ---------------------------------------------------------------------------
# Tests — non-JAX data types
# ---------------------------------------------------------------------------

class TestPredictiveCheckNonJax:
    """predictive_check should work with non-JAX data types."""

    def test_numpy_data(self):
        """Likelihood that generates numpy arrays (not JAX arrays)."""
        prior = Normal(loc=0.0, scale=2.0, name="mu")
        lik = NumpyGaussianLikelihood(rng_seed=42)
        observed = np.array([1.2, 0.8, 1.5, 0.3, 1.1])

        result = predictive_check(
            prior, lik,
            test_fn=lambda d: float(np.mean(d)),
            observed_data=observed,
            n_replications=100,
            key=jax.random.PRNGKey(0),
        )
        assert "replicated_statistics" in result
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["replicated_statistics"].n == 100

    def test_numpy_prior_check(self):
        """Prior check with numpy-based likelihood."""
        prior = Normal(loc=0.0, scale=1.0, name="mu")
        lik = NumpyGaussianLikelihood(rng_seed=7)

        result = predictive_check(
            prior, lik,
            test_fn=lambda d: float(np.std(d)),
            n_samples=50,
            n_replications=30,
            key=jax.random.PRNGKey(1),
        )
        assert result["replicated_statistics"].n == 30
        assert "p_value" not in result

    def test_categorical_string_data(self):
        """Likelihood that generates lists of strings."""
        prior = MultivariateNormal(
            loc=jnp.array([1.0, 1.0, 1.0]),
            cov=0.1 * jnp.eye(3),
            name="logits",
        )
        lik = CategoricalLikelihood(rng_seed=99)
        observed = ["cat", "dog", "cat", "fish", "cat",
                     "dog", "cat", "cat", "fish", "cat"]

        def cat_fraction(data):
            return sum(1 for x in data if x == "cat") / len(data)

        result = predictive_check(
            prior, lik,
            test_fn=cat_fraction,
            observed_data=observed,
            n_replications=50,
            key=jax.random.PRNGKey(2),
        )
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["replicated_statistics"].n == 50

    def test_empirical_distribution_as_source(self):
        """Use an EmpiricalDistribution (non-parametric) as the source."""
        samples = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
        dist = EmpiricalDistribution(samples)
        lik = NumpyGaussianLikelihood(rng_seed=11)

        result = predictive_check(
            dist, lik,
            test_fn=lambda d: float(np.mean(d)),
            n_samples=10,
            n_replications=20,
            key=jax.random.PRNGKey(3),
        )
        assert result["replicated_statistics"].n == 20


# ---------------------------------------------------------------------------
# Tests — batched fast path
# ---------------------------------------------------------------------------

class TestPredictiveCheckBatched:
    """Tests for the vectorized (batched) predictive check path."""

    @pytest.fixture
    def glm_setup(self):
        x = jnp.linspace(-1, 1, 20)
        X = jnp.column_stack([jnp.ones_like(x), x])
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="beta")
        lik = GLMLikelihood(tfp_glm.Poisson(), X)
        return prior, lik

    def test_supports_key_arg_glm(self, glm_setup):
        _, lik = glm_setup
        assert _supports_key_arg(lik)

    def test_supports_key_arg_plain(self, likelihood):
        """PoissonLikelihood (no key arg) → False."""
        assert not _supports_key_arg(likelihood)

    def test_batched_path_used_for_glm(self, glm_setup):
        """GLMLikelihood triggers the batched path and produces correct results."""
        prior, lik = glm_setup
        result = predictive_check(
            prior, lik,
            test_fn=lambda d: jnp.mean(d),
            n_samples=20,
            n_replications=50,
            key=jax.random.PRNGKey(42),
        )
        assert result["replicated_statistics"].n == 50

    def test_batched_posterior_check(self, glm_setup):
        """Batched path works with observed data and returns a p-value."""
        prior, lik = glm_setup
        observed = jnp.ones(20, dtype=jnp.float32) * 2
        result = predictive_check(
            prior, lik,
            test_fn=lambda d: jnp.mean(d),
            observed_data=observed,
            n_replications=100,
            key=jax.random.PRNGKey(7),
        )
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_vmap_fallback(self, glm_setup):
        """When test_fn is not vmap-able, the batched path falls back to a loop."""
        prior, lik = glm_setup

        def non_vmapable(d):
            # Python control flow that breaks vmap tracing
            val = float(jnp.mean(d))
            if val > 0:
                return val
            return -val

        result = predictive_check(
            prior, lik,
            test_fn=non_vmapable,
            n_samples=20,
            n_replications=30,
            key=jax.random.PRNGKey(99),
        )
        assert result["replicated_statistics"].n == 30

    def test_loop_path_for_plain_likelihood(self, prior, likelihood):
        """PoissonLikelihood (no key arg) uses the loop path."""
        result = predictive_check(
            prior, likelihood,
            test_fn=lambda d: float(jnp.mean(d)),
            n_samples=20,
            n_replications=30,
            key=jax.random.PRNGKey(5),
        )
        assert result["replicated_statistics"].n == 30
