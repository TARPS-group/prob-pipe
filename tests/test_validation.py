"""Tests for probpipe.validation module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Normal, MultivariateNormal, predictive_check
from probpipe.validation import predictive_check as pc_direct


# ---------------------------------------------------------------------------
# Helper: a simple generative likelihood that satisfies both protocols
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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prior():
    return MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))


@pytest.fixture
def likelihood():
    x = jnp.linspace(-1, 1, 20)
    return PoissonLikelihood(x)


@pytest.fixture
def observed_data():
    return jnp.array([2, 1, 3, 0, 5, 1, 2, 4, 3, 1,
                       0, 2, 6, 1, 3, 2, 0, 4, 1, 3])


# ---------------------------------------------------------------------------
# Tests
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

    def test_type_error_non_sampling(self, likelihood, observed_data):
        with pytest.raises(TypeError, match="does not support sampling"):
            predictive_check(
                "not_a_distribution", likelihood,
                test_fn=lambda d: float(jnp.mean(d)),
                observed_data=observed_data,
            )

    def test_type_error_non_generative(self, prior, observed_data):
        class BadLikelihood:
            def log_likelihood(self, params, data):
                return 0.0

        with pytest.raises(TypeError, match="GenerativeLikelihood"):
            predictive_check(
                prior, BadLikelihood(),
                test_fn=lambda d: float(jnp.mean(d)),
                observed_data=observed_data,
            )

    def test_is_workflow_function(self):
        from probpipe.core.node import WorkflowFunction
        assert isinstance(predictive_check, WorkflowFunction)

    def test_importable_from_top_level(self):
        from probpipe import predictive_check as pc
        assert callable(pc)

    def test_importable_from_subpackage(self):
        assert callable(pc_direct)
