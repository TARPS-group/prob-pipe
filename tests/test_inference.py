"""Tests for the probpipe.inference package.

Covers:
- ApproximateDistribution: chain access, warmup, inference_data, draws
- rwmh workflow function: basic sampling with SupportsLogProb
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApproximateDistribution,
    Normal,
    mean,
    sample,
    variance,
)
from unittest.mock import MagicMock

from probpipe.distributions.multivariate import MultivariateNormal
from probpipe.inference import rwmh


# ---------------------------------------------------------------------------
# ApproximateDistribution
# ---------------------------------------------------------------------------


class TestApproximateDistribution:
    """Test chain-structured empirical distribution."""

    @pytest.fixture
    def two_chain_dist(self):
        """Two chains, 50 draws each, 2D event."""
        chain1 = jax.random.normal(jax.random.PRNGKey(0), (50, 2))
        chain2 = jax.random.normal(jax.random.PRNGKey(1), (50, 2))
        warmup1 = jax.random.normal(jax.random.PRNGKey(2), (10, 2))
        warmup2 = jax.random.normal(jax.random.PRNGKey(3), (10, 2))
        return ApproximateDistribution(
            [chain1, chain2],
            algorithm="test",
            warmup_samples=[warmup1, warmup2],
            name="test_posterior",
        )

    def test_empty_chains_raises(self):
        with pytest.raises(ValueError, match="at least one chain"):
            ApproximateDistribution([])

    def test_num_chains(self, two_chain_dist):
        assert two_chain_dist.num_chains == 2

    def test_num_draws(self, two_chain_dist):
        assert two_chain_dist.num_draws == 50

    def test_event_shape(self, two_chain_dist):
        assert two_chain_dist.event_shape == (2,)

    def test_total_samples(self, two_chain_dist):
        assert two_chain_dist.n == 100  # 50 * 2 chains

    def test_algorithm(self, two_chain_dist):
        assert two_chain_dist.algorithm == "test"

    def test_warmup_samples(self, two_chain_dist):
        assert two_chain_dist.warmup_samples is not None
        assert len(two_chain_dist.warmup_samples) == 2
        assert two_chain_dist.warmup_samples[0].shape == (10, 2)

    def test_draws_single_chain(self, two_chain_dist):
        d = two_chain_dist.draws(chain=0)
        assert d.shape == (50, 2)

    def test_draws_all_chains(self, two_chain_dist):
        d = two_chain_dist.draws()
        assert d.shape == (100, 2)

    def test_draws_with_warmup(self, two_chain_dist):
        d = two_chain_dist.draws(chain=0, include_warmup=True)
        assert d.shape == (60, 2)  # 10 warmup + 50 draws

    def test_draws_all_with_warmup(self, two_chain_dist):
        d = two_chain_dist.draws(include_warmup=True)
        assert d.shape == (120, 2)  # (10 + 50) * 2

    def test_mean_and_variance(self, two_chain_dist):
        m = mean(two_chain_dist)
        v = variance(two_chain_dist)
        assert m.shape == (2,)
        assert v.shape == (2,)
        assert jnp.all(jnp.isfinite(m))
        assert jnp.all(jnp.isfinite(v))

    def test_sample(self, two_chain_dist):
        key = jax.random.PRNGKey(42)
        s = sample(two_chain_dist, key=key)
        assert s.shape == (2,)

    def test_repr(self, two_chain_dist):
        r = repr(two_chain_dist)
        assert "ApproximateDistribution" in r
        assert "num_chains=2" in r
        assert "num_draws=50" in r
        assert "test" in r

    def test_without_warmup(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = ApproximateDistribution([chain])
        assert dist.warmup_samples is None
        assert dist.num_chains == 1
        assert dist.num_draws == 20

    def test_inference_data_default_none(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = ApproximateDistribution([chain])
        assert dist.inference_data is None

    def test_inference_data_stored(self):
        import arviz as az
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        idata = az.from_dict({"posterior": {"params": np.random.randn(1, 20, 3)}})
        dist = ApproximateDistribution(
            [chain], inference_data=idata,
        )
        assert dist.inference_data is idata
        assert hasattr(dist.inference_data, "posterior")

    def test_algorithm_default(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = ApproximateDistribution([chain])
        assert dist.algorithm == "unknown"


# ---------------------------------------------------------------------------
# rwmh workflow function
# ---------------------------------------------------------------------------


class TestRWMH:
    """Test the standalone rwmh workflow function."""

    def test_basic_sampling(self):
        """RWMH samples from a simple Normal distribution."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        result = rwmh(
            dist=dist,
            num_results=100,
            num_warmup=50,
            step_size=0.5,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)
        assert result.num_draws == 100
        assert result.num_chains == 1
        assert result.event_shape == (2,)
        assert result.algorithm == "rwmh"

    def test_inference_data_produced(self):
        """RWMH produces InferenceData with posterior and sample_stats."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=20,
            step_size=0.5,
            random_seed=42,
        )
        assert result.inference_data is not None
        assert hasattr(result.inference_data, "posterior")
        assert hasattr(result.inference_data, "sample_stats")
        assert "acceptance_rate" in result.inference_data.sample_stats

    def test_multi_chain(self):
        """RWMH with multiple chains."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=20,
            num_chains=3,
            step_size=0.5,
            random_seed=42,
        )
        assert result.num_chains == 3
        assert result.num_draws == 50
        assert result.n == 150  # 50 * 3

    def test_warmup_stored(self):
        """RWMH stores warmup samples."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=20,
            step_size=0.5,
            random_seed=42,
        )
        assert result.warmup_samples is not None
        assert result.warmup_samples[0].shape == (20, 2)

    def test_provenance(self):
        """RWMH attaches provenance."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=20,
            step_size=0.5,
            random_seed=42,
        )
        assert result.source is not None
        assert result.source.operation == "rwmh"

    def test_with_log_prob_fn_normal_normal_conjugate(self):
        """RWMH posterior must recover the analytical Normal-Normal conjugate.

        Prior: params ~ N(0, sigma_p^2 I).
        Likelihood: y_i ~ N(params, sigma_y^2 I), i.i.d.
        Closed-form posterior:
            mean = (sigma_y^2 * 0 + n * sigma_p^2 * y_bar) / (sigma_y^2 + n * sigma_p^2)
            var  = (sigma_p^2 * sigma_y^2) / (sigma_y^2 + n * sigma_p^2)
        """
        sigma_p = np.sqrt(10.0)  # prior std
        sigma_y = 1.0            # likelihood std
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=sigma_p ** 2 * jnp.eye(2))
        data = jnp.array([[1.0, 2.0], [1.5, 2.5], [0.8, 1.8]])
        n = data.shape[0]

        def log_lik(params, data):
            return -0.5 / sigma_y ** 2 * jnp.sum((data - params) ** 2)

        result = rwmh(
            dist=prior,
            data=data,
            log_prob_fn=log_lik,
            num_results=8000,
            num_warmup=2000,
            step_size=0.3,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)

        # Analytical posterior.
        y_bar = np.asarray(jnp.mean(data, axis=0))
        denom = sigma_y ** 2 + n * sigma_p ** 2
        analytical_mean = (n * sigma_p ** 2 / denom) * y_bar
        analytical_var = (sigma_p ** 2 * sigma_y ** 2) / denom

        draws = np.asarray(result.draws()).reshape(-1, 2)
        np.testing.assert_allclose(draws.mean(0), analytical_mean, atol=0.08)
        np.testing.assert_allclose(draws.var(0, ddof=1), [analytical_var] * 2, atol=0.1)

    def test_requires_log_prob(self):
        """RWMH raises for distributions without SupportsLogProb and no conversion path."""
        from probpipe import ArrayDistribution

        class NoLogProbNoSample(ArrayDistribution):
            @property
            def event_shape(self):
                return (2,)

        dist = NoLogProbNoSample()
        with pytest.raises(TypeError):
            rwmh(dist=dist, num_results=10, num_warmup=5)

    def test_custom_init(self):
        """RWMH with custom initial state."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=20,
            step_size=0.5,
            init=jnp.array([5.0, 5.0]),
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)

    def test_zero_warmup(self):
        """RWMH with num_warmup=0 stores no warmup samples."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=0,
            step_size=0.5,
            random_seed=42,
        )
        assert result.warmup_samples is None
        assert result.num_draws == 50

    def test_non_supports_mean_init(self):
        """RWMH falls back to zeros init when dist has no SupportsMean."""
        from probpipe import ArrayDistribution
        from probpipe.core.protocols import SupportsLogProb

        class LogProbOnlyDist(ArrayDistribution, SupportsLogProb):
            @property
            def event_shape(self):
                return (2,)

            def _log_prob(self, value):
                return -0.5 * jnp.sum(value ** 2)

            def _prob(self, value):
                return jnp.exp(self._log_prob(value))

            def _unnormalized_log_prob(self, value):
                return self._log_prob(value)

            def _unnormalized_prob(self, value):
                return self._prob(value)

        dist = LogProbOnlyDist()
        result = rwmh(
            dist=dist,
            num_results=30,
            num_warmup=10,
            step_size=0.5,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)

    def test_mean_exception_fallback(self):
        """RWMH falls back to zeros init when _mean() raises."""
        from probpipe import ArrayDistribution
        from probpipe.core.protocols import SupportsLogProb, SupportsMean

        class BrokenMeanLogProbDist(ArrayDistribution, SupportsLogProb, SupportsMean):
            @property
            def event_shape(self):
                return (2,)

            def _log_prob(self, value):
                return -0.5 * jnp.sum(value ** 2)

            def _prob(self, value):
                return jnp.exp(self._log_prob(value))

            def _unnormalized_log_prob(self, value):
                return self._log_prob(value)

            def _unnormalized_prob(self, value):
                return self._prob(value)

            def _mean(self):
                raise RuntimeError("broken")

        dist = BrokenMeanLogProbDist()
        result = rwmh(
            dist=dist,
            num_results=30,
            num_warmup=10,
            step_size=0.5,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)
