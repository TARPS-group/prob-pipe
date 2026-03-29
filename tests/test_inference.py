"""Tests for the probpipe.inference package.

Covers:
- MCMCDiagnostics: properties, summary
- MCMCApproximateDistribution: chain access, warmup, diagnostics, draws
- rwmh workflow function: basic sampling with SupportsLogProb
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    MCMCApproximateDistribution,
    MCMCDiagnostics,
    Normal,
    mean,
    sample,
    variance,
)
from probpipe.distributions.multivariate import MultivariateNormal
from probpipe.inference import rwmh


# ---------------------------------------------------------------------------
# MCMCDiagnostics
# ---------------------------------------------------------------------------


class TestMCMCDiagnostics:
    """Test diagnostics dataclass properties."""

    def test_accept_rate_from_is_accepted(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(10),
            step_size=0.1,
            is_accepted=jnp.array([True, True, False, True, True,
                                    False, True, True, True, False]),
            algorithm="test",
        )
        np.testing.assert_allclose(diag.accept_rate, 0.7, atol=1e-5)

    def test_accept_rate_from_log_ratio(self):
        # All accepted (log_accept_ratio = 0 means ratio = 1)
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(10),
            step_size=0.1,
            algorithm="test",
        )
        np.testing.assert_allclose(diag.accept_rate, 1.0, atol=1e-5)

    def test_accept_rate_numpy_override(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(10),
            step_size=0.1,
            algorithm="test",
        )
        diag._numpy_accept_rate = 0.42
        np.testing.assert_allclose(diag.accept_rate, 0.42)

    def test_final_step_size(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(10),
            step_size=jnp.array([0.1, 0.2, 0.3]),
            algorithm="test",
        )
        np.testing.assert_allclose(diag.final_step_size, 0.2, atol=1e-5)

    def test_summary(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(5),
            step_size=0.1,
            algorithm="nuts",
        )
        s = diag.summary()
        assert "nuts" in s
        assert "accept_rate" in s
        assert "final_step_size" in s

    # -- dict-style extra diagnostics ----------------------------------------

    def test_extra_getitem_setitem(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(5),
            step_size=0.1,
        )
        diag["diverging"] = jnp.array([False, True, False, False, True])
        assert "diverging" in diag
        assert diag["diverging"].shape == (5,)

    def test_extra_get_default(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(3),
            step_size=0.1,
        )
        assert diag.get("missing") is None
        assert diag.get("missing", 42) == 42

    def test_extra_keys_values_items(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(3),
            step_size=0.1,
            extra={"a": 1, "b": 2},
        )
        assert set(diag.keys()) == {"a", "b"}
        assert list(diag.values()) == [1, 2]
        assert dict(diag.items()) == {"a": 1, "b": 2}

    def test_extra_iter(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(3),
            step_size=0.1,
            extra={"x": 10, "y": 20},
        )
        assert set(diag) == {"x", "y"}

    def test_extra_missing_key_raises(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(3),
            step_size=0.1,
        )
        with pytest.raises(KeyError, match="No diagnostic named"):
            diag["nonexistent"]

    def test_extra_in_constructor(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(3),
            step_size=0.1,
            extra={"n_divergences": 5, "tree_depth": jnp.array([3, 4, 5])},
        )
        assert diag["n_divergences"] == 5
        assert diag["tree_depth"].shape == (3,)

    def test_summary_includes_extras(self):
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(3),
            step_size=0.1,
            algorithm="nuts",
            extra={"n_divergences": 2, "tree_depth": jnp.array([3, 4, 5])},
        )
        s = diag.summary()
        assert "n_divergences=2" in s
        assert "tree_depth=Array(3,)" in s

    def test_extra_default_empty(self):
        """Extra dict defaults to empty, not shared across instances."""
        d1 = MCMCDiagnostics(log_accept_ratio=jnp.zeros(1), step_size=0.1)
        d2 = MCMCDiagnostics(log_accept_ratio=jnp.zeros(1), step_size=0.1)
        d1["foo"] = 1
        assert "foo" not in d2


# ---------------------------------------------------------------------------
# MCMCApproximateDistribution
# ---------------------------------------------------------------------------


class TestMCMCApproximateDistribution:
    """Test chain-structured empirical distribution."""

    @pytest.fixture
    def two_chain_dist(self):
        """Two chains, 50 draws each, 2D event."""
        chain1 = jax.random.normal(jax.random.PRNGKey(0), (50, 2))
        chain2 = jax.random.normal(jax.random.PRNGKey(1), (50, 2))
        warmup1 = jax.random.normal(jax.random.PRNGKey(2), (10, 2))
        warmup2 = jax.random.normal(jax.random.PRNGKey(3), (10, 2))
        diag = MCMCDiagnostics(
            log_accept_ratio=jnp.zeros(100),
            step_size=0.1,
            algorithm="test",
        )
        return MCMCApproximateDistribution(
            [chain1, chain2],
            diagnostics=diag,
            warmup_samples=[warmup1, warmup2],
            name="test_posterior",
        )

    def test_empty_chains_raises(self):
        with pytest.raises(ValueError, match="at least one chain"):
            MCMCApproximateDistribution([])

    def test_num_chains(self, two_chain_dist):
        assert two_chain_dist.num_chains == 2

    def test_num_draws(self, two_chain_dist):
        assert two_chain_dist.num_draws == 50

    def test_event_shape(self, two_chain_dist):
        assert two_chain_dist.event_shape == (2,)

    def test_total_samples(self, two_chain_dist):
        assert two_chain_dist.n == 100  # 50 * 2 chains

    def test_diagnostics(self, two_chain_dist):
        assert two_chain_dist.diagnostics is not None
        assert two_chain_dist.diagnostics.algorithm == "test"

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
        assert "MCMCApproximateDistribution" in r
        assert "num_chains=2" in r
        assert "num_draws=50" in r

    def test_without_warmup(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = MCMCApproximateDistribution([chain])
        assert dist.warmup_samples is None
        assert dist.num_chains == 1
        assert dist.num_draws == 20

    def test_without_diagnostics(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = MCMCApproximateDistribution([chain])
        assert dist.diagnostics is None
        r = repr(dist)
        assert "MCMCApproximateDistribution" in r


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
        assert isinstance(result, MCMCApproximateDistribution)
        assert result.num_draws == 100
        assert result.num_chains == 1
        assert result.event_shape == (2,)

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

    def test_diagnostics(self):
        """RWMH populates diagnostics."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=20,
            step_size=0.5,
            random_seed=42,
        )
        assert result.diagnostics is not None
        assert result.diagnostics.algorithm == "rwmh"
        assert 0.0 < result.diagnostics.accept_rate <= 1.0

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

    def test_with_log_prob_fn(self):
        """RWMH with external log_prob_fn and data."""
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10)
        data = jnp.array([[1.0, 2.0], [1.5, 2.5], [0.8, 1.8]])

        def log_lik(params, data):
            return -0.5 * jnp.sum((data - params) ** 2)

        result = rwmh(
            dist=prior,
            data=data,
            log_prob_fn=log_lik,
            num_results=100,
            num_warmup=50,
            step_size=0.3,
            random_seed=42,
        )
        assert isinstance(result, MCMCApproximateDistribution)
        # Posterior mean should be pulled toward data mean
        post_mean = jnp.mean(result.draws(), axis=0)
        data_mean = jnp.mean(data, axis=0)
        assert jnp.linalg.norm(post_mean - data_mean) < 2.0

    def test_requires_log_prob(self):
        """RWMH raises for distributions without SupportsLogProb."""
        from probpipe import EmpiricalDistribution

        emp = EmpiricalDistribution(jnp.ones((10, 2)))
        with pytest.raises(TypeError, match="does not support log_prob"):
            rwmh(dist=emp, num_results=10, num_warmup=5)

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
        assert isinstance(result, MCMCApproximateDistribution)

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
        assert isinstance(result, MCMCApproximateDistribution)

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
        assert isinstance(result, MCMCApproximateDistribution)
