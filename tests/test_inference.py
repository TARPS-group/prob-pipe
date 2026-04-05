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
    InferenceDiagnostics,
    Normal,
    mean,
    sample,
    variance,
)
from unittest.mock import MagicMock

from probpipe.distributions.multivariate import MultivariateNormal
from probpipe.inference import rwmh, extract_arviz_diagnostics


# ---------------------------------------------------------------------------
# InferenceDiagnostics
# ---------------------------------------------------------------------------


class TestInferenceDiagnostics:
    """Test InferenceDiagnostics dict-like container."""

    def test_accept_rate_from_is_accepted(self):
        diag = InferenceDiagnostics(
            algorithm="test",
            log_accept_ratio=jnp.zeros(10),
            step_size=0.1,
            is_accepted=jnp.array([True, True, False, True, True,
                                    False, True, True, True, False]),
        )
        np.testing.assert_allclose(diag.accept_rate, 0.7, atol=1e-5)

    def test_accept_rate_from_log_ratio(self):
        # All accepted (log_accept_ratio = 0 means ratio = 1)
        diag = InferenceDiagnostics(
            algorithm="test",
            log_accept_ratio=jnp.zeros(10),
            step_size=0.1,
        )
        np.testing.assert_allclose(diag.accept_rate, 1.0, atol=1e-5)

    def test_accept_rate_override(self):
        diag = InferenceDiagnostics(
            algorithm="test",
            log_accept_ratio=jnp.zeros(10),
            step_size=0.1,
        )
        diag["_accept_rate_override"] = 0.42
        np.testing.assert_allclose(diag.accept_rate, 0.42)

    def test_accept_rate_nan_when_missing(self):
        diag = InferenceDiagnostics(algorithm="test")
        assert jnp.isnan(diag.accept_rate)

    def test_final_step_size(self):
        diag = InferenceDiagnostics(
            algorithm="test",
            log_accept_ratio=jnp.zeros(10),
            step_size=jnp.array([0.1, 0.2, 0.3]),
        )
        np.testing.assert_allclose(diag.final_step_size, 0.2, atol=1e-5)

    def test_final_step_size_nan_when_missing(self):
        diag = InferenceDiagnostics(algorithm="test")
        assert jnp.isnan(diag.final_step_size)

    def test_summary(self):
        diag = InferenceDiagnostics(
            algorithm="nuts",
            log_accept_ratio=jnp.zeros(5),
            step_size=0.1,
        )
        s = diag.summary()
        assert "nuts" in s
        assert "accept_rate" in s
        assert "final_step_size" in s

    # -- dict-style access ---------------------------------------------------

    def test_getitem_setitem(self):
        diag = InferenceDiagnostics(algorithm="test")
        diag["diverging"] = jnp.array([False, True, False, False, True])
        assert "diverging" in diag
        assert diag["diverging"].shape == (5,)

    def test_get_default(self):
        diag = InferenceDiagnostics(algorithm="test")
        assert diag.get("missing") is None
        assert diag.get("missing", 42) == 42

    def test_keys_values_items(self):
        diag = InferenceDiagnostics(algorithm="test", a=1, b=2)
        assert set(diag.keys()) == {"a", "b"}
        assert list(diag.values()) == [1, 2]
        assert dict(diag.items()) == {"a": 1, "b": 2}

    def test_iter(self):
        diag = InferenceDiagnostics(algorithm="test", x=10, y=20)
        assert set(diag) == {"x", "y"}

    def test_len(self):
        diag = InferenceDiagnostics(algorithm="test", a=1, b=2, c=3)
        assert len(diag) == 3

    def test_missing_key_raises(self):
        diag = InferenceDiagnostics(algorithm="test")
        with pytest.raises(KeyError, match="No diagnostic named"):
            diag["nonexistent"]

    def test_kwargs_in_constructor(self):
        diag = InferenceDiagnostics(
            algorithm="test",
            n_divergences=5,
            tree_depth=jnp.array([3, 4, 5]),
        )
        assert diag["n_divergences"] == 5
        assert diag["tree_depth"].shape == (3,)

    def test_summary_includes_extras(self):
        diag = InferenceDiagnostics(
            algorithm="nuts",
            log_accept_ratio=jnp.zeros(3),
            step_size=0.1,
            n_divergences=2,
            tree_depth=jnp.array([3, 4, 5]),
        )
        s = diag.summary()
        assert "n_divergences=2" in s
        assert "tree_depth=Array(3,)" in s

    def test_instances_independent(self):
        """Separate instances don't share state."""
        d1 = InferenceDiagnostics(algorithm="test", log_accept_ratio=jnp.zeros(1))
        d2 = InferenceDiagnostics(algorithm="test", log_accept_ratio=jnp.zeros(1))
        d1["foo"] = 1
        assert "foo" not in d2

    def test_constructor_stores_all_in_dict(self):
        """All kwargs (including log_accept_ratio, step_size) are in the dict."""
        diag = InferenceDiagnostics(
            algorithm="test",
            log_accept_ratio=jnp.zeros(5),
            step_size=0.1,
        )
        assert "log_accept_ratio" in diag
        assert "step_size" in diag
        assert diag["step_size"] == 0.1

    def test_repr(self):
        diag = InferenceDiagnostics(algorithm="nuts", step_size=0.1)
        r = repr(diag)
        assert "InferenceDiagnostics" in r
        assert "nuts" in r


# ---------------------------------------------------------------------------
# extract_arviz_diagnostics
# ---------------------------------------------------------------------------


def _make_arviz_sample_stats(num_chains=2, num_draws=20, *, fields=None):
    """Build a mock ArviZ sample_stats with xarray-like .values access."""
    all_fields = {
        "acceptance_rate": np.random.uniform(0.5, 1.0, (num_chains, num_draws)),
        "step_size": np.full((num_chains, num_draws), 0.05),
        "diverging": np.zeros((num_chains, num_draws), dtype=bool),
        "tree_depth": np.random.randint(1, 8, (num_chains, num_draws)),
        "n_steps": np.random.randint(1, 128, (num_chains, num_draws)),
        "energy": np.random.randn(num_chains, num_draws),
        "energy_error": np.random.randn(num_chains, num_draws) * 0.01,
        "lp": np.random.randn(num_chains, num_draws),
    }
    if fields is not None:
        all_fields = {k: v for k, v in all_fields.items() if k in fields}

    stats = MagicMock()
    stats.__contains__ = lambda self, k: k in all_fields
    stats.__getitem__ = lambda self, k: MagicMock(values=all_fields[k])
    return stats


class TestExtractArvizDiagnostics:
    """Test the shared ArviZ diagnostic extraction."""

    def test_full_extraction(self):
        trace = MagicMock()
        trace.sample_stats = _make_arviz_sample_stats(num_chains=2, num_draws=10)

        diag = extract_arviz_diagnostics(trace, "test_nuts", 10, 2)

        assert diag.algorithm == "test_nuts"
        assert diag["log_accept_ratio"].shape == (20,)
        assert diag["step_size"].shape == (20,)
        assert 0.0 < diag.accept_rate <= 1.0
        assert "diverging" in diag
        assert diag["n_divergences"] == 0
        assert "tree_depth" in diag
        assert "n_steps" in diag
        assert "energy" in diag
        assert "energy_error" in diag
        assert "lp" in diag

    def test_no_sample_stats(self):
        trace = MagicMock(spec=["posterior"])

        diag = extract_arviz_diagnostics(trace, "test", 5, 2)

        assert diag.algorithm == "test"
        assert diag["log_accept_ratio"].shape == (10,)
        # Only log_accept_ratio and step_size (fallback)
        assert "diverging" not in diag

    def test_partial_stats(self):
        trace = MagicMock()
        trace.sample_stats = _make_arviz_sample_stats(
            num_chains=1, num_draws=5,
            fields={"acceptance_rate", "diverging"},
        )

        diag = extract_arviz_diagnostics(trace, "partial", 5, 1)

        assert "diverging" in diag
        assert "tree_depth" not in diag
        assert "step_size" not in diag  # not in fields

    def test_mean_tree_accept_fallback(self):
        """PyMC uses 'mean_tree_accept' instead of 'acceptance_rate'."""
        stats = MagicMock()
        ar = np.random.uniform(0.7, 1.0, (1, 10))
        fields = {"mean_tree_accept": ar}
        stats.__contains__ = lambda self, k: k in fields
        stats.__getitem__ = lambda self, k: MagicMock(values=fields[k])

        trace = MagicMock()
        trace.sample_stats = stats

        diag = extract_arviz_diagnostics(trace, "pymc_nuts", 10, 1)

        assert diag["log_accept_ratio"].shape == (10,)
        assert 0.0 < diag.accept_rate <= 1.0

    def test_step_size_bar_fallback(self):
        """CmdStanPy may use 'step_size_bar'."""
        stats = MagicMock()
        fields = {"step_size_bar": np.full((1, 5), 0.1)}
        stats.__contains__ = lambda self, k: k in fields
        stats.__getitem__ = lambda self, k: MagicMock(values=fields[k])

        trace = MagicMock()
        trace.sample_stats = stats

        diag = extract_arviz_diagnostics(trace, "test", 5, 1)

        np.testing.assert_allclose(diag.final_step_size, 0.1, atol=1e-5)

    def test_divergence_count(self):
        stats = MagicMock()
        div = np.zeros((1, 10), dtype=bool)
        div[0, [1, 4, 7]] = True
        fields = {"diverging": div}
        stats.__contains__ = lambda self, k: k in fields
        stats.__getitem__ = lambda self, k: MagicMock(values=fields[k])

        trace = MagicMock()
        trace.sample_stats = stats

        diag = extract_arviz_diagnostics(trace, "test", 10, 1)

        assert diag["n_divergences"] == 3


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
        diag = InferenceDiagnostics(
            algorithm="test",
            log_accept_ratio=jnp.zeros(100),
            step_size=0.1,
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

    def test_inference_data_default_none(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = MCMCApproximateDistribution([chain])
        assert dist.inference_data is None

    def test_inference_data_stored(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        sentinel = {"posterior": "mock_inference_data"}
        dist = MCMCApproximateDistribution(
            [chain], inference_data=sentinel,
        )
        assert dist.inference_data is sentinel


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
        assert result.inference_data is None

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
