"""Tests for the probpipe.inference package.

Covers:
- ApproximateDistribution: chain access, warmup, inference_data, draws
- ApproximateDistribution with Values template: named draws
- _ValuesDistributionView: component views, select, broadcasting
- rwmh workflow function: basic sampling with SupportsLogProb
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApproximateDistribution,
    MultivariateNormal,
    Normal,
    Values,
    mean,
    sample,
    variance,
)
from unittest.mock import MagicMock

from probpipe.inference import rwmh
from probpipe.inference._approximate_distribution import (
    _ValuesDistributionView,
    make_posterior,
)
from probpipe.inference._tfp_mcmc import _build_mcmc_datatree


# ---------------------------------------------------------------------------
# ApproximateDistribution
# ---------------------------------------------------------------------------


class TestApproximateDistribution:
    """Test chain-structured empirical distribution."""

    @pytest.fixture
    def two_chain_dist(self):
        """Two chains, 50 draws each, 2D event, built via make_posterior."""
        chain1 = jax.random.normal(jax.random.PRNGKey(0), (50, 2))
        chain2 = jax.random.normal(jax.random.PRNGKey(1), (50, 2))
        warmup1 = jax.random.normal(jax.random.PRNGKey(2), (10, 2))
        warmup2 = jax.random.normal(jax.random.PRNGKey(3), (10, 2))
        chains = [chain1, chain2]
        auxiliary = _build_mcmc_datatree(chains, warmup_chains=[warmup1, warmup2])
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        return make_posterior(
            chains, parents=(prior,), algorithm="test", auxiliary=auxiliary,
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

    def test_algorithm_from_provenance(self, two_chain_dist):
        assert two_chain_dist.algorithm == "test"
        assert two_chain_dist.source.metadata["algorithm"] == "test"

    def test_auxiliary_is_datatree(self, two_chain_dist):
        assert two_chain_dist.auxiliary is not None
        assert hasattr(two_chain_dist.auxiliary, "children")

    def test_inference_data_alias(self, two_chain_dist):
        assert two_chain_dist.inference_data is two_chain_dist.auxiliary

    def test_warmup_from_auxiliary(self, two_chain_dist):
        warmup = two_chain_dist.warmup_samples
        assert warmup is not None
        assert len(warmup) == 2
        assert warmup[0].shape == (10, 2)

    def test_draws_single_chain(self, two_chain_dist):
        d = two_chain_dist.draws(chain=0)
        assert d.shape == (50, 2)

    def test_draws_all_chains(self, two_chain_dist):
        d = two_chain_dist.draws()
        assert d.shape == (100, 2)

    def test_draws_with_warmup(self, two_chain_dist):
        d = two_chain_dist.draws(chain=0, include_warmup=True)
        assert d.shape == (60, 2)

    def test_draws_all_with_warmup(self, two_chain_dist):
        d = two_chain_dist.draws(include_warmup=True)
        assert d.shape == (120, 2)

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


class TestApproximateDistributionValuesTemplate:
    """draws() returns named Values when a values_template is provided."""

    @pytest.fixture
    def template(self):
        return Values(r=jnp.array(0.0), K=jnp.array(0.0), phi=jnp.array(0.0))

    @pytest.fixture
    def posterior_with_template(self, template):
        # 3 scalar params → flat draw vectors of size 3
        chain = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3))
        return make_posterior(
            [chain], parents=(prior,), algorithm="test",
            values_template=template,
        )

    def test_draws_returns_values(self, posterior_with_template):
        draws = posterior_with_template.draws()
        assert isinstance(draws, Values)

    def test_draws_has_correct_fields(self, posterior_with_template):
        draws = posterior_with_template.draws()
        assert draws.fields() == ("K", "phi", "r")

    def test_draws_field_shapes(self, posterior_with_template):
        draws = posterior_with_template.draws()
        assert draws.r.shape == (100,)
        assert draws.K.shape == (100,)
        assert draws.phi.shape == (100,)

    def test_draws_values_match_raw(self, posterior_with_template):
        """Named draws must contain the same data as raw flat draws."""
        raw = posterior_with_template.draws()
        # Reconstruct flat from named
        flat = jnp.stack([raw.K, raw.phi, raw.r], axis=-1)  # sorted order
        chain = posterior_with_template.chains[0]
        np.testing.assert_allclose(flat, chain, atol=1e-6)

    def test_draws_single_chain_returns_values(self, posterior_with_template):
        draws = posterior_with_template.draws(chain=0)
        assert isinstance(draws, Values)
        assert draws.r.shape == (100,)

    def test_without_template_returns_array(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (50, 3))
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3))
        post = make_posterior([chain], parents=(prior,), algorithm="test")
        draws = post.draws()
        assert isinstance(draws, jnp.ndarray)
        assert draws.shape == (50, 3)

    def test_values_template_property(self, posterior_with_template, template):
        assert posterior_with_template.values_template is template

    def test_array_shaped_fields(self):
        """Template with non-scalar fields unflattens correctly."""
        template = Values(
            mean=jnp.zeros(3),
            cov=jnp.zeros((2, 2)),
        )
        flat_size = 3 + 4  # 3 + 2*2
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, flat_size))
        prior = MultivariateNormal(loc=jnp.zeros(flat_size), cov=jnp.eye(flat_size))
        post = make_posterior(
            [chain], parents=(prior,), algorithm="test",
            values_template=template,
        )
        draws = post.draws()
        assert draws.mean.shape == (20, 3)
        assert draws.cov.shape == (20, 2, 2)

    def test_draws_with_warmup_and_template(self):
        """draws(include_warmup=True) returns Values when template is set."""
        template = Values(a=jnp.array(0.0), b=jnp.array(0.0))
        chain = jax.random.normal(jax.random.PRNGKey(0), (50, 2))
        warmup = jax.random.normal(jax.random.PRNGKey(1), (10, 2))
        auxiliary = _build_mcmc_datatree([chain], warmup_chains=[warmup])
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        post = make_posterior(
            [chain], parents=(prior,), algorithm="test",
            auxiliary=auxiliary, values_template=template,
        )
        draws = post.draws(include_warmup=True)
        assert isinstance(draws, Values)
        assert draws.a.shape == (60,)  # 10 warmup + 50 draws
        assert draws.b.shape == (60,)

    def test_nested_values_template_unflatten(self):
        """Nested Values template unflattens draws into nested structure."""
        template = Values(
            params=Values(a=jnp.array(0.0), b=jnp.array(0.0)),
            scale=jnp.array(0.0),
        )
        flat_size = 3  # a + b + scale
        chain = jax.random.normal(jax.random.PRNGKey(0), (30, flat_size))
        prior = MultivariateNormal(loc=jnp.zeros(flat_size), cov=jnp.eye(flat_size))
        post = make_posterior(
            [chain], parents=(prior,), algorithm="test",
            values_template=template,
        )
        draws = post.draws()
        assert isinstance(draws, Values)
        assert isinstance(draws.params, Values)
        assert draws.params.a.shape == (30,)
        assert draws.params.b.shape == (30,)
        assert draws.scale.shape == (30,)

    def test_without_warmup(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = ApproximateDistribution([chain])
        assert dist.warmup_samples is None
        assert dist.num_chains == 1
        assert dist.num_draws == 20

    def test_bare_dist_no_auxiliary(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = ApproximateDistribution([chain])
        assert dist.auxiliary is None
        assert dist.inference_data is None

    def test_auxiliary_has_posterior_group(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        auxiliary = _build_mcmc_datatree([chain])
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3))
        post = make_posterior([chain], parents=(prior,), algorithm="test",
                             auxiliary=auxiliary)
        assert "posterior" in post.auxiliary.children

    def test_algorithm_default_without_auxiliary(self):
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
        """RWMH produces auxiliary DataTree with posterior group."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=20,
            step_size=0.5,
            random_seed=42,
        )
        assert result.inference_data is not None
        assert "posterior" in result.inference_data.children
        # RWMH scalar stats (accept_rate, step_size) live in provenance,
        # not as per-draw arrays in sample_stats.
        assert result.source.metadata["accept_rate"] > 0
        assert result.source.metadata["step_size"] == 0.5

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


# ---------------------------------------------------------------------------
# _ValuesDistributionView + select
# ---------------------------------------------------------------------------


class TestValuesDistributionView:
    """Component views from Values-based posteriors."""

    @pytest.fixture
    def template(self):
        return Values(K=jnp.array(0.0), phi=jnp.array(0.0), r=jnp.array(0.0))

    @pytest.fixture
    def posterior(self, template):
        chain = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3))
        return make_posterior(
            [chain], parents=(prior,), algorithm="test",
            values_template=template,
        )

    def test_getitem_returns_view(self, posterior):
        view = posterior["r"]
        assert isinstance(view, _ValuesDistributionView)

    def test_getitem_missing_field_raises(self, posterior):
        with pytest.raises(KeyError, match="nonexistent"):
            posterior["nonexistent"]

    def test_getitem_without_template_raises(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = ApproximateDistribution([chain])
        with pytest.raises(KeyError):
            dist["x"]

    def test_component_names(self, posterior):
        assert posterior.component_names == ("K", "phi", "r")

    def test_component_names_without_template(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = ApproximateDistribution([chain])
        assert dist.component_names == ()

    def test_view_event_shape_scalar(self, posterior):
        view = posterior["r"]
        assert view.event_shape == ()

    def test_view_event_shape_vector(self):
        template = Values(vec=jnp.zeros(5), scalar=jnp.array(0.0))
        chain = jax.random.normal(jax.random.PRNGKey(0), (50, 6))
        prior = MultivariateNormal(loc=jnp.zeros(6), cov=jnp.eye(6))
        post = make_posterior([chain], parents=(prior,), algorithm="test",
                             values_template=template)
        assert post["scalar"].event_shape == ()
        assert post["vec"].event_shape == (5,)

    def test_view_mean(self, posterior):
        view = posterior["K"]
        draws = posterior.draws()
        np.testing.assert_allclose(
            float(view._mean()), float(jnp.mean(draws.K)), atol=1e-5
        )

    def test_view_variance(self, posterior):
        view = posterior["K"]
        draws = posterior.draws()
        np.testing.assert_allclose(
            float(view._variance()), float(jnp.var(draws.K)), atol=1e-5
        )

    def test_view_sample(self, posterior):
        view = posterior["r"]
        s = view._sample(jax.random.PRNGKey(42), (10,))
        assert s.shape == (10,)

    def test_select_positional(self, posterior):
        sel = posterior.select("r", "K")
        assert set(sel.keys()) == {"r", "K"}
        assert all(isinstance(v, _ValuesDistributionView) for v in sel.values())

    def test_select_keyword_remap(self, posterior):
        sel = posterior.select(growth_rate="r")
        assert "growth_rate" in sel
        assert isinstance(sel["growth_rate"], _ValuesDistributionView)

    def test_select_mixed(self, posterior):
        sel = posterior.select("phi", growth_rate="r")
        assert set(sel.keys()) == {"phi", "growth_rate"}

    def test_repr(self, posterior):
        view = posterior["r"]
        r = repr(view)
        assert "ApproximateDistribution" in r
        assert "r" in r


class TestValuesSelect:
    """Values.select() for concrete data."""

    def test_positional(self):
        v = Values(r=1.0, K=70.0, phi=10.0)
        sel = v.select("r", "K")
        assert set(sel.keys()) == {"r", "K"}
        np.testing.assert_allclose(float(sel["r"]), 1.0)
        np.testing.assert_allclose(float(sel["K"]), 70.0)

    def test_keyword_remap(self):
        v = Values(r=1.0, K=70.0)
        sel = v.select(growth_rate="r")
        assert "growth_rate" in sel
        np.testing.assert_allclose(float(sel["growth_rate"]), 1.0)

    def test_mixed(self):
        v = Values(r=1.0, K=70.0, phi=10.0)
        sel = v.select("phi", growth_rate="r")
        assert set(sel.keys()) == {"phi", "growth_rate"}

    def test_missing_field_raises(self):
        v = Values(r=1.0)
        with pytest.raises(KeyError, match="nonexistent"):
            v.select("nonexistent")

    def test_missing_mapping_target_raises(self):
        v = Values(r=1.0)
        with pytest.raises(KeyError, match="z"):
            v.select(x="z")
