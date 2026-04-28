"""Tests for the probpipe.inference package.

Covers:
- ApproximateDistribution: chain access, warmup, inference_data, draws
- ApproximateDistribution with Record template: named draws
- _RecordDistributionView: component views, select, broadcasting
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
    ProductDistribution,
    Record,
    RecordArray,
    RecordTemplate,
    mean,
    sample,
    variance,
)
from unittest.mock import MagicMock

from probpipe.inference import rwmh
from probpipe.core.distribution import _RecordDistributionView
from probpipe.inference._approximate_distribution import make_posterior
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
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
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

    def test_auxiliary_is_inference_data(self, two_chain_dist):
        assert two_chain_dist.auxiliary is not None
        # arviz InferenceData (0.x has .groups(), 1.x DataTree has .children)
        assert hasattr(two_chain_dist.auxiliary, "groups") or hasattr(two_chain_dist.auxiliary, "children")

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
    """draws() returns named Record when a record_template is provided."""

    @pytest.fixture
    def template(self):
        return RecordTemplate(r=(), K=(), phi=())

    @pytest.fixture
    def posterior_with_template(self, template):
        # 3 scalar params → flat draw vectors of size 3
        chain = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="z")
        return make_posterior(
            [chain], parents=(prior,), algorithm="test",
            record_template=template,
        )

    def test_draws_returns_values(self, posterior_with_template):
        draws = posterior_with_template.draws()
        assert isinstance(draws, (Record, RecordArray))
        # Insertion order from the template fixture: r, K, phi.
        assert draws.fields == ("r", "K", "phi")
        assert draws["r"].shape == (100,)

    def test_draws_has_correct_fields(self, posterior_with_template):
        draws = posterior_with_template.draws()
        assert draws.fields == ("r", "K", "phi")

    def test_draws_field_shapes(self, posterior_with_template):
        draws = posterior_with_template.draws()
        assert draws["r"].shape == (100,)
        assert draws["K"].shape == (100,)
        assert draws["phi"].shape == (100,)

    def test_draws_values_match_raw(self, posterior_with_template):
        """Named draws must contain the same data as raw flat draws."""
        raw = posterior_with_template.draws()
        # Reconstruct flat from named (template insertion order).
        flat = jnp.stack([raw["r"], raw["K"], raw["phi"]], axis=-1)
        chain = posterior_with_template.chains[0]
        np.testing.assert_allclose(flat, chain, atol=1e-6)

    def test_draws_single_chain_returns_values(self, posterior_with_template):
        draws = posterior_with_template.draws(chain=0)
        assert isinstance(draws, (Record, RecordArray))
        assert draws["r"].shape == (100,)

    def test_without_template_returns_array(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (50, 3))
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="z")
        post = make_posterior([chain], parents=(prior,), algorithm="test")
        draws = post.draws()
        assert isinstance(draws, jnp.ndarray)
        assert draws.shape == (50, 3)

    def test_record_template_property(self, posterior_with_template, template):
        assert posterior_with_template.record_template is template

    def test_array_shaped_fields(self):
        """Template with non-scalar fields unflattens correctly."""
        template = RecordTemplate(
            mean=(3,),
            cov=(2, 2),
        )
        flat_size = 3 + 4  # 3 + 2*2
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, flat_size))
        prior = MultivariateNormal(loc=jnp.zeros(flat_size), cov=jnp.eye(flat_size), name="z")
        post = make_posterior(
            [chain], parents=(prior,), algorithm="test",
            record_template=template,
        )
        draws = post.draws()
        assert draws["mean"].shape == (20, 3)
        assert draws["cov"].shape == (20, 2, 2)

    def test_draws_with_warmup_and_template(self):
        """draws(include_warmup=True) returns Record when template is set."""
        template = RecordTemplate(a=(), b=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (50, 2))
        warmup = jax.random.normal(jax.random.PRNGKey(1), (10, 2))
        auxiliary = _build_mcmc_datatree([chain], warmup_chains=[warmup])
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
        post = make_posterior(
            [chain], parents=(prior,), algorithm="test",
            auxiliary=auxiliary, record_template=template,
        )
        draws = post.draws(include_warmup=True)
        assert isinstance(draws, (Record, RecordArray))
        assert draws["a"].shape == (60,)  # 10 warmup + 50 draws
        assert draws["b"].shape == (60,)

    def test_nested_record_template_unflatten(self):
        """Nested Record template unflattens draws into nested structure."""
        template = RecordTemplate(
            params=RecordTemplate(a=(), b=()),
            scale=(),
        )
        flat_size = 3  # a + b + scale
        chain = jax.random.normal(jax.random.PRNGKey(0), (30, flat_size))
        prior = MultivariateNormal(loc=jnp.zeros(flat_size), cov=jnp.eye(flat_size), name="z")
        post = make_posterior(
            [chain], parents=(prior,), algorithm="test",
            record_template=template,
        )
        draws = post.draws()
        assert isinstance(draws, (Record, RecordArray))
        assert isinstance(draws["params"], (Record, RecordArray))
        assert draws["params"]["a"].shape == (30,)
        assert draws["params"]["b"].shape == (30,)
        assert draws["scale"].shape == (30,)

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
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="z")
        post = make_posterior([chain], parents=(prior,), algorithm="test",
                             auxiliary=auxiliary)
        assert "posterior" in post.auxiliary

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
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
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
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=20,
            step_size=0.5,
            random_seed=42,
        )
        assert result.inference_data is not None
        assert "posterior" in result.inference_data
        # RWMH scalar stats (accept_rate, step_size) live in provenance,
        # not as per-draw arrays in sample_stats.
        assert result.source.metadata["accept_rate"] > 0
        assert result.source.metadata["step_size"] == 0.5

    def test_multi_chain(self):
        """RWMH with multiple chains."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
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
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
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
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
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
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=sigma_p ** 2 * jnp.eye(2), name="params")
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

        raw_draws = result.draws()
        if hasattr(raw_draws, 'fields'):
            raw_draws = jnp.concatenate(
                [raw_draws[f] for f in raw_draws.fields], axis=-1
            )
        draws = np.asarray(raw_draws).reshape(-1, 2)
        # MC standard error: posterior_sd / sqrt(effective_n).
        # RWMH on this 2D target with step_size=0.3 has heavy autocorrelation,
        # so effective_n << 8000. Assume n_eff ~ 300 conservatively.
        n_eff = 300
        mc_se_mean = 4.0 * np.sqrt(analytical_var / n_eff)
        np.testing.assert_allclose(draws.mean(0), analytical_mean, atol=mc_se_mean)
        mc_se_var = 4.0 * np.sqrt(2.0 * analytical_var ** 2 / (n_eff - 1))
        np.testing.assert_allclose(draws.var(0, ddof=1), [analytical_var] * 2, atol=mc_se_var)

    def test_requires_log_prob(self):
        """RWMH raises for distributions without SupportsLogProb and no conversion path."""
        from probpipe import NumericRecordDistribution

        class NoLogProbNoSample(NumericRecordDistribution):
            @property
            def event_shape(self):
                return (2,)

        dist = NoLogProbNoSample(name="test")
        with pytest.raises(TypeError):
            rwmh(dist=dist, num_results=10, num_warmup=5)

    def test_custom_init(self):
        """RWMH with custom initial state."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
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
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
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
        from probpipe import NumericRecordDistribution
        from probpipe.core.protocols import SupportsLogProb

        class LogProbOnlyDist(NumericRecordDistribution, SupportsLogProb):
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

        dist = LogProbOnlyDist(name="test")
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
        from probpipe import NumericRecordDistribution
        from probpipe.core.protocols import SupportsLogProb, SupportsMean

        class BrokenMeanLogProbDist(NumericRecordDistribution, SupportsLogProb, SupportsMean):
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

        dist = BrokenMeanLogProbDist(name="test")
        result = rwmh(
            dist=dist,
            num_results=30,
            num_warmup=10,
            step_size=0.5,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)


# ---------------------------------------------------------------------------
# _RecordDistributionView + select
# ---------------------------------------------------------------------------


class TestRecordDistributionView:
    """Component views from Record-based posteriors."""

    @pytest.fixture
    def template(self):
        return RecordTemplate(K=(), phi=(), r=())

    @pytest.fixture
    def posterior(self, template):
        chain = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="z")
        return make_posterior(
            [chain], parents=(prior,), algorithm="test",
            record_template=template,
        )

    def test_getitem_returns_view(self, posterior):
        view = posterior["r"]
        assert isinstance(view, _RecordDistributionView)

    def test_getitem_missing_field_raises(self, posterior):
        with pytest.raises(KeyError, match="nonexistent"):
            posterior["nonexistent"]

    def test_getitem_without_template_raises(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = ApproximateDistribution([chain])
        with pytest.raises(KeyError):
            dist["x"]

    def test_fields(self, posterior):
        assert posterior.fields == ("K", "phi", "r")

    def test_fields_without_template(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = ApproximateDistribution([chain])
        assert dist.fields == ()

    def test_view_event_shape_scalar(self, posterior):
        view = posterior["r"]
        assert view.event_shape == ()

    def test_view_event_shape_vector(self):
        template = RecordTemplate(vec=(5,), scalar=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (50, 6))
        prior = MultivariateNormal(loc=jnp.zeros(6), cov=jnp.eye(6), name="z")
        post = make_posterior([chain], parents=(prior,), algorithm="test",
                             record_template=template)
        assert post["scalar"].event_shape == ()
        assert post["vec"].event_shape == (5,)

    def test_view_mean(self, posterior):
        view = posterior["K"]
        draws = posterior.draws()
        np.testing.assert_allclose(
            float(view._mean()), float(jnp.mean(draws["K"])), atol=1e-5
        )

    def test_view_variance(self, posterior):
        view = posterior["K"]
        draws = posterior.draws()
        np.testing.assert_allclose(
            float(view._variance()), float(jnp.var(draws["K"])), atol=1e-5
        )

    def test_view_sample(self, posterior):
        view = posterior["r"]
        s = view._sample(jax.random.PRNGKey(42), (10,))
        assert s.shape == (10,)

    def test_select_positional(self, posterior):
        sel = posterior.select("r", "K")
        assert set(sel.keys()) == {"r", "K"}
        assert all(isinstance(v, _RecordDistributionView) for v in sel.values())

    def test_select_keyword_remap(self, posterior):
        sel = posterior.select(growth_rate="r")
        assert "growth_rate" in sel
        assert isinstance(sel["growth_rate"], _RecordDistributionView)

    def test_select_mixed(self, posterior):
        sel = posterior.select("phi", growth_rate="r")
        assert set(sel.keys()) == {"phi", "growth_rate"}

    def test_repr(self, posterior):
        view = posterior["r"]
        r = repr(view)
        assert "ApproximateDistribution" in r
        assert "r" in r

    def test_view_mean_fallback_without_supports_mean(self):
        """_mean() falls back to _field_draws() when parent lacks SupportsMean."""
        # ApproximateDistribution IS SupportsMean, so we test the fallback
        # by checking the empirical mean matches the draws directly.
        template = RecordTemplate(a=(), b=())
        chain = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
        post = make_posterior([chain], parents=(prior,), algorithm="test",
                             record_template=template)
        view = post["a"]
        # Mean of column 0 (field "a"): (1+3+5)/3 = 3.0
        np.testing.assert_allclose(float(view._mean()), 3.0, atol=1e-5)
        # Variance of column 0: var([1,3,5]) = 8/3
        np.testing.assert_allclose(float(view._variance()), jnp.var(chain[:, 0]), atol=1e-5)

    def test_view_name_matches_field(self, posterior):
        """View.name should return the field name."""
        assert posterior["K"].name == "K"
        assert posterior["phi"].name == "phi"
        assert posterior["r"].name == "r"

    def test_view_name_from_product(self):
        """View.name works on ProductDistribution views."""
        p = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=0.0, scale=1.0, name="y"),
        )
        assert p["x"].name == "x"
        assert p["y"].name == "y"


class TestViewProtocolDuckTyping:
    """_RecordDistributionView dynamically inherits protocol support from its parent.

    When the parent supports SupportsLogProb, the view's dynamic subclass
    also inherits SupportsLogProb — so isinstance checks work correctly.
    """

    def test_view_from_product_isinstance_log_prob(self):
        """ProductDistribution supports SupportsLogProb → so does view."""
        from probpipe import ProductDistribution, SupportsLogProb
        joint = ProductDistribution(x=Normal(0, 1, name="x"), y=Normal(3, 2, name="y"))
        view = joint["x"]
        assert isinstance(view, SupportsLogProb)

    def test_view_from_posterior_not_isinstance_log_prob(self):
        """ApproximateDistribution lacks SupportsLogProb → view doesn't have it."""
        from probpipe import SupportsLogProb
        template = RecordTemplate(a=(), b=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (50, 2))
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
        post = make_posterior([chain], parents=(prior,), algorithm="test",
                             record_template=template)
        view = post["a"]
        assert not isinstance(view, SupportsLogProb)

    def test_view_always_isinstance_sampling(self):
        """Every view is SupportsSampling regardless of parent type."""
        from probpipe import SupportsSampling, ProductDistribution
        joint = ProductDistribution(x=Normal(0, 1, name="x"), y=Normal(3, 2, name="y"))
        assert isinstance(joint["x"], SupportsSampling)

        template = RecordTemplate(a=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 1))
        prior = Normal(0, 1, name="x")
        post = make_posterior([chain], parents=(prior,), algorithm="test",
                             record_template=template)
        assert isinstance(post["a"], SupportsSampling)

    def test_view_always_isinstance_mean_variance(self):
        """Every view is SupportsMean and SupportsVariance."""
        from probpipe import SupportsMean, SupportsVariance, ProductDistribution
        joint = ProductDistribution(x=Normal(0, 1, name="x"), y=Normal(3, 2, name="y"))
        view = joint["x"]
        assert isinstance(view, SupportsMean)
        assert isinstance(view, SupportsVariance)

    def test_view_log_prob_delegates_to_component(self):
        """View _log_prob delegates to the underlying component distribution."""
        from probpipe import ProductDistribution
        import scipy.stats
        joint = ProductDistribution(x=Normal(loc=2.0, scale=0.5, name="x"), y=Normal(0, 1, name="y"))
        view = joint["x"]
        lp = float(view._log_prob(jnp.array(2.0)))
        expected = scipy.stats.norm.logpdf(2.0, loc=2.0, scale=0.5)
        np.testing.assert_allclose(lp, expected, rtol=1e-5)

    def test_view_no_cov_when_parent_lacks_it(self):
        """View lacks SupportsCovariance when parent doesn't have it."""
        from probpipe import SupportsCovariance, ProductDistribution
        joint = ProductDistribution(x=Normal(0, 1, name="x"), y=Normal(3, 2, name="y"))
        view = joint["x"]
        assert not isinstance(view, SupportsCovariance)

    def test_dynamic_protocol_depends_on_parent(self):
        """Same _RecordDistributionView base, different isinstance results."""
        from probpipe import SupportsLogProb, ProductDistribution
        # ProductDistribution parent → isinstance True
        joint = ProductDistribution(x=Normal(0, 1, name="x"), y=Normal(3, 2, name="y"))
        view_with = joint["x"]
        assert isinstance(view_with, SupportsLogProb)

        # ApproximateDistribution parent → isinstance False
        template = RecordTemplate(a=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 1))
        prior = Normal(0, 1, name="x")
        post = make_posterior([chain], parents=(prior,), algorithm="test",
                             record_template=template)
        view_without = post["a"]
        assert not isinstance(view_without, SupportsLogProb)

    def test_view_still_isinstance_base_class(self):
        """Dynamic subclass is still isinstance of _RecordDistributionView."""
        from probpipe import ProductDistribution
        joint = ProductDistribution(x=Normal(0, 1, name="x"), y=Normal(3, 2, name="y"))
        view = joint["x"]
        assert isinstance(view, _RecordDistributionView)


class TestRecordDistributionProperties:
    """RecordDistribution base class properties on ApproximateDistribution."""

    @pytest.fixture
    def template(self):
        return RecordTemplate(K=(), phi=(), r=())

    @pytest.fixture
    def posterior(self, template):
        chain = jax.random.normal(jax.random.PRNGKey(0), (50, 3))
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="z")
        return make_posterior(
            [chain], parents=(prior,), algorithm="test",
            record_template=template,
        )

    def test_record_distribution_flatten_unflatten(self, posterior):
        """RecordDistribution.flatten_value / unflatten_value round-trip."""
        from probpipe.core._record_distribution import RecordDistribution
        v = Record(K=jnp.array(1.0), phi=jnp.array(2.0), r=jnp.array(3.0))
        flat = RecordDistribution.flatten_value(posterior, v)
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])  # insertion: K, phi, r
        v2 = RecordDistribution.unflatten_value(posterior, flat)
        assert isinstance(v2, Record)
        np.testing.assert_allclose(float(v2["K"]), 1.0)
        np.testing.assert_allclose(float(v2["r"]), 3.0)

    def test_flatten_unflatten_roundtrip(self, posterior, template):
        from probpipe.core._record_distribution import RecordDistribution
        v = Record(K=jnp.array(1.0), phi=jnp.array(2.0), r=jnp.array(3.0))
        flat = RecordDistribution.flatten_value(posterior, v)
        assert flat.shape == (3,)
        v2 = RecordDistribution.unflatten_value(posterior, flat)
        assert isinstance(v2, Record)
        np.testing.assert_allclose(float(v2["K"]), 1.0)
        np.testing.assert_allclose(float(v2["r"]), 3.0)

    def test_unflatten_without_template_raises(self):
        from probpipe.core._record_distribution import RecordDistribution
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = ApproximateDistribution([chain])
        with pytest.raises(RuntimeError, match="record_template"):
            RecordDistribution.unflatten_value(dist, jnp.zeros(3))

    def test_record_distribution_event_shapes(self, posterior):
        """RecordDistribution.event_shapes returns per-field dict."""
        from probpipe.core._record_distribution import RecordDistribution
        shapes = RecordDistribution.event_shapes.fget(posterior)
        assert shapes == {"K": (), "phi": (), "r": ()}

    def test_record_distribution_event_size(self, posterior, template):
        """RecordDistribution.event_size matches template.flat_size."""
        from probpipe.core._record_distribution import RecordDistribution
        assert RecordDistribution.event_size.fget(posterior) == template.flat_size


class TestValuesSelect:
    """Record.select() for concrete data."""

    def test_positional(self):
        v = Record(r=1.0, K=70.0, phi=10.0)
        sel = v.select("r", "K")
        assert set(sel.keys()) == {"r", "K"}
        np.testing.assert_allclose(float(sel["r"]), 1.0)
        np.testing.assert_allclose(float(sel["K"]), 70.0)

    def test_keyword_remap(self):
        v = Record(r=1.0, K=70.0)
        sel = v.select(growth_rate="r")
        assert "growth_rate" in sel
        np.testing.assert_allclose(float(sel["growth_rate"]), 1.0)

    def test_mixed(self):
        v = Record(r=1.0, K=70.0, phi=10.0)
        sel = v.select("phi", growth_rate="r")
        assert set(sel.keys()) == {"phi", "growth_rate"}

    def test_missing_field_raises(self):
        v = Record(r=1.0)
        with pytest.raises(KeyError, match="nonexistent"):
            v.select("nonexistent")

    def test_missing_mapping_target_raises(self):
        v = Record(r=1.0)
        with pytest.raises(KeyError, match="z"):
            v.select(x="z")

    def test_empty_select(self):
        v = Record(r=1.0, K=70.0)
        sel = v.select()
        assert sel == {}


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEndValuesPipeline:
    """Full pipeline: named prior → inference → named draws → views → broadcasting.

    Validates correctness at every step, not just types and shapes.
    """

    @pytest.fixture
    def posterior(self):
        """Run inference once for all end-to-end tests."""
        prior = MultivariateNormal(
            loc=jnp.zeros(2), cov=jnp.eye(2) * 10, name="params"
        )

        class _Lik:
            def log_likelihood(self, params, data):
                return -0.5 * jnp.sum((data - params) ** 2)

        from probpipe import SimpleModel, condition_on
        model = SimpleModel(prior, _Lik())
        return condition_on(
            model, jnp.array([1.0, 2.0]),
            num_results=500, num_warmup=200, step_size=0.3, random_seed=42,
        )

    def test_template_propagation(self, posterior):
        """record_template flows from named prior through to posterior."""
        tpl = posterior.record_template
        assert tpl is not None
        assert tpl.fields == ("params",)
        assert tpl["params"] == (2,)

    def test_draws_are_named_values(self, posterior):
        """draws() returns Record with correct field names and shapes."""
        draws = posterior.draws()
        assert isinstance(draws, (Record, RecordArray))
        assert draws.fields == ("params",)
        assert draws["params"].shape == (500, 2)

    def test_draws_values_correct(self, posterior):
        """Posterior mean and std match analytical conjugate values."""
        # Prior N(0, 10I) + likelihood N(data, I), data=[1,2], n=1
        # Posterior mean = sigma_prior^2 / (sigma_lik^2 + sigma_prior^2) * data
        #                = 10/11 * [1, 2] ≈ [0.909, 1.818]
        # Posterior var  = sigma_prior^2 * sigma_lik^2 / (sigma_lik^2 + sigma_prior^2)
        #                = 10/11 ≈ 0.909
        draws = posterior.draws()
        post_mean = np.asarray(draws["params"].mean(axis=0))
        post_std = np.asarray(draws["params"].std(axis=0))
        analytical_mean = np.array([10 / 11, 20 / 11])
        analytical_std = np.sqrt(10 / 11)
        np.testing.assert_allclose(post_mean, analytical_mean, atol=0.15)
        np.testing.assert_allclose(post_std, analytical_std, atol=0.15)

    def test_view_values_match_draws(self, posterior):
        """View _mean() matches draws and analytical posterior mean."""
        view = posterior["params"]
        assert isinstance(view, _RecordDistributionView)
        assert view.event_shape == (2,)

        # Delegation check: view._mean() == draws().params.mean()
        draws = posterior.draws()
        np.testing.assert_allclose(
            np.asarray(view._mean()),
            np.asarray(draws["params"].mean(axis=0)),
            atol=1e-5,
        )
        # Analytical check: view._mean() near analytical posterior mean
        np.testing.assert_allclose(
            np.asarray(view._mean()),
            np.array([10 / 11, 20 / 11]),
            atol=0.15,
        )

    def test_select_returns_views(self, posterior):
        """select() returns dict of views matching component names."""
        sel = posterior.select("params")
        assert set(sel.keys()) == {"params"}
        assert isinstance(sel["params"], _RecordDistributionView)

    def test_workflow_broadcasting_values_correct(self, posterior):
        """Broadcast predict(params, x) computes correct function of posterior."""
        from probpipe.core.node import workflow_function

        @workflow_function(n_broadcast_samples=100, vectorize="loop", seed=0)
        def predict(params, x):
            return params[0] + params[1] * x

        result = predict(**posterior.select("params"), x=0.5)
        assert result.n == 100
        # predict([~0.91, ~1.82], 0.5) ≈ 0.91 + 1.82*0.5 ≈ 1.82
        analytical = 10 / 11 + 0.5 * 20 / 11
        np.testing.assert_allclose(float(mean(result)), analytical, atol=0.2)

    def test_workflow_broadcasting_preserves_correlation(self, posterior):
        """Two views from same posterior sample jointly (not independently).

        Both mean AND variance of a-b must be ~0.  An independent-sampling
        bug would produce mean≈0 (by symmetry) but var≈2*var(params),
        so checking var is the real correlation test.
        """
        from probpipe.core.node import workflow_function

        @workflow_function(n_broadcast_samples=50, vectorize="loop", seed=0)
        def identity_pair(a, b):
            return a - b

        sel = posterior.select(a="params", b="params")
        result = identity_pair(**sel)
        # Mean check: necessary but insufficient
        np.testing.assert_allclose(np.asarray(mean(result)), 0.0, atol=1e-5)
        # Variance check: this is what actually validates correlation
        np.testing.assert_allclose(np.asarray(variance(result)), 0.0, atol=1e-5)

    def test_multi_field_posterior(self):
        """Posterior with multiple named scalar fields."""
        template = RecordTemplate(a=(), b=(), c=())
        # 3 scalar fields → flat draw vectors of size 3
        chain = jax.random.normal(jax.random.PRNGKey(0), (200, 3))
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="z")
        post = make_posterior(
            [chain], parents=(prior,), algorithm="test",
            record_template=template,
        )
        draws = post.draws()
        assert isinstance(draws, (Record, RecordArray))
        assert draws.fields == ("a", "b", "c")
        assert draws["a"].shape == (200,)

        # Per-field views
        view_a = post["a"]
        view_b = post["b"]
        assert isinstance(view_a, _RecordDistributionView)
        np.testing.assert_allclose(
            float(view_a._mean()), float(draws["a"].mean()), atol=1e-5
        )

        # Select multiple fields
        sel = post.select("a", "c")
        assert set(sel.keys()) == {"a", "c"}

    def test_workflow_mixed_posterior_and_independent(self, posterior):
        """Workflow with both posterior views and an independent distribution."""
        from probpipe.core.node import workflow_function

        @workflow_function(n_broadcast_samples=50, vectorize="loop", seed=0)
        def noisy_predict(params, noise):
            return params[0] + params[1] * 0.5 + noise

        result = noisy_predict(
            **posterior.select("params"),
            noise=Normal(0, 0.01, name="noise"),
        )
        assert result.n == 50
        # Mean should be close to predict without noise
        analytical = 10 / 11 + 0.5 * 20 / 11
        np.testing.assert_allclose(float(mean(result)), analytical, atol=0.3)
