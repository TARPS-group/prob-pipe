"""Unit tests for BroadcastDistribution and MarginalizedBroadcastDistribution."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BroadcastDistribution,
    EmpiricalDistribution,
    Normal,
    Provenance,
    SupportsSampling,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsLogProb,
)
from probpipe.core.distribution import (
    _ArrayMarginal,
    _ListMarginal,
    _MixtureMarginal,
    _make_marginal,
    _make_mixture_marginal,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# BroadcastDistribution construction
# ---------------------------------------------------------------------------


class TestBroadcastDistributionConstruction:
    def test_basic_construction(self):
        n = 10
        inputs = {"x": jnp.ones((n, 2))}
        outputs = jnp.zeros((n, 3))
        bd = BroadcastDistribution(
            input_samples=inputs,
            output_samples=outputs,
            weights=None,
            broadcast_args=["x"],
        )
        assert bd.n == n
        assert bd.component_names == ("x", "_output")

    def test_with_weights(self):
        n = 5
        inputs = {"a": jnp.ones((n, 1))}
        outputs = jnp.zeros((n, 1))
        w = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BroadcastDistribution(
            input_samples=inputs,
            output_samples=outputs,
            weights=w,
            broadcast_args=["a"],
        )
        # Weights should be normalised
        np.testing.assert_allclose(float(jnp.sum(bd.weights)), 1.0, atol=1e-5)

    def test_multiple_broadcast_args(self):
        n = 8
        inputs = {"a": jnp.ones((n, 1)), "b": jnp.zeros((n, 2))}
        outputs = jnp.ones((n,))
        bd = BroadcastDistribution(
            input_samples=inputs,
            output_samples=outputs,
            weights=None,
            broadcast_args=["a", "b"],
        )
        assert bd.component_names == ("a", "b", "_output")
        assert "a" in bd.input_samples
        assert "b" in bd.input_samples


# ---------------------------------------------------------------------------
# BroadcastDistribution protocols
# ---------------------------------------------------------------------------


class TestBroadcastDistributionProtocols:
    def test_supports_sampling(self):
        bd = BroadcastDistribution(
            input_samples={"x": jnp.ones((5, 1))},
            output_samples=jnp.zeros((5, 1)),
            weights=None,
            broadcast_args=["x"],
        )
        assert isinstance(bd, SupportsSampling)

    def test_supports_named_components(self):
        bd = BroadcastDistribution(
            input_samples={"x": jnp.ones((5, 1))},
            output_samples=jnp.zeros((5, 1)),
            weights=None,
            broadcast_args=["x"],
        )
        assert hasattr(bd, 'component_names')


# ---------------------------------------------------------------------------
# BroadcastDistribution joint sampling
# ---------------------------------------------------------------------------


class TestBroadcastDistributionSampling:
    def test_joint_sample_structure(self, key):
        n = 10
        inputs = {"a": jnp.arange(n, dtype=jnp.float32).reshape(-1, 1)}
        outputs = jnp.arange(n, dtype=jnp.float32).reshape(-1, 1) * 2
        bd = BroadcastDistribution(
            input_samples=inputs,
            output_samples=outputs,
            weights=None,
            broadcast_args=["a"],
        )
        sample = bd._sample(key, ())
        assert "a" in sample
        assert "_output" in sample

    def test_joint_sample_batch(self, key):
        n = 10
        inputs = {"x": jnp.ones((n, 2))}
        outputs = jnp.zeros((n, 3))
        bd = BroadcastDistribution(
            input_samples=inputs,
            output_samples=outputs,
            weights=None,
            broadcast_args=["x"],
        )
        batch = bd._sample(key, (5,))
        assert batch["x"].shape == (5, 2)
        assert batch["_output"].shape == (5, 3)

    def test_joint_sample_preserves_pairing(self, key):
        """Resampled pairs should match original input–output pairs."""
        n = 3
        inputs = {"x": jnp.array([[1.0], [2.0], [3.0]])}
        outputs = jnp.array([[10.0], [20.0], [30.0]])
        bd = BroadcastDistribution(
            input_samples=inputs,
            output_samples=outputs,
            weights=None,
            broadcast_args=["x"],
        )
        batch = bd._sample(key, (100,))
        # For each resampled pair, output should be 10x input
        np.testing.assert_allclose(batch["_output"], batch["x"] * 10, atol=1e-5)


# ---------------------------------------------------------------------------
# BroadcastDistribution named component access
# ---------------------------------------------------------------------------


class TestBroadcastDistributionComponents:
    def test_getitem_input_returns_empirical(self):
        n = 5
        x_data = jnp.arange(n, dtype=jnp.float32).reshape(-1, 1)
        bd = BroadcastDistribution(
            input_samples={"x": x_data},
            output_samples=jnp.zeros((n, 1)),
            weights=None,
            broadcast_args=["x"],
        )
        x_dist = bd["x"]
        assert isinstance(x_dist, EmpiricalDistribution)
        np.testing.assert_allclose(x_dist.samples, x_data, atol=1e-6)

    def test_getitem_output_returns_marginal(self):
        n = 5
        bd = BroadcastDistribution(
            input_samples={"x": jnp.ones((n, 1))},
            output_samples=jnp.zeros((n, 2)),
            weights=None,
            broadcast_args=["x"],
        )
        out = bd["_output"]
        assert isinstance(out, _ArrayMarginal)

    def test_getitem_invalid_key(self):
        bd = BroadcastDistribution(
            input_samples={"x": jnp.ones((5, 1))},
            output_samples=jnp.zeros((5, 1)),
            weights=None,
            broadcast_args=["x"],
        )
        with pytest.raises(KeyError):
            bd["nonexistent"]


# ---------------------------------------------------------------------------
# BroadcastDistribution provenance
# ---------------------------------------------------------------------------


class TestBroadcastDistributionProvenance:
    def test_provenance_attachment(self):
        bd = BroadcastDistribution(
            input_samples={"x": jnp.ones((5, 1))},
            output_samples=jnp.zeros((5, 1)),
            weights=None,
            broadcast_args=["x"],
        )
        prov = Provenance("broadcast", metadata={"func": "test"})
        bd.with_source(prov)
        assert bd.source is prov
        assert bd.source.operation == "broadcast"


# ---------------------------------------------------------------------------
# _ArrayMarginal (output marginal for array outputs)
# ---------------------------------------------------------------------------


class TestArrayMarginal:
    def test_protocols(self):
        m = _ArrayMarginal(jnp.ones((10, 3)), None)
        assert isinstance(m, SupportsSampling)
        assert isinstance(m, SupportsMean)
        assert isinstance(m, SupportsVariance)
        assert isinstance(m, SupportsCovariance)

    def test_mean_uniform(self):
        samples = jnp.array([[1.0], [2.0], [3.0]])
        m = _ArrayMarginal(samples, None)
        np.testing.assert_allclose(m._mean(), jnp.array([2.0]), atol=1e-5)

    def test_mean_weighted(self):
        samples = jnp.array([[0.0], [10.0]])
        weights = jnp.array([0.75, 0.25])
        m = _ArrayMarginal(samples, weights)
        np.testing.assert_allclose(m._mean(), jnp.array([2.5]), atol=1e-5)

    def test_variance(self):
        samples = jnp.array([[1.0], [3.0]])
        m = _ArrayMarginal(samples, None)
        np.testing.assert_allclose(m._variance(), jnp.array([1.0]), atol=1e-5)

    def test_cov(self):
        samples = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        m = _ArrayMarginal(samples, None)
        cov = m._cov()
        assert cov.shape == (2, 2)

    def test_sample(self, key):
        samples = jnp.arange(100, dtype=jnp.float32).reshape(-1, 1)
        m = _ArrayMarginal(samples, None)
        drawn = m._sample(key, (50,))
        assert drawn.shape == (50, 1)

    def test_properties(self):
        samples = jnp.ones((10, 3))
        m = _ArrayMarginal(samples, None)
        assert m.n == 10
        assert m.event_shape == (3,)
        assert m.dim == 3


# ---------------------------------------------------------------------------
# _MixtureMarginal (output marginal for distribution outputs)
# ---------------------------------------------------------------------------


class TestMixtureMarginal:
    def test_sampling_protocol_when_components_support_it(self):
        components = [Normal(loc=0.0, scale=1.0, name="x"), Normal(loc=5.0, scale=1.0, name="y")]
        m = _make_mixture_marginal(components, None)
        assert isinstance(m, SupportsSampling)

    def test_mean_protocol_when_components_support_it(self):
        components = [Normal(loc=0.0, scale=1.0, name="x"), Normal(loc=10.0, scale=1.0, name="y")]
        m = _make_mixture_marginal(components, None)
        assert isinstance(m, SupportsMean)
        np.testing.assert_allclose(m._mean(), 5.0, atol=1e-5)

    def test_mean_weighted(self):
        components = [Normal(loc=0.0, scale=1.0, name="x"), Normal(loc=10.0, scale=1.0, name="y")]
        weights = jnp.array([0.75, 0.25])
        m = _make_mixture_marginal(components, weights)
        np.testing.assert_allclose(m._mean(), 2.5, atol=1e-5)

    def test_variance_protocol(self):
        components = [Normal(loc=0.0, scale=1.0, name="x"), Normal(loc=10.0, scale=1.0, name="y")]
        m = _make_mixture_marginal(components, None)
        assert isinstance(m, SupportsVariance)
        v = m._variance()
        # Law of total variance: E[Var] + Var[E]
        # E[Var] = 0.5*(1+1) = 1, Var[E] = 0.5*(0-5)^2 + 0.5*(10-5)^2 = 25
        np.testing.assert_allclose(v, 26.0, atol=1e-4)

    def test_log_prob_protocol(self):
        components = [Normal(loc=0.0, scale=1.0, name="x"), Normal(loc=5.0, scale=1.0, name="y")]
        m = _make_mixture_marginal(components, None)
        assert isinstance(m, SupportsLogProb)
        lp = m._log_prob(jnp.array(0.0))
        assert jnp.isfinite(lp)

    def test_sample_draws(self, key):
        components = [Normal(loc=-100.0, scale=0.01, name="a"), Normal(loc=100.0, scale=0.01, name="b")]
        m = _make_mixture_marginal(components, None)
        draws = m._sample(key, (1000,))
        # Should be bimodal around -100 and 100
        assert float(jnp.min(draws)) < -50
        assert float(jnp.max(draws)) > 50

    def test_no_sampling_when_components_lack_it(self):
        """Components without SupportsSampling → marginal shouldn't support it."""
        from probpipe.core.distribution import Distribution

        class NoSampleDist(Distribution):
            pass

        components = [NoSampleDist(name="test"), NoSampleDist(name="test")]
        m = _make_mixture_marginal(components, None)
        assert not isinstance(m, SupportsSampling)

    def test_properties(self):
        components = [Normal(loc=0.0, scale=1.0, name="x"), Normal(loc=5.0, scale=1.0, name="y")]
        m = _make_mixture_marginal(components, None)
        assert m.n == 2
        assert m.components is components


# ---------------------------------------------------------------------------
# _ListMarginal (output marginal for non-stackable outputs)
# ---------------------------------------------------------------------------


class TestListMarginal:
    def test_no_protocols(self):
        m = _ListMarginal(["a", "b", "c"], None)
        assert not isinstance(m, SupportsSampling)
        assert not isinstance(m, SupportsMean)

    def test_properties(self):
        items = ["hello", "world"]
        m = _ListMarginal(items, None)
        assert m.n == 2
        assert m.items == items


# ---------------------------------------------------------------------------
# _make_marginal factory
# ---------------------------------------------------------------------------


class TestMakeMarginal:
    def test_array_output(self):
        m = _make_marginal(jnp.ones((5, 2)), None)
        assert isinstance(m, _ArrayMarginal)

    def test_list_of_arrays(self):
        m = _make_marginal([jnp.array(1.0), jnp.array(2.0)], None)
        assert isinstance(m, _ArrayMarginal)

    def test_list_of_strings(self):
        m = _make_marginal(["a", "b", "c"], None)
        assert isinstance(m, _ListMarginal)

    def test_list_of_distributions(self):
        dists = [Normal(loc=0.0, scale=1.0, name="x"), Normal(loc=1.0, scale=1.0, name="y")]
        m = _make_marginal(dists, None)
        assert isinstance(m, _MixtureMarginal)

    def test_explicit_output_distributions(self):
        dists = [Normal(loc=0.0, scale=1.0, name="x"), Normal(loc=1.0, scale=1.0, name="y")]
        m = _make_marginal(None, None, output_distributions=dists)
        assert isinstance(m, _MixtureMarginal)


# ---------------------------------------------------------------------------
# BroadcastDistribution.marginalize()
# ---------------------------------------------------------------------------


class TestMarginalize:
    def test_marginal_is_cached(self):
        bd = BroadcastDistribution(
            input_samples={"x": jnp.ones((5, 1))},
            output_samples=jnp.zeros((5, 2)),
            weights=None,
            broadcast_args=["x"],
        )
        m1 = bd.marginalize()
        m2 = bd.marginalize()
        assert m1 is m2

    def test_output_property_is_alias(self):
        bd = BroadcastDistribution(
            input_samples={"x": jnp.ones((5, 1))},
            output_samples=jnp.zeros((5, 2)),
            weights=None,
            broadcast_args=["x"],
        )
        assert bd.output is bd.marginalize()

    def test_marginal_inherits_weights(self):
        w = jnp.array([0.1, 0.2, 0.3, 0.2, 0.2])
        bd = BroadcastDistribution(
            input_samples={"x": jnp.ones((5, 1))},
            output_samples=jnp.zeros((5, 2)),
            weights=w,
            broadcast_args=["x"],
        )
        m = bd.marginalize()
        np.testing.assert_allclose(m.weights, w, atol=1e-5)

    def test_backward_compat_samples_property(self):
        """bd.samples should forward to marginal."""
        data = jnp.arange(10, dtype=jnp.float32).reshape(-1, 1)
        bd = BroadcastDistribution(
            input_samples={"x": jnp.ones((10, 1))},
            output_samples=data,
            weights=None,
            broadcast_args=["x"],
        )
        np.testing.assert_allclose(bd.samples, data, atol=1e-6)

    def test_samples_property_list_output(self):
        """bd.samples should forward to marginal.items for list outputs."""
        items = ["a", "b", "c"]
        bd = BroadcastDistribution(
            input_samples={"x": jnp.ones((3, 1))},
            output_samples=items,
            weights=None,
            broadcast_args=["x"],
        )
        assert bd.samples == items


# ---------------------------------------------------------------------------
# BroadcastDistribution joint sampling with list outputs
# ---------------------------------------------------------------------------


class TestBroadcastDistributionListOutputSampling:
    def test_joint_sample_list_outputs(self, key):
        """Resampling with list outputs returns list of items."""
        items = ["alpha", "beta", "gamma"]
        bd = BroadcastDistribution(
            input_samples={"x": jnp.array([[1.0], [2.0], [3.0]])},
            output_samples=items,
            weights=None,
            broadcast_args=["x"],
        )
        batch = bd._sample(key, (5,))
        assert isinstance(batch["_output"], list)
        assert len(batch["_output"]) == 5
        assert all(item in items for item in batch["_output"])

    def test_joint_sample_with_weights(self, key):
        """Weighted joint sampling selects pairs according to weights."""
        n = 3
        inputs = {"x": jnp.array([[1.0], [2.0], [3.0]])}
        outputs = jnp.array([[10.0], [20.0], [30.0]])
        # Put all weight on the last sample
        w = jnp.array([0.0, 0.0, 1.0])
        bd = BroadcastDistribution(
            input_samples=inputs,
            output_samples=outputs,
            weights=w,
            broadcast_args=["x"],
        )
        batch = bd._sample(key, (20,))
        # All samples should be the third pair
        np.testing.assert_allclose(batch["x"], jnp.full((20, 1), 3.0), atol=1e-5)
        np.testing.assert_allclose(batch["_output"], jnp.full((20, 1), 30.0), atol=1e-5)


# ---------------------------------------------------------------------------
# _ArrayMarginal additional coverage
# ---------------------------------------------------------------------------


class TestArrayMarginalAdditional:
    def test_repr(self):
        m = _ArrayMarginal(jnp.ones((10, 3)), None)
        r = repr(m)
        assert "MarginalizedBroadcastDistribution" in r
        assert "n=10" in r
        assert "event_shape=(3,)" in r

    def test_sample_scalar(self, key):
        """Single draw (sample_shape=()) returns unbatched array."""
        samples = jnp.arange(100, dtype=jnp.float32).reshape(-1, 1)
        m = _ArrayMarginal(samples, None)
        drawn = m._sample(key, ())
        assert drawn.shape == (1,)

    def test_sample_weighted_scalar(self, key):
        """Weighted single draw."""
        samples = jnp.array([[0.0], [100.0]])
        w = jnp.array([0.0, 1.0])
        m = _ArrayMarginal(samples, w)
        drawn = m._sample(key, ())
        np.testing.assert_allclose(drawn, jnp.array([100.0]), atol=1e-5)

    def test_sample_weighted_batch(self, key):
        """Weighted batch draw."""
        samples = jnp.array([[0.0], [100.0]])
        w = jnp.array([0.0, 1.0])
        m = _ArrayMarginal(samples, w)
        drawn = m._sample(key, (20,))
        np.testing.assert_allclose(drawn, jnp.full((20, 1), 100.0), atol=1e-5)

    def test_expectation_full(self):
        """Full expectation over all samples."""
        samples = jnp.array([[1.0], [2.0], [3.0]])
        m = _ArrayMarginal(samples, None)
        result = m._expectation(lambda x: x ** 2)
        # E[X^2] = (1+4+9)/3 = 14/3
        np.testing.assert_allclose(result, jnp.array([14.0 / 3]), atol=1e-4)

    def test_expectation_full_weighted(self):
        """Weighted expectation."""
        samples = jnp.array([[0.0], [10.0]])
        w = jnp.array([0.75, 0.25])
        m = _ArrayMarginal(samples, w)
        result = m._expectation(lambda x: x)
        np.testing.assert_allclose(result, jnp.array([2.5]), atol=1e-4)

    def test_expectation_subsampled(self, key):
        """Subsampled expectation returns BootstrapDistribution by default."""
        from probpipe.core.distribution import BootstrapDistribution

        samples = jnp.arange(100, dtype=jnp.float32).reshape(-1, 1)
        m = _ArrayMarginal(samples, None)
        result = m._expectation(lambda x: x, key=key, num_evaluations=20)
        assert isinstance(result, BootstrapDistribution)

    def test_expectation_subsampled_no_dist(self, key):
        """Subsampled expectation with return_dist=False returns array."""
        samples = jnp.arange(100, dtype=jnp.float32).reshape(-1, 1)
        m = _ArrayMarginal(samples, None)
        result = m._expectation(
            lambda x: x, key=key, num_evaluations=20, return_dist=False
        )
        assert isinstance(result, jnp.ndarray)

    def test_expectation_subsampled_weighted(self, key):
        """Subsampled expectation with weights."""
        from probpipe.core.distribution import BootstrapDistribution

        n = 50
        samples = jnp.arange(n, dtype=jnp.float32).reshape(-1, 1)
        w = jnp.ones(n) / n
        m = _ArrayMarginal(samples, w)
        result = m._expectation(lambda x: x, key=key, num_evaluations=10)
        assert isinstance(result, BootstrapDistribution)

    def test_expectation_subsampled_weighted_no_dist(self, key):
        """Subsampled weighted expectation with return_dist=False."""
        n = 50
        samples = jnp.arange(n, dtype=jnp.float32).reshape(-1, 1)
        w = jnp.ones(n) / n
        m = _ArrayMarginal(samples, w)
        result = m._expectation(
            lambda x: x, key=key, num_evaluations=10, return_dist=False
        )
        assert isinstance(result, jnp.ndarray)

    def test_cov_weighted(self):
        """Weighted covariance."""
        samples = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        w = jnp.array([0.5, 0.5])
        m = _ArrayMarginal(samples, w)
        cov = m._cov()
        assert cov.shape == (2, 2)

    def test_variance_weighted(self):
        """Weighted variance."""
        samples = jnp.array([[0.0], [10.0]])
        w = jnp.array([0.5, 0.5])
        m = _ArrayMarginal(samples, w)
        v = m._variance()
        # Var = 0.5*(0-5)^2 + 0.5*(10-5)^2 = 25
        np.testing.assert_allclose(v, jnp.array([25.0]), atol=1e-4)


# ---------------------------------------------------------------------------
# _MixtureMarginal repr
# ---------------------------------------------------------------------------


class TestMixtureMarginalAdditional:
    def test_repr(self):
        components = [Normal(loc=0.0, scale=1.0, name="x"), Normal(loc=5.0, scale=1.0, name="y")]
        m = _make_mixture_marginal(components, None)
        r = repr(m)
        assert "MarginalizedBroadcastDistribution" in r
        assert "mixture" in r
        assert "n=2" in r

    def test_sample_scalar(self, key):
        """Single draw (sample_shape=()) returns scalar."""
        components = [Normal(loc=0.0, scale=1.0, name="x"), Normal(loc=5.0, scale=1.0, name="y")]
        m = _make_mixture_marginal(components, None)
        drawn = m._sample(key, ())
        assert drawn.shape == ()

    def test_weights_property(self):
        components = [Normal(loc=0.0, scale=1.0, name="x"), Normal(loc=5.0, scale=1.0, name="y")]
        m = _make_mixture_marginal(components, None)
        np.testing.assert_allclose(m.weights, jnp.array([0.5, 0.5]), atol=1e-5)


# ---------------------------------------------------------------------------
# _ListMarginal repr and weights
# ---------------------------------------------------------------------------


class TestListMarginalAdditional:
    def test_repr(self):
        m = _ListMarginal(["a", "b"], None)
        r = repr(m)
        assert "MarginalizedBroadcastDistribution" in r
        assert "list" in r
        assert "n=2" in r

    def test_weights_uniform(self):
        m = _ListMarginal(["a", "b"], None)
        np.testing.assert_allclose(m.weights, jnp.array([0.5, 0.5]))

    def test_weights_provided(self):
        w = jnp.array([0.3, 0.7])
        m = _ListMarginal(["a", "b"], w)
        np.testing.assert_allclose(m.weights, w, atol=1e-5)


# ---------------------------------------------------------------------------
# _make_marginal edge cases
# ---------------------------------------------------------------------------


class TestMakeMarginalEdgeCases:
    def test_scalar_output(self):
        """Single scalar value (e.g., from vmap)."""
        m = _make_marginal(3.14, None)
        assert isinstance(m, _ArrayMarginal)

    def test_with_name(self):
        """Name propagates."""
        m = _make_marginal(jnp.ones((5, 2)), None, name="test_output")
        assert m._name == "test_output"


# ---------------------------------------------------------------------------
# _RecordArrayMarginal (Record-returning WorkflowFunctions)
# ---------------------------------------------------------------------------


from probpipe import Record, RecordArray, ProductDistribution  # noqa: E402
from probpipe.core._broadcast_distributions import _RecordArrayMarginal  # noqa: E402
from probpipe.core.node import workflow_function  # noqa: E402
from probpipe import mean, variance, sample  # noqa: E402


class TestRecordArrayMarginal:
    """Record-returning WorkflowFunction outputs should be RecordArrayMarginal,
    not _ListMarginal, and must support mean/variance/sample."""

    @pytest.fixture
    def record_workflow(self):
        @workflow_function
        def transform(x, y):
            return Record(sum=x + y, diff=x - y)
        return transform

    @pytest.fixture
    def prior(self):
        return ProductDistribution(
            Normal(loc=1.0, scale=0.1, name="x"),
            Normal(loc=2.0, scale=0.1, name="y"),
        )

    def test_record_output_produces_record_array_marginal(
        self, record_workflow, prior,
    ):
        result = record_workflow(**prior.select("x", "y"))
        assert isinstance(result, _RecordArrayMarginal)

    def test_mean_per_field(self, record_workflow, prior):
        result = record_workflow(**prior.select("x", "y"))
        m = mean(result)
        assert isinstance(m, Record)
        # sum = x + y ~ N(3, sqrt(0.02)); diff = x - y ~ N(-1, sqrt(0.02))
        mc_se = 3.0 * np.sqrt(0.02) / np.sqrt(128)
        np.testing.assert_allclose(float(m["sum"]), 3.0, atol=mc_se)
        np.testing.assert_allclose(float(m["diff"]), -1.0, atol=mc_se)

    def test_variance_per_field(self, record_workflow, prior):
        result = record_workflow(**prior.select("x", "y"))
        v = variance(result)
        assert isinstance(v, Record)
        # Var(x+y) = Var(x) + Var(y) = 0.02 when jointly sampled independently.
        # Allow for MC SE on variance: ~ 2 * var / sqrt(n) = 2 * 0.02 / sqrt(128).
        np.testing.assert_allclose(float(v["sum"]), 0.02, atol=0.02)
        np.testing.assert_allclose(float(v["diff"]), 0.02, atol=0.02)

    def test_sample_returns_record(self, record_workflow, prior, key):
        result = record_workflow(**prior.select("x", "y"))
        s = sample(result, key=key, sample_shape=(5,))
        assert "sum" in s
        assert s["sum"].shape == (5,)
        assert s["diff"].shape == (5,)

    def test_repr_mentions_fields(self, record_workflow, prior):
        result = record_workflow(**prior.select("x", "y"))
        r = repr(result)
        assert "sum" in r and "diff" in r


# ===========================================================================
# _make_stack — RecordArray-broadcast sibling of _make_marginal (issue #130)
# ===========================================================================


class TestMakeStack:
    """Dispatch rules for wrapping n inner-function outputs as a
    shape-(n,) aggregate. Every case is a parameter-sweep-like scenario
    where row identity must survive; there is no marginalisation."""

    def test_list_of_scalars_wraps_as_numeric_record_array(self):
        from probpipe import NumericRecordArray
        from probpipe.core._broadcast_distributions import _make_stack, AUTO_WRAP_FIELD
        out = _make_stack([1.0, 2.0, 3.0, 4.0], n=4)
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (4,)
        assert out.fields == (AUTO_WRAP_FIELD,)
        np.testing.assert_allclose(out[AUTO_WRAP_FIELD], [1.0, 2.0, 3.0, 4.0])

    def test_list_of_arrays_preserves_event_shape(self):
        from probpipe import NumericRecordArray
        from probpipe.core._broadcast_distributions import _make_stack, AUTO_WRAP_FIELD
        values = [jnp.arange(3.0) + 10.0 * i for i in range(4)]
        out = _make_stack(values, n=4)
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (4,)
        assert out[AUTO_WRAP_FIELD].shape == (4, 3)

    def test_list_of_numeric_records_promotes_to_numeric_array(self):
        from probpipe import NumericRecord, NumericRecordArray
        from probpipe.core._broadcast_distributions import _make_stack
        records = [NumericRecord(a=float(i), b=float(i) * 2) for i in range(5)]
        out = _make_stack(records, n=5)
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (5,)
        np.testing.assert_allclose(out["a"], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(out["b"], [0, 2, 4, 6, 8])

    def test_list_of_mixed_records_falls_back_to_recordarray(self):
        """Records with a string (non-numeric) leaf can't go through
        ``NumericRecordArray.stack``. The fallback path builds each
        field independently — numeric leaves via ``jnp.stack``,
        opaque leaves via ``np.asarray(dtype=object)``."""
        from probpipe import Record, NumericRecordArray, RecordArray
        from probpipe.core._broadcast_distributions import _make_stack
        records = [Record(a=float(i), label=f"row{i}") for i in range(3)]
        out = _make_stack(records, n=3)
        assert isinstance(out, RecordArray)
        assert not isinstance(out, NumericRecordArray)
        np.testing.assert_allclose(out["a"], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(out["label"], ["row0", "row1", "row2"])

    def test_list_of_distributions_gives_distribution_array(self):
        from probpipe import DistributionArray, Normal
        from probpipe.core._broadcast_distributions import _make_stack
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(3)]
        out = _make_stack(comps, n=3)
        assert isinstance(out, DistributionArray)
        assert out.batch_shape == (3,)
        assert out[0] is comps[0]

    def test_list_of_record_arrays_nests_batch_shape(self):
        """Each inner RecordArray has its own batch_shape (m,). Stacking
        n of them produces a RecordArray with batch_shape (n, m)."""
        from probpipe import NumericRecord, NumericRecordArray
        from probpipe.core._broadcast_distributions import _make_stack
        inner = [
            NumericRecordArray.stack(
                [NumericRecord(x=float(i * 10 + j)) for j in range(4)]
            )
            for i in range(3)
        ]
        out = _make_stack(inner, n=3)
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (3, 4)
        np.testing.assert_allclose(out["x"][0], [0, 1, 2, 3])
        np.testing.assert_allclose(out["x"][2], [20, 21, 22, 23])

    def test_vmap_ndarray_wraps_as_numeric_record_array(self):
        """A bare ``jnp.ndarray`` with leading axis n (typical ``jax.vmap``
        output for scalar-returning fns) wraps without unstacking."""
        from probpipe import NumericRecordArray
        from probpipe.core._broadcast_distributions import _make_stack, AUTO_WRAP_FIELD
        arr = jnp.arange(12.0).reshape(4, 3)
        out = _make_stack(arr, n=4)
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (4,)
        assert out[AUTO_WRAP_FIELD].shape == (4, 3)

    def test_vmap_record_with_batched_leaves_promotes_to_ra(self):
        """``jax.vmap`` of a Record-returning fn produces a Record whose
        leaves are already batched along a leading axis. That's the
        input form for the pytree branch of ``_make_stack``."""
        from probpipe import NumericRecordArray, Record
        from probpipe.core._broadcast_distributions import _make_stack
        rec = Record(x=jnp.arange(5.0), y=jnp.arange(5.0) + 10)
        out = _make_stack(rec, n=5)
        assert isinstance(out, NumericRecordArray)
        assert out.batch_shape == (5,)

    def test_length_mismatch_raises(self):
        from probpipe.core._broadcast_distributions import _make_stack
        with pytest.raises(ValueError, match=r"expected prod\(batch_shape\)=5"):
            _make_stack([1.0, 2.0, 3.0], n=5)

    def test_ndarray_leading_axis_mismatch_raises(self):
        from probpipe.core._broadcast_distributions import _make_stack
        with pytest.raises(ValueError, match="expected leading axis"):
            _make_stack(jnp.arange(6.0), n=4)


# ===========================================================================
# _coerce_output — attaches provenance to broadcast outputs
# ===========================================================================


class TestCoerceOutput:
    """``_coerce_output`` is the single entry point where broadcast
    outputs pick up their provenance. Non-broadcast values pass through
    unchanged (scalars / ndarrays / callables stay usable in idiomatic
    arithmetic / attribute access)."""

    def test_none_mode_passes_through(self):
        from probpipe.core.node import _coerce_output
        # Non-Record/Dist values wouldn't normally have .with_source
        # but the "none" mode short-circuits before the check anyway.
        assert _coerce_output(3.14, broadcast_mode="none", provenance=None) == 3.14
        # None provenance also short-circuits.
        prov = Provenance("x", parents=())
        assert _coerce_output(3.14, broadcast_mode="none", provenance=prov) == 3.14

    def test_stack_mode_attaches_to_recordarray(self):
        from probpipe import NumericRecord, NumericRecordArray
        from probpipe.core.node import _coerce_output
        ra = NumericRecordArray.stack(
            [NumericRecord(x=float(i)) for i in range(3)]
        )
        assert ra.source is None
        prov = Provenance("sweep", parents=())
        out = _coerce_output(ra, broadcast_mode="stack", provenance=prov)
        assert out is ra
        assert ra.source.operation == "sweep"

    def test_attaches_to_distribution_array(self):
        from probpipe import DistributionArray, Normal
        from probpipe.core._broadcast_distributions import _make_stack
        from probpipe.core.node import _coerce_output
        da = _make_stack(
            [Normal(loc=0.0, scale=1.0, name=f"d{i}") for i in range(3)], n=3,
        )
        assert isinstance(da, DistributionArray)
        assert da.source is None
        prov = Provenance("nested", parents=())
        _coerce_output(da, broadcast_mode="nested", provenance=prov)
        assert da.source.operation == "nested"

    def test_existing_source_is_not_overwritten(self):
        """If the broadcasting layer has already wired a fresh inner
        marginal with a source (e.g., a _MixtureMarginal was built with
        its own provenance), ``_coerce_output`` must not crash and the
        existing source must remain."""
        from probpipe import NumericRecord
        from probpipe.core.node import _coerce_output
        nr = NumericRecord(x=1.0).with_source(Provenance("inner", parents=()))
        # Second set would normally raise RuntimeError; _coerce_output
        # swallows it.
        _coerce_output(
            nr,
            broadcast_mode="stack",
            provenance=Provenance("outer", parents=()),
        )
        assert nr.source.operation == "inner"
