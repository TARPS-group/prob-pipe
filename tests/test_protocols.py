"""Tests for protocol compliance across distribution classes."""

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
    Normal,
    Beta,
    Gamma,
    Bernoulli,
    Categorical,
    MultivariateNormal,
    EmpiricalDistribution,
    NumericEmpiricalDistribution,
    BootstrapDistribution,
    TransformedDistribution,
    ProductDistribution,
    SequentialJointDistribution,
    JointEmpirical,
    JointGaussian,
)
from probpipe.core.protocols import (
    SupportsExpectation,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsLogProb,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsConditioning,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal():
    return Normal(loc=0.0, scale=1.0, name="x")


@pytest.fixture
def empirical():
    samples = jax.random.normal(jax.random.PRNGKey(0), (100, 2))
    return EmpiricalDistribution(samples)


@pytest.fixture
def bootstrap():
    evals = jax.random.normal(jax.random.PRNGKey(1), (50,))
    return BootstrapDistribution(evals)


@pytest.fixture
def joint():
    return ProductDistribution(x=Normal(0, 1, name="x"), y=Normal(1, 2, name="y"))


# ---------------------------------------------------------------------------
# SupportsSampling
# ---------------------------------------------------------------------------

class TestSupportsSampling:
    """All distributions should support sampling."""

    @pytest.mark.parametrize("dist_cls,kwargs", [
        (Normal, {"loc": 0.0, "scale": 1.0, "name": "x"}),
        (Beta, {"alpha": 2.0, "beta": 5.0, "name": "b"}),
        (Gamma, {"concentration": 3.0, "rate": 1.0, "name": "g"}),
        (Bernoulli, {"probs": 0.5, "name": "d"}),
        (MultivariateNormal, {"loc": jnp.zeros(2), "cov": jnp.eye(2), "name": "z"}),
    ])
    def test_tfp_distributions(self, dist_cls, kwargs):
        dist = dist_cls(**kwargs)
        assert isinstance(dist, SupportsSampling)

    def test_empirical(self, empirical):
        assert isinstance(empirical, SupportsSampling)

    def test_bootstrap(self, bootstrap):
        assert isinstance(bootstrap, SupportsSampling)

    def test_joint(self, joint):
        assert isinstance(joint, SupportsSampling)


# ---------------------------------------------------------------------------
# SupportsExpectation
# ---------------------------------------------------------------------------

class TestSupportsExpectation:
    def test_normal(self, normal):
        assert isinstance(normal, SupportsExpectation)

    def test_empirical(self, empirical):
        assert isinstance(empirical, SupportsExpectation)

    def test_joint(self, joint):
        assert isinstance(joint, SupportsExpectation)


# ---------------------------------------------------------------------------
# SupportsLogProb
# ---------------------------------------------------------------------------

class TestSupportsLogProb:
    @pytest.mark.parametrize("dist_cls,kwargs", [
        (Normal, {"loc": 0.0, "scale": 1.0, "name": "x"}),
        (Beta, {"alpha": 2.0, "beta": 5.0, "name": "b"}),
        (MultivariateNormal, {"loc": jnp.zeros(2), "cov": jnp.eye(2), "name": "z"}),
    ])
    def test_tfp_distributions(self, dist_cls, kwargs):
        dist = dist_cls(**kwargs)
        assert isinstance(dist, SupportsLogProb)

    def test_empirical_not_log_prob(self, empirical):
        assert not isinstance(empirical, SupportsLogProb)


# ---------------------------------------------------------------------------
# Protocol hierarchy
# ---------------------------------------------------------------------------

class TestProtocolHierarchy:
    """Verify that protocol inheritance relationships hold."""

    def test_sampling_and_expectation_independent(self, normal):
        """SupportsSampling and SupportsExpectation are independent protocols."""
        assert isinstance(normal, SupportsSampling)
        assert isinstance(normal, SupportsExpectation)

    def test_log_prob_implies_unnormalized(self, normal):
        """SupportsLogProb extends SupportsUnnormalizedLogProb."""
        assert isinstance(normal, SupportsLogProb)
        assert isinstance(normal, SupportsUnnormalizedLogProb)

    def test_mean_independent_of_expectation(self):
        """SupportsMean does NOT extend SupportsExpectation."""
        assert not issubclass(SupportsMean, SupportsExpectation)

    def test_variance_independent_of_expectation(self):
        """SupportsVariance does NOT extend SupportsExpectation."""
        assert not issubclass(SupportsVariance, SupportsExpectation)

    def test_covariance_independent_of_expectation(self):
        """SupportsCovariance does NOT extend SupportsExpectation."""
        assert not issubclass(SupportsCovariance, SupportsExpectation)

    def test_concrete_dist_supports_both_mean_and_expectation(self, normal):
        """Concrete distributions like Normal support both independently."""
        assert isinstance(normal, SupportsMean)
        assert isinstance(normal, SupportsExpectation)

    def test_sampling_independent_of_expectation(self):
        """SupportsSampling does NOT extend SupportsExpectation."""
        assert not issubclass(SupportsSampling, SupportsExpectation)

    def test_log_prob_subclass_check(self):
        assert issubclass(SupportsLogProb, SupportsUnnormalizedLogProb)


# ---------------------------------------------------------------------------
# SupportsMean / SupportsVariance / SupportsCovariance
# ---------------------------------------------------------------------------

class TestSupportsMean:
    """Only distributions with exact (non-MC) moments satisfy these."""

    def test_tfp_normal(self, normal):
        assert isinstance(normal, SupportsMean)
        assert isinstance(normal, SupportsVariance)

    def test_empirical_numeric_has_moments(self, empirical):
        """Numeric EmpiricalDistribution dispatches to Array variant with moments."""
        assert isinstance(empirical, SupportsMean)
        assert isinstance(empirical, SupportsVariance)
        assert isinstance(empirical, SupportsCovariance)

    def test_empirical_generic_no_moments(self):
        """Non-numeric EmpiricalDistribution does not support moments."""
        dist = EmpiricalDistribution(["a", "b", "c"])
        assert not isinstance(dist, SupportsMean)
        assert not isinstance(dist, SupportsVariance)
        assert not isinstance(dist, SupportsCovariance)

    def test_array_empirical(self):
        samples = jax.random.normal(jax.random.PRNGKey(0), (100, 2))
        dist = NumericEmpiricalDistribution(samples)
        assert isinstance(dist, SupportsMean)
        assert isinstance(dist, SupportsVariance)
        assert isinstance(dist, SupportsCovariance)

    def test_bootstrap(self, bootstrap):
        assert isinstance(bootstrap, SupportsMean)
        assert isinstance(bootstrap, SupportsVariance)


# ---------------------------------------------------------------------------
# SupportsConditioning
# ---------------------------------------------------------------------------

class TestSupportsConditioning:
    def test_product_distribution(self, joint):
        assert isinstance(joint, SupportsConditioning)

    def test_sequential_joint(self):
        sjd = SequentialJointDistribution(
            x=Normal(0, 1, name="x"),
            y=lambda x: Normal(loc=x, scale=1.0, name="y"),
        )
        assert isinstance(sjd, SupportsConditioning)

    def test_joint_gaussian(self):
        jg = JointGaussian(
            mean=jnp.zeros(4),
            cov=jnp.eye(4),
            x=2,
            y=2,
        )
        assert isinstance(jg, SupportsConditioning)

    def test_normal_not_conditionable(self, normal):
        assert not isinstance(normal, SupportsConditioning)


# ---------------------------------------------------------------------------
# Named components (duck-typing check)
# ---------------------------------------------------------------------------

class TestNamedComponents:
    def test_product_distribution(self, joint):
        assert hasattr(joint, 'component_names')

    def test_normal_component_names_has_name(self, normal):
        assert normal.component_names == ("x",)


# ---------------------------------------------------------------------------
# Orchestration hints
# ---------------------------------------------------------------------------

class TestOrchestrationHints:
    def test_default_sampling_cost(self, normal):
        assert normal._sampling_cost == "low"

    def test_default_preferred_orchestration(self, normal):
        assert normal._preferred_orchestration is None


# ---------------------------------------------------------------------------
# Dynamic-protocol views (regression for hard-coded SupportsMean/Variance/
# Sampling on _RecordDistributionView and FlattenedView)
# ---------------------------------------------------------------------------


class TestRecordDistributionViewDynamicProtocols:
    """A view over a field must only claim protocols its parent supports."""

    def test_view_over_log_prob_only_parent_is_not_sampling(self):
        """Build a parent with a RecordTemplate that supports only
        log_prob, and verify the view doesn't claim to be
        SupportsSampling / SupportsMean / SupportsVariance."""
        from probpipe.core._distribution_base import Distribution
        from probpipe.core._record_distribution import (
            RecordDistribution, _RecordDistributionView,
        )
        from probpipe.core.record import RecordTemplate

        class _LogProbOnlyParent(RecordDistribution, SupportsLogProb):
            record_template = RecordTemplate(x=(), y=())

            def __init__(self):
                self._name = "lp_only"

            def _log_prob(self, value):
                import jax.numpy as jnp
                return jnp.asarray(0.0)

        parent = _LogProbOnlyParent()
        view = _RecordDistributionView(parent, "x")
        assert isinstance(view, SupportsLogProb)
        assert not isinstance(view, SupportsSampling)
        assert not isinstance(view, SupportsMean)
        assert not isinstance(view, SupportsVariance)

    def test_view_over_full_parent_gets_all_protocols(self):
        """ProductDistribution supports sampling and (through its TFP
        leaves) mean / variance — its views should match."""
        dist = ProductDistribution(
            intercept=Normal(loc=0.0, scale=1.0, name="intercept"),
            slope=Normal(loc=0.0, scale=1.0, name="slope"),
        )
        view = dist["intercept"]
        assert isinstance(view, SupportsSampling)
        assert isinstance(view, SupportsMean)
        assert isinstance(view, SupportsVariance)


class TestFlattenedViewDynamicProtocols:
    """FlattenedView must only claim the protocols its base supports."""

    def test_flattened_view_inherits_sampling_and_log_prob(self):
        from probpipe.core._array_distributions import FlattenedView
        dist = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=0.0, scale=1.0, name="y"),
        )
        flat = FlattenedView(dist)
        assert isinstance(flat, SupportsSampling)
        assert isinstance(flat, SupportsLogProb)

    def test_flattened_view_over_sampling_only_base(self):
        """A sampling-only base produces a FlattenedView that isn't
        SupportsLogProb."""
        import jax.numpy as jnp
        import jax
        from probpipe.core._array_distributions import (
            FlattenedView, NumericRecordDistribution,
        )
        from probpipe.core.record import RecordTemplate

        class _SampleOnlyBase(NumericRecordDistribution, SupportsSampling):
            record_template = RecordTemplate(x=())

            def __init__(self):
                self._name = "sample_only"

            @property
            def event_shape(self):
                return ()

            def _sample_one(self, key):
                return jnp.asarray(0.0)

            def _sample(self, key, sample_shape=()):
                return jax.random.normal(key, sample_shape)

        base = _SampleOnlyBase()
        flat = FlattenedView(base)
        assert isinstance(flat, SupportsSampling)
        assert not isinstance(flat, SupportsLogProb)


# ---------------------------------------------------------------------------
# Sample / sample_one return-type convention
# ---------------------------------------------------------------------------


class TestSampleReturnTypeConvention:
    """Pin down the contract documented on ``SupportsSampling``.

    - Numeric distributions return ``Array`` (sample_shape + event_shape).
    - Record-based joints return ``Record`` / ``NumericRecord`` for an
      unbatched draw and ``NumericRecordArray`` for a batched draw.
    - ``_sample_one(key)`` must be equivalent to ``_sample(key, ())``.
    """

    def test_numeric_distribution_returns_array(self):
        import jax.numpy as jnp
        dist = Normal(loc=0.0, scale=1.0, name="x")
        k = jax.random.PRNGKey(0)
        assert isinstance(dist._sample(k, ()), jnp.ndarray)
        assert dist._sample(k, (5,)).shape == (5,)
        assert dist._sample(k, (3, 4)).shape == (3, 4)

    def test_product_distribution_return_types(self):
        from probpipe import NumericRecord, Record
        from probpipe.core._record_array import NumericRecordArray
        dist = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=0.0, scale=1.0, name="y"),
        )
        k = jax.random.PRNGKey(0)
        # unbatched
        s0 = dist._sample(k, ())
        assert isinstance(s0, Record)
        # batched
        s1 = dist._sample(k, (5,))
        assert isinstance(s1, NumericRecordArray)
        assert s1.batch_shape == (5,)

    def test_sample_one_matches_sample_empty(self):
        """Every joint's ``_sample_one(key)`` must equal
        ``_sample(key, ())`` (same type, same fields, same values)."""
        from probpipe import Record
        k = jax.random.PRNGKey(42)

        # ProductDistribution
        dist = ProductDistribution(
            a=Normal(loc=0.0, scale=1.0, name="a"),
            b=Normal(loc=0.0, scale=1.0, name="b"),
        )
        s_one = dist._sample_one(k)
        s_zero = dist._sample(k, ())
        assert type(s_one) is type(s_zero)
        assert s_one.fields == s_zero.fields
        for f in s_one.fields:
            assert float(s_one[f]) == float(s_zero[f])

    def test_joint_empirical_return_types(self):
        import numpy as np
        from probpipe import Record
        from probpipe.core._record_array import NumericRecordArray
        # Build a small JointEmpirical from stored per-component samples
        je = JointEmpirical(
            x=np.asarray([[1.0], [2.0], [3.0]]),
            y=np.asarray([[0.5], [1.5], [2.5]]),
        )
        k = jax.random.PRNGKey(0)
        assert isinstance(je._sample(k, ()), Record)
        assert isinstance(je._sample(k, (4,)), NumericRecordArray)
        assert je._sample(k, (4,)).batch_shape == (4,)

    def test_joint_gaussian_return_types(self):
        from probpipe import Record
        from probpipe.core._record_array import NumericRecordArray
        jg = JointGaussian(
            x=1, y=1,
            mean=jnp.zeros(2),
            cov=jnp.eye(2),
        )
        k = jax.random.PRNGKey(0)
        assert isinstance(jg._sample(k, ()), Record)
        assert isinstance(jg._sample(k, (5,)), NumericRecordArray)
        assert jg._sample(k, (5,)).batch_shape == (5,)


class TestMixtureSamplingDispatch:
    """``_MixtureSampling._sample`` dispatches on component sample type.

    Numeric components → Array; Record components → RecordArray;
    incompatible types → clear TypeError.
    """

    def test_array_components_stacked(self):
        import jax.numpy as jnp
        from probpipe.core._broadcast_distributions import _make_mixture_marginal
        comps = [Normal(loc=0.0, scale=1.0, name=f"c{i}") for i in range(3)]
        mix = _make_mixture_marginal(comps)
        s = mix._sample(jax.random.PRNGKey(0), (4,))
        assert isinstance(s, jnp.ndarray)
        assert s.shape == (4,)

    def test_record_components_stacked_as_record_array(self):
        from probpipe.core._broadcast_distributions import _make_mixture_marginal
        from probpipe.core._record_array import RecordArray, NumericRecordArray
        from probpipe import Record
        comps = [
            ProductDistribution(
                a=Normal(loc=float(i), scale=1.0, name="a"),
                b=Normal(loc=float(-i), scale=1.0, name="b"),
            )
            for i in range(3)
        ]
        mix = _make_mixture_marginal(comps)
        # Batched → RecordArray
        s_batched = mix._sample(jax.random.PRNGKey(0), (5,))
        assert isinstance(s_batched, (RecordArray, NumericRecordArray))
        assert s_batched.batch_shape == (5,)
        # Unbatched → Record (first row of the stacked RecordArray)
        s_one = mix._sample(jax.random.PRNGKey(0), ())
        assert isinstance(s_one, Record)
