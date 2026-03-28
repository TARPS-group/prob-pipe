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
    SupportsNamedComponents,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal():
    return Normal(loc=0.0, scale=1.0)


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
    return ProductDistribution(x=Normal(0, 1), y=Normal(1, 2))


# ---------------------------------------------------------------------------
# SupportsSampling
# ---------------------------------------------------------------------------

class TestSupportsSampling:
    """All distributions should support sampling."""

    @pytest.mark.parametrize("dist_cls,kwargs", [
        (Normal, {"loc": 0.0, "scale": 1.0}),
        (Beta, {"alpha": 2.0, "beta": 5.0}),
        (Gamma, {"concentration": 3.0, "rate": 1.0}),
        (Bernoulli, {"probs": 0.5}),
        (MultivariateNormal, {"loc": jnp.zeros(2), "cov": jnp.eye(2)}),
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
        (Normal, {"loc": 0.0, "scale": 1.0}),
        (Beta, {"alpha": 2.0, "beta": 5.0}),
        (MultivariateNormal, {"loc": jnp.zeros(2), "cov": jnp.eye(2)}),
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

    def test_empirical(self, empirical):
        assert isinstance(empirical, SupportsMean)
        assert isinstance(empirical, SupportsVariance)
        assert isinstance(empirical, SupportsCovariance)

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
            x=Normal(0, 1),
            y=lambda x: Normal(loc=x, scale=1.0),
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
# SupportsNamedComponents
# ---------------------------------------------------------------------------

class TestSupportsNamedComponents:
    def test_product_distribution(self, joint):
        assert isinstance(joint, SupportsNamedComponents)

    def test_normal_not_named_components(self, normal):
        assert not isinstance(normal, SupportsNamedComponents)


# ---------------------------------------------------------------------------
# Orchestration hints
# ---------------------------------------------------------------------------

class TestOrchestrationHints:
    def test_default_sampling_cost(self, normal):
        assert normal._sampling_cost == "low"

    def test_default_preferred_orchestration(self, normal):
        assert normal._preferred_orchestration is None
