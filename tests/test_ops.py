"""Tests for standalone operations in probpipe.core.ops."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Normal,
    MultivariateNormal,
    ArrayEmpiricalDistribution,
    EmpiricalDistribution,
    BootstrapDistribution,
    ProductDistribution,
    SequentialJointDistribution,
)
from probpipe.core import ops


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal():
    return Normal(loc=2.0, scale=0.5)


@pytest.fixture
def mvn():
    return MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3))


@pytest.fixture
def empirical():
    samples = jax.random.normal(jax.random.PRNGKey(0), (200, 2))
    return ArrayEmpiricalDistribution(samples)


@pytest.fixture
def joint():
    return ProductDistribution(x=Normal(0, 1), y=Normal(1, 2))


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

class TestSample:
    def test_sample_scalar(self, normal):
        s = ops.sample(normal, key=jax.random.PRNGKey(0))
        assert s.shape == ()

    def test_sample_with_shape(self, normal):
        s = ops.sample(normal, key=jax.random.PRNGKey(0), sample_shape=(50,))
        assert s.shape == (50,)

    def test_sample_mvn(self, mvn):
        s = ops.sample(mvn, key=jax.random.PRNGKey(0), sample_shape=(10,))
        assert s.shape == (10, 3)

    def test_sample_empirical(self, empirical):
        s = ops.sample(empirical, key=jax.random.PRNGKey(0), sample_shape=(5,))
        assert s.shape == (5, 2)

    def test_sample_type_error(self):
        with pytest.raises(TypeError, match="does not support sampling"):
            ops.sample("not a distribution")


# ---------------------------------------------------------------------------
# log_prob
# ---------------------------------------------------------------------------

class TestLogProb:
    def test_log_prob_scalar(self, normal):
        lp = ops.log_prob(normal, jnp.array(2.0))
        expected = -0.5 * jnp.log(2 * jnp.pi * 0.25)
        np.testing.assert_allclose(float(lp), float(expected), atol=1e-5)

    def test_log_prob_mvn(self, mvn):
        lp = ops.log_prob(mvn, jnp.zeros(3))
        assert lp.shape == ()

    def test_log_prob_type_error(self):
        with pytest.raises(TypeError, match="does not support log_prob"):
            ops.log_prob("not a distribution", 1.0)


# ---------------------------------------------------------------------------
# prob
# ---------------------------------------------------------------------------

class TestProb:
    def test_prob_equals_exp_log_prob(self, normal):
        x = jnp.array(2.0)
        p = ops.prob(normal, x)
        lp = ops.log_prob(normal, x)
        np.testing.assert_allclose(float(p), float(jnp.exp(lp)), atol=1e-6)


# ---------------------------------------------------------------------------
# unnormalized_log_prob
# ---------------------------------------------------------------------------

class TestUnnormalizedLogProb:
    def test_equals_log_prob_for_normalized(self, normal):
        x = jnp.array(1.5)
        ulp = ops.unnormalized_log_prob(normal, x)
        lp = ops.log_prob(normal, x)
        np.testing.assert_allclose(float(ulp), float(lp), atol=1e-6)

    def test_unnormalized_prob_equals_exp(self, normal):
        x = jnp.array(1.5)
        up = ops.unnormalized_prob(normal, x)
        ulp = ops.unnormalized_log_prob(normal, x)
        np.testing.assert_allclose(float(up), float(jnp.exp(ulp)), atol=1e-6)


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------

class TestMean:
    def test_exact_mean_normal(self, normal):
        m = ops.mean(normal)
        np.testing.assert_allclose(float(m), 2.0, atol=1e-5)

    def test_exact_mean_mvn(self, mvn):
        m = ops.mean(mvn)
        np.testing.assert_allclose(m, jnp.zeros(3), atol=1e-5)

    def test_exact_mean_empirical(self, empirical):
        m = ops.mean(empirical)
        assert m.shape == (2,)

    def test_exact_mean_bootstrap(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BootstrapDistribution(evals)
        np.testing.assert_allclose(float(ops.mean(bd)), 3.0)

    def test_raises_without_supports_mean(self):
        """mean op raises TypeError for distributions without SupportsMean."""
        from probpipe.core.protocols import SupportsSampling, SupportsExpectation
        from probpipe.core.distribution import _vmap_sample, _mc_expectation, ArrayDistribution

        class NoMeanDist(ArrayDistribution, SupportsSampling, SupportsExpectation):
            _sampling_cost = "low"
            _preferred_orchestration = None
            @property
            def event_shape(self):
                return ()
            def _sample_one(self, key):
                return jax.random.normal(key)
            def _sample(self, key, sample_shape=()):
                return _vmap_sample(self, key, sample_shape)
            def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
                return _mc_expectation(self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist)

        with pytest.raises(TypeError, match="does not support mean"):
            ops.mean(NoMeanDist())


# ---------------------------------------------------------------------------
# variance
# ---------------------------------------------------------------------------

class TestVariance:
    def test_exact_variance_normal(self, normal):
        v = ops.variance(normal)
        np.testing.assert_allclose(float(v), 0.25, atol=1e-5)

    def test_exact_variance_empirical(self, empirical):
        v = ops.variance(empirical)
        assert v.shape == (2,)


# ---------------------------------------------------------------------------
# cov
# ---------------------------------------------------------------------------

class TestCov:
    def test_exact_cov_empirical(self, empirical):
        c = ops.cov(empirical)
        assert c.shape == (2, 2)
        np.testing.assert_allclose(c, c.T, atol=1e-5)


# ---------------------------------------------------------------------------
# expectation
# ---------------------------------------------------------------------------

class TestExpectation:
    def test_expectation_identity(self, normal):
        result = ops.expectation(
            normal, lambda x: x,
            key=jax.random.PRNGKey(0),
            num_evaluations=5000,
            return_dist=False,
        )
        np.testing.assert_allclose(float(result), 2.0, atol=0.1)

    def test_expectation_returns_bootstrap(self, normal):
        result = ops.expectation(
            normal, lambda x: x,
            key=jax.random.PRNGKey(0),
            num_evaluations=500,
            return_dist=True,
        )
        assert isinstance(result, BootstrapDistribution)


# ---------------------------------------------------------------------------
# condition_on
# ---------------------------------------------------------------------------

class TestConditionOn:
    def test_condition_product(self, joint):
        conditioned = ops.condition_on(joint, x=jnp.array(2.0))
        assert conditioned.component_names == ("y",)

    def test_condition_sequential(self):
        sjd = SequentialJointDistribution(
            x=Normal(0, 1),
            y=lambda x: Normal(loc=x, scale=1.0),
        )
        conditioned = ops.condition_on(sjd, x=jnp.array(3.0))
        assert conditioned.component_names == ("y",)

    def test_condition_type_error(self):
        """Objects with no protocols raise TypeError."""
        with pytest.raises(TypeError, match="does not support conditioning"):
            ops.condition_on("not_a_distribution", jnp.array(1.0))


# ---------------------------------------------------------------------------
# WorkflowFunction routing
# ---------------------------------------------------------------------------

class TestWorkflowFunctionRouting:
    """Verify public ops are WorkflowFunction instances."""

    def test_ops_are_workflow_functions(self):
        from probpipe.core.node import WorkflowFunction
        for name in ops.__all__:
            fn = getattr(ops, name)
            assert isinstance(fn, WorkflowFunction), (
                f"ops.{name} is not a WorkflowFunction"
            )

    def test_public_ops_are_callable(self):
        for name in ops.__all__:
            fn = getattr(ops, name)
            assert callable(fn), f"ops.{name} is not callable"

    def test_ops_accept_positional_args(self, normal):
        """Ops should accept positional arguments (not just keyword)."""
        # log_prob(dist, value) — both positional
        lp = ops.log_prob(normal, jnp.array(2.0))
        assert lp.shape == ()

        # mean(dist) — single positional
        m = ops.mean(normal)
        assert m.shape == ()

    def test_positional_and_keyword_duplicate_raises(self, normal):
        """Passing the same arg both positionally and as keyword raises."""
        with pytest.raises(TypeError, match="multiple values"):
            ops.log_prob(normal, value=jnp.array(1.0), dist=normal)


# ---------------------------------------------------------------------------
# Top-level imports work
# ---------------------------------------------------------------------------

class TestTopLevelImports:
    """Verify ops are importable from probpipe top level."""

    def test_import_sample(self):
        from probpipe import sample as s
        assert callable(s)

    def test_import_mean(self):
        from probpipe import mean as m
        assert callable(m)

    def test_import_log_prob(self):
        from probpipe import log_prob as lp
        assert callable(lp)

    def test_import_condition_on(self):
        from probpipe import condition_on as co
        assert callable(co)
