"""Tests for standalone operations in probpipe.core.ops."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Normal,
    MultivariateNormal,
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
    return EmpiricalDistribution(samples)


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

    def test_mc_fallback_for_non_exact(self):
        """Distributions without _mean() fall back to expectation()."""
        from probpipe import TransformedDistribution
        import tensorflow_probability.substrates.jax.bijectors as tfb

        base = Normal(loc=0.0, scale=1.0)
        td = TransformedDistribution(base, tfb.Exp())
        # TransformedDistribution doesn't define _mean, so should use MC
        m = ops.mean(td)
        # E[exp(X)] for X ~ Normal(0, 1) = exp(0.5) ≈ 1.6487
        np.testing.assert_allclose(float(m), 1.6487, atol=0.2)


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

    def test_condition_type_error(self, normal):
        with pytest.raises(TypeError, match="does not support conditioning"):
            ops.condition_on(normal, x=jnp.array(1.0))


# ---------------------------------------------------------------------------
# WorkflowFunction wrappers
# ---------------------------------------------------------------------------

class TestWorkflowFunctionWrappers:
    """Verify wf_* accessors are WorkflowFunction instances."""

    def test_all_wf_ops_exist(self):
        from probpipe.core.node import WorkflowFunction
        wf_names = [
            "wf_sample", "wf_log_prob", "wf_prob",
            "wf_unnormalized_log_prob", "wf_unnormalized_prob",
            "wf_mean", "wf_variance", "wf_cov", "wf_expectation", "wf_condition_on",
        ]
        for name in wf_names:
            wf = getattr(ops, name)
            assert isinstance(wf, WorkflowFunction), f"{name} is not a WorkflowFunction"


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
