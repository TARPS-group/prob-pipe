"""Tests for JointGaussian."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    JointGaussian,
    MultivariateNormal,
    DistributionView,
    EmpiricalDistribution,
    JointDistribution,
)
from probpipe.core.node import Workflow


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_basic_construction(self):
        jg = JointGaussian(
            mean=jnp.array([0.0, 1.0]),
            cov=jnp.eye(2),
            x=1,
            y=1,
        )
        assert isinstance(jg, JointGaussian)
        assert isinstance(jg, JointDistribution)

    def test_component_names(self):
        jg = JointGaussian(
            mean=jnp.array([0.0, 1.0, 2.0]),
            cov=jnp.eye(3),
            a=1,
            bc=2,
        )
        assert jg.component_names == ("a", "bc")

    def test_event_shape(self):
        jg = JointGaussian(
            mean=jnp.array([0.0, 1.0, 2.0, 3.0]),
            cov=jnp.eye(4),
            x=1,
            yz=3,
        )
        assert jg.event_shape == (4,)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            JointGaussian(mean=jnp.array([]), cov=jnp.zeros((0, 0)))

    def test_raises_on_mean_shape_mismatch(self):
        with pytest.raises(ValueError, match="mean shape"):
            JointGaussian(
                mean=jnp.array([0.0, 1.0]),
                cov=jnp.eye(3),
                x=1,
                y=2,
            )

    def test_raises_on_cov_shape_mismatch(self):
        with pytest.raises(ValueError, match="cov shape"):
            JointGaussian(
                mean=jnp.array([0.0, 1.0, 2.0]),
                cov=jnp.eye(2),
                x=1,
                y=2,
            )

    def test_mean_vector_property(self):
        m = jnp.array([1.0, 2.0, 3.0])
        jg = JointGaussian(mean=m, cov=jnp.eye(3), a=1, b=2)
        np.testing.assert_allclose(jg.mean_vector, m, atol=1e-6)

    def test_covariance_property(self):
        c = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        jg = JointGaussian(mean=jnp.zeros(2), cov=c, x=1, y=1)
        np.testing.assert_allclose(jg.covariance, c, atol=1e-6)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:

    def test_sample_flat_shape(self):
        jg = JointGaussian(
            mean=jnp.array([0.0, 1.0]),
            cov=jnp.eye(2),
            x=1, y=1,
        )
        key = jax.random.PRNGKey(0)
        s = jg.sample(key)
        assert s.shape == (2,)

    def test_sample_flat_batch(self):
        jg = JointGaussian(
            mean=jnp.array([0.0, 1.0]),
            cov=jnp.eye(2),
            x=1, y=1,
        )
        key = jax.random.PRNGKey(1)
        s = jg.sample(key, (10,))
        assert s.shape == (10, 2)

    def test_sample_structured(self):
        jg = JointGaussian(
            mean=jnp.array([0.0, 1.0, 2.0]),
            cov=jnp.eye(3),
            a=1, bc=2,
        )
        key = jax.random.PRNGKey(2)
        structured = jg.sample_structured(key, (5,))
        assert structured["a"].shape == (5, 1)
        assert structured["bc"].shape == (5, 2)

    def test_sample_mean_convergence(self):
        m = jnp.array([3.0, -1.0])
        jg = JointGaussian(mean=m, cov=0.01 * jnp.eye(2), x=1, y=1)
        key = jax.random.PRNGKey(3)
        s = jg.sample(key, (1000,))
        np.testing.assert_allclose(jnp.mean(s, axis=0), m, atol=0.1)

    def test_sample_preserves_cross_covariance(self):
        """Strong positive cross-covariance should appear in samples."""
        cov = jnp.array([[1.0, 0.9], [0.9, 1.0]])
        jg = JointGaussian(mean=jnp.zeros(2), cov=cov, x=1, y=1)
        key = jax.random.PRNGKey(4)
        s = jg.sample(key, (2000,))
        empirical_corr = jnp.corrcoef(s.T)[0, 1]
        assert float(empirical_corr) > 0.7


# ---------------------------------------------------------------------------
# log_prob
# ---------------------------------------------------------------------------

class TestLogProb:

    def test_log_prob_shape(self):
        jg = JointGaussian(
            mean=jnp.array([0.0, 1.0]),
            cov=jnp.eye(2),
            x=1, y=1,
        )
        key = jax.random.PRNGKey(10)
        s = jg.sample(key, (5,))
        lps = jg.log_prob(s)
        assert lps.shape == (5,)

    def test_log_prob_matches_mvn(self):
        """log_prob should match a full MultivariateNormal."""
        m = jnp.array([1.0, 2.0, 3.0])
        c = jnp.array([[2.0, 0.5, 0.1],
                        [0.5, 1.0, 0.3],
                        [0.1, 0.3, 1.5]])
        jg = JointGaussian(mean=m, cov=c, a=1, b=2)
        mvn = MultivariateNormal(loc=m, cov=c)

        key = jax.random.PRNGKey(11)
        s = jg.sample(key, (10,))
        lp_jg = jg.log_prob(s)
        lp_mvn = mvn.log_prob(s)
        np.testing.assert_allclose(lp_jg, lp_mvn, atol=1e-4)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

class TestMoments:

    def test_mean(self):
        m = jnp.array([1.0, 2.0, 3.0])
        jg = JointGaussian(mean=m, cov=jnp.eye(3), a=1, b=2)
        np.testing.assert_allclose(jg.mean(), m, atol=1e-6)

    def test_variance(self):
        c = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        jg = JointGaussian(mean=jnp.zeros(2), cov=c, x=1, y=1)
        np.testing.assert_allclose(jg.variance(), jnp.array([2.0, 3.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

class TestViews:

    def test_marginal_is_mvn(self):
        jg = JointGaussian(
            mean=jnp.array([0.0, 1.0, 2.0]),
            cov=jnp.eye(3),
            a=1, bc=2,
        )
        assert isinstance(jg.components["a"], MultivariateNormal)
        assert isinstance(jg.components["bc"], MultivariateNormal)

    def test_marginal_mean_correct(self):
        jg = JointGaussian(
            mean=jnp.array([5.0, 10.0, 15.0]),
            cov=jnp.eye(3),
            x=1, yz=2,
        )
        np.testing.assert_allclose(
            jg.components["x"].mean(), jnp.array([5.0]), atol=1e-6
        )
        np.testing.assert_allclose(
            jg.components["yz"].mean(), jnp.array([10.0, 15.0]), atol=1e-6
        )

    def test_marginal_cov_correct(self):
        cov = jnp.array([[2.0, 0.5, 0.1],
                          [0.5, 1.0, 0.3],
                          [0.1, 0.3, 1.5]])
        jg = JointGaussian(
            mean=jnp.zeros(3),
            cov=cov,
            x=1, yz=2,
        )
        np.testing.assert_allclose(
            jg.components["x"].cov, jnp.array([[2.0]]), atol=1e-5
        )
        np.testing.assert_allclose(
            jg.components["yz"].cov, jnp.array([[1.0, 0.3], [0.3, 1.5]]), atol=1e-5
        )


# ---------------------------------------------------------------------------
# condition_on
# ---------------------------------------------------------------------------

class TestConditionOn:

    def test_basic_conditioning(self):
        """Condition x=1 → should get a JointGaussian with only y."""
        cov = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        jg = JointGaussian(mean=jnp.zeros(2), cov=cov, x=1, y=1)
        cond = jg.condition_on(x=jnp.array([1.0]))
        assert isinstance(cond, JointGaussian)
        assert cond.component_names == ("y",)
        assert cond.event_shape == (1,)

    def test_conditional_mean(self):
        """Verify the Gaussian conditioning formula for the mean."""
        mu = jnp.array([0.0, 0.0])
        cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
        jg = JointGaussian(mean=mu, cov=cov, x=1, y=1)

        # Condition on x=2.0
        # mu_y|x = mu_y + Sigma_yx @ Sigma_xx^{-1} @ (x - mu_x)
        # = 0 + 0.8 * 1.0 * (2 - 0) = 1.6
        cond = jg.condition_on(x=jnp.array([2.0]))
        np.testing.assert_allclose(float(cond.mean()[0]), 1.6, atol=1e-5)

    def test_conditional_variance(self):
        """Verify the Gaussian conditioning formula for the variance."""
        cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
        jg = JointGaussian(mean=jnp.zeros(2), cov=cov, x=1, y=1)

        # Var_y|x = Sigma_yy - Sigma_yx @ Sigma_xx^{-1} @ Sigma_xy
        # = 1.0 - 0.8 * 1.0 * 0.8 = 0.36
        cond = jg.condition_on(x=jnp.array([0.0]))
        np.testing.assert_allclose(float(cond.variance()[0]), 0.36, atol=1e-5)

    def test_conditioning_reduces_dimensions(self):
        jg = JointGaussian(
            mean=jnp.array([0.0, 1.0, 2.0, 3.0]),
            cov=jnp.eye(4),
            a=1, b=1, c=2,
        )
        cond = jg.condition_on(a=jnp.array([0.0]))
        assert cond.event_shape == (3,)
        assert cond.component_names == ("b", "c")

    def test_conditioning_multiple(self):
        jg = JointGaussian(
            mean=jnp.array([0.0, 1.0, 2.0, 3.0]),
            cov=jnp.eye(4),
            a=1, b=1, c=2,
        )
        cond = jg.condition_on(a=jnp.array([0.0]), c=jnp.array([2.0, 3.0]))
        assert cond.event_shape == (1,)
        assert cond.component_names == ("b",)

    def test_raises_on_unknown_component(self):
        jg = JointGaussian(mean=jnp.zeros(2), cov=jnp.eye(2), x=1, y=1)
        with pytest.raises(KeyError, match="Unknown"):
            jg.condition_on(z=jnp.array([0.0]))

    def test_raises_on_conditioning_all(self):
        jg = JointGaussian(mean=jnp.zeros(2), cov=jnp.eye(2), x=1, y=1)
        with pytest.raises(ValueError, match="Cannot condition on all"):
            jg.condition_on(x=jnp.array([0.0]), y=jnp.array([0.0]))

    def test_provenance(self):
        jg = JointGaussian(mean=jnp.zeros(2), cov=jnp.eye(2), x=1, y=1)
        cond = jg.condition_on(x=jnp.array([0.0]))
        assert cond.source is not None
        assert cond.source.operation == "condition_on"
        assert "x" in cond.source.metadata["conditioned"]

    def test_conditional_sampling(self):
        """Draw samples from conditional and verify they're centered correctly."""
        cov = jnp.array([[1.0, 0.9], [0.9, 1.0]])
        jg = JointGaussian(mean=jnp.zeros(2), cov=cov, x=1, y=1)
        cond = jg.condition_on(x=jnp.array([3.0]))
        # mu_y|x=3 = 0 + 0.9 * 1 * 3 = 2.7
        key = jax.random.PRNGKey(20)
        s = cond.sample(key, (1000,))
        np.testing.assert_allclose(float(jnp.mean(s)), 2.7, atol=0.2)


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------

class TestBroadcasting:

    def test_views_in_workflow(self):
        cov = jnp.array([[1.0, 0.9], [0.9, 1.0]])
        jg = JointGaussian(mean=jnp.zeros(2), cov=cov, x=1, y=1)

        def add(a: float, b: float) -> float:
            return a + b

        wf = Workflow(
            func=add,
            broadcast_backend="loop",
            n_broadcast_samples=50,
            seed=42,
        )
        result = wf(a=jg["x"], b=jg["y"])
        assert isinstance(result, EmpiricalDistribution)
        # Both ~ N(0,1) with corr=0.9, so sum ~ N(0, 1+1+2*0.9) = N(0, 2.8)
        assert abs(float(jnp.mean(result.samples))) < 1.5


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr(self):
        jg = JointGaussian(
            mean=jnp.zeros(3),
            cov=jnp.eye(3),
            x=1, yz=2,
            name="my_gauss",
        )
        r = repr(jg)
        assert "JointGaussian" in r
        assert "x=1" in r
        assert "yz=2" in r
        assert "my_gauss" in r
