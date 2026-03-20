"""Tests for JointEmpirical."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    JointEmpirical,
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
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        assert isinstance(je, JointEmpirical)
        assert isinstance(je, JointDistribution)

    def test_component_names(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0]),
            y=jnp.array([3.0, 4.0]),
        )
        assert je.component_names == ("x", "y")

    def test_n_property(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        assert je.n == 3

    def test_event_shape(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0]),  # scalar → dim 1
            y=jnp.array([[1.0, 2.0], [3.0, 4.0]]),  # dim 2
        )
        # total_dim = 1 + 2 = 3
        assert je.event_shape == (3,)

    def test_raises_on_mismatched_n(self):
        with pytest.raises(ValueError, match="same number of samples"):
            JointEmpirical(
                x=jnp.array([1.0, 2.0]),
                y=jnp.array([1.0, 2.0, 3.0]),
            )

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            JointEmpirical()

    def test_raises_on_scalar(self):
        with pytest.raises(ValueError, match="at least 1 dimension"):
            JointEmpirical(x=jnp.array(1.0))

    def test_with_weights(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
            weights=jnp.array([0.2, 0.3, 0.5]),
        )
        assert not je.is_uniform
        np.testing.assert_allclose(jnp.sum(je.weights), 1.0, atol=1e-5)

    def test_with_log_weights(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
            log_weights=jnp.array([0.0, 1.0, 2.0]),
        )
        assert not je.is_uniform

    def test_uniform_default(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        assert je.is_uniform

    def test_weights_and_log_weights_exclusive(self):
        with pytest.raises(ValueError, match="either weights or log_weights"):
            JointEmpirical(
                x=jnp.array([1.0, 2.0]),
                weights=jnp.array([0.5, 0.5]),
                log_weights=jnp.array([0.0, 0.0]),
            )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:

    def test_sample_flat_shape(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        key = jax.random.PRNGKey(0)
        s = je.sample(key)
        assert s.shape == (2,)

    def test_sample_flat_batch(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        key = jax.random.PRNGKey(1)
        s = je.sample(key, (10,))
        assert s.shape == (10, 2)

    def test_sample_structured(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        key = jax.random.PRNGKey(2)
        structured = je.sample_structured(key, (5,))
        assert set(structured.keys()) == {"x", "y"}
        # 1D inputs have event_shape=() per TFP convention
        assert structured["x"].shape == (5,)
        assert structured["y"].shape == (5,)

    def test_joint_resampling_preserves_correlation(self):
        """Rows should be resampled jointly."""
        x = jnp.array([10.0, 20.0, 30.0])
        y = jnp.array([100.0, 200.0, 300.0])  # y = 10 * x
        je = JointEmpirical(x=x, y=y)

        key = jax.random.PRNGKey(3)
        structured = je.sample_structured(key, (100,))

        # For each sample, y should be 10 * x
        np.testing.assert_allclose(
            structured["y"], 10.0 * structured["x"], atol=1e-5
        )

    def test_multidimensional_components(self):
        je = JointEmpirical(
            a=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            b=jnp.array([10.0, 20.0, 30.0]),
        )
        key = jax.random.PRNGKey(4)
        structured = je.sample_structured(key, (5,))
        assert structured["a"].shape == (5, 2)
        assert structured["b"].shape == (5,)

    def test_weighted_sampling(self):
        """Heavily weighted sample should appear most often."""
        je = JointEmpirical(
            x=jnp.array([0.0, 100.0]),
            weights=jnp.array([0.01, 0.99]),
        )
        key = jax.random.PRNGKey(5)
        structured = je.sample_structured(key, (200,))
        # Most samples should be close to 100
        assert float(jnp.mean(structured["x"])) > 90.0


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

class TestViews:

    def test_getitem_returns_view(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0]),
            y=jnp.array([3.0, 4.0]),
        )
        view = je["x"]
        assert isinstance(view, DistributionView)

    def test_view_sample(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0]),
            y=jnp.array([3.0, 4.0]),
        )
        key = jax.random.PRNGKey(10)
        s = je["x"].sample(key, (5,))
        assert s.shape == (5,)

    def test_marginal_is_empirical(self):
        """Component distributions should be EmpiricalDistribution."""
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        assert isinstance(je.components["x"], EmpiricalDistribution)
        assert isinstance(je.components["y"], EmpiricalDistribution)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

class TestMoments:

    def test_mean_uniform(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 3.0]),
            y=jnp.array([10.0, 20.0]),
        )
        m = je.mean()
        np.testing.assert_allclose(m, jnp.array([2.0, 15.0]), atol=1e-5)

    def test_mean_weighted(self):
        je = JointEmpirical(
            x=jnp.array([0.0, 10.0]),
            weights=jnp.array([0.25, 0.75]),
        )
        m = je.mean()
        np.testing.assert_allclose(m, jnp.array([7.5]), atol=1e-5)

    def test_variance_uniform(self):
        je = JointEmpirical(
            x=jnp.array([0.0, 2.0]),
        )
        v = je.variance()
        np.testing.assert_allclose(v, jnp.array([1.0]), atol=1e-5)

    def test_log_prob_finite(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        key = jax.random.PRNGKey(20)
        s = je.sample(key)
        lp = je.log_prob(s)
        assert jnp.isfinite(lp)


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------

class TestBroadcasting:

    def test_views_in_workflow(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([10.0, 20.0, 30.0]),
        )

        def add(a: float, b: float) -> float:
            return a + b

        wf = Workflow(
            func=add,
            broadcast_backend="loop",
            n_broadcast_samples=30,
            seed=42,
        )
        result = wf(a=je["x"], b=je["y"])
        assert isinstance(result, EmpiricalDistribution)
        # Joint resampling means y = 10*x, so a+b = 11*x
        # Mean of x is 2.0, so mean of a+b should be ~22
        mean_val = float(jnp.mean(result.samples))
        assert abs(mean_val - 22.0) < 5.0
