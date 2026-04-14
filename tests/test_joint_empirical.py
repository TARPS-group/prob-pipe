"""Tests for JointEmpirical."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    JointEmpirical,
    EmpiricalDistribution,
    Record,
    RecordDistribution,
)
from probpipe.core._record_distribution import _RecordDistributionView
from probpipe.core.node import WorkflowFunction
from probpipe import condition_on, log_prob, mean, sample, variance


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
        assert isinstance(je, RecordDistribution)

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

    def test_event_shapes(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0]),  # scalar → event_shape ()
            y=jnp.array([[1.0, 2.0], [3.0, 4.0]]),  # event_shape (2,)
        )
        assert je.event_shapes == {"x": (), "y": (2,)}
        assert je.event_size == 3  # 1 + 2

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

    def test_sample_returns_values(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        key = jax.random.PRNGKey(0)
        s = sample(je, key=key)
        assert isinstance(s, Record)
        assert set(s.fields()) == {"x", "y"}
        assert s["x"].shape == ()
        assert s["y"].shape == ()

    def test_sample_batch(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        key = jax.random.PRNGKey(1)
        s = sample(je, key=key, sample_shape=(10,))
        assert isinstance(s, Record)
        assert s["x"].shape == (10,)
        assert s["y"].shape == (10,)

    def test_joint_resampling_preserves_correlation(self):
        """Rows should be resampled jointly."""
        x = jnp.array([10.0, 20.0, 30.0])
        y = jnp.array([100.0, 200.0, 300.0])  # y = 10 * x
        je = JointEmpirical(x=x, y=y)

        key = jax.random.PRNGKey(3)
        s = sample(je, key=key, sample_shape=(100,))

        # For each sample, y should be 10 * x
        np.testing.assert_allclose(s["y"], 10.0 * s["x"], atol=1e-5)

    def test_multidimensional_components(self):
        je = JointEmpirical(
            a=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            b=jnp.array([10.0, 20.0, 30.0]),
        )
        key = jax.random.PRNGKey(4)
        s = sample(je, key=key, sample_shape=(5,))
        assert s["a"].shape == (5, 2)
        assert s["b"].shape == (5,)

    def test_weighted_sampling(self):
        """Weighted sample mean must match E[X] = 0*0.01 + 100*0.99 = 99."""
        je = JointEmpirical(
            x=jnp.array([0.0, 100.0]),
            weights=jnp.array([0.01, 0.99]),
        )
        key = jax.random.PRNGKey(5)
        s = sample(je, key=key, sample_shape=(50_000,))
        # MC std ~ sqrt(0.01*0.99*100^2) / sqrt(50_000) ~ 0.044
        np.testing.assert_allclose(
            float(jnp.mean(s["x"])), 99.0, atol=0.15,
        )


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
        assert isinstance(view, _RecordDistributionView)

    def test_view_sample(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0]),
            y=jnp.array([3.0, 4.0]),
        )
        key = jax.random.PRNGKey(10)
        s = sample(je["x"], key=key, sample_shape=(5,))
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
        m = mean(je)
        assert isinstance(m, Record)
        np.testing.assert_allclose(m["x"], 2.0, atol=1e-5)
        np.testing.assert_allclose(m["y"], 15.0, atol=1e-5)

    def test_mean_weighted(self):
        je = JointEmpirical(
            x=jnp.array([0.0, 10.0]),
            weights=jnp.array([0.25, 0.75]),
        )
        m = mean(je)
        assert isinstance(m, Record)
        np.testing.assert_allclose(m["x"], 7.5, atol=1e-5)

    def test_variance_uniform(self):
        je = JointEmpirical(
            x=jnp.array([0.0, 2.0]),
        )
        v = variance(je)
        assert isinstance(v, Record)
        np.testing.assert_allclose(v["x"], 1.0, atol=1e-5)



# ---------------------------------------------------------------------------
# LogProb
# ---------------------------------------------------------------------------

class TestLogProb:

    def test_isinstance_log_prob(self):
        from probpipe import SupportsLogProb
        je = JointEmpirical(x=jnp.array([1.0, 2.0, 3.0]),
                            y=jnp.array([4.0, 5.0, 6.0]))
        assert isinstance(je, SupportsLogProb)

    def test_log_prob_finite(self):
        je = JointEmpirical(x=jnp.array([1.0, 2.0, 3.0]),
                            y=jnp.array([4.0, 5.0, 6.0]))
        s = sample(je, key=jax.random.PRNGKey(0))
        lp = log_prob(je, s)
        assert jnp.isfinite(lp)


# ---------------------------------------------------------------------------
# Flatten / Unflatten
# ---------------------------------------------------------------------------

class TestFlattenUnflatten:

    def test_flatten_roundtrip(self):
        je = JointEmpirical(
            a=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            b=jnp.array([10.0, 20.0, 30.0]),
        )
        key = jax.random.PRNGKey(30)
        s = sample(je, key=key, sample_shape=(5,))
        flat = je.flatten_value(s)
        assert flat.shape == (5, 3)  # 2 + 1 = 3
        unflat = je.unflatten_value(flat)
        np.testing.assert_allclose(unflat["a"], s["a"], atol=1e-6)
        np.testing.assert_allclose(unflat["b"], s["b"], atol=1e-6)


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------

class TestConditionOn:

    def test_condition_on_removes_component(self):
        """Conditioning removes the specified component."""
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        cond = condition_on(je, x=jnp.array(1.0))
        assert cond.component_names == ("y",)
        assert isinstance(cond, JointEmpirical)

    def test_conditioned_sample_shape(self):
        """Conditioned distribution samples the remaining components."""
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        cond = condition_on(je, x=jnp.array(1.0))
        s = sample(cond, key=jax.random.PRNGKey(0), sample_shape=(5,))
        assert set(s.fields()) == {"y"}
        assert s["y"].shape == (5,)

    def test_condition_on_preserves_correlation(self):
        """Row-wise correlation is preserved after conditioning."""
        x = jnp.array([10.0, 20.0, 30.0])
        y = jnp.array([100.0, 200.0, 300.0])  # y = 10 * x
        z = jnp.array([1.0, 2.0, 3.0])        # z = x / 10
        je = JointEmpirical(x=x, y=y, z=z)
        cond = condition_on(je, x=jnp.array(0.0))  # remove x
        s = sample(cond, key=jax.random.PRNGKey(1), sample_shape=(100,))
        # y and z still come from the same row, so y = 100 * z
        np.testing.assert_allclose(s["y"], 100.0 * s["z"], atol=1e-5)

    def test_condition_on_weighted(self):
        """Conditioning on weighted JointEmpirical preserves weights."""
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
            log_weights=jnp.array([0.0, 1.0, 2.0]),
        )
        cond = condition_on(je, x=jnp.array(1.0))
        assert not cond.is_uniform
        assert cond.n == 3

    def test_condition_on_uniform_stays_uniform(self):
        """Conditioning on uniform JointEmpirical stays uniform."""
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        cond = condition_on(je, x=jnp.array(1.0))
        assert cond.is_uniform

    def test_condition_on_provenance(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0]),
            y=jnp.array([3.0, 4.0]),
        )
        cond = condition_on(je, x=jnp.array(1.0))
        assert cond.source is not None
        assert cond.source.operation == "condition_on"
        assert "x" in cond.source.metadata["conditioned"]

    def test_condition_on_unknown_raises(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0]),
            y=jnp.array([3.0, 4.0]),
        )
        with pytest.raises(KeyError, match="not found"):
            condition_on(je, z=jnp.array(1.0))

    def test_condition_on_all_raises(self):
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0]),
            y=jnp.array([3.0, 4.0]),
        )
        with pytest.raises(ValueError, match="Cannot condition on all"):
            condition_on(je, x=jnp.array(1.0), y=jnp.array(2.0))

    def test_condition_on_dict_for_leaf_raises(self):
        """Passing a dict value for a leaf component should raise TypeError."""
        je = JointEmpirical(
            x=jnp.array([1.0, 2.0]),
            y=jnp.array([3.0, 4.0]),
        )
        with pytest.raises(TypeError, match="component distribution"):
            condition_on(je, x={"sub": jnp.array(1.0)})


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

        wf = WorkflowFunction(
            func=add,
            vectorize="loop",
            n_broadcast_samples=30,
            seed=42,
        )
        result = wf(a=je["x"], b=je["y"])
        assert hasattr(result, "samples")
        # Joint resampling means y = 10*x, so a+b = 11*x
        # Mean of x is 2.0, so mean of a+b should be ~22
        mean_val = float(jnp.mean(result.samples))
        assert abs(mean_val - 22.0) < 5.0
