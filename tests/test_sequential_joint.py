"""Tests for SequentialJointDistribution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Normal,
    MultivariateNormal,
    SequentialJointDistribution,
    DistributionView,
    ConditionedComponent,
    EmpiricalDistribution,
    JointDistribution,
)
from probpipe.core.node import Workflow


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_basic_construction(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        assert isinstance(joint, SequentialJointDistribution)
        assert isinstance(joint, JointDistribution)

    def test_component_names(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        assert joint.component_names == ("z", "x")

    def test_event_shape(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        # Two scalars → total_dim = 2
        assert joint.event_shape == (2,)

    def test_three_components(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
            y=lambda z, x: Normal(loc=z + x, scale=0.1),
        )
        assert joint.event_shape == (3,)
        assert joint.component_names == ("z", "x", "y")

    def test_independent_root(self):
        """All root distributions (no callables) should work."""
        joint = SequentialJointDistribution(
            a=Normal(loc=0.0, scale=1.0),
            b=Normal(loc=1.0, scale=2.0),
        )
        assert joint.event_shape == (2,)

    def test_raises_on_invalid_dependency(self):
        with pytest.raises(ValueError, match="not defined before"):
            SequentialJointDistribution(
                x=lambda z: Normal(loc=z, scale=1.0),  # z not yet defined
                z=Normal(loc=0.0, scale=1.0),
            )

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            SequentialJointDistribution()

    def test_named(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
            name="my_seq",
        )
        assert joint._name == "my_seq"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:

    def test_sample_flat_shape(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        key = jax.random.PRNGKey(0)
        s = joint.sample(key)
        assert s.shape == (2,)

    def test_sample_flat_batch(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        key = jax.random.PRNGKey(1)
        s = joint.sample(key, (10,))
        assert s.shape == (10, 2)

    def test_sample_structured(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        key = jax.random.PRNGKey(2)
        structured = joint.sample_structured(key, (20,))
        assert set(structured.keys()) == {"z", "x"}
        assert structured["z"].shape == (20,)
        assert structured["x"].shape == (20,)

    def test_conditional_sampling_correlation(self):
        """x depends on z, so they should be correlated."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.01),  # x ≈ z
        )
        key = jax.random.PRNGKey(3)
        structured = joint.sample_structured(key, (500,))
        # x should be very close to z
        np.testing.assert_allclose(
            structured["x"], structured["z"], atol=0.1
        )

    def test_auto_key(self):
        """sample() without explicit key should work."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        s = joint.sample(sample_shape=(5,))
        assert s.shape == (5, 2)


# ---------------------------------------------------------------------------
# log_prob
# ---------------------------------------------------------------------------

class TestLogProb:

    def test_log_prob_shape(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        key = jax.random.PRNGKey(10)
        samples = joint.sample(key, (5,))
        lps = joint.log_prob(samples)
        assert lps.shape == (5,)

    def test_log_prob_finite(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        key = jax.random.PRNGKey(11)
        samples = joint.sample(key, (10,))
        lps = joint.log_prob(samples)
        assert jnp.all(jnp.isfinite(lps))

    def test_log_prob_manual(self):
        """Verify log_prob = log p(z) + log p(x|z)."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        z_val = jnp.array(1.0)
        x_val = jnp.array(1.5)
        flat = jnp.array([z_val, x_val])
        lp = joint.log_prob(flat)

        lp_z = Normal(loc=0.0, scale=1.0).log_prob(z_val)
        lp_x_given_z = Normal(loc=z_val, scale=0.5).log_prob(x_val)
        expected = lp_z + lp_x_given_z
        np.testing.assert_allclose(float(lp), float(expected), atol=1e-5)

    def test_log_prob_three_components(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
            y=lambda z, x: Normal(loc=z + x, scale=0.1),
        )
        z_val, x_val, y_val = 0.5, 0.8, 1.2
        flat = jnp.array([z_val, x_val, y_val])
        lp = joint.log_prob(flat)

        expected = (
            float(Normal(loc=0.0, scale=1.0).log_prob(z_val))
            + float(Normal(loc=z_val, scale=0.5).log_prob(x_val))
            + float(Normal(loc=z_val + x_val, scale=0.1).log_prob(y_val))
        )
        np.testing.assert_allclose(float(lp), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# DistributionView
# ---------------------------------------------------------------------------

class TestDistributionView:

    def test_getitem_returns_view(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        view = joint["z"]
        assert isinstance(view, DistributionView)
        assert view._component_name == "z"

    def test_view_event_shape(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        assert joint["z"].event_shape == ()
        assert joint["x"].event_shape == ()

    def test_view_sample(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        key = jax.random.PRNGKey(20)
        s = joint["z"].sample(key, (5,))
        assert s.shape == (5,)


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------

class TestConditionOn:

    def test_condition_on_removes_conditioned_component(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        cond = joint.condition_on(z=jnp.array(2.0))
        assert "z" not in cond.components
        assert "x" in cond.components
        assert cond.component_names == ("x",)
        assert cond.event_shape == (1,)

    def test_conditioned_sampling_uses_observed_value(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        cond = joint.condition_on(z=jnp.array(3.0))
        key = jax.random.PRNGKey(30)
        structured = cond.sample_structured(key, (50,))
        # z should not be in structured (it's conditioned)
        assert "z" not in structured
        # x should be centered around 3.0 (since x = N(z=3, 0.5))
        assert abs(float(jnp.mean(structured["x"])) - 3.0) < 0.3

    def test_condition_on_downstream_effect(self):
        """Conditioning on z should make downstream x use the observed value."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.01),  # x ≈ z
        )
        cond = joint.condition_on(z=jnp.array(5.0))
        key = jax.random.PRNGKey(31)
        structured = cond.sample_structured(key, (100,))
        assert "z" not in structured
        # x should be very close to 5.0
        np.testing.assert_allclose(
            structured["x"], jnp.full((100,), 5.0), atol=0.1
        )

    def test_condition_on_provenance(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        cond = joint.condition_on(z=jnp.array(0.0))
        assert cond.source is not None
        assert cond.source.operation == "condition_on"

    def test_condition_on_unknown_raises(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        with pytest.raises(KeyError, match="Unknown"):
            joint.condition_on(nonexistent=jnp.array(0.0))

    def test_condition_on_non_root_removes_component(self):
        """Conditioning on a non-root removes it from components."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        cond = joint.condition_on(x=jnp.array(1.0))
        assert "x" not in cond.components
        assert "z" in cond.components
        assert cond.component_names == ("z",)

    def test_condition_on_non_root_with_unconditioned_parent_raises(self):
        """Sampling raises if a conditioned non-root has unconditioned parents."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        cond = joint.condition_on(x=jnp.array(1.0))
        with pytest.raises(NotImplementedError, match="unconditioned parent"):
            cond.sample(sample_shape=(5,))

    def test_condition_on_non_root_with_all_parents_conditioned_is_sampleable(self):
        """If all parents of a conditioned non-root are also conditioned, sampling works."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
            y=lambda z, x: Normal(loc=z + x, scale=0.1),
        )
        # Condition on z (root) and x (non-root, parent=z which is conditioned)
        cond = joint.condition_on(z=jnp.array(1.0), x=jnp.array(2.0))
        key = jax.random.PRNGKey(40)
        structured = cond.sample_structured(key, (50,))
        # z and x should not be in structured
        assert "z" not in structured
        assert "x" not in structured
        # y ~ N(1+2, 0.1) = N(3, 0.1)
        assert abs(float(jnp.mean(structured["y"])) - 3.0) < 0.1

    def test_condition_on_non_root_log_prob_works(self):
        """log_prob should still work after conditioning on a non-root."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        cond = joint.condition_on(x=jnp.array(1.0))
        # Only z is unconditioned; flat input is just z value
        flat = jnp.array([0.0])
        lp = cond.log_prob(flat)
        assert jnp.isfinite(lp)
        # Should be log p(z=0) + log p(x=1|z=0) (unnormalized conditional)
        expected = (
            float(Normal(loc=0.0, scale=1.0).log_prob(0.0))
            + float(Normal(loc=0.0, scale=0.5).log_prob(1.0))
        )
        np.testing.assert_allclose(float(lp), expected, atol=1e-5)

    def test_condition_on_chain_accumulates(self):
        """Successive condition_on calls accumulate conditioned names."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
            y=lambda z, x: Normal(loc=z + x, scale=0.1),
        )
        # First condition on x alone — not sampleable (z unconditioned parent)
        cond1 = joint.condition_on(x=jnp.array(1.0))
        assert cond1.component_names == ("z", "y")
        with pytest.raises(NotImplementedError):
            cond1.sample(sample_shape=(5,))
        # Then also condition on z — now sampleable, only y remains
        cond2 = cond1.condition_on(z=jnp.array(0.0))
        assert cond2.component_names == ("y",)
        s = cond2.sample(sample_shape=(5,))
        assert s.shape == (5, 1)

    def test_raises_on_conditioning_all(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        with pytest.raises(ValueError, match="Cannot condition on all"):
            joint.condition_on(z=jnp.array(0.0), x=jnp.array(0.0))


# ---------------------------------------------------------------------------
# Broadcasting reconnection
# ---------------------------------------------------------------------------

class TestBroadcastingReconnection:

    def test_views_from_sequential_joint_loop(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.01),  # x ≈ z
        )

        def subtract(a: float, b: float) -> float:
            return a - b

        wf = Workflow(
            func=subtract,
            vectorize="loop",
            n_broadcast_samples=30,
            seed=42,
        )
        result = wf(a=joint["z"], b=joint["x"])
        assert isinstance(result, EmpiricalDistribution)
        # z and x are jointly sampled, x ≈ z, so a - b ≈ 0
        np.testing.assert_allclose(
            np.array(result.samples), 0.0, atol=0.15
        )

    def test_views_from_sequential_joint_jax(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.01),
        )

        def subtract(a: float, b: float) -> float:
            return a - b

        wf = Workflow(
            func=subtract,
            vectorize="jax",
            n_broadcast_samples=30,
            seed=55,
        )
        result = wf(a=joint["z"], b=joint["x"])
        assert isinstance(result, EmpiricalDistribution)
        np.testing.assert_allclose(
            np.array(result.samples), 0.0, atol=0.15
        )


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_includes_callable(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        r = repr(joint)
        assert "SequentialJointDistribution" in r
        assert "z=Normal" in r
        assert "x=<callable>" in r
