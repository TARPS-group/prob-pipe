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
    EmpiricalDistribution,
    JointDistribution,
)
from probpipe.core.distribution import PyTreeArrayDistribution
from probpipe.core.node import WorkflowFunction
from probpipe import condition_on, log_prob, sample, unnormalized_log_prob


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
        assert isinstance(joint, PyTreeArrayDistribution)

    def test_component_names(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        assert joint.component_names == ("z", "x")

    def test_event_shapes(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        assert joint.event_shapes == {"z": (), "x": ()}
        assert joint.event_size == 2

    def test_three_components(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
            y=lambda z, x: Normal(loc=z + x, scale=0.1),
        )
        assert joint.event_shapes == {"z": (), "x": (), "y": ()}
        assert joint.event_size == 3
        assert joint.component_names == ("z", "x", "y")

    def test_independent_root(self):
        """All root distributions (no callables) should work."""
        joint = SequentialJointDistribution(
            a=Normal(loc=0.0, scale=1.0),
            b=Normal(loc=1.0, scale=2.0),
        )
        assert joint.event_shapes == {"a": (), "b": ()}
        assert joint.event_size == 2

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

    def test_sample_returns_dict(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        key = jax.random.PRNGKey(0)
        s = sample(joint, key=key)
        assert isinstance(s, dict)
        assert set(s.keys()) == {"z", "x"}
        assert s["z"].shape == ()
        assert s["x"].shape == ()

    def test_sample_batch(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        key = jax.random.PRNGKey(1)
        s = sample(joint, key=key, sample_shape=(10,))
        assert isinstance(s, dict)
        assert s["z"].shape == (10,)
        assert s["x"].shape == (10,)

    def test_conditional_sampling_correlation(self):
        """x depends on z, so they should be correlated."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.01),  # x ≈ z
        )
        key = jax.random.PRNGKey(3)
        s = sample(joint, key=key, sample_shape=(500,))
        # x should be very close to z
        np.testing.assert_allclose(s["x"], s["z"], atol=0.1)

    def test_auto_key(self):
        """sample() without explicit key should work."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        s = sample(joint, sample_shape=(5,))
        assert isinstance(s, dict)
        assert s["z"].shape == (5,)
        assert s["x"].shape == (5,)


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
        samples = sample(joint, key=key, sample_shape=(5,))
        lps = log_prob(joint, samples)
        assert lps.shape == (5,)

    def test_log_prob_finite(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        key = jax.random.PRNGKey(11)
        samples = sample(joint, key=key, sample_shape=(10,))
        lps = log_prob(joint, samples)
        assert jnp.all(jnp.isfinite(lps))

    def test_log_prob_manual(self):
        """Verify log_prob = log p(z) + log p(x|z)."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        z_val = jnp.array(1.0)
        x_val = jnp.array(1.5)
        value = {"z": z_val, "x": x_val}
        lp = log_prob(joint, value)

        lp_z = Normal(loc=0.0, scale=1.0)._log_prob(z_val)
        lp_x_given_z = Normal(loc=z_val, scale=0.5)._log_prob(x_val)
        expected = lp_z + lp_x_given_z
        np.testing.assert_allclose(float(lp), float(expected), atol=1e-5)

    def test_log_prob_three_components(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
            y=lambda z, x: Normal(loc=z + x, scale=0.1),
        )
        z_val, x_val, y_val = 0.5, 0.8, 1.2
        value = {"z": jnp.array(z_val), "x": jnp.array(x_val), "y": jnp.array(y_val)}
        lp = log_prob(joint, value)

        expected = (
            float(Normal(loc=0.0, scale=1.0)._log_prob(z_val))
            + float(Normal(loc=z_val, scale=0.5)._log_prob(x_val))
            + float(Normal(loc=z_val + x_val, scale=0.1)._log_prob(y_val))
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
        s = sample(joint["z"], key=key, sample_shape=(5,))
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
        cond = condition_on(joint, z=jnp.array(2.0))
        assert "z" not in cond.components
        assert "x" in cond.components
        assert cond.component_names == ("x",)
        assert cond.event_shapes == {"x": ()}
        assert cond.event_size == 1

    def test_conditioned_sampling_uses_observed_value(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        cond = condition_on(joint, z=jnp.array(3.0))
        key = jax.random.PRNGKey(30)
        s = sample(cond, key=key, sample_shape=(50,))
        # z should not be in s (it's conditioned)
        assert "z" not in s
        # x should be centered around 3.0 (since x = N(z=3, 0.5))
        assert abs(float(jnp.mean(s["x"])) - 3.0) < 0.3

    def test_condition_on_downstream_effect(self):
        """Conditioning on z should make downstream x use the observed value."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.01),  # x ≈ z
        )
        cond = condition_on(joint, z=jnp.array(5.0))
        key = jax.random.PRNGKey(31)
        s = sample(cond, key=key, sample_shape=(100,))
        assert "z" not in s
        # x should be very close to 5.0
        np.testing.assert_allclose(s["x"], jnp.full((100,), 5.0), atol=0.1)

    def test_condition_on_provenance(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        cond = condition_on(joint, z=jnp.array(0.0))
        assert cond.source is not None
        assert cond.source.operation == "condition_on"

    def test_condition_on_unknown_raises(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        with pytest.raises(KeyError, match="not found"):
            condition_on(joint, nonexistent=jnp.array(0.0))

    def test_condition_on_non_root_removes_component(self):
        """Conditioning on a non-root removes it from components."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        cond = condition_on(joint, x=jnp.array(1.0))
        assert "x" not in cond.components
        assert "z" in cond.components
        assert cond.component_names == ("z",)

    def test_condition_on_non_root_with_unconditioned_parent_raises(self):
        """Sampling raises if a conditioned non-root has unconditioned parents."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        cond = condition_on(joint, x=jnp.array(1.0))
        with pytest.raises(NotImplementedError, match="unconditioned parent"):
            sample(cond, sample_shape=(5,))

    def test_condition_on_non_root_with_all_parents_conditioned_is_sampleable(self):
        """If all parents of a conditioned non-root are also conditioned, sampling works."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
            y=lambda z, x: Normal(loc=z + x, scale=0.1),
        )
        # Condition on z (root) and x (non-root, parent=z which is conditioned)
        cond = condition_on(joint, z=jnp.array(1.0), x=jnp.array(2.0))
        key = jax.random.PRNGKey(40)
        s = sample(cond, key=key, sample_shape=(50,))
        # z and x should not be in s
        assert "z" not in s
        assert "x" not in s
        # y ~ N(1+2, 0.1) = N(3, 0.1)
        assert abs(float(jnp.mean(s["y"])) - 3.0) < 0.1

    def test_condition_on_root_log_prob_works(self):
        """log_prob should work when conditioning on root components."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        # Condition on root z — x's conditional p(x|z=2) is normalized
        cond = condition_on(joint, z=jnp.array(2.0))
        value = {"x": jnp.array(1.5)}
        lp = log_prob(cond, value)
        assert jnp.isfinite(lp)
        # Should equal log p(x=1.5 | z=2.0) = log N(1.5; 2.0, 0.5)
        expected = float(Normal(loc=2.0, scale=0.5)._log_prob(1.5))
        np.testing.assert_allclose(float(lp), expected, atol=1e-5)

    def test_condition_on_non_root_log_prob_raises(self):
        """log_prob should raise when conditioning on non-root with free parents."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        cond = condition_on(joint, x=jnp.array(1.0))
        value = {"z": jnp.array(0.0)}
        with pytest.raises(NotImplementedError, match="unnormalized_log_prob"):
            log_prob(cond, value)

    def test_condition_on_non_root_unnormalized_log_prob_works(self):
        """unnormalized_log_prob should work after conditioning on a non-root."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        cond = condition_on(joint, x=jnp.array(1.0))
        # Only z is unconditioned; input is dict with just z
        value = {"z": jnp.array(0.0)}
        lp = unnormalized_log_prob(cond, value)
        assert jnp.isfinite(lp)
        # Should be log p(z=0) + log p(x=1|z=0) (unnormalized conditional)
        expected = (
            float(Normal(loc=0.0, scale=1.0)._log_prob(0.0))
            + float(Normal(loc=0.0, scale=0.5)._log_prob(1.0))
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
        cond1 = condition_on(joint, x=jnp.array(1.0))
        assert cond1.component_names == ("z", "y")
        with pytest.raises(NotImplementedError):
            sample(cond1, sample_shape=(5,))
        # Then also condition on z — now sampleable, only y remains
        cond2 = condition_on(cond1, z=jnp.array(0.0))
        assert cond2.component_names == ("y",)
        s = sample(cond2, sample_shape=(5,))
        assert isinstance(s, dict)
        assert set(s.keys()) == {"y"}
        assert s["y"].shape == (5,)

    def test_raises_on_conditioning_all(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        with pytest.raises(ValueError, match="Cannot condition on all"):
            condition_on(joint, z=jnp.array(0.0), x=jnp.array(0.0))

    def test_dict_for_leaf_raises(self):
        """Passing a dict value for a leaf component should raise TypeError."""
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        with pytest.raises(TypeError, match="component distribution"):
            condition_on(joint, z={"sub": jnp.array(0.0)})


# ---------------------------------------------------------------------------
# Flatten / Unflatten
# ---------------------------------------------------------------------------

class TestFlattenUnflatten:

    def test_flatten_roundtrip(self):
        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
        )
        key = jax.random.PRNGKey(50)
        s = sample(joint, key=key, sample_shape=(5,))
        flat = joint.flatten_value(s)
        assert flat.shape == (5, 2)
        unflat = joint.unflatten_value(flat)
        np.testing.assert_allclose(unflat["z"], s["z"], atol=1e-6)
        np.testing.assert_allclose(unflat["x"], s["x"], atol=1e-6)


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

        wf = WorkflowFunction(
            func=subtract,
            vectorize="loop",
            n_broadcast_samples=30,
            seed=42,
        )
        result = wf(a=joint["z"], b=joint["x"])
        assert hasattr(result, "samples")
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

        wf = WorkflowFunction(
            func=subtract,
            vectorize="jax",
            n_broadcast_samples=30,
            seed=55,
        )
        result = wf(a=joint["z"], b=joint["x"])
        assert hasattr(result, "samples")
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
