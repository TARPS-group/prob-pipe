"""Comprehensive tests for joint distribution classes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from probpipe import (
    Normal,
    MultivariateNormal,
    Gamma,
    ProductDistribution,
    DistributionView,
    ConditionedComponent,
    JointDistribution,
    EmpiricalDistribution,
    ArrayDistribution,
    PyTreeArrayDistribution,
    FlattenedView,
)
from probpipe.core.node import WorkflowFunction
from probpipe import condition_on, from_distribution, log_prob, mean, sample, variance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_x():
    return Normal(loc=0.0, scale=1.0, name="x")


@pytest.fixture
def normal_y():
    return Normal(loc=3.0, scale=2.0, name="y")


@pytest.fixture
def mvn_z():
    loc = jnp.array([1.0, 2.0, 3.0])
    cov = jnp.eye(3) * 0.5
    return MultivariateNormal(loc=loc, cov=cov, name="z")


@pytest.fixture
def joint_xy(normal_x, normal_y):
    return ProductDistribution(x=normal_x, y=normal_y)


@pytest.fixture
def joint_xz(normal_x, mvn_z):
    return ProductDistribution(x=normal_x, z=mvn_z)


# ===========================================================================
# 1. TestProductDistribution
# ===========================================================================

class TestProductDistribution:

    def test_construction_normal_components(self, normal_x, normal_y):
        joint = ProductDistribution(x=normal_x, y=normal_y)
        assert isinstance(joint, ProductDistribution)
        assert isinstance(joint, JointDistribution)

    def test_construction_with_mvn(self, normal_x, mvn_z):
        joint = ProductDistribution(x=normal_x, z=mvn_z)
        assert isinstance(joint, ProductDistribution)

    def test_isinstance_pytree_array_distribution(self, joint_xy):
        assert isinstance(joint_xy, PyTreeArrayDistribution)
        assert not isinstance(joint_xy, ArrayDistribution)

    def test_event_shapes(self, joint_xy):
        assert joint_xy.event_shapes == {"x": (), "y": ()}

    def test_event_shapes_mixed(self, joint_xz):
        assert joint_xz.event_shapes == {"x": (), "z": (3,)}

    def test_event_size_scalar_components(self, joint_xy):
        # Two scalar Normals: total dim = 1 + 1 = 2
        assert joint_xy.event_size == 2

    def test_event_size_mixed_components(self, joint_xz):
        # Scalar Normal (dim 1) + MVN(dim 3): total dim = 4
        assert joint_xz.event_size == 4

    def test_component_names(self, joint_xy):
        assert joint_xy.component_names == ("x", "y")

    def test_component_names_order_preserved(self, normal_x, normal_y):
        joint = ProductDistribution(y=normal_y, x=normal_x)
        assert joint.component_names == ("y", "x")

    def test_sample_returns_dict(self, joint_xy):
        key = jax.random.PRNGKey(0)
        s = sample(joint_xy, key=key)
        assert isinstance(s, dict)
        assert set(s.keys()) == {"x", "y"}

    def test_sample_shapes_scalar(self, joint_xy):
        key = jax.random.PRNGKey(0)
        s = sample(joint_xy, key=key)
        assert s["x"].shape == ()
        assert s["y"].shape == ()

    def test_sample_shapes_with_sample_shape(self, joint_xz):
        key = jax.random.PRNGKey(1)
        s = sample(joint_xz, key=key, sample_shape=(10,))
        assert s["x"].shape == (10,)
        assert s["z"].shape == (10, 3)

    def test_log_prob_accepts_dict(self, joint_xy, normal_x, normal_y):
        key = jax.random.PRNGKey(4)
        s = sample(joint_xy, key=key)
        lp_joint = log_prob(joint_xy, s)
        lp_sum = log_prob(normal_x, s["x"]) + log_prob(normal_y, s["y"])
        np.testing.assert_allclose(float(lp_joint), float(lp_sum), atol=1e-5)

    def test_log_prob_batch(self, joint_xy):
        key = jax.random.PRNGKey(5)
        samples = sample(joint_xy, key=key, sample_shape=(20,))
        lps = log_prob(joint_xy, samples)
        assert lps.shape == (20,)

    def test_mean_returns_dict(self, joint_xz, normal_x, mvn_z):
        m = mean(joint_xz)
        assert isinstance(m, dict)
        np.testing.assert_allclose(m["x"], mean(normal_x), atol=1e-6)
        np.testing.assert_allclose(m["z"], mean(mvn_z), atol=1e-6)

    def test_variance_returns_dict(self, joint_xz, normal_x, mvn_z):
        v = variance(joint_xz)
        assert isinstance(v, dict)
        np.testing.assert_allclose(v["x"], variance(normal_x), atol=1e-6)
        np.testing.assert_allclose(v["z"], variance(mvn_z), atol=1e-6)

    def test_repr_includes_class_and_names(self, joint_xy):
        r = repr(joint_xy)
        assert "ProductDistribution" in r
        assert "x=" in r
        assert "y=" in r

    def test_repr_named_joint(self, normal_x, normal_y):
        joint = ProductDistribution(x=normal_x, y=normal_y, name="my_joint")
        r = repr(joint)
        assert "my_joint" in r


# ===========================================================================
# 2. TestFlattenUnflatten
# ===========================================================================

class TestFlattenUnflatten:
    """Test flatten_value / unflatten_value (inherited from PyTreeArrayDistribution)."""

    def test_flatten_value_scalar_components(self, joint_xy):
        key = jax.random.PRNGKey(60)
        s = sample(joint_xy, key=key)
        flat = joint_xy.flatten_value(s)
        assert flat.shape == (2,)

    def test_flatten_value_mixed_components(self, joint_xz):
        key = jax.random.PRNGKey(61)
        s = sample(joint_xz, key=key)
        flat = joint_xz.flatten_value(s)
        assert flat.shape == (4,)

    def test_roundtrip_scalar_components(self, joint_xy):
        key = jax.random.PRNGKey(60)
        s = sample(joint_xy, key=key, sample_shape=(5,))
        flat = joint_xy.flatten_value(s)
        recovered = joint_xy.unflatten_value(flat)
        np.testing.assert_allclose(recovered["x"], s["x"], atol=1e-6)
        np.testing.assert_allclose(recovered["y"], s["y"], atol=1e-6)

    def test_roundtrip_mixed_components(self, joint_xz):
        key = jax.random.PRNGKey(61)
        s = sample(joint_xz, key=key, sample_shape=(3,))
        flat = joint_xz.flatten_value(s)
        assert flat.shape == (3, 4)
        recovered = joint_xz.unflatten_value(flat)
        np.testing.assert_allclose(recovered["x"], s["x"], atol=1e-6)
        np.testing.assert_allclose(recovered["z"], s["z"], atol=1e-6)

    def test_roundtrip_no_batch_dim(self, joint_xy):
        key = jax.random.PRNGKey(62)
        s = sample(joint_xy, key=key)
        flat = joint_xy.flatten_value(s)
        assert flat.shape == (2,)
        recovered = joint_xy.unflatten_value(flat)
        np.testing.assert_allclose(recovered["x"], s["x"], atol=1e-6)
        np.testing.assert_allclose(recovered["y"], s["y"], atol=1e-6)



# ===========================================================================
# 3. TestFlattenedView
# ===========================================================================

class TestFlattenedViewInterop:
    """Test as_flat_distribution() returns a usable FlattenedView."""

    def test_as_flat_distribution_type(self, joint_xy):
        flat_dist = joint_xy.as_flat_distribution()
        assert isinstance(flat_dist, FlattenedView)
        assert isinstance(flat_dist, ArrayDistribution)

    def test_flat_event_shape(self, joint_xz):
        flat_dist = joint_xz.as_flat_distribution()
        assert flat_dist.event_shape == (4,)

    def test_flat_sample_shape(self, joint_xz):
        flat_dist = joint_xz.as_flat_distribution()
        s = sample(flat_dist, key=jax.random.PRNGKey(0), sample_shape=(10,))
        assert s.shape == (10, 4)

    def test_flat_log_prob_consistent(self, joint_xy):
        flat_dist = joint_xy.as_flat_distribution()
        key = jax.random.PRNGKey(1)
        s = sample(joint_xy, key=key)
        flat_s = joint_xy.flatten_value(s)
        lp_dict = log_prob(joint_xy, s)
        lp_flat = log_prob(flat_dist, flat_s)
        np.testing.assert_allclose(float(lp_dict), float(lp_flat), atol=1e-5)

    def test_unflatten_sample(self, joint_xz):
        flat_dist = joint_xz.as_flat_distribution()
        flat_s = sample(flat_dist, key=jax.random.PRNGKey(2))
        restored = flat_dist.unflatten_sample(flat_s)
        assert isinstance(restored, dict)
        assert set(restored.keys()) == {"x", "z"}
        assert restored["z"].shape == (3,)


# ===========================================================================
# 4. TestDistributionView
# ===========================================================================

class TestDistributionView:

    def test_event_shape_matches_component(self, joint_xz):
        view_x = joint_xz["x"]
        view_z = joint_xz["z"]
        assert view_x.event_shape == ()
        assert view_z.event_shape == (3,)

    def test_sample_correct_shape_scalar(self, joint_xy):
        key = jax.random.PRNGKey(10)
        view = joint_xy["x"]
        s = sample(view, key=key)
        assert s.shape == ()

    def test_sample_correct_shape_mvn(self, joint_xz):
        key = jax.random.PRNGKey(11)
        view = joint_xz["z"]
        s = sample(view, key=key, sample_shape=(7,))
        assert s.shape == (7, 3)

    def test_log_prob_matches_component(self, joint_xy, normal_x):
        view = joint_xy["x"]
        x_val = jnp.array(1.5)
        lp_view = log_prob(view, x_val)
        lp_direct = log_prob(normal_x, x_val)
        np.testing.assert_allclose(float(lp_view), float(lp_direct), atol=1e-6)

    def test_mean_matches_component(self, joint_xz, mvn_z):
        view = joint_xz["z"]
        np.testing.assert_allclose(mean(view), mean(mvn_z), atol=1e-6)

    def test_parent_reference(self, joint_xy):
        view = joint_xy["x"]
        assert view._parent is joint_xy

    def test_component_name(self, joint_xy):
        view = joint_xy["y"]
        assert view._component_name == "y"

    def test_keyerror_invalid_component(self, joint_xy):
        with pytest.raises(KeyError, match="not_a_component"):
            joint_xy["not_a_component"]

    def test_repr(self, joint_xy):
        view = joint_xy["x"]
        r = repr(view)
        assert "DistributionView" in r
        assert "ProductDistribution" in r
        assert "'x'" in r


# ===========================================================================
# 5. TestBind
# ===========================================================================

class TestBind:

    def test_bind_returns_dict_of_views(self, joint_xy):
        views = joint_xy.bind(a="x", b="y")
        assert isinstance(views, dict)
        assert set(views.keys()) == {"a", "b"}
        assert all(isinstance(v, DistributionView) for v in views.values())

    def test_bind_component_name_mapping(self, joint_xy):
        views = joint_xy.bind(alpha="x", beta="y")
        assert views["alpha"]._component_name == "x"
        assert views["beta"]._component_name == "y"

    def test_bind_single(self, joint_xy):
        views = joint_xy.bind(only_x="x")
        assert len(views) == 1
        assert views["only_x"]._component_name == "x"


# ===========================================================================
# 7. TestConditionOn
# ===========================================================================

class TestConditionOn:

    def test_condition_on_removes_conditioned_component(self, joint_xy):
        cond = condition_on(joint_xy, x=jnp.array(2.0))
        assert "x" not in cond.components
        assert "y" in cond.components
        assert cond.component_names == ("y",)
        assert cond.event_size == 1

    def test_conditioned_sample_excludes_conditioned(self, joint_xy):
        cond = condition_on(joint_xy, x=jnp.array(2.0))
        key = jax.random.PRNGKey(20)
        s = sample(cond, key=key, sample_shape=(10,))
        assert "x" not in s
        assert "y" in s

    def test_unconditioned_component_still_varies(self, joint_xy):
        cond = condition_on(joint_xy, x=jnp.array(0.0))
        key = jax.random.PRNGKey(21)
        s = sample(cond, key=key, sample_shape=(100,))
        # y should have non-trivial variance
        assert jnp.std(s["y"]) > 0.1

    def test_log_prob_works_on_conditioned(self, joint_xy):
        cond = condition_on(joint_xy, x=jnp.array(1.0))
        key = jax.random.PRNGKey(22)
        s = sample(cond, key=key, sample_shape=(5,))
        lps = log_prob(cond, s)
        assert lps.shape == (5,)
        assert jnp.all(jnp.isfinite(lps))

    def test_provenance_attached(self, joint_xy):
        cond = condition_on(joint_xy, y=jnp.array(0.5))
        assert cond.source is not None
        assert cond.source.operation == "condition_on"
        assert "y" in cond.source.metadata["conditioned"]

    def test_keyerror_on_unknown_component(self, joint_xy):
        with pytest.raises(KeyError, match="not found"):
            condition_on(joint_xy, nonexistent=jnp.array(0.0))

    def test_raises_on_conditioning_all(self, joint_xy):
        with pytest.raises(ValueError, match="Cannot condition on all"):
            condition_on(joint_xy, x=jnp.array(0.0), y=jnp.array(0.0))


# ===========================================================================
# 8. TestConditionedComponent
# ===========================================================================

class TestConditionedComponent:

    def test_event_shape_matches_base(self, normal_x):
        pc = ConditionedComponent(normal_x, jnp.array(1.0))
        assert pc.event_shape == normal_x.event_shape

    def test_event_shape_mvn(self, mvn_z):
        val = jnp.array([1.0, 2.0, 3.0])
        pc = ConditionedComponent(mvn_z, val)
        assert pc.event_shape == (3,)

    def test_sample_always_returns_pinned(self, normal_x):
        val = jnp.array(42.0)
        pc = ConditionedComponent(normal_x, val)
        key = jax.random.PRNGKey(30)
        s1 = sample(pc, key=key)
        s2 = sample(pc, key=jax.random.PRNGKey(31))
        np.testing.assert_allclose(float(s1), 42.0, atol=1e-6)
        np.testing.assert_allclose(float(s2), 42.0, atol=1e-6)

    def test_sample_broadcast_shape(self, normal_x):
        val = jnp.array(5.0)
        pc = ConditionedComponent(normal_x, val)
        key = jax.random.PRNGKey(32)
        s = sample(pc, key=key, sample_shape=(8,))
        assert s.shape == (8,)
        np.testing.assert_allclose(s, jnp.full((8,), 5.0), atol=1e-6)

    def test_value_error_shape_mismatch(self, mvn_z):
        with pytest.raises(ValueError, match="shape"):
            ConditionedComponent(mvn_z, jnp.array(1.0))  # scalar vs (3,)

    def test_log_prob_is_constant(self, normal_x):
        val = jnp.array(0.0)
        pc = ConditionedComponent(normal_x, val)
        lp1 = log_prob(pc, jnp.array(999.0))
        lp2 = log_prob(pc, jnp.array(-999.0))
        # log_prob always evaluates base at the pinned value, ignoring input
        np.testing.assert_allclose(float(lp1), float(lp2), atol=1e-6)

    def test_mean_returns_pinned_value(self, normal_x):
        val = jnp.array(7.0)
        pc = ConditionedComponent(normal_x, val)
        np.testing.assert_allclose(float(mean(pc)), 7.0, atol=1e-6)

    def test_variance_is_zero(self, normal_x):
        pc = ConditionedComponent(normal_x, jnp.array(1.0))
        np.testing.assert_allclose(float(variance(pc)), 0.0, atol=1e-6)


# ===========================================================================
# 9. TestComponentLogProb
# ===========================================================================

class TestComponentLogProb:

    def test_component_log_prob_returns_dict(self, joint_xy, normal_x, normal_y):
        key = jax.random.PRNGKey(90)
        s = sample(joint_xy, key=key)
        clp = joint_xy.component_log_prob(s)
        assert isinstance(clp, dict)
        assert set(clp.keys()) == {"x", "y"}

    def test_component_log_prob_matches_individual(self, joint_xz, normal_x, mvn_z):
        key = jax.random.PRNGKey(91)
        s = sample(joint_xz, key=key)
        clp = joint_xz.component_log_prob(s)
        np.testing.assert_allclose(
            float(clp["x"]), float(log_prob(normal_x, s["x"])), atol=1e-5
        )
        np.testing.assert_allclose(
            float(clp["z"]), float(log_prob(mvn_z, s["z"])), atol=1e-5
        )

    def test_component_log_prob_sums_to_joint(self, joint_xy):
        key = jax.random.PRNGKey(92)
        s = sample(joint_xy, key=key)
        clp = joint_xy.component_log_prob(s)
        joint_lp = log_prob(joint_xy, s)
        np.testing.assert_allclose(
            float(clp["x"] + clp["y"]), float(joint_lp), atol=1e-5
        )


# ===========================================================================
# 10. TestBroadcastingReconnection
# ===========================================================================

class TestBroadcastingReconnection:
    """Test that DistributionViews from the same parent are sampled jointly."""

    @staticmethod
    def _make_add_workflow(backend="loop"):
        """Create a workflow that adds two arrays."""
        def add(a: float, b: float) -> float:
            return a + b

        return WorkflowFunction(
            func=add,
            vectorize=backend,
            n_broadcast_samples=50,
            seed=42,
        )

    def test_joint_views_sampled_together_loop(self):
        """Two views from same parent passed to workflow -> sampled jointly."""
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0),
            y=Normal(loc=10.0, scale=1.0),
        )
        wf = self._make_add_workflow("loop")
        result = wf(a=joint["x"], b=joint["y"])
        assert isinstance(result, EmpiricalDistribution)
        # x ~ N(0,1), y ~ N(10,1) => a+b ~ N(10, sqrt(2))
        mean_val = float(jnp.mean(result.samples))
        assert abs(mean_val - 10.0) < 2.0

    def test_same_view_twice_gives_identical_values_loop(self):
        """If joint['x'] is passed to both args, they get the same values."""
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0),
            y=Normal(loc=10.0, scale=1.0),
        )
        view_x = joint["x"]

        def subtract(a: float, b: float) -> float:
            return a - b

        wf = WorkflowFunction(
            func=subtract,
            vectorize="loop",
            n_broadcast_samples=20,
            seed=99,
        )
        result = wf(a=view_x, b=view_x)
        assert isinstance(result, EmpiricalDistribution)
        # a and b are the same samples, so a - b = 0 for every sample
        np.testing.assert_allclose(
            np.array(result.samples), 0.0, atol=1e-5
        )

    def test_mix_of_view_and_independent(self):
        """Mix of DistributionView and independent Normal both work."""
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0),
            y=Normal(loc=5.0, scale=1.0),
        )
        independent = Normal(loc=100.0, scale=0.1)

        def add3(a: float, b: float, c: float) -> float:
            return a + b + c

        wf = WorkflowFunction(
            func=add3,
            vectorize="loop",
            n_broadcast_samples=50,
            seed=77,
        )
        result = wf(a=joint["x"], b=joint["y"], c=independent)
        assert isinstance(result, EmpiricalDistribution)
        # x ~ N(0,1), y ~ N(5,1), c ~ N(100, 0.1) => sum ~ N(105, ...)
        mean_val = float(jnp.mean(result.samples))
        assert abs(mean_val - 105.0) < 3.0

    def test_joint_views_sampled_together_jax(self):
        """Two views from same parent using jax backend."""
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0),
            y=Normal(loc=10.0, scale=1.0),
        )

        def add(a: float, b: float) -> float:
            return a + b

        wf = WorkflowFunction(
            func=add,
            vectorize="jax",
            n_broadcast_samples=50,
            seed=55,
        )
        result = wf(a=joint["x"], b=joint["y"])
        assert isinstance(result, EmpiricalDistribution)
        mean_val = float(jnp.mean(result.samples))
        assert abs(mean_val - 10.0) < 2.0

    def test_same_view_twice_jax_backend(self):
        """Same DistributionView to both args with jax backend."""
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0),
            y=Normal(loc=10.0, scale=1.0),
        )
        view_x = joint["x"]

        def subtract(a: float, b: float) -> float:
            return a - b

        wf = WorkflowFunction(
            func=subtract,
            vectorize="jax",
            n_broadcast_samples=20,
            seed=88,
        )
        result = wf(a=view_x, b=view_x)
        assert isinstance(result, EmpiricalDistribution)
        np.testing.assert_allclose(
            np.array(result.samples), 0.0, atol=1e-5
        )


# ===========================================================================
# 11. TestPytreeRegistration
# ===========================================================================

class TestPytreeRegistration:

    def test_tree_flatten_unflatten_roundtrip(self, normal_x, normal_y):
        joint = ProductDistribution(x=normal_x, y=normal_y, name="rt")
        children, aux = jax.tree_util.tree_flatten(joint)
        reconstructed = jax.tree_util.tree_unflatten(aux, children)
        assert isinstance(reconstructed, ProductDistribution)
        assert reconstructed.component_names == ("x", "y")
        assert reconstructed.event_size == 2

    def test_tree_flatten_preserves_components(self, normal_x, mvn_z):
        joint = ProductDistribution(x=normal_x, z=mvn_z)
        children, aux = jax.tree_util.tree_flatten(joint)
        reconstructed = jax.tree_util.tree_unflatten(aux, children)
        # Verify component distributions survived the roundtrip
        key = jax.random.PRNGKey(50)
        s_orig = sample(joint, key=key)
        s_recon = sample(reconstructed, key=key)
        np.testing.assert_allclose(s_orig["x"], s_recon["x"], atol=1e-6)
        np.testing.assert_allclose(s_orig["z"], s_recon["z"], atol=1e-6)

    def test_tree_leaves_returns_components(self, normal_x, normal_y):
        joint = ProductDistribution(x=normal_x, y=normal_y)
        leaves = jax.tree.leaves(joint)
        # The leaves should be the component distributions
        assert len(leaves) == 2

    def test_tree_flatten_preserves_name(self, normal_x, normal_y):
        joint = ProductDistribution(x=normal_x, y=normal_y, name="named_joint")
        children, aux = jax.tree_util.tree_flatten(joint)
        reconstructed = jax.tree_util.tree_unflatten(aux, children)
        assert reconstructed._name == "named_joint"

    def test_unflatten_with_modified_children(self):
        """Verify tree_unflatten works with different component distributions."""
        a = Normal(loc=0.0, scale=1.0, name="a")
        b = Normal(loc=1.0, scale=1.0, name="b")
        joint = ProductDistribution(a=a, b=b)
        _, aux = jax.tree_util.tree_flatten(joint)

        # Substitute different distributions
        new_a = Normal(loc=10.0, scale=0.1, name="a")
        new_b = Normal(loc=20.0, scale=0.2, name="b")
        reconstructed = jax.tree_util.tree_unflatten(aux, [new_a, new_b])
        m = mean(reconstructed)
        np.testing.assert_allclose(float(m["a"]), 10.0, atol=1e-5)
        np.testing.assert_allclose(float(m["b"]), 20.0, atol=1e-5)


# ===========================================================================
# 12. Additional coverage
# ===========================================================================

class TestSingleComponent:
    """ProductDistribution with one component."""

    def test_single_component_event_size(self):
        joint = ProductDistribution(a=Normal(loc=0.0, scale=1.0))
        assert joint.event_size == 1

    def test_single_component_sample(self):
        joint = ProductDistribution(a=Normal(loc=0.0, scale=1.0))
        key = jax.random.PRNGKey(70)
        s = sample(joint, key=key, sample_shape=(5,))
        assert s["a"].shape == (5,)

    def test_single_component_log_prob(self):
        n = Normal(loc=0.0, scale=1.0)
        joint = ProductDistribution(a=n)
        s = sample(joint, key=jax.random.PRNGKey(71), sample_shape=(2,))
        lps = log_prob(joint, s)
        expected = jnp.array([float(log_prob(n, s["a"][0])), float(log_prob(n, s["a"][1]))])
        np.testing.assert_allclose(lps, expected, atol=1e-5)


class TestLogProbBatchValues:
    """Verify log_prob batch values are numerically correct."""

    def test_batch_log_prob_matches_individual(self, joint_xy, normal_x, normal_y):
        key = jax.random.PRNGKey(80)
        samples = sample(joint_xy, key=key, sample_shape=(10,))
        batch_lps = log_prob(joint_xy, samples)

        for i in range(10):
            s_i = {k: v[i] for k, v in samples.items()}
            expected = float(log_prob(normal_x, s_i["x"])) + float(log_prob(normal_y, s_i["y"]))
            np.testing.assert_allclose(float(batch_lps[i]), expected, atol=1e-5)


class TestEmptyComponentsValidation:

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            ProductDistribution()


class TestDistributionViewFromDistribution:

    def test_from_distribution_raises(self, joint_xy):
        with pytest.raises(TypeError):
            from_distribution(Normal(loc=0.0, scale=1.0), DistributionView)


class TestEnumerateWithDistributionViews:
    """Verify DistributionView reconnection works in the enumerate path."""

    def test_empirical_plus_views_preserves_correlation(self):
        """When an empirical and two correlated views are mixed, views stay paired."""
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0),
            y=Normal(loc=0.0, scale=1.0),
        )
        view_x = joint["x"]
        view_y = joint["y"]

        # Small empirical that will be enumerated
        ed = EmpiricalDistribution(jnp.array([[10.0], [20.0]]))

        def compute(a: float, b: float, c: float) -> float:
            return (a - b) + c

        wf = WorkflowFunction(
            func=compute,
            vectorize="loop",
            n_broadcast_samples=50,
            seed=123,
        )
        result = wf(a=view_x, b=view_y, c=ed)
        assert isinstance(result, EmpiricalDistribution)
        assert result.n == 50


# ===========================================================================
# 8. TestNestedProductDistribution
# ===========================================================================

class TestNestedProductDistribution:
    """Tests for ProductDistribution with nested dict components.

    A nested ProductDistribution groups components into sub-dicts::

        ProductDistribution(
            physics={"force": Normal(0, 1), "mass": Gamma(2, 1)},
            observation=Normal(0, 0.1),
        )

    The nesting is purely organizational — all leaf components remain
    statistically independent.  Samples, event_shapes, log_prob, etc.
    mirror the nested dict structure.
    """

    @pytest.fixture
    def nested_joint(self):
        return ProductDistribution(
            physics={"force": Normal(loc=0.0, scale=1.0),
                     "mass": Gamma(concentration=2.0, rate=1.0)},
            observation=Normal(loc=0.0, scale=0.1),
        )

    # -- Construction and introspection ------------------------------------

    def test_isinstance(self, nested_joint):
        assert isinstance(nested_joint, ProductDistribution)
        assert isinstance(nested_joint, JointDistribution)
        assert isinstance(nested_joint, PyTreeArrayDistribution)
        assert not isinstance(nested_joint, ArrayDistribution)

    def test_is_not_flat(self, nested_joint):
        assert not nested_joint._is_flat

    def test_event_shapes_nested(self, nested_joint):
        es = nested_joint.event_shapes
        assert isinstance(es, dict)
        assert isinstance(es["physics"], dict)
        assert es["physics"]["force"] == ()
        assert es["physics"]["mass"] == ()
        assert es["observation"] == ()

    def test_event_size(self, nested_joint):
        # 3 scalar leaves → event_size = 3
        assert nested_joint.event_size == 3

    def test_component_names_are_key_paths(self, nested_joint):
        names = nested_joint.component_names
        # JAX sorts dict keys, so order is: observation, physics.force, physics.mass
        assert len(names) == 3
        # Each name is a tuple (since the dict is nested)
        assert all(isinstance(n, tuple) for n in names)
        # Check expected key paths (JAX canonical order: sorted keys, depth-first)
        assert ("observation",) in names
        assert ("physics", "force") in names
        assert ("physics", "mass") in names

    def test_treedef_matches_sample_structure(self, nested_joint):
        key = jax.random.PRNGKey(0)
        s = sample(nested_joint, key=key)
        assert jax.tree.structure(s) == nested_joint.treedef

    # -- Sampling ----------------------------------------------------------

    def test_sample_returns_nested_dict(self, nested_joint):
        key = jax.random.PRNGKey(1)
        s = sample(nested_joint, key=key)
        assert isinstance(s, dict)
        assert isinstance(s["physics"], dict)
        assert "force" in s["physics"]
        assert "mass" in s["physics"]
        assert "observation" in s

    def test_sample_leaf_shapes(self, nested_joint):
        key = jax.random.PRNGKey(2)
        s = sample(nested_joint, key=key, sample_shape=(10,))
        assert s["physics"]["force"].shape == (10,)
        assert s["physics"]["mass"].shape == (10,)
        assert s["observation"].shape == (10,)

    def test_sample_single(self, nested_joint):
        key = jax.random.PRNGKey(3)
        s = sample(nested_joint, key=key)
        # Each leaf should be a scalar
        assert s["physics"]["force"].shape == ()
        assert s["physics"]["mass"].shape == ()
        assert s["observation"].shape == ()

    # -- log_prob ----------------------------------------------------------

    def test_log_prob_accepts_nested_dict(self, nested_joint):
        key = jax.random.PRNGKey(10)
        s = sample(nested_joint, key=key, sample_shape=(5,))
        lps = log_prob(nested_joint, s)
        assert lps.shape == (5,)
        assert jnp.all(jnp.isfinite(lps))

    def test_log_prob_equals_sum_of_components(self, nested_joint):
        key = jax.random.PRNGKey(11)
        s = sample(nested_joint, key=key)
        lp_joint = log_prob(nested_joint, s)

        # Manual sum of leaf log-probs
        force_dist = Normal(loc=0.0, scale=1.0)
        mass_dist = Gamma(concentration=2.0, rate=1.0)
        obs_dist = Normal(loc=0.0, scale=0.1)
        lp_manual = (
            log_prob(force_dist, s["physics"]["force"])
            + log_prob(mass_dist, s["physics"]["mass"])
            + log_prob(obs_dist, s["observation"])
        )
        np.testing.assert_allclose(float(lp_joint), float(lp_manual), atol=1e-5)

    # -- component_log_prob ------------------------------------------------

    def test_component_log_prob_nested(self, nested_joint):
        key = jax.random.PRNGKey(12)
        s = sample(nested_joint, key=key)
        clp = nested_joint.component_log_prob(s)
        assert isinstance(clp, dict)
        assert isinstance(clp["physics"], dict)
        assert jnp.isfinite(clp["physics"]["force"])
        assert jnp.isfinite(clp["physics"]["mass"])
        assert jnp.isfinite(clp["observation"])

    # -- mean / variance ---------------------------------------------------

    def test_mean_nested(self, nested_joint):
        m = mean(nested_joint)
        assert isinstance(m, dict)
        assert isinstance(m["physics"], dict)
        assert m["physics"]["force"].shape == ()
        assert m["physics"]["mass"].shape == ()
        assert m["observation"].shape == ()

    def test_variance_nested(self, nested_joint):
        v = variance(nested_joint)
        assert isinstance(v, dict)
        assert isinstance(v["physics"], dict)
        assert jnp.all(jnp.asarray(jax.tree.leaves(v)) >= 0)

    # -- flatten / unflatten -----------------------------------------------

    def test_flatten_unflatten_roundtrip(self, nested_joint):
        key = jax.random.PRNGKey(20)
        s = sample(nested_joint, key=key, sample_shape=(5,))
        flat = nested_joint.flatten_value(s)
        assert flat.shape == (5, 3)
        recovered = nested_joint.unflatten_value(flat)
        # Check all leaves match
        for orig, rec in zip(jax.tree.leaves(s), jax.tree.leaves(recovered)):
            np.testing.assert_allclose(orig, rec, atol=1e-6)

    def test_as_flat_distribution(self, nested_joint):
        flat_dist = nested_joint.as_flat_distribution()
        assert isinstance(flat_dist, FlattenedView)
        assert flat_dist.event_shape == (3,)
        key = jax.random.PRNGKey(21)
        s = sample(flat_dist, key=key, sample_shape=(5,))
        assert s.shape == (5, 3)

    # -- __getitem__ with key paths ----------------------------------------

    def test_getitem_nested_key_path(self, nested_joint):
        view = nested_joint["physics", "force"]
        assert isinstance(view, DistributionView)
        assert view._key_path == ("physics", "force")

    def test_getitem_top_level(self, nested_joint):
        view = nested_joint["observation"]
        assert isinstance(view, DistributionView)
        assert view._key_path == ("observation",)

    def test_getitem_internal_node_returns_sub_joint(self, nested_joint):
        """Indexing to a sub-dict returns a ProductDistribution."""
        sub = nested_joint["physics"]
        assert isinstance(sub, ProductDistribution)
        assert set(sub._components.keys()) == {"force", "mass"}

    def test_getitem_internal_node_sample(self, nested_joint):
        """Sub-joint from internal node should sample correctly."""
        sub = nested_joint["physics"]
        key = jax.random.PRNGKey(99)
        s = sample(sub, key=key, sample_shape=(10,))
        assert isinstance(s, dict)
        assert set(s.keys()) == {"force", "mass"}
        assert s["force"].shape == (10,)
        assert s["mass"].shape == (10,)
        # Gamma samples should be positive
        assert jnp.all(s["mass"] > 0)

    def test_getitem_internal_node_log_prob(self, nested_joint):
        """Sub-joint log_prob should work."""
        sub = nested_joint["physics"]
        key = jax.random.PRNGKey(100)
        s = sample(sub, key=key)
        lp = log_prob(sub, s)
        assert jnp.isfinite(lp)

    def test_getitem_internal_node_moments(self, nested_joint):
        """Sub-joint mean/variance should work."""
        sub = nested_joint["physics"]
        m = mean(sub)
        assert isinstance(m, dict)
        assert set(m.keys()) == {"force", "mass"}

    def test_getitem_internal_node_event_shapes(self, nested_joint):
        """Sub-joint should have correct event_shapes."""
        sub = nested_joint["physics"]
        es = sub.event_shapes
        assert es == {"force": (), "mass": ()}
        assert sub.event_size == 2

    def test_getitem_internal_node_is_independent(self, nested_joint):
        """Sub-joint is a marginal — it does not share state with parent."""
        sub = nested_joint["physics"]
        parent_names = nested_joint.component_names
        sub_names = sub.component_names
        # Sub-joint is flat, so names are plain strings
        assert sub_names == ("force", "mass")

    def test_getitem_invalid_raises(self, nested_joint):
        with pytest.raises(KeyError, match="not found"):
            nested_joint["nonexistent"]

    def test_view_sample_nested(self, nested_joint):
        key = jax.random.PRNGKey(30)
        view = nested_joint["physics", "mass"]
        s = sample(view, key=key, sample_shape=(10,))
        assert s.shape == (10,)
        # Gamma samples should be positive
        assert jnp.all(s > 0)

    def test_view_event_shape_nested(self, nested_joint):
        assert nested_joint["physics", "force"].event_shape == ()
        assert nested_joint["observation"].event_shape == ()

    # -- bind with key paths -----------------------------------------------

    def test_bind_nested(self, nested_joint):
        views = nested_joint.bind(
            f=("physics", "force"),
            o="observation",
        )
        assert isinstance(views["f"], DistributionView)
        assert isinstance(views["o"], DistributionView)
        assert views["f"]._key_path == ("physics", "force")
        assert views["o"]._key_path == ("observation",)

    # -- condition_on -------------------------------------------------------

    def test_condition_on_top_level_leaf(self, nested_joint):
        """Condition on a top-level leaf component."""
        cond = condition_on(nested_joint, observation=jnp.array(0.5))
        assert "observation" not in cond._components
        assert "physics" in cond._components
        assert cond.event_size == 2  # force + mass

    def test_condition_on_all_leaves_in_group(self, nested_joint):
        """Conditioning on all components under a dict removes it."""
        cond = nested_joint.condition_on(
            physics={"force": jnp.array(1.0), "mass": jnp.array(2.0)}
        )
        assert "physics" not in cond._components
        assert "observation" in cond._components
        assert cond.event_size == 1

    def test_condition_on_single_nested_leaf(self, nested_joint):
        """Condition on one component, keeping the sibling."""
        cond = nested_joint.condition_on(
            physics={"force": jnp.array(1.0)}
        )
        # physics dict still exists but only contains mass
        assert "physics" in cond._components
        assert isinstance(cond._components["physics"], dict)
        assert "mass" in cond._components["physics"]
        assert "force" not in cond._components["physics"]
        assert "observation" in cond._components
        assert cond.event_size == 2  # mass(1) + observation(1)

    def test_condition_on_nested_leaf_sample(self, nested_joint):
        """Samples from conditioned nested joint should exclude conditioned leaf."""
        cond = nested_joint.condition_on(
            physics={"force": jnp.array(1.0)}
        )
        key = jax.random.PRNGKey(50)
        s = sample(cond, key=key, sample_shape=(5,))
        assert "physics" in s
        assert "mass" in s["physics"]
        assert "force" not in s["physics"]
        assert "observation" in s
        assert s["physics"]["mass"].shape == (5,)
        assert s["observation"].shape == (5,)

    def test_condition_on_nested_leaf_log_prob(self, nested_joint):
        """Log-prob should work on conditioned nested joint."""
        cond = nested_joint.condition_on(
            physics={"force": jnp.array(1.0)}
        )
        key = jax.random.PRNGKey(51)
        s = sample(cond, key=key)
        lp = log_prob(cond, s)
        assert jnp.isfinite(lp)

    def test_condition_on_multiple_leaves_across_groups(self, nested_joint):
        """Condition on leaves from different groups simultaneously."""
        cond = nested_joint.condition_on(
            physics={"force": jnp.array(1.0)},
            observation=jnp.array(0.5),
        )
        # Only mass remains
        assert "physics" in cond._components
        assert "mass" in cond._components["physics"]
        assert "observation" not in cond._components
        assert cond.event_size == 1

    def test_condition_on_positional_dict(self, nested_joint):
        """Condition using a positional dict instead of kwargs."""
        cond = nested_joint.condition_on(
            {"physics": {"force": jnp.array(1.0)}}
        )
        assert "physics" in cond._components
        assert "mass" in cond._components["physics"]
        assert "force" not in cond._components["physics"]

    def test_condition_on_positional_and_kwargs_exclusive(self, nested_joint):
        """Cannot mix positional dict and kwargs."""
        with pytest.raises(TypeError, match="either a positional dict or keyword"):
            nested_joint.condition_on(
                {"physics": {"force": jnp.array(1.0)}},
                observation=jnp.array(0.5),
            )

    def test_condition_on_group_with_scalar_raises(self, nested_joint):
        """Conditioning on a dict node with a scalar should raise TypeError."""
        with pytest.raises(TypeError, match="contains component distributions"):
            condition_on(nested_joint, physics=jnp.array(1.0))

    def test_condition_on_leaf_with_dict_raises(self, nested_joint):
        """Passing a dict for a component distribution should raise TypeError."""
        with pytest.raises(TypeError, match="component distribution"):
            nested_joint.condition_on(
                observation={"sub": jnp.array(1.0)}
            )

    def test_condition_on_unknown_nested_key_raises(self, nested_joint):
        """Unknown key inside a nested dict should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            nested_joint.condition_on(
                physics={"nonexistent": jnp.array(1.0)}
            )

    def test_condition_on_all_raises(self, nested_joint):
        """Conditioning on all leaves should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot condition on all"):
            nested_joint.condition_on(
                physics={"force": jnp.array(1.0), "mass": jnp.array(2.0)},
                observation=jnp.array(0.5),
            )

    def test_condition_on_provenance(self, nested_joint):
        """Conditioned distribution should have provenance."""
        cond = nested_joint.condition_on(
            physics={"force": jnp.array(1.0)}
        )
        assert cond.source is not None
        assert cond.source.operation == "condition_on"

    def test_condition_on_empty_raises(self, nested_joint):
        """Calling condition_on with no arguments should raise."""
        with pytest.raises(ValueError, match="at least one"):
            condition_on(nested_joint)

    def test_condition_on_empty_dict_raises(self, nested_joint):
        """Calling condition_on with an empty dict should raise."""
        with pytest.raises(ValueError, match="at least one"):
            condition_on(nested_joint, {})

    # -- repr --------------------------------------------------------------

    def test_repr_nested(self, nested_joint):
        r = repr(nested_joint)
        assert "ProductDistribution" in r
        # Should include component names from the nested structure
        assert "force" in r
        assert "mass" in r
        assert "observation" in r

    # -- JAX pytree registration -------------------------------------------

    def test_pytree_roundtrip(self, nested_joint):
        children, aux = jax.tree_util.tree_flatten(nested_joint)
        reconstructed = jax.tree_util.tree_unflatten(aux, children)
        assert isinstance(reconstructed, ProductDistribution)
        assert reconstructed.event_size == nested_joint.event_size
        # Verify sample structure matches
        key = jax.random.PRNGKey(40)
        s1 = sample(nested_joint, key=key)
        s2 = sample(reconstructed, key=key)
        for l1, l2 in zip(jax.tree.leaves(s1), jax.tree.leaves(s2)):
            np.testing.assert_allclose(l1, l2, atol=1e-6)

    # -- broadcasting with nested views ------------------------------------

    def test_workflow_broadcasting_nested(self, nested_joint):
        """DistributionViews from nested joints should work in WorkflowFunction."""
        view_force = nested_joint["physics", "force"]
        view_obs = nested_joint["observation"]

        def add(a: float, b: float) -> float:
            return a + b

        wf = WorkflowFunction(
            func=add,
            vectorize="loop",
            n_broadcast_samples=30,
            seed=42,
        )
        result = wf(a=view_force, b=view_obs)
        assert isinstance(result, EmpiricalDistribution)
        assert result.n == 30


# ===========================================================================
# 9. TestNestedWithMVN
# ===========================================================================

class TestNestedWithMVN:
    """Nested dicts where some leaves are multivariate (non-scalar event)."""

    @pytest.fixture
    def nested_mvn(self):
        return ProductDistribution(
            group={"position": MultivariateNormal(
                        loc=jnp.zeros(2), cov=jnp.eye(2)),
                   "scale": Normal(loc=1.0, scale=0.1)},
            label=Normal(loc=0.0, scale=1.0),
        )

    def test_event_shapes(self, nested_mvn):
        es = nested_mvn.event_shapes
        assert es["group"]["position"] == (2,)
        assert es["group"]["scale"] == ()
        assert es["label"] == ()

    def test_event_size(self, nested_mvn):
        # 2 (MVN) + 1 (scalar) + 1 (scalar) = 4
        assert nested_mvn.event_size == 4

    def test_sample_and_flatten(self, nested_mvn):
        key = jax.random.PRNGKey(50)
        s = sample(nested_mvn, key=key, sample_shape=(5,))
        assert s["group"]["position"].shape == (5, 2)
        assert s["group"]["scale"].shape == (5,)
        assert s["label"].shape == (5,)

        flat = nested_mvn.flatten_value(s)
        assert flat.shape == (5, 4)

        recovered = nested_mvn.unflatten_value(flat)
        np.testing.assert_allclose(
            recovered["group"]["position"], s["group"]["position"], atol=1e-6
        )

    def test_getitem_mvn_leaf(self, nested_mvn):
        view = nested_mvn["group", "position"]
        assert view.event_shape == (2,)
        key = jax.random.PRNGKey(51)
        s = sample(view, key=key, sample_shape=(3,))
        assert s.shape == (3, 2)
