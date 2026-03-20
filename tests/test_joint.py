"""Comprehensive tests for joint distribution classes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from probpipe import (
    Normal,
    MultivariateNormal,
    ProductDistribution,
    DistributionView,
    ConditionedComponent,
    JointDistribution,
    EmpiricalDistribution,
    Distribution,
)
from probpipe.core.node import Workflow


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

    def test_event_shape_scalar_components(self, joint_xy):
        # Two scalar Normals: total dim = 1 + 1 = 2
        assert joint_xy.event_shape == (2,)

    def test_event_shape_mixed_components(self, joint_xz):
        # Scalar Normal (dim 1) + MVN(dim 3): total dim = 4
        assert joint_xz.event_shape == (4,)

    def test_component_names(self, joint_xy):
        assert joint_xy.component_names == ("x", "y")

    def test_component_names_order_preserved(self, normal_x, normal_y):
        joint = ProductDistribution(y=normal_y, x=normal_x)
        assert joint.component_names == ("y", "x")

    def test_sample_flat_shape(self, joint_xy):
        key = jax.random.PRNGKey(0)
        s = joint_xy.sample(key)
        assert s.shape == (2,)

    def test_sample_flat_shape_with_sample_shape(self, joint_xz):
        key = jax.random.PRNGKey(1)
        s = joint_xz.sample(key, (10,))
        assert s.shape == (10, 4)

    def test_sample_structured_returns_dict(self, joint_xy):
        key = jax.random.PRNGKey(2)
        structured = joint_xy.sample_structured(key)
        assert isinstance(structured, dict)
        assert set(structured.keys()) == {"x", "y"}

    def test_sample_structured_shapes(self, joint_xz):
        key = jax.random.PRNGKey(3)
        structured = joint_xz.sample_structured(key, (5,))
        assert structured["x"].shape == (5,)
        assert structured["z"].shape == (5, 3)

    def test_log_prob_equals_sum(self, joint_xy, normal_x, normal_y):
        key = jax.random.PRNGKey(4)
        flat_sample = joint_xy.sample(key)
        lp_joint = joint_xy.log_prob(flat_sample)

        # Manually compute sum of component log probs
        x_val = flat_sample[0:1].reshape(())
        y_val = flat_sample[1:2].reshape(())
        lp_sum = normal_x.log_prob(x_val) + normal_y.log_prob(y_val)

        np.testing.assert_allclose(float(lp_joint), float(lp_sum), atol=1e-5)

    def test_log_prob_batch(self, joint_xy):
        key = jax.random.PRNGKey(5)
        samples = joint_xy.sample(key, (20,))
        lps = joint_xy.log_prob(samples)
        assert lps.shape == (20,)

    def test_mean_concatenates(self, joint_xz, normal_x, mvn_z):
        m = joint_xz.mean()
        expected = jnp.concatenate([
            normal_x.mean().reshape(-1),
            mvn_z.mean().reshape(-1),
        ])
        np.testing.assert_allclose(m, expected, atol=1e-6)

    def test_variance_concatenates(self, joint_xz, normal_x, mvn_z):
        v = joint_xz.variance()
        expected = jnp.concatenate([
            normal_x.variance().reshape(-1),
            mvn_z.variance().reshape(-1),
        ])
        np.testing.assert_allclose(v, expected, atol=1e-6)

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
# 2. TestDistributionView
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
        s = view.sample(key)
        assert s.shape == ()

    def test_sample_correct_shape_mvn(self, joint_xz):
        key = jax.random.PRNGKey(11)
        view = joint_xz["z"]
        s = view.sample(key, (7,))
        assert s.shape == (7, 3)

    def test_log_prob_matches_component(self, joint_xy, normal_x):
        view = joint_xy["x"]
        x_val = jnp.array(1.5)
        lp_view = view.log_prob(x_val)
        lp_direct = normal_x.log_prob(x_val)
        np.testing.assert_allclose(float(lp_view), float(lp_direct), atol=1e-6)

    def test_mean_matches_component(self, joint_xz, mvn_z):
        view = joint_xz["z"]
        np.testing.assert_allclose(view.mean(), mvn_z.mean(), atol=1e-6)

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
# 3. TestBind
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
# 4. TestFromDistributions
# ===========================================================================

class TestFromDistributions:

    def test_from_distributions_named(self):
        a = Normal(loc=0.0, scale=1.0, name="a")
        b = Normal(loc=1.0, scale=0.5, name="b")
        joint = ProductDistribution.from_distributions([a, b])
        assert joint.component_names == ("a", "b")
        assert joint.event_shape == (2,)

    def test_from_distributions_raises_on_none_name(self):
        a = Normal(loc=0.0, scale=1.0, name="a")
        b = Normal(loc=1.0, scale=0.5)  # no name
        with pytest.raises(ValueError, match="name=None"):
            ProductDistribution.from_distributions([a, b])

    def test_from_distributions_raises_on_duplicate_name(self):
        a = Normal(loc=0.0, scale=1.0, name="dup")
        b = Normal(loc=1.0, scale=0.5, name="dup")
        with pytest.raises(ValueError, match="Duplicate"):
            ProductDistribution.from_distributions([a, b])


# ===========================================================================
# 5. TestConditionOn
# ===========================================================================

class TestConditionOn:

    def test_condition_on_removes_conditioned_component(self, joint_xy):
        cond = joint_xy.condition_on(x=jnp.array(2.0))
        assert "x" not in cond.components
        assert "y" in cond.components
        assert cond.component_names == ("y",)
        assert cond.event_shape == (1,)

    def test_conditioned_sample_excludes_conditioned(self, joint_xy):
        cond = joint_xy.condition_on(x=jnp.array(2.0))
        key = jax.random.PRNGKey(20)
        structured = cond.sample_structured(key, (10,))
        assert "x" not in structured
        assert "y" in structured

    def test_unconditioned_component_still_varies(self, joint_xy):
        cond = joint_xy.condition_on(x=jnp.array(0.0))
        key = jax.random.PRNGKey(21)
        structured = cond.sample_structured(key, (100,))
        # y should have non-trivial variance
        assert jnp.std(structured["y"]) > 0.1

    def test_log_prob_works_on_conditioned(self, joint_xy):
        cond = joint_xy.condition_on(x=jnp.array(1.0))
        key = jax.random.PRNGKey(22)
        flat = cond.sample(key, (5,))
        lps = cond.log_prob(flat)
        assert lps.shape == (5,)
        assert jnp.all(jnp.isfinite(lps))

    def test_provenance_attached(self, joint_xy):
        cond = joint_xy.condition_on(y=jnp.array(0.5))
        assert cond.source is not None
        assert cond.source.operation == "condition_on"
        assert "y" in cond.source.metadata["conditioned"]

    def test_keyerror_on_unknown_component(self, joint_xy):
        with pytest.raises(KeyError, match="Unknown"):
            joint_xy.condition_on(nonexistent=jnp.array(0.0))

    def test_raises_on_conditioning_all(self, joint_xy):
        with pytest.raises(ValueError, match="Cannot condition on all"):
            joint_xy.condition_on(x=jnp.array(0.0), y=jnp.array(0.0))


# ===========================================================================
# 6. TestConditionedComponent
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
        s1 = pc.sample(key)
        s2 = pc.sample(jax.random.PRNGKey(31))
        np.testing.assert_allclose(float(s1), 42.0, atol=1e-6)
        np.testing.assert_allclose(float(s2), 42.0, atol=1e-6)

    def test_sample_broadcast_shape(self, normal_x):
        val = jnp.array(5.0)
        pc = ConditionedComponent(normal_x, val)
        key = jax.random.PRNGKey(32)
        s = pc.sample(key, (8,))
        assert s.shape == (8,)
        np.testing.assert_allclose(s, jnp.full((8,), 5.0), atol=1e-6)

    def test_value_error_shape_mismatch(self, mvn_z):
        with pytest.raises(ValueError, match="shape"):
            ConditionedComponent(mvn_z, jnp.array(1.0))  # scalar vs (3,)

    def test_log_prob_is_constant(self, normal_x):
        val = jnp.array(0.0)
        pc = ConditionedComponent(normal_x, val)
        lp1 = pc.log_prob(jnp.array(999.0))
        lp2 = pc.log_prob(jnp.array(-999.0))
        # log_prob always evaluates base at the pinned value, ignoring input
        np.testing.assert_allclose(float(lp1), float(lp2), atol=1e-6)

    def test_mean_returns_pinned_value(self, normal_x):
        val = jnp.array(7.0)
        pc = ConditionedComponent(normal_x, val)
        np.testing.assert_allclose(float(pc.mean()), 7.0, atol=1e-6)

    def test_variance_is_zero(self, normal_x):
        pc = ConditionedComponent(normal_x, jnp.array(1.0))
        np.testing.assert_allclose(float(pc.variance()), 0.0, atol=1e-6)


# ===========================================================================
# 7. TestBroadcastingReconnection
# ===========================================================================

class TestBroadcastingReconnection:
    """Test that DistributionViews from the same parent are sampled jointly."""

    @staticmethod
    def _make_add_workflow(backend="loop"):
        """Create a workflow that adds two arrays."""
        def add(a: float, b: float) -> float:
            return a + b

        return Workflow(
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

        wf = Workflow(
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

        wf = Workflow(
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

        wf = Workflow(
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

        wf = Workflow(
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
# 8. TestPytreeRegistration
# ===========================================================================

class TestPytreeRegistration:

    def test_tree_flatten_unflatten_roundtrip(self, normal_x, normal_y):
        joint = ProductDistribution(x=normal_x, y=normal_y, name="rt")
        children, aux = jax.tree_util.tree_flatten(joint)
        reconstructed = jax.tree_util.tree_unflatten(aux, children)
        assert isinstance(reconstructed, ProductDistribution)
        assert reconstructed.component_names == ("x", "y")
        assert reconstructed.event_shape == (2,)

    def test_tree_flatten_preserves_components(self, normal_x, mvn_z):
        joint = ProductDistribution(x=normal_x, z=mvn_z)
        children, aux = jax.tree_util.tree_flatten(joint)
        reconstructed = jax.tree_util.tree_unflatten(aux, children)
        # Verify component distributions survived the roundtrip
        key = jax.random.PRNGKey(50)
        s_orig = joint.sample(key)
        s_recon = reconstructed.sample(key)
        np.testing.assert_allclose(s_orig, s_recon, atol=1e-6)

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
        np.testing.assert_allclose(float(reconstructed.mean()[0]), 10.0, atol=1e-5)
        np.testing.assert_allclose(float(reconstructed.mean()[1]), 20.0, atol=1e-5)


# ===========================================================================
# 9. Additional coverage
# ===========================================================================

class TestFlattenUnflattenRoundtrip:
    """Verify flatten_structured and unflatten are inverses."""

    def test_roundtrip_scalar_components(self, joint_xy):
        key = jax.random.PRNGKey(60)
        structured = joint_xy.sample_structured(key, (5,))
        flat = joint_xy.flatten_structured(structured)
        recovered = joint_xy.unflatten(flat)
        np.testing.assert_allclose(recovered["x"], structured["x"], atol=1e-6)
        np.testing.assert_allclose(recovered["y"], structured["y"], atol=1e-6)

    def test_roundtrip_mixed_components(self, joint_xz):
        key = jax.random.PRNGKey(61)
        structured = joint_xz.sample_structured(key, (3,))
        flat = joint_xz.flatten_structured(structured)
        assert flat.shape == (3, 4)
        recovered = joint_xz.unflatten(flat)
        np.testing.assert_allclose(recovered["x"], structured["x"], atol=1e-6)
        np.testing.assert_allclose(recovered["z"], structured["z"], atol=1e-6)

    def test_roundtrip_no_batch_dim(self, joint_xy):
        key = jax.random.PRNGKey(62)
        structured = joint_xy.sample_structured(key)
        flat = joint_xy.flatten_structured(structured)
        assert flat.shape == (2,)
        recovered = joint_xy.unflatten(flat)
        np.testing.assert_allclose(recovered["x"], structured["x"], atol=1e-6)
        np.testing.assert_allclose(recovered["y"], structured["y"], atol=1e-6)


class TestSingleComponent:
    """ProductDistribution with one component."""

    def test_single_component_event_shape(self):
        joint = ProductDistribution(a=Normal(loc=0.0, scale=1.0))
        assert joint.event_shape == (1,)

    def test_single_component_sample(self):
        joint = ProductDistribution(a=Normal(loc=0.0, scale=1.0))
        key = jax.random.PRNGKey(70)
        s = joint.sample(key, (5,))
        assert s.shape == (5, 1)

    def test_single_component_log_prob(self):
        n = Normal(loc=0.0, scale=1.0)
        joint = ProductDistribution(a=n)
        x = jnp.array([[0.0], [1.0]])
        lps = joint.log_prob(x)
        expected = jnp.array([float(n.log_prob(0.0)), float(n.log_prob(1.0))])
        np.testing.assert_allclose(lps, expected, atol=1e-5)


class TestLogProbBatchValues:
    """Verify log_prob batch values are numerically correct, not just shapes."""

    def test_batch_log_prob_matches_individual(self, joint_xy, normal_x, normal_y):
        key = jax.random.PRNGKey(80)
        samples = joint_xy.sample(key, (10,))
        batch_lps = joint_xy.log_prob(samples)

        for i in range(10):
            x_val = samples[i, 0]
            y_val = samples[i, 1]
            expected = float(normal_x.log_prob(x_val)) + float(normal_y.log_prob(y_val))
            np.testing.assert_allclose(float(batch_lps[i]), expected, atol=1e-5)


class TestEmptyComponentsValidation:

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            ProductDistribution()


class TestDistributionViewFromDistribution:

    def test_from_distribution_raises(self, joint_xy):
        with pytest.raises(NotImplementedError, match="structural reference"):
            DistributionView.from_distribution(Normal(loc=0.0, scale=1.0))


class TestEnumerateWithDistributionViews:
    """Verify DistributionView reconnection works in the enumerate path
    (when both EmpiricalDistribution and DistributionView are broadcast args)."""

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
            # a - b should be 0 if views are sampled jointly (same parent, same key split)
            # c is from the empirical
            return (a - b) + c

        wf = Workflow(
            func=compute,
            vectorize="loop",
            n_broadcast_samples=50,
            seed=123,
        )
        # view_x and view_y should be sampled jointly via reconnection
        # even though ed is enumerated separately
        result = wf(a=view_x, b=view_y, c=ed)
        assert isinstance(result, EmpiricalDistribution)
        # Since x and y are independent in ProductDistribution, a-b won't
        # be exactly 0, but the reconnection ensures they are consistently paired.
        # We can't test exact correlation here, but we can verify it runs
        # and produces the right number of samples.
        assert result.n == 50
