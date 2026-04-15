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
    EmpiricalDistribution,
    ArrayDistribution,
    RecordDistribution,
)
from probpipe.core._record_distribution import _RecordDistributionView
from probpipe.core.record import Record
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
        assert isinstance(joint, RecordDistribution)

    def test_construction_with_mvn(self, normal_x, mvn_z):
        joint = ProductDistribution(x=normal_x, z=mvn_z)
        assert isinstance(joint, ProductDistribution)

    def test_isinstance_record_distribution(self, joint_xy):
        assert isinstance(joint_xy, RecordDistribution)
        assert not isinstance(joint_xy, ArrayDistribution)
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

    def test_component_names_sorted(self, normal_x, normal_y):
        joint = ProductDistribution(y=normal_y, x=normal_x)
        # RecordDistribution stores fields in sorted order
        assert joint.component_names == ("x", "y")

    def test_sample_returns_values(self, joint_xy):
        key = jax.random.PRNGKey(0)
        s = sample(joint_xy, key=key)
        assert isinstance(s, Record)
        assert set(s.fields) == {"x", "y"}

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

    def test_log_prob_accepts_values(self, joint_xy, normal_x, normal_y):
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

    def test_mean_returns_values(self, joint_xz, normal_x, mvn_z):
        m = mean(joint_xz)
        assert isinstance(m, Record)
        np.testing.assert_allclose(m["x"], mean(normal_x), atol=1e-6)
        np.testing.assert_allclose(m["z"], mean(mvn_z), atol=1e-6)

    def test_variance_returns_values(self, joint_xz, normal_x, mvn_z):
        v = variance(joint_xz)
        assert isinstance(v, Record)
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

    # -- Dict-like interface (shared with Record) -----------------------------

    def test_fields_matches_component_names(self, joint_xy):
        assert joint_xy.fields == joint_xy.component_names

    def test_contains_existing(self, joint_xy):
        assert "x" in joint_xy
        assert "y" in joint_xy

    def test_contains_missing(self, joint_xy):
        assert "z" not in joint_xy

    def test_keys(self, joint_xy):
        assert list(joint_xy.keys()) == list(joint_xy.component_names)

    def test_values(self, joint_xy):
        vals = list(joint_xy.values())
        assert len(vals) == 2
        assert all(isinstance(v, _RecordDistributionView) for v in vals)

    def test_items(self, joint_xy):
        items = dict(joint_xy.items())
        assert set(items.keys()) == {"x", "y"}
        assert all(isinstance(v, _RecordDistributionView) for v in items.values())


# ===========================================================================
# 2. TestFlattenUnflatten
# ===========================================================================

class TestFlattenUnflatten:
    """Test flatten_value / unflatten_value (inherited from RecordDistribution)."""

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
        s = sample(joint_xy, key=key)
        assert isinstance(s, Record)
        flat = joint_xy.flatten_value(s)
        assert flat.shape == (2,)
        recovered = joint_xy.unflatten_value(flat)
        assert isinstance(recovered, Record)
        np.testing.assert_allclose(recovered["x"], s["x"], atol=1e-6)
        np.testing.assert_allclose(recovered["y"], s["y"], atol=1e-6)

    def test_roundtrip_mixed_components(self, joint_xz):
        key = jax.random.PRNGKey(61)
        s = sample(joint_xz, key=key)
        assert isinstance(s, Record)
        flat = joint_xz.flatten_value(s)
        assert flat.shape == (4,)
        recovered = joint_xz.unflatten_value(flat)
        assert isinstance(recovered, Record)
        np.testing.assert_allclose(recovered["x"], s["x"], atol=1e-6)
        np.testing.assert_allclose(recovered["z"], s["z"], atol=1e-6)

    def test_roundtrip_no_batch_dim(self, joint_xy):
        key = jax.random.PRNGKey(62)
        s = sample(joint_xy, key=key)
        assert isinstance(s, Record)
        flat = joint_xy.flatten_value(s)
        assert flat.shape == (2,)
        recovered = joint_xy.unflatten_value(flat)
        assert isinstance(recovered, Record)
        np.testing.assert_allclose(recovered["x"], s["x"], atol=1e-6)
        np.testing.assert_allclose(recovered["y"], s["y"], atol=1e-6)



# ===========================================================================
# 4. TestDistributionView (RecordDistributionView for ProductDistribution)
# ===========================================================================

class TestDistributionView:

    def test_view_type(self, joint_xy):
        view = joint_xy["x"]
        assert isinstance(view, _RecordDistributionView)

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

    def test_mean_matches_component(self, joint_xz, mvn_z):
        view = joint_xz["z"]
        np.testing.assert_allclose(mean(view), mean(mvn_z), atol=1e-6)

    def test_parent_reference(self, joint_xy):
        view = joint_xy["x"]
        assert view._parent is joint_xy

    def test_key(self, joint_xy):
        view = joint_xy["y"]
        assert view._key == "y"

    def test_keyerror_invalid_component(self, joint_xy):
        with pytest.raises(KeyError, match="not_a_component"):
            joint_xy["not_a_component"]

    def test_repr(self, joint_xy):
        view = joint_xy["x"]
        r = repr(view)
        assert "_RecordDistributionView" in r
        assert "ProductDistribution" in r
        assert "'x'" in r




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
        assert isinstance(s, Record)
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
# 10. TestBroadcastingReconnection
# ===========================================================================

class TestBroadcastingReconnection:
    """Test that _RecordDistributionViews from the same parent are sampled jointly."""

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
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=10.0, scale=1.0, name="y"),
        )
        wf = self._make_add_workflow("loop")
        result = wf(a=joint["x"], b=joint["y"])
        assert hasattr(result, "samples")
        # x ~ N(0,1), y ~ N(10,1) => a+b ~ N(10, sqrt(2))
        mean_val = float(jnp.mean(result.samples))
        assert abs(mean_val - 10.0) < 2.0

    def test_same_view_twice_gives_identical_values_loop(self):
        """If joint['x'] is passed to both args, they get the same values (loop)."""
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=10.0, scale=1.0, name="y"),
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
        assert hasattr(result, "samples")
        # a and b are the same samples, so a - b = 0 for every sample
        np.testing.assert_allclose(
            np.array(result.samples), 0.0, atol=1e-5
        )

    def test_mix_of_view_and_independent(self):
        """Mix of _RecordDistributionView and independent Normal both work."""
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=5.0, scale=1.0, name="y"),
        )
        independent = Normal(loc=100.0, scale=0.1, name="c")

        def add3(a: float, b: float, c: float) -> float:
            return a + b + c

        wf = WorkflowFunction(
            func=add3,
            vectorize="loop",
            n_broadcast_samples=50,
            seed=77,
        )
        result = wf(a=joint["x"], b=joint["y"], c=independent)
        assert hasattr(result, "samples")
        # x ~ N(0,1), y ~ N(5,1), c ~ N(100, 0.1) => sum ~ N(105, ...)
        mean_val = float(jnp.mean(result.samples))
        assert abs(mean_val - 105.0) < 3.0

    def test_joint_views_sampled_together_jax(self):
        """Two views from same parent using jax backend."""
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=10.0, scale=1.0, name="y"),
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
        assert hasattr(result, "samples")
        mean_val = float(jnp.mean(result.samples))
        assert abs(mean_val - 10.0) < 2.0

    def test_same_view_twice_jax_backend(self):
        """Same _RecordDistributionView to both args with jax backend."""
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=10.0, scale=1.0, name="y"),
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
        assert hasattr(result, "samples")
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

    def test_pytree_leaves_returns_components(self, normal_x, normal_y):
        joint = ProductDistribution(x=normal_x, y=normal_y)
        leaves = jax.tree.leaves(joint)
        assert len(leaves) == 2

    def test_pytree_roundtrip(self, normal_x, normal_y):
        joint = ProductDistribution(x=normal_x, y=normal_y, name="j")
        children, aux = jax.tree.flatten(joint)
        reconstructed = jax.tree.unflatten(aux, children)
        assert isinstance(reconstructed, ProductDistribution)
        assert reconstructed.component_names == ("x", "y")
        assert reconstructed._name == "j"


# ===========================================================================
# 11. Dynamic protocol support
# ===========================================================================

class TestProductProtocolDuckTyping:
    """ProductDistribution dynamically includes protocols based on components."""

    def test_all_log_prob_components(self):
        """All Normal components → isinstance SupportsLogProb True."""
        from probpipe import SupportsLogProb
        joint = ProductDistribution(x=Normal(0, 1, name="x"), y=Normal(1, 2, name="y"))
        assert isinstance(joint, SupportsLogProb)

    def test_all_mean_variance_components(self):
        """All Normal components → isinstance SupportsMean/SupportsVariance True."""
        from probpipe import SupportsMean, SupportsVariance
        joint = ProductDistribution(x=Normal(0, 1, name="x"), y=Normal(1, 2, name="y"))
        assert isinstance(joint, SupportsMean)
        assert isinstance(joint, SupportsVariance)

    def test_mixed_no_log_prob(self):
        """Component lacking SupportsLogProb → product lacks it too."""
        from probpipe import SupportsLogProb, BootstrapDistribution
        boot = BootstrapDistribution(jnp.array([1.0, 2.0, 3.0]), name="y")
        joint = ProductDistribution(x=Normal(0, 1, name="x"), y=boot)
        assert not isinstance(joint, SupportsLogProb)

    def test_always_supports_sampling(self):
        """ProductDistribution always supports SupportsSampling."""
        from probpipe import SupportsSampling
        joint = ProductDistribution(x=Normal(0, 1, name="x"), y=Normal(1, 2, name="y"))
        assert isinstance(joint, SupportsSampling)

    def test_always_supports_conditioning(self):
        """ProductDistribution always supports SupportsConditioning."""
        from probpipe import SupportsConditioning
        joint = ProductDistribution(x=Normal(0, 1, name="x"), y=Normal(1, 2, name="y"))
        assert isinstance(joint, SupportsConditioning)

    def test_dynamic_subclass_pytree_roundtrip(self):
        """Dynamic ProductDistribution subclass is JAX pytree-compatible."""
        joint = ProductDistribution(x=Normal(0, 1, name="x"), y=Normal(1, 2, name="y"))
        children, aux = jax.tree.flatten(joint)
        reconstructed = jax.tree.unflatten(aux, children)
        assert isinstance(reconstructed, ProductDistribution)
        assert reconstructed.component_names == ("x", "y")


# ===========================================================================
# 12. Additional coverage
# ===========================================================================

class TestSingleComponent:
    """ProductDistribution with one component."""

    def test_single_component_event_size(self):
        joint = ProductDistribution(a=Normal(loc=0.0, scale=1.0, name="a"))
        assert joint.event_size == 1

    def test_single_component_sample(self):
        joint = ProductDistribution(a=Normal(loc=0.0, scale=1.0, name="a"))
        key = jax.random.PRNGKey(70)
        s = sample(joint, key=key, sample_shape=(5,))
        assert s["a"].shape == (5,)

    def test_single_component_log_prob(self):
        n = Normal(loc=0.0, scale=1.0, name="a")
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
            s_i = Record({k: v[i] for k, v in samples.items()})
            expected = float(log_prob(normal_x, s_i["x"])) + float(log_prob(normal_y, s_i["y"]))
            np.testing.assert_allclose(float(batch_lps[i]), expected, atol=1e-5)


class TestEmptyComponentsValidation:

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            ProductDistribution()


class TestDistributionViewFromDistribution:

    def test_from_distribution_raises(self, joint_xy):
        with pytest.raises(TypeError):
            from_distribution(Normal(loc=0.0, scale=1.0, name="x"), _RecordDistributionView)


class TestEnumerateWithDistributionViews:
    """Verify _RecordDistributionView reconnection works in the enumerate path."""

    def test_empirical_plus_views_preserves_correlation(self):
        """When an empirical and two correlated views are mixed, views stay paired."""
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=0.0, scale=1.0, name="y"),
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
        assert hasattr(result, "samples")
        assert result.n == 50


# ===========================================================================
# 8. TestNestedProductDistribution
# ===========================================================================

class TestNestedProductDistribution:
    """Tests for ProductDistribution with nested dict components.

    A nested ProductDistribution groups components into sub-dicts::

        ProductDistribution(
            physics={"force": Normal(0, 1, name="force"), "mass": Gamma(2, 1, name="mass")},
            observation=Normal(0, 0.1, name="observation"),
        )

    The nesting is purely organizational — all leaf components remain
    statistically independent.  Samples return nested Record objects.
    """

    @pytest.fixture
    def nested_joint(self):
        return ProductDistribution(
            physics={"force": Normal(loc=0.0, scale=1.0, name="force"),
                     "mass": Gamma(concentration=2.0, rate=1.0, name="mass")},
            observation=Normal(loc=0.0, scale=0.1, name="observation"),
        )

    # -- Construction and introspection ------------------------------------

    def test_isinstance(self, nested_joint):
        assert isinstance(nested_joint, ProductDistribution)
        assert isinstance(nested_joint, RecordDistribution)
        assert not isinstance(nested_joint, ArrayDistribution)
        assert not isinstance(nested_joint, ArrayDistribution)

    def test_event_shapes_nested(self, nested_joint):
        es = nested_joint.event_shapes
        assert isinstance(es, dict)
        # RecordDistribution.event_shapes returns top-level fields;
        # nested Record fields report () for sub-Record
        assert "observation" in es
        assert "physics" in es

    def test_event_size(self, nested_joint):
        # 3 scalar leaves -> event_size = 3
        assert nested_joint.event_size == 3

    def test_component_names_are_top_level(self, nested_joint):
        names = nested_joint.component_names
        # RecordDistribution.component_names returns top-level field names
        assert len(names) == 2
        assert "observation" in names
        assert "physics" in names

    # -- Sampling ----------------------------------------------------------

    def test_sample_returns_nested_values(self, nested_joint):
        key = jax.random.PRNGKey(1)
        s = sample(nested_joint, key=key)
        assert isinstance(s, Record)
        assert isinstance(s["physics"], Record)
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

    def test_log_prob_accepts_nested_values(self, nested_joint):
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
        force_dist = Normal(loc=0.0, scale=1.0, name="force")
        mass_dist = Gamma(concentration=2.0, rate=1.0, name="mass")
        obs_dist = Normal(loc=0.0, scale=0.1, name="observation")
        lp_manual = (
            log_prob(force_dist, s["physics"]["force"])
            + log_prob(mass_dist, s["physics"]["mass"])
            + log_prob(obs_dist, s["observation"])
        )
        np.testing.assert_allclose(float(lp_joint), float(lp_manual), atol=1e-5)

    # -- mean / variance ---------------------------------------------------

    def test_mean_nested(self, nested_joint):
        m = mean(nested_joint)
        assert isinstance(m, Record)
        assert isinstance(m["physics"], Record)
        assert m["physics"]["force"].shape == ()
        assert m["physics"]["mass"].shape == ()
        assert m["observation"].shape == ()

    def test_variance_nested(self, nested_joint):
        v = variance(nested_joint)
        assert isinstance(v, Record)
        assert isinstance(v["physics"], Record)
        assert jnp.all(jnp.asarray(jax.tree.leaves(v)) >= 0)

    # -- flatten / unflatten -----------------------------------------------

    def test_flatten_unflatten_roundtrip(self, nested_joint):
        key = jax.random.PRNGKey(20)
        s = sample(nested_joint, key=key)
        assert isinstance(s, Record)
        flat = nested_joint.flatten_value(s)
        assert flat.shape == (3,)
        recovered = nested_joint.unflatten_value(flat)
        assert isinstance(recovered, Record)
        # Check all leaves match
        for orig, rec in zip(jax.tree.leaves(s), jax.tree.leaves(recovered)):
            np.testing.assert_allclose(orig, rec, atol=1e-6)

    # -- __getitem__ --------------------------------------------------------

    def test_getitem_top_level(self, nested_joint):
        view = nested_joint["observation"]
        assert isinstance(view, _RecordDistributionView)
        assert view._key == "observation"

    def test_getitem_nested_field_returns_view(self, nested_joint):
        """Indexing a nested field returns a _RecordDistributionView."""
        view = nested_joint["physics"]
        assert isinstance(view, _RecordDistributionView)
        assert view._key == "physics"

    def test_getitem_invalid_raises(self, nested_joint):
        with pytest.raises(KeyError, match="nonexistent"):
            nested_joint["nonexistent"]

    def test_view_event_shape(self, nested_joint):
        assert nested_joint["observation"].event_shape == ()

    # -- select with field names --------------------------------------------

    def test_select_nested(self, nested_joint):
        views = nested_joint.select(
            p="physics",
            o="observation",
        )
        assert isinstance(views["p"], _RecordDistributionView)
        assert isinstance(views["o"], _RecordDistributionView)
        assert views["p"]._key == "physics"
        assert views["o"]._key == "observation"

    # -- condition_on -------------------------------------------------------

    def test_condition_on_top_level_leaf(self, nested_joint):
        """Condition on a top-level leaf component."""
        cond = condition_on(nested_joint, observation=jnp.array(0.5))
        assert "observation" not in cond._components
        assert "physics" in cond._components
        assert cond.event_size == 2  # force + mass

    def test_condition_on_all_leaves_in_group(self, nested_joint):
        """Conditioning on all components under a dict removes it."""
        cond = condition_on(nested_joint,
            physics={"force": jnp.array(1.0), "mass": jnp.array(2.0)}
        )
        assert "physics" not in cond._components
        assert "observation" in cond._components
        assert cond.event_size == 1

    def test_condition_on_single_nested_leaf(self, nested_joint):
        """Condition on one component, keeping the sibling."""
        cond = condition_on(nested_joint,
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
        cond = condition_on(nested_joint,
            physics={"force": jnp.array(1.0)}
        )
        key = jax.random.PRNGKey(50)
        s = sample(cond, key=key, sample_shape=(5,))
        assert isinstance(s, Record)
        assert "physics" in s
        assert isinstance(s["physics"], Record)
        assert "mass" in s["physics"]
        assert "force" not in s["physics"]
        assert "observation" in s
        assert s["physics"]["mass"].shape == (5,)
        assert s["observation"].shape == (5,)

    def test_condition_on_nested_leaf_log_prob(self, nested_joint):
        """Log-prob should work on conditioned nested joint."""
        cond = condition_on(nested_joint,
            physics={"force": jnp.array(1.0)}
        )
        key = jax.random.PRNGKey(51)
        s = sample(cond, key=key)
        lp = log_prob(cond, s)
        assert jnp.isfinite(lp)

    def test_condition_on_multiple_leaves_across_groups(self, nested_joint):
        """Condition on leaves from different groups simultaneously."""
        cond = condition_on(nested_joint,
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
        cond = condition_on(nested_joint,
            {"physics": {"force": jnp.array(1.0)}}
        )
        assert "physics" in cond._components
        assert "mass" in cond._components["physics"]
        assert "force" not in cond._components["physics"]

    def test_condition_on_positional_and_kwargs_exclusive(self, nested_joint):
        """Cannot mix positional dict and kwargs."""
        with pytest.raises(TypeError, match="either a positional dict or keyword"):
            condition_on(nested_joint,
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
            condition_on(nested_joint,
                observation={"sub": jnp.array(1.0)}
            )

    def test_condition_on_unknown_nested_key_raises(self, nested_joint):
        """Unknown key inside a nested dict should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            condition_on(nested_joint,
                physics={"nonexistent": jnp.array(1.0)}
            )

    def test_condition_on_all_raises(self, nested_joint):
        """Conditioning on all leaves should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot condition on all"):
            condition_on(nested_joint,
                physics={"force": jnp.array(1.0), "mass": jnp.array(2.0)},
                observation=jnp.array(0.5),
            )

    def test_condition_on_provenance(self, nested_joint):
        """Conditioned distribution should have provenance."""
        cond = condition_on(nested_joint,
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
        assert "physics" in r
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

    def test_workflow_broadcasting_nested(self):
        """_RecordDistributionViews from nested joints should work in WorkflowFunction."""
        # Use a flat joint so each view is a scalar for the add function
        joint = ProductDistribution(
            force=Normal(loc=0.0, scale=1.0, name="force"),
            observation=Normal(loc=0.0, scale=0.1, name="observation"),
        )
        view_force = joint["force"]
        view_obs = joint["observation"]

        def add(a: float, b: float) -> float:
            return a + b

        wf = WorkflowFunction(
            func=add,
            vectorize="loop",
            n_broadcast_samples=30,
            seed=42,
        )
        result = wf(a=view_force, b=view_obs)
        assert hasattr(result, "samples")
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
                        loc=jnp.zeros(2), cov=jnp.eye(2), name="position"),
                   "scale": Normal(loc=1.0, scale=0.1, name="scale")},
            label=Normal(loc=0.0, scale=1.0, name="label"),
        )

    def test_event_shapes(self, nested_mvn):
        es = nested_mvn.event_shapes
        # RecordDistribution.event_shapes returns top-level fields;
        # nested Record fields report () for sub-Record
        assert es["group"] == ()
        assert es["label"] == ()

    def test_event_size(self, nested_mvn):
        # 2 (MVN) + 1 (scalar) + 1 (scalar) = 4
        assert nested_mvn.event_size == 4

    def test_sample_and_flatten(self, nested_mvn):
        key = jax.random.PRNGKey(50)
        s = sample(nested_mvn, key=key, sample_shape=(5,))
        assert isinstance(s, Record)
        assert isinstance(s["group"], Record)
        assert s["group"]["position"].shape == (5, 2)
        assert s["group"]["scale"].shape == (5,)
        assert s["label"].shape == (5,)

        # Flatten a single sample (no batch dim)
        s_single = sample(nested_mvn, key=key)
        flat = nested_mvn.flatten_value(s_single)
        assert flat.shape == (4,)

        recovered = nested_mvn.unflatten_value(flat)
        assert isinstance(recovered, Record)
        np.testing.assert_allclose(
            recovered["group"]["position"], s_single["group"]["position"], atol=1e-6
        )

    def test_getitem_top_level(self, nested_mvn):
        view = nested_mvn["label"]
        assert isinstance(view, _RecordDistributionView)
        assert view._key == "label"
        assert view.event_shape == ()
