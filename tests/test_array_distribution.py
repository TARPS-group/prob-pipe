"""Tests for NumericRecordDistribution, FlattenedView, and shape semantics."""

import jax
import jax.numpy as jnp
import math
import numpy as np
import pytest

from probpipe import (
    Distribution,
    NumericRecordDistribution,
    RecordEmpiricalDistribution,
    FlattenedView,
    Normal,
    MultivariateNormal,
    from_distribution,
    EmpiricalDistribution,
)
from probpipe import log_prob, sample, unnormalized_log_prob


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def scalar_normal():
    return Normal(loc=0.0, scale=1.0, name="x")


@pytest.fixture
def vector_mvn():
    return MultivariateNormal(
        loc=jnp.zeros(3),
        cov=jnp.eye(3),
        name="z",
    )


@pytest.fixture
def matrix_mvn():
    """MVN with event_shape (4,) to test flatten/unflatten with non-trivial shapes."""
    return MultivariateNormal(
        loc=jnp.zeros(4),
        cov=jnp.eye(4),
        name="w",
    )


# ---------------------------------------------------------------------------
# Hierarchy checks
# ---------------------------------------------------------------------------

class TestHierarchy:
    def test_arraydist_is_pytreearraydist(self, scalar_normal):
        assert isinstance(scalar_normal, NumericRecordDistribution)

    def test_arraydist_is_distribution(self, scalar_normal):
        assert isinstance(scalar_normal, Distribution)

    def test_pytreearraydist_is_distribution(self, scalar_normal):
        """NumericRecordDistribution inherits NumericRecordDistribution which inherits Distribution."""
        assert isinstance(scalar_normal, Distribution)
        assert isinstance(scalar_normal, NumericRecordDistribution)
        assert isinstance(scalar_normal, NumericRecordDistribution)


# ---------------------------------------------------------------------------
# Distribution[T] base class methods
# ---------------------------------------------------------------------------

class TestDistributionBase:
    """Tests for methods defined on Distribution[T] itself."""

    def test_log_prob_raises_by_default(self):
        """Distribution[T] without SupportsLogProb raises TypeError."""
        class StubDist(Distribution):
            pass
        d = StubDist(name="stub")
        with pytest.raises(TypeError, match="does not support log_prob"):
            log_prob(d, jnp.array(0.0))

    def test_unnormalized_log_prob_delegates_to_log_prob(self, scalar_normal):
        """unnormalized_log_prob defaults to log_prob."""
        val = jnp.array(0.5)
        np.testing.assert_allclose(
            unnormalized_log_prob(scalar_normal, val),
            log_prob(scalar_normal, val),
            atol=1e-6,
        )

    def test_repr_with_name(self):
        """Distribution.__repr__ includes the name when set."""
        n = Normal(loc=0.0, scale=1.0, name="my_normal")
        r = repr(n)
        assert "my_normal" in r

    def test_repr_includes_class_and_name(self):
        """Distribution.__repr__ includes both the class name and the name."""
        n = Normal(loc=0.0, scale=1.0, name="x")
        r = repr(n)
        assert "Normal" in r
        assert "x" in r

    def test_from_distribution_on_base_class(self, scalar_normal):
        """from_distribution is accessible on Distribution[T] base."""
        # Normal inherits from_distribution from Distribution[T]
        result = from_distribution(scalar_normal, Normal, num_samples=100)
        assert isinstance(result, Normal)


# ---------------------------------------------------------------------------
# NumericRecordDistribution trivial pytree interface
# ---------------------------------------------------------------------------

class TestArrayDistributionPyTreeInterface:
    def test_treedef_scalar(self, scalar_normal):
        td = scalar_normal.treedef
        assert td == jax.tree.structure(None)

    def test_treedef_vector(self, vector_mvn):
        td = vector_mvn.treedef
        assert td == jax.tree.structure(None)

    def test_event_shapes_dict(self, scalar_normal):
        assert scalar_normal.event_shapes == {"x": ()}

    def test_event_shapes_vector(self, vector_mvn):
        assert vector_mvn.event_shapes == {"z": (3,)}
        assert vector_mvn.event_shape == (3,)

    def test_flat_event_shapes_scalar(self, scalar_normal):
        fes = scalar_normal.flat_event_shapes
        assert fes == [()]

    def test_event_size_scalar(self, scalar_normal):
        assert scalar_normal.event_size == 1

    def test_event_size_vector(self, vector_mvn):
        assert vector_mvn.event_size == 3

    def test_event_size_4d(self, matrix_mvn):
        assert matrix_mvn.event_size == 4


# ---------------------------------------------------------------------------
# flatten_value / unflatten_value on NumericRecordDistribution
# ---------------------------------------------------------------------------

class TestArrayDistFlattenUnflatten:
    def test_flatten_vector_sample(self, vector_mvn, key):
        s = sample(vector_mvn, key=key)
        flat = vector_mvn.flatten_value(s)
        assert flat.shape == (3,)
        np.testing.assert_allclose(flat, s, atol=1e-6)

    def test_unflatten_vector_sample(self, vector_mvn, key):
        s = sample(vector_mvn, key=key)
        flat = vector_mvn.flatten_value(s)
        restored = vector_mvn.unflatten_value(flat)
        np.testing.assert_allclose(restored, s, atol=1e-6)

    def test_flatten_unflatten_roundtrip_batched(self, vector_mvn, key):
        samples = jnp.asarray(sample(vector_mvn, key=key, sample_shape=(5,)))
        flat = vector_mvn.flatten_value(samples)
        assert flat.shape == (5, 3)
        restored = vector_mvn.unflatten_value(flat)
        np.testing.assert_allclose(restored, samples, atol=1e-6)

    def test_flatten_unflatten_4d(self, matrix_mvn, key):
        s = sample(matrix_mvn, key=key)
        flat = matrix_mvn.flatten_value(s)
        assert flat.shape == (4,)
        restored = matrix_mvn.unflatten_value(flat)
        np.testing.assert_allclose(restored, s, atol=1e-6)


# ---------------------------------------------------------------------------
# as_flat_distribution / FlattenedView
# ---------------------------------------------------------------------------

class TestFlattenedView:
    def test_as_flat_returns_flattened_view(self, vector_mvn):
        flat_dist = vector_mvn.as_flat_distribution()
        assert isinstance(flat_dist, FlattenedView)
        assert isinstance(flat_dist, NumericRecordDistribution)

    def test_event_shape(self, vector_mvn):
        flat_dist = vector_mvn.as_flat_distribution()
        assert flat_dist.event_shape == (3,)

    def test_sample_shape(self, vector_mvn, key):
        flat_dist = vector_mvn.as_flat_distribution()
        s = sample(flat_dist, key=key)
        assert s.shape == (3,)

    def test_sample_batched(self, vector_mvn, key):
        flat_dist = vector_mvn.as_flat_distribution()
        samples = sample(flat_dist, key=key, sample_shape=(10,))
        assert samples.shape == (10, 3)

    def test_log_prob_matches(self, vector_mvn, key):
        flat_dist = vector_mvn.as_flat_distribution()
        s = sample(vector_mvn, key=key)
        flat_sample = vector_mvn.flatten_value(s)

        lp_original = log_prob(vector_mvn, s)
        lp_flat = log_prob(flat_dist, flat_sample)
        np.testing.assert_allclose(lp_flat, lp_original, atol=1e-5)

    def test_base_distribution(self, vector_mvn):
        flat_dist = vector_mvn.as_flat_distribution()
        assert flat_dist.base_distribution is vector_mvn

    def test_unflatten_sample(self, vector_mvn, key):
        flat_dist = vector_mvn.as_flat_distribution()
        flat_sample = sample(flat_dist, key=key)
        restored = flat_dist.unflatten_sample(flat_sample)
        np.testing.assert_allclose(
            restored, vector_mvn.unflatten_value(flat_sample), atol=1e-6
        )

    def test_repr(self, vector_mvn):
        flat_dist = vector_mvn.as_flat_distribution()
        r = repr(flat_dist)
        assert "FlattenedView" in r
        assert "MultivariateNormal" in r

    def test_4d_event_shape(self, matrix_mvn):
        flat_dist = matrix_mvn.as_flat_distribution()
        assert flat_dist.event_shape == (4,)

    def test_log_prob_roundtrip_4d(self, matrix_mvn, key):
        flat_dist = matrix_mvn.as_flat_distribution()
        s = sample(matrix_mvn, key=key)
        flat_sample = matrix_mvn.flatten_value(s)

        lp_original = log_prob(matrix_mvn, s)
        lp_flat = log_prob(flat_dist, flat_sample)
        np.testing.assert_allclose(lp_flat, lp_original, atol=1e-5)


# ---------------------------------------------------------------------------
# supports property
# ---------------------------------------------------------------------------

class TestSupports:
    def test_supports_is_per_field_dict(self, scalar_normal):
        """supports returns a per-field dict of constraints."""
        from probpipe import real
        result = scalar_normal.supports
        assert isinstance(result, dict)
        assert len(result) == 1
        # The single field's constraint should match .support
        assert list(result.values())[0] == scalar_normal.support
        assert list(result.values())[0] == real

    def test_dtypes_is_per_field_dict(self, scalar_normal):
        """dtypes returns a per-field dict of dtypes."""
        result = scalar_normal.dtypes
        assert isinstance(result, dict)
        assert len(result) == 1
        assert list(result.values())[0] == scalar_normal.dtype


# ---------------------------------------------------------------------------
# Canonical / convenience accessor pairs on NumericRecordDistribution (PR-D)
# ---------------------------------------------------------------------------


class TestCanonicalConvenience:
    """Pin the canonical / convenience accessor split documented in
    ``NumericRecordDistribution.__doc__``: canonical per-field
    accessors (``event_shapes`` / ``dtypes`` / ``supports``) are the
    source of truth; scalar convenience accessors (``event_shape`` /
    ``dtype`` / ``support``) derive and raise (or return ``None``)
    on multi-leaf templates.
    """

    @pytest.fixture
    def multi_leaf_dist(self):
        """A synthetic ``NumericRecordDistribution`` with a multi-leaf
        template (two scalar fields). Exercises the convenience-
        accessor multi-leaf guards that no concrete subclass shipping
        today triggers (every shipped class is single-leaf via the
        auto-template helper)."""
        from probpipe.core.record import RecordTemplate

        class TwoField(NumericRecordDistribution):
            # ``event_shape`` is still abstract on the base (the
            # auto-template helper reads it for single-leaf subclasses);
            # multi-leaf subclasses bypass that by overriding
            # ``record_template`` directly and let the convenience raise.
            @property
            def event_shape(self):
                raise TypeError("multi-leaf; use event_shapes")

            @property
            def record_template(self):
                return RecordTemplate(a=(), b=(2,))

            @property
            def dtypes(self):
                return {"a": jnp.float32, "b": jnp.int32}

            @property
            def supports(self):
                from probpipe import real
                return {"a": real, "b": real}

        return TwoField(name="two_field")

    def test_dtype_derives_from_dtypes_single_leaf(self, scalar_normal):
        """Single-leaf: ``dtype`` returns the sole dtype in ``dtypes``."""
        assert scalar_normal.dtype == list(scalar_normal.dtypes.values())[0]

    def test_dtype_returns_none_when_dtypes_mixed(self, multi_leaf_dist):
        """Multi-leaf with mixed dtypes: ``dtype`` is ``None``."""
        # Two fields, different dtypes → convenience returns None.
        assert multi_leaf_dist.dtype is None

    def test_support_raises_typeerror_on_multi_leaf(self, multi_leaf_dist):
        """Multi-leaf: ``support`` (the convenience) raises TypeError."""
        with pytest.raises(TypeError, match="not single-field"):
            _ = multi_leaf_dist.support

    def test_supports_canonical_on_multi_leaf(self, multi_leaf_dist):
        """Multi-leaf: ``supports`` returns the per-field dict."""
        from probpipe import real
        assert multi_leaf_dist.supports == {"a": real, "b": real}

    def test_treedef_leaf_for_single_leaf(self, scalar_normal):
        """Single-leaf: ``treedef`` is the leaf treedef (one-leaf pytree)."""
        assert scalar_normal.treedef == jax.tree.structure(None)

    def test_treedef_record_for_multi_leaf(self, multi_leaf_dist):
        """Multi-leaf: ``treedef`` matches a ``NumericRecord`` skeleton
        with the same field names — locks the relationship between
        ``record_template`` and the sample pytree (Story A)."""
        from probpipe import NumericRecord
        expected = jax.tree.structure(
            NumericRecord(a=jnp.zeros(()), b=jnp.zeros((2,)))
        )
        assert multi_leaf_dist.treedef == expected

    def test_flat_event_shapes_tree_walks_multi_leaf(self, multi_leaf_dist):
        """``flat_event_shapes`` is one entry per leaf in template
        field order — not a single-leaf-only ``[event_shape]``."""
        assert multi_leaf_dist.flat_event_shapes == [(), (2,)]


# ---------------------------------------------------------------------------
# Bernoulli / Categorical no longer report float32 (PR-D commit B fix)
# ---------------------------------------------------------------------------


class TestIntegerDtypeReporting:
    """Pre-PR-D, the base ``dtypes`` silently returned
    ``{name: default_float_dtype()}`` for every field, so every
    integer-valued distribution reported a float dtype. With
    ``dtypes`` canonical (subclasses must override), TFP's int
    dtypes flow through correctly.
    """

    def test_bernoulli_dtype_is_int32(self):
        from probpipe import Bernoulli
        assert Bernoulli(probs=0.5, name="x").dtype == jnp.int32

    def test_categorical_dtype_is_int32(self):
        from probpipe import Categorical
        assert Categorical(probs=jnp.array([0.5, 0.5]), name="x").dtype == jnp.int32

    def test_normal_dtype_is_float(self):
        """Normal continues to report float (no regression on the
        always-float family)."""
        from probpipe import Normal
        assert jnp.issubdtype(Normal(loc=0.0, scale=1.0, name="x").dtype, jnp.floating)


# ---------------------------------------------------------------------------
# FlattenedView on EmpiricalDistribution
# ---------------------------------------------------------------------------

class TestFlattenedViewEmpirical:
    def test_empirical_flatten_roundtrip(self, key):
        samples = jax.random.normal(key, shape=(100, 5))
        emp = RecordEmpiricalDistribution(samples, name="x")

        flat_dist = emp.as_flat_distribution()
        assert flat_dist.event_shape == (5,)

        flat_sample = sample(flat_dist, key=key)
        assert flat_sample.shape == (5,)
