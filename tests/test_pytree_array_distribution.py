"""Tests for PyTreeArrayDistribution, FlattenedView, and ArrayDistribution pytree interface (Phase 2)."""

import jax
import jax.numpy as jnp
import math
import numpy as np
import pytest

from probpipe import (
    Distribution,
    PyTreeArrayDistribution,
    ArrayDistribution,
    ArrayEmpiricalDistribution,
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
    return Normal(loc=0.0, scale=1.0)


@pytest.fixture
def vector_mvn():
    return MultivariateNormal(
        loc=jnp.zeros(3),
        cov=jnp.eye(3),
    )


@pytest.fixture
def matrix_mvn():
    """MVN with event_shape (4,) to test flatten/unflatten with non-trivial shapes."""
    return MultivariateNormal(
        loc=jnp.zeros(4),
        cov=jnp.eye(4),
    )


# ---------------------------------------------------------------------------
# Hierarchy checks
# ---------------------------------------------------------------------------

class TestHierarchy:
    def test_arraydist_is_pytreearraydist(self, scalar_normal):
        assert isinstance(scalar_normal, PyTreeArrayDistribution)

    def test_arraydist_is_distribution(self, scalar_normal):
        assert isinstance(scalar_normal, Distribution)

    def test_pytreearraydist_is_distribution(self, scalar_normal):
        """ArrayDistribution inherits PyTreeArrayDistribution which inherits Distribution."""
        assert isinstance(scalar_normal, Distribution)
        assert isinstance(scalar_normal, PyTreeArrayDistribution)
        assert isinstance(scalar_normal, ArrayDistribution)


# ---------------------------------------------------------------------------
# Distribution[T] base class methods
# ---------------------------------------------------------------------------

class TestDistributionBase:
    """Tests for methods defined on Distribution[T] itself."""

    def test_log_prob_raises_by_default(self):
        """Distribution[T] without SupportsLogProb raises TypeError."""
        class StubDist(Distribution):
            def _sample_one(self, key):
                return jnp.array(0.0)
        d = StubDist()
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

    def test_repr_without_name(self):
        """Distribution.__repr__ works without a name."""
        n = Normal(loc=0.0, scale=1.0)
        r = repr(n)
        assert "Normal" in r

    def test_from_distribution_on_base_class(self, scalar_normal):
        """from_distribution is accessible on Distribution[T] base."""
        # Normal inherits from_distribution from Distribution[T]
        result = from_distribution(scalar_normal, Normal, num_samples=100)
        assert isinstance(result, Normal)


# ---------------------------------------------------------------------------
# ArrayDistribution trivial pytree interface
# ---------------------------------------------------------------------------

class TestArrayDistributionPyTreeInterface:
    def test_treedef_scalar(self, scalar_normal):
        td = scalar_normal.treedef
        assert td == jax.tree.structure(None)

    def test_treedef_vector(self, vector_mvn):
        td = vector_mvn.treedef
        assert td == jax.tree.structure(None)

    def test_event_shapes_equals_event_shape(self, scalar_normal):
        assert scalar_normal.event_shapes == scalar_normal.event_shape

    def test_event_shapes_vector(self, vector_mvn):
        assert vector_mvn.event_shapes == (3,)
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
# flatten_value / unflatten_value on ArrayDistribution
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
        samples = sample(vector_mvn, key=key, sample_shape=(5,))
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
        assert isinstance(flat_dist, ArrayDistribution)

    def test_event_shape(self, vector_mvn):
        flat_dist = vector_mvn.as_flat_distribution()
        assert flat_dist.event_shape == (3,)

    def test_batch_shape(self, vector_mvn):
        flat_dist = vector_mvn.as_flat_distribution()
        assert flat_dist.batch_shape == vector_mvn.batch_shape

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
    def test_supports_delegates_to_support(self, scalar_normal):
        """For ArrayDistribution, supports should equal support."""
        from probpipe import real
        assert scalar_normal.supports == scalar_normal.support
        assert scalar_normal.supports == real


# ---------------------------------------------------------------------------
# FlattenedView on EmpiricalDistribution
# ---------------------------------------------------------------------------

class TestFlattenedViewEmpirical:
    def test_empirical_flatten_roundtrip(self, key):
        samples = jax.random.normal(key, shape=(100, 5))
        emp = ArrayEmpiricalDistribution(samples)

        flat_dist = emp.as_flat_distribution()
        assert flat_dist.event_shape == (5,)

        flat_sample = sample(flat_dist, key=key)
        assert flat_sample.shape == (5,)
