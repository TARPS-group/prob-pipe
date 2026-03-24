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
    FlattenedView,
    Normal,
    MultivariateNormal,
    EmpiricalDistribution,
)


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

    def test_flat_dim_scalar(self, scalar_normal):
        assert scalar_normal.flat_dim == 1

    def test_flat_dim_vector(self, vector_mvn):
        assert vector_mvn.flat_dim == 3

    def test_flat_dim_4d(self, matrix_mvn):
        assert matrix_mvn.flat_dim == 4


# ---------------------------------------------------------------------------
# flatten_value / unflatten_value on ArrayDistribution
# ---------------------------------------------------------------------------

class TestArrayDistFlattenUnflatten:
    def test_flatten_vector_sample(self, vector_mvn, key):
        sample = vector_mvn.sample(key)
        flat = vector_mvn.flatten_value(sample)
        assert flat.shape == (3,)
        np.testing.assert_allclose(flat, sample, atol=1e-6)

    def test_unflatten_vector_sample(self, vector_mvn, key):
        sample = vector_mvn.sample(key)
        flat = vector_mvn.flatten_value(sample)
        restored = vector_mvn.unflatten_value(flat)
        np.testing.assert_allclose(restored, sample, atol=1e-6)

    def test_flatten_unflatten_roundtrip_batched(self, vector_mvn, key):
        samples = vector_mvn.sample(key, sample_shape=(5,))
        flat = vector_mvn.flatten_value(samples)
        assert flat.shape == (5, 3)
        restored = vector_mvn.unflatten_value(flat)
        np.testing.assert_allclose(restored, samples, atol=1e-6)

    def test_flatten_unflatten_4d(self, matrix_mvn, key):
        sample = matrix_mvn.sample(key)
        flat = matrix_mvn.flatten_value(sample)
        assert flat.shape == (4,)
        restored = matrix_mvn.unflatten_value(flat)
        np.testing.assert_allclose(restored, sample, atol=1e-6)


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
        sample = flat_dist.sample(key)
        assert sample.shape == (3,)

    def test_sample_batched(self, vector_mvn, key):
        flat_dist = vector_mvn.as_flat_distribution()
        samples = flat_dist.sample(key, sample_shape=(10,))
        assert samples.shape == (10, 3)

    def test_log_prob_matches(self, vector_mvn, key):
        flat_dist = vector_mvn.as_flat_distribution()
        sample = vector_mvn.sample(key)
        flat_sample = vector_mvn.flatten_value(sample)

        lp_original = vector_mvn.log_prob(sample)
        lp_flat = flat_dist.log_prob(flat_sample)
        np.testing.assert_allclose(lp_flat, lp_original, atol=1e-5)

    def test_base_distribution(self, vector_mvn):
        flat_dist = vector_mvn.as_flat_distribution()
        assert flat_dist.base_distribution is vector_mvn

    def test_unflatten_sample(self, vector_mvn, key):
        flat_dist = vector_mvn.as_flat_distribution()
        flat_sample = flat_dist.sample(key)
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
        sample = matrix_mvn.sample(key)
        flat_sample = matrix_mvn.flatten_value(sample)

        lp_original = matrix_mvn.log_prob(sample)
        lp_flat = flat_dist.log_prob(flat_sample)
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
        emp = EmpiricalDistribution(samples)

        flat_dist = emp.as_flat_distribution()
        assert flat_dist.event_shape == (5,)

        flat_sample = flat_dist.sample(key)
        assert flat_sample.shape == (5,)
