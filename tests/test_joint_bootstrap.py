"""Tests for JointBootstrapDistribution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    JointBootstrapDistribution,
    SupportsSampling,
    sample,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_empirical(self):
        data = jnp.arange(20.0).reshape(10, 2)
        emp = EmpiricalDistribution(data)
        dist = JointBootstrapDistribution(emp)
        assert dist.n == 10

    def test_from_array(self):
        data = jnp.arange(20.0).reshape(10, 2)
        dist = JointBootstrapDistribution(data)
        assert dist.n == 10

    def test_custom_n(self):
        data = jnp.ones((50, 3))
        dist = JointBootstrapDistribution(data, n=30)
        assert dist.n == 30

    def test_n_from_empirical_default(self):
        emp = EmpiricalDistribution(jnp.ones((20, 4)))
        dist = JointBootstrapDistribution(emp)
        assert dist.n == 20

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="positive"):
            JointBootstrapDistribution(jnp.ones((5, 2)), n=0)

    def test_scalar_source_raises(self):
        with pytest.raises(ValueError, match="at least 1 dimension"):
            JointBootstrapDistribution(jnp.array(1.0))

    def test_name(self):
        dist = JointBootstrapDistribution(jnp.ones((5, 2)), name="boot")
        assert dist.name == "boot"

    def test_name_default_none(self):
        dist = JointBootstrapDistribution(jnp.ones((5, 2)))
        assert dist.name is None


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_supports_sampling(self):
        dist = JointBootstrapDistribution(jnp.ones((5, 2)))
        assert isinstance(dist, SupportsSampling)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    @pytest.fixture
    def dist(self):
        data = jnp.arange(30.0).reshape(10, 3)
        return JointBootstrapDistribution(data)

    def test_sample_one_shape(self, dist):
        key = jax.random.PRNGKey(0)
        s = dist._sample_one(key)
        assert s.shape == (10, 3)

    def test_sample_no_shape(self, dist):
        key = jax.random.PRNGKey(1)
        s = dist._sample(key)
        assert s.shape == (10, 3)

    def test_sample_with_shape(self, dist):
        key = jax.random.PRNGKey(2)
        s = dist._sample(key, sample_shape=(5,))
        assert s.shape == (5, 10, 3)

    def test_sample_2d_shape(self, dist):
        key = jax.random.PRNGKey(3)
        s = dist._sample(key, sample_shape=(2, 3))
        assert s.shape == (2, 3, 10, 3)

    def test_sample_op(self, dist):
        s = sample(dist, key=jax.random.PRNGKey(4))
        assert s.shape == (10, 3)

    def test_samples_are_rows_of_data(self, dist):
        """Each row of a bootstrap sample should be a row from the original data."""
        key = jax.random.PRNGKey(5)
        s = dist._sample_one(key)
        data = jnp.arange(30.0).reshape(10, 3)
        # Each row in s should appear in data
        for i in range(s.shape[0]):
            matches = jnp.any(jnp.all(data == s[i], axis=1))
            assert matches

    def test_custom_n_changes_shape(self):
        data = jnp.ones((50, 2))
        dist = JointBootstrapDistribution(data, n=20)
        s = dist._sample(jax.random.PRNGKey(0))
        assert s.shape == (20, 2)

    def test_1d_source(self):
        """1D source (scalar observations)."""
        data = jnp.arange(10.0)
        dist = JointBootstrapDistribution(data)
        s = dist._sample(jax.random.PRNGKey(0))
        assert s.shape == (10,)

    def test_weighted_empirical(self):
        """Weighted EmpiricalDistribution uses weights for sampling."""
        data = jnp.array([[0.0], [1.0], [2.0]])
        weights = jnp.array([0.0, 0.0, 1.0])  # all weight on last row
        emp = EmpiricalDistribution(data, weights=weights)
        dist = JointBootstrapDistribution(emp)
        s = dist._sample_one(jax.random.PRNGKey(0))
        # All rows should be [2.0]
        np.testing.assert_allclose(s, jnp.full((3, 1), 2.0))


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr(self):
        data = jnp.ones((10, 3))
        dist = JointBootstrapDistribution(data, n=8)
        r = repr(dist)
        assert "JointBootstrapDistribution" in r
        assert "n=8" in r
        assert "source_n=10" in r
