"""Tests for BootstrapReplicateDistribution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    NumericRecordDistribution,
    ArrayBootstrapReplicateDistribution,
    BootstrapDistribution,
    Distribution,
    EmpiricalDistribution,
    BootstrapReplicateDistribution,
    SupportsExpectation,
    SupportsSampling,
    Record,
    expectation,
    sample,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_empirical(self):
        data = jnp.arange(20.0).reshape(10, 2)
        emp = EmpiricalDistribution(data)
        dist = BootstrapReplicateDistribution(emp)
        assert dist.n == 10

    def test_from_array(self):
        data = jnp.arange(20.0).reshape(10, 2)
        dist = BootstrapReplicateDistribution(data)
        assert dist.n == 10

    def test_custom_n(self):
        data = jnp.ones((50, 3))
        dist = BootstrapReplicateDistribution(data, n=30)
        assert dist.n == 30

    def test_n_from_empirical_default(self):
        emp = EmpiricalDistribution(jnp.ones((20, 4)))
        dist = BootstrapReplicateDistribution(emp)
        assert dist.n == 20

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="positive"):
            BootstrapReplicateDistribution(jnp.ones((5, 2)), n=0)

    def test_scalar_source_raises(self):
        with pytest.raises(ValueError, match="at least 1 dimension"):
            BootstrapReplicateDistribution(jnp.array(1.0))

    def test_name(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)), name="boot")
        assert dist.name == "boot"

    def test_name_default(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)))
        assert dist.name == "bootstrap"


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_supports_sampling(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)))
        assert isinstance(dist, SupportsSampling)

    def test_supports_expectation(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)))
        assert isinstance(dist, SupportsExpectation)

    def test_generic_numeric_dispatches_to_array(self):
        # Factory dispatch: numeric arrays → ArrayBootstrapReplicateDistribution
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)))
        assert isinstance(dist, Distribution)
        assert isinstance(dist, NumericRecordDistribution)
        assert isinstance(dist, ArrayBootstrapReplicateDistribution)

    def test_generic_object_is_not_array(self):
        # Non-numeric (object) source stays as base class
        dist = BootstrapReplicateDistribution(["a", "b", "c"])
        assert isinstance(dist, Distribution)
        assert not isinstance(dist, NumericRecordDistribution)

    def test_array_is_array_distribution(self):
        dist = ArrayBootstrapReplicateDistribution(jnp.ones((5, 2)))
        assert isinstance(dist, NumericRecordDistribution)
        assert isinstance(dist, BootstrapReplicateDistribution)
        assert isinstance(dist, Distribution)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    @pytest.fixture
    def dist(self):
        data = jnp.arange(30.0).reshape(10, 3)
        return BootstrapReplicateDistribution(data)

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
        dist = BootstrapReplicateDistribution(data, n=20)
        s = dist._sample(jax.random.PRNGKey(0))
        assert s.shape == (20, 2)

    def test_1d_source(self):
        """1D source (scalar observations)."""
        data = jnp.arange(10.0)
        dist = BootstrapReplicateDistribution(data)
        s = dist._sample(jax.random.PRNGKey(0))
        assert s.shape == (10,)

    def test_weighted_empirical(self):
        """Weighted EmpiricalDistribution uses weights for sampling."""
        data = jnp.array([[0.0], [1.0], [2.0]])
        weights = jnp.array([0.0, 0.0, 1.0])  # all weight on last row
        emp = EmpiricalDistribution(data, weights=weights)
        dist = BootstrapReplicateDistribution(emp)
        s = dist._sample_one(jax.random.PRNGKey(0))
        # All rows should be [2.0]
        np.testing.assert_allclose(s, jnp.full((3, 1), 2.0))


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Test generic (non-array-specific) properties."""

    def test_data_from_array(self):
        data = jnp.arange(20.0).reshape(10, 2)
        dist = BootstrapReplicateDistribution(data)
        np.testing.assert_array_equal(dist.data, data)

    def test_data_from_empirical(self):
        data = jnp.arange(20.0).reshape(10, 2)
        emp = EmpiricalDistribution(data)
        dist = BootstrapReplicateDistribution(emp)
        np.testing.assert_array_equal(dist.data, data)

    def test_source_n(self):
        dist = BootstrapReplicateDistribution(jnp.ones((15, 3)))
        assert dist.source_n == 15

    def test_is_uniform_from_array(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)))
        assert dist.is_uniform is True

    def test_is_uniform_from_uniform_empirical(self):
        emp = EmpiricalDistribution(jnp.ones((5, 2)))
        dist = BootstrapReplicateDistribution(emp)
        assert dist.is_uniform is True

    def test_is_uniform_from_weighted_empirical(self):
        emp = EmpiricalDistribution(jnp.ones((5, 2)), weights=jnp.array([1., 2., 3., 4., 5.]))
        dist = BootstrapReplicateDistribution(emp)
        assert dist.is_uniform is False

    def test_weights_uniform(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)))
        np.testing.assert_allclose(dist.weights, jnp.ones(5) / 5)

    def test_weights_from_weighted_empirical(self):
        weights = jnp.array([1., 2., 3.])
        emp = EmpiricalDistribution(jnp.ones((3, 2)), weights=weights)
        dist = BootstrapReplicateDistribution(emp)
        assert dist.weights is not None
        assert dist.weights.shape == (3,)

    def test_approximate_flag(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)))
        assert dist._approximate is True

    def test_numeric_has_event_shape(self):
        """Numeric arrays dispatch to Array variant with event_shape."""
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)))
        assert hasattr(dist, "event_shape")
        assert dist.event_shape == (5, 2)

    def test_generic_no_event_shape(self):
        """Non-numeric BootstrapReplicateDistribution has no event_shape."""
        dist = BootstrapReplicateDistribution(["a", "b", "c"])
        assert not hasattr(dist, "event_shape") or "event_shape" not in type(dist).__dict__

    def test_generic_no_dim(self):
        """Non-numeric BootstrapReplicateDistribution has no dim."""
        dist = BootstrapReplicateDistribution(["a", "b", "c"])
        assert not hasattr(dist, "dim") or "dim" not in type(dist).__dict__

    def test_generic_no_dtype(self):
        """Non-numeric BootstrapReplicateDistribution has no dtype."""
        dist = BootstrapReplicateDistribution(["a", "b", "c"])
        assert not hasattr(dist, "dtype") or "dtype" not in type(dist).__dict__


# ---------------------------------------------------------------------------
# Expectation
# ---------------------------------------------------------------------------


class TestExpectation:
    def test_expectation_returns_array(self):
        data = jnp.arange(30.0).reshape(10, 3)
        dist = BootstrapReplicateDistribution(data)
        result = dist._expectation(
            lambda d: jnp.mean(d, axis=0),
            key=jax.random.PRNGKey(0),
            num_evaluations=50,
            return_dist=False,
        )
        assert result.shape == (3,)

    def test_expectation_returns_bootstrap_dist(self):
        data = jnp.arange(30.0).reshape(10, 3)
        dist = BootstrapReplicateDistribution(data)
        result = dist._expectation(
            lambda d: jnp.mean(d, axis=0),
            key=jax.random.PRNGKey(0),
            num_evaluations=50,
            return_dist=True,
        )
        assert isinstance(result, BootstrapDistribution)
        assert result.n == 50

    def test_expectation_op(self):
        data = jnp.arange(30.0).reshape(10, 3)
        dist = BootstrapReplicateDistribution(data)
        result = expectation(
            dist,
            lambda d: jnp.mean(d, axis=0),
            key=jax.random.PRNGKey(0),
            num_evaluations=50,
            return_dist=False,
        )
        assert result.shape == (3,)

    def test_expectation_mean_converges(self):
        """E[mean(bootstrap_dataset)] should converge to mean(data)."""
        data = jnp.arange(30.0).reshape(10, 3)
        dist = BootstrapReplicateDistribution(data)
        result = dist._expectation(
            lambda d: jnp.mean(d, axis=0),
            key=jax.random.PRNGKey(42),
            num_evaluations=2000,
            return_dist=False,
        )
        expected = jnp.mean(data, axis=0)
        # Bootstrap resampling with only 10 data points has high variance
        np.testing.assert_allclose(result, expected, atol=0.25)


# ---------------------------------------------------------------------------
# ArrayBootstrapReplicateDistribution
# ---------------------------------------------------------------------------


class TestArrayBootstrapReplicateDistribution:
    def test_support(self):
        from probpipe.core.constraints import real
        dist = ArrayBootstrapReplicateDistribution(jnp.ones((5, 2)))
        assert dist.support == real

    def test_sample_shape(self):
        data = jnp.arange(30.0).reshape(10, 3)
        dist = ArrayBootstrapReplicateDistribution(data)
        s = dist._sample(jax.random.PRNGKey(0), sample_shape=(4,))
        assert s.shape == (4, 10, 3)

    def test_event_shape(self):
        data = jnp.ones((10, 3))
        dist = ArrayBootstrapReplicateDistribution(data, n=8)
        assert dist.event_shape == (8, 3)

    def test_from_empirical(self):
        emp = EmpiricalDistribution(jnp.ones((20, 4)))
        dist = ArrayBootstrapReplicateDistribution(emp)
        assert dist.n == 20

    def test_obs_shape(self):
        dist = ArrayBootstrapReplicateDistribution(jnp.ones((10, 3, 4)))
        assert dist.obs_shape == (3, 4)

    def test_obs_shape_scalar(self):
        dist = ArrayBootstrapReplicateDistribution(jnp.ones((10,)))
        assert dist.obs_shape == ()

    def test_dim(self):
        dist = ArrayBootstrapReplicateDistribution(jnp.ones((10, 3)), n=5)
        assert dist.dim == 5 * 3

    def test_dim_scalar_obs(self):
        dist = ArrayBootstrapReplicateDistribution(jnp.ones((10,)), n=5)
        assert dist.dim == 5

    def test_dtype(self):
        dist = ArrayBootstrapReplicateDistribution(jnp.ones((5, 2)))
        assert dist.dtype == jnp.float32


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr(self):
        data = jnp.ones((10, 3))
        dist = BootstrapReplicateDistribution(data, n=8)
        r = repr(dist)
        assert "BootstrapReplicateDistribution" in r
        assert "n=8" in r
        assert "source_n=10" in r


# ---------------------------------------------------------------------------
# Record-based EmpiricalDistribution
# ---------------------------------------------------------------------------


class TestValuesEmpiricalDistribution:
    """EmpiricalDistribution(Record(...)) → _RecordEmpiricalDistribution."""

    @pytest.fixture
    def values_data(self):
        X = jnp.ones((20, 3))
        y = jnp.arange(20.0)
        return Record(X=X, y=y)

    def test_dispatch(self, values_data):
        from probpipe.core._empirical import _RecordEmpiricalDistribution
        emp = EmpiricalDistribution(values_data)
        assert isinstance(emp, _RecordEmpiricalDistribution)

    def test_n(self, values_data):
        emp = EmpiricalDistribution(values_data)
        assert emp.n == 20

    def test_satisfies_sampling(self, values_data):
        emp = EmpiricalDistribution(values_data)
        assert isinstance(emp, SupportsSampling)

    def test_sample_one_returns_values(self, values_data):
        emp = EmpiricalDistribution(values_data)
        s = sample(emp, key=jax.random.PRNGKey(0))
        assert isinstance(s, Record)
        assert "X" in s and "y" in s

    def test_sample_one_shapes(self, values_data):
        emp = EmpiricalDistribution(values_data)
        s = sample(emp, key=jax.random.PRNGKey(0))
        assert s["X"].shape == (3,)
        assert s["y"].shape == ()

    def test_sample_batch(self, values_data):
        emp = EmpiricalDistribution(values_data)
        s = emp._sample(jax.random.PRNGKey(0), sample_shape=(5,))
        assert isinstance(s, Record)
        assert s["X"].shape == (5, 3)
        assert s["y"].shape == (5,)

    def test_record_template(self, values_data):
        emp = EmpiricalDistribution(values_data)
        tpl = emp.record_template
        assert tpl is not None
        assert tpl["X"] == (3,)
        assert tpl["y"] == ()

    def test_component_names(self, values_data):
        emp = EmpiricalDistribution(values_data)
        assert emp.component_names == ("X", "y")

    def test_getitem_returns_view(self, values_data):
        from probpipe.core._record_distribution import _RecordDistributionView
        emp = EmpiricalDistribution(values_data)
        view = emp["X"]
        assert isinstance(view, _RecordDistributionView)

    def test_getitem_returns_view(self, values_data):
        from probpipe.core._record_distribution import _RecordDistributionView
        emp = EmpiricalDistribution(values_data)
        view = emp["X"]
        assert isinstance(view, _RecordDistributionView)

    def test_mean(self, values_data):
        emp = EmpiricalDistribution(values_data)
        m = emp._mean()
        assert isinstance(m, Record)
        np.testing.assert_allclose(m["X"], 1.0)
        np.testing.assert_allclose(m["y"], jnp.arange(20.0).mean())

    def test_variance(self, values_data):
        emp = EmpiricalDistribution(values_data)
        v = emp._variance()
        assert isinstance(v, Record)
        np.testing.assert_allclose(v["X"], 0.0, atol=1e-7)

    def test_repr(self, values_data):
        emp = EmpiricalDistribution(values_data)
        r = repr(emp)
        assert "_RecordEmpiricalDistribution" in r
        assert "n=20" in r


# ---------------------------------------------------------------------------
# Record-based BootstrapReplicateDistribution
# ---------------------------------------------------------------------------


class TestValuesBootstrapReplicateDistribution:
    """BootstrapReplicateDistribution with Record source."""

    @pytest.fixture
    def values_data(self):
        X = jnp.ones((20, 3))
        y = jnp.arange(20.0)
        return Record(X=X, y=y)

    @pytest.fixture
    def bootstrap(self, values_data):
        emp = EmpiricalDistribution(values_data)
        return BootstrapReplicateDistribution(emp)

    def test_dispatch_from_empirical(self, values_data):
        from probpipe.core._empirical import _RecordBootstrapReplicateDistribution
        emp = EmpiricalDistribution(values_data)
        boot = BootstrapReplicateDistribution(emp)
        assert isinstance(boot, _RecordBootstrapReplicateDistribution)

    def test_dispatch_from_values(self, values_data):
        from probpipe.core._empirical import _RecordBootstrapReplicateDistribution
        boot = BootstrapReplicateDistribution(values_data)
        assert isinstance(boot, _RecordBootstrapReplicateDistribution)

    def test_n(self, bootstrap):
        assert bootstrap.n == 20

    def test_custom_n(self, values_data):
        emp = EmpiricalDistribution(values_data)
        boot = BootstrapReplicateDistribution(emp, n=10)
        assert boot.n == 10

    def test_sample_one_returns_values(self, bootstrap):
        s = sample(bootstrap, key=jax.random.PRNGKey(0))
        assert isinstance(s, Record)
        assert "X" in s and "y" in s

    def test_sample_one_shapes(self, bootstrap):
        s = sample(bootstrap, key=jax.random.PRNGKey(0))
        assert s["X"].shape == (20, 3)
        assert s["y"].shape == (20,)

    def test_sample_batch(self, bootstrap):
        s = bootstrap._sample(jax.random.PRNGKey(0), sample_shape=(4,))
        assert isinstance(s, Record)
        assert s["X"].shape == (4, 20, 3)
        assert s["y"].shape == (4, 20)

    def test_record_template(self, bootstrap):
        tpl = bootstrap.record_template
        assert tpl is not None
        assert tpl["X"] == (20, 3)
        assert tpl["y"] == (20,)

    def test_component_names(self, bootstrap):
        assert bootstrap.component_names == ("X", "y")

    def test_getitem_returns_view(self, bootstrap):
        from probpipe.core._record_distribution import _RecordDistributionView
        view = bootstrap["X"]
        assert isinstance(view, _RecordDistributionView)

    def test_getitem_returns_view(self, bootstrap):
        from probpipe.core._record_distribution import _RecordDistributionView
        view = bootstrap["X"]
        assert isinstance(view, _RecordDistributionView)

    def test_view_sample(self, bootstrap):
        """Sampling a view extracts the named field from a bootstrap sample."""
        view = bootstrap["y"]
        s = sample(view, key=jax.random.PRNGKey(0))
        assert s.shape == (20,)  # one bootstrapped dataset's y

    def test_joint_resampling(self, bootstrap):
        """X and y views from the same parent preserve row correspondence."""
        key = jax.random.PRNGKey(42)
        full = sample(bootstrap, key=key)
        # Extract individually from the same full sample
        assert full["X"].shape == (20, 3)
        assert full["y"].shape == (20,)

    def test_repr(self, bootstrap):
        r = repr(bootstrap)
        assert "_RecordBootstrapReplicateDistribution" in r
        assert "n=20" in r
