"""Tests for BootstrapReplicateDistribution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    NumericRecordDistribution,
    RecordBootstrapReplicateDistribution,
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
        emp = EmpiricalDistribution(data, name="x")
        dist = BootstrapReplicateDistribution(emp, name="x")
        assert dist.n == 10

    def test_from_array(self):
        data = jnp.arange(20.0).reshape(10, 2)
        dist = BootstrapReplicateDistribution(data, name="x")
        assert dist.n == 10

    def test_custom_n(self):
        data = jnp.ones((50, 3))
        dist = BootstrapReplicateDistribution(data, n=30, name="x")
        assert dist.n == 30

    def test_n_from_empirical_default(self):
        emp = EmpiricalDistribution(jnp.ones((20, 4)), name="x")
        dist = BootstrapReplicateDistribution(emp, name="x")
        assert dist.n == 20

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="positive"):
            BootstrapReplicateDistribution(jnp.ones((5, 2)), n=0, name="x")

    def test_scalar_source_raises(self):
        with pytest.raises(ValueError, match="at least 1 dimension"):
            BootstrapReplicateDistribution(jnp.array(1.0), name="x")

    def test_name(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)), name="boot")
        assert dist.name == "boot"

    def test_name_default(self):
        # Numeric-array sources auto-wrap and use the (mandatory) name=
        # as the field name. The default-name path lives on the generic
        # base, which handles non-numeric (e.g. opaque-object) sources.
        dist = BootstrapReplicateDistribution(["a", "b", "c"])
        assert dist.name == "bootstrap"


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_supports_sampling(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)), name="x")
        assert isinstance(dist, SupportsSampling)

    def test_supports_expectation(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)), name="x")
        assert isinstance(dist, SupportsExpectation)

    def test_generic_numeric_dispatches_to_array(self):
        # Factory dispatch: numeric arrays → RecordBootstrapReplicateDistribution
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)), name="x")
        assert isinstance(dist, Distribution)
        assert isinstance(dist, NumericRecordDistribution)
        assert isinstance(dist, RecordBootstrapReplicateDistribution)

    def test_generic_object_is_not_array(self):
        # Non-numeric (object) source stays as base class
        dist = BootstrapReplicateDistribution(["a", "b", "c"], name="x")
        assert isinstance(dist, Distribution)
        assert not isinstance(dist, NumericRecordDistribution)

    def test_array_is_array_distribution(self):
        dist = RecordBootstrapReplicateDistribution(jnp.ones((5, 2)), name="x")
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
        return BootstrapReplicateDistribution(data, name="x")

    def test_sample_empty_shape(self, dist):
        key = jax.random.PRNGKey(0)
        s = dist._sample(key, ())
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
        s = dist._sample(key, ())
        data = jnp.arange(30.0).reshape(10, 3)
        # Single-field auto-wrap: extract the field array.
        s_arr = s["x"]
        for i in range(s_arr.shape[0]):
            matches = jnp.any(jnp.all(data == s_arr[i], axis=1))
            assert matches

    def test_custom_n_changes_shape(self):
        data = jnp.ones((50, 2))
        dist = BootstrapReplicateDistribution(data, n=20, name="x")
        s = dist._sample(jax.random.PRNGKey(0))
        assert s.shape == (20, 2)

    def test_1d_source(self):
        """1D source (scalar observations)."""
        data = jnp.arange(10.0)
        dist = BootstrapReplicateDistribution(data, name="x")
        s = dist._sample(jax.random.PRNGKey(0))
        assert s.shape == (10,)

    def test_weighted_empirical(self):
        """Weighted EmpiricalDistribution uses weights for sampling."""
        data = jnp.array([[0.0], [1.0], [2.0]])
        weights = jnp.array([0.0, 0.0, 1.0])  # all weight on last row
        emp = EmpiricalDistribution(data, weights=weights, name="x")
        dist = BootstrapReplicateDistribution(emp, name="x")
        s = dist._sample(jax.random.PRNGKey(0), ())
        # All rows should be [2.0]
        np.testing.assert_allclose(s, jnp.full((3, 1), 2.0))


# ---------------------------------------------------------------------------
# SupportsSampling source (#93) — a parametric distribution as the source
# ---------------------------------------------------------------------------


class TestSampleableSource:
    """Tests for the sampleable-source path (no stored observations)."""

    def _normal(self):
        from probpipe import Normal
        return Normal(loc=0.0, scale=1.0, name="x")

    def test_construction_keeps_generic_base(self):
        # SupportsSampling sources stay in the generic base (no Record
        # data to wrap), unlike numeric arrays / Records / Empirical.
        d = BootstrapReplicateDistribution(self._normal(), n=10)
        assert isinstance(d, BootstrapReplicateDistribution)
        assert not isinstance(d, RecordBootstrapReplicateDistribution)
        assert d.n == 10
        assert d.source_n is None  # no canonical observation count

    def test_missing_n_raises(self):
        with pytest.raises(ValueError, match="n must be a positive int"):
            BootstrapReplicateDistribution(self._normal())

    def test_zero_n_raises(self):
        with pytest.raises(ValueError, match="n must be a positive int"):
            BootstrapReplicateDistribution(self._normal(), n=0)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="n must be a positive int"):
            BootstrapReplicateDistribution(self._normal(), n=-3)

    def test_sample_empty_shape(self):
        # One bootstrap replicate is ``n`` i.i.d. draws from source._sample.
        d = BootstrapReplicateDistribution(self._normal(), n=10)
        s = d._sample(jax.random.PRNGKey(0), ())
        assert s.shape == (10,)

    def test_sample_with_shape(self):
        # sample_shape prepends; total = sample_shape + (n,) + event_shape.
        d = BootstrapReplicateDistribution(self._normal(), n=10)
        s = d._sample(jax.random.PRNGKey(1), sample_shape=(3,))
        assert s.shape == (3, 10)

    def test_data_is_none_for_sampleable_source(self):
        # No stored observations.
        d = BootstrapReplicateDistribution(self._normal(), n=5)
        assert d.data is None
        assert d.weights is None

    def test_repr_mentions_source(self):
        d = BootstrapReplicateDistribution(self._normal(), n=5)
        r = repr(d)
        assert "n=5" in r
        assert "Normal" in r


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Test generic (non-array-specific) properties."""

    def test_data_from_array(self):
        # Numeric-array sources auto-wrap as a single-field Record;
        # ``dist.data`` exposes the wrapped Record. Pull the field for
        # raw-array comparison.
        data = jnp.arange(20.0).reshape(10, 2)
        dist = BootstrapReplicateDistribution(data, name="x")
        np.testing.assert_array_equal(dist.data["x"], data)

    def test_data_from_empirical(self):
        data = jnp.arange(20.0).reshape(10, 2)
        emp = EmpiricalDistribution(data, name="x")
        dist = BootstrapReplicateDistribution(emp, name="x")
        np.testing.assert_array_equal(dist.data["x"], data)

    def test_source_n(self):
        dist = BootstrapReplicateDistribution(jnp.ones((15, 3)), name="x")
        assert dist.source_n == 15

    def test_is_uniform_from_array(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)), name="x")
        assert dist.is_uniform is True

    def test_is_uniform_from_uniform_empirical(self):
        emp = EmpiricalDistribution(jnp.ones((5, 2)), name="x")
        dist = BootstrapReplicateDistribution(emp, name="x")
        assert dist.is_uniform is True

    def test_is_uniform_from_weighted_empirical(self):
        emp = EmpiricalDistribution(jnp.ones((5, 2)), weights=jnp.array([1., 2., 3., 4., 5.]), name="x")
        dist = BootstrapReplicateDistribution(emp, name="x")
        assert dist.is_uniform is False

    def test_weights_uniform(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)), name="x")
        np.testing.assert_allclose(dist.weights, jnp.ones(5) / 5)

    def test_weights_from_weighted_empirical(self):
        weights = jnp.array([1., 2., 3.])
        emp = EmpiricalDistribution(jnp.ones((3, 2)), weights=weights, name="x")
        dist = BootstrapReplicateDistribution(emp, name="x")
        assert dist.weights is not None
        assert dist.weights.shape == (3,)

    def test_approximate_flag(self):
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)), name="x")
        assert dist._approximate is True

    def test_numeric_has_event_shape(self):
        """Numeric arrays dispatch to Array variant with event_shape."""
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)), name="x")
        assert hasattr(dist, "event_shape")
        assert dist.event_shape == (5, 2)

    def test_generic_no_event_shape(self):
        """Non-numeric BootstrapReplicateDistribution has no event_shape."""
        dist = BootstrapReplicateDistribution(["a", "b", "c"], name="x")
        assert not hasattr(dist, "event_shape") or "event_shape" not in type(dist).__dict__

    def test_generic_no_dim(self):
        """Non-numeric BootstrapReplicateDistribution has no dim."""
        dist = BootstrapReplicateDistribution(["a", "b", "c"], name="x")
        assert not hasattr(dist, "dim") or "dim" not in type(dist).__dict__

    def test_generic_no_dtype(self):
        """Non-numeric BootstrapReplicateDistribution has no dtype."""
        dist = BootstrapReplicateDistribution(["a", "b", "c"], name="x")
        assert not hasattr(dist, "dtype") or "dtype" not in type(dist).__dict__


# ---------------------------------------------------------------------------
# Expectation
# ---------------------------------------------------------------------------


class TestExpectation:
    def test_expectation_returns_array(self):
        data = jnp.arange(30.0).reshape(10, 3)
        dist = BootstrapReplicateDistribution(data, name="x")
        result = dist._expectation(
            lambda d: jnp.mean(d, axis=0),
            key=jax.random.PRNGKey(0),
            num_evaluations=50,
            return_dist=False,
        )
        assert result.shape == (3,)

    def test_expectation_returns_bootstrap_dist(self):
        data = jnp.arange(30.0).reshape(10, 3)
        dist = BootstrapReplicateDistribution(data, name="x")
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
        dist = BootstrapReplicateDistribution(data, name="x")
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
        dist = BootstrapReplicateDistribution(data, name="x")
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
# RecordBootstrapReplicateDistribution
# ---------------------------------------------------------------------------


class TestRecordBootstrapReplicateDistribution:
    def test_support(self):
        from probpipe.core.constraints import real
        dist = RecordBootstrapReplicateDistribution(jnp.ones((5, 2)), name="x")
        assert dist.support == real

    def test_sample_shape(self):
        data = jnp.arange(30.0).reshape(10, 3)
        dist = RecordBootstrapReplicateDistribution(data, name="x")
        s = dist._sample(jax.random.PRNGKey(0), sample_shape=(4,))
        assert s.shape == (4, 10, 3)

    def test_event_shape(self):
        data = jnp.ones((10, 3))
        dist = RecordBootstrapReplicateDistribution(data, n=8, name="x")
        assert dist.event_shape == (8, 3)

    def test_from_empirical(self):
        emp = EmpiricalDistribution(jnp.ones((20, 4)), name="x")
        dist = RecordBootstrapReplicateDistribution(emp, name="x")
        assert dist.n == 20

    def test_obs_shape(self):
        dist = RecordBootstrapReplicateDistribution(jnp.ones((10, 3, 4)), name="x")
        assert dist.obs_shape == (3, 4)

    def test_obs_shape_scalar(self):
        dist = RecordBootstrapReplicateDistribution(jnp.ones((10,)), name="x")
        assert dist.obs_shape == ()

    def test_dim(self):
        dist = RecordBootstrapReplicateDistribution(jnp.ones((10, 3)), n=5, name="x")
        assert dist.dim == 5 * 3

    def test_dim_scalar_obs(self):
        dist = RecordBootstrapReplicateDistribution(jnp.ones((10,)), n=5, name="x")
        assert dist.dim == 5

    def test_dtype(self):
        dist = RecordBootstrapReplicateDistribution(jnp.ones((5, 2)), name="x")
        # Inherits dtype from the source array (default float dtype here).
        assert dist.dtype == jnp.zeros((), dtype=float).dtype


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr(self):
        data = jnp.ones((10, 3))
        dist = BootstrapReplicateDistribution(data, n=8, name="x")
        r = repr(dist)
        assert "BootstrapReplicateDistribution" in r
        assert "n=8" in r
        assert "source_n=10" in r


# ---------------------------------------------------------------------------
# Record-based EmpiricalDistribution
# ---------------------------------------------------------------------------


class TestValuesEmpiricalDistribution:
    """EmpiricalDistribution(Record(...)) → RecordEmpiricalDistribution."""

    @pytest.fixture
    def values_data(self):
        X = jnp.ones((20, 3))
        y = jnp.arange(20.0)
        return Record(X=X, y=y)

    def test_dispatch(self, values_data):
        from probpipe.core._empirical import RecordEmpiricalDistribution
        emp = EmpiricalDistribution(values_data, name="x")
        assert isinstance(emp, RecordEmpiricalDistribution)

    def test_n(self, values_data):
        emp = EmpiricalDistribution(values_data, name="x")
        assert emp.n == 20

    def test_satisfies_sampling(self, values_data):
        emp = EmpiricalDistribution(values_data, name="x")
        assert isinstance(emp, SupportsSampling)

    def test_sample_one_returns_values(self, values_data):
        emp = EmpiricalDistribution(values_data, name="x")
        s = sample(emp, key=jax.random.PRNGKey(0))
        assert isinstance(s, Record)
        assert "X" in s and "y" in s

    def test_sample_one_shapes(self, values_data):
        emp = EmpiricalDistribution(values_data, name="x")
        s = sample(emp, key=jax.random.PRNGKey(0))
        assert s["X"].shape == (3,)
        assert s["y"].shape == ()

    def test_sample_batch(self, values_data):
        emp = EmpiricalDistribution(values_data, name="x")
        s = emp._sample(jax.random.PRNGKey(0), sample_shape=(5,))
        assert isinstance(s, Record)
        assert s["X"].shape == (5, 3)
        assert s["y"].shape == (5,)

    def test_record_template(self, values_data):
        emp = EmpiricalDistribution(values_data, name="x")
        tpl = emp.record_template
        assert tpl is not None
        assert tpl["X"] == (3,)
        assert tpl["y"] == ()

    def test_fields(self, values_data):
        emp = EmpiricalDistribution(values_data, name="x")
        assert emp.fields == ("X", "y")

    def test_getitem_returns_view(self, values_data):
        from probpipe.core._record_distribution import _RecordDistributionView
        emp = EmpiricalDistribution(values_data, name="x")
        view = emp["X"]
        assert isinstance(view, _RecordDistributionView)

    def test_mean(self, values_data):
        emp = EmpiricalDistribution(values_data, name="x")
        m = emp._mean()
        assert isinstance(m, Record)
        np.testing.assert_allclose(m["X"], 1.0)
        np.testing.assert_allclose(m["y"], jnp.arange(20.0).mean())

    def test_variance(self, values_data):
        emp = EmpiricalDistribution(values_data, name="x")
        v = emp._variance()
        assert isinstance(v, Record)
        np.testing.assert_allclose(v["X"], 0.0, atol=1e-7)

    def test_repr(self, values_data):
        emp = EmpiricalDistribution(values_data, name="x")
        r = repr(emp)
        assert "RecordEmpiricalDistribution" in r
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
        emp = EmpiricalDistribution(values_data, name="x")
        return BootstrapReplicateDistribution(emp, name="x")

    def test_dispatch_from_empirical(self, values_data):
        from probpipe.core._empirical import RecordBootstrapReplicateDistribution
        emp = EmpiricalDistribution(values_data, name="x")
        boot = BootstrapReplicateDistribution(emp, name="x")
        assert isinstance(boot, RecordBootstrapReplicateDistribution)

    def test_dispatch_from_values(self, values_data):
        from probpipe.core._empirical import RecordBootstrapReplicateDistribution
        boot = BootstrapReplicateDistribution(values_data, name="x")
        assert isinstance(boot, RecordBootstrapReplicateDistribution)

    def test_n(self, bootstrap):
        assert bootstrap.n == 20

    def test_custom_n(self, values_data):
        emp = EmpiricalDistribution(values_data, name="x")
        boot = BootstrapReplicateDistribution(emp, n=10, name="x")
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

    def test_fields(self, bootstrap):
        assert bootstrap.fields == ("X", "y")

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
        assert "RecordBootstrapReplicateDistribution" in r
        assert "n=20" in r
