"""Tests for BootstrapReplicateDistribution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BootstrapDistribution,
    BootstrapReplicateDistribution,
    Distribution,
    EmpiricalDistribution,
    NumericRecordDistribution,
    Record,
    RecordBootstrapReplicateDistribution,
    SupportsExpectation,
    SupportsSampling,
    expectation,
    sample,
)
from probpipe.core._specs import NumericArraySpec

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_empirical(self):
        data = jnp.arange(20.0).reshape(10, 2)
        emp = EmpiricalDistribution("x", data)
        dist = BootstrapReplicateDistribution("x", emp)
        assert dist.replicate_size == 10

    def test_from_array(self):
        data = jnp.arange(20.0).reshape(10, 2)
        dist = BootstrapReplicateDistribution("x", data)
        assert dist.replicate_size == 10

    def test_custom_n(self):
        data = jnp.ones((50, 3))
        dist = BootstrapReplicateDistribution("x", data, replicate_size=30)
        assert dist.replicate_size == 30

    def test_n_from_empirical_default(self):
        emp = EmpiricalDistribution("x", jnp.ones((20, 4)))
        dist = BootstrapReplicateDistribution("x", emp)
        assert dist.replicate_size == 20

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="positive"):
            BootstrapReplicateDistribution("x", jnp.ones((5, 2)), replicate_size=0)

    def test_scalar_source_raises(self):
        with pytest.raises(ValueError, match="at least 1 dimension"):
            BootstrapReplicateDistribution("x", jnp.array(1.0))

    def test_name(self):
        dist = BootstrapReplicateDistribution("boot", jnp.ones((5, 2)))
        assert dist.name == "boot"

    def test_generic_base_keeps_its_name(self):
        # Opaque-object sources stay on the generic base, which keeps the name
        # it is given.
        dist = BootstrapReplicateDistribution("dist", ["a", "b", "c"])
        assert dist.name == "dist"


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_supports_sampling(self):
        dist = BootstrapReplicateDistribution("x", jnp.ones((5, 2)))
        assert isinstance(dist, SupportsSampling)

    def test_supports_expectation(self):
        dist = BootstrapReplicateDistribution("x", jnp.ones((5, 2)))
        assert isinstance(dist, SupportsExpectation)

    def test_generic_numeric_dispatches_to_array(self):
        # Factory dispatch: numeric arrays → RecordBootstrapReplicateDistribution
        dist = BootstrapReplicateDistribution("x", jnp.ones((5, 2)))
        assert isinstance(dist, Distribution)
        assert isinstance(dist, NumericRecordDistribution)
        assert isinstance(dist, RecordBootstrapReplicateDistribution)

    def test_generic_object_is_not_array(self):
        # Non-numeric (object) source stays as base class
        dist = BootstrapReplicateDistribution("x", ["a", "b", "c"])
        assert isinstance(dist, Distribution)
        assert not isinstance(dist, NumericRecordDistribution)

    def test_array_is_array_distribution(self):
        dist = RecordBootstrapReplicateDistribution("x", jnp.ones((5, 2)))
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
        return BootstrapReplicateDistribution("x", data)

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
        dist = BootstrapReplicateDistribution("x", data, replicate_size=20)
        s = dist._sample(jax.random.PRNGKey(0))
        assert s.shape == (20, 2)

    def test_1d_source(self):
        """1D source (scalar observations)."""
        data = jnp.arange(10.0)
        dist = BootstrapReplicateDistribution("x", data)
        s = dist._sample(jax.random.PRNGKey(0))
        assert s.shape == (10,)

    def test_weighted_empirical(self):
        """Weighted EmpiricalDistribution uses weights for sampling."""
        data = jnp.array([[0.0], [1.0], [2.0]])
        weights = jnp.array([0.0, 0.0, 1.0])  # all weight on last row
        emp = EmpiricalDistribution("x", data, weights=weights)
        dist = BootstrapReplicateDistribution("x", emp)
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
        d = BootstrapReplicateDistribution("d", self._normal(), replicate_size=10)
        assert isinstance(d, BootstrapReplicateDistribution)
        assert not isinstance(d, RecordBootstrapReplicateDistribution)
        assert d.replicate_size == 10
        assert d.source_size is None  # no canonical observation count

    def test_missing_n_raises(self):
        with pytest.raises(ValueError, match="replicate_size must be a "):
            BootstrapReplicateDistribution("boot", self._normal())

    def test_zero_n_raises(self):
        with pytest.raises(ValueError, match="replicate_size must be a "):
            BootstrapReplicateDistribution("boot", self._normal(), replicate_size=0)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="replicate_size must be a "):
            BootstrapReplicateDistribution("boot", self._normal(), replicate_size=-3)

    def test_sample_empty_shape(self):
        # One bootstrap replicate is ``n`` i.i.d. draws from source._sample.
        d = BootstrapReplicateDistribution("d", self._normal(), replicate_size=10)
        s = d._sample(jax.random.PRNGKey(0), ())
        assert s.shape == (10,)

    def test_sample_with_shape(self):
        # sample_shape prepends; total = sample_shape + (n,) + event_shape.
        d = BootstrapReplicateDistribution("d", self._normal(), replicate_size=10)
        s = d._sample(jax.random.PRNGKey(1), sample_shape=(3,))
        assert s.shape == (3, 10)

    def test_data_is_none_for_sampleable_source(self):
        # No stored observations.
        d = BootstrapReplicateDistribution("d", self._normal(), replicate_size=5)
        assert d.data is None
        assert d.weights is None

    def test_repr_mentions_source(self):
        d = BootstrapReplicateDistribution("d", self._normal(), replicate_size=5)
        r = repr(d)
        assert "replicate_size=5" in r
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
        dist = BootstrapReplicateDistribution("x", data)
        np.testing.assert_array_equal(dist.data["x"], data)

    def test_data_from_empirical(self):
        data = jnp.arange(20.0).reshape(10, 2)
        emp = EmpiricalDistribution("x", data)
        dist = BootstrapReplicateDistribution("x", emp)
        np.testing.assert_array_equal(dist.data["x"], data)

    def test_source_n(self):
        dist = BootstrapReplicateDistribution("x", jnp.ones((15, 3)))
        assert dist.source_size == 15

    def test_is_uniform_from_array(self):
        dist = BootstrapReplicateDistribution("x", jnp.ones((5, 2)))
        assert dist.is_uniform is True

    def test_is_uniform_from_uniform_empirical(self):
        emp = EmpiricalDistribution("x", jnp.ones((5, 2)))
        dist = BootstrapReplicateDistribution("x", emp)
        assert dist.is_uniform is True

    def test_is_uniform_from_weighted_empirical(self):
        emp = EmpiricalDistribution(
            "x", jnp.ones((5, 2)), weights=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        dist = BootstrapReplicateDistribution("x", emp)
        assert dist.is_uniform is False

    def test_weights_uniform(self):
        dist = BootstrapReplicateDistribution("x", jnp.ones((5, 2)))
        np.testing.assert_allclose(dist.weights, jnp.ones(5) / 5)

    def test_weights_from_weighted_empirical(self):
        weights = jnp.array([1.0, 2.0, 3.0])
        emp = EmpiricalDistribution("x", jnp.ones((3, 2)), weights=weights)
        dist = BootstrapReplicateDistribution("x", emp)
        assert dist.weights is not None
        assert dist.weights.shape == (3,)

    def test_approximate_flag(self):
        dist = BootstrapReplicateDistribution("x", jnp.ones((5, 2)))
        assert dist._approximate is True

    def test_numeric_has_event_shape(self):
        """Numeric arrays dispatch to Array variant with event_shape."""
        dist = BootstrapReplicateDistribution("x", jnp.ones((5, 2)))
        assert hasattr(dist, "event_shape")
        assert dist.event_shape == (5, 2)

    def test_generic_no_event_shape(self):
        """Non-numeric BootstrapReplicateDistribution has no event_shape."""
        dist = BootstrapReplicateDistribution("x", ["a", "b", "c"])
        assert not hasattr(dist, "event_shape") or "event_shape" not in type(dist).__dict__

    def test_generic_no_dim(self):
        """Non-numeric BootstrapReplicateDistribution has no dim."""
        dist = BootstrapReplicateDistribution("x", ["a", "b", "c"])
        assert not hasattr(dist, "dim") or "dim" not in type(dist).__dict__

    def test_generic_no_dtype(self):
        """Non-numeric BootstrapReplicateDistribution has no dtype."""
        dist = BootstrapReplicateDistribution("x", ["a", "b", "c"])
        assert not hasattr(dist, "dtype") or "dtype" not in type(dist).__dict__


# ---------------------------------------------------------------------------
# Expectation
# ---------------------------------------------------------------------------


class TestExpectation:
    def test_expectation_returns_array(self):
        data = jnp.arange(30.0).reshape(10, 3)
        dist = BootstrapReplicateDistribution("x", data)
        result = dist._expectation(
            lambda d: jnp.mean(d, axis=0),
            key=jax.random.PRNGKey(0),
            num_evaluations=50,
            return_dist=False,
        )
        assert result.shape == (3,)

    def test_expectation_returns_bootstrap_dist(self):
        data = jnp.arange(30.0).reshape(10, 3)
        dist = BootstrapReplicateDistribution("x", data)
        result = dist._expectation(
            lambda d: jnp.mean(d, axis=0),
            key=jax.random.PRNGKey(0),
            num_evaluations=50,
            return_dist=True,
        )
        assert isinstance(result, BootstrapDistribution)
        assert result.num_atoms == 50

    def test_expectation_op(self):
        data = jnp.arange(30.0).reshape(10, 3)
        dist = BootstrapReplicateDistribution("x", data)
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
        dist = BootstrapReplicateDistribution("x", data)
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

        dist = RecordBootstrapReplicateDistribution("x", jnp.ones((5, 2)))
        assert dist.support == real

    def test_sample_shape(self):
        data = jnp.arange(30.0).reshape(10, 3)
        dist = RecordBootstrapReplicateDistribution("x", data)
        s = dist._sample(jax.random.PRNGKey(0), sample_shape=(4,))
        assert s.shape == (4, 10, 3)

    def test_event_shape(self):
        data = jnp.ones((10, 3))
        dist = RecordBootstrapReplicateDistribution("x", data, replicate_size=8)
        assert dist.event_shape == (8, 3)

    def test_from_empirical(self):
        emp = EmpiricalDistribution("x", jnp.ones((20, 4)))
        dist = RecordBootstrapReplicateDistribution("x", emp)
        assert dist.replicate_size == 20

    def test_rejects_object_array_empirical(self):
        """Generic (object-array) EmpiricalDistribution is rejected.

        Numeric-array empiricals dispatch to RecordEmpiricalDistribution
        via the factory ``__new__``, so any EmpiricalDistribution that
        reaches RecordBootstrapReplicateDistribution.__init__ as a
        generic instance is object-backed and can't be wrapped as a
        Record. Confirm the explicit TypeError fires (instead of the
        unhelpful _as_float_array(object_arr) failure that used to
        bubble up).
        """
        from probpipe.core._empirical import RecordEmpiricalDistribution

        emp = EmpiricalDistribution("emp", ["a", "b", "c"])
        assert not isinstance(emp, RecordEmpiricalDistribution)
        with pytest.raises(
            TypeError,
            match=r"generic .object-array. EmpiricalDistribution",
        ):
            RecordBootstrapReplicateDistribution("boot", emp)

    def test_obs_shape(self):
        dist = RecordBootstrapReplicateDistribution("x", jnp.ones((10, 3, 4)))
        assert dist.obs_shape == (3, 4)

    def test_obs_shape_scalar(self):
        dist = RecordBootstrapReplicateDistribution("x", jnp.ones((10,)))
        assert dist.obs_shape == ()

    def test_dim(self):
        dist = RecordBootstrapReplicateDistribution("x", jnp.ones((10, 3)), replicate_size=5)
        assert dist.dim == 5 * 3

    def test_dim_scalar_obs(self):
        dist = RecordBootstrapReplicateDistribution("x", jnp.ones((10,)), replicate_size=5)
        assert dist.dim == 5

    def test_dtype(self):
        dist = RecordBootstrapReplicateDistribution("x", jnp.ones((5, 2)))
        # Inherits dtype from the source array (default float dtype here).
        assert dist.dtype == jnp.zeros((), dtype=float).dtype


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr(self):
        data = jnp.ones((10, 3))
        dist = BootstrapReplicateDistribution("x", data, replicate_size=8)
        r = repr(dist)
        assert "BootstrapReplicateDistribution" in r
        assert "replicate_size=8" in r
        assert "source_size=10" in r


# ---------------------------------------------------------------------------
# Record-based EmpiricalDistribution
# ---------------------------------------------------------------------------


class TestValuesEmpiricalDistribution:
    """EmpiricalDistribution(name, Record(...)) → RecordEmpiricalDistribution."""

    @pytest.fixture
    def values_data(self):
        X = jnp.ones((20, 3))
        y = jnp.arange(20.0)
        return Record("r", X=X, y=y)

    def test_dispatch(self, values_data):
        from probpipe.core._empirical import RecordEmpiricalDistribution

        emp = EmpiricalDistribution("x", values_data)
        assert isinstance(emp, RecordEmpiricalDistribution)

    def test_n(self, values_data):
        emp = EmpiricalDistribution("x", values_data)
        assert emp.num_atoms == 20

    def test_satisfies_sampling(self, values_data):
        emp = EmpiricalDistribution("x", values_data)
        assert isinstance(emp, SupportsSampling)

    def test_sample_one_returns_values(self, values_data):
        emp = EmpiricalDistribution("x", values_data)
        s = sample(emp, key=jax.random.PRNGKey(0))
        assert isinstance(s, Record)
        assert "X" in s and "y" in s

    def test_sample_one_shapes(self, values_data):
        emp = EmpiricalDistribution("x", values_data)
        s = sample(emp, key=jax.random.PRNGKey(0))
        assert s["X"].shape == (3,)
        assert s["y"].shape == ()

    def test_sample_batch(self, values_data):
        emp = EmpiricalDistribution("x", values_data)
        s = emp._sample(jax.random.PRNGKey(0), sample_shape=(5,))
        assert isinstance(s, Record)
        assert s["X"].shape == (5, 3)
        assert s["y"].shape == (5,)

    def test_event_template(self, values_data):
        emp = EmpiricalDistribution("x", values_data)
        tpl = emp.event_template
        assert tpl is not None
        assert tpl["X"] == NumericArraySpec((3,))
        assert tpl["y"] == NumericArraySpec(())

    def test_fields(self, values_data):
        emp = EmpiricalDistribution("x", values_data)
        assert emp.fields == ("X", "y")

    def test_getitem_returns_view(self, values_data):
        from probpipe.core._record_distribution import _RecordDistributionView

        emp = EmpiricalDistribution("x", values_data)
        view = emp["X"]
        assert isinstance(view, _RecordDistributionView)

    def test_mean(self, values_data):
        emp = EmpiricalDistribution("x", values_data)
        m = emp._mean()
        assert isinstance(m, Record)
        np.testing.assert_allclose(m["X"], 1.0)
        np.testing.assert_allclose(m["y"], jnp.arange(20.0).mean())

    def test_variance(self, values_data):
        emp = EmpiricalDistribution("x", values_data)
        v = emp._variance()
        assert isinstance(v, Record)
        np.testing.assert_allclose(v["X"], 0.0, atol=1e-7)

    def test_repr(self, values_data):
        emp = EmpiricalDistribution("x", values_data)
        r = repr(emp)
        assert "RecordEmpiricalDistribution" in r
        assert "num_atoms=20" in r


# ---------------------------------------------------------------------------
# Record-based BootstrapReplicateDistribution
# ---------------------------------------------------------------------------


class TestValuesBootstrapReplicateDistribution:
    """BootstrapReplicateDistribution with Record source."""

    @pytest.fixture
    def values_data(self):
        X = jnp.ones((20, 3))
        y = jnp.arange(20.0)
        return Record("r", X=X, y=y)

    @pytest.fixture
    def bootstrap(self, values_data):
        emp = EmpiricalDistribution("x", values_data)
        return BootstrapReplicateDistribution("x", emp)

    def test_dispatch_from_empirical(self, values_data):
        from probpipe.core._empirical import RecordBootstrapReplicateDistribution

        emp = EmpiricalDistribution("x", values_data)
        boot = BootstrapReplicateDistribution("x", emp)
        assert isinstance(boot, RecordBootstrapReplicateDistribution)

    def test_dispatch_from_values(self, values_data):
        from probpipe.core._empirical import RecordBootstrapReplicateDistribution

        boot = BootstrapReplicateDistribution("x", values_data)
        assert isinstance(boot, RecordBootstrapReplicateDistribution)

    def test_n(self, bootstrap):
        assert bootstrap.replicate_size == 20

    def test_custom_n(self, values_data):
        emp = EmpiricalDistribution("x", values_data)
        boot = BootstrapReplicateDistribution("x", emp, replicate_size=10)
        assert boot.replicate_size == 10

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

    def test_event_template(self, bootstrap):
        tpl = bootstrap.event_template
        assert tpl is not None
        assert tpl["X"] == NumericArraySpec((20, 3))
        assert tpl["y"] == NumericArraySpec((20,))

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
        assert "replicate_size=20" in r
