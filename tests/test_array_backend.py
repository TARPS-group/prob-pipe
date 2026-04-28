"""Tests for probpipe.core._array_backend (aux registry + Record/NumericRecord round-trip)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    AuxHooks,
    NumericRecord,
    Record,
    RecordArray,
    aux_for,
    register_aux,
)
from probpipe.core._array_backend import aux_registry


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------


class TestRegistryLookup:
    def test_aux_for_numpy_returns_none(self):
        assert aux_for(np.zeros(3)) is None

    def test_aux_for_jax_returns_none(self):
        assert aux_for(jnp.zeros(3)) is None

    def test_aux_for_python_scalar_returns_none(self):
        assert aux_for(1.5) is None
        assert aux_for(42) is None
        assert aux_for(True) is None

    def test_aux_for_xarray_returns_hooks(self):
        xr = pytest.importorskip("xarray")
        da = xr.DataArray([1.0, 2.0])
        hooks = aux_for(da)
        assert hooks is not None
        assert isinstance(hooks, AuxHooks)

    def test_aux_for_pandas_series_returns_hooks(self):
        pd = pytest.importorskip("pandas")
        s = pd.Series([1.0, 2.0])
        hooks = aux_for(s)
        assert hooks is not None

    def test_aux_for_pandas_dataframe_returns_hooks(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"x": [1.0, 2.0]})
        hooks = aux_for(df)
        assert hooks is not None

    def test_register_aux_then_lookup(self):
        # Custom backend type — not in registry by default.
        class MyArrayLike:
            shape = (3,)
            dtype = jnp.float32
            ndim = 1

            def __array__(self, dtype=None, copy=None):
                arr = np.array([1.0, 2.0, 3.0])
                return arr if copy is False else arr.copy()

        assert aux_for(MyArrayLike()) is None
        register_aux(
            MyArrayLike,
            capture=lambda leaf: ("custom",),
            restore=lambda arr, aux: arr,
        )
        try:
            hooks = aux_for(MyArrayLike())
            assert hooks is not None
            assert hooks.capture(MyArrayLike()) == ("custom",)
        finally:
            del aux_registry[MyArrayLike]

    def test_aux_for_walks_mro(self):
        """An instance of a registered type's subclass picks up the
        base-class hooks (regression for the MRO-walk semantics
        documented at ``_array_backend.py:103-108``)."""
        class Base:
            pass

        class Sub(Base):
            pass

        register_aux(
            Base,
            capture=lambda leaf: "base-aux",
            restore=lambda arr, aux: ("restored", aux),
        )
        try:
            hooks = aux_for(Sub())
            assert hooks is not None
            assert hooks.capture(Sub()) == "base-aux"
        finally:
            del aux_registry[Base]

    def test_register_aux_overwrites_silently(self):
        """Re-registering an existing leaf type silently overwrites the
        previous hooks (documented at ``_array_backend.py:93-99``)."""
        class MyType:
            pass

        register_aux(
            MyType,
            capture=lambda leaf: "first",
            restore=lambda arr, aux: ("first", aux),
        )
        try:
            assert aux_for(MyType()).capture(MyType()) == "first"
            register_aux(
                MyType,
                capture=lambda leaf: "second",
                restore=lambda arr, aux: ("second", aux),
            )
            hooks = aux_for(MyType())
            assert hooks.capture(MyType()) == "second"
        finally:
            del aux_registry[MyType]


# ---------------------------------------------------------------------------
# xarray round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def da():
    xr = pytest.importorskip("xarray")
    return xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=["t"],
        coords={"t": [10, 20, 30]},
        attrs={"units": "meters"},
        name="temps",
    )


class TestXarrayRoundTrip:
    def test_to_numeric_then_to_native_preserves_dims(self, da):
        back = Record(temps=da).to_numeric().to_native()
        assert back["temps"].dims == ("t",)

    def test_to_numeric_then_to_native_preserves_coords(self, da):
        back = Record(temps=da).to_numeric().to_native()
        np.testing.assert_array_equal(back["temps"].coords["t"].values, [10, 20, 30])

    def test_to_numeric_then_to_native_preserves_attrs(self, da):
        back = Record(temps=da).to_numeric().to_native()
        assert back["temps"].attrs == {"units": "meters"}

    def test_to_numeric_then_to_native_preserves_name(self, da):
        back = Record(temps=da).to_numeric().to_native()
        assert back["temps"].name == "temps"

    def test_direct_numeric_record_construction_preserves_metadata(self, da):
        # The "no to_numeric() detour" path must produce identical results.
        back = NumericRecord(temps=da).to_native()
        assert back["temps"].dims == ("t",)
        np.testing.assert_array_equal(back["temps"].coords["t"].values, [10, 20, 30])

    def test_to_native_returns_plain_record_not_numeric(self, da):
        nr = NumericRecord(temps=da)
        back = nr.to_native()
        assert isinstance(back, Record)
        assert not isinstance(back, NumericRecord)

    def test_values_round_trip_within_dtype_tolerance(self, da):
        back = NumericRecord(temps=da).to_native()
        np.testing.assert_allclose(np.asarray(back["temps"]), [1.0, 2.0, 3.0])

    def test_nested_numeric_record_round_trips_xarray(self, da):
        """Aux on a nested ``NumericRecord`` round-trips: ``to_native``
        recurses into nested children (``_numeric_record.py:286-289``).
        """
        outer = NumericRecord(inner=NumericRecord(temps=da))
        back = outer.to_native()
        # Outer is a permissive Record after to_native (restored leaves
        # may not satisfy the NumericRecord invariant). The inner gets
        # the same treatment recursively.
        assert isinstance(back, Record)
        inner = back["inner"]
        assert isinstance(inner, Record)
        assert inner["temps"].dims == ("t",)
        np.testing.assert_array_equal(
            inner["temps"].coords["t"].values, [10, 20, 30],
        )
        assert inner["temps"].attrs == {"units": "meters"}

    def test_xarray_multidim_coord_round_trips(self):
        """Per-coord dims and attrs round-trip even for multi-dim coords
        (regression for the richer-capture tweak)."""
        xr = pytest.importorskip("xarray")
        # 2-D ``area`` coord aligned with both axes; 1-D ``t`` coord
        # with its own attrs.
        area = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        da_md = xr.DataArray(
            np.ones((2, 3)),
            dims=("x", "t"),
            coords={
                "t": xr.DataArray([10, 20, 30], dims=["t"], attrs={"unit": "s"}),
                "area": xr.DataArray(area, dims=("x", "t")),
            },
        )
        back = NumericRecord(field=da_md).to_native()["field"]
        assert back.coords["t"].dims == ("t",)
        assert back.coords["t"].attrs == {"unit": "s"}
        assert back.coords["area"].dims == ("x", "t")
        np.testing.assert_array_equal(
            back.coords["area"].values, area,
        )


# ---------------------------------------------------------------------------
# pandas round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def pd_module():
    return pytest.importorskip("pandas")


@pytest.fixture
def datetime_series(pd_module):
    return pd_module.Series(
        [10.0, 20.0, 30.0],
        index=pd_module.DatetimeIndex(
            ["2024-01-01", "2024-01-02", "2024-01-03"]
        ),
        name="counts",
    )


class TestPandasRoundTrip:
    def test_series_round_trip_preserves_index(self, pd_module, datetime_series):
        back = NumericRecord(counts=datetime_series).to_native()
        assert isinstance(back["counts"], pd_module.Series)
        np.testing.assert_array_equal(back["counts"].index, datetime_series.index)

    def test_series_round_trip_preserves_name(self, datetime_series):
        back = NumericRecord(counts=datetime_series).to_native()
        assert back["counts"].name == "counts"

    def test_dataframe_round_trip_preserves_columns(self, pd_module):
        df = pd_module.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        back = NumericRecord(data=df).to_native()
        assert isinstance(back["data"], pd_module.DataFrame)
        assert list(back["data"].columns) == ["x", "y"]

    def test_dataframe_round_trip_preserves_index(self, pd_module):
        idx = pd_module.Index(["a", "b", "c"], name="row")
        df = pd_module.DataFrame({"v": [1.0, 2.0, 3.0]}, index=idx)
        back = NumericRecord(data=df).to_native()
        np.testing.assert_array_equal(back["data"].index.values, ["a", "b", "c"])
        assert back["data"].index.name == "row"


# ---------------------------------------------------------------------------
# Mixed-backend Record
# ---------------------------------------------------------------------------


class TestMixedBackendRecord:
    def test_mixed_record_round_trip_per_field(self):
        xr = pytest.importorskip("xarray")
        pd = pytest.importorskip("pandas")
        da = xr.DataArray([1.0, 2.0, 3.0], dims=["t"], coords={"t": [0, 1, 2]})
        s = pd.Series([10, 20, 30], name="cs")
        r = Record(temp=da, counts=s, intercept=1.5)
        back = r.to_numeric().to_native()
        assert isinstance(back["temp"], xr.DataArray)
        assert isinstance(back["counts"], pd.Series)
        # Plain numeric leaf → restored as jax array.
        assert isinstance(back["intercept"], jax.Array)


# ---------------------------------------------------------------------------
# Path equivalence: NumericRecord(...) and Record(...).to_numeric()
# ---------------------------------------------------------------------------


@pytest.fixture
def da_simple():
    xr = pytest.importorskip("xarray")
    return xr.DataArray(
        [1.0, 2.0, 3.0], dims=["t"], coords={"t": [10, 20, 30]}
    )


class TestPathEquivalence:
    def test_aux_keys_match(self, da_simple):
        nr_direct = NumericRecord(x=da_simple, y=jnp.array(1.5))
        nr_via_to_numeric = Record(x=da_simple, y=jnp.array(1.5)).to_numeric()
        keys_direct = set((nr_direct.aux or {}).keys())
        keys_via = set((nr_via_to_numeric.aux or {}).keys())
        assert keys_direct == keys_via == {"x"}

    def test_store_arrays_bitwise_equal(self, da_simple):
        nr_direct = NumericRecord(x=da_simple, y=2.5)
        nr_via_to_numeric = Record(x=da_simple, y=2.5).to_numeric()
        for f in nr_direct.fields:
            np.testing.assert_array_equal(
                np.asarray(nr_direct[f]), np.asarray(nr_via_to_numeric[f])
            )

    def test_to_native_results_match(self, da_simple):
        b1 = NumericRecord(x=da_simple, y=2.5).to_native()
        b2 = Record(x=da_simple, y=2.5).to_numeric().to_native()
        # xarray fields restored identically (dims + coord values).
        assert b1["x"].dims == b2["x"].dims
        np.testing.assert_array_equal(
            b1["x"].coords["t"].values, b2["x"].coords["t"].values
        )
        # Numeric pass-through fields.
        np.testing.assert_array_equal(np.asarray(b1["y"]), np.asarray(b2["y"]))


# ---------------------------------------------------------------------------
# Non-coercible leaves raise
# ---------------------------------------------------------------------------


class TestNonCoercibleLeavesRaise:
    def test_numeric_record_string_raises(self):
        # ``name=`` is the Record-name kwarg, so use a different field.
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord(label="alice")

    def test_to_numeric_string_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            Record(label="alice").to_numeric()

    def test_to_numeric_opaque_raises(self):
        class Opaque:
            pass

        with pytest.raises(TypeError, match="numeric"):
            Record(thing=Opaque()).to_numeric()


# ---------------------------------------------------------------------------
# Transforms drop aux
# ---------------------------------------------------------------------------


@pytest.fixture
def da_zero_indexed():
    xr = pytest.importorskip("xarray")
    return xr.DataArray(
        [1.0, 2.0, 3.0], dims=["t"], coords={"t": [0, 1, 2]}
    )


class TestTransformsDropAux:
    def test_map_drops_aux(self, da_zero_indexed):
        nr = NumericRecord(x=da_zero_indexed)
        assert nr.aux is not None
        out = nr.map(jnp.log)
        assert out.aux is None

    def test_replace_drops_aux(self, da_zero_indexed):
        nr = NumericRecord(x=da_zero_indexed, y=jnp.array(1.5))
        assert nr.aux is not None
        out = nr.replace(y=jnp.array(2.0))
        assert out.aux is None

    def test_jax_tree_map_drops_aux(self, da_zero_indexed):
        nr = NumericRecord(x=da_zero_indexed)
        out = jax.tree.map(lambda v: v + 1, nr)
        assert isinstance(out, NumericRecord)
        assert out.aux is None


# ---------------------------------------------------------------------------
# RecordArray.stack drops aux
# ---------------------------------------------------------------------------


class TestRecordArrayStackDropsAux:
    def test_stack_produces_plain_jax_array(self):
        xr = pytest.importorskip("xarray")
        # Build per-row NumericRecords carrying xarray leaves so the
        # source records *do* carry aux.
        records = [
            NumericRecord(x=xr.DataArray([float(i), float(i + 1)], dims=["t"]))
            for i in range(3)
        ]
        # Each source record has aux for ``x``.
        assert all(r.aux is not None for r in records)
        ra = RecordArray.stack(records)
        # RecordArray has no aux slot today; the stacked column is a
        # plain ``jax.Array``, not a per-row xarray structure.
        col = ra["x"]
        assert isinstance(col, jax.Array)
        assert col.shape == (3, 2)


# ---------------------------------------------------------------------------
# Pytree round-trip drops aux
# ---------------------------------------------------------------------------


class TestPytreeRoundTrip:
    def test_flatten_unflatten_drops_aux(self):
        xr = pytest.importorskip("xarray")
        da = xr.DataArray([1.0, 2.0, 3.0], dims=["t"], coords={"t": [0, 1, 2]})
        nr = NumericRecord(x=da)
        leaves, treedef = jax.tree_util.tree_flatten(nr)
        out = jax.tree_util.tree_unflatten(treedef, leaves)
        # Aux is reconstructed from leaf types after unflatten, and the
        # leaves are now jax.Array, so no aux is recaptured.
        assert isinstance(out, NumericRecord)
        assert out.aux is None
