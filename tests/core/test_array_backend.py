"""Tests for the array-backend registry and native-form NumericRecord leaves.

Covers the :class:`~probpipe.ArrayBackend` registry (lookup, registration,
MRO walk), the native-storage contract (leaves stored verbatim; metadata
survives construction, structural transforms, and pickling with no capture
step), the lazy compute boundary (no materialisation at construction; the
set-once conversion cache), the JAX boundary (native types do not cross), the
eager batch boundary, and the end-to-end effect of registering a backend for
a type the duck path cannot see.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import ArrayBackend, NumericRecord, Record, array_backend_for, register_array_backend
from probpipe.core import _array_backend
from probpipe.core.event_template import ArraySpec, EventTemplate, NumericEventTemplate

xr = pytest.importorskip("xarray")
pd = pytest.importorskip("pandas")


@pytest.fixture
def clean_registry():
    """Snapshot and restore the backend registry around a test's registrations."""
    saved = dict(_array_backend._backend_registry)
    yield
    _array_backend._backend_registry.clear()
    _array_backend._backend_registry.update(saved)


@pytest.fixture
def da():
    return xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=["t"],
        coords={"t": [10, 20, 30]},
        attrs={"units": "meters"},
        name="temps",
    )


# ---------------------------------------------------------------------------
# Registry lookup and registration
# ---------------------------------------------------------------------------


class TestRegistryLookup:
    def test_numpy_and_jax_unregistered(self):
        # Bare arrays are their own native form: the duck path covers them.
        assert array_backend_for(np.zeros(2)) is None
        assert array_backend_for(jnp.zeros(2)) is None

    def test_python_scalar_unregistered(self):
        assert array_backend_for(1.5) is None

    def test_xarray_and_series_registered_for_metadata(self):
        # xarray / pandas are registered (not left to the duck path) so their
        # identity-bearing metadata (coords / index) is available to
        # fingerprint and __eq__ via the metadata hook.
        da_backend = array_backend_for(xr.DataArray(np.zeros(2)))
        s_backend = array_backend_for(pd.Series([1.0, 2.0]))
        assert da_backend is not None and s_backend is not None
        # metadata carries the identity-bearing container attributes
        da = xr.DataArray(np.zeros(2), dims=["t"], coords={"t": [0, 1]})
        assert "coords" in da_backend.metadata(da)
        assert "index" in s_backend.metadata(pd.Series([1.0, 2.0]))

    def test_bare_arrays_have_no_metadata(self):
        # An unregistered numpy-protocol leaf carries no identity metadata:
        # its identity is its values alone.
        assert array_backend_for(np.zeros(2)) is None

    def test_dataframe_registered_builtin(self):
        backend = array_backend_for(pd.DataFrame({"a": [1.0]}))
        assert backend is not None
        assert backend.event_shape(pd.DataFrame({"a": [1.0, 2.0]})) == (2, 1)

    def test_dataframe_numpy_dtype_homogeneous_and_mixed(self):
        backend = array_backend_for(pd.DataFrame({"a": [1.0]}))
        homo = pd.DataFrame({"a": [1.0], "b": [2.0]})
        mixed = pd.DataFrame({"a": [1.0], "b": [2]})
        assert backend.numpy_dtype(homo) == np.dtype("float64")
        assert backend.numpy_dtype(mixed) is None  # no single dtype

    def test_dataframe_is_numeric_per_instance(self):
        backend = array_backend_for(pd.DataFrame({"a": [1.0]}))
        assert backend.is_numeric(pd.DataFrame({"a": [1.0]}))
        assert not backend.is_numeric(pd.DataFrame({"a": ["s"]}))

    def test_register_then_lookup(self, clean_registry):
        class Box:
            pass

        backend = ArrayBackend(event_shape=lambda b: (1,))
        register_array_backend(Box, backend)
        assert array_backend_for(Box()) is backend

    def test_lookup_walks_mro(self, clean_registry):
        class Base:
            pass

        class Sub(Base):
            pass

        backend = ArrayBackend(event_shape=lambda b: (1,))
        register_array_backend(Base, backend)
        assert array_backend_for(Sub()) is backend

    def test_reregister_warns_and_overwrites(self, clean_registry):
        class Box:
            pass

        first = ArrayBackend(event_shape=lambda b: (1,))
        second = ArrayBackend(event_shape=lambda b: (2,))
        register_array_backend(Box, first)
        with pytest.warns(UserWarning, match="overwriting the existing ArrayBackend"):
            register_array_backend(Box, second)
        assert array_backend_for(Box()) is second


# ---------------------------------------------------------------------------
# Native storage: metadata survives with no capture/restore step
# ---------------------------------------------------------------------------


class TestNativeStorage:
    def test_leaf_stored_verbatim(self, da):
        nr = NumericRecord("nr", temps=da)
        assert nr["temps"] is da

    def test_metadata_intact_after_construction(self, da):
        nr = NumericRecord("nr", temps=da)
        assert nr["temps"].dims == ("t",)
        assert [int(v) for v in nr["temps"].coords["t"].values] == [10, 20, 30]
        assert nr["temps"].attrs == {"units": "meters"}
        assert nr["temps"].name == "temps"

    def test_to_numeric_is_identity(self, da):
        nr = NumericRecord("nr", temps=da)
        assert nr.to_numeric() is nr

    def test_construction_paths_agree(self, da):
        # Direct construction, promoted construction, and to_numeric() store
        # the identical native leaf and produce equal records.
        direct = NumericRecord("r", temps=da, x=1.0)
        promoted = Record("r", temps=da, x=1.0)
        converted = Record("r", temps=da, x=1.0).to_numeric()
        for rec in (direct, promoted, converted):
            assert type(rec) is NumericRecord
            assert rec["temps"] is da
        assert direct == promoted == converted

    def test_nested_native_leaf(self, da):
        inner = NumericRecord("grp", temps=da)
        outer = NumericRecord("outer", grp=inner, extra=1.0)
        assert outer.at_path("grp") is inner
        assert outer["grp/temps"] is da

    def test_series_stored_verbatim(self):
        s = pd.Series([1.0, 2.0], index=["a", "b"], name="s")
        nr = NumericRecord("nr", s=s)
        assert nr["s"] is s
        assert list(nr["s"].index) == ["a", "b"]

    def test_dataframe_stored_verbatim_and_vectorized(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        nr = NumericRecord("nr", tbl=df)
        assert nr["tbl"] is df
        assert nr.vector_size == 4
        np.testing.assert_allclose(nr.to_vector(), [1.0, 3.0, 2.0, 4.0])

    def test_mixed_backend_record(self, da):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        nr = NumericRecord("nr", temps=da, tbl=df, x=jnp.array(1.0))
        assert nr["temps"] is da
        assert nr["tbl"] is df
        assert isinstance(nr["x"], jnp.ndarray)

    def test_non_numeric_leaves_raise(self):
        with pytest.raises(TypeError, match="must be a numeric"):
            NumericRecord("nr", s="text")
        with pytest.raises(TypeError, match="must be a numeric"):
            Record("r", x=1.0, o=object()).to_numeric()
        with pytest.raises(TypeError, match="must be a numeric"):
            NumericRecord("nr", tbl=pd.DataFrame({"a": ["s"]}))


# ---------------------------------------------------------------------------
# Structural transforms preserve native leaves (ported metadata-survival suite)
# ---------------------------------------------------------------------------


class TestTransformsPreserveNativeLeaves:
    def test_without_preserves_top_level_native_leaf(self, da):
        nr = NumericRecord("nr", temps=da, extra=jnp.array(1.0))
        edited = nr.without("extra")
        assert edited["temps"] is da
        assert type(edited["temps"]) is xr.DataArray

    def test_without_preserves_nested_native_leaf(self, da):
        outer = NumericRecord("outer", grp=NumericRecord("grp", temps=da), extra=jnp.array(1.0))
        edited = outer.without("extra")
        assert edited["grp/temps"] is da

    def test_replace_preserves_untouched_native_leaf(self, da):
        nr = NumericRecord("nr", temps=da, extra=jnp.array(1.0))
        edited = nr.replace(extra=jnp.array(2.0))
        assert edited["temps"] is da

    def test_replace_swaps_in_new_native_leaf(self, da):
        nr = NumericRecord("nr", temps=da, extra=jnp.array(1.0))
        new_da = xr.DataArray(np.zeros(2), dims=["s"])
        edited = nr.replace(temps=new_da)
        assert edited["temps"] is new_da

    def test_merge_preserves_native_leaves_from_both_sides(self, da):
        left = NumericRecord("left", temps=da)
        s = pd.Series([1.0, 2.0])
        right = NumericRecord("right", s=s)
        merged = left.merge(right)
        assert merged["temps"] is da
        assert merged["s"] is s

    def test_with_path_names_preserves_native_leaf(self, da):
        nr = NumericRecord("nr", temps=da, extra=jnp.array(1.0))
        renamed = nr.with_path_names(temps="warmth")
        assert renamed["warmth"] is da

    def test_identity_map_preserves_native_leaf(self, da):
        # map rebuilds the record but stores whatever f returns verbatim — it
        # does not coerce the leaf — so an identity f yields a *new* record
        # whose leaf is the same native object.
        nr = NumericRecord("nr", temps=da)
        mapped = nr.map(lambda x: x)
        assert mapped is not nr
        assert mapped["temps"] is da

    def test_map_through_native_arithmetic_stays_native(self, da):
        # xarray arithmetic returns a DataArray, so the mapped record keeps a
        # native (new) leaf rather than a coerced array.
        nr = NumericRecord("nr", temps=da)
        doubled = nr.map(lambda x: x * 2)
        assert type(doubled["temps"]) is xr.DataArray
        np.testing.assert_allclose(np.asarray(doubled["temps"]), [2.0, 4.0, 6.0])

    def test_source_record_unchanged_after_transform(self, da):
        nr = NumericRecord("nr", temps=da, extra=jnp.array(1.0))
        nr.without("extra")
        nr.with_path_names(temps="warmth")
        assert nr["temps"] is da
        assert tuple(nr.children) == ("temps", "extra")


# ---------------------------------------------------------------------------
# The JAX boundary: conversion on flatten; native types do not cross
# ---------------------------------------------------------------------------


class TestJaxBoundary:
    def test_flatten_converts_to_jax(self, da):
        nr = NumericRecord("nr", temps=da, x=jnp.array(1.0))
        leaves, treedef = jax.tree_util.tree_flatten(nr)
        assert all(isinstance(leaf, jnp.ndarray) for leaf in leaves)
        back = jax.tree_util.tree_unflatten(treedef, leaves)
        assert type(back) is NumericRecord
        assert isinstance(back["temps"], jnp.ndarray)  # native type does not cross
        assert back.name == "nr"
        assert back.event_template == nr.event_template

    def test_tree_map_returns_bare_arrays(self, da):
        nr = NumericRecord("nr", temps=da)
        out = jax.tree_util.tree_map(lambda x: x * 2, nr)
        assert isinstance(out["temps"], jnp.ndarray)
        np.testing.assert_allclose(out["temps"], [2.0, 4.0, 6.0])

    def test_jit_over_native_leaf_record(self, da):
        nr = NumericRecord("nr", temps=da)

        @jax.jit
        def double(v):
            return jax.tree_util.tree_map(lambda x: x * 2, v)

        out = double(nr)
        assert type(out) is NumericRecord
        np.testing.assert_allclose(out["temps"], [2.0, 4.0, 6.0])


# ---------------------------------------------------------------------------
# Lazy conversion: no materialisation at construction; set-once cache
# ---------------------------------------------------------------------------


class _LazyLeaf:
    """A stand-in for a lazily-loaded array: metadata is free, values count.

    Exposes a numeric numpy ``dtype`` and a ``shape`` (the duck path sees
    it), and counts every materialisation through ``__array__`` — the test
    hook for "construction must not touch values".
    """

    def __init__(self, shape=(3,)):
        self.shape = shape
        self.dtype = np.dtype("float32")
        self.materialisations = 0

    def __array__(self, dtype=None, copy=None):
        self.materialisations += 1
        return np.ones(self.shape, dtype=self.dtype)


class TestLazyConversion:
    def test_construction_and_navigation_do_not_materialise(self):
        leaf = _LazyLeaf()
        nr = NumericRecord("nr", lazy=leaf, x=1.0)
        assert nr["lazy"] is leaf
        assert nr.event_template["lazy"] == ArraySpec((3,))
        assert nr.vector_size == 4
        assert leaf.materialisations == 0

    def test_transforms_do_not_materialise(self):
        leaf = _LazyLeaf()
        nr = NumericRecord("nr", lazy=leaf, x=1.0)
        edited = nr.without("x").with_path_names(lazy="late")
        assert edited["late"] is leaf
        assert leaf.materialisations == 0

    def test_promotion_and_inference_do_not_materialise(self):
        leaf = _LazyLeaf()
        r = Record("r", lazy=leaf)
        assert type(r) is NumericRecord
        assert isinstance(r.event_template, NumericEventTemplate)
        assert leaf.materialisations == 0

    def test_compute_boundary_materialises_exactly_once(self):
        leaf = _LazyLeaf()
        nr = NumericRecord("nr", lazy=leaf)
        v1 = nr.to_vector()
        v2 = nr.to_vector()
        np.testing.assert_allclose(v1, v2)
        assert leaf.materialisations == 1  # set-once cache

    def test_cache_shared_across_boundaries(self):
        leaf = _LazyLeaf()
        nr = NumericRecord("nr", lazy=leaf)
        nr.to_vector()
        jax.tree_util.tree_flatten(nr)  # flatten reuses the cached conversion
        assert leaf.materialisations == 1


# ---------------------------------------------------------------------------
# Pickling: native leaves round-trip themselves (no capture/restore)
# ---------------------------------------------------------------------------


class TestNativePickle:
    def test_pickle_preserves_native_types_at_every_level(self, da):
        import pickle

        inner = NumericRecord("grp", temps=da)
        outer = NumericRecord("outer", grp=inner, tbl=pd.DataFrame({"a": [1.0]}), x=1.0)
        back = pickle.loads(pickle.dumps(outer))
        assert type(back["grp/temps"]) is xr.DataArray
        assert back["grp/temps"].dims == ("t",)
        assert back["grp/temps"].attrs == {"units": "meters"}
        assert type(back["tbl"]) is pd.DataFrame
        assert back == outer


# ---------------------------------------------------------------------------
# The eager batch boundary converts through the registry
# ---------------------------------------------------------------------------


class TestEagerBatchBoundary:
    def test_stack_of_native_leaf_records_coerces_columns(self, da):
        from probpipe import RecordArray

        records = [NumericRecord("r", temps=da, x=float(i)) for i in range(3)]
        ra = RecordArray.stack(records)
        assert isinstance(ra["temps"], jnp.ndarray)
        assert ra["temps"].shape == (3, 3)
        assert ra.batch_shape == (3,)


# ---------------------------------------------------------------------------
# End-to-end: one registration lights up the whole surface
# ---------------------------------------------------------------------------


class _FakeTensor:
    """A torch-like container the duck path cannot see or convert.

    Its ``dtype`` is a non-numpy sentinel (so duck recognition fails) and it
    has no ``__array__`` (so ``jnp.asarray`` cannot convert it) — exactly the
    case the registry exists for.
    """

    class _Dtype:
        pass

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)
        self.dtype = self._Dtype()
        self.shape = self._values.shape
        self.to_jax_calls = 0


def _fake_backend() -> ArrayBackend:
    def _to_jax(t: _FakeTensor):
        t.to_jax_calls += 1
        return jnp.asarray(t._values)

    return ArrayBackend(
        event_shape=lambda t: tuple(t.shape),
        numpy_dtype=lambda t: np.dtype("float32"),
        to_jax=_to_jax,
        to_numpy=lambda t: t._values,
    )


class TestBackendRegistrationEndToEnd:
    def test_unregistered_fake_tensor_is_rejected(self):
        t = _FakeTensor([1.0, 2.0])
        assert EventTemplate.infer_from({"t": t})["t"] != ArraySpec((2,))
        with pytest.raises(TypeError, match="must be a numeric"):
            NumericRecord("nr", t=t)

    def test_one_registration_lights_up_everything(self, clean_registry):
        register_array_backend(_FakeTensor, _fake_backend())
        t = _FakeTensor([1.0, 2.0])

        # Recognition: template inference and spec validation.
        tpl = EventTemplate.infer_from({"t": t})
        assert isinstance(tpl, NumericEventTemplate)
        assert tpl["t"] == ArraySpec((2,))
        assert ArraySpec((2,)).is_valid(t)
        assert ArraySpec((2,), dtype=np.float32).is_valid(t)
        assert not ArraySpec((3,)).is_valid(t)

        # Promotion: an all-numeric record holding the tensor promotes.
        r = Record("r", t=t)
        assert type(r) is NumericRecord
        assert r["t"] is t

        # Conversion: the compute boundary routes through the custom to_jax.
        np.testing.assert_allclose(r.to_vector(), [1.0, 2.0])
        assert t.to_jax_calls == 1
        r.to_vector()
        assert t.to_jax_calls == 1  # set-once cache

        # Fingerprint: content-stable (not repr-based) via to_numpy.
        from probpipe.core._fingerprint import fingerprint

        same = fingerprint(_FakeTensor([1.0, 2.0]))
        assert fingerprint(t) == same
        assert fingerprint(_FakeTensor([9.0, 2.0])) != same

    def test_batch_stack_uses_registered_converter(self, clean_registry):
        from probpipe import RecordArray

        register_array_backend(_FakeTensor, _fake_backend())
        records = [Record("r", t=_FakeTensor([float(i), 2.0])) for i in range(3)]
        ra = RecordArray.stack(records)
        assert isinstance(ra["t"], jnp.ndarray)
        assert ra["t"].shape == (3, 2)
        assert all(r["t"].to_jax_calls == 1 for r in records)


class TestRegisteredBackendIdentity:
    """A registered non-numpy backend (a container the duck path cannot see or
    convert) must be first-class on the identity / conversion paths too — not
    just construction and to_vector. Regression for the review-round finding
    that __eq__ / to_numpy bypassed the registry.
    """

    @pytest.fixture
    def box_backend(self, clean_registry):
        class Box:
            def __init__(self, vals):
                self._v = np.asarray(vals, dtype=float)

        register_array_backend(
            Box,
            ArrayBackend(
                event_shape=lambda b: b._v.shape,
                numpy_dtype=lambda b: b._v.dtype,
                to_jax=lambda b: jnp.asarray(b._v),
                to_numpy=lambda b: b._v,
            ),
        )
        return Box

    def test_eq_routes_through_registry(self, box_backend):
        Box = box_backend
        a = NumericRecord("r", x=Box([1.0, 2.0]))
        b = NumericRecord("r", x=Box([1.0, 2.0]))
        assert (a.to_vector() == b.to_vector()).all()
        assert a == b  # was False when __eq__ used raw jnp.asarray
        assert a != NumericRecord("r", x=Box([1.0, 9.0]))

    def test_to_numpy_routes_through_registry(self, box_backend):
        Box = box_backend
        out = NumericRecord("r", x=Box([1.0, 2.0])).to_numpy()["x"]
        np.testing.assert_array_equal(out, [1.0, 2.0])  # not a 0-d object array
        assert out.dtype == np.float64

    def test_hash_does_not_materialise_registered_leaf(self, box_backend):
        # __hash__ is structural (shape + dtype from metadata); it must not
        # call the backend's to_jax/to_numpy.
        class Counting(box_backend):
            def __init__(self, vals):
                super().__init__(vals)
                self.converted = 0

            def __array__(self, dtype=None, copy=None):
                self.converted += 1
                return self._v

        leaf = Counting([1.0, 2.0])
        hash(NumericRecord("r", x=leaf))
        assert leaf.converted == 0


class TestNullableDtypeNotNumeric:
    """pandas nullable / masked dtypes report a numpy kind but are not
    np.dtype-coercible; a frame using them must stay a plain Record rather
    than promoting into a form jax cannot hold (which crashed on
    dtype / is_valid / to_vector). Regression for the review-round finding.
    """

    def test_nullable_frame_does_not_promote(self):
        df = pd.DataFrame({"a": pd.array([1, 2], dtype="Int64")})
        r = Record("r", m=df)
        assert type(r) is Record  # not NumericRecord
        assert not isinstance(r.event_template, NumericEventTemplate)

    def test_numpy_backed_frame_still_promotes(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        r = Record("r", m=df)
        assert type(r) is NumericRecord
        np.testing.assert_allclose(r.to_vector(), [1.0, 3.0, 2.0, 4.0])

    def test_is_numeric_dtype_rejects_extension_dtype(self):
        from probpipe.core.event_template import _is_numeric_dtype

        assert not _is_numeric_dtype(pd.Int64Dtype())
        assert _is_numeric_dtype(np.dtype("int64"))

    def test_is_numeric_dtype_returns_false_on_uninterpretable(self):
        # np.dtype(x) raises ValueError (not TypeError) for some objects — e.g.
        # a mock stand-in flowed through a numeric gate. The predicate must
        # return False, not propagate. (Regression for the stan-suite mock.)
        from unittest.mock import MagicMock

        from probpipe.core.event_template import _full_array_shape_or_none, _is_numeric_dtype

        assert _is_numeric_dtype(MagicMock()) is False
        assert _full_array_shape_or_none(MagicMock()) is None


class TestNativeMetadataInIdentity:
    """Option A: native-container metadata (coords / index) is part of a
    record's identity — __eq__ distinguishes it, matching fingerprint().
    """

    def test_eq_distinguishes_coords(self, da):
        other = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=["t"],
            coords={"t": [99, 98, 97]},  # different coords, same values
            attrs={"units": "meters"},
            name="temps",
        )
        r1 = NumericRecord("r", temps=da)
        r2 = NumericRecord("r", temps=other)
        assert (r1.to_vector() == r2.to_vector()).all()
        assert r1 != r2  # coords are identity-bearing

    def test_eq_equal_when_coords_match(self, da):
        r1 = NumericRecord("r", temps=da)
        r2 = NumericRecord("r", temps=da)
        assert r1 == r2
