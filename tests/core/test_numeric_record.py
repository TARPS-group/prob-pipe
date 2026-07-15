"""Tests for probpipe.core._numeric_record.NumericRecord."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import NumericRecord, Record
from probpipe.core.event_template import EventTemplate

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_scalars(self):
        nr = NumericRecord("nr", a=1.0, b=2.0)
        assert nr.fields == ("a", "b")

    def test_arrays(self):
        nr = NumericRecord("nr", x=jnp.zeros(3), y=np.ones(2))
        assert nr.fields == ("x", "y")

    def test_nested(self):
        inner = NumericRecord("nr", x=1.0, y=2.0)
        outer = NumericRecord("nr", params=inner, z=3.0)
        assert isinstance(outer.at_path("params"), NumericRecord)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord("nr", label="hello")

    def test_bool_accepted(self):
        """Python ``bool`` is a numeric leaf (JAX treats it as dtype bool)."""
        import jax.numpy as jnp

        v = NumericRecord("nr", flag=True)
        assert v["flag"].dtype == jnp.bool_
        # ``bool`` array fields also work.
        import numpy as np

        v2 = NumericRecord("nr", mask=np.array([True, False, True]))
        assert v2["mask"].dtype == jnp.bool_

    def test_bytes_rejected(self):
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord("nr", label=b"binary")

    def test_list_of_strings_rejected(self):
        """Lists must contain numbers, not strings."""
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord("nr", tags=["a", "b"])

    def test_none_rejected(self):
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord("nr", missing=None)

    def test_nested_record_not_numeric_raises(self):
        """A nested non-numeric Record in NumericRecord raises. (An
        all-numeric inner Record promotes to NumericRecord and nests fine.)"""
        inner = Record("r", x="tag")
        with pytest.raises(TypeError, match="NumericRecord"):
            NumericRecord("nr", params=inner, z=2.0)

    def test_dict_positional(self):
        nr = NumericRecord("nr", {"a": 1.0, "b": jnp.zeros(3)})
        assert nr.fields == ("a", "b")

    def test_is_record(self):
        nr = NumericRecord("nr", x=1.0)
        assert isinstance(nr, Record)

    def test_coerces_python_scalar_to_jax_array(self):
        """Every leaf is a jnp.ndarray after construction (uniform type)."""
        nr = NumericRecord("nr", a=1.0, b=2)
        assert isinstance(nr["a"], jnp.ndarray)
        assert isinstance(nr["b"], jnp.ndarray)

    def test_numpy_stored_native_converted_at_boundary(self):
        arr = np.array([1.0, 2.0])
        nr = NumericRecord("nr", x=arr)
        # Native storage: navigation returns the numpy leaf verbatim; the
        # compute boundary (to_vector) converts to jax.
        assert nr["x"] is arr
        vec = nr.to_vector()
        assert isinstance(vec, jnp.ndarray)
        np.testing.assert_allclose(vec, [1.0, 2.0])

    def test_jax_array_passthrough(self):
        """An existing jnp.ndarray is stored without conversion or copy."""
        arr = jnp.array([1.0, 2.0])
        nr = NumericRecord("nr", x=arr)
        assert nr["x"] is arr

    def test_xarray_accepted_and_stored_native(self):
        """xarray.DataArray wraps numeric data and is accepted **verbatim**:
        the leaf keeps its dims / coords, and conversion to ``jnp.ndarray``
        happens only at the compute boundary."""
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["time"],
            coords={"time": [10, 20, 30]},
        )
        nr = NumericRecord("nr", y=da)
        assert nr["y"] is da
        assert nr["y"].dims == ("time",)
        np.testing.assert_allclose(nr.to_vector(), [1.0, 2.0, 3.0])

    # Regression: previous ``_is_numeric_leaf`` short-circuited True on
    # ``isinstance(val, np.ndarray)`` without checking ``val.dtype.kind``,
    # so a numpy object-dtype array slipped past our validation and
    # blew up later inside JAX with a cryptic "Dtype object is not a
    # valid JAX array type" error. The check should happen on *every*
    # array-like leaf, not just "unknown" ones.

    def test_rejects_numpy_object_dtype_array_of_strings(self):
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord("nr", x=np.array(["a", "b"], dtype=object))

    def test_rejects_numpy_object_dtype_array_of_dicts(self):
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord("nr", x=np.array([{"a": 1}, {"b": 2}], dtype=object))

    def test_rejects_numpy_string_dtype_array(self):
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord("nr", x=np.array(["a", "b"]))  # dtype='<U1'

    def test_rejects_jax_error_never_surfaces(self):
        """The TypeError must come from NumericRecord validation (clear
        message naming the field), not from ``jnp.asarray`` downstream.
        Otherwise users get the JAX-internal 'Dtype object is not a
        valid JAX array type' and can't tell which field is to blame."""
        with pytest.raises(TypeError, match="'x'"):
            NumericRecord("nr", x=np.array(["a", "b"], dtype=object))

    def test_accepts_all_numeric_array_dtypes(self):
        """The other side of the regression: float / int / uint /
        complex / bool arrays all pass validation."""
        dtypes = [np.float32, np.float64, np.int32, np.uint8, np.complex64, np.bool_]
        for dt in dtypes:
            arr = np.array([1, 2, 3]).astype(dt)
            nr = NumericRecord("nr", x=arr)
            # Stored verbatim (native form); the boundary converts on demand.
            assert nr["x"] is arr, f"failed for dtype {dt}"

    def test_numeric_dtype_predicate_shared(self):
        """Every numeric gate must agree on what counts as numeric by consuming
        a shared predicate rather than duplicating the logic. Two levels are
        shared: the dtype-level ``_is_numeric_dtype`` (used directly where only
        a dtype is in hand — ``NumericRecordArray``, the broadcast-template
        builder, the ``Design`` marginals probe) and the leaf-level
        ``_is_numeric_leaf`` (the registry-first resolver that wraps it,
        consumed by ``NumericRecord`` and template inference)."""
        from probpipe.core import (
            _array_backend,
            _broadcast_distributions,
            _numeric_record,
            _record_array,
            event_template,
        )
        from probpipe.record import design

        # dtype-level predicate (lives in _array_backend): imported directly from
        # there wherever only a dtype is in hand
        assert _record_array._is_numeric_dtype is _array_backend._is_numeric_dtype
        assert _broadcast_distributions._is_numeric_dtype is _array_backend._is_numeric_dtype
        assert design._is_numeric_dtype is _array_backend._is_numeric_dtype
        # leaf-level predicate: one resolver shared by the record gate and inference
        assert _numeric_record._is_numeric_leaf is _array_backend._is_numeric_leaf
        assert event_template._is_numeric_leaf is _array_backend._is_numeric_leaf

    def test_bfloat16_leaf_accepted(self):
        # ml_dtypes numerics (kind "V") are numeric leaves.
        nr = NumericRecord("nr", x=jnp.ones(3, dtype=jnp.bfloat16))
        assert nr["x"].dtype == jnp.bfloat16

    def test_bfloat16_vector_round_trip(self):
        nr = NumericRecord(
            "nr", x=jnp.ones(3, dtype=jnp.bfloat16), y=jnp.zeros((), dtype=jnp.bfloat16)
        )
        tpl = nr.event_template
        vec = nr.to_vector()
        assert vec.shape == (4,)
        assert vec.dtype == jnp.bfloat16
        assert NumericRecord.from_vector("nr", tpl, vec) == nr

    def test_dtype_pinned_vector_round_trip(self):
        # A dtype-pinned (int32) field must round-trip: to_vector promotes to a
        # common float across the mixed-dtype fields, and from_vector casts each
        # block back to its declared dtype. Before the cast + skeleton fix, the
        # int32 template made from_vector raise on the float32 placeholder.
        from probpipe.core.event_template import ArraySpec, EventTemplate

        tpl = EventTemplate(
            k=ArraySpec(shape=(3,), dtype=jnp.int32), x=ArraySpec(shape=(2,), dtype=jnp.float32)
        )
        nr = NumericRecord(
            "nr", k=jnp.array([1, 2, 3], dtype=jnp.int32), x=jnp.zeros(2), event_template=tpl
        )
        back = NumericRecord.from_vector("nr", nr.event_template, nr.to_vector())
        assert back["k"].dtype == jnp.int32
        assert list(np.asarray(back["k"])) == [1, 2, 3]
        assert back == nr


# ---------------------------------------------------------------------------
# to_vector / from_vector (numeric 1-D serialization)
# ---------------------------------------------------------------------------


class TestToVectorFromVector:
    def test_flatten_scalars(self):
        nr = NumericRecord("nr", a=1.0, b=2.0, c=3.0)
        flat = nr.to_vector()
        assert flat.shape == (3,)
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])

    def test_flatten_arrays(self):
        nr = NumericRecord("nr", x=jnp.array([1.0, 2.0]), y=jnp.array([3.0]))
        flat = nr.to_vector()
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])

    def test_flatten_nested(self):
        inner = NumericRecord("nr", x=1.0, y=2.0)
        outer = NumericRecord("nr", a=inner, b=3.0)
        flat = outer.to_vector()
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])

    def test_unflatten_with_event_template(self):
        tpl = EventTemplate(a=(), b=(3,))
        flat = jnp.array([1.0, 2.0, 3.0, 4.0])
        nr = NumericRecord.from_vector("nr", tpl, flat)
        assert isinstance(nr, NumericRecord)
        np.testing.assert_allclose(nr["a"], 1.0)
        np.testing.assert_allclose(nr["b"], [2.0, 3.0, 4.0])

    def test_roundtrip_with_template(self):
        tpl = EventTemplate(r=(), K=(), phi=())
        nr = NumericRecord("nr", r=1.8, K=70.0, phi=10.0)
        flat = nr.to_vector()
        nr2 = NumericRecord.from_vector("nr2", tpl, flat)
        np.testing.assert_allclose(float(nr2["r"]), 1.8)
        np.testing.assert_allclose(float(nr2["K"]), 70.0)
        np.testing.assert_allclose(float(nr2["phi"]), 10.0)

    def test_roundtrip_nested_template(self):
        from probpipe.core.event_template import NumericEventTemplate

        inner_tpl = NumericEventTemplate(x=(), y=(2,))
        outer_tpl = NumericEventTemplate(params=inner_tpl, z=(3,))
        flat = jnp.arange(6.0)  # x=0, y=[1,2], z=[3,4,5]
        nr = NumericRecord.from_vector("nr", outer_tpl, flat)
        assert isinstance(nr.at_path("params"), NumericRecord)
        np.testing.assert_allclose(nr["params/x"], 0.0)
        np.testing.assert_allclose(nr["params/y"], [1.0, 2.0])
        np.testing.assert_allclose(nr["z"], [3.0, 4.0, 5.0])

    def test_vector_size(self):
        nr = NumericRecord("nr", a=1.0, b=jnp.zeros(4), c=jnp.zeros((2, 3)))
        assert nr.vector_size == 11


# ---------------------------------------------------------------------------
# Record -> NumericRecord conversion (to_numeric)
# ---------------------------------------------------------------------------


class TestToNumeric:
    def test_simple(self):
        r = Record("r", a=1.0, b=jnp.zeros(3))
        nr = r.to_numeric()
        assert isinstance(nr, NumericRecord)
        assert nr.fields == ("a", "b")

    def test_nested(self):
        inner = Record("r", x=1.0, y=2.0)
        outer = Record("r", params=inner, z=3.0)
        nr = outer.to_numeric()
        assert isinstance(nr, NumericRecord)
        assert isinstance(nr.at_path("params"), NumericRecord)


# ---------------------------------------------------------------------------
# JAX PyTree
# ---------------------------------------------------------------------------


class TestPyTree:
    def test_tree_map(self):
        nr = NumericRecord("nr", a=1.0, b=2.0)
        nr2 = jax.tree.map(lambda x: x * 2, nr)
        assert isinstance(nr2, NumericRecord)
        np.testing.assert_allclose(float(nr2["a"]), 2.0)
        np.testing.assert_allclose(float(nr2["b"]), 4.0)

    def test_tree_leaves(self):
        nr = NumericRecord("nr", a=jnp.array(1.0), b=jnp.array(2.0))
        leaves = jax.tree.leaves(nr)
        assert len(leaves) == 2

    def test_roundtrip(self):
        # NumericRecord uses its own pytree registration (separate from the
        # base Record's), so pin that a flatten/unflatten round-trip
        # preserves the leaf values, the template, and the full identity
        # pair (name + name_is_auto) — not just the field names.
        nr = NumericRecord("nr", x=jnp.array([1.0, 2.0]), y=jnp.array(3.0), name_is_auto=True)
        leaves, treedef = jax.tree.flatten(nr)
        nr2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(nr2, NumericRecord)
        assert nr2.fields == nr.fields
        assert nr2 == nr  # structural equality: template + field values
        np.testing.assert_allclose(np.asarray(nr2["x"]), [1.0, 2.0])
        assert nr2.name == "nr"
        assert nr2.name_is_auto is True

    def test_jit(self):
        nr = NumericRecord("nr", a=1.0, b=2.0)

        @jax.jit
        def f(vals):
            return vals["a"] + vals["b"]

        result = f(nr)
        np.testing.assert_allclose(float(result), 3.0)

    def test_grad(self):
        nr = NumericRecord("nr", x=1.0)

        def f(vals):
            return vals["x"] ** 2

        grads = jax.grad(f)(nr)
        assert isinstance(grads, NumericRecord)
        np.testing.assert_allclose(float(grads["x"]), 2.0)


# ---------------------------------------------------------------------------
# Single-field scalar-like coercion (issue #130 PR 1.5)
# ---------------------------------------------------------------------------


class TestSingleFieldCoercion:
    """A single-field NumericRecord behaves like a thin wrapper around
    its one numeric value for coercion purposes. Multi-field records
    raise a clear error — ``record["field"]`` is the explicit way.

    The shim is what keeps idiomatic expressions like
    ``float(mean(dist))`` working once ``mean`` returns a
    ``NumericRecord("nr", result=...)`` under the full output-type contract.
    """

    def test_float_scalar(self):
        nr = NumericRecord("nr", result=3.14)
        np.testing.assert_allclose(float(nr), 3.14, rtol=1e-5)

    def test_int_scalar(self):
        nr = NumericRecord("nr", result=3.14)
        assert int(nr) == 3

    def test_bool_scalar(self):
        assert bool(NumericRecord("nr", result=1.0)) is True
        assert bool(NumericRecord("nr", result=0.0)) is False

    def test_np_asarray_scalar(self):
        nr = NumericRecord("nr", result=2.5)
        arr = np.asarray(nr)
        assert arr.shape == ()
        assert float(arr) == 2.5

    def test_np_asarray_vector(self):
        nr = NumericRecord("nr", result=jnp.array([1.0, 2.0, 3.0]))
        arr = np.asarray(nr)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_np_asarray_with_dtype(self):
        nr = NumericRecord("nr", result=3.14)
        arr = np.asarray(nr, dtype=np.float64)
        assert arr.dtype == np.float64
        np.testing.assert_allclose(arr, 3.14)

    def test_jnp_asarray_preserves_jax_type(self):
        nr = NumericRecord("nr", result=jnp.array([1.0, 2.0]))
        arr = jnp.asarray(nr)
        assert isinstance(arr, jnp.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0])

    # Multi-field raises — silently unwrapping one of many fields is
    # ambiguous. Users say what they want with explicit indexing.

    def test_multi_field_float_raises(self):
        nr = NumericRecord("nr", a=1.0, b=2.0)
        with pytest.raises(TypeError, match="2 fields"):
            float(nr)

    def test_multi_field_int_raises(self):
        with pytest.raises(TypeError, match="not scalar-like"):
            int(NumericRecord("nr", a=1.0, b=2.0))

    def test_multi_field_asarray_raises(self):
        with pytest.raises(TypeError, match="not scalar-like"):
            np.asarray(NumericRecord("nr", a=1.0, b=2.0))

    def test_nested_numeric_record_raises(self):
        inner = NumericRecord("nr", x=1.0, y=2.0)
        outer = NumericRecord("nr", inner=inner)
        with pytest.raises(TypeError, match="nested NumericRecord"):
            float(outer)
