"""Tests for probpipe.core._numeric_record.NumericRecord."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import NumericRecord, Record
from probpipe.core.record import RecordTemplate


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_scalars(self):
        nr = NumericRecord(a=1.0, b=2.0)
        assert nr.fields == ("a", "b")

    def test_arrays(self):
        nr = NumericRecord(x=jnp.zeros(3), y=np.ones(2))
        assert nr.fields == ("x", "y")

    def test_nested(self):
        inner = NumericRecord(x=1.0, y=2.0)
        outer = NumericRecord(params=inner, z=3.0)
        assert isinstance(outer["params"], NumericRecord)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord(label="hello")

    def test_bool_accepted(self):
        """Python ``bool`` is a numeric leaf (JAX treats it as dtype bool)."""
        import jax.numpy as jnp
        v = NumericRecord(flag=True)
        assert v["flag"].dtype == jnp.bool_
        # ``bool`` array fields also work.
        import numpy as np
        v2 = NumericRecord(mask=np.array([True, False, True]))
        assert v2["mask"].dtype == jnp.bool_

    def test_bytes_rejected(self):
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord(label=b"binary")

    def test_list_of_strings_rejected(self):
        """Lists must contain numbers, not strings."""
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord(tags=["a", "b"])

    def test_none_rejected(self):
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord(missing=None)

    def test_nested_record_not_numeric_raises(self):
        """Nested plain Record in NumericRecord raises."""
        inner = Record(x=1.0)
        with pytest.raises(TypeError, match="NumericRecord"):
            NumericRecord(params=inner, z=2.0)

    def test_dict_positional(self):
        nr = NumericRecord({"a": 1.0, "b": jnp.zeros(3)})
        assert nr.fields == ("a", "b")

    def test_is_record(self):
        nr = NumericRecord(x=1.0)
        assert isinstance(nr, Record)

    def test_coerces_python_scalar_to_jax_array(self):
        """Every leaf is a jnp.ndarray after construction (uniform type)."""
        nr = NumericRecord(a=1.0, b=2)
        assert isinstance(nr["a"], jnp.ndarray)
        assert isinstance(nr["b"], jnp.ndarray)

    def test_coerces_numpy_to_jax_array(self):
        nr = NumericRecord(x=np.array([1.0, 2.0]))
        assert isinstance(nr["x"], jnp.ndarray)

    def test_jax_array_passthrough(self):
        """An existing jnp.ndarray is stored without conversion or copy."""
        arr = jnp.array([1.0, 2.0])
        nr = NumericRecord(x=arr)
        assert nr["x"] is arr

    def test_xarray_accepted_and_coerced(self):
        """xarray.DataArray wraps numeric data and is accepted, but it is
        coerced to ``jnp.ndarray`` at construction — labels / coords are
        dropped. Use a plain ``Record`` if you need to preserve xarray
        metadata."""
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            [1.0, 2.0, 3.0], dims=["time"], coords={"time": [10, 20, 30]},
        )
        nr = NumericRecord(y=da)
        assert isinstance(nr["y"], jnp.ndarray)
        np.testing.assert_allclose(nr["y"], [1.0, 2.0, 3.0])

    # Regression: previous ``_is_numeric_leaf`` short-circuited True on
    # ``isinstance(val, np.ndarray)`` without checking ``val.dtype.kind``,
    # so a numpy object-dtype array slipped past our validation and
    # blew up later inside JAX with a cryptic "Dtype object is not a
    # valid JAX array type" error. The check should happen on *every*
    # array-like leaf, not just "unknown" ones.

    def test_rejects_numpy_object_dtype_array_of_strings(self):
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord(x=np.array(["a", "b"], dtype=object))

    def test_rejects_numpy_object_dtype_array_of_dicts(self):
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord(x=np.array([{"a": 1}, {"b": 2}], dtype=object))

    def test_rejects_numpy_string_dtype_array(self):
        with pytest.raises(TypeError, match="numeric"):
            NumericRecord(x=np.array(["a", "b"]))  # dtype='<U1'

    def test_rejects_jax_error_never_surfaces(self):
        """The TypeError must come from NumericRecord validation (clear
        message naming the field), not from ``jnp.asarray`` downstream.
        Otherwise users get the JAX-internal 'Dtype object is not a
        valid JAX array type' and can't tell which field is to blame."""
        with pytest.raises(TypeError, match="'x'"):
            NumericRecord(x=np.array(["a", "b"], dtype=object))

    def test_accepts_all_numeric_array_dtypes(self):
        """The other side of the regression: float / int / uint /
        complex / bool arrays all pass validation."""
        dtypes = [np.float32, np.float64, np.int32, np.uint8, np.complex64, np.bool_]
        for dt in dtypes:
            arr = np.array([1, 2, 3]).astype(dt)
            nr = NumericRecord(x=arr)
            assert isinstance(nr["x"], jnp.ndarray), f"failed for dtype {dt}"

    def test_numeric_dtype_kinds_shared_constant(self):
        """``NumericRecord`` and ``NumericRecordArray`` must agree on
        which dtype kinds are numeric. Duplicated literals would let
        the two validation sites drift silently — this test pins down
        that they consume the same frozenset."""
        from probpipe.core._numeric_record import _NUMERIC_DTYPE_KINDS
        assert _NUMERIC_DTYPE_KINDS == frozenset("biufc")
        # Import path is also the import path used by
        # NumericRecordArray._validate_fields (see _record_array.py).
        from probpipe.core import _record_array
        assert _record_array._NUMERIC_DTYPE_KINDS is _NUMERIC_DTYPE_KINDS


# ---------------------------------------------------------------------------
# Flatten / unflatten
# ---------------------------------------------------------------------------


class TestFlattenUnflatten:
    def test_flatten_scalars(self):
        nr = NumericRecord(a=1.0, b=2.0, c=3.0)
        flat = nr.flatten()
        assert flat.shape == (3,)
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])

    def test_flatten_arrays(self):
        nr = NumericRecord(x=jnp.array([1.0, 2.0]), y=jnp.array([3.0]))
        flat = nr.flatten()
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])

    def test_flatten_nested(self):
        inner = NumericRecord(x=1.0, y=2.0)
        outer = NumericRecord(a=inner, b=3.0)
        flat = outer.flatten()
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])

    def test_unflatten_with_record_template(self):
        tpl = RecordTemplate(a=(), b=(3,))
        flat = jnp.array([1.0, 2.0, 3.0, 4.0])
        nr = NumericRecord.unflatten(flat, template=tpl)
        assert isinstance(nr, NumericRecord)
        np.testing.assert_allclose(nr["a"], 1.0)
        np.testing.assert_allclose(nr["b"], [2.0, 3.0, 4.0])

    def test_roundtrip_with_template(self):
        tpl = RecordTemplate(r=(), K=(), phi=())
        nr = NumericRecord(r=1.8, K=70.0, phi=10.0)
        flat = nr.flatten()
        nr2 = NumericRecord.unflatten(flat, template=tpl)
        np.testing.assert_allclose(float(nr2["r"]), 1.8)
        np.testing.assert_allclose(float(nr2["K"]), 70.0)
        np.testing.assert_allclose(float(nr2["phi"]), 10.0)

    def test_roundtrip_nested_template(self):
        from probpipe.core.record import NumericRecordTemplate
        inner_tpl = NumericRecordTemplate(x=(), y=(2,))
        outer_tpl = NumericRecordTemplate(params=inner_tpl, z=(3,))
        flat = jnp.arange(6.0)  # x=0, y=[1,2], z=[3,4,5]
        nr = NumericRecord.unflatten(flat, template=outer_tpl)
        assert isinstance(nr["params"], NumericRecord)
        np.testing.assert_allclose(nr["params"]["x"], 0.0)
        np.testing.assert_allclose(nr["params"]["y"], [1.0, 2.0])
        np.testing.assert_allclose(nr["z"], [3.0, 4.0, 5.0])

    def test_unflatten_opaque_raises(self):
        tpl = RecordTemplate(label=None, x=())
        with pytest.raises(TypeError, match="opaque"):
            NumericRecord.unflatten(jnp.array([1.0]), template=tpl)

    def test_flat_size(self):
        nr = NumericRecord(a=1.0, b=jnp.zeros(4), c=jnp.zeros((2, 3)))
        assert nr.flat_size == 11


# ---------------------------------------------------------------------------
# from_record conversion
# ---------------------------------------------------------------------------


class TestFromRecord:
    def test_simple(self):
        r = Record(a=1.0, b=jnp.zeros(3))
        nr = NumericRecord.from_record(r)
        assert isinstance(nr, NumericRecord)
        assert nr.fields == ("a", "b")

    def test_nested(self):
        inner = Record(x=1.0, y=2.0)
        outer = Record(params=inner, z=3.0)
        nr = NumericRecord.from_record(outer)
        assert isinstance(nr, NumericRecord)
        assert isinstance(nr["params"], NumericRecord)


# ---------------------------------------------------------------------------
# JAX PyTree
# ---------------------------------------------------------------------------


class TestPyTree:
    def test_tree_map(self):
        nr = NumericRecord(a=1.0, b=2.0)
        nr2 = jax.tree.map(lambda x: x * 2, nr)
        assert isinstance(nr2, NumericRecord)
        np.testing.assert_allclose(float(nr2["a"]), 2.0)
        np.testing.assert_allclose(float(nr2["b"]), 4.0)

    def test_tree_leaves(self):
        nr = NumericRecord(a=jnp.array(1.0), b=jnp.array(2.0))
        leaves = jax.tree.leaves(nr)
        assert len(leaves) == 2

    def test_roundtrip(self):
        nr = NumericRecord(x=jnp.array([1.0, 2.0]), y=jnp.array(3.0))
        leaves, treedef = jax.tree.flatten(nr)
        nr2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(nr2, NumericRecord)
        assert nr2.fields == nr.fields

    def test_jit(self):
        nr = NumericRecord(a=1.0, b=2.0)

        @jax.jit
        def f(vals):
            return vals["a"] + vals["b"]

        result = f(nr)
        np.testing.assert_allclose(float(result), 3.0)

    def test_grad(self):
        nr = NumericRecord(x=1.0)

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
    ``NumericRecord(result=...)`` under the full output-type contract.
    """

    def test_float_scalar(self):
        nr = NumericRecord(result=3.14)
        np.testing.assert_allclose(float(nr), 3.14, rtol=1e-5)

    def test_int_scalar(self):
        nr = NumericRecord(result=3.14)
        assert int(nr) == 3

    def test_bool_scalar(self):
        assert bool(NumericRecord(result=1.0)) is True
        assert bool(NumericRecord(result=0.0)) is False

    def test_np_asarray_scalar(self):
        nr = NumericRecord(result=2.5)
        arr = np.asarray(nr)
        assert arr.shape == ()
        assert float(arr) == 2.5

    def test_np_asarray_vector(self):
        nr = NumericRecord(result=jnp.array([1.0, 2.0, 3.0]))
        arr = np.asarray(nr)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_np_asarray_with_dtype(self):
        nr = NumericRecord(result=3.14)
        arr = np.asarray(nr, dtype=np.float64)
        assert arr.dtype == np.float64
        np.testing.assert_allclose(arr, 3.14)

    def test_jnp_asarray_preserves_jax_type(self):
        nr = NumericRecord(result=jnp.array([1.0, 2.0]))
        arr = jnp.asarray(nr)
        assert isinstance(arr, jnp.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0])

    # Multi-field raises — silently unwrapping one of many fields is
    # ambiguous. Users say what they want with explicit indexing.

    def test_multi_field_float_raises(self):
        nr = NumericRecord(a=1.0, b=2.0)
        with pytest.raises(TypeError, match="2 fields"):
            float(nr)

    def test_multi_field_int_raises(self):
        with pytest.raises(TypeError, match="not scalar-like"):
            int(NumericRecord(a=1.0, b=2.0))

    def test_multi_field_asarray_raises(self):
        with pytest.raises(TypeError, match="not scalar-like"):
            np.asarray(NumericRecord(a=1.0, b=2.0))

    def test_nested_numeric_record_raises(self):
        inner = NumericRecord(x=1.0, y=2.0)
        outer = NumericRecord(inner=inner)
        with pytest.raises(TypeError, match="nested NumericRecord"):
            float(outer)
