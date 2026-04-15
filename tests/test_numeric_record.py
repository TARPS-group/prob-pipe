"""Tests for probpipe.core.record.NumericRecord and ArrayBackend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import NumericRecord, Record
from probpipe.core._array_backend import (
    ArrayBackend,
    JaxBackend,
    NumpyBackend,
    detect_backend,
)
from probpipe.core.record import RecordTemplate


# ---------------------------------------------------------------------------
# ArrayBackend
# ---------------------------------------------------------------------------


class TestArrayBackend:
    def test_numpy_backend_is_protocol_compliant(self):
        assert isinstance(NumpyBackend(), ArrayBackend)

    def test_jax_backend_is_protocol_compliant(self):
        assert isinstance(JaxBackend(), ArrayBackend)

    def test_detect_numpy(self):
        backend = detect_backend(np.array(1.0))
        assert isinstance(backend, NumpyBackend)

    def test_detect_jax(self):
        backend = detect_backend(jnp.array(1.0))
        assert isinstance(backend, JaxBackend)

    def test_detect_scalar_defaults_to_jax(self):
        backend = detect_backend(1.0)
        assert isinstance(backend, JaxBackend)

    def test_numpy_operations(self):
        b = NumpyBackend()
        a = np.array([1.0, 2.0, 3.0])
        assert b.ravel(a).shape == (3,)
        assert b.reshape(a, (3, 1)).shape == (3, 1)
        assert b.zeros((2, 3)).shape == (2, 3)
        np.testing.assert_allclose(b.mean(a, axis=0), 2.0)
        stacked = b.stack([a, a], axis=0)
        assert stacked.shape == (2, 3)
        catted = b.concatenate([a, a], axis=0)
        assert catted.shape == (6,)

    def test_jax_operations(self):
        b = JaxBackend()
        a = jnp.array([1.0, 2.0, 3.0])
        assert b.ravel(a).shape == (3,)
        assert b.reshape(a, (3, 1)).shape == (3, 1)
        assert b.zeros((2, 3)).shape == (2, 3)
        np.testing.assert_allclose(b.mean(a, axis=0), 2.0)


# ---------------------------------------------------------------------------
# NumericRecord construction
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


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


class TestBackend:
    def test_jax_arrays(self):
        nr = NumericRecord(x=jnp.array(1.0))
        assert isinstance(nr.backend, JaxBackend)

    def test_numpy_arrays(self):
        nr = NumericRecord(x=np.array(1.0))
        assert isinstance(nr.backend, NumpyBackend)

    def test_scalar_defaults_to_jax(self):
        nr = NumericRecord(x=1.0)
        assert isinstance(nr.backend, JaxBackend)


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
        np.testing.assert_allclose(nr.a, 1.0)
        np.testing.assert_allclose(nr.b, [2.0, 3.0, 4.0])

    def test_unflatten_with_record(self):
        template = NumericRecord(a=1.0, b=jnp.zeros(3))
        flat = jnp.array([10.0, 20.0, 30.0, 40.0])
        nr = NumericRecord.unflatten(flat, template=template)
        assert isinstance(nr, NumericRecord)
        np.testing.assert_allclose(nr.a, 10.0)
        np.testing.assert_allclose(nr.b, [20.0, 30.0, 40.0])

    def test_roundtrip(self):
        nr = NumericRecord(r=1.8, K=70.0, phi=10.0)
        flat = nr.flatten()
        nr2 = NumericRecord.unflatten(flat, template=nr)
        np.testing.assert_allclose(float(nr2.r), 1.8)
        np.testing.assert_allclose(float(nr2.K), 70.0)
        np.testing.assert_allclose(float(nr2.phi), 10.0)

    def test_roundtrip_nested_template(self):
        inner_tpl = RecordTemplate(x=(), y=(2,))
        outer_tpl = RecordTemplate(params=inner_tpl, z=(3,))
        flat = jnp.arange(6.0)  # x=0, y=[1,2], z=[3,4,5]
        nr = NumericRecord.unflatten(flat, template=outer_tpl)
        assert isinstance(nr["params"], NumericRecord)
        np.testing.assert_allclose(nr.params.x, 0.0)
        np.testing.assert_allclose(nr.params.y, [1.0, 2.0])
        np.testing.assert_allclose(nr.z, [3.0, 4.0, 5.0])

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
        np.testing.assert_allclose(float(nr2.a), 2.0)
        np.testing.assert_allclose(float(nr2.b), 4.0)

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
            return vals.a + vals.b

        result = f(nr)
        np.testing.assert_allclose(float(result), 3.0)

    def test_grad(self):
        nr = NumericRecord(x=1.0)

        def f(vals):
            return vals.x ** 2

        grads = jax.grad(f)(nr)
        assert isinstance(grads, NumericRecord)
        np.testing.assert_allclose(float(grads.x), 2.0)
