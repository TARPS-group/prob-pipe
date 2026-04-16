"""Tests for RecordArray and NumericRecordArray."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import NumericRecord, NumericRecordArray, Record, RecordArray
from probpipe.core.record import RecordTemplate


# ---------------------------------------------------------------------------
# RecordArray construction
# ---------------------------------------------------------------------------


class TestRecordArrayConstruction:
    def test_basic(self):
        tpl = RecordTemplate(x=(3,), y=())
        ra = RecordArray(
            x=jnp.zeros((10, 3)),
            y=jnp.ones(10),
            batch_shape=(10,),
            template=tpl,
        )
        assert ra.fields == ("x", "y")
        assert ra.batch_shape == (10,)

    def test_multidim_batch(self):
        tpl = RecordTemplate(x=())
        ra = RecordArray(
            x=jnp.zeros((4, 5)),
            batch_shape=(4, 5),
            template=tpl,
        )
        assert ra.batch_shape == (4, 5)
        assert len(ra) == 20

    def test_dict_positional(self):
        tpl = RecordTemplate(a=(), b=(2,))
        ra = RecordArray(
            {"a": jnp.zeros(5), "b": jnp.ones((5, 2))},
            batch_shape=(5,),
            template=tpl,
        )
        assert ra.fields == ("a", "b")

    def test_field_mismatch_raises(self):
        tpl = RecordTemplate(x=(), y=())
        with pytest.raises(ValueError, match="do not match"):
            RecordArray(x=jnp.zeros(5), batch_shape=(5,), template=tpl)

    def test_empty_raises(self):
        tpl = RecordTemplate(x=())
        with pytest.raises(ValueError, match="at least one"):
            RecordArray(batch_shape=(5,), template=tpl)


# ---------------------------------------------------------------------------
# RecordArray field access
# ---------------------------------------------------------------------------


class TestRecordArrayAccess:
    @pytest.fixture
    def ra(self):
        tpl = RecordTemplate(x=(3,), y=())
        return RecordArray(
            x=jnp.arange(30).reshape(10, 3),
            y=jnp.arange(10.0),
            batch_shape=(10,),
            template=tpl,
        )

    def test_getitem_str(self, ra):
        assert ra["x"].shape == (10, 3)
        assert ra["y"].shape == (10,)

    def test_getitem_int(self, ra):
        r = ra[0]
        assert isinstance(r, Record)
        np.testing.assert_allclose(r["x"], [0, 1, 2])
        np.testing.assert_allclose(r["y"], 0.0)

    def test_getitem_int_last(self, ra):
        r = ra[9]
        np.testing.assert_allclose(r["x"], [27, 28, 29])
        np.testing.assert_allclose(r["y"], 9.0)

    def test_contains(self, ra):
        assert "x" in ra
        assert "z" not in ra

    def test_len(self, ra):
        assert len(ra) == 10

    def test_keys(self, ra):
        assert list(ra.keys()) == ["x", "y"]

    def test_missing_key_raises(self, ra):
        with pytest.raises(KeyError):
            ra["nonexistent"]


# ---------------------------------------------------------------------------
# RecordArray immutability
# ---------------------------------------------------------------------------


class TestRecordArrayImmutability:
    def test_setattr_raises(self):
        tpl = RecordTemplate(x=())
        ra = RecordArray(x=jnp.zeros(5), batch_shape=(5,), template=tpl)
        with pytest.raises(AttributeError, match="immutable"):
            ra.x = jnp.ones(5)


# ---------------------------------------------------------------------------
# RecordArray.stack
# ---------------------------------------------------------------------------


class TestRecordArrayStack:
    def test_stack_records(self):
        records = [Record(a=float(i), b=jnp.array([i, i + 1])) for i in range(5)]
        ra = RecordArray.stack(records)
        assert ra.batch_shape == (5,)
        assert ra["a"].shape == (5,)
        assert ra["b"].shape == (5, 2)

    def test_stack_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            RecordArray.stack([])

    def test_stack_with_template(self):
        tpl = RecordTemplate(a=(), b=(2,))
        records = [Record(a=1.0, b=jnp.zeros(2)) for _ in range(3)]
        ra = RecordArray.stack(records, template=tpl)
        assert ra.template == tpl
        assert ra.batch_shape == (3,)


# ---------------------------------------------------------------------------
# NumericRecordArray construction
# ---------------------------------------------------------------------------


class TestNumericRecordArrayConstruction:
    def test_basic(self):
        tpl = RecordTemplate(x=(3,), y=())
        nra = NumericRecordArray(
            x=jnp.zeros((10, 3)),
            y=jnp.ones(10),
            batch_shape=(10,),
            template=tpl,
        )
        assert isinstance(nra, RecordArray)
        assert nra.fields == ("x", "y")


# ---------------------------------------------------------------------------
# NumericRecordArray flatten / unflatten
# ---------------------------------------------------------------------------


class TestNumericRecordArrayFlatten:
    def test_flatten_shape(self):
        tpl = RecordTemplate(x=(3,), y=())
        nra = NumericRecordArray(
            x=jnp.zeros((10, 3)),
            y=jnp.ones(10),
            batch_shape=(10,),
            template=tpl,
        )
        flat = nra.flatten()
        assert flat.shape == (10, 4)  # 3 + 1

    def test_flatten_multidim_batch(self):
        tpl = RecordTemplate(a=(2,), b=())
        nra = NumericRecordArray(
            a=jnp.zeros((4, 5, 2)),
            b=jnp.ones((4, 5)),
            batch_shape=(4, 5),
            template=tpl,
        )
        flat = nra.flatten()
        assert flat.shape == (4, 5, 3)  # 2 + 1

    def test_flatten_values(self):
        tpl = RecordTemplate(a=(), b=(2,))
        nra = NumericRecordArray(
            a=jnp.array([10.0, 20.0]),
            b=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            batch_shape=(2,),
            template=tpl,
        )
        flat = nra.flatten()
        # a comes first (sorted), then b
        np.testing.assert_allclose(flat[0], [10.0, 1.0, 2.0])
        np.testing.assert_allclose(flat[1], [20.0, 3.0, 4.0])

    def test_unflatten_roundtrip(self):
        tpl = RecordTemplate(x=(3,), y=())
        nra = NumericRecordArray(
            x=jnp.arange(30.0).reshape(10, 3),
            y=jnp.arange(10.0) * 100,
            batch_shape=(10,),
            template=tpl,
        )
        flat = nra.flatten()
        nra2 = NumericRecordArray.unflatten(flat, template=tpl)
        np.testing.assert_allclose(nra2["x"], nra["x"])
        np.testing.assert_allclose(nra2["y"], nra["y"])

    def test_unflatten_infers_batch(self):
        tpl = RecordTemplate(a=(), b=(2,))
        flat = jnp.zeros((8, 3))
        nra = NumericRecordArray.unflatten(flat, template=tpl)
        assert nra.batch_shape == (8,)
        assert nra["a"].shape == (8,)
        assert nra["b"].shape == (8, 2)

    def test_unflatten_explicit_batch(self):
        tpl = RecordTemplate(a=(), b=(2,))
        flat = jnp.zeros((4, 5, 3))
        nra = NumericRecordArray.unflatten(
            flat, template=tpl, batch_shape=(4, 5)
        )
        assert nra.batch_shape == (4, 5)
        assert nra["a"].shape == (4, 5)
        assert nra["b"].shape == (4, 5, 2)

    def test_unflatten_scalar_fields(self):
        tpl = RecordTemplate(a=(), b=(), c=())
        flat = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        nra = NumericRecordArray.unflatten(flat, template=tpl)
        np.testing.assert_allclose(nra["a"], [1.0, 4.0])
        np.testing.assert_allclose(nra["b"], [2.0, 5.0])
        np.testing.assert_allclose(nra["c"], [3.0, 6.0])

    def test_unflatten_opaque_raises(self):
        tpl = RecordTemplate(label=None, x=())
        with pytest.raises(TypeError, match="opaque"):
            NumericRecordArray.unflatten(jnp.zeros((5, 1)), template=tpl)


# ---------------------------------------------------------------------------
# NumericRecordArray mean / var
# ---------------------------------------------------------------------------


class TestNumericRecordArrayReductions:
    @pytest.fixture
    def nra(self):
        tpl = RecordTemplate(x=(3,), y=())
        return NumericRecordArray(
            x=jnp.arange(30.0).reshape(10, 3),
            y=jnp.arange(10.0),
            batch_shape=(10,),
            template=tpl,
        )

    def test_mean_returns_numeric_record(self, nra):
        m = nra.mean(axis=0)
        assert isinstance(m, NumericRecord)
        assert m["x"].shape == (3,)
        assert m["y"].shape == ()

    def test_mean_values(self, nra):
        m = nra.mean(axis=0)
        np.testing.assert_allclose(m["y"], 4.5)
        np.testing.assert_allclose(m["x"], [13.5, 14.5, 15.5])

    def test_var_returns_numeric_record(self, nra):
        v = nra.var(axis=0)
        assert isinstance(v, NumericRecord)

    def test_mean_multidim_batch(self):
        tpl = RecordTemplate(x=())
        nra = NumericRecordArray(
            x=jnp.arange(12.0).reshape(3, 4),
            batch_shape=(3, 4),
            template=tpl,
        )
        m = nra.mean(axis=0)
        assert isinstance(m, NumericRecordArray)
        assert m.batch_shape == (4,)
        np.testing.assert_allclose(m["x"], [4.0, 5.0, 6.0, 7.0])

    def test_mean_then_mean_collapses(self):
        tpl = RecordTemplate(x=())
        nra = NumericRecordArray(
            x=jnp.arange(12.0).reshape(3, 4),
            batch_shape=(3, 4),
            template=tpl,
        )
        m = nra.mean(axis=0).mean(axis=0)
        assert isinstance(m, NumericRecord)
        np.testing.assert_allclose(m["x"], 5.5)


# ---------------------------------------------------------------------------
# JAX PyTree
# ---------------------------------------------------------------------------


class TestRecordArrayPyTree:
    def test_tree_map(self):
        tpl = RecordTemplate(x=(2,), y=())
        nra = NumericRecordArray(
            x=jnp.ones((5, 2)),
            y=jnp.ones(5),
            batch_shape=(5,),
            template=tpl,
        )
        nra2 = jax.tree.map(lambda a: a * 3, nra)
        assert isinstance(nra2, NumericRecordArray)
        np.testing.assert_allclose(nra2["x"], 3.0 * jnp.ones((5, 2)))

    def test_tree_leaves(self):
        tpl = RecordTemplate(a=(), b=(3,))
        nra = NumericRecordArray(
            a=jnp.zeros(5),
            b=jnp.zeros((5, 3)),
            batch_shape=(5,),
            template=tpl,
        )
        leaves = jax.tree.leaves(nra)
        assert len(leaves) == 2

    def test_roundtrip(self):
        tpl = RecordTemplate(x=())
        nra = NumericRecordArray(
            x=jnp.arange(5.0),
            batch_shape=(5,),
            template=tpl,
        )
        leaves, treedef = jax.tree.flatten(nra)
        nra2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(nra2, NumericRecordArray)
        assert nra2.batch_shape == (5,)
        np.testing.assert_allclose(nra2["x"], nra["x"])

    def test_jit(self):
        tpl = RecordTemplate(x=(2,))
        nra = NumericRecordArray(
            x=jnp.ones((5, 2)),
            batch_shape=(5,),
            template=tpl,
        )

        @jax.jit
        def f(arr):
            return jnp.sum(arr["x"])

        result = f(nra)
        np.testing.assert_allclose(result, 10.0)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_record_array_repr(self):
        tpl = RecordTemplate(x=(3,))
        ra = RecordArray(
            x=jnp.zeros((5, 3)), batch_shape=(5,), template=tpl
        )
        r = repr(ra)
        assert "RecordArray" in r
        assert "batch_shape=(5,)" in r

    def test_numeric_record_array_repr(self):
        tpl = RecordTemplate(x=())
        nra = NumericRecordArray(
            x=jnp.zeros(5), batch_shape=(5,), template=tpl
        )
        r = repr(nra)
        assert "NumericRecordArray" in r
