"""Tests for RecordArray and NumericRecordArray."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Normal,
    NumericRecord,
    NumericRecordArray,
    Provenance,
    Record,
    RecordArray,
    provenance_ancestors,
)
from probpipe.core.event_template import EventTemplate

# ---------------------------------------------------------------------------
# RecordArray construction
# ---------------------------------------------------------------------------


class TestRecordArrayConstruction:
    def test_basic(self):
        tpl = EventTemplate(x=(3,), y=())
        ra = RecordArray(
            x=jnp.zeros((10, 3)),
            y=jnp.ones(10),
            batch_shape=(10,),
            template=tpl,
        )
        assert ra.fields == ("x", "y")
        assert ra.batch_shape == (10,)

    def test_multidim_batch(self):
        tpl = EventTemplate(x=())
        ra = RecordArray(
            x=jnp.zeros((4, 5)),
            batch_shape=(4, 5),
            template=tpl,
        )
        assert ra.batch_shape == (4, 5)
        assert len(ra) == 1  # field count, not batch size

    def test_dict_positional(self):
        tpl = EventTemplate(a=(), b=(2,))
        ra = RecordArray(
            {"a": jnp.zeros(5), "b": jnp.ones((5, 2))},
            batch_shape=(5,),
            template=tpl,
        )
        assert ra.fields == ("a", "b")

    def test_field_mismatch_raises(self):
        tpl = EventTemplate(x=(), y=())
        with pytest.raises(ValueError, match="do not match"):
            RecordArray(x=jnp.zeros(5), batch_shape=(5,), template=tpl)

    def test_empty_raises(self):
        tpl = EventTemplate(x=())
        with pytest.raises(ValueError, match="at least one"):
            RecordArray(batch_shape=(5,), template=tpl)

    def test_zero_length_batch(self):
        """batch_shape=(0,) is a valid edge case (no samples)."""
        tpl = EventTemplate(x=())
        ra = RecordArray(x=jnp.zeros(0), batch_shape=(0,), template=tpl)
        assert ra.batch_shape == (0,)
        assert len(ra) == 1  # field count
        assert ra["x"].shape == (0,)

    def test_flatten_preserves_nan(self):
        """flatten() must not silently replace NaN (numerical stability)."""
        tpl = EventTemplate(a=(), b=(2,))
        nra = NumericRecordArray(
            a=jnp.array([jnp.nan, 1.0]),
            b=jnp.array([[jnp.inf, -jnp.inf], [0.0, jnp.nan]]),
            batch_shape=(2,),
            template=tpl,
        )
        flat = nra.to_vector()
        # flat[0] = [nan, inf, -inf]; flat[1] = [1.0, 0.0, nan]
        row0 = np.asarray(flat[0])
        row1 = np.asarray(flat[1])
        assert np.isnan(row0[0])
        assert np.isposinf(row0[1])
        assert np.isneginf(row0[2])
        assert np.isnan(row1[2])


# ---------------------------------------------------------------------------
# RecordArray field access
# ---------------------------------------------------------------------------


class TestRecordArrayAccess:
    @pytest.fixture
    def ra(self):
        tpl = EventTemplate(x=(3,), y=())
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

    def test_getitem_int_on_recordarray_returns_record(self):
        """The base ``RecordArray`` materialises elements as ``Record``."""
        tpl = EventTemplate(x=(3,))
        ra = RecordArray(
            x=jnp.arange(30.0).reshape(10, 3),
            batch_shape=(10,),
            template=tpl,
        )
        assert type(ra[0]) is Record

    def test_getitem_int_on_numeric_recordarray_returns_numeric_record(self):
        """``NumericRecordArray[int]`` must return a ``NumericRecord`` so
        the numeric invariant survives slicing (fix for PR #123 review
        comment #4)."""
        from probpipe import NumericRecord

        tpl = EventTemplate(x=(3,), y=())
        nra = NumericRecordArray(
            x=jnp.arange(30.0).reshape(10, 3),
            y=jnp.arange(10.0),
            batch_shape=(10,),
            template=tpl,
        )
        elem = nra[0]
        assert type(elem) is NumericRecord
        assert isinstance(elem["x"], jnp.ndarray)
        assert isinstance(elem["y"], jnp.ndarray)

    def test_contains(self, ra):
        assert "x" in ra
        assert "z" not in ra

    def test_len(self, ra):
        # ``len`` reports field count; for batch size use ``batch_shape``.
        assert len(ra) == 2

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
        tpl = EventTemplate(x=())
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
        tpl = EventTemplate(a=(), b=(2,))
        records = [Record(a=1.0, b=jnp.zeros(2)) for _ in range(3)]
        ra = RecordArray.stack(records, template=tpl)
        assert ra.template == tpl
        assert ra.batch_shape == (3,)


# ---------------------------------------------------------------------------
# NumericRecordArray construction
# ---------------------------------------------------------------------------


class TestNumericRecordArrayConstruction:
    def test_basic(self):
        tpl = EventTemplate(x=(3,), y=())
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
        tpl = EventTemplate(x=(3,), y=())
        nra = NumericRecordArray(
            x=jnp.zeros((10, 3)),
            y=jnp.ones(10),
            batch_shape=(10,),
            template=tpl,
        )
        flat = nra.to_vector()
        assert flat.shape == (10, 4)  # 3 + 1

    def test_flatten_multidim_batch(self):
        tpl = EventTemplate(a=(2,), b=())
        nra = NumericRecordArray(
            a=jnp.zeros((4, 5, 2)),
            b=jnp.ones((4, 5)),
            batch_shape=(4, 5),
            template=tpl,
        )
        flat = nra.to_vector()
        assert flat.shape == (4, 5, 3)  # 2 + 1

    def test_flatten_values(self):
        tpl = EventTemplate(a=(), b=(2,))
        nra = NumericRecordArray(
            a=jnp.array([10.0, 20.0]),
            b=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            batch_shape=(2,),
            template=tpl,
        )
        flat = nra.to_vector()
        # Template insertion order: a first, then b.
        np.testing.assert_allclose(flat[0], [10.0, 1.0, 2.0])
        np.testing.assert_allclose(flat[1], [20.0, 3.0, 4.0])

    def test_unflatten_roundtrip(self):
        tpl = EventTemplate(x=(3,), y=())
        nra = NumericRecordArray(
            x=jnp.arange(30.0).reshape(10, 3),
            y=jnp.arange(10.0) * 100,
            batch_shape=(10,),
            template=tpl,
        )
        flat = nra.to_vector()
        nra2 = tpl.from_vector(flat)
        np.testing.assert_allclose(nra2["x"], nra["x"])
        np.testing.assert_allclose(nra2["y"], nra["y"])

    def test_unflatten_infers_batch(self):
        tpl = EventTemplate(a=(), b=(2,))
        flat = jnp.zeros((8, 3))
        nra = tpl.from_vector(flat)
        assert nra.batch_shape == (8,)
        assert nra["a"].shape == (8,)
        assert nra["b"].shape == (8, 2)

    def test_unflatten_explicit_batch(self):
        tpl = EventTemplate(a=(), b=(2,))
        flat = jnp.zeros((4, 5, 3))
        nra = tpl.from_vector(flat)
        assert nra.batch_shape == (4, 5)
        assert nra["a"].shape == (4, 5)
        assert nra["b"].shape == (4, 5, 2)

    def test_unflatten_scalar_fields(self):
        tpl = EventTemplate(a=(), b=(), c=())
        flat = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        nra = tpl.from_vector(flat)
        np.testing.assert_allclose(nra["a"], [1.0, 4.0])
        np.testing.assert_allclose(nra["b"], [2.0, 5.0])
        np.testing.assert_allclose(nra["c"], [3.0, 6.0])


# ---------------------------------------------------------------------------
# Nested-record flatten / slash indexing (issue #262)
# ---------------------------------------------------------------------------


class TestNumericRecordArrayNested:
    """flatten() recurses into nested record fields, in depth-first leaf order,
    and round-trips through unflatten; slash paths index leaves."""

    @staticmethod
    def _nested_tpl():
        return EventTemplate(outer=EventTemplate(a=(), b=()), m=())

    def test_flatten_nested_record_array_field(self):
        # The unflatten / canonical-_sample shape: the nested field is itself a
        # NumericRecordArray (exercises the RecordArray branch).
        tpl = self._nested_tpl()
        flat = jnp.arange(15.0).reshape(5, 3)  # columns = outer/a, outer/b, m
        nra = tpl.from_vector(flat)
        assert isinstance(nra.at_path("outer"), NumericRecordArray)
        np.testing.assert_allclose(nra.to_vector(), flat)  # round-trips exactly

    def test_flatten_nested_record_field(self):
        # The pre-canonical shape: the nested field is a plain Record holding
        # batch-shaped leaves (exercises the Record branch).
        tpl = self._nested_tpl()
        inner = Record(a=jnp.arange(5.0), b=jnp.arange(5.0) + 10)
        nra = NumericRecordArray(
            {"outer": inner, "m": jnp.arange(5.0) + 100},
            batch_shape=(5,),
            template=tpl,
        )
        flat = nra.to_vector()
        assert flat.shape == (5, 3)
        np.testing.assert_allclose(flat[:, 0], inner["a"])
        np.testing.assert_allclose(flat[:, 1], inner["b"])
        np.testing.assert_allclose(flat[:, 2], nra["m"])
        # Integer-indexing descends the plain-Record nested field (the Record
        # branch of _get_record), returning a nested record element.
        elem = nra[2]
        np.testing.assert_allclose(elem["outer/a"], inner["a"][2])
        np.testing.assert_allclose(elem["outer/b"], inner["b"][2])

    def test_flatten_nested_depth2_roundtrip(self):
        tpl = EventTemplate(outer=EventTemplate(deep=EventTemplate(g=(), h=()), a=()), m=())
        flat = jnp.arange(20.0).reshape(5, 4)  # outer/deep/g, outer/deep/h, outer/a, m
        nra = tpl.from_vector(flat)
        np.testing.assert_allclose(nra.to_vector(), flat)
        np.testing.assert_allclose(nra["outer/deep/g"], flat[:, 0])

    def test_flatten_nested_vector_leaf(self):
        # A non-scalar (vector) leaf under nesting: flatten lays out its columns
        # at the leaf's event size, in canonical leaf order, and round-trips.
        tpl = EventTemplate(outer=EventTemplate(a=(2,), b=()), m=())
        flat = jnp.arange(20.0).reshape(5, 4)  # outer/a (2), outer/b (1), m (1)
        nra = tpl.from_vector(flat)
        assert np.asarray(nra["outer/a"]).shape == (5, 2)
        np.testing.assert_allclose(nra["outer/a"], flat[:, 0:2])
        np.testing.assert_allclose(nra["outer/b"], flat[:, 2])
        np.testing.assert_allclose(nra["m"], flat[:, 3])
        np.testing.assert_allclose(nra.to_vector(), flat)

    def test_getitem_slash_path(self):
        tpl = self._nested_tpl()
        nra = tpl.from_vector(jnp.arange(15.0).reshape(5, 3))
        np.testing.assert_allclose(nra["outer/a"], nra.at_path("outer")["a"])
        with pytest.raises(KeyError):
            nra["outer/missing"]  # leaf missing inside the sub-record
        with pytest.raises(KeyError):
            nra["nope/a"]  # head field not in the store
        with pytest.raises(KeyError):
            nra["m/x"]  # slash descends into a (scalar) leaf

    def test_getitem_int_nested_element(self):
        # Integer indexing of a nested record array descends into the nested
        # field, returning a (nested) record element — not an indexing error.
        tpl = self._nested_tpl()
        nra = tpl.from_vector(jnp.arange(15.0).reshape(5, 3))
        elem = nra[2]
        assert isinstance(elem, Record)
        np.testing.assert_allclose(elem["outer/a"], nra.at_path("outer")["a"][2])
        np.testing.assert_allclose(elem.to_vector(), nra.to_vector()[2])

    def test_getitem_int_nested_multidim_batch(self):
        tpl = self._nested_tpl()
        nra = tpl.from_vector(jnp.arange(24.0).reshape(2, 4, 3))
        elem = nra[5]  # flat index into the (2, 4) batch
        np.testing.assert_allclose(
            elem["outer/a"], np.asarray(nra.at_path("outer")["a"]).reshape(-1)[5]
        )


# ---------------------------------------------------------------------------
# NumericRecordArray mean / var
# ---------------------------------------------------------------------------


class TestNumericRecordArrayReductions:
    @pytest.fixture
    def nra(self):
        tpl = EventTemplate(x=(3,), y=())
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
        tpl = EventTemplate(x=())
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
        tpl = EventTemplate(x=())
        nra = NumericRecordArray(
            x=jnp.arange(12.0).reshape(3, 4),
            batch_shape=(3, 4),
            template=tpl,
        )
        m = nra.mean(axis=0).mean(axis=0)
        assert isinstance(m, NumericRecord)
        np.testing.assert_allclose(m["x"], 5.5)

    def test_mean_nested_records(self):
        tpl = EventTemplate(outer=EventTemplate(a=(), b=(2,)), m=())
        flat = jnp.asarray(
            [
                [0.0, 0.0, 1.0, 10.0],
                [1.0, 1.0, 2.0, 11.0],
                [2.0, 2.0, 3.0, 12.0],
            ]
        )
        nra = tpl.from_vector(flat)
        m = nra.mean(axis=0)
        assert isinstance(m, NumericRecord)
        assert isinstance(m.at_path("outer"), NumericRecord)
        np.testing.assert_allclose(m["outer/a"], 1.0)
        np.testing.assert_allclose(m["outer/b"], [1.0, 2.0])
        np.testing.assert_allclose(m["m"], 11.0)

    def test_var_nested_records(self):
        tpl = EventTemplate(outer=EventTemplate(a=(), b=(2,)), m=())
        flat = jnp.asarray(
            [
                [0.0, 0.0, 1.0, 10.0],
                [1.0, 1.0, 2.0, 11.0],
                [2.0, 2.0, 3.0, 12.0],
            ]
        )
        nra = tpl.from_vector(flat)
        v = nra.var(axis=0)
        assert isinstance(v, NumericRecord)
        assert isinstance(v.at_path("outer"), NumericRecord)
        np.testing.assert_allclose(v["outer/a"], 2.0 / 3.0)
        np.testing.assert_allclose(v["outer/b"], [2.0 / 3.0, 2.0 / 3.0])
        np.testing.assert_allclose(v["m"], 2.0 / 3.0)

    def test_reduce_plain_nested_record_field(self):
        tpl = EventTemplate(outer=EventTemplate(a=(), b=(2,)), m=())
        inner = Record(
            a=jnp.asarray([0.0, 1.0, 2.0]),
            b=jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        )
        nra = NumericRecordArray(
            {"outer": inner, "m": jnp.asarray([10.0, 11.0, 12.0])},
            batch_shape=(3,),
            template=tpl,
        )

        m = nra.mean(axis=0)
        assert isinstance(m, NumericRecord)
        assert isinstance(m.at_path("outer"), NumericRecord)
        np.testing.assert_allclose(m["outer/a"], 1.0)
        np.testing.assert_allclose(m["outer/b"], [1.0, 2.0])
        np.testing.assert_allclose(m["m"], 11.0)

        v = nra.var(axis=0)
        assert isinstance(v, NumericRecord)
        assert isinstance(v.at_path("outer"), NumericRecord)
        np.testing.assert_allclose(v["outer/a"], 2.0 / 3.0)
        np.testing.assert_allclose(v["outer/b"], [2.0 / 3.0, 2.0 / 3.0])
        np.testing.assert_allclose(v["m"], 2.0 / 3.0)

    def test_mean_nested_records_multidim_batch(self):
        tpl = EventTemplate(outer=EventTemplate(a=()), m=())
        flat = jnp.arange(12.0).reshape(2, 3, tpl.vector_size)
        nra = tpl.from_vector(flat)
        m = nra.mean(axis=0)
        assert isinstance(m, NumericRecordArray)
        assert isinstance(m.at_path("outer"), NumericRecordArray)
        assert m.batch_shape == (3,)
        assert m.at_path("outer").batch_shape == (3,)
        np.testing.assert_allclose(m["outer/a"], [3.0, 5.0, 7.0])
        np.testing.assert_allclose(m["m"], [4.0, 6.0, 8.0])

    def test_reduce_plain_nested_record_field_multidim_batch(self):
        tpl = EventTemplate(outer=EventTemplate(a=()), m=())
        inner = Record(a=jnp.arange(6.0).reshape(2, 3))
        nra = NumericRecordArray(
            {"outer": inner, "m": jnp.arange(6.0).reshape(2, 3) + 10.0},
            batch_shape=(2, 3),
            template=tpl,
        )

        m = nra.mean(axis=0)
        assert isinstance(m, NumericRecordArray)
        assert isinstance(m.at_path("outer"), NumericRecordArray)
        assert m.batch_shape == (3,)
        assert m.at_path("outer").batch_shape == (3,)
        np.testing.assert_allclose(m["outer/a"], [1.5, 2.5, 3.5])
        np.testing.assert_allclose(m["m"], [11.5, 12.5, 13.5])


# ---------------------------------------------------------------------------
# JAX PyTree
# ---------------------------------------------------------------------------


class TestRecordArrayPyTree:
    def test_tree_map(self):
        tpl = EventTemplate(x=(2,), y=())
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
        tpl = EventTemplate(a=(), b=(3,))
        nra = NumericRecordArray(
            a=jnp.zeros(5),
            b=jnp.zeros((5, 3)),
            batch_shape=(5,),
            template=tpl,
        )
        leaves = jax.tree.leaves(nra)
        assert len(leaves) == 2

    def test_roundtrip(self):
        tpl = EventTemplate(x=())
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
        tpl = EventTemplate(x=(2,))
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
        tpl = EventTemplate(x=(3,))
        ra = RecordArray(x=jnp.zeros((5, 3)), batch_shape=(5,), template=tpl)
        r = repr(ra)
        assert "RecordArray" in r
        assert "batch_shape=(5,)" in r

    def test_numeric_record_array_repr(self):
        tpl = EventTemplate(x=())
        nra = NumericRecordArray(x=jnp.zeros(5), batch_shape=(5,), template=tpl)
        r = repr(nra)
        assert "NumericRecordArray" in r


# ---------------------------------------------------------------------------
# Equality / hashability
# ---------------------------------------------------------------------------


class TestEquality:
    def test_equal_same_contents(self):
        tpl = EventTemplate(x=())
        a = RecordArray(x=jnp.arange(3.0), batch_shape=(3,), template=tpl)
        b = RecordArray(x=jnp.arange(3.0), batch_shape=(3,), template=tpl)
        assert a == b

    def test_unequal_values(self):
        tpl = EventTemplate(x=())
        a = RecordArray(x=jnp.arange(3.0), batch_shape=(3,), template=tpl)
        b = RecordArray(x=jnp.arange(3.0) + 1, batch_shape=(3,), template=tpl)
        assert a != b

    def test_unequal_batch_shape(self):
        tpl = EventTemplate(x=())
        a = RecordArray(x=jnp.zeros(3), batch_shape=(3,), template=tpl)
        b = RecordArray(x=jnp.zeros((1, 3)), batch_shape=(1, 3), template=tpl)
        assert a != b

    def test_unequal_template(self):
        a = RecordArray(x=jnp.zeros(3), batch_shape=(3,), template=EventTemplate(x=()))
        b = RecordArray(x=jnp.zeros((3, 2)), batch_shape=(3,), template=EventTemplate(x=(2,)))
        assert a != b

    def test_recordarray_not_equal_to_numeric_recordarray(self):
        """Type-strict equality: different concrete classes are not equal
        even when contents match."""
        tpl = EventTemplate(x=())
        ra = RecordArray(x=jnp.zeros(3), batch_shape=(3,), template=tpl)
        nra = NumericRecordArray(x=jnp.zeros(3), batch_shape=(3,), template=tpl)
        assert ra != nra

    def test_self_equality_with_nan(self):
        """Regression for PR #123 review comment (f-ii): ``a == a`` must
        hold even when leaves contain NaN. Without the identity fast
        path ``jnp.array_equal`` treats NaN != NaN and self-equality
        returns False."""
        tpl = EventTemplate(x=())
        ra = RecordArray(
            x=jnp.array([jnp.nan, 1.0, jnp.nan]),
            batch_shape=(3,),
            template=tpl,
        )
        assert ra == ra


class TestUnhashable:
    """``RecordArray`` is intentionally unhashable (mirrors NumPy).

    Rationale: ``__eq__`` compares leaves elementwise, so any
    consistent ``__hash__`` would have to materialise every byte (and
    crash inside ``jit`` / ``vmap``). Users who need a structural key
    build one explicitly from ``(type(ra), ra.batch_shape, ra.fields,
    ra.template)``.
    """

    def test_recordarray_unhashable(self):
        tpl = EventTemplate(x=())
        ra = RecordArray(x=jnp.zeros(3), batch_shape=(3,), template=tpl)
        with pytest.raises(TypeError, match="unhashable"):
            hash(ra)

    def test_numeric_recordarray_unhashable(self):
        tpl = EventTemplate(x=())
        nra = NumericRecordArray(x=jnp.zeros(3), batch_shape=(3,), template=tpl)
        with pytest.raises(TypeError, match="unhashable"):
            hash(nra)

    def test_cannot_be_used_as_dict_key(self):
        tpl = EventTemplate(x=())
        ra = RecordArray(x=jnp.zeros(3), batch_shape=(3,), template=tpl)
        with pytest.raises(TypeError, match="unhashable"):
            _ = {ra: 1}


# ---------------------------------------------------------------------------
# Numeric-leaf validation on NumericRecordArray
# ---------------------------------------------------------------------------


class TestNumericRecordArrayValidation:
    """Regression for PR #123 review comment (c) on a8be0b3.

    Before the fix, ``NumericRecordArray.__init__`` inherited
    ``RecordArray.__init__`` unchanged and did no leaf-type check, so
    non-numeric leaves (from direct construction or from
    ``jax.tree.map`` returning strings) produced silently-corrupt
    instances that only failed downstream.
    """

    def test_rejects_list_leaf(self):
        tpl = EventTemplate(x=(2,))
        with pytest.raises(TypeError, match="numeric"):
            NumericRecordArray(
                {"x": ["a", "b"]},
                batch_shape=(),
                template=tpl,
            )

    def test_rejects_object_dtype_array(self):
        tpl = EventTemplate(x=())
        arr = np.array(["a", "b"], dtype=object)
        with pytest.raises(TypeError, match="numeric"):
            NumericRecordArray({"x": arr}, batch_shape=(2,), template=tpl)

    def test_rejects_wrong_shape(self):
        """Leaf shape must match ``(*batch_shape, *event_shape)``."""
        tpl = EventTemplate(x=(3,))
        with pytest.raises(ValueError, match="shape"):
            NumericRecordArray(
                {"x": jnp.zeros((5, 2))},  # expected (5, 3)
                batch_shape=(5,),
                template=tpl,
            )

    def test_tree_map_to_string_rejected(self):
        """``jax.tree.map(lambda x: "hi", nra)`` must raise at
        unflatten time rather than produce a corrupt NumericRecordArray."""
        tpl = EventTemplate(x=())
        nra = NumericRecordArray(
            x=jnp.arange(3.0),
            batch_shape=(3,),
            template=tpl,
        )
        with pytest.raises(TypeError, match="numeric"):
            jax.tree.map(lambda x: "hi", nra)

    def test_accepts_numpy_leaf_and_coerces_to_jnp(self):
        tpl = EventTemplate(x=())
        nra = NumericRecordArray(
            x=np.arange(3.0),
            batch_shape=(3,),
            template=tpl,
        )
        assert isinstance(nra["x"], jnp.ndarray)

    def test_rejects_numpy_object_dtype_batch(self):
        """Parallel regression for the _is_numeric_leaf object-dtype fix:
        NumericRecordArray._validate_fields shares the same
        _NUMERIC_DTYPE_KINDS set, so an object-dtype batched field
        must be rejected up front (not left to blow up inside JAX)."""
        tpl = EventTemplate(x=())
        with pytest.raises(TypeError, match="non-numeric dtype"):
            NumericRecordArray(
                {"x": np.array([{"a": 1}, {"b": 2}], dtype=object)},
                batch_shape=(2,),
                template=tpl,
            )

    def test_rejects_numpy_string_dtype_batch(self):
        tpl = EventTemplate(x=())
        with pytest.raises(TypeError, match="non-numeric dtype"):
            NumericRecordArray(
                {"x": np.array(["a", "b"])},  # dtype='<U1'
                batch_shape=(2,),
                template=tpl,
            )

    def test_nested_template_field_allowed(self):
        """Fields whose template spec is a nested ``EventTemplate`` are
        allowed to hold a Record / RecordArray leaf without triggering
        the numeric-dtype check. This is how nested ProductDistributions
        materialise samples."""
        from probpipe import Record

        inner_tpl = EventTemplate(x=(), y=())
        outer_tpl = EventTemplate(physics=inner_tpl, obs=(3,))
        inner = Record(x=1.0, y=2.0)
        nra = NumericRecordArray(
            {"physics": inner, "obs": jnp.zeros(3)},
            batch_shape=(),
            template=outer_tpl,
        )
        assert nra.at_path("physics") is inner


# ---------------------------------------------------------------------------
# Provenance (issue #130)
# ---------------------------------------------------------------------------


class TestProvenance:
    """RecordArray carries the same ``.provenance`` / ``.with_provenance`` slot
    as Record and Distribution so sweep outputs can record which
    parameters / distributions produced them.
    """

    @pytest.fixture
    def ra(self):
        return NumericRecordArray.stack([NumericRecord(x=float(i), y=2.0 * i) for i in range(3)])

    def test_initial_provenance_is_none(self, ra):
        assert ra.provenance is None

    def test_with_provenance_sets_and_returns_self(self, ra):
        out = ra.with_provenance(Provenance("sweep", parents=()))
        assert out is ra
        assert ra.provenance.operation == "sweep"

    def test_with_provenance_is_write_once(self, ra):
        ra.with_provenance(Provenance("first", parents=()))
        with pytest.raises(RuntimeError, match="write-once"):
            ra.with_provenance(Provenance("second", parents=()))

    def test_eq_ignores_provenance(self, ra):
        ra2 = NumericRecordArray.stack([NumericRecord(x=float(i), y=2.0 * i) for i in range(3)])
        ra.with_provenance(Provenance("a", parents=()))
        ra2.with_provenance(Provenance("b", parents=()))
        assert ra == ra2

    def test_pytree_roundtrip_drops_provenance(self, ra):
        ra.with_provenance(Provenance("sweep", parents=()))
        leaves, treedef = jax.tree_util.tree_flatten(ra)
        ra2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert ra2.provenance is None
        assert ra2 == ra

    def test_numeric_record_array_inherits_slot(self, ra):
        # ``ra`` is a NumericRecordArray (subclass of RecordArray) —
        # verify the slot is inherited without redeclaration.
        assert isinstance(ra, NumericRecordArray)
        assert hasattr(ra, "provenance")

    # Integration: provenance_ancestors walks RecordArray → parent
    # distributions.

    def test_provenance_ancestors_through_distribution(self, ra):
        prior = Normal(loc=0.0, scale=1.0, name="prior")
        ra.with_provenance(Provenance("sweep", parents=(prior,)))
        ancestors = provenance_ancestors(ra)
        assert len(ancestors) == 1
        assert ancestors[0] is prior


# ---------------------------------------------------------------------------
# Single-field array-like coercion (issue #130 PR 1.5)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Hierarchy — RecordArray IS-A Record (issue #130 symmetry fix)
# ---------------------------------------------------------------------------


class TestHierarchy:
    """``RecordArray`` inherits from ``Record`` so that the Record family
    mirrors Distribution / DistributionArray's relationship. Users who
    want to discriminate batched vs. scalar write
    ``isinstance(x, RecordArray)`` or
    ``isinstance(x, Record) and not isinstance(x, RecordArray)``.
    """

    def test_recordarray_is_record(self):
        """Base ``RecordArray`` — permissive leaf storage, no numeric
        validation — must still ``isinstance`` as a ``Record``."""
        tpl = EventTemplate(label=None, value=())
        ra = RecordArray(
            {"label": np.asarray(["a", "b", "c"], dtype=object), "value": jnp.arange(3.0)},
            batch_shape=(3,),
            template=tpl,
        )
        assert isinstance(ra, Record)
        assert not isinstance(ra, NumericRecordArray)

    def test_numericrecordarray_is_record(self):
        """The ``NumericRecordArray`` subclass inherits through the
        chain NumericRecordArray → RecordArray → Record."""
        ra = NumericRecordArray.stack([NumericRecord(x=float(i)) for i in range(3)])
        assert isinstance(ra, NumericRecordArray)
        assert isinstance(ra, RecordArray)
        assert isinstance(ra, Record)

    def test_numericrecord_is_not_recordarray(self):
        """Guard against the inverse mistake: NumericRecord is still
        a scalar Record, not a RecordArray."""
        nr = NumericRecord(x=1.0)
        assert isinstance(nr, Record)
        assert not isinstance(nr, RecordArray)

    def test_mro_order(self):
        """Subclass chain is NumericRecordArray → RecordArray → Record.
        The linear MRO keeps the provenance / name / hash plumbing on
        Record and lets RecordArray add batch-shape / template
        specialisation."""
        mro = [c.__name__ for c in NumericRecordArray.mro()]
        # Record must appear before object, after RecordArray.
        assert mro.index("RecordArray") < mro.index("Record")
        assert mro.index("Record") < mro.index("object")

    def test_provenance_slot_inherited_from_record(self):
        ra = NumericRecordArray.stack([NumericRecord(x=float(i)) for i in range(2)])
        # Record defines the ``_provenance`` slot; RecordArray should
        # *not* redeclare it (that would raise a layout conflict on
        # construction). Confirm the attribute works end-to-end.
        assert ra.provenance is None
        ra.with_provenance(Provenance("test", parents=()))
        assert ra.provenance.operation == "test"

    def test_name_slot_inherited_from_record(self):
        """RecordArray uses Record's ``_name`` slot for its stored name,
        not a separate property on RecordArray. The default name is
        derived from the class name + fields at construction time."""
        ra = NumericRecordArray.stack([NumericRecord(x=float(i)) for i in range(2)])
        assert "numericrecordarray" in ra.name.lower()

    def test_custom_name_kwarg_honored(self):
        """RecordArray accepts a ``name=`` kwarg at construction,
        matching Record's API."""
        from probpipe.core.event_template import EventTemplate

        tpl = EventTemplate(x=())
        ra = NumericRecordArray(
            {"x": jnp.arange(3.0)},
            batch_shape=(3,),
            template=tpl,
            name="my_sweep",
        )
        assert ra.name == "my_sweep"


class TestSingleFieldCoercion:
    """A single-field NumericRecordArray is array-like via ``__array__``
    / ``__jax_array__``. Multi-field raises.

    This is the NumericRecordArray counterpart to the scalar shim on
    NumericRecord — it keeps ``np.asarray(result)`` working when a
    workflow function auto-wrapped its scalar output as a
    ``NumericRecordArray(result=...)`` under the PR 1.5 output-type
    contract.
    """

    def test_np_asarray_single_field(self):
        nra = NumericRecordArray.stack([NumericRecord(result=float(i)) for i in range(4)])
        arr = np.asarray(nra)
        np.testing.assert_allclose(arr, [0.0, 1.0, 2.0, 3.0])

    def test_jnp_asarray_single_field(self):
        nra = NumericRecordArray.stack([NumericRecord(result=float(i)) for i in range(4)])
        arr = jnp.asarray(nra)
        assert isinstance(arr, jnp.ndarray)
        np.testing.assert_allclose(arr, [0.0, 1.0, 2.0, 3.0])

    def test_multi_field_raises(self):
        nra = NumericRecordArray.stack(
            [NumericRecord(a=float(i), b=float(i) * 2) for i in range(3)]
        )
        with pytest.raises(TypeError, match="2 fields"):
            np.asarray(nra)

    # ``float()`` / ``int()`` / ``bool()`` are deliberately not exposed
    # on NumericRecordArray — the value isn't scalar.
    def test_float_not_supported(self):
        nra = NumericRecordArray.stack([NumericRecord(result=float(i)) for i in range(4)])
        with pytest.raises(TypeError):
            float(nra)
