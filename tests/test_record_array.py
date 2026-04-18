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

    def test_zero_length_batch(self):
        """batch_shape=(0,) is a valid edge case (no samples)."""
        tpl = RecordTemplate(x=())
        ra = RecordArray(x=jnp.zeros(0), batch_shape=(0,), template=tpl)
        assert ra.batch_shape == (0,)
        assert len(ra) == 0
        assert ra["x"].shape == (0,)

    def test_flatten_preserves_nan(self):
        """flatten() must not silently replace NaN (numerical stability)."""
        tpl = RecordTemplate(a=(), b=(2,))
        nra = NumericRecordArray(
            a=jnp.array([jnp.nan, 1.0]),
            b=jnp.array([[jnp.inf, -jnp.inf], [0.0, jnp.nan]]),
            batch_shape=(2,),
            template=tpl,
        )
        flat = nra.flatten()
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

    def test_getitem_int_on_recordarray_returns_record(self):
        """The base ``RecordArray`` materialises elements as ``Record``."""
        tpl = RecordTemplate(x=(3,))
        ra = RecordArray(
            x=jnp.arange(30.0).reshape(10, 3),
            batch_shape=(10,), template=tpl,
        )
        assert type(ra[0]) is Record

    def test_getitem_int_on_numeric_recordarray_returns_numeric_record(self):
        """``NumericRecordArray[int]`` must return a ``NumericRecord`` so
        the numeric invariant survives slicing (fix for PR #123 review
        comment #4)."""
        from probpipe import NumericRecord
        tpl = RecordTemplate(x=(3,), y=())
        nra = NumericRecordArray(
            x=jnp.arange(30.0).reshape(10, 3),
            y=jnp.arange(10.0),
            batch_shape=(10,), template=tpl,
        )
        elem = nra[0]
        assert type(elem) is NumericRecord
        assert isinstance(elem["x"], jnp.ndarray)
        assert isinstance(elem["y"], jnp.ndarray)

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


# ---------------------------------------------------------------------------
# Equality / hashability
# ---------------------------------------------------------------------------


class TestEquality:
    def test_equal_same_contents(self):
        tpl = RecordTemplate(x=())
        a = RecordArray(x=jnp.arange(3.0), batch_shape=(3,), template=tpl)
        b = RecordArray(x=jnp.arange(3.0), batch_shape=(3,), template=tpl)
        assert a == b

    def test_unequal_values(self):
        tpl = RecordTemplate(x=())
        a = RecordArray(x=jnp.arange(3.0), batch_shape=(3,), template=tpl)
        b = RecordArray(x=jnp.arange(3.0) + 1, batch_shape=(3,), template=tpl)
        assert a != b

    def test_unequal_batch_shape(self):
        tpl = RecordTemplate(x=())
        a = RecordArray(x=jnp.zeros(3), batch_shape=(3,), template=tpl)
        b = RecordArray(x=jnp.zeros((1, 3)), batch_shape=(1, 3), template=tpl)
        assert a != b

    def test_unequal_template(self):
        a = RecordArray(x=jnp.zeros(3), batch_shape=(3,),
                        template=RecordTemplate(x=()))
        b = RecordArray(x=jnp.zeros((3, 2)), batch_shape=(3,),
                        template=RecordTemplate(x=(2,)))
        assert a != b

    def test_recordarray_not_equal_to_numeric_recordarray(self):
        """Type-strict equality: different concrete classes are not equal
        even when contents match."""
        tpl = RecordTemplate(x=())
        ra = RecordArray(x=jnp.zeros(3), batch_shape=(3,), template=tpl)
        nra = NumericRecordArray(x=jnp.zeros(3), batch_shape=(3,), template=tpl)
        assert ra != nra

    def test_self_equality_with_nan(self):
        """Regression for PR #123 review comment (f-ii): ``a == a`` must
        hold even when leaves contain NaN. Without the identity fast
        path ``jnp.array_equal`` treats NaN != NaN and self-equality
        returns False."""
        tpl = RecordTemplate(x=())
        ra = RecordArray(
            x=jnp.array([jnp.nan, 1.0, jnp.nan]),
            batch_shape=(3,), template=tpl,
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
        tpl = RecordTemplate(x=())
        ra = RecordArray(x=jnp.zeros(3), batch_shape=(3,), template=tpl)
        with pytest.raises(TypeError, match="unhashable"):
            hash(ra)

    def test_numeric_recordarray_unhashable(self):
        tpl = RecordTemplate(x=())
        nra = NumericRecordArray(x=jnp.zeros(3), batch_shape=(3,), template=tpl)
        with pytest.raises(TypeError, match="unhashable"):
            hash(nra)

    def test_cannot_be_used_as_dict_key(self):
        tpl = RecordTemplate(x=())
        ra = RecordArray(x=jnp.zeros(3), batch_shape=(3,), template=tpl)
        with pytest.raises(TypeError, match="unhashable"):
            {ra: 1}


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
        tpl = RecordTemplate(x=(2,))
        with pytest.raises(TypeError, match="numeric"):
            NumericRecordArray(
                {"x": ["a", "b"]}, batch_shape=(), template=tpl,
            )

    def test_rejects_object_dtype_array(self):
        tpl = RecordTemplate(x=())
        arr = np.array(["a", "b"], dtype=object)
        with pytest.raises(TypeError, match="numeric"):
            NumericRecordArray({"x": arr}, batch_shape=(2,), template=tpl)

    def test_rejects_wrong_shape(self):
        """Leaf shape must match ``(*batch_shape, *event_shape)``."""
        tpl = RecordTemplate(x=(3,))
        with pytest.raises(ValueError, match="shape"):
            NumericRecordArray(
                {"x": jnp.zeros((5, 2))},  # expected (5, 3)
                batch_shape=(5,), template=tpl,
            )

    def test_tree_map_to_string_rejected(self):
        """``jax.tree.map(lambda x: "hi", nra)`` must raise at
        unflatten time rather than produce a corrupt NumericRecordArray."""
        tpl = RecordTemplate(x=())
        nra = NumericRecordArray(
            x=jnp.arange(3.0), batch_shape=(3,), template=tpl,
        )
        with pytest.raises(TypeError, match="numeric"):
            jax.tree.map(lambda x: "hi", nra)

    def test_accepts_numpy_leaf_and_coerces_to_jnp(self):
        tpl = RecordTemplate(x=())
        nra = NumericRecordArray(
            x=np.arange(3.0), batch_shape=(3,), template=tpl,
        )
        assert isinstance(nra["x"], jnp.ndarray)

    def test_rejects_numpy_object_dtype_batch(self):
        """Parallel regression for the _is_numeric_leaf object-dtype fix:
        NumericRecordArray._validate_fields shares the same
        _NUMERIC_DTYPE_KINDS set, so an object-dtype batched field
        must be rejected up front (not left to blow up inside JAX)."""
        tpl = RecordTemplate(x=())
        with pytest.raises(TypeError, match="non-numeric dtype"):
            NumericRecordArray(
                {"x": np.array([{"a": 1}, {"b": 2}], dtype=object)},
                batch_shape=(2,), template=tpl,
            )

    def test_rejects_numpy_string_dtype_batch(self):
        tpl = RecordTemplate(x=())
        with pytest.raises(TypeError, match="non-numeric dtype"):
            NumericRecordArray(
                {"x": np.array(["a", "b"])},  # dtype='<U1'
                batch_shape=(2,), template=tpl,
            )

    def test_nested_template_field_allowed(self):
        """Fields whose template spec is a nested ``RecordTemplate`` are
        allowed to hold a Record / RecordArray leaf without triggering
        the numeric-dtype check. This is how nested ProductDistributions
        materialise samples."""
        from probpipe import Record
        inner_tpl = RecordTemplate(x=(), y=())
        outer_tpl = RecordTemplate(physics=inner_tpl, obs=(3,))
        inner = Record(x=1.0, y=2.0)
        nra = NumericRecordArray(
            {"physics": inner, "obs": jnp.zeros(3)},
            batch_shape=(),
            template=outer_tpl,
        )
        assert nra["physics"] is inner
