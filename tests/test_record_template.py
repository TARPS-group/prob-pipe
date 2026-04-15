"""Tests for probpipe.core.record.RecordTemplate."""

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Record
from probpipe.core.record import RecordTemplate


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_kwargs(self):
        tpl = RecordTemplate(x=(), y=(3,))
        assert tpl.fields == ("x", "y")

    def test_dict_positional(self):
        tpl = RecordTemplate({"a": (), "b": (2,)})
        assert tpl.fields == ("a", "b")

    def test_fields_sorted(self):
        tpl = RecordTemplate(z=(), a=(3,), m=None)
        assert tpl.fields == ("a", "m", "z")

    def test_dict_and_kwargs_raises(self):
        with pytest.raises(ValueError, match="Cannot pass both"):
            RecordTemplate({"a": ()}, b=(2,))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            RecordTemplate()

    def test_none_spec(self):
        tpl = RecordTemplate(label=None, x=())
        assert tpl["label"] is None
        assert tpl["x"] == ()

    def test_nested(self):
        inner = RecordTemplate(force=(), mass=())
        outer = RecordTemplate(physics=inner, obs=())
        assert isinstance(outer["physics"], RecordTemplate)
        assert outer["physics"]["force"] == ()

    def test_invalid_spec_raises(self):
        with pytest.raises(TypeError, match="spec must be"):
            RecordTemplate(x=3.0)

    def test_invalid_shape_raises(self):
        with pytest.raises(TypeError, match="non-negative ints"):
            RecordTemplate(x=(-1,))

    def test_invalid_shape_float_raises(self):
        with pytest.raises(TypeError, match="non-negative ints"):
            RecordTemplate(x=(1.5,))


# ---------------------------------------------------------------------------
# Field access
# ---------------------------------------------------------------------------


class TestFieldAccess:
    @pytest.fixture
    def tpl(self):
        return RecordTemplate(a=(), b=(3,), c=(2, 4))

    def test_getitem(self, tpl):
        assert tpl["a"] == ()
        assert tpl["b"] == (3,)
        assert tpl["c"] == (2, 4)

    def test_contains(self, tpl):
        assert "a" in tpl
        assert "z" not in tpl

    def test_len(self, tpl):
        assert len(tpl) == 3

    def test_missing_key_raises(self, tpl):
        with pytest.raises(KeyError):
            tpl["nonexistent"]


# ---------------------------------------------------------------------------
# Leaf shapes
# ---------------------------------------------------------------------------


class TestLeafShapes:
    def test_flat_fields(self):
        tpl = RecordTemplate(x=(), y=(3,))
        assert tpl.leaf_shapes == {"x": (), "y": (3,)}

    def test_opaque_leaf(self):
        tpl = RecordTemplate(label=None, x=())
        assert tpl.leaf_shapes == {"label": None, "x": ()}

    def test_nested_flattens(self):
        inner = RecordTemplate(a=(), b=(2,))
        outer = RecordTemplate(inner=inner, z=(3,))
        shapes = outer.leaf_shapes
        assert shapes == {"inner.a": (), "inner.b": (2,), "z": (3,)}

    def test_numeric_leaf_shapes_excludes_opaque(self):
        tpl = RecordTemplate(label=None, x=(), y=(3,))
        assert tpl.numeric_leaf_shapes == {"x": (), "y": (3,)}


# ---------------------------------------------------------------------------
# flat_size
# ---------------------------------------------------------------------------


class TestFlatSize:
    def test_scalars(self):
        tpl = RecordTemplate(a=(), b=(), c=())
        assert tpl.flat_size == 3

    def test_arrays(self):
        tpl = RecordTemplate(x=(5,), y=(2, 3))
        assert tpl.flat_size == 11

    def test_nested(self):
        inner = RecordTemplate(r=(), K=())
        outer = RecordTemplate(params=inner, obs=(4,))
        assert outer.flat_size == 6

    def test_opaque_excluded(self):
        tpl = RecordTemplate(label=None, x=(3,))
        assert tpl.flat_size == 3

    def test_scalar_only(self):
        tpl = RecordTemplate(a=())
        assert tpl.flat_size == 1


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_setattr_raises(self):
        tpl = RecordTemplate(x=())
        with pytest.raises(AttributeError, match="immutable"):
            tpl.x = (3,)

    def test_delattr_raises(self):
        tpl = RecordTemplate(x=())
        with pytest.raises(AttributeError, match="immutable"):
            del tpl.x


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------


class TestEqualityAndHashing:
    def test_equal(self):
        t1 = RecordTemplate(x=(), y=(3,))
        t2 = RecordTemplate(x=(), y=(3,))
        assert t1 == t2

    def test_not_equal_shapes(self):
        t1 = RecordTemplate(x=())
        t2 = RecordTemplate(x=(3,))
        assert t1 != t2

    def test_not_equal_fields(self):
        t1 = RecordTemplate(x=())
        t2 = RecordTemplate(y=())
        assert t1 != t2

    def test_not_equal_to_other_types(self):
        tpl = RecordTemplate(x=())
        assert tpl != "not a template"

    def test_hash_equal(self):
        t1 = RecordTemplate(x=(), y=(3,))
        t2 = RecordTemplate(x=(), y=(3,))
        assert hash(t1) == hash(t2)

    def test_hash_usable_in_set(self):
        t1 = RecordTemplate(x=(), y=(3,))
        t2 = RecordTemplate(x=(), y=(3,))
        assert len({t1, t2}) == 1

    def test_nested_equality(self):
        inner = RecordTemplate(a=(), b=())
        t1 = RecordTemplate(sub=inner, z=())
        t2 = RecordTemplate(sub=RecordTemplate(a=(), b=()), z=())
        assert t1 == t2
        assert hash(t1) == hash(t2)


# ---------------------------------------------------------------------------
# from_record factory
# ---------------------------------------------------------------------------


class TestFromRecord:
    def test_scalar_fields(self):
        r = Record(a=1.0, b=2.0)
        tpl = RecordTemplate.from_record(r)
        assert tpl.fields == ("a", "b")
        assert tpl["a"] == ()
        assert tpl["b"] == ()

    def test_array_fields(self):
        r = Record(x=jnp.zeros(5), y=jnp.zeros((2, 3)))
        tpl = RecordTemplate.from_record(r)
        assert tpl["x"] == (5,)
        assert tpl["y"] == (2, 3)

    def test_nested_record(self):
        inner = Record(x=1.0, y=jnp.zeros(3))
        outer = Record(params=inner, z=2.0)
        tpl = RecordTemplate.from_record(outer)
        assert isinstance(tpl["params"], RecordTemplate)
        assert tpl["params"]["x"] == ()
        assert tpl["params"]["y"] == (3,)
        assert tpl["z"] == ()

    def test_batch_shape_stripping(self):
        r = Record(x=jnp.zeros((100, 3)), y=jnp.zeros((100,)))
        tpl = RecordTemplate.from_record(r, batch_shape=(100,))
        assert tpl["x"] == (3,)
        assert tpl["y"] == ()

    def test_roundtrip_flat_size(self):
        r = Record(a=1.0, b=jnp.zeros(4), c=jnp.zeros((2, 3)))
        tpl = RecordTemplate.from_record(r)
        assert tpl.flat_size == r.flat_size


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_simple(self):
        tpl = RecordTemplate(x=(), y=(3,))
        r = repr(tpl)
        assert "RecordTemplate(" in r
        assert "x=()" in r
        assert "y=(3,)" in r

    def test_nested(self):
        inner = RecordTemplate(a=())
        outer = RecordTemplate(sub=inner, z=(2,))
        r = repr(outer)
        assert "sub=RecordTemplate(" in r

    def test_opaque(self):
        tpl = RecordTemplate(label=None, x=())
        assert "label=None" in repr(tpl)
