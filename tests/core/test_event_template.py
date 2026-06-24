"""Tests for probpipe.core.record.EventTemplate."""

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import NumericRecord, NumericRecordArray, Record
from probpipe.core.event_template import (
    ArraySpec,
    DistributionSpec,
    EventTemplate,
    FunctionSpec,
    NumericEventTemplate,
    OpaqueSpec,
)

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_kwargs(self):
        tpl = EventTemplate(x=(), y=(3,))
        assert tpl.fields == ("x", "y")

    def test_dict_positional(self):
        tpl = EventTemplate({"a": (), "b": (2,)})
        assert tpl.fields == ("a", "b")

    def test_fields_insertion_order(self):
        tpl = EventTemplate(z=(), a=(3,), m=None)
        assert tpl.fields == ("z", "a", "m")

    def test_slash_in_field_name_rejected(self):
        with pytest.raises(ValueError, match="must not contain '/'"):
            EventTemplate(**{"a/b": ()})

    def test_dict_and_kwargs_raises(self):
        with pytest.raises(ValueError, match="Cannot pass both"):
            EventTemplate({"a": ()}, b=(2,))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            EventTemplate()

    def test_none_spec(self):
        tpl = EventTemplate(label=None, x=())
        assert tpl["label"] == OpaqueSpec()
        assert tpl["x"] == ArraySpec(())

    def test_nested(self):
        inner = EventTemplate(force=(), mass=())
        outer = EventTemplate(physics=inner, obs=())
        assert isinstance(outer["physics"], EventTemplate)
        assert outer["physics"]["force"] == ArraySpec(())

    def test_invalid_spec_raises(self):
        with pytest.raises(TypeError, match="spec must be"):
            EventTemplate(x=3.0)

    def test_invalid_shape_raises(self):
        with pytest.raises(TypeError, match="non-negative ints"):
            EventTemplate(x=(-1,))

    def test_invalid_shape_float_raises(self):
        with pytest.raises(TypeError, match="non-negative ints"):
            EventTemplate(x=(1.5,))


# ---------------------------------------------------------------------------
# Field access
# ---------------------------------------------------------------------------


class TestFieldAccess:
    @pytest.fixture
    def tpl(self):
        return EventTemplate(a=(), b=(3,), c=(2, 4))

    def test_getitem(self, tpl):
        assert tpl["a"] == ArraySpec(())
        assert tpl["b"] == ArraySpec((3,))
        assert tpl["c"] == ArraySpec((2, 4))

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
        tpl = EventTemplate(x=(), y=(3,))
        assert tpl.leaf_shapes == {"x": (), "y": (3,)}

    def test_nested_flattens(self):
        inner = EventTemplate(a=(), b=(2,))
        outer = EventTemplate(inner=inner, z=(3,))
        shapes = outer.leaf_shapes
        # Slash-delimited keys for consistency with ``Record["a/b"]``
        # path access.
        assert shapes == {"inner/a": (), "inner/b": (2,), "z": (3,)}

    def test_leaf_shapes_on_numeric_template(self):
        tpl = NumericEventTemplate(x=(), y=(3,))
        assert tpl.leaf_shapes == {"x": (), "y": (3,)}

    def test_leaf_shapes_not_on_base_template(self):
        """``vector_size`` / ``leaf_shapes`` are only meaningful
        when every leaf is numeric — they live on
        :class:`NumericEventTemplate`, not the base ``EventTemplate``.
        """
        tpl = EventTemplate(label=None, x=(), y=(3,))
        assert not hasattr(tpl, "leaf_shapes")
        assert not hasattr(tpl, "vector_size")


# ---------------------------------------------------------------------------
# leaf_paths (canonical leaf order — the single source of truth)
# ---------------------------------------------------------------------------


class TestLeafPaths:
    def test_flat_equals_fields(self):
        # For a flat template (every field a leaf), leaf_paths == fields.
        tpl = EventTemplate(x=(), y=(3,))
        assert tpl.leaf_paths == ("x", "y") == tpl.fields

    def test_includes_opaque_leaves(self):
        # leaf_paths enumerates every leaf, numeric or opaque.
        tpl = EventTemplate(label=None, x=())
        assert tpl.leaf_paths == ("label", "x")

    def test_nested_depth_first_insertion_order(self):
        # A nested field expands into one path per nested leaf; fields stays
        # top-level only.
        inner = EventTemplate(a=(), b=(2,))
        outer = EventTemplate(inner=inner, z=(3,))
        assert outer.leaf_paths == ("inner/a", "inner/b", "z")
        assert outer.fields == ("inner", "z")

    def test_depth2(self):
        tpl = EventTemplate(outer=EventTemplate(deep=EventTemplate(g=(), h=()), a=()), m=())
        assert tpl.leaf_paths == ("outer/deep/g", "outer/deep/h", "outer/a", "m")

    def test_keys_match_leaf_shapes_in_order(self):
        # leaf_shapes (numeric template) is keyed by leaf_paths, same order.
        tpl = EventTemplate(outer=EventTemplate(a=(2,), b=()), m=())
        assert tuple(tpl.leaf_shapes) == tpl.leaf_paths

    def test_order_matches_flatten_and_to_vector(self):
        # The canonical leaf order is the order flatten() / to_vector() use.
        v = NumericRecord(
            x=jnp.array([1.0, 2.0]),
            nested=NumericRecord(a=jnp.array(3.0), b=jnp.array([4.0, 5.0])),
        )
        tpl = EventTemplate.infer_from(v)
        assert tpl.leaf_paths == ("x", "nested/a", "nested/b")
        # to_vector concatenates leaves in leaf_paths order: x(2) | a(1) | b(2).
        np.testing.assert_allclose(tpl.to_vector(v), [1.0, 2.0, 3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# vector_size (on NumericEventTemplate)
# ---------------------------------------------------------------------------


class TestFlatSize:
    def test_scalars(self):
        tpl = NumericEventTemplate(a=(), b=(), c=())
        assert tpl.vector_size == 3

    def test_arrays(self):
        tpl = NumericEventTemplate(x=(5,), y=(2, 3))
        assert tpl.vector_size == 11

    def test_nested(self):
        inner = NumericEventTemplate(r=(), K=())
        outer = NumericEventTemplate(params=inner, obs=(4,))
        assert outer.vector_size == 6

    def test_scalar_only(self):
        tpl = NumericEventTemplate(a=())
        assert tpl.vector_size == 1

    def test_rejects_opaque_leaf(self):
        with pytest.raises(TypeError, match="opaque"):
            NumericEventTemplate(label=None, x=(3,))

    def test_rejects_non_numeric_nested(self):
        # ``EventTemplate(x=(), label=None)`` stays a plain base template
        # (mixed leaves block auto-promotion), so embedding it inside a
        # ``NumericEventTemplate`` must be rejected.
        inner = EventTemplate(x=(), label=None)
        with pytest.raises(TypeError, match="NumericEventTemplate"):
            NumericEventTemplate(nested=inner, y=())


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_setattr_raises(self):
        tpl = EventTemplate(x=())
        with pytest.raises(AttributeError, match="immutable"):
            tpl.x = (3,)

    def test_delattr_raises(self):
        tpl = EventTemplate(x=())
        with pytest.raises(AttributeError, match="immutable"):
            del tpl.x


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------


class TestEqualityAndHashing:
    def test_equal(self):
        t1 = EventTemplate(x=(), y=(3,))
        t2 = EventTemplate(x=(), y=(3,))
        assert t1 == t2

    def test_not_equal_shapes(self):
        t1 = EventTemplate(x=())
        t2 = EventTemplate(x=(3,))
        assert t1 != t2

    def test_not_equal_fields(self):
        t1 = EventTemplate(x=())
        t2 = EventTemplate(y=())
        assert t1 != t2

    def test_not_equal_to_other_types(self):
        tpl = EventTemplate(x=())
        assert tpl != "not a template"

    def test_hash_equal(self):
        t1 = EventTemplate(x=(), y=(3,))
        t2 = EventTemplate(x=(), y=(3,))
        assert hash(t1) == hash(t2)

    def test_hash_usable_in_set(self):
        t1 = EventTemplate(x=(), y=(3,))
        t2 = EventTemplate(x=(), y=(3,))
        assert len({t1, t2}) == 1

    def test_nested_equality(self):
        inner = EventTemplate(a=(), b=())
        t1 = EventTemplate(sub=inner, z=())
        t2 = EventTemplate(sub=EventTemplate(a=(), b=()), z=())
        assert t1 == t2
        assert hash(t1) == hash(t2)

    def test_eq_is_order_sensitive(self):
        """Insertion-order is part of the template's identity (#124),
        and ``__hash__`` is order-sensitive — so ``__eq__`` must agree
        to satisfy Python's eq/hash contract.
        """
        t1 = EventTemplate(a=(), b=(2,))
        t2 = EventTemplate(b=(2,), a=())
        assert t1 != t2
        assert hash(t1) != hash(t2)
        # And the contract holds: equal templates hash the same.
        t3 = EventTemplate(a=(), b=(2,))
        assert t1 == t3
        assert hash(t1) == hash(t3)


# ---------------------------------------------------------------------------
# from_record factory
# ---------------------------------------------------------------------------


class TestFromRecord:
    def test_scalar_fields(self):
        r = Record(a=1.0, b=2.0)
        tpl = EventTemplate.infer_from(r)
        assert tpl.fields == ("a", "b")
        assert tpl["a"] == ArraySpec(())
        assert tpl["b"] == ArraySpec(())

    def test_array_fields(self):
        r = Record(x=jnp.zeros(5), y=jnp.zeros((2, 3)))
        tpl = EventTemplate.infer_from(r)
        assert tpl["x"] == ArraySpec((5,))
        assert tpl["y"] == ArraySpec((2, 3))

    def test_nested_record(self):
        inner = Record(x=1.0, y=jnp.zeros(3))
        outer = Record(params=inner, z=2.0)
        tpl = EventTemplate.infer_from(outer)
        assert isinstance(tpl["params"], EventTemplate)
        assert tpl["params"]["x"] == ArraySpec(())
        assert tpl["params"]["y"] == ArraySpec((3,))
        assert tpl["z"] == ArraySpec(())

    def test_roundtrip_vector_size(self):
        from probpipe.core._numeric_record import NumericRecord

        r = NumericRecord(a=1.0, b=jnp.zeros(4), c=jnp.zeros((2, 3)))
        tpl = EventTemplate.infer_from(r)
        # Auto-promoted to NumericEventTemplate because the input was a
        # NumericRecord, so ``vector_size`` is reachable.
        assert isinstance(tpl, NumericEventTemplate)
        assert tpl.vector_size == r.vector_size

    def test_from_numeric_record_promotes(self):
        """Calling ``from_record`` on a ``NumericRecord`` returns a
        :class:`NumericEventTemplate`, even through the base
        ``EventTemplate.infer_from`` classmethod, so downstream code
        that needs ``vector_size`` keeps working without the caller having
        to name the subclass explicitly."""
        from probpipe.core._numeric_record import NumericRecord

        r = NumericRecord(a=1.0, b=jnp.zeros(2))
        tpl = EventTemplate.infer_from(r)
        assert isinstance(tpl, NumericEventTemplate)

    def test_from_mixed_record_stays_base(self):
        """A plain ``Record`` with a non-numeric leaf can't be promoted —
        the result is a plain :class:`EventTemplate` with an opaque
        slot."""
        r = Record(x=1.0, label="tag")
        tpl = EventTemplate.infer_from(r)
        assert type(tpl) is EventTemplate
        assert tpl["label"] == OpaqueSpec()

    def test_list_leaf_is_opaque(self):
        """A Python list leaf has no .shape / .dtype, so the field is
        recorded as opaque (``None``) even when it contains numbers.
        Users should wrap lists in np.asarray/jnp.asarray for a numeric
        template entry — this test pins down that behavior so the
        documented guidance stays in sync with the implementation."""
        r = Record(xs=[1.0, 2.0, 3.0])
        tpl = EventTemplate.infer_from(r)
        assert tpl["xs"] == OpaqueSpec()

    def test_list_leaf_after_asarray_is_numeric(self):
        """The opposite end of the list-leaf story: wrapping the list
        in ``np.asarray`` produces a numeric template entry."""

        r = Record(xs=np.asarray([1.0, 2.0, 3.0]))
        tpl = EventTemplate.infer_from(r)
        assert tpl["xs"] == ArraySpec((3,))


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_simple(self):
        # All-numeric → auto-promotes to NumericEventTemplate.
        tpl = EventTemplate(x=(), y=(3,))
        r = repr(tpl)
        assert r.startswith("NumericEventTemplate(")
        assert "x=()" in r
        assert "y=(3,)" in r

    def test_nested(self):
        inner = EventTemplate(a=())
        outer = EventTemplate(sub=inner, z=(2,))
        r = repr(outer)
        # Both auto-promote; inner's repr is nested under the outer.
        assert "sub=NumericEventTemplate(" in r

    def test_mixed_stays_base(self):
        tpl = EventTemplate(label=None, x=())
        assert repr(tpl).startswith("EventTemplate(")

    def test_opaque(self):
        tpl = EventTemplate(label=None, x=())
        assert "label=None" in repr(tpl)

    def test_populated_array_spec_shows_full_repr(self):
        # A spec carrying dtype/support is not bare, so repr falls back to the
        # full dataclass repr rather than the bare-shape shorthand.
        tpl = EventTemplate(x=ArraySpec((3,), dtype="float32"))
        r = repr(tpl)
        assert "ArraySpec(" in r
        assert "dtype='float32'" in r

    def test_populated_opaque_spec_shows_full_repr(self):
        tpl = EventTemplate(label=OpaqueSpec(meta="tag"), x=())
        r = repr(tpl)
        assert "OpaqueSpec(meta='tag')" in r


# ---------------------------------------------------------------------------
# Leaf specs — the closed sum (ArraySpec / OpaqueSpec / DistributionSpec /
# FunctionSpec)
# ---------------------------------------------------------------------------


class TestLeafSpecs:
    def test_array_spec_defaults(self):
        spec = ArraySpec((3,))
        assert spec.shape == (3,)
        assert spec.dtype is None
        assert spec.support is None

    def test_array_spec_coerces_shape_to_tuple(self):
        # A list shape is normalised to a tuple so the spec stays hashable.
        spec = ArraySpec([2, 4])
        assert spec.shape == (2, 4)
        assert isinstance(spec.shape, tuple)

    def test_array_spec_rejects_negative_dims(self):
        with pytest.raises(TypeError, match="non-negative ints"):
            ArraySpec((-1,))

    def test_specs_are_frozen(self):
        from dataclasses import FrozenInstanceError

        for spec in (
            ArraySpec((3,)),
            OpaqueSpec(),
            DistributionSpec(inner_template=EventTemplate(x=())),
            FunctionSpec(input_template=EventTemplate(x=()), output_template=EventTemplate(y=())),
        ):
            with pytest.raises(FrozenInstanceError):
                spec.shape = (1,)  # type: ignore[misc]

    def test_specs_are_hashable(self):
        # Usable as dict keys / set members — required for treedef caching.
        specs = {
            ArraySpec((3,)): 1,
            OpaqueSpec(): 2,
            DistributionSpec(inner_template=EventTemplate(x=())): 3,
            FunctionSpec(
                input_template=EventTemplate(x=()), output_template=EventTemplate(y=())
            ): 4,
        }
        assert len(specs) == 4

    def test_specs_value_equality(self):
        assert ArraySpec((3,)) == ArraySpec((3,))
        assert ArraySpec((3,), dtype="float32") == ArraySpec((3,), dtype="float32")
        assert ArraySpec((3,)) != ArraySpec((2,))
        assert ArraySpec((3,)) != ArraySpec((3,), dtype="float32")
        assert OpaqueSpec() == OpaqueSpec()
        assert OpaqueSpec(meta="a") == OpaqueSpec(meta="a")
        assert OpaqueSpec(meta="a") != OpaqueSpec(meta="b")
        inner = EventTemplate(x=())
        assert DistributionSpec(inner_template=inner) == DistributionSpec(inner_template=inner)
        assert FunctionSpec(
            input_template=EventTemplate(x=()), output_template=EventTemplate(y=())
        ) == FunctionSpec(input_template=EventTemplate(x=()), output_template=EventTemplate(y=()))

    def test_array_and_opaque_specs_are_distinct(self):
        assert ArraySpec(()) != OpaqueSpec()


# ---------------------------------------------------------------------------
# Construction sugar + explicit specs
# ---------------------------------------------------------------------------


class TestConstructionSpecs:
    def test_tuple_becomes_array_spec(self):
        tpl = EventTemplate(x=(3,))
        assert tpl["x"] == ArraySpec((3,))

    def test_none_becomes_opaque_spec(self):
        tpl = EventTemplate(label=None)
        assert tpl["label"] == OpaqueSpec()

    def test_nested_template_preserved(self):
        inner = EventTemplate(a=(), b=(3,))
        tpl = EventTemplate(sub=inner, z=())
        assert tpl["sub"] is inner

    def test_explicit_array_spec_accepted(self):
        spec = ArraySpec((2,), dtype="float32")
        tpl = EventTemplate(x=spec)
        assert tpl["x"] is spec

    def test_explicit_opaque_spec_accepted(self):
        spec = OpaqueSpec(meta="tag")
        tpl = EventTemplate(label=spec, x=())
        assert tpl["label"] is spec

    def test_explicit_distribution_and_function_specs_accepted(self):
        dspec = DistributionSpec(inner_template=EventTemplate(x=()))
        fspec = FunctionSpec(
            input_template=EventTemplate(a=()), output_template=EventTemplate(b=())
        )
        tpl = EventTemplate(d=dspec, f=fspec)
        assert tpl["d"] is dspec
        assert tpl["f"] is fspec

    def test_unsupported_spec_rejected(self):
        with pytest.raises(TypeError, match="spec must be"):
            EventTemplate(x=3.0)


# ---------------------------------------------------------------------------
# Auto-promotion to NumericEventTemplate (iff every leaf is an ArraySpec)
# ---------------------------------------------------------------------------


class TestAutoPromotionSpecs:
    def test_explicit_array_specs_promote(self):
        tpl = EventTemplate(x=ArraySpec(()), y=ArraySpec((3,)))
        assert isinstance(tpl, NumericEventTemplate)

    def test_nested_numeric_promotes(self):
        tpl = EventTemplate(sub=EventTemplate(a=(), b=(2,)), z=())
        assert isinstance(tpl, NumericEventTemplate)

    def test_opaque_spec_blocks_promotion(self):
        tpl = EventTemplate(x=(), label=OpaqueSpec())
        assert type(tpl) is EventTemplate

    def test_distribution_spec_blocks_promotion(self):
        tpl = EventTemplate(x=(), d=DistributionSpec(inner_template=EventTemplate(a=())))
        assert type(tpl) is EventTemplate

    def test_function_spec_blocks_promotion(self):
        tpl = EventTemplate(
            x=(),
            f=FunctionSpec(input_template=EventTemplate(a=()), output_template=EventTemplate(b=())),
        )
        assert type(tpl) is EventTemplate

    def test_numeric_rejects_opaque_spec(self):
        with pytest.raises(TypeError, match="opaque"):
            NumericEventTemplate(x=(), label=OpaqueSpec())

    def test_numeric_rejects_distribution_spec(self):
        with pytest.raises(TypeError, match="non-numeric"):
            NumericEventTemplate(x=(), d=DistributionSpec(inner_template=EventTemplate(a=())))

    def test_numeric_rejects_function_spec(self):
        with pytest.raises(TypeError, match="non-numeric"):
            NumericEventTemplate(
                x=(),
                f=FunctionSpec(
                    input_template=EventTemplate(a=()), output_template=EventTemplate(b=())
                ),
            )


# ---------------------------------------------------------------------------
# Back-compat: shape-shaped accessors keep their pre-spec return values
# ---------------------------------------------------------------------------


class TestShapeAccessorBackCompat:
    def test_vector_size_unchanged(self):
        tpl = EventTemplate(x=(), y=(3,), z=(2, 4))
        assert isinstance(tpl, NumericEventTemplate)
        assert tpl.vector_size == 1 + 3 + 8

    def test_hash_eq_order_sensitive(self):
        a = EventTemplate(x=(), y=(3,))
        b = EventTemplate(x=(), y=(3,))
        c = EventTemplate(y=(3,), x=())
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)


# ---------------------------------------------------------------------------
# Numeric queries & projection: is_numeric / is_multi_field /
# non_numeric_fields / numeric_subset
# ---------------------------------------------------------------------------


def _dist_spec() -> DistributionSpec:
    return DistributionSpec(inner_template=EventTemplate(a=()))


def _func_spec() -> FunctionSpec:
    return FunctionSpec(input_template=EventTemplate(a=()), output_template=EventTemplate(b=()))


class TestIsNumeric:
    def test_all_arrayspec(self):
        assert EventTemplate(x=(), y=(3,)).is_numeric is True

    def test_nested_all_numeric(self):
        tpl = EventTemplate(x=(), params=EventTemplate(a=(), b=(3,)))
        assert tpl.is_numeric is True

    def test_mixed_opaque(self):
        assert EventTemplate(x=(), label=None).is_numeric is False

    def test_distribution_leaf(self):
        assert EventTemplate(x=(), d=_dist_spec()).is_numeric is False

    def test_function_leaf(self):
        assert EventTemplate(x=(), f=_func_spec()).is_numeric is False

    def test_nested_mixed(self):
        tpl = EventTemplate(x=(), nested=EventTemplate(a=(), label=None))
        assert tpl.is_numeric is False


class TestIsMultiField:
    def test_single_field(self):
        assert EventTemplate(x=()).is_multi_field is False

    def test_multi_field(self):
        assert EventTemplate(x=(), y=()).is_multi_field is True

    def test_single_opaque_leaf(self):
        assert EventTemplate(label=None).is_multi_field is False

    def test_two_leaves_mixed(self):
        assert EventTemplate(x=(), label=None).is_multi_field is True

    def test_single_leaf_under_nested_field(self):
        # One top-level field nesting a single leaf -> one leaf -> not multi.
        assert EventTemplate(a=EventTemplate(b=())).is_multi_field is False

    def test_multiple_leaves_under_one_nested_field(self):
        # jhuggins' case: a single top-level field 'a' with leaves a/b, a/c.
        tpl = EventTemplate(a=EventTemplate(b=(), c=()))
        assert tpl.fields == ("a",)  # one top-level field ...
        assert tpl.is_multi_field is True  # ... but two leaves -> multi-field

    def test_deeply_nested_single_leaf(self):
        tpl = EventTemplate(a=EventTemplate(b=EventTemplate(c=())))
        assert tpl.is_multi_field is False


class TestNumericSubset:
    def test_drops_non_numeric_keeps_numeric(self):
        tpl = EventTemplate(x=(), label=None, d=_dist_spec(), y=(3,))
        sub = tpl.numeric_subset()
        assert isinstance(sub, NumericEventTemplate)
        assert sub.fields == ("x", "y")

    def test_recurses_into_nested(self):
        tpl = EventTemplate(x=(), nested=EventTemplate(a=(), label=None, b=(3,)))
        sub = tpl.numeric_subset()
        assert sub.fields == ("x", "nested")
        assert isinstance(sub["nested"], NumericEventTemplate)
        assert sub["nested"].fields == ("a", "b")

    def test_prunes_emptied_nested(self):
        tpl = EventTemplate(x=(), nested=EventTemplate(label=None, tag=None))
        sub = tpl.numeric_subset()
        assert sub.fields == ("x",)

    def test_path_stable(self):
        tpl = EventTemplate(x=(), nested=EventTemplate(a=(), label=None, b=(3,)))
        sub = tpl.numeric_subset()
        assert sub.leaf_shapes == {"x": (), "nested/a": (), "nested/b": (3,)}

    def test_idempotent_on_all_numeric(self):
        tpl = EventTemplate(x=(), y=(3,), nested=EventTemplate(a=(), b=(2,)))
        sub = tpl.numeric_subset()
        assert sub == tpl
        assert sub.numeric_subset() == sub

    def test_returns_numeric_template_with_vector_size(self):
        tpl = EventTemplate(x=(), label=None, y=(3,), z=(2, 4))
        sub = tpl.numeric_subset()
        assert isinstance(sub, NumericEventTemplate)
        assert sub.vector_size == 1 + 3 + 8

    def test_raises_when_no_numeric_leaves(self):
        tpl = EventTemplate(label=None, tag=None)
        with pytest.raises(ValueError, match="ArraySpec leaves survive"):
            tpl.numeric_subset()

    def test_raises_names_dropped_fields(self):
        tpl = EventTemplate(label=None, d=_dist_spec())
        with pytest.raises(ValueError, match="label"):
            tpl.numeric_subset()

    def test_raises_when_only_nested_empties(self):
        tpl = EventTemplate(nested=EventTemplate(label=None, tag=None))
        with pytest.raises(ValueError, match="nested"):
            tpl.numeric_subset()


# ---------------------------------------------------------------------------
# to_vector / from_vector — 1-D numeric (de)serialization
# ---------------------------------------------------------------------------


class TestToVector:
    def test_scalar_value(self):
        v = NumericRecord(x=1.5)
        tpl = EventTemplate.infer_from(v)
        vec = tpl.to_vector(v)
        assert vec.shape == (1,)
        assert jnp.array_equal(vec, jnp.asarray([1.5]))

    def test_vector_value(self):
        v = NumericRecord(y=jnp.arange(3.0))
        tpl = EventTemplate.infer_from(v)
        vec = tpl.to_vector(v)
        assert vec.shape == (3,)
        assert jnp.array_equal(vec, jnp.arange(3.0))

    def test_multi_field_value(self):
        v = NumericRecord(x=1.0, y=jnp.arange(3.0), z=jnp.ones((2, 4)))
        tpl = EventTemplate.infer_from(v)
        vec = tpl.to_vector(v)
        assert vec.shape == (1 + 3 + 8,)

    def test_to_vector_shape_is_vector_size(self):
        tpl = EventTemplate(x=(), y=(3,), z=(2, 4))
        v = NumericRecord(x=0.0, y=jnp.zeros(3), z=jnp.zeros((2, 4)))
        assert tpl.to_vector(v).shape == (tpl.vector_size,)

    def test_order_and_value_match_instance_to_vector(self):
        # The template-level to_vector must agree with the instance-level
        # NumericRecord.to_vector (same canonical leaf order), so the two are
        # interchangeable.
        v = NumericRecord(x=1.0, y=jnp.arange(3.0), nested=NumericRecord(a=2.0, b=jnp.arange(2.0)))
        tpl = EventTemplate.infer_from(v)
        assert jnp.array_equal(tpl.to_vector(v), v.to_vector())

    def test_batched_shape_is_batch_shape_plus_vector_size(self):
        tpl = EventTemplate(x=(), y=(3,))
        flat = jnp.arange(2 * 5 * tpl.vector_size, dtype=float).reshape(2, 5, tpl.vector_size)
        v = tpl.from_vector(flat)
        assert isinstance(v, NumericRecordArray)
        assert tpl.to_vector(v).shape == (2, 5, tpl.vector_size)

    def test_non_record_value_raises(self):
        tpl = EventTemplate(x=())
        with pytest.raises(TypeError, match="NumericRecord"):
            tpl.to_vector(jnp.asarray([1.0]))


class TestFromVectorRoundTripSingle:
    def test_scalar(self):
        v = NumericRecord(x=1.5)
        tpl = EventTemplate.infer_from(v)
        assert tpl.from_vector(tpl.to_vector(v)) == v

    def test_vector(self):
        v = NumericRecord(y=jnp.arange(3.0))
        tpl = EventTemplate.infer_from(v)
        assert tpl.from_vector(tpl.to_vector(v)) == v

    def test_multi_field(self):
        v = NumericRecord(x=1.0, y=jnp.arange(3.0), z=jnp.arange(8.0).reshape(2, 4))
        tpl = EventTemplate.infer_from(v)
        assert tpl.from_vector(tpl.to_vector(v)) == v

    def test_nested(self):
        v = NumericRecord(x=1.0, y=jnp.arange(3.0), nested=NumericRecord(a=2.0, b=jnp.arange(2.0)))
        tpl = EventTemplate.infer_from(v)
        round_tripped = tpl.from_vector(tpl.to_vector(v))
        assert isinstance(round_tripped, NumericRecord)
        assert round_tripped == v

    def test_returns_single_for_1d_vec(self):
        tpl = EventTemplate(x=(), y=(3,))
        v = tpl.from_vector(jnp.arange(4.0))
        assert isinstance(v, NumericRecord)


class TestFromVectorRoundTripBatched:
    def test_single_batch_axis(self):
        tpl = EventTemplate(x=(), y=(3,))
        flat = jnp.arange(4 * tpl.vector_size, dtype=float).reshape(4, tpl.vector_size)
        v = tpl.from_vector(flat)
        assert isinstance(v, NumericRecordArray)
        assert v.batch_shape == (4,)
        assert tpl.from_vector(tpl.to_vector(v)) == v

    def test_multi_axis_batch_shape(self):
        # batch_shape=(2, 3) catches trailing-axis split / reshape bugs.
        tpl = EventTemplate(x=(), y=(3,), z=(2, 2))
        flat = jnp.arange(2 * 3 * tpl.vector_size, dtype=float).reshape(2, 3, tpl.vector_size)
        v = tpl.from_vector(flat)
        assert isinstance(v, NumericRecordArray)
        assert v.batch_shape == (2, 3)
        assert jnp.array_equal(tpl.to_vector(v), flat)
        assert tpl.from_vector(tpl.to_vector(v)) == v

    def test_nested_multi_axis_batch_shape(self):
        # Nested numeric subtree + multi-axis batch: from_vector builds a nested
        # NumericRecordArray as a field of the outer NumericRecordArray.
        tpl = EventTemplate(x=(), nested=EventTemplate(a=(), b=(2,)), y=(3,))
        flat = jnp.arange(2 * 3 * tpl.vector_size, dtype=float).reshape(2, 3, tpl.vector_size)
        v = tpl.from_vector(flat)
        assert isinstance(v, NumericRecordArray)
        assert v.batch_shape == (2, 3)
        assert isinstance(v["nested"], NumericRecordArray)
        assert v["nested/b"].shape == (2, 3, 2)
        # NOTE: assert via the vector round-trip, not ``from_vector(...) == v``.
        # RecordArray.__eq__ is currently broken for a nested *Array field (it
        # calls jnp.array_equal on the nested array, which raises, then falls
        # back to an identity check). Tracked in #235 (batch-records PR); once
        # that is fixed, simplify to ``tpl.from_vector(tpl.to_vector(v)) == v``.
        assert jnp.array_equal(tpl.to_vector(v), flat)


class TestFromVectorErrors:
    def test_wrong_trailing_size_raises(self):
        tpl = EventTemplate(x=(), y=(3,))
        with pytest.raises(ValueError, match="vector_size"):
            tpl.from_vector(jnp.zeros(5))
