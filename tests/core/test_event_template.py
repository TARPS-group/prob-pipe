"""Tests for probpipe.core.record.EventTemplate."""

import jax
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
    ValueSpec,
)

# ---------------------------------------------------------------------------
# Path separator
# ---------------------------------------------------------------------------


def test_path_separator_is_slash():
    """Docstrings spell the nested-path separator literally as ``/``.

    Pin the constant so changing it trips CI and forces a conscious sweep of
    the docstrings that hardcode the character.
    """
    from probpipe.core.event_template import _PATH_SEP

    assert _PATH_SEP == "/"


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
        assert isinstance(outer.at_path("physics"), EventTemplate)
        assert outer["physics/force"] == ArraySpec(())

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


class TestNamedTreeSurfaceOnEventTemplate:
    """EventTemplate gained the shared ``_NamedTree`` collection protocol:
    ``/``-path and tuple indexing, path membership, iteration, and
    ``keys``/``values``/``items`` — over its value specs, keyed by leaf path.
    Exercised on a *nested* template (the flat case is covered above).
    """

    @pytest.fixture
    def nested(self):
        return EventTemplate(theta=EventTemplate(loc=(2,), scale=()), sigma=(3,))

    def test_path_indexing_returns_leaf_spec(self, nested):
        assert nested["theta/loc"] == ArraySpec((2,))
        assert nested["theta/scale"] == ArraySpec(())
        assert nested["sigma"] == ArraySpec((3,))

    def test_tuple_indexing_matches_slash_path(self, nested):
        assert nested["theta", "loc"] == nested["theta/loc"]

    def test_partial_path_navigates_to_subtree(self, nested):
        # A partial path reaches a subtree via at_path; [] is leaf-only.
        assert nested.at_path("theta") == EventTemplate(loc=(2,), scale=())
        with pytest.raises(KeyError):
            nested["theta"]

    def test_path_membership_is_leaf_only(self, nested):
        # Leaf-keyed membership: a key (leaf path) is a member; a partial path is
        # navigable but not a member.
        assert "theta/loc" in nested
        assert "sigma" in nested
        assert "theta" not in nested  # subtree, not a field
        assert "theta/missing" not in nested
        assert "nonexistent" not in nested
        # children is the one-level view that does include subtrees
        assert "theta" in nested.children

    def test_iteration_and_len_over_leaf_fields(self, nested):
        # __iter__ / keys / __len__ range over leaf fields, keyed by path.
        assert tuple(nested) == ("theta/loc", "theta/scale", "sigma")
        assert tuple(nested.keys()) == ("theta/loc", "theta/scale", "sigma")
        assert len(nested) == 3
        # children gives the one-level (top-level) names
        assert tuple(nested.children) == ("theta", "sigma")

    def test_values_and_items(self, nested):
        values = list(nested.values())
        assert values == [ArraySpec((2,)), ArraySpec(()), ArraySpec((3,))]
        assert list(nested.items()) == list(zip(nested.keys(), values))


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
# keys() (canonical leaf order — the single source of truth)
# ---------------------------------------------------------------------------


class TestKeys:
    def test_flat_equals_fields(self):
        # For a flat template (every field a leaf), keys() == fields.
        tpl = EventTemplate(x=(), y=(3,))
        assert tuple(tpl.keys()) == ("x", "y") == tpl.fields

    def test_includes_opaque_leaves(self):
        # keys() enumerates every leaf, numeric or opaque.
        tpl = EventTemplate(label=None, x=())
        assert tuple(tpl.keys()) == ("label", "x")

    def test_nested_depth_first_insertion_order(self):
        # A nested field expands into one key per nested leaf; fields stays
        # top-level only.
        inner = EventTemplate(a=(), b=(2,))
        outer = EventTemplate(inner=inner, z=(3,))
        assert tuple(outer.keys()) == ("inner/a", "inner/b", "z")
        assert outer.fields == ("inner", "z")

    def test_depth2(self):
        tpl = EventTemplate(outer=EventTemplate(deep=EventTemplate(g=(), h=()), a=()), m=())
        assert tuple(tpl.keys()) == ("outer/deep/g", "outer/deep/h", "outer/a", "m")

    def test_keys_match_leaf_shapes_in_order(self):
        # leaf_shapes (numeric template) is keyed by keys(), same order.
        tpl = EventTemplate(outer=EventTemplate(a=(2,), b=()), m=())
        assert tuple(tpl.leaf_shapes) == tuple(tpl.keys())

    def test_order_matches_flatten_and_to_vector(self):
        # The canonical leaf order is the order flatten() / to_vector() use.
        v = NumericRecord(
            x=jnp.array([1.0, 2.0]),
            nested=NumericRecord(a=jnp.array(3.0), b=jnp.array([4.0, 5.0])),
        )
        tpl = EventTemplate.infer_from(v)
        assert tuple(tpl.keys()) == ("x", "nested/a", "nested/b")
        # to_vector concatenates leaves in keys() order: x(2) | a(1) | b(2).
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
# infer_from factory
# ---------------------------------------------------------------------------


class TestInferFrom:
    def test_scalar_fields(self):
        r = Record(a=1.0, b=2.0)
        tpl = EventTemplate.infer_from(r)
        assert tpl.fields == ("a", "b")
        assert tpl["a"] == ArraySpec(())
        assert tpl["b"] == ArraySpec(())

    def test_mapping_input_inferred_field_by_field(self):
        # The non-Record branch: a bare mapping is inferred field by field
        # (a nested Record contributes its own event_template).
        tpl = EventTemplate.infer_from(
            {"a": 1.0, "x": jnp.zeros(3), "params": Record(m=jnp.zeros(2))}
        )
        assert tuple(tpl.children) == ("a", "x", "params")
        assert tpl["a"] == ArraySpec(())
        assert tpl["x"] == ArraySpec((3,))
        assert isinstance(tpl.at_path("params"), EventTemplate)
        assert tpl["params/m"] == ArraySpec((2,))

    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError, match="at least one field"):
            EventTemplate.infer_from({})

    def test_array_fields(self):
        r = Record(x=jnp.zeros(5), y=jnp.zeros((2, 3)))
        tpl = EventTemplate.infer_from(r)
        assert tpl["x"] == ArraySpec((5,))
        assert tpl["y"] == ArraySpec((2, 3))

    def test_nested_record(self):
        inner = Record(x=1.0, y=jnp.zeros(3))
        outer = Record(params=inner, z=2.0)
        tpl = EventTemplate.infer_from(outer)
        assert isinstance(tpl.at_path("params"), EventTemplate)
        assert tpl["params/x"] == ArraySpec(())
        assert tpl["params/y"] == ArraySpec((3,))
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
        """Calling ``infer_from`` on a ``NumericRecord`` returns a
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
        # full dataclass repr rather than the bare-shape shorthand. The dtype
        # renders in its normalised ``numpy.dtype`` form.
        tpl = EventTemplate(x=ArraySpec((3,), dtype="float32"))
        r = repr(tpl)
        assert "ArraySpec(" in r
        assert "dtype=dtype('float32')" in r

    def test_populated_opaque_spec_shows_full_repr(self):
        tpl = EventTemplate(label=OpaqueSpec(meta="tag"), x=())
        r = repr(tpl)
        assert "OpaqueSpec(meta='tag')" in r


# ---------------------------------------------------------------------------
# Value specs — the ValueSpec base and its concrete subclasses (ArraySpec /
# OpaqueSpec / DistributionSpec / FunctionSpec)
# ---------------------------------------------------------------------------


class TestValueSpecs:
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
            DistributionSpec(event_template=EventTemplate(x=())),
            FunctionSpec(input_template=EventTemplate(x=()), output_template=EventTemplate(y=())),
        ):
            with pytest.raises(FrozenInstanceError):
                spec.shape = (1,)  # type: ignore[misc]

    def test_specs_are_hashable(self):
        # Usable as dict keys / set members — required for treedef caching.
        specs = {
            ArraySpec((3,)): 1,
            OpaqueSpec(): 2,
            DistributionSpec(event_template=EventTemplate(x=())): 3,
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
        # Distinct-but-equal templates, so this pins value equality (a
        # shared object would also pass under identity-based equality).
        assert DistributionSpec(event_template=EventTemplate(x=())) == DistributionSpec(
            event_template=EventTemplate(x=())
        )
        assert FunctionSpec(
            input_template=EventTemplate(x=()), output_template=EventTemplate(y=())
        ) == FunctionSpec(input_template=EventTemplate(x=()), output_template=EventTemplate(y=()))

    def test_array_and_opaque_specs_are_distinct(self):
        assert ArraySpec(()) != OpaqueSpec()

    def test_value_spec_is_abstract_base(self):
        for cls in (ArraySpec, OpaqueSpec, DistributionSpec, FunctionSpec):
            assert issubclass(cls, ValueSpec)
        with pytest.raises(TypeError, match="abstract"):
            ValueSpec()  # type: ignore[abstract]

    def test_value_spec_exported_leaf_spec_removed(self):
        import probpipe

        assert probpipe.ValueSpec is ValueSpec
        assert "ValueSpec" in probpipe.__all__
        assert not hasattr(probpipe, "LeafSpec")
        assert "LeafSpec" not in probpipe.__all__

    def test_equal_specs_hash_equal(self):
        # ``ArraySpec.__hash__`` is hand-written; pin the eq/hash contract for
        # every spec kind, including a populated ArraySpec.
        from probpipe.core.constraints import positive

        inner_a, inner_b = EventTemplate(x=()), EventTemplate(x=())
        pairs = [
            (ArraySpec((3,)), ArraySpec((3,))),
            (
                ArraySpec((2,), dtype="float32", support=positive),
                ArraySpec((2,), dtype=jnp.float32, support=positive),
            ),
            (OpaqueSpec(meta="a"), OpaqueSpec(meta="a")),
            (DistributionSpec(event_template=inner_a), DistributionSpec(event_template=inner_b)),
        ]
        for a, b in pairs:
            assert a == b
            assert hash(a) == hash(b)

    def test_distribution_and_function_spec_inequality(self):
        # Distinct-but-equal templates compare equal; different templates
        # do not (the equality is by value, not object identity).
        assert DistributionSpec(event_template=EventTemplate(x=())) != DistributionSpec(
            event_template=EventTemplate(y=())
        )
        assert FunctionSpec(EventTemplate(a=()), EventTemplate(b=())) != FunctionSpec(
            EventTemplate(a=()), EventTemplate(c=())
        )

    def test_array_spec_unset_dtype_not_equal_to_set(self):
        # numpy treats ``np.dtype(None)`` as the default dtype, so a naive
        # field comparison would report these equal (while the eq/hash
        # contract requires equal objects to hash equal).
        assert ArraySpec(()) != ArraySpec((), dtype=jnp.float64)
        assert ArraySpec((), dtype=jnp.float64) != ArraySpec(())

    def test_array_spec_dtype_spellings_normalise(self):
        # Any numpy-coercible dtype spelling yields the same (equal, and
        # equal-hashing) spec.
        specs = [
            ArraySpec((), dtype="float32"),
            ArraySpec((), dtype=jnp.float32),
            ArraySpec((), dtype=np.dtype("float32")),
        ]
        assert len(set(specs)) == 1
        assert all(s.dtype == np.dtype("float32") for s in specs)

    def test_array_spec_pickle_round_trip(self):
        import pickle

        spec = ArraySpec((3,), dtype="float32")
        restored = pickle.loads(pickle.dumps(spec))
        assert restored == spec
        assert hash(restored) == hash(spec)
        assert restored.dtype == np.dtype("float32")

    def test_template_with_all_spec_kinds_pickle_round_trip(self):
        import pickle

        tpl = EventTemplate(
            x=ArraySpec((2,), dtype="float32"),
            label=OpaqueSpec(meta="tag"),
            d=DistributionSpec(event_template=EventTemplate(a=())),
            f=FunctionSpec(ArraySpec(()), ArraySpec(())),
        )
        restored = pickle.loads(pickle.dumps(tpl))
        assert restored == tpl
        assert hash(restored) == hash(tpl)

    def test_array_spec_zero_dim_allowed(self):
        spec = ArraySpec((0,))
        assert spec.shape == (0,)
        assert spec.is_valid(jnp.ones(0))
        assert not spec.is_valid(jnp.ones(1))

    def test_distribution_spec_requires_event_template(self):
        with pytest.raises(TypeError, match="must be an EventTemplate"):
            DistributionSpec(event_template=(3,))  # type: ignore[arg-type]
        # Unlike FunctionSpec, a bare ValueSpec is not auto-wrapped here.
        with pytest.raises(TypeError, match="must be an EventTemplate"):
            DistributionSpec(event_template=ArraySpec(()))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ValueSpec.is_valid — does a concrete value match the spec?
# ---------------------------------------------------------------------------


class TestArraySpecIsValid:
    def test_shape_match(self):
        spec = ArraySpec((3,))
        assert spec.is_valid(jnp.ones(3))
        assert spec.is_valid(np.ones(3))
        assert not spec.is_valid(jnp.ones(2))
        assert not spec.is_valid(jnp.ones((3, 1)))

    def test_scalar_shape(self):
        spec = ArraySpec(())
        assert spec.is_valid(1.5)
        assert spec.is_valid(2)
        assert spec.is_valid(True)
        assert spec.is_valid(jnp.asarray(1.0))
        assert not spec.is_valid(jnp.ones(1))

    def test_non_numeric_values_invalid(self):
        spec = ArraySpec(())
        assert not spec.is_valid("text")
        assert not spec.is_valid([1.0, 2.0])
        assert not spec.is_valid((1.0, 2.0))
        assert not spec.is_valid({"a": 1.0})
        assert not spec.is_valid(None)
        assert not spec.is_valid(np.asarray(["a"]))

    def test_dtype_checked_when_set(self):
        spec = ArraySpec((2,), dtype=jnp.float32)
        assert spec.is_valid(jnp.ones(2, dtype=jnp.float32))
        assert not spec.is_valid(jnp.ones(2, dtype=jnp.int32))

    def test_dtype_unset_accepts_any_numeric_dtype(self):
        spec = ArraySpec((2,))
        assert spec.is_valid(jnp.ones(2, dtype=jnp.float32))
        assert spec.is_valid(np.ones(2, dtype=np.int64))

    def test_python_scalar_dtype_is_numpy_default(self):
        # A bare Python scalar reports the dtype ``np.asarray`` gives it.
        assert ArraySpec((), dtype=np.asarray(1.0).dtype).is_valid(1.0)
        assert not ArraySpec((), dtype=jnp.int32).is_valid(1.0)

    def test_support_checked_when_set(self):
        from probpipe.core.constraints import positive

        spec = ArraySpec((2,), support=positive)
        assert spec.is_valid(jnp.asarray([1.0, 2.0]))
        assert not spec.is_valid(jnp.asarray([1.0, -2.0]))

    def test_support_checked_on_python_scalar(self):
        from probpipe.core.constraints import positive

        spec = ArraySpec((), support=positive)
        assert spec.is_valid(2.0)
        assert not spec.is_valid(-2.0)

    def test_support_vacuous_on_empty_array(self):
        from probpipe.core.constraints import positive

        assert ArraySpec((0,), support=positive).is_valid(jnp.ones(0))

    def test_dtype_and_support_together(self):
        # The checks compose: shape, then dtype, then support — each can
        # independently reject.
        from probpipe.core.constraints import positive

        spec = ArraySpec((2,), dtype=jnp.float32, support=positive)
        assert spec.is_valid(jnp.asarray([1.0, 2.0], dtype=jnp.float32))
        # np, not jnp: JAX without x64 would silently downcast to float32.
        assert not spec.is_valid(np.asarray([1.0, 2.0], dtype=np.float64))
        assert not spec.is_valid(jnp.asarray([1.0, -2.0], dtype=jnp.float32))

    def test_shape_dtype_path_is_jit_traceable(self):
        # The Notes claim: shape/dtype checks read only static (shape, dtype)
        # attributes, so a support-less spec runs under jax.jit without forcing
        # concretization — it returns rather than raising. (jit wraps the
        # Python bool result as a scalar Array, hence bool(...).)
        spec = ArraySpec((3,), dtype=jnp.float32)
        assert bool(jax.jit(spec.is_valid)(jnp.ones(3, dtype=jnp.float32))) is True
        assert bool(jax.jit(spec.is_valid)(jnp.ones(2, dtype=jnp.float32))) is False

    def test_support_path_not_jit_traceable(self):
        # The counter-claim: a support-carrying spec forces concretization of
        # a traced value (bool(jnp.all(...))), so it cannot run under trace.
        from probpipe.core.constraints import positive

        spec = ArraySpec((3,), support=positive)
        with pytest.raises(jax.errors.TracerBoolConversionError):
            jax.jit(spec.is_valid)(jnp.ones(3))


class TestOpaqueSpecIsValid:
    def test_non_mapping_objects_valid(self):
        spec = OpaqueSpec()
        assert spec.is_valid("label")
        assert spec.is_valid(object())
        assert spec.is_valid(None)
        assert spec.is_valid([1, 2, 3])

    def test_numeric_values_valid(self):
        # As the fallback spec, OpaqueSpec accepts any non-mapping value,
        # including numerics (though infer_from routes those to ArraySpec).
        spec = OpaqueSpec()
        assert spec.is_valid(1.5)
        assert spec.is_valid(jnp.ones(2))

    def test_mapping_invalid(self):
        # A mapping denotes tree structure (a subtree), never a leaf value —
        # the one thing OpaqueSpec rejects.
        assert not OpaqueSpec().is_valid({"a": 1})
        from types import MappingProxyType

        assert not OpaqueSpec().is_valid(MappingProxyType({"a": 1}))

    def test_dict_leaf_divergence_is_deliberate(self):
        # Interim divergence, pinned on purpose: Record still stores a plain
        # dict value as an opaque leaf, so the spec inferred FROM such a value
        # does not validate it. is_valid states the target contract (mappings
        # are structure, never leaves); the record layer aligns later.
        r = Record(x={"a": 1})
        spec = r.event_template["x"]
        assert spec == OpaqueSpec()
        assert not spec.is_valid(r["x"])

    def test_meta_not_checked(self):
        assert OpaqueSpec(meta="tag").is_valid("anything")


class TestDistributionSpecIsValid:
    def test_matching_distribution_valid(self):
        from probpipe import Normal

        dist = Normal(name="x", loc=0.0, scale=1.0)
        assert DistributionSpec(event_template=dist.event_template).is_valid(dist)

    def test_template_mismatch_invalid(self):
        from probpipe import Normal

        dist = Normal(name="x", loc=0.0, scale=1.0)
        assert not DistributionSpec(event_template=EventTemplate(y=())).is_valid(dist)

    def test_non_distribution_invalid(self):
        spec = DistributionSpec(event_template=EventTemplate(x=()))
        assert not spec.is_valid(42)
        assert not spec.is_valid(EventTemplate(x=()))

    def test_distribution_without_template_invalid(self):
        # A distribution always carries the schema of its draws; one that
        # exposes no event template cannot satisfy any DistributionSpec.
        from probpipe.core._distribution_base import Distribution

        class _NoTemplate(Distribution):
            def __init__(self):
                super().__init__(name="d")

        spec = DistributionSpec(event_template=EventTemplate(x=()))
        assert not spec.is_valid(_NoTemplate())

    def test_distribution_with_none_template_invalid(self):
        from probpipe.core._distribution_base import Distribution

        class _NoneTemplate(Distribution):
            def __init__(self):
                super().__init__(name="d")

            @property
            def event_template(self):
                return None

        spec = DistributionSpec(event_template=EventTemplate(x=()))
        assert not spec.is_valid(_NoneTemplate())

    @pytest.mark.parametrize("error", [TypeError, RuntimeError, ValueError])
    def test_raising_template_property_returns_false(self, error):
        # An ``event_template`` property that fails to run — whatever it
        # raises — leaves the schema unknown; is_valid must return False
        # rather than propagate (the probe is total).
        from probpipe.core._distribution_base import Distribution

        class _RaisingTemplate(Distribution):
            def __init__(self):
                super().__init__(name="d")

            @property
            def event_template(self):
                raise error("template not derivable")

        spec = DistributionSpec(event_template=EventTemplate(x=()))
        assert not spec.is_valid(_RaisingTemplate())


class TestFunctionSpecIsValid:
    def test_callable_valid(self):
        spec = FunctionSpec(input_template=EventTemplate(a=()), output_template=EventTemplate(b=()))
        assert spec.is_valid(lambda a: a)
        assert spec.is_valid(np.sin)

    def test_non_callable_invalid(self):
        spec = FunctionSpec(input_template=EventTemplate(a=()), output_template=EventTemplate(b=()))
        assert not spec.is_valid(3.0)
        assert not spec.is_valid("f")


# ---------------------------------------------------------------------------
# FunctionSpec — bare-spec convenience
# ---------------------------------------------------------------------------


class TestFunctionSpecBareSpec:
    def test_bare_specs_wrap_in_single_field_templates(self):
        spec = FunctionSpec(ArraySpec(()), ArraySpec((2,)))
        assert spec.input_template == EventTemplate(input=ArraySpec(()))
        assert spec.output_template == EventTemplate(output=ArraySpec((2,)))
        # The stored attributes are always EventTemplates, whatever the
        # constructor input form.
        assert isinstance(spec.input_template, EventTemplate)
        assert isinstance(spec.output_template, EventTemplate)

    def test_bare_spec_on_one_side_only(self):
        out = EventTemplate(y=(), z=(3,))
        spec = FunctionSpec(OpaqueSpec(), out)
        assert spec.input_template == EventTemplate(input=OpaqueSpec())
        assert spec.output_template is out

    def test_templates_pass_through_unwrapped(self):
        inp, out = EventTemplate(a=()), EventTemplate(b=())
        spec = FunctionSpec(inp, out)
        assert spec.input_template is inp
        assert spec.output_template is out

    def test_invalid_side_rejected(self):
        with pytest.raises(TypeError, match="input_template"):
            FunctionSpec((3,), EventTemplate(b=()))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="output_template"):
            FunctionSpec(EventTemplate(a=()), "not a template")  # type: ignore[arg-type]

    def test_wrapped_spec_equality_and_hash(self):
        # The wrapped form and the explicit single-field form are one spec.
        a = FunctionSpec(ArraySpec(()), ArraySpec(()))
        b = FunctionSpec(EventTemplate(input=ArraySpec(())), EventTemplate(output=ArraySpec(())))
        assert a == b
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# FunctionSpec — optional (None) templates: "some callable, structure unknown"
# ---------------------------------------------------------------------------


class TestFunctionSpecOptionalTemplates:
    def test_bare_function_spec_is_any_callable(self):
        spec = FunctionSpec()
        assert spec.input_template is None
        assert spec.output_template is None
        assert spec.is_valid(lambda x: x)
        assert spec.is_valid(np.sin)
        assert not spec.is_valid(3.0)

    def test_one_side_specified(self):
        spec = FunctionSpec(output_template=ArraySpec(()))
        assert spec.input_template is None
        assert spec.output_template == EventTemplate(output=ArraySpec(()))

    def test_none_specs_are_hashable_and_equal(self):
        assert FunctionSpec() == FunctionSpec()
        assert hash(FunctionSpec()) == hash(FunctionSpec())
        # A template-less spec differs from a typed one.
        assert FunctionSpec() != FunctionSpec(ArraySpec(()), ArraySpec(()))

    def test_none_spec_usable_as_template_leaf(self):
        # A FunctionSpec leaf (with unspecified signature) lives in a template
        # and blocks numeric auto-promotion like any non-array leaf.
        tpl = EventTemplate(f=FunctionSpec(), x=())
        assert tpl["f"] == FunctionSpec()
        assert type(tpl) is EventTemplate

    def test_none_spec_pickle_round_trip(self):
        import pickle

        spec = FunctionSpec()
        restored = pickle.loads(pickle.dumps(spec))
        assert restored == spec
        assert hash(restored) == hash(spec)


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
        assert tpl.at_path("sub") is inner

    def test_explicit_array_spec_accepted(self):
        spec = ArraySpec((2,), dtype="float32")
        tpl = EventTemplate(x=spec)
        assert tpl["x"] is spec

    def test_explicit_opaque_spec_accepted(self):
        spec = OpaqueSpec(meta="tag")
        tpl = EventTemplate(label=spec, x=())
        assert tpl["label"] is spec

    def test_explicit_distribution_and_function_specs_accepted(self):
        dspec = DistributionSpec(event_template=EventTemplate(x=()))
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
        tpl = EventTemplate(x=(), d=DistributionSpec(event_template=EventTemplate(a=())))
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
            NumericEventTemplate(x=(), d=DistributionSpec(event_template=EventTemplate(a=())))

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
# Numeric queries & projection: is_numeric / is_multi_field / numeric_subset
# ---------------------------------------------------------------------------


def _dist_spec() -> DistributionSpec:
    return DistributionSpec(event_template=EventTemplate(a=()))


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
        assert isinstance(sub.at_path("nested"), NumericEventTemplate)
        assert tuple(sub.at_path("nested").children) == ("a", "b")

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
        assert isinstance(v.at_path("nested"), NumericRecordArray)
        assert v["nested/b"].shape == (2, 3, 2)
        assert tpl.from_vector(tpl.to_vector(v)) == v


class TestFromVectorErrors:
    def test_wrong_trailing_size_raises(self):
        tpl = EventTemplate(x=(), y=(3,))
        with pytest.raises(ValueError, match="vector_size"):
            tpl.from_vector(jnp.zeros(5))
