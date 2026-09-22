"""Tests for probpipe.core.record.RecordSpec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, get_type_hints

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Function, NumericRecord, Record
from probpipe.core._batch import BatchSpec
from probpipe.core._numeric_record_batch import NumericRecordBatch
from probpipe.core._opaque import OpaqueSpec
from probpipe.core._opaque_batch import OpaqueBatch
from probpipe.core._record_spec import (
    _concretize_event_template,
    _unify_event_template_with_value,
)
from probpipe.core._specs import (
    DistributionSpec,
    FunctionSpec,
    NumericArraySpec,
    NumericRecordSpec,
    NumericSpec,
    RecordSpec,
    TermSpec,
)


@dataclass(frozen=True)
class _UnhashableTermSpec(TermSpec):
    metadata: list[str]

    def is_valid(self, value: Any) -> bool:
        return True


@dataclass(frozen=True)
class _TaggedTermSpec(TermSpec):
    tag: str

    def is_valid(self, value: Any) -> bool:
        return True


# ---------------------------------------------------------------------------
# Path separator
# ---------------------------------------------------------------------------


def test_path_separator_is_slash():
    """Docstrings spell the nested-path separator literally as ``/``.

    Pin the constant so changing it trips CI and forces a conscious sweep of
    the docstrings that hardcode the character.
    """
    from probpipe.core.named_tree import _PATH_SEP

    assert _PATH_SEP == "/"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_kwargs(self):
        tpl = RecordSpec(x=(), y=(3,))
        assert tpl.fields == ("x", "y")

    def test_dict_positional(self):
        tpl = RecordSpec({"a": (), "b": (2,)})
        assert tpl.fields == ("a", "b")

    def test_fields_insertion_order(self):
        tpl = RecordSpec(z=(), a=(3,), m=None)
        assert tpl.fields == ("z", "a", "m")

    @pytest.mark.parametrize("empty_type", [RecordSpec, NumericRecordSpec])
    @pytest.mark.parametrize("with_leaf", [False, True])
    def test_copy_constructor_preserves_empty_subtrees_and_numeric_kind(
        self, empty_type, with_leaf
    ):
        empty = empty_type()
        fields = {"x": ()} if with_leaf else {}
        original = RecordSpec(fields | {"empty": empty})

        copied = RecordSpec(original)

        assert type(copied) is type(original)
        assert copied == original
        assert tuple(copied.children) == tuple(original.children)
        assert copied.children["empty"] is original.children["empty"]

    def test_slash_in_field_name_rejected(self):
        with pytest.raises(ValueError, match="must not contain '/'"):
            RecordSpec(**{"a/b": ()})

    def test_dict_and_kwargs_raises(self):
        with pytest.raises(ValueError, match="Cannot pass both"):
            RecordSpec({"a": ()}, b=(2,))

    def test_the_empty_template_is_legal(self):
        """A template is a finite map of fields, and the empty map is one.

        It is also the identity of template composition, which a floor of one
        field would leave inexpressible.
        """
        empty = RecordSpec()

        assert len(empty) == 0
        assert list(empty) == []

    def test_the_empty_template_is_not_promoted_to_numeric(self):
        """Vacuously every leaf is numeric, which is not a reason to claim it."""
        assert not isinstance(RecordSpec(), NumericRecordSpec)

    def test_none_spec(self):
        tpl = RecordSpec(label=None, x=())
        assert tpl["label"] == OpaqueSpec()
        assert tpl["x"] == NumericArraySpec(())

    def test_nested(self):
        inner = RecordSpec(force=(), mass=())
        outer = RecordSpec(physics=inner, obs=())
        assert isinstance(outer.at_path("physics"), RecordSpec)
        assert outer["physics/force"] == NumericArraySpec(())

    def test_invalid_spec_raises(self):
        with pytest.raises(TypeError, match="spec must be"):
            RecordSpec(x=3.0)

    def test_invalid_shape_raises(self):
        with pytest.raises(TypeError, match="non-negative ints"):
            RecordSpec(x=(-1,))

    def test_invalid_shape_float_raises(self):
        with pytest.raises(TypeError, match="non-negative ints"):
            RecordSpec(x=(1.5,))


# ---------------------------------------------------------------------------
# Field access
# ---------------------------------------------------------------------------


class TestFieldAccess:
    @pytest.fixture
    def tpl(self):
        return RecordSpec(a=(), b=(3,), c=(2, 4))

    def test_getitem(self, tpl):
        assert tpl["a"] == NumericArraySpec(())
        assert tpl["b"] == NumericArraySpec((3,))
        assert tpl["c"] == NumericArraySpec((2, 4))

    def test_contains(self, tpl):
        assert "a" in tpl
        assert "z" not in tpl

    def test_len(self, tpl):
        assert len(tpl) == 3

    def test_missing_key_raises(self, tpl):
        with pytest.raises(KeyError):
            tpl["nonexistent"]


class TestNamedTreeSurfaceOnRecordSpec:
    """RecordSpec gained the shared ``NamedTree`` collection protocol:
    ``/``-path and tuple indexing, path membership, iteration, and
    ``keys``/``values``/``items`` — over its value specs, keyed by leaf path.
    Exercised on a *nested* template (the flat case is covered above).
    """

    @pytest.fixture
    def nested(self):
        return RecordSpec(theta=RecordSpec(loc=(2,), scale=()), sigma=(3,))

    def test_path_indexing_returns_leaf_spec(self, nested):
        assert nested["theta/loc"] == NumericArraySpec((2,))
        assert nested["theta/scale"] == NumericArraySpec(())
        assert nested["sigma"] == NumericArraySpec((3,))

    def test_tuple_indexing_matches_slash_path(self, nested):
        assert nested["theta", "loc"] == nested["theta/loc"]

    def test_partial_path_navigates_to_subtree(self, nested):
        # A partial path reaches a subtree via at_path; [] is leaf-only.
        assert nested.at_path("theta") == RecordSpec(loc=(2,), scale=())
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
        assert values == [NumericArraySpec((2,)), NumericArraySpec(()), NumericArraySpec((3,))]
        assert list(nested.items()) == list(zip(nested.keys(), values))


# ---------------------------------------------------------------------------
# Leaf shapes
# ---------------------------------------------------------------------------


class TestLeafShapes:
    def test_flat_fields(self):
        tpl = RecordSpec(x=(), y=(3,))
        assert tpl.leaf_shapes == {"x": (), "y": (3,)}

    def test_nested_flattens(self):
        inner = RecordSpec(a=(), b=(2,))
        outer = RecordSpec(inner=inner, z=(3,))
        shapes = outer.leaf_shapes
        # Slash-delimited keys for consistency with ``Record["a/b"]``
        # path access.
        assert shapes == {"inner/a": (), "inner/b": (2,), "z": (3,)}

    def test_leaf_shapes_on_numeric_template(self):
        tpl = NumericRecordSpec(x=(), y=(3,))
        assert tpl.leaf_shapes == {"x": (), "y": (3,)}

    def test_leaf_shapes_not_on_base_template(self):
        """``vector_size`` / ``leaf_shapes`` are only meaningful
        when every leaf is numeric — they live on
        :class:`NumericRecordSpec`, not the base ``RecordSpec``.
        """
        tpl = RecordSpec(label=None, x=(), y=(3,))
        assert not hasattr(tpl, "leaf_shapes")
        assert not hasattr(tpl, "vector_size")


# ---------------------------------------------------------------------------
# keys() (canonical leaf order — the single source of truth)
# ---------------------------------------------------------------------------


class TestKeys:
    def test_flat_equals_fields(self):
        # For a flat template (every field a leaf), keys() == fields.
        tpl = RecordSpec(x=(), y=(3,))
        assert tuple(tpl.keys()) == ("x", "y") == tpl.fields

    def test_includes_opaque_leaves(self):
        # keys() enumerates every leaf, numeric or opaque.
        tpl = RecordSpec(label=None, x=())
        assert tuple(tpl.keys()) == ("label", "x")

    def test_nested_depth_first_insertion_order(self):
        # A nested field expands into one key per nested leaf; fields stays
        # top-level only.
        inner = RecordSpec(a=(), b=(2,))
        outer = RecordSpec(inner=inner, z=(3,))
        assert tuple(outer.keys()) == ("inner/a", "inner/b", "z")
        assert outer.fields == ("inner", "z")

    def test_depth2(self):
        tpl = RecordSpec(outer=RecordSpec(deep=RecordSpec(g=(), h=()), a=()), m=())
        assert tuple(tpl.keys()) == ("outer/deep/g", "outer/deep/h", "outer/a", "m")

    def test_keys_match_leaf_shapes_in_order(self):
        # leaf_shapes (numeric template) is keyed by keys(), same order.
        tpl = RecordSpec(outer=RecordSpec(a=(2,), b=()), m=())
        assert tuple(tpl.leaf_shapes) == tuple(tpl.keys())

    def test_order_matches_flatten_and_to_vector(self):
        # The canonical leaf order is the order flatten() / to_vector() use.
        v = NumericRecord(
            "nr",
            x=jnp.array([1.0, 2.0]),
            nested=NumericRecord("nr", a=jnp.array(3.0), b=jnp.array([4.0, 5.0])),
        )
        tpl = RecordSpec.infer_from(v)
        assert tuple(tpl.keys()) == ("x", "nested/a", "nested/b")
        # to_vector concatenates leaves in keys() order: x(2) | a(1) | b(2).
        np.testing.assert_allclose(v.to_vector(), [1.0, 2.0, 3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# vector_size (on NumericRecordSpec)
# ---------------------------------------------------------------------------


class TestFlatSize:
    def test_scalars(self):
        tpl = NumericRecordSpec(a=(), b=(), c=())
        assert tpl.vector_size == 3

    def test_arrays(self):
        tpl = NumericRecordSpec(x=(5,), y=(2, 3))
        assert tpl.vector_size == 11

    def test_nested(self):
        inner = NumericRecordSpec(r=(), K=())
        outer = NumericRecordSpec(params=inner, obs=(4,))
        assert outer.vector_size == 6

    def test_scalar_only(self):
        tpl = NumericRecordSpec(a=())
        assert tpl.vector_size == 1

    def test_symbolic_template_raises(self):
        tpl = NumericRecordSpec(x=("obs", 3), y=("obs",))

        assert tpl.free_dims == frozenset({"obs"})
        assert not tpl.is_concrete
        with pytest.raises(ValueError, match="unbound dimensions: obs"):
            _ = tpl.vector_size

    def test_rejects_opaque_leaf(self):
        with pytest.raises(TypeError, match="only NumericArraySpec"):
            NumericRecordSpec(label=None, x=(3,))

    def test_rejects_non_numeric_nested(self):
        # ``RecordSpec(x=(), label=None)`` stays a plain base template
        # (mixed leaves block auto-promotion), so embedding it inside a
        # ``NumericRecordSpec`` must be rejected.
        inner = RecordSpec(x=(), label=None)
        with pytest.raises(TypeError, match="NumericRecordSpec"):
            NumericRecordSpec(nested=inner, y=())


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_setattr_raises(self):
        tpl = RecordSpec(x=())
        with pytest.raises(AttributeError, match="immutable"):
            tpl.x = (3,)

    def test_delattr_raises(self):
        tpl = RecordSpec(x=())
        with pytest.raises(AttributeError, match="immutable"):
            del tpl.x


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------


class TestEqualityAndHashing:
    def test_equal(self):
        t1 = RecordSpec(x=(), y=(3,))
        t2 = RecordSpec(x=(), y=(3,))
        assert t1 == t2

    def test_not_equal_shapes(self):
        t1 = RecordSpec(x=())
        t2 = RecordSpec(x=(3,))
        assert t1 != t2

    def test_not_equal_fields(self):
        t1 = RecordSpec(x=())
        t2 = RecordSpec(y=())
        assert t1 != t2

    def test_not_equal_to_other_types(self):
        tpl = RecordSpec(x=())
        assert tpl != "not a template"

    def test_hash_equal(self):
        t1 = RecordSpec(x=(), y=(3,))
        t2 = RecordSpec(x=(), y=(3,))
        assert hash(t1) == hash(t2)

    def test_hash_usable_in_set(self):
        t1 = RecordSpec(x=(), y=(3,))
        t2 = RecordSpec(x=(), y=(3,))
        assert len({t1, t2}) == 1

    def test_nested_equality(self):
        inner = RecordSpec(a=(), b=())
        t1 = RecordSpec(sub=inner, z=())
        t2 = RecordSpec(sub=RecordSpec(a=(), b=()), z=())
        assert t1 == t2
        assert hash(t1) == hash(t2)

    def test_eq_is_order_sensitive(self):
        """Insertion-order is part of the template's identity (#124),
        and ``__hash__`` is order-sensitive — so ``__eq__`` must agree
        to satisfy Python's eq/hash contract.
        """
        t1 = RecordSpec(a=(), b=(2,))
        t2 = RecordSpec(b=(2,), a=())
        assert t1 != t2
        assert hash(t1) != hash(t2)
        # And the contract holds: equal templates hash the same.
        t3 = RecordSpec(a=(), b=(2,))
        assert t1 == t3
        assert hash(t1) == hash(t3)


# ---------------------------------------------------------------------------
# infer_from factory
# ---------------------------------------------------------------------------


class TestInferFrom:
    def test_scalar_fields(self):
        r = Record("r", a=1.0, b=2.0)
        tpl = RecordSpec.infer_from(r)
        assert tpl.fields == ("a", "b")
        assert tpl["a"] == NumericArraySpec(())
        assert tpl["b"] == NumericArraySpec(())

    def test_nested_mapping_is_structure_not_opaque(self):
        # A mapping is never a leaf: a nested dict value is inferred as a
        # nested template (structure), not an opaque leaf spec — otherwise the
        # inferred OpaqueSpec would reject the very dict it came from.
        tpl = RecordSpec.infer_from({"cfg": {"lr": 0.1}, "x": 2.0})
        assert tuple(tpl.keys()) == ("cfg/lr", "x")
        assert not isinstance(tpl["cfg/lr"], OpaqueSpec)

    def test_mapping_input_inferred_field_by_field(self):
        # The non-Record branch: a bare mapping is inferred field by field
        # (a nested Record contributes its own event_template).
        tpl = RecordSpec.infer_from(
            {"a": 1.0, "x": jnp.zeros(3), "params": Record("r", m=jnp.zeros(2))}
        )
        assert tuple(tpl.children) == ("a", "x", "params")
        assert tpl["a"] == NumericArraySpec(())
        assert tpl["x"] == NumericArraySpec((3,))
        assert isinstance(tpl.at_path("params"), RecordSpec)
        assert tpl["params/m"] == NumericArraySpec((2,))

    def test_an_empty_mapping_infers_the_empty_template(self):
        assert len(RecordSpec.infer_from({})) == 0

    def test_array_fields(self):
        r = Record("r", x=jnp.zeros(5), y=jnp.zeros((2, 3)))
        tpl = RecordSpec.infer_from(r)
        assert tpl["x"] == NumericArraySpec((5,))
        assert tpl["y"] == NumericArraySpec((2, 3))

    def test_nested_record(self):
        inner = Record("r", x=1.0, y=jnp.zeros(3))
        outer = Record("r", params=inner, z=2.0)
        tpl = RecordSpec.infer_from(outer)
        assert isinstance(tpl.at_path("params"), RecordSpec)
        assert tpl["params/x"] == NumericArraySpec(())
        assert tpl["params/y"] == NumericArraySpec((3,))
        assert tpl["z"] == NumericArraySpec(())

    def test_roundtrip_vector_size(self):
        from probpipe.core._numeric_record import NumericRecord

        r = NumericRecord("nr", a=1.0, b=jnp.zeros(4), c=jnp.zeros((2, 3)))
        tpl = RecordSpec.infer_from(r)
        # Auto-promoted to NumericRecordSpec because the input was a
        # NumericRecord, so ``vector_size`` is reachable.
        assert isinstance(tpl, NumericRecordSpec)
        assert tpl.vector_size == r.vector_size

    def test_from_numeric_record_promotes(self):
        """Calling ``infer_from`` on a ``NumericRecord`` returns a
        :class:`NumericRecordSpec`, even through the base
        ``RecordSpec.infer_from`` classmethod, so downstream code
        that needs ``vector_size`` keeps working without the caller having
        to name the subclass explicitly."""
        from probpipe.core._numeric_record import NumericRecord

        r = NumericRecord("nr", a=1.0, b=jnp.zeros(2))
        tpl = RecordSpec.infer_from(r)
        assert isinstance(tpl, NumericRecordSpec)

    def test_from_mixed_record_stays_base(self):
        """A plain ``Record`` with a non-numeric leaf can't be promoted —
        the result is a plain :class:`RecordSpec` with an opaque
        slot."""
        r = Record("r", x=1.0, label="tag")
        tpl = RecordSpec.infer_from(r)
        assert type(tpl) is RecordSpec
        assert tpl["label"] == OpaqueSpec()

    def test_list_leaf_is_opaque(self):
        """A Python list leaf has no .shape / .dtype, so the field is
        recorded as opaque (``None``) even when it contains numbers.
        Users should wrap lists in np.asarray/jnp.asarray for a numeric
        template entry — this test pins down that behavior so the
        documented guidance stays in sync with the implementation."""
        r = Record("r", xs=[1.0, 2.0, 3.0])
        tpl = RecordSpec.infer_from(r)
        assert tpl["xs"] == OpaqueSpec()

    def test_list_leaf_after_asarray_is_numeric(self):
        """The opposite end of the list-leaf story: wrapping the list
        in ``np.asarray`` produces a numeric template entry."""

        r = Record("r", xs=np.asarray([1.0, 2.0, 3.0]))
        tpl = RecordSpec.infer_from(r)
        assert tpl["xs"] == NumericArraySpec((3,))


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_simple(self):
        # All-numeric → auto-promotes to NumericRecordSpec.
        tpl = RecordSpec(x=(), y=(3,))
        r = repr(tpl)
        assert r.startswith("NumericRecordSpec(")
        assert "x=()" in r
        assert "y=(3,)" in r

    def test_nested(self):
        inner = RecordSpec(a=())
        outer = RecordSpec(sub=inner, z=(2,))
        r = repr(outer)
        # Both auto-promote; inner's repr is nested under the outer.
        assert "sub=NumericRecordSpec(" in r

    def test_mixed_stays_base(self):
        tpl = RecordSpec(label=None, x=())
        assert repr(tpl).startswith("RecordSpec(")

    def test_opaque(self):
        tpl = RecordSpec(label=None, x=())
        assert "label=None" in repr(tpl)

    def test_populated_numeric_array_spec_shows_full_repr(self):
        # A spec carrying dtype/support is not bare, so repr falls back to the
        # full dataclass repr rather than the bare-shape shorthand. The dtype
        # renders in its normalised ``numpy.dtype`` form.
        tpl = RecordSpec(x=NumericArraySpec((3,), dtype="float32"))
        r = repr(tpl)
        assert "NumericArraySpec(" in r
        assert "dtype=dtype('float32')" in r

    def test_populated_opaque_spec_shows_full_repr(self):
        tpl = RecordSpec(label=OpaqueSpec(meta="tag"), x=())
        r = repr(tpl)
        assert "OpaqueSpec(meta='tag')" in r


# ---------------------------------------------------------------------------
# Value specs — the TermSpec base and its concrete subclasses (NumericArraySpec /
# OpaqueSpec / DistributionSpec / FunctionSpec)
# ---------------------------------------------------------------------------


class TestTermSpecs:
    def test_numeric_array_spec_defaults(self):
        spec = NumericArraySpec((3,))
        assert spec.shape == (3,)
        assert spec.dtype is None
        assert spec.support is None

    def test_numeric_array_spec_coerces_shape_to_tuple(self):
        # A list shape is normalised to a tuple so the spec stays hashable.
        spec = NumericArraySpec([2, 4])
        assert spec.shape == (2, 4)
        assert isinstance(spec.shape, tuple)

    def test_numeric_array_spec_rejects_negative_dims(self):
        with pytest.raises(TypeError, match="non-negative ints"):
            NumericArraySpec((-1,))

    def test_specs_are_frozen(self):

        for spec in (
            NumericArraySpec((3,)),
            OpaqueSpec(),
            RecordSpec(x=()),
            DistributionSpec(event_spec=RecordSpec(x=())),
            FunctionSpec(input_template=RecordSpec(x=()), output_spec=RecordSpec(y=())),
        ):
            with pytest.raises(AttributeError):
                spec.shape = (1,)  # type: ignore[misc]

    def test_specs_are_hashable(self):
        # Usable as dict keys / set members — required for treedef caching.
        specs = {
            NumericArraySpec((3,)): 1,
            OpaqueSpec(): 2,
            DistributionSpec(event_spec=RecordSpec(x=())): 3,
            FunctionSpec(input_template=RecordSpec(x=()), output_spec=RecordSpec(y=())): 4,
            RecordSpec(x=()): 5,
        }
        assert len(specs) == 5

    def test_opaque_spec_rejects_unhashable_meta_at_construction(self):
        with pytest.raises(TypeError, match=r"OpaqueSpec\.meta must be hashable"):
            OpaqueSpec(meta=[])  # type: ignore[arg-type]

    def test_numeric_array_spec_rejects_unhashable_support_at_construction(self):
        with pytest.raises(TypeError, match=r"NumericArraySpec\.support must be hashable"):
            NumericArraySpec((), support=[])  # type: ignore[arg-type]

    def test_template_rejects_unhashable_custom_value_spec_at_construction(self):
        spec = _UnhashableTermSpec(metadata=["mutable"])

        with pytest.raises(TypeError, match=r"Field 'custom' spec must be hashable"):
            RecordSpec(custom=spec)

    def test_template_accepts_hashable_custom_value_spec(self):
        spec = _TaggedTermSpec(tag="custom")

        template = RecordSpec(custom=spec)

        assert template["custom"] is spec
        assert hash(template) == hash(RecordSpec(custom=_TaggedTermSpec(tag="custom")))

    def test_specs_value_equality(self):
        assert NumericArraySpec((3,)) == NumericArraySpec((3,))
        assert NumericArraySpec((3,), dtype="float32") == NumericArraySpec((3,), dtype="float32")
        assert NumericArraySpec((3,)) != NumericArraySpec((2,))
        assert NumericArraySpec((3,)) != NumericArraySpec((3,), dtype="float32")
        assert OpaqueSpec() == OpaqueSpec()
        assert OpaqueSpec(meta="a") == OpaqueSpec(meta="a")
        assert OpaqueSpec(meta="a") != OpaqueSpec(meta="b")
        # Distinct-but-equal templates, so this pins value equality (a
        # shared object would also pass under identity-based equality).
        assert DistributionSpec(event_spec=RecordSpec(x=())) == DistributionSpec(
            event_spec=RecordSpec(x=())
        )
        assert FunctionSpec(
            input_template=RecordSpec(x=()), output_spec=RecordSpec(y=())
        ) == FunctionSpec(input_template=RecordSpec(x=()), output_spec=RecordSpec(y=()))

    def test_array_and_opaque_specs_are_distinct(self):
        assert NumericArraySpec(()) != OpaqueSpec()

    def test_value_spec_is_abstract_base(self):
        for cls in (NumericArraySpec, OpaqueSpec, DistributionSpec, FunctionSpec):
            assert issubclass(cls, TermSpec)
        with pytest.raises(TypeError, match="abstract"):
            TermSpec()  # type: ignore[abstract]

    def test_value_spec_exported_leaf_spec_removed(self):
        import probpipe

        assert probpipe.TermSpec is TermSpec
        assert "TermSpec" in probpipe.__all__
        assert not hasattr(probpipe, "LeafSpec")
        assert "LeafSpec" not in probpipe.__all__

    def test_equal_specs_hash_equal(self):
        # ``NumericArraySpec.__hash__`` is hand-written; pin the eq/hash contract for
        # every spec kind, including a populated NumericArraySpec.
        from probpipe.core.constraints import positive

        inner_a, inner_b = RecordSpec(x=()), RecordSpec(x=())
        pairs = [
            (NumericArraySpec((3,)), NumericArraySpec((3,))),
            (
                NumericArraySpec((2,), dtype="float32", support=positive),
                NumericArraySpec((2,), dtype=jnp.float32, support=positive),
            ),
            (OpaqueSpec(meta="a"), OpaqueSpec(meta="a")),
            (DistributionSpec(event_spec=inner_a), DistributionSpec(event_spec=inner_b)),
        ]
        for a, b in pairs:
            assert a == b
            assert hash(a) == hash(b)

    def test_distribution_and_function_spec_inequality(self):
        # Distinct-but-equal templates compare equal; different templates
        # do not (the equality is by value, not object identity).
        assert DistributionSpec(event_spec=RecordSpec(x=())) != DistributionSpec(
            event_spec=RecordSpec(y=())
        )
        assert FunctionSpec(RecordSpec(a=()), RecordSpec(b=())) != FunctionSpec(
            RecordSpec(a=()), RecordSpec(c=())
        )

    def test_numeric_array_spec_unset_dtype_not_equal_to_set(self):
        # numpy treats ``np.dtype(None)`` as the default dtype, so a naive
        # field comparison would report these equal (while the eq/hash
        # contract requires equal objects to hash equal).
        assert NumericArraySpec(()) != NumericArraySpec((), dtype=jnp.float64)
        assert NumericArraySpec((), dtype=jnp.float64) != NumericArraySpec(())

    def test_numeric_array_spec_dtype_spellings_normalise(self):
        # Any numpy-coercible dtype spelling yields the same (equal, and
        # equal-hashing) spec.
        specs = [
            NumericArraySpec((), dtype="float32"),
            NumericArraySpec((), dtype=jnp.float32),
            NumericArraySpec((), dtype=np.dtype("float32")),
        ]
        assert len(set(specs)) == 1
        assert all(s.dtype == np.dtype("float32") for s in specs)

    def test_numeric_array_spec_pickle_round_trip(self):
        import pickle

        spec = NumericArraySpec((3,), dtype="float32")
        restored = pickle.loads(pickle.dumps(spec))
        assert restored == spec
        assert hash(restored) == hash(spec)
        assert restored.dtype == np.dtype("float32")

    def test_template_with_all_spec_kinds_pickle_round_trip(self):
        import pickle

        tpl = RecordSpec(
            x=NumericArraySpec((2,), dtype="float32"),
            label=OpaqueSpec(meta="tag"),
            d=DistributionSpec(event_spec=RecordSpec(a=())),
            f=FunctionSpec(
                RecordSpec(inp=NumericArraySpec(())), RecordSpec(out=NumericArraySpec(()))
            ),
            r=RecordSpec(c=NumericArraySpec(())),
        )
        restored = pickle.loads(pickle.dumps(tpl))
        assert restored == tpl
        assert hash(restored) == hash(tpl)

    def test_numeric_array_spec_zero_dim_allowed(self):
        spec = NumericArraySpec((0,))
        assert spec.shape == (0,)
        assert spec.is_valid(jnp.ones(0))
        assert not spec.is_valid(jnp.ones(1))

    def test_numeric_array_spec_symbolic_dimensions(self):
        spec = NumericArraySpec(("obs", 3, "obs"))

        assert spec.is_valid(np.zeros((4, 3, 4)))
        assert not spec.is_valid(np.zeros((4, 3, 5)))

    @pytest.mark.parametrize("dimension", ["", -1, 1.5, None])
    def test_numeric_array_spec_rejects_invalid_symbolic_dimensions(self, dimension):
        with pytest.raises(TypeError, match="symbolic dimension"):
            NumericArraySpec((dimension,))

    def test_distribution_spec_requires_record_declaration(self):
        with pytest.raises(TypeError, match="must be a RecordSpec"):
            DistributionSpec(event_spec=(3,))  # type: ignore[arg-type]
        # The current distribution declaration remains record-valued.
        with pytest.raises(TypeError, match="must be a RecordSpec"):
            DistributionSpec(event_spec=NumericArraySpec(()))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TermSpec.is_valid — does a concrete value match the spec?
# ---------------------------------------------------------------------------


class TestNumericArraySpecIsValid:
    def test_shape_match(self):
        spec = NumericArraySpec((3,))
        assert spec.is_valid(jnp.ones(3))
        assert spec.is_valid(np.ones(3))
        assert not spec.is_valid(jnp.ones(2))
        assert not spec.is_valid(jnp.ones((3, 1)))

    def test_scalar_shape(self):
        spec = NumericArraySpec(())
        assert spec.is_valid(1.5)
        assert spec.is_valid(2)
        assert spec.is_valid(True)
        assert spec.is_valid(jnp.asarray(1.0))
        assert not spec.is_valid(jnp.ones(1))

    def test_non_numeric_values_invalid(self):
        spec = NumericArraySpec(())
        assert not spec.is_valid("text")
        assert not spec.is_valid([1.0, 2.0])
        assert not spec.is_valid((1.0, 2.0))
        assert not spec.is_valid({"a": 1.0})
        assert not spec.is_valid(None)
        assert not spec.is_valid(np.asarray(["a"]))

    def test_dtype_checked_when_set(self):
        spec = NumericArraySpec((2,), dtype=jnp.float32)
        assert spec.is_valid(jnp.ones(2, dtype=jnp.float32))
        # A same-kind cast satisfies the spec (numpy treats int->float as
        # same_kind); a cross-kind cast (a float value against an int-dtype
        # spec) does not.
        assert spec.is_valid(jnp.ones(2, dtype=jnp.int32))
        assert not NumericArraySpec((2,), dtype=jnp.int32).is_valid(jnp.ones(2, dtype=jnp.float32))

    def test_dtype_unset_accepts_any_numeric_dtype(self):
        spec = NumericArraySpec((2,))
        assert spec.is_valid(jnp.ones(2, dtype=jnp.float32))
        assert spec.is_valid(np.ones(2, dtype=np.int64))

    def test_ml_dtypes_are_numeric(self):
        # bfloat16 / float8 report numpy kind "V" but are numeric JAX arrays;
        # they satisfy NumericArraySpec (shape-only and dtype-pinned). infer_from
        # therefore routes them to the array side (see the infer_from test).
        bf16 = jnp.ones(2, dtype=jnp.bfloat16)
        assert NumericArraySpec((2,)).is_valid(bf16)
        assert NumericArraySpec((2,), dtype=jnp.bfloat16).is_valid(bf16)
        # bf16 -> float32 is a safe widening (same-kind), so it satisfies a
        # float32 spec; a cross-kind int spec does not.
        assert NumericArraySpec((2,), dtype=jnp.float32).is_valid(bf16)
        assert not NumericArraySpec((2,), dtype=jnp.int32).is_valid(bf16)
        f8 = jnp.ones((), dtype=jnp.float8_e4m3fn)
        assert NumericArraySpec(()).is_valid(f8)

    def test_structured_dtype_stays_non_numeric(self):
        # numpy structured dtypes are also kind "V" but are not numeric:
        # they fail NumericArraySpec, so infer_from routes them to OpaqueSpec.
        rec = np.zeros(2, dtype=[("a", "f4")])
        assert not NumericArraySpec((2,)).is_valid(rec)
        assert RecordSpec.infer_from({"r": rec})["r"] == OpaqueSpec()

    def test_infer_from_bfloat16_is_numeric(self):
        tpl = RecordSpec.infer_from({"x": jnp.ones((2, 3), dtype=jnp.bfloat16)})
        assert isinstance(tpl, NumericRecordSpec)
        assert tpl["x"] == NumericArraySpec((2, 3))

    def test_python_scalar_dtype_is_numpy_default(self):
        # A bare Python scalar reports the dtype ``np.asarray`` gives it.
        assert NumericArraySpec((), dtype=np.asarray(1.0).dtype).is_valid(1.0)
        assert not NumericArraySpec((), dtype=jnp.int32).is_valid(1.0)

    def test_support_not_checked_by_is_valid(self):
        # ``support`` is descriptive metadata on the spec; ``is_valid`` validates
        # structure only (shape + dtype), so an out-of-support value still
        # passes. (Support is a data-dependent check that isn't jit-traceable,
        # and is_valid runs at Record construction, which happens under trace.)
        from probpipe.core.constraints import positive

        spec = NumericArraySpec((2,), support=positive)
        assert spec.support is positive  # still stored on the spec
        assert spec.is_valid(jnp.asarray([1.0, 2.0]))
        assert spec.is_valid(jnp.asarray([1.0, -2.0]))  # support not enforced

    def test_is_valid_is_jit_traceable(self):
        # is_valid reads only static shape/dtype (no data-dependent support
        # check), so it runs under jax.jit for any spec — including one that
        # carries a support — returning rather than concretizing. (jit wraps the
        # Python bool as a scalar Array, hence bool(...).)
        from probpipe.core.constraints import positive

        spec = NumericArraySpec((3,), dtype=jnp.float32)
        assert bool(jax.jit(spec.is_valid)(jnp.ones(3, dtype=jnp.float32))) is True
        assert bool(jax.jit(spec.is_valid)(jnp.ones(2, dtype=jnp.float32))) is False
        # even with a support set, no concretization is forced:
        assert bool(jax.jit(NumericArraySpec((3,), support=positive).is_valid)(jnp.ones(3))) is True


class TestOpaqueSpecIsValid:
    def test_non_mapping_objects_valid(self):
        spec = OpaqueSpec()
        assert spec.is_valid("label")
        assert spec.is_valid(object())
        assert spec.is_valid(None)
        assert spec.is_valid([1, 2, 3])

    def test_numeric_values_valid(self):
        # As the fallback spec, OpaqueSpec accepts any non-mapping value,
        # including numerics (though infer_from routes those to NumericArraySpec).
        spec = OpaqueSpec()
        assert spec.is_valid(1.5)
        assert spec.is_valid(jnp.ones(2))

    def test_mapping_invalid(self):
        # A mapping denotes tree structure (a subtree), never a leaf value —
        # the one thing OpaqueSpec rejects.
        assert not OpaqueSpec().is_valid({"a": 1})
        from types import MappingProxyType

        assert not OpaqueSpec().is_valid(MappingProxyType({"a": 1}))

    def test_record_layer_agrees_mappings_are_never_leaves(self):
        # The record layer honours the same rule as the spec: a mapping value
        # denotes tree structure, so it is materialised into a subtree rather
        # than stored as an opaque leaf.
        r = Record("r", x={"a": 1})
        assert tuple(r.keys()) == ("x/a",)

    def test_meta_not_checked(self):
        assert OpaqueSpec(meta="tag").is_valid("anything")


class TestDistributionSpecIsValid:
    def test_matching_distribution_valid(self):
        from probpipe import Normal

        dist = Normal(name="x", loc=0.0, scale=1.0)
        assert DistributionSpec(event_spec=dist.event_template).is_valid(dist)

    def test_template_mismatch_invalid(self):
        from probpipe import Normal

        dist = Normal(name="x", loc=0.0, scale=1.0)
        assert not DistributionSpec(event_spec=RecordSpec(y=())).is_valid(dist)

    def test_non_distribution_invalid(self):
        spec = DistributionSpec(event_spec=RecordSpec(x=()))
        assert not spec.is_valid(42)
        assert not spec.is_valid(RecordSpec(x=()))

    def test_distribution_without_template_invalid(self):
        # A distribution always carries the schema of its draws; one that
        # exposes no event template cannot satisfy any DistributionSpec.
        from probpipe.core._distribution_base import Distribution

        class _NoTemplate(Distribution):
            def __init__(self):
                super().__init__(name="d")

        spec = DistributionSpec(event_spec=RecordSpec(x=()))
        assert not spec.is_valid(_NoTemplate())

    def test_distribution_with_none_template_invalid(self):
        from probpipe.core._distribution_base import Distribution

        class _NoneTemplate(Distribution):
            def __init__(self):
                super().__init__(name="d")

            @property
            def event_template(self):
                return None

        spec = DistributionSpec(event_spec=RecordSpec(x=()))
        assert not spec.is_valid(_NoneTemplate())

    def test_type_error_template_is_not_a_match(self):
        # TypeError is the documented "template not derivable" signal (e.g. an
        # un-named auto-deriving distribution): a non-match, so is_valid
        # returns False.
        from probpipe.core._distribution_base import Distribution

        class _NotDerivable(Distribution):
            def __init__(self):
                super().__init__(name="d")

            @property
            def event_template(self):
                raise TypeError("template not derivable")

        spec = DistributionSpec(event_spec=RecordSpec(x=()))
        assert not spec.is_valid(_NotDerivable())

    @pytest.mark.parametrize("error", [RuntimeError, ValueError, KeyError])
    def test_unexpected_template_error_propagates(self, error):
        # An unexpected error from event_template is a malfunctioning
        # distribution, not a clean non-match — is_valid must not mask it as
        # invalid; it propagates so the bug surfaces.
        from probpipe.core._distribution_base import Distribution

        class _Broken(Distribution):
            def __init__(self):
                super().__init__(name="d")

            @property
            def event_template(self):
                raise error("boom")

        spec = DistributionSpec(event_spec=RecordSpec(x=()))
        with pytest.raises(error):
            spec.is_valid(_Broken())


class TestFunctionSpecIsValid:
    def test_callable_valid(self):
        spec = FunctionSpec(input_template=RecordSpec(a=()), output_spec=RecordSpec(b=()))
        assert spec.is_valid(lambda a: a)
        assert spec.is_valid(np.sin)

    def test_non_callable_invalid(self):
        spec = FunctionSpec(input_template=RecordSpec(a=()), output_spec=RecordSpec(b=()))
        assert not spec.is_valid(3.0)
        assert not spec.is_valid("f")


# ---------------------------------------------------------------------------
# FunctionSpec — input/output must be explicit RecordSpecs
# ---------------------------------------------------------------------------


class TestFunctionSpecTemplatesRequired:
    def test_explicit_sides_stored_per_the_storage_rule(self):
        # The input side is a schema and is stored as given; the output side is
        # a declaration, so a bare template is stored wrapped.
        inp, out = RecordSpec(a=()), RecordSpec(b=())
        spec = FunctionSpec(inp, out)
        assert spec.input_template is inp
        assert spec.output_spec == RecordSpec(out)

    def test_bare_value_spec_rejected_on_the_input_side_only(self):
        # The input side is a record schema, written out as a RecordSpec, so
        # a bare TermSpec is not wrapped into one. The output side is a
        # declaration and accepts any value specification.
        with pytest.raises(TypeError, match="input_template must be None or a RecordSpec"):
            FunctionSpec(NumericArraySpec(()), RecordSpec(b=()))  # type: ignore[arg-type]

        assert FunctionSpec(RecordSpec(a=()), OpaqueSpec()).output_spec == OpaqueSpec()

    def test_non_spec_rejected(self):
        with pytest.raises(TypeError, match="input_template must be None or a RecordSpec"):
            FunctionSpec((3,), RecordSpec(b=()))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="output_spec must be None or a TermSpec"):
            FunctionSpec(RecordSpec(a=()), "not a template")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FunctionSpec — optional (None) templates: "some callable, structure unknown"
# ---------------------------------------------------------------------------


class TestFunctionSpecOptionalTemplates:
    def test_bare_function_spec_is_any_callable(self):
        spec = FunctionSpec()
        assert spec.input_template is None
        assert spec.output_spec is None
        assert spec.is_valid(lambda x: x)
        assert spec.is_valid(np.sin)
        assert not spec.is_valid(3.0)

    def test_one_side_specified(self):
        spec = FunctionSpec(output_spec=RecordSpec(out=NumericArraySpec(())))
        assert spec.input_template is None
        # A record output is stored as its declaration (the storage rule).
        assert spec.output_spec == RecordSpec(out=NumericArraySpec(()))

    def test_none_specs_are_hashable_and_equal(self):
        assert FunctionSpec() == FunctionSpec()
        assert hash(FunctionSpec()) == hash(FunctionSpec())
        # A template-less spec differs from a typed one.
        assert FunctionSpec() != FunctionSpec(RecordSpec(inp=()), RecordSpec(out=()))

    def test_none_spec_usable_as_template_leaf(self):
        # A FunctionSpec leaf (with unspecified signature) lives in a template
        # and blocks numeric auto-promotion like any non-array leaf.
        tpl = RecordSpec(f=FunctionSpec(), x=())
        assert tpl["f"] == FunctionSpec()
        assert type(tpl) is RecordSpec

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
    def test_tuple_becomes_numeric_array_spec(self):
        tpl = RecordSpec(x=(3,))
        assert tpl["x"] == NumericArraySpec((3,))

    def test_none_becomes_opaque_spec(self):
        tpl = RecordSpec(label=None)
        assert tpl["label"] == OpaqueSpec()

    def test_nested_template_preserved(self):
        inner = RecordSpec(a=(), b=(3,))
        tpl = RecordSpec(sub=inner, z=())
        assert tpl.at_path("sub") is inner

    def test_explicit_numeric_array_spec_accepted(self):
        spec = NumericArraySpec((2,), dtype="float32")
        tpl = RecordSpec(x=spec)
        assert tpl["x"] is spec

    def test_explicit_opaque_spec_accepted(self):
        spec = OpaqueSpec(meta="tag")
        tpl = RecordSpec(label=spec, x=())
        assert tpl["label"] is spec

    def test_explicit_distribution_and_function_specs_accepted(self):
        dspec = DistributionSpec(event_spec=RecordSpec(x=()))
        fspec = FunctionSpec(input_template=RecordSpec(a=()), output_spec=RecordSpec(b=()))
        tpl = RecordSpec(d=dspec, f=fspec)
        assert tpl["d"] is dspec
        assert tpl["f"] is fspec

    def test_unsupported_spec_rejected(self):
        with pytest.raises(TypeError, match="spec must be"):
            RecordSpec(x=3.0)


# ---------------------------------------------------------------------------
# Auto-promotion to NumericRecordSpec (iff every leaf is a NumericArraySpec)
# ---------------------------------------------------------------------------


class TestAutoPromotionSpecs:
    def test_explicit_numeric_array_specs_promote(self):
        tpl = RecordSpec(x=NumericArraySpec(()), y=NumericArraySpec((3,)))
        assert isinstance(tpl, NumericRecordSpec)

    def test_nested_numeric_promotes(self):
        tpl = RecordSpec(sub=RecordSpec(a=(), b=(2,)), z=())
        assert isinstance(tpl, NumericRecordSpec)

    @pytest.mark.parametrize("empty", [{}, RecordSpec(), NumericRecordSpec(), {"nested": {}}])
    def test_empty_subtree_forms_have_the_same_numeric_layout(self, empty):
        implicit = RecordSpec(x=(), empty=empty)
        explicit = RecordSpec(x=(), empty=RecordSpec(empty))
        assert type(implicit) is type(explicit) is NumericRecordSpec
        assert implicit == explicit
        assert hash(implicit) == hash(explicit)
        assert implicit.is_numeric
        assert implicit.vector_size == explicit.vector_size == 1
        assert implicit.leaf_shapes == explicit.leaf_shapes == {"x": ()}
        assert isinstance(implicit.children["empty"], NumericRecordSpec)
        assert implicit.numeric_subset() == NumericRecordSpec(x=())

    def test_numeric_parent_normalizes_empty_child_without_mutating_it(self):
        empty = RecordSpec()
        parent = NumericRecordSpec(empty=empty)
        assert type(empty) is RecordSpec
        assert isinstance(parent.children["empty"], NumericRecordSpec)
        assert parent.vector_size == 0
        assert parent.leaf_shapes == {}
        assert list(parent.children) == ["empty"]

    def test_dimension_transforms_preserve_empty_numeric_subtrees(self):
        from probpipe import positive

        spec = NumericRecordSpec(
            theta=NumericArraySpec(("d",), dtype="float64", support=positive),
            aux=NumericRecordSpec(nested=NumericRecordSpec()),
        )
        renamed = spec.with_dim_names(d="n")
        assert isinstance(renamed, NumericRecordSpec)
        assert isinstance(renamed.at_path("aux/nested"), NumericRecordSpec)
        assert renamed["theta"].shape == ("n",)
        values = {"theta": np.ones(3), "aux": {"nested": {}}}
        actual = RecordSpec(theta=(3,), aux={"nested": {}})
        for bound in (
            spec.with_dims(d=3),
            renamed.with_dims(n=3),
            spec.bind_dims_from_value(values),
            spec.bind_dims_from_spec(actual),
        ):
            assert isinstance(bound, NumericRecordSpec)
            assert isinstance(bound.at_path("aux/nested"), NumericRecordSpec)
            assert list(bound.children) == ["theta", "aux"]
            assert bound.vector_size == 3
            assert bound.leaf_shapes == {"theta": (3,)}
            assert bound["theta"].dtype == np.dtype("float64")
            assert bound["theta"].support is positive
        assert spec.free_dims == {"d"}

    @pytest.mark.parametrize("spec_type", [RecordSpec, NumericRecordSpec])
    def test_dimension_transforms_preserve_empty_root_kind(self, spec_type):
        spec = spec_type()
        for result in (
            spec.with_dims(d=3),
            spec.with_dim_names(d="n"),
            spec.bind_dims_from_value({}),
            spec.bind_dims_from_spec(RecordSpec()),
        ):
            assert type(result) is spec_type
            assert result == spec

    def test_opaque_spec_blocks_promotion(self):
        tpl = RecordSpec(x=(), label=OpaqueSpec())
        assert type(tpl) is RecordSpec

    def test_distribution_spec_blocks_promotion(self):
        tpl = RecordSpec(x=(), d=DistributionSpec(event_spec=RecordSpec(a=())))
        assert type(tpl) is RecordSpec

    def test_numeric_record_spec_preserves_promotion(self):
        tpl = RecordSpec(x=(), r=RecordSpec(a=()))
        assert isinstance(tpl, NumericRecordSpec)
        assert tpl.vector_size == 2

    def test_function_spec_blocks_promotion(self):
        tpl = RecordSpec(
            x=(),
            f=FunctionSpec(input_template=RecordSpec(a=()), output_spec=RecordSpec(b=())),
        )
        assert type(tpl) is RecordSpec

    def test_numeric_rejects_opaque_spec(self):
        with pytest.raises(TypeError, match="only NumericArraySpec"):
            NumericRecordSpec(x=(), label=OpaqueSpec())

    def test_numeric_rejects_distribution_spec(self):
        with pytest.raises(TypeError, match="only NumericArraySpec"):
            NumericRecordSpec(x=(), d=DistributionSpec(event_spec=RecordSpec(a=())))

    def test_numeric_rejects_mixed_record_spec(self):
        with pytest.raises(TypeError, match="nested sub-templates"):
            NumericRecordSpec(x=(), r=RecordSpec(a=None))

    def test_numeric_rejects_function_spec(self):
        with pytest.raises(TypeError, match="only NumericArraySpec"):
            NumericRecordSpec(
                x=(),
                f=FunctionSpec(input_template=RecordSpec(a=()), output_spec=RecordSpec(b=())),
            )


# ---------------------------------------------------------------------------
# Back-compat: shape-shaped accessors keep their pre-spec return values
# ---------------------------------------------------------------------------


class TestShapeAccessorBackCompat:
    def test_vector_size_unchanged(self):
        tpl = RecordSpec(x=(), y=(3,), z=(2, 4))
        assert isinstance(tpl, NumericRecordSpec)
        assert tpl.vector_size == 1 + 3 + 8

    def test_hash_eq_order_sensitive(self):
        a = RecordSpec(x=(), y=(3,))
        b = RecordSpec(x=(), y=(3,))
        c = RecordSpec(y=(3,), x=())
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
        assert hash(a) != hash(c)


# ---------------------------------------------------------------------------
# Numeric queries & projection: is_numeric / is_multi_field / numeric_subset
# ---------------------------------------------------------------------------


def _dist_spec() -> DistributionSpec:
    return DistributionSpec(event_spec=RecordSpec(a=()))


def _func_spec() -> FunctionSpec:
    return FunctionSpec(input_template=RecordSpec(a=()), output_spec=RecordSpec(b=()))


class TestIsNumeric:
    def test_all_numeric_array_specs(self):
        assert RecordSpec(x=(), y=(3,)).is_numeric is True

    def test_nested_all_numeric(self):
        tpl = RecordSpec(x=(), params=RecordSpec(a=(), b=(3,)))
        assert tpl.is_numeric is True

    def test_mixed_opaque(self):
        assert RecordSpec(x=(), label=None).is_numeric is False

    def test_distribution_leaf(self):
        assert RecordSpec(x=(), d=_dist_spec()).is_numeric is False

    def test_function_leaf(self):
        assert RecordSpec(x=(), f=_func_spec()).is_numeric is False

    def test_nested_mixed(self):
        tpl = RecordSpec(x=(), nested=RecordSpec(a=(), label=None))
        assert tpl.is_numeric is False


class TestIsMultiField:
    def test_single_field(self):
        assert RecordSpec(x=()).is_multi_field is False

    def test_multi_field(self):
        assert RecordSpec(x=(), y=()).is_multi_field is True

    def test_single_opaque_leaf(self):
        assert RecordSpec(label=None).is_multi_field is False

    def test_two_leaves_mixed(self):
        assert RecordSpec(x=(), label=None).is_multi_field is True

    def test_single_leaf_under_nested_field(self):
        # One top-level field nesting a single leaf -> one leaf -> not multi.
        assert RecordSpec(a=RecordSpec(b=())).is_multi_field is False

    def test_multiple_leaves_under_one_nested_field(self):
        # jhuggins' case: a single top-level field 'a' with leaves a/b, a/c.
        tpl = RecordSpec(a=RecordSpec(b=(), c=()))
        assert tpl.fields == ("a",)  # one top-level field ...
        assert tpl.is_multi_field is True  # ... but two leaves -> multi-field

    def test_deeply_nested_single_leaf(self):
        tpl = RecordSpec(a=RecordSpec(b=RecordSpec(c=())))
        assert tpl.is_multi_field is False


class TestNumericSubset:
    def test_drops_non_numeric_keeps_numeric(self):
        tpl = RecordSpec(x=(), label=None, d=_dist_spec(), y=(3,))
        sub = tpl.numeric_subset()
        assert isinstance(sub, NumericRecordSpec)
        assert sub.fields == ("x", "y")

    def test_recurses_into_nested(self):
        tpl = RecordSpec(x=(), nested=RecordSpec(a=(), label=None, b=(3,)))
        sub = tpl.numeric_subset()
        assert sub.fields == ("x", "nested")
        assert isinstance(sub.at_path("nested"), NumericRecordSpec)
        assert tuple(sub.at_path("nested").children) == ("a", "b")

    def test_prunes_emptied_nested(self):
        tpl = RecordSpec(x=(), nested=RecordSpec(label=None, tag=None))
        sub = tpl.numeric_subset()
        assert sub.fields == ("x",)

    def test_path_stable(self):
        tpl = RecordSpec(x=(), nested=RecordSpec(a=(), label=None, b=(3,)))
        sub = tpl.numeric_subset()
        assert sub.leaf_shapes == {"x": (), "nested/a": (), "nested/b": (3,)}

    def test_idempotent_on_all_numeric(self):
        tpl = RecordSpec(x=(), y=(3,), nested=RecordSpec(a=(), b=(2,)))
        sub = tpl.numeric_subset()
        assert sub == tpl
        assert sub.numeric_subset() == sub

    def test_returns_numeric_template_with_vector_size(self):
        tpl = RecordSpec(x=(), label=None, y=(3,), z=(2, 4))
        sub = tpl.numeric_subset()
        assert isinstance(sub, NumericRecordSpec)
        assert sub.vector_size == 1 + 3 + 8

    def test_raises_when_no_numeric_leaves(self):
        tpl = RecordSpec(label=None, tag=None)
        with pytest.raises(ValueError, match="NumericSpec leaves survive"):
            tpl.numeric_subset()

    def test_raises_names_dropped_fields(self):
        tpl = RecordSpec(label=None, d=_dist_spec())
        with pytest.raises(ValueError, match="label"):
            tpl.numeric_subset()

    def test_raises_when_only_nested_empties(self):
        tpl = RecordSpec(nested=RecordSpec(label=None, tag=None))
        with pytest.raises(ValueError, match="nested"):
            tpl.numeric_subset()

    @pytest.mark.parametrize("empty", [RecordSpec(), NumericRecordSpec()])
    def test_empty_subtrees_do_not_count_as_numeric_leaves(self, empty):
        with pytest.raises(ValueError, match="no NumericSpec leaves survive"):
            empty.numeric_subset()
        tpl = RecordSpec(nested=empty, label=None)
        with pytest.raises(ValueError, match=r"Dropped fields:.*nested.*label"):
            tpl.numeric_subset()
        assert RecordSpec(nested=empty, x=(0,)).numeric_subset() == NumericRecordSpec(x=(0,))

    def test_prunes_empty_subtrees_inside_numeric_records(self):
        tpl = RecordSpec(nested=NumericRecordSpec(empty=NumericRecordSpec(), x=()), label=None)
        sub = tpl.numeric_subset()
        assert sub == NumericRecordSpec(nested=NumericRecordSpec(x=()))
        assert sub.numeric_subset() == sub

    def test_preserves_leaf_metadata_order_and_symbolic_dimensions(self):
        from probpipe.core.constraints import positive

        leaf = NumericArraySpec(("n",), dtype="float32", support=positive)
        tpl = RecordSpec(z=(0,), nested=RecordSpec(label=None, a=leaf), x=())
        sub = tpl.numeric_subset()
        assert sub.keys() == ("z", "nested/a", "x")
        assert sub["nested/a"] is leaf
        assert sub.free_dims == {"n"}
        assert sub.with_dims(n=2).vector_size == 3
        assert tpl.keys() == ("z", "nested/label", "nested/a", "x")

    def test_numeric_layout_errors_are_not_silently_pruned(self):
        @dataclass(frozen=True)
        class UnavailableLayout(NumericSpec):
            @property
            def vector_size(self):
                raise ValueError("numeric layout unavailable")

            def is_valid(self, value):
                return True

        tpl = RecordSpec(x=(), nested=RecordSpec(value=UnavailableLayout(), label=None))
        with pytest.raises(ValueError, match="numeric layout unavailable"):
            tpl.numeric_subset()

    def test_projected_layout_reconstructs_values_in_canonical_order(self):
        tpl = RecordSpec(z=(), nested=RecordSpec(label=None, x=(2,), empty=(0,)))
        sub = tpl.numeric_subset()
        record = NumericRecord.from_vector("value", sub, jnp.array([1.0, 2.0, 3.0]))
        assert record.spec == sub
        assert record.keys() == ("z", "nested/x", "nested/empty")
        assert record["nested/empty"].shape == (0,)
        np.testing.assert_array_equal(record["z"], 1.0)
        np.testing.assert_array_equal(record["nested/x"], [2.0, 3.0])


# ---------------------------------------------------------------------------
# to_vector / from_vector — 1-D numeric (de)serialization
# ---------------------------------------------------------------------------


class TestToVector:
    def test_to_vector_and_from_vector_are_value_only_not_on_template(self):
        # Both ``to_vector`` (value → vector) and ``from_vector`` (vector →
        # value) are value operations per the design contract; the template
        # must not carry either (that would make the template layer depend
        # on the value type).
        assert not hasattr(NumericRecordSpec, "to_vector")
        assert not hasattr(NumericRecordSpec, "from_vector")
        nr = NumericRecord("nr", x=1.0, y=jnp.arange(3.0))
        assert NumericRecord.from_vector("nr", nr.event_template, nr.to_vector()) == nr

    def test_scalar_value(self):
        v = NumericRecord("nr", x=1.5)
        vec = v.to_vector()
        assert vec.shape == (1,)
        assert jnp.array_equal(vec, jnp.asarray([1.5]))

    def test_vector_value(self):
        v = NumericRecord("nr", y=jnp.arange(3.0))
        vec = v.to_vector()
        assert vec.shape == (3,)
        assert jnp.array_equal(vec, jnp.arange(3.0))

    def test_multi_field_value(self):
        v = NumericRecord("nr", x=1.0, y=jnp.arange(3.0), z=jnp.ones((2, 4)))
        vec = v.to_vector()
        assert vec.shape == (1 + 3 + 8,)
        # Exact concatenation in canonical field order (x, y, z) — also pins
        # the ordering, which a shape-only check would miss.
        np.testing.assert_array_equal(np.asarray(vec), np.array([1.0, 0.0, 1.0, 2.0, *([1.0] * 8)]))

    def test_to_vector_shape_is_vector_size(self):
        tpl = RecordSpec(x=(), y=(3,), z=(2, 4))
        v = NumericRecord("nr", x=0.0, y=jnp.zeros(3), z=jnp.zeros((2, 4)))
        assert v.to_vector().shape == (tpl.vector_size,)

    def test_batched_shape_is_batch_shape_plus_vector_size(self):
        tpl = RecordSpec(x=(), y=(3,))
        flat = jnp.arange(2 * 5 * tpl.vector_size, dtype=float).reshape(2, 5, tpl.vector_size)
        v = NumericRecordBatch.from_vector("nrb", tpl, flat, level_names="draw")
        assert isinstance(v, NumericRecordBatch)
        assert v.to_vector().shape == (2, 5, tpl.vector_size)


class TestFromVectorRoundTripSingle:
    def test_empty_numeric_subtree_keeps_its_kind(self):
        tpl = NumericRecordSpec(theta=(3,), aux=NumericRecordSpec())
        flat = jnp.arange(3.0)
        value = NumericRecord.from_vector("value", tpl, flat)
        assert isinstance(value, NumericRecord)
        assert isinstance(value.at_path("aux"), NumericRecord)
        assert value.event_template == tpl
        assert list(value.children) == ["theta", "aux"]
        np.testing.assert_array_equal(value.to_vector(), flat)

    def test_scalar(self):
        v = NumericRecord("nr", x=1.5)
        tpl = RecordSpec.infer_from(v)
        assert NumericRecord.from_vector("nr", tpl, v.to_vector()) == v

    def test_vector(self):
        v = NumericRecord("nr", y=jnp.arange(3.0))
        tpl = RecordSpec.infer_from(v)
        assert NumericRecord.from_vector("nr", tpl, v.to_vector()) == v

    def test_multi_field(self):
        v = NumericRecord("nr", x=1.0, y=jnp.arange(3.0), z=jnp.arange(8.0).reshape(2, 4))
        tpl = RecordSpec.infer_from(v)
        assert NumericRecord.from_vector("nr", tpl, v.to_vector()) == v

    def test_nested(self):
        v = NumericRecord(
            "nr", x=1.0, y=jnp.arange(3.0), nested=NumericRecord("nr", a=2.0, b=jnp.arange(2.0))
        )
        tpl = RecordSpec.infer_from(v)
        round_tripped = NumericRecord.from_vector("nr", tpl, v.to_vector())
        assert isinstance(round_tripped, NumericRecord)
        assert round_tripped == v

    def test_returns_single_for_1d_vec(self):
        tpl = RecordSpec(x=(), y=(3,))
        v = NumericRecord.from_vector("nr", tpl, jnp.arange(4.0))
        assert isinstance(v, NumericRecord)

    def test_zero_d_vec_raises_type_error(self):
        # A 0-d scalar is not a 1-D vector; ``NumericRecord.from_vector`` raises
        # the documented TypeError rather than an IndexError from indexing
        # ``vec.shape[-1]``.
        tpl = RecordSpec(x=())
        with pytest.raises(TypeError, match="1-D vector"):
            NumericRecord.from_vector("nr", tpl, 5.0)


class TestFromVectorRoundTripBatched:
    def test_empty_numeric_subtree_survives_vector_round_trip(self):
        tpl = NumericRecordSpec(theta=(3,), aux=NumericRecordSpec())
        flat = jnp.arange(6.0).reshape(2, 3)
        value = NumericRecordBatch.from_vector("values", tpl, flat, level_names="draw")
        assert isinstance(value, NumericRecordBatch)
        assert value.batch_shape == (2,)
        assert value.level_names == ("draw",)
        assert value.event_template == tpl
        assert isinstance(value.event_template.children["aux"], NumericRecordSpec)
        np.testing.assert_array_equal(value.to_vector(), flat)

    def test_single_batch_axis(self):
        tpl = RecordSpec(x=(), y=(3,))
        flat = jnp.arange(4 * tpl.vector_size, dtype=float).reshape(4, tpl.vector_size)
        v = NumericRecordBatch.from_vector("nrb", tpl, flat, level_names="draw")
        assert isinstance(v, NumericRecordBatch)
        assert v.batch_shape == (4,)
        assert NumericRecordBatch.from_vector("nrb", tpl, v.to_vector(), level_names="draw") == v

    def test_multi_axis_batch_shape(self):
        # batch_shape=(2, 3) catches trailing-axis split / reshape bugs.
        tpl = RecordSpec(x=(), y=(3,), z=(2, 2))
        flat = jnp.arange(2 * 3 * tpl.vector_size, dtype=float).reshape(2, 3, tpl.vector_size)
        v = NumericRecordBatch.from_vector("nrb", tpl, flat, level_names="draw")
        assert isinstance(v, NumericRecordBatch)
        assert v.batch_shape == (2, 3)
        assert jnp.array_equal(v.to_vector(), flat)
        assert NumericRecordBatch.from_vector("nrb", tpl, v.to_vector(), level_names="draw") == v

    def test_nested_multi_axis_batch_shape(self):
        # Nested numeric subtree + multi-axis batch: from_vector builds a nested
        # NumericRecordBatch as a field of the outer NumericRecordBatch.
        tpl = RecordSpec(x=(), nested=RecordSpec(a=(), b=(2,)), y=(3,))
        flat = jnp.arange(2 * 3 * tpl.vector_size, dtype=float).reshape(2, 3, tpl.vector_size)
        v = NumericRecordBatch.from_vector("nrb", tpl, flat, level_names="draw")
        assert isinstance(v, NumericRecordBatch)
        assert v.batch_shape == (2, 3)
        assert isinstance(v["nested"], NumericRecordBatch)
        assert v["nested/b"].shape == (2, 3, 2)
        assert NumericRecordBatch.from_vector("nrb", tpl, v.to_vector(), level_names="draw") == v


class TestFromVectorErrors:
    def test_wrong_trailing_size_raises(self):
        tpl = RecordSpec(x=(), y=(3,))
        with pytest.raises(ValueError, match="vector_size"):
            NumericRecord.from_vector("nr", tpl, jnp.zeros(5))


# ---------------------------------------------------------------------------
# TermSpec taxonomy (RecordSpec / DistributionSpec / FunctionSpec)
# ---------------------------------------------------------------------------


class TestTermSpecTaxonomy:
    """The term-spec sub-hierarchy: one spec per kind, all also TermSpecs."""

    def test_term_specs_are_value_specs(self):

        tau = RecordSpec(x=())
        for spec in (
            RecordSpec(tau),
            DistributionSpec(tau),
            FunctionSpec(),
        ):
            assert isinstance(spec, TermSpec)

    def test_raw_value_kinds_are_term_specs(self):

        assert isinstance(NumericArraySpec(()), TermSpec)
        assert isinstance(OpaqueSpec(), TermSpec)

    # --- the storage rule: a declaration is stored as a TermSpec ---

    def test_record_schema_is_stored_directly(self):
        """The schema already declares the kind; no wrapper is stored."""

        tau = RecordSpec(x=())
        assert DistributionSpec(tau).event_spec is tau
        assert isinstance(DistributionSpec(tau).event_spec, TermSpec)
        assert FunctionSpec(tau, tau).output_spec is tau
        assert isinstance(FunctionSpec(tau, tau).output_spec, TermSpec)

    def test_equal_schema_copies_produce_equal_declarations(self):
        """Schema copying does not change the declared kind or structure."""

        tau = RecordSpec(x=())
        assert DistributionSpec(RecordSpec(tau)) == DistributionSpec(tau)
        assert FunctionSpec(tau, RecordSpec(tau)) == FunctionSpec(tau, tau)

    def test_a_declaration_field_is_declared_at_the_type_it_stores(self):
        """The annotation is the post-construction guarantee, not the input sugar.

        The wider template spelling is carried by the constructor signature, so a
        type checker and the generated API reference read the stored type. Pins
        the split against a rewidening of the field annotations.
        """
        assert get_type_hints(DistributionSpec)["event_spec"] is RecordSpec
        assert get_type_hints(FunctionSpec)["output_spec"] == TermSpec | None
        assert get_type_hints(NumericArraySpec)["dtype"] == np.dtype | None

    def test_the_old_parameter_names_are_gone(self):
        """Positional construction survives the rename; keyword construction does not.

        The migration rule the CHANGELOG states: a keyword call moves to the new
        parameter name, as does every read of the old attribute.
        """
        tau = RecordSpec(x=())
        assert DistributionSpec(tau) == DistributionSpec(event_spec=tau)
        assert FunctionSpec(tau, tau) == FunctionSpec(tau, output_spec=tau)
        with pytest.raises(TypeError, match="event_template"):
            DistributionSpec(event_template=tau)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="output_template"):
            FunctionSpec(tau, output_template=tau)  # type: ignore[call-arg]
        assert not hasattr(DistributionSpec(tau), "event_template")
        assert not hasattr(FunctionSpec(tau, tau), "output_template")

    def test_term_valued_output_is_kept_not_wrapped(self):
        """A term output declaration names its own kind and passes through."""
        tau = RecordSpec(x=())
        inner = DistributionSpec(tau)
        assert FunctionSpec(tau, inner).output_spec is inner
        assert FunctionSpec(tau, FunctionSpec()).output_spec == FunctionSpec()

    def test_term_valued_event_declaration_is_rejected(self):
        """An event declaration is record-valued: a term draw is not yet checkable.

        A ``Distribution`` exposes a ``RecordSpec`` and nothing that reports
        a term-valued draw kind, so a random-measure declaration could be
        written but never satisfied. It is refused at construction rather than
        accepted and always reported invalid.
        """
        tau = RecordSpec(x=())
        for decl in (DistributionSpec(tau), FunctionSpec()):
            with pytest.raises(TypeError, match="must be a RecordSpec"):
                DistributionSpec(decl)  # type: ignore[arg-type]

    def test_declared_kind_is_the_stored_spec_class(self):
        """The declaration's class is the declared kind — a structural test."""
        tau = RecordSpec(x=())
        assert DistributionSpec(tau).event_spec is tau
        assert FunctionSpec(tau, tau).output_spec is tau
        assert type(FunctionSpec(tau, FunctionSpec()).output_spec) is FunctionSpec

    def test_raw_value_output_declaration_is_stored_as_given(self):
        """An output declaration is any value specification, as in Fun(sigma, rho).

        A numeric-array output declares the array itself. No record schema or
        field name is inserted.
        """
        tau = RecordSpec(x=())
        for raw in (NumericArraySpec((3,)), OpaqueSpec(meta="m")):
            assert FunctionSpec(tau, raw).output_spec is raw

    def test_unspecified_output_stays_none(self):
        """None means "unspecified" and is not wrapped into a record declaration."""
        assert FunctionSpec().output_spec is None
        assert FunctionSpec(RecordSpec(x=())).output_spec is None

    def test_record_schema_is_its_kind_spec(self):
        tau = RecordSpec(x=())
        assert isinstance(tau, TermSpec)
        assert tau["x"] == NumericArraySpec(())

    def test_term_specs_compare_and_hash_by_value(self):
        # Distinct-but-equal templates throughout, so this pins value equality
        # rather than sharing one template object.
        assert RecordSpec(x=()) == RecordSpec(x=())
        assert hash(RecordSpec(x=())) == hash(RecordSpec(x=()))

    def test_record_declarations_preserve_schema_identity(self):
        """The storage rule at the value level, not merely by class."""
        tau = RecordSpec(x=())
        assert DistributionSpec(tau).event_spec is tau
        assert FunctionSpec(tau, tau).output_spec is tau

    def test_term_spec_declares_the_abstract_validation_protocol(self):
        """The hierarchy's headline claim: is_valid stays declared once."""
        with pytest.raises(TypeError, match="abstract"):
            TermSpec()  # type: ignore[abstract]
        assert "is_valid" in TermSpec.__abstractmethods__

    def test_the_old_attribute_names_are_gone(self):
        """The breaking half of the rename, which the CHANGELOG advertises."""
        tau = RecordSpec(x=())
        assert not hasattr(DistributionSpec(tau), "event_template")
        assert not hasattr(FunctionSpec(tau, tau), "output_template")
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            DistributionSpec(event_template=tau)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            FunctionSpec(tau, output_template=tau)  # type: ignore[call-arg]


class TestRecordSpec:
    def test_requires_mapping(self):

        with pytest.raises(TypeError):
            RecordSpec(NumericArraySpec(()))  # not a RecordSpec

    def test_requires_mapping_message(self):
        with pytest.raises(TypeError, match="must be a mapping or a RecordSpec"):
            RecordSpec(NumericArraySpec(()))

    def test_is_valid_accepts_matching_record_only(self):

        rec = Record("r", x=jnp.asarray(1.0))
        spec = RecordSpec(rec.event_template)
        assert spec.is_valid(rec)
        assert not spec.is_valid(jnp.asarray(1.0))  # not a Record
        assert not spec.is_valid(Record("r", y=jnp.asarray(1.0)))  # wrong template

    def test_is_valid_false_when_fields_cannot_be_read(self):
        class _Unreadable(Record):
            @property
            def children(self):
                raise TypeError("fields unavailable")

        rec = _Unreadable("r", x=jnp.asarray(1.0))
        assert not RecordSpec(x=()).is_valid(rec)

    def test_is_valid_propagates_an_unexpected_error(self):
        class _Broken(Record):
            @property
            def children(self):
                raise RuntimeError("malfunctioning record")

        rec = _Broken("r", x=jnp.asarray(1.0))
        with pytest.raises(RuntimeError, match="malfunctioning record"):
            RecordSpec(x=()).is_valid(rec)


class TestFunctionSpecOutputWidening:
    """An output declaration admits every kind, and raw values too.

    The record-output, non-spec-rejection, and callability cases are covered by
    TestFunctionSpecTemplatesRequired and TestFunctionSpecOptionalTemplates; only
    the widening itself lives here.
    """

    def test_term_spec_output_accepted(self):
        tau = RecordSpec(out=())
        assert FunctionSpec(RecordSpec(x=()), DistributionSpec(tau)).output_spec == (
            DistributionSpec(tau)
        )
        assert isinstance(FunctionSpec(output_spec=RecordSpec(y=())).output_spec, RecordSpec)

    def test_input_template_still_event_template_only(self):
        # The input side is a schema, so a spec is not accepted there even though
        # the output side takes one.
        with pytest.raises(TypeError, match="input_template must be None or a RecordSpec"):
            FunctionSpec(input_template=DistributionSpec(RecordSpec(x=())))


def test_public_exports():
    import probpipe

    for name in ("TermSpec", "RecordSpec", "NumericArraySpec"):
        assert hasattr(probpipe, name), name
        assert name in probpipe.__all__, name

    removed_name = "Array" + "Spec"
    assert not hasattr(probpipe, removed_name)
    assert removed_name not in probpipe.__all__


class TestFreeDimsReachThroughTermSpecs:
    """A name is visible wherever it is declared, not only at the outermost level.

    Design II.3: "A template with **any** symbolic entry is polymorphic, with
    `is_concrete` false and `free_dims` listing the unbound names." A term spec
    carries a schema, so a symbolic dimension inside one is such an entry.
    """

    @staticmethod
    def _symbolic():
        return RecordSpec(x=NumericArraySpec(shape=("obs",)))

    @pytest.mark.parametrize(
        "declare",
        [
            lambda sym: RecordSpec(x=NumericArraySpec(shape=("obs",))),
            lambda sym: RecordSpec(r=RecordSpec(sym)),
            lambda sym: RecordSpec(law=DistributionSpec(sym)),
            lambda sym: RecordSpec(f=FunctionSpec(sym, None)),
            lambda sym: RecordSpec(f=FunctionSpec(None, RecordSpec(sym))),
            lambda sym: RecordSpec(law=DistributionSpec(RecordSpec(sym))),
        ],
        ids=["array", "record", "distribution", "function-in", "function-out", "nested"],
    )
    def test_a_symbolic_dimension_is_reported_wherever_it_is_declared(self, declare):
        template = declare(self._symbolic())

        assert template.free_dims == frozenset({"obs"})
        assert not template.is_concrete

    def test_one_scope_across_a_term_spec_boundary(self):
        """The same name inside and outside a term spec is one dimension."""
        template = RecordSpec(
            data=NumericArraySpec(shape=("obs",)),
            law=DistributionSpec(self._symbolic()),
        )

        assert template.free_dims == frozenset({"obs"})

    def test_a_concrete_term_spec_reports_nothing(self):
        assert RecordSpec(law=DistributionSpec(RecordSpec(x=(3,)))).is_concrete

    def test_a_spec_declaring_no_dimensions_reports_none(self):
        assert OpaqueSpec().free_dims == frozenset()
        assert NumericArraySpec(shape=(3,)).free_dims == frozenset()


class TestWithDims:
    def test_binding_reaches_through_a_term_spec(self):
        sym = RecordSpec(x=NumericArraySpec(shape=("obs",)))
        template = RecordSpec(law=DistributionSpec(sym), data=NumericArraySpec(shape=("obs",)))

        bound = template.with_dims(obs=4)

        assert bound.is_concrete
        assert bound["data"].shape == (4,)
        assert bound["law"].event_spec["x"].shape == (4,)

    def test_binding_returns_a_new_template(self):
        template = RecordSpec(x=NumericArraySpec(shape=("obs",)))

        assert template.with_dims(obs=2) is not template
        assert not template.is_concrete

    def test_an_all_numeric_bound_template_gains_its_flat_layout(self):
        bound = RecordSpec(x=NumericArraySpec(shape=("n",))).with_dims(n=3)

        assert isinstance(bound, NumericRecordSpec)
        assert bound.vector_size == 3

    def test_unsupplied_dimensions_remain_symbolic(self):
        template = RecordSpec(
            x=NumericArraySpec(shape=("obs",)), y=NumericArraySpec(shape=("features",))
        )

        assert template.with_dims().free_dims == {"features", "obs"}

    def test_a_batch_spec_axis_is_bindable(self):
        """It is reported by `free_dims`, so it must be substitutable."""
        from probpipe import BatchSpec

        template = RecordSpec(b=BatchSpec(NumericArraySpec(shape=(3,)), [("S",)], ["draw"]))

        bound = template.with_dims(S=4)

        assert bound["b"].axis_groups == ((4,),)
        assert bound.is_concrete

    def test_a_size_must_be_an_integer(self):
        """A string would be read as a dimension *name*, silently renaming it."""
        template = RecordSpec(x=NumericArraySpec(shape=("n",)))

        for size in ("m", 2.0, None):
            with pytest.raises(TypeError, match="must be an integer"):
                template.with_dims(n=size)

    def test_binding_some_names_reports_only_the_rest(self):
        template = RecordSpec(x=NumericArraySpec(shape=("a",)), y=NumericArraySpec(shape=("b",)))

        bound = template.with_dims(a=2)
        assert bound.free_dims == {"b"}
        assert bound["x"].shape == (2,)
        assert template.free_dims == {"a", "b"}

    def test_binding_an_already_concrete_template_is_a_no_op_copy(self):
        template = RecordSpec(x=NumericArraySpec(shape=(3,)))

        bound = template.with_dims()

        assert bound == template
        assert bound is not template

    def test_a_name_the_template_does_not_declare_is_ignored(self):
        """So one mapping can bind several templates."""
        assert RecordSpec(x=NumericArraySpec(shape=("n",))).with_dims(n=2, other=9).is_concrete


class TestTemplateConcretization:
    @pytest.mark.parametrize(
        "bindings, missing",
        [({}, "features, obs"), ({"obs": 2}, "features")],
    )
    def test_missing_dimensions_are_named_in_sorted_order(self, bindings, missing):
        template = RecordSpec(x=("obs",), y=("features",))
        original_bindings = bindings.copy()
        with pytest.raises(ValueError) as caught:
            _concretize_event_template(template, bindings, context="output")
        assert str(caught.value) == f"output has unbound symbolic dimensions: {missing}"
        assert template == RecordSpec(x=("obs",), y=("features",))
        assert bindings == original_bindings

    def test_complete_bindings_return_a_concrete_schema(self):
        template = RecordSpec(x=("obs",), y=("features",))
        bindings = {"obs": 2, "features": 3}
        concrete = _concretize_event_template(template, bindings, context="output")
        assert concrete == NumericRecordSpec(x=(2,), y=(3,))
        assert concrete.is_concrete
        assert template == RecordSpec(x=("obs",), y=("features",))
        assert bindings == {"obs": 2, "features": 3}


class TestBindingAFunctionSpec:
    """A `FunctionSpec`'s two sides bind from the callable's own declaration.

    Not a conformance check: a function is contravariant in its input, so whether
    a value *is* an acceptable function is a separate question. Only the sizes
    bind.
    """

    @staticmethod
    def _sym():
        return RecordSpec(x=NumericArraySpec(shape=("obs",)))

    def test_a_bare_callable_leaves_the_dimensions_free(self):
        """It declares nothing, so there is nothing to bind from — and no refusal."""
        declared = RecordSpec(f=FunctionSpec(self._sym(), None))

        record = Record("r", f=lambda x: x, event_template=declared)

        assert record.event_template["f"].input_template["x"].shape == ("obs",)

    def test_the_input_side_binds_from_the_callable_declaration(self):
        declared = RecordSpec(f=FunctionSpec(self._sym(), None))
        typed = Function(
            func=lambda x: x, name="g", input_template=RecordSpec(x=NumericArraySpec(shape=(7,)))
        )

        record = Record("r", f=typed, event_template=declared)

        assert record.event_template["f"].input_template["x"].shape == (7,)

    def test_the_output_side_binds_from_the_callable_declaration(self):
        declared = RecordSpec(f=FunctionSpec(None, RecordSpec(y=NumericArraySpec(("m",)))))
        typed = Function(
            func=lambda x: x,
            name="g",
            output_template=RecordSpec(y=NumericArraySpec(shape=(5,))),
        )

        record = Record("r", f=typed, event_template=declared)

        assert record.event_template["f"].output_spec["y"].shape == (5,)

    def test_a_non_callable_is_refused(self):
        declared = RecordSpec(f=FunctionSpec(self._sym(), None))

        with pytest.raises(ValueError, match="does not conform to its field spec"):
            Record("r", f=3, event_template=declared)


class TestInferenceThroughTermSpecs:
    """Sizes are bound from the term a spec is matched against, as for an array.

    `NumericArraySpec` has always accepted a concrete value against a symbolic shape and
    left the binding to the one pass; these tests hold the term specs to the same
    rule.
    """

    @staticmethod
    def _law(size=3):
        import jax.numpy as jnp

        from probpipe import MultivariateNormal

        return MultivariateNormal(jnp.zeros(size), jnp.eye(size), name="x")

    def test_a_distribution_binds_the_declared_dimension(self):
        sym = RecordSpec(x=NumericArraySpec(shape=("obs",)))
        record = Record("r", law=self._law(3), event_template=RecordSpec(law=DistributionSpec(sym)))

        assert record.event_template.is_concrete
        assert record.event_template["law"].event_spec["x"].shape == (3,)

    def test_a_name_shared_across_the_boundary_binds_once(self):
        declared = RecordSpec(
            data=NumericArraySpec(shape=("obs",)),
            law=DistributionSpec(RecordSpec(x=NumericArraySpec(shape=("obs",)))),
        )
        record = Record("r", data=jnp.zeros(3), law=self._law(3), event_template=declared)

        assert record.event_template["data"].shape == (3,)
        assert record.event_template["law"].event_spec["x"].shape == (3,)

    def test_a_disagreement_binds_inner_first_then_outer(self):
        """The direction that proves the scope is shared, not merely inherited.

        With the term spec declared *first*, `obs` is bound by the law and the
        outer array must agree with it. An implementation that gave the term spec
        its own copy of the bindings would accept this, since the copy flows only
        inward — so this is the case that pins one scope rather than two.
        """
        declared = RecordSpec(
            law=DistributionSpec(RecordSpec(x=NumericArraySpec(shape=("obs",)))),
            data=NumericArraySpec(shape=("obs",)),
        )

        with pytest.raises(
            ValueError, match=r"/data binds symbolic dimension 'obs' to 5, .*already bound to 3"
        ):
            Record("r", law=self._law(3), data=jnp.zeros(5), event_template=declared)

    def test_field_order_does_not_change_the_outcome(self):
        """The same declaration either way round: one scope, one answer."""
        term_first = RecordSpec(
            law=DistributionSpec(RecordSpec(x=NumericArraySpec(shape=("obs",)))),
            data=NumericArraySpec(shape=("obs",)),
        )
        array_first = RecordSpec(
            data=NumericArraySpec(shape=("obs",)),
            law=DistributionSpec(RecordSpec(x=NumericArraySpec(shape=("obs",)))),
        )

        for declared in (term_first, array_first):
            record = Record("r", law=self._law(3), data=jnp.zeros(3), event_template=declared)
            assert record.event_template.is_concrete
            assert record.event_template["data"].shape == (3,)

    def test_a_disagreement_across_the_boundary_raises(self):
        """The point of one scope: 5 outside and 3 inside is a contradiction."""
        declared = RecordSpec(
            data=NumericArraySpec(shape=("obs",)),
            law=DistributionSpec(RecordSpec(x=NumericArraySpec(shape=("obs",)))),
        )

        with pytest.raises(
            ValueError, match=r"/law/x binds symbolic dimension 'obs' to 3, .*already bound to 5"
        ):
            Record("r", data=jnp.zeros(5), law=self._law(3), event_template=declared)

    def test_the_kind_is_checked_by_the_pass_itself(self):
        """Asserted at the unifier, not through `Record`.

        `Record` validates again afterwards, so routing through it cannot tell
        whether the pass checks the kind or merely lets a later check catch it.
        A polymorphic declaration must refuse a wrong-kind value on its own.
        """
        law = self._law(3)
        record = Record("w", x=jnp.zeros(3))
        sym = RecordSpec(x=NumericArraySpec(shape=("obs",)))

        for declared, value in (
            (RecordSpec(sym), law),
            (DistributionSpec(sym), record),
        ):
            with pytest.raises(ValueError, match=r"does not conform|expected named fields"):
                _unify_event_template_with_value(
                    RecordSpec(field=declared), {"field": value}, context="v"
                )

    def test_a_callable_declaration_refuses_a_non_callable_in_the_pass(self):
        """Likewise for the FunctionSpec branch, which has its own refusal."""
        declared = RecordSpec(f=FunctionSpec(RecordSpec(x=NumericArraySpec(shape=("obs",))), None))

        with pytest.raises(ValueError, match="does not conform to its field spec"):
            _unify_event_template_with_value(declared, {"f": 3}, context="v")

    def test_a_value_carrying_no_schema_says_so(self):
        """A polymorphic schema needs one to bind against."""
        declared = RecordSpec(law=DistributionSpec(RecordSpec(x=NumericArraySpec(shape=("obs",)))))

        with pytest.raises(ValueError, match="exposes no schema to bind it against"):
            Record("r", law=object(), event_template=declared)

    def test_a_concrete_declaration_still_requires_an_exact_match(self):
        """Inference is for the symbolic case; a fixed size is still a fixed size."""
        declared = RecordSpec(law=DistributionSpec(RecordSpec(x=NumericArraySpec(shape=(4,)))))

        with pytest.raises(ValueError, match="does not conform"):
            Record("r", law=self._law(3), event_template=declared)


class TestAFunctionOutputBindsWhateverItDeclares:
    """A `FunctionSpec`'s output binds whether or not it declares a record.

    `output_spec` is any value spec, since a callable may return a term of any
    kind. A record declaration meets the callable's output template as a whole;
    any other declaration describes the one value returned and meets that
    template's sole immediate field. Both bind, so a name shared with the input is one
    dimension on either route.
    """

    @staticmethod
    def _function(input_size=3, output_size=5):
        return Function(
            func=lambda x: jnp.zeros(output_size),
            name="f",
            input_template=RecordSpec(x=NumericArraySpec(shape=(input_size,))),
            output_template=RecordSpec(out=NumericArraySpec(shape=(output_size,))),
        )

    @staticmethod
    def _declared(output_spec):
        return RecordSpec(f=FunctionSpec(RecordSpec(x=NumericArraySpec(shape=("n",))), output_spec))

    def test_a_shared_name_binds_from_a_non_record_output(self):
        """`n` on both sides binds once when the two agree."""
        declared = self._declared(NumericArraySpec(shape=("n",)))

        record = Record("r", f=self._function(4, 4), event_template=declared)

        assert record.event_template.is_concrete
        assert record.event_template["f"].output_spec.shape == (4,)

    def test_a_non_record_output_that_disagrees_with_the_input_raises(self):
        """The case a skipped output hid: the input says 3, the output says 5.

        Binding only the input would leave the declaration reporting an output of
        `(3,)` for a callable that returns `(5,)` — a schema that is not merely
        unbound but wrong.
        """
        declared = self._declared(NumericArraySpec(shape=("n",)))

        with pytest.raises(ValueError, match=r"symbolic dimension 'n' to 5, .*already bound to 3"):
            Record("r", f=self._function(3, 5), event_template=declared)

    def test_a_record_output_that_disagrees_raises_the_same_way(self):
        """The route that already worked, asserted beside the one that did not."""
        declared = self._declared(RecordSpec(out=NumericArraySpec(shape=("n",))))

        with pytest.raises(ValueError, match=r"symbolic dimension 'n' to 5, .*already bound to 3"):
            Record("r", f=self._function(3, 5), event_template=declared)

    def test_one_declared_output_value_does_not_match_several_fields(self):
        """A single value declaration meets a single field, so two is a mismatch."""
        function = Function(
            func=lambda x: x,
            name="f",
            input_template=RecordSpec(x=NumericArraySpec(shape=(3,))),
            output_template=RecordSpec(
                a=NumericArraySpec(shape=(3,)), b=NumericArraySpec(shape=(4,))
            ),
        )
        declared = self._declared(NumericArraySpec(shape=("n",)))

        with pytest.raises(ValueError, match=r"declares one output value.*output fields"):
            Record("r", f=function, event_template=declared)

    @pytest.mark.parametrize(
        "template",
        [RecordSpec(outer=RecordSpec(inner=(3,))), RecordSpec(x=(3,), empty=RecordSpec())],
        ids=["nested_leaf", "empty_sibling"],
    )
    def test_one_array_output_does_not_flatten_record_structure(self, template):
        function = Function(func=lambda: None, output_template=template)
        for size in (3, "n"):
            spec = FunctionSpec(output_spec=NumericArraySpec((size,)))
            with pytest.raises(ValueError):
                spec.bind_dims_from_value(function)
            with pytest.raises(ValueError):
                RecordSpec(f=spec).bind_dims_from_value({"f": function})
        record_output = FunctionSpec(output_spec=template)
        assert record_output.bind_dims_from_value(function) == record_output

    def test_a_bare_callable_still_binds_nothing_from_its_output(self):
        """No declaration to read, so the output stays free rather than raising."""
        declared = self._declared(NumericArraySpec(shape=("k",)))

        record = Record("r", f=lambda x: x, event_template=declared)

        assert record.event_template.free_dims == frozenset({"n", "k"})


class TestMultiplicityBindsFromAValue:
    """A declared batch axis binds from the batch it is matched against.

    `TestSymbolicMultiplicity` in ``tests/core/test_batch.py`` holds the
    *declaration* side, what a spec reports and what it substitutes. These hold
    the *pass*: a symbolic axis meets an actual `Batch` and takes its size, in the
    same scope every other dimension binds in. A batch axis is a dimension like
    any other, so the rules `TestInferenceThroughTermSpecs` pins for a term spec's
    schema are the rules here.
    """

    @staticmethod
    def _batch(size=3, level="item"):
        return OpaqueBatch(
            "batch",
            [object() for _ in range(size)],
            level,
        )

    @staticmethod
    def _grid(shape, level_names="grid", axes_per_level=None):
        """A batch whose store has *shape*, for the multi-axis and multi-level cases."""
        store = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            store[index] = object()
        return OpaqueBatch(
            "batch",
            store,
            level_names,
            axes_per_level=axes_per_level or (len(shape),),
        )

    @staticmethod
    def _declared(axis="n", field=None):
        fields: dict[str, Any] = {}
        if field is not None:
            fields["data"] = NumericArraySpec(shape=(field,))
        fields["b"] = BatchSpec(OpaqueSpec(), [(axis,)], ["item"])
        return RecordSpec(fields)

    def test_an_axis_size_is_inferred_from_the_batch(self):
        """How many elements there are is read off the batch."""
        record = Record("r", b=self._batch(3), event_template=self._declared())

        assert record.event_template.is_concrete
        assert record.event_template["b"].axis_groups == ((3,),)
        assert record.event_template["b"].batch_size == 3

    def test_an_axis_and_a_field_share_one_dimension(self):
        """`("n",)` of elements beside an array of shape `("n",)` binds `n` once."""
        record = Record(
            "r", data=jnp.zeros(3), b=self._batch(3), event_template=self._declared(field="n")
        )

        assert record.event_template.is_concrete
        assert record.event_template["data"].shape == (3,)
        assert record.event_template["b"].axis_groups == ((3,),)

    def test_field_order_does_not_change_the_outcome(self):
        """One scope, so the axis may bind first or second and still agree.

        Declared array-first, `n` is bound by the array and the multiplicity must
        agree; declared batch-first, the reverse. An implementation giving the
        batch its own copy of the bindings would pass one order and fail the
        other, so both directions are asserted.
        """
        array_first = RecordSpec(
            data=NumericArraySpec(shape=("n",)), b=BatchSpec(OpaqueSpec(), [("n",)], ["item"])
        )
        batch_first = RecordSpec(
            b=BatchSpec(OpaqueSpec(), [("n",)], ["item"]), data=NumericArraySpec(shape=("n",))
        )

        for declared in (array_first, batch_first):
            record = Record("r", data=jnp.zeros(3), b=self._batch(3), event_template=declared)
            assert record.event_template.is_concrete
            assert record.event_template["data"].shape == (3,)
            assert record.event_template["b"].axis_groups == ((3,),)

    def test_a_disagreement_between_the_two_sides_raises(self):
        """A batch of 3 beside an array of 5 is a contradiction, not a rebinding."""
        with pytest.raises(
            ValueError, match=r"binds symbolic dimension 'n' to 3, .*already bound to 5"
        ):
            Record(
                "r", data=jnp.zeros(5), b=self._batch(3), event_template=self._declared(field="n")
            )

    def test_the_disagreement_raises_in_either_order(self):
        """The batch-first direction, which a copied scope would let through."""
        declared = RecordSpec(
            b=BatchSpec(OpaqueSpec(), [("n",)], ["item"]), data=NumericArraySpec(shape=("n",))
        )

        with pytest.raises(
            ValueError, match=r"/data binds symbolic dimension 'n' to 5, .*already bound to 3"
        ):
            Record("r", b=self._batch(3), data=jnp.zeros(5), event_template=declared)

    def test_a_bound_declaration_equals_the_concrete_one(self):
        """Binding is inference, not a second dialect: the two declarations agree."""
        inferred = Record("r", b=self._batch(3), event_template=self._declared())
        concrete = Record(
            "r",
            b=self._batch(3),
            event_template=RecordSpec(b=BatchSpec(OpaqueSpec(), [(3,)], ["item"])),
        )

        assert inferred.event_template == concrete.event_template
        assert inferred.event_template["b"] == concrete.event_template["b"]

    def test_a_concrete_axis_still_requires_an_exact_match(self):
        """A fixed multiplicity is fixed, as a fixed array dimension is."""
        declared = RecordSpec(b=BatchSpec(OpaqueSpec(), [(4,)], ["item"]))

        with pytest.raises(ValueError, match="does not conform to its field spec"):
            Record("r", b=self._batch(3), event_template=declared)

    def test_a_value_carrying_no_multiplicity_says_so(self):
        """A raw value is not a batch, so there is nothing to bind an axis from."""
        with pytest.raises(ValueError, match="exposes no schema to bind it against"):
            Record("r", b=jnp.zeros(3), event_template=self._declared())

    def test_a_level_name_mismatch_is_refused_rather_than_bound(self):
        """The tiling is structure, so it is checked rather than inferred."""
        declared = RecordSpec(b=BatchSpec(OpaqueSpec(), [("n",)], ["draw"]))

        with pytest.raises(ValueError, match=r"has levels \['item'\], expected \['draw'\]"):
            Record("r", b=self._batch(3), event_template=declared)

    def test_one_level_may_hold_several_symbolic_axes(self):
        """A level holding two axes binds each in turn."""
        declared = RecordSpec(b=BatchSpec(OpaqueSpec(), [("rows", "cols")], ["grid"]))

        record = Record("r", b=self._grid((3, 4)), event_template=declared)

        assert record.event_template["b"].axis_groups == ((3, 4),)

    def test_a_name_repeated_within_one_level_declares_a_square_grid(self):
        """`("n", "n")` binds once and demands both axes agree."""
        declared = RecordSpec(b=BatchSpec(OpaqueSpec(), [("n", "n")], ["grid"]))

        record = Record("r", b=self._grid((3, 3)), event_template=declared)
        assert record.event_template["b"].axis_groups == ((3, 3),)

        with pytest.raises(ValueError, match=r"symbolic dimension 'n' to 4, .*already bound to 3"):
            Record("r", b=self._grid((3, 4)), event_template=declared)

    def test_levels_bind_independently(self):
        """Two levels, two dimensions, each read off its own axis."""
        declared = RecordSpec(b=BatchSpec(OpaqueSpec(), [("c",), ("d",)], ["chain", "draw"]))

        record = Record(
            "r",
            b=self._grid((2, 4), ["chain", "draw"], (1, 1)),
            event_template=declared,
        )

        assert record.event_template["b"].axis_groups == ((2,), (4,))

    def test_a_level_arity_mismatch_names_the_tiling(self):
        """Two declared axes in a level do not bind against an actual one."""
        declared = RecordSpec(b=BatchSpec(OpaqueSpec(), [("a", "b")], ["grid"]))

        with pytest.raises(ValueError, match=r"tiles its axes as \[1\], expected \[2\]"):
            Record("r", b=self._batch(3, level="grid"), event_template=declared)

    def test_a_partially_bindable_template_binds_what_it_can(self):
        """Binding is a refinement, so an unbindable name is left free.

        A bare callable declares neither side, so `k` has nothing to bind
        against, while the batch beside it still binds `n`. The result is a
        template that is neither concrete nor refused.
        """
        declared = RecordSpec(
            f=FunctionSpec(RecordSpec(x=NumericArraySpec(shape=("k",))), None),
            b=BatchSpec(OpaqueSpec(), [("n",)], ["item"]),
        )

        record = Record("r", f=lambda x: x, b=self._batch(3), event_template=declared)

        assert not record.event_template.is_concrete
        assert record.event_template.free_dims == frozenset({"k"})
        assert record.event_template["b"].axis_groups == ((3,),)


class TestEverySpecBindsWhatItDeclares:
    """A spec that reports dimensions implements binding for them.

    `free_dims`, `_substitute_dims`, and the two binding methods are one contract:
    a spec that reports a name and leaves binding to the base class would raise
    the base's refusal at the moment the name had to be resolved. The check is
    over the live subclasses, so a spec added later is held to it too.
    """

    @staticmethod
    def _concrete_specs():
        seen: list[type[TermSpec]] = []
        pending = [TermSpec]
        while pending:
            for subclass in pending.pop().__subclasses__():
                if subclass not in seen:
                    seen.append(subclass)
                    pending.append(subclass)
        return [spec for spec in seen if not getattr(spec, "__abstractmethods__", False)]

    def test_the_inventory_is_not_vacuous(self):
        """The walk finds the specs it is meant to hold."""
        found = set(self._concrete_specs())

        assert {NumericArraySpec, OpaqueSpec, RecordSpec, DistributionSpec, FunctionSpec} <= found
        assert BatchSpec in found

    @pytest.mark.parametrize("method", ["_bind_dims_from_value", "_bind_dims_from_spec"])
    def test_a_spec_reporting_dimensions_overrides_binding(self, method):
        """Whatever reports a dimension resolves it, rather than inheriting a refusal."""
        for spec in self._concrete_specs():
            if spec.free_dims is TermSpec.free_dims:
                continue  # declares no dimensions, so the default is the answer
            assert getattr(spec, method) is not getattr(TermSpec, method), (
                f"{spec.__name__} reports free_dims but inherits {method}"
            )

    def test_a_spec_declaring_no_dimensions_keeps_the_default(self):
        """`OpaqueSpec` declares none, so the base class answers for it."""
        assert OpaqueSpec.free_dims is TermSpec.free_dims
        assert OpaqueSpec._bind_dims_from_value is TermSpec._bind_dims_from_value

    def test_the_default_refuses_rather_than_passing_silently(self):
        """A spec that reported a name it could not bind would say so."""

        @dataclass(frozen=True)
        class _DimlessButClaiming(TermSpec):
            @property
            def free_dims(self):
                return frozenset({"n"})

            def is_valid(self, value: Any) -> bool:
                return True

        with pytest.raises(ValueError, match="cannot bind from a value"):
            _DimlessButClaiming()._bind_dims_from_value(object(), {}, "p")

    def test_an_array_binds_its_own_shape(self):
        """`NumericArraySpec` owns its binding rather than being special-cased by the pass."""
        bindings: dict[str, int] = {}

        NumericArraySpec(shape=("n", "m"))._bind_dims_from_value(jnp.zeros((2, 5)), bindings, "p")

        assert bindings == {"n": 2, "m": 5}
