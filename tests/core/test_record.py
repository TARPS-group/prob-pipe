"""Tests for probpipe.core.record.Record."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Normal,
    NumericArray,
    OpaqueSpec,
    Provenance,
    Record,
    RecordSpec,
    provenance_ancestors,
)
from probpipe.core.record import _pack_fields

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_kwargs(self):
        v = Record(
            {"r": 1.8, "K": 70.0, "phi": 10.0},
            label="r",
        )
        assert v.fields == ("r", "K", "phi")  # insertion order

    def test_dict_positional(self):
        v = Record(
            {"a": 1.0, "b": 2.0},
            label="r",
        )
        assert v.fields == ("a", "b")

    def test_polymorphic_template_is_jointly_bound_and_stored_concrete(self):
        declaration = RecordSpec(x=("obs", 2), nested=RecordSpec(y=("obs",)))

        record = Record(
            {"x": np.zeros((4, 2)), "nested": {"y": np.ones((4,))}},
            event_template=declaration,
            label="data",
        )

        assert record.event_template == RecordSpec(x=(4, 2), nested=RecordSpec(y=(4,)))
        assert record.event_template.is_concrete
        assert declaration.free_dims == frozenset({"obs"})

    def test_polymorphic_template_reports_joint_binding_conflict(self):
        declaration = RecordSpec(x=("obs", 2), y=("obs",))

        with pytest.raises(
            ValueError,
            match=r"Record 'data'/y.*'obs'.*already bound to 4",
        ):
            Record(
                {"x": np.zeros((4, 2)), "y": np.ones((5,))},
                event_template=declaration,
                label="data",
            )

    def test_positional_accepts_any_mapping(self):
        # The positional arg accepts any collections.abc.Mapping, not just dict.
        from collections import OrderedDict
        from types import MappingProxyType

        from probpipe import NumericRecord

        v = Record(
            MappingProxyType({"a": 1.0, "b": 2.0}),
            label="r",
        )
        assert v.fields == ("a", "b")
        nr = NumericRecord(
            OrderedDict([("z", jnp.zeros(2)), ("a", 1.0)]),
            label="nr",
        )
        assert nr.fields == ("z", "a")  # mapping iteration order preserved

    def test_insertion_order_preserved(self):
        v = Record(
            {"z": 1.0, "a": 2.0, "m": 3.0},
            label="r",
        )
        # Insertion order, NOT alphabetical.
        assert v.fields == ("z", "a", "m")

    def test_slash_in_field_name_rejected(self):
        with pytest.raises(ValueError, match="must not contain '/'"):
            Record.from_fields(**{"a/b": 1.0})

    def test_merge_refuses_a_field_against_a_group(self):
        with pytest.raises(ValueError, match="single field in one record but a group of fields"):
            Record(
                {"x": 1.0},
                label="a",
            ).merge(
                Record(
                    {
                        "x": Record(
                            {"z": 1.0},
                            label="c",
                        )
                    },
                    label="b",
                )
            )

    def test_dict_and_kwargs_raises(self):
        with pytest.raises(TypeError, match="unexpected keyword"):
            Record(
                {"a": 1.0},
                b=2.0,
                label="r",
            )

    def test_the_empty_record_is_legal(self):
        """A record is a named tree, and the tree with no branches is one."""
        empty = Record({}, label="r")

        assert (empty.label, list(empty.event_template)) == ("r", [])

    def test_all_numeric_promotes_and_coerces(self):
        from probpipe import NumericRecord

        arr = np.array([1.0, 2.0, 3.0])
        v = Record(
            {"x": arr},
            label="r",
        )
        # All-numeric construction promotes to NumericRecord; leaves are
        # stored in native form (nothing is coerced at construction).
        assert type(v) is NumericRecord
        assert v.raw("x") is arr

    @pytest.mark.parametrize(
        "fields",
        [
            pytest.param({"a": np.zeros(3)}, id="all-numeric"),
            pytest.param({"a": np.zeros(3), "b": {}}, id="empty-nested-mapping"),
            pytest.param({"a": np.zeros(3), "b": {"c": {}}}, id="empty-nested-twice"),
            pytest.param(
                {
                    "a": np.zeros(3),
                    "b": Record(
                        {},
                        label="b",
                    ),
                },
                id="empty-nested-record",
            ),
            pytest.param({}, id="empty-root"),
            pytest.param({"a": np.zeros(3), "s": "text"}, id="non-numeric-field"),
        ],
    )
    def test_the_class_agrees_with_the_schema_it_carries(self, fields):
        """The value probe and the schema probe answer the same question.

        ``Record.__new__`` picks the class from the raw values while the
        carried schema is inferred separately, so the two predicates must
        decide alike. An empty record holds no non-numeric leaf and lays out
        flat at length zero, so both count it numeric, nested or at the root.
        """
        from probpipe import NumericRecord, NumericRecordSpec

        record = Record(
            fields,
            label="r",
        )
        assert isinstance(record, NumericRecord) == isinstance(record.spec, NumericRecordSpec), (
            f"{type(record).__name__} carries {type(record.spec).__name__}"
        )
        if isinstance(record, NumericRecord):
            # The promised numeric API is actually usable.
            assert record.to_vector().shape == (record.spec.vector_size,)

    def test_mixed_record_stores_values_verbatim(self):
        arr = np.array([1.0, 2.0, 3.0])
        v = Record(
            {"x": arr, "label": "tag"},
            label="r",
        )
        # A mixed record stays a plain Record and stores leaves as-is.
        assert v.raw("x") is arr

    def test_accepts_opaque_leaves(self):
        v = Record(
            {"label": "horseshoe", "x": 1.0},
            label="r",
        )
        assert v.raw("label") == "horseshoe"
        assert v["x"] == 1.0

    def test_jax_arrays(self):
        arr = jnp.array([1.0, 2.0])
        v = Record(
            {"x": arr},
            label="r",
        )
        assert v.raw("x") is arr

    def test_scalars(self):
        v = Record(
            {"a": 1, "b": 2.5, "c": True},
            label="r",
        )
        # All-numeric records promote; scalar values coerce to jax scalars
        # and stay value-equal.
        assert v["a"] == 1
        assert v["b"] == 2.5
        assert bool(v["c"]) is True

    def test_nested(self):
        inner = Record(
            {"x": 1.0, "y": 2.0},
            label="r",
        )
        outer = Record(
            {"params": inner, "z": 3.0},
            label="r",
        )
        assert isinstance(outer.at_path("params"), Record)
        assert outer["params/x"] == 1.0

    def test_from_dict(self):
        v = Record.from_dict("r", {"a": 1.0, "b": 2.0})
        assert v.fields == ("a", "b")

    def test_list_input(self):
        v = Record(
            {"x": [1.0, 2.0, 3.0]},
            label="r",
        )
        # Stored as-is — caller decides conversion.
        assert v.raw("x") == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Field access
# ---------------------------------------------------------------------------


class TestFieldAccess:
    @pytest.fixture
    def v(self):
        return Record(
            {"r": 1.8, "K": 70.0, "phi": 10.0},
            label="r",
        )

    def test_item(self, v):
        np.testing.assert_allclose(float(v["K"]), 70.0, rtol=1e-5)

    def test_key_path_tuple(self):
        v = Record(
            {
                "params": Record(
                    {"r": 1.8, "K": 70.0},
                    label="r",
                ),
                "obs": Record(
                    {"y": np.zeros(5)},
                    label="r",
                ),
            },
            label="r",
        )
        np.testing.assert_allclose(float(v["params", "r"]), 1.8, rtol=1e-5)

    def test_key_path_string(self):
        v = Record(
            {
                "params": Record(
                    {"r": 1.8, "K": 70.0},
                    label="r",
                ),
                "obs": Record(
                    {"y": np.zeros(5)},
                    label="r",
                ),
            },
            label="r",
        )
        np.testing.assert_allclose(float(v["params/r"]), 1.8, rtol=1e-5)

    def test_key_path_string_three_levels(self):
        v = Record(
            {
                "a": Record(
                    {
                        "b": Record(
                            {"c": 42.0},
                            label="r",
                        )
                    },
                    label="r",
                )
            },
            label="r",
        )
        assert v["a/b/c"] == 42.0

    def test_path_in_membership(self):
        v = Record(
            {
                "params": Record(
                    {"r": 1.8, "K": 70.0},
                    label="r",
                )
            },
            label="r",
        )
        assert "params/r" in v
        assert "params/missing" not in v
        assert "missing/r" not in v

    def test_path_through_non_record_raises_clear_keyerror(self):
        """Descending past a leaf via path syntax must raise ``KeyError`` with
        a path-aware message — not a numpy ``IndexError``."""
        v = Record(
            {"a": np.array([1.0, 2.0])},
            label="r",
        )
        with pytest.raises(KeyError, match="is a field, not a group"):
            v["a/b"]
        # __contains__ swallows the same case to False.
        assert "a/b" not in v

    def test_fields(self, v):
        assert v.fields == ("r", "K", "phi")

    def test_len(self, v):
        assert len(v) == 3

    def test_contains(self, v):
        assert "r" in v
        assert "missing" not in v

    def test_iter(self, v):
        assert list(v) == ["r", "K", "phi"]

    def test_items(self, v):
        items = list(v.items())
        assert len(items) == 3
        assert items[0][0] == "r"

    def test_keys(self, v):
        assert list(v.keys()) == ["r", "K", "phi"]

    def test_values_iter(self, v):
        vals = list(v.values())
        assert len(vals) == 3

    def test_missing_item_raises(self, v):
        with pytest.raises(KeyError):
            v["nonexistent"]

    def test_bad_key_type_raises(self, v):
        with pytest.raises(TypeError, match="key must be str"):
            v[42]


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_setattr_raises(self):
        v = Record(
            {"x": 1.0},
            label="r",
        )
        with pytest.raises(AttributeError, match="immutable"):
            v.x = 2.0

    def test_delattr_raises(self):
        v = Record(
            {"x": 1.0},
            label="r",
        )
        with pytest.raises(AttributeError, match="immutable"):
            del v.x

    def test_replace(self):
        v = Record(
            {"a": 1.0, "b": 2.0},
            label="r",
        )
        v2 = v.replace(b=3.0)
        assert v["b"] == 2.0  # original unchanged
        assert v2["b"] == 3.0

    def test_replace_nonexistent_raises(self):
        v = Record(
            {"a": 1.0},
            label="r",
        )
        with pytest.raises(KeyError, match="z"):
            v.replace(z=5.0)

    def test_replace_nested_path(self):
        v = Record(
            {
                "physics": Record(
                    {"force": 1.0, "mass": 2.0},
                    label="r",
                ),
                "obs": 3.0,
            },
            label="r",
        )
        v2 = v.replace({"physics/mass": 9.0})
        assert v2["physics/mass"] == 9.0
        assert v2["physics/force"] == 1.0  # untouched

    def test_merge(self):
        v1 = Record(
            {"a": 1.0},
            label="r",
        )
        v2 = Record(
            {"b": 2.0},
            label="r",
        )
        merged = v1.merge(v2)
        assert merged.fields == ("a", "b")

    def test_merge_overlap_raises(self):
        v1 = Record(
            {"a": 1.0},
            label="r",
        )
        v2 = Record(
            {"a": 2.0},
            label="r",
        )
        with pytest.raises(ValueError, match="cannot merge: both have the field 'a'"):
            v1.merge(v2)

    def test_without(self):
        v = Record(
            {"a": 1.0, "b": 2.0, "c": 3.0},
            label="r",
        )
        v2 = v.without("b")
        assert v2.fields == ("a", "c")

    def test_without_nonexistent_key_raises(self):
        """Removing a key that doesn't exist raises KeyError (leaf-keyed contract)."""
        v = Record(
            {"a": 1.0, "b": 2.0},
            label="r",
        )
        with pytest.raises(KeyError):
            v.without("z")

    def test_without_nested_path(self):
        v = Record(
            {
                "physics": Record(
                    {"force": 1.0, "mass": 2.0},
                    label="r",
                ),
                "obs": 3.0,
            },
            label="r",
        )
        assert tuple(v.without("physics/mass").keys()) == ("physics/force", "obs")
        assert tuple(v.without("physics").keys()) == ("obs",)

    def test_without_all_raises(self):
        v = Record(
            {"a": 1.0},
            label="r",
        )
        with pytest.raises(ValueError, match=r"without\(\) cannot remove every field"):
            v.without("a")

    # replace / merge / without must preserve the subclass (regression:
    # NumericRecord was silently downgraded to Record by these methods).

    def test_replace_preserves_numeric_record(self):
        from probpipe import NumericRecord

        nr = NumericRecord(
            {"a": 1.0, "b": 2.0},
            label="nr",
        )
        assert type(nr.replace(a=3.0)) is NumericRecord

    def test_merge_preserves_numeric_record(self):
        from probpipe import NumericRecord

        nr = NumericRecord(
            {"a": 1.0},
            label="nr",
        )
        assert (
            type(
                nr.merge(
                    NumericRecord(
                        {"b": 2.0},
                        label="nr",
                    )
                )
            )
            is NumericRecord
        )

    def test_without_preserves_numeric_record(self):
        from probpipe import NumericRecord

        nr = NumericRecord(
            {"a": 1.0, "b": 2.0},
            label="nr",
        )
        assert type(nr.without("a")) is NumericRecord


# ---------------------------------------------------------------------------
# Storage policy (no auto-conversion)
# ---------------------------------------------------------------------------


class TestStorage:
    """Record stores leaves verbatim — no coercion at construction.

    Together these tests pin down the storage policy: any time a new
    leaf type (or conversion layer) is added to ``Record``, one of
    these assertions will be the first to fail.
    """

    def test_numpy_stored_verbatim(self):
        arr = np.array([1.0, 2.0])
        v = Record(
            {"x": arr},
            label="r",
        )
        # Native storage: promotion never coerces — the numpy leaf is stored
        # verbatim on the promoted and the mixed record alike.
        assert v.raw("x") is arr
        mixed = Record(
            {"x": arr, "label": "tag"},
            label="r",
        )
        assert mixed.raw("x") is arr

    def test_scalar_coerced_by_promotion(self):
        v = Record(
            {"x": 42.0},
            label="r",
        )
        assert v["x"] == 42.0
        assert isinstance(v.raw("x"), jnp.ndarray)
        mixed = Record(
            {"x": 42.0, "label": "tag"},
            label="r",
        )
        assert isinstance(mixed.raw("x"), float)

    def test_jax_stored_verbatim(self):
        arr = jnp.array([1.0, 2.0])
        v = Record(
            {"x": arr},
            label="r",
        )
        assert v.raw("x") is arr

    def test_string_stored_verbatim(self):
        v = Record(
            {"x": "hello", "y": 1.0},
            label="r",
        )
        assert v.raw("x") == "hello"

    def test_heterogeneous_leaves(self):
        """Strings, numbers, and arrays co-exist in a plain Record."""
        v = Record(
            {"label": "x", "count": 1.0, "array": jnp.zeros(3)},
            label="r",
        )
        assert v.raw("label") == "x"
        assert v["count"] == 1.0
        assert v["array"].shape == (3,)

    def test_xarray_stored_verbatim(self):
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["time"],
            coords={"time": [10, 20, 30]},
        )
        v = Record(
            {"y": da},
            label="r",
        )
        # DataArray is preserved, coords and all.
        assert v.raw("y") is da
        assert v.raw("y").dims == ("time",)
        np.testing.assert_array_equal(v.raw("y").coords["time"].values, [10, 20, 30])

    def test_xarray_leaf_survives_structural_edit(self):
        from probpipe import NumericRecord

        xr = pytest.importorskip("xarray")
        da = xr.DataArray([1.0, 2.0, 3.0], dims=["t"], coords={"t": [0, 1, 2]})
        v = Record(
            {"y": da, "z": jnp.array(1.0)},
            label="r",
        )
        # Backend leaves are first-class numeric: the record promotes, and a
        # structural edit keeps the native leaf verbatim.
        assert isinstance(v, NumericRecord)
        edited = v.without("z")
        assert isinstance(edited, NumericRecord)
        assert edited.raw("y") is da
        assert v.replace(z=jnp.array(2.0)).raw("y") is da

    def test_backend_leaf_gives_numeric_template(self):
        # A native backend leaf infers a NumericArraySpec, so the template is
        # numeric and stays in step with the record's promoted class.
        from probpipe.core._specs import NumericArraySpec, NumericRecordSpec

        xr = pytest.importorskip("xarray")
        da = xr.DataArray([1.0, 2.0, 3.0], dims=["t"])
        v = Record(
            {"y": da},
            label="r",
        )
        assert isinstance(v.event_template, NumericRecordSpec)
        assert v.event_template["y"] == NumericArraySpec((3,))


class TestNumericAPIOnRecord:
    """The numeric-1-D APIs live on NumericRecord, not on Record.

    ``to_vector`` / ``vector_size`` require numeric leaves, so they belong
    only on ``NumericRecord``; if someone re-adds one to ``Record``, this
    fails. The general decomposition (``values()`` export +
    ``from_field_values`` reconstruction; leaves kept whole, any type) DOES
    live on ``Record`` — asserted present here so the two vocabularies don't
    drift. The JAX-pytree ``flatten`` / ``unflatten`` are *not* Record methods
    (use ``jax.tree_util`` directly). Implementation-detail attributes like
    ``_resolved`` / ``_coords`` are intentionally not checked here.
    """

    def test_numeric_vector_ops_absent_from_mixed_record(self):
        v = Record(
            {"a": 1.0, "label": "tag"},
            label="r",
        )
        for attr in ("to_vector", "vector_size", "zip"):
            assert not hasattr(v, attr), f"Record should not expose {attr!r}"
        assert not hasattr(Record, "zip")

    def test_general_decomposition_present_flatten_absent(self):
        # values() / from_field_values is the general (any-leaf) ProbPipe leaf
        # traversal on Record; the JAX-pytree flatten/unflatten are NOT Record
        # methods.
        v = Record(
            {"a": 1.0, "label": "x"},
            label="r",
        )
        assert Record.from_field_values(v.label, v.event_template, v.values()) == v
        assert not hasattr(Record, "flatten")
        assert not hasattr(Record, "unflatten")


class TestGeneralDecomposition:
    """``values()`` / ``from_field_values`` are the general (any-leaf-type)
    leaf (de)composition at the *template's* granularity: each leaf is kept
    whole (never raveled), visited in canonical ``keys()`` order. They are
    distinct from the numeric ``to_vector`` / ``from_vector`` (which ravel and
    concatenate numeric leaves) and from JAX's finer pytree view (which
    descends into container-valued opaque leaves — see
    ``test_container_leaf_is_one_whole_leaf``).
    """

    def test_values_keeps_leaves_whole_in_canonical_order(self):
        v = Record(
            {"x": jnp.array([1.0, 2.0]), "label": "horseshoe"},
            label="r",
        )
        leaves = list(v.values())
        assert leaves[0].shape == (2,)  # kept whole, not raveled
        assert leaves[1].raw() == "horseshoe"  # a view of the opaque leaf stored as-is
        assert list(v.keys()) == ["x", "label"]  # canonical order

    def test_container_leaf_is_one_whole_leaf(self):
        # A tuple field is ONE opaque leaf at template granularity; JAX's
        # pytree view descends into it (documented divergence).
        v = Record(
            {"x": jnp.zeros(2), "pair": (jnp.array(1.0), jnp.array(2.0))},
            label="r",
        )
        leaves = list(v.values())
        assert len(leaves) == 2  # x, pair (whole)
        assert isinstance(leaves[1].raw(), tuple)
        assert len(jax.tree_util.tree_leaves(v)) == 3  # JAX descends the tuple

    def test_roundtrip_with_opaque_leaf(self):
        # Opaque (non-numeric) leaves round-trip — unlike to_vector.
        v = Record(
            {"x": jnp.array([1.0, 2.0]), "label": "horseshoe", "count": 3},
            label="r",
        )
        assert Record.from_field_values(v.label, v.event_template, v.values()) == v

    def test_roundtrip_with_backend_leaf(self):
        # A native backend leaf (xarray) round-trips through
        # from_field_values with its class, template, and native leaf intact:
        # nothing is coerced, so the round-trip is exact.
        from probpipe import NumericRecord
        from probpipe.core._specs import NumericRecordSpec

        xr = pytest.importorskip("xarray")
        da = xr.DataArray([1.0, 2.0, 3.0], dims=["t"], coords={"t": [0, 1, 2]})
        v = Record(
            {"x": da},
            label="obs",
        )
        assert isinstance(v, NumericRecord)
        assert isinstance(v.event_template, NumericRecordSpec)
        rebuilt = Record.from_field_values(v.label, v.event_template, v.values())
        assert type(rebuilt) is type(v)
        assert rebuilt.raw("x") is da
        assert rebuilt == v

    def test_roundtrip_with_out_of_order_template(self):
        # ``values()`` yields in canonical order and ``from_field_values`` walks
        # the template in that same order, so a record built with an explicitly
        # out-of-order template round-trips without transposing field values.
        v = Record(
            {"b": 2.0, "a": 1.0},
            event_template=RecordSpec(a=(), b=()),
            label="r",
        )
        rebuilt = Record.from_field_values(v.label, v.event_template, v.values())
        assert rebuilt == v
        assert float(rebuilt["a"]) == 1.0
        assert float(rebuilt["b"]) == 2.0

    def test_roundtrip_preserves_user_name(self):
        # ``==`` ignores the label, so assert label fidelity separately: the
        # reconstructed record carries exactly the label passed in.
        v = Record(
            {
                "theta": Record(
                    {"loc": jnp.array([0.0, 1.0]), "label": "p"},
                    label="theta",
                ),
                "tag": "t",
            },
            label="mine",
        )
        rebuilt = Record.from_field_values(v.label, v.event_template, v.values())
        assert rebuilt.label == "mine"

    def test_numeric_record_roundtrip(self):
        from probpipe import NumericRecord

        v = NumericRecord(
            {
                "a": jnp.array([1.0, 2.0, 3.0]),
                "b": NumericRecord(
                    {"c": jnp.array(5.0)},
                    label="nr",
                ),
            },
            label="nr",
        )
        rebuilt = Record.from_field_values(v.label, v.event_template, v.values())
        assert rebuilt == v
        assert isinstance(rebuilt, NumericRecord)
        assert isinstance(rebuilt.at_path("b"), NumericRecord)

    def test_mixed_nested_record_roundtrip(self):
        # A nested mixed record with both numeric and opaque leaves.
        v = Record(
            {
                "theta": Record(
                    {"loc": jnp.array([0.0, 1.0]), "label": "prior"},
                    label="theta",
                ),
                "tag": "run-7",
            },
            label="r",
        )
        assert Record.from_field_values(v.label, v.event_template, v.values()) == v

    def test_wrong_leaf_count_raises(self):
        v = Record(
            {"a": 1.0, "b": 2.0},
            label="r",
        )
        with pytest.raises(ValueError, match="expected 2"):
            Record.from_field_values("v", v.event_template, [1.0])

    def test_jax_pytree_roundtrip_still_works(self):
        # Record stays a registered pytree; the JAX path round-trips via
        # jax.tree_util (the documented finer-granularity escape hatch).
        v = Record(
            {"x": jnp.array([1.0, 2.0]), "label": "horseshoe"},
            label="r",
        )
        leaves, treedef = jax.tree_util.tree_flatten(v)
        assert jax.tree_util.tree_unflatten(treedef, leaves) == v


class TestKeysAgreement:
    """A Record's own ``keys()`` must always equal its ``event_template``'s
    ``keys()`` — both define "what is a leaf" and the ``event_template`` is the
    source of truth. In particular, a cross-type nested value (an
    ``RecordSpec`` stored as a ``Record`` field value) is one opaque leaf,
    not an internal node to descend into.
    """

    def test_flat(self):
        v = Record(
            {"a": 1.0, "b": jnp.zeros(3), "c": "x"},
            label="r",
        )
        assert list(v.keys()) == list(v.event_template.keys())

    def test_nested_in_family(self):
        v = Record(
            {
                "theta": Record(
                    {"loc": jnp.zeros(2), "s": jnp.ones(3)},
                    label="r",
                ),
                "top": jnp.array(1.0),
            },
            label="r",
        )
        assert list(v.keys()) == list(v.event_template.keys())
        assert list(v.keys()) == ["theta/loc", "theta/s", "top"]

    def test_cross_type_value_is_one_opaque_leaf(self):
        # A RecordSpec stored as a Record field value is an opaque leaf,
        # NOT an internal node: keys() must not descend into it.
        v = Record(
            {"weird": RecordSpec(a=(2,)), "x": jnp.array([1.0, 2.0])},
            label="r",
        )
        assert list(v.keys()) == ["weird", "x"]
        assert list(v.keys()) == list(v.event_template.keys())


# ---------------------------------------------------------------------------
# JAX PyTree
# ---------------------------------------------------------------------------


class TestPyTree:
    def test_tree_map(self):
        v = Record(
            {"a": 1.0, "b": 2.0},
            label="r",
        )
        v2 = jax.tree.map(lambda x: x * 2, v)
        assert isinstance(v2, Record)
        assert v2["a"] == 2.0
        assert v2["b"] == 4.0

    def test_tree_leaves(self):
        v = Record(
            {"a": jnp.array(1.0), "b": jnp.array(2.0)},
            label="r",
        )
        leaves = jax.tree.leaves(v)
        assert len(leaves) == 2

    def test_tree_structure_roundtrip(self):
        v = Record(
            {"x": jnp.array([1.0, 2.0]), "y": jnp.array(3.0)},
            label="r",
        )
        leaves, treedef = jax.tree.flatten(v)
        v2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(v2, Record)
        assert v2.fields == v.fields

    def test_nested_tree_map(self):
        v = Record(
            {
                "params": Record(
                    {"r": 1.0, "K": 2.0},
                    label="r",
                ),
                "z": 3.0,
            },
            label="r",
        )
        v2 = jax.tree.map(lambda x: x + 10, v)
        assert isinstance(v2, Record)
        assert isinstance(v2.at_path("params"), Record)
        assert v2["params/r"] == 11.0
        assert v2["z"] == 13.0

    def test_tree_map_realigns_values_by_template_order(self):
        # Flatten emits children in the template's field order and unflatten
        # zips them back against that same order, so a record whose stored
        # field order differs from its explicit template survives a
        # round-trip with each value on its own field (never transposed).
        v = Record(
            {"b": 2.0, "a": 1.0},
            event_template=RecordSpec(a=(), b=()),
            label="r",
        )
        v2 = jax.tree.map(lambda x: x, v)
        assert float(v2["a"]) == 1.0
        assert float(v2["b"]) == 2.0

    def test_tree_roundtrip_equals_out_of_order_template(self):
        # Storage is canonicalized to template order at construction, so a
        # flatten/unflatten round-trip reproduces an equal record (equality is
        # order-sensitive) rather than one whose fields silently reordered.
        v = Record(
            {"b": 2.0, "a": 1.0},
            event_template=RecordSpec(a=(), b=()),
            label="r",
        )
        leaves, treedef = jax.tree_util.tree_flatten(v)
        assert jax.tree_util.tree_unflatten(treedef, leaves) == v

    def test_jit(self):
        v = Record(
            {"a": 1.0, "b": 2.0},
            label="r",
        )

        @jax.jit
        def f(vals):
            return vals["a"] + vals["b"]

        result = f(v)
        np.testing.assert_allclose(float(result), 3.0)

    def test_jit_returns_values(self):
        v = Record(
            {"a": 1.0, "b": 2.0},
            label="r",
        )

        @jax.jit
        def f(vals):
            return jax.tree.map(lambda x: x * 2, vals)

        result = f(v)
        assert isinstance(result, Record)
        np.testing.assert_allclose(float(result["a"]), 2.0)

    def test_vmap(self):
        batch = Record(
            {"x": jnp.array([1.0, 2.0, 3.0])},
            label="r",
        )

        @jax.vmap
        def f(vals):
            return vals["x"] ** 2

        result = f(batch)
        np.testing.assert_allclose(result, [1.0, 4.0, 9.0])

    def test_grad(self):
        v = Record(
            {"x": 1.0},
            label="r",
        )

        def f(vals):
            return vals["x"] ** 2

        grads = jax.grad(f)(v)
        assert isinstance(grads, Record)
        np.testing.assert_allclose(float(grads["x"]), 2.0)


# ---------------------------------------------------------------------------
# Backend conversion
# ---------------------------------------------------------------------------


class TestConversion:
    def test_to_dict_verbatim(self):
        arr = np.array([2.0, 3.0])
        v = Record(
            {"a": 1.0, "b": arr, "label": "tag"},
            label="r",
        )
        d = v.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) == {"a", "b", "label"}
        # No coercion beyond construction — values returned as stored.
        assert d["a"] == 1.0
        assert d["b"] is arr

    def test_to_numpy(self):
        v = Record(
            {"a": jnp.array(1.0), "b": jnp.array([2.0])},
            label="r",
        )
        d = v.to_numpy()
        assert isinstance(d["a"], np.ndarray)
        assert isinstance(d["b"], np.ndarray)

    def test_to_numpy_preserves_opaque(self):
        v = Record(
            {"label": "x", "y": np.array([1.0, 2.0])},
            label="r",
        )
        d = v.to_numpy()
        assert d["label"] == "x"
        assert isinstance(d["y"], np.ndarray)

    def test_to_dict_nested(self):
        v = Record(
            {
                "inner": Record(
                    {"x": 1.0},
                    label="r",
                ),
                "y": 2.0,
            },
            label="r",
        )
        d = v.to_dict()
        assert isinstance(d["inner"], dict)
        assert d["inner"]["x"] == 1.0

    def test_to_numeric_returns_numeric_record(self):
        from probpipe import NumericRecord

        v = Record(
            {"a": 1.0, "b": jnp.array([2.0, 3.0])},
            label="r",
        )
        nr = v.to_numeric()
        assert isinstance(nr, NumericRecord)
        assert nr.fields == ("a", "b")
        np.testing.assert_array_equal(np.asarray(nr["b"]), [2.0, 3.0])

    def test_xarray_metadata_kept_natively(self):
        """xarray dims / coords / attrs never leave the record: the leaf is
        stored natively (the migration replacement for the old
        ``to_datatree`` / ``from_datatree`` pair), and ``to_numeric()`` is
        validation, not conversion.
        """
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["time"],
            coords={"time": [10, 20, 30]},
            attrs={"units": "m"},
        )
        back = Record(
            {"y": da},
            label="r",
        ).to_numeric()
        assert back.raw("y") is da
        assert back.raw("y").dims == ("time",)
        np.testing.assert_array_equal(back.raw("y").coords["time"].values, [10, 20, 30])
        assert back.raw("y").attrs == {"units": "m"}

    def test_to_numeric_recurses_into_nested_records(self):
        """``to_numeric()`` recurses into nested non-NumericRecord children."""
        from probpipe import NumericRecord

        outer = Record(
            {
                "inner": Record(
                    {"a": 1.0},
                    label="r",
                ),
                "z": 2.0,
            },
            label="r",
        )
        nr = outer.to_numeric()
        assert isinstance(nr, NumericRecord)
        assert isinstance(nr.at_path("inner"), NumericRecord)
        assert nr["inner", "a"] == 1.0
        # Deterministic: a second conversion yields the same structure.
        nr2 = outer.to_numeric()
        assert nr.fields == nr2.fields
        for field in nr.fields:
            assert type(nr.at_path(field)) is type(nr2.at_path(field))


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


class TestEnsure:
    def test_values_passthrough(self):
        v = Record(
            {"x": 1.0},
            label="r",
        )
        assert Record.ensure(v) is v

    def test_dict_coercion(self):
        v = Record.ensure({"a": 1.0, "b": 2.0})
        assert isinstance(v, Record)
        assert v.fields == ("a", "b")

    def test_nested_dict_explodes_into_subtree(self):
        # A nested dict value denotes tree structure (mappings are never
        # leaves): it becomes a nested subtree, not a rejected mapping leaf.
        v = Record.ensure({"summary": {"mean": 1.0, "count": 2.0}, "x": 3.0})
        assert isinstance(v, Record)
        assert list(v.keys()) == ["summary/mean", "summary/count", "x"]
        assert v.label == "record(summary,x)"

    def test_array_coercion(self):
        v = Record.ensure(jnp.array([1.0, 2.0]), label="measurements")
        assert isinstance(v, Record)
        assert "data" in v
        np.testing.assert_allclose(np.asarray(v["data"]), [1.0, 2.0])

    def test_numpy_coercion(self):
        v = Record.ensure(np.array([1.0]), label="measurements")
        assert isinstance(v, Record)
        assert "data" in v

    def test_a_path_keyed_mapping_is_labeled_by_its_top_level_fields(self):
        v = Record.ensure({"y/a": 1.0, "y/b": 2.0})
        assert v.label == "record(y)"
        assert v.label == Record({"y": {"a": 1.0, "b": 2.0}}).label

    def test_a_mapping_takes_a_supplied_label(self):
        assert Record.ensure({"a": 1.0}, label="measurements").label == "measurements"

    def test_a_wrapped_tracked_term_lends_the_record_its_label(self):
        term = NumericArray(jnp.ones(3), label="temperature")
        v = Record.ensure(term)
        assert v.label == "temperature"
        assert term.label == "temperature"

    def test_a_supplied_label_overrides_a_wrapped_terms_label(self):
        term = NumericArray(jnp.ones(3), label="temperature")
        assert Record.ensure(term, label="reading").label == "reading"

    def test_a_record_passes_through_and_ignores_a_supplied_label(self):
        v = Record({"x": 1.0}, label="r")
        assert Record.ensure(v, label="other") is v

    def test_a_value_without_a_label_requires_one(self):
        with pytest.raises(
            TypeError, match=r"^Record\.ensure\(\) needs a label for a jax\.Array, .*label=\.\.\.$"
        ):
            Record.ensure(jnp.ones(3))

    def test_an_empty_mapping_without_a_label_is_refused(self):
        with pytest.raises(TypeError, match="label"):
            Record.ensure({})
        assert Record.ensure({}, label="empty").label == "empty"


class TestPackFields:
    """``_pack_fields`` builds a record labeled as the constructor labels it."""

    def test_the_record_is_labeled_by_its_fields_in_the_given_order(self):
        packed = _pack_fields(("b", "a"), {"a": 1.0, "b": 2.0})
        assert tuple(packed) == ("b", "a")
        assert packed.label == "record(b,a)"

    def test_path_keyed_fields_are_labeled_by_their_top_level_field(self):
        packed = _pack_fields(("y/a", "y/b"), {"y/a": 1.0, "y/b": 2.0})
        assert packed.label == "record(y)"

    def test_a_missing_or_unexpected_field_is_refused(self):
        with pytest.raises(TypeError, match=r"^Owner: expected exactly the fields"):
            _pack_fields(("a",), {"b": 1.0}, owner="Owner")


# ---------------------------------------------------------------------------
# Leaf-wise operations
# ---------------------------------------------------------------------------


class TestLeafOps:
    def test_map(self):
        v = Record(
            {"a": 2.0, "b": 3.0},
            label="r",
        )
        v2 = v.map(lambda x: x**2)
        assert v2["a"] == 4.0
        assert v2["b"] == 9.0

    def test_map_rederives_the_numeric_axis(self):
        """``map`` rebuilds through the base class, re-deciding promotion."""
        from probpipe import NumericRecord

        mixed = Record(
            {"a": 2.0, "label": "tag"},
            label="r",
        )
        assert type(mixed.map(lambda x: 1.0)) is NumericRecord  # promoted
        numeric = Record(
            {"a": 2.0, "b": 3.0},
            label="r",
        )
        assert type(numeric.map(lambda x: "s")) is Record  # demoted

    def test_map_preserves_numeric_record(self):
        """``map`` on a NumericRecord returns a NumericRecord as long as
        the mapped values remain numeric (validated by ``__init__``)."""
        from probpipe import NumericRecord

        nr = NumericRecord(
            {"a": 1.0, "b": 2.0},
            label="nr",
        )
        out = nr.map(lambda x: x * 2)
        assert type(out) is NumericRecord

    def test_map_on_numeric_record_demotes_on_non_numeric_output(self):
        """A map whose outputs are no longer numeric re-derives the numeric
        axis: the rebuild routes through the base class and demotes."""
        from probpipe import NumericRecord

        nr = NumericRecord(
            {"a": 1.0, "b": 2.0},
            label="nr",
        )
        out = nr.map(lambda x: "not numeric")
        assert type(out) is Record
        assert out.raw("a") == "not numeric"

    def test_map_nested(self):
        v = Record(
            {
                "inner": Record(
                    {"x": 2.0},
                    label="r",
                ),
                "y": 3.0,
            },
            label="r",
        )
        v2 = v.map(lambda x: x + 1)
        assert v2["inner/x"] == 3.0
        assert v2["y"] == 4.0

    def test_map_with_keys(self):
        v = Record(
            {"a": 1.0, "b": 2.0},
            label="r",
        )
        keys_seen = []
        v.map_with_keys(lambda k, x: keys_seen.append(k) or x)
        assert keys_seen == ["a", "b"]

    def test_map_with_keys_passes_full_path(self):
        v = Record(
            {
                "inner": Record(
                    {"x": 2.0},
                    label="r",
                ),
                "y": 3.0,
            },
            label="r",
        )
        keys_seen = []
        v.map_with_keys(lambda k, x: keys_seen.append(k) or x)
        assert keys_seen == ["inner/x", "y"]

    def test_map_forwards_args(self):
        v = Record(
            {"a": 1.0, "b": 2.0},
            label="r",
        )
        v2 = v.map(lambda x, bump: x + bump, bump=10.0)
        assert v2["a"] == 11.0 and v2["b"] == 12.0

    def test_map_rejects_node_return(self):
        v = Record(
            {"a": 1.0},
            label="r",
        )
        with pytest.raises(ValueError, match="must return a single value"):
            v.map(
                lambda x: Record(
                    {"z": x},
                    label="r",
                )
            )


# ---------------------------------------------------------------------------
# Repr and equality
# ---------------------------------------------------------------------------


class TestReprAndEquality:
    def test_repr_names_the_label_and_the_field_paths(self):
        assert (
            repr(
                Record(
                    {"a": 1.0, "b": "tag"},
                    label="r",
                )
            )
            == "Record('r', fields=('a', 'b'))"
        )

    def test_repr_of_a_numeric_record_names_its_class(self):
        assert (
            repr(
                Record(
                    {"x": jnp.zeros((3, 4))},
                    label="r",
                )
            )
            == "NumericRecord('r', fields=('x',))"
        )

    def test_repr_lists_nested_fields_by_path(self):
        school = Record(
            {"data": {"effect": 28.0, "se": 15.0}, "label": "A"},
            label="school",
        )
        assert repr(school) == "Record('school', fields=('data/effect', 'data/se', 'label'))"

    def test_equality(self):
        v1 = Record(
            {"a": 1.0, "b": 2.0},
            label="r",
        )
        v2 = Record(
            {"a": 1.0, "b": 2.0},
            label="r",
        )
        assert v1 == v2

    def test_inequality_values(self):
        v1 = Record(
            {"a": 1.0},
            label="r",
        )
        v2 = Record(
            {"a": 2.0},
            label="r",
        )
        assert v1 != v2

    def test_inequality_fields(self):
        v1 = Record(
            {"a": 1.0},
            label="r",
        )
        v2 = Record(
            {"b": 1.0},
            label="r",
        )
        assert v1 != v2

    def test_hash_includes_shape(self):
        """Records with the same field names but different shapes should
        hash differently."""
        v1 = Record(
            {"a": jnp.zeros(3)},
            label="r",
        )
        v2 = Record(
            {"a": jnp.zeros(5)},
            label="r",
        )
        assert hash(v1) != hash(v2)

    def test_hash_excludes_value(self):
        """Records with the same shape+dtype but different values hash
        the same (structural hash)."""
        v1 = Record(
            {"a": jnp.zeros(3)},
            label="r",
        )
        v2 = Record(
            {"a": jnp.ones(3)},
            label="r",
        )
        assert hash(v1) == hash(v2)

    def test_hash_distinguishes_dtype(self):
        """Records with the same shape but different dtype hash differently."""
        v1 = Record(
            {"a": jnp.zeros(3, dtype=jnp.float32)},
            label="r",
        )
        v2 = Record(
            {"a": jnp.zeros(3, dtype=jnp.int32)},
            label="r",
        )
        assert hash(v1) != hash(v2)

    def test_eq_type_strict(self):
        """Equality is class-strict; promotion aligns classes by content, so
        an all-numeric Record construction equals its NumericRecord twin."""
        from probpipe import NumericRecord

        r = Record(
            {"a": 1.0, "b": 2.0},
            label="r",
        )
        nr = NumericRecord(
            {"a": 1.0, "b": 2.0},
            label="nr",
        )
        assert type(r) is NumericRecord  # promoted
        assert r == nr
        mixed = Record(
            {"a": 1.0, "label": "tag"},
            label="r",
        )
        assert mixed != Record(
            {"a": 1.0, "label": "other"},
            label="r",
        )

    # Hash / eq contract: ``a == b`` must imply ``hash(a) == hash(b)``.
    # Regression: ``__hash__`` used to read raw ``.shape`` / ``.dtype`` while
    # ``__eq__`` coerced via ``jnp.asarray``, so
    # Record("r", a=1.0) == Record("r", a=jnp.asarray(1.0)) but the hashes differed.

    def test_hash_eq_contract_scalar_vs_zero_d_array(self):
        r1 = Record(
            {"a": 1.0},
            label="r",
        )
        r2 = Record(
            {"a": jnp.asarray(1.0)},
            label="r",
        )
        assert r1 == r2
        assert hash(r1) == hash(r2)

    def test_hash_eq_contract_python_int_vs_numpy(self):
        r1 = Record(
            {"a": 1},
            label="r",
        )
        r2 = Record(
            {"a": np.asarray(1)},
            label="r",
        )
        assert r1 == r2
        assert hash(r1) == hash(r2)

    def test_hash_eq_contract_opaque_leaf(self):
        """Two Records with the same string leaves must hash the same."""
        r1 = Record(
            {"label": "x", "count": 1.0},
            label="r",
        )
        r2 = Record(
            {"label": "x", "count": 1.0},
            label="r",
        )
        assert r1 == r2
        assert hash(r1) == hash(r2)

    # NaN leaves: equality is reflexive across INDEPENDENT copies, not only via
    # the identity fast-path. ``__eq__`` compares native values with
    # ``equal_nan`` (the same basis as the content fingerprint), so two
    # separately built records with a NaN leaf are equal and hash the same.
    # Regression: NaN leaves made ``__eq__`` non-reflexive across copies
    # (``jnp.array_equal`` treats NaN != NaN).

    def test_self_equality_with_nan(self):
        r = Record(
            {"x": jnp.array([jnp.nan, 1.0, jnp.nan])},
            label="r",
        )
        assert r == r

    def test_self_equality_with_nan_nested(self):
        r = Record(
            {
                "inner": Record(
                    {"x": jnp.array([jnp.nan])},
                    label="r",
                ),
                "y": jnp.nan,
            },
            label="r",
        )
        assert r == r

    def test_independent_copies_equal_with_nan(self):
        r1 = Record(
            {"x": jnp.array([jnp.nan, 1.0, jnp.nan])},
            label="r",
        )
        r2 = Record(
            {"x": jnp.array([jnp.nan, 1.0, jnp.nan])},
            label="r",
        )
        assert r1 is not r2
        assert r1 == r2
        assert hash(r1) == hash(r2)

    def test_independent_copies_equal_with_nan_nested(self):
        def mk():
            return Record(
                {
                    "inner": Record(
                        {"x": jnp.array([jnp.nan])},
                        label="r",
                    ),
                    "y": jnp.nan,
                },
                label="r",
            )

        assert mk() == mk()

    def test_nan_differs_from_finite(self):
        # equal_nan aligns NaN positions; a NaN vs a finite value at the same
        # position is still unequal.
        assert Record(
            {"x": jnp.array([jnp.nan, 1.0])},
            label="r",
        ) != Record(
            {"x": jnp.array([2.0, 1.0])},
            label="r",
        )

    def test_eq_agrees_with_fingerprint(self):
        # ``__eq__`` and ``fingerprint()`` share one native, full-precision
        # basis, so they agree in both directions: NaN copies are equal *and*
        # fingerprint-equal; a sub-float32 difference in float64 is unequal
        # *and* fingerprint-different (not silently equated under x32).
        from probpipe.core._fingerprint import fingerprint

        nan1 = Record(
            {"x": np.array([1.0, np.nan])},
            label="r",
        )
        nan2 = Record(
            {"x": np.array([1.0, np.nan])},
            label="r",
        )
        assert (nan1 == nan2) is True
        assert fingerprint(nan1) == fingerprint(nan2)

        near1 = Record(
            {"x": np.array([1.0])},
            label="r",
        )
        near2 = Record(
            {"x": np.array([1.0 + 1e-9])},
            label="r",
        )
        assert (near1 == near2) == (fingerprint(near1) == fingerprint(near2))
        assert near1 != near2


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Record carries the same ``.provenance`` / ``.with_provenance`` slot as
    Distribution, so workflow outputs can attach a Provenance node
    regardless of which of the three output types (Record, RecordBatch,
    Distribution) the broadcasting layer produced.
    """

    def test_initial_provenance_is_none(self):
        r = Record(
            {"x": 1.0, "y": 2.0},
            label="r",
        )
        assert r.provenance is None

    def test_with_provenance_sets_and_returns_self(self):
        r = Record(
            {"x": 1.0},
            label="r",
        )
        out = r.with_provenance(Provenance("op", parents=()))
        assert out is r
        assert r.provenance.operation == "op"

    def test_with_provenance_is_write_once(self):
        r = Record(
            {"x": 1.0},
            label="r",
        )
        r.with_provenance(Provenance("first", parents=()))
        with pytest.raises(RuntimeError, match="set only once"):
            r.with_provenance(Provenance("second", parents=()))

    # Semantic transformations reset the source — the new Record is a
    # different logical value even though the class preserves.

    def test_replace_resets_provenance(self):
        r = Record(
            {"x": 1.0},
            label="r",
        ).with_provenance(Provenance("orig", parents=()))
        r2 = r.replace(x=2.0)
        assert r2.provenance is None
        assert r.provenance.operation == "orig"  # original unaffected

    def test_merge_resets_provenance(self):
        r = Record(
            {"x": 1.0},
            label="r",
        ).with_provenance(Provenance("orig", parents=()))
        merged = r.merge(
            Record(
                {"y": 2.0},
                label="r",
            )
        )
        assert merged.provenance is None

    def test_without_resets_provenance(self):
        r = Record(
            {"x": 1.0, "y": 2.0},
            label="r",
        ).with_provenance(Provenance("orig", parents=()))
        r2 = r.without("y")
        assert r2.provenance is None

    def test_map_resets_provenance(self):
        r = Record(
            {"x": 1.0},
            label="r",
        ).with_provenance(Provenance("orig", parents=()))
        r2 = r.map(lambda v: v + 1)
        assert r2.provenance is None

    # Structural equality / hashing ignore source — two Records with the
    # same fields but different provenance are still equal.

    def test_eq_ignores_provenance(self):
        r1 = Record(
            {"x": 1.0},
            label="r",
        ).with_provenance(Provenance("a", parents=()))
        r2 = Record(
            {"x": 1.0},
            label="r",
        ).with_provenance(Provenance("b", parents=()))
        assert r1 == r2

    def test_hash_ignores_provenance(self):
        r1 = Record(
            {"x": 1.0},
            label="r",
        ).with_provenance(Provenance("a", parents=()))
        r2 = Record(
            {"x": 1.0},
            label="r",
        )
        assert hash(r1) == hash(r2)

    # Pytree roundtrip drops the source (runtime-only metadata — a
    # Provenance parent isn't hashable by structure, so pushing it into
    # the aux tuple would break jax.tree_util.tree_unflatten's equality
    # semantics). Document this caveat with a test.

    def test_pytree_roundtrip_drops_provenance(self):
        r = Record(
            {"x": 1.0, "y": jnp.array([2.0, 3.0])},
            label="r",
        )
        r.with_provenance(Provenance("op", parents=()))
        leaves, treedef = jax.tree_util.tree_flatten(r)
        r2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert r2.provenance is None
        # But the Record is otherwise structurally identical.
        assert r2 == r

    # Integration: walk provenance from a Record through a Distribution
    # ancestor via provenance_ancestors.

    def test_provenance_ancestors_walks_through_distribution(self):
        prior = Normal("prior", loc=0.0, scale=1.0)
        r = Record(
            {"theta": 1.0},
            label="r",
        ).with_provenance(Provenance("draw", parents=(prior,)))
        ancestors = provenance_ancestors(r)
        assert len(ancestors) == 1
        assert ancestors[0] is prior

    def test_provenance_ancestors_walks_nested_records(self):
        prior = Normal("theta", loc=0.0, scale=1.0, label="prior")
        middle = Record(
            {"theta": 1.0},
            label="r",
        ).with_provenance(Provenance("draw", parents=(prior,)))
        outer = Record(
            {"result": 2.0},
            label="r",
        ).with_provenance(Provenance("transform", parents=(middle,)))
        ancestors = provenance_ancestors(outer)
        names = [getattr(a, "label", None) for a in ancestors]
        assert names == [middle.label, "prior"]


# ---------------------------------------------------------------------------
# The stored spec
# ---------------------------------------------------------------------------


class TestSpecStorage:
    """``spec`` and ``event_template`` return the same stored ``RecordSpec``."""

    def test_inferred_spec_and_event_template_are_the_same_record_spec(self):
        r = Record(
            {"x": jnp.asarray(1.0), "label": "a"},
            label="r",
        )
        assert isinstance(r.spec, RecordSpec)
        assert r.spec is r.event_template

    def test_explicit_event_template_and_spec_are_the_same_object(self):
        tpl = RecordSpec(x=())
        r = Record(
            {"x": jnp.asarray(1.0)},
            event_template=tpl,
            label="r",
        )
        assert r.event_template is r.spec

    def test_a_schema_is_stored_directly(self):
        tpl = RecordSpec(x=())
        r = Record(
            {"x": jnp.asarray(1.0)},
            event_template=tpl,
            label="r",
        )
        assert r.spec is tpl
        assert r.event_template is tpl

    def test_a_concrete_spec_is_stored_verbatim(self):
        spec = RecordSpec(x=())
        r = Record(
            {"x": jnp.asarray(1.0)},
            event_template=spec,
            label="r",
        )
        assert r.spec is spec

    def test_the_two_declaration_forms_agree(self):
        tpl = RecordSpec(x=(2,), label=OpaqueSpec())
        fields = {"x": jnp.zeros(2), "label": "a"}
        assert Record(
            dict(fields),
            event_template=tpl,
            label="r",
        ) == Record(
            dict(fields),
            event_template=RecordSpec(tpl),
            label="r",
        )

    def test_spec_admits_the_record_it_types(self):
        r = Record(
            {"x": jnp.asarray(1.0), "label": "a"},
            label="r",
        )
        assert r.spec.is_valid(r)

    def test_a_spec_declaration_still_drives_numeric_promotion(self):
        from probpipe import NumericRecord

        numeric = RecordSpec(x=(2,))
        assert isinstance(
            Record(
                {"x": jnp.zeros(2)},
                event_template=numeric,
                label="r",
            ),
            NumericRecord,
        )
        # A non-numeric leaf in the declaration vetoes promotion.
        mixed = RecordSpec(x=(2,), label=OpaqueSpec())
        r = Record(
            {"x": jnp.zeros(2), "label": "a"},
            event_template=mixed,
            label="r",
        )
        assert not isinstance(r, NumericRecord)

    def test_a_polymorphic_declaration_is_stored_bound(self):
        spec = RecordSpec(x=("obs", 2), y=("obs",))
        r = Record(
            {"x": jnp.zeros((5, 2)), "y": jnp.zeros(5)},
            event_template=spec,
            label="r",
        )
        assert r.spec.is_concrete
        assert r.spec != spec  # the stored declaration is the bound one

    def test_a_nested_record_carries_its_own_spec(self):
        r = Record(
            {
                "theta": Record(
                    {"loc": jnp.zeros(2)},
                    label="theta",
                ),
                "obs": jnp.zeros(5),
            },
            label="r",
        )
        child = r.at_path("theta")
        assert isinstance(child.spec, RecordSpec)
        assert child.spec == r.event_template.at_path("theta")

    def test_the_spec_rides_in_the_pytree_aux(self):
        r = Record(
            {"x": jnp.asarray(1.0), "label": "a"},
            label="r",
        )
        leaves, treedef = jax.tree_util.tree_flatten(r)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        # ``is``, not ``==``: the aux spec is reused without a wrapper, so
        # crossing a JAX transform boundary allocates no declaration. An equality
        # check would still pass if that reuse were lost.
        assert rebuilt.spec is r.spec

    def test_records_sharing_a_spec_share_a_treedef(self):
        tpl = RecordSpec(x=(2,))
        first = Record(
            {"x": jnp.zeros(2)},
            event_template=tpl,
            label="a",
        )
        second = Record(
            {"x": jnp.ones(2)},
            event_template=tpl,
            label="a",
        )
        assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(second)

    def test_the_declaration_form_does_not_reach_the_treedef(self):
        """Equal declarations give one treedef however they were written.

        The aux is the spec, so a treedef compares by declaration. Writing that
        declaration as a schema or an equal schema copy must not
        split it in two — a treedef is a jit cache key, so a split there would
        silently retrace.
        """
        tpl = RecordSpec(x=(2,))
        forms = (
            tpl,
            RecordSpec(tpl),
            RecordSpec(x=(2,)),  # equal but distinct
            RecordSpec(x=(2,)),
        )
        treedefs = {
            jax.tree_util.tree_structure(
                Record(
                    {"x": jnp.zeros(2)},
                    event_template=form,
                    label="a",
                )
            )
            for form in forms
        }
        assert len(treedefs) == 1
        # Hashing agrees with equality, which is what a cache lookup relies on.
        assert len({hash(treedef) for treedef in treedefs}) == 1

    def test_a_different_declaration_gives_a_different_treedef(self):
        two = Record(
            {"x": jnp.zeros(2)},
            event_template=RecordSpec(x=(2,)),
            label="a",
        )
        three = Record(
            {"x": jnp.zeros(3)},
            event_template=RecordSpec(x=(3,)),
            label="a",
        )
        assert jax.tree_util.tree_structure(two) != jax.tree_util.tree_structure(three)

    def test_jit_traces_once_across_the_declaration_forms(self):
        """The end the treedef test is a proxy for: one trace, not four."""
        traces = []

        @jax.jit
        def total(record):
            traces.append(1)
            return record["x"].sum()

        tpl = RecordSpec(x=(2,))
        for form in (
            tpl,
            RecordSpec(tpl),
            RecordSpec(x=(2,)),
            RecordSpec(x=(2,)),
        ):
            total(
                Record(
                    {"x": jnp.zeros(2)},
                    event_template=form,
                    label="a",
                )
            )
        assert len(traces) == 1

    def test_the_spec_survives_a_pickle_roundtrip(self):
        import pickle

        from probpipe.core._specs import NumericArraySpec

        # An explicit dtype is not recoverable by inference, so the spec must
        # be the thing that was serialized.
        spec = RecordSpec(x=NumericArraySpec(shape=(), dtype=jnp.float32))
        r = Record(
            {"x": jnp.asarray(1.0, dtype=jnp.float32)},
            event_template=spec,
            label="r",
        )
        rebuilt = pickle.loads(pickle.dumps(r))
        assert rebuilt.spec == spec

    def test_numeric_record_accepts_a_spec_declaration(self):
        from probpipe import NumericRecord

        spec = RecordSpec(x=(2,), y=())
        nr = NumericRecord(
            {"x": jnp.zeros(2), "y": jnp.asarray(1.0)},
            event_template=spec,
            label="nr",
        )
        assert nr.spec is spec
        assert nr.spec is nr.event_template


# ---------------------------------------------------------------------------
# Authoritative RecordSpec storage
# ---------------------------------------------------------------------------


class TestRecordSpecStorage:
    def test_inferred_when_not_supplied(self):
        r = Record(
            {"x": jnp.asarray(1.0), "label": "a"},
            label="r",
        )
        from probpipe.core._specs import RecordSpec

        assert isinstance(r.event_template, RecordSpec)
        assert r.event_template.fields == ("x", "label")

    def test_inferred_template_is_cached(self):
        r = Record(
            {"x": jnp.asarray(1.0)},
            label="r",
        )
        # Same object on repeated access — inferred once, never recomputed.
        assert r.event_template is r.event_template

    def test_explicit_template_returned_verbatim(self):
        from probpipe.core._specs import NumericArraySpec, RecordSpec

        tpl = RecordSpec(x=NumericArraySpec(shape=(), dtype=jnp.float32))
        r = Record(
            {"x": jnp.asarray(1.0)},
            event_template=tpl,
            label="r",
        )
        assert r.event_template is tpl

    def test_explicit_template_field_mismatch_raises(self):
        from probpipe.core._specs import RecordSpec

        with pytest.raises(ValueError, match="do not"):
            Record(
                {"x": jnp.asarray(1.0)},
                event_template=RecordSpec(y=()),
                label="r",
            )

    def test_numeric_record_carries_numeric_template(self):
        from probpipe.core._specs import NumericRecordSpec

        nr = Record(
            {"a": 1.0, "b": jnp.zeros(3)},
            label="r",
        ).to_numeric()
        assert isinstance(nr.event_template, NumericRecordSpec)
        # to_vector ravels and concatenates the leaves in canonical order:
        # a (scalar) then b (length 3), so the flat vector is [a, b0, b1, b2].
        vec = nr.to_vector()
        np.testing.assert_array_equal(np.asarray(vec), np.asarray([1.0, 0.0, 0.0, 0.0]))
        assert vec.shape[0] == nr.event_template.vector_size

    def test_equality_distinguishes_structurally_different_templates(self):
        from probpipe.core._specs import NumericArraySpec, RecordSpec

        # float32 data is same-kind valid against both a float32 spec (exact)
        # and a float64 spec (a widening), so both records construct; their
        # templates differ, so the records are unequal despite identical data.
        data = {"x": jnp.asarray(1.0, dtype=jnp.float32)}
        r_f32 = Record(
            dict(data),
            event_template=RecordSpec(x=NumericArraySpec(shape=(), dtype=jnp.float32)),
            label="r",
        )
        r_f64 = Record(
            dict(data),
            event_template=RecordSpec(x=NumericArraySpec(shape=(), dtype=jnp.float64)),
            label="r",
        )
        assert r_f32 != r_f64
        # Same data, both inferred -> equal templates -> equal records.
        assert Record(
            {"x": jnp.asarray(1.0)},
            label="r",
        ) == Record(
            {"x": jnp.asarray(1.0)},
            label="r",
        )

    def test_construction_enforces_leaf_shape_and_dtype(self):
        from probpipe.core._specs import NumericArraySpec, RecordSpec

        # A cross-kind dtype (a float value against an int-dtype spec) fails the
        # spec's is_valid -> construction raises.
        with pytest.raises(ValueError, match="does not match event_template"):
            Record(
                {"x": jnp.asarray(1.0, dtype=jnp.float32)},
                event_template=RecordSpec(x=NumericArraySpec(shape=(), dtype=jnp.int32)),
                label="r",
            )
        # A shape mismatch also raises.
        with pytest.raises(ValueError, match="does not match event_template"):
            Record(
                {"x": jnp.zeros(3)},
                event_template=RecordSpec(x=NumericArraySpec(shape=(2,))),
                label="r",
            )
        # A same-kind cast (int value against a float spec) satisfies is_valid,
        # so construction succeeds.
        Record(
            {"x": jnp.ones(2, dtype=jnp.int32)},
            event_template=RecordSpec(x=NumericArraySpec(shape=(2,), dtype=jnp.float32)),
            label="r",
        )

    def test_pytree_roundtrip_threads_the_declaration(self):
        # The declaration rides in the aux and comes back with the value; it is
        # not re-derived from the rebuilt leaves.
        r = Record(
            {"x": jnp.asarray(1.0), "y": jnp.zeros(2)},
            label="r",
        )
        leaves, treedef = jax.tree_util.tree_flatten(r)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt == r
        assert rebuilt.event_template == r.event_template

    def test_record_batch_event_template_is_template(self):
        from probpipe import NumericRecord, RecordBatch

        ra = RecordBatch.stack(
            [
                NumericRecord(
                    {"x": 1.0},
                    label="nr",
                ),
                NumericRecord(
                    {"x": 2.0},
                    label="nr",
                ),
            ],
            level_name="draw",
        )
        assert ra.event_template is ra.element_spec

    def test_explicit_nested_template_validates_recursively(self):
        from probpipe.core._specs import RecordSpec

        tpl = RecordSpec(physics=RecordSpec(force=(), mass=()), obs=(5,))
        r = Record(
            {
                "physics": Record(
                    {"force": 1.0, "mass": 2.0},
                    label="physics",
                ),
                "obs": jnp.zeros(5),
            },
            event_template=tpl,
            label="r",
        )
        assert r.event_template is tpl

    def test_nested_field_name_mismatch_raises_with_path(self):
        from probpipe.core._specs import RecordSpec

        tpl = RecordSpec(physics=RecordSpec(force=(), mass=()), obs=(5,))
        with pytest.raises(ValueError, match="physics"):
            Record(
                {
                    "physics": Record(
                        {"force": 1.0, "momentum": 2.0},
                        label="physics",
                    ),
                    "obs": jnp.zeros(5),
                },
                event_template=tpl,
                label="r",
            )

    def test_structure_vs_leaf_mismatch_raises(self):
        from probpipe.core._specs import RecordSpec

        # Template says ``physics`` is a leaf; record has a nested Record there.
        with pytest.raises(ValueError, match="structure mismatch"):
            Record(
                {
                    "physics": Record(
                        {"a": 1.0},
                        label="physics",
                    ),
                    "x": 2.0,
                },
                event_template=RecordSpec(physics=(), x=()),
                label="r",
            )


class TestConstructionMessages:
    """Each refusal names the constructor, what it got, and the fix."""

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda: Record("r", {"a": 1.0}), id="label-and-fields"),
            pytest.param(lambda: Record("r"), id="label-alone"),
        ],
    )
    def test_the_label_first_form_names_the_new_form(self, build):
        with pytest.raises(
            TypeError,
            match=(
                r"^Record takes the fields first and the label as the keyword label, but got "
                r"the string 'r' as the fields; write Record\(fields, label='r'\)$"
            ),
        ):
            build()

    def test_fields_that_are_not_a_mapping_are_refused_with_their_type(self):
        with pytest.raises(
            TypeError,
            match=r"^Record takes a mapping of field names to values, got list; pass a dict",
        ):
            Record([1.0])

    def test_an_empty_record_without_a_label_is_refused(self):
        with pytest.raises(
            TypeError,
            match=r"^cannot derive a default label for a Record with no fields; pass label=\.\.\.$",
        ):
            Record({})
        assert Record({}, label="empty").label == "empty"

    def test_from_fields_with_no_fields_points_to_the_constructor(self):
        with pytest.raises(
            TypeError,
            match=r"^Record\.from_fields\(\) requires at least one field; .*Record\(\{\}, label=",
        ):
            Record.from_fields()

    def test_from_fields_refuses_a_path_separator_in_a_name(self):
        with pytest.raises(ValueError, match="/"):
            Record.from_fields(**{"a/b": 1.0})
