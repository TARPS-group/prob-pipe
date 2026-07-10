"""Tests for probpipe.core.record.Record."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import EventTemplate, Normal, Provenance, Record, provenance_ancestors

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_kwargs(self):
        v = Record(r=1.8, K=70.0, phi=10.0)
        assert v.fields == ("r", "K", "phi")  # insertion order

    def test_dict_positional(self):
        v = Record({"a": 1.0, "b": 2.0})
        assert v.fields == ("a", "b")

    def test_positional_accepts_any_mapping(self):
        # The positional arg accepts any collections.abc.Mapping, not just dict.
        from collections import OrderedDict
        from types import MappingProxyType

        from probpipe import NumericRecord

        v = Record(MappingProxyType({"a": 1.0, "b": 2.0}))
        assert v.fields == ("a", "b")
        nr = NumericRecord(OrderedDict([("z", jnp.zeros(2)), ("a", 1.0)]))
        assert nr.fields == ("z", "a")  # mapping iteration order preserved

    def test_insertion_order_preserved(self):
        v = Record(z=1.0, a=2.0, m=3.0)
        # Insertion order, NOT alphabetical.
        assert v.fields == ("z", "a", "m")

    def test_slash_in_field_name_rejected(self):
        with pytest.raises(ValueError, match="must not contain '/'"):
            Record(**{"a/b": 1.0})

    def test_dict_and_kwargs_raises(self):
        with pytest.raises(ValueError, match="Cannot pass both"):
            Record({"a": 1.0}, b=2.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Record()

    def test_stores_values_verbatim(self):
        arr = np.array([1.0, 2.0, 3.0])
        v = Record(x=arr)
        # Record no longer performs any conversion — stored as-is.
        assert v["x"] is arr

    def test_accepts_opaque_leaves(self):
        v = Record(label="horseshoe", x=1.0)
        assert v["label"] == "horseshoe"
        assert v["x"] == 1.0

    def test_jax_arrays(self):
        arr = jnp.array([1.0, 2.0])
        v = Record(x=arr)
        assert v["x"] is arr

    def test_scalars(self):
        v = Record(a=1, b=2.5, c=True)
        assert v["a"] == 1
        assert v["b"] == 2.5
        assert v["c"] is True

    def test_nested(self):
        inner = Record(x=1.0, y=2.0)
        outer = Record(params=inner, z=3.0)
        assert isinstance(outer.at_path("params"), Record)
        assert outer["params/x"] == 1.0

    def test_from_dict(self):
        v = Record.from_dict({"a": 1.0, "b": 2.0})
        assert v.fields == ("a", "b")

    def test_list_input(self):
        v = Record(x=[1.0, 2.0, 3.0])
        # Stored as-is — caller decides conversion.
        assert v["x"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Field access
# ---------------------------------------------------------------------------


class TestFieldAccess:
    @pytest.fixture
    def v(self):
        return Record(r=1.8, K=70.0, phi=10.0)

    def test_item(self, v):
        np.testing.assert_allclose(float(v["K"]), 70.0, rtol=1e-5)

    def test_key_path_tuple(self):
        v = Record(params=Record(r=1.8, K=70.0), obs=Record(y=np.zeros(5)))
        np.testing.assert_allclose(float(v["params", "r"]), 1.8, rtol=1e-5)

    def test_key_path_string(self):
        v = Record(params=Record(r=1.8, K=70.0), obs=Record(y=np.zeros(5)))
        np.testing.assert_allclose(float(v["params/r"]), 1.8, rtol=1e-5)

    def test_key_path_string_three_levels(self):
        v = Record(a=Record(b=Record(c=42.0)))
        assert v["a/b/c"] == 42.0

    def test_path_in_membership(self):
        v = Record(params=Record(r=1.8, K=70.0))
        assert "params/r" in v
        assert "params/missing" not in v
        assert "missing/r" not in v

    def test_path_through_non_record_raises_clear_keyerror(self):
        """Descending past a leaf via path syntax must raise ``KeyError`` with
        a path-aware message — not a numpy ``IndexError`` (regression for PR-A
        review finding #5)."""
        v = Record(a=np.array([1.0, 2.0]))
        with pytest.raises(KeyError, match="non-tree leaf"):
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
        v = Record(x=1.0)
        with pytest.raises(AttributeError, match="immutable"):
            v.x = 2.0

    def test_delattr_raises(self):
        v = Record(x=1.0)
        with pytest.raises(AttributeError, match="immutable"):
            del v.x

    def test_replace(self):
        v = Record(a=1.0, b=2.0)
        v2 = v.replace(b=3.0)
        assert v["b"] == 2.0  # original unchanged
        assert v2["b"] == 3.0

    def test_replace_nonexistent_raises(self):
        v = Record(a=1.0)
        with pytest.raises(KeyError, match="z"):
            v.replace(z=5.0)

    def test_replace_nested_path(self):
        v = Record(physics=Record(force=1.0, mass=2.0), obs=3.0)
        v2 = v.replace({"physics/mass": 9.0})
        assert v2["physics/mass"] == 9.0
        assert v2["physics/force"] == 1.0  # untouched

    def test_merge(self):
        v1 = Record(a=1.0)
        v2 = Record(b=2.0)
        merged = v1.merge(v2)
        assert merged.fields == ("a", "b")

    def test_merge_overlap_raises(self):
        v1 = Record(a=1.0)
        v2 = Record(a=2.0)
        with pytest.raises(ValueError, match="Overlapping"):
            v1.merge(v2)

    def test_without(self):
        v = Record(a=1.0, b=2.0, c=3.0)
        v2 = v.without("b")
        assert v2.fields == ("a", "c")

    def test_without_nonexistent_key_raises(self):
        """Removing a key that doesn't exist raises KeyError (leaf-keyed contract)."""
        v = Record(a=1.0, b=2.0)
        with pytest.raises(KeyError):
            v.without("z")

    def test_without_nested_path(self):
        v = Record(physics=Record(force=1.0, mass=2.0), obs=3.0)
        assert tuple(v.without("physics/mass").keys()) == ("physics/force", "obs")
        assert tuple(v.without("physics").keys()) == ("obs",)

    def test_without_all_raises(self):
        v = Record(a=1.0)
        with pytest.raises(ValueError, match="Cannot remove all"):
            v.without("a")

    # replace / merge / without must preserve the subclass (regression
    # for review comment (b) on a8be0b3: NumericRecord was silently
    # downgraded to Record by these methods).

    def test_replace_preserves_numeric_record(self):
        from probpipe import NumericRecord

        nr = NumericRecord(a=1.0, b=2.0)
        assert type(nr.replace(a=3.0)) is NumericRecord

    def test_merge_preserves_numeric_record(self):
        from probpipe import NumericRecord

        nr = NumericRecord(a=1.0)
        assert type(nr.merge(NumericRecord(b=2.0))) is NumericRecord

    def test_without_preserves_numeric_record(self):
        from probpipe import NumericRecord

        nr = NumericRecord(a=1.0, b=2.0)
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
        v = Record(x=arr)
        assert v["x"] is arr
        assert isinstance(v["x"], np.ndarray)

    def test_scalar_stored_verbatim(self):
        v = Record(x=42.0)
        assert v["x"] == 42.0
        assert isinstance(v["x"], float)

    def test_jax_stored_verbatim(self):
        arr = jnp.array([1.0, 2.0])
        v = Record(x=arr)
        assert v["x"] is arr

    def test_string_stored_verbatim(self):
        v = Record(x="hello", y=1.0)
        assert v["x"] == "hello"

    def test_heterogeneous_leaves(self):
        """Strings, numbers, and arrays co-exist in a plain Record."""
        v = Record(label="x", count=1.0, array=jnp.zeros(3))
        assert v["label"] == "x"
        assert v["count"] == 1.0
        assert v["array"].shape == (3,)

    def test_xarray_stored_verbatim(self):
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["time"],
            coords={"time": [10, 20, 30]},
        )
        v = Record(y=da)
        # DataArray is preserved, coords and all.
        assert v["y"] is da
        assert v["y"].dims == ("time",)
        np.testing.assert_array_equal(v["y"].coords["time"].values, [10, 20, 30])


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

    def test_numeric_vector_ops_absent_from_record(self):
        v = Record(a=1.0)
        for attr in ("to_vector", "vector_size", "zip"):
            assert not hasattr(v, attr), f"Record should not expose {attr!r}"
        assert not hasattr(Record, "zip")

    def test_general_decomposition_present_flatten_absent(self):
        # values() / from_field_values is the general (any-leaf) ProbPipe leaf
        # traversal on Record; the JAX-pytree flatten/unflatten are NOT Record
        # methods.
        v = Record(a=1.0, label="x")
        assert v.event_template.from_field_values(list(v.values())) == v
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
        v = Record(x=jnp.array([1.0, 2.0]), label="horseshoe")
        leaves = list(v.values())
        assert leaves[0].shape == (2,)  # kept whole, not raveled
        assert leaves[1] == "horseshoe"  # opaque leaf preserved as-is
        assert list(v.keys()) == ["x", "label"]  # canonical order

    def test_container_leaf_is_one_whole_leaf(self):
        # A tuple field is ONE opaque leaf at template granularity; JAX's
        # pytree view descends into it (documented divergence).
        v = Record(x=jnp.zeros(2), pair=(jnp.array(1.0), jnp.array(2.0)))
        leaves = list(v.values())
        assert len(leaves) == 2  # x, pair (whole)
        assert isinstance(leaves[1], tuple)
        assert len(jax.tree_util.tree_leaves(v)) == 3  # JAX descends the tuple

    def test_roundtrip_with_opaque_leaf(self):
        # Opaque (non-numeric) leaves round-trip — unlike to_vector.
        v = Record(x=jnp.array([1.0, 2.0]), label="horseshoe", count=3)
        assert v.event_template.from_field_values(list(v.values())) == v

    def test_numeric_record_roundtrip(self):
        from probpipe import NumericRecord

        v = NumericRecord(a=jnp.array([1.0, 2.0, 3.0]), b=NumericRecord(c=jnp.array(5.0)))
        rebuilt = v.event_template.from_field_values(list(v.values()))
        assert rebuilt == v
        assert isinstance(rebuilt, NumericRecord)
        assert isinstance(rebuilt.at_path("b"), NumericRecord)

    def test_mixed_nested_record_roundtrip(self):
        # A nested mixed record with both numeric and opaque leaves.
        v = Record(
            theta=Record(loc=jnp.array([0.0, 1.0]), label="prior"),
            tag="run-7",
        )
        assert v.event_template.from_field_values(list(v.values())) == v

    def test_wrong_leaf_count_raises(self):
        v = Record(a=1.0, b=2.0)
        with pytest.raises(ValueError, match="expected 2"):
            v.event_template.from_field_values([1.0])

    def test_jax_pytree_roundtrip_still_works(self):
        # Record stays a registered pytree; the JAX path round-trips via
        # jax.tree_util (the documented finer-granularity escape hatch).
        v = Record(x=jnp.array([1.0, 2.0]), label="horseshoe")
        leaves, treedef = jax.tree_util.tree_flatten(v)
        assert jax.tree_util.tree_unflatten(treedef, leaves) == v


class TestKeysAgreement:
    """A Record's own ``keys()`` must always equal its ``event_template``'s
    ``keys()`` — both define "what is a leaf" and the ``event_template`` is the
    source of truth. In particular, a cross-type nested value (an
    ``EventTemplate`` stored as a ``Record`` field value) is one opaque leaf,
    not an internal node to descend into.
    """

    def test_flat(self):
        v = Record(a=1.0, b=jnp.zeros(3), c="x")
        assert list(v.keys()) == list(v.event_template.keys())

    def test_nested_in_family(self):
        v = Record(theta=Record(loc=jnp.zeros(2), s=jnp.ones(3)), top=jnp.array(1.0))
        assert list(v.keys()) == list(v.event_template.keys())
        assert list(v.keys()) == ["theta/loc", "theta/s", "top"]

    def test_cross_type_value_is_one_opaque_leaf(self):
        # An EventTemplate stored as a Record field value is an opaque leaf,
        # NOT an internal node: keys() must not descend into it.
        v = Record(weird=EventTemplate(a=(2,)), x=jnp.array([1.0, 2.0]))
        assert list(v.keys()) == ["weird", "x"]
        assert list(v.keys()) == list(v.event_template.keys())


# ---------------------------------------------------------------------------
# JAX PyTree
# ---------------------------------------------------------------------------


class TestPyTree:
    def test_tree_map(self):
        v = Record(a=1.0, b=2.0)
        v2 = jax.tree.map(lambda x: x * 2, v)
        assert isinstance(v2, Record)
        assert v2["a"] == 2.0
        assert v2["b"] == 4.0

    def test_tree_leaves(self):
        v = Record(a=jnp.array(1.0), b=jnp.array(2.0))
        leaves = jax.tree.leaves(v)
        assert len(leaves) == 2

    def test_tree_structure_roundtrip(self):
        v = Record(x=jnp.array([1.0, 2.0]), y=jnp.array(3.0))
        leaves, treedef = jax.tree.flatten(v)
        v2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(v2, Record)
        assert v2.fields == v.fields

    def test_nested_tree_map(self):
        v = Record(params=Record(r=1.0, K=2.0), z=3.0)
        v2 = jax.tree.map(lambda x: x + 10, v)
        assert isinstance(v2, Record)
        assert isinstance(v2.at_path("params"), Record)
        assert v2["params/r"] == 11.0
        assert v2["z"] == 13.0

    def test_jit(self):
        v = Record(a=1.0, b=2.0)

        @jax.jit
        def f(vals):
            return vals["a"] + vals["b"]

        result = f(v)
        np.testing.assert_allclose(float(result), 3.0)

    def test_jit_returns_values(self):
        v = Record(a=1.0, b=2.0)

        @jax.jit
        def f(vals):
            return jax.tree.map(lambda x: x * 2, vals)

        result = f(v)
        assert isinstance(result, Record)
        np.testing.assert_allclose(float(result["a"]), 2.0)

    def test_vmap(self):
        batch = Record(x=jnp.array([1.0, 2.0, 3.0]))

        @jax.vmap
        def f(vals):
            return vals["x"] ** 2

        result = f(batch)
        np.testing.assert_allclose(result, [1.0, 4.0, 9.0])

    def test_grad(self):
        v = Record(x=1.0)

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
        v = Record(a=1.0, b=arr)
        d = v.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) == {"a", "b"}
        # No coercion — values returned as stored.
        assert d["a"] == 1.0
        assert d["b"] is arr

    def test_to_numpy(self):
        v = Record(a=jnp.array(1.0), b=jnp.array([2.0]))
        d = v.to_numpy()
        assert isinstance(d["a"], np.ndarray)
        assert isinstance(d["b"], np.ndarray)

    def test_to_numpy_preserves_opaque(self):
        v = Record(label="x", y=np.array([1.0, 2.0]))
        d = v.to_numpy()
        assert d["label"] == "x"
        assert isinstance(d["y"], np.ndarray)

    def test_to_dict_nested(self):
        v = Record(inner=Record(x=1.0), y=2.0)
        d = v.to_dict()
        assert isinstance(d["inner"], dict)
        assert d["inner"]["x"] == 1.0

    def test_to_numeric_returns_numeric_record(self):
        from probpipe import NumericRecord

        v = Record(a=1.0, b=jnp.array([2.0, 3.0]))
        nr = v.to_numeric()
        assert isinstance(nr, NumericRecord)
        assert nr.fields == ("a", "b")
        np.testing.assert_array_equal(np.asarray(nr["b"]), [2.0, 3.0])

    def test_to_numeric_to_native_round_trip_xarray(self):
        """xarray dims / coords / attrs survive the ``Record`` round-trip
        via the aux registry (the migration replacement for the old
        ``to_datatree`` / ``from_datatree`` pair).
        """
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["time"],
            coords={"time": [10, 20, 30]},
            attrs={"units": "m"},
        )
        back = Record(y=da).to_numeric().to_native()
        assert back["y"].dims == ("time",)
        np.testing.assert_array_equal(back["y"].coords["time"].values, [10, 20, 30])
        assert back["y"].attrs == {"units": "m"}

    def test_to_numeric_recurses_into_nested_records(self):
        """``to_numeric()`` recurses into nested non-NumericRecord children
        (regression for the divergence flagged in PR-A review)."""
        from probpipe import NumericRecord

        outer = Record(inner=Record(a=1.0), z=2.0)
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
        v = Record(x=1.0)
        assert Record.ensure(v) is v

    def test_dict_coercion(self):
        v = Record.ensure({"a": 1.0, "b": 2.0})
        assert isinstance(v, Record)
        assert v.fields == ("a", "b")

    def test_array_coercion(self):
        v = Record.ensure(jnp.array([1.0, 2.0]))
        assert isinstance(v, Record)
        assert "data" in v
        np.testing.assert_allclose(np.asarray(v["data"]), [1.0, 2.0])

    def test_numpy_coercion(self):
        v = Record.ensure(np.array([1.0]))
        assert isinstance(v, Record)
        assert "data" in v


# ---------------------------------------------------------------------------
# Leaf-wise operations
# ---------------------------------------------------------------------------


class TestLeafOps:
    def test_map(self):
        v = Record(a=2.0, b=3.0)
        v2 = v.map(lambda x: x**2)
        assert v2["a"] == 4.0
        assert v2["b"] == 9.0

    def test_map_returns_same_class(self):
        """``map`` on a plain Record returns a Record."""
        v = Record(a=2.0, b=3.0)
        assert type(v.map(lambda x: x + 1)) is Record

    def test_map_preserves_numeric_record(self):
        """``map`` on a NumericRecord returns a NumericRecord as long as
        the mapped values remain numeric (validated by ``__init__``)."""
        from probpipe import NumericRecord

        nr = NumericRecord(a=1.0, b=2.0)
        out = nr.map(lambda x: x * 2)
        assert type(out) is NumericRecord

    def test_map_on_numeric_record_rejects_non_numeric_output(self):
        """A map that returns a string violates the NumericRecord
        invariant and must fail loudly at reconstruction."""
        from probpipe import NumericRecord

        nr = NumericRecord(a=1.0, b=2.0)
        with pytest.raises(TypeError, match="numeric"):
            nr.map(lambda x: "not numeric")

    def test_map_nested(self):
        v = Record(inner=Record(x=2.0), y=3.0)
        v2 = v.map(lambda x: x + 1)
        assert v2["inner/x"] == 3.0
        assert v2["y"] == 4.0

    def test_map_with_keys(self):
        v = Record(a=1.0, b=2.0)
        keys_seen = []
        v.map_with_keys(lambda k, x: keys_seen.append(k) or x)
        assert keys_seen == ["a", "b"]

    def test_map_with_keys_passes_full_path(self):
        v = Record(inner=Record(x=2.0), y=3.0)
        keys_seen = []
        v.map_with_keys(lambda k, x: keys_seen.append(k) or x)
        assert keys_seen == ["inner/x", "y"]

    def test_map_forwards_args(self):
        v = Record(a=1.0, b=2.0)
        v2 = v.map(lambda x, bump: x + bump, bump=10.0)
        assert v2["a"] == 11.0 and v2["b"] == 12.0

    def test_map_rejects_node_return(self):
        v = Record(a=1.0)
        with pytest.raises(ValueError, match="introduce nesting"):
            v.map(lambda x: Record(z=x))


# ---------------------------------------------------------------------------
# Repr and equality
# ---------------------------------------------------------------------------


class TestReprAndEquality:
    def test_repr_scalars(self):
        v = Record(a=1.0, b=2.0)
        r = repr(v)
        assert "Record(" in r
        assert "a=" in r
        assert "b=" in r

    def test_repr_arrays(self):
        v = Record(x=jnp.zeros((3, 4)))
        r = repr(v)
        assert "shape=(3, 4)" in r

    def test_repr_nested(self):
        v = Record(inner=Record(x=1.0))
        r = repr(v)
        assert "inner=Record(" in r

    def test_equality(self):
        v1 = Record(a=1.0, b=2.0)
        v2 = Record(a=1.0, b=2.0)
        assert v1 == v2

    def test_inequality_values(self):
        v1 = Record(a=1.0)
        v2 = Record(a=2.0)
        assert v1 != v2

    def test_inequality_fields(self):
        v1 = Record(a=1.0)
        v2 = Record(b=1.0)
        assert v1 != v2

    def test_hash_includes_shape(self):
        """Records with the same field names but different shapes should
        hash differently (follow-up to review comment #10)."""
        v1 = Record(a=jnp.zeros(3))
        v2 = Record(a=jnp.zeros(5))
        assert hash(v1) != hash(v2)

    def test_hash_excludes_value(self):
        """Records with the same shape+dtype but different values hash
        the same (structural hash)."""
        v1 = Record(a=jnp.zeros(3))
        v2 = Record(a=jnp.ones(3))
        assert hash(v1) == hash(v2)

    def test_hash_distinguishes_dtype(self):
        """Records with the same shape but different dtype hash differently."""
        v1 = Record(a=jnp.zeros(3, dtype=jnp.float32))
        v2 = Record(a=jnp.zeros(3, dtype=jnp.int32))
        assert hash(v1) != hash(v2)

    def test_eq_type_strict(self):
        """Record == NumericRecord is False even with identical fields."""
        from probpipe import NumericRecord

        r = Record(a=1.0, b=2.0)
        nr = NumericRecord(a=1.0, b=2.0)
        assert r != nr

    # Hash / eq contract: ``a == b`` must imply ``hash(a) == hash(b)``.
    # Regression for review comment (a) on a8be0b3 — ``__hash__`` used
    # to read raw ``.shape`` / ``.dtype`` while ``__eq__`` coerced via
    # ``jnp.asarray``, so Record(a=1.0) == Record(a=jnp.asarray(1.0))
    # but the hashes differed.

    def test_hash_eq_contract_scalar_vs_zero_d_array(self):
        r1 = Record(a=1.0)
        r2 = Record(a=jnp.asarray(1.0))
        assert r1 == r2
        assert hash(r1) == hash(r2)

    def test_hash_eq_contract_python_int_vs_numpy(self):
        r1 = Record(a=1)
        r2 = Record(a=np.asarray(1))
        assert r1 == r2
        assert hash(r1) == hash(r2)

    def test_hash_eq_contract_opaque_leaf(self):
        """Two Records with the same string leaves must hash the same."""
        r1 = Record(label="x", count=1.0)
        r2 = Record(label="x", count=1.0)
        assert r1 == r2
        assert hash(r1) == hash(r2)

    # Regression for (f-ii): self-equality must hold even when leaves
    # contain NaN. ``jnp.array_equal`` treats NaN != NaN, so we need an
    # identity fast-path.

    def test_self_equality_with_nan(self):
        r = Record(x=jnp.array([jnp.nan, 1.0, jnp.nan]))
        assert r == r

    def test_self_equality_with_nan_nested(self):
        r = Record(inner=Record(x=jnp.array([jnp.nan])), y=jnp.nan)
        assert r == r


# ---------------------------------------------------------------------------
# Provenance (issue #130)
# ---------------------------------------------------------------------------


class TestProvenance:
    """Record carries the same ``.provenance`` / ``.with_provenance`` slot as
    Distribution, so workflow outputs can attach a Provenance node
    regardless of which of the three output types (Record, RecordArray,
    Distribution) the broadcasting layer produced.
    """

    def test_initial_provenance_is_none(self):
        r = Record(x=1.0, y=2.0)
        assert r.provenance is None

    def test_with_provenance_sets_and_returns_self(self):
        r = Record(x=1.0)
        out = r.with_provenance(Provenance("op", parents=()))
        assert out is r
        assert r.provenance.operation == "op"

    def test_with_provenance_is_write_once(self):
        r = Record(x=1.0)
        r.with_provenance(Provenance("first", parents=()))
        with pytest.raises(RuntimeError, match="write-once"):
            r.with_provenance(Provenance("second", parents=()))

    # Semantic transformations reset the source — the new Record is a
    # different logical value even though the class preserves.

    def test_replace_resets_provenance(self):
        r = Record(x=1.0).with_provenance(Provenance("orig", parents=()))
        r2 = r.replace(x=2.0)
        assert r2.provenance is None
        assert r.provenance.operation == "orig"  # original unaffected

    def test_merge_resets_provenance(self):
        r = Record(x=1.0).with_provenance(Provenance("orig", parents=()))
        merged = r.merge(Record(y=2.0))
        assert merged.provenance is None

    def test_without_resets_provenance(self):
        r = Record(x=1.0, y=2.0).with_provenance(Provenance("orig", parents=()))
        r2 = r.without("y")
        assert r2.provenance is None

    def test_map_resets_provenance(self):
        r = Record(x=1.0).with_provenance(Provenance("orig", parents=()))
        r2 = r.map(lambda v: v + 1)
        assert r2.provenance is None

    # Structural equality / hashing ignore source — two Records with the
    # same fields but different provenance are still equal.

    def test_eq_ignores_provenance(self):
        r1 = Record(x=1.0).with_provenance(Provenance("a", parents=()))
        r2 = Record(x=1.0).with_provenance(Provenance("b", parents=()))
        assert r1 == r2

    def test_hash_ignores_provenance(self):
        r1 = Record(x=1.0).with_provenance(Provenance("a", parents=()))
        r2 = Record(x=1.0)
        assert hash(r1) == hash(r2)

    # Pytree roundtrip drops the source (runtime-only metadata — a
    # Provenance parent isn't hashable by structure, so pushing it into
    # the aux tuple would break jax.tree_util.tree_unflatten's equality
    # semantics). Document this caveat with a test.

    def test_pytree_roundtrip_drops_provenance(self):
        r = Record(x=1.0, y=jnp.array([2.0, 3.0]))
        r.with_provenance(Provenance("op", parents=()))
        leaves, treedef = jax.tree_util.tree_flatten(r)
        r2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert r2.provenance is None
        # But the Record is otherwise structurally identical.
        assert r2 == r

    # Integration: walk provenance from a Record through a Distribution
    # ancestor via provenance_ancestors.

    def test_provenance_ancestors_walks_through_distribution(self):
        prior = Normal(loc=0.0, scale=1.0, name="prior")
        r = Record(theta=1.0).with_provenance(Provenance("draw", parents=(prior,)))
        ancestors = provenance_ancestors(r)
        assert len(ancestors) == 1
        assert ancestors[0] is prior

    def test_provenance_ancestors_walks_nested_records(self):
        prior = Normal(loc=0.0, scale=1.0, name="prior")
        middle = Record(theta=1.0).with_provenance(Provenance("draw", parents=(prior,)))
        outer = Record(result=2.0).with_provenance(Provenance("transform", parents=(middle,)))
        ancestors = provenance_ancestors(outer)
        names = [getattr(a, "name", None) for a in ancestors]
        assert names == [middle.name, "prior"]


# ---------------------------------------------------------------------------
# Authoritative EventTemplate storage
# ---------------------------------------------------------------------------


class TestEventTemplateStorage:
    def test_inferred_when_not_supplied(self):
        r = Record(x=jnp.asarray(1.0), label="a")
        from probpipe.core.event_template import EventTemplate

        assert isinstance(r.event_template, EventTemplate)
        assert r.event_template.fields == ("x", "label")

    def test_inferred_template_is_cached(self):
        r = Record(x=jnp.asarray(1.0))
        # Same object on repeated access — inferred once, never recomputed.
        assert r.event_template is r.event_template

    def test_explicit_template_returned_verbatim(self):
        from probpipe.core.event_template import ArraySpec, EventTemplate

        tpl = EventTemplate(x=ArraySpec(shape=(), dtype=jnp.float32))
        r = Record({"x": jnp.asarray(1.0)}, event_template=tpl)
        assert r.event_template is tpl

    def test_explicit_template_field_mismatch_raises(self):
        from probpipe.core.event_template import EventTemplate

        with pytest.raises(ValueError, match="do not"):
            Record({"x": jnp.asarray(1.0)}, event_template=EventTemplate(y=()))

    def test_numeric_record_carries_numeric_template(self):
        from probpipe.core.event_template import NumericEventTemplate

        nr = Record(a=1.0, b=jnp.zeros(3)).to_numeric()
        assert isinstance(nr.event_template, NumericEventTemplate)
        # to_vector delegates to the stored template.
        np.testing.assert_array_equal(
            np.asarray(nr.to_vector()),
            np.asarray(nr.event_template.to_vector(nr)),
        )

    def test_equality_distinguishes_structurally_different_templates(self):
        from probpipe.core.event_template import ArraySpec, EventTemplate

        data = {"x": jnp.asarray(1.0)}
        r_f32 = Record(
            dict(data), event_template=EventTemplate(x=ArraySpec(shape=(), dtype=jnp.float32))
        )
        r_i32 = Record(
            dict(data), event_template=EventTemplate(x=ArraySpec(shape=(), dtype=jnp.int32))
        )
        # Identical bytes, structurally different schemas -> unequal.
        assert r_f32 != r_i32
        # Same data, both inferred -> equal templates -> equal records.
        assert Record(x=jnp.asarray(1.0)) == Record(x=jnp.asarray(1.0))

    def test_pytree_roundtrip_reinfers_template(self):
        r = Record(x=jnp.asarray(1.0), y=jnp.zeros(2))
        leaves, treedef = jax.tree_util.tree_flatten(r)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt == r
        assert rebuilt.event_template == r.event_template

    def test_record_array_event_template_is_template(self):
        from probpipe import NumericRecord, RecordArray

        ra = RecordArray.stack([NumericRecord(x=1.0), NumericRecord(x=2.0)])
        assert ra.event_template is ra.template

    def test_explicit_nested_template_validates_recursively(self):
        from probpipe.core.event_template import EventTemplate

        tpl = EventTemplate(physics=EventTemplate(force=(), mass=()), obs=(5,))
        r = Record(
            physics=Record(force=1.0, mass=2.0),
            obs=jnp.zeros(5),
            event_template=tpl,
        )
        assert r.event_template is tpl

    def test_nested_field_name_mismatch_raises_with_path(self):
        from probpipe.core.event_template import EventTemplate

        tpl = EventTemplate(physics=EventTemplate(force=(), mass=()), obs=(5,))
        with pytest.raises(ValueError, match="physics"):
            Record(
                physics=Record(force=1.0, momentum=2.0),  # wrong nested field name
                obs=jnp.zeros(5),
                event_template=tpl,
            )

    def test_structure_vs_leaf_mismatch_raises(self):
        from probpipe.core.event_template import EventTemplate

        # Template says ``physics`` is a leaf; record has a nested Record there.
        with pytest.raises(ValueError, match="structure mismatch"):
            Record(
                physics=Record(a=1.0),
                x=2.0,
                event_template=EventTemplate(physics=(), x=()),
            )
