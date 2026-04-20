"""Tests for probpipe.core.record.Record."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Normal, Provenance, Record, provenance_ancestors


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_kwargs(self):
        v = Record(r=1.8, K=70.0, phi=10.0)
        assert v.fields == ("K", "phi", "r")  # sorted

    def test_dict_positional(self):
        v = Record({"a": 1.0, "b": 2.0})
        assert v.fields == ("a", "b")

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
        assert isinstance(outer["params"], Record)
        assert outer["params"]["x"] == 1.0

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

    def test_key_path(self):
        v = Record(params=Record(r=1.8, K=70.0), obs=Record(y=np.zeros(5)))
        np.testing.assert_allclose(float(v["params", "r"]), 1.8, rtol=1e-5)

    def test_fields(self, v):
        assert v.fields == ("K", "phi", "r")

    def test_len(self, v):
        assert len(v) == 3

    def test_contains(self, v):
        assert "r" in v
        assert "missing" not in v

    def test_iter(self, v):
        assert list(v) == ["K", "phi", "r"]

    def test_items(self, v):
        items = list(v.items())
        assert len(items) == 3
        assert items[0][0] == "K"

    def test_keys(self, v):
        assert list(v.keys()) == ["K", "phi", "r"]

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
        with pytest.raises(KeyError, match="non-existent"):
            v.replace(z=5.0)

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

    def test_without_nonexistent_key(self):
        """Removing a key that doesn't exist silently keeps all fields."""
        v = Record(a=1.0, b=2.0)
        v2 = v.without("z")
        assert v2.fields == ("a", "b")

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
    """Numeric-only APIs live on NumericRecord, not on Record.

    A single focused test — if someone tries to re-add ``flatten`` or
    similar to ``Record`` (rather than ``NumericRecord``), this fails.
    Implementation-detail attributes like ``_resolved`` / ``_coords``
    are intentionally not checked here: they are private and covered
    indirectly by the storage-verbatim tests above.
    """

    def test_numeric_ops_absent_from_record(self):
        v = Record(a=1.0)
        for attr in ("flatten", "unflatten", "flat_size", "zip"):
            assert not hasattr(v, attr), f"Record should not expose {attr!r}"
        assert not hasattr(Record, "unflatten")
        assert not hasattr(Record, "zip")


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
        assert isinstance(v2["params"], Record)
        assert v2["params"]["r"] == 11.0
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

    def test_to_datatree(self):
        xr = pytest.importorskip("xarray")
        v = Record(x=np.array([1.0, 2.0]), y=np.array(3.0))
        dt = v.to_datatree()
        np.testing.assert_allclose(dt["/x"]["x"].values, [1.0, 2.0])
        np.testing.assert_allclose(float(dt["/y"]["y"].values), 3.0)

    def test_to_datatree_preserves_xarray_coords(self):
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            [1.0, 2.0, 3.0], dims=["time"],
            coords={"time": [10, 20, 30]},
        )
        v = Record(y=da)
        dt = v.to_datatree()
        y_out = dt["/y"]["y"]
        np.testing.assert_array_equal(y_out.coords["time"].values, [10, 20, 30])

    def test_to_datatree_loses_coords_when_leaf_converted(self):
        """Coord fidelity is only as good as the leaf survives transforms.

        If the user applies a function that converts the leaf to a plain
        JAX array (e.g., ``jnp.asarray``), the xarray structure is gone
        and ``to_datatree`` wraps it as a bare DataArray.
        """
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            [1.0, 2.0, 3.0], dims=["time"], coords={"time": [10, 20, 30]},
        )
        v = Record(y=da)
        # A real numeric transform: materialize to JAX array.
        v_t = v.map(lambda x: jnp.asarray(x))
        dt = v_t.to_datatree()
        assert "time" not in dt["/y"]["y"].coords

    def test_from_datatree_roundtrip(self):
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            [1.0, 2.0, 3.0], dims=["time"],
            coords={"time": [10, 20, 30]},
        )
        ds = xr.Dataset({"y": da})
        dt = xr.DataTree.from_dict({"/root": ds})
        v = Record.from_datatree(dt["root"])
        assert "y" in v
        np.testing.assert_allclose(np.asarray(v["y"]), [1.0, 2.0, 3.0])

    def test_from_datatree_nested(self):
        """from_datatree reconstructs nested Record from child groups."""
        xr = pytest.importorskip("xarray")
        inner = Record(a=np.array(1.0), b=np.array(2.0))
        outer = Record(params=inner, z=np.array(3.0))
        dt = outer.to_datatree()
        roundtripped = Record.from_datatree(dt)
        assert isinstance(roundtripped["params"], Record)
        np.testing.assert_allclose(float(np.asarray(roundtripped["params"]["a"])), 1.0)
        np.testing.assert_allclose(float(np.asarray(roundtripped["params"]["b"])), 2.0)
        np.testing.assert_allclose(float(np.asarray(roundtripped["z"])), 3.0)


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
        v2 = v.map(lambda x: x ** 2)
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
        assert v2["inner"]["x"] == 3.0
        assert v2["y"] == 4.0

    def test_map_with_names(self):
        v = Record(a=1.0, b=2.0)
        names_seen = []
        v.map_with_names(lambda n, x: names_seen.append(n) or x)
        assert names_seen == ["a", "b"]


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
    """Record carries the same ``.source`` / ``.with_source`` slot as
    Distribution, so workflow outputs can attach a Provenance node
    regardless of which of the three output types (Record, RecordArray,
    Distribution) the broadcasting layer produced.
    """

    def test_initial_source_is_none(self):
        r = Record(x=1.0, y=2.0)
        assert r.source is None

    def test_with_source_sets_and_returns_self(self):
        r = Record(x=1.0)
        out = r.with_source(Provenance("op", parents=()))
        assert out is r
        assert r.source.operation == "op"

    def test_with_source_is_write_once(self):
        r = Record(x=1.0)
        r.with_source(Provenance("first", parents=()))
        with pytest.raises(RuntimeError, match="write-once"):
            r.with_source(Provenance("second", parents=()))

    # Semantic transformations reset the source — the new Record is a
    # different logical value even though the class preserves.

    def test_replace_resets_source(self):
        r = Record(x=1.0).with_source(Provenance("orig", parents=()))
        r2 = r.replace(x=2.0)
        assert r2.source is None
        assert r.source.operation == "orig"  # original unaffected

    def test_merge_resets_source(self):
        r = Record(x=1.0).with_source(Provenance("orig", parents=()))
        merged = r.merge(Record(y=2.0))
        assert merged.source is None

    def test_without_resets_source(self):
        r = Record(x=1.0, y=2.0).with_source(Provenance("orig", parents=()))
        r2 = r.without("y")
        assert r2.source is None

    def test_map_resets_source(self):
        r = Record(x=1.0).with_source(Provenance("orig", parents=()))
        r2 = r.map(lambda v: v + 1)
        assert r2.source is None

    # Structural equality / hashing ignore source — two Records with the
    # same fields but different provenance are still equal.

    def test_eq_ignores_source(self):
        r1 = Record(x=1.0).with_source(Provenance("a", parents=()))
        r2 = Record(x=1.0).with_source(Provenance("b", parents=()))
        assert r1 == r2

    def test_hash_ignores_source(self):
        r1 = Record(x=1.0).with_source(Provenance("a", parents=()))
        r2 = Record(x=1.0)
        assert hash(r1) == hash(r2)

    # Pytree roundtrip drops the source (runtime-only metadata — a
    # Provenance parent isn't hashable by structure, so pushing it into
    # the aux tuple would break jax.tree_util.tree_unflatten's equality
    # semantics). Document this caveat with a test.

    def test_pytree_roundtrip_drops_source(self):
        r = Record(x=1.0, y=jnp.array([2.0, 3.0]))
        r.with_source(Provenance("op", parents=()))
        leaves, treedef = jax.tree_util.tree_flatten(r)
        r2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert r2.source is None
        # But the Record is otherwise structurally identical.
        assert r2 == r

    # Integration: walk provenance from a Record through a Distribution
    # ancestor via provenance_ancestors.

    def test_provenance_ancestors_walks_through_distribution(self):
        prior = Normal(loc=0.0, scale=1.0, name="prior")
        r = Record(theta=1.0).with_source(
            Provenance("draw", parents=(prior,))
        )
        ancestors = provenance_ancestors(r)
        assert len(ancestors) == 1
        assert ancestors[0] is prior

    def test_provenance_ancestors_walks_nested_records(self):
        prior = Normal(loc=0.0, scale=1.0, name="prior")
        middle = Record(theta=1.0).with_source(
            Provenance("draw", parents=(prior,))
        )
        outer = Record(result=2.0).with_source(
            Provenance("transform", parents=(middle,))
        )
        ancestors = provenance_ancestors(outer)
        names = [getattr(a, "name", None) for a in ancestors]
        assert names == [middle.name, "prior"]
