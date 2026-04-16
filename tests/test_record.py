"""Tests for probpipe.core.record.Record."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Record


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

    def test_numpy_arrays(self):
        arr = np.array([1.0, 2.0, 3.0])
        v = Record(x=arr)
        assert isinstance(v.raw("x"), np.ndarray)

    def test_jax_arrays(self):
        arr = jnp.array([1.0, 2.0])
        v = Record(x=arr)
        assert isinstance(v["x"], jnp.ndarray)

    def test_scalars(self):
        v = Record(a=1, b=2.5, c=True)
        assert v["a"].shape == ()
        assert float(v["b"]) == 2.5

    def test_nested(self):
        inner = Record(x=1.0, y=2.0)
        outer = Record(params=inner, z=3.0)
        assert isinstance(outer["params"], Record)
        assert float(outer["params"]["x"]) == 1.0

    def test_from_dict(self):
        v = Record.from_dict({"a": 1.0, "b": 2.0})
        assert v.fields == ("a", "b")

    def test_list_input(self):
        v = Record(x=[1.0, 2.0, 3.0])
        assert v["x"].shape == (3,)


# ---------------------------------------------------------------------------
# Field access
# ---------------------------------------------------------------------------


class TestFieldAccess:
    @pytest.fixture
    def v(self):
        return Record(r=1.8, K=70.0, phi=10.0)

    def test_attribute(self, v):
        np.testing.assert_allclose(float(v["r"]), 1.8, rtol=1e-5)

    def test_item(self, v):
        np.testing.assert_allclose(float(v["K"]), 70.0, rtol=1e-5)

    def test_key_path(self):
        v = Record(params=Record(r=1.8, K=70.0), obs=Record(y=np.zeros(5)))
        np.testing.assert_allclose(float(v["params", "r"]), 1.8, rtol=1e-5)

    def test_fields(self, v):
        assert v.fields == ("K", "phi", "r")

    def test_raw(self):
        arr = np.array([1.0, 2.0])
        v = Record(x=arr)
        assert v.raw("x") is arr

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
        assert float(v["b"]) == 2.0  # original unchanged
        assert float(v2["b"]) == 3.0

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


# ---------------------------------------------------------------------------
# Lazy resolution
# ---------------------------------------------------------------------------


class TestLazyResolution:
    def test_numpy_resolves_to_jax(self):
        arr = np.array([1.0, 2.0])
        v = Record(x=arr)
        result = v["x"]
        assert isinstance(result, jnp.ndarray)
        np.testing.assert_allclose(result, arr)

    def test_scalar_resolves_to_jax(self):
        v = Record(x=42.0)
        result = v["x"]
        assert isinstance(result, jnp.ndarray)
        assert result.shape == ()

    def test_jax_passthrough(self):
        arr = jnp.array([1.0, 2.0])
        v = Record(x=arr)
        assert v["x"] is arr  # no copy

    def test_resolution_cached(self):
        arr = np.array([1.0, 2.0])
        v = Record(x=arr)
        r1 = v["x"]
        r2 = v["x"]
        assert r1 is r2  # same object

    def test_xarray_resolves(self):
        xr = pytest.importorskip("xarray")
        da = xr.DataArray([1.0, 2.0, 3.0], dims=["time"])
        v = Record(y=da)
        result = v["y"]
        assert isinstance(result, jnp.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_xarray_coords_preserved(self):
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["time"],
            coords={"time": [10, 20, 30]},
        )
        v = Record(y=da)
        assert v._coords is not None
        assert "y" in v._coords
        assert v._coords["y"]["dims"] == ("time",)


# ---------------------------------------------------------------------------
# Flatten / unflatten
# ---------------------------------------------------------------------------


class TestFlatten:
    def test_flat_size_scalars(self):
        v = Record(a=1.0, b=2.0, c=3.0)
        assert v.flat_size == 3

    def test_flat_size_arrays(self):
        v = Record(x=jnp.zeros(5), y=jnp.zeros((2, 3)))
        assert v.flat_size == 11

    def test_flat_size_nested(self):
        v = Record(params=Record(r=1.0, K=2.0), obs=Record(y=jnp.zeros(4)))
        assert v.flat_size == 6

    def test_flatten_scalars(self):
        v = Record(a=1.0, b=2.0, c=3.0)
        flat = v.flatten()
        assert flat.shape == (3,)
        # Sorted order: a=1, b=2, c=3
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])

    def test_flatten_arrays(self):
        v = Record(x=jnp.array([1.0, 2.0]), y=jnp.array([3.0]))
        flat = v.flatten()
        # Sorted: x first, then y
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])

    def test_flatten_nested(self):
        v = Record(a=Record(x=1.0, y=2.0), b=3.0)
        flat = v.flatten()
        # a.x=1, a.y=2, b=3
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])

    def test_unflatten_roundtrip(self):
        v = Record(r=1.8, K=70.0, phi=10.0)
        flat = v.flatten()
        v2 = Record.unflatten(flat, template=v)
        assert v2.fields == v.fields
        np.testing.assert_allclose(float(v2["r"]), 1.8)
        np.testing.assert_allclose(float(v2["K"]), 70.0)
        np.testing.assert_allclose(float(v2["phi"]), 10.0)

    def test_unflatten_nested_roundtrip(self):
        v = Record(
            params=Record(r=1.0, K=2.0),
            obs=Record(y=jnp.array([3.0, 4.0, 5.0])),
        )
        flat = v.flatten()
        v2 = Record.unflatten(flat, template=v)
        np.testing.assert_allclose(float(v2["params"]["r"]), 1.0)
        np.testing.assert_allclose(float(v2["params"]["K"]), 2.0)
        np.testing.assert_allclose(v2["obs"]["y"], [3.0, 4.0, 5.0])

    def test_unflatten_preserves_shapes(self):
        v = Record(mat=jnp.zeros((2, 3)), vec=jnp.zeros(4))
        flat = v.flatten()
        v2 = Record.unflatten(flat, template=v)
        assert v2["mat"].shape == (2, 3)
        assert v2["vec"].shape == (4,)


# ---------------------------------------------------------------------------
# JAX PyTree
# ---------------------------------------------------------------------------


class TestPyTree:
    def test_tree_map(self):
        v = Record(a=1.0, b=2.0)
        v2 = jax.tree.map(lambda x: x * 2, v)
        assert isinstance(v2, Record)
        np.testing.assert_allclose(float(v2["a"]), 2.0)
        np.testing.assert_allclose(float(v2["b"]), 4.0)

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
        np.testing.assert_allclose(float(v2["params"]["r"]), 11.0)
        np.testing.assert_allclose(float(v2["z"]), 13.0)

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
    def test_to_dict(self):
        v = Record(a=1.0, b=np.array([2.0, 3.0]))
        d = v.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) == {"a", "b"}
        assert isinstance(d["a"], jnp.ndarray)

    def test_to_numpy(self):
        v = Record(a=jnp.array(1.0), b=jnp.array([2.0]))
        d = v.to_numpy()
        assert isinstance(d["a"], np.ndarray)
        assert isinstance(d["b"], np.ndarray)

    def test_to_dict_nested(self):
        v = Record(inner=Record(x=1.0), y=2.0)
        d = v.to_dict()
        assert isinstance(d["inner"], dict)
        assert float(d["inner"]["x"]) == 1.0

    def test_to_datatree(self):
        xr = pytest.importorskip("xarray")
        v = Record(x=np.array([1.0, 2.0]), y=np.array(3.0))
        dt = v.to_datatree()
        np.testing.assert_allclose(dt["/x"]["x"].values, [1.0, 2.0])
        np.testing.assert_allclose(float(dt["/y"]["y"].values), 3.0)

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
        np.testing.assert_allclose(v["y"], [1.0, 2.0, 3.0])

    def test_from_datatree_nested(self):
        """from_datatree reconstructs nested Record from child groups."""
        xr = pytest.importorskip("xarray")
        inner = Record(a=np.array(1.0), b=np.array(2.0))
        outer = Record(params=inner, z=np.array(3.0))
        dt = outer.to_datatree()
        roundtripped = Record.from_datatree(dt)
        assert isinstance(roundtripped["params"], Record)
        np.testing.assert_allclose(float(roundtripped["params"]["a"]), 1.0)
        np.testing.assert_allclose(float(roundtripped["params"]["b"]), 2.0)
        np.testing.assert_allclose(float(roundtripped["z"]), 3.0)


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
        np.testing.assert_allclose(v["data"], [1.0, 2.0])

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
        np.testing.assert_allclose(float(v2["a"]), 4.0)
        np.testing.assert_allclose(float(v2["b"]), 9.0)

    def test_map_nested(self):
        v = Record(inner=Record(x=2.0), y=3.0)
        v2 = v.map(lambda x: x + 1)
        np.testing.assert_allclose(float(v2["inner"]["x"]), 3.0)
        np.testing.assert_allclose(float(v2["y"]), 4.0)

    def test_map_with_names(self):
        v = Record(a=1.0, b=2.0)
        names_seen = []
        v.map_with_names(lambda n, x: names_seen.append(n) or x)
        assert names_seen == ["a", "b"]

    def test_zip(self):
        v1 = Record(a=1.0, b=2.0)
        v2 = Record(a=10.0, b=20.0)
        zipped = Record.zip(v1, v2)
        # zip stacks along a new leading axis
        np.testing.assert_allclose(zipped["a"], [1.0, 10.0])
        np.testing.assert_allclose(zipped["b"], [2.0, 20.0])

    def test_zip_mismatched_raises(self):
        v1 = Record(a=1.0)
        v2 = Record(b=2.0)
        with pytest.raises(ValueError, match="different fields"):
            Record.zip(v1, v2)


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

    def test_hash_by_fields(self):
        v1 = Record(a=1.0, b=2.0)
        v2 = Record(a=99.0, b=99.0)
        # Same field names → same hash (structural hash, not value hash)
        assert hash(v1) == hash(v2)
