"""Record — ProbPipe's universal structured value type.

A named, immutable, lazy, JAX-pytree-registered container for structured
non-random values.  ``Record`` is the non-random counterpart to
:class:`~probpipe.core._distribution_base.Distribution`: it carries named
fields, stores backend-agnostic data (numpy, xarray, JAX), and converts
lazily on demand.

Usage::

    from probpipe import Record

    params = Record(r=1.8, K=70.0, phi=10.0)
    data = Record(counts=np.array([2, 1, 3, 0, 5]))

    params.r          # → jnp.array(1.8)
    params.fields()   # → ('K', 'phi', 'r')
    params.flatten()  # → jnp.array([70., 10., 1.8])

    jax.tree.map(jnp.log, params)   # leaf-wise transform
    jax.jit(lambda v: v.r + v.K)(params)  # JIT-compatible
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import ArrayLike

if TYPE_CHECKING:
    import xarray as xr

__all__ = ["Record"]

# Resolved field: JAX array for leaves, ``Record`` for nested structure.
_ResolvedField: TypeAlias = "jnp.ndarray | Record"


# ---------------------------------------------------------------------------
# Lazy resolution helpers
# ---------------------------------------------------------------------------


def _resolve(raw: Any) -> jnp.ndarray:
    """Convert a raw stored object to a JAX array."""
    if isinstance(raw, jnp.ndarray):
        return raw
    # xarray DataArray — extract .values before conversion
    if hasattr(raw, "values") and hasattr(raw, "dims"):
        return jnp.asarray(raw.values)
    # pandas Series / DataFrame — use .to_numpy() for clean conversion
    if hasattr(raw, "to_numpy"):
        return jnp.asarray(raw.to_numpy())
    return jnp.asarray(raw)


def _extract_coords(raw: Any) -> dict[str, Any] | None:
    """Extract coordinate metadata from xarray objects for round-tripping."""
    if hasattr(raw, "coords") and hasattr(raw, "dims"):
        return {
            "dims": raw.dims,
            "coords": {k: np.asarray(v) for k, v in raw.coords.items()},
            "attrs": dict(getattr(raw, "attrs", {})),
        }
    return None


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


class Record:
    """Named, immutable, lazy, pytree-registered container for structured values.

    Fields are stored in sorted order by name for deterministic flattening.
    Each field is lazily resolved to a JAX array on first access.  Nested
    ``Record`` objects are supported for hierarchical structure.

    Parameters
    ----------
    **fields
        Named values.  Each value can be a JAX array, numpy array, scalar,
        xarray DataArray, or a nested ``Record`` object.
    """

    __slots__ = ("_store", "_resolved", "_coords", "_name")

    def __init__(
        self,
        _dict: dict[str, ArrayLike | Record] | None = None,
        /,
        *,
        name: str | None = None,
        **fields: ArrayLike | Record,
    ):
        if _dict is not None:
            if fields:
                raise ValueError("Cannot pass both positional dict and keyword arguments")
            fields = _dict
        if not fields:
            raise ValueError("Record requires at least one named field")
        store = OrderedDict(sorted(fields.items()))
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_resolved", {})
        # Auto-generate name from field names if not provided
        if name is None:
            name = "record(" + ",".join(store.keys()) + ")"
        object.__setattr__(self, "_name", name)
        # Preserve xarray coordinate metadata so to_datatree() can
        # reconstruct labelled axes after a JAX round-trip.
        coords: dict[str, dict[str, Any]] = {}
        for n, raw in store.items():
            c = _extract_coords(raw)
            if c is not None:
                coords[n] = c
        object.__setattr__(self, "_coords", coords if coords else None)

    # -- Name ---------------------------------------------------------------

    @property
    def name(self) -> str:
        """Name of this Record."""
        return self._name

    # -- Immutability -------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Record is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Record is immutable")

    # -- Field access -------------------------------------------------------

    def __getattr__(self, name: str) -> _ResolvedField:
        store = object.__getattribute__(self, "_store")
        if name in store:
            return self._resolve_field(name)
        raise AttributeError(f"Record has no field {name!r}")

    def __getitem__(self, key: str | tuple[str, ...]) -> _ResolvedField:
        if isinstance(key, str):
            store = self._store
            if key not in store:
                raise KeyError(key)
            return self._resolve_field(key)
        if isinstance(key, tuple):
            v = self
            for k in key:
                v = v[k]
            return v
        raise TypeError(f"key must be str or tuple[str, ...], got {type(key).__name__}")

    def _resolve_field(self, name: str) -> _ResolvedField:
        """Resolve a single field, caching the result."""
        resolved = object.__getattribute__(self, "_resolved")
        if name in resolved:
            return resolved[name]
        raw = self._store[name]
        if isinstance(raw, Record):
            resolved[name] = raw
            return raw
        val = _resolve(raw)
        resolved[name] = val
        return val

    def fields(self) -> tuple[str, ...]:
        """Field names in sorted order."""
        return tuple(self._store.keys())

    def raw(self, name: str) -> Any:
        """Return the original un-resolved object for a field."""
        return self._store[name]

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def items(self) -> Iterator[tuple[str, _ResolvedField]]:
        """Iterate over (name, resolved_value) pairs."""
        for name in self._store:
            yield name, self._resolve_field(name)

    def keys(self) -> Iterator[str]:
        """Iterate over field names."""
        return iter(self._store)

    def values(self) -> Iterator[_ResolvedField]:
        """Iterate over resolved values."""
        for name in self._store:
            yield self._resolve_field(name)

    # -- Selection ----------------------------------------------------------

    def select(self, *fields: str, **mapping: str) -> dict[str, _ResolvedField]:
        """Select fields as a dict, for splatting into function calls.

        Positional args use the field name as the key (identity mapping).
        Keyword args remap: ``select(x="field_name")`` → ``{"x": self.field_name}``.

        Usage::

            predict(**params.select("r", "K"), x=x_grid)
            predict(**params.select(growth_rate="r"), x=x_grid)
        """
        result: dict[str, _ResolvedField] = {}
        for f in fields:
            if f not in self._store:
                raise KeyError(f"No field {f!r} in Record")
            result[f] = self._resolve_field(f)
        for arg_name, field_name in mapping.items():
            if field_name not in self._store:
                raise KeyError(f"No field {field_name!r} in Record")
            result[arg_name] = self._resolve_field(field_name)
        return result

    # -- Immutable updates --------------------------------------------------

    def replace(self, **updates: ArrayLike | Record) -> Record:
        """Return a new Record with specified fields replaced."""
        new = dict(self._store)
        for k, v in updates.items():
            if k not in new:
                raise KeyError(f"Cannot replace non-existent field {k!r}")
            new[k] = v
        return Record(new)

    def merge(self, other: Record) -> Record:
        """Return a new Record combining fields from self and other.

        Raises ``ValueError`` if any field names overlap.
        """
        overlap = set(self._store) & set(other._store)
        if overlap:
            raise ValueError(f"Overlapping field names: {overlap}")
        combined = dict(self._store)
        combined.update(other._store)
        return Record(combined)

    def without(self, *names: str) -> Record:
        """Return a new Record with the specified fields removed."""
        new = {k: v for k, v in self._store.items() if k not in names}
        if not new:
            raise ValueError("Cannot remove all fields from Record")
        return Record(new)

    # -- Flat-array conversion ----------------------------------------------

    @property
    def flat_size(self) -> int:
        """Total number of scalar elements across all leaf arrays."""
        total = 0
        for name in self._store:
            val = self._resolve_field(name)
            if isinstance(val, Record):
                total += val.flat_size
            else:
                total += val.size
        return total

    def flatten(self) -> jnp.ndarray:
        """Concatenate all leaf arrays into a single 1-D vector.

        Fields are traversed in sorted name order, depth-first for nested
        Record.  Each leaf is raveled before concatenation.
        """
        parts = []
        for name in self._store:
            val = self._resolve_field(name)
            if isinstance(val, Record):
                parts.append(val.flatten())
            else:
                parts.append(jnp.ravel(val))
        return jnp.concatenate(parts)

    @classmethod
    def unflatten(cls, flat: jnp.ndarray, *, template: Record) -> Record:
        """Reconstruct a Record from a flat array using a template for structure.

        The template provides field names, shapes, and nesting structure.
        """
        fields: dict[str, jnp.ndarray | Record] = {}
        offset = 0
        for name in template._store:
            tval = template._resolve_field(name)
            if isinstance(tval, Record):
                size = tval.flat_size
                child_flat = flat[offset:offset + size]
                fields[name] = cls.unflatten(child_flat, template=tval)
                offset += size
            else:
                size = tval.size
                fields[name] = flat[offset:offset + size].reshape(tval.shape)
                offset += size
        return cls(fields)

    # -- Backend conversion -------------------------------------------------

    def to_dict(self) -> dict[str, jnp.ndarray | dict]:
        """Return a dict of resolved JAX arrays (recursive for nested)."""
        result: dict[str, jnp.ndarray | dict] = {}
        for name in self._store:
            val = self._resolve_field(name)
            if isinstance(val, Record):
                result[name] = val.to_dict()
            else:
                result[name] = val
        return result

    def to_numpy(self) -> dict[str, np.ndarray | dict]:
        """Return a dict of numpy arrays (recursive for nested)."""
        result: dict[str, np.ndarray | dict] = {}
        for name in self._store:
            val = self._resolve_field(name)
            if isinstance(val, Record):
                result[name] = val.to_numpy()
            else:
                result[name] = np.asarray(val)
        return result

    def to_datatree(self) -> xr.DataTree:
        """Reconstruct an xarray DataTree, reattaching preserved coordinates.

        Requires ``xarray`` to be installed.
        """
        import xarray as xr

        datasets: dict[str, xr.Dataset | xr.DataTree] = {}
        coords_map = self._coords or {}
        for name in self._store:
            val = self._resolve_field(name)
            if isinstance(val, Record):
                datasets[f"/{name}"] = val.to_datatree()
                continue
            arr = np.asarray(val)
            if name in coords_map:
                meta = coords_map[name]
                da = xr.DataArray(
                    arr, dims=meta["dims"],
                    coords=meta["coords"], attrs=meta["attrs"],
                )
            else:
                da = xr.DataArray(arr)
            datasets[f"/{name}"] = xr.Dataset({name: da})
        return xr.DataTree.from_dict(datasets)

    # -- Coercion -----------------------------------------------------------

    @classmethod
    def ensure(cls, x: Any) -> Record:
        """Coerce *x* to Record if it isn't already.

        - ``Record`` → pass through
        - ``dict`` → ``Record(**x)``
        - array-like → ``Record(data=x)``
        """
        if isinstance(x, cls):
            return x
        if isinstance(x, dict):
            return cls(x)
        return cls(data=x)

    # -- Constructors -------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, ArrayLike | Record]) -> Record:
        """Construct Record from a dict of arrays."""
        return cls(d)

    @classmethod
    def from_datatree(cls, dt) -> Record:
        """Construct Record from an xarray DataTree.

        Extracts arrays and preserves coordinate metadata for round-tripping.
        """
        fields: dict[str, ArrayLike | Record] = {}
        if hasattr(dt, "data_vars"):
            for var_name in dt.data_vars:
                fields[var_name] = dt[var_name]
        if hasattr(dt, "children"):
            for child_name, child_node in dt.children.items():
                child_vars = list(child_node.data_vars) if hasattr(child_node, "data_vars") else []
                child_kids = list(child_node.children) if hasattr(child_node, "children") else []
                # Leaf group with a single variable matching its name →
                # extract the DataArray directly (avoids double-wrapping).
                if child_vars == [child_name] and not child_kids:
                    fields[child_name] = child_node[child_name]
                else:
                    fields[child_name] = cls.from_datatree(child_node)
        return cls(fields)

    # -- Leaf-wise operations -----------------------------------------------

    def map(self, fn: Callable[[jnp.ndarray], jnp.ndarray]) -> Record:
        """Apply *fn* to each leaf array, returning a new Record."""
        fields: dict[str, jnp.ndarray | Record] = {}
        for name in self._store:
            val = self._resolve_field(name)
            if isinstance(val, Record):
                fields[name] = val.map(fn)
            else:
                fields[name] = fn(val)
        return Record(fields)

    def map_with_names(self, fn: Callable[[str, jnp.ndarray], jnp.ndarray]) -> Record:
        """Apply *fn(name, array)* to each leaf, returning a new Record."""
        fields: dict[str, jnp.ndarray | Record] = {}
        for name in self._store:
            val = self._resolve_field(name)
            if isinstance(val, Record):
                fields[name] = val.map_with_names(fn)
            else:
                fields[name] = fn(name, val)
        return Record(fields)

    @staticmethod
    def zip(v1: Record, v2: Record) -> Record:
        """Stack matching fields from two Record along a new leading axis."""
        if v1.fields() != v2.fields():
            raise ValueError(
                f"Cannot zip Record with different fields: "
                f"{v1.fields()} vs {v2.fields()}"
            )
        fields: dict[str, jnp.ndarray | Record] = {}
        for name in v1._store:
            a = v1._resolve_field(name)
            b = v2._resolve_field(name)
            if isinstance(a, Record) and isinstance(b, Record):
                fields[name] = Record.zip(a, b)
            else:
                fields[name] = jnp.stack([a, b])
        return Record(fields)

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for name in self._store:
            raw = self._store[name]
            if isinstance(raw, Record):
                parts.append(f"{name}={raw!r}")
            else:
                val = self._resolve_field(name)
                if val.ndim == 0:
                    parts.append(f"{name}={float(val):.6g}")
                else:
                    parts.append(f"{name}=array(shape={val.shape})")
        return f"Record({', '.join(parts)})"

    # -- Equality (structural, not value) -----------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Record):
            return NotImplemented
        if self.fields() != other.fields():
            return False
        for name in self._store:
            a = self._resolve_field(name)
            b = other._resolve_field(name)
            if isinstance(a, Record) and isinstance(b, Record):
                if a != b:
                    return False
            elif isinstance(a, Record) or isinstance(b, Record):
                return False
            else:
                if not jnp.array_equal(a, b):
                    return False
        return True

    def __hash__(self) -> int:
        # Hash by field names only (like JAX pytree aux_data).
        return hash(self.fields())


# ---------------------------------------------------------------------------
# JAX PyTree registration
# ---------------------------------------------------------------------------


def _record_flatten(v: Record) -> tuple[list, tuple[str, ...]]:
    """Flatten Record for JAX pytree traversal."""
    children = []
    for name in v._store:
        raw = v._store[name]
        if isinstance(raw, Record):
            children.append(raw)
        else:
            children.append(v._resolve_field(name))
    return children, v.fields()


def _record_unflatten(aux: tuple[str, ...], children: list) -> Record:
    """Unflatten Record from JAX pytree traversal."""
    return Record(dict(zip(aux, children)))


jax.tree_util.register_pytree_node(Record, _record_flatten, _record_unflatten)
