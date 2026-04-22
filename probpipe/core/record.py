"""Record — ProbPipe's universal structured value type.

A named, immutable, JAX-pytree-registered container for structured
non-random values.  ``Record`` is the non-random counterpart to
:class:`~probpipe.core._distribution_base.Distribution`: it carries
named fields of arbitrary types and stores them as-is, with no
automatic coercion or caching.

The Record family
-----------------

| Class                                                       | Purpose                                                                                 |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| :class:`Record`                                             | Single structured value; fields may be arrays, scalars, strings, xarray, nested Record. |
| :class:`~probpipe.NumericRecord` (subclass)                 | Single structured value; every leaf is a numeric array (validated at construction).     |
| :class:`~probpipe.RecordArray`                              | Batch of ``Record`` elements sharing a :class:`RecordTemplate`.                         |
| :class:`~probpipe.NumericRecordArray` (subclass)            | Batch of :class:`~probpipe.NumericRecord` elements with ``flatten`` / ``mean`` / ``var``. |
| :class:`RecordTemplate`                                     | Structural skeleton: field names, per-field leaf shapes or ``None`` for opaque leaves.  |

**When to reach for which:**

* Use :class:`Record` when fields are heterogeneous (numeric array plus a
  label string, a DataFrame, an xarray object, ...). No method on
  ``Record`` assumes numeric leaves.
* Use :class:`~probpipe.NumericRecord` when you want to flatten / unflatten
  to a 1-D vector, take reductions, or pass the value through
  :func:`jax.numpy` operations. Construction validates that every leaf
  is a numeric scalar or array and coerces each to :class:`jnp.ndarray`
  so downstream code sees a uniform type.
* Use :class:`~probpipe.RecordArray` / :class:`~probpipe.NumericRecordArray`
  for collections (e.g., posterior draws): each field has shape
  ``(*batch_shape, *leaf_shape)``. Integer indexing materialises a
  single element (``Record`` from ``RecordArray``, ``NumericRecord``
  from ``NumericRecordArray``); field indexing returns the batched
  array.
* Use :class:`RecordTemplate` whenever you need to round-trip
  unflatten → flatten without an example instance, or describe the
  expected structure of a distribution's sample.

Usage::

    from probpipe import Record, NumericRecord

    params = NumericRecord(r=1.8, K=70.0, phi=10.0)
    data = Record(counts=np.array([2, 1, 3, 0, 5]), label="horseshoe")

    params["r"]            # → jnp.array(1.8)
    params.fields          # → ('K', 'phi', 'r')
    params.flatten()       # → jnp.array([70., 10., 1.8])

    data["counts"]         # → np.array([2, 1, 3, 0, 5]) (stored verbatim)
    data["label"]          # → "horseshoe"

    jax.tree.map(jnp.log, params)           # NumericRecord: all leaves are JAX-ready
    jax.jit(lambda v: v["r"] + v["K"])(params)

Storage policy
--------------

``Record`` performs no coercion at construction. ``record[name]``
returns whichever object was passed in — ``numpy.ndarray``,
``jnp.ndarray``, ``xarray.DataArray``, Python scalar, string, dict,
nested ``Record``, anything. Implications:

* ``jax.tree.map(fn, record)`` invokes ``fn`` on every non-``Record``
  leaf regardless of type, so the function must handle the leaf types
  the caller provided.
* ``jnp`` operations on raw Python scalars or ``xarray`` objects work
  exactly as they do outside ``Record`` — i.e., whatever JAX / xarray
  interop provides, no better and no worse.
* If you need a uniform ``jnp.ndarray`` type across leaves, either
  convert at the boundary (``jnp.asarray(rec[name])``) or use
  :class:`~probpipe.NumericRecord`, which coerces at construction.

``NumericRecord`` is the one place conversion happens automatically,
and only for a validated set of numeric inputs (numeric arrays,
numeric scalars including ``bool``, and objects with a numeric dtype
such as ``xarray.DataArray`` or ``pandas.Series``). Non-numeric
leaves raise ``TypeError`` at construction time with a message that
names the offending field and its type.

A side effect of the no-coercion policy: Python ``list`` / ``tuple``
leaves have no ``.shape`` or ``.dtype``, so :meth:`RecordTemplate.from_record`
sees them as opaque (``None``) — even if they contain numbers.
Wrap numeric lists in ``np.asarray`` or ``jnp.asarray`` before
storing them if you want a numeric template entry.

Coord / label lifecycle
-----------------------

The only structural metadata ``Record`` respects is what's encoded in
the stored leaf itself. If you pass in an ``xr.DataArray`` with
dims / coords / attrs, the leaf keeps those as long as no transform
replaces it. ``to_datatree`` re-reads them on export.

Coords are **not** carried as a Record-level sidecar, so any
operation that swaps a leaf for a plain array loses them. That
includes ``record.map(jnp.asarray)``, ``jax.tree.map(jnp.sqrt, ...)``
when applied to an xarray leaf, and construction of a
:class:`~probpipe.NumericRecord` from a ``Record``. Treat coords as a
construction-time snapshot that export re-attaches, not a property
that follows the data through computation. The xarray decoupling
tracked in
https://github.com/TARPS-group/prob-pipe/issues/125 will eventually
move coord handling to a dedicated subclass.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import ArrayLike
from .provenance import Provenance

if TYPE_CHECKING:
    import xarray as xr

__all__ = ["Record", "RecordTemplate", "NumericRecordTemplate"]

# A field value: nested ``Record`` or anything else (stored as-is).
_FieldValue: TypeAlias = "Any"


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


class Record:
    """Named, immutable, pytree-registered container for structured values.

    Fields are stored in alphabetical order by name and returned verbatim;
    ``Record`` performs no coercion between backends (numpy, JAX, xarray,
    Python scalars, strings, nested Records are all accepted). Use
    :class:`NumericRecord` when you want numeric-leaf validation and
    flatten / unflatten support.

    Parameters
    ----------
    **fields
        Named values.  Values may be JAX or numpy arrays, Python scalars,
        strings, xarray / pandas objects, nested ``Record``, or any other
        opaque object. Nothing is converted at construction.
    name : str, optional
        Name for provenance / introspection. Auto-generated from field
        names if not provided.

    Notes
    -----
    Field order is currently alphabetical for deterministic flattening; see
    https://github.com/TARPS-group/prob-pipe/issues/124 for the planned
    switch to insertion order.
    """

    __slots__ = ("_store", "_name", "_source")

    def __init__(
        self,
        _dict: dict[str, _FieldValue] | None = None,
        /,
        *,
        name: str | None = None,
        **fields: _FieldValue,
    ):
        if _dict is not None:
            if fields:
                raise ValueError("Cannot pass both positional dict and keyword arguments")
            fields = _dict
        if not fields:
            raise ValueError("Record requires at least one named field")
        store = OrderedDict(sorted(fields.items()))
        object.__setattr__(self, "_store", store)
        # Auto-generate name from field names if not provided
        if name is None:
            name = "record(" + ",".join(store.keys()) + ")"
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_source", None)

    # -- Name & provenance --------------------------------------------------

    @property
    def name(self) -> str:
        """Name of this Record."""
        return self._name

    @property
    def source(self) -> Provenance | None:
        """Provenance describing how this Record was created, or ``None``."""
        return self._source

    def with_source(self, source: Provenance) -> Record:
        """Attach provenance to this Record (write-once).

        Mirrors ``Distribution.with_source`` — `_source` is set once and
        subsequent calls raise. Semantic transformations (``replace``,
        ``merge``, ``without``, ``map``, ``map_with_names``) return a
        *new* Record with an empty source; the caller attaches fresh
        provenance there if desired.

        Notes
        -----
        ``_source`` is runtime-only metadata — it is not serialised into
        the JAX pytree aux (a ``Provenance`` parent is a ``Distribution``
        or ``Record``, neither of which is hashable by structure).
        Round-tripping through ``jax.tree_util.tree_flatten`` /
        ``tree_unflatten`` therefore drops the source; re-attach it on
        the reconstructed Record if you need to preserve the chain.
        """
        if self._source is not None:
            raise RuntimeError(
                f"Source already set on {self!r}. "
                "Provenance is write-once; create a new Record instead."
            )
        object.__setattr__(self, "_source", source)
        return self

    # -- Immutability -------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Record is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Record is immutable")

    # -- Field access -------------------------------------------------------

    def __getitem__(self, key: str | tuple[str, ...]) -> _FieldValue:
        if isinstance(key, str):
            store = self._store
            if key not in store:
                raise KeyError(key)
            return store[key]
        if isinstance(key, tuple):
            v: Any = self
            for k in key:
                v = v[k]
            return v
        raise TypeError(f"key must be str or tuple[str, ...], got {type(key).__name__}")

    @property
    def fields(self) -> tuple[str, ...]:
        """Field names in sorted order."""
        return tuple(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def items(self) -> Iterator[tuple[str, _FieldValue]]:
        """Iterate over (name, value) pairs."""
        return iter(self._store.items())

    def keys(self) -> Iterator[str]:
        """Iterate over field names."""
        return iter(self._store)

    def values(self) -> Iterator[_FieldValue]:
        """Iterate over values."""
        return iter(self._store.values())

    # -- Selection ----------------------------------------------------------

    def select(self, *fields: str, **mapping: str) -> dict[str, _FieldValue]:
        """Select fields as a dict, for splatting into function calls.

        Positional args use the field name as the key (identity mapping).
        Keyword args remap: ``select(x="field_name")`` → ``{"x": self.field_name}``.

        Usage::

            predict(**params.select("r", "K"), x=x_grid)
            predict(**params.select(growth_rate="r"), x=x_grid)
        """
        result: dict[str, _FieldValue] = {}
        for f in fields:
            if f not in self._store:
                raise KeyError(f"No field {f!r} in Record")
            result[f] = self[f]
        for arg_name, field_name in mapping.items():
            if field_name not in self._store:
                raise KeyError(f"No field {field_name!r} in Record")
            result[arg_name] = self[field_name]
        return result

    def select_all(self) -> dict[str, _FieldValue]:
        """Return every field as a dict, for splatting into function calls.

        Sugar for ``select(*self.fields)``. Subclasses whose
        ``__getitem__`` returns a view (``RecordArray`` →
        ``_RecordArrayView``, ``RecordDistribution`` →
        ``_RecordDistributionView``) inherit this method and return
        per-field views — so ``f(**ra.select_all())`` triggers the
        parent-identity zip sweep in ``WorkflowFunction``, and
        ``f(**dist.select_all())`` similarly preserves cross-field
        correlation.
        """
        return self.select(*self.fields)

    # -- Immutable updates --------------------------------------------------

    def replace(self, **updates: ArrayLike | Record) -> Record:
        """Return a new Record with specified fields replaced.

        Returns an instance of ``type(self)`` so that subclasses
        (``NumericRecord``) preserve their class through the update.
        """
        new = dict(self._store)
        for k, v in updates.items():
            if k not in new:
                raise KeyError(f"Cannot replace non-existent field {k!r}")
            new[k] = v
        return type(self)(new)

    def merge(self, other: Record) -> Record:
        """Return a new Record combining fields from self and other.

        Raises ``ValueError`` if any field names overlap. Returns an
        instance of ``type(self)``.
        """
        overlap = set(self._store) & set(other._store)
        if overlap:
            raise ValueError(f"Overlapping field names: {overlap}")
        combined = dict(self._store)
        combined.update(other._store)
        return type(self)(combined)

    def without(self, *names: str) -> Record:
        """Return a new Record with the specified fields removed.

        Returns an instance of ``type(self)``.
        """
        new = {k: v for k, v in self._store.items() if k not in names}
        if not new:
            raise ValueError("Cannot remove all fields from Record")
        return type(self)(new)

    # -- Backend conversion -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a dict of stored values (recursive for nested Record).

        Leaves are returned verbatim; no coercion to numpy or JAX.
        """
        result: dict[str, Any] = {}
        for name, val in self._store.items():
            if isinstance(val, Record):
                result[name] = val.to_dict()
            else:
                result[name] = val
        return result

    def to_numpy(self) -> dict[str, Any]:
        """Return a dict of numpy arrays (recursive for nested Record).

        Each numeric leaf is converted via ``np.asarray``. Non-numeric
        leaves (strings, opaque objects) are returned as-is.

        Notes
        -----
        xarray coord metadata is not preserved: an ``xr.DataArray``
        leaf becomes a plain numpy array with its dims / coords / attrs
        stripped. Use :meth:`to_datatree` instead to keep the xarray
        structure.
        """
        result: dict[str, Any] = {}
        for name, val in self._store.items():
            if isinstance(val, Record):
                result[name] = val.to_numpy()
            elif hasattr(val, "shape") or isinstance(val, (int, float, complex)):
                result[name] = np.asarray(val)
            else:
                result[name] = val
        return result

    def to_datatree(self) -> xr.DataTree:
        """Export to an xarray DataTree.

        Requires ``xarray`` to be installed. If a leaf is already an
        ``xr.DataArray``, its dims / coords / attrs are preserved. Any
        other leaf is wrapped as a bare ``DataArray`` without coord
        metadata.

        Note: coordinate metadata only survives a round-trip through
        ``Record`` for leaves that were ``xr.DataArray`` at construction
        time. It is **not** preserved through JAX transforms.
        """
        import xarray as xr

        datasets: dict[str, xr.Dataset | xr.DataTree] = {}
        for name, val in self._store.items():
            if isinstance(val, Record):
                datasets[f"/{name}"] = val.to_datatree()
                continue
            if isinstance(val, xr.DataArray):
                da = val
            else:
                da = xr.DataArray(np.asarray(val))
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

    def map(self, fn: Callable[[Any], Any]) -> Record:
        """Apply *fn* to each leaf, returning a new Record.

        Nested ``Record`` objects are traversed and rebuilt with the same
        class. ``fn`` sees leaves as stored (no coercion).
        """
        fields: dict[str, Any] = {}
        for name, val in self._store.items():
            if isinstance(val, Record):
                fields[name] = val.map(fn)
            else:
                fields[name] = fn(val)
        return type(self)(fields)

    def map_with_names(self, fn: Callable[[str, Any], Any]) -> Record:
        """Apply *fn(name, value)* to each leaf, returning a new Record."""
        fields: dict[str, Any] = {}
        for name, val in self._store.items():
            if isinstance(val, Record):
                fields[name] = val.map_with_names(fn)
            else:
                fields[name] = fn(name, val)
        return type(self)(fields)

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for name, val in self._store.items():
            if isinstance(val, Record):
                parts.append(f"{name}={val!r}")
            elif hasattr(val, "shape") and val.shape != ():
                parts.append(f"{name}=array(shape={val.shape})")
            else:
                parts.append(f"{name}={val!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    # -- Call-forwarding shim for single-field Records ----------------------
    #
    # Mirrors the single-field ``__jax_array__`` / ``__float__`` etc.
    # shims on ``NumericRecord``: when a WorkflowFunction wraps a
    # callable return as ``Record({fn_name: callable})``, the caller
    # can keep invoking the callable transparently via ``result(args)``.
    # Multi-field records raise — silently unwrapping one of many
    # fields would be ambiguous.
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(self._store) != 1:
            raise TypeError(
                f"{type(self).__name__} with {len(self._store)} fields is not "
                f"callable; access a specific field with record['field_name'] "
                f"first."
            )
        only = next(iter(self._store.values()))
        if not callable(only):
            raise TypeError(
                f"{type(self).__name__} single field is not callable "
                f"(got {type(only).__name__})."
            )
        return only(*args, **kwargs)

    # -- Equality / hash ----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        # Identity fast-path: self-equality must return True even when
        # leaves contain NaN (``jnp.array_equal`` treats NaN != NaN).
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        if self.fields != other.fields:
            return False
        for name, a in self._store.items():
            b = other._store[name]
            if isinstance(a, Record) and isinstance(b, Record):
                if a != b:
                    return False
            elif isinstance(a, Record) or isinstance(b, Record):
                return False
            elif hasattr(a, "shape") or hasattr(b, "shape"):
                try:
                    if not jnp.array_equal(jnp.asarray(a), jnp.asarray(b)):
                        return False
                except Exception:
                    if a is not b and a != b:
                        return False
            else:
                if a != b:
                    return False
        return True

    def __hash__(self) -> int:
        # Structural hash: class, field names, and per-field shape+dtype.
        # Numeric leaves are coerced via ``jnp.asarray`` first so a bare
        # Python scalar and its array wrapping hash identically
        # (``Record(a=1.0)`` and ``Record(a=jnp.asarray(1.0))`` compare
        # equal under ``__eq__`` and must therefore hash the same).
        # Non-numeric (opaque) leaves use ``type(val)`` as the signature
        # so identical types collide while different types don't.
        parts: list[Any] = [type(self).__name__]
        for name, val in self._store.items():
            if isinstance(val, Record):
                parts.append((name, hash(val)))
                continue
            try:
                arr = jnp.asarray(val)
            except (TypeError, ValueError):
                parts.append((name, "opaque", type(val).__name__))
                continue
            parts.append((name, tuple(arr.shape), str(arr.dtype)))
        return hash(tuple(parts))


# ---------------------------------------------------------------------------
# RecordTemplate — structural skeleton
# ---------------------------------------------------------------------------

# Leaf spec: shape tuple for numeric fields, None for opaque leaves.
_LeafSpec: TypeAlias = "tuple[int, ...] | None"
_FieldSpec: TypeAlias = "_LeafSpec | RecordTemplate"


def _all_numeric(specs) -> bool:
    """True iff every spec is either a shape tuple or an already-promoted
    :class:`NumericRecordTemplate`. Used by the base-class auto-promotion
    hook so ``RecordTemplate(x=(), y=(3,))`` returns a
    ``NumericRecordTemplate`` instance without opting in explicitly."""
    for spec in specs:
        if spec is None:
            return False
        if isinstance(spec, RecordTemplate) and not isinstance(
            spec, NumericRecordTemplate,
        ):
            return False
        if not isinstance(spec, (tuple, RecordTemplate)):
            # Leave validation of unsupported spec types to __init__.
            return False
    return True


class RecordTemplate:
    """Structural description of a Record: field names, leaf shapes, nesting.

    Stores the skeleton of a Record without data — field names, per-field
    shapes (for numeric leaves) or ``None`` (for opaque leaves), and
    optional nested ``RecordTemplate`` for hierarchical structure.

    Inspired by JAX's ``PyTreeDef``: a template can reconstruct a Record
    from flat data, and describes the expected structure for type-checking
    and flattening.

    Parameters
    ----------
    **field_specs
        Named fields.  Each value is one of:

        - ``tuple[int, ...]`` — shape of a numeric array leaf
          (e.g., ``()`` for scalar, ``(3,)`` for 3-vector).
        - ``None`` — opaque (non-array) leaf.
        - ``RecordTemplate`` — nested sub-structure.

    Examples
    --------
    ::

        RecordTemplate(x=(), y=(3,))                    # -> NumericRecordTemplate
        RecordTemplate(label=None, x=())                 # -> RecordTemplate (mixed)
        RecordTemplate(physics=RecordTemplate(force=(), mass=()), obs=())

    Notes
    -----
    Calling ``RecordTemplate(...)`` directly auto-promotes to a
    :class:`NumericRecordTemplate` when every spec is numeric (and
    every nested sub-template is itself all-numeric). That keeps
    ``flat_size`` and ``numeric_leaf_shapes`` reachable in the common
    all-numeric case without requiring the caller to name the subclass.
    Mixed templates (any ``None`` spec) stay as plain ``RecordTemplate``
    and do not expose ``flat_size`` — it isn't a meaningful quantity
    once opaque leaves are in the mix.
    """

    __slots__ = ("_specs",)

    def __new__(
        cls,
        _dict: dict[str, _FieldSpec] | None = None,
        /,
        **field_specs: _FieldSpec,
    ):
        # Only auto-promote when invoked directly on the base class —
        # explicit ``NumericRecordTemplate(...)`` calls bypass this path
        # and run their own strict validation.
        if cls is RecordTemplate:
            specs = _dict if _dict is not None else field_specs
            if specs and _all_numeric(specs.values()):
                return object.__new__(NumericRecordTemplate)
        return object.__new__(cls)

    def __init__(
        self,
        _dict: dict[str, _FieldSpec] | None = None,
        /,
        **field_specs: _FieldSpec,
    ):
        if _dict is not None:
            if field_specs:
                raise ValueError(
                    "Cannot pass both positional dict and keyword arguments"
                )
            field_specs = _dict
        if not field_specs:
            raise ValueError(f"{type(self).__name__} requires at least one field")
        # Validate specs
        for name, spec in field_specs.items():
            if spec is not None and not isinstance(spec, (tuple, RecordTemplate)):
                raise TypeError(
                    f"Field {name!r}: spec must be a shape tuple, None, "
                    f"or RecordTemplate, got {type(spec).__name__}"
                )
            if isinstance(spec, tuple):
                if not all(isinstance(d, int) and d >= 0 for d in spec):
                    raise TypeError(
                        f"Field {name!r}: shape must be a tuple of "
                        f"non-negative ints, got {spec!r}"
                    )
        self._post_validate(field_specs)
        specs = OrderedDict(sorted(field_specs.items()))
        object.__setattr__(self, "_specs", specs)

    def _post_validate(self, field_specs: dict[str, _FieldSpec]) -> None:
        """Subclass hook for stricter spec validation. No-op on the base."""
        return

    # -- Immutability -------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("RecordTemplate is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("RecordTemplate is immutable")

    # -- Field access -------------------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        """Field names in sorted order."""
        return tuple(self._specs.keys())

    @property
    def leaf_shapes(self) -> dict[str, tuple[int, ...] | None]:
        """Per-field leaf shapes.  ``None`` for opaque (non-array) leaves.

        For nested ``RecordTemplate`` fields, returns the nested
        template's ``leaf_shapes`` (not the template itself).
        """
        result: dict[str, tuple[int, ...] | None] = {}
        for name, spec in self._specs.items():
            if isinstance(spec, RecordTemplate):
                # Flatten nested template into dotted names
                for sub_name, sub_shape in spec.leaf_shapes.items():
                    result[f"{name}.{sub_name}"] = sub_shape
            else:
                result[name] = spec
        return result

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> _FieldSpec:
        return self._specs[name]

    def __len__(self) -> int:
        return len(self._specs)

    # -- Equality and hashing -----------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RecordTemplate):
            return NotImplemented
        return self._specs == other._specs

    def __hash__(self) -> int:
        def _spec_key(spec: _FieldSpec):
            if spec is None:
                return None
            if isinstance(spec, tuple):
                return spec
            return hash(spec)  # RecordTemplate
        return hash(tuple(
            (name, _spec_key(spec)) for name, spec in self._specs.items()
        ))

    # -- Factory methods ----------------------------------------------------

    @classmethod
    def from_record(
        cls,
        record: Record,
        *,
        batch_shape: tuple[int, ...] = (),
    ) -> RecordTemplate:
        """Infer a template from an existing Record.

        Numeric leaves are recorded with their shape (after stripping the
        leading ``batch_shape``). Python numeric scalars are treated as
        shape-``()`` leaves. Non-numeric leaves (strings, opaque objects)
        are recorded as ``None``.

        Parameters
        ----------
        record : Record
            Source record whose fields define the template structure.
        batch_shape : tuple of int
            Leading dimensions to strip from field shapes to get event
            shapes.  For a single-sample Record, use ``()`` (default).

        Notes
        -----
        A Python ``list`` or ``tuple`` leaf has no ``.shape`` / ``.dtype``
        and is treated as opaque (``None``) even if it contains numbers.
        Wrap it in ``np.asarray(...)`` or ``jnp.asarray(...)`` before
        putting it in the Record if you want a numeric template entry.
        Downstream operations that call ``NumericRecord.unflatten`` will
        otherwise raise on the opaque field.
        """
        # Promote plain ``RecordTemplate.from_record`` to
        # ``NumericRecordTemplate`` when the source signals it is all-numeric
        # (a ``NumericRecord`` or any Record whose recursive leaves are
        # numeric). That keeps ``flat_size`` reachable for the common
        # all-numeric case without requiring callers to name the subclass.
        target_cls = cls
        if cls is RecordTemplate:
            from ._numeric_record import NumericRecord
            if isinstance(record, NumericRecord):
                target_cls = NumericRecordTemplate
        n_batch = len(batch_shape)
        specs: dict[str, _FieldSpec] = {}
        for name in record.fields:
            val = record[name]
            if isinstance(val, Record):
                specs[name] = target_cls.from_record(val, batch_shape=batch_shape)
                continue
            # Numeric scalar / numeric array → strip leading batch dims.
            if isinstance(val, (bool, int, float, complex, np.integer, np.floating, np.bool_)):
                full_shape: tuple[int, ...] = ()
            elif hasattr(val, "shape") and hasattr(val, "dtype"):
                kind = getattr(val.dtype, "kind", None)
                if kind in {"b", "i", "u", "f", "c"}:
                    full_shape = tuple(val.shape)
                else:
                    specs[name] = None
                    continue
            else:
                specs[name] = None
                continue
            event_shape = full_shape[n_batch:] if n_batch else full_shape
            specs[name] = event_shape
        return target_cls(specs)

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for name, spec in self._specs.items():
            if isinstance(spec, RecordTemplate):
                parts.append(f"{name}={spec!r}")
            elif spec is None:
                parts.append(f"{name}=None")
            else:
                parts.append(f"{name}={spec}")
        return f"{type(self).__name__}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# NumericRecordTemplate — all-numeric specialisation
# ---------------------------------------------------------------------------


class NumericRecordTemplate(RecordTemplate):
    """RecordTemplate where every leaf is numeric.

    Extends :class:`RecordTemplate` by requiring each spec to be a shape
    tuple (or a nested :class:`NumericRecordTemplate`) — no opaque
    ``None`` leaves are allowed. That restriction is what makes
    :attr:`flat_size` and :meth:`numeric_leaf_shapes` meaningful:
    ``flat_size`` is the total number of scalar elements across every
    numeric leaf, and the unflatten machinery (``NumericRecord.unflatten``
    / ``NumericRecordArray.unflatten``) requires a template of this
    class so that every field can be reconstructed from a slice of the
    flat buffer.

    Use :meth:`RecordTemplate.from_record` on a :class:`NumericRecord`
    (it auto-promotes) or call this constructor directly when you have
    the shape specs in hand.
    """

    __slots__ = ("_flat_size",)

    def _post_validate(self, field_specs: dict[str, _FieldSpec]) -> None:
        for name, spec in field_specs.items():
            if spec is None:
                raise TypeError(
                    f"NumericRecordTemplate: field {name!r} is opaque "
                    f"(spec=None); opaque leaves are not allowed — use "
                    f"RecordTemplate if you need a mixed template."
                )
            if isinstance(spec, RecordTemplate) and not isinstance(
                spec, NumericRecordTemplate,
            ):
                raise TypeError(
                    f"NumericRecordTemplate: nested field {name!r} is a "
                    f"{type(spec).__name__}; nested sub-templates must "
                    f"themselves be NumericRecordTemplate."
                )

    def __init__(
        self,
        _dict: dict[str, _FieldSpec] | None = None,
        /,
        **field_specs: _FieldSpec,
    ):
        super().__init__(_dict, **field_specs)
        object.__setattr__(self, "_flat_size", self._compute_flat_size())

    @property
    def numeric_leaf_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-field shapes for numeric leaves.

        On :class:`NumericRecordTemplate` every leaf is numeric, so this
        is equivalent to :attr:`leaf_shapes`. Kept as a distinct name for
        symmetry with historical callers that used it as a filter.
        """
        return dict(self.leaf_shapes)

    def _compute_flat_size(self) -> int:
        """Total scalar count across all numeric leaves."""
        from .._utils import prod
        total = 0
        for spec in self._specs.values():
            if isinstance(spec, NumericRecordTemplate):
                total += spec.flat_size
            else:
                # spec is a shape tuple — validated by ``_post_validate``.
                total += prod(spec) if spec else 1
        return total

    @property
    def flat_size(self) -> int:
        """Total number of scalar elements across all numeric leaves."""
        return self._flat_size


# ---------------------------------------------------------------------------
# Template walking helpers
# ---------------------------------------------------------------------------


def _spec_size(spec: _FieldSpec) -> int:
    """Number of scalar elements a leaf-spec stands for.

    Shared by ``NumericRecord.unflatten`` and ``NumericRecordArray.unflatten``
    when walking a template and slicing a flat buffer. Nested specs must be
    :class:`NumericRecordTemplate` so that ``.flat_size`` is defined;
    opaque leaves (``spec is None``) have no flat size and raise.
    """
    if isinstance(spec, NumericRecordTemplate):
        return spec.flat_size
    if isinstance(spec, RecordTemplate):
        raise TypeError(
            f"nested {type(spec).__name__} contains opaque leaves; "
            f"unflatten requires a NumericRecordTemplate."
        )
    if spec is None:
        raise TypeError(
            "opaque template fields (shape=None) have no flat size; "
            "unflatten is only defined for numeric-leaf fields."
        )
    from .._utils import prod
    return prod(spec) if spec else 1


# ---------------------------------------------------------------------------
# JAX PyTree registration
# ---------------------------------------------------------------------------


def _record_flatten(v: Record) -> tuple[list, tuple[str, ...]]:
    """Flatten Record for JAX pytree traversal.

    Leaves are the stored values exactly as-is. JAX will further traverse
    any nested ``Record`` children it encounters because ``Record`` is a
    registered pytree type. Non-pytree objects (strings, opaque objects)
    become pytree leaves themselves, and any leaf-wise transformation
    applied by JAX must accept them.
    """
    children = list(v._store.values())
    return children, v.fields


def _record_unflatten(aux: tuple[str, ...], children: list) -> Record:
    """Unflatten Record from JAX pytree traversal."""
    return Record(dict(zip(aux, children)))


jax.tree_util.register_pytree_node(Record, _record_flatten, _record_unflatten)
