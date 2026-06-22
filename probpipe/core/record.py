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
| :class:`~probpipe.NumericRecord` (subclass)                 | Single structured value; every leaf is a ``jax.Array`` (post-construction invariant).   |
| :class:`~probpipe.RecordArray`                              | Batch of ``Record`` elements sharing a :class:`EventTemplate`.                         |
| :class:`~probpipe.NumericRecordArray` (subclass)            | Batch of :class:`~probpipe.NumericRecord` elements with ``flatten`` / ``mean`` / ``var``. |
| :class:`EventTemplate`                                     | Structural skeleton: field names, per-field leaf shapes or ``None`` for opaque leaves.  |

**When to reach for which:**

* Use :class:`Record` when fields are heterogeneous (numeric array
  plus a label string, a DataFrame, an xarray object, ...) — or when
  you want to keep the original backend objects intact. ``Record``
  performs no coercion and accepts arbitrary leaves.
* Use :class:`~probpipe.NumericRecord` when you want to flatten /
  unflatten to a 1-D vector, take reductions, or pass the value
  through :func:`jax.numpy` operations. Construction coerces every
  leaf to a ``jax.Array`` (the post-construction invariant) and
  captures backend-specific metadata (xarray dims/coords, pandas
  index) via the aux registry so :meth:`NumericRecord.to_native` can
  restore the original backend on the reverse trip.
* Use :class:`~probpipe.RecordArray` / :class:`~probpipe.NumericRecordArray`
  for collections (e.g., posterior draws): each field has shape
  ``(*batch_shape, *leaf_shape)``. Integer indexing materialises a
  single element (``Record`` from ``RecordArray``, ``NumericRecord``
  from ``NumericRecordArray``); field indexing returns the batched
  array.
* Use :class:`EventTemplate` whenever you need to round-trip
  unflatten → flatten without an example instance, or describe the
  expected structure of a distribution's sample.

Usage::

    from probpipe import Record, NumericRecord

    params = NumericRecord(r=1.8, K=70.0, phi=10.0)
    data = Record(counts=np.array([2, 1, 3, 0, 5]), label="horseshoe")

    params["r"]            # → jnp.array(1.8)
    params.fields          # → ('r', 'K', 'phi')   # insertion order
    params.flatten()       # → jnp.array([1.8, 70., 10.])

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

A side effect of the no-coercion policy: Python ``list`` / ``tuple``
leaves have no ``.shape`` or ``.dtype``, so :meth:`EventTemplate.from_record`
sees them as opaque (``None``) — even if they contain numbers.
Wrap numeric lists in ``np.asarray`` or ``jnp.asarray`` before
storing them if you want a numeric template entry.

Round-trip to / from JAX
------------------------

ProbPipe's native array form is the JAX array. Use :meth:`Record.to_numeric`
to convert a ``Record`` (any leaves) to a :class:`NumericRecord` (every
leaf a ``jax.Array``), and :meth:`NumericRecord.to_native` to go back.
The reverse trip uses the per-type aux registry in
:mod:`probpipe.core._array_backend` to restore backend-specific metadata
(xarray dims / coords / attrs, pandas index / columns / dtypes) that
``jnp.asarray`` would otherwise drop. Direct ``NumericRecord(...)``
construction consults the same registry, so the two paths are
semantically identical.

Field ordering
--------------

Fields iterate in **insertion order** (the order keyword arguments are
passed, or the order of the input ``dict``). The ``/`` character is
reserved as a path separator on nested ``Record``s and ``EventTemplate``s
and is rejected at construction.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array, ArrayLike
from .constraints import Constraint
from .provenance import Provenance

if TYPE_CHECKING:
    # Annotation-only back-references: these live in modules that import
    # Record from here, so TYPE_CHECKING avoids the runtime import cycle.
    from ._numeric_record import NumericRecord
    from ._record_array import NumericRecordArray

__all__ = [
    "ArraySpec",
    "DistributionSpec",
    "EventTemplate",
    "FunctionSpec",
    "LeafSpec",
    "NumericEventTemplate",
    "OpaqueSpec",
    "Record",
]

# A field value: nested ``Record`` or anything else (stored as-is).
type _FieldValue = Any

# Separator for nested-path access (``record["a/b/c"]``); reserved in
# field names.
_PATH_SEP = "/"


def _check_no_path_sep(name: str) -> None:
    if _PATH_SEP in name:
        raise ValueError(
            f"Field name {name!r} must not contain {_PATH_SEP!r} "
            f"(reserved as the nested-path separator)."
        )


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


class Record:
    """Named, immutable, pytree-registered container for structured values.

    Fields iterate in insertion order and are returned verbatim;
    ``Record`` performs no coercion between backends (numpy, JAX, xarray,
    Python scalars, strings, nested Records are all accepted). Use
    :class:`NumericRecord` when you want a uniform ``jax.Array`` leaf
    type and flatten / unflatten support.

    Parameters
    ----------
    **fields
        Named values.  Values may be JAX or numpy arrays, Python scalars,
        strings, xarray / pandas objects, nested ``Record``, or any other
        opaque object. Nothing is converted at construction. Field names
        must not contain ``/`` (reserved as the nested-path separator).
    name : str, optional
        Name for provenance / introspection. Auto-generated from field
        names if not provided.
    """

    __slots__ = ("_name", "_source", "_store")

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
        for field_name in fields:
            _check_no_path_sep(field_name)
        store = dict(fields)
        object.__setattr__(self, "_store", store)
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

    def with_source(self, source: Provenance | None) -> Record:
        """Attach provenance to this Record (write-once).

        Passing ``None`` (e.g. the result of ``Provenance.create()`` under
        :attr:`ProvenanceMode.OFF`) is a no-op.

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
        if source is None:
            return self
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

    def __reduce__(self):
        return (_unpickle_record, (dict(self._store), self._name, self._source))

    # -- Field access -------------------------------------------------------

    def __getitem__(self, key: str | tuple[str, ...]) -> _FieldValue:
        if isinstance(key, str):
            if _PATH_SEP in key:
                return self[tuple(key.split(_PATH_SEP))]
            store = self._store
            if key not in store:
                raise KeyError(key)
            return store[key]
        if isinstance(key, tuple):
            v: Any = self
            for i, k in enumerate(key):
                if i > 0 and not isinstance(v, Record):
                    # Descending past a non-Record leaf is a path
                    # error, not an indexing error — translate so that
                    # ``"a/b" in record`` and ``record["a/b"]`` both
                    # raise the user-friendly KeyError instead of
                    # leaking a numpy/list-side IndexError.
                    raise KeyError(
                        f"path {'/'.join(key)!r} descends through "
                        f"non-Record leaf {type(v).__name__} at "
                        f"{'/'.join(key[:i])!r}"
                    )
                v = v[k]
            return v
        raise TypeError(f"key must be str or tuple[str, ...], got {type(key).__name__}")

    @property
    def fields(self) -> tuple[str, ...]:
        """Field names in insertion order."""
        return tuple(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, name: str) -> bool:
        if _PATH_SEP in name:
            try:
                self[name]
            except (KeyError, TypeError, IndexError):
                return False
            return True
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
        leaves (strings, opaque objects) are returned as-is. Backend
        metadata (xarray dims / coords, pandas index) is stripped — use
        :meth:`to_numeric` followed by :meth:`NumericRecord.to_native`
        if you need a metadata-preserving round-trip.
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

    def to_numeric(self) -> NumericRecord:
        """Convert to a :class:`NumericRecord` with every leaf a ``jax.Array``.

        Per-field metadata that ``jnp.asarray`` would drop (xarray
        dims / coords / attrs, pandas index / columns / dtypes) is
        captured via the aux registry in
        :mod:`probpipe.core._array_backend` and stored on the resulting
        ``NumericRecord``. Calling :meth:`NumericRecord.to_native`
        on the result reverses the conversion, restoring each leaf to
        its original backend type. Nested ``Record`` children recurse
        — every level becomes a ``NumericRecord``.

        Equivalent to :meth:`NumericRecord.from_record`.

        Raises
        ------
        TypeError
            If any leaf is not coercible via ``jnp.asarray`` (e.g.
            strings, opaque Python objects).
        """
        # Lazy import to avoid the module-level circular dep:
        # _numeric_record.py imports Record from this module.
        from ._numeric_record import NumericRecord

        return NumericRecord.from_record(self)

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
    # When a WorkflowFunction wraps a callable return as
    # ``Record({fn_name: callable})``, the caller can keep invoking it via
    # ``result(args)``. Multi-field records raise — unwrapping one of many
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
                f"{type(self).__name__} single field is not callable (got {type(only).__name__})."
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
        # Structural hash over class, field names, and per-field shape+dtype.
        # Numeric leaves are coerced via ``jnp.asarray`` so a scalar and its
        # array wrapping hash alike (they compare equal under ``__eq__``);
        # opaque leaves fall back to ``type(val)``.
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
# Leaf specs — the closed sum describing what one leaf of a template is.
# ---------------------------------------------------------------------------
#
# Each spec is a frozen, hashable dataclass so that an ``EventTemplate`` (which
# stores them) stays hashable for jit treedef caching and keeps the
# order-sensitive ``__eq__`` / ``__hash__`` contract.


@dataclass(frozen=True)
class ArraySpec:
    """A numeric array leaf: a fixed event ``shape`` plus optional metadata.

    ``dtype`` and ``support`` are optional (default ``None``); the current
    auto-build path stores only shape, and distributions populate the extra
    fields in a later phase. Both must be hashable when set.
    """

    shape: tuple[int, ...]
    dtype: jnp.dtype | None = None
    support: Constraint | None = None

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if not all(isinstance(d, int) and d >= 0 for d in shape):
            raise TypeError(
                f"ArraySpec.shape must be a tuple of non-negative ints, got {self.shape!r}"
            )
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class OpaqueSpec:
    """A non-array Python-object leaf (str, DataFrame, ...).

    ``meta`` is optional opaque metadata and must be hashable (or ``None``).
    """

    meta: Hashable = None


@dataclass(frozen=True)
class DistributionSpec:
    """A leaf whose value is a ``Distribution``.

    ``inner_template`` is the :class:`EventTemplate` of one draw from that
    distribution. Not yet produced in this phase.
    """

    inner_template: EventTemplate


@dataclass(frozen=True)
class FunctionSpec:
    """A leaf whose value is a callable.

    ``input_template`` / ``output_template`` are the :class:`EventTemplate`\\ s
    of the callable's input and output. Not yet produced in this phase.
    """

    input_template: EventTemplate
    output_template: EventTemplate


# ``_FieldSpec`` adds nested templates to the public leaf union;
# ``_FieldSpecInput`` also admits the construction-time sugar the constructor
# normalises (a bare shape tuple or ``None``).
type LeafSpec = ArraySpec | OpaqueSpec | DistributionSpec | FunctionSpec
type _FieldSpec = LeafSpec | EventTemplate
type _FieldSpecInput = _FieldSpec | tuple[int, ...] | None


def _to_spec(spec: _FieldSpecInput) -> _FieldSpec:
    """Normalise a constructor input to a stored field spec.

    Construction-time sugar (preserved): a bare shape ``tuple`` becomes an
    :class:`ArraySpec`, ``None`` becomes an :class:`OpaqueSpec`, and a nested
    :class:`EventTemplate` is kept as-is. Already-built specs pass through, so
    new code may supply explicit ``ArraySpec(...)`` / ``OpaqueSpec(...)`` etc.
    """
    # NB: an explicit class tuple rather than ``(LeafSpec, EventTemplate)`` —
    # pyright doesn't narrow ``spec`` through a union-alias inside isinstance,
    # which would leave the ``return spec`` typed as the wider input alias.
    if isinstance(spec, (ArraySpec, OpaqueSpec, DistributionSpec, FunctionSpec, EventTemplate)):
        return spec
    if spec is None:
        return OpaqueSpec()
    if isinstance(spec, tuple):
        return ArraySpec(shape=spec)
    raise TypeError(
        f"spec must be a shape tuple, None, a leaf spec "
        f"(ArraySpec/OpaqueSpec/DistributionSpec/FunctionSpec), or an "
        f"EventTemplate, got {type(spec).__name__}"
    )


def _is_numeric_spec(spec: Any) -> bool:
    """A numeric leaf: an :class:`ArraySpec` or a (nested) :class:`NumericEventTemplate`."""
    return isinstance(spec, (ArraySpec, NumericEventTemplate))


def _all_numeric(specs: Iterable[Any]) -> bool:
    """True iff every (raw, pre-normalisation) input spec is numeric.

    Drives the base-class auto-promotion hook so ``EventTemplate(x=(), y=(3,))``
    returns a ``NumericEventTemplate`` without opting in explicitly. Raw inputs
    also allow the shape-tuple sugar; ``None`` / ``OpaqueSpec`` /
    ``DistributionSpec`` / ``FunctionSpec`` / mixed nested templates and any
    unsupported type are non-numeric (``__init__`` rejects the latter).
    """
    return all(isinstance(s, tuple) or _is_numeric_spec(s) for s in specs)


def _pack_fields(
    fields: tuple[str, ...],
    field_kwargs: dict[str, Any],
    *,
    owner: str = "",
) -> Record:
    """Validate that *field_kwargs* names exactly *fields*, then build a Record.

    The general "named values → validated :class:`Record`" operation behind
    :meth:`EventTemplate.pack`. Raises ``TypeError`` if any field is missing
    or unexpected; otherwise returns a ``Record`` keyed in *fields* order.
    *owner* (optional) prefixes the error message — the calling distribution or
    template class name.

    :meth:`Distribution._pack_value` calls this directly rather than through
    :meth:`EventTemplate.pack` because some distributions expose ``fields``
    without a ``EventTemplate`` instance.
    """
    given = set(field_kwargs)
    expected = set(fields)
    missing = [f for f in fields if f not in given]
    extra = [k for k in field_kwargs if k not in expected]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if extra:
            parts.append(f"unexpected {extra}")
        prefix = f"{owner}: " if owner else ""
        raise TypeError(
            f"{prefix}expected exactly the fields {tuple(fields)} — {'; '.join(parts)}."
        )
    return Record(**{f: field_kwargs[f] for f in fields})


# dtype.kind codes for numeric arrays: b=bool, i=int, u=uint, f=float, c=complex.
_NUMERIC_KINDS = frozenset("biufc")


def _full_leaf_shape(val: Any) -> tuple[int, ...] | None:
    """Infer the full (pre-batch-strip) array shape of a Record leaf.

    Returns the leaf's shape for numeric scalars (``()``) and numeric arrays,
    or ``None`` for opaque leaves (strings, object arrays, Python lists/tuples,
    and anything without a numeric ``dtype``/``shape``). Used by
    :meth:`EventTemplate.from_record`.
    """
    if isinstance(val, (bool, int, float, complex, np.integer, np.floating, np.bool_)):
        return ()
    if (
        hasattr(val, "shape")
        and hasattr(val, "dtype")
        and getattr(val.dtype, "kind", None) in _NUMERIC_KINDS
    ):
        return tuple(val.shape)
    return None


# ---------------------------------------------------------------------------
# EventTemplate — structural skeleton
# ---------------------------------------------------------------------------


class EventTemplate:
    """Structural description of a Record: field names, leaf shapes, nesting.

    Stores the skeleton of a Record without data — field names, per-field
    shapes (for numeric leaves) or ``None`` (for opaque leaves), and
    optional nested ``EventTemplate`` for hierarchical structure.

    Inspired by JAX's ``PyTreeDef``: a template can reconstruct a Record
    from flat data, and describes the expected structure for type-checking
    and flattening.

    Parameters
    ----------
    **field_specs
        Named fields.  Each value is one of:

        - ``tuple[int, ...]`` — shape of a numeric array leaf
          (e.g., ``()`` for scalar, ``(3,)`` for 3-vector); normalised to
          :class:`ArraySpec`.
        - ``None`` — opaque (non-array) leaf; normalised to :class:`OpaqueSpec`.
        - a leaf spec — :class:`ArraySpec` / :class:`OpaqueSpec` /
          :class:`DistributionSpec` / :class:`FunctionSpec` (passed through).
        - ``EventTemplate`` — nested sub-structure (kept as-is).

    Examples
    --------
    ::

        EventTemplate(x=(), y=(3,))                    # -> NumericEventTemplate
        EventTemplate(label=None, x=())                 # -> EventTemplate (mixed)
        EventTemplate(physics=EventTemplate(force=(), mass=()), obs=())

    Notes
    -----
    Leaves are stored as frozen, hashable spec objects
    (``ArraySpec`` / ``OpaqueSpec`` / ``DistributionSpec`` / ``FunctionSpec``);
    ``__getitem__`` returns the stored spec (or nested template), while
    shape-shaped access stays on ``leaf_shapes`` / ``event_shapes`` /
    ``field_event_shape``.

    Calling ``EventTemplate(...)`` directly auto-promotes to a
    :class:`NumericEventTemplate` when every spec is numeric (and
    every nested sub-template is itself all-numeric). That keeps
    ``flat_size`` and ``numeric_leaf_shapes`` reachable in the common
    all-numeric case without requiring the caller to name the subclass.
    Mixed templates (any ``None`` spec) stay as plain ``EventTemplate``
    and do not expose ``flat_size`` — it isn't a meaningful quantity
    once opaque leaves are in the mix.
    """

    __slots__ = ("_specs",)

    def __new__(
        cls,
        _dict: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        # Only auto-promote when invoked directly on the base class —
        # explicit ``NumericEventTemplate(...)`` calls bypass this path
        # and run their own strict validation.
        if cls is EventTemplate:
            specs = _dict if _dict is not None else field_specs
            if specs and _all_numeric(specs.values()):
                return object.__new__(NumericEventTemplate)
        return object.__new__(cls)

    def __init__(
        self,
        _dict: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        source: Mapping[str, _FieldSpecInput]
        if _dict is not None:
            if field_specs:
                raise ValueError("Cannot pass both positional dict and keyword arguments")
            source = _dict
        else:
            source = field_specs
        if not source:
            raise ValueError(f"{type(self).__name__} requires at least one field")
        specs: dict[str, _FieldSpec] = {}
        for name, spec in source.items():
            _check_no_path_sep(name)
            try:
                specs[name] = _to_spec(spec)
            except TypeError as exc:
                raise TypeError(f"Field {name!r}: {exc}") from None
        self._post_validate(specs)
        object.__setattr__(self, "_specs", specs)

    def _post_validate(self, field_specs: dict[str, _FieldSpec]) -> None:
        """Subclass hook for stricter spec validation. No-op on the base."""
        return

    # -- Immutability -------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("EventTemplate is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("EventTemplate is immutable")

    def __reduce__(self):
        return (_unpickle_event_template, (dict(self._specs),))

    # -- Field access -------------------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        """Field names in insertion order."""
        return tuple(self._specs.keys())

    @property
    def leaf_shapes(self) -> dict[str, tuple[int, ...] | None]:
        """Per-field leaf shapes.  ``None`` for opaque (non-array) leaves.

        For nested ``EventTemplate`` fields, returns the nested
        template's ``leaf_shapes`` (not the template itself), keyed by
        ``/``-delimited paths so the keys round-trip with
        :meth:`Record.__getitem__`'s path syntax.
        """
        result: dict[str, tuple[int, ...] | None] = {}
        for name, spec in self._specs.items():
            if isinstance(spec, EventTemplate):
                for sub_name, sub_shape in spec.leaf_shapes.items():
                    result[f"{name}{_PATH_SEP}{sub_name}"] = sub_shape
            elif isinstance(spec, ArraySpec):
                result[name] = spec.shape
            else:
                # Opaque / distribution / function leaves have no array shape.
                result[name] = None
        return result

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-top-level-field event shapes.

        Unlike :attr:`leaf_shapes` (which descends into nested
        sub-templates and emits one entry per leaf), ``event_shapes``
        emits one entry per top-level field. Nested sub-templates and
        opaque leaves collapse to ``()``. This is the view that downstream
        Distribution code wants when answering "what is the per-field event
        shape of one draw?".
        """
        return {name: self.field_event_shape(name) for name in self._specs}

    def field_event_shape(self, name: str) -> tuple[int, ...]:
        """Event shape for one top-level field.

        :class:`ArraySpec` leaves return their ``shape`` verbatim; opaque /
        distribution / function leaves and nested ``EventTemplate``
        sub-structures collapse to ``()``. Raises ``KeyError`` if ``name`` is
        not a top-level field.
        """
        spec = self._specs[name]
        if isinstance(spec, ArraySpec):
            return spec.shape
        return ()

    def pack(self, **field_kwargs: Any) -> Record:
        """Build a :class:`Record` from named values matching this template.

        Validates that *field_kwargs* names exactly this template's top-level
        :attr:`fields` (no missing or unexpected names) and returns a
        ``Record`` keyed in field order. Values pass through unchanged (a
        nested-template field takes a sub-``Record``). Object form of
        :func:`_pack_fields`.
        """
        return _pack_fields(self.fields, field_kwargs, owner=type(self).__name__)

    # -- Numeric queries & projection ---------------------------------------

    @property
    def is_numeric(self) -> bool:
        """Whether every reachable leaf is an :class:`ArraySpec`.

        Recursive: descends into nested :class:`EventTemplate` fields and
        returns ``True`` only if *all* leaves (at every depth) are numeric
        array leaves. Any :class:`OpaqueSpec` / :class:`DistributionSpec` /
        :class:`FunctionSpec` leaf — or a nested sub-template that is not
        itself all-numeric — makes the whole template non-numeric.

        This is computed as an explicit recursive leaf check rather than
        ``isinstance(self, NumericEventTemplate)``. Under the ``__new__``
        auto-promotion invariant the two agree, but the recursive form is
        also correct for hand-built mixed nestings.

        Returns
        -------
        bool
            ``True`` iff every reachable leaf is an :class:`ArraySpec`.
        """
        for spec in self._specs.values():
            if isinstance(spec, ArraySpec):
                continue
            if isinstance(spec, EventTemplate):
                if not spec.is_numeric:
                    return False
                continue
            # Opaque / distribution / function leaf — not numeric.
            return False
        return True

    @property
    def is_multi_field(self) -> bool:
        """Whether this template describes more than one leaf.

        Counts *reachable* leaves recursively (descending into nested
        sub-templates), not top-level fields — so a single top-level field that
        nests several leaves is multi-field. For example
        ``EventTemplate(a=EventTemplate(b=(), c=()))`` has leaves ``a/b`` and
        ``a/c`` and is multi-field, whereas ``EventTemplate(a=EventTemplate(b=()))``
        describes the single leaf ``a/b`` and is not. Equivalent to
        ``len(self.leaf_shapes) > 1``.

        Returns
        -------
        bool
            ``True`` iff the template has more than one leaf; ``False`` iff it
            describes exactly one leaf.
        """
        return len(self.leaf_shapes) > 1

    def numeric_fields(self) -> tuple[str, ...]:
        """Top-level field names whose leaf is numeric.

        A top-level field is numeric if its spec is an :class:`ArraySpec` or a
        nested :class:`EventTemplate` that is itself all-numeric (see
        :attr:`is_numeric`). The exact complement of :meth:`non_numeric_fields`:
        together the two partition :attr:`fields` (each in insertion order).

        Note that this reports *top-level* fields only, so a field is numeric
        here iff its nested sub-template is *entirely* numeric — it does not
        descend to count partially-numeric nestings. Use :meth:`numeric_subset`
        for the recursive, path-stable projection to numeric leaves.

        Returns
        -------
        tuple of str
            Names of the numeric top-level fields, in insertion order. Empty
            when every top-level field is non-numeric.
        """
        result: list[str] = []
        for name, spec in self._specs.items():
            if isinstance(spec, ArraySpec) or (isinstance(spec, EventTemplate) and spec.is_numeric):
                result.append(name)
        return tuple(result)

    def non_numeric_fields(self) -> tuple[str, ...]:
        """Top-level field names whose leaf is non-numeric.

        A top-level field is non-numeric if its spec is an
        :class:`OpaqueSpec` / :class:`DistributionSpec` / :class:`FunctionSpec`
        leaf, or a nested :class:`EventTemplate` that is not itself all-numeric
        (see :attr:`is_numeric`). The exact complement of :meth:`numeric_fields`:
        together the two partition :attr:`fields`. Used to build clear inference
        error messages when a template cannot be projected to a numeric subset.

        Returns
        -------
        tuple of str
            Names of the non-numeric top-level fields, in insertion order.
            Empty when the template :attr:`is_numeric`.
        """
        result: list[str] = []
        for name, spec in self._specs.items():
            if isinstance(spec, ArraySpec):
                continue
            if isinstance(spec, EventTemplate) and spec.is_numeric:
                continue
            result.append(name)
        return tuple(result)

    def numeric_subset(self) -> NumericEventTemplate:
        """Project to the :class:`ArraySpec`-leaf sub-template.

        Keeps every numeric leaf, recursing into nested
        :class:`EventTemplate` fields (each contributes its own
        ``numeric_subset()``); drops :class:`OpaqueSpec` /
        :class:`DistributionSpec` / :class:`FunctionSpec` leaves; and prunes
        any nested template that becomes empty. Surviving leaves keep their
        ``/``-delimited paths (the projection is path-stable). Inference uses
        this to recover the numeric core of a mixed template.

        On an already-all-numeric template the result is an equal
        :class:`NumericEventTemplate` (the projection is idempotent).

        Returns
        -------
        NumericEventTemplate
            The numeric-leaf sub-template, so that :attr:`flat_size` and
            :attr:`numeric_leaf_shapes` are available.

        Raises
        ------
        ValueError
            If no numeric leaves survive — an :class:`EventTemplate` needs at
            least one field, so an empty numeric subset is meaningless. The
            message names the dropped (non-numeric) fields.
        """
        specs: dict[str, _FieldSpec] = {}
        for name, spec in self._specs.items():
            if isinstance(spec, ArraySpec):
                specs[name] = spec
            elif isinstance(spec, EventTemplate):
                try:
                    specs[name] = spec.numeric_subset()
                except ValueError:
                    # The empty-projection guard below is the *only* ValueError
                    # numeric_subset() raises, so catching it here means the
                    # nested template had no numeric leaves — prune it. If a
                    # future change adds another ValueError path, narrow this
                    # catch so it can't mask an unrelated failure.
                    continue
            # Opaque / distribution / function leaves are dropped.
        if not specs:
            raise ValueError(
                f"numeric_subset() of {type(self).__name__} is empty: no "
                f"ArraySpec leaves survive. Dropped non-numeric fields: "
                f"{self.non_numeric_fields()}."
            )
        return NumericEventTemplate(specs)

    # -- 1-D numeric (de)serialization --------------------------------------

    def to_vector(self, value: NumericRecord | NumericRecordArray) -> Array:
        """Serialize the numeric leaves of *value* into its 1-D vector representation.

        ``to_vector`` / :meth:`from_vector` convert between the structured and
        vector representations of a **numeric** value. A numeric value is in
        general a structured tree of array-valued leaves, with
        :attr:`~NumericEventTemplate.flat_size` the total number of scalar
        values making up those arrays. ``to_vector`` converts the structured
        value to its unique flat representation: an array of shape
        ``(flat_size,)``. A batch of numeric values is flattened to a matrix:
        an array of shape ``(*batch_shape, flat_size)``.

        This method is distinct from ``flatten``. The latter follows the
        JAX-pytree terminology, implying the mapping ``value -> (leaves, aux)``
        in the sense of :func:`jax.tree_util.tree_flatten`. ``flatten`` returns
        the list of the leaves of the value object, and thus is defined for all
        values (not just numeric ones). ``to_vector`` applies only to numeric
        values, and goes further in that it ravels the leaf arrays to a 1-D
        representation.

        Leaves are visited in **canonical leaf order** — the deterministic,
        depth-first, insertion-order traversal also used by :attr:`leaf_shapes`.
        The structural operation lives on the event template; a
        :class:`~probpipe.NumericRecord` inherits the functionality from the
        template.

        Parameters
        ----------
        value : NumericRecord or NumericRecordArray
            The value to serialize; its structure must match this template. A
            single :class:`~probpipe.NumericRecord` yields a 1-D vector; a
            batched :class:`~probpipe.NumericRecordArray` with
            ``batch_shape == B`` yields one vector per batch element.

        Returns
        -------
        jax.Array
            The concatenated numeric leaves, dtype-promoted by
            ``jnp.concatenate``. Shape ``(vector_size,)`` for a single
            ``value`` and ``(*B, vector_size)`` for a batched ``value`` with
            ``batch_shape == B``, where ``vector_size == self.flat_size``.

        Raises
        ------
        TypeError
            If this template is not :attr:`is_numeric` — it has non-numeric
            leaves, so there is no canonical numeric vector. The message names
            the offending fields and points at :meth:`numeric_subset` (which
            projects to the numeric leaves first); ``to_vector`` never silently
            drops non-numeric leaves. Also raised if *value* is not a
            :class:`~probpipe.NumericRecord` / :class:`~probpipe.NumericRecordArray`.

        See Also
        --------
        from_vector : Reconstruct a value from a flat vector (the inverse).
        numeric_subset : Project a mixed template to its numeric leaves.
        """
        if not self.is_numeric:
            raise TypeError(
                f"{type(self).__name__}.to_vector requires an all-numeric "
                f"template, but these fields are non-numeric: "
                f"{self.non_numeric_fields()}. Project to the numeric leaves "
                f"first with numeric_subset()."
            )
        from ._numeric_record import NumericRecord
        from ._record_array import NumericRecordArray

        if isinstance(value, NumericRecordArray):
            batch_shape = value.batch_shape
        elif isinstance(value, NumericRecord):
            batch_shape = ()
        else:
            raise TypeError(
                f"to_vector expects a NumericRecord (single) or "
                f"NumericRecordArray (batched), got {type(value).__name__}."
            )

        # ``tree_leaves`` yields the numeric leaves in canonical leaf order; ravel
        # each leaf's event dimensions and concatenate along the trailing axis.
        # Reshaping to ``(*batch_shape, -1)`` preserves leading batch axes
        # (``batch_shape == ()`` for a single value gives a 1-D result).
        leaves = jax.tree_util.tree_leaves(value)
        return jnp.concatenate([jnp.reshape(leaf, (*batch_shape, -1)) for leaf in leaves], axis=-1)

    def from_vector(
        self,
        vec: ArrayLike,
        *,
        non_numeric: Mapping[str, Any] | None = None,
    ) -> Any:
        """Reconstruct a numeric value from its 1-D vector representation.

        :meth:`to_vector` / ``from_vector`` convert between the structured and
        vector representations of a **numeric** value (see :meth:`to_vector`).
        ``from_vector`` is the inverse: it splits *vec* along its trailing axis
        into the template's leaves — in the same canonical leaf order
        :meth:`to_vector` uses — reshapes each chunk to its event shape, and
        rebuilds the structured value. The **rank of** *vec* selects single vs.
        batched: a vector of shape ``(vector_size,)`` rebuilds a single value,
        while a matrix of shape ``(*batch_shape, vector_size)`` rebuilds a batch
        with that ``batch_shape``. Here ``vector_size`` is the total scalar
        count, :attr:`~NumericEventTemplate.flat_size` (for a mixed template, the
        ``flat_size`` of its :meth:`numeric_subset`).

        This method is distinct from ``unflatten``. The latter is the inverse of
        the JAX-pytree ``flatten`` (the mapping ``(leaves, aux) -> value`` in the
        sense of :func:`jax.tree_util.tree_unflatten`), rebuilding an arbitrary
        value from its leaves and structural ``aux``. ``from_vector`` applies
        only to numeric values, and rebuilds one from its dense 1-D
        representation alone — splitting and reshaping the vector using this
        template's leaf shapes.

        When this template is mixed (not :attr:`is_numeric`), *vec* carries only
        the numeric leaves that :meth:`numeric_subset` keeps; pass *non_numeric*
        to supply the dropped leaves and rebuild the full mixed value. Like
        :meth:`to_vector`, the structural operation lives on the event template.

        Parameters
        ----------
        vec : array-like
            The flat numeric vector; its trailing axis must have length
            ``vector_size``.
        non_numeric : Mapping[str, Any], optional
            Values for the leaves that :meth:`numeric_subset` dropped, keyed by
            their ``/``-delimited leaf path, used to rebuild a *full* mixed
            value from a numeric-only vector. Required — and only meaningful —
            when this template is **not** :attr:`is_numeric`; must be ``None``
            or empty when it is. For a batched result each supplied value must
            itself carry the leading ``batch_shape``.

        Returns
        -------
        NumericRecord or NumericRecordArray
            When this template :attr:`is_numeric`: the reconstructed numeric
            value (single → :class:`~probpipe.NumericRecord`; batched →
            :class:`~probpipe.NumericRecordArray`).
        Record or RecordArray
            When this template is mixed and *non_numeric* is supplied: the full
            mixed value, with numeric leaves taken from *vec* and the remaining
            leaves taken from *non_numeric* (single → :class:`~probpipe.Record`;
            batched → :class:`~probpipe.RecordArray`).

        Raises
        ------
        ValueError
            If *vec*'s trailing axis is not ``vector_size``; if *non_numeric*
            is ``None`` while this template is non-numeric (the dropped leaves
            cannot be supplied); if *non_numeric* is non-empty while this
            template is numeric (there are no dropped leaves); if a required
            dropped-leaf path is missing from *non_numeric*; or if a batched
            *non_numeric* value's leading axes do not match ``batch_shape``.

        Notes
        -----
        **Round-trip invariant.** ``self.from_vector(self.to_vector(v)) == v``
        for any numeric ``value`` ``v`` matching this (numeric) template. For a
        mixed template ``T`` whose numeric leaves form ``Tn = T.numeric_subset()``,
        the analogous round trip
        ``T.from_vector(Tn.to_vector(vn), non_numeric=dropped)`` rebuilds the
        full mixed value (a plain :class:`~probpipe.Record` /
        :class:`~probpipe.RecordArray`) from its numeric part ``vn`` and the
        dropped leaves.

        See Also
        --------
        to_vector : Serialize a value to a flat vector (the inverse).
        numeric_subset : Project a mixed template to its numeric leaves.
        """
        from ._numeric_record import NumericRecord
        from ._record_array import NumericRecordArray, RecordArray

        vec = jnp.asarray(vec)
        numeric = self.is_numeric

        if numeric and non_numeric:
            raise ValueError(
                f"{type(self).__name__}.from_vector: this template is numeric "
                f"(numeric_subset drops no leaves), so non_numeric must be None "
                f"or empty, got keys {sorted(non_numeric)}."
            )
        if not numeric and not non_numeric:
            raise ValueError(
                f"{type(self).__name__}.from_vector: this template is mixed; its "
                f"non-numeric leaves {self.non_numeric_fields()} were dropped by "
                f"numeric_subset and must be supplied via non_numeric to rebuild "
                f"a full value."
            )

        # vector_size is the total scalar count across the numeric leaves;
        # numeric_subset() is idempotent on an already-numeric template.
        num_tpl = self if isinstance(self, NumericEventTemplate) else self.numeric_subset()
        vector_size = num_tpl.flat_size
        if vec.shape[-1] != vector_size:
            raise ValueError(
                f"{type(self).__name__}.from_vector: vec trailing axis is "
                f"{vec.shape[-1]}, expected vector_size={vector_size}."
            )

        batched = vec.ndim > 1
        batch_shape = tuple(vec.shape[:-1]) if batched else ()
        supplied = dict(non_numeric) if non_numeric else {}

        # Record classes for the rebuilt value: an all-numeric template rebuilds
        # NumericRecord(Array); a mixed template rebuilds a plain Record(Array)
        # carrying the supplied non-numeric leaves alongside the numeric ones.
        single_cls = NumericRecord if numeric else Record
        batched_cls = NumericRecordArray if numeric else RecordArray

        def _check_batch(path: str, val: Any) -> None:
            shape = getattr(val, "shape", None)
            if shape is not None and tuple(shape[: len(batch_shape)]) != batch_shape:
                raise ValueError(
                    f"{type(self).__name__}.from_vector: non_numeric[{path!r}] has "
                    f"leading shape {tuple(shape[: len(batch_shape)])}, expected "
                    f"batch_shape {batch_shape}."
                )

        # Walk the template in canonical leaf order, slicing ``vec`` into each
        # numeric leaf and reshaping it to (*batch_shape, *event_shape); a single
        # offset advances along vec's trailing axis. Non-numeric leaves (mixed
        # templates only) are filled from ``non_numeric`` and consume no vector.
        offset = 0

        def _build(template: EventTemplate, prefix: str) -> Record:
            nonlocal offset
            fields: dict[str, Any] = {}
            for name in template.fields:
                spec = template[name]
                path = f"{prefix}{name}"
                if isinstance(spec, EventTemplate):
                    fields[name] = _build(spec, f"{path}{_PATH_SEP}")
                elif isinstance(spec, ArraySpec):
                    size = prod(spec.shape) if spec.shape else 1
                    chunk = vec[..., offset : offset + size]
                    offset += size
                    fields[name] = jnp.reshape(chunk, (*batch_shape, *spec.shape))
                else:
                    # Opaque / distribution / function leaf — supplied by caller.
                    if path not in supplied:
                        raise ValueError(
                            f"{type(self).__name__}.from_vector: missing non_numeric "
                            f"value for dropped leaf {path!r}."
                        )
                    val = supplied[path]
                    if batched:
                        _check_batch(path, val)
                    fields[name] = val
            if batched:
                return batched_cls(fields, batch_shape=batch_shape, template=template)
            return single_cls(fields)

        return _build(self, "")

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> _FieldSpec:
        """Return the stored spec for *name*.

        Returns the leaf spec object (:class:`ArraySpec` / :class:`OpaqueSpec`
        / :class:`DistributionSpec` / :class:`FunctionSpec`) or, for a nested
        field, the nested :class:`EventTemplate`. For shape-shaped access use
        :attr:`leaf_shapes` / :attr:`event_shapes` / :meth:`field_event_shape`.
        """
        return self._specs[name]

    def __len__(self) -> int:
        return len(self._specs)

    # -- Equality and hashing -----------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EventTemplate):
            return NotImplemented
        # Order-sensitive comparison so equality matches the
        # order-sensitive ``__hash__`` (insertion order is part of the
        # template's identity). dict.__eq__ alone would ignore order,
        # breaking the eq/hash contract.
        return tuple(self._specs.items()) == tuple(other._specs.items())

    def __hash__(self) -> int:
        # All field specs (leaf specs and nested templates) are hashable, so
        # the order-sensitive item tuple hashes directly. Insertion order is
        # part of the template's identity.
        return hash(tuple(self._specs.items()))

    # -- Factory methods ----------------------------------------------------

    @classmethod
    def from_record(
        cls,
        record: Record,
        *,
        batch_shape: tuple[int, ...] = (),
    ) -> EventTemplate:
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
        # Promote plain ``EventTemplate.from_record`` to
        # ``NumericEventTemplate`` when the source signals it is all-numeric
        # (a ``NumericRecord`` or any Record whose recursive leaves are
        # numeric). That keeps ``flat_size`` reachable for the common
        # all-numeric case without requiring callers to name the subclass.
        target_cls = cls
        if cls is EventTemplate:
            from ._numeric_record import NumericRecord

            if isinstance(record, NumericRecord):
                target_cls = NumericEventTemplate
        n_batch = len(batch_shape)
        # Construction-time sugar; the constructor normalises to stored specs.
        specs: dict[str, _FieldSpecInput] = {}
        for name in record.fields:
            val = record[name]
            if isinstance(val, Record):
                specs[name] = target_cls.from_record(val, batch_shape=batch_shape)
                continue
            # Numeric leaf → event shape (drop leading batch dims); opaque → None.
            full_shape = _full_leaf_shape(val)
            specs[name] = None if full_shape is None else full_shape[n_batch:]
        return target_cls(specs)

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for name, spec in self._specs.items():
            if isinstance(spec, EventTemplate):
                parts.append(f"{name}={spec!r}")
            elif isinstance(spec, ArraySpec) and spec.dtype is None and spec.support is None:
                # Bare specs render as their sugar form (shape tuple / None).
                parts.append(f"{name}={spec.shape}")
            elif isinstance(spec, OpaqueSpec) and spec.meta is None:
                parts.append(f"{name}=None")
            else:
                parts.append(f"{name}={spec!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# NumericEventTemplate — all-numeric specialisation
# ---------------------------------------------------------------------------


class NumericEventTemplate(EventTemplate):
    """EventTemplate where every leaf is numeric.

    Extends :class:`EventTemplate` by requiring each spec to be a shape
    tuple (or a nested :class:`NumericEventTemplate`) — no opaque
    ``None`` leaves are allowed. That restriction is what makes
    :attr:`flat_size` and :meth:`numeric_leaf_shapes` meaningful:
    ``flat_size`` is the total number of scalar elements across every
    numeric leaf, and the unflatten machinery (``NumericRecord.unflatten``
    / ``NumericRecordArray.unflatten``) requires a template of this
    class so that every field can be reconstructed from a slice of the
    flat buffer.

    Use :meth:`EventTemplate.from_record` on a :class:`NumericRecord`
    (it auto-promotes) or call this constructor directly when you have
    the shape specs in hand.
    """

    __slots__ = ("_flat_size",)

    def _post_validate(self, field_specs: dict[str, _FieldSpec]) -> None:
        for name, spec in field_specs.items():
            if _is_numeric_spec(spec):
                continue
            if isinstance(spec, EventTemplate):
                raise TypeError(
                    f"NumericEventTemplate: nested field {name!r} is a "
                    f"{type(spec).__name__}; nested sub-templates must "
                    f"themselves be NumericEventTemplate."
                )
            if isinstance(spec, OpaqueSpec):
                raise TypeError(
                    f"NumericEventTemplate: field {name!r} is opaque "
                    f"(OpaqueSpec); opaque leaves are not allowed — use "
                    f"EventTemplate if you need a mixed template."
                )
            # DistributionSpec / FunctionSpec — non-array leaves.
            raise TypeError(
                f"NumericEventTemplate: field {name!r} has a non-numeric leaf "
                f"({type(spec).__name__}); only ArraySpec leaves (or nested "
                f"NumericEventTemplate) are allowed — use EventTemplate if you "
                f"need a mixed template."
            )

    def __init__(
        self,
        _dict: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        super().__init__(_dict, **field_specs)
        object.__setattr__(self, "_flat_size", self._compute_flat_size())

    @property
    def numeric_leaf_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-field shapes for numeric leaves.

        On :class:`NumericEventTemplate` every leaf is numeric, so this
        is equivalent to :attr:`leaf_shapes`. Kept as a distinct name for
        symmetry with historical callers that used it as a filter.
        """
        return dict(self.leaf_shapes)

    def _compute_flat_size(self) -> int:
        """Total scalar count across all numeric leaves."""
        total = 0
        for spec in self._specs.values():
            if isinstance(spec, NumericEventTemplate):
                total += spec.flat_size
            else:
                # spec is an ArraySpec — validated by ``_post_validate``.
                total += prod(spec.shape) if spec.shape else 1
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
    :class:`NumericEventTemplate` so that ``.flat_size`` is defined;
    non-array leaves (:class:`OpaqueSpec` / :class:`DistributionSpec` /
    :class:`FunctionSpec`) have no flat size and raise.
    """
    if isinstance(spec, NumericEventTemplate):
        return spec.flat_size
    if isinstance(spec, EventTemplate):
        raise TypeError(
            f"nested {type(spec).__name__} contains non-numeric leaves; "
            f"unflatten requires a NumericEventTemplate."
        )
    if isinstance(spec, ArraySpec):
        return prod(spec.shape) if spec.shape else 1
    if isinstance(spec, OpaqueSpec):
        raise TypeError(
            "opaque template fields have no flat size; unflatten is only "
            "defined for numeric-leaf (ArraySpec) fields."
        )
    raise TypeError(
        f"non-array template field ({type(spec).__name__}) has no flat size; "
        f"unflatten is only defined for numeric-leaf (ArraySpec) fields."
    )


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------


def _unpickle_record(store: dict, name: str, source) -> Record:
    r = Record(name=name, **store)
    if source is not None:
        object.__setattr__(r, "_source", source)
    return r


def _unpickle_event_template(specs: dict) -> EventTemplate:
    return EventTemplate(specs)


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
