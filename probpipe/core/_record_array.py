"""RecordArray / NumericRecordArray — batched Record containers.

A ``RecordArray`` stores a batch of Records with consistent field structure.
Each field has shape ``(*batch_shape, *leaf_shape)``.  ``NumericRecordArray``
adds numeric operations: ``to_vector`` (1-D serialization, inverse
``NumericEventTemplate.from_vector``), ``mean``, ``var``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array
from ._numeric_record import _NUMERIC_DTYPE_KINDS, NumericRecord
from .event_template import ArraySpec, EventTemplate
from .record import Record

__all__ = [
    "NumericRecordArray",
    "RecordArray",
    "_RecordArrayView",
]


class RecordArray(Record):
    """Batch of Records with consistent field structure.

    Each field stores values with shape ``(*batch_shape, *leaf_shape)``.
    A ``RecordArray`` *is* a :class:`Record` — the batched variant,
    parallel to the way :class:`DistributionArray` is a
    :class:`Distribution`. Consolidating the two in a single hierarchy
    means:

    - ``isinstance(x, Record)`` accepts both scalar and batched Records.
      Code that needs to distinguish uses
      ``isinstance(x, RecordArray)`` for the batched case, or
      ``isinstance(x, Record) and not isinstance(x, RecordArray)`` for
      scalar-only.
    - The :class:`~probpipe.core.tracked.Tracked` identity surface
      (``.name`` / ``.name_is_auto`` / ``.provenance`` / ``.with_name`` /
      ``.with_provenance``) is inherited from Record (stored on the slots
      declared on Record). A batch is contractually ``Tracked`` only; the
      ``annotations`` slot inherited from Record is an interim artifact of
      ``RecordArray`` subclassing ``Record`` and will go away when the batch
      types are reworked onto the generic ``Batch``.
    - The leaf-keyed *field-navigation* surface (``at_path`` / ``children`` /
      ``is_field``) is inherited from ``_NamedTree`` and works the same as on a
      single ``Record``; string ``[]`` is likewise leaf-only (use ``at_path`` for
      a sub-batch). The batch-axis operators (``len`` / iteration / integer ``[]``
      and the meaning of ``keys`` / ``values`` / ``items`` / ``in``) keep their
      current top-level batch behavior — their reconciliation, and the immutable
      edits ``replace`` / ``merge`` / ``without`` (which raise here), are not yet
      defined and will be settled when the batch axis is generalized. One
      consequence of this split is that ``in`` (top-level) and ``[]`` (leaf-only)
      can disagree: ``"outer" in arr`` is ``True`` for a top-level field that is an
      interior node, yet ``arr["outer"]`` raises — reach it with ``at_path``.

    Parameters
    ----------
    batch_shape : tuple of int
        Shape of the batch dimensions.
    template : EventTemplate
        Structural description of each element.
    name : str, optional
        Human-readable name for provenance / introspection. Defaults
        to ``"{class_name}({field list in template order})"``.
    **fields
        Named values, each with shape ``(*batch_shape, *leaf_shape)``.

    Notes
    -----
    Construct from a list of Records with :meth:`RecordArray.stack`.
    Indexing is either integer (``arr[i]`` → single :class:`Record`) or
    field name (``arr["x"]`` → batched leaf array).
    """

    __slots__ = ("_batch_shape", "_template")

    def __init__(
        self,
        _fields: Mapping[str, Any] | None = None,
        /,
        *,
        batch_shape: tuple[int, ...],
        template: EventTemplate,
        name: str | None = None,
        **fields: Any,
    ):
        if _fields is not None:
            if fields:
                raise ValueError("Cannot pass both positional dict and keyword arguments")
            fields = _fields
        if not fields:
            raise ValueError("RecordArray requires at least one field")
        if set(fields.keys()) != set(template.fields):
            raise ValueError(
                f"Field names {sorted(fields)} do not match template "
                f"fields {sorted(template.fields)}"
            )
        # Reorder to match the template so iteration order is canonical
        # regardless of kwarg order.
        store: OrderedDict[str, Any] = OrderedDict((name, fields[name]) for name in template.fields)
        # Subclass validation hook. Runs after sort / name-check so
        # subclasses (e.g. NumericRecordArray) see a canonicalised view
        # of the leaves. Raises from ``_validate_fields`` propagate.
        store = type(self)._validate_fields(store, batch_shape, template)
        # Inherit the Record plumbing for the field store and identity slots.
        # We bypass Record's normal constructor path because RecordArray
        # requires its own field-validation hook and an auto-name that
        # reflects the class name, not the "record(...)" default.
        name_is_auto = name is None
        if name is None:
            name = f"{type(self).__name__.lower()}({','.join(store.keys())})"
        object.__setattr__(self, "_tree", store)
        self._init_tracked(name, name_is_auto=name_is_auto)
        object.__setattr__(self, "_annotations", None)
        object.__setattr__(self, "_batch_shape", batch_shape)
        object.__setattr__(self, "_template", template)

    @classmethod
    def _validate_fields(
        cls,
        store: OrderedDict[str, Any],
        batch_shape: tuple[int, ...],
        template: EventTemplate,
    ) -> OrderedDict[str, Any]:
        """Hook for subclasses to validate / coerce leaves at construction.

        The base implementation is a no-op — ``RecordArray`` accepts any
        leaves, matching the permissive storage policy of ``Record``.

        Subclasses may return a new ``OrderedDict`` with the same keys
        (in template order) and optionally coerced values, or raise
        ``TypeError`` / ``ValueError`` on invalid input.
        """
        return store

    # ``__setattr__`` / ``__delattr__`` / ``.name`` / ``.provenance`` /
    # ``.with_provenance`` / ``.fields`` are inherited from :class:`Record`.

    def __reduce__(self):
        return (
            _unpickle_record_array,
            (
                dict(self._tree),
                self._batch_shape,
                self._template,
                self._name,
                self._name_is_auto,
                self._provenance,
            ),
        )

    # -- Properties ---------------------------------------------------------

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Shape of the batch dimensions."""
        return self._batch_shape

    @property
    def template(self) -> EventTemplate:
        """Structural description of each element."""
        return self._template

    @property
    def event_template(self) -> EventTemplate:
        """The authoritative :class:`EventTemplate` describing one element.

        Uniform accessor shared with single :class:`Record` (and
        :class:`~probpipe.core._distribution_base.Distribution`). A batch always
        stores its template explicitly — required to recover an element's event
        shape from a batched leaf — so this returns it directly (never inferred);
        it is the same object as :attr:`template`.
        """
        return self._template

    # -- Field access -------------------------------------------------------

    def __getitem__(self, key: str | tuple[str, ...] | int) -> Any:
        """Index a batched **field** by key, or one element by integer batch index.

        Indexing is dual along two orthogonal axes:

        - a **string / tuple key** selects a field along the *field axis*
          (``arr["x"]`` / ``arr["outer/a"]`` → the batched leaf array). Like a
          single ``Record``, this is leaf-only: a partial path that stops at a
          subtree raises ``KeyError`` — reach a sub-batch with :meth:`at_path`.
        - an **integer** selects one element along the *batch axis*
          (``arr[i]`` → a single, possibly nested ``Record``).

        A missing key, or a path that descends through a leaf, raises ``KeyError``;
        a non-str/tuple/int key raises ``TypeError``.
        """
        if isinstance(key, (str, tuple)):
            node = self.at_path(key)
            if isinstance(node, self._node_type()):
                raise KeyError(
                    f"{key!r} is a subtree, not a field; use at_path() to navigate to it"
                )
            return node
        if isinstance(key, (int, np.integer)):
            return self._get_record(int(key))
        raise TypeError(f"key must be str, tuple, or int, got {type(key).__name__}")

    def view(self, field: str) -> _RecordArrayView:
        """Return a single-field view carrying parent identity.

        Unlike ``ra[field]`` (which returns the raw column), a view
        remembers the parent ``RecordArray``. When multiple views of
        the same parent land in a single ``WorkflowFunction`` call,
        the sweep layer groups them by parent identity and iterates
        them in lockstep (zip) rather than cartesian-producting.

        Used internally by :meth:`~probpipe.record.Design.select_all`.
        Direct construction by end-users is supported but rarely needed.
        """
        return _RecordArrayView(self, field)

    _record_cls: type = Record
    """Class used to materialise a single element via integer indexing.

    Overridden on :class:`NumericRecordArray` so element extraction
    returns a :class:`NumericRecord` (preserving the numeric guarantee).
    """

    def _get_record(self, index: int) -> Record:
        """Extract a single Record at a flat batch index.

        Nested record fields (a co-batched ``RecordArray`` or a plain
        ``Record`` with batch-shaped leaves) are descended recursively so
        the element is itself a (nested) record, not indexed by the raw
        batch tuple.
        """
        nd_index = np.unravel_index(index, self._batch_shape)

        def _elem(val: Record | Array) -> Record | Array:
            if isinstance(val, RecordArray):
                return val._get_record(index)
            if isinstance(val, Record):
                # A NumericRecord requires NumericRecord children, so a plain-Record
                # nested field (batch-shaped leaves stored on a plain Record) is
                # promoted; a non-numeric parent keeps the child's own type.
                cls = NumericRecord if isinstance(self, NumericRecordArray) else type(val)
                return cls({k: _elem(child) for k, child in val.children.items()})
            return val[nd_index]

        return self._record_cls({name: _elem(self._tree[name]) for name in self._tree})

    def __contains__(self, name: str) -> bool:
        return name in self._tree

    def __len__(self) -> int:
        # Transitional: the top-level field count — this matches neither a
        # single Record's len (the leaf count) nor the batch count (use
        # ``prod(ra.batch_shape)`` for that).
        return len(self._tree)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tree)

    def keys(self) -> Iterator[str]:
        """Iterate over field names."""
        return iter(self._tree)

    def values(self) -> Iterator[Any]:
        """Iterate over field values (batched)."""
        for name in self._tree:
            yield self._tree[name]

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over (name, batched_value) pairs."""
        for name in self._tree:
            yield name, self._tree[name]

    # -- Selection (override to return views) ------------------------------

    def select(self, *fields: str, **mapping: str) -> dict[str, Any]:
        """Select fields as a dict of single-field **views**.

        Mirrors :meth:`Record.select` but each entry is a
        :class:`_RecordArrayView` rather than the raw column. The views
        carry this ``RecordArray`` as their parent, so splatting the
        result into a ``@workflow_function`` triggers the
        parent-identity **zip sweep** (one inner call per row, matching
        ``f(p=self)``) instead of cartesian-producting the fields as
        independent axes.

        For raw-column access, use ``self["field"]`` per field or
        iterate ``self.items()``.
        """
        result: dict[str, Any] = {}
        for f in fields:
            if f not in self._tree:
                raise KeyError(f"No field {f!r} in {type(self).__name__}")
            result[f] = self.view(f)
        for arg_name, field_name in mapping.items():
            if field_name not in self._tree:
                raise KeyError(f"No field {field_name!r} in {type(self).__name__}")
            result[arg_name] = self.view(field_name)
        return result

    # ``select_all`` is inherited from ``Record`` and calls
    # ``self.select(*self.fields)``, which dispatches here and returns
    # per-field views.

    # -- Factory methods ----------------------------------------------------

    @classmethod
    def stack(cls, records: list[Record], *, template: EventTemplate | None = None) -> RecordArray:
        """Stack a list of Records into a RecordArray with batch_shape=(n,).

        Parameters
        ----------
        records : list of Record
            Records with consistent field structure.
        template : EventTemplate, optional
            If not provided, inferred from the first record.

        Notes
        -----
        Any backend metadata captured on the source ``NumericRecord``
        instances (xarray dims / coords, pandas index) is dropped — the
        stacked leaves are plain ``jax.Array`` objects. ``RecordArray``
        does not currently carry per-row aux.
        """
        if not records:
            raise ValueError("Cannot stack empty list of Records")
        if template is None:
            template = EventTemplate.infer_from(records[0])
        if any(isinstance(c, EventTemplate) for c in template.children.values()):
            # Stacking into a nested batch is not yet supported. This raises
            # ``TypeError`` — not the ``NotImplementedError`` used by the batch
            # edit methods (``replace`` / ``merge`` / ``without`` / ``map``) and
            # the broadcast nested-record guards — on purpose: the broadcast
            # sweep calls ``stack`` inside ``except (TypeError, ValueError)``
            # blocks (see ``_broadcast_distributions``) and converts the failure
            # into a clearer message. A ``NotImplementedError`` would escape
            # those handlers and surface raw. Keep it a ``TypeError`` until the
            # *Batch rework makes nested stacking a first-class construction.
            raise TypeError("RecordArray.stack does not yet support nested templates.")
        fields: dict[str, Any] = {}
        for name in template.children:
            field_vals = [r[name] for r in records]
            fields[name] = jnp.stack(field_vals, axis=0)
        return cls(fields, batch_shape=(len(records),), template=template)

    # -- Structural transforms & edits (deferred for the batch types) --------
    #
    # ``replace`` / ``merge`` / ``without`` / ``map`` / ``map_with_keys`` /
    # ``from_nested_dict`` are defined on ``_NamedTree`` for the single
    # value/spec collections and rebuild through single-value construction. On
    # a batch their semantics (which axis they act on, how the batch shape
    # composes) are not yet defined, so they all raise ``NotImplementedError``
    # rather than dying inside the batch constructor.

    def replace(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "replace() is not supported on a batched record; edit the underlying "
            "field arrays directly."
        )

    def merge(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("merge() is not supported on a batched record.")

    def without(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("without() is not supported on a batched record.")

    def map(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "map() is not supported on a batched record; map over the underlying "
            "field arrays directly."
        )

    def map_with_keys(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("map_with_keys() is not supported on a batched record.")

    @classmethod
    def from_nested_dict(cls, data: Mapping[str, Any], **kwargs: Any) -> Any:
        raise NotImplementedError(
            "from_nested_dict() is not supported on a batched record; construct "
            "with batch_shape= and template= directly."
        )

    # -- Equality / hash ----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        # Identity fast-path preserves reflexivity when leaves contain
        # NaN (``jnp.array_equal`` treats NaN != NaN).
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        if self._batch_shape != other._batch_shape:
            return False
        if self.fields != other.fields:
            return False
        if self._template != other._template:
            return False
        for name, a in self._tree.items():
            b = other._tree[name]
            if isinstance(a, Record) or isinstance(b, Record):
                # A nested child (Record / RecordArray) cannot go through
                # ``jnp.array_equal`` (a multi-field record is not array-like);
                # delegate to its own ``__eq__`` so nested batches compare by
                # value and the documented ``from_vector``/``to_vector`` and
                # pickle round-trips hold.
                if type(a) is not type(b) or a != b:
                    return False
                continue
            try:
                if not bool(jnp.array_equal(a, b)):
                    return False
            except Exception:
                if a is not b:
                    return False
        return True

    # RecordArray is intentionally unhashable. ``__eq__`` compares leaf
    # arrays elementwise; a value-based hash would require materialising
    # every byte (prohibitive for large posterior batches) and would
    # crash inside JIT / vmap because traced arrays are not hashable by
    # content. We follow the NumPy precedent of making array-carrying
    # containers unhashable rather than silently O(n). If you need a
    # structural key, use ``(type(ra), ra.batch_shape, ra.fields,
    # ra.template)`` explicitly.
    __hash__ = None

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        field_parts = []
        for name in self._tree:
            val = self._tree[name]
            if hasattr(val, "shape"):
                field_parts.append(f"{name}=array(shape={val.shape})")
            else:
                field_parts.append(f"{name}=...")
        cls_name = type(self).__name__
        return f"{cls_name}(batch_shape={self._batch_shape}, {', '.join(field_parts)})"


# ---------------------------------------------------------------------------
# NumericRecordArray
# ---------------------------------------------------------------------------


class NumericRecordArray(RecordArray):
    """Batch of NumericRecords — all leaves are numeric arrays.

    Adds ``to_vector``, ``mean``, ``var`` operations.
    Construction validates that every leaf has a numeric dtype and
    shape ``(*batch_shape, *event_shape)`` matching the template, so
    pytree round-trips (``jax.tree.map``) cannot silently produce a
    ``NumericRecordArray`` with non-numeric or ill-shaped leaves.

    Each field has shape ``(*batch_shape, *event_shape)``.

    Notes
    -----
    The **canonical** storage for a nested top-level field is a nested
    ``NumericRecordArray`` (what :meth:`NumericEventTemplate.from_vector`
    builds). Construction also *accepts* the field as a ``NumericRecord``
    whose leaves carry the batch shape, but the two forms have different JAX
    pytree treedefs and compare unequal — prefer the canonical form when
    building batches by hand.
    """

    __slots__ = ()

    # Integer indexing (``arr[i]``) returns a NumericRecord so the numeric
    # guarantee is preserved through slicing.
    _record_cls: type = NumericRecord

    @classmethod
    def _validate_fields(
        cls,
        store: OrderedDict[str, Any],
        batch_shape: tuple[int, ...],
        template: EventTemplate,
    ) -> OrderedDict[str, Any]:
        """Require numeric dtype and matching event shape on every leaf.

        Raises ``TypeError`` if a leaf is non-numeric, ``ValueError`` if
        its shape is not ``(*batch_shape, *event_shape)`` for the
        corresponding template entry. Fields whose template spec is a
        nested ``EventTemplate`` are forwarded as-is — the nested
        element is allowed to be a ``Record`` / ``NumericRecord`` /
        ``RecordArray`` and is validated at its own construction site.
        """
        out: OrderedDict[str, Any] = OrderedDict()
        for name, raw in store.items():
            spec = template.children[name]
            # Nested structure: skip numeric validation, let the nested
            # type enforce its own invariant.
            if isinstance(spec, EventTemplate):
                out[name] = raw
                continue
            # Batched numeric leaves must expose dtype + shape. Plain
            # Python scalars / strings / lists never do, so this is the
            # one check we need to reject them as a batched field.
            if not hasattr(raw, "dtype") or not hasattr(raw, "shape"):
                raise TypeError(
                    f"NumericRecordArray: field {name!r} must be a "
                    f"numeric array, got {type(raw).__name__}"
                )
            kind = getattr(raw.dtype, "kind", None)
            if kind not in _NUMERIC_DTYPE_KINDS:
                raise TypeError(
                    f"NumericRecordArray: field {name!r} has non-numeric dtype {raw.dtype!r}"
                )
            event_shape = spec.shape if isinstance(spec, ArraySpec) else ()
            expected = tuple(batch_shape) + event_shape
            actual = tuple(raw.shape)
            if actual != expected:
                raise ValueError(
                    f"NumericRecordArray: field {name!r} has shape "
                    f"{actual}, expected {expected} "
                    f"(batch_shape={batch_shape}, event_shape={event_shape})"
                )
            out[name] = jnp.asarray(raw)
        return out

    def __reduce__(self):
        return (
            _unpickle_numeric_record_array,
            (
                dict(self._tree),
                self._batch_shape,
                self._template,
                self._name,
                self._name_is_auto,
                self._provenance,
            ),
        )

    # -- 1-D vector conversion ----------------------------------------------

    def to_vector(self) -> jnp.ndarray:
        """Serialize each batch element to its 1-D vector.

        Instance-level convenience for the numeric 1-D serialization whose
        structural definition lives on :meth:`NumericEventTemplate.to_vector`:
        delegates to this array's :attr:`template`. Returns a matrix of shape
        ``(*batch_shape, vector_size)`` — one raveled vector per batch element,
        leaves visited in canonical leaf order (insertion order, depth-first
        into nested records). The inverse,
        :meth:`NumericEventTemplate.from_vector`, reconstructs the batch.

        Distinct from ``list(record.values())`` (which keeps each batched leaf
        whole); ``to_vector`` ravels and concatenates each element's event
        dimensions into a dense matrix.
        """
        return self.template.to_vector(self)

    # -- Reductions ---------------------------------------------------------

    def _reduce(
        self,
        fn: Callable[[jnp.ndarray, int], jnp.ndarray],
        axis: int = 0,
    ) -> NumericRecordArray | Any:
        """Apply a reduction function over a batch axis."""
        new_batch = self._batch_shape[:axis] + self._batch_shape[axis + 1 :]

        def _reduce_value(value: Any, spec: Any) -> Any:
            if isinstance(value, NumericRecordArray):
                return value._reduce(fn, axis)
            if isinstance(value, Record):
                if not isinstance(spec, EventTemplate):
                    return fn(value, axis)
                fields = {
                    name: _reduce_value(child, spec.children[name])
                    for name, child in value.children.items()
                }
                if not new_batch:
                    return NumericRecord(fields)
                return NumericRecordArray(fields, batch_shape=new_batch, template=spec)
            return fn(value, axis)

        fields = {
            name: _reduce_value(value, self._template.children[name])
            for name, value in self._tree.items()
        }
        if not new_batch:
            return NumericRecord(fields)
        return NumericRecordArray(fields, batch_shape=new_batch, template=self._template)

    def mean(self, axis: int = 0) -> Any:
        """Mean over a batch axis.

        Returns ``NumericRecord`` if no batch dims remain, else
        ``NumericRecordArray``.
        """
        return self._reduce(jnp.mean, axis)

    def var(self, axis: int = 0) -> Any:
        """Variance over a batch axis.

        Returns ``NumericRecord`` if no batch dims remain, else
        ``NumericRecordArray``.
        """
        return self._reduce(jnp.var, axis)

    # -- Single-field array-like coercion ---------------------------------
    #
    # When a NumericRecordArray has exactly one numeric field, it behaves
    # like a thin wrapper around that field's batched array for the
    # ``np.asarray`` / ``jnp.asarray`` coercion paths. Mirrors the
    # single-field shim on ``NumericRecord`` (see ``_numeric_record.py``)
    # but for batched values — ``float()`` / ``int()`` / ``bool()`` are
    # intentionally **not** exposed here because the value isn't scalar.
    # ---------------------------------------------------------------------

    def _single_numeric_leaf(self):
        """Return the sole numeric leaf array, or raise ``TypeError``."""
        if len(self._tree) != 1:
            raise TypeError(
                f"NumericRecordArray with {len(self._tree)} fields is "
                f"not array-like; access a specific field with "
                f"array['field_name'] first."
            )
        only = next(iter(self._tree.values()))
        if isinstance(only, (Record, RecordArray)):
            raise TypeError(
                "NumericRecordArray with a nested Record field is not "
                "array-like; access the nested record explicitly."
            )
        return only

    def __array__(self, dtype=None, copy=None):
        leaf = self._single_numeric_leaf()
        arr = np.asarray(leaf, dtype=dtype) if dtype is not None else np.asarray(leaf)
        return arr.copy() if copy else arr

    def __jax_array__(self):
        return jnp.asarray(self._single_numeric_leaf())

    # Single-field shape / dtype / ndim shims — return the sole leaf's
    # full array shape (``batch_shape + leaf_shape``), matching the
    # one-thing-in-here philosophy of the other single-field shims.
    @property
    def shape(self) -> tuple[int, ...]:
        leaf = self._single_numeric_leaf()
        return tuple(getattr(leaf, "shape", ()))

    @property
    def dtype(self):
        leaf = self._single_numeric_leaf()
        return getattr(leaf, "dtype", None)

    @property
    def ndim(self) -> int:
        leaf = self._single_numeric_leaf()
        return int(getattr(leaf, "ndim", 0))


# ---------------------------------------------------------------------------
# Single-field view
# ---------------------------------------------------------------------------
#
# A view is a ``RecordArray`` with exactly one field, aliased into the
# parent's storage, plus a ``_parent`` reference. It's consciously a
# *plain* ``RecordArray`` subclass — not a ``NumericRecordArray``
# subclass — to avoid inheriting per-field batch-reduction methods
# (``.mean`` / ``.var`` / ``.to_vector``) that clash with the "act like
# the underlying column" intuition. Users who want numeric column ops
# convert explicitly: ``jnp.asarray(view).mean()``.
#
# Parallel to :class:`~probpipe.core._record_distribution._RecordDistributionView`:
# same "thin wrapper carrying parent identity for WF-layer sweep
# grouping" role. The WF layer detects ``view._parent`` as the
# shared-identity key; sibling views from the same parent zip into a
# single sweep axis, different parents product.
# ---------------------------------------------------------------------------


class _RecordArrayView(RecordArray):
    """View of a single named field in a :class:`RecordArray`.

    Constructed via ``parent[field]``. The underlying column is aliased
    — no copy — and ``view._parent`` carries the parent reference used
    by the ``WorkflowFunction`` sweep layer to group sibling views.

    Minimal surface: ``__array__`` / ``__jax_array__`` for conversion,
    ``.shape`` / ``.dtype`` / ``.ndim`` for introspection,
    ``view[i]`` / ``view[slice]`` to slice the underlying column,
    ``.parent`` / ``.field`` for sweep metadata. Arithmetic /
    reductions / reshaping require explicit
    ``jnp.asarray(view)`` conversion, matching the explicit-conversion
    policy already used for ``NumericRecord`` / ``NumericRecordArray``.

    Parameters
    ----------
    parent : RecordArray
        The source RecordArray.
    field : str
        Name of the field to view. Must be present in ``parent``.
    """

    __slots__ = ("_field", "_parent")

    def __init__(self, parent: RecordArray, field: str):
        if field not in parent._tree:
            raise KeyError(
                f"field {field!r} not in parent {type(parent).__name__}(fields={parent.fields})"
            )
        # ``children`` (not ``[]``): a top-level field may be a nested subtree,
        # and template ``[]`` is leaf-only.
        field_spec = parent.template.children[field]
        store = OrderedDict([(field, parent._tree[field])])
        template = EventTemplate({field: field_spec})
        # Populate RecordArray state directly — data was validated at
        # the parent's construction.
        object.__setattr__(self, "_tree", store)
        object.__setattr__(self, "_template", template)
        object.__setattr__(self, "_batch_shape", parent._batch_shape)
        self._init_tracked(f"{parent._name}[{field!r}]", name_is_auto=True)
        object.__setattr__(self, "_annotations", None)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_field", field)

    @property
    def parent(self) -> RecordArray:
        """The parent ``RecordArray`` this view points at.

        Shared-identity signal for the ``WorkflowFunction`` sweep layer:
        views with the same ``parent`` zip into one sweep axis; views
        with different parents (and plain ``RecordArray``s) product.
        """
        return self._parent

    @property
    def field(self) -> str:
        """The name of the viewed field."""
        return self._field

    # -- Column conversion + introspection ------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(getattr(self._tree[self._field], "shape", ()))

    @property
    def dtype(self) -> jnp.dtype | None:
        return getattr(self._tree[self._field], "dtype", None)

    @property
    def ndim(self) -> int:
        return int(getattr(self._tree[self._field], "ndim", 0))

    def __array__(self, dtype=None, copy=None):
        leaf = self._tree[self._field]
        arr = np.asarray(leaf, dtype=dtype) if dtype is not None else np.asarray(leaf)
        return arr.copy() if copy else arr

    def __jax_array__(self):
        return jnp.asarray(self._tree[self._field])

    # ``view[i]`` indexes the underlying column; ``view[name]`` is
    # idempotent (same field) or raises.
    def __getitem__(self, key):
        if isinstance(key, str):
            if key == self._field:
                return self
            raise KeyError(key)
        return self._tree[self._field][key]

    def __len__(self) -> int:
        # Row count, matching the column-like intuition. (``RecordArray``
        # itself returns the field count.)
        leaf = self._tree[self._field]
        return int(leaf.shape[0]) if getattr(leaf, "shape", ()) else 0

    def __iter__(self):
        return iter(self._tree[self._field])

    def __repr__(self) -> str:
        return (
            f"_RecordArrayView(parent={type(self._parent).__name__}, "
            f"field={self._field!r}, batch_shape={self._batch_shape})"
        )


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------


def _unpickle_record_array(
    store: dict, batch_shape: tuple, template, name: str, name_is_auto: bool, provenance
) -> RecordArray:
    ra = RecordArray(store, batch_shape=batch_shape, template=template, name=name)
    object.__setattr__(ra, "_name_is_auto", name_is_auto)
    if provenance is not None:
        object.__setattr__(ra, "_provenance", provenance)
    return ra


def _unpickle_numeric_record_array(
    store: dict, batch_shape: tuple, template, name: str, name_is_auto: bool, provenance
) -> NumericRecordArray:
    nra = NumericRecordArray(store, batch_shape=batch_shape, template=template, name=name)
    object.__setattr__(nra, "_name_is_auto", name_is_auto)
    if provenance is not None:
        object.__setattr__(nra, "_provenance", provenance)
    return nra


# ---------------------------------------------------------------------------
# JAX PyTree registration
# ---------------------------------------------------------------------------


def _record_array_flatten(
    ra: RecordArray,
) -> tuple[list, tuple[tuple[str, ...], tuple[int, ...], EventTemplate]]:
    """Flatten RecordArray for JAX pytree."""
    children = [ra._tree[name] for name in ra._tree]
    aux = (ra.fields, ra._batch_shape, ra._template)
    return children, aux


def _record_array_unflatten(
    aux: tuple[tuple[str, ...], tuple[int, ...], EventTemplate],
    children: list,
) -> RecordArray:
    """Unflatten RecordArray from JAX pytree."""
    field_names, batch_shape, template = aux
    return RecordArray(
        dict(zip(field_names, children)),
        batch_shape=batch_shape,
        template=template,
    )


def _numeric_record_array_unflatten(
    aux: tuple[tuple[str, ...], tuple[int, ...], EventTemplate],
    children: list,
) -> NumericRecordArray:
    """Unflatten NumericRecordArray from JAX pytree."""
    field_names, batch_shape, template = aux
    return NumericRecordArray(
        dict(zip(field_names, children)),
        batch_shape=batch_shape,
        template=template,
    )


jax.tree_util.register_pytree_node(RecordArray, _record_array_flatten, _record_array_unflatten)
jax.tree_util.register_pytree_node(
    NumericRecordArray, _record_array_flatten, _numeric_record_array_unflatten
)


# Views are intentionally pytree leaves, not nodes — JAX routes
# ``jnp.sum(view)`` / etc. through ``__jax_array__`` to operate on
# the underlying column, and we avoid an unflatten that would drop
# the ``_parent`` reference.
