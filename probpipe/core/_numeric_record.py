"""The numeric, all-array specialization of :class:`~probpipe.Record`.

This module provides :class:`NumericRecord`, the :class:`~probpipe.Record`
whose every field is numeric. Leaves are stored **in their native form** —
a ``jax`` / ``numpy`` array, an ``xarray.DataArray``, a ``pandas`` object, or
any registered array backend — and convert to ``jax.Array`` lazily, at the
compute boundary (JAX pytree flatten, :meth:`NumericRecord.to_vector`, the
single-field scalar shim), with a set-once per-leaf cache. Because the
compute boundary presents an ordinary PyTree of arrays, a ``NumericRecord``
passes through ``jit`` / ``vmap`` / ``grad`` unchanged and gains a flat 1-D
vector form, while navigation (``record["x"]`` / ``children`` / ``at_path``)
returns the native leaves verbatim.

See :class:`NumericRecord` for the specifics: the numeric-leaf invariant, the
lazy conversion contract, the flat-vector layout (``to_vector`` /
``from_vector``), and the single-field scalar coercion.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array, ArrayLike
from ._array_backend import _to_jax_array, array_backend_for
from .event_template import EventTemplate, NumericEventTemplate, _is_numeric_dtype
from .named_tree import _PATH_SEP, _check_no_path_sep, _unflatten_paths
from .record import Record

__all__ = ["NumericRecord", "_is_numeric_leaf"]


# Scalar types accepted as numeric leaves. ``bool`` is intentionally
# included: JAX treats it as dtype ``bool_`` and numpy arrays of bools
# participate in arithmetic as 0/1.
_NUMERIC_SCALARS = (bool, int, float, complex, np.integer, np.floating, np.bool_)


def _is_numeric_leaf(val: Any) -> bool:
    """True if *val* is a numeric array, numeric container, or numeric scalar.

    Resolution is registry-first: a value whose type has a registered
    :class:`~probpipe.ArrayBackend` answers through its ``is_numeric`` hook
    (so e.g. a ``pandas.DataFrame`` counts iff every column is numeric);
    everything else duck-types on a numeric ``dtype`` / ``shape``. Rejects
    object-dtype arrays, string-like scalars, and opaque types. Dtype
    numericness is decided by the shared ``_is_numeric_dtype`` predicate, so
    this gate, ``NumericRecordArray._validate_fields``, and template
    inference agree on what counts as numeric.
    """
    if isinstance(val, (str, bytes)):
        return False
    if isinstance(val, _NUMERIC_SCALARS):
        return True
    backend = array_backend_for(val)
    if backend is not None:
        return backend.is_numeric(val)
    if hasattr(val, "dtype") and hasattr(val, "shape"):
        return _is_numeric_dtype(val.dtype)
    return False


class NumericRecord(Record):
    """A :class:`Record` whose fields are all numeric, stored in native form.

    A ``NumericRecord`` is the numeric specialization of :class:`Record`. It
    holds the same named, ordered, possibly-nested collection of fields, but
    constrains every field to be numeric. It inherits the full
    :class:`Record` interface — the leaf-keyed mapping, the tree navigation,
    the metadata (:attr:`~Record.name`, :attr:`~Record.provenance`), and the
    equality and hashing rules — and adds the array-only features described
    below.

    The numeric-leaf invariant, and native storage
    ----------------------------------------------
    Every leaf must be numeric: a numeric array or container from any
    supported backend (``jax.numpy``, ``numpy``, ``xarray.DataArray``, a
    ``pandas`` object with numeric dtypes, or any type registered via
    :func:`~probpipe.register_array_backend`), a numeric Python scalar
    (``int``, ``float``, ``complex``, or ``bool``), or a nested
    ``NumericRecord``. Construction **validates without converting**: an
    array-like leaf is stored verbatim in its native form, reading only its
    container metadata (shape / dtype), so a lazy or disk-backed value is
    not materialised. A bare Python scalar — which carries no metadata — is
    normalised to a 0-d ``jax.Array``. Any non-numeric leaf raises
    ``TypeError`` at construction, naming the offending field.

    Lazy conversion at the compute boundary
    ---------------------------------------
    Navigation returns native leaves verbatim: ``record["x"]``,
    :attr:`~Record.children`, and :meth:`~Record.at_path` never convert.
    Conversion to ``jax.Array`` happens at the compute boundary — the JAX
    pytree flatten that ``jit`` / ``vmap`` / ``grad`` traverse,
    :meth:`to_vector`, and the single-field scalar shim — through a set-once
    per-leaf cache, so each leaf materialises at most once. A JAX transform
    therefore returns a record with bare ``jax.Array`` leaves (unflatten
    cannot rebuild native containers); structural transforms
    (:meth:`~Record.without` / :meth:`~Record.merge` / :meth:`~Record.replace`
    / :meth:`~Record.with_path_names`) and pickling reuse the native leaves
    verbatim, so native types survive them.

    Aliasing and mutation
    ---------------------
    Native leaves are stored **by reference**, exactly as a plain
    :class:`Record` stores opaque leaves. Mutating a passed-in container in
    place after construction therefore reaches the record — and, after the
    leaf has crossed a compute boundary, additionally desynchronises the
    cached converted array from the native view (navigation shows the
    mutation, compute uses the snapshot taken at first conversion). Records
    assume their data is not externally mutated mid-pipeline; no defensive
    copies are made.

    Equality, hashing, and lazy leaves
    ----------------------------------
    :meth:`~Record.__eq__`, :meth:`~Record.__hash__`, and content
    fingerprints compare converted values, so computing them on a record
    with lazy / disk-backed leaves forces materialisation on demand.

    The flat vector form
    --------------------
    Because every leaf is numeric, the whole value can be flattened into a
    single dense 1-D array. :meth:`to_vector` converts and ravels the leaves,
    in canonical leaf order (depth-first, insertion order), into a vector of
    length :attr:`vector_size`; :meth:`from_vector` rebuilds the record from
    such a vector (with bare ``jax.Array`` leaves — a reconstructed value has
    no native provenance to restore). Note that this is different from
    ``list(record.values())``, which returns the ordered list of native
    fields and is supported by any ``Record``.

    Single-field records as scalars
    --------------------------------
    A ``NumericRecord`` with exactly one field behaves like a thin wrapper
    around that field's value: ``float(r)``, ``int(r)``, ``bool(r)``,
    ``np.asarray(r)``, ``jnp.asarray(r)``, and the ``r.shape`` / ``r.dtype``
    / ``r.ndim`` attributes all forward to the sole leaf (the value
    coercions convert; the shape / dtype / ndim attributes read container
    metadata without materialising). A record with more than one field, or
    whose one field is itself a nested record, raises ``TypeError`` from
    these conversions, since unwrapping one field of several would be
    ambiguous; access a specific field explicitly in that case.

    Parameters
    ----------
    name : str
        The record's name — the required first positional argument, exactly
        as on :class:`Record`.
    _fields : Mapping, optional
        Fields as a positional mapping — an alternative to keyword ``**fields``
        (passing both raises). As on :class:`Record`, use it when a field name
        would collide with the ``event_template`` / ``name_is_auto`` keywords.
    **fields
        Named numeric values: a numeric array or container (``jax`` /
        ``numpy`` / ``xarray`` / ``pandas`` / registered backends), a numeric
        Python scalar, or a nested ``NumericRecord``. At least one field is
        required.
    name_is_auto : bool, optional
        ``True`` when *name* was derived by the producing operation rather
        than supplied by the user. Defaults to ``False``.
    event_template : NumericEventTemplate, optional
        The value's authoritative schema. When omitted it is inferred from the
        field data at construction; when supplied it is validated against the
        fields and stored. Either way it is fixed for the life of the record.

    Raises
    ------
    TypeError
        If any leaf is not a numeric array/container, a numeric scalar, or a
        nested ``NumericRecord``.
    ValueError
        If no fields are given, a field name contains ``/``, or both ``_fields``
        and keyword fields are passed (inherited from :class:`Record`).

    Notes
    -----
    Constructing ``NumericRecord(name, **fields)``, constructing
    ``Record(name, **fields)`` from all-numeric fields (which auto-promotes),
    and calling ``to_numeric()`` follow the same validation path and produce
    identical results; ``to_numeric()`` on an existing ``NumericRecord`` is
    the identity.

    The compute boundary presents a plain PyTree of arrays: the JAX pytree
    children are the converted leaves, so ``jit`` / ``vmap`` / ``grad`` see
    exactly the ProbPipe structure. As on :class:`Record`, the PyTree aux
    carries the ``(event_template, name, name_is_auto)`` triple, so the
    template and name survive a flatten/unflatten round-trip;
    :attr:`provenance`, :attr:`annotations`, and the native container types
    do not cross a JAX transform boundary.
    """

    __slots__ = ("_jax_cache", "_vector_size")

    def __init__(
        self,
        name: str,
        _fields: Mapping[str, ArrayLike | NumericRecord] | None = None,
        /,
        *,
        event_template: EventTemplate | None = None,
        name_is_auto: bool = False,
        _validate_leaves: bool = True,
        **fields: ArrayLike | NumericRecord,
    ):
        # Build the validated field dict *before* Record's __init__ runs, so
        # ``_fields`` is populated exactly once and the "constructed once,
        # never touched" invariant implied by ``__slots__`` + the
        # ``__setattr__`` guard holds.
        if _fields is not None:
            if fields:
                raise ValueError("Cannot pass both positional dict and keyword arguments")
            raw_inputs = _unflatten_paths(_fields)
        else:
            for field_name in fields:
                _check_no_path_sep(field_name)
            raw_inputs = dict(fields)
        # Materialise structural nesting (path-keyed construction) into nested
        # NumericRecords *before* leaf validation, so the numeric check happens
        # on the nested record that owns each leaf.
        raw_fields: dict[str, Any] = {}
        for field_name, value in raw_inputs.items():
            if isinstance(value, Mapping):
                # A mapping value is nested structure: materialise the child.
                sub_template: EventTemplate | None = None
                if event_template is not None:
                    child = event_template.children.get(field_name)
                    if isinstance(child, EventTemplate):
                        sub_template = child
                raw_fields[field_name] = type(self)(
                    field_name, value, event_template=sub_template, name_is_auto=True
                )
            else:
                raw_fields[field_name] = value
        validated = self._validate(raw_fields)
        super().__init__(
            name,
            validated,
            event_template=event_template,
            name_is_auto=name_is_auto,
            _validate_leaves=_validate_leaves,
        )
        # Cache vector_size, reading only container metadata (shapes) — a
        # lazy / disk-backed leaf is not materialised here.
        total = 0
        for val in self._tree.values():
            if isinstance(val, NumericRecord):
                total += val.vector_size
            else:
                backend = array_backend_for(val)
                shape = backend.event_shape(val) if backend is not None else val.shape
                total += int(prod(shape))
        object.__setattr__(self, "_vector_size", total)
        # The set-once converted-leaf cache backing the compute boundary.
        # Mutable by design (an internal memo, not value state); shallow
        # copies share it, so a conversion benefits every identity copy.
        object.__setattr__(self, "_jax_cache", {})

    @classmethod
    def _validate(cls, raw_fields: dict[str, Any]) -> dict[str, Any]:
        """Return the validated field dict, leaves in native form.

        Validation reads container metadata only (shape / dtype /
        registered-backend hooks) — values are not touched, so lazy leaves
        stay lazy. A bare Python scalar is normalised to a 0-d ``jax.Array``
        (it carries no metadata to preserve); every other numeric leaf is
        stored verbatim.

        Raises ``TypeError`` on non-numeric input with a message that
        names the offending field and its type.
        """
        cls_name = cls.__name__
        out: dict[str, Any] = {}
        for field_name, raw in raw_fields.items():
            if isinstance(raw, Record):
                if not isinstance(raw, NumericRecord):
                    raise TypeError(
                        f"{cls_name}: field {field_name!r} is a "
                        f"{type(raw).__name__}; nested records must be "
                        f"NumericRecord (got fields {raw.fields})"
                    )
                out[field_name] = raw
                continue
            if not _is_numeric_leaf(raw):
                raise TypeError(
                    f"{cls_name}: field {field_name!r} must be a numeric "
                    f"array, numeric scalar, or nested NumericRecord, got "
                    f"{type(raw).__name__}"
                )
            out[field_name] = jnp.asarray(raw) if isinstance(raw, _NUMERIC_SCALARS) else raw
        return out

    # -- The compute boundary: lazy, cached conversion -----------------------

    def _as_jax(self, field_name: str) -> jnp.ndarray:
        """The converted ``jax.Array`` for a top-level leaf field (cached).

        The single conversion point every compute boundary routes through.
        A leaf stored as a ``jax.Array`` (including a tracer inside a JAX
        transform) passes through untouched; a native container converts via
        its registered backend's ``to_jax`` (or ``jnp.asarray``) exactly
        once, with the result memoised in the set-once cache.
        """
        val = self._tree[field_name]
        if isinstance(val, jnp.ndarray):
            return val
        cache = self._jax_cache
        arr = cache.get(field_name)
        if arr is None:
            arr = _to_jax_array(val)
            cache[field_name] = arr
        return arr

    def _leaf_as_jax(self, path: str) -> jnp.ndarray:
        """The converted ``jax.Array`` for the leaf at *path* (any depth)."""
        head, sep, rest = path.partition(_PATH_SEP)
        if sep:
            return self._tree[head]._leaf_as_jax(rest)
        return self._as_jax(head)

    # -- 1-D vector conversion ----------------------------------------------

    @property
    def vector_size(self) -> int:
        """Length of this record's 1-D vector (``to_vector`` / ``from_vector``).

        The total number of scalar elements across all numeric leaves — the
        length of :meth:`to_vector`'s output. Computed from container
        metadata at construction; no values are materialised.
        """
        return self._vector_size

    def to_vector(self) -> jnp.ndarray:
        """Serialize to the dense 1-D vector of shape ``(vector_size,)``.

        The numeric 1-D serialization: the record's numeric leaves, visited in
        canonical leaf order (:meth:`~probpipe.core.named_tree.NamedTree.keys` —
        insertion order, depth-first into nested records), each converted to
        ``jax.Array`` (a compute boundary — lazy leaves materialise, once),
        raveled, and concatenated into one dense vector. The inverse is
        :meth:`from_vector`.

        This is distinct from ``list(record.values())``, which keeps each leaf
        whole and native (any type); ``to_vector`` ravels and concatenates the
        numeric leaves into a single dense vector.
        """
        leaves = [self._leaf_as_jax(key) for key in self.event_template]
        return jnp.concatenate([jnp.reshape(leaf, -1) for leaf in leaves])

    @classmethod
    def from_vector(cls, name: str, template: NumericEventTemplate, vec: Array) -> NumericRecord:
        """Reconstruct a single record from its dense 1-D vector.

        The value-level inverse of :meth:`to_vector`: splits *vec* into the
        template's per-field blocks, reshapes each to its ``ArraySpec`` shape
        in canonical leaf order, and returns a ``NumericRecord`` carrying
        *template* as its authoritative schema under the user-given *name*.
        The reconstructed leaves are bare ``jax.Array``\\ s — a flat vector
        carries no native container to restore.

        Parameters
        ----------
        name : str
            Name for the reconstructed record (user-given).
        template : NumericEventTemplate
            The flat layout supplying field names, shapes, and order.
        vec : Array
            A vector of shape ``(template.vector_size,)`` — one single
            (unbatched) value.

        Returns
        -------
        NumericRecord
            The reconstructed record, with ``record.to_vector()`` equal to
            *vec*.

        Raises
        ------
        TypeError
            If *vec* carries leading batch axes — batched reconstruction is
            the batch type's concern; use :meth:`NumericRecordArray.from_vector`
            for a batched matrix.
        ValueError
            If the vector length does not equal ``template.vector_size``.
        """
        vec = jnp.asarray(vec)
        if vec.ndim != 1:
            raise TypeError(
                f"NumericRecord.from_vector expects a 1-D vector (one value); "
                f"got shape {tuple(vec.shape)}. Reconstruct a batch with "
                f"NumericRecordArray.from_vector."
            )
        return _reconstruct_from_vector(name, template, vec, name_is_auto=False)

    def to_numeric(self) -> NumericRecord:
        """Return ``self`` — a ``NumericRecord`` is already numeric (identity)."""
        return self

    # -- Single-field scalar-like coercion ---------------------------------
    #
    # When a NumericRecord has exactly one numeric field, it behaves like
    # a thin wrapper around that field's value for the common coercion
    # paths: ``float()``, ``int()``, ``bool()``, ``np.asarray()``,
    # ``jnp.asarray()``, plus ``.shape`` / ``.dtype`` / ``.ndim``. The
    # shim lets workflow authors who return a single-field NumericRecord
    # from a ``@workflow_function`` keep using idiomatic expressions
    # like ``float(result)`` / ``result.shape`` / ``np.asarray(result)``
    # without a manual ``result["field"]`` unwrap at every callsite.
    #
    # The shim is intentionally narrow — only single-field records
    # qualify, and only the explicit coercion entry points are exposed
    # (no ``.mean()`` / ``.ravel()`` / arithmetic / slicing). Multi-field
    # records raise ``TypeError`` with a message pointing at explicit
    # field access, because silently unwrapping one field of many would
    # be ambiguous. The error is loud, not silent — but the shim only
    # covers the documented surface; for a multi-field-friendly flat
    # matrix view, empirical / bootstrap distributions expose
    # ``.flat_samples`` (an explicit ``(n, dim)`` accessor).
    # ---------------------------------------------------------------------

    def __reduce__(self):
        # Native leaves pickle themselves, so one branch suffices: the stored
        # field dict round-trips with native types intact at every nesting
        # level, and the authoritative template is threaded back so an
        # explicit (non-inferred) schema survives rather than being
        # re-inferred. The conversion cache is deliberately not serialized —
        # it is a memo, rebuilt on demand.
        return (
            _unpickle_numeric_record,
            (
                dict(self._tree),
                self._name,
                self._name_is_auto,
                self._provenance,
                self._event_template,
            ),
        )

    def _single_numeric_field(self) -> str:
        """Return the sole numeric field's name, or raise ``TypeError``."""
        if len(self._tree) != 1:
            raise TypeError(
                f"NumericRecord with {len(self._tree)} fields is not "
                f"scalar-like; access a specific field with "
                f"record['field_name'] first."
            )
        only = next(iter(self._tree))
        if isinstance(self._tree[only], NumericRecord):
            raise TypeError(
                "NumericRecord with a nested NumericRecord field is not "
                "scalar-like; access the nested record explicitly."
            )
        return only

    def __float__(self) -> float:
        return float(self._as_jax(self._single_numeric_field()))

    def __int__(self) -> int:
        return int(self._as_jax(self._single_numeric_field()))

    def __bool__(self) -> bool:
        return bool(self._as_jax(self._single_numeric_field()))

    def __array__(self, dtype=None, copy=None):
        leaf = self._as_jax(self._single_numeric_field())
        arr = np.asarray(leaf, dtype=dtype) if dtype is not None else np.asarray(leaf)
        return arr.copy() if copy else arr

    # JAX treats ``__jax_array__`` as the conversion hook for
    # ``jnp.asarray`` — without it, ``jnp.asarray(nr)`` goes through
    # ``__array__`` (numpy) and loses JAX tracing support.
    def __jax_array__(self):
        return self._as_jax(self._single_numeric_field())

    # Single-field shape / dtype / ndim — same "there's only one
    # thing in here; forward to it" ergonomic as the array-conversion
    # shims above, reading container metadata only (no materialisation).
    # Multi-field raises ``TypeError`` (loud, not silent).
    @property
    def shape(self) -> tuple[int, ...]:
        leaf = self._tree[self._single_numeric_field()]
        backend = array_backend_for(leaf)
        if backend is not None:
            return tuple(backend.event_shape(leaf))
        return tuple(getattr(leaf, "shape", ()))

    @property
    def dtype(self):
        """The sole leaf's dtype, or ``None`` when it has no single dtype.

        A registered backend reports the leaf's dtype (``None`` for a
        heterogeneous container such as a mixed-column frame); a bare
        array-like forwards its own ``.dtype``. A leaf that exposes no single
        dtype yields ``None`` rather than a stand-in.
        """
        leaf = self._tree[self._single_numeric_field()]
        backend = array_backend_for(leaf)
        if backend is not None:
            return backend.numpy_dtype(leaf)
        return getattr(leaf, "dtype", None)

    @property
    def ndim(self) -> int:
        return len(self.shape)


# ---------------------------------------------------------------------------
# 1-D vector reconstruction (value-level; the template supplies only layout)
# ---------------------------------------------------------------------------


def _value_treedef(
    template: NumericEventTemplate, batch_shape: tuple[int, ...]
) -> jax.tree_util.PyTreeDef:
    """PyTreeDef of the value :func:`_reconstruct_from_vector` builds.

    A throwaway ``NumericRecord`` / ``NumericRecordArray`` skeleton mirroring
    *template*; its structure (field names, nesting, ``batch_shape``, template)
    is all the treedef needs, so the placeholder leaves are cheap zero-stride
    broadcasts. Pairing this treedef with the real ordered leaves in
    ``tree_unflatten`` assembles the value in one place.
    """
    numeric_fill = jnp.zeros((), dtype=jnp.float32)

    def _build(tpl: NumericEventTemplate) -> NumericRecord:
        fields: dict[str, Any] = {}
        for name, spec in tpl.children.items():
            if isinstance(spec, NumericEventTemplate):
                fields[name] = _build(spec)
            else:
                fields[name] = jnp.broadcast_to(numeric_fill, (*batch_shape, *spec.shape))
        if batch_shape:
            from ._record_array import NumericRecordArray

            return NumericRecordArray(fields, batch_shape=batch_shape, template=tpl)
        # A template carries no name; the caller renames the reconstructed
        # value. Skip leaf validation: the placeholder fill is float32 and the
        # template may pin another dtype (int32 / bool) — this skeleton exists
        # only to capture the treedef structure, and the real leaves are cast
        # to the field dtype in ``_reconstruct_from_vector``.
        return Record(
            "value", fields, event_template=tpl, name_is_auto=True, _validate_leaves=False
        )

    return jax.tree_util.tree_structure(_build(template))


def _reconstruct_from_vector(
    name: str, template: NumericEventTemplate, vec: Array, *, name_is_auto: bool
) -> NumericRecord | Any:
    """Reconstruct a numeric value from its flat vector, under *name*.

    Splits *vec* along its trailing axis into *template*'s leaves (canonical
    leaf order), reshapes each to its event shape, and rebuilds the structured
    value: a single :class:`NumericRecord` when *vec* is 1-D, a batched
    :class:`~probpipe.NumericRecordArray` (``batch_shape == vec.shape[:-1]``)
    otherwise. This is the value-level machinery behind
    :meth:`NumericRecord.from_vector` / :meth:`NumericRecordArray.from_vector`;
    the template supplies only the leaf layout (shapes, order), never
    constructing the value itself.

    Raises
    ------
    ValueError
        If *vec* is 0-dimensional, or its trailing axis is not
        ``template.vector_size``.
    """
    vec = jnp.asarray(vec)
    if vec.ndim == 0:
        raise ValueError(
            f"from_vector: vec must have a trailing axis of size "
            f"vector_size={template.vector_size}; got a 0-d scalar."
        )
    if vec.shape[-1] != template.vector_size:
        raise ValueError(
            f"from_vector: vec trailing axis is {vec.shape[-1]}, expected "
            f"vector_size={template.vector_size}."
        )
    batch_shape = tuple(vec.shape[:-1])

    offset = 0
    leaves: list[Any] = []

    def _collect(tpl: NumericEventTemplate) -> None:
        nonlocal offset
        for spec in tpl.children.values():
            if isinstance(spec, NumericEventTemplate):
                _collect(spec)
            else:
                size = prod(spec.shape) if spec.shape else 1
                chunk = vec[..., offset : offset + size]
                offset += size
                leaf = jnp.reshape(chunk, (*batch_shape, *spec.shape))
                # Cast back to the field's declared dtype so a dtype-pinned
                # template (e.g. int32 / bool) round-trips faithfully — the flat
                # vector is typically float (``to_vector`` concatenates, which
                # promotes across mixed-dtype fields).
                if spec.dtype is not None:
                    leaf = leaf.astype(spec.dtype)
                leaves.append(leaf)

    _collect(template)
    value = jax.tree_util.tree_unflatten(_value_treedef(template, batch_shape), leaves)
    object.__setattr__(value, "_name", name)
    object.__setattr__(value, "_name_is_auto", name_is_auto)
    return value


# ---------------------------------------------------------------------------
# Pickle helper
# ---------------------------------------------------------------------------


def _unpickle_numeric_record(
    store: dict, name: str, name_is_auto: bool, provenance, event_template=None
) -> NumericRecord:
    # ``store`` holds the native leaves verbatim (they pickle themselves), so
    # reconstruction is ordinary validation-without-conversion. The threaded
    # template preserves an explicit (non-inferred) schema across the
    # round-trip; ``None`` falls back to inference.
    nr = NumericRecord(name, store, event_template=event_template)
    return nr._restore_identity(name_is_auto=name_is_auto, provenance=provenance)


# ---------------------------------------------------------------------------
# JAX PyTree registration — converting flatten, custom unflatten
# ---------------------------------------------------------------------------


def _numeric_record_flatten(v: NumericRecord) -> tuple[list, tuple[EventTemplate, str, bool]]:
    """Flatten NumericRecord for JAX pytree traversal, converting at the boundary.

    Children are emitted in the template's field order (matching
    :func:`~probpipe.core.record._record_flatten`), with each non-record leaf
    converted to ``jax.Array`` through the set-once cache — this is the
    compute boundary where native containers materialise. Nested
    ``NumericRecord`` children pass through whole; JAX recurses into them via
    their own registration. The static aux is the
    ``(event_template, name, name_is_auto)`` triple; provenance, annotations,
    and the native container types do not cross a JAX transform boundary.
    """
    children = [
        child if isinstance(child, Record) else v._as_jax(name)
        for name, child in ((n, v._tree[n]) for n in v._event_template.children)
    ]
    return children, (v._event_template, v._name, v._name_is_auto)


def _numeric_record_unflatten(
    aux: tuple[EventTemplate, str, bool], children: list
) -> NumericRecord:
    """Unflatten NumericRecord from JAX pytree traversal, threading the aux template."""
    template, name, name_is_auto = aux
    nr = NumericRecord(
        name,
        dict(zip(tuple(template.children), children)),
        event_template=template,
        _validate_leaves=False,
    )
    return nr._restore_identity(name_is_auto=name_is_auto, provenance=None)


jax.tree_util.register_pytree_node(
    NumericRecord, _numeric_record_flatten, _numeric_record_unflatten
)
