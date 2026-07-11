"""NumericRecord — the numeric, all-array :class:`Record`.

A :class:`NumericRecord` is a :class:`~probpipe.Record` in which every field is a
numeric ``jax.Array``. It is the specialization of ``Record`` that gives access
to all of the standard, efficient array-based computation offered by JAX
(e.g., batched operations, JIT compilation, and automatic differentiation). A
plain ``Record`` may hold arbitrary Python objects; a ``NumericRecord`` narrows
that to numbers, and in return gains array-native features a general record
cannot offer.

What the numeric restriction buys
---------------------------------
Because every leaf is a JAX array, a ``NumericRecord`` is an ordinary JAX PyTree
of arrays. It passes through ``jit`` / ``vmap`` / ``grad`` unchanged, and JAX's
view of the tree coincides with ProbPipe's — unlike a general ``Record``, where
the two can differ. It also has a flat one-dimensional vector form
(:meth:`~NumericRecord.to_vector` and its structural inverse
:meth:`NumericEventTemplate.from_vector`), providing compatibility with inference
algorithms that assume parameters are represented as 1-D arrays. A
``NumericRecord`` stores a :class:`NumericEventTemplate` describing its structure.

Backends and round-tripping
---------------------------
A ``NumericRecord`` accepts numeric leaves from several array backends (``jax``,
``numpy``, ``xarray``, ``pandas``) and coerces each to a ``jax.Array`` at
construction. Metadata those backends carry that a bare array cannot — an
``xarray`` array's dimensions, coordinates, and attributes, or a ``pandas``
object's index, columns, and dtypes — is captured and stored so that
:meth:`~NumericRecord.to_native` can later restore each leaf to its original
backend type.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import ArrayLike
from ._array_backend import aux_for
from .event_template import EventTemplate, _check_no_path_sep, _PathSubtree, _unflatten_paths
from .record import Record, _record_flatten

__all__ = ["_NUMERIC_DTYPE_KINDS", "NumericRecord", "_is_numeric_leaf"]


# Scalar types accepted as numeric leaves. ``bool`` is intentionally
# included: JAX treats it as dtype ``bool_`` and numpy arrays of bools
# participate in arithmetic as 0/1.
_NUMERIC_SCALARS = (bool, int, float, complex, np.integer, np.floating, np.bool_)

# dtype.kind codes for numeric arrays: b=bool, i=int, u=uint, f=float,
# c=complex. Shared with ``NumericRecordArray._validate_fields`` so the
# two validation sites stay in lockstep.
_NUMERIC_DTYPE_KINDS = frozenset("biufc")


def _is_numeric_leaf(val: Any) -> bool:
    """True if *val* is a numeric array or numeric scalar.

    Rejects object-dtype arrays, string-like scalars, and opaque types
    that don't expose ``dtype`` / ``shape``. ``pandas.DataFrame``-like
    types whose elements are numeric are accepted via their ``.dtypes``
    summary.
    """
    if isinstance(val, (str, bytes)):
        return False
    if isinstance(val, _NUMERIC_SCALARS):
        return True
    if hasattr(val, "dtype") and hasattr(val, "shape"):
        kind = getattr(val.dtype, "kind", None)
        return kind in _NUMERIC_DTYPE_KINDS
    # ``pandas.DataFrame`` has ``.dtypes`` (per-column) but no ``.dtype``.
    # Accept when every column is numeric.
    dtypes = getattr(val, "dtypes", None)
    if dtypes is not None and hasattr(val, "shape"):
        try:
            kinds = {getattr(d, "kind", None) for d in dtypes}
        except TypeError:
            return False
        return bool(kinds) and kinds.issubset(_NUMERIC_DTYPE_KINDS)
    return False


class NumericRecord(Record):
    """A :class:`Record` whose fields are all arrays.

    A ``NumericRecord`` is the numeric specialization of :class:`Record`. It
    holds the same named, ordered, possibly-nested collection of fields, but
    constrains every field to be a numeric array. It inherits the full
    :class:`Record` interface — the leaf-keyed mapping, the tree navigation, the
    metadata (:attr:`~Record.name`, :attr:`~Record.provenance`), and the
    equality and hashing rules — and adds the array-only features described
    below.

    The numeric-leaf invariant
    --------------------------
    After construction, every leaf is a ``jax.Array`` and every interior node is
    a nested ``NumericRecord``. A leaf may be given as a numeric array from any
    supported backend (``jax.numpy``, ``numpy``, ``xarray.DataArray``, or a
    ``pandas.Series`` / ``DataFrame`` with a numeric dtype) or as a numeric
    Python scalar (``int``, ``float``, ``complex``, or ``bool``); each is coerced
    to a ``jax.Array`` so downstream code always sees a single, uniform array
    type. A ``bool`` value, or an array of ``bool`` dtype, counts as numeric —
    matching JAX and NumPy, where booleans are a valid array dtype. Any leaf that
    is not numeric, such as a string or an opaque Python object, raises
    ``TypeError`` at construction, naming the offending field.

    Backend metadata and ``to_native``
    ----------------------------------
    Coercing a field to a bare ``jax.Array`` discards any metadata its original
    backend carried — an ``xarray`` array's dimensions, coordinates, and
    attributes, or a ``pandas`` object's index, columns, and dtypes. That
    metadata is captured at construction, stored on the record keyed by field,
    and readable through :attr:`aux`. :meth:`to_native` reverses the conversion,
    restoring each field to its original backend type. Because restored ``xarray``
    or ``pandas`` leaves are no longer ``jax.Array``\\ s, :meth:`to_native`
    returns a plain :class:`Record` rather than a ``NumericRecord``.

    The flat vector form
    --------------------
    Because every leaf is numeric, the whole value can be flattened into a single
    dense 1-D array. :meth:`to_vector` ravels and concatenates the leaves, in
    canonical leaf order (depth-first, insertion order), into a vector of length
    :attr:`vector_size`; :meth:`NumericEventTemplate.from_vector` rebuilds the
    record from such a vector. Note that this is different from
    ``list(record.values())``, which returns the ordered list of fields and is
    supported by any ``Record``. :meth:`to_vector`, supported only by the numeric
    specialization, goes a step farther by raveling the fields into a single
    1-D array.

    Single-field records as scalars
    --------------------------------
    A ``NumericRecord`` with exactly one field behaves like a thin wrapper around
    that field's value: ``float(r)``, ``int(r)``, ``bool(r)``, ``np.asarray(r)``,
    ``jnp.asarray(r)``, and the ``r.shape`` / ``r.dtype`` / ``r.ndim`` attributes
    all forward to the sole leaf. This lets a workflow function that returns a
    single-field record be used in ordinary numeric expressions without
    unwrapping the field by hand. A record with more than one field, or whose one
    field is itself a nested record, raises ``TypeError`` from these conversions,
    since unwrapping one field of several would be ambiguous; access a specific
    field explicitly in that case.

    Parameters
    ----------
    _fields : Mapping, optional
        Fields as a positional mapping — an alternative to keyword ``**fields``
        (passing both raises). As on :class:`Record`, use it when a field name
        would collide with the ``name`` / ``event_template`` keywords.
    **fields
        Named numeric values: a numeric array (``jax`` / ``numpy`` / ``xarray`` /
        ``pandas`` with a numeric dtype), a numeric Python scalar, or a nested
        ``NumericRecord``. At least one field is required.
    name : str, optional
        Human-readable label for introspection / provenance. Defaults to a label
        derived from the field names.
    event_template : NumericEventTemplate, optional
        The value's authoritative schema. When omitted it is inferred from the
        field data at construction; when supplied it is validated against the
        fields and stored. Either way it is fixed for the life of the record.

    Raises
    ------
    TypeError
        If any leaf is not a numeric array, a numeric scalar, or a nested
        ``NumericRecord``.
    ValueError
        If no fields are given, a field name contains ``/``, or both ``_fields``
        and keyword fields are passed (inherited from :class:`Record`).

    Notes
    -----
    Constructing ``NumericRecord(**fields)`` and calling
    ``Record(**fields).to_numeric()`` follow the same validation and coercion
    path and produce identical results.

    Unlike a general :class:`Record`, whose JAX PyTree structure can be finer
    than its ProbPipe structure, a ``NumericRecord`` is a plain PyTree of arrays:
    the two views coincide, and it passes through ``jit`` / ``vmap`` / ``grad``
    unchanged. As on :class:`Record`, :attr:`name`, :attr:`provenance`, and
    :attr:`event_template` are runtime metadata and are not serialized into the
    PyTree aux.
    """

    __slots__ = ("_aux", "_vector_size")

    def __init__(
        self,
        _fields: Mapping[str, ArrayLike | NumericRecord] | None = None,
        /,
        *,
        name: str | None = None,
        event_template: EventTemplate | None = None,
        **fields: ArrayLike | NumericRecord,
    ):
        # Build the validated + coerced field dict *before* Record's
        # __init__ runs, so ``_fields`` is populated exactly once and the
        # "constructed once, never touched" invariant implied by
        # ``__slots__`` + the ``__setattr__`` guard holds.
        if _fields is not None:
            if fields:
                raise ValueError("Cannot pass both positional dict and keyword arguments")
            raw_inputs = _unflatten_paths(_fields)
        else:
            for field_name in fields:
                _check_no_path_sep(field_name)
            raw_inputs = dict(fields)
        # Materialise structural nesting (path-keyed construction) into nested
        # NumericRecords *before* leaf validation, so the numeric check and the
        # backend-aux capture happen on the nested record that owns each leaf —
        # keying aux by a "/"-path the storage tree cannot see would orphan it.
        raw_fields: dict[str, Any] = {}
        for field_name, value in raw_inputs.items():
            if isinstance(value, _PathSubtree):
                sub_template: EventTemplate | None = None
                if event_template is not None:
                    child = event_template.children.get(field_name)
                    if isinstance(child, EventTemplate):
                        sub_template = child
                raw_fields[field_name] = type(self)(value, event_template=sub_template)
            else:
                raw_fields[field_name] = value
        validated, aux = self._validate_and_coerce(raw_fields)
        super().__init__(validated, name=name, event_template=event_template)
        # Cache vector_size — leaves are immutable arrays after construction.
        total = 0
        for val in self._tree.values():
            if isinstance(val, NumericRecord):
                total += val.vector_size
            else:
                total += int(val.size)
        object.__setattr__(self, "_vector_size", total)
        # Aux is ``None`` if no field had a registered hook — keeps the
        # common all-jax case allocation-free and lets ``to_native``
        # short-circuit.
        object.__setattr__(self, "_aux", aux if aux else None)

    @classmethod
    def _validate_and_coerce(
        cls, raw_fields: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return ``(validated_fields, aux)`` for the raw fields.

        ``validated_fields`` has every leaf coerced to ``jnp.ndarray``
        (or kept as a nested :class:`NumericRecord`). ``aux`` captures
        backend-specific metadata for any field whose original leaf
        type is in the aux registry; the caller stores it on
        ``self._aux`` for :meth:`to_native` to consume.

        Raises ``TypeError`` on non-numeric input with a message that
        names the offending field and its type.
        """
        cls_name = cls.__name__
        out: dict[str, Any] = {}
        aux: dict[str, Any] = {}
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
            hooks = aux_for(raw)
            if hooks is not None:
                aux[field_name] = (hooks, hooks.capture(raw))
            out[field_name] = raw if isinstance(raw, jnp.ndarray) else jnp.asarray(raw)
        return out, aux

    # -- 1-D vector conversion ----------------------------------------------

    @property
    def vector_size(self) -> int:
        """Length of this record's 1-D vector (``to_vector`` / ``from_vector``).

        The total number of scalar elements across all numeric leaves — the
        length of :meth:`to_vector`'s output.
        """
        return self._vector_size

    def to_vector(self) -> jnp.ndarray:
        """Serialize to the dense 1-D vector of shape ``(vector_size,)``.

        Instance-level convenience for the numeric 1-D serialization whose
        structural definition lives on :meth:`NumericEventTemplate.to_vector`:
        delegates to this record's authoritative :attr:`~Record.event_template`
        (``nr.to_vector() == nr.event_template.to_vector(nr)``). Leaves are
        visited in canonical leaf order (insertion order, depth-first into nested
        records) and raveled before concatenation. The inverse,
        :meth:`NumericEventTemplate.from_vector`, reconstructs the record from
        such a vector.

        This is distinct from ``list(record.values())``, which keeps each leaf
        whole (any type); ``to_vector`` ravels and concatenates the numeric
        leaves into a single dense vector.
        """
        return self.event_template.to_vector(self)

    # -- Conversion back to native backends --------------------------------

    @property
    def aux(self) -> dict[str, Any] | None:
        """Captured backend metadata blobs, keyed by field name (or ``None``).

        Each entry is the opaque ``aux_blob`` returned by the registered
        ``capture`` hook for that field's original leaf type. Fields whose
        leaf type wasn't in the registry (plain numpy / jax / Python
        scalars) are absent.

        The hook pair is intentionally not exposed here — call
        :meth:`to_native` to materialise the original backend objects.
        """
        if self._aux is None:
            return None
        return {name: blob for name, (_hooks, blob) in self._aux.items()}

    def to_native(self) -> Record:
        """Restore each leaf to its original backend type, returning a :class:`Record`.

        Fields whose original leaf type was registered in
        :mod:`probpipe.core._array_backend` are restored via
        ``hooks.restore(jax_array, aux)``. Fields without captured aux
        pass through as their stored ``jax.Array``. Nested
        :class:`NumericRecord` fields recurse.

        The result is a permissive :class:`Record`, not a
        ``NumericRecord`` — restored xarray / pandas leaves are no
        longer ``jax.Array`` and would fail the numeric invariant.
        """
        fields: dict[str, Any] = {}
        aux = self._aux or {}
        for field_name, val in self._tree.items():
            if isinstance(val, NumericRecord):
                fields[field_name] = val.to_native()
                continue
            entry = aux.get(field_name)
            if entry is None:
                fields[field_name] = val
            else:
                hooks, blob = entry
                fields[field_name] = hooks.restore(val, blob)
        return Record(fields)

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
        return (
            _unpickle_numeric_record,
            (dict(self._tree), self._name, self._name_is_auto, self._provenance),
        )

    def _single_numeric_leaf(self):
        """Return the sole numeric leaf, or raise ``TypeError``."""
        if len(self._tree) != 1:
            raise TypeError(
                f"NumericRecord with {len(self._tree)} fields is not "
                f"scalar-like; access a specific field with "
                f"record['field_name'] first."
            )
        only = next(iter(self._tree.values()))
        if isinstance(only, NumericRecord):
            raise TypeError(
                "NumericRecord with a nested NumericRecord field is not "
                "scalar-like; access the nested record explicitly."
            )
        return only

    def __float__(self) -> float:
        return float(self._single_numeric_leaf())

    def __int__(self) -> int:
        return int(self._single_numeric_leaf())

    def __bool__(self) -> bool:
        return bool(self._single_numeric_leaf())

    def __array__(self, dtype=None, copy=None):
        leaf = self._single_numeric_leaf()
        arr = np.asarray(leaf, dtype=dtype) if dtype is not None else np.asarray(leaf)
        return arr.copy() if copy else arr

    # JAX treats ``__jax_array__`` as the conversion hook for
    # ``jnp.asarray`` — without it, ``jnp.asarray(nr)`` goes through
    # ``__array__`` (numpy) and loses JAX tracing support.
    def __jax_array__(self):
        return jnp.asarray(self._single_numeric_leaf())

    # Single-field shape / dtype / ndim — same "there's only one
    # thing in here; forward to it" ergonomic as the array-conversion
    # shims above. Multi-field raises ``TypeError`` (loud, not silent).
    @property
    def shape(self) -> tuple[int, ...]:
        leaf = self._single_numeric_leaf()
        return tuple(getattr(leaf, "shape", ()))

    @property
    def dtype(self):
        leaf = self._single_numeric_leaf()
        return getattr(leaf, "dtype", type(leaf))

    @property
    def ndim(self) -> int:
        leaf = self._single_numeric_leaf()
        return int(getattr(leaf, "ndim", 0))


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------


def _unpickle_numeric_record(
    store: dict, name: str, name_is_auto: bool, provenance
) -> NumericRecord:
    nr = NumericRecord(name=name, **store)
    return nr._restore_identity(name_is_auto=name_is_auto, provenance=provenance)


# ---------------------------------------------------------------------------
# JAX PyTree registration — reuse Record's flatten, custom unflatten
# ---------------------------------------------------------------------------


def _numeric_record_unflatten(aux: tuple[str, ...], children: list) -> NumericRecord:
    """Unflatten NumericRecord from JAX pytree traversal."""
    return NumericRecord(dict(zip(aux, children)))


jax.tree_util.register_pytree_node(NumericRecord, _record_flatten, _numeric_record_unflatten)
