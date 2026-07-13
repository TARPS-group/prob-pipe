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
(:meth:`~NumericRecord.to_vector` and its inverse
:meth:`~NumericRecord.from_vector`), providing compatibility with inference
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
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array, ArrayLike
from ._array_backend import aux_for
from .event_template import EventTemplate, NumericEventTemplate, _is_numeric_dtype
from .named_tree import _check_no_path_sep, _unflatten_paths
from .record import Record, _record_flatten

__all__ = ["NumericRecord", "_is_numeric_leaf"]


# Scalar types accepted as numeric leaves. ``bool`` is intentionally
# included: JAX treats it as dtype ``bool_`` and numpy arrays of bools
# participate in arithmetic as 0/1.
_NUMERIC_SCALARS = (bool, int, float, complex, np.integer, np.floating, np.bool_)


def _is_numeric_leaf(val: Any) -> bool:
    """True if *val* is a numeric array or numeric scalar.

    Rejects object-dtype arrays, string-like scalars, and opaque types
    that don't expose ``dtype`` / ``shape``. ``pandas.DataFrame``-like
    types whose elements are numeric are accepted via their ``.dtypes``
    summary. Dtype numericness is decided by the shared
    ``_is_numeric_dtype`` predicate, so this gate,
    ``NumericRecordArray._validate_fields``, and template inference agree
    on what counts as numeric.
    """
    if isinstance(val, (str, bytes)):
        return False
    if isinstance(val, _NUMERIC_SCALARS):
        return True
    if hasattr(val, "dtype") and hasattr(val, "shape"):
        return _is_numeric_dtype(val.dtype)
    # ``pandas.DataFrame`` has ``.dtypes`` (per-column) but no ``.dtype``.
    # Accept when every column is numeric.
    dtypes = getattr(val, "dtypes", None)
    if dtypes is not None and hasattr(val, "shape"):
        try:
            return len(dtypes) > 0 and all(_is_numeric_dtype(d) for d in dtypes)
        except TypeError:
            return False
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
    :attr:`vector_size`; :meth:`from_vector` rebuilds the
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
    name : str
        The record's name — the required first positional argument, exactly
        as on :class:`Record`.
    _fields : Mapping, optional
        Fields as a positional mapping — an alternative to keyword ``**fields``
        (passing both raises). As on :class:`Record`, use it when a field name
        would collide with the ``event_template`` / ``name_is_auto`` keywords.
    **fields
        Named numeric values: a numeric array (``jax`` / ``numpy`` / ``xarray`` /
        ``pandas`` with a numeric dtype), a numeric Python scalar, or a nested
        ``NumericRecord``. At least one field is required.
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
        If any leaf is not a numeric array, a numeric scalar, or a nested
        ``NumericRecord``.
    ValueError
        If no fields are given, a field name contains ``/``, or both ``_fields``
        and keyword fields are passed (inherited from :class:`Record`).

    Notes
    -----
    Constructing ``NumericRecord(name, **fields)``, constructing
    ``Record(name, **fields)`` from all-numeric fields (which auto-promotes),
    and calling ``to_numeric()`` follow the same validation and coercion path
    and produce identical results.

    Unlike a general :class:`Record`, whose JAX PyTree structure can be finer
    than its ProbPipe structure, a ``NumericRecord`` is a plain PyTree of arrays:
    the two views coincide, and it passes through ``jit`` / ``vmap`` / ``grad``
    unchanged. As on :class:`Record`, the PyTree aux carries the
    ``(event_template, name, name_is_auto)`` triple, so the template and name
    survive a flatten/unflatten round-trip; :attr:`provenance`,
    :attr:`annotations`, and backend (``xarray`` / ``pandas``) aux metadata
    do not cross a JAX transform boundary.
    """

    __slots__ = ("_aux", "_vector_size")

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
        validated, aux = self._validate_and_coerce(raw_fields)
        super().__init__(
            name,
            validated,
            event_template=event_template,
            name_is_auto=name_is_auto,
            _validate_leaves=_validate_leaves,
        )
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

        The numeric 1-D serialization: the record's numeric leaves, visited in
        canonical leaf order (:meth:`~probpipe.core.named_tree.NamedTree.keys` —
        insertion order, depth-first into nested records), each raveled and
        concatenated into one dense vector. The inverse is :meth:`from_vector`.

        This is distinct from ``list(record.values())``, which keeps each leaf
        whole (any type); ``to_vector`` ravels and concatenates the numeric
        leaves into a single dense vector.
        """
        leaves = [self[key] for key in self.event_template]
        return jnp.concatenate([jnp.reshape(leaf, -1) for leaf in leaves])

    @classmethod
    def from_vector(cls, name: str, template: NumericEventTemplate, vec: Array) -> NumericRecord:
        """Reconstruct a single record from its dense 1-D vector.

        The value-level inverse of :meth:`to_vector`: splits *vec* into the
        template's per-field blocks, reshapes each to its ``ArraySpec`` shape
        in canonical leaf order, and returns a ``NumericRecord`` carrying
        *template* as its authoritative schema under the user-given *name*.

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
        return Record(self._name, fields, name_is_auto=self._name_is_auto)

    # -- Immutable updates: carry backend aux across structural transforms --
    #
    # ``without`` / ``merge`` / ``replace`` / ``with_path_names`` rebuild the
    # record from its (mostly reused-by-reference) leaves. Backend aux lives
    # per ``NumericRecord`` level, keyed by field name, so a *rebuilt* level
    # would otherwise lose it: a top-level ``xarray`` / ``pandas`` leaf came
    # back bare from ``to_native`` after an edit, while a nested one — reused
    # verbatim as a whole child — survived. These overrides re-attach the aux
    # of every surviving leaf by object identity, at whatever level it now
    # sits, so the result is consistent regardless of nesting depth. A leaf
    # whose *value* changed (``replace`` with a new value, or ``map``) is a new
    # object, is not matched, and correctly loses its stale aux. A JAX pytree
    # round-trip still drops aux — see the class docstring.

    def without(self, *paths: str) -> Record:
        """See :meth:`Record.without`.

        Backend (``xarray`` / ``pandas``) aux for leaves that survive the edit
        is carried onto the result, so ``to_native`` still restores them.
        """
        return self._carry_aux(super().without(*paths), (self,))

    def merge(self, other: Record) -> Record:
        """See :meth:`Record.merge`.

        Backend aux from both operands' surviving leaves is carried onto the
        merged result.
        """
        return self._carry_aux(super().merge(other), (self, other))

    def replace(self, _updates: Mapping[str, Any] | None = None, /, **updates: Any) -> Record:
        """See :meth:`Record.replace`.

        Backend aux for untouched leaves is carried onto the result; a replaced
        leaf takes its new value's aux (captured fresh) or none.
        """
        return self._carry_aux(super().replace(_updates, **updates), (self,))

    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Record:
        """See :meth:`Record.with_path_names`.

        Backend aux follows each renamed leaf to its new key, at any nesting
        depth, so ``to_native`` still restores them.
        """
        return self._carry_aux(super().with_path_names(mapping, **kwargs), (self,))

    @staticmethod
    def _carry_aux(result: Record, sources: tuple[Record, ...]) -> Record:
        """Re-attach backend aux carried (by leaf identity) from *sources* onto *result*."""
        if not isinstance(result, NumericRecord):
            return result
        by_id: dict[int, tuple[Any, Any]] = {}
        for src in sources:
            if isinstance(src, NumericRecord):
                by_id.update(src._backend_aux_by_leaf_id())
        if by_id:
            result._reattach_backend_aux(by_id)
        return result

    def _backend_aux_by_leaf_id(self) -> dict[int, tuple[Any, Any]]:
        """Map ``id(leaf) -> (hooks, blob)`` across this record and its nested levels.

        Aux is stored per level, keyed by that level's field name; keying by
        the leaf object's identity instead lets a structural transform
        re-associate each surviving leaf's aux with its (possibly renamed or
        re-nested) position in the rebuilt tree.
        """
        by_id: dict[int, tuple[Any, Any]] = {}
        if self._aux:
            for field_name, entry in self._aux.items():
                leaf = self._tree.get(field_name)
                if leaf is not None:
                    by_id[id(leaf)] = entry
        for child in self._tree.values():
            if isinstance(child, NumericRecord):
                by_id.update(child._backend_aux_by_leaf_id())
        return by_id

    def _reattach_backend_aux(self, by_id: dict[int, tuple[Any, Any]]) -> None:
        """Recursively re-attach carried aux to this record and its nested levels.

        A level's own freshly captured aux (e.g. a value newly ``replace``d
        with a backend object) is kept; carried entries only *add* the aux for
        leaves reused verbatim from the source, matched by object identity. The
        keys-subset guard skips the write when nothing new is carried, so a
        subtree reused verbatim from the source is never mutated.
        """
        carried: dict[str, tuple[Any, Any]] = {}
        for field_name, leaf in self._tree.items():
            if isinstance(leaf, NumericRecord):
                leaf._reattach_backend_aux(by_id)
            else:
                entry = by_id.get(id(leaf))
                if entry is not None:
                    carried[field_name] = entry
        current = self._aux or {}
        if carried and not (set(carried) <= set(current)):
            object.__setattr__(self, "_aux", {**carried, **current})

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
        # A record carrying backend aux (xarray / pandas metadata) round-trips
        # through its native form: the backend objects pickle themselves and
        # aux is re-captured on load. This keeps ``to_native`` faithful across a
        # pickle round-trip without pickling the aux hook closures (which are
        # not stdlib-picklable). Aux-free records take the direct path over the
        # bare arrays; a nested aux-carrying child is handled by its own
        # ``__reduce__`` when this record is pickled through that path.
        if self._aux:
            native = dict(self.to_native())
            return (
                _unpickle_numeric_record_native,
                (
                    native,
                    self._name,
                    self._name_is_auto,
                    self._provenance,
                    self._event_template,
                ),
            )
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
        # A template carries no name; the caller renames the reconstructed value.
        return Record("value", fields, event_template=tpl, name_is_auto=True)

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
                leaves.append(jnp.reshape(chunk, (*batch_shape, *spec.shape)))

    _collect(template)
    value = jax.tree_util.tree_unflatten(_value_treedef(template, batch_shape), leaves)
    object.__setattr__(value, "_name", name)
    object.__setattr__(value, "_name_is_auto", name_is_auto)
    return value


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------


def _unpickle_numeric_record(
    store: dict, name: str, name_is_auto: bool, provenance, event_template=None
) -> NumericRecord:
    # ``event_template`` defaults to None for in-flight pickles predating the
    # template being serialized; the threaded template preserves the exact
    # schema instead of re-inferring a weaker one.
    nr = NumericRecord(name, store, event_template=event_template)
    return nr._restore_identity(name_is_auto=name_is_auto, provenance=provenance)


def _unpickle_numeric_record_native(
    native: dict, name: str, name_is_auto: bool, provenance, event_template=None
) -> NumericRecord:
    # Inverse of the aux-carrying ``__reduce__`` branch: ``native`` is the
    # path-keyed mapping of restored backend leaves, so rebuilding a
    # ``NumericRecord`` from it re-captures the backend aux at every level.
    # The authoritative template is threaded back so an explicit (non-inferred)
    # schema survives the round-trip rather than being re-inferred.
    nr = NumericRecord(name, native, event_template=event_template)
    return nr._restore_identity(name_is_auto=name_is_auto, provenance=provenance)


# ---------------------------------------------------------------------------
# JAX PyTree registration — reuse Record's flatten, custom unflatten
# ---------------------------------------------------------------------------


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


jax.tree_util.register_pytree_node(NumericRecord, _record_flatten, _numeric_record_unflatten)
