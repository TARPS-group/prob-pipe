"""NumericRecord — Record subclass where every leaf is a ``jax.Array``.

Adds ``flatten`` / ``unflatten`` / ``flat_size`` for 1-D serialisation.
Construction validates that every leaf is a numeric value (numeric
array, numeric scalar, or nested ``NumericRecord``) and coerces each
to ``jnp.ndarray`` so the post-construction invariant is "every leaf
is a ``jax.Array``" (or a nested ``NumericRecord``).

Per-field metadata that ``jnp.asarray`` would drop — ``xarray`` dims /
coords / attrs, ``pandas`` index / columns / dtypes — is captured via
the registry in :mod:`probpipe.core._array_backend` and stored on the
new instance. :meth:`NumericRecord.to_native` reverses the conversion,
restoring each leaf to its original backend type. Direct
``NumericRecord(**fields)`` and ``Record(**fields).to_numeric()``
follow the same code path and produce identical results.

Bool handling
-------------
Python ``bool`` and arrays with ``bool`` dtype are treated as numeric
leaves (consistent with JAX / NumPy, where ``bool`` is a valid array
dtype that participates in arithmetic as ``0`` / ``1``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import ArrayLike
from ._array_backend import aux_for
from .record import Record, RecordTemplate, _record_flatten, _spec_size

__all__ = ["NumericRecord", "_is_numeric_leaf", "_NUMERIC_DTYPE_KINDS"]


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
    """``Record`` where every leaf is a ``jax.Array``.

    Adds :meth:`flatten` / :meth:`unflatten` / :attr:`flat_size` for
    serialising the record to / from a flat 1-D vector. Construction
    validates that every leaf is a numeric value (or a nested
    :class:`NumericRecord`) and coerces scalar / numpy / xarray /
    pandas leaves to ``jnp.ndarray`` so downstream code sees a uniform
    JAX array type. Backend-specific metadata (xarray dims / coords /
    attrs, pandas index / columns / dtypes) is captured via the aux
    registry in :mod:`probpipe.core._array_backend` and stored on the
    instance; :meth:`to_native` reverses the conversion.

    Parameters
    ----------
    **fields
        Named values. Every leaf must be a numeric array (``jax.numpy``,
        ``numpy``, ``xarray.DataArray``, ``pandas.Series`` /
        ``DataFrame`` with numeric dtype), a numeric Python scalar
        (``int``, ``float``, ``complex``, ``bool``), or a nested
        ``NumericRecord``. Non-numeric values raise ``TypeError`` at
        construction time.

    Notes
    -----
    ``NumericRecord(**fields)`` and ``Record(**fields).to_numeric()``
    are semantically identical — both consult the aux registry to
    capture metadata, both coerce leaves via ``jnp.asarray``, both
    raise ``TypeError`` on non-coercible leaves.

    Validation and coercion happen *before* the underlying ``Record`` is
    constructed, so ``_store`` is populated exactly once and remains
    immutable from the moment ``__init__`` returns — consistent with the
    ``__slots__`` + ``__setattr__`` guard on the base class.
    """

    __slots__ = ("_flat_size", "_aux")

    def __init__(
        self,
        _dict: dict[str, ArrayLike | NumericRecord] | None = None,
        /,
        *,
        name: str | None = None,
        **fields: ArrayLike | NumericRecord,
    ):
        # Build the validated + coerced field dict *before* Record's
        # __init__ runs, so ``_store`` is populated exactly once and the
        # "constructed once, never touched" invariant implied by
        # ``__slots__`` + the ``__setattr__`` guard holds.
        if _dict is not None:
            if fields:
                raise ValueError(
                    "Cannot pass both positional dict and keyword arguments"
                )
            raw_fields = _dict
        else:
            raw_fields = fields
        validated, aux = self._validate_and_coerce(raw_fields)
        super().__init__(validated, name=name)
        # Cache flat_size — leaves are immutable arrays after construction.
        total = 0
        for val in self._store.values():
            if isinstance(val, NumericRecord):
                total += val.flat_size
            else:
                total += int(val.size)
        object.__setattr__(self, "_flat_size", total)
        # Aux is ``None`` if no field had a registered hook — keeps the
        # common all-jax case allocation-free and lets ``to_native``
        # short-circuit.
        object.__setattr__(self, "_aux", aux if aux else None)

    @classmethod
    def _validate_and_coerce(
        cls, raw_fields: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return ``(validated_store, aux)`` for the raw fields.

        ``validated_store`` has every leaf coerced to ``jnp.ndarray``
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

    # -- Flat-array conversion ----------------------------------------------

    @property
    def flat_size(self) -> int:
        """Total number of scalar elements across all numeric leaves."""
        return self._flat_size

    def flatten(self) -> jnp.ndarray:
        """Concatenate all leaf arrays into a single 1-D vector.

        Fields are traversed in sorted order; nested ``NumericRecord``
        are traversed depth-first. Each leaf is raveled before
        concatenation.
        """
        parts: list[jnp.ndarray] = []
        for val in self._store.values():
            if isinstance(val, NumericRecord):
                parts.append(val.flatten())
            else:
                parts.append(jnp.ravel(val))
        return jnp.concatenate(parts)

    @classmethod
    def unflatten(
        cls,
        flat: jnp.ndarray,
        *,
        template: RecordTemplate,
    ) -> NumericRecord:
        """Reconstruct a ``NumericRecord`` from a flat array.

        Parameters
        ----------
        flat : array
            1-D array of concatenated scalars.
        template : RecordTemplate
            Provides field names and shapes for reconstruction.
        """
        fields: dict[str, jnp.ndarray | NumericRecord] = {}
        offset = 0

        for field_name in template.fields:
            spec = template[field_name]
            size = _spec_size(spec)
            chunk = flat[offset : offset + size]
            if isinstance(spec, RecordTemplate):
                fields[field_name] = cls.unflatten(chunk, template=spec)
            else:
                fields[field_name] = chunk.reshape(spec)
            offset += size

        return cls(fields)

    @classmethod
    def from_record(cls, record: Record) -> NumericRecord:
        """Convert a ``Record`` to ``NumericRecord``, validating leaves.

        Equivalent to ``record.to_numeric()``; both paths consult the
        aux registry, coerce every leaf via ``jnp.asarray``, and raise
        ``TypeError`` on non-coercible leaves. Nested ``Record``
        children recurse, preserving structure.
        """
        return cls({
            field_name: cls.from_record(val) if isinstance(val, Record) else val
            for field_name, val in record._store.items()
        })

    # -- Conversion back to native backends --------------------------------

    @property
    def aux(self) -> dict[str, Any] | None:
        """Captured backend metadata, keyed by field name (or ``None``).

        Each entry is an ``(AuxHooks, aux_blob)`` pair captured from the
        original leaf at construction. Fields whose leaf type wasn't in
        the registry (plain numpy / jax / Python scalars) are absent.
        """
        return self._aux

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
        for field_name, val in self._store.items():
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
    # ``jnp.asarray()``. The shim lets workflow authors who return a
    # single-field NumericRecord from a ``@workflow_function`` keep using
    # idiomatic expressions like ``float(result)`` / ``np.asarray(result)``
    # without a manual ``result["field"]`` unwrap at every callsite.
    #
    # The shim is intentionally narrow — only single-field records
    # qualify, and only scalar-like coercions are exposed. Multi-field
    # records raise ``TypeError`` with a message pointing at explicit
    # field access, because silently unwrapping one field of many would
    # be ambiguous.
    # ---------------------------------------------------------------------

    def _single_numeric_leaf(self):
        """Return the sole numeric leaf, or raise ``TypeError``."""
        if len(self._store) != 1:
            raise TypeError(
                f"NumericRecord with {len(self._store)} fields is not "
                f"scalar-like; access a specific field with "
                f"record['field_name'] first."
            )
        only = next(iter(self._store.values()))
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

    # Single-field shape / dtype / ndim — the same "there's only one
    # thing in here; forward to it" ergonomic as the arithmetic and
    # array-conversion shims above.
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
# JAX PyTree registration — reuse Record's flatten, custom unflatten
# ---------------------------------------------------------------------------


def _numeric_record_unflatten(
    aux: tuple[str, ...], children: list
) -> NumericRecord:
    """Unflatten NumericRecord from JAX pytree traversal."""
    return NumericRecord(dict(zip(aux, children)))


jax.tree_util.register_pytree_node(
    NumericRecord, _record_flatten, _numeric_record_unflatten
)
