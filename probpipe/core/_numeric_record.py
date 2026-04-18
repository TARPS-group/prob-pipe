"""NumericRecord — Record subclass where every leaf is a numeric array.

Adds ``flatten`` / ``unflatten`` / ``flat_size`` for 1-D serialisation.
Construction validates that every leaf is a numeric array (or nested
``NumericRecord``), coerces plain Python scalars and numpy arrays to
``jnp.ndarray`` so downstream JAX code receives a uniform array type,
and raises a clear error otherwise.

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
    that don't expose ``dtype`` / ``shape``.
    """
    if isinstance(val, (str, bytes)):
        return False
    if isinstance(val, _NUMERIC_SCALARS):
        return True
    if hasattr(val, "dtype") and hasattr(val, "shape"):
        kind = getattr(val.dtype, "kind", None)
        return kind in _NUMERIC_DTYPE_KINDS
    return False


class NumericRecord(Record):
    """``Record`` where every leaf is a numeric array.

    Adds :meth:`flatten` / :meth:`unflatten` / :attr:`flat_size` for
    serialising the record to / from a flat 1-D vector. Construction
    validates that every leaf is a numeric value (or a nested
    :class:`NumericRecord`) and coerces scalar / numpy leaves to
    ``jnp.ndarray`` so downstream code sees a uniform JAX array type.

    Parameters
    ----------
    **fields
        Named values. Every leaf must be a numeric array (``jax.numpy``,
        ``numpy``), a numeric Python scalar (``int``, ``float``,
        ``complex``, ``bool``), or a nested ``NumericRecord``. Non-numeric
        values raise ``TypeError`` at construction time.

    Notes
    -----
    Validation and coercion happen *before* the underlying ``Record`` is
    constructed, so ``_store`` is populated exactly once and remains
    immutable from the moment ``__init__`` returns — consistent with the
    ``__slots__`` + ``__setattr__`` guard on the base class.
    """

    __slots__ = ("_flat_size",)

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
        validated = self._validate_and_coerce(raw_fields)
        super().__init__(validated, name=name)
        # Cache flat_size — leaves are immutable arrays after construction.
        total = 0
        for val in self._store.values():
            if isinstance(val, NumericRecord):
                total += val.flat_size
            else:
                total += int(val.size)
        object.__setattr__(self, "_flat_size", total)

    @classmethod
    def _validate_and_coerce(
        cls, raw_fields: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a new dict with every leaf validated and (if numeric)
        coerced to ``jnp.ndarray``. Raises ``TypeError`` on non-numeric
        input with a message that names the offending field and its type.
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
            out[field_name] = raw if isinstance(raw, jnp.ndarray) else jnp.asarray(raw)
        return out

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
        """Convert a ``Record`` to ``NumericRecord``, validating leaves."""
        raw_fields: dict[str, Any] = {}
        for field_name, val in record._store.items():
            if isinstance(val, Record):
                raw_fields[field_name] = cls.from_record(val)
            else:
                raw_fields[field_name] = val
        return cls(raw_fields)


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
