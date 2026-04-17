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

from .._utils import prod
from ..custom_types import ArrayLike
from .record import Record, RecordTemplate, _record_flatten

__all__ = ["NumericRecord"]


# Scalar types accepted as numeric leaves. ``bool`` is intentionally
# included: JAX treats it as dtype ``bool_`` and numpy arrays of bools
# participate in arithmetic as 0/1.
_NUMERIC_SCALARS = (bool, int, float, complex, np.integer, np.floating, np.bool_)


def _is_numeric_leaf(val: Any) -> bool:
    """True if *val* is a numeric array or numeric scalar."""
    if isinstance(val, (np.ndarray, jnp.ndarray)):
        return True
    if isinstance(val, _NUMERIC_SCALARS):
        return True
    if isinstance(val, (str, bytes)):
        return False
    # Anything else that quacks like an array: require both shape and dtype
    # attributes, and reject non-numeric dtypes (e.g. numpy object arrays).
    if hasattr(val, "shape") and hasattr(val, "dtype"):
        dtype = val.dtype
        kind = getattr(dtype, "kind", None)
        return kind in {"b", "i", "u", "f", "c"}
    return False


def _coerce_leaf(val: Any, *, field_name: str, cls_name: str) -> jnp.ndarray:
    """Convert a validated numeric leaf to ``jnp.ndarray``.

    Validation should already have happened via :func:`_is_numeric_leaf`;
    this function simply performs the actual conversion with a useful
    error message if it fails.
    """
    if isinstance(val, jnp.ndarray):
        return val
    try:
        return jnp.asarray(val)
    except Exception as exc:  # pragma: no cover — defensive
        raise TypeError(
            f"{cls_name}: field {field_name!r} passed numeric-leaf check "
            f"but could not be converted to jnp.ndarray ({type(val).__name__}: {exc})"
        ) from exc


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
        super().__init__(_dict, name=name, **fields)
        # Validate + coerce leaves in-place in _store so downstream
        # accessors return jnp.ndarray uniformly.
        store = self._store
        cls_name = type(self).__name__
        for field_name, raw in list(store.items()):
            if isinstance(raw, Record):
                if not isinstance(raw, NumericRecord):
                    raise TypeError(
                        f"{cls_name}: field {field_name!r} is a "
                        f"{type(raw).__name__}; nested records must be "
                        f"NumericRecord (got fields {raw.fields})"
                    )
                continue
            if not _is_numeric_leaf(raw):
                raise TypeError(
                    f"{cls_name}: field {field_name!r} must be a numeric "
                    f"array, numeric scalar, or nested NumericRecord, got "
                    f"{type(raw).__name__}"
                )
            store[field_name] = _coerce_leaf(
                raw, field_name=field_name, cls_name=cls_name,
            )
        # Cache flat_size — leaves are immutable arrays after construction.
        total = 0
        for val in store.values():
            if isinstance(val, NumericRecord):
                total += val.flat_size
            else:
                total += int(val.size)
        object.__setattr__(self, "_flat_size", total)

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
            if isinstance(spec, RecordTemplate):
                size = spec.flat_size
                child_flat = flat[offset : offset + size]
                fields[field_name] = cls.unflatten(child_flat, template=spec)
                offset += size
            elif spec is None:
                raise TypeError(
                    f"Cannot unflatten opaque field {field_name!r} from a "
                    f"flat array: template has shape=None (non-numeric "
                    f"leaf). Opaque fields are only supported on Record, "
                    f"not NumericRecord."
                )
            else:
                size = prod(spec) if spec else 1
                fields[field_name] = flat[offset : offset + size].reshape(spec)
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
