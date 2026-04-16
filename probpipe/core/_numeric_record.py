"""NumericRecord — Record subclass where every leaf is a numeric array.

Adds ``unflatten()`` for 1-D deserialisation (``flatten()`` is inherited
from ``Record``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import ArrayLike
from .record import Record, RecordTemplate, _record_flatten

__all__ = ["NumericRecord"]


def _is_numeric(val: Any) -> bool:
    """Check if a value is a numeric array or numeric scalar."""
    # bool before int — bool is a subclass of int
    if isinstance(val, bool):
        return False
    if isinstance(val, (np.ndarray, jnp.ndarray)):
        return True
    if isinstance(val, (int, float, complex, np.integer, np.floating)):
        return True
    if isinstance(val, (str, bytes)):
        return False
    return hasattr(val, "shape") and hasattr(val, "dtype")


class NumericRecord(Record):
    """Record where every leaf is a numeric array.

    Inherits ``flatten()`` from ``Record`` and adds
    ``unflatten()`` for 1-D deserialisation.
    Construction validates that all leaves are numeric.

    Parameters
    ----------
    **fields
        Named values.  Every leaf must be a numeric array (numpy, JAX,
        or scalar).  Nested ``NumericRecord`` objects are supported.
    """

    __slots__ = ()

    def __init__(
        self,
        _dict: dict[str, ArrayLike | NumericRecord] | None = None,
        /,
        *,
        name: str | None = None,
        **fields: ArrayLike | NumericRecord,
    ):
        super().__init__(_dict, name=name, **fields)
        self._validate_numeric()

    def _validate_numeric(self) -> None:
        """Ensure all leaves are numeric arrays."""
        for field_name in self._store:
            raw = self._store[field_name]
            if isinstance(raw, Record):
                if not isinstance(raw, NumericRecord):
                    raise TypeError(
                        f"Field {field_name!r}: nested Record must be "
                        f"NumericRecord in a NumericRecord, got "
                        f"{type(raw).__name__}"
                    )
            elif not _is_numeric(raw):
                raise TypeError(
                    f"Field {field_name!r}: NumericRecord requires numeric "
                    f"values, got {type(raw).__name__}"
                )

    # -- Unflatten ------------------------------------------------------------

    @classmethod
    def unflatten(
        cls,
        flat: jnp.ndarray,
        *,
        template: RecordTemplate,
    ) -> NumericRecord:
        """Reconstruct a NumericRecord from a flat array.

        Parameters
        ----------
        flat : array
            1-D array of concatenated scalars.
        template : RecordTemplate
            Provides field names and shapes.
        """
        from .._utils import prod

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
                    f"Cannot unflatten opaque field {field_name!r} "
                    f"from flat array"
                )
            else:
                size = prod(spec) if spec else 1
                fields[field_name] = flat[offset : offset + size].reshape(spec)
                offset += size

        return cls(fields)

    @classmethod
    def from_record(cls, record: Record) -> NumericRecord:
        """Convert a Record to NumericRecord (validates all leaves numeric)."""
        raw_fields = {}
        for field_name in record._store:
            val = record._store[field_name]
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
