"""RecordArray / NumericRecordArray — batched Record containers.

A ``RecordArray`` stores a batch of Records with consistent field structure.
Each field has shape ``(*batch_shape, *leaf_shape)``.  ``NumericRecordArray``
adds numeric operations: ``flatten``, ``unflatten``, ``mean``, ``var``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .._utils import prod
from ..custom_types import ArrayLike
from .record import Record, RecordTemplate

__all__ = ["RecordArray", "NumericRecordArray"]


class RecordArray:
    """Batch of Records with consistent field structure.

    Each field stores values with shape ``(*batch_shape, *leaf_shape)``.

    Parameters
    ----------
    batch_shape : tuple of int
        Shape of the batch dimensions.
    template : RecordTemplate
        Structural description of each element.
    **fields
        Named values, each with shape ``(*batch_shape, *leaf_shape)``.
    """

    __slots__ = ("_store", "_batch_shape", "_template")

    def __init__(
        self,
        _dict: dict[str, Any] | None = None,
        /,
        *,
        batch_shape: tuple[int, ...],
        template: RecordTemplate,
        **fields: Any,
    ):
        if _dict is not None:
            if fields:
                raise ValueError(
                    "Cannot pass both positional dict and keyword arguments"
                )
            fields = _dict
        if not fields:
            raise ValueError("RecordArray requires at least one field")
        if set(fields.keys()) != set(template.fields):
            raise ValueError(
                f"Field names {sorted(fields)} do not match template "
                f"fields {sorted(template.fields)}"
            )
        object.__setattr__(
            self, "_store", OrderedDict(sorted(fields.items()))
        )
        object.__setattr__(self, "_batch_shape", batch_shape)
        object.__setattr__(self, "_template", template)

    # -- Immutability -------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("RecordArray is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("RecordArray is immutable")

    # -- Properties ---------------------------------------------------------

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Shape of the batch dimensions."""
        return self._batch_shape

    @property
    def template(self) -> RecordTemplate:
        """Structural description of each element."""
        return self._template

    @property
    def fields(self) -> tuple[str, ...]:
        """Field names in sorted order."""
        return tuple(self._store.keys())

    # -- Field access -------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        store = object.__getattribute__(self, "_store")
        if name in store:
            return store[name]
        raise AttributeError(f"RecordArray has no field {name!r}")

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, str):
            if key not in self._store:
                raise KeyError(key)
            return self._store[key]
        if isinstance(key, (int, np.integer)):
            return self._get_record(int(key))
        raise TypeError(
            f"key must be str or int, got {type(key).__name__}"
        )

    def _get_record(self, index: int) -> Record:
        """Extract a single Record at a flat batch index."""
        # Convert flat index to multi-dimensional index
        nd_index = np.unravel_index(index, self._batch_shape)
        fields: dict[str, Any] = {}
        for name in self._store:
            val = self._store[name]
            spec = self._template[name]
            if isinstance(spec, RecordTemplate):
                # Nested: extract sub-RecordArray then index
                sub_batch = self._batch_shape
                sub_ra = RecordArray(
                    {sub_name: val[sub_name] for sub_name in spec.fields},
                    batch_shape=sub_batch,
                    template=spec,
                )
                fields[name] = sub_ra[index]
            else:
                fields[name] = val[nd_index]
        return Record(fields)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __len__(self) -> int:
        return prod(self._batch_shape)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def keys(self) -> Iterator[str]:
        """Iterate over field names."""
        return iter(self._store)

    def values(self) -> Iterator[Any]:
        """Iterate over field values (batched)."""
        for name in self._store:
            yield self._store[name]

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over (name, batched_value) pairs."""
        for name in self._store:
            yield name, self._store[name]

    # -- Factory methods ----------------------------------------------------

    @classmethod
    def stack(cls, records: list[Record], *, template: RecordTemplate | None = None) -> RecordArray:
        """Stack a list of Records into a RecordArray with batch_shape=(n,).

        Parameters
        ----------
        records : list of Record
            Records with consistent field structure.
        template : RecordTemplate, optional
            If not provided, inferred from the first record.
        """
        if not records:
            raise ValueError("Cannot stack empty list of Records")
        if template is None:
            template = RecordTemplate.from_record(records[0])
        fields: dict[str, Any] = {}
        for name in template.fields:
            field_vals = [r[name] for r in records]
            fields[name] = jnp.stack(field_vals, axis=0)
        return cls(fields, batch_shape=(len(records),), template=template)

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        field_parts = []
        for name in self._store:
            val = self._store[name]
            if hasattr(val, "shape"):
                field_parts.append(f"{name}=array(shape={val.shape})")
            else:
                field_parts.append(f"{name}=...")
        return (
            f"RecordArray(batch_shape={self._batch_shape}, "
            f"{', '.join(field_parts)})"
        )


# ---------------------------------------------------------------------------
# NumericRecordArray
# ---------------------------------------------------------------------------


class NumericRecordArray(RecordArray):
    """Batch of NumericRecords — all leaves are numeric arrays.

    Adds ``flatten``/``unflatten``, ``mean``, ``var`` operations.

    Each field has shape ``(*batch_shape, *event_shape)``.
    """

    __slots__ = ()

    # -- Flatten / unflatten ------------------------------------------------

    def flatten(self) -> jnp.ndarray:
        """Flatten event dimensions into a single trailing axis.

        Returns array of shape ``(*batch_shape, flat_event_size)``.
        """
        n_batch = len(self._batch_shape)
        parts = []
        for name in self._store:
            val = self._store[name]
            # Reshape to (*batch_shape, field_event_size)
            event_shape = val.shape[n_batch:]
            field_size = prod(event_shape)
            new_shape = self._batch_shape + (field_size,)
            parts.append(jnp.reshape(val, new_shape))
        return jnp.concatenate(parts, axis=-1)

    @classmethod
    def unflatten(
        cls,
        flat: jnp.ndarray,
        *,
        template: RecordTemplate,
        batch_shape: tuple[int, ...] | None = None,
    ) -> NumericRecordArray:
        """Reconstruct from a flat array.

        Parameters
        ----------
        flat : array
            Shape ``(*batch_shape, flat_event_size)``.
        template : RecordTemplate
            Structural description providing field names and event shapes.
        batch_shape : tuple of int, optional
            If not provided, inferred as ``flat.shape[:-1]``.
        """
        if batch_shape is None:
            batch_shape = flat.shape[:-1]

        fields: dict[str, jnp.ndarray] = {}
        offset = 0
        for name in template.fields:
            spec = template[name]
            if isinstance(spec, RecordTemplate):
                size = spec.flat_size
                chunk = flat[..., offset : offset + size]
                # Recursively unflatten nested templates
                sub_fields = cls._unflatten_nested(
                    chunk, spec, batch_shape
                )
                # Store nested fields with dotted access pattern
                # For now, store as a sub-dict that gets wrapped
                fields[name] = sub_fields
                offset += size
            elif spec is None:
                raise TypeError(
                    f"Cannot unflatten opaque field {name!r}"
                )
            else:
                size = prod(spec) if spec else 1
                chunk = flat[..., offset : offset + size]
                fields[name] = jnp.reshape(chunk, batch_shape + spec)
                offset += size

        return cls(fields, batch_shape=batch_shape, template=template)

    @staticmethod
    def _unflatten_nested(
        flat_chunk: jnp.ndarray,
        template: RecordTemplate,
        batch_shape: tuple[int, ...],
    ) -> jnp.ndarray:
        """Unflatten a nested RecordTemplate chunk into a sub-RecordArray.

        For now, returns the reshaped arrays as a dict; nested
        RecordArray support will come in a future phase.
        """
        # TODO: return a nested NumericRecordArray once nesting is needed
        raise NotImplementedError(
            "Nested RecordTemplate in NumericRecordArray.unflatten "
            "is not yet supported"
        )

    # -- Reductions ---------------------------------------------------------

    def mean(self, axis: int = 0) -> Any:
        """Mean over a batch axis.

        Parameters
        ----------
        axis : int
            Batch axis to reduce.  Must be in ``range(len(batch_shape))``.

        Returns
        -------
        NumericRecordArray or NumericRecord
            If the result has no batch dimensions left, returns a
            ``NumericRecord``.  Otherwise returns a
            ``NumericRecordArray`` with one fewer batch dimension.
        """
        from ._numeric_record import NumericRecord

        new_batch = self._batch_shape[:axis] + self._batch_shape[axis + 1 :]
        fields: dict[str, jnp.ndarray] = {}
        for name in self._store:
            fields[name] = jnp.mean(self._store[name], axis=axis)

        if not new_batch:
            return NumericRecord(fields)
        return NumericRecordArray(
            fields, batch_shape=new_batch, template=self._template
        )

    def var(self, axis: int = 0) -> Any:
        """Variance over a batch axis.

        Parameters
        ----------
        axis : int
            Batch axis to reduce.

        Returns
        -------
        NumericRecordArray or NumericRecord
        """
        from ._numeric_record import NumericRecord

        new_batch = self._batch_shape[:axis] + self._batch_shape[axis + 1 :]
        fields: dict[str, jnp.ndarray] = {}
        for name in self._store:
            fields[name] = jnp.var(self._store[name], axis=axis)

        if not new_batch:
            return NumericRecord(fields)
        return NumericRecordArray(
            fields, batch_shape=new_batch, template=self._template
        )

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        field_parts = []
        for name in self._store:
            val = self._store[name]
            field_parts.append(f"{name}=array(shape={val.shape})")
        return (
            f"NumericRecordArray(batch_shape={self._batch_shape}, "
            f"{', '.join(field_parts)})"
        )


# ---------------------------------------------------------------------------
# JAX PyTree registration
# ---------------------------------------------------------------------------


def _record_array_flatten(
    ra: RecordArray,
) -> tuple[list, tuple[tuple[str, ...], tuple[int, ...], RecordTemplate]]:
    """Flatten RecordArray for JAX pytree."""
    children = [ra._store[name] for name in ra._store]
    aux = (ra.fields, ra._batch_shape, ra._template)
    return children, aux


def _record_array_unflatten(
    aux: tuple[tuple[str, ...], tuple[int, ...], RecordTemplate],
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
    aux: tuple[tuple[str, ...], tuple[int, ...], RecordTemplate],
    children: list,
) -> NumericRecordArray:
    """Unflatten NumericRecordArray from JAX pytree."""
    field_names, batch_shape, template = aux
    return NumericRecordArray(
        dict(zip(field_names, children)),
        batch_shape=batch_shape,
        template=template,
    )


jax.tree_util.register_pytree_node(
    RecordArray, _record_array_flatten, _record_array_unflatten
)
jax.tree_util.register_pytree_node(
    NumericRecordArray, _record_array_flatten, _numeric_record_array_unflatten
)
