"""RecordArray / NumericRecordArray — batched Record containers.

A ``RecordArray`` stores a batch of Records with consistent field structure.
Each field has shape ``(*batch_shape, *leaf_shape)``.  ``NumericRecordArray``
adds numeric operations: ``flatten``, ``unflatten``, ``mean``, ``var``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
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

    __slots__ = ("_store", "_batch_shape", "_template", "_fields")

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
        store = OrderedDict(sorted(fields.items()))
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_batch_shape", batch_shape)
        object.__setattr__(self, "_template", template)
        object.__setattr__(self, "_fields", tuple(store.keys()))

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
        return self._fields

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
        nd_index = np.unravel_index(index, self._batch_shape)
        fields: dict[str, Any] = {}
        for name in self._store:
            fields[name] = self._store[name][nd_index]
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
        cls_name = type(self).__name__
        return (
            f"{cls_name}(batch_shape={self._batch_shape}, "
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
                raise NotImplementedError(
                    "Nested RecordTemplate in NumericRecordArray.unflatten "
                    "is not yet supported"
                )
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

    # -- Reductions ---------------------------------------------------------

    def _reduce(
        self,
        fn: Callable[[jnp.ndarray, int], jnp.ndarray],
        axis: int = 0,
    ) -> NumericRecordArray | Any:
        """Apply a reduction function over a batch axis."""
        from ._numeric_record import NumericRecord

        new_batch = self._batch_shape[:axis] + self._batch_shape[axis + 1:]
        fields = {name: fn(self._store[name], axis) for name in self._store}
        if not new_batch:
            return NumericRecord(fields)
        return NumericRecordArray(
            fields, batch_shape=new_batch, template=self._template
        )

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
