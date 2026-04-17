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

    Notes
    -----
    Construct from a list of Records with :meth:`RecordArray.stack`.
    Indexing is either integer (``arr[i]`` → single :class:`Record`) or
    field name (``arr["x"]`` → batched leaf array).
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
        # Subclass validation hook. Runs after sort / name-check so
        # subclasses (e.g. NumericRecordArray) see a canonicalised view
        # of the leaves. Raises from ``_validate_fields`` propagate.
        store = type(self)._validate_fields(store, batch_shape, template)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_batch_shape", batch_shape)
        object.__setattr__(self, "_template", template)
        object.__setattr__(self, "_fields", tuple(store.keys()))

    @classmethod
    def _validate_fields(
        cls,
        store: "OrderedDict[str, Any]",
        batch_shape: tuple[int, ...],
        template: RecordTemplate,
    ) -> "OrderedDict[str, Any]":
        """Hook for subclasses to validate / coerce leaves at construction.

        The base implementation is a no-op — ``RecordArray`` accepts any
        leaves, matching the permissive storage policy of ``Record``.
        Subclasses may return a new ``OrderedDict`` with coerced values
        or raise ``TypeError`` / ``ValueError`` on invalid input.
        """
        return store

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

    _record_cls: type = Record
    """Class used to materialise a single element via integer indexing.

    Overridden on :class:`NumericRecordArray` so element extraction
    returns a :class:`NumericRecord` (preserving the numeric guarantee).
    """

    def _get_record(self, index: int) -> Record:
        """Extract a single Record at a flat batch index."""
        nd_index = np.unravel_index(index, self._batch_shape)
        fields: dict[str, Any] = {}
        for name in self._store:
            fields[name] = self._store[name][nd_index]
        return self._record_cls(fields)

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
        if self._fields != other._fields:
            return False
        if self._template != other._template:
            return False
        for name, a in self._store.items():
            b = other._store[name]
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
    Construction validates that every leaf has a numeric dtype and
    shape ``(*batch_shape, *event_shape)`` matching the template, so
    pytree round-trips (``jax.tree.map``) cannot silently produce a
    ``NumericRecordArray`` with non-numeric or ill-shaped leaves.

    Each field has shape ``(*batch_shape, *event_shape)``.
    """

    __slots__ = ()

    # Integer indexing (``arr[i]``) returns a NumericRecord so the numeric
    # guarantee is preserved through slicing. Set at module load in
    # _numeric_record.py to avoid a circular import here.
    _record_cls: type = Record  # overridden below

    @classmethod
    def _validate_fields(
        cls,
        store: "OrderedDict[str, Any]",
        batch_shape: tuple[int, ...],
        template: RecordTemplate,
    ) -> "OrderedDict[str, Any]":
        """Require numeric dtype and matching event shape on every leaf.

        Raises ``TypeError`` if a leaf is non-numeric, ``ValueError`` if
        its shape is not ``(*batch_shape, *event_shape)`` for the
        corresponding template entry. Fields whose template spec is a
        nested ``RecordTemplate`` are forwarded as-is — the nested
        element is allowed to be a ``Record`` / ``NumericRecord`` /
        ``RecordArray`` and is validated at its own construction site.
        """
        out: "OrderedDict[str, Any]" = OrderedDict()
        for name, raw in store.items():
            spec = template[name]
            # Nested structure: skip numeric validation, let the nested
            # type enforce its own invariant.
            if isinstance(spec, RecordTemplate):
                out[name] = raw
                continue
            if not hasattr(raw, "dtype") or not hasattr(raw, "shape"):
                raise TypeError(
                    f"NumericRecordArray: field {name!r} must be a "
                    f"numeric array, got {type(raw).__name__}"
                )
            kind = getattr(raw.dtype, "kind", None)
            if kind not in {"b", "i", "u", "f", "c"}:
                raise TypeError(
                    f"NumericRecordArray: field {name!r} has non-numeric "
                    f"dtype {raw.dtype!r}"
                )
            event_shape = () if spec is None else tuple(spec)
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
                size = spec.flat_size
                chunk = flat[..., offset : offset + size]
                fields[name] = cls.unflatten(
                    chunk, template=spec, batch_shape=batch_shape,
                )
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
