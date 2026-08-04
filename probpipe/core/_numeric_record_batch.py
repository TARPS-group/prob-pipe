"""NumericRecordBatch — the all-numeric specialization of :class:`RecordBatch`.

Every field of a :class:`NumericRecordBatch` is a numeric array, so the whole
batch is a bare pytree of arrays whose leading axes are the batch axes: it passes
through ``jit`` / ``vmap`` / ``grad`` unchanged, and it gains the batched flat
layout, :meth:`NumericRecordBatch.to_vector` and
:meth:`NumericRecordBatch.from_vector`.

The split mirrors the single-record side, where :class:`~probpipe.Record` and
:class:`~probpipe.NumericRecord` live apart for the same reason: the numeric
specialization adds a flat-vector contract of its own, and nothing in the general
case depends on it.

See design III.3.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array
from ._record_batch import (
    RecordBatch,
    _record_batch_flatten,
    _record_element_spec,
    _unflatten_with,
)
from .event_template import ArraySpec, EventTemplate, NumericEventTemplate, RecordSpec
from .provenance import Provenance

__all__ = ["NumericRecordBatch"]


class NumericRecordBatch(RecordBatch):
    """A :class:`RecordBatch` whose every column is a numeric array.

    The all-numeric specialization, carrying a ``NumericEventTemplate``: a bare
    pytree of arrays whose leading axes are the ``batch_shape``, so it passes
    through ``jit`` / ``vmap`` / ``grad`` unchanged. It adds the batched flat
    layout, :meth:`to_vector` and :meth:`from_vector`.

    Construction is that of :class:`RecordBatch`, narrowed: *element_spec* must
    describe an all-numeric element, and every column must carry a numeric dtype.

    Raises
    ------
    TypeError
        If *element_spec* does not describe an all-numeric element, or a column
        is not a numeric array.
    """

    __slots__ = ()

    def __init__(
        self,
        fields: Mapping[str, Any],
        level_names: str | Iterable[str],
        *,
        element_spec: RecordSpec | EventTemplate,
        axis_groups: Iterable[Iterable[int]] | None = None,
        name: str | None = None,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
    ) -> None:
        template = _record_element_spec(element_spec, kind=type(self).__name__).event_template
        if not isinstance(template, NumericEventTemplate):
            raise TypeError(
                f"{type(self).__name__} describes an all-numeric element, so its element_spec "
                f"carries a NumericEventTemplate; got one over {type(template).__name__} with "
                f"fields {list(template.keys())}"
            )
        super().__init__(
            fields,
            level_names,
            element_spec=element_spec,
            axis_groups=axis_groups,
            name=name,
            name_is_auto=name_is_auto,
            provenance=provenance,
        )

    # ``element_spec`` is not overridden: it already reports the stored
    # ``RecordSpec``, and only ``event_template`` has anything narrower to say.

    @property
    def event_template(self) -> NumericEventTemplate:
        """The numeric structure of one element — a view on :attr:`element_spec`."""
        template = self.element_spec.event_template
        assert isinstance(template, NumericEventTemplate)  # narrowed at construction
        return template

    # ``_check_columns`` is not overridden: every field of a numeric template is
    # an ``ArraySpec``, so the base already checks each column for a numeric
    # dtype the declaration admits.

    # -- single-field coercion ----------------------------------------------
    #
    # With exactly one field, a batch is a thin wrapper around that field's
    # values, so the array-conversion and introspection entry points forward to
    # them. Deliberately narrower than the single-record shim: ``float`` / ``int``
    # / ``bool`` are absent, because a batch of values is not one scalar however
    # few fields it has.

    def _sole_field(self) -> Any:
        """The one field's values, or a refusal naming what to do instead."""
        if len(self._columns) != 1:
            raise TypeError(
                f"a {type(self).__name__} of {len(self._columns)} fields is not array-like; "
                f"read one field first, as batch['field']"
            )
        return next(iter(self._columns.values()))

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """The sole field's values as a numpy array."""
        leaf = self._sole_field()
        array = np.asarray(leaf, dtype=dtype) if dtype is not None else np.asarray(leaf)
        return array.copy() if copy else array

    def __jax_array__(self) -> Array:
        """The sole field's values as a ``jax.Array``."""
        return jnp.asarray(self._sole_field())

    @property
    def shape(self) -> tuple[int, ...]:
        """The sole field's full shape, ``(*batch_shape, *event_shape)``."""
        return tuple(self._sole_field().shape)

    @property
    def dtype(self) -> Any:
        """The sole field's dtype."""
        return self._sole_field().dtype

    @property
    def ndim(self) -> int:
        """The rank of the sole field's values, batch axes included."""
        return int(self._sole_field().ndim)

    # -- flat layout --------------------------------------------------------

    def to_vector(self) -> Array:
        """Every element's flat vector, stacked.

        Returns
        -------
        Array
            Shape ``(*batch_shape, vector_size)``: one raveled vector per
            element, fields visited in the template's canonical order, each
            field's event axes raveled and the fields concatenated. The inverse
            is :meth:`from_vector`.

        Notes
        -----
        Distinct from reading the columns, which keeps each field whole and its
        event axes intact. The batch axes are left as they are: a multi-level
        batch keeps its levels as the leading axes of the result, so the flat
        dimension is always the last one and the levels read outermost-first.
        """
        batch_shape = self.batch_shape
        template = self.event_template
        leaf_shapes = template.leaf_shapes
        return jnp.concatenate(
            [
                # The field's own flat width rather than ``-1``: an empty batch
                # axis leaves nothing for ``-1`` to be inferred from.
                jnp.reshape(
                    jnp.asarray(self._columns[key]),
                    (*batch_shape, int(np.prod(leaf_shapes[key], dtype=int))),
                )
                for key in template
            ],
            axis=-1,
        )

    @classmethod
    def from_vector(
        cls,
        name: str,
        template: NumericEventTemplate,
        vec: Array,
        *,
        level_names: str | Iterable[str],
        axis_groups: Iterable[Iterable[int]] | None = None,
    ) -> Self:
        """Rebuild a batch from its elements' flat vectors, inverting :meth:`to_vector`.

        Parameters
        ----------
        name : str
            The reconstructed batch's name (user-given).
        template : NumericEventTemplate
            The flat layout: field names, event shapes, and canonical order.
        vec : Array
            Shape ``(*batch_shape, vector_size)`` — the trailing axis is the flat
            dimension, and every leading axis is a batch axis.
        level_names : str or iterable of str
            One name per level of the reconstructed batch, outermost first; a
            single string names a single level. Required for the reason
            :meth:`RecordBatch.stack` states, and plural because *vec* may carry
            several batch axes: naming them is how a multi-level batch round-trips.
        axis_groups : iterable of iterable of int, optional
            The axis sizes each level holds, as for the constructor. Defaults to
            one axis per level.

        Returns
        -------
        NumericRecordBatch
            The batch, satisfying ``batch.to_vector() == vec``.

        Raises
        ------
        TypeError
            If *vec* has no batch axis — reconstruct a single value with
            ``NumericRecord.from_vector``.
        ValueError
            If the trailing axis is not ``template.vector_size``, or if the level
            names do not account for *vec*'s leading axes.

        Examples
        --------
        A two-level batch round-trips when both levels are named:

        >>> import jax.numpy as jnp
        >>> from probpipe import EventTemplate
        >>> template = EventTemplate(x=(2,))
        >>> batch = NumericRecordBatch({"x": jnp.zeros((4, 5, 2))}, ("chain", "draw"),
        ...                            element_spec=template)
        >>> rebuilt = NumericRecordBatch.from_vector(
        ...     "post", template, batch.to_vector(), level_names=("chain", "draw"))
        >>> rebuilt.batch_shape
        (4, 5)
        """
        vec = jnp.asarray(vec)
        if vec.ndim < 2:
            raise TypeError(
                f"{cls.__name__}.from_vector takes a batched matrix, shaped "
                f"(*batch_shape, vector_size); got shape {tuple(vec.shape)}. Reconstruct a "
                f"single value with NumericRecord.from_vector"
            )
        if vec.shape[-1] != template.vector_size:
            raise ValueError(
                f"{cls.__name__}.from_vector: the trailing axis is {vec.shape[-1]}, expected "
                f"{template.vector_size} for this template"
            )
        batch_shape = tuple(vec.shape[:-1])
        columns: dict[str, Any] = {}
        offset = 0
        for key, event_shape in template.leaf_shapes.items():
            size = int(np.prod(event_shape, dtype=int))
            block = jnp.reshape(vec[..., offset : offset + size], (*batch_shape, *event_shape))
            # Concatenating promoted the fields to one dtype, so a field that
            # declares its own is cast back to it — otherwise the reconstruction
            # contradicts the very template it was rebuilt from.
            declared = template[key]
            if isinstance(declared, ArraySpec) and declared.dtype is not None:
                block = block.astype(declared.dtype)
            columns[key] = block
            offset += size
        return cls(
            columns,
            level_names,
            element_spec=template,
            axis_groups=axis_groups,
            name=name,
        )


jax.tree_util.register_pytree_node(
    NumericRecordBatch, _record_batch_flatten, _unflatten_with(NumericRecordBatch)
)
