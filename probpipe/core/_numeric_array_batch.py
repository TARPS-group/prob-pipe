"""NumericArrayBatch — the batch form of the numeric-array kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Self

import numpy as np

from ._array_backend import _event_shape_of, _is_numeric_leaf, _numpy_dtype_of, _take_at
from ._batch import Batch, BatchSpec, _axis_groups_for
from ._numeric_array import NumericArray
from .event_template import NumericArraySpec
from .provenance import Provenance

__all__ = ["NumericArrayBatch"]


class NumericArrayBatch(Batch[NumericArray]):
    """A batch of numeric arrays sharing one :class:`NumericArraySpec`.

    Storage is a single array with the batch axes leading — the split
    :class:`~probpipe.RecordBatch` uses, with one column instead of many. This
    is where a `draw` level lives for an array-valued law.

    Parameters
    ----------
    values : array-like
        One array holding every element, shaped ``(*batch_shape, *event_shape)``.
        Stored verbatim in its native form, as a :class:`NumericArray`'s value
        is, so a lazy or disk-backed column is not materialised to be batched.
    level_names : str or iterable of str
        One name per level, outermost first; a single string names one level.
    element_spec : NumericArraySpec
        What every element satisfies. Its ``shape`` is the event shape, so it is
        what splits the stored array's axes into batch and event.
    axis_groups : iterable of iterable of int, optional
        The axis sizes each level holds, in order, tiling ``batch_shape``.
        Defaults to one axis per level.
    name : str, optional
        The batch's name; defaults to ``"numericarraybatch"``, marked auto.
    name_is_auto : bool, default False
        Whether *name* is auto-derived rather than user-given.
    provenance : Provenance, optional
        How this batch was produced.

    Raises
    ------
    TypeError
        If *element_spec* is not a :class:`NumericArraySpec`, or *values* does
        not convert to an array.
    TypeError
        Also if the stored array's dtype is one the declared element does not
        admit — a cross-kind conversion.
    ValueError
        If *values* has fewer axes than the event shape it must end with, which
        would leave no batch axis; if its trailing axes are not that event
        shape; if a declared dimension is symbolic, which gives the event shape
        no size to split by; or if *axis_groups* does not tile ``batch_shape``.

    Notes
    -----
    Selection yields the element kind, as it does for every batch:
    ``batch[i]`` is a :class:`NumericArray`. It materializes, so each element
    takes the derived name and inherits this batch's lineage.
    """

    _values: Any

    __slots__ = ("_values",)

    def __init__(
        self,
        values: Any,
        level_names: str | Iterable[str],
        *,
        element_spec: NumericArraySpec,
        axis_groups: Iterable[Iterable[int]] | None = None,
        name: str | None = None,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
    ) -> None:
        if not isinstance(element_spec, NumericArraySpec):
            raise TypeError(
                f"NumericArrayBatch element_spec must be a NumericArraySpec, "
                f"got {type(element_spec).__name__}"
            )
        event_shape = tuple(element_spec.shape)
        if any(not isinstance(axis, int) for axis in event_shape):
            raise ValueError(
                f"a symbolic dimension gives the event shape no size to split the stored "
                f"axes by; bind {element_spec.shape} with with_dims before batching"
            )
        if not _is_numeric_leaf(values):
            raise TypeError(
                f"NumericArrayBatch stores one array; {type(values).__name__} is not a numeric leaf"
            )
        stored = _event_shape_of(values)
        n_event = len(event_shape)
        if n_event and stored[len(stored) - n_event :] != event_shape:
            raise ValueError(
                f"the stored array ends with {stored[len(stored) - n_event :]} where its "
                f"elements declare the event shape {event_shape}"
            )
        # The batch asserts *element_spec* of every element, so a column whose
        # dtype the declaration does not admit makes that assertion false. Caught
        # here rather than at the first selection, which is where NumericArray
        # would otherwise raise — one layer too late to name the batch.
        dtype = _numpy_dtype_of(values)
        if element_spec.dtype is not None and dtype is not None:
            if not np.can_cast(dtype, element_spec.dtype, casting="same_kind"):
                raise TypeError(
                    f"the stored array has dtype {dtype}, which the declared element "
                    f"{np.dtype(element_spec.dtype)} does not admit; a widening or a "
                    f"within-kind narrowing passes, a cross-kind conversion does not"
                )
        batch_shape = stored[: len(stored) - n_event] if n_event else stored
        if not batch_shape:
            raise ValueError(
                f"a batch has at least one batch axis, and an array of shape {stored} over "
                f"elements of event shape {event_shape} leaves none; a single value is a "
                f"NumericArray"
            )

        names = (level_names,) if isinstance(level_names, str) else tuple(level_names)
        groups = _axis_groups_for(batch_shape, names, axis_groups, kind="NumericArrayBatch")
        if name is None:
            name, name_is_auto = "numericarraybatch", True
        object.__setattr__(self, "_values", values)
        self._init_batch(
            BatchSpec(element_spec, groups, names),
            name=name,
            name_is_auto=name_is_auto,
            provenance=provenance,
        )

    # -- what it holds ------------------------------------------------------

    @property
    def element_spec(self) -> NumericArraySpec:
        """What every element satisfies — a view on :attr:`spec`."""
        spec = self.spec.element_spec
        assert isinstance(spec, NumericArraySpec)
        return spec

    @property
    def values(self) -> Any:
        """The stored array, batch axes leading, in native form. Untracked."""
        return self._values

    @property
    def dtype(self) -> Any:
        return self.element_spec.dtype

    def __repr__(self) -> str:
        return (
            f"NumericArrayBatch(batch_shape={self.batch_shape}, "
            f"levels={self.level_names}, event_shape={tuple(self.element_spec.shape)})"
        )

    # -- the concrete-storage seam ------------------------------------------

    def _element_at(self, index: tuple[int, ...], *, name: str) -> NumericArray:
        """The array at a fully-integer positional *index*, as a `NumericArray`.

        Selection is positional and goes through the value's backend, since
        ``[]`` reads labels rather than positions on a ``pandas`` container.

        Indexing a stored array does not produce the element on its own — the
        element is the *tracked* value around it — so this is the materializing
        side of both rules :meth:`~probpipe.core._batch.Batch._element_at`
        states: it takes the derived *name*, marked auto, and inherits this
        batch's provenance.
        """
        return self._inherit_provenance(
            NumericArray(
                _take_at(self._values, index),
                name=name,
                name_is_auto=True,
                spec=self.element_spec,
            )
        )

    def _sub_batch_at(self, index: tuple[int | slice, ...], *, spec: BatchSpec, name: str) -> Self:
        """A view over the same array, indexed on the batch axes as given.

        *index* addresses the leading axes only, so the event axes are untouched
        and selection yields a view rather than a copy. It goes through the
        value's backend, since ``[]`` is not positional on every container. Built without
        ``__init__`` for the reason ``RecordBatch`` gives: the spec and the name
        are already decided, and re-deriving them from the view's own shape
        would lose the levels a dropped axis came from.
        """
        view = object.__new__(self._view_type)
        object.__setattr__(view, "_values", _take_at(self._values, index))
        view._init_batch(spec, name=name, name_is_auto=True)
        return view
