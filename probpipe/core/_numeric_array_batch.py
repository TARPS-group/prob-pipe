"""NumericArrayBatch — the batch form of the numeric-array kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Self

from ._array_backend import _to_jax_array
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
        try:
            array = _to_jax_array(values)
        except Exception as exc:
            raise TypeError(
                f"NumericArrayBatch stores one array; {type(values).__name__} does not convert"
            ) from exc

        stored = tuple(array.shape)
        n_event = len(event_shape)
        if n_event and stored[len(stored) - n_event :] != event_shape:
            raise ValueError(
                f"the stored array ends with {stored[len(stored) - n_event :]} where its "
                f"elements declare the event shape {event_shape}"
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
        object.__setattr__(self, "_values", array)
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
        """The stored array, batch axes leading. Untracked."""
        return self._values

    @property
    def dtype(self) -> Any:
        return self._values.dtype

    def __repr__(self) -> str:
        return (
            f"NumericArrayBatch(batch_shape={self.batch_shape}, "
            f"levels={self.level_names}, event_shape={tuple(self.element_spec.shape)})"
        )

    # -- the concrete-storage seam ------------------------------------------

    def _element_at(self, index: tuple[int, ...], *, name: str) -> NumericArray:
        """The array at a fully-integer positional *index*, as a `NumericArray`.

        Indexing a stored array does not produce the element on its own — the
        element is the *tracked* value around it — so this is the materializing
        side of both rules :meth:`~probpipe.core._batch.Batch._element_at`
        states: it takes the derived *name*, marked auto, and inherits this
        batch's provenance.
        """
        return self._inherit_provenance(
            NumericArray(
                self._values[index],
                name=name,
                name_is_auto=True,
                spec=self.element_spec,
            )
        )

    def _sub_batch_at(self, index: tuple[int | slice, ...], *, spec: BatchSpec, name: str) -> Self:
        """A view over the same array, indexed on the batch axes as given.

        *index* addresses the leading axes only, so the event axes are untouched
        and array indexing yields a view rather than a copy. Built without
        ``__init__`` for the reason ``RecordBatch`` gives: the spec and the name
        are already decided, and re-deriving them from the view's own shape
        would lose the levels a dropped axis came from.
        """
        view = object.__new__(type(self))
        object.__setattr__(view, "_values", self._values[index])
        view._init_batch(spec, name=name, name_is_auto=True)
        return view
