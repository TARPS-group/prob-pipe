"""Object-array storage for the batch forms of values that do not stack natively.

An ``ArraySpec`` value batches natively — an array with the batch axes leading —
so no class is needed for it. A callable and an opaque object have no such form:
there is nothing to stack them *into*. :class:`_ObjectBatch` supplies the
storage those two batch forms share, a numpy object array, leaving each public
class to say only what its elements are and which spec they satisfy.

The object array earns its place by answering the storage contract
:class:`~probpipe.core._batch.Batch` states rather than by holding arrays:
numpy's basic indexing returns a **view** over the same objects, so a
sub-batch shares its parent's store, and it honors a descending or stepped
slice in the order given, which the derived names of a view are stated in.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Self

import numpy as np

from ._batch import Batch, BatchSpec
from .event_template import ValueSpec
from .provenance import Provenance

__all__ = ["_ObjectBatch"]


class _ObjectBatch[E](Batch[E]):
    """A :class:`Batch` storing its elements in a numpy object array.

    Parameters
    ----------
    elements : numpy.ndarray or sequence
        The elements, as an object array of any shape or a flat sequence. A
        nested sequence is not unpacked: build the array to state a shape of
        more than one axis, since what nesting means for an arbitrary Python
        object is the caller's to decide.
    level_names : str or sequence of str
        One name per level, outermost first; a single string names a single
        level. There is no default, deliberately — see *Notes*.
    axis_groups : sequence of sequence of int, optional
        The axes each level holds. Defaults to one axis per level, which
        requires as many names as ``elements`` has axes; a level spanning
        several axes is stated explicitly.

    Raises
    ------
    TypeError
        If ``elements`` is neither an ndarray nor a sequence.
    ValueError
        If ``elements`` is empty, or if ``axis_groups`` is omitted and the
        number of names does not match the number of axes.

    Notes
    -----
    A level name is required rather than defaulted because a batch's levels are
    named so that operations can align operands by meaning: a placeholder would
    read as meaning something while naming nothing, which is the same reason
    :class:`~probpipe.core._batch.Batch` refuses to resolve a clash by
    suffixing. The caller that mints a level knows what it means.
    """

    __slots__ = ("_store",)

    def __init__(
        self,
        elements: np.ndarray | Sequence[E],
        level_names: str | Iterable[str],
        *,
        element_spec: ValueSpec,
        axis_groups: Iterable[Iterable[int]] | None = None,
        name: str | None = None,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
    ) -> None:
        store = _as_object_array(elements, kind=type(self).__name__)
        names = (level_names,) if isinstance(level_names, str) else tuple(level_names)
        groups = _axis_groups_for(store.shape, names, axis_groups, kind=type(self).__name__)

        object.__setattr__(self, "_store", store)
        self._init_batch(
            BatchSpec(element_spec, groups, names),
            name=name if name is not None else type(self).__name__.lower(),
            name_is_auto=name is None or name_is_auto,
            provenance=provenance,
        )

    # -- the storage seam ---------------------------------------------------

    def _element_at(self, index: tuple[int, ...], *, name: str) -> E:
        """The stored object at *index*, as it was given.

        *name* is unused, which is the whole of the identity rule for a batch
        that **stores** its elements: the object handed back is the caller's own,
        so whatever identity it arrived with is what it keeps. A tracked element
        takes a derived name where a batch *materializes* one per index — a row
        of columnar storage has no identity until it is built — but a callable
        placed here by name already means something, and renaming it to its
        position would lose that and hand back a copy besides. The batch remains
        the one place the position is recorded.
        """
        return self._store[index]

    def _sub_batch_at(self, index: tuple[int | slice, ...], *, spec: BatchSpec, name: str) -> Self:
        """A view over the same store, indexed as given.

        numpy basic indexing returns a view, so the selection shares its
        parent's objects and is presented in the order *index* states, a
        descending slice included. Built without ``__init__``, since the spec and
        the name are already decided and re-deriving them from the view's own
        shape would lose the levels a dropped axis came from.
        """
        view = type(self).__new__(type(self))
        object.__setattr__(view, "_store", self._store[index])
        view._init_batch(spec, name=name, name_is_auto=True)
        return view


def _as_object_array(elements: np.ndarray | Sequence[Any], *, kind: str) -> np.ndarray:
    """*elements* as an object array, without unpacking what it holds.

    ``np.asarray`` would look inside each element and stack anything array-like,
    turning a batch of two arrays into one 2-d numeric array. Allocating empty
    and assigning keeps each object whole, whatever it is.
    """
    if isinstance(elements, np.ndarray):
        if elements.dtype != object:
            raise TypeError(
                f"{kind} stores objects, so an ndarray of elements must have dtype=object; "
                f"got dtype={elements.dtype}"
            )
        store = elements
    else:
        try:
            flat = list(elements)
        except TypeError:
            raise TypeError(
                f"{kind} takes an object ndarray or a sequence of elements; "
                f"got {type(elements).__name__}"
            ) from None
        store = np.empty(len(flat), dtype=object)
        for position, element in enumerate(flat):
            store[position] = element

    if store.size == 0:
        raise ValueError(f"{kind} requires at least one element")
    if store.ndim == 0:
        raise ValueError(f"{kind} requires at least one batch axis; got a single object")
    return store


def _axis_groups_for(
    shape: tuple[int, ...],
    names: tuple[str, ...],
    axis_groups: Iterable[Iterable[int]] | None,
    *,
    kind: str,
) -> tuple[tuple[int, ...], ...]:
    """The axis groups for *shape*, defaulting to one axis per level."""
    if axis_groups is not None:
        return tuple(tuple(group) for group in axis_groups)
    if len(names) != len(shape):
        raise ValueError(
            f"{kind} places one axis per level unless axis_groups says otherwise, so "
            f"{len(shape)} axes need {len(shape)} level names; got {len(names)}: {list(names)}"
        )
    return tuple((size,) for size in shape)
