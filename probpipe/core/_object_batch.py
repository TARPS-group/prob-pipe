"""Object-array storage for the batch forms of values that do not stack natively.

A ``NumericArraySpec`` value batches natively — an array with the batch axes leading —
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

from collections.abc import Iterable, Mapping
from typing import Any, Self

import jax
import numpy as np

from ._batch import Batch, BatchSpec, _axis_groups_for
from .event_template import ValueSpec
from .provenance import Provenance


class _ObjectBatch[E](Batch[E]):
    """A :class:`Batch` storing its elements in a numpy object array.

    Parameters
    ----------
    name : str
        The batch's name. Required, as it is for every batch: a batch is a value a
        caller holds, and a name derived from its class says nothing about what it
        holds.
    elements : numpy.ndarray or iterable
        The elements, as an object array of any shape or a flat iterable. A
        nested sequence is not unpacked: build the array to state a shape of
        more than one axis, since what nesting means for an arbitrary Python
        object is the caller's to decide. A supplied array is copied and the
        store frozen, so the batch holds the elements it validated.
    level_names : str or iterable of str
        One name per level, outermost first; a single string names a single
        level. There is no default, deliberately — see *Notes*.
    element_spec : ValueSpec
        What every element satisfies, checked against each at construction.
    axes_per_level : iterable of int, optional
        How many axes each level holds, outermost first; they must account for
        every batch axis. Defaults to one axis per level, which requires as many
        names as there are batch axes. The *sizes* are read off the elements
        rather than restated here — they are already fixed by the data, so the
        only thing left to say is where one level ends and the next begins.
    name_is_auto : bool, default False
        Whether *name* is auto-derived rather than user-given, which is what an
        operation naming its own result states.
    provenance : Provenance, optional
        How this batch was produced.

    Raises
    ------
    TypeError
        If ``elements`` is a string, a mapping, or a non-object array — each of
        which iterates into something other than its elements — if it is not
        iterable at all, or if an ndarray of elements is not ``dtype=object``.
    ValueError
        If ``elements`` is a zero-dimensional array (one object, with no batch
        axis to count along), if ``axis_groups`` does not tile the stored shape, or if
        ``axis_groups`` is omitted and the number of names does not match the
        number of axes.

    Notes
    -----
    A level name is required rather than defaulted because a batch's levels are
    named so that operations can align operands by meaning: a placeholder would
    read as meaning something while naming nothing, which is the same reason
    :class:`~probpipe.core._batch.Batch` refuses to resolve a clash by
    suffixing. The caller that mints a level knows what it means.

    Construction admits no elements, as selection always did: ``batch[0:0]`` and
    ``OpaqueBatch("draws", [], "draw")`` are both a batch of nothing. Zero is a count the
    level can carry, and an object array of no elements still reports the shape
    ``(0,)`` to read it from. What is refused is a missing *axis*: a
    zero-dimensional store is one object, with no level to count along.
    """

    _store: np.ndarray

    __slots__ = ("_store",)

    #: What the shared spec admits, phrased for the refusal a bad element earns.
    _element_rule = "satisfy this batch's element specification"

    def __init__(
        self,
        name: str,
        elements: np.ndarray | Iterable[E],
        /,
        level_names: str | Iterable[str],
        *,
        element_spec: ValueSpec,
        axes_per_level: Iterable[int] | None = None,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
    ) -> None:
        store = _as_object_array(elements, kind=type(self).__name__)
        names = (level_names,) if isinstance(level_names, str) else tuple(level_names)
        groups = _axis_groups_for(store.shape, names, axes_per_level, kind=type(self).__name__)

        object.__setattr__(self, "_store", store)
        _check_elements(
            store, element_spec, describing=self._element_rule, kind=type(self).__name__
        )
        self._init_batch(
            BatchSpec(element_spec, groups, names),
            name=name,
            name_is_auto=name_is_auto,
            provenance=provenance,
        )

    @classmethod
    def _over_store(cls, store: np.ndarray, *, spec: BatchSpec, name: str) -> Self:
        """This batch over *store* as given, without copying or re-checking it.

        The public constructor copies the elements, freezes the copy, and checks
        every entry against the element spec — each O(batch_size), and each
        earning its cost against a caller who owns the array and may write to it
        or have filled it with the wrong thing. A caller holding a store it
        already froze and already validated has neither to defend against, and
        entering through ``__init__`` would make presenting one field a walk over
        the whole batch.

        The store is shared, not copied, so this is a view: it is the caller's
        responsibility that the buffer is frozen and its entries satisfy *spec*'s
        element spec.
        """
        # ``object.__new__`` for the reason :meth:`_sub_batch_at` gives: a host's
        # own ``__new__`` may select a class from constructor arguments.
        batch = object.__new__(cls)
        object.__setattr__(batch, "_store", store)
        batch._init_batch(spec, name=name, name_is_auto=True)
        return batch

    # -- the storage seam ---------------------------------------------------

    def _element_at(self, index: tuple[int, ...], *, name: str) -> E:
        """The stored object at *index*: the caller's own, under its own identity.

        *name*, the identity derived for the position, is unused, and no
        provenance is written — this is the *storing* side of both rules
        :meth:`~probpipe.core._batch.Batch._element_at` states.

        Notes
        -----
        A derived name belongs to an element a batch *materializes*, since a row
        of columnar storage has no identity until it is built. An object placed
        here already means something, so renaming it to its position would lose
        that and hand back a copy besides. The batch stays the one place the
        position is recorded — in the name of a sub-batch, which is a view rather
        than a caller's object.
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
        # ``object.__new__`` for the reason ``TrackedTerm._shallow_copy`` gives: a
        # host's own ``__new__`` may select a class from constructor arguments and
        # must not run again where there are none.
        view = object.__new__(type(self))
        object.__setattr__(view, "_store", self._store[index])
        view._init_batch(spec, name=name, name_is_auto=True)
        return view


def _frozen_object_column(column: np.ndarray) -> np.ndarray:
    """*column* as an object array nobody can write through.

    A batch holds the columns it validated. An object column is the one kind a
    caller can still mutate after construction — a JAX array is already immutable
    and a numpy numeric column follows the aliasing convention the single-record
    types already set — so it is copied and frozen, for the reason
    ``_ObjectBatch`` states: a caller keeping a handle on what they passed cannot
    write a value into the batch that its spec does not admit. Only the pointer
    array is copied, so the elements stay shared.
    """
    frozen = np.array(column, dtype=object, subok=False)
    frozen.setflags(write=False)
    return frozen


def _is_object_array(column: Any) -> bool:
    """Whether *column* is a numpy array of objects, the non-array column form."""
    return isinstance(column, np.ndarray) and column.dtype == object


def _as_object_array(elements: np.ndarray | Iterable[Any], *, kind: str) -> np.ndarray:
    """*elements* as a writable-by-nobody object array, without unpacking it.

    ``np.asarray`` would look inside each element and stack anything array-like,
    turning a batch of two arrays into one 2-d numeric array. Allocating empty
    and assigning keeps each object whole, whatever it is.

    A supplied array is copied and the store is frozen, so the batch's elements
    are the ones it validated: a caller who keeps a handle on the array they
    passed cannot write a mapping into an ``OpaqueBatch`` afterwards, and a view,
    which shares this buffer, cannot write through to its parent. Only the
    pointer array is copied — the elements themselves stay shared.
    """
    if isinstance(elements, np.ndarray):
        if elements.dtype != object:
            raise TypeError(
                f"{kind} stores objects, so an ndarray of elements must have dtype=object; "
                f"got dtype={elements.dtype}"
            )
        # A subclass — np.matrix, a masked array — indexes by its own rules and
        # would not hand back the objects that were stored.
        store = np.array(elements, dtype=object, subok=False)
    else:
        _refuse_container(elements, kind=kind)
        store = _from_iterable(elements, kind=kind)

    if store.ndim == 0:
        raise ValueError(f"{kind} requires at least one batch axis; got a single object")
    store.setflags(write=False)
    return store


def _refuse_container(elements: Any, *, kind: str) -> None:
    """Refuse an *elements* that iterates into something other than its elements.

    A string iterates into characters, a mapping into its keys, and a numeric
    array into scalars — each a batch of parts of one object rather than a batch
    of objects, and the mapping case would slip past the per-element check that
    refuses a mapping *as* an element. Wrap the one object in a list to mean a
    batch of one.
    """
    if isinstance(elements, str | bytes | Mapping):
        raise TypeError(
            f"{kind} takes a sequence of elements, and a {type(elements).__name__} iterates "
            f"into its parts rather than into elements; wrap it in a list to batch it as one"
        )
    if isinstance(elements, np.ndarray | jax.Array):
        raise TypeError(
            f"{kind} stores objects, so an array of elements must have dtype=object; "
            f"got a {type(elements).__name__} of {elements.dtype}"
        )


def _from_iterable(elements: Iterable[Any], *, kind: str) -> np.ndarray:
    """An object array holding each of *elements*, whole."""
    try:
        iterator = iter(elements)
    except TypeError:
        raise TypeError(
            f"{kind} takes an object ndarray or an iterable of elements; "
            f"got {type(elements).__name__}"
        ) from None
    flat = list(iterator)
    store = np.empty(len(flat), dtype=object)
    for position, element in enumerate(flat):
        store[position] = element
    return store


def _check_elements(
    store: np.ndarray, element_spec: ValueSpec, *, describing: str, kind: str
) -> None:
    """Fail on the first element the shared spec does not admit, naming its position.

    Checked at construction rather than left to ``is_valid`` because a batch
    asserts its ``element_spec`` of *every* element: one that does not satisfy it
    makes the batch's own spec a false statement, and where it sits is what a
    caller needs to hear. *describing* states positively what an element must be,
    so each class supplies only its own phrase.
    """
    for index, element in np.ndenumerate(store):
        if not element_spec.is_valid(element):
            position = index[0] if len(index) == 1 else index
            raise TypeError(
                f"every element of a {kind} must {describing}; the element at {position} "
                f"is a {type(element).__name__}"
            )
