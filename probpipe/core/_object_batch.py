"""Object-array storage for the batch forms of values that do not stack natively.

A ``NumericArraySpec`` value batches natively — an array with the batch axes leading —
so no class is needed for it. A callable, an opaque object, and a law have no such
form: there is nothing to stack them *into*. :class:`_ObjectBatch` supplies the
storage those batch forms share, a numpy object array, leaving each public
class to say only what its elements are and which spec they satisfy.

The object array earns its place by answering the storage contract
:class:`~probpipe.core._batch.Batch` states rather than by holding arrays:
numpy's basic indexing returns a **view** over the same objects, so a
sub-batch shares its parent's store, and it honors a descending or stepped
slice in the order given, which the derived labels of a view are stated in.

An element is a view as well: the stored object under the label derived from
its position, sharing the stored object's representation, with provenance
naming the batch and the stored object.

See design II.4, II.5, and III.1.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Self

import jax
import numpy as np

from ._batch import Batch, BatchSpec, _axis_groups_for
from ._expression import Applied, Expression
from ._repr import type_name
from ._shapes import AxisCountsLike, NamesLike, _as_axis_counts, _as_names
from ._specs import TermSpec
from .provenance import Provenance
from .tracked import TrackedTerm


class _ObjectBatch[E](Batch[E]):
    """A :class:`Batch` storing its elements in a numpy object array.

    Parameters
    ----------
    label : str
        The batch's label. Required, as it is for every batch: a batch is a value a
        caller holds, and a label derived from its class says nothing about what it
        holds.
    elements : numpy.ndarray or iterable
        The elements, as an object array of any shape or a flat iterable. A
        nested sequence is not unpacked: build the array to state a shape of
        more than one axis, since what nesting means for an arbitrary Python
        object is the caller's to decide. A supplied array is copied and the
        store frozen, so the batch holds the elements it validated.
    level_names : str or sequence of str
        One name per level, outermost first; a single string names a single
        level. There is no default, deliberately — see *Notes*.
    element_spec : TermSpec
        What every element satisfies, checked against each at construction.
    axes_per_level : int or sequence of int, optional
        How many axes each level holds, outermost first (a single int is one level's
        count); they must account for every batch axis. Defaults to one axis per
        level, which requires as many names as there are batch axes. The *sizes* are
        read off the elements rather than restated here — they are already fixed by
        the data, so the only thing left to say is where one level ends and the next
        begins.
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
        axis to count along), if ``axes_per_level`` does not account for every
        stored axis or gives a count that is not one per level, or if it is omitted
        and the number of names does not match the number of axes.
    TypeError
        If *level_names* is not a str or a sequence of str, or *axes_per_level* is
        not an int or a sequence of ints; a generator, a set, ``bytes``, and a
        mapping are refused for both.
    ValueError
        If an *axes_per_level* count is less than 1, or a level name is empty or
        contains ``/``.

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

    #: What the shared spec admits, worded to follow "elements must" in the
    #: refusal a bad element earns.
    _element_rule = "match element_spec"

    def __init__(
        self,
        label: str,
        elements: np.ndarray | Iterable[E],
        /,
        level_names: NamesLike,
        *,
        element_spec: TermSpec,
        axes_per_level: AxisCountsLike | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        store = _as_object_array(elements, kind=type(self).__name__)
        kind = type(self).__name__
        names = _as_names(level_names, what=f"{kind} level_names")
        axes = (
            None
            if axes_per_level is None
            else _as_axis_counts(axes_per_level, what=f"{kind} axes_per_level")
        )
        groups = _axis_groups_for(store.shape, names, axes, kind=kind)

        object.__setattr__(self, "_store", store)
        _check_elements(
            store,
            element_spec,
            refusal=lambda element: self._element_refusal(element, element_spec),
            kind=type(self).__name__,
        )
        self._init_batch(
            BatchSpec._from_groups(element_spec, groups, names),
            label=label,
            provenance=provenance,
        )

    @classmethod
    def _over_store(cls, store: np.ndarray, *, spec: BatchSpec, label: str) -> Self:
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
        batch._init_batch(spec, label=label)
        return batch

    def raw(self) -> np.ndarray:
        """The storage view: the frozen object array of the stored elements, batch axes leading."""
        return self._store

    # -- the storage seam ---------------------------------------------------

    def _element_at(self, index: tuple[int, ...], *, label: str) -> E:
        """The stored object at *index*, as a view under the derived *label*.

        A stored tracked term is returned as a copy under *label* that shares its
        representation, so a law keeps its parameters and a function its callable,
        and the stored object itself is left untouched. A stored value that is
        not a tracked term is wrapped as the term of the batch's element kind by
        :meth:`_wrap_element`. Either way the view's provenance records the
        batch and, for a stored term, that term as its source, with the position
        in the metadata.

        Parameters
        ----------
        index : tuple of int
            One position per batch axis.
        label : str
            The label of the view, derived from its position.

        Returns
        -------
        E
            A tracked term of the element kind.

        Raises
        ------
        TypeError
            If a stored value is not a tracked term and the element kind gives
            it no term, as :meth:`_wrap_element` states.
        """
        stored = self._store[index]
        source = stored if isinstance(stored, TrackedTerm) else None
        provenance = Provenance.of_view(self, source, metadata={"position": list(index)})
        if isinstance(stored, TrackedTerm):
            view = stored.with_label(label)
            # ``with_label`` records a relabeling; the view's lineage is its selection.
            object.__setattr__(view, "_provenance", None)
            return view.with_provenance(provenance)
        return self._wrap_element(stored, label).with_provenance(provenance)

    def _element_call(self, index: tuple[int, ...]) -> Expression | None:
        """The call the stored law at *index* carries when a function lifted over laws gave it, else ``None``."""
        stored = self._store[index]
        expression = stored._expression if isinstance(stored, TrackedTerm) else None
        return expression if isinstance(expression, Applied) else None

    def _wrap_element(self, value: Any, label: str) -> Any:
        """The term of this batch's element kind holding the raw *value*, labeled *label*.

        A batch whose elements are always tracked terms, as a batch of laws is,
        keeps this default, which refuses a raw value.

        Parameters
        ----------
        value : Any
            The raw value stored at the element's position.
        label : str
            The label of the element view, derived from its position.

        Returns
        -------
        Any
            The element term that an override builds; this default raises instead.

        Raises
        ------
        TypeError
            Always, naming the value's type.
        """
        raise TypeError(
            f"a {type(self).__name__} element is a tracked term, and the value stored at "
            f"{label!r} is a {type(value).__name__}"
        )

    def _element_refusal(self, element: Any, element_spec: TermSpec) -> str:
        """Why *element_spec* refuses *element*, worded to follow "but" in the message.

        The default states :attr:`_element_rule`; a class whose spec refuses
        elements for more than one reason words each one.
        """
        return f"elements must {self._element_rule}"

    def _sub_batch_at(self, index: tuple[int | slice, ...], *, spec: BatchSpec, label: str) -> Self:
        """A view over the same store, indexed as given.

        numpy basic indexing returns a view, so the selection shares its
        parent's objects and is presented in the order *index* states, a
        descending slice included. Built without ``__init__``, since the spec and
        the label are already decided and re-deriving them from the view's own
        shape would lose the levels a dropped axis came from.
        """
        # ``object.__new__`` for the reason ``TrackedTerm._shallow_copy`` gives: a
        # host's own ``__new__`` may select a class from constructor arguments and
        # must not run again where there are none.
        view = object.__new__(type(self))
        object.__setattr__(view, "_store", self._store[index])
        view._init_batch(spec, label=label)
        return view


def _frozen_object_column(column: np.ndarray) -> np.ndarray:
    """*column* as an object array nobody can write through.

    A batch holds the columns it validated, so an object column is copied and
    frozen, for the reason ``_ObjectBatch`` states: a caller keeping a handle on
    what they passed cannot write a value into the batch that its spec does not
    admit. Only the pointer array is copied, so the elements stay shared. A
    numeric NumPy column is marked read-only in place by ``_read_only``.
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
            raise TypeError(_not_object_dtype(kind, elements))
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
        parts = "keys" if isinstance(elements, Mapping) else "characters"
        raise TypeError(
            f"{kind}: elements must be a sequence of elements, got {type(elements).__name__}, "
            f"which would be split into its {parts}; wrap it in a list to batch it as one element"
        )
    if isinstance(elements, np.ndarray | jax.Array):
        raise TypeError(_not_object_dtype(kind, elements))


def _not_object_dtype(kind: str, elements: Any) -> str:
    """The message for an array of elements whose dtype is not ``object``."""
    return (
        f"{kind}: an array of elements must have dtype=object, got {type_name(elements)} "
        f"with dtype {elements.dtype}; use NumericArrayBatch for numeric values"
    )


def _from_iterable(elements: Iterable[Any], *, kind: str) -> np.ndarray:
    """An object array holding each of *elements*, whole."""
    try:
        iterator = iter(elements)
    except TypeError:
        raise TypeError(
            f"{kind}: elements must be an object array or an iterable, "
            f"got {type(elements).__name__}"
        ) from None
    flat = list(iterator)
    store = np.empty(len(flat), dtype=object)
    for position, element in enumerate(flat):
        store[position] = element
    return store


def _check_elements(
    store: np.ndarray, element_spec: TermSpec, *, refusal: Callable[[Any], str], kind: str
) -> None:
    """Fail on the first element the shared spec does not admit, naming its position.

    Checked at construction rather than left to ``is_valid`` because a batch
    asserts its ``element_spec`` of *every* element: one that does not satisfy it
    makes the batch's own spec a false statement, and where it sits is what a
    caller needs to hear. *refusal* says why the spec refuses an element, so each
    class supplies only its own wording.
    """
    for index, element in np.ndenumerate(store):
        if not element_spec.is_valid(element):
            position = index[0] if len(index) == 1 else index
            raise TypeError(
                f"{kind}: element {position} is {type_name(element)}, but {refusal(element)}"
            )
