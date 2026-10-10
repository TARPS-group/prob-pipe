"""OpaqueBatch — the batch form of the opaque kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np

from ._kinds import register_kind
from ._object_batch import _as_object_array, _ObjectBatch
from ._opaque import Opaque, OpaqueSpec
from ._shapes import AxisCountsLike, NamesLike
from ._spec_base import _opaque_spec_of
from ._specs import TermSpec
from .provenance import Provenance

__all__ = ["OpaqueBatch"]


class OpaqueBatch(_ObjectBatch[Any]):
    """A batch of opaque objects sharing one :class:`OpaqueSpec`.

    Parameters
    ----------
    label : str
        The batch's label. Required, as it is for every batch: a batch is a value a
        caller holds, and a label derived from its class says nothing about what it
        holds.
    elements : numpy.ndarray or iterable
        The objects, as an object array of any shape or a flat iterable.
    level_names : str or sequence of str
        One name per level, outermost first.
    element_spec : OpaqueSpec, optional
        What every element satisfies. Defaults to the :class:`OpaqueSpec` of
        the type the elements share exactly, which admits any value when they
        differ.
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
        If ``element_spec`` is not an :class:`OpaqueSpec`; if an element is a
        mapping, naming the position that failed; if ``elements`` is a string, a
        mapping, or an array that is not ``dtype=object`` — each iterates into
        something other than its elements — or is not iterable at all.
    ValueError
        If ``elements`` is a zero-dimensional array (one object, with no batch
        axis to count along); if ``axes_per_level`` does not account for every axis
        the elements are stored in, or gives a count that is not one per level; or
        if it is omitted and the number of level names does not match the number
        of axes.
    TypeError
        If *level_names* is not a str or a sequence of str, or *axes_per_level* is
        not an int or a sequence of ints; a generator, a set, ``bytes``, and a
        mapping are refused for both.
    ValueError
        If an *axes_per_level* count is less than 1, or a level name is empty or
        contains ``/``.

    Notes
    -----
    An opaque value exposes no structure, so there is nothing to stack it into
    and the collection is a batch. An element may be any value except a mapping,
    which the value layer reads as a subtree rather than a leaf.

    This is the case a batch's own spec exists for: an ``OpaqueSpec`` names no
    ProbPipe kind, yet the batch is specified all the same, at the family kind
    over it.

    This batch **stores** its elements, and ``batch[i]`` is a view of the
    stored object: an :class:`~probpipe.Opaque` holding it under the label
    derived from the position, or, for a stored tracked term, a copy of that
    term under the derived label that shares its representation. Its provenance
    records the batch and the stored term. A sub-batch is a view and takes a
    derived label as any view does.

    Examples
    --------
    >>> batch = OpaqueBatch("labels", ["north", "south"], "site")
    >>> batch.batch_shape
    (2,)
    >>> batch[0].value
    'north'
    >>> batch[0].label
    'labels[site=0]'
    """

    __slots__ = ()

    def _element_refusal(self, element: Any, element_spec: TermSpec) -> str:
        """Why *element_spec* refuses *element*: it is a mapping, or not of the spec's type."""
        if isinstance(element, Mapping):
            return "elements cannot be mappings; use RecordBatch for structured values"
        return f"elements must match element_spec {element_spec!r}"

    def __init__(
        self,
        label: str,
        elements: np.ndarray | Iterable[Any],
        /,
        level_names: NamesLike,
        *,
        element_spec: OpaqueSpec | None = None,
        axes_per_level: AxisCountsLike | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        if element_spec is None:
            elements = _as_object_array(elements, kind=type(self).__name__)
            element_spec = _opaque_spec_of(elements.flat)
        elif not isinstance(element_spec, OpaqueSpec):
            raise TypeError(
                f"OpaqueBatch.element_spec must be an OpaqueSpec, got {type(element_spec).__name__}"
            )
        super().__init__(
            label,
            elements,
            level_names,
            element_spec=element_spec,
            axes_per_level=axes_per_level,
            provenance=provenance,
        )

    @property
    def element_spec(self) -> OpaqueSpec:
        """The :class:`OpaqueSpec` every element satisfies — a view on ``spec``."""
        return cast(OpaqueSpec, self._spec.element_spec)

    def _wrap_element(self, value: Any, label: str) -> Opaque:
        """The stored *value* as an :class:`~probpipe.Opaque` labeled *label*."""
        return Opaque(label, value, spec=self.element_spec)


register_kind(OpaqueSpec, term_class=Opaque, batch_class=OpaqueBatch)
