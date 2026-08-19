"""OpaqueBatch — the batch form of the opaque kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ._object_batch import _ObjectBatch
from ._opaque import OpaqueSpec
from .provenance import Provenance

__all__ = ["OpaqueBatch"]


class OpaqueBatch(_ObjectBatch[Any]):
    """A batch of opaque objects sharing one :class:`OpaqueSpec`.

    Parameters
    ----------
    elements : numpy.ndarray or iterable
        The objects, as an object array of any shape or a flat iterable.
    level_names : str or iterable of str
        One name per level, outermost first.
    element_spec : OpaqueSpec, optional
        What every element satisfies. Defaults to ``OpaqueSpec()``.
    axis_groups : iterable of iterable of int, optional
        The axis sizes each level holds; defaults to one axis per level.
    name : str, optional
        The batch's name; defaults to ``"opaquebatch"``, marked auto-derived.
    name_is_auto : bool, default False
        Whether *name* is auto-derived rather than user-given.
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
        If ``elements`` is empty, or is a zero-dimensional array (one object with
        no batch axis); if ``axis_groups`` does not tile the shape the elements are
        stored in; or if ``axis_groups`` is omitted and the number of level names
        does not match the number of axes.

    Notes
    -----
    An opaque value exposes no structure, so there is nothing to stack it into
    and the collection is a batch. An element may be any value except a mapping,
    which the value layer reads as a subtree rather than a leaf.

    This is the case a batch's own spec exists for: an ``OpaqueSpec`` names no
    ProbPipe kind, yet the batch is specified all the same, at the family kind
    over it.

    This batch **stores** its elements, so ``batch[i]`` is the object that was
    put in — the same object, under whatever identity it already had, not a copy
    renamed to its position. A sub-batch is a view and takes a derived name as any
    view does.

    Examples
    --------
    >>> batch = OpaqueBatch(["north", "south"], "site", name="labels")
    >>> batch.batch_shape
    (2,)
    >>> batch[0]
    'north'
    """

    __slots__ = ()

    _element_rule = "be any value but a mapping, which denotes a subtree"

    def __init__(
        self,
        elements: np.ndarray | Iterable[Any],
        level_names: str | Iterable[str],
        *,
        element_spec: OpaqueSpec | None = None,
        axis_groups: Iterable[Iterable[int]] | None = None,
        name: str | None = None,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
    ) -> None:
        if element_spec is None:
            element_spec = OpaqueSpec()
        elif not isinstance(element_spec, OpaqueSpec):
            raise TypeError(
                f"OpaqueBatch.element_spec must be an OpaqueSpec, got {type(element_spec).__name__}"
            )
        super().__init__(
            elements,
            level_names,
            element_spec=element_spec,
            axis_groups=axis_groups,
            name=name,
            name_is_auto=name_is_auto,
            provenance=provenance,
        )

    @property
    def element_spec(self) -> OpaqueSpec:
        """The :class:`OpaqueSpec` every element satisfies — a view on ``spec``."""
        spec = self._spec.element_spec
        assert isinstance(spec, OpaqueSpec)  # narrowed at construction
        return spec
