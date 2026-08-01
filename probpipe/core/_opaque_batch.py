"""OpaqueBatch — the batch form of the opaque kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ._object_batch import _ObjectBatch
from .event_template import OpaqueSpec
from .provenance import Provenance

__all__ = ["OpaqueBatch"]


class OpaqueBatch(_ObjectBatch[Any]):
    """A batch of opaque objects sharing one :class:`OpaqueSpec`.

    An opaque value exposes no structure, so there is nothing to stack it into
    and the collection is a batch. This is the case a batch's own spec exists
    for: an ``OpaqueSpec`` names no ProbPipe kind, yet the batch is specified all
    the same, at the family kind over it.

    Parameters
    ----------
    elements : numpy.ndarray or sequence
        The objects, as an object array of any shape or a flat sequence.
    level_names : str or sequence of str
        One name per level, outermost first.
    element_spec : OpaqueSpec, optional
        What every element satisfies. Defaults to ``OpaqueSpec()``.
    axis_groups : sequence of sequence of int, optional
        The axes each level holds; defaults to one axis per level.

    Raises
    ------
    TypeError
        If ``element_spec`` is not an :class:`OpaqueSpec`, or an element is a
        mapping — which denotes tree structure, never a leaf, as everywhere else
        in the value layer.

    Examples
    --------
    >>> batch = OpaqueBatch(["north", "south"], "site", name="labels")
    >>> batch.batch_shape
    (2,)
    >>> batch[0]
    'north'
    """

    __slots__ = ()

    def __init__(
        self,
        elements: np.ndarray | Sequence[Any],
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
        _reject_mappings(self)

    @property
    def element_spec(self) -> OpaqueSpec:
        """The :class:`OpaqueSpec` every element satisfies — a view on ``spec``."""
        spec = self._spec.element_spec
        assert isinstance(spec, OpaqueSpec)  # narrowed at construction
        return spec


def _reject_mappings(batch: OpaqueBatch) -> None:
    """Fail at construction on any element the shared spec does not admit.

    ``OpaqueSpec`` accepts every value except a mapping, which the value layer
    reads as a subtree rather than a leaf. Reporting it here names the position,
    which is what a caller needs; leaving it to ``is_valid`` would make the
    batch's own spec a false statement about one of its elements.
    """
    for index, element in np.ndenumerate(batch._store):
        if not batch.element_spec.is_valid(element):
            position = index[0] if len(index) == 1 else index
            raise TypeError(
                f"an opaque element is any value but a mapping, which denotes a subtree; "
                f"the element at {position} is a {type(element).__name__}"
            )
