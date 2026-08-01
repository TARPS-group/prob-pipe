"""FunctionBatch — the batch form of the function kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ._object_batch import _ObjectBatch
from .event_template import FunctionSpec
from .provenance import Provenance

__all__ = ["FunctionBatch"]


class FunctionBatch(_ObjectBatch[Callable]):
    """A batch of callables sharing one :class:`FunctionSpec`.

    A callable has no native stacked form, so the collection is a batch rather
    than an array. Every element satisfies the shared ``element_spec``, and the
    spec is callable-generic: a plain lambda, a NumPy function, and a
    ``Function`` are all admitted, the wrapper being one such element and not the
    required type.

    Parameters
    ----------
    elements : numpy.ndarray or sequence of callable
        The callables, as an object array of any shape or a flat sequence.
    level_names : str or sequence of str
        One name per level, outermost first.
    element_spec : FunctionSpec, optional
        What every element satisfies. Defaults to ``FunctionSpec()``, which
        specifies a callable and neither of its templates.
    axis_groups : sequence of sequence of int, optional
        The axes each level holds; defaults to one axis per level.

    Raises
    ------
    TypeError
        If ``element_spec`` is not a :class:`FunctionSpec`, or an element is not
        callable.

    Examples
    --------
    >>> batch = FunctionBatch([lambda x: x, lambda x: 2 * x], "variant", name="f")
    >>> batch.batch_shape
    (2,)
    >>> batch[1](3)
    6
    """

    __slots__ = ()

    def __init__(
        self,
        elements: np.ndarray | Sequence[Callable],
        level_names: str | Iterable[str],
        *,
        element_spec: FunctionSpec | None = None,
        axis_groups: Iterable[Iterable[int]] | None = None,
        name: str | None = None,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
    ) -> None:
        if element_spec is None:
            element_spec = FunctionSpec()
        elif not isinstance(element_spec, FunctionSpec):
            raise TypeError(
                f"FunctionBatch.element_spec must be a FunctionSpec, "
                f"got {type(element_spec).__name__}"
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
        _require_callable(self)

    @property
    def element_spec(self) -> FunctionSpec:
        """The :class:`FunctionSpec` every element satisfies — a view on ``spec``."""
        spec = self._spec.element_spec
        assert isinstance(spec, FunctionSpec)  # narrowed at construction
        return spec


def _require_callable(batch: FunctionBatch) -> None:
    """Fail at construction on any element the shared spec does not admit.

    Checked here rather than left to ``is_valid`` because a batch asserts that
    *every* element satisfies one spec: an element that does not makes the
    batch's own spec a false statement about it, and the position it sits at is
    what a caller needs to hear about.
    """
    for index, element in np.ndenumerate(batch._store):
        if not batch.element_spec.is_valid(element):
            position = index[0] if len(index) == 1 else index
            raise TypeError(
                f"every element of a FunctionBatch is callable; the element at "
                f"{position} is a {type(element).__name__}"
            )
