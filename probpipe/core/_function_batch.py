"""FunctionBatch — the batch form of the function kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from ._kinds import register_kind
from ._object_batch import _ObjectBatch
from .event_template import FunctionSpec
from .provenance import Provenance

__all__ = ["FunctionBatch"]


class FunctionBatch(_ObjectBatch[Callable]):
    """A batch of callables sharing one :class:`FunctionSpec`.

    Parameters
    ----------
    name : str
        The batch's name. Required, as it is for every batch: a batch is a value a
        caller holds, and a name derived from its class says nothing about what it
        holds.
    elements : numpy.ndarray or iterable of callable
        The callables, as an object array of any shape or a flat iterable.
    level_names : str or iterable of str
        One name per level, outermost first.
    element_spec : FunctionSpec, optional
        What every element satisfies. Defaults to ``FunctionSpec()``, which
        specifies a callable and neither of its templates.
    axes_per_level : iterable of int, optional
        How many axes each level holds, outermost first; they must account for
        every batch axis. Defaults to one axis per level, which requires as many
        names as there are batch axes. The *sizes* are read off the elements
        rather than restated here — they are already fixed by the data, so the
        only thing left to say is where one level ends and the next begins.
    name_is_auto : bool, default False
        Whether *name* is auto-derived rather than user-given.
    provenance : Provenance, optional
        How this batch was produced.

    Raises
    ------
    TypeError
        If ``element_spec`` is not a :class:`FunctionSpec`; if an element is not
        callable, naming the position that failed; if ``elements`` is a string, a
        mapping, or an array that is not ``dtype=object`` — each iterates into
        something other than its elements — or is not iterable at all.
    ValueError
        If ``elements`` is a zero-dimensional array (one object, with no batch
        axis to count along); if ``axis_groups`` does not tile the shape the elements are
        stored in; or if ``axis_groups`` is omitted and the number of level names
        does not match the number of axes.

    Notes
    -----
    A callable has no native stacked form, so the collection is a batch rather
    than an array. The spec is callable-generic: a plain lambda, a NumPy
    function, and a ``Function`` are all admitted, the wrapper being one such
    element and not the required type.

    This batch **stores** its elements, so ``batch[i]`` is the callable that was
    put in — the same object, under its own name and lineage, not a copy renamed
    to its position. A sub-batch is a view and takes a derived name as any view
    does.

    Examples
    --------
    >>> batch = FunctionBatch("f", [lambda x: x, lambda x: 2 * x], "variant")
    >>> batch.batch_shape
    (2,)
    >>> batch[1](3)
    6
    """

    __slots__ = ()

    _element_rule = "be callable"

    def __init__(
        self,
        name: str,
        elements: np.ndarray | Iterable[Callable],
        /,
        level_names: str | Iterable[str],
        *,
        element_spec: FunctionSpec | None = None,
        axes_per_level: Iterable[int] | None = None,
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
            name,
            elements,
            level_names,
            element_spec=element_spec,
            axes_per_level=axes_per_level,
            name_is_auto=name_is_auto,
            provenance=provenance,
        )

    @property
    def element_spec(self) -> FunctionSpec:
        """The :class:`FunctionSpec` every element satisfies — a view on ``spec``."""
        spec = self._spec.element_spec
        assert isinstance(spec, FunctionSpec)  # narrowed at construction
        return spec


register_kind(FunctionSpec, batch_class=FunctionBatch)
