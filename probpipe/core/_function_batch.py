"""FunctionBatch — the batch form of the function kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import cast

import numpy as np

from ..values._function_base import Function, FunctionSpec
from ._kinds import register_kind
from ._object_batch import _as_object_array, _collection_expression, _ObjectBatch
from ._shapes import AxisCountsLike, NamesLike
from .provenance import Provenance

__all__ = ["FunctionBatch"]


class FunctionBatch(_ObjectBatch[Callable]):
    """A batch of callables sharing one :class:`FunctionSpec`.

    Parameters
    ----------
    elements : numpy.ndarray or iterable of callable
        The callables, as an object array of any shape or a flat iterable.
    level_names : str or sequence of str
        One name per level, outermost first.
    label : str, optional
        The batch's display alias. Defaults to a bounded description of its
        members. Empty unnamed collections require an alias.
    element_spec : FunctionSpec, optional
        What every element satisfies. Defaults to ``FunctionSpec()``, which
        specifies a callable and neither of its input/output declarations.
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
        If ``element_spec`` is not a :class:`FunctionSpec`; if an element is not
        callable, naming the position that failed; if ``elements`` is a string, a
        mapping, or an array that is not ``dtype=object`` — each iterates into
        something other than its elements — or is not iterable at all; or an empty
        collection has no explicit label.
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
    A callable has no native stacked form, so the collection is a batch rather
    than an array. The spec is callable-generic: a plain lambda, a NumPy
    function, and a ``Function`` are all admitted, the wrapper being one such
    element and not the required type.

    This batch **stores** its elements, and ``batch[i]`` is a view of the
    stored callable: a :class:`~probpipe.Function` wrapping it under the label
    derived from the position and the batch's declarations, or, for a stored
    ``Function``, a copy under the derived label that shares its callable. Its
    provenance records the batch and the stored term. A callable whose
    signature cannot be inspected, as for some builtins, has no ``Function``
    view and raises ``ValueError`` when indexed. A sub-batch is a view and
    takes a derived label as any view does.

    Examples
    --------
    >>> batch = FunctionBatch(
    ...     [lambda x: x, lambda x: 2 * x],
    ...     "variant",
    ...     label="f",
    ... )
    >>> batch.batch_shape
    (2,)
    >>> batch[1].apply(3)
    6
    """

    __slots__ = ()

    _element_rule = "be callable"

    def __init__(
        self,
        elements: np.ndarray | Iterable[Callable],
        /,
        level_names: NamesLike,
        *,
        label: str | None = None,
        element_spec: FunctionSpec | None = None,
        axes_per_level: AxisCountsLike | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        if element_spec is None:
            element_spec = FunctionSpec()
        elif not isinstance(element_spec, FunctionSpec):
            raise TypeError(
                f"FunctionBatch.element_spec must be a FunctionSpec, "
                f"got {type(element_spec).__name__}"
            )
        elements = _as_object_array(elements, kind=type(self).__name__)
        expression = _collection_expression(elements) if label is None else None
        if expression is not None:
            label = expression.render_label()
        super().__init__(
            label,
            elements,
            level_names,
            element_spec=element_spec,
            axes_per_level=axes_per_level,
            provenance=provenance,
        )
        if expression is not None:
            self._store_expression(expression)

    @property
    def element_spec(self) -> FunctionSpec:
        """The :class:`FunctionSpec` every element satisfies — a view on ``spec``."""
        return cast(FunctionSpec, self._spec.element_spec)

    def _wrap_element(self, value: Callable, label: str) -> Function:
        """The callable *value* as a ``Function`` labeled *label* under the batch's declarations.

        Parameters
        ----------
        value : callable
            The object stored at the element's position.
        label : str, optional
            The label of the element view, derived from its position.

        Returns
        -------
        Function
            A new ``Function``, which the caller gives the view's provenance.

        Raises
        ------
        ValueError
            If the callable's signature cannot be inspected, or does not match the
            declared input slots.
        """
        spec = self.element_spec
        return Function(
            value,
            input_spec=spec.input_spec,
            output_spec=spec.output_spec,
            label=label,
        )


register_kind(FunctionSpec, batch_class=FunctionBatch)
