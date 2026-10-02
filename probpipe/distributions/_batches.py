"""The batch forms of the two distribution kinds.

Provides:
  - ``DistributionBatch`` – separate laws sharing one event declaration.
  - ``ConditionalDistributionBatch`` – separate kernels sharing one given
    declaration and one event declaration.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np

from ..core._kinds import register_kind
from ..core._object_batch import _as_object_array, _ObjectBatch
from ..core._specs import InputSpec, OutputSpec
from ..core.provenance import Provenance
from ._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ._distribution import _ELEMENT_SOURCE, Distribution, DistributionSpec

__all__ = ["ConditionalDistributionBatch", "DistributionBatch"]


def _first_element_spec(store: np.ndarray, kind: type, owner: str) -> object:
    """The spec of the first element of *store*, which the others must satisfy.

    Raises
    ------
    ValueError
        If *store* holds no element, so there is no declaration to share.
    TypeError
        If the first element is not an instance of *kind*.
    """
    if store.size == 0:
        raise ValueError(
            f"{owner} of no elements has no declaration to read; pass element_spec explicitly"
        )
    first = store.flat[0]
    if not isinstance(first, kind):
        position = (0,) * store.ndim
        raise TypeError(
            f"{owner} holds {kind.__name__} elements, got {type(first).__name__} at position "
            f"{position}"
        )
    return first.spec


class DistributionBatch(_ObjectBatch[Distribution]):
    """``N`` separate distributions sharing one event declaration, along batch axes.

    A ``DistributionBatch`` is a collection of separate measures, distinct from
    one joint law over a product space, as a batch of draws is distinct from one
    record of many fields. It is the batch form of ``DistributionSpec``-valued
    terms, and what a sweep of a kernel over a batch of given values produces.
    The batch stores its elements, and ``batch[i]`` is a view of the stored law:
    a copy under the name derived from the position, such as ``"laws[law=1]"``,
    sharing the stored law's representation, whose provenance records the batch
    and the stored law. A lift groups an element with its stored law, so every
    access of one element draws together.

    Parameters
    ----------
    name : str
        The batch's name.
    elements : numpy.ndarray or iterable of Distribution
        The laws, as an object array of any shape or a flat iterable.
    level_names : str or iterable of str
        One name per level, outermost first.
    element_spec : DistributionSpec, optional
        What every element satisfies. Defaults to the first element's spec, so
        the elements share its event declaration.
    axes_per_level : iterable of int, optional
        How many axes each level holds, outermost first. Defaults to one axis
        per level.
    provenance : Provenance, optional
        How this batch was produced.

    Raises
    ------
    TypeError
        If *element_spec* is not a ``DistributionSpec``, or an element is not a
        ``Distribution`` whose declaration unifies with it, naming the position.
    ValueError
        If *elements* is empty and *element_spec* is omitted, or the axes and
        level names disagree as for every batch.
    """

    __slots__ = ()

    _element_rule = "be a Distribution whose event declaration matches the batch's"

    def __init__(
        self,
        name: str,
        elements: np.ndarray | Iterable[Distribution],
        /,
        level_names: str | Iterable[str],
        *,
        element_spec: DistributionSpec | None = None,
        axes_per_level: Iterable[int] | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        if element_spec is None:
            elements = _as_object_array(elements, kind=type(self).__name__)
            element_spec = cast(
                DistributionSpec,
                _first_element_spec(elements, Distribution, "a DistributionBatch"),
            )
        elif not isinstance(element_spec, DistributionSpec):
            raise TypeError(
                f"DistributionBatch.element_spec must be a DistributionSpec, "
                f"got {type(element_spec).__name__}"
            )
        super().__init__(
            name,
            elements,
            level_names,
            element_spec=element_spec,
            axes_per_level=axes_per_level,
            provenance=provenance,
        )

    @property
    def element_spec(self) -> DistributionSpec:
        """The ``DistributionSpec`` every element satisfies, a view on ``spec``."""
        return cast(DistributionSpec, self._spec.element_spec)

    @property
    def event_spec(self) -> OutputSpec:
        """The event declaration the elements share, a view on ``spec``."""
        return self.element_spec.event_spec

    def _element_at(self, index: tuple[int, ...], *, name: str) -> Distribution:
        """The stored law at *index*, as a view that records the stored law as its source.

        The view is the stored law under the derived *name*, as every object
        batch presents an element. Its source is the root the lift's capture
        follows, so two accesses of one element, and an element and its stored
        law, draw together (V.5).
        """
        view = super()._element_at(index, name=name)
        object.__setattr__(view, _ELEMENT_SOURCE, self._store[index])
        return view


def _element_source(law: Distribution) -> Distribution | None:
    """The stored law *law* is a batch element of, or None when no batch presented it."""
    return getattr(law, _ELEMENT_SOURCE, None)


class ConditionalDistributionBatch(_ObjectBatch[ConditionalDistribution]):
    """``N`` separate conditional distributions sharing their declarations, along batch axes.

    The elements share one given declaration and one event declaration. It is
    the batch form of ``ConditionalDistributionSpec``-valued terms. As for
    :class:`DistributionBatch`, ``batch[i]`` is a view of the stored kernel under
    the name derived from the position, with provenance recording the batch and
    the stored kernel.

    Parameters
    ----------
    name : str
        The batch's name.
    elements : numpy.ndarray or iterable of ConditionalDistribution
        The kernels, as an object array of any shape or a flat iterable.
    level_names : str or iterable of str
        One name per level, outermost first.
    element_spec : ConditionalDistributionSpec, optional
        What every element satisfies. Defaults to the first element's spec.
    axes_per_level : iterable of int, optional
        How many axes each level holds, outermost first.
    provenance : Provenance, optional
        How this batch was produced.

    Raises
    ------
    TypeError
        If *element_spec* is not a ``ConditionalDistributionSpec``, or an element
        does not satisfy it, naming the position.
    ValueError
        If *elements* is empty and *element_spec* is omitted, or the axes and
        level names disagree as for every batch.
    """

    __slots__ = ()

    _element_rule = "be a ConditionalDistribution whose declarations match the batch's"

    def __init__(
        self,
        name: str,
        elements: np.ndarray | Iterable[ConditionalDistribution],
        /,
        level_names: str | Iterable[str],
        *,
        element_spec: ConditionalDistributionSpec | None = None,
        axes_per_level: Iterable[int] | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        if element_spec is None:
            elements = _as_object_array(elements, kind=type(self).__name__)
            element_spec = cast(
                ConditionalDistributionSpec,
                _first_element_spec(
                    elements, ConditionalDistribution, "a ConditionalDistributionBatch"
                ),
            )
        elif not isinstance(element_spec, ConditionalDistributionSpec):
            raise TypeError(
                f"ConditionalDistributionBatch.element_spec must be a "
                f"ConditionalDistributionSpec, got {type(element_spec).__name__}"
            )
        super().__init__(
            name,
            elements,
            level_names,
            element_spec=element_spec,
            axes_per_level=axes_per_level,
            provenance=provenance,
        )

    @property
    def element_spec(self) -> ConditionalDistributionSpec:
        """The ``ConditionalDistributionSpec`` every element satisfies, a view on ``spec``."""
        return cast(ConditionalDistributionSpec, self._spec.element_spec)

    @property
    def given_spec(self) -> InputSpec:
        """The given declaration the elements share, a view on ``spec``."""
        return self.element_spec.given_spec

    @property
    def event_spec(self) -> OutputSpec:
        """The event declaration the elements share, a view on ``spec``."""
        return self.element_spec.event_spec


register_kind(DistributionSpec, term_class=Distribution, batch_class=DistributionBatch)
register_kind(
    ConditionalDistributionSpec,
    term_class=ConditionalDistribution,
    batch_class=ConditionalDistributionBatch,
)
