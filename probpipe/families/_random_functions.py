"""The random functions and random measures.

A ``RandomFunction`` is a distribution whose event is a ``FunctionSpec``: a
draw is a callable, and calling the random function at a point returns the law
of the function's value there. A ``RandomMeasure`` is a distribution whose
event is a ``DistributionSpec``: a draw is a ``Distribution``, and its mean is
the marginalized law.

Provides:
  - ``RandomFunction`` – a distribution over functions, evaluated by calling it.
  - ``RandomMeasure`` – a distribution over distributions.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..core._spec_base import TermSpec
from ..core._specs import OpaqueSpec, OutputSpec
from ..distributions._distribution import (
    Distribution,
    DistributionSpec,
    _class_label,
    _constructor_label,
    _whole_term_event,
)
from ..values._function_base import FunctionSpec

__all__ = ["RandomFunction", "RandomMeasure"]


def _event_of_kind(
    component: str,
    event_spec: OutputSpec | TermSpec | None,
    kind: type[TermSpec],
    default: TermSpec,
    owner: str,
) -> OutputSpec:
    """The whole-term event under *component* of a law whose draws are of *kind*.

    Parameters
    ----------
    component : str
        The component of the event.
    event_spec : OutputSpec or TermSpec or None
        The declaration the constructor received: a declaration of *component*,
        whose type hole *default* fills, the type of a draw, or None for
        *default*.
    kind : type of TermSpec
        The class a declared type must be an instance of, such as ``FunctionSpec``.
    default : TermSpec
        The type of a draw when *event_spec* leaves it open.
    owner : str
        The constructor, as error messages name it.

    Returns
    -------
    OutputSpec
        The declaration of a whole term under *component*.

    Raises
    ------
    TypeError
        If *component* is not a string, or *event_spec* declares a type that is
        not a *kind*.
    ValueError
        If *component* is not a valid component name, or *event_spec* names
        another component.
    """
    if event_spec is None:
        return _whole_term_event(component, default, None, owner)
    declared = event_spec.spec if isinstance(event_spec, OutputSpec) else event_spec
    if declared is not None and not isinstance(declared, kind):
        raise TypeError(
            f"event_spec of {owner} must declare a {kind.__name__}, got {type(declared).__name__}"
        )
    if not isinstance(event_spec, OutputSpec):
        return _whole_term_event(component, declared, None, owner)
    if declared is None:
        return _whole_term_event(component, default, event_spec, owner)
    # A complete declaration of the component is kept as it is, so a law derived
    # from another shares its declaration.
    _whole_term_event(component, declared, event_spec, owner)
    return event_spec


class RandomFunction(Distribution):
    """A distribution over functions, whose value at a point is a distribution.

    One draw is a callable, declared by a ``FunctionSpec``. Calling the random
    function at ``x`` returns the law of ``f(x)`` for ``f`` drawn from it; at
    stacked points that law is the finite-dimensional law there. A family
    claims ``SupportsMean`` when it has a mean function, ``SupportsVariance``
    when it has a pointwise variance function, and ``SupportsSampling`` when
    it draws whole functions, each returning a callable. A law over functions
    has no density in general, so the base claims none.

    Parameters
    ----------
    component : str
        The component of the event, a whole term.
    event_spec : OutputSpec or TermSpec, optional
        The declaration of *component* or the type of one draw, a
        ``FunctionSpec``. The type defaults to a callable whose input and
        output are unspecified, which also fills a type hole.
    label : str, optional
        The random function's label, its class name by default.

    Raises
    ------
    TypeError
        If *component* is not a string, or *event_spec* declares a type that is
        not a ``FunctionSpec``.
    ValueError
        If *component* is not a valid component name, or *event_spec* names
        another component.
    """

    def __init__(
        self,
        component: str,
        event_spec: OutputSpec | TermSpec | None = None,
        *,
        label: str | None = None,
    ) -> None:
        owner = _class_label(self)
        declaration = _event_of_kind(component, event_spec, FunctionSpec, FunctionSpec(), owner)
        super().__init__(
            declaration,
            label=_constructor_label(self, label, owner),
        )

    @abstractmethod
    def __call__(self, x: Any) -> Distribution:
        """The law of the drawn function's value at *x*."""


class RandomMeasure(Distribution):
    """A distribution over distributions: one draw is a ``Distribution``.

    The event is a ``DistributionSpec``, which declares the event of a drawn
    law. A family claims ``SupportsSampling`` when it draws laws,
    ``SupportsMean`` when it computes the marginalized law
    ``D̄(A) = ∫ D(A) dM(D)``, and ``SupportsRandomLogProb`` or
    ``SupportsRandomUnnormalizedLogProb`` only when it computes the law of
    ``x ↦ log D(x)`` for ``D ~ M``, which is a ``RandomFunction``. A random
    measure claims no variance, since a law-valued draw has no event-typed
    second moment in general.

    Parameters
    ----------
    component : str
        The component of the event, a whole term.
    event_spec : OutputSpec or TermSpec, optional
        The declaration of *component* or the type of one draw, a
        ``DistributionSpec``. The type defaults to a law whose event is opaque
        under *component*, which also fills a type hole.
    label : str, optional
        The random measure's label, its class name by default.

    Raises
    ------
    TypeError
        If *component* is not a string, or *event_spec* declares a type that is
        not a ``DistributionSpec``.
    ValueError
        If *component* is not a valid component name, or *event_spec* names
        another component.
    """

    def __init__(
        self,
        component: str,
        event_spec: OutputSpec | TermSpec | None = None,
        *,
        label: str | None = None,
    ) -> None:
        owner = _class_label(self)
        if not isinstance(component, str):
            raise TypeError(
                f"{owner} takes the component of its event as its first argument, a string; got "
                f"{type(component).__name__}"
            )
        opaque_law = DistributionSpec(OutputSpec(**{component: OpaqueSpec()}))
        declaration = _event_of_kind(component, event_spec, DistributionSpec, opaque_law, owner)
        super().__init__(
            declaration,
            label=_constructor_label(self, label, owner),
        )
