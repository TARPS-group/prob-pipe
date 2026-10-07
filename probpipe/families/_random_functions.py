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
from ..distributions._distribution import Distribution, DistributionSpec
from ..values._function_base import FunctionSpec

__all__ = ["RandomFunction", "RandomMeasure"]


def _event_of_kind(
    name: str, event_spec: OutputSpec | TermSpec | None, kind: type[TermSpec], default: TermSpec
) -> OutputSpec | TermSpec:
    """The event declaration of a law whose draws are of *kind*, with a hole filled by *default*.

    Parameters
    ----------
    name : str
        The law's label, which the error message names.
    event_spec : OutputSpec or TermSpec or None
        The declaration the constructor received, or None for the default.
    kind : type of TermSpec
        The class a declared type must be an instance of, such as ``FunctionSpec``.
    default : TermSpec
        The type of a draw when *event_spec* leaves it open.

    Returns
    -------
    OutputSpec or TermSpec
        The declaration to pass to ``Distribution``: *event_spec* with any type hole filled
        by *default*, or *default* itself when *event_spec* is None.

    Raises
    ------
    TypeError
        If *event_spec* declares a type that is not a *kind*.
    """
    if event_spec is None:
        return default
    declared = event_spec.spec if isinstance(event_spec, OutputSpec) else event_spec
    if declared is None:
        return event_spec._with_spec(default)
    if not isinstance(declared, kind):
        raise TypeError(
            f"the event of {name!r} declares a {kind.__name__}, got {type(declared).__name__}"
        )
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
    label : str
        The random function's label.
    event_spec : OutputSpec or TermSpec, optional
        The declaration of one draw, whose type is a ``FunctionSpec``; a bare
        term spec completes as for ``Distribution``. The type defaults to a
        callable whose input and output are unspecified, which also fills a
        type hole, and the declaration to a whole term under *label*.

    Raises
    ------
    TypeError
        If *event_spec* declares a type that is not a ``FunctionSpec``.
    """

    def __init__(self, label: str, event_spec: OutputSpec | TermSpec | None = None) -> None:
        super().__init__(label, _event_of_kind(label, event_spec, FunctionSpec, FunctionSpec()))

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
    label : str
        The random measure's label.
    event_spec : OutputSpec or TermSpec, optional
        The declaration of one draw, whose type is a ``DistributionSpec``. The
        type defaults to a law whose event is opaque, which also fills a type
        hole, and the declaration to a whole term under *label*.

    Raises
    ------
    TypeError
        If *event_spec* declares a type that is not a ``DistributionSpec``.
    """

    def __init__(self, label: str, event_spec: OutputSpec | TermSpec | None = None) -> None:
        opaque_law = DistributionSpec(OutputSpec(**{label: OpaqueSpec()}))
        super().__init__(label, _event_of_kind(label, event_spec, DistributionSpec, opaque_law))
