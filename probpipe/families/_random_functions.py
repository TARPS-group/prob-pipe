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
    name : str
        The random function's label.
    event_spec : OutputSpec or TermSpec, optional
        The declaration of one draw, a bare term spec completing as for
        ``Distribution``. By default a draw is a callable whose input and
        output are unspecified, a whole term under *name*.
    """

    def __init__(self, name: str, event_spec: OutputSpec | TermSpec | None = None) -> None:
        super().__init__(name, FunctionSpec() if event_spec is None else event_spec)

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
    name : str
        The random measure's label.
    event_spec : OutputSpec or TermSpec, optional
        The declaration of one draw, a ``DistributionSpec``. By default a draw
        is a law whose event is opaque, a whole term under *name*.
    """

    def __init__(self, name: str, event_spec: OutputSpec | TermSpec | None = None) -> None:
        if event_spec is None:
            event_spec = DistributionSpec(OutputSpec(**{name: OpaqueSpec()}))
        super().__init__(name, event_spec)
