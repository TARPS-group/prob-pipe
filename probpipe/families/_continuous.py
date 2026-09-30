"""The continuous parametric families.

``Normal``, ``Beta``, ``Gamma``, ``InverseGamma``, ``Exponential``,
``LogNormal``, ``StudentT``, ``Uniform``, ``Cauchy``, ``Laplace``,
``HalfNormal``, ``HalfCauchy``, ``Pareto``, and ``TruncatedNormal`` each derive
their event term spec from their parameters and take an ``event_spec``
declaration that names the event's component.

The families are defined in :mod:`probpipe.distributions.continuous`.
"""

from __future__ import annotations

__all__: list[str] = []
