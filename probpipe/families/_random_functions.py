"""The random functions and random measures.

A ``RandomFunction`` is a distribution whose event is a ``FunctionSpec``: a
draw is a callable, and calling the random function at a point returns the law
of the function's value there. A ``RandomMeasure`` is a distribution whose
event is a ``DistributionSpec``: a draw is a ``Distribution``, and its mean is
the marginalized law.

``RandomFunction`` is defined in :mod:`probpipe.core._random_functions`, and
``RandomMeasure`` in :mod:`probpipe.core._random_measures`.
"""

from __future__ import annotations

__all__: list[str] = []
