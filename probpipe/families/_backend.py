"""The backend adapter of the parametric families.

``TFPDistribution`` implements the capability set on raw arrays over a wrapped
backend distribution, and every parametric family is a thin constructor over
it. It is the only class that knows the backend exists, and its ``raw()`` is
the wrapped backend distribution.

``TFPDistribution`` is defined in :mod:`probpipe.distributions._tfp_base`.
"""

from __future__ import annotations

__all__: list[str] = []
