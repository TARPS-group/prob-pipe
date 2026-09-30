"""The mixture family: a convex combination of laws over one event declaration.

Provides:
  - ``MixtureDistribution`` – the finite mixture of its components, what
    ``mixture`` returns for a finite mixing distribution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..distributions._distribution import Distribution

if TYPE_CHECKING:
    from ..custom_types import Array

__all__ = ["MixtureDistribution"]


class MixtureDistribution(Distribution):
    """A convex combination of component laws that share one event declaration.

    The components share one event declaration, which includes the component
    names, the kind, and the packaging, while their labels may differ. The mixture samples
    when every component samples, and it has a log-density, the weighted
    log-sum-exp of the components', when every component has one. Its moments
    combine componentwise when every component provides them: the mean is
    ``Σ wᵢ mᵢ`` and the covariance is ``Σ wᵢ (Σᵢ + mᵢ mᵢᵀ) − m mᵀ``.

    Parameters
    ----------
    name : str
        The mixture's label.
    components : Sequence[Distribution]
        The component laws, at least one, sharing one event declaration.
    weights : Array
        One nonnegative weight per component, summing to one.

    Raises
    ------
    NotImplementedError
        Always, until the mixture family is implemented.
    """

    def __init__(self, name: str, components: Sequence[Distribution], weights: Array) -> None:
        raise NotImplementedError("MixtureDistribution.__init__")
