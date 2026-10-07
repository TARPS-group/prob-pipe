"""The mixture operation: a kernel's output law under a mixing distribution.

``mixture(K, mixing)`` returns ``μK = ∫ K(s, ·) μ(ds)``, the law of ``T`` for
``S ~ mixing`` and ``T ~ K(S, ·)``. It is a derived operation: the
reconstructed kernel output of the composed joint ``K * mixing``.
"""

from __future__ import annotations

from typing import Any

from ..core._specs import OutputSpec
from ..distributions._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ..distributions._distribution import Distribution, DistributionSpec
from ._evaluate import evaluate
from ._operation import operation

__all__ = ["mixture"]


def _mixture_result(K: ConditionalDistributionSpec, mixing: DistributionSpec) -> OutputSpec:
    """A law that exposes the kernel's event declaration unchanged."""
    return OutputSpec(DistributionSpec(K.event_spec))


def _kernel_output_projection(K: ConditionalDistribution) -> Any:
    """The map from a joint draw to *K*'s produced components, reconstructed by ``K.event_spec``."""
    raise NotImplementedError("mixture.kernel_output_projection")


@operation(result=_mixture_result)
def mixture(K: ConditionalDistribution, mixing: Distribution):
    """The mixture of the kernel *K* under the mixing distribution *mixing*.

    The mixing distribution's produced slots meet the kernel's given slots by
    name.

    Parameters
    ----------
    K : ConditionalDistribution
        The kernel ``K(s, ·)``, which draws ``T`` given ``S = s``.
    mixing : Distribution
        The law ``μ`` of ``S``.

    Returns
    -------
    Distribution
        The standalone law of the kernel's output, with the kernel's event
        declaration.

    Raises
    ------
    ResolutionError
        If neither the identity nor a direct route applies.
    """
    return evaluate(_kernel_output_projection(K), K * mixing)
