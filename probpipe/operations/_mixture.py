"""The mixture operation: a kernel's output law under a mixing distribution.

``mixture(K, mixing)`` returns ``μK = ∫ K(s, ·) μ(ds)``, the law of ``T`` for
``S ~ mixing`` and ``T ~ K(S, ·)``. It is a derived operation: the
reconstructed kernel output of the merged joint ``K * mixing``.
"""

from __future__ import annotations

from typing import Any

from ..core._specs import OutputSpec
from ..distributions._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ..distributions._distribution import Distribution, DistributionSpec
from ._operation import operation
from ._pushforward import pushforward

__all__ = ["mixture"]


def _mixture_result(K: ConditionalDistributionSpec, mixing: DistributionSpec) -> OutputSpec:
    """A law under the kernel's event spec, unchanged."""
    return OutputSpec(DistributionSpec(K.event_spec))


def _kernel_output_projection(K: ConditionalDistribution) -> Any:
    """The map from a joint draw to *K*'s outputs, reconstructed by ``K.event_spec``."""
    raise NotImplementedError("mixture is not implemented yet")


@operation(result=_mixture_result)
def mixture(K: ConditionalDistribution, mixing: Distribution):
    """The mixture of the kernel *K* under the mixing distribution *mixing*.

    The mixing distribution's outputs meet the kernel's inputs of the same
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
        spec.

    Raises
    ------
    ResolutionError
        If neither the formula nor a direct route applies.
    """
    return pushforward((K & mixing), _kernel_output_projection(K))
