"""The mixture operation: a kernel's output law under a mixing distribution.

``mixture(K, mixing)`` returns ``μK = ∫ K(s, ·) μ(ds)``, the law of ``T`` for
``S ~ mixing`` and ``T ~ K(S, ·)``. It is a derived operation: the
reconstructed kernel output of the merged joint ``K & mixing``.
"""

from __future__ import annotations

from typing import Any

from ..core._specs import OutputSpec
from ..core.record import Record
from ..distributions._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._factored import _event_of
from ..functions._call import ApplicabilityError
from ..values import Function
from ._operation import operation
from ._pushforward import pushforward

__all__ = ["mixture"]


def _mixture_result(K: ConditionalDistributionSpec, mixing: DistributionSpec) -> OutputSpec:
    """A law under the kernel's event spec, unchanged."""
    missing = [name for name in K.given_spec.required if name not in mixing.event_spec.outputs]
    if missing:
        raise ApplicabilityError(
            f"mixture: mixing does not supply the required inputs {missing}; "
            "bind them with `condition_on` before calling mixture. "
            "Returning a kernel over remaining inputs is not implemented yet"
        )
    return OutputSpec(DistributionSpec(K.event_spec))


def _kernel_output_projection(K: ConditionalDistribution) -> Function:
    """The map from a joint draw to *K*'s outputs, reconstructed by ``K.event_spec``."""
    return Function(
        lambda draw: _event_of(K.event_spec, draw.children),
        output_spec=K.event_spec,
        label=K.label,
    )


def _mixture_check(K: ConditionalDistribution, mixing: Distribution) -> Any:
    """The merged joint can be pushed forward onto the kernel's outputs."""
    return pushforward.check(K & mixing, _kernel_output_projection(K))


@operation(result=_mixture_result, derived_route_check=_mixture_check)
def mixture(K: ConditionalDistribution, mixing: Distribution):
    """The mixture of the kernel *K* under the mixing distribution *mixing*.

    The mixing distribution's outputs meet the kernel's inputs of the same
    name. An empirical mixing law is enumerated into a finite
    ``MixtureDistribution`` with its original weights. Otherwise the derived
    route projects joint draws through ``pushforward``. Its sampling route
    returns an empirical approximation, with the draw count controlled by
    ``max_evaluations``.

    Interim implementation detail: mixing must supply every required input.
    Returning a kernel over the remaining inputs is not implemented yet.

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
    ApplicabilityError
        If mixing leaves a required kernel input unfed.
    ValueError
        If the factors' outputs overlap or matched input and output specs
        do not unify, as in a merge.
    ResolutionError
        If neither the formula nor a direct route applies.
    """
    return pushforward((K & mixing), _kernel_output_projection(K))
