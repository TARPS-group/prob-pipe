"""The mixture operation: a kernel's output law under a mixing distribution.

``mixture(K, mixing)`` returns ``μK = ∫ K(s, ·) μ(ds)``, the law of ``T`` for
``S ~ mixing`` and ``T ~ K(S, ·)``. It is a derived operation: the
reconstructed kernel output of the merged joint ``K & mixing``.
"""

from __future__ import annotations

from ..core._specs import OutputSpec
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._factored import _event_of
from ..values import Function
from ._operation import operation
from ._pushforward import pushforward

__all__ = ["mixture"]


def _kernel_output_projection(K: ConditionalDistribution) -> Function:
    """The map from a joint draw to *K*'s outputs, reconstructed by ``K.event_spec``."""
    return Function(
        lambda draw: _event_of(K.event_spec, draw.children),
        output_spec=K.event_spec,
        label=K.label,
    )


@operation(
    result=lambda K, mixing: (
        None
        if any(name not in mixing.event_spec.outputs for name in K.given_spec.required)
        else OutputSpec(DistributionSpec(K.event_spec))
    ),
    derived_route_check=lambda K, mixing: pushforward.check(
        K & mixing, _kernel_output_projection(K)
    ),
)
def mixture(K: ConditionalDistribution, mixing: Distribution):
    """The mixture of the kernel *K* under the mixing distribution *mixing*.

    The mixing distribution's outputs meet the kernel's inputs of the same
    name. An empirical mixing law is enumerated into a finite
    ``MixtureDistribution`` with its original weights. Otherwise the derived
    route projects joint draws through ``pushforward``. Its sampling route
    returns an empirical approximation, with the draw count controlled by
    ``max_evaluations``.

    The finite route builds one component law per atom, so its cost grows
    with the number of atoms.

    Inputs that mixing leaves unfed remain inputs of the returned kernel.
    Binding its last required input computes the mixture eagerly. Conditional
    sampling draws from the merged joint and projects its outputs directly.

    Parameters
    ----------
    K : ConditionalDistribution
        The kernel ``K(s, ·)``, which draws ``T`` given ``S = s``.
    mixing : Distribution
        The law ``μ`` of ``S``.

    Returns
    -------
    Distribution or ConditionalDistribution
        The standalone law or kernel of the kernel's output, with the kernel's event
        spec.

    Raises
    ------
    ValueError
        If the factors' outputs overlap or matched input and output specs
        do not unify, as in a merge.
    ResolutionError
        If neither the formula nor a direct route applies.
    """
    return pushforward((K & mixing), _kernel_output_projection(K))
