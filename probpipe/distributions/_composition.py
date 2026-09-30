"""The ``*`` operator: composing distributions and kernels into a joint.

``A * B`` composes two operands conditional-first: the left operand may
condition on what the right produces, so ``lik * prior`` reads as the density
``p(y | β) · p(β)``. Each operand enters as its flattened factors, so ``A * B *
C`` is one flat joint, and composition reads only component and slot names,
never a label. The result is a ``FactoredDistribution`` when no given is left
unmet and a ``FactoredConditionalDistribution`` over the unmet givens
otherwise; its label joins the operands' current labels with ``·``.

The base classes expose the operator through ``__mul__``, which delegates to the
engine this module installs at import.
"""

from __future__ import annotations

from ._conditional import ConditionalDistribution
from ._distribution import Distribution, _install_composition
from ._factored import (
    FactoredConditionalDistribution,
    FactoredDistribution,
    SupportsFactors,
    _factor_graph,
)

__all__: list[str] = []

#: The separator the joint's label places between the operands' labels.
_LABEL_SEP = "·"


def _flat_factors(
    operand: Distribution | ConditionalDistribution,
) -> tuple[Distribution | ConditionalDistribution, ...]:
    """*operand*'s factors if it is factored, and *operand* alone otherwise."""
    if isinstance(operand, SupportsFactors):
        return tuple(operand.factors)
    return (operand,)


def _compose(
    left: Distribution | ConditionalDistribution, right: Distribution | ConditionalDistribution
) -> FactoredDistribution | FactoredConditionalDistribution:
    """The joint ``left * right``, the most specific factored kind of the flat factor graph.

    Returns
    -------
    FactoredDistribution or FactoredConditionalDistribution
        The joint, or ``NotImplemented`` when *right* is neither distribution
        kind.

    Raises
    ------
    ValueError
        If a component is produced twice, the right operand consumes a
        component the left produces, or a matched spec does not unify.
    """
    if not isinstance(right, (Distribution, ConditionalDistribution)):
        return NotImplemented
    factors = (*_flat_factors(left), *_flat_factors(right))
    label = f"{left.name}{_LABEL_SEP}{right.name}"
    if _factor_graph(factors).unmet is None:
        return FactoredDistribution(label, factors)
    return FactoredConditionalDistribution(label, factors)


_install_composition(_compose)
