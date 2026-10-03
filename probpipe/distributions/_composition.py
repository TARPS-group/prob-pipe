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
    _factor_graph,
    _joined_label,
)

__all__: list[str] = []


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
    # Each operand enters with its factors and its dimension scope; the factor
    # graph flattens a factored operand and carries its scope.
    operands = (left, right)
    label = _joined_label((left.label, right.label))
    if _factor_graph(operands).unmet is None:
        return FactoredDistribution(label, operands)
    return FactoredConditionalDistribution(label, operands)


_install_composition(_compose)
