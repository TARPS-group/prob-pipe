"""The ``&`` operator: merging distributions and kernels into a joint.

``A & B`` merges two operands conditional-first: the left operand may
condition on what the right produces, so ``lik & prior`` reads as the density
``p(y | β) · p(β)``. Each operand enters as its flattened factors, so ``A & B &
C`` is one flat joint, and merge reads only output and input names,
never a label. The result is a ``FactoredDistribution`` when no input is left
unmet and a ``FactoredConditionalDistribution`` over the unmet inputs
otherwise; it is unlabeled, so its expression is the product of the operands'
expressions: its label joins the operands' current labels with ``·`` and it
displays operand by operand, a labeled operand by its label.

The base classes expose the operator through ``__and__``, which delegates to the
engine this module installs at import.
"""

from __future__ import annotations

from collections.abc import Callable

from ..core._specs import _root_output_name
from ._conditional import ConditionalDistribution
from ._distribution import Distribution, _install_merge
from ._factored import (
    FactoredConditionalDistribution,
    FactoredDistribution,
    _factor_graph,
    _FactorGraph,
    _joined_label,
    _product_of,
)
from ._views import CoupledMarginal, _is_within

__all__: list[str] = []


def _merge(
    left: Distribution | ConditionalDistribution, right: Distribution | ConditionalDistribution
) -> FactoredDistribution | FactoredConditionalDistribution:
    """The joint ``left & right``, the most specific factored kind of the flat factor graph.

    Parameters
    ----------
    left : Distribution or ConditionalDistribution
        The left operand, whose inputs may consume the outputs *right*
        produces.
    right : Distribution or ConditionalDistribution
        The right operand, whose unmet inputs remain unmet in the joint.

    Returns
    -------
    FactoredDistribution or FactoredConditionalDistribution
        The joint, or ``NotImplemented`` when *right* is neither distribution
        kind. The joint is unlabeled, so its notation joins its operands'
        notations with ``·``, an unlabeled product operand entering as its
        factors, as ``lik(y | mu)·prior(mu)`` or ``model(y, mu)·d(z)``.

    Raises
    ------
    TypeError
        If an operand's draws are distributions or batches, which have no
        outputs to merge by name, until they are wrapped under a name.
    ValueError
        If factors in the two operands share a root ancestor, an output is
        produced twice, the right operand consumes an output the left
        produces, or a matched spec does not unify. Shared roots are checked
        before the other merge conditions.
    """
    if not isinstance(right, (Distribution, ConditionalDistribution)):
        return NotImplemented
    # Each operand enters with its factors and its dimension scope; the factor
    # graph flattens a factored operand and carries its scope.
    _refuse_coupled_operands(left, right)
    operands = (left, right)
    label = _joined_label((left.label, right.label))
    if _factor_graph(operands).unmet is None:
        joint = FactoredDistribution(
            operands,
            label=label,
        )
    else:
        joint = FactoredConditionalDistribution(
            operands,
            label=label,
        )
    joint._store_expression(_product_of(operands))
    return joint


_root_ancestor: Callable[[Distribution], Distribution] | None = None


def _install_root_ancestor(reader: Callable[[Distribution], Distribution]) -> None:
    """Install the root reader shared with lifted distribution arguments."""
    global _root_ancestor
    _root_ancestor = reader


def _merge_factors(
    operand: Distribution | ConditionalDistribution,
) -> tuple[Distribution | ConditionalDistribution, ...]:
    """The existing flattened factors, without validating dimensions or outputs."""
    graph = getattr(operand, "_graph", None)
    if isinstance(graph, _FactorGraph) and _root_output_name(operand.event_spec) is None:
        return graph.factors
    return (operand,)


def _refuse_coupled_operands(
    left: Distribution | ConditionalDistribution,
    right: Distribution | ConditionalDistribution,
) -> None:
    """Refuse shared roots before checking either operand's outputs or dimensions."""
    if _root_ancestor is None:
        raise RuntimeError(
            "cannot check distribution coupling before its reader is installed. Import probpipe first"
        )
    for left_factor in _merge_factors(left):
        if not isinstance(left_factor, Distribution):
            continue
        left_root = _root_ancestor(left_factor)
        for right_factor in _merge_factors(right):
            if not isinstance(right_factor, Distribution):
                continue
            if left_root is not _root_ancestor(right_factor):
                continue
            root_name = left_root.label
            if isinstance(left_factor, CoupledMarginal) and isinstance(
                right_factor, CoupledMarginal
            ):
                if left_factor.parent is right_factor.parent:
                    root_name = left_factor.parent.label
                left_paths = (
                    (left_factor.path,) if isinstance(left_factor.path, str) else left_factor.path
                )
                right_paths = (
                    (right_factor.path,)
                    if isinstance(right_factor.path, str)
                    else right_factor.path
                )
                left_selection = ", ".join(repr(path) for path in left_paths)
                right_selection = ", ".join(repr(path) for path in right_paths)
                unique_paths = tuple(dict.fromkeys((*left_paths, *right_paths)))
                joint_paths = tuple(
                    path
                    for path in unique_paths
                    if not any(
                        path != parent and _is_within(path, parent) for parent in unique_paths
                    )
                )
                joint_selection = ", ".join(repr(path) for path in joint_paths)
                independent_advice = (
                    "For independent laws, rename shared outputs of the standalone marginals "
                    "before merging them"
                    if left_factor.event_spec.outputs.keys()
                    & right_factor.event_spec.outputs.keys()
                    else f"For independent laws, write marginal({root_name}, {left_selection}) & "
                    f"marginal({root_name}, {right_selection})"
                )
                raise ValueError(
                    f"cannot merge {root_name}[{left_selection}] & {root_name}[{right_selection}]: "
                    f"both are coupled marginals of {root_name}, which a merge would treat as "
                    f"independent. For their joint law, write {root_name}[{joint_selection}]. "
                    f"{independent_advice}"
                )
            raise ValueError(
                f"cannot merge {left_factor.label} & {right_factor.label}: both read draws of "
                f"{root_name}, which a merge would treat as independent. Select their joint "
                f"law from {root_name} instead"
            )


_install_merge(_merge)
