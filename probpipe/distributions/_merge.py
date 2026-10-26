"""The ``&`` operator: merging distributions and kernels into a joint.

``A & B`` merges two operands conditional-first: the left operand may
condition on what the right produces, so ``lik & prior`` reads as the density
``p(y | β) · p(β)``. Each operand enters as its flattened factors, so ``A & B &
C`` is one flat joint, and merge reads only output and input names,
never a label. The result is a ``FactoredDistribution`` when no input is left
unmet and a ``FactoredConditionalDistribution`` over the unmet inputs
otherwise; it is unlabeled, so its expression is the product of the operands'
expressions: its label joins the operands' current labels with ``·`` and it
displays operand by operand, a labeled operand by its label. Its provenance
records the step ``&`` with the two operands as its parents, left first.

The base classes expose the operator through ``__and__``, which delegates to the
engine this module installs at import.
"""

from __future__ import annotations

from collections.abc import Callable

from ..core.provenance import _MERGE_OPERATION, Provenance
from ._capabilities import SupportsMarginals, _capability_guard
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


def _merge_distributions(
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
        factors, as ``lik(y | mu)·prior(mu)`` or ``model(y, mu)·d(z)``. Its
        provenance records the operation ``&`` with *left* and *right* as its
        parents, in that order, as written rather than flattened, wherever
        :meth:`~probpipe.core.provenance.Provenance.create` records one.

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
    return joint.with_provenance(Provenance.create(_MERGE_OPERATION, parents=operands))


_root_ancestor: Callable[[Distribution], Distribution] | None = None


def _install_root_ancestor(reader: Callable[[Distribution], Distribution]) -> None:
    """Install the root reader shared with lifted distribution arguments."""
    global _root_ancestor
    _root_ancestor = reader


def _coupling_parts(
    operand: Distribution | ConditionalDistribution,
) -> tuple[Distribution, ...]:
    """The laws of *operand* that a coupling can pass through: the operand and every factor at any depth.

    A factored operand contributes itself and its factors, and a factor that is
    itself factored, such as a grouped sub-joint, contributes its own factors in
    turn. A kernel draws nothing of its own, so it contributes nothing.
    """
    parts: list[Distribution] = []
    pending: list[Distribution | ConditionalDistribution] = [operand]
    while pending:
        part = pending.pop(0)
        if isinstance(part, Distribution):
            parts.append(part)
        graph = getattr(part, "_graph", None)
        if isinstance(graph, _FactorGraph):
            pending.extend(graph.factors)
    return tuple(parts)


def _shown(part: Distribution) -> str:
    """*part* as a merge message names it: a coupled marginal by its parent and selection."""
    if isinstance(part, CoupledMarginal):
        return f"{part.parent.label}[{_selection(_paths_of(part))}]"
    return part.label


def _paths_of(marginal: CoupledMarginal) -> tuple[str, ...]:
    """The paths a coupled marginal selects from its parent."""
    return (marginal.path,) if isinstance(marginal.path, str) else tuple(marginal.path)


def _selection(paths: tuple[str, ...]) -> str:
    """*paths* as they appear between the brackets of an index."""
    return ", ".join(repr(path) for path in paths)


def _has_exact_marginals(law: Distribution, *selections: tuple[str, ...]) -> bool:
    """Whether ``marginal(law, s)`` computes for every selection, which takes an exact marginal today."""
    if not isinstance(law, SupportsMarginals):
        return False
    return all(
        _capability_guard(law, "_marginal", paths[0] if len(paths) == 1 else paths).feasible
        for paths in selections
    )


def _refuse_coupled_operands(
    left: Distribution | ConditionalDistribution,
    right: Distribution | ConditionalDistribution,
) -> None:
    """Refuse operands with a part on each side that shares one root ancestor (IV.2, condition 9).

    Each side's parts are the operand and every factor at any depth, so a law
    is refused beside its coupled marginal, its rename, or a joint that holds
    either. Two coupled marginals of one parent are reported as such, since
    their joint law is the parent's coupled marginal at both selections.

    Parameters
    ----------
    left : Distribution or ConditionalDistribution
        The left operand of the merge.
    right : Distribution or ConditionalDistribution
        The right operand of the merge.

    Raises
    ------
    ValueError
        If a part of *left* and a part of *right* share a root ancestor.
    """
    if _root_ancestor is None:
        raise RuntimeError(
            "cannot check distribution coupling before its reader is installed. Import probpipe first"
        )
    left_parts = [(part, _root_ancestor(part)) for part in _coupling_parts(left)]
    right_parts = [(part, _root_ancestor(part)) for part in _coupling_parts(right)]
    pairs = [
        (left_part, right_part, left_root)
        for left_part, left_root in left_parts
        for right_part, right_root in right_parts
        if left_root is right_root
    ]
    if not pairs:
        return
    for left_part, right_part, _ in pairs:
        if (
            isinstance(left_part, CoupledMarginal)
            and isinstance(right_part, CoupledMarginal)
            and left_part.parent is right_part.parent
        ):
            raise ValueError(_coupled_marginals_message(left_part, right_part))
    left_part, right_part, root = pairs[0]
    if left_part is right_part:
        raise ValueError(
            f"cannot merge {_shown(left)} & {_shown(right)}: the law {_shown(left_part)} is in "
            f"both operands. Merge it once"
        )
    written = (left, right, left_part, right_part)
    source = root.label if any(root is law for law in written) else "the same law"
    named = (_shown(left_part), _shown(right_part))
    parts = (
        "both are"
        if (left_part, right_part) == (left, right) or named[0] == named[1]
        else f"{named[0]} and {named[1]} are both"
    )
    raise ValueError(
        f"cannot merge {_shown(left)} & {_shown(right)}: {parts} drawn from {source}, which a "
        f"merge would treat as independent"
    )


def _coupled_marginals_message(left: CoupledMarginal, right: CoupledMarginal) -> str:
    """The refusal of two coupled marginals of one parent, with the fixes that hold for it."""
    parent = left.parent.label
    left_paths, right_paths = _paths_of(left), _paths_of(right)
    unique_paths = tuple(dict.fromkeys((*left_paths, *right_paths)))
    joint_paths = tuple(
        path
        for path in unique_paths
        if not any(path != other and _is_within(path, other) for other in unique_paths)
    )
    sentences = [
        f"cannot merge {parent}[{_selection(left_paths)}] & {parent}[{_selection(right_paths)}]: "
        f"both are coupled marginals of {parent}, which a merge would treat as independent",
        f"For their joint law, write {parent}[{_selection(joint_paths)}]",
    ]
    if _has_exact_marginals(left.parent, left_paths, right_paths):
        if left.event_spec.outputs.keys() & right.event_spec.outputs.keys():
            sentences.append(
                "For independent laws, rename shared outputs of the standalone marginals before "
                "merging them"
            )
        else:
            sentences.append(
                f"For independent laws, write marginal({parent}, {_selection(left_paths)}) & "
                f"marginal({parent}, {_selection(right_paths)})"
            )
    return ". ".join(sentences)


_install_merge(_merge_distributions)
