"""The joint operation: merge after realigning field names.

``A & B`` merges two factors when a producer's field names match the names
its consumer conditions on. ``joint(A, B, **align)`` renames ``B``'s fields
first, so it equals ``A & B.rename(**align)``, which connects
mismatched factors without altering their joint law.
"""

from __future__ import annotations

from typing import Any

from ..core._specs import OutputSpec
from ..distributions._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ..distributions._distribution import Distribution, DistributionSpec
from ._operation import BoundCall, _install_expression_rule, operation

__all__ = ["joint"]

_FACTOR_KINDS = (DistributionSpec, ConditionalDistributionSpec)


def _joint_result(A: Any, B: Any, align: Any) -> None:
    """None: the merged law, or a kernel over the unmet givens, carries the declaration merge derives."""
    return None


def _merged_expression() -> None:
    """None: the joint carries the product of its operands' expressions, as ``&`` gives it (IV.2).

    The route merges *A* with *B* after renaming *B*'s fields, so the joint
    reads factor by factor under the realigned names.
    """
    return None


@operation(
    result=_joint_result,
    roles={"A": _FACTOR_KINDS, "B": _FACTOR_KINDS},
)
def joint(A: Any, B: Any, **align: str):
    """merge *A* with *B* after renaming *B*'s fields, as ``A & B.rename(**align)``.

    The factors are consumed as objects, so a batch of laws is refused rather
    than swept (VI.11).

    Parameters
    ----------
    A : Distribution or ConditionalDistribution
        The first factor, which may condition on what *B* produces, since the
        factors are merged conditional-first.
    B : Distribution or ConditionalDistribution
        The second factor, whose paths *align* renames.
    **align : str
        Renames of *B*'s paths, ``old=new``, as ``rename`` takes them.

    Returns
    -------
    Distribution or ConditionalDistribution
        The joint; a kernel when a given is left unmet. The joint is unlabeled,
        as one that ``&`` builds is, so its notation joins its factors'
        notations with ``·``.

    Raises
    ------
    ApplicabilityError
        If a factor is neither distribution kind, a batch of laws included.
    ValueError
        If merge rejects the realigned factors.
    """


def _can_merge(call: BoundCall, result: OutputSpec | None) -> bool:
    """Both factors are distributions or conditional distributions, which merge joins."""
    kinds = (Distribution, ConditionalDistribution)
    return isinstance(call.operands["A"], kinds) and isinstance(call.operands["B"], kinds)


def _merge(call: BoundCall, result: OutputSpec | None) -> Any:
    """``A & B`` after ``B.rename(align)``."""
    A, B = call.operands["A"], call.operands["B"]
    align = call.operands.get("align") or {}
    return A & (B.rename(dict(align)) if align else B)


joint.structural_route("merge", check=_can_merge, execute=_merge, exact=True)
_install_expression_rule(joint, _merged_expression)
