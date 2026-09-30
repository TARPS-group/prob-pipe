"""The joint operation: composition after realigning field names.

``A * B`` composes two factors when a producer's field names match the names
its consumer conditions on. ``joint(A, B, **align)`` renames ``B``'s fields
first, so it equals ``A * B.with_path_names(**align)``, which connects
mismatched factors without altering their joint law.
"""

from __future__ import annotations

from typing import Any

from ..core._specs import OutputSpec
from ..distributions._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ..distributions._distribution import Distribution, DistributionSpec
from ._operation import BoundCall, operation

__all__ = ["joint"]

_FACTOR_KINDS = (DistributionSpec, ConditionalDistributionSpec)


def _joint_result(A: Any, B: Any, align: Any) -> OutputSpec:
    """A law, or a kernel over the unmet givens, whose declaration composition derives."""
    return OutputSpec(joint=None)


@operation(result=_joint_result, roles={"A": _FACTOR_KINDS, "B": _FACTOR_KINDS})
def joint(A: Distribution, B: Distribution, **align: str):
    """Compose *A* with *B* after renaming *B*'s fields, as ``A * B.with_path_names(**align)``.

    Parameters
    ----------
    A, B : Distribution or ConditionalDistribution
        The factors, composed conditional-first, so *A* may condition on what
        *B* produces.
    **align : str
        Renames of *B*'s paths, ``old=new``, as ``with_path_names`` takes them.

    Returns
    -------
    Distribution or ConditionalDistribution
        The joint; a kernel when a given is left unmet.

    Raises
    ------
    ApplicabilityError
        If a factor is neither distribution kind.
    ValueError
        If composition rejects the realigned factors.
    """


def _can_compose(call: BoundCall, result: OutputSpec | None) -> bool:
    """Both factors are distributions or conditional distributions, which composition joins."""
    kinds = (Distribution, ConditionalDistribution)
    return isinstance(call.operands["A"], kinds) and isinstance(call.operands["B"], kinds)


def _compose(call: BoundCall, result: OutputSpec | None) -> Any:
    """``A * B`` after ``B.with_path_names(align)``."""
    A, B = call.operands["A"], call.operands["B"]
    align = call.operands.get("align") or {}
    return A * (B.with_path_names(dict(align)) if align else B)


joint.structural_route("compose", check=_can_compose, execute=_compose, exact=True)
