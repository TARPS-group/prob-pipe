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
from ..distributions._factored import _joined_label
from ._operation import BoundCall, operation

__all__ = ["joint"]

_FACTOR_KINDS = (DistributionSpec, ConditionalDistributionSpec)


def _joint_result(A: Any, B: Any, align: Any) -> None:
    """None: the composed law, or a kernel over the unmet givens, carries the declaration composition derives."""
    return None


def _composed_label(A: Any, B: Any) -> str:
    """The factors' labels joined as composition joins them (IV.2); a rename keeps a label."""
    return _joined_label((A.label, B.label))


@operation(
    result=_joint_result,
    roles={"A": _FACTOR_KINDS, "B": _FACTOR_KINDS},
    label=_composed_label,
)
def joint(A: Any, B: Any, **align: str):
    """Compose *A* with *B* after renaming *B*'s fields, as ``A * B.with_path_names(**align)``.

    The factors are consumed as objects, so a batch of laws is refused rather
    than swept (VI.11).

    Parameters
    ----------
    A : Distribution or ConditionalDistribution
        The first factor, which may condition on what *B* produces, since the
        factors are composed conditional-first.
    B : Distribution or ConditionalDistribution
        The second factor, whose paths *align* renames.
    **align : str
        Renames of *B*'s paths, ``old=new``, as ``with_path_names`` takes them.

    Returns
    -------
    Distribution or ConditionalDistribution
        The joint; a kernel when a given is left unmet. The joint is unlabeled,
        as one that ``*`` builds is, so its notation joins its factors'
        notations with ``·``.

    Raises
    ------
    ApplicabilityError
        If a factor is neither distribution kind, a batch of laws included.
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
