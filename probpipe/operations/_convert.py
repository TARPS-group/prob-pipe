"""The convert operation: a change of representation with the record of any other result.

``convert(d, target)`` returns a distribution of the requested class, or one
claiming the requested capability protocol, carrying ``d``'s event
declaration. A source that already satisfies the target returns under fresh
identity, and any other conversion is the converter registry's.
"""

from __future__ import annotations

from typing import Any

from ..core._dispatch import Feasibility
from ..core._specs import OutputSpec
from ..distributions._conversion import _satisfaction, converter_registry
from ..distributions._distribution import Distribution, DistributionSpec
from ._operation import BoundCall, operation

__all__ = ["convert"]


def _is_target(target: Any) -> bool:
    """The target is a distribution class or a capability protocol."""
    return isinstance(target, type)


def _convert_result(d: DistributionSpec, target: Any) -> OutputSpec:
    """The converted law carries the source's event declaration."""
    return OutputSpec(convert=DistributionSpec(d.event_spec))


@operation(result=_convert_result, conditions=(_is_target,))
def convert(d: Distribution, target: type):
    """Convert *d* to the representation *target* names.

    ``with_options(method=..., exact_only=...)`` selects the converter.

    Parameters
    ----------
    d : Distribution
        The source law.
    target : type
        A distribution class or a capability protocol.

    Returns
    -------
    Distribution
        A law of the target class, or claiming the target protocol, with *d*'s
        event declaration.

    Raises
    ------
    ApplicabilityError
        If *target* is not a class.
    ResolutionError
        If *d* does not satisfy *target* and no converter applies.
    ValueError
        If the converted law does not carry *d*'s event declaration.
    """


def _satisfies_target(call: BoundCall, result: OutputSpec | None) -> Feasibility:
    """The source satisfies the target as it is, as the converter registry tests it first.

    A class target is satisfied by its instances, and a capability protocol by
    a law that claims it and whose guard admits the call. The rejection says
    which the source is not.
    """
    return _satisfaction(call.operands["d"], call.operands["target"])


def _unchanged(call: BoundCall, result: OutputSpec | None) -> Any:
    """The source itself, which the return gives fresh identity."""
    return call.operands["d"]


convert.structural_route("identity", check=_satisfies_target, execute=_unchanged, exact=True)
convert.registry_route("converters", registry=converter_registry)
