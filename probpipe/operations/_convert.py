"""The convert operation: a change of representation with the record of any other result.

``convert(d, target)`` returns a distribution of the requested class, or one
claiming the requested capability protocol, carrying ``d``'s event
declaration. A source that already satisfies the target returns under fresh
identity, and any other conversion is the converter registry's. The result
keeps ``d``'s support unless the converter option ``check_support=False``
overrides the support check. The result then has the converted law's own
support.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

from ..core._dispatch import Feasibility
from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericArraySpec, TermSpec
from ..core._specs import OutputSpec
from ..distributions._conversion import _satisfaction, converter_registry
from ..distributions._distribution import Distribution, DistributionSpec
from ._operation import BoundCall, Operation, operation_registry

__all__ = ["convert"]


def _is_target(target: Any) -> bool:
    """The target is a distribution class or a capability protocol."""
    return isinstance(target, type)


def _convert_result(d: DistributionSpec, target: Any) -> OutputSpec:
    """The converted law, which exposes the source's event declaration."""
    return OutputSpec(DistributionSpec(d.event_spec))


def _open_support(spec: TermSpec) -> TermSpec:
    """*spec* with the support of each array left open."""
    if isinstance(spec, NumericArraySpec):
        return replace(spec, support=None)
    if isinstance(spec, RecordSpec):
        return spec.map(_open_support)
    return spec


class _Conversion(Operation):
    """The convert operation, whose planning reads the converter option ``check_support``.

    The result rule declares the source's event declaration with its support.
    A call whose ``method_options`` set ``check_support=False`` overrides the
    support check. Its planning leaves each support of that declaration open,
    so the result has the converted law's own support.

    Parameters
    ----------
    declaration : callable
        The authored signature of ``convert``.
    """

    def __init__(self, declaration: Callable[..., Any]) -> None:
        super().__init__(declaration, result=_convert_result, conditions=(_is_target,))

    def _plan(self, call: BoundCall) -> tuple[OutputSpec | None, tuple[str, ...]]:
        """The result rule's declaration, with each support open under ``check_support=False``.

        Parameters
        ----------
        call : BoundCall
            The bound call of one point, whose ``method_options`` control may set
            ``check_support``.

        Returns
        -------
        result : OutputSpec or None
            The declaration, or ``None`` when the declarations leave it open.
        deferred : tuple of str
            The checks deferred to the return, as :meth:`Operation._plan` gives
            them.

        Raises
        ------
        ApplicabilityError, TypeError
            As :meth:`Operation._plan` raises them.
        """
        result, deferred = super()._plan(call)
        if result is None or call.controls["method_options"].get("check_support", True):
            return result, deferred
        # The result rule declares a law, and a law's event declaration has no type hole.
        event = cast(DistributionSpec, result.spec).event_spec
        opened = event._with_spec(_open_support(cast(TermSpec, event.spec)))
        return OutputSpec(DistributionSpec(opened)), deferred


@_Conversion
def convert(d: Distribution, target: type):
    """Convert *d* to the representation *target* names.

    ``with_options(method=..., exact_only=...)`` selects the converter, and
    ``with_options(method_options=...)`` sets the options the selected
    converter reads, such as ``num_samples``. The converted law keeps *d*'s
    support. A converter refuses a result on another support unless the option
    ``check_support=False`` overrides the refusal.

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
        event declaration. Under ``check_support=False`` its support is the
        converted law's own, such as the support of a fitted family.

    Raises
    ------
    ApplicabilityError
        If *target* is not a class.
    ResolutionError
        If *d* does not satisfy *target* and no converter applies.
    ValueError
        If the converted law does not carry *d*'s event declaration, which
        includes its support unless ``check_support=False``.
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


operation_registry.register(convert)
convert.structural_route("identity", check=_satisfies_target, execute=_unchanged, exact=True)
convert.registry_route("converters", registry=converter_registry)
