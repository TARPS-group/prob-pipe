"""A call-local law view for an operation at a kernel's given values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np

from ..core._dispatch import Feasibility
from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericArraySpec, OpaqueSpec, TermSpec, _unify_specs
from ..core.record import Record
from ..distributions._capabilities import (
    _CONDITIONAL_TWINS,
    _DELEGATING_DEFAULTS,
    _capability_guard,
    _capability_subclass,
    _kernel_claims,
)
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution
from ..functions._call import ApplicabilityError
from ..functions._resolution import StandIn


def _bind_given_dims_from_spec(
    expected: TermSpec, actual: TermSpec, bindings: dict[str, int], path: str
) -> None:
    """Bind dimensions while leaving value-only admission to concrete binding."""
    if isinstance(expected, OpaqueSpec) and not isinstance(actual, (RecordSpec, OpaqueSpec)):
        # An opaque declaration admits numeric values too. Their numeric spec
        # cannot establish their Python type; concrete binding checks that.
        return
    if (
        isinstance(expected, NumericArraySpec)
        and isinstance(actual, OpaqueSpec)
        and actual.type in (list, tuple)
    ):
        # Sequence payloads are opaque during planning. Concrete numeric
        # binding checks their shape and dtype before the kernel runs.
        return
    if isinstance(expected, RecordSpec) and isinstance(actual, RecordSpec):
        if expected.children.keys() != actual.children.keys():
            _unify_specs(expected, actual, bindings, path)
        for name, spec in expected.children.items():
            _bind_given_dims_from_spec(spec, actual.children[name], bindings, f"{path}/{name}")
        return
    _unify_specs(expected, actual, bindings, path)


def _bind_given_dims_from_value(
    expected: TermSpec, value: Any, bindings: dict[str, int], path: str
) -> None:
    """Bind dimensions from numeric sequences without replacing the kernel input."""
    if isinstance(expected, RecordSpec):
        children = value.children if isinstance(value, Record) else value
        if isinstance(children, Mapping) and expected.children.keys() == children.keys():
            for name, spec in expected.children.items():
                _bind_given_dims_from_value(spec, children[name], bindings, f"{path}/{name}")
            return
    if isinstance(expected, NumericArraySpec) and isinstance(value, (list, tuple)):
        value = np.asarray(value)
    expected._bind_dims_from_value(value, bindings, path)


def _forward(method: str) -> Callable[..., Any]:
    """Forward one unconditional primitive to the kernel's conditional twin."""
    conditional = "_conditional" + method

    def forward(self: _GivenLaw, *args: Any, **kwargs: Any) -> Any:
        return getattr(self.kernel, conditional)(self.given, *args, **kwargs)

    forward.__name__ = method
    forward.__doc__ = f"The kernel's ``{conditional}`` at this call's given."
    return forward


def _guard(method: str) -> Callable[..., Any]:
    """Forward the primitive's guard, including the given argument."""
    conditional = "_conditional" + method

    def guard(self: _GivenLaw, *args: Any, **kwargs: Any) -> Feasibility:
        if isinstance(self.given, StandIn):
            target = conditional
            while target is not None and getattr(self.kernel, target + "_guard", None) is None:
                target = _DELEGATING_DEFAULTS.get(getattr(type(self.kernel), target, None))
            if target is not None:
                return Feasibility(None, pending=("the conditional guard needs the given values",))
        return _capability_guard(self.kernel, conditional, self.given, *args, **kwargs)

    guard.__name__ = method + "_guard"
    guard.__doc__ = f"The kernel's ``{conditional}`` guard admits the given."
    return guard


def _methods(protocol: type) -> dict[str, Callable[..., Any]]:
    """The unconditional protocol's primitives and their delegated guards."""
    return {
        name: implementation
        for cls in protocol.__mro__
        for method, value in vars(cls).items()
        if method.startswith("_") and not method.startswith("__") and callable(value)
        for name, implementation in (
            (method, _forward(method)),
            (method + "_guard", _guard(method)),
        )
    }


class _GivenLaw(Distribution):
    """The kernel's law at complete given slots, without evaluating the kernel.

    It claims exactly the unconditional capabilities whose conditional twins
    the kernel claims. All given values unify in one dimension scope before
    the event declaration is substituted. No density, draw, kernel application,
    or inference method runs during construction. Internal declaration-only
    planning also accepts a record-valued StandIn: its field declarations bind
    the dimensions, and guards that need its values remain unresolved.

    Parameters
    ----------
    kernel : ConditionalDistribution
        The kernel whose conditional capabilities serve this call.
    given : Record, Mapping[str, Any], or StandIn
        Values of every required given slot; optional slots may be omitted. A
        private planning stand-in carries a RecordSpec for those slots.

    Raises
    ------
    ApplicabilityError
        If the given is not a record, mapping, or record-valued planning stand-in,
        names an unknown slot, leaves a required slot unbound, or fails its declaration.
    """

    _capability_table: ClassVar[dict[type, dict[str, Callable[..., Any]]]] = {
        protocol: _methods(protocol) for protocol in _CONDITIONAL_TWINS
    }

    def __new__(cls, kernel: ConditionalDistribution, given: Any) -> _GivenLaw:
        return object.__new__(_capability_subclass(cls, _kernel_claims(kernel)))

    def __init__(self, kernel: ConditionalDistribution, given: Any) -> None:
        if isinstance(given, StandIn) and isinstance(given.spec, RecordSpec):
            values = {name: StandIn(spec) for name, spec in given.spec.children.items()}
        elif isinstance(given, Record):
            values = dict(given.children)
        elif isinstance(given, Mapping):
            values = dict(given)
        else:
            raise ApplicabilityError("given must be a Record or a mapping of given slots")
        unknown = set(values) - set(kernel.given_spec)
        missing = set(kernel.given_spec.required) - set(values)
        if unknown or missing:
            raise ApplicabilityError(
                f"given for {kernel.label!r}: unknown slots {sorted(unknown)}; "
                f"missing required slots {sorted(missing)}; "
                f"available slots {list(kernel.given_spec)}"
            )
        bindings: dict[str, int] = {}
        try:
            for name, value in values.items():
                if isinstance(value, StandIn):
                    _bind_given_dims_from_spec(
                        kernel.given_spec[name], value.spec, bindings, f"given/{name}"
                    )
                else:
                    _bind_given_dims_from_value(
                        kernel.given_spec[name], value, bindings, f"given/{name}"
                    )
        except (TypeError, ValueError) as error:
            raise ApplicabilityError(f"given for {kernel.label!r}: {error}") from None
        super().__init__(kernel.event_spec.with_dim_sizes(**bindings), label=kernel.label)
        object.__setattr__(self, "kernel", kernel)
        object.__setattr__(self, "given", given if isinstance(given, StandIn) else values)
        self._store_expression(kernel._embedded_expression().with_fixed(tuple(values)))
