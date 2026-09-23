"""Static checks on the shape of ``supported_types``; pyright reads this file, pytest does not.

``UnaryWithBinaryShape`` must be rejected. The ignore on its ``supported_types``
line is required to be necessary, so if pyright ever stops rejecting the
override, the ignore is reported as unnecessary and the check still fails.
"""

# pyright: reportIncompatibleMethodOverride=true, reportUnnecessaryTypeIgnoreComment=true

from __future__ import annotations

from typing import Any, assert_type

from probpipe.inference import (
    BinaryDispatchMethod,
    BinaryDispatchRegistry,
    BinarySupportedTypes,
    Feasibility,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
    UnarySupportedTypes,
)


class GoodUnary(UnaryDispatchMethod):
    @property
    def name(self) -> str:
        return "good_unary"

    @property
    def exact(self) -> bool:
        return True

    def supported_types(self) -> UnarySupportedTypes:
        return (int,)

    def check(self, *args: Any, **kwargs: Any) -> Feasibility:
        return Feasibility(feasible=True)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return None


class GoodBinary(BinaryDispatchMethod):
    @property
    def name(self) -> str:
        return "good_binary"

    @property
    def exact(self) -> bool:
        return True

    def supported_types(self) -> BinarySupportedTypes:
        return ((int,), (str,))

    def check(self, *args: Any, **kwargs: Any) -> Feasibility:
        return Feasibility(feasible=True)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return None


class UnaryWithBinaryShape(GoodUnary):
    def supported_types(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> BinarySupportedTypes:
        return ((int,), (str,))


unary_registry: UnaryDispatchRegistry[UnaryDispatchMethod] = UnaryDispatchRegistry()
binary_registry: BinaryDispatchRegistry[BinaryDispatchMethod] = BinaryDispatchRegistry()
assert_type(unary_registry.get_method("good_unary").supported_types(), tuple[type, ...])
assert_type(
    binary_registry.get_method("good_binary").supported_types(),
    tuple[tuple[type, ...], tuple[type, ...]],
)
