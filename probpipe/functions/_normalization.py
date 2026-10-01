"""Function distribution-input normalization helpers.

This private module handles only distribution-valued workflow inputs.
It is not a general normalization layer for all values entering a
``Function`` call.

The normalization step runs after call resolution and before broadcast
planning. It performs value-changing work that the planner should not
do: converting external distribution objects through the converter
registry, and converting distributions to satisfy the ``Distribution`` class
or the distribution capability protocol a parameter names. A union annotation
names what its arms other than ``None`` name, so ``Normal | None`` converts as
``Normal`` does.

Keeping those conversions here lets broadcast planning remain a pure
classification step over already-normalized values.
"""

from __future__ import annotations

from types import UnionType
from typing import Any, Union, get_args, get_origin

from ..converters import converter_registry
from ..distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsCovariance,
    SupportsExactConditioning,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
)
from ..distributions._distribution import Distribution, NumericDistribution
from ..values import _binding

DISTRIBUTION_HINT_PROTOCOLS: tuple[type, ...] = (
    SupportsExpectation,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsLogProb,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsQuantile,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsExactConditioning,
    SupportsApproximateConditioning,
    SupportsMarginals,
)


def is_distribution_hint(expected: Any) -> bool:
    """Return whether a type hint asks for a distribution object.

    A union asks for one when any of its arms does, so an optional
    distribution annotation consumes the distribution as the plain one does.
    """
    if get_origin(expected) in (Union, UnionType):
        return any(is_distribution_hint(arm) for arm in get_args(expected))
    origin = getattr(expected, "__origin__", None)
    expected_type = origin if isinstance(origin, type) else expected
    try:
        if (
            expected_type is not None
            and isinstance(expected_type, type)
            and issubclass(expected_type, Distribution)
        ):
            return True
    except TypeError:
        pass
    return expected in DISTRIBUTION_HINT_PROTOCOLS


def normalize_distribution_values(
    *,
    values: dict[str, Any],
    signature_info: _binding.WorkflowSignatureInfo,
) -> dict[str, Any]:
    """Normalize distribution-valued inputs before broadcast planning.

    Non-distribution values are copied through unchanged. Distribution
    values may be converted according to the function's type hints, and
    external distribution objects in non-distribution slots are converted
    to their ProbPipe law so the distribution-broadcast
    path can sample them uniformly.
    """
    out = dict(values)

    for ref in _binding.iter_input_refs(signature_info, values):
        value = _binding.input_ref_value(out, ref)
        expected = _binding.input_ref_hint(signature_info, ref)

        if expected is not None:
            value = _convert_hinted_distribution(value, expected, label=ref.label)
            out = _binding.replace_input_ref(out, ref, value)

        if (
            not is_distribution_hint(expected)
            and converter_registry.is_distribution_type(value)
            and not isinstance(value, Distribution)
        ):
            out = _binding.replace_input_ref(
                out,
                ref,
                converter_registry.convert(value, Distribution),
            )

    return out


def plan_distribution_values(
    *,
    values: dict[str, Any],
    signature_info: _binding.WorkflowSignatureInfo,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    """Plan the conversions :func:`normalize_distribution_values` executes, executing none.

    Returns
    -------
    tuple
        The values, unconverted; the converter registry's report of each planned
        conversion, by the argument's label; and, for each backend object a
        conversion brings into ProbPipe, a sentence saying that the call's lift
        waits on the law the conversion constructs.

    Raises
    ------
    ApplicabilityError
        If a distribution argument matches none of several named classes.
    """
    conversions: dict[str, Any] = {}
    waiting: list[str] = []
    for ref in _binding.iter_input_refs(signature_info, values):
        value = _binding.input_ref_value(values, ref)
        expected = _binding.input_ref_hint(signature_info, ref)
        target = None if expected is None else _conversion_target(value, expected, label=ref.label)
        if (
            target is None
            and not is_distribution_hint(expected)
            and converter_registry.is_distribution_type(value)
            and not isinstance(value, Distribution)
        ):
            target = Distribution
        if target is None:
            continue
        conversions[ref.label] = converter_registry.check(value, target)
        if not isinstance(value, Distribution):
            name = getattr(target, "__name__", repr(target))
            waiting.append(f"{ref.label!r}: the call lifts the {name} its conversion constructs")
    return dict(values), conversions, tuple(waiting)


def _arms(expected: Any) -> tuple[Any, ...]:
    """The arms of a union annotation other than ``None``, or the annotation itself."""
    if get_origin(expected) in (Union, UnionType):
        return tuple(arm for arm in get_args(expected) if arm is not type(None))
    return (expected,)


def _hint_class(arm: Any) -> Any:
    """The class a parametrized annotation names, or the annotation itself."""
    origin = getattr(arm, "__origin__", None)
    return origin if isinstance(origin, type) else arm


def _convert_hinted_distribution(value: Any, expected: Any, *, label: str) -> Any:
    """The argument converted to the distribution class or capability its parameter names.

    *expected* names the distribution classes and capability protocols among its
    arms, with ``None`` and value types set aside. A distribution of one of them
    passes unchanged, and any other converts to the single one named.

    Raises
    ------
    ApplicabilityError
        If *value* is a distribution of none of several named classes, which
        leaves no single conversion target.
    TypeError
        If no converter produces the named class.
    """
    target = _conversion_target(value, expected, label=label)
    if target is None:
        return value
    if isinstance(target, type) and issubclass(target, Distribution):
        return converter_registry.convert(value, target)
    try:
        return converter_registry.convert(value, target)
    except (TypeError, AttributeError):
        return value


def _conversion_target(value: Any, expected: Any, *, label: str) -> Any:
    """The class or capability the argument converts to at its parameter, or ``None``.

    A distribution of a class or capability *expected* names converts to nothing;
    any other converts to the single one named, and a backend object to the
    representation the registry brings it in as where that is an instance of the
    named class. A ProbPipe law converts to a named capability it may lack.

    Raises
    ------
    ApplicabilityError
        If *value* is a distribution of none of several named classes, which
        leaves no single conversion target.
    """
    arms = tuple(arm for arm in _arms(expected) if is_distribution_hint(arm))
    if not arms or not converter_registry.is_distribution_type(value):
        return None
    if any(isinstance(value, _hint_class(arm)) for arm in arms):
        return None
    if len(arms) > 1:
        from ._call import ApplicabilityError

        accepted = " | ".join(getattr(_hint_class(arm), "__name__", repr(arm)) for arm in arms)
        raise ApplicabilityError(
            f"parameter {label!r} accepts {accepted}, and got a {type(value).__name__}, which is "
            f"none of them. A union of several distribution classes names no single conversion "
            f"target: pass a law of one of those classes, or annotate the parameter with the "
            f"class to convert to"
        )
    (arm,) = arms
    if _is_concrete_distribution_hint(arm):
        target = _hint_class(arm)
        if not isinstance(value, Distribution) and issubclass(NumericDistribution, target):
            # A backend object enters ProbPipe as the law the registry converts it
            # to, which is an instance of the class the parameter names.
            target = Distribution
        return target
    if arm in DISTRIBUTION_HINT_PROTOCOLS and isinstance(value, Distribution):
        return arm
    return None


def _is_concrete_distribution_hint(expected: Any) -> bool:
    try:
        expected_class = _hint_class(expected)
        return isinstance(expected_class, type) and issubclass(expected_class, Distribution)
    except TypeError:
        return False
