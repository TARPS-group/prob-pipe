"""Function distribution-input normalization helpers.

This private module handles only distribution-valued workflow inputs.
It is not a general normalization layer for all values entering a
``Function`` call.

The normalization step runs after call resolution and before broadcast
planning. It performs value-changing work that the planner should not
do: converting external distribution objects through the converter
registry, converting ProbPipe distributions to satisfy concrete
``Distribution`` hints or distribution capability protocols, and
unwrapping scalar ``DistributionArray`` inputs when the function expects
a scalar distribution value.

Keeping those conversions here lets broadcast planning remain a pure
classification step over already-normalized values.
"""

from __future__ import annotations

from typing import Any

from ..converters import converter_registry
from . import _workflow_call
from ._distribution_array import DistributionArray
from .distribution import Distribution, NumericRecordDistribution
from .protocols import (
    SupportsConditioning,
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsQuantile,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
)

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
    SupportsConditioning,
)


def is_distribution_hint(expected: Any) -> bool:
    """Return whether a type hint asks for a distribution object."""
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
    signature_info: _workflow_call.WorkflowSignatureInfo,
) -> dict[str, Any]:
    """Normalize distribution-valued inputs before broadcast planning.

    Non-distribution values are copied through unchanged. Distribution
    values may be converted according to the function's type hints, and
    external distribution objects in non-distribution slots are converted
    to ProbPipe ``NumericRecordDistribution`` so the distribution-broadcast
    path can sample them uniformly.
    """
    out = dict(values)

    for ref in _workflow_call.iter_input_refs(signature_info, values):
        value = _workflow_call.input_ref_value(out, ref)
        expected = _workflow_call.input_ref_hint(signature_info, ref)

        if isinstance(value, DistributionArray):
            if (
                value.batch_shape == ()
                and not _is_distribution_array_hint(expected)
                and expected is not Any
            ):
                out = _workflow_call.replace_input_ref(out, ref, value._flat_component(0))
            continue

        if expected is not None:
            value = _convert_hinted_distribution(value, expected)
            out = _workflow_call.replace_input_ref(out, ref, value)

        if (
            not is_distribution_hint(expected)
            and converter_registry.is_distribution_type(value)
            and not isinstance(value, Distribution)
        ):
            out = _workflow_call.replace_input_ref(
                out,
                ref,
                converter_registry.convert(value, NumericRecordDistribution),
            )

    return out


def _convert_hinted_distribution(value: Any, expected: Any) -> Any:
    is_dist_subclass = _is_concrete_distribution_hint(expected)

    if (
        is_dist_subclass
        and converter_registry.is_distribution_type(value)
        and not isinstance(value, expected)
    ):
        return converter_registry.convert(value, expected)

    if (
        not is_dist_subclass
        and expected in DISTRIBUTION_HINT_PROTOCOLS
        and isinstance(value, Distribution)
        and not isinstance(value, expected)
    ):
        try:
            return converter_registry.convert(value, expected)
        except (TypeError, AttributeError):
            return value

    return value


def _is_concrete_distribution_hint(expected: Any) -> bool:
    try:
        return isinstance(expected, type) and issubclass(expected, Distribution)
    except TypeError:
        return False


def _is_distribution_array_hint(expected: Any) -> bool:
    try:
        return isinstance(expected, type) and issubclass(expected, DistributionArray)
    except TypeError:
        return False
