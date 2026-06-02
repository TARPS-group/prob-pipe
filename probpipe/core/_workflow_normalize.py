"""WorkflowFunction distribution-normalization helpers.

This private module owns distribution-valued input conversion before
broadcast planning. It centralizes the existing converter-registry use
so the planner can stay a pure classification step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..converters import converter_registry
from ._distribution_array import DistributionArray
from .distribution import Distribution, NumericRecordDistribution
from .protocols import (
    SupportsConditioning,
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
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


def normalize_workflow_values(
    *,
    values: dict[str, Any],
    hints: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize distribution-valued workflow inputs before planning."""
    out = dict(values)

    for name, value in values.items():
        expected = hints.get(name)

        if isinstance(value, DistributionArray):
            if (
                value.batch_shape == ()
                and not _is_distribution_array_hint(expected)
                and expected is not Any
            ):
                out[name] = value._flat_component(0)
            continue

        if expected is not None:
            out[name] = _convert_hinted_distribution(value, expected)
            value = out[name]

        if (
            not is_distribution_hint(expected)
            and converter_registry.is_distribution_type(value)
            and not isinstance(value, Distribution)
        ):
            out[name] = converter_registry.convert(value, NumericRecordDistribution)

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
