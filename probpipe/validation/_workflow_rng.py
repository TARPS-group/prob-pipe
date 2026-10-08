"""Private workflow-RNG contracts for validation operations."""

from __future__ import annotations

import operator
from typing import Any

from ..custom_types import PRNGKey
from ..functions import _broker

_VALIDATION_SAMPLING_ABI = "probpipe.validation/v1"
_SLICED_WASSERSTEIN_PROVIDER_ABI = "probpipe.validation.sliced_wasserstein/v1"


def _validate_positive_int(name: str, value: Any) -> int:
    """Validate a positive integer control before stochastic commit."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    try:
        count = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer; got {value!r}") from None
    if count <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return count


def _claim_validation_key(
    *,
    operation_kind: str,
    execution_mode: str,
    sample_shape: tuple[int, ...] | None,
    provider_abi: str,
) -> PRNGKey:
    """Claim one validation singleton event of the workflow scope and return its key."""
    return _broker._resolve_automatic_key(
        None,
        _broker._singleton_effect_plan(
            operation_kind=operation_kind,
            execution_mode=execution_mode,
            sample_shape=sample_shape,
            sampling_abi=_VALIDATION_SAMPLING_ABI,
            provider_abi=provider_abi,
        ),
    )
