"""Private workflow-RNG contracts for validation operations."""

from __future__ import annotations

from typing import Any

from ..core import _workflow_broker
from ..custom_types import PRNGKey
from ..modeling._glm import GLMLikelihood

_VALIDATION_SAMPLING_ABI = "probpipe.validation/v1"
_GLM_GENERATIVE_PROVIDER_ABI = "probpipe.modeling.GLMLikelihood.generate_data/v1"
_SLICED_WASSERSTEIN_PROVIDER_ABI = "probpipe.validation.sliced_wasserstein/v1"
_CERTIFIED_GLM_GENERATE_DATA = GLMLikelihood.generate_data


def _validate_positive_int(name: str, value: Any) -> int:
    """Validate a positive integer control before stochastic commit."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _certified_generative_provider_abi(provider: Any) -> str | None:
    """Return the closed GLM provider ABI, or ``None`` for opaque providers."""
    if type(provider) is not GLMLikelihood:
        return None
    if GLMLikelihood.generate_data is not _CERTIFIED_GLM_GENERATE_DATA:
        return None
    if "generate_data" in vars(provider):
        return None
    return _GLM_GENERATIVE_PROVIDER_ABI


def _require_certified_generative_provider(provider: Any, operation: str) -> str:
    """Require the closed omitted-key generative provider contract."""
    provider_abi = _certified_generative_provider_abi(provider)
    if provider_abi is None:
        raise TypeError(
            f"{operation} cannot use workflow-owned randomness with opaque provider "
            f"{type(provider).__name__}; pass an explicit key= value."
        )
    if provider._x is None:
        raise ValueError(
            f"{operation} requires GLMLikelihood to have a stored design matrix "
            "before requesting workflow-owned randomness"
        )
    return provider_abi


def _resolve_validation_key(
    key: PRNGKey | None,
    *,
    operation_kind: str,
    execution_mode: str,
    sample_shape: tuple[int, ...] | None,
    provider_abi: str,
) -> PRNGKey:
    """Preserve a caller key or claim one validation singleton event."""
    if key is not None:
        return key
    return _workflow_broker._resolve_automatic_key(
        None,
        _workflow_broker._singleton_effect_plan(
            operation_kind=operation_kind,
            execution_mode=execution_mode,
            sample_shape=sample_shape,
            sampling_abi=_VALIDATION_SAMPLING_ABI,
            provider_abi=provider_abi,
        ),
    )
