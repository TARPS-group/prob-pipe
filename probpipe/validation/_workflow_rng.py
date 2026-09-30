"""Private workflow-RNG contracts for validation operations."""

from __future__ import annotations

import operator
from typing import Any

from ..core import _workflow_broker
from ..core.protocols import _WorkflowGenerativeProviderCertificate
from ..custom_types import PRNGKey

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


def _certified_generative_provider(
    provider: Any,
) -> _WorkflowGenerativeProviderCertificate | None:
    """Return exact private provider authority, or ``None`` when opaque."""
    provider_type = type(provider)
    certificate = vars(provider_type).get("_workflow_generative_provider_certificate")
    if not isinstance(certificate, _WorkflowGenerativeProviderCertificate):
        return None
    if certificate.provider_type is not provider_type:
        return None
    if getattr(provider_type, "generate_data", None) is not certificate.generate_data:
        return None
    if "generate_data" in vars(provider):
        return None
    return certificate


def _require_certified_generative_provider(provider: Any, operation: str) -> str:
    """Require the closed omitted-key generative provider contract."""
    certificate = _certified_generative_provider(provider)
    if certificate is None:
        raise TypeError(
            f"{operation} cannot use workflow-owned randomness with opaque provider "
            f"{type(provider).__name__}; pass an explicit key= value."
        )
    certificate.preflight(provider, operation)
    return certificate.provider_abi


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
