"""Private workflow-RNG contracts for stochastic diagnostics."""

from __future__ import annotations

from ..core import _workflow_broker
from ..custom_types import PRNGKey

_PPC_SAMPLING_ABI = "probpipe.diagnostics.ppc/v1"


def _resolve_ppc_key(
    key: PRNGKey | None,
    *,
    source_index: int,
    n_replications: int,
    provider_abi: str,
) -> PRNGKey:
    """Preserve a caller key or claim one ordered PPC source event."""
    if key is not None:
        return key
    return _workflow_broker._resolve_automatic_key(
        None,
        _workflow_broker._singleton_effect_plan(
            operation_kind="diagnostics-ppc",
            execution_mode="sampled",
            sample_shape=(n_replications,),
            source_index=source_index,
            sampling_abi=_PPC_SAMPLING_ABI,
            provider_abi=provider_abi,
        ),
    )
