"""Private workflow-RNG contracts for stochastic diagnostics."""

from __future__ import annotations

from ..custom_types import PRNGKey
from ..functions import _broker

_PPC_SAMPLING_ABI = "probpipe.diagnostics.ppc/v1"


def _claim_ppc_key(
    *,
    source_index: int,
    n_replications: int,
    provider_abi: str,
) -> PRNGKey:
    """Claim one ordered PPC source event of the workflow scope and return its key."""
    return _broker._resolve_automatic_key(
        None,
        _broker._singleton_effect_plan(
            operation_kind="diagnostics-ppc",
            execution_mode="sampled",
            sample_shape=(n_replications,),
            source_index=source_index,
            sampling_abi=_PPC_SAMPLING_ABI,
            provider_abi=provider_abi,
        ),
    )
