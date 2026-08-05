"""Versioned capability contract for workflow RNG execution routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from . import _workflow_plan
from .config import WorkflowKind

_EXECUTION_CONTRACT_ABI = "probpipe.workflow_rng_execution/v1"
_RNG_ABI = "ProbPipe-RNG-v1"
_SAMPLING_ABI = "probpipe.distribution_sampling/v1"
_PROVIDER_ABI = "probpipe.distribution/v1"
_DESCENDANT_ADAPTER_ABI = "probpipe.transformed_descendant/v1"
_DESCENDANT_PROVIDER_ABI = "tensorflow_probability.substrates.jax.bijector.forward/v1"
_JAX_KEY_ABI = "jax.random.wrap_key_data/threefry2x32/v1"

WorkflowEvaluator = Literal["rowwise", "jax_vmap"]
WorkflowTransport = Literal[
    "local_inline",
    "local_thread",
    "prefect_task",
    "prefect_flow",
]


@dataclass(frozen=True, slots=True)
class WorkflowRngExecutionContract:
    """Canonical evaluator/transport and stochastic ABI requirements."""

    abi: str
    evaluator: WorkflowEvaluator
    transport: WorkflowTransport
    rng_abi: str
    sampling_abi: str
    provider_abis: tuple[str, ...]
    descendant_adapter_abis: tuple[str, ...]
    jax_key_abi: str
    plan_evaluation_mode: str | None


def make_execution_contract(
    *,
    evaluator: WorkflowEvaluator,
    transport: WorkflowTransport,
    stochastic_plan: _workflow_plan.StochasticPlan | None,
) -> WorkflowRngExecutionContract:
    """Build the exact execution contract for one planned route."""
    provider_abis = {_PROVIDER_ABI}
    descendant_adapter_abis: set[str] = set()
    if stochastic_plan is not None:
        for group in stochastic_plan.source_groups:
            for consumer in group.consumers:
                descriptor = consumer.descendant_descriptor
                if descriptor is None:
                    continue
                labels = dict(_iter_descriptor_fields(descriptor))
                provider = labels.get("provider_abi")
                if isinstance(provider, str):
                    provider_abis.add(provider)
                adapter = labels.get("descendant_adapter_abi")
                if isinstance(adapter, str):
                    descendant_adapter_abis.add(adapter)
    return WorkflowRngExecutionContract(
        abi=_EXECUTION_CONTRACT_ABI,
        evaluator=evaluator,
        transport=transport,
        rng_abi=_RNG_ABI,
        sampling_abi=_SAMPLING_ABI,
        provider_abis=tuple(sorted(provider_abis)),
        descendant_adapter_abis=tuple(sorted(descendant_adapter_abis)),
        jax_key_abi=_JAX_KEY_ABI,
        plan_evaluation_mode=(None if stochastic_plan is None else stochastic_plan.evaluation_mode),
    )


def supports_execution_contract(
    contract: WorkflowRngExecutionContract,
    stochastic_plan: _workflow_plan.StochasticPlan | None,
    *,
    jax_structure_supported: bool = True,
) -> bool:
    """Return whether one route satisfies all versioned RNG capabilities."""
    if not isinstance(contract, WorkflowRngExecutionContract):
        return False
    if (
        contract.abi != _EXECUTION_CONTRACT_ABI
        or contract.rng_abi != _RNG_ABI
        or contract.sampling_abi != _SAMPLING_ABI
        or contract.jax_key_abi != _JAX_KEY_ABI
        or contract.evaluator not in ("rowwise", "jax_vmap")
        or contract.transport
        not in ("local_inline", "local_thread", "prefect_task", "prefect_flow")
    ):
        return False
    if _PROVIDER_ABI not in contract.provider_abis:
        return False
    if any(
        provider not in (_PROVIDER_ABI, _DESCENDANT_PROVIDER_ABI)
        for provider in contract.provider_abis
    ):
        return False
    if any(adapter != _DESCENDANT_ADAPTER_ABI for adapter in contract.descendant_adapter_abis):
        return False
    if contract.evaluator == "jax_vmap":
        if not jax_structure_supported:
            return False
        evaluation_mode = (
            contract.plan_evaluation_mode
            if stochastic_plan is None
            else stochastic_plan.evaluation_mode
        )
        if evaluation_mode not in (None, "sampled"):
            return False
        if stochastic_plan is not None and any(
            group.execution_mode != "sampled" for group in stochastic_plan.source_groups
        ):
            return False
    return True


def execution_contract_abi() -> str:
    """Return the immutable route-neutral execution contract ABI."""
    return _EXECUTION_CONTRACT_ABI


def key_adapter_abi() -> str:
    """Return the immutable JAX key adapter ABI used by admitted routes."""
    return _JAX_KEY_ABI


def transport_for_execution_mode(mode: str) -> WorkflowTransport:
    """Map row-wise execution configuration to its canonical transport."""
    match mode:
        case "sequential":
            return "local_inline"
        case "thread":
            return "local_thread"
        case "prefect_task":
            return "prefect_task"
        case "prefect_flow":
            return "prefect_flow"
        case _:
            raise ValueError(f"unsupported workflow execution mode: {mode!r}")


def transport_for_workflow_kind(kind: WorkflowKind) -> WorkflowTransport:
    """Map a JAX evaluator's orchestration mode to canonical transport."""
    match kind:
        case WorkflowKind.TASK:
            return "prefect_task"
        case WorkflowKind.FLOW:
            return "prefect_flow"
        case _:
            return "local_inline"


def _iter_descriptor_fields(value: Any):
    if isinstance(value, tuple):
        if len(value) == 2 and isinstance(value[0], str):
            yield value[0], value[1]
        for item in value:
            yield from _iter_descriptor_fields(item)
