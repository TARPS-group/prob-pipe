"""Canonical provenance recipes for successful workflow RNG invocations."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from . import _workflow_broker, _workflow_plan
from ._workflow_managed import ManagedEffectClaim
from ._workflow_rng import RandomEventIdentity, encode_random_event
from .config import ProvenanceMode, provenance_config

_RNG_RECIPE_ABI = "probpipe.rng_recipe/v1"
_REPLAY_ANCHOR_ABI = "probpipe.replay_anchor/v1"
_STOCHASTIC_PLAN_ABI = "probpipe.stochastic_plan/v1"
_MANAGED_CHILD_POLICY_ABI = "probpipe.managed_child/v1"


def provenance_recipe_fields(
    stochastic_plan: _workflow_plan.StochasticPlan | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return exact controls/diagnostics for one successful active invocation."""
    if provenance_config.mode is ProvenanceMode.OFF:
        return {}, {}
    snapshot = _workflow_broker._snapshot_active_recipe_state()
    if snapshot is None:
        return {}, {}

    effects = _sort_effects(snapshot.effects, stochastic_plan)
    nested_automatic_parent = any(
        (
            effect.occurrence_kind == "invocation"
            and effect.occurrence_path != snapshot.occurrence_path
        )
        or sum(
            isinstance(segment, tuple) and len(segment) > 0 and segment[0] == "managed-unit"
            for segment in effect.occurrence_path
        )
        > 1
        for effect in effects
    )
    compatibility_contracts = _unique_json_values(
        [_contract_controls(contract) for contract in snapshot.execution_contracts]
    )
    execution_diagnostics = [
        {
            "evaluator": contract.evaluator,
            "transport": contract.transport,
            "contract_abi": contract.abi,
        }
        for contract in snapshot.execution_contracts
    ]
    anchor = snapshot.callable_anchor
    callable_controls = (
        anchor.controls()
        if anchor is not None
        else {
            "supported": False,
            "module": None,
            "qualname": None,
            "definition_abi": "probpipe.callable_definition/v1",
            "form": "missing_function_anchor",
        }
    )
    random_recipe = {
        "schema": _RNG_RECIPE_ABI,
        "rng_abi": "ProbPipe-RNG-v1",
        "root_words": list(snapshot.root_words),
        "occurrence_path": _json_value(snapshot.occurrence_path),
        "events": [_serialize_random_event(effect) for effect in effects],
        "expected_event_count": len(effects),
    }
    replay_anchor = {
        "schema": _REPLAY_ANCHOR_ABI,
        "standalone": {
            "eligibility": (
                "nested_workflow_rng_execution" if nested_automatic_parent else "supported"
            ),
            "restriction": ("nested_automatic_function" if nested_automatic_parent else None),
        },
        "callable": callable_controls,
        "plan": {
            "schema": _STOCHASTIC_PLAN_ABI,
            "canonical_fields": serialize_stochastic_plan(stochastic_plan),
            "expected_effects": [_serialize_effect_anchor(effect) for effect in effects],
        },
        "compatibility": {
            "execution_contract": "probpipe.workflow_rng_execution/v1",
            "capabilities": compatibility_contracts,
            "sampling_abi": _ordered_unique([effect.sampling_abi for effect in effects]),
            "provider_abi": _ordered_unique([effect.provider_abi for effect in effects]),
        },
    }
    return (
        {"randomness": random_recipe, "replay": replay_anchor},
        {
            "rng_origin": snapshot.rng_origin,
            "callable_source": anchor.diagnostics() if anchor is not None else {},
            "execution": execution_diagnostics,
        },
    )


def serialize_stochastic_plan(
    plan: _workflow_plan.StochasticPlan | None,
) -> dict[str, Any]:
    """Serialize only canonical plan fields under the version-1 plan ABI."""
    if plan is None:
        return {
            "kind": "direct_operation",
            "evaluation_mode": None,
            "arg_refs": [],
            "source_groups": [],
            "logical_units": [],
            "n_broadcast_samples": None,
            "sample_shape": None,
            "exact_group_order": [],
            "exact_combination_order": [],
            "repetitions_per_combination": None,
            "n_evaluations": None,
            "managed_child_policy": _MANAGED_CHILD_POLICY_ABI,
            "key_ownership": "automatic",
        }
    return {
        "kind": "function_lifting",
        "evaluation_mode": plan.evaluation_mode,
        "arg_refs": [_serialize_ref(ref) for ref in plan.arg_refs],
        "source_groups": [
            {
                "index": group.index,
                "source_id": _json_value(group.stochastic_source_id),
                "execution_mode": group.execution_mode,
                "exact_size": group.exact_size,
                "consumers": [
                    {
                        "arg_ref": _serialize_ref(consumer.arg_ref),
                        "record_path": list(consumer.record_path),
                        "descendant_descriptor": _json_value(consumer.descendant_descriptor),
                    }
                    for consumer in group.consumers
                ],
            }
            for group in plan.source_groups
        ],
        "logical_units": [
            {
                "layout": unit.layout,
                "flat_index": unit.flat_index,
                "coordinates": list(unit.coordinates),
                "logical_unit_id": _json_value(unit.logical_unit_id),
            }
            for unit in plan.logical_units
        ],
        "n_broadcast_samples": plan.n_broadcast_samples,
        "sample_shape": _json_value(plan.sample_shape),
        "exact_group_order": list(plan.exact_group_order),
        "exact_combination_order": _json_value(plan.exact_combination_order),
        "repetitions_per_combination": plan.repetitions_per_combination,
        "n_evaluations": plan.n_evaluations,
        "managed_child_policy": _MANAGED_CHILD_POLICY_ABI,
        "key_ownership": "automatic" if plan.random_events else "none",
    }


def _serialize_ref(ref: Any) -> dict[str, Any]:
    return {
        "parameter_name": ref.parameter_name,
        "subscript": ref.subscript,
        "label": ref.label,
    }


def _serialize_random_event(effect: ManagedEffectClaim) -> dict[str, Any]:
    return {
        "occurrence_path": _json_value(effect.occurrence_path),
        "occurrence_kind": effect.occurrence_kind,
        "source": _json_value(effect.stochastic_source_id),
        "unit": _json_value(effect.logical_unit_id),
        "key_ownership": "automatic",
    }


def _serialize_effect_anchor(effect: ManagedEffectClaim) -> dict[str, Any]:
    return {
        "operation_kind": effect.operation_kind,
        "execution_mode": effect.execution_mode,
        "sample_shape": _json_value(effect.sample_shape),
        "sampling_abi": effect.sampling_abi,
        "provider_abi": effect.provider_abi,
    }


def _sort_effects(
    effects: tuple[ManagedEffectClaim, ...],
    plan: _workflow_plan.StochasticPlan | None,
) -> tuple[ManagedEffectClaim, ...]:
    plan_order = (
        {
            (event.stochastic_source_id, event.logical_unit_id): index
            for index, event in enumerate(plan.random_events)
        }
        if plan is not None
        else {}
    )

    def key(effect: ManagedEffectClaim):
        identity = RandomEventIdentity(
            occurrence_path=effect.occurrence_path,
            stochastic_source_id=effect.stochastic_source_id,
            logical_unit_id=effect.logical_unit_id,
        )
        return (
            _canonical_json(_json_value(effect.occurrence_path)),
            plan_order.get(
                (effect.stochastic_source_id, effect.logical_unit_id),
                len(plan_order),
            ),
            encode_random_event(identity),
        )

    return tuple(sorted(effects, key=key))


def _contract_controls(contract: Any) -> dict[str, Any]:
    raw = asdict(contract)
    raw.pop("evaluator", None)
    raw.pop("transport", None)
    return _json_value(raw)


def _unique_json_values(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[bytes, dict[str, Any]] = {}
    for value in values:
        unique.setdefault(_canonical_json(value), value)
    return list(unique.values())


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"workflow recipe contains unsupported canonical value {type(value).__name__}")
