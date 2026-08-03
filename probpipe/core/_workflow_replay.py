"""Standalone workflow RNG replay admission and preflight validation."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from . import _workflow_context, _workflow_execution_contract
from ._workflow_callable import CallableAnchor
from ._workflow_errors import (
    ReplayCompatibilityError,
    ReplayUnsupportedCallableError,
)
from ._workflow_rng import RandomEventIdentity, encode_random_event
from .provenance import Provenance

_RNG_RECIPE_ABI = "probpipe.rng_recipe/v1"
_RNG_ABI = "ProbPipe-RNG-v1"
_REPLAY_ANCHOR_ABI = "probpipe.replay_anchor/v1"
_STOCHASTIC_PLAN_ABI = "probpipe.stochastic_plan/v1"
_CALLABLE_DEFINITION_ABI = "probpipe.callable_definition/v1"
_PROBPIPE_REPLAY_ABI = "probpipe.replay/v1"
_MANAGED_CHILD_POLICY_ABI = "probpipe.managed_child/v1"


@dataclass(frozen=True)
class _ExpectedReplayEvent:
    """Validated event identity and its parallel effect anchor."""

    occurrence_path: tuple[Any, ...]
    occurrence_kind: str
    source: tuple[Any, ...]
    unit: tuple[Any, ...]
    effect: dict[str, Any]
    encoded_identity: bytes


@dataclass
class _ReplayState:
    """One validated standalone replay scope."""

    provenance: Provenance
    root_words: tuple[int, int]
    occurrence_path: tuple[Any, ...]
    callable_anchor: dict[str, Any]
    canonical_plan: dict[str, Any]
    execution_capabilities: tuple[dict[str, Any], ...]
    expected_events: tuple[_ExpectedReplayEvent, ...]
    recorded_source: dict[str, Any]
    recorded_execution: tuple[dict[str, Any], ...]
    root_started: bool = False
    root_completed: bool = False
    root_failed: bool = False
    source_artifact_drift: bool = False
    actual_execution: list[dict[str, Any]] = field(default_factory=list)

    def validate_callable(self, current: CallableAnchor) -> None:
        """Require the current Function to match the strong recorded anchor."""
        controls = current.controls()
        if not current.supported:
            raise ReplayUnsupportedCallableError(
                "replay requires an importable module-level closure-free Python def; "
                f"the supplied Function is recorded as {current.form!r}"
            )
        recorded = self.callable_anchor
        if controls.get("module") != recorded.get("module") or controls.get(
            "qualname"
        ) != recorded.get("qualname"):
            raise ReplayCompatibilityError(
                "the supplied Function callable import anchor changed since recording"
            )
        if (
            controls.get("definition_abi") != recorded.get("definition_abi")
            or controls.get("python_replay_abi") != recorded.get("python_replay_abi")
            or controls.get("probpipe_replay_abi") != recorded.get("probpipe_replay_abi")
        ):
            raise ReplayCompatibilityError(
                "the supplied Function callable replay ABI is incompatible"
            )
        if controls != recorded:
            raise ReplayCompatibilityError(
                "the supplied Function callable definition changed since recording"
            )
        self.source_artifact_drift = current.diagnostics() != self.recorded_source

    def validate_plan(self, current: dict[str, Any]) -> None:
        """Require exact canonical lifting/direct-operation plan equality."""
        if current != self.canonical_plan:
            raise ReplayCompatibilityError(
                "the current stochastic plan differs from the recorded replay plan"
            )

    def validate_execution_contract(self, contract: Any) -> None:
        """Require the current route to satisfy a recorded route-neutral capability."""
        capability = _workflow_execution_contract.execution_capability_fields(contract)
        if capability not in self.execution_capabilities:
            raise ReplayCompatibilityError(
                "the current evaluator or transport cannot satisfy the recorded "
                "workflow RNG execution contract"
            )
        route = {
            "evaluator": contract.evaluator,
            "transport": contract.transport,
            "contract_abi": contract.abi,
        }
        if route not in self.actual_execution:
            self.actual_execution.append(route)

    def diagnostics(self) -> dict[str, Any]:
        """Return non-authoritative source and route drift observations."""
        recorded_routes = [
            {
                "evaluator": item.get("evaluator"),
                "transport": item.get("transport"),
                "contract_abi": item.get("contract_abi"),
            }
            for item in self.recorded_execution
        ]
        return {
            "source_artifact_drift": self.source_artifact_drift,
            "execution_drift": recorded_routes != self.actual_execution,
            "recorded_execution": recorded_routes,
            "current_execution": copy.deepcopy(self.actual_execution),
        }


@dataclass(frozen=True)
class _ReplayFunctionCall:
    """Root-call controller supplied to Function.__call__."""

    state: _ReplayState

    @property
    def occurrence_path(self) -> tuple[Any, ...]:
        return self.state.occurrence_path

    def validate_callable(self, anchor: CallableAnchor) -> None:
        self.state.validate_callable(anchor)


_ACTIVE_REPLAY_STATE: ContextVar[_ReplayState | None] = ContextVar(
    "probpipe_active_replay_state",
    default=None,
)
_REPLAY_FUNCTION_DEPTH: ContextVar[int] = ContextVar(
    "probpipe_replay_function_depth",
    default=0,
)


class _ReplayRunScope:
    """Synchronous implementation backing the public replay_run context."""

    def __init__(self, provenance: Provenance):
        self._provenance = provenance
        self._state: _ReplayState | None = None
        self._state_token: Token[_ReplayState | None] | None = None
        self._frame_scope: Any = None

    def __enter__(self) -> None:
        if self._state_token is not None:
            raise RuntimeError("replay_run context is already active")
        if _ACTIVE_REPLAY_STATE.get() is not None:
            raise ReplayCompatibilityError("replay_run contexts cannot be nested")
        if _workflow_context._capture_active_workflow_frame() is not None:
            raise ReplayCompatibilityError(
                "replay_run must be entered outside an active workflow_run"
            )
        state = _validate_provenance(self._provenance)
        frame_scope = _workflow_context._replay_workflow_frame(state.root_words)
        frame_scope.__enter__()
        try:
            token = _ACTIVE_REPLAY_STATE.set(state)
        except BaseException:
            frame_scope.__exit__(*sys_exc_info())
            raise
        self._state = state
        self._state_token = token
        self._frame_scope = frame_scope
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._state is None or self._state_token is None or self._frame_scope is None:
            raise RuntimeError("replay_run context is not active")
        state = self._state
        token = self._state_token
        frame_scope = self._frame_scope
        pending: BaseException | None = None
        if exc_type is None and not state.root_started:
            pending = ReplayCompatibilityError(
                "replay_run must contain exactly one top-level Function.__call__"
            )
        try:
            _ACTIVE_REPLAY_STATE.reset(token)
        finally:
            frame_scope.__exit__(exc_type, exc_value, traceback)
            self._state = None
            self._state_token = None
            self._frame_scope = None
        if pending is not None:
            raise pending


def replay_run(provenance: Provenance) -> _ReplayRunScope:
    """Restore and validate one recorded workflow-owned stochastic invocation."""
    return _ReplayRunScope(provenance)


def _replay_is_active() -> bool:
    return _ACTIVE_REPLAY_STATE.get() is not None


@contextmanager
def _function_replay_scope() -> Iterator[_ReplayFunctionCall | None]:
    """Bind at most one top-level Function call to an active replay scope."""
    state = _ACTIVE_REPLAY_STATE.get()
    if state is None:
        yield None
        return
    depth = _REPLAY_FUNCTION_DEPTH.get()
    if depth > 0:
        token = _REPLAY_FUNCTION_DEPTH.set(depth + 1)
        try:
            yield None
        finally:
            _REPLAY_FUNCTION_DEPTH.reset(token)
        return
    if state.root_started:
        raise ReplayCompatibilityError("replay_run accepts only one top-level Function.__call__")
    state.root_started = True
    token = _REPLAY_FUNCTION_DEPTH.set(1)
    try:
        yield _ReplayFunctionCall(state)
    except BaseException:
        state.root_failed = True
        raise
    else:
        state.root_completed = True
    finally:
        _REPLAY_FUNCTION_DEPTH.reset(token)


def _reject_function_apply() -> None:
    if _ACTIVE_REPLAY_STATE.get() is not None:
        raise ReplayCompatibilityError(
            "Function.apply is not permitted inside replay_run; call the Function normally"
        )


def _validate_active_plan(current: dict[str, Any]) -> None:
    state = _ACTIVE_REPLAY_STATE.get()
    if state is not None and _REPLAY_FUNCTION_DEPTH.get() == 1:
        state.validate_plan(current)


def _validate_active_execution_contract(contract: Any) -> None:
    state = _ACTIVE_REPLAY_STATE.get()
    if state is not None and _REPLAY_FUNCTION_DEPTH.get() == 1:
        state.validate_execution_contract(contract)


def _active_replay_diagnostics() -> dict[str, Any] | None:
    state = _ACTIVE_REPLAY_STATE.get()
    if state is None or _REPLAY_FUNCTION_DEPTH.get() != 1:
        return None
    return state.diagnostics()


def _validate_provenance(provenance: Provenance) -> _ReplayState:
    if not isinstance(provenance, Provenance):
        raise ReplayCompatibilityError("replay_run requires a Provenance RNG recipe")
    controls = provenance.controls
    randomness = _mapping(controls.get("randomness"), "randomness RNG recipe")
    replay = _mapping(controls.get("replay"), "replay anchor")
    if randomness.get("schema") != _RNG_RECIPE_ABI:
        raise ReplayCompatibilityError("unknown or missing workflow RNG recipe schema")
    if randomness.get("rng_abi") != _RNG_ABI:
        raise ReplayCompatibilityError("recorded workflow RNG ABI is incompatible")
    if replay.get("schema") != _REPLAY_ANCHOR_ABI:
        raise ReplayCompatibilityError("recorded replay anchor schema is incompatible")

    root_words = _root_words(randomness.get("root_words"))
    occurrence_path = _structural_tuple(
        randomness.get("occurrence_path"),
        field_name="randomness.occurrence_path",
    )
    standalone = _mapping(replay.get("standalone"), "replay.standalone")
    eligibility = standalone.get("eligibility")
    if eligibility != "supported":
        if eligibility == "nested_workflow_rng_execution":
            raise ReplayCompatibilityError(
                "standalone replay does not support a parent with nested automatic "
                "workflow randomness"
            )
        raise ReplayCompatibilityError("recorded standalone replay eligibility is invalid")

    callable_anchor = copy.deepcopy(dict(_mapping(replay.get("callable"), "replay.callable")))
    if callable_anchor.get("definition_abi") != _CALLABLE_DEFINITION_ABI:
        raise ReplayCompatibilityError("recorded callable definition ABI is incompatible")
    if callable_anchor.get("supported") is not True:
        form = callable_anchor.get("form", "unsupported")
        raise ReplayUnsupportedCallableError(
            "replay requires an importable module-level closure-free Python def; "
            f"the recorded Function uses {form!r}"
        )
    if callable_anchor.get("probpipe_replay_abi") != _PROBPIPE_REPLAY_ABI:
        raise ReplayCompatibilityError("recorded ProbPipe replay ABI is incompatible")
    for field_name in ("module", "qualname", "python_replay_abi", "sha256"):
        if not isinstance(callable_anchor.get(field_name), str):
            raise ReplayCompatibilityError(f"recorded callable anchor has invalid {field_name}")
    if not isinstance(callable_anchor.get("signature_and_templates"), dict):
        raise ReplayCompatibilityError(
            "recorded callable anchor has invalid signature_and_templates"
        )

    plan_anchor = _mapping(replay.get("plan"), "replay.plan")
    if plan_anchor.get("schema") != _STOCHASTIC_PLAN_ABI:
        raise ReplayCompatibilityError("recorded stochastic plan ABI is incompatible")
    canonical_plan = copy.deepcopy(
        dict(_mapping(plan_anchor.get("canonical_fields"), "replay.plan.canonical_fields"))
    )
    if canonical_plan.get("managed_child_policy") != _MANAGED_CHILD_POLICY_ABI:
        raise ReplayCompatibilityError("recorded managed-child replay policy is incompatible")
    if canonical_plan.get("key_ownership") != "automatic":
        raise ReplayCompatibilityError("recorded replay plan is not workflow-key-owned")

    compatibility = _mapping(replay.get("compatibility"), "replay.compatibility")
    if (
        compatibility.get("execution_contract")
        != _workflow_execution_contract.execution_contract_abi()
    ):
        raise ReplayCompatibilityError("recorded workflow RNG execution contract is incompatible")
    capabilities_raw = _list(compatibility.get("capabilities"), "compatibility.capabilities")
    capabilities = tuple(
        copy.deepcopy(dict(_mapping(item, "compatibility capability"))) for item in capabilities_raw
    )
    if not capabilities:
        raise ReplayCompatibilityError("recorded replay has no execution capability")

    events_raw = _list(randomness.get("events"), "randomness.events")
    effects_raw = _list(plan_anchor.get("expected_effects"), "replay.plan.expected_effects")
    expected_count = randomness.get("expected_event_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(events_raw)
        or len(events_raw) != len(effects_raw)
        or expected_count < 1
    ):
        raise ReplayCompatibilityError("recorded replay event count is inconsistent")
    expected_events = _expected_events(
        events_raw,
        effects_raw,
        outer_occurrence_path=occurrence_path,
    )

    diagnostics = provenance.diagnostics
    recorded_source = copy.deepcopy(
        dict(_mapping_or_empty(diagnostics.get("callable_source"), "callable_source"))
    )
    recorded_execution = tuple(
        copy.deepcopy(dict(_mapping(item, "execution diagnostic")))
        for item in _list_or_empty(diagnostics.get("execution"), "execution")
    )
    return _ReplayState(
        provenance=provenance,
        root_words=root_words,
        occurrence_path=occurrence_path,
        callable_anchor=callable_anchor,
        canonical_plan=canonical_plan,
        execution_capabilities=capabilities,
        expected_events=expected_events,
        recorded_source=recorded_source,
        recorded_execution=recorded_execution,
    )


def _expected_events(
    events: list[Any],
    effects: list[Any],
    *,
    outer_occurrence_path: tuple[Any, ...],
) -> tuple[_ExpectedReplayEvent, ...]:
    result: list[_ExpectedReplayEvent] = []
    seen: set[bytes] = set()
    for index, (raw_event, raw_effect) in enumerate(zip(events, effects, strict=True)):
        event = _mapping(raw_event, f"randomness.events[{index}]")
        effect = copy.deepcopy(dict(_mapping(raw_effect, f"replay.plan.expected_effects[{index}]")))
        occurrence_path = _structural_tuple(
            event.get("occurrence_path"),
            field_name=f"randomness.events[{index}].occurrence_path",
        )
        if occurrence_path[: len(outer_occurrence_path)] != outer_occurrence_path:
            raise ReplayCompatibilityError(
                "recorded replay event is outside its anchored occurrence path"
            )
        occurrence_kind = event.get("occurrence_kind")
        if occurrence_kind not in ("invocation", "operation"):
            raise ReplayCompatibilityError("recorded replay event occurrence kind is invalid")
        if event.get("key_ownership") != "automatic":
            raise ReplayCompatibilityError("recorded replay event is not workflow-key-owned")
        source = _structural_tuple(
            event.get("source"),
            field_name=f"randomness.events[{index}].source",
        )
        unit = _structural_tuple(
            event.get("unit"),
            field_name=f"randomness.events[{index}].unit",
        )
        _validate_effect(effect, index=index)
        encoded = encode_random_event(
            RandomEventIdentity(
                occurrence_path=occurrence_path,
                stochastic_source_id=source,
                logical_unit_id=unit,
            )
        )
        if encoded in seen:
            raise ReplayCompatibilityError("recorded replay contains a duplicate event identity")
        seen.add(encoded)
        result.append(
            _ExpectedReplayEvent(
                occurrence_path=occurrence_path,
                occurrence_kind=occurrence_kind,
                source=source,
                unit=unit,
                effect=effect,
                encoded_identity=encoded,
            )
        )
    return tuple(result)


def _validate_effect(effect: dict[str, Any], *, index: int) -> None:
    for field_name in (
        "operation_kind",
        "execution_mode",
        "sampling_abi",
        "provider_abi",
    ):
        if not isinstance(effect.get(field_name), str) or not effect[field_name]:
            raise ReplayCompatibilityError(
                f"recorded replay effect {index} has invalid {field_name}"
            )
    sample_shape = effect.get("sample_shape")
    if sample_shape is not None:
        if not isinstance(sample_shape, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in sample_shape
        ):
            raise ReplayCompatibilityError(
                f"recorded replay effect {index} has invalid sample_shape"
            )


def _root_words(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(word, bool) or not isinstance(word, int) or not 0 <= word <= 0xFFFFFFFF
            for word in value
        )
    ):
        raise ReplayCompatibilityError(
            "recorded randomness.root_words must contain two uint32 values"
        )
    return value[0], value[1]


def _structural_tuple(value: Any, *, field_name: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ReplayCompatibilityError(f"recorded {field_name} must be a JSON sequence")

    def convert(item: Any) -> Any:
        if isinstance(item, list):
            return tuple(convert(child) for child in item)
        if isinstance(item, str):
            return item
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 2**64 - 1:
            raise ReplayCompatibilityError(
                f"recorded {field_name} contains an invalid structural value"
            )
        return item

    return tuple(convert(item) for item in value)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayCompatibilityError(f"recorded {field_name} must be a mapping")
    return value


def _mapping_or_empty(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _mapping(value, field_name)


def _list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReplayCompatibilityError(f"recorded {field_name} must be a sequence")
    return value


def _list_or_empty(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    return _list(value, field_name)


def sys_exc_info() -> tuple[type[BaseException] | None, BaseException | None, Any]:
    """Return exception state without importing sys into the public namespace."""
    import sys

    return sys.exc_info()
