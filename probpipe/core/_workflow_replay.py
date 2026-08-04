"""Standalone workflow RNG replay admission and preflight validation."""

from __future__ import annotations

import copy
import sys
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import Lock
from types import TracebackType
from typing import Any

from . import _workflow_context, _workflow_execution_contract
from ._workflow_callable import CallableAnchor
from ._workflow_errors import (
    ReplayCompatibilityError,
    ReplayUnsupportedCallableError,
)
from ._workflow_managed import (
    _MANAGED_WORK_ITEM_ABI,
    ManagedAttemptState,
    ManagedEffectClaim,
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

    def managed_effect(self) -> ManagedEffectClaim:
        """Return the transport-safe expected effect descriptor."""
        sample_shape = self.effect["sample_shape"]
        return ManagedEffectClaim(
            occurrence_path=self.occurrence_path,
            occurrence_kind=self.occurrence_kind,
            stochastic_source_id=self.source,
            logical_unit_id=self.unit,
            operation_kind=self.effect["operation_kind"],
            execution_mode=self.effect["execution_mode"],
            sample_shape=None if sample_shape is None else tuple(sample_shape),
            sampling_abi=self.effect["sampling_abi"],
            provider_abi=self.effect["provider_abi"],
            record_path=tuple(self.effect["record_path"]),
            descendant_descriptor=_descriptor_tuple(self.effect["descendant_descriptor"]),
        )


@dataclass
class _ReplayEventClaim:
    """Transient ownership for one expected logical event."""

    expected: _ExpectedReplayEvent
    direct_claimed: bool = False
    work_item_token: bytes | None = None
    attempt_tokens: set[bytes] = field(default_factory=set)
    successful: bool = False


@dataclass
class _ReplayState:
    """One validated standalone replay scope."""

    provenance: Provenance
    root_words: tuple[int, int]
    occurrence_path: tuple[Any, ...]
    callable_anchor: dict[str, Any]
    canonical_plan: dict[str, Any]
    execution_contract_abi: str
    sampling_abis: tuple[str, ...]
    provider_abis: tuple[str, ...]
    descendant_adapter_abis: tuple[str, ...]
    key_adapter_abi: str
    expected_events: tuple[_ExpectedReplayEvent, ...]
    recorded_source: dict[str, Any]
    recorded_execution: tuple[dict[str, Any], ...]
    root_started: bool = False
    root_completed: bool = False
    root_failed: bool = False
    source_artifact_drift: bool = False
    requested_dispatch: str | None = None
    requested_workflow_kind: str | None = None
    actual_execution: list[dict[str, Any]] = field(default_factory=list)
    claims: dict[bytes, _ReplayEventClaim] = field(init=False)
    claims_lock: Any = field(default_factory=Lock, repr=False)

    def __post_init__(self) -> None:
        self.claims = {
            expected.encoded_identity: _ReplayEventClaim(expected)
            for expected in self.expected_events
        }

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
        """Require the current route to satisfy the one route-neutral contract."""
        if (
            contract.abi != self.execution_contract_abi
            or contract.rng_abi != _RNG_ABI
            or contract.jax_key_abi != self.key_adapter_abi
            or (
                self.canonical_plan.get("kind") != "direct_operation"
                and tuple(contract.descendant_adapter_abis) != self.descendant_adapter_abis
            )
        ):
            raise ReplayCompatibilityError(
                "the current evaluator or transport cannot satisfy the recorded "
                "workflow RNG execution contract"
            )
        if self.requested_dispatch is None or self.requested_workflow_kind is None:
            raise ReplayCompatibilityError(
                "the current Function did not declare its requested execution route"
            )
        route = {
            "requested_dispatch": self.requested_dispatch,
            "requested_workflow_kind": self.requested_workflow_kind,
            "resolved_evaluator": contract.evaluator,
            "resolved_transport": contract.transport,
            "contract_abi": contract.abi,
        }
        if route not in self.actual_execution:
            self.actual_execution.append(route)

    def validate_effect_plan(self, plan: Any) -> None:
        """Match one planned effect before its stochastic occurrence is committed."""
        current_effect = _effect_plan_anchor(plan)
        if not any(
            expected.source == plan.event.stochastic_source_id
            and expected.unit == plan.event.logical_unit_id
            and expected.effect == current_effect
            for expected in self.expected_events
        ):
            raise ReplayCompatibilityError(
                "unexpected replay event: the current stochastic effect plan differs "
                "from the recorded replay plan"
            )

    def record_requested_execution(self, dispatch: str, workflow_kind: str) -> None:
        """Capture current request diagnostics before route resolution."""
        requested = (dispatch, workflow_kind)
        existing = (self.requested_dispatch, self.requested_workflow_kind)
        if existing != (None, None) and existing != requested:
            raise ReplayCompatibilityError(
                "one replayed Function declared conflicting execution requests"
            )
        self.requested_dispatch, self.requested_workflow_kind = requested

    def diagnostics(self) -> dict[str, Any]:
        """Return non-authoritative source and route drift observations."""
        recorded_routes = [
            {
                "requested_dispatch": item.get("requested_dispatch"),
                "requested_workflow_kind": item.get("requested_workflow_kind"),
                "resolved_evaluator": item.get("resolved_evaluator"),
                "resolved_transport": item.get("resolved_transport"),
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

    def claim_effect(
        self,
        effect: ManagedEffectClaim,
        *,
        attempt: ManagedAttemptState | None,
    ) -> None:
        """Validate and atomically claim an event before key derivation."""
        encoded = _encoded_effect_identity(effect)
        with self.claims_lock:
            claim = self.claims.get(encoded)
            if claim is None:
                raise ReplayCompatibilityError(
                    "workflow execution requested an unexpected replay event"
                )
            _require_matching_effect(claim.expected, effect)
            if attempt is None:
                if claim.direct_claimed or claim.work_item_token is not None:
                    raise ReplayCompatibilityError(
                        "workflow execution duplicated an already claimed replay event"
                    )
                claim.direct_claimed = True
                return

            work_item_token = attempt.work_item_token.value
            if claim.direct_claimed:
                raise ReplayCompatibilityError(
                    "a managed work item cannot reuse a directly claimed replay event"
                )
            if claim.work_item_token is None:
                claim.work_item_token = work_item_token
            elif claim.work_item_token != work_item_token:
                raise ReplayCompatibilityError(
                    "a different managed work-item token attempted to reuse a replay event"
                )
            if attempt.attempt_token in claim.attempt_tokens:
                raise ReplayCompatibilityError(
                    "one managed attempt duplicated a replay event claim"
                )
            claim.attempt_tokens.add(attempt.attempt_token)

    def assert_all_events_claimed(self) -> None:
        """Reject a successful invocation that omitted any expected event."""
        with self.claims_lock:
            missing_count = sum(not claim.successful for claim in self.claims.values())
        if missing_count:
            raise ReplayCompatibilityError(
                f"workflow replay completed with {missing_count} missing expected event(s)"
            )

    def mark_successful_effects(self, effects: tuple[ManagedEffectClaim, ...]) -> None:
        """Confirm the exact events retained by the successful root invocation."""
        with self.claims_lock:
            for effect in effects:
                encoded = _encoded_effect_identity(effect)
                claim = self.claims.get(encoded)
                if claim is None:
                    raise ReplayCompatibilityError(
                        "workflow execution completed with an unexpected replay event"
                    )
                _require_matching_effect(claim.expected, effect)
                if not claim.direct_claimed and claim.work_item_token is None:
                    raise ReplayCompatibilityError(
                        "workflow execution completed an unclaimed replay event"
                    )
                claim.successful = True

    def expected_effects_for_unit(
        self,
        parent_occurrence_path: tuple[Any, ...],
        unit_segment: tuple[Any, ...],
    ) -> tuple[ManagedEffectClaim, ...]:
        """Return the exact replay namespace assigned to one remote work item."""
        prefix = (*parent_occurrence_path, unit_segment)
        return tuple(
            expected.managed_effect()
            for expected in self.expected_events
            if expected.occurrence_path[: len(prefix)] == prefix
        )


@dataclass
class _RemoteReplayClaims:
    """Worker-local pre-derivation validator for one assigned namespace."""

    expected_by_identity: dict[bytes, ManagedEffectClaim]
    attempt: ManagedAttemptState
    claimed: set[bytes] = field(default_factory=set)

    def validate_plan(self, plan: Any) -> None:
        """Validate a transported plan before its child occurrence is committed."""
        current_effect = _effect_plan_anchor(plan)
        if not any(
            effect.stochastic_source_id == plan.event.stochastic_source_id
            and effect.logical_unit_id == plan.event.logical_unit_id
            and _managed_effect_anchor(effect) == current_effect
            for effect in self.expected_by_identity.values()
        ):
            raise ReplayCompatibilityError(
                "the remote stochastic effect plan differs from its assigned event namespace"
            )

    def claim(
        self,
        effect: ManagedEffectClaim,
        attempt: ManagedAttemptState | None,
    ) -> None:
        if attempt != self.attempt:
            raise ReplayCompatibilityError(
                "remote replay event does not belong to its assigned work-item attempt"
            )
        encoded = _encoded_effect_identity(effect)
        expected = self.expected_by_identity.get(encoded)
        if expected is None:
            raise ReplayCompatibilityError(
                "remote workflow execution requested an unexpected replay event"
            )
        if expected != effect:
            raise ReplayCompatibilityError(
                "remote workflow replay effect differs from its assigned event namespace"
            )
        if encoded in self.claimed:
            raise ReplayCompatibilityError(
                "one remote managed attempt duplicated a replay event claim"
            )
        self.claimed.add(encoded)


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
_REMOTE_REPLAY_CLAIMS: ContextVar[_RemoteReplayClaims | None] = ContextVar(
    "probpipe_remote_replay_claims",
    default=None,
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
            frame_scope.__exit__(*sys.exc_info())
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
        if exc_type is None:
            if not state.root_started:
                pending = ReplayCompatibilityError(
                    "replay_run must contain exactly one top-level Function.__call__"
                )
            elif not state.root_completed:
                pending = ReplayCompatibilityError(
                    "the top-level Function call in replay_run did not complete successfully"
                )
        try:
            _ACTIVE_REPLAY_STATE.reset(token)
        finally:
            try:
                frame_scope.__exit__(exc_type, exc_value, traceback)
            finally:
                self._state = None
                self._state_token = None
                self._frame_scope = None
        if pending is not None:
            raise pending


def replay_run(provenance: Provenance) -> _ReplayRunScope:
    """Replay one recorded workflow-owned stochastic invocation.

    Parameters
    ----------
    provenance : Provenance
        Provenance from a successful workflow-owned stochastic result. The
        record must contain a compatible RNG recipe and replay anchor.

    Returns
    -------
    context manager
        A synchronous scope in which exactly one top-level
        :meth:`Function.__call__ <probpipe.Function.__call__>` must run.

    Raises
    ------
    ReplayCompatibilityError
        If the record, stochastic plan, execution capability, or observed
        events are incompatible with the replay attempt.
    ReplayUnsupportedCallableError
        If the recorded or supplied callable lacks a strong replay anchor.

    Notes
    -----
    Replay restores ProbPipe's structural RNG root and validates the current
    call before deriving keys. It does not load user code or inputs from the
    provenance record. The caller must invoke the same supported Function
    definition with compatible arguments inside the scope.
    """
    return _ReplayRunScope(provenance)


def _replay_is_active() -> bool:
    return _ACTIVE_REPLAY_STATE.get() is not None


def _capture_active_replay_state() -> _ReplayState | None:
    return _ACTIVE_REPLAY_STATE.get()


@contextmanager
def _function_replay_scope() -> Generator[_ReplayFunctionCall | None, None, None]:
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
        try:
            state.assert_all_events_claimed()
        except BaseException:
            state.root_failed = True
            raise
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


def _record_active_requested_execution(dispatch: str, workflow_kind: str) -> None:
    state = _ACTIVE_REPLAY_STATE.get()
    if state is not None and _REPLAY_FUNCTION_DEPTH.get() == 1:
        state.record_requested_execution(dispatch, workflow_kind)


def _active_replay_diagnostics() -> dict[str, Any] | None:
    state = _ACTIVE_REPLAY_STATE.get()
    if state is None or _REPLAY_FUNCTION_DEPTH.get() != 1:
        return None
    return state.diagnostics()


def _claim_effect_before_derivation(
    effect: ManagedEffectClaim,
    *,
    attempt: ManagedAttemptState | None,
) -> None:
    """Claim a local or transported event before deriving its key."""
    remote = _REMOTE_REPLAY_CLAIMS.get()
    if remote is not None:
        remote.claim(effect, attempt)
        return
    state = _ACTIVE_REPLAY_STATE.get()
    if state is not None:
        state.claim_effect(effect, attempt=attempt)


def _validate_effect_plan_before_commit(plan: Any) -> None:
    """Validate effect semantics without allocating an occurrence or deriving a key."""
    remote = _REMOTE_REPLAY_CLAIMS.get()
    if remote is not None:
        remote.validate_plan(plan)
        return
    state = _ACTIVE_REPLAY_STATE.get()
    if state is not None:
        state.validate_effect_plan(plan)


def _expected_effects_for_managed_unit(
    parent_occurrence_path: tuple[Any, ...],
    unit_segment: tuple[Any, ...],
) -> tuple[ManagedEffectClaim, ...] | None:
    state = _ACTIVE_REPLAY_STATE.get()
    if state is None:
        return None
    return state.expected_effects_for_unit(parent_occurrence_path, unit_segment)


@contextmanager
def _remote_replay_claim_scope(
    expected: tuple[ManagedEffectClaim, ...] | None,
    attempt: ManagedAttemptState,
) -> Generator[None, None, None]:
    """Install a worker-local expected namespace when replay authority exists."""
    if expected is None:
        yield
        return
    registry = _RemoteReplayClaims(
        expected_by_identity={_encoded_effect_identity(effect): effect for effect in expected},
        attempt=attempt,
    )
    token = _REMOTE_REPLAY_CLAIMS.set(registry)
    try:
        yield
    finally:
        _REMOTE_REPLAY_CLAIMS.reset(token)


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
    _validate_function_occurrence_path(occurrence_path)
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

    compatibility = _mapping(replay.get("compatibility"), "replay.compatibility")
    expected_compatibility_fields = {
        "execution_contract",
        "sampling_abi",
        "provider_abi",
        "descendant_adapter_abi",
        "key_adapter_abi",
    }
    if set(compatibility) != expected_compatibility_fields:
        raise ReplayCompatibilityError(
            "recorded replay compatibility fields do not match the version-1 schema"
        )
    execution_contract_abi = compatibility.get("execution_contract")
    if execution_contract_abi != _workflow_execution_contract.execution_contract_abi():
        raise ReplayCompatibilityError("recorded workflow RNG execution contract is incompatible")
    sampling_abis = _abi_sequence(compatibility.get("sampling_abi"), "sampling ABI")
    provider_abis = _abi_sequence(compatibility.get("provider_abi"), "provider ABI")
    descendant_adapter_abis = _abi_sequence(
        compatibility.get("descendant_adapter_abi"),
        "descendant-adapter ABI",
    )
    key_adapter_abi = compatibility.get("key_adapter_abi")
    if key_adapter_abi != _workflow_execution_contract.key_adapter_abi():
        raise ReplayCompatibilityError("recorded workflow key-adapter ABI is incompatible")

    compatibility_material = (canonical_plan, [event.effect for event in expected_events])
    expected_sampling_abis = _ordered_unique(
        [event.effect["sampling_abi"] for event in expected_events]
        + list(_iter_named_abi(compatibility_material, "sampling_abi"))
    )
    expected_provider_abis = _ordered_unique(
        [event.effect["provider_abi"] for event in expected_events]
        + list(_iter_named_abi(compatibility_material, "provider_abi"))
    )
    expected_descendant_adapter_abis = _ordered_unique(
        list(_iter_named_abi(compatibility_material, "descendant_adapter_abi"))
    )
    if sampling_abis != expected_sampling_abis:
        raise ReplayCompatibilityError(
            "recorded sampling ABI fields disagree with the expected effects and plan"
        )
    if provider_abis != expected_provider_abis:
        raise ReplayCompatibilityError(
            "recorded provider ABI fields disagree with the expected effects and plan"
        )
    if descendant_adapter_abis != expected_descendant_adapter_abis:
        raise ReplayCompatibilityError(
            "recorded descendant-adapter ABI fields disagree with the canonical plan"
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
        execution_contract_abi=execution_contract_abi,
        sampling_abis=sampling_abis,
        provider_abis=provider_abis,
        descendant_adapter_abis=descendant_adapter_abis,
        key_adapter_abi=key_adapter_abi,
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
        _validate_occurrence_path(occurrence_path)
        if occurrence_path[: len(outer_occurrence_path)] != outer_occurrence_path:
            raise ReplayCompatibilityError(
                "recorded replay event is outside its anchored occurrence_path"
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
    expected_fields = {
        "operation_kind",
        "execution_mode",
        "sample_shape",
        "sampling_abi",
        "provider_abi",
        "record_path",
        "descendant_descriptor",
    }
    if set(effect) != expected_fields:
        raise ReplayCompatibilityError(f"recorded replay effect {index} has incompatible fields")
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
    record_path = effect.get("record_path")
    if not isinstance(record_path, list) or any(
        not isinstance(field, str) for field in record_path
    ):
        raise ReplayCompatibilityError(f"recorded replay effect {index} has invalid record_path")
    _validate_descriptor_value(effect.get("descendant_descriptor"), index=index)


def _encoded_effect_identity(effect: ManagedEffectClaim) -> bytes:
    return encode_random_event(
        RandomEventIdentity(
            occurrence_path=effect.occurrence_path,
            stochastic_source_id=effect.stochastic_source_id,
            logical_unit_id=effect.logical_unit_id,
        )
    )


def _require_matching_effect(
    expected: _ExpectedReplayEvent,
    actual: ManagedEffectClaim,
) -> None:
    if expected.occurrence_kind != actual.occurrence_kind:
        raise ReplayCompatibilityError(
            "workflow replay event occurrence kind differs from the recorded recipe"
        )
    current_effect = _managed_effect_anchor(actual)
    if current_effect != expected.effect:
        raise ReplayCompatibilityError(
            "workflow replay effect or provider ABI differs from the recorded plan"
        )


def _effect_plan_anchor(plan: Any) -> dict[str, Any]:
    return {
        "operation_kind": plan.operation_kind,
        "execution_mode": plan.execution_mode,
        "sample_shape": None if plan.sample_shape is None else list(plan.sample_shape),
        "sampling_abi": plan.sampling_abi,
        "provider_abi": plan.provider_abi,
        "record_path": list(plan.record_path),
        "descendant_descriptor": _effect_json_value(plan.descendant_descriptor),
    }


def _managed_effect_anchor(effect: ManagedEffectClaim) -> dict[str, Any]:
    return {
        "operation_kind": effect.operation_kind,
        "execution_mode": effect.execution_mode,
        "sample_shape": None if effect.sample_shape is None else list(effect.sample_shape),
        "sampling_abi": effect.sampling_abi,
        "provider_abi": effect.provider_abi,
        "record_path": list(effect.record_path),
        "descendant_descriptor": _effect_json_value(effect.descendant_descriptor),
    }


def _effect_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_effect_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        f"stochastic effect descriptor contains unsupported value {type(value).__name__}"
    )


def _descriptor_tuple(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return tuple(_descriptor_item(item) for item in value)


def _descriptor_item(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_descriptor_item(item) for item in value)
    return value


def _validate_descriptor_value(value: Any, *, index: int) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise ReplayCompatibilityError(
            f"recorded replay effect {index} has invalid descendant_descriptor"
        )

    def validate(item: Any) -> bool:
        if isinstance(item, list):
            return all(validate(child) for child in item)
        return item is None or isinstance(item, (str, bool, int, float))

    if not validate(value):
        raise ReplayCompatibilityError(
            f"recorded replay effect {index} has invalid descendant_descriptor"
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


def _validate_function_occurrence_path(path: tuple[Any, ...]) -> None:
    """Reject paths that cannot identify a committed public Function call."""
    _validate_occurrence_path(path)
    index = 0
    while index < len(path) and path[index][0] == "scope":
        index += 1
    if index == len(path) or path[index][0] != "invocation":
        raise ReplayCompatibilityError(
            "recorded randomness.occurrence_path has no root Function invocation"
        )
    index += 1

    while index < len(path):
        if path[index][0] != "managed-unit":
            raise ReplayCompatibilityError(
                "recorded randomness.occurrence_path has a dangling Function segment"
            )
        index += 1
        if index == len(path) or path[index][0] != "child":
            raise ReplayCompatibilityError(
                "recorded randomness.occurrence_path has a dangling managed unit"
            )
        index += 1

        scope_start = index
        while index < len(path) and path[index][0] == "scope":
            index += 1
        if index != scope_start:
            if index == len(path) or path[index][0] != "invocation":
                raise ReplayCompatibilityError(
                    "recorded randomness.occurrence_path has a dangling nested scope"
                )
            index += 1


def _validate_occurrence_path(path: tuple[Any, ...]) -> None:
    """Validate the closed version-1 occurrence-path segment grammar."""
    if not path:
        raise ReplayCompatibilityError("recorded randomness.occurrence_path must not be empty")
    for segment in path:
        if not isinstance(segment, tuple) or not segment:
            raise ReplayCompatibilityError(
                "recorded randomness.occurrence_path contains an invalid path segment"
            )
        tag = segment[0]
        if tag in ("scope", "invocation", "operation", "child"):
            if len(segment) != 2 or not _is_nonnegative_int(segment[1]):
                raise ReplayCompatibilityError(
                    f"recorded randomness.occurrence_path contains an invalid {tag!r} segment"
                )
            continue
        if tag == "managed-unit":
            _validate_managed_unit_segment(segment)
            continue
        raise ReplayCompatibilityError(
            "recorded randomness.occurrence_path contains an unknown path segment"
        )
    if path[-1][0] not in ("invocation", "operation", "child"):
        raise ReplayCompatibilityError(
            "recorded randomness.occurrence_path does not end at a stochastic occurrence"
        )


def _validate_managed_unit_segment(segment: tuple[Any, ...]) -> None:
    if len(segment) < 3 or segment[1] != _MANAGED_WORK_ITEM_ABI:
        raise ReplayCompatibilityError(
            "recorded randomness.occurrence_path contains an incompatible managed unit"
        )
    layout = segment[2]
    if layout == "point":
        valid = len(segment) == 4 and segment[3] == 0
    elif layout == "sweep-cell":
        valid = len(segment) >= 4 and all(_is_nonnegative_int(item) for item in segment[3:])
    elif layout == "lifted-evaluation":
        valid = (
            len(segment) == 5
            and _is_logical_unit_id(segment[3])
            and _is_nonnegative_int(segment[4])
        )
    else:
        valid = False
    if not valid:
        raise ReplayCompatibilityError(
            "recorded randomness.occurrence_path contains an invalid managed unit"
        )


def _is_logical_unit_id(value: Any) -> bool:
    if not isinstance(value, tuple) or not value:
        return False
    if value[0] == "singleton":
        return len(value) == 1
    return (
        value[0] == "cell"
        and len(value) >= 2
        and all(_is_nonnegative_int(item) for item in value[1:])
    )


def _is_nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


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


def _abi_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    items = _list(value, field_name)
    if any(not isinstance(item, str) or not item for item in items):
        raise ReplayCompatibilityError(f"recorded {field_name} must contain ABI strings")
    result = tuple(items)
    if len(set(result)) != len(result):
        raise ReplayCompatibilityError(f"recorded {field_name} contains duplicate entries")
    return result


def _ordered_unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _iter_named_abi(value: Any, field_name: str):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == field_name and isinstance(item, str):
                yield item
            yield from _iter_named_abi(item, field_name)
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and value[0] == field_name and isinstance(value[1], str):
            yield value[1]
        for item in value:
            yield from _iter_named_abi(item, field_name)
