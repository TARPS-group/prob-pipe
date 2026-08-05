"""Private automatic-key broker for workflow-owned stochastic effects."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ..custom_types import PRNGKey
from . import _workflow_context
from ._workflow_rng import (
    _RandomEventPath,
    _validate_random_event_value,
)

if TYPE_CHECKING:
    from ._workflow_callable import CallableAnchor
    from ._workflow_execution_contract import WorkflowRngExecutionContract
from ._workflow_managed import (
    ManagedAttemptState,
    ManagedClaimReport,
    ManagedEffectClaim,
    ManagedParentEnvelope,
    ManagedUnitFrame,
    ManagedWorkItem,
    ManagedWorkItemToken,
    _CanonicalDescriptor,
    _validate_stochastic_effect_fields,
)

_OccurrenceKind = Literal["invocation", "operation"]
_StructuralRngId = _RandomEventPath


class _ManagedUnitStatus(Enum):
    """Lifecycle state of one parent-issued managed unit."""

    ISSUED = "issued"
    ACTIVE = "active"
    JOINED = "joined"
    CANCELLED = "cancelled"


_DISTRIBUTION_SAMPLING_ABI = "probpipe.distribution_sampling/v1"
_PROBPIPE_DISTRIBUTION_PROVIDER_ABI = "probpipe.distribution/v1"


class _RandomEventPlan(Protocol):
    """Structural event fields accepted by the broker."""

    stochastic_source_id: _StructuralRngId
    logical_unit_id: _StructuralRngId


@dataclass(frozen=True, slots=True)
class _DirectRandomEventPlan:
    """Singleton source/unit event used outside lifting."""

    stochastic_source_id: _StructuralRngId
    logical_unit_id: _StructuralRngId


def _validate_stochastic_event(
    event: _RandomEventPlan,
) -> tuple[_StructuralRngId, _StructuralRngId]:
    """Validate a broker event before it can commit an occurrence ordinal."""
    try:
        source_id = event.stochastic_source_id
        unit_id = event.logical_unit_id
    except AttributeError as error:
        raise TypeError(
            "stochastic effect events must define source and logical-unit identities"
        ) from error
    if not isinstance(source_id, tuple) or not isinstance(unit_id, tuple):
        raise TypeError("stochastic effect source and unit identities must be tuples")
    _validate_random_event_value(source_id)
    _validate_random_event_value(unit_id)
    return source_id, unit_id


@dataclass(frozen=True, slots=True)
class StochasticEffectPlan:
    """Immutable plan anchor for one automatic-key request."""

    operation_kind: str
    execution_mode: str
    event: _RandomEventPlan
    sample_shape: tuple[int, ...] | None
    sampling_abi: str
    provider_abi: str
    record_path: tuple[str, ...] = ()
    descendant_descriptor: _CanonicalDescriptor | None = None

    def __post_init__(self) -> None:
        source_id, unit_id = _validate_stochastic_event(self.event)
        object.__setattr__(self, "event", _DirectRandomEventPlan(source_id, unit_id))
        _validate_stochastic_effect_fields(
            operation_kind=self.operation_kind,
            execution_mode=self.execution_mode,
            sample_shape=self.sample_shape,
            sampling_abi=self.sampling_abi,
            provider_abi=self.provider_abi,
            record_path=self.record_path,
            descendant_descriptor=self.descendant_descriptor,
        )


@dataclass(slots=True)
class _ManagedUnitClaimState:
    """Operational retry state for one canonical managed unit."""

    frame: ManagedUnitFrame
    child_invocations: list[_workflow_context._WorkflowInvocation] = field(default_factory=list)
    active_attempt: bytes | None = None
    active_transport: Literal["local", "remote"] | None = None
    active_parent_occurrence_path: _RandomEventPath | None = None
    seen_attempts: set[bytes] = field(default_factory=set)
    active_effect_identities: set[tuple[Any, ...]] = field(default_factory=set)
    effect_claims_by_identity: dict[tuple[Any, ...], ManagedEffectClaim] = field(
        default_factory=dict
    )
    status: _ManagedUnitStatus = _ManagedUnitStatus.ISSUED


@dataclass(slots=True)
class _ManagedClaimRegistry:
    """Parent-broker registry for managed units and their retry claims."""

    by_unit: dict[tuple[Any, ...], _ManagedUnitClaimState] = field(default_factory=dict)
    by_token: dict[ManagedWorkItemToken, _ManagedUnitClaimState] = field(default_factory=dict)
    closed: bool = False
    lock: Any = field(default_factory=Lock, repr=False)


@dataclass(slots=True)
class _ManagedAttemptContext:
    """Attempt-local child cursor bound inside one managed work item."""

    parent_broker: Any
    frame: ManagedUnitFrame
    attempt: ManagedAttemptState
    workflow_frame: _workflow_context._WorkflowFrame
    next_child_ordinal: int = 0
    successful_effects_by_identity: dict[tuple[Any, ...], ManagedEffectClaim] = field(
        default_factory=dict
    )

    def claim_child_invocation(self) -> _workflow_context._WorkflowInvocation:
        """Claim or retry the next child occurrence in canonical order."""
        ordinal = self.next_child_ordinal
        self.next_child_ordinal += 1
        return self.parent_broker._claim_managed_child(
            frame=self.frame,
            attempt=self.attempt,
            child_ordinal=ordinal,
        )

    def claim_scoped_child_invocation(
        self,
        workflow_frame: _workflow_context._WorkflowFrame,
        occurrence_kind: _OccurrenceKind,
    ) -> _workflow_context._WorkflowInvocation:
        """Claim a child whose active nested run supplies its root and scope path."""
        managed_child = self.claim_child_invocation()
        scope_path = _workflow_context._materialize_descendant_path(
            workflow_frame,
            self.workflow_frame,
        )
        local_ordinal = workflow_frame.ledger.commit()
        return _workflow_context._WorkflowInvocation(
            frame=workflow_frame,
            occurrence_path=(
                *managed_child.occurrence_path,
                *scope_path,
                (occurrence_kind, local_ordinal),
            ),
        )

    def claim_effect(self, effect: ManagedEffectClaim) -> None:
        """Retain one attempt claim without making it durable recipe state."""
        self.parent_broker.claim_managed_effect(
            effect,
            frame=self.frame,
            attempt=self.attempt,
        )

    def accept_successful_effects(self, effects: tuple[ManagedEffectClaim, ...]) -> None:
        """Collect effects from child scopes that completed successfully."""
        for effect in effects:
            identity = _effect_identity(effect)
            existing = self.successful_effects_by_identity.get(identity)
            if existing is not None and existing != effect:
                raise RuntimeError(
                    "a stochastic event identity completed with a different effect plan"
                )
            self.successful_effects_by_identity[identity] = effect

    def publish_successful_effects(self) -> None:
        """Promote successful effects through the containing broker boundary."""
        self.parent_broker.accept_successful_managed_effects(
            tuple(self.successful_effects_by_identity.values())
        )


def _effect_identity(effect: ManagedEffectClaim) -> tuple[Any, ...]:
    """Return the canonical identity shared by retry and recipe ledgers."""
    return (
        effect.occurrence_path,
        effect.stochastic_source_id,
        effect.logical_unit_id,
    )


def _managed_effect_child_ordinal(
    effect: ManagedEffectClaim,
    *,
    prefix: _RandomEventPath,
) -> int:
    """Validate one report occurrence namespace and return its child ordinal."""
    occurrence_path = effect.occurrence_path
    if occurrence_path[: len(prefix)] != prefix or len(occurrence_path) <= len(prefix):
        raise RuntimeError("a remote effect occurrence path is outside its managed unit namespace")
    child_segment = occurrence_path[len(prefix)]
    if (
        not isinstance(child_segment, tuple)
        or len(child_segment) != 2
        or child_segment[0] != "child"
        or isinstance(child_segment[1], bool)
        or not isinstance(child_segment[1], int)
        or child_segment[1] < 0
    ):
        raise RuntimeError("a remote effect occurrence path has an invalid child ordinal")
    return child_segment[1]


@dataclass(frozen=True, slots=True)
class _BrokerRecipeSnapshot:
    """Successful-invocation data safe for canonical recipe serialization."""

    root_words: tuple[int, int]
    occurrence_path: tuple[Any, ...]
    rng_origin: dict[str, str | int | None]
    effects: tuple[ManagedEffectClaim, ...]
    execution_contracts: tuple[WorkflowRngExecutionContract, ...]
    requested_dispatch: str | None
    requested_workflow_kind: str | None
    callable_anchor: CallableAnchor | None


def _singleton_effect_plan(
    *,
    operation_kind: str,
    execution_mode: str,
    sample_shape: tuple[int, ...] | None,
    source_index: int = 0,
    sampling_abi: str = _DISTRIBUTION_SAMPLING_ABI,
    provider_abi: str = _PROBPIPE_DISTRIBUTION_PROVIDER_ABI,
    record_path: tuple[str, ...] = (),
    descendant_descriptor: tuple[Any, ...] | None = None,
) -> StochasticEffectPlan:
    """Build the standard singleton event plan for a direct operation."""
    return StochasticEffectPlan(
        operation_kind=operation_kind,
        execution_mode=execution_mode,
        event=_DirectRandomEventPlan(
            ("source-group", source_index),
            ("singleton",),
        ),
        sample_shape=sample_shape,
        sampling_abi=sampling_abi,
        provider_abi=provider_abi,
        record_path=record_path,
        descendant_descriptor=descendant_descriptor,
    )


@dataclass(slots=True)
class _AutomaticKeyBroker:
    """Lazily commit and serve keys for one stochastic occurrence."""

    occurrence_kind: _OccurrenceKind
    _frame: _workflow_context._WorkflowFrame | None = None
    _invocation: _workflow_context._WorkflowInvocation | None = None
    _managed_attempt: _ManagedAttemptContext | None = None
    _managed_claims: _ManagedClaimRegistry = field(default_factory=_ManagedClaimRegistry)
    _execution_contracts: list[WorkflowRngExecutionContract] = field(default_factory=list)
    _effects_by_identity: dict[tuple[Any, ...], ManagedEffectClaim] = field(default_factory=dict)
    _effects_lock: Any = field(default_factory=Lock, repr=False)
    _lock: Any = field(default_factory=Lock, repr=False)
    _requested_dispatch: str | None = None
    _requested_workflow_kind: str | None = None
    _callable_anchor: CallableAnchor | None = None
    _replay_state: Any = None

    def key_for(self, plan: StochasticEffectPlan) -> PRNGKey:
        """Return the workflow-owned key for one planned effect."""
        _workflow_context._assert_workflow_admission(self._frame)
        self._assert_managed_registry_open()
        if not isinstance(plan, StochasticEffectPlan):
            raise TypeError("automatic key requests require a StochasticEffectPlan")
        source_id, unit_id = _validate_stochastic_event(plan.event)
        _validate_stochastic_effect_fields(
            operation_kind=plan.operation_kind,
            execution_mode=plan.execution_mode,
            sample_shape=plan.sample_shape,
            sampling_abi=plan.sampling_abi,
            provider_abi=plan.provider_abi,
            record_path=plan.record_path,
            descendant_descriptor=plan.descendant_descriptor,
        )
        coordination_probe = _REMOTE_COORDINATION_PROBE.get()
        if coordination_probe is not None:
            coordination_probe.effect_observed = True
            raise _ManagedCoordinationRequired
        _workflow_context._guard_automatic_key_request()
        self.validate_replay_effect_plan(plan)
        with self._lock:
            if self._invocation is None:
                self._invocation = self._claim_own_invocation()
            invocation = self._invocation
        effect = ManagedEffectClaim(
            occurrence_path=invocation.occurrence_path,
            occurrence_kind=self.occurrence_kind,
            stochastic_source_id=source_id,
            logical_unit_id=unit_id,
            operation_kind=plan.operation_kind,
            execution_mode=plan.execution_mode,
            sample_shape=plan.sample_shape,
            sampling_abi=plan.sampling_abi,
            provider_abi=plan.provider_abi,
            record_path=plan.record_path,
            descendant_descriptor=plan.descendant_descriptor,
        )
        if self._managed_attempt is None:
            self.claim_replay_effect(effect, attempt=None)
            self._record_effect(effect)
        else:
            parent_broker = self._managed_attempt.parent_broker
            claim_replay_effect = getattr(parent_broker, "claim_replay_effect", None)
            if claim_replay_effect is None:
                from . import _workflow_replay

                _workflow_replay._claim_effect_before_derivation(
                    effect,
                    attempt=self._managed_attempt.attempt,
                )
            else:
                claim_replay_effect(
                    effect,
                    attempt=self._managed_attempt.attempt,
                )
            self._managed_attempt.claim_effect(effect)
            self._record_effect(effect)
        return invocation.key_for(
            stochastic_source_id=source_id,
            logical_unit_id=unit_id,
        )

    def register_managed_work_items(self, items: tuple[ManagedWorkItem, ...]) -> None:
        """Register every issued unit token before request submission."""
        with self._managed_claims.lock:
            self._assert_managed_registry_open_unlocked()
            for item in items:
                unit = item.frame.unit_segment
                existing = self._managed_claims.by_unit.get(unit)
                if existing is not None and existing.frame.token != item.frame.token:
                    raise RuntimeError(
                        "a managed workflow unit cannot be reused with a different token"
                    )
                token_owner = self._managed_claims.by_token.get(item.frame.token)
                if token_owner is not None and token_owner.frame.unit_segment != unit:
                    raise RuntimeError(
                        "a managed work-item token cannot own multiple logical units"
                    )
                if existing is None:
                    existing = _ManagedUnitClaimState(frame=item.frame)
                    self._managed_claims.by_unit[unit] = existing
                    self._managed_claims.by_token[item.frame.token] = existing

    def record_execution_contract(self, contract: WorkflowRngExecutionContract) -> None:
        """Record one distinct capability-checked route for later diagnostics."""
        if contract not in self._execution_contracts:
            self._execution_contracts.append(contract)

    def set_requested_execution(self, dispatch: str, workflow_kind: str) -> None:
        """Record the public execution request independently from its resolution."""
        requested = (dispatch, workflow_kind)
        existing = (self._requested_dispatch, self._requested_workflow_kind)
        if existing != (None, None) and existing != requested:
            raise RuntimeError("a Function broker already has a different execution request")
        self._requested_dispatch, self._requested_workflow_kind = requested

    def set_callable_anchor(self, anchor: CallableAnchor) -> None:
        """Attach the immutable definition anchor for this public Function."""
        if self._callable_anchor is not None:
            raise RuntimeError("a Function broker already has a callable anchor")
        self._callable_anchor = anchor

    def claim_replay_effect(
        self,
        effect: ManagedEffectClaim,
        *,
        attempt: ManagedAttemptState | None,
    ) -> None:
        """Validate one local/transported effect against captured replay state."""
        if self._replay_state is not None:
            self._replay_state.claim_effect(effect, attempt=attempt)
            return
        if self._managed_attempt is not None:
            parent = self._managed_attempt.parent_broker
            parent_claim = getattr(parent, "claim_replay_effect", None)
            if parent_claim is not None:
                parent_claim(effect, attempt=attempt)
                return
        from . import _workflow_replay

        _workflow_replay._claim_effect_before_derivation(effect, attempt=attempt)

    def validate_replay_effect_plan(self, plan: StochasticEffectPlan) -> None:
        """Reject direct-operation plan drift before committing its occurrence."""
        if self._replay_state is not None:
            self._replay_state.validate_effect_plan(plan)
            return
        if self._managed_attempt is not None:
            parent = self._managed_attempt.parent_broker
            parent_validate = getattr(parent, "validate_replay_effect_plan", None)
            if parent_validate is not None:
                parent_validate(plan)
                return
        from . import _workflow_replay

        _workflow_replay._validate_effect_plan_before_commit(plan)

    def claim_managed_effect(
        self,
        effect: ManagedEffectClaim,
        *,
        frame: ManagedUnitFrame,
        attempt: ManagedAttemptState,
    ) -> None:
        """Retain retry compatibility state before deriving a managed key."""
        with self._managed_claims.lock:
            state = self._managed_claims.by_token.get(attempt.work_item_token)
            if (
                state is None
                or state.frame != frame
                or state.active_attempt != attempt.attempt_token
            ):
                raise RuntimeError("managed effect claim does not own the active attempt")
            identity = _effect_identity(effect)
            existing = state.effect_claims_by_identity.get(identity)
            if existing is not None and existing != effect:
                raise RuntimeError(
                    "a stochastic event identity was retried with a different effect plan"
                )
            if identity in state.active_effect_identities:
                raise RuntimeError("one managed attempt duplicated a stochastic effect claim")
            state.active_effect_identities.add(identity)
            state.effect_claims_by_identity[identity] = effect

    def accept_successful_managed_effects(
        self,
        effects: tuple[ManagedEffectClaim, ...],
    ) -> None:
        """Make effects from one successful managed boundary recipe-visible."""
        with self._effects_lock:
            for effect in effects:
                existing = self._effects_by_identity.get(_effect_identity(effect))
                if existing is not None and existing != effect:
                    raise RuntimeError(
                        "a stochastic event identity completed with a different effect plan"
                    )
            self._effects_by_identity.update(
                (_effect_identity(effect), effect) for effect in effects
            )

    def _publish_managed_effects(self) -> None:
        """Publish this successful broker's effects to its managed attempt."""
        if self._managed_attempt is None:
            return
        with self._effects_lock:
            effects = tuple(self._effects_by_identity.values())
        self._managed_attempt.accept_successful_effects(effects)

    def _mark_replay_effects_successful(self) -> None:
        """Confirm the recipe-visible effects of a successful replay root."""
        if (
            self._replay_state is None
            or self._invocation is None
            or self._invocation.occurrence_path != self._replay_state.occurrence_path
        ):
            return
        with self._effects_lock:
            effects = tuple(self._effects_by_identity.values())
        self._replay_state.mark_successful_effects(effects)

    def _record_effect(self, effect: ManagedEffectClaim) -> None:
        identity = _effect_identity(effect)
        with self._effects_lock:
            existing = self._effects_by_identity.get(identity)
            if existing is not None:
                if existing != effect:
                    raise RuntimeError(
                        "a stochastic event identity was retried with a different effect plan"
                    )
                raise RuntimeError(
                    "one stochastic broker scope duplicated a stochastic effect claim"
                )
            self._effects_by_identity[identity] = effect

    def validate_managed_attempt_preflight(
        self,
        attempt: ManagedAttemptState,
        frame: ManagedUnitFrame,
    ) -> None:
        """Validate an attempt and frame without reserving registry state."""
        with self._managed_claims.lock:
            self._validate_managed_attempt_unlocked(attempt, frame)

    def begin_managed_attempt(
        self,
        attempt: ManagedAttemptState,
        frame: ManagedUnitFrame,
    ) -> ManagedUnitFrame:
        """Admit one fresh attempt for a previously registered work-item token."""
        with self._managed_claims.lock:
            state = self._validate_managed_attempt_unlocked(attempt, frame)
            state.seen_attempts.add(attempt.attempt_token)
            state.active_attempt = attempt.attempt_token
            state.active_transport = "local"
            state.active_parent_occurrence_path = None
            state.active_effect_identities.clear()
            state.status = _ManagedUnitStatus.ACTIVE
            return state.frame

    def finish_managed_attempt(self, attempt: ManagedAttemptState) -> None:
        """Join one active attempt without discarding its retry claims."""
        with self._managed_claims.lock:
            state = self._managed_claims.by_token.get(attempt.work_item_token)
            if (
                state is None
                or state.status is not _ManagedUnitStatus.ACTIVE
                or state.active_attempt != attempt.attempt_token
                or state.active_transport != "local"
            ):
                raise RuntimeError("managed work-item attempt is not active")
            state.active_attempt = None
            state.active_transport = None
            state.active_parent_occurrence_path = None
            state.active_effect_identities.clear()
            state.status = _ManagedUnitStatus.JOINED

    def cancel_unstarted_managed_items(self, items: tuple[ManagedWorkItem, ...]) -> None:
        """Cancel issued items that were never submitted after an earlier failure."""
        with self._managed_claims.lock:
            self._assert_managed_registry_open_unlocked()
            for item in items:
                state = self._managed_claims.by_token[item.frame.token]
                if state.status is _ManagedUnitStatus.ISSUED:
                    state.status = _ManagedUnitStatus.CANCELLED

    def assert_managed_items_joined(self, items: tuple[ManagedWorkItem, ...]) -> None:
        """Require all issued request tokens to be inactive and joined."""
        with self._managed_claims.lock:
            for item in items:
                state = self._managed_claims.by_token[item.frame.token]
                if state.active_attempt is not None or state.status not in {
                    _ManagedUnitStatus.JOINED,
                    _ManagedUnitStatus.CANCELLED,
                }:
                    raise RuntimeError(
                        "managed workflow request exited before all work items joined"
                    )

    def assert_all_managed_items_joined(self) -> None:
        """Prevent the parent Function broker from releasing active ownership."""
        with self._managed_claims.lock:
            if any(
                state.active_attempt is not None
                or state.status not in {_ManagedUnitStatus.JOINED, _ManagedUnitStatus.CANCELLED}
                for state in self._managed_claims.by_unit.values()
            ):
                raise RuntimeError(
                    "a Function workflow scope cannot exit before all managed work items join"
                )

    def close_managed_claim_registry(self) -> None:
        """Atomically join-check and close this broker's managed registry."""
        with self._managed_claims.lock:
            if self._managed_claims.closed:
                return
            if any(
                state.active_attempt is not None
                or state.status not in {_ManagedUnitStatus.JOINED, _ManagedUnitStatus.CANCELLED}
                for state in self._managed_claims.by_unit.values()
            ):
                raise RuntimeError(
                    "a stochastic broker scope cannot exit before all managed work items join"
                )
            self._managed_claims.closed = True

    def _assert_managed_registry_open(self) -> None:
        """Reject broker operations after the Function scope has closed."""
        with self._managed_claims.lock:
            self._assert_managed_registry_open_unlocked()

    def _assert_managed_registry_open_unlocked(self) -> None:
        if self._managed_claims.closed:
            raise RuntimeError("the managed workflow broker is closed")

    def _validate_managed_attempt_unlocked(
        self,
        attempt: ManagedAttemptState,
        frame: ManagedUnitFrame,
    ) -> _ManagedUnitClaimState:
        """Validate one attempt while the managed registry lock is held."""
        self._assert_managed_registry_open_unlocked()
        state = self._managed_claims.by_token.get(attempt.work_item_token)
        if state is None:
            raise RuntimeError("managed work-item token was not registered by its parent")
        if state.frame != frame:
            raise RuntimeError("managed work-item frame does not match its registered token")
        if state.status is _ManagedUnitStatus.CANCELLED:
            raise RuntimeError("a cancelled managed work item cannot start an attempt")
        if state.status is _ManagedUnitStatus.ACTIVE or state.active_attempt is not None:
            raise RuntimeError(
                "a managed work-item token already has an active attempt; "
                "duplicate or concurrent attempts are not allowed"
            )
        if attempt.attempt_token in state.seen_attempts:
            raise RuntimeError("a managed attempt token cannot be reused")
        return state

    def reserve_remote_managed_attempt(
        self,
        frame: ManagedUnitFrame,
        attempt: ManagedAttemptState,
        *,
        parent_authority: bool,
    ) -> ManagedParentEnvelope | None:
        """Reserve a remote attempt before submission and optionally bind authority."""
        with self._managed_claims.lock:
            state = self._validate_managed_attempt_unlocked(attempt, frame)
            state.seen_attempts.add(attempt.attempt_token)
            state.active_attempt = attempt.attempt_token
            state.active_transport = "remote"
            state.active_parent_occurrence_path = None
            state.active_effect_identities.clear()
            state.status = _ManagedUnitStatus.ACTIVE
            retry_effects = tuple(state.effect_claims_by_identity.values())
        if not parent_authority:
            return None

        try:
            envelope = self._make_remote_parent_envelope(
                frame,
                attempt,
                retry_effects=retry_effects,
            )
            with self._managed_claims.lock:
                state = self._require_active_remote_attempt_unlocked(attempt, frame)
                state.active_parent_occurrence_path = envelope.parent_occurrence_path
            return envelope
        except BaseException:
            self.abort_remote_managed_attempt(attempt)
            raise

    def _make_remote_parent_envelope(
        self,
        frame: ManagedUnitFrame,
        attempt: ManagedAttemptState,
        *,
        retry_effects: tuple[ManagedEffectClaim, ...],
    ) -> ManagedParentEnvelope:
        """Materialize authority for an already-reserved remote attempt."""
        parent_invocation = self._ensure_parent_invocation()
        from . import _workflow_replay

        return ManagedParentEnvelope(
            root_words=_workflow_context._resolve_root_words(parent_invocation.frame),
            parent_occurrence_path=parent_invocation.occurrence_path,
            frame=frame,
            attempt=attempt,
            replay_expected_effects=_workflow_replay._expected_effects_for_managed_unit(
                parent_invocation.occurrence_path,
                frame.unit_segment,
            ),
            retry_effects=retry_effects,
        )

    def abort_remote_managed_attempt(self, attempt: ManagedAttemptState) -> None:
        """Release one exact remote reservation after failure or cancellation."""
        with self._managed_claims.lock:
            self._assert_managed_registry_open_unlocked()
            state = self._managed_claims.by_token.get(attempt.work_item_token)
            if state is None or attempt.attempt_token not in state.seen_attempts:
                raise RuntimeError("remote managed attempt was not reserved by this parent")
            if state.active_attempt is None and state.status is _ManagedUnitStatus.JOINED:
                return
            if (
                state.status is not _ManagedUnitStatus.ACTIVE
                or state.active_attempt != attempt.attempt_token
                or state.active_transport != "remote"
            ):
                raise RuntimeError("remote managed attempt does not own the active reservation")
            state.active_attempt = None
            state.active_transport = None
            state.active_parent_occurrence_path = None
            state.active_effect_identities.clear()
            state.status = _ManagedUnitStatus.JOINED

    def accept_remote_claim_report(self, report: ManagedClaimReport) -> None:
        """Validate and atomically reconcile one reserved remote report."""
        replay_transaction = (
            nullcontext(None)
            if self._replay_state is None
            else self._replay_state.claim_effects_transaction(
                report.effects,
                attempt=report.attempt,
            )
        )
        with replay_transaction as replay_batch, self._managed_claims.lock:
            state, child_invocations = self._validate_remote_report_unlocked(report)
            with self._effects_lock:
                for effect in report.successful_effects:
                    existing = self._effects_by_identity.get(_effect_identity(effect))
                    if existing is not None and existing != effect:
                        raise RuntimeError(
                            "a stochastic event identity completed with a different effect plan"
                        )

                if replay_batch is not None:
                    replay_batch.commit()
                for effect in report.effects:
                    state.effect_claims_by_identity[_effect_identity(effect)] = effect
                state.child_invocations.extend(child_invocations)
                self._effects_by_identity.update(
                    (_effect_identity(effect), effect) for effect in report.successful_effects
                )
                state.active_attempt = None
                state.active_transport = None
                state.active_parent_occurrence_path = None
                state.active_effect_identities.clear()
                state.status = _ManagedUnitStatus.JOINED

    def _validate_remote_report_unlocked(
        self,
        report: ManagedClaimReport,
    ) -> tuple[
        _ManagedUnitClaimState,
        tuple[_workflow_context._WorkflowInvocation, ...],
    ]:
        """Validate a remote report without changing any parent ledger."""
        state = self._require_active_remote_attempt_unlocked(report.attempt, report.frame)
        parent_path = state.active_parent_occurrence_path
        if parent_path is None:
            if report.child_count or report.effects or report.successful_effects:
                raise RuntimeError(
                    "a rootless remote coordination probe cannot report stochastic claims"
                )
            return state, ()

        prefix = (*parent_path, report.frame.unit_segment)
        for effect in report.effects:
            child_ordinal = _managed_effect_child_ordinal(effect, prefix=prefix)
            if child_ordinal >= report.child_count:
                raise RuntimeError(
                    "a remote effect child ordinal exceeds the reported child namespace"
                )
            identity = _effect_identity(effect)
            existing = state.effect_claims_by_identity.get(identity)
            if existing is not None and existing != effect:
                raise RuntimeError(
                    "a remote stochastic event was retried with a different effect plan"
                )

        parent_invocation = self._invocation
        if parent_invocation is None or parent_invocation.occurrence_path != parent_path:
            raise RuntimeError("remote report lost its reserved parent occurrence authority")
        child_invocations = tuple(
            _workflow_context._WorkflowInvocation(
                frame=parent_invocation.frame,
                occurrence_path=(
                    *parent_path,
                    report.frame.unit_segment,
                    ("child", child_ordinal),
                ),
            )
            for child_ordinal in range(
                len(state.child_invocations),
                report.child_count,
            )
        )
        return state, child_invocations

    def _require_active_remote_attempt_unlocked(
        self,
        attempt: ManagedAttemptState,
        frame: ManagedUnitFrame,
    ) -> _ManagedUnitClaimState:
        """Return the exact active remote reservation while the registry is locked."""
        self._assert_managed_registry_open_unlocked()
        state = self._managed_claims.by_token.get(attempt.work_item_token)
        if state is None or state.frame != frame:
            raise RuntimeError("remote managed unit is not registered by this parent")
        if (
            state.status is not _ManagedUnitStatus.ACTIVE
            or state.active_attempt != attempt.attempt_token
            or state.active_transport != "remote"
        ):
            raise RuntimeError("remote managed attempt does not own the active reservation")
        return state

    def _claim_managed_child(
        self,
        *,
        frame: ManagedUnitFrame,
        attempt: ManagedAttemptState,
        child_ordinal: int,
    ) -> _workflow_context._WorkflowInvocation:
        """Claim one retry-stable child occurrence for a managed attempt."""
        with self._managed_claims.lock:
            state = self._managed_claims.by_token.get(frame.token)
            if (
                state is None
                or state.frame.unit_segment != frame.unit_segment
                or state.active_attempt != attempt.attempt_token
            ):
                raise RuntimeError("managed child claim does not own the active attempt")
            if child_ordinal < len(state.child_invocations):
                return state.child_invocations[child_ordinal]
            if child_ordinal != len(state.child_invocations):
                raise RuntimeError("managed child claims must be made in ordinal order")

            parent_invocation = self._ensure_parent_invocation()
            invocation = _workflow_context._WorkflowInvocation(
                frame=parent_invocation.frame,
                occurrence_path=(
                    *parent_invocation.occurrence_path,
                    frame.unit_segment,
                    ("child", child_ordinal),
                ),
            )
            state.child_invocations.append(invocation)
            return invocation

    def _ensure_parent_invocation(self) -> _workflow_context._WorkflowInvocation:
        """Lazily materialize the containing public Function occurrence."""
        with self._lock:
            if self._invocation is None:
                self._invocation = self._claim_own_invocation()
            return self._invocation

    def _claim_own_invocation(self) -> _workflow_context._WorkflowInvocation:
        """Commit this broker while preserving a nested run inside a managed unit."""
        if self._managed_attempt is not None:
            if self._frame is None or self._frame is self._managed_attempt.workflow_frame:
                return self._managed_attempt.claim_child_invocation()
            return self._managed_attempt.claim_scoped_child_invocation(
                self._frame,
                self.occurrence_kind,
            )
        if self._frame is None:
            raise RuntimeError("automatic-key broker has no workflow frame")
        if _workflow_context._capture_active_workflow_frame() is self._frame:
            return _workflow_context._commit_stochastic_invocation(self.occurrence_kind)
        return _workflow_context._commit_stochastic_invocation_in_frame(
            self._frame,
            self.occurrence_kind,
        )


_ACTIVE_AUTOMATIC_KEY_BROKER: ContextVar[_AutomaticKeyBroker | None] = ContextVar(
    "probpipe_active_automatic_key_broker",
    default=None,
)

_ACTIVE_MANAGED_ATTEMPT: ContextVar[_ManagedAttemptContext | None] = ContextVar(
    "probpipe_active_managed_attempt",
    default=None,
)


@dataclass(slots=True)
class _RemoteCoordinationObservation:
    """Attempt-local observation that survives caught probe exceptions."""

    attempt: ManagedAttemptState
    effect_observed: bool = False


_REMOTE_COORDINATION_PROBE: ContextVar[_RemoteCoordinationObservation | None] = ContextVar(
    "probpipe_remote_coordination_probe",
    default=None,
)


class _ManagedCoordinationRequired(RuntimeError):
    """Signal that a remote work item needs parent RNG authority."""


@dataclass(slots=True)
class _RemoteManagedParent:
    """Worker-local adapter for a parent-authorized child namespace."""

    envelope: ManagedParentEnvelope
    attempt: ManagedAttemptState
    workflow_frame: _workflow_context._WorkflowFrame
    child_invocations: list[_workflow_context._WorkflowInvocation] = field(default_factory=list)
    effect_claims_by_identity: dict[tuple[Any, ...], ManagedEffectClaim] = field(
        default_factory=dict
    )
    successful_effects_by_identity: dict[tuple[Any, ...], ManagedEffectClaim] = field(
        default_factory=dict
    )

    def _claim_managed_child(
        self,
        *,
        frame: ManagedUnitFrame,
        attempt: ManagedAttemptState,
        child_ordinal: int,
    ) -> _workflow_context._WorkflowInvocation:
        if frame != self.envelope.frame or attempt != self.attempt:
            raise RuntimeError("remote managed child does not own its envelope")
        if child_ordinal != len(self.child_invocations):
            raise RuntimeError("remote managed child claims must be made in order")
        invocation = _workflow_context._WorkflowInvocation(
            frame=self.workflow_frame,
            occurrence_path=(
                *self.envelope.parent_occurrence_path,
                frame.unit_segment,
                ("child", child_ordinal),
            ),
        )
        self.child_invocations.append(invocation)
        return invocation

    def claim_managed_effect(
        self,
        effect: ManagedEffectClaim,
        *,
        frame: ManagedUnitFrame,
        attempt: ManagedAttemptState,
    ) -> None:
        """Validate and retain one remote retry claim before key derivation."""
        if frame != self.envelope.frame or attempt != self.attempt:
            raise RuntimeError("remote managed effect does not own its envelope")
        identity = _effect_identity(effect)
        retry_effect = next(
            (
                candidate
                for candidate in self.envelope.retry_effects
                if _effect_identity(candidate) == identity
            ),
            None,
        )
        if retry_effect is not None and retry_effect != effect:
            raise RuntimeError("remote event identity changed effect plan during retry")
        existing = self.effect_claims_by_identity.get(identity)
        if existing is not None:
            if existing != effect:
                raise RuntimeError("remote event identity changed effect plan during retry")
            raise RuntimeError("one managed attempt duplicated a stochastic effect claim")
        self.effect_claims_by_identity[identity] = effect

    def accept_successful_managed_effects(
        self,
        effects: tuple[ManagedEffectClaim, ...],
    ) -> None:
        """Mark the worker effects that crossed a successful scope boundary."""
        for effect in effects:
            identity = _effect_identity(effect)
            if self.effect_claims_by_identity.get(identity) != effect:
                raise RuntimeError("a successful remote effect was not claimed by its attempt")
            self.successful_effects_by_identity[identity] = effect

    def report(self) -> ManagedClaimReport:
        """Return the serializable claim count for parent reconciliation."""
        return ManagedClaimReport(
            frame=self.envelope.frame,
            attempt=self.attempt,
            child_count=len(self.child_invocations),
            effects=tuple(self.effect_claims_by_identity.values()),
            successful_effects=tuple(self.successful_effects_by_identity.values()),
        )


@contextmanager
def _installed_stochastic_broker_scope(
    broker: _AutomaticKeyBroker,
) -> Generator[_AutomaticKeyBroker, None, None]:
    """Install and finalize one newly owned stochastic broker."""
    token: Token[_AutomaticKeyBroker | None] = _ACTIVE_AUTOMATIC_KEY_BROKER.set(broker)
    try:
        yield broker
    except BaseException:
        broker.close_managed_claim_registry()
        raise
    else:
        broker.close_managed_claim_registry()
        broker._publish_managed_effects()
        broker._mark_replay_effects_successful()
    finally:
        _ACTIVE_AUTOMATIC_KEY_BROKER.reset(token)


@contextmanager
def _function_stochastic_scope(
    *,
    occurrence_path: tuple[Any, ...] | None = None,
) -> Generator[_AutomaticKeyBroker, None, None]:
    """Install a lazy broker for one public Function invocation."""
    _workflow_context._assert_workflow_admission()
    frame = _workflow_context._capture_active_workflow_frame()
    from . import _workflow_replay

    broker = _AutomaticKeyBroker(
        "invocation",
        _frame=frame,
        _managed_attempt=_ACTIVE_MANAGED_ATTEMPT.get(),
        _replay_state=_workflow_replay._capture_active_replay_state(),
    )
    if occurrence_path is not None:
        if frame is None or broker._managed_attempt is not None:
            raise RuntimeError("a replay occurrence requires a standalone workflow frame")
        broker._invocation = _workflow_context._WorkflowInvocation(
            frame=frame,
            occurrence_path=occurrence_path,
        )
    with _installed_stochastic_broker_scope(broker):
        yield broker


@contextmanager
def _managed_stochastic_scope() -> Generator[_AutomaticKeyBroker, None, None]:
    """Reuse an active broker or install one managed-operation broker."""
    _workflow_context._assert_workflow_admission()
    active = _ACTIVE_AUTOMATIC_KEY_BROKER.get()
    if active is not None:
        yield active
        return

    managed_attempt = _ACTIVE_MANAGED_ATTEMPT.get()
    if managed_attempt is not None:
        broker = _AutomaticKeyBroker(
            "operation",
            _frame=_workflow_context._capture_active_workflow_frame(),
            _managed_attempt=managed_attempt,
        )
        with _installed_stochastic_broker_scope(broker):
            yield broker
        return

    with _workflow_context._ephemeral_workflow_run():
        broker = _AutomaticKeyBroker(
            "operation",
            _frame=_workflow_context._capture_active_workflow_frame(),
        )
        with _installed_stochastic_broker_scope(broker):
            yield broker


def _capture_active_broker() -> _AutomaticKeyBroker | None:
    """Capture the admitted parent broker for managed execution transport."""
    _workflow_context._assert_workflow_admission()
    return _ACTIVE_AUTOMATIC_KEY_BROKER.get()


def _record_active_execution_contract(
    contract: WorkflowRngExecutionContract,
) -> None:
    """Attach an actual route contract to the current public invocation."""
    from . import _workflow_replay

    _workflow_replay._validate_active_execution_contract(contract)
    broker = _ACTIVE_AUTOMATIC_KEY_BROKER.get()
    if broker is not None:
        broker.record_execution_contract(contract)


def _record_active_requested_execution(dispatch: str, workflow_kind: str) -> None:
    """Attach requested route diagnostics to the current public invocation."""
    from . import _workflow_replay

    _workflow_replay._record_active_requested_execution(dispatch, workflow_kind)
    broker = _ACTIVE_AUTOMATIC_KEY_BROKER.get()
    if broker is not None:
        broker.set_requested_execution(dispatch, workflow_kind)


def _snapshot_active_recipe_state() -> _BrokerRecipeSnapshot | None:
    """Snapshot a successful active invocation without operational ownership data."""
    broker = _ACTIVE_AUTOMATIC_KEY_BROKER.get()
    if broker is None or broker._invocation is None:
        return None
    with broker._effects_lock:
        effects = tuple(broker._effects_by_identity.values())
    if not effects:
        return None
    invocation = broker._invocation
    return _BrokerRecipeSnapshot(
        root_words=_workflow_context._resolve_root_words(invocation.frame),
        occurrence_path=invocation.occurrence_path,
        rng_origin=_workflow_context._describe_rng_origin(invocation.frame),
        effects=effects,
        execution_contracts=tuple(broker._execution_contracts),
        requested_dispatch=broker._requested_dispatch,
        requested_workflow_kind=broker._requested_workflow_kind,
        callable_anchor=broker._callable_anchor,
    )


@contextmanager
def _remote_coordination_probe_scope(
    attempt: ManagedAttemptState,
) -> Generator[_RemoteCoordinationObservation, None, None]:
    """Run a remote item without permitting automatic stochastic commit."""
    observation = _RemoteCoordinationObservation(attempt)
    probe_token = _REMOTE_COORDINATION_PROBE.set(observation)
    attempt_token = _ACTIVE_MANAGED_ATTEMPT.set(None)
    broker_token = _ACTIVE_AUTOMATIC_KEY_BROKER.set(None)
    try:
        yield observation
    finally:
        _ACTIVE_AUTOMATIC_KEY_BROKER.reset(broker_token)
        _ACTIVE_MANAGED_ATTEMPT.reset(attempt_token)
        _REMOTE_COORDINATION_PROBE.reset(probe_token)


@contextmanager
def _remote_managed_work_item_stochastic_scope(
    envelope: ManagedParentEnvelope,
    attempt: ManagedAttemptState,
) -> Generator[_RemoteManagedParent, None, None]:
    """Install parent-authorized RNG derivation inside a remote worker."""
    if envelope.attempt != attempt or envelope.frame.token != attempt.work_item_token:
        raise RuntimeError("remote managed attempt does not own its parent envelope")
    frame = _workflow_context._capture_active_workflow_frame()
    if frame is None:
        raise RuntimeError("remote managed randomness requires a transported frame")
    _workflow_context._assert_transported_root_authority(frame, envelope.root_words)
    parent = _RemoteManagedParent(
        envelope=envelope,
        attempt=attempt,
        workflow_frame=frame,
    )
    state = _ManagedAttemptContext(
        parent_broker=parent,
        frame=envelope.frame,
        attempt=attempt,
        workflow_frame=frame,
    )
    from . import _workflow_replay

    attempt_token = _ACTIVE_MANAGED_ATTEMPT.set(state)
    broker_token = _ACTIVE_AUTOMATIC_KEY_BROKER.set(None)
    try:
        with _workflow_replay._remote_replay_claim_scope(
            envelope.replay_expected_effects,
            attempt,
        ):
            yield parent
    except BaseException:
        raise
    else:
        state.publish_successful_effects()
    finally:
        _ACTIVE_AUTOMATIC_KEY_BROKER.reset(broker_token)
        _ACTIVE_MANAGED_ATTEMPT.reset(attempt_token)


@contextmanager
def _managed_work_item_stochastic_scope(
    parent_broker: _AutomaticKeyBroker,
    frame: ManagedUnitFrame,
    *,
    attempt: ManagedAttemptState | None = None,
) -> Generator[ManagedAttemptState, None, None]:
    """Install one retry attempt and an empty child-broker slot."""
    if attempt is None:
        attempt = ManagedAttemptState.create(frame.token)
    parent_broker.validate_managed_attempt_preflight(attempt, frame)
    workflow_frame = _workflow_context._capture_active_workflow_frame()
    if workflow_frame is None:
        raise RuntimeError("managed randomness requires an active workflow frame")
    with workflow_frame.state.lock:
        managed_unit_segment = workflow_frame.state.managed_unit_segment
    if managed_unit_segment != frame.unit_segment:
        raise RuntimeError("managed randomness requires the matching active managed workflow frame")
    state = _ManagedAttemptContext(
        parent_broker=parent_broker,
        frame=frame,
        attempt=attempt,
        workflow_frame=workflow_frame,
    )
    parent_broker.begin_managed_attempt(attempt, frame)
    attempt_token: Token[_ManagedAttemptContext | None] | None = None
    broker_token: Token[_AutomaticKeyBroker | None] | None = None
    try:
        attempt_token = _ACTIVE_MANAGED_ATTEMPT.set(state)
        broker_token = _ACTIVE_AUTOMATIC_KEY_BROKER.set(None)
        yield attempt
    except BaseException:
        raise
    else:
        state.publish_successful_effects()
    finally:
        try:
            if broker_token is not None:
                _ACTIVE_AUTOMATIC_KEY_BROKER.reset(broker_token)
            if attempt_token is not None:
                _ACTIVE_MANAGED_ATTEMPT.reset(attempt_token)
        finally:
            parent_broker.finish_managed_attempt(attempt)


def _resolve_automatic_key(
    explicit_key: PRNGKey | None,
    plan: StochasticEffectPlan | None,
) -> PRNGKey:
    """Preserve an explicit key or resolve one through the active broker."""
    if explicit_key is not None:
        return explicit_key
    if plan is None:
        raise TypeError("an omitted key requires a stochastic effect plan")

    broker = _ACTIVE_AUTOMATIC_KEY_BROKER.get()
    if broker is not None:
        return broker.key_for(plan)
    with _managed_stochastic_scope() as managed_broker:
        return managed_broker.key_for(plan)
