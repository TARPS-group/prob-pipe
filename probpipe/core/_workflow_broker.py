"""Private automatic-key broker for workflow-owned stochastic effects."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ..custom_types import PRNGKey
from . import _workflow_context

if TYPE_CHECKING:
    from ._workflow_execution_contract import WorkflowRngExecutionContract
from ._workflow_managed import (
    ManagedAttemptState,
    ManagedClaimReport,
    ManagedParentEnvelope,
    ManagedUnitFrame,
    ManagedWorkItem,
    ManagedWorkItemToken,
)

_OccurrenceKind = Literal["invocation", "operation"]
_StructuralRngId = tuple[str | int, ...]

_DISTRIBUTION_SAMPLING_ABI = "probpipe.distribution_sampling/v1"
_PROBPIPE_DISTRIBUTION_PROVIDER_ABI = "probpipe.distribution/v1"


class _RandomEventPlan(Protocol):
    """Structural event fields accepted by the broker."""

    stochastic_source_id: _StructuralRngId
    logical_unit_id: _StructuralRngId


@dataclass(frozen=True)
class _DirectRandomEventPlan:
    """Singleton source/unit event used outside lifting."""

    stochastic_source_id: _StructuralRngId
    logical_unit_id: _StructuralRngId


@dataclass(frozen=True)
class StochasticEffectPlan:
    """Immutable plan anchor for one automatic-key request."""

    operation_kind: str
    execution_mode: str
    event: _RandomEventPlan
    sample_shape: tuple[int, ...] | None
    sampling_abi: str
    provider_abi: str


@dataclass
class _ManagedUnitClaimState:
    """Operational retry state for one canonical managed unit."""

    frame: ManagedUnitFrame
    child_invocations: list[_workflow_context._WorkflowInvocation] = field(default_factory=list)
    active_attempt: bytes | None = None
    seen_attempts: set[bytes] = field(default_factory=set)
    has_started: bool = False
    joined: bool = False


@dataclass
class _ManagedClaimRegistry:
    """Parent-broker registry for managed units and their retry claims."""

    by_unit: dict[tuple[Any, ...], _ManagedUnitClaimState] = field(default_factory=dict)
    by_token: dict[ManagedWorkItemToken, _ManagedUnitClaimState] = field(default_factory=dict)
    lock: Any = field(default_factory=Lock, repr=False)


@dataclass
class _ManagedAttemptContext:
    """Attempt-local child cursor bound inside one managed work item."""

    parent_broker: Any
    frame: ManagedUnitFrame
    attempt: ManagedAttemptState
    next_child_ordinal: int = 0

    def claim_child_invocation(self) -> _workflow_context._WorkflowInvocation:
        """Claim or retry the next child occurrence in canonical order."""
        ordinal = self.next_child_ordinal
        self.next_child_ordinal += 1
        return self.parent_broker._claim_managed_child(
            frame=self.frame,
            attempt=self.attempt,
            child_ordinal=ordinal,
        )


def _singleton_effect_plan(
    *,
    operation_kind: str,
    execution_mode: str,
    sample_shape: tuple[int, ...] | None,
    source_index: int = 0,
    sampling_abi: str = _DISTRIBUTION_SAMPLING_ABI,
    provider_abi: str = _PROBPIPE_DISTRIBUTION_PROVIDER_ABI,
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
    )


@dataclass
class _AutomaticKeyBroker:
    """Lazily commit and serve keys for one stochastic occurrence."""

    occurrence_kind: _OccurrenceKind
    _frame: _workflow_context._WorkflowFrame | None = None
    _invocation: _workflow_context._WorkflowInvocation | None = None
    _managed_attempt: _ManagedAttemptContext | None = None
    _managed_claims: _ManagedClaimRegistry = field(default_factory=_ManagedClaimRegistry)
    _execution_contracts: list[WorkflowRngExecutionContract] = field(default_factory=list)
    _lock: Any = field(default_factory=Lock, repr=False)

    def key_for(self, plan: StochasticEffectPlan) -> PRNGKey:
        """Return the workflow-owned key for one planned effect."""
        if not isinstance(plan, StochasticEffectPlan):
            raise TypeError("automatic key requests require a StochasticEffectPlan")
        _workflow_context._guard_automatic_key_request()
        if _REMOTE_COORDINATION_PROBE.get():
            raise _ManagedCoordinationRequired
        with self._lock:
            if self._invocation is None:
                if self._managed_attempt is None:
                    self._invocation = _workflow_context._commit_stochastic_invocation(
                        self.occurrence_kind
                    )
                else:
                    self._invocation = self._managed_attempt.claim_child_invocation()
            invocation = self._invocation
        return invocation.key_for(
            stochastic_source_id=plan.event.stochastic_source_id,
            logical_unit_id=plan.event.logical_unit_id,
        )

    def register_managed_work_items(self, items: tuple[ManagedWorkItem, ...]) -> None:
        """Register every issued unit token before request submission."""
        with self._managed_claims.lock:
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

    def begin_managed_attempt(self, attempt: ManagedAttemptState) -> ManagedUnitFrame:
        """Admit one fresh attempt for a previously registered work-item token."""
        with self._managed_claims.lock:
            state = self._managed_claims.by_token.get(attempt.work_item_token)
            if state is None:
                raise RuntimeError("managed work-item token was not registered by its parent")
            if state.active_attempt is not None:
                raise RuntimeError(
                    "a managed work-item token already has an active attempt; "
                    "duplicate or concurrent attempts are not allowed"
                )
            if attempt.attempt_token in state.seen_attempts:
                raise RuntimeError("a managed attempt token cannot be reused")
            state.seen_attempts.add(attempt.attempt_token)
            state.active_attempt = attempt.attempt_token
            state.has_started = True
            state.joined = False
            return state.frame

    def finish_managed_attempt(self, attempt: ManagedAttemptState) -> None:
        """Join one active attempt without discarding its retry claims."""
        with self._managed_claims.lock:
            state = self._managed_claims.by_token.get(attempt.work_item_token)
            if state is None or state.active_attempt != attempt.attempt_token:
                raise RuntimeError("managed work-item attempt is not active")
            state.active_attempt = None
            state.joined = True

    def cancel_unstarted_managed_items(self, items: tuple[ManagedWorkItem, ...]) -> None:
        """Join issued items that were never submitted after an earlier failure."""
        with self._managed_claims.lock:
            for item in items:
                state = self._managed_claims.by_token[item.frame.token]
                if not state.has_started:
                    state.joined = True

    def assert_managed_items_joined(self, items: tuple[ManagedWorkItem, ...]) -> None:
        """Require all issued request tokens to be inactive and joined."""
        with self._managed_claims.lock:
            for item in items:
                state = self._managed_claims.by_token[item.frame.token]
                if state.active_attempt is not None or not state.joined:
                    raise RuntimeError(
                        "managed workflow request exited before all work items joined"
                    )

    def assert_all_managed_items_joined(self) -> None:
        """Prevent the parent Function broker from releasing active ownership."""
        with self._managed_claims.lock:
            if any(
                state.active_attempt is not None or not state.joined
                for state in self._managed_claims.by_unit.values()
            ):
                raise RuntimeError(
                    "a Function workflow scope cannot exit before all managed work items join"
                )

    def prepare_remote_managed_unit(
        self,
        frame: ManagedUnitFrame,
    ) -> ManagedParentEnvelope:
        """Materialize parent authority after a remote task requests randomness."""
        with self._managed_claims.lock:
            state = self._managed_claims.by_token.get(frame.token)
            if state is None or state.frame.unit_segment != frame.unit_segment:
                raise RuntimeError("remote managed unit is not registered by this parent")
        parent_invocation = self._ensure_parent_invocation()
        return ManagedParentEnvelope(
            root_words=_workflow_context._resolve_root_words(parent_invocation.frame),
            parent_occurrence_path=parent_invocation.occurrence_path,
            frame=frame,
        )

    def accept_remote_claim_report(self, report: ManagedClaimReport) -> None:
        """Join and reconcile the canonical child namespace from a remote attempt."""
        with self._managed_claims.lock:
            state = self._managed_claims.by_token.get(report.frame.token)
            if state is None or state.frame.unit_segment != report.frame.unit_segment:
                raise RuntimeError("remote claim report does not own a registered unit")
            if state.active_attempt is not None:
                raise RuntimeError("remote claim report conflicts with an active local attempt")
            if report.attempt.attempt_token in state.seen_attempts:
                raise RuntimeError("a managed attempt token cannot be reused")
            state.seen_attempts.add(report.attempt.attempt_token)
            state.has_started = True
            state.joined = True

            if report.child_count == 0:
                return
            parent_invocation = self._ensure_parent_invocation()
            while len(state.child_invocations) < report.child_count:
                child_ordinal = len(state.child_invocations)
                state.child_invocations.append(
                    _workflow_context._WorkflowInvocation(
                        frame=parent_invocation.frame,
                        occurrence_path=(
                            *parent_invocation.occurrence_path,
                            report.frame.unit_segment,
                            ("child", child_ordinal),
                        ),
                    )
                )

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
                if self._frame is None:
                    raise RuntimeError("parent broker has no workflow frame")
                self._invocation = _workflow_context._commit_stochastic_invocation_in_frame(
                    self._frame,
                    self.occurrence_kind,
                )
            return self._invocation


_ACTIVE_AUTOMATIC_KEY_BROKER: ContextVar[_AutomaticKeyBroker | None] = ContextVar(
    "probpipe_active_automatic_key_broker",
    default=None,
)

_ACTIVE_MANAGED_ATTEMPT: ContextVar[_ManagedAttemptContext | None] = ContextVar(
    "probpipe_active_managed_attempt",
    default=None,
)

_REMOTE_COORDINATION_PROBE: ContextVar[bool] = ContextVar(
    "probpipe_remote_coordination_probe",
    default=False,
)


class _ManagedCoordinationRequired(RuntimeError):
    """Signal that a remote work item needs parent RNG authority."""


@dataclass
class _RemoteManagedParent:
    """Worker-local adapter for a parent-authorized child namespace."""

    envelope: ManagedParentEnvelope
    attempt: ManagedAttemptState
    workflow_frame: _workflow_context._WorkflowFrame
    child_invocations: list[_workflow_context._WorkflowInvocation] = field(default_factory=list)

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

    def report(self) -> ManagedClaimReport:
        """Return the serializable claim count for parent reconciliation."""
        return ManagedClaimReport(
            frame=self.envelope.frame,
            attempt=self.attempt,
            child_count=len(self.child_invocations),
        )


@contextmanager
def _function_stochastic_scope() -> Iterator[_AutomaticKeyBroker]:
    """Install a lazy broker for one public Function invocation."""
    _workflow_context._assert_workflow_admission()
    broker = _AutomaticKeyBroker(
        "invocation",
        _frame=_workflow_context._capture_active_workflow_frame(),
        _managed_attempt=_ACTIVE_MANAGED_ATTEMPT.get(),
    )
    token: Token[_AutomaticKeyBroker | None] = _ACTIVE_AUTOMATIC_KEY_BROKER.set(broker)
    try:
        yield broker
    finally:
        try:
            broker.assert_all_managed_items_joined()
        finally:
            _ACTIVE_AUTOMATIC_KEY_BROKER.reset(token)


@contextmanager
def _managed_stochastic_scope() -> Iterator[_AutomaticKeyBroker]:
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
            _managed_attempt=managed_attempt,
        )
        token: Token[_AutomaticKeyBroker | None] = _ACTIVE_AUTOMATIC_KEY_BROKER.set(broker)
        try:
            yield broker
        finally:
            _ACTIVE_AUTOMATIC_KEY_BROKER.reset(token)
        return

    with _workflow_context._ephemeral_workflow_run():
        broker = _AutomaticKeyBroker(
            "operation",
            _frame=_workflow_context._capture_active_workflow_frame(),
        )
        token: Token[_AutomaticKeyBroker | None] = _ACTIVE_AUTOMATIC_KEY_BROKER.set(broker)
        try:
            yield broker
        finally:
            _ACTIVE_AUTOMATIC_KEY_BROKER.reset(token)


def _capture_active_broker() -> _AutomaticKeyBroker | None:
    """Capture the admitted parent broker for managed execution transport."""
    _workflow_context._assert_workflow_admission()
    return _ACTIVE_AUTOMATIC_KEY_BROKER.get()


def _record_active_execution_contract(
    contract: WorkflowRngExecutionContract,
) -> None:
    """Attach an actual route contract to the current public invocation."""
    broker = _ACTIVE_AUTOMATIC_KEY_BROKER.get()
    if broker is not None:
        broker.record_execution_contract(contract)


@contextmanager
def _remote_coordination_probe_scope() -> Iterator[None]:
    """Run a remote item without permitting automatic stochastic commit."""
    probe_token = _REMOTE_COORDINATION_PROBE.set(True)
    attempt_token = _ACTIVE_MANAGED_ATTEMPT.set(None)
    broker_token = _ACTIVE_AUTOMATIC_KEY_BROKER.set(None)
    try:
        yield
    finally:
        _ACTIVE_AUTOMATIC_KEY_BROKER.reset(broker_token)
        _ACTIVE_MANAGED_ATTEMPT.reset(attempt_token)
        _REMOTE_COORDINATION_PROBE.reset(probe_token)


@contextmanager
def _remote_managed_work_item_stochastic_scope(
    envelope: ManagedParentEnvelope,
    attempt: ManagedAttemptState,
) -> Iterator[_RemoteManagedParent]:
    """Install parent-authorized RNG derivation inside a remote worker."""
    frame = _workflow_context._capture_active_workflow_frame()
    if frame is None:
        raise RuntimeError("remote managed randomness requires a transported frame")
    parent = _RemoteManagedParent(
        envelope=envelope,
        attempt=attempt,
        workflow_frame=frame,
    )
    state = _ManagedAttemptContext(
        parent_broker=parent,
        frame=envelope.frame,
        attempt=attempt,
    )
    attempt_token = _ACTIVE_MANAGED_ATTEMPT.set(state)
    broker_token = _ACTIVE_AUTOMATIC_KEY_BROKER.set(None)
    try:
        yield parent
    finally:
        _ACTIVE_AUTOMATIC_KEY_BROKER.reset(broker_token)
        _ACTIVE_MANAGED_ATTEMPT.reset(attempt_token)


@contextmanager
def _managed_work_item_stochastic_scope(
    parent_broker: _AutomaticKeyBroker,
    frame: ManagedUnitFrame,
    *,
    attempt: ManagedAttemptState | None = None,
) -> Iterator[ManagedAttemptState]:
    """Install one retry attempt and an empty child-broker slot."""
    if attempt is None:
        attempt = ManagedAttemptState.create(frame.token)
    registered_frame = parent_broker.begin_managed_attempt(attempt)
    if registered_frame != frame:
        raise RuntimeError("managed work-item frame does not match its registered token")
    state = _ManagedAttemptContext(
        parent_broker=parent_broker,
        frame=frame,
        attempt=attempt,
    )
    attempt_token = _ACTIVE_MANAGED_ATTEMPT.set(state)
    broker_token = _ACTIVE_AUTOMATIC_KEY_BROKER.set(None)
    try:
        yield attempt
    finally:
        _ACTIVE_AUTOMATIC_KEY_BROKER.reset(broker_token)
        _ACTIVE_MANAGED_ATTEMPT.reset(attempt_token)
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
