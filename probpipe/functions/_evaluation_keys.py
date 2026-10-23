"""Keys for the draws made inside evaluation points, and the automatic-key broker.

An evaluation point is one evaluation of a callable at one set of arguments,
which ProbPipe runs for the caller in place, in a worker thread, or in a Prefect
task. This private module holds the serializable payloads that carry an
evaluation point to where it runs, the segment of the occurrence path that names
it, and the broker that serves the key of each workflow-owned draw. A draw made
inside an evaluation point claims the next child occurrence of that point, so its
key is the same whichever thread, task, or retry runs the point.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ..custom_types import PRNGKey
from . import _context
from ._rng import _RandomEventPath, _validate_random_event_value

if TYPE_CHECKING:
    from ._callable import CallableAnchor
    from ._execution import WorkflowRngExecutionContract

#: The tag and the version that start each evaluation point's segment of the
#: occurrence path, and the version of the rule that numbers the draws made
#: inside an evaluation point. Each value enters every key derived inside an
#: evaluation point or every replay record, so changing one changes those
#: keys and refuses the replay records written before the change.
_EVALUATION_POINT_TAG = "evaluation-point"
_EVALUATION_POINT_ABI = "probpipe.evaluation_point/v1"
_EVALUATION_POINT_CHILD_POLICY_ABI = "probpipe.evaluation_point_child/v1"

_DISTRIBUTION_SAMPLING_ABI = "probpipe.distribution_sampling/v1"
_PROBPIPE_DISTRIBUTION_PROVIDER_ABI = "probpipe.distribution/v1"

type _CanonicalDescriptorValue = (
    None | bool | int | str | bytes | tuple[_CanonicalDescriptorValue, ...]
)
type _CanonicalDescriptor = tuple[_CanonicalDescriptorValue, ...]


@dataclass(frozen=True)
class EvaluationPointToken:
    """Opaque, serializable ownership token for one evaluation point."""

    value: bytes

    def __post_init__(self) -> None:
        _validate_evaluation_point_token_value(self.value)

    @classmethod
    def create(cls) -> EvaluationPointToken:
        """Create one process-independent operational token."""
        return cls(uuid.uuid4().bytes)


@dataclass(frozen=True)
class EvaluationPointFrame:
    """Canonical point-segment binding transported with one evaluation point."""

    point_segment: _RandomEventPath
    token: EvaluationPointToken
    derivation_abi: str = _EVALUATION_POINT_ABI

    def __post_init__(self) -> None:
        _validate_evaluation_point_frame_fields(
            point_segment=self.point_segment,
            token=self.token,
            derivation_abi=self.derivation_abi,
        )


@dataclass(frozen=True)
class IssuedEvaluationPoint:
    """One evaluation point that a parent broker issued, frozen for its runs.

    An evaluation point is one evaluation of a callable at one set of
    arguments. :func:`issue_evaluation_points` makes one for each evaluation
    point of a set, in point order, and the parent broker registers each
    before any runs. The point then runs in place, in a worker thread, or in a
    Prefect task, once per attempt, and every attempt runs under the same
    frame, so its draws keep their keys across retries. It is immutable and
    can be pickled. Construction raises ``TypeError`` for a negative or
    non-integer index or for a frame that is not valid.

    Attributes
    ----------
    index : int
        Its point index: its position within its set, from zero.
    frame : EvaluationPointFrame
        Its segment of the occurrence path and its ownership token, which
        name it in the keys of its draws and in its parent's registry.
    """

    index: int
    frame: EvaluationPointFrame

    def __post_init__(self) -> None:
        _validate_issued_evaluation_point_fields(index=self.index, frame=self.frame)


@dataclass(frozen=True)
class EvaluationPointAttempt:
    """Operational identity for one execution attempt of an evaluation point."""

    evaluation_point_token: EvaluationPointToken
    attempt_token: bytes

    def __post_init__(self) -> None:
        _validate_evaluation_point_attempt_fields(
            evaluation_point_token=self.evaluation_point_token,
            attempt_token=self.attempt_token,
        )

    @classmethod
    def create(cls, evaluation_point_token: EvaluationPointToken) -> EvaluationPointAttempt:
        """Create a fresh attempt identity for an existing evaluation point."""
        return cls(evaluation_point_token=evaluation_point_token, attempt_token=uuid.uuid4().bytes)


@dataclass(frozen=True)
class EvaluationPointParentEnvelope:
    """Serializable root and occurrence authority for one remote evaluation point."""

    root_words: tuple[int, int]
    parent_occurrence_path: _RandomEventPath
    frame: EvaluationPointFrame
    attempt: EvaluationPointAttempt
    replay_expected_effects: tuple[StochasticEffectClaim, ...] | None = None
    retry_effects: tuple[StochasticEffectClaim, ...] = ()

    def __post_init__(self) -> None:
        _validate_evaluation_point_parent_envelope_fields(
            root_words=self.root_words,
            parent_occurrence_path=self.parent_occurrence_path,
            frame=self.frame,
            attempt=self.attempt,
            replay_expected_effects=self.replay_expected_effects,
            retry_effects=self.retry_effects,
        )


@dataclass(frozen=True)
class StochasticEffectClaim:
    """Serializable descriptor of one automatic stochastic effect claim."""

    occurrence_path: _RandomEventPath
    occurrence_kind: str
    stochastic_source_id: _RandomEventPath
    logical_unit_id: _RandomEventPath
    operation_kind: str
    execution_mode: str
    sample_shape: tuple[int, ...] | None
    sampling_abi: str
    provider_abi: str
    record_path: tuple[str, ...] = ()
    descendant_descriptor: _CanonicalDescriptor | None = None

    def __post_init__(self) -> None:
        _validate_effect_claim_fields(
            occurrence_path=self.occurrence_path,
            occurrence_kind=self.occurrence_kind,
            stochastic_source_id=self.stochastic_source_id,
            logical_unit_id=self.logical_unit_id,
            operation_kind=self.operation_kind,
            execution_mode=self.execution_mode,
            sample_shape=self.sample_shape,
            sampling_abi=self.sampling_abi,
            provider_abi=self.provider_abi,
            record_path=self.record_path,
            descendant_descriptor=self.descendant_descriptor,
        )


@dataclass(frozen=True)
class EvaluationPointClaimReport:
    """Serializable claim summary returned by one remote attempt."""

    frame: EvaluationPointFrame
    attempt: EvaluationPointAttempt
    child_count: int
    effects: tuple[StochasticEffectClaim, ...] = ()
    successful_effects: tuple[StochasticEffectClaim, ...] = ()

    def __post_init__(self) -> None:
        _validate_evaluation_point_claim_report_fields(
            frame=self.frame,
            attempt=self.attempt,
            child_count=self.child_count,
            effects=self.effects,
            successful_effects=self.successful_effects,
        )


def plain_call_point_segment() -> _RandomEventPath:
    """Return the canonical segment of the one evaluation point of a plain call."""
    return (_EVALUATION_POINT_TAG, _EVALUATION_POINT_ABI, "point", 0)


def sweep_cell_point_segment(coordinates: tuple[int, ...]) -> _RandomEventPath:
    """Return the canonical segment for one row-major sweep cell."""
    if not isinstance(coordinates, tuple):
        raise TypeError("sweep cell coordinates must be a tuple")
    if not coordinates:
        raise ValueError("sweep cell segments require at least one coordinate")
    if any(
        isinstance(coordinate, bool) or not isinstance(coordinate, int)
        for coordinate in coordinates
    ):
        raise TypeError("sweep cell coordinates must be non-boolean integers")
    if any(coordinate < 0 for coordinate in coordinates):
        raise ValueError("sweep cell coordinates must be non-negative")
    return (_EVALUATION_POINT_TAG, _EVALUATION_POINT_ABI, "sweep-cell", *coordinates)


def lifted_evaluation_point_segment(
    logical_unit_id: tuple[str | int, ...],
    flat_index: int,
) -> _RandomEventPath:
    """Return the canonical segment for one evaluation within a lifted unit."""
    if not isinstance(logical_unit_id, tuple):
        raise TypeError("lifted logical-unit identities must be tuples")
    if not _is_logical_unit_id(logical_unit_id):
        raise ValueError("lifted logical-unit identities must be canonical")
    if isinstance(flat_index, bool) or not isinstance(flat_index, int):
        raise TypeError("lifted evaluation indexes must be non-boolean integers")
    if flat_index < 0:
        raise ValueError("lifted evaluation indexes must be non-negative")
    return (
        _EVALUATION_POINT_TAG,
        _EVALUATION_POINT_ABI,
        "lifted-evaluation",
        logical_unit_id,
        flat_index,
    )


def issue_evaluation_points(
    point_segments: Sequence[_RandomEventPath],
) -> tuple[IssuedEvaluationPoint, ...]:
    """One issued evaluation point for each of *point_segments*, in order, each with a new token."""
    return tuple(
        IssuedEvaluationPoint(
            index=index,
            frame=EvaluationPointFrame(
                point_segment=tuple(point_segment),
                token=EvaluationPointToken.create(),
            ),
        )
        for index, point_segment in enumerate(point_segments)
    )


# -- The automatic-key broker ------------------------------------------------

_OccurrenceKind = Literal["invocation", "operation"]
_StructuralRngId = _RandomEventPath


class _EvaluationPointStatus(Enum):
    """Lifecycle state of one parent-issued evaluation point."""

    ISSUED = "issued"
    ACTIVE = "active"
    JOINED = "joined"
    CANCELLED = "cancelled"


class _BrokerLifecycle(Enum):
    """Lifecycle of one owned automatic-key broker."""

    OPEN = "open"
    FINALIZING = "finalizing"
    SEALED = "sealed"


class _RandomEventPlan(Protocol):
    """Structural event fields accepted by the broker."""

    @property
    def stochastic_source_id(self) -> _StructuralRngId: ...
    @property
    def logical_unit_id(self) -> _StructuralRngId: ...


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
class _EvaluationPointClaimState:
    """Operational retry state for one canonical evaluation point."""

    frame: EvaluationPointFrame
    child_invocations: list[_context._WorkflowInvocation] = field(default_factory=list)
    active_attempt: bytes | None = None
    active_transport: Literal["local", "remote"] | None = None
    active_parent_occurrence_path: _RandomEventPath | None = None
    seen_attempts: set[bytes] = field(default_factory=set)
    active_effect_identities: set[tuple[Any, ...]] = field(default_factory=set)
    effect_claims_by_identity: dict[tuple[Any, ...], StochasticEffectClaim] = field(
        default_factory=dict
    )
    status: _EvaluationPointStatus = _EvaluationPointStatus.ISSUED


@dataclass(slots=True)
class _EvaluationPointClaimRegistry:
    """Parent-broker registry for evaluation points and their retry claims."""

    by_point_segment: dict[tuple[Any, ...], _EvaluationPointClaimState] = field(
        default_factory=dict
    )
    by_token: dict[EvaluationPointToken, _EvaluationPointClaimState] = field(default_factory=dict)
    lock: Any = field(default_factory=Lock, repr=False)


@dataclass(slots=True)
class _EvaluationPointAttemptContext:
    """Attempt-local child cursor bound inside one evaluation point."""

    parent_broker: Any
    frame: EvaluationPointFrame
    attempt: EvaluationPointAttempt
    workflow_frame: _context._WorkflowFrame
    next_child_ordinal: int = 0
    claimed_effects_by_identity: dict[tuple[Any, ...], StochasticEffectClaim] = field(
        default_factory=dict
    )
    successful_effects_by_identity: dict[tuple[Any, ...], StochasticEffectClaim] = field(
        default_factory=dict
    )

    def claim_child_invocation(self) -> _context._WorkflowInvocation:
        """Claim or retry the next child occurrence in canonical order."""
        ordinal = self.next_child_ordinal
        self.next_child_ordinal += 1
        return self.parent_broker._claim_evaluation_point_child(
            frame=self.frame,
            attempt=self.attempt,
            child_ordinal=ordinal,
        )

    def planned_child_invocation(self) -> _context._WorkflowInvocation:
        """The child occurrence ``claim_child_invocation`` would claim next, claiming nothing."""
        return self.parent_broker._planned_evaluation_point_child(
            frame=self.frame,
            attempt=self.attempt,
            child_ordinal=self.next_child_ordinal,
        )

    def claim_scoped_child_invocation(
        self,
        workflow_frame: _context._WorkflowFrame,
        occurrence_kind: _OccurrenceKind,
    ) -> _context._WorkflowInvocation:
        """Claim a child whose active nested run supplies its root and scope path."""
        point_child_invocation = self.claim_child_invocation()
        scope_path = _context._materialize_descendant_path(
            workflow_frame,
            self.workflow_frame,
        )
        local_ordinal = workflow_frame.ledger.commit()
        return _context._WorkflowInvocation(
            frame=workflow_frame,
            occurrence_path=(
                *point_child_invocation.occurrence_path,
                *scope_path,
                (occurrence_kind, local_ordinal),
            ),
        )

    def claim_effect(self, effect: StochasticEffectClaim) -> None:
        """Retain one attempt claim without making it durable recipe state."""
        self.parent_broker.claim_evaluation_point_effect(
            effect,
            frame=self.frame,
            attempt=self.attempt,
        )
        self.claimed_effects_by_identity[_effect_identity(effect)] = effect

    def accept_successful_effects(self, effects: tuple[StochasticEffectClaim, ...]) -> None:
        """Collect effects from child scopes that completed successfully."""
        for effect in effects:
            identity = _effect_identity(effect)
            claimed = self.claimed_effects_by_identity.get(identity)
            if claimed is None:
                self.claim_effect(effect)
            elif claimed != effect:
                raise RuntimeError(
                    "a stochastic event identity completed with a different effect plan"
                )
            existing_effect = self.successful_effects_by_identity.get(identity)
            if existing_effect is not None and existing_effect != effect:
                raise RuntimeError(
                    "a stochastic event identity completed with a different effect plan"
                )
            self.successful_effects_by_identity[identity] = effect

    def publish_successful_effects(self) -> None:
        """Promote successful effects through the containing broker boundary."""
        self.parent_broker.accept_successful_evaluation_point_effects(
            tuple(self.successful_effects_by_identity.values()),
            frame=self.frame,
            attempt=self.attempt,
        )


def _effect_identity(effect: StochasticEffectClaim) -> tuple[Any, ...]:
    """Return the canonical identity shared by retry and recipe ledgers."""
    return (
        effect.occurrence_path,
        effect.stochastic_source_id,
        effect.logical_unit_id,
    )


def _evaluation_point_effect_child_ordinal(
    effect: StochasticEffectClaim,
    *,
    prefix: _RandomEventPath,
) -> int:
    """Validate one report occurrence namespace and return its child ordinal."""
    occurrence_path = effect.occurrence_path
    if occurrence_path[: len(prefix)] != prefix or len(occurrence_path) <= len(prefix):
        raise RuntimeError(
            "a remote effect occurrence path is outside its evaluation point namespace"
        )
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


@dataclass(frozen=True)
class _BrokerRecipeSnapshot:
    """Successful-invocation data safe for canonical recipe serialization."""

    root_words: tuple[int, int]
    occurrence_path: tuple[Any, ...]
    rng_origin: dict[str, str | int | None]
    effects: tuple[StochasticEffectClaim, ...]
    execution_contracts: tuple[WorkflowRngExecutionContract, ...]
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
    _frame: _context._WorkflowFrame | None = None
    _invocation: _context._WorkflowInvocation | None = None
    _planned_invocation: _context._WorkflowInvocation | None = None
    _evaluation_point_attempt: _EvaluationPointAttemptContext | None = None
    _evaluation_point_claims: _EvaluationPointClaimRegistry = field(
        default_factory=_EvaluationPointClaimRegistry
    )
    _lifecycle: _BrokerLifecycle = field(
        default=_BrokerLifecycle.OPEN,
        init=False,
    )
    _execution_contracts: list[WorkflowRngExecutionContract] = field(default_factory=list)
    _effects_by_identity: dict[tuple[Any, ...], StochasticEffectClaim] = field(default_factory=dict)
    _effects_lock: Any = field(default_factory=Lock, repr=False)
    _lock: Any = field(default_factory=Lock, repr=False)
    _callable_anchor: CallableAnchor | None = None
    _replay_state: Any = None

    def key_for(self, plan: StochasticEffectPlan) -> PRNGKey:
        """Return the workflow-owned key for one planned effect."""
        _context._assert_workflow_admission(self._frame)
        self._assert_broker_open()
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
        _context._guard_automatic_key_request()
        _context._guard_caller_jax_trace(plan.operation_kind)
        self.validate_replay_effect_plan(plan)
        invocation = self._ensure_parent_invocation()
        effect = StochasticEffectClaim(
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
        if self._evaluation_point_attempt is None:
            self.claim_replay_effect(effect, attempt=None)
            self._record_effect(effect)
        else:
            parent_broker = self._evaluation_point_attempt.parent_broker
            claim_replay_effect = getattr(parent_broker, "claim_replay_effect", None)
            if claim_replay_effect is None:
                from . import _replay

                _replay._claim_effect_before_derivation(
                    effect,
                    attempt=self._evaluation_point_attempt.attempt,
                )
            else:
                claim_replay_effect(
                    effect,
                    attempt=self._evaluation_point_attempt.attempt,
                )
            self._evaluation_point_attempt.claim_effect(effect)
            self._record_effect(effect)
        return invocation.key_for(
            stochastic_source_id=source_id,
            logical_unit_id=unit_id,
        )

    def register_evaluation_points(
        self, evaluation_points: tuple[IssuedEvaluationPoint, ...]
    ) -> None:
        """Register every issued evaluation point token before request submission."""
        with self._evaluation_point_claims.lock:
            self._assert_broker_open_unlocked()
            by_point_segment = dict(self._evaluation_point_claims.by_point_segment)
            by_token = dict(self._evaluation_point_claims.by_token)
            for evaluation_point in evaluation_points:
                point_segment = evaluation_point.frame.point_segment
                existing_claim_state = by_point_segment.get(point_segment)
                if (
                    existing_claim_state is not None
                    and existing_claim_state.frame.token != evaluation_point.frame.token
                ):
                    raise RuntimeError(
                        "an evaluation point cannot be reused with a different token"
                    )
                token_owner = by_token.get(evaluation_point.frame.token)
                if token_owner is not None and token_owner.frame.point_segment != point_segment:
                    raise RuntimeError(
                        "an evaluation point token cannot own multiple point segments"
                    )
                if existing_claim_state is None:
                    existing_claim_state = _EvaluationPointClaimState(frame=evaluation_point.frame)
                    by_point_segment[point_segment] = existing_claim_state
                    by_token[evaluation_point.frame.token] = existing_claim_state
            self._evaluation_point_claims.by_point_segment = by_point_segment
            self._evaluation_point_claims.by_token = by_token

    def record_execution_contract(self, contract: WorkflowRngExecutionContract) -> None:
        """Record one distinct capability-checked route for later diagnostics."""
        self._assert_broker_open()
        if contract not in self._execution_contracts:
            self._execution_contracts.append(contract)

    def set_callable_anchor(self, anchor: CallableAnchor) -> None:
        """Attach the immutable definition anchor for this public Function."""
        self._assert_broker_open()
        if self._callable_anchor is not None:
            raise RuntimeError("a Function broker already has a callable anchor")
        self._callable_anchor = anchor

    def claim_replay_effect(
        self,
        effect: StochasticEffectClaim,
        *,
        attempt: EvaluationPointAttempt | None,
    ) -> None:
        """Validate one local/transported effect against captured replay state."""
        self._assert_broker_open()
        if self._replay_state is not None:
            self._replay_state.claim_effect(effect, attempt=attempt)
            return
        if self._evaluation_point_attempt is not None:
            parent_broker = self._evaluation_point_attempt.parent_broker
            parent_claim = getattr(parent_broker, "claim_replay_effect", None)
            if parent_claim is not None:
                parent_claim(effect, attempt=attempt)
                return
        from . import _replay

        _replay._claim_effect_before_derivation(effect, attempt=attempt)

    def validate_replay_effect_plan(self, plan: StochasticEffectPlan) -> None:
        """Reject direct-operation plan drift before committing its occurrence."""
        self._assert_broker_open()
        if self._replay_state is not None:
            self._replay_state.validate_effect_plan(plan)
            return
        if self._evaluation_point_attempt is not None:
            parent_broker = self._evaluation_point_attempt.parent_broker
            parent_validate = getattr(parent_broker, "validate_replay_effect_plan", None)
            if parent_validate is not None:
                parent_validate(plan)
                return
        from . import _replay

        _replay._validate_effect_plan_before_commit(plan)

    def claim_evaluation_point_effect(
        self,
        effect: StochasticEffectClaim,
        *,
        frame: EvaluationPointFrame,
        attempt: EvaluationPointAttempt,
    ) -> None:
        """Retain retry compatibility state before deriving an evaluation point's key."""
        with self._evaluation_point_claims.lock:
            self._assert_broker_open_unlocked()
            claim_state = self._evaluation_point_claims.by_token.get(attempt.evaluation_point_token)
            if (
                claim_state is None
                or claim_state.frame != frame
                or claim_state.status is not _EvaluationPointStatus.ACTIVE
                or claim_state.active_attempt != attempt.attempt_token
                or claim_state.active_transport != "local"
            ):
                raise RuntimeError("evaluation point effect claim does not own the active attempt")
            identity = _effect_identity(effect)
            existing_effect = claim_state.effect_claims_by_identity.get(identity)
            if existing_effect is not None and existing_effect != effect:
                raise RuntimeError(
                    "a stochastic event identity was retried with a different effect plan"
                )
            if identity in claim_state.active_effect_identities:
                raise RuntimeError(
                    "one evaluation point attempt duplicated a stochastic effect claim"
                )
            claim_state.active_effect_identities.add(identity)
            claim_state.effect_claims_by_identity[identity] = effect

    def accept_successful_evaluation_point_effects(
        self,
        effects: tuple[StochasticEffectClaim, ...],
        *,
        frame: EvaluationPointFrame,
        attempt: EvaluationPointAttempt,
    ) -> None:
        """Make effects from one successful evaluation point boundary recipe-visible."""
        with self._evaluation_point_claims.lock:
            self._assert_broker_open_unlocked()
            if not isinstance(effects, tuple) or any(
                not isinstance(effect, StochasticEffectClaim) for effect in effects
            ):
                raise TypeError("successful evaluation point effects must be an effect tuple")
            claim_state = self._evaluation_point_claims.by_token.get(attempt.evaluation_point_token)
            if (
                claim_state is None
                or claim_state.frame != frame
                or claim_state.status is not _EvaluationPointStatus.ACTIVE
                or claim_state.active_attempt != attempt.attempt_token
                or claim_state.active_transport != "local"
            ):
                raise RuntimeError(
                    "successful evaluation point effects do not own the active attempt"
                )
            identities = set()
            for effect in effects:
                identity = _effect_identity(effect)
                if identity in identities:
                    raise RuntimeError(
                        "successful evaluation point effects contain a duplicate identity"
                    )
                identities.add(identity)
                if (
                    identity not in claim_state.active_effect_identities
                    or claim_state.effect_claims_by_identity.get(identity) != effect
                ):
                    raise RuntimeError(
                        "a successful evaluation point effect was not claimed by its active attempt"
                    )
            with self._effects_lock:
                for effect in effects:
                    existing_effect = self._effects_by_identity.get(_effect_identity(effect))
                    if existing_effect is not None and existing_effect != effect:
                        raise RuntimeError(
                            "a stochastic event identity completed with a different effect plan"
                        )
                if self._replay_state is not None:
                    self._replay_state.mark_successful_effects(
                        effects,
                        attempt=attempt,
                    )
                self._effects_by_identity.update(
                    (_effect_identity(effect), effect) for effect in effects
                )

    def _publish_effects_to_evaluation_point(
        self,
        effects: tuple[StochasticEffectClaim, ...],
    ) -> None:
        """Publish this successful broker's effects to its evaluation point attempt."""
        self._assert_broker_finalizing()
        if self._evaluation_point_attempt is None:
            return
        self._evaluation_point_attempt.accept_successful_effects(effects)

    def _mark_replay_effects_successful(
        self,
        effects: tuple[StochasticEffectClaim, ...],
    ) -> None:
        """Confirm the recipe-visible effects of a successful replay root."""
        self._assert_broker_finalizing()
        if (
            self._replay_state is None
            or self._invocation is None
            or self._invocation.occurrence_path != self._replay_state.occurrence_path
        ):
            return
        self._replay_state.finalize_successful_effects(effects)

    def _record_effect(self, effect: StochasticEffectClaim) -> None:
        identity = _effect_identity(effect)
        with self._evaluation_point_claims.lock:
            self._assert_broker_open_unlocked()
            with self._effects_lock:
                existing_effect = self._effects_by_identity.get(identity)
                if existing_effect is not None:
                    if existing_effect != effect:
                        raise RuntimeError(
                            "a stochastic event identity was retried with a different effect plan"
                        )
                    raise RuntimeError(
                        "one stochastic broker scope duplicated a stochastic effect claim"
                    )
                self._effects_by_identity[identity] = effect

    def validate_evaluation_point_attempt_preflight(
        self,
        attempt: EvaluationPointAttempt,
        frame: EvaluationPointFrame,
    ) -> None:
        """Validate an attempt and frame without reserving registry state."""
        with self._evaluation_point_claims.lock:
            self._validate_evaluation_point_attempt_unlocked(attempt, frame)

    def begin_evaluation_point_attempt(
        self,
        attempt: EvaluationPointAttempt,
        frame: EvaluationPointFrame,
    ) -> EvaluationPointFrame:
        """Admit one fresh attempt for a previously registered evaluation point token."""
        with self._evaluation_point_claims.lock:
            claim_state = self._validate_evaluation_point_attempt_unlocked(attempt, frame)
            claim_state.seen_attempts.add(attempt.attempt_token)
            claim_state.active_attempt = attempt.attempt_token
            claim_state.active_transport = "local"
            claim_state.active_parent_occurrence_path = None
            claim_state.active_effect_identities.clear()
            claim_state.status = _EvaluationPointStatus.ACTIVE
            return claim_state.frame

    def finish_evaluation_point_attempt(self, attempt: EvaluationPointAttempt) -> None:
        """Join one active attempt without discarding its retry claims."""
        with self._evaluation_point_claims.lock:
            self._assert_broker_open_unlocked()
            claim_state = self._evaluation_point_claims.by_token.get(attempt.evaluation_point_token)
            if (
                claim_state is None
                or claim_state.status is not _EvaluationPointStatus.ACTIVE
                or claim_state.active_attempt != attempt.attempt_token
                or claim_state.active_transport != "local"
            ):
                raise RuntimeError("evaluation point attempt is not active")
            claim_state.active_attempt = None
            claim_state.active_transport = None
            claim_state.active_parent_occurrence_path = None
            claim_state.active_effect_identities.clear()
            claim_state.status = _EvaluationPointStatus.JOINED

    def cancel_unstarted_evaluation_points(
        self, evaluation_points: tuple[IssuedEvaluationPoint, ...]
    ) -> None:
        """Cancel issued evaluation points that were never submitted after an earlier failure."""
        with self._evaluation_point_claims.lock:
            self._assert_broker_open_unlocked()
            for evaluation_point in evaluation_points:
                claim_state = self._evaluation_point_claims.by_token[evaluation_point.frame.token]
                if claim_state.status is _EvaluationPointStatus.ISSUED:
                    claim_state.status = _EvaluationPointStatus.CANCELLED

    def assert_evaluation_points_joined(
        self, evaluation_points: tuple[IssuedEvaluationPoint, ...]
    ) -> None:
        """Require all issued request tokens to be inactive and joined."""
        with self._evaluation_point_claims.lock:
            for evaluation_point in evaluation_points:
                claim_state = self._evaluation_point_claims.by_token[evaluation_point.frame.token]
                if claim_state.active_attempt is not None or claim_state.status not in {
                    _EvaluationPointStatus.JOINED,
                    _EvaluationPointStatus.CANCELLED,
                }:
                    raise RuntimeError(
                        "evaluation point request exited before all evaluation points joined"
                    )

    def _begin_finalization(self) -> tuple[StochasticEffectClaim, ...]:
        """Join-check, freeze successful effects, and enter finalization."""
        with self._evaluation_point_claims.lock:
            self._assert_broker_open_unlocked()
            if any(
                claim_state.active_attempt is not None
                or claim_state.status
                not in {_EvaluationPointStatus.JOINED, _EvaluationPointStatus.CANCELLED}
                for claim_state in self._evaluation_point_claims.by_point_segment.values()
            ):
                raise RuntimeError(
                    "a stochastic broker scope cannot exit before all evaluation points join"
                )
            self._lifecycle = _BrokerLifecycle.FINALIZING
            with self._effects_lock:
                return tuple(self._effects_by_identity.values())

    def _seal_finalization(self) -> None:
        """Permanently seal a broker after its terminal publication attempt."""
        with self._evaluation_point_claims.lock:
            if self._lifecycle is not _BrokerLifecycle.FINALIZING:
                raise RuntimeError("only a finalizing stochastic broker can be sealed")
            self._lifecycle = _BrokerLifecycle.SEALED

    def _finalize(self, *, successful: bool) -> None:
        """Seal one owned scope and publish only its pre-seal success snapshot."""
        effects = self._begin_finalization()
        try:
            if successful:
                self._publish_effects_to_evaluation_point(effects)
                self._mark_replay_effects_successful(effects)
        finally:
            self._seal_finalization()

    def _assert_broker_open(self) -> None:
        """Reject broker activity after scope finalization starts."""
        with self._evaluation_point_claims.lock:
            self._assert_broker_open_unlocked()

    def _assert_broker_open_unlocked(self) -> None:
        if self._lifecycle is _BrokerLifecycle.FINALIZING:
            raise RuntimeError("the stochastic workflow broker is finalizing")
        if self._lifecycle is _BrokerLifecycle.SEALED:
            raise RuntimeError("the stochastic workflow broker is sealed")

    def _assert_broker_finalizing(self) -> None:
        with self._evaluation_point_claims.lock:
            if self._lifecycle is not _BrokerLifecycle.FINALIZING:
                raise RuntimeError("terminal broker effects can only be published while finalizing")

    def _validate_evaluation_point_attempt_unlocked(
        self,
        attempt: EvaluationPointAttempt,
        frame: EvaluationPointFrame,
    ) -> _EvaluationPointClaimState:
        """Validate one attempt while the evaluation point registry lock is held."""
        self._assert_broker_open_unlocked()
        claim_state = self._evaluation_point_claims.by_token.get(attempt.evaluation_point_token)
        if claim_state is None:
            raise RuntimeError("evaluation point token was not registered by its parent")
        if claim_state.frame != frame:
            raise RuntimeError("evaluation point frame does not match its registered token")
        if claim_state.status is _EvaluationPointStatus.CANCELLED:
            raise RuntimeError("a cancelled evaluation point cannot start an attempt")
        if (
            claim_state.status is _EvaluationPointStatus.ACTIVE
            or claim_state.active_attempt is not None
        ):
            raise RuntimeError(
                "an evaluation point token already has an active attempt. "
                "Duplicate or concurrent attempts are not allowed"
            )
        if attempt.attempt_token in claim_state.seen_attempts:
            raise RuntimeError("an evaluation point attempt token cannot be reused")
        return claim_state

    def reserve_remote_evaluation_point_attempt(
        self,
        frame: EvaluationPointFrame,
        attempt: EvaluationPointAttempt,
    ) -> EvaluationPointParentEnvelope:
        """Reserve a remote attempt before submission and bind the parent's authority to it."""
        with self._evaluation_point_claims.lock:
            claim_state = self._validate_evaluation_point_attempt_unlocked(attempt, frame)
            claim_state.seen_attempts.add(attempt.attempt_token)
            claim_state.active_attempt = attempt.attempt_token
            claim_state.active_transport = "remote"
            claim_state.active_parent_occurrence_path = None
            claim_state.active_effect_identities.clear()
            claim_state.status = _EvaluationPointStatus.ACTIVE
            retry_effects = tuple(claim_state.effect_claims_by_identity.values())
        try:
            envelope = self._make_remote_parent_envelope(
                frame,
                attempt,
                retry_effects=retry_effects,
            )
            with self._evaluation_point_claims.lock:
                claim_state = self._require_active_remote_attempt_unlocked(attempt, frame)
                claim_state.active_parent_occurrence_path = envelope.parent_occurrence_path
            return envelope
        except BaseException:
            self.abort_remote_evaluation_point_attempt(attempt)
            raise

    def _make_remote_parent_envelope(
        self,
        frame: EvaluationPointFrame,
        attempt: EvaluationPointAttempt,
        *,
        retry_effects: tuple[StochasticEffectClaim, ...],
    ) -> EvaluationPointParentEnvelope:
        """Materialize authority for an already-reserved remote attempt.

        The envelope carries the call's occurrence as planned, which a report
        commits only when it names a draw, so an evaluation point that draws
        nothing leaves the workflow's later keys unchanged, as it does in place.
        """
        parent_invocation = self._reserved_parent_invocation()
        from . import _replay

        return EvaluationPointParentEnvelope(
            root_words=_context._resolve_root_words(parent_invocation.frame),
            parent_occurrence_path=parent_invocation.occurrence_path,
            frame=frame,
            attempt=attempt,
            replay_expected_effects=_replay._expected_effects_for_evaluation_point(
                parent_invocation.occurrence_path,
                frame.point_segment,
            ),
            retry_effects=retry_effects,
        )

    def abort_remote_evaluation_point_attempt(self, attempt: EvaluationPointAttempt) -> None:
        """Release one exact remote reservation after failure or cancellation."""
        with self._evaluation_point_claims.lock:
            self._assert_broker_open_unlocked()
            claim_state = self._evaluation_point_claims.by_token.get(attempt.evaluation_point_token)
            if claim_state is None or attempt.attempt_token not in claim_state.seen_attempts:
                raise RuntimeError(
                    "remote evaluation point attempt was not reserved by this parent"
                )
            if (
                claim_state.active_attempt is None
                and claim_state.status is _EvaluationPointStatus.JOINED
            ):
                return
            if (
                claim_state.status is not _EvaluationPointStatus.ACTIVE
                or claim_state.active_attempt != attempt.attempt_token
                or claim_state.active_transport != "remote"
            ):
                raise RuntimeError(
                    "remote evaluation point attempt does not own the active reservation"
                )
            claim_state.active_attempt = None
            claim_state.active_transport = None
            claim_state.active_parent_occurrence_path = None
            claim_state.active_effect_identities.clear()
            claim_state.status = _EvaluationPointStatus.JOINED

    def accept_remote_claim_report(self, report: EvaluationPointClaimReport) -> None:
        """Validate and atomically reconcile one reserved remote report."""
        self._assert_broker_open()
        report = _validated_evaluation_point_claim_report_snapshot(report)
        with self._evaluation_point_claims.lock:
            claim_state, child_invocations = self._validate_remote_report_unlocked(report)
            with self._effects_lock:
                for effect in report.successful_effects:
                    existing_effect = self._effects_by_identity.get(_effect_identity(effect))
                    if existing_effect is not None and existing_effect != effect:
                        raise RuntimeError(
                            "a stochastic event identity completed with a different effect plan"
                        )

                replay_claim_scope = (
                    self._replay_state._effect_batch_scope(
                        report.effects,
                        successful_effects=report.successful_effects,
                        attempt=report.attempt,
                    )
                    if self._replay_state is not None
                    else nullcontext()
                )
                with replay_claim_scope:
                    if report.effects or child_invocations:
                        self._ensure_parent_invocation()
                    for effect in report.effects:
                        claim_state.effect_claims_by_identity[_effect_identity(effect)] = effect
                    claim_state.child_invocations.extend(child_invocations)
                    self._effects_by_identity.update(
                        (_effect_identity(effect), effect) for effect in report.successful_effects
                    )
                    claim_state.active_attempt = None
                    claim_state.active_transport = None
                    claim_state.active_parent_occurrence_path = None
                    claim_state.active_effect_identities.clear()
                    claim_state.status = _EvaluationPointStatus.JOINED

    def _validate_remote_report_unlocked(
        self,
        report: EvaluationPointClaimReport,
    ) -> tuple[
        _EvaluationPointClaimState,
        tuple[_context._WorkflowInvocation, ...],
    ]:
        """Validate a remote report without changing any parent ledger."""
        claim_state = self._require_active_remote_attempt_unlocked(report.attempt, report.frame)
        parent_path = claim_state.active_parent_occurrence_path
        if parent_path is None:
            raise RuntimeError(
                "a remote evaluation point attempt reported before its parent authorized it"
            )

        point_path = (*parent_path, report.frame.point_segment)
        for effect in (*report.effects, *report.successful_effects):
            child_ordinal = _evaluation_point_effect_child_ordinal(effect, prefix=point_path)
            if child_ordinal >= report.child_count:
                raise RuntimeError(
                    "a remote effect child ordinal exceeds the reported child namespace"
                )
            identity = _effect_identity(effect)
            existing_effect = claim_state.effect_claims_by_identity.get(identity)
            if existing_effect is not None and existing_effect != effect:
                raise RuntimeError(
                    "a remote stochastic event was retried with a different effect plan"
                )

        if not report.effects and report.child_count <= len(claim_state.child_invocations):
            return claim_state, ()
        parent_invocation = self._reserved_parent_invocation()
        if parent_invocation.occurrence_path != parent_path:
            raise RuntimeError("remote report lost its reserved parent occurrence authority")
        child_invocations = tuple(
            _context._WorkflowInvocation(
                frame=parent_invocation.frame,
                occurrence_path=(
                    *parent_path,
                    report.frame.point_segment,
                    ("child", child_ordinal),
                ),
            )
            for child_ordinal in range(
                len(claim_state.child_invocations),
                report.child_count,
            )
        )
        return claim_state, child_invocations

    def _require_active_remote_attempt_unlocked(
        self,
        attempt: EvaluationPointAttempt,
        frame: EvaluationPointFrame,
    ) -> _EvaluationPointClaimState:
        """Return the exact active remote reservation while the registry is locked."""
        self._assert_broker_open_unlocked()
        claim_state = self._evaluation_point_claims.by_token.get(attempt.evaluation_point_token)
        if claim_state is None or claim_state.frame != frame:
            raise RuntimeError("remote evaluation point is not registered by this parent")
        if (
            claim_state.status is not _EvaluationPointStatus.ACTIVE
            or claim_state.active_attempt != attempt.attempt_token
            or claim_state.active_transport != "remote"
        ):
            raise RuntimeError(
                "remote evaluation point attempt does not own the active reservation"
            )
        return claim_state

    def _claim_evaluation_point_child(
        self,
        *,
        frame: EvaluationPointFrame,
        attempt: EvaluationPointAttempt,
        child_ordinal: int,
    ) -> _context._WorkflowInvocation:
        """Claim one retry-stable child occurrence for an evaluation point attempt."""
        with self._evaluation_point_claims.lock:
            self._assert_broker_open_unlocked()
            claim_state = self._evaluation_point_claims.by_token.get(frame.token)
            if (
                claim_state is None
                or claim_state.frame.point_segment != frame.point_segment
                or claim_state.status is not _EvaluationPointStatus.ACTIVE
                or claim_state.active_attempt != attempt.attempt_token
                or claim_state.active_transport != "local"
            ):
                raise RuntimeError("evaluation point child claim does not own the active attempt")
            if child_ordinal < len(claim_state.child_invocations):
                return claim_state.child_invocations[child_ordinal]
            if child_ordinal != len(claim_state.child_invocations):
                raise RuntimeError("evaluation point child claims must be made in ordinal order")

            parent_invocation = self._ensure_parent_invocation()
            invocation = _context._WorkflowInvocation(
                frame=parent_invocation.frame,
                occurrence_path=(
                    *parent_invocation.occurrence_path,
                    frame.point_segment,
                    ("child", child_ordinal),
                ),
            )
            claim_state.child_invocations.append(invocation)
            return invocation

    def _planned_evaluation_point_child(
        self,
        *,
        frame: EvaluationPointFrame,
        attempt: EvaluationPointAttempt,
        child_ordinal: int,
    ) -> _context._WorkflowInvocation:
        """The child occurrence ``_claim_evaluation_point_child`` would claim, claiming nothing."""
        with self._evaluation_point_claims.lock:
            claim_state = self._evaluation_point_claims.by_token.get(frame.token)
            if (
                claim_state is None
                or claim_state.frame.point_segment != frame.point_segment
                or claim_state.active_attempt != attempt.attempt_token
            ):
                raise RuntimeError("evaluation point child plan does not own the active attempt")
            if child_ordinal < len(claim_state.child_invocations):
                return claim_state.child_invocations[child_ordinal]
            parent_invocation = self._reserved_parent_invocation()
            return _context._WorkflowInvocation(
                frame=parent_invocation.frame,
                occurrence_path=(
                    *parent_invocation.occurrence_path,
                    frame.point_segment,
                    ("child", child_ordinal),
                ),
            )

    def _ensure_parent_invocation(self) -> _context._WorkflowInvocation:
        """Lazily commit the containing public Function occurrence.

        When evaluation points shipped elsewhere hold the occurrence as
        planned, the commit must give the same occurrence, since their keys
        derive from it.
        """
        with self._lock:
            if self._invocation is None:
                invocation = self._claim_own_invocation()
                if (
                    self._planned_invocation is not None
                    and invocation.occurrence_path != self._planned_invocation.occurrence_path
                ):
                    raise RuntimeError(
                        "another occurrence committed in the workflow while evaluation points "
                        "of this call ran in other processes, so their keys no longer match "
                        "the call's place in the workflow"
                    )
                self._invocation = invocation
            return self._invocation

    def _reserved_parent_invocation(self) -> _context._WorkflowInvocation:
        """The call's occurrence, committed if a draw committed it, or else as planned."""
        with self._lock:
            if self._invocation is not None:
                return self._invocation
            if self._planned_invocation is None:
                self._planned_invocation = self._planned_own_invocation()
            return self._planned_invocation

    def _planned_own_invocation(self) -> _context._WorkflowInvocation:
        """The occurrence ``_claim_own_invocation`` would commit, committing nothing."""
        attempt_context = self._evaluation_point_attempt
        if attempt_context is not None:
            point_child_invocation = attempt_context.planned_child_invocation()
            if self._frame is None or self._frame is attempt_context.workflow_frame:
                return point_child_invocation
            scope_path = _context._planned_descendant_path(
                self._frame,
                attempt_context.workflow_frame,
            )
            return _context._WorkflowInvocation(
                frame=self._frame,
                occurrence_path=(
                    *point_child_invocation.occurrence_path,
                    *scope_path,
                    (self.occurrence_kind, self._frame.ledger.peek()),
                ),
            )
        if self._frame is None:
            raise RuntimeError("automatic-key broker has no workflow frame")
        return _context._planned_stochastic_invocation_in_frame(self._frame, self.occurrence_kind)

    def _claim_own_invocation(self) -> _context._WorkflowInvocation:
        """Commit this broker while preserving a nested run inside an evaluation point."""
        if self._evaluation_point_attempt is not None:
            if self._frame is None or self._frame is self._evaluation_point_attempt.workflow_frame:
                return self._evaluation_point_attempt.claim_child_invocation()
            return self._evaluation_point_attempt.claim_scoped_child_invocation(
                self._frame,
                self.occurrence_kind,
            )
        if self._frame is None:
            raise RuntimeError("automatic-key broker has no workflow frame")
        if _context._capture_active_workflow_frame() is self._frame:
            return _context._commit_stochastic_invocation(self.occurrence_kind)
        return _context._commit_stochastic_invocation_in_frame(
            self._frame,
            self.occurrence_kind,
        )


_ACTIVE_AUTOMATIC_KEY_BROKER: ContextVar[_AutomaticKeyBroker | None] = ContextVar(
    "probpipe_active_automatic_key_broker",
    default=None,
)

_ACTIVE_EVALUATION_POINT_ATTEMPT: ContextVar[_EvaluationPointAttemptContext | None] = ContextVar(
    "probpipe_active_evaluation_point_attempt",
    default=None,
)


@dataclass(slots=True)
class _RemoteEvaluationPointParent:
    """Worker-local adapter for a parent-authorized child namespace."""

    envelope: EvaluationPointParentEnvelope
    attempt: EvaluationPointAttempt
    workflow_frame: _context._WorkflowFrame
    child_invocations: list[_context._WorkflowInvocation] = field(default_factory=list)
    effect_claims_by_identity: dict[tuple[Any, ...], StochasticEffectClaim] = field(
        default_factory=dict
    )
    successful_effects_by_identity: dict[tuple[Any, ...], StochasticEffectClaim] = field(
        default_factory=dict
    )
    _retry_effects_by_identity: dict[tuple[Any, ...], StochasticEffectClaim] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        for effect in self.envelope.retry_effects:
            identity = _effect_identity(effect)
            existing_effect = self._retry_effects_by_identity.get(identity)
            if existing_effect is None:
                self._retry_effects_by_identity[identity] = effect
                continue
            if existing_effect != effect:
                raise RuntimeError(
                    "remote retry envelope contains conflicting effect plans "
                    "for the same stochastic event identity"
                )
            raise RuntimeError(
                "remote retry envelope contains a duplicated stochastic event identity"
            )

    def _claim_evaluation_point_child(
        self,
        *,
        frame: EvaluationPointFrame,
        attempt: EvaluationPointAttempt,
        child_ordinal: int,
    ) -> _context._WorkflowInvocation:
        if frame != self.envelope.frame or attempt != self.attempt:
            raise RuntimeError("remote evaluation point child does not own its envelope")
        if child_ordinal != len(self.child_invocations):
            raise RuntimeError("remote evaluation point child claims must be made in order")
        invocation = _context._WorkflowInvocation(
            frame=self.workflow_frame,
            occurrence_path=(
                *self.envelope.parent_occurrence_path,
                frame.point_segment,
                ("child", child_ordinal),
            ),
        )
        self.child_invocations.append(invocation)
        return invocation

    def _planned_evaluation_point_child(
        self,
        *,
        frame: EvaluationPointFrame,
        attempt: EvaluationPointAttempt,
        child_ordinal: int,
    ) -> _context._WorkflowInvocation:
        """The child occurrence ``_claim_evaluation_point_child`` would claim, claiming nothing."""
        if frame != self.envelope.frame or attempt != self.attempt:
            raise RuntimeError("remote evaluation point child does not own its envelope")
        if child_ordinal < len(self.child_invocations):
            return self.child_invocations[child_ordinal]
        return _context._WorkflowInvocation(
            frame=self.workflow_frame,
            occurrence_path=(
                *self.envelope.parent_occurrence_path,
                frame.point_segment,
                ("child", child_ordinal),
            ),
        )

    def claim_evaluation_point_effect(
        self,
        effect: StochasticEffectClaim,
        *,
        frame: EvaluationPointFrame,
        attempt: EvaluationPointAttempt,
    ) -> None:
        """Validate and retain one remote retry claim before key derivation."""
        if frame != self.envelope.frame or attempt != self.attempt:
            raise RuntimeError("remote evaluation point effect does not own its envelope")
        identity = _effect_identity(effect)
        retry_effect = self._retry_effects_by_identity.get(identity)
        if retry_effect is not None and retry_effect != effect:
            raise RuntimeError("remote event identity changed effect plan during retry")
        existing_effect = self.effect_claims_by_identity.get(identity)
        if existing_effect is not None:
            if existing_effect != effect:
                raise RuntimeError("remote event identity changed effect plan during retry")
            raise RuntimeError("one evaluation point attempt duplicated a stochastic effect claim")
        self.effect_claims_by_identity[identity] = effect

    def accept_successful_evaluation_point_effects(
        self,
        effects: tuple[StochasticEffectClaim, ...],
        *,
        frame: EvaluationPointFrame,
        attempt: EvaluationPointAttempt,
    ) -> None:
        """Mark the worker effects that crossed a successful scope boundary."""
        if frame != self.envelope.frame or attempt != self.attempt:
            raise RuntimeError("successful remote effects do not own their envelope")
        if not isinstance(effects, tuple) or any(
            not isinstance(effect, StochasticEffectClaim) for effect in effects
        ):
            raise TypeError("successful remote effects must be an effect tuple")
        identities = set()
        for effect in effects:
            identity = _effect_identity(effect)
            if identity in identities:
                raise RuntimeError("successful remote effects contain a duplicate identity")
            identities.add(identity)
            if self.effect_claims_by_identity.get(identity) != effect:
                raise RuntimeError("a successful remote effect was not claimed by its attempt")
            self.successful_effects_by_identity[identity] = effect

    def report(self) -> EvaluationPointClaimReport:
        """Return the serializable claim count for parent reconciliation."""
        return EvaluationPointClaimReport(
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
        broker._finalize(successful=False)
        raise
    else:
        broker._finalize(successful=True)
    finally:
        _ACTIVE_AUTOMATIC_KEY_BROKER.reset(token)


@contextmanager
def _function_stochastic_scope(
    *,
    occurrence_path: tuple[Any, ...] | None = None,
) -> Generator[_AutomaticKeyBroker, None, None]:
    """Install a lazy broker for one public Function invocation."""
    _context._assert_workflow_admission()
    frame = _context._capture_active_workflow_frame()
    from . import _replay

    broker = _AutomaticKeyBroker(
        "invocation",
        _frame=frame,
        _evaluation_point_attempt=_ACTIVE_EVALUATION_POINT_ATTEMPT.get(),
        _replay_state=_replay._capture_active_replay_state(),
    )
    if occurrence_path is not None:
        if frame is None or broker._evaluation_point_attempt is not None:
            raise RuntimeError("a replay occurrence requires a standalone workflow frame")
        broker._invocation = _context._WorkflowInvocation(
            frame=frame,
            occurrence_path=occurrence_path,
        )
    with _installed_stochastic_broker_scope(broker):
        yield broker


@contextmanager
def _operation_stochastic_scope() -> Generator[_AutomaticKeyBroker, None, None]:
    """Reuse an active broker or install one operation broker."""
    _context._assert_workflow_admission()
    active_broker = _ACTIVE_AUTOMATIC_KEY_BROKER.get()
    if active_broker is not None:
        yield active_broker
        return

    evaluation_point_attempt = _ACTIVE_EVALUATION_POINT_ATTEMPT.get()
    if evaluation_point_attempt is not None:
        broker = _AutomaticKeyBroker(
            "operation",
            _frame=_context._capture_active_workflow_frame(),
            _evaluation_point_attempt=evaluation_point_attempt,
        )
        with _installed_stochastic_broker_scope(broker):
            yield broker
        return

    with _context._ephemeral_workflow_run():
        broker = _AutomaticKeyBroker(
            "operation",
            _frame=_context._capture_active_workflow_frame(),
        )
        with _installed_stochastic_broker_scope(broker):
            yield broker


def _capture_active_broker() -> _AutomaticKeyBroker | None:
    """Capture the admitted parent broker for evaluation point transport."""
    _context._assert_workflow_admission()
    return _ACTIVE_AUTOMATIC_KEY_BROKER.get()


def _record_active_execution_contract(
    contract: WorkflowRngExecutionContract,
) -> None:
    """Attach an actual route contract to the current public invocation."""
    from . import _replay

    _replay._validate_active_execution_contract(contract)
    broker = _ACTIVE_AUTOMATIC_KEY_BROKER.get()
    if broker is not None:
        broker.record_execution_contract(contract)


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
        root_words=_context._resolve_root_words(invocation.frame),
        occurrence_path=invocation.occurrence_path,
        rng_origin=_context._describe_rng_origin(invocation.frame),
        effects=effects,
        execution_contracts=tuple(broker._execution_contracts),
        callable_anchor=broker._callable_anchor,
    )


@contextmanager
def _remote_evaluation_point_stochastic_scope(
    envelope: EvaluationPointParentEnvelope,
    attempt: EvaluationPointAttempt,
) -> Generator[_RemoteEvaluationPointParent, None, None]:
    """Install parent-authorized RNG derivation inside a remote worker."""
    if envelope.attempt != attempt or envelope.frame.token != attempt.evaluation_point_token:
        raise RuntimeError("remote evaluation point attempt does not own its parent envelope")
    frame = _context._capture_active_workflow_frame()
    if frame is None:
        raise RuntimeError("remote evaluation point randomness requires a transported frame")
    _context._assert_transported_frame_consistency(frame, envelope.root_words)
    remote_parent = _RemoteEvaluationPointParent(
        envelope=envelope,
        attempt=attempt,
        workflow_frame=frame,
    )
    attempt_context = _EvaluationPointAttemptContext(
        parent_broker=remote_parent,
        frame=envelope.frame,
        attempt=attempt,
        workflow_frame=frame,
    )
    from . import _replay

    attempt_token = _ACTIVE_EVALUATION_POINT_ATTEMPT.set(attempt_context)
    broker_token = _ACTIVE_AUTOMATIC_KEY_BROKER.set(None)
    try:
        with _replay._remote_replay_claim_scope(
            envelope.replay_expected_effects,
            attempt,
        ):
            yield remote_parent
        attempt_context.publish_successful_effects()
    finally:
        _ACTIVE_AUTOMATIC_KEY_BROKER.reset(broker_token)
        _ACTIVE_EVALUATION_POINT_ATTEMPT.reset(attempt_token)


@contextmanager
def _evaluation_point_stochastic_scope(
    parent_broker: _AutomaticKeyBroker,
    frame: EvaluationPointFrame,
    *,
    attempt: EvaluationPointAttempt | None = None,
) -> Generator[EvaluationPointAttempt, None, None]:
    """Install one retry attempt, with no child broker yet."""
    if attempt is None:
        attempt = EvaluationPointAttempt.create(frame.token)
    parent_broker.validate_evaluation_point_attempt_preflight(attempt, frame)
    workflow_frame = _context._capture_active_workflow_frame()
    if workflow_frame is None:
        raise RuntimeError("evaluation point randomness requires an active workflow frame")
    with workflow_frame.state.lock:
        point_segment = workflow_frame.state.point_segment
    if point_segment != frame.point_segment:
        raise RuntimeError(
            "evaluation point randomness requires the matching active evaluation point workflow frame"
        )
    attempt_context = _EvaluationPointAttemptContext(
        parent_broker=parent_broker,
        frame=frame,
        attempt=attempt,
        workflow_frame=workflow_frame,
    )
    parent_broker.begin_evaluation_point_attempt(attempt, frame)
    attempt_token: Token[_EvaluationPointAttemptContext | None] | None = None
    broker_token: Token[_AutomaticKeyBroker | None] | None = None
    try:
        attempt_token = _ACTIVE_EVALUATION_POINT_ATTEMPT.set(attempt_context)
        broker_token = _ACTIVE_AUTOMATIC_KEY_BROKER.set(None)
        yield attempt
        attempt_context.publish_successful_effects()
    finally:
        try:
            if broker_token is not None:
                _ACTIVE_AUTOMATIC_KEY_BROKER.reset(broker_token)
            if attempt_token is not None:
                _ACTIVE_EVALUATION_POINT_ATTEMPT.reset(attempt_token)
        finally:
            parent_broker.finish_evaluation_point_attempt(attempt)


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
    with _operation_stochastic_scope() as operation_broker:
        return operation_broker.key_for(plan)


# -- Validation of the payloads ----------------------------------------------


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


def _validate_evaluation_point_token_value(value: object) -> None:
    """Validate the serialized bytes of one evaluation point token."""
    if not isinstance(value, bytes) or len(value) != 16:
        raise TypeError("evaluation point tokens must contain exactly 16 bytes")


def _validate_stochastic_effect_fields(
    *,
    operation_kind: object,
    execution_mode: object,
    sample_shape: object,
    sampling_abi: object,
    provider_abi: object,
    record_path: object,
    descendant_descriptor: object,
) -> None:
    """Validate canonical fields shared by local plans and transport claims."""
    string_fields = {
        "operation_kind": operation_kind,
        "execution_mode": execution_mode,
        "sampling_abi": sampling_abi,
        "provider_abi": provider_abi,
    }
    for field_name, value in string_fields.items():
        if not isinstance(value, str):
            raise TypeError(f"stochastic effect {field_name} must be a string")
        if not value:
            raise ValueError(f"stochastic effect {field_name} must be non-empty")

    if sample_shape is not None:
        if not isinstance(sample_shape, tuple):
            raise TypeError("stochastic effect sample shapes must be tuples or None")
        if any(isinstance(size, bool) or not isinstance(size, int) for size in sample_shape):
            raise TypeError(
                "stochastic effect sample shape dimensions must be non-boolean integers"
            )
        if any(size < 0 for size in sample_shape):
            raise ValueError("stochastic effect sample shape dimensions must be non-negative")

    if not isinstance(record_path, tuple):
        raise TypeError("stochastic effect record paths must be tuples")
    if any(not isinstance(field, str) for field in record_path):
        raise TypeError("stochastic effect record path fields must be strings")

    if descendant_descriptor is not None:
        if not isinstance(descendant_descriptor, tuple):
            raise TypeError("stochastic effect descendant descriptors must be tuples or None")
        if not _is_canonical_descriptor_value(descendant_descriptor):
            raise TypeError(
                "stochastic effect descendant descriptors must contain only canonical tuple values"
            )


def _is_canonical_descriptor_value(value: object) -> bool:
    """Return whether a descriptor value follows the immutable encoder grammar."""
    if isinstance(value, tuple):
        return all(_is_canonical_descriptor_value(entry) for entry in value)
    return value is None or isinstance(value, (bool, int, str, bytes))


def _validate_effect_claim_fields(
    *,
    occurrence_path: object,
    occurrence_kind: object,
    stochastic_source_id: object,
    logical_unit_id: object,
    operation_kind: object,
    execution_mode: object,
    sample_shape: object,
    sampling_abi: object,
    provider_abi: object,
    record_path: object,
    descendant_descriptor: object,
) -> None:
    """Validate one complete effect claim at construction or admission."""
    if not isinstance(occurrence_path, tuple):
        raise TypeError("effect occurrence paths must be tuples")
    if not isinstance(stochastic_source_id, tuple) or not isinstance(
        logical_unit_id,
        tuple,
    ):
        raise TypeError("effect source and unit identities must be tuples")
    _validate_random_event_value(occurrence_path)
    _validate_random_event_value(stochastic_source_id)
    _validate_random_event_value(logical_unit_id)
    if not isinstance(occurrence_kind, str):
        raise TypeError("effect occurrence_kind must be a string")
    if occurrence_kind not in {"invocation", "operation"}:
        raise ValueError("effect occurrence_kind must be 'invocation' or 'operation'")
    _validate_stochastic_effect_fields(
        operation_kind=operation_kind,
        execution_mode=execution_mode,
        sample_shape=sample_shape,
        sampling_abi=sampling_abi,
        provider_abi=provider_abi,
        record_path=record_path,
        descendant_descriptor=descendant_descriptor,
    )


def _validate_evaluation_point_frame_fields(
    *,
    point_segment: object,
    token: object,
    derivation_abi: object,
) -> None:
    """Validate the complete authority fields of one evaluation point frame."""
    if not isinstance(point_segment, tuple):
        raise TypeError("evaluation point segments must be tuples")
    if not _is_evaluation_point_segment(point_segment):
        raise ValueError("evaluation point segments must use a canonical evaluation point segment")
    if not isinstance(token, EvaluationPointToken):
        raise TypeError("evaluation point frames require an evaluation point token")
    _validate_evaluation_point_token_value(token.value)
    if derivation_abi != _EVALUATION_POINT_ABI:
        raise ValueError(f"unsupported evaluation point ABI: {derivation_abi!r}")


def _validate_evaluation_point_attempt_fields(
    *,
    evaluation_point_token: object,
    attempt_token: object,
) -> None:
    """Validate the complete authority fields of one evaluation point attempt."""
    if not isinstance(evaluation_point_token, EvaluationPointToken):
        raise TypeError("evaluation point attempts require an evaluation point token")
    _validate_evaluation_point_token_value(evaluation_point_token.value)
    if not isinstance(attempt_token, bytes) or len(attempt_token) != 16:
        raise TypeError("evaluation point attempt tokens must contain exactly 16 bytes")


def _validate_issued_evaluation_point_fields(*, index: object, frame: object) -> None:
    """Validate the index and frame of one issued evaluation point."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise TypeError("evaluation point indexes must be non-negative integers")
    if not isinstance(frame, EvaluationPointFrame):
        raise TypeError("evaluation points require a valid evaluation point frame")
    _validate_evaluation_point_frame_fields(
        point_segment=frame.point_segment,
        token=frame.token,
        derivation_abi=frame.derivation_abi,
    )


def _validate_effect_claim_instance(
    effect: object,
) -> StochasticEffectClaim:
    """Rerun complete effect validation on an existing immutable instance."""
    if not isinstance(effect, StochasticEffectClaim):
        raise TypeError("evaluation point reports must contain StochasticEffectClaim values")
    _validate_effect_claim_fields(
        occurrence_path=effect.occurrence_path,
        occurrence_kind=effect.occurrence_kind,
        stochastic_source_id=effect.stochastic_source_id,
        logical_unit_id=effect.logical_unit_id,
        operation_kind=effect.operation_kind,
        execution_mode=effect.execution_mode,
        sample_shape=effect.sample_shape,
        sampling_abi=effect.sampling_abi,
        provider_abi=effect.provider_abi,
        record_path=effect.record_path,
        descendant_descriptor=effect.descendant_descriptor,
    )
    return effect


def _validate_evaluation_point_parent_envelope_fields(
    *,
    root_words: object,
    parent_occurrence_path: object,
    frame: object,
    attempt: object,
    replay_expected_effects: object,
    retry_effects: object,
) -> None:
    """Validate parent authority and all nested transported effect snapshots."""
    if not isinstance(frame, EvaluationPointFrame) or not isinstance(
        attempt,
        EvaluationPointAttempt,
    ):
        raise TypeError("evaluation point parent authority requires a frame and attempt")
    _validate_evaluation_point_frame_fields(
        point_segment=frame.point_segment,
        token=frame.token,
        derivation_abi=frame.derivation_abi,
    )
    _validate_evaluation_point_attempt_fields(
        evaluation_point_token=attempt.evaluation_point_token,
        attempt_token=attempt.attempt_token,
    )
    if (
        not isinstance(root_words, tuple)
        or len(root_words) != 2
        or any(
            isinstance(word, bool) or not isinstance(word, int) or not 0 <= word <= 0xFFFFFFFF
            for word in root_words
        )
    ):
        raise TypeError("evaluation point parent root words must be two uint32 integers")
    if not isinstance(parent_occurrence_path, tuple):
        raise TypeError("evaluation point parent occurrence paths must be tuples")
    _validate_random_event_value(parent_occurrence_path)
    if attempt.evaluation_point_token != frame.token:
        raise ValueError("evaluation point parent attempt must own its frame")
    if replay_expected_effects is not None and (
        not isinstance(replay_expected_effects, tuple)
        or any(not isinstance(effect, StochasticEffectClaim) for effect in replay_expected_effects)
    ):
        raise TypeError("evaluation point replay expectations must be effect tuples or None")
    if not isinstance(retry_effects, tuple) or any(
        not isinstance(effect, StochasticEffectClaim) for effect in retry_effects
    ):
        raise TypeError("evaluation point retry effects must be an effect tuple")
    if replay_expected_effects is not None:
        for effect in replay_expected_effects:
            _validate_effect_claim_instance(effect)
        _unique_effects(
            replay_expected_effects,
            field_name="evaluation point replay expectations",
        )
    for effect in retry_effects:
        _validate_effect_claim_instance(effect)
    _unique_effects(
        retry_effects,
        field_name="evaluation point retry effects",
    )


def _validate_evaluation_point_claim_report_fields(
    *,
    frame: object,
    attempt: object,
    child_count: object,
    effects: object,
    successful_effects: object,
) -> None:
    """Validate a complete remote claim report without mutating state."""
    if not isinstance(frame, EvaluationPointFrame) or not isinstance(
        attempt,
        EvaluationPointAttempt,
    ):
        raise TypeError("evaluation point claim reports require a frame and attempt")
    _validate_evaluation_point_frame_fields(
        point_segment=frame.point_segment,
        token=frame.token,
        derivation_abi=frame.derivation_abi,
    )
    _validate_evaluation_point_attempt_fields(
        evaluation_point_token=attempt.evaluation_point_token,
        attempt_token=attempt.attempt_token,
    )
    if attempt.evaluation_point_token != frame.token:
        raise ValueError("evaluation point report attempt must own its frame")
    if isinstance(child_count, bool) or not isinstance(child_count, int) or child_count < 0:
        raise TypeError("evaluation point child counts must be non-negative integers")
    if not isinstance(effects, tuple) or any(
        not isinstance(effect, StochasticEffectClaim) for effect in effects
    ):
        raise TypeError("evaluation point effect reports must contain a tuple of effects")
    if not isinstance(successful_effects, tuple) or any(
        not isinstance(effect, StochasticEffectClaim) for effect in successful_effects
    ):
        raise TypeError("successful evaluation point effects must contain a tuple of effects")
    for effect in (*effects, *successful_effects):
        _validate_effect_claim_instance(effect)
    effects_by_identity = _unique_effects(
        effects,
        field_name="evaluation point effect report",
    )
    successful_by_identity = _unique_effects(
        successful_effects,
        field_name="successful evaluation point effect report",
    )
    if any(
        effects_by_identity.get(identity) != effect
        for identity, effect in successful_by_identity.items()
    ):
        raise ValueError("successful evaluation point effects must be claimed by the same attempt")


def _validated_effect_claim_snapshot(
    effect: object,
) -> StochasticEffectClaim:
    """Reconstruct one immutable effect while rerunning its full validation."""
    if not isinstance(effect, StochasticEffectClaim):
        raise TypeError("evaluation point reports must contain StochasticEffectClaim values")
    return StochasticEffectClaim(
        occurrence_path=effect.occurrence_path,
        occurrence_kind=effect.occurrence_kind,
        stochastic_source_id=effect.stochastic_source_id,
        logical_unit_id=effect.logical_unit_id,
        operation_kind=effect.operation_kind,
        execution_mode=effect.execution_mode,
        sample_shape=effect.sample_shape,
        sampling_abi=effect.sampling_abi,
        provider_abi=effect.provider_abi,
        record_path=effect.record_path,
        descendant_descriptor=effect.descendant_descriptor,
    )


def _validated_evaluation_point_claim_report_snapshot(
    report: object,
) -> EvaluationPointClaimReport:
    """Deeply revalidate and freeze an untrusted transported report."""
    if not isinstance(report, EvaluationPointClaimReport):
        raise TypeError("remote evaluation point reports must be EvaluationPointClaimReport values")
    if not isinstance(report.frame, EvaluationPointFrame) or not isinstance(
        report.attempt,
        EvaluationPointAttempt,
    ):
        raise TypeError("evaluation point claim reports require a frame and attempt")
    if not isinstance(report.effects, tuple) or any(
        not isinstance(effect, StochasticEffectClaim) for effect in report.effects
    ):
        raise TypeError("evaluation point effect reports must contain a tuple of effects")
    if not isinstance(report.successful_effects, tuple) or any(
        not isinstance(effect, StochasticEffectClaim) for effect in report.successful_effects
    ):
        raise TypeError("successful evaluation point effects must contain a tuple of effects")
    return EvaluationPointClaimReport(
        frame=_validated_evaluation_point_frame_snapshot(report.frame),
        attempt=_validated_evaluation_point_attempt_snapshot(report.attempt),
        child_count=report.child_count,
        effects=tuple(_validated_effect_claim_snapshot(effect) for effect in report.effects),
        successful_effects=tuple(
            _validated_effect_claim_snapshot(effect) for effect in report.successful_effects
        ),
    )


def _validated_evaluation_point_frame_snapshot(frame: object) -> EvaluationPointFrame:
    """Deeply reconstruct one transported evaluation point frame."""
    if not isinstance(frame, EvaluationPointFrame):
        raise TypeError("transported evaluation point frames must be EvaluationPointFrame values")
    if not isinstance(frame.token, EvaluationPointToken):
        raise TypeError("evaluation point frames require an evaluation point token")
    return EvaluationPointFrame(
        point_segment=frame.point_segment,
        token=EvaluationPointToken(frame.token.value),
        derivation_abi=frame.derivation_abi,
    )


def _validated_evaluation_point_attempt_snapshot(attempt: object) -> EvaluationPointAttempt:
    """Deeply reconstruct one transported evaluation point attempt."""
    if not isinstance(attempt, EvaluationPointAttempt):
        raise TypeError(
            "transported evaluation point attempts must be EvaluationPointAttempt values"
        )
    if not isinstance(attempt.evaluation_point_token, EvaluationPointToken):
        raise TypeError("evaluation point attempts require an evaluation point token")
    return EvaluationPointAttempt(
        evaluation_point_token=EvaluationPointToken(attempt.evaluation_point_token.value),
        attempt_token=attempt.attempt_token,
    )


def _is_evaluation_point_segment(segment: tuple[object, ...]) -> bool:
    """Return whether a segment follows the closed grammar of an evaluation point's segment."""
    if len(segment) < 3 or segment[:2] != (_EVALUATION_POINT_TAG, _EVALUATION_POINT_ABI):
        return False
    layout = segment[2]
    if layout == "point":
        return len(segment) == 4 and _is_nonnegative_int(segment[3]) and segment[3] == 0
    if layout == "sweep-cell":
        return len(segment) >= 4 and all(
            _is_nonnegative_int(coordinate) for coordinate in segment[3:]
        )
    if layout == "lifted-evaluation":
        return (
            len(segment) == 5
            and _is_logical_unit_id(segment[3])
            and _is_nonnegative_int(segment[4])
        )
    return False


def _is_logical_unit_id(value: object) -> bool:
    if not isinstance(value, tuple) or not value:
        return False
    if value[0] == "singleton":
        return len(value) == 1
    return (
        value[0] == "cell"
        and len(value) >= 2
        and all(_is_nonnegative_int(coordinate) for coordinate in value[1:])
    )


def _is_nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _unique_effects(
    effects: tuple[StochasticEffectClaim, ...],
    *,
    field_name: str,
) -> dict[tuple[object, ...], StochasticEffectClaim]:
    """Index an effect tuple while rejecting duplicate structural identities."""
    effects_by_identity = {}
    for effect in effects:
        identity = (
            effect.occurrence_path,
            effect.stochastic_source_id,
            effect.logical_unit_id,
        )
        if identity in effects_by_identity:
            raise ValueError(f"{field_name} contains a duplicate effect identity")
        effects_by_identity[identity] = effect
    return effects_by_identity
