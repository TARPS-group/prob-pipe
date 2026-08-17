"""Immutable payloads for ProbPipe-managed workflow execution."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._workflow_rng import (
    _RandomEventPath,
    _validate_random_event_value,
)
from .config import ProvenanceMode

_MANAGED_WORK_ITEM_ABI = "probpipe.managed_work_item/v1"

type _CanonicalDescriptorValue = (
    None | bool | int | str | bytes | tuple[_CanonicalDescriptorValue, ...]
)
type _CanonicalDescriptor = tuple[_CanonicalDescriptorValue, ...]


def _validate_managed_work_item_token_value(value: object) -> None:
    """Validate the serialized bytes of one managed work-item token."""
    if not isinstance(value, bytes) or len(value) != 16:
        raise TypeError("managed work-item tokens must contain exactly 16 bytes")


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
        return all(_is_canonical_descriptor_value(item) for item in value)
    return value is None or isinstance(value, (bool, int, str, bytes))


def _validate_managed_effect_claim_fields(
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
        raise TypeError("managed effect occurrence paths must be tuples")
    if not isinstance(stochastic_source_id, tuple) or not isinstance(
        logical_unit_id,
        tuple,
    ):
        raise TypeError("managed effect source and unit identities must be tuples")
    _validate_random_event_value(occurrence_path)
    _validate_random_event_value(stochastic_source_id)
    _validate_random_event_value(logical_unit_id)
    if not isinstance(occurrence_kind, str):
        raise TypeError("managed effect occurrence_kind must be a string")
    if occurrence_kind not in {"invocation", "operation"}:
        raise ValueError("managed effect occurrence_kind must be 'invocation' or 'operation'")
    _validate_stochastic_effect_fields(
        operation_kind=operation_kind,
        execution_mode=execution_mode,
        sample_shape=sample_shape,
        sampling_abi=sampling_abi,
        provider_abi=provider_abi,
        record_path=record_path,
        descendant_descriptor=descendant_descriptor,
    )


@dataclass(frozen=True, slots=True)
class ManagedWorkItemToken:
    """Opaque, serializable ownership token for one managed work item."""

    value: bytes

    def __post_init__(self) -> None:
        _validate_managed_work_item_token_value(self.value)

    @classmethod
    def create(cls) -> ManagedWorkItemToken:
        """Create one process-independent operational token."""
        return cls(uuid.uuid4().bytes)


def _validate_managed_unit_frame_fields(
    *,
    unit_segment: object,
    token: object,
    derivation_abi: object,
) -> None:
    """Validate the complete authority fields of one managed unit frame."""
    if not isinstance(unit_segment, tuple):
        raise TypeError("managed unit segments must be tuples")
    if not _is_managed_unit_segment(unit_segment):
        raise ValueError("managed unit segments must use a canonical managed unit segment")
    if not isinstance(token, ManagedWorkItemToken):
        raise TypeError("managed unit frames require a managed work-item token")
    _validate_managed_work_item_token_value(token.value)
    if derivation_abi != _MANAGED_WORK_ITEM_ABI:
        raise ValueError(f"unsupported managed work-item ABI: {derivation_abi!r}")


def _validate_managed_attempt_fields(
    *,
    work_item_token: object,
    attempt_token: object,
) -> None:
    """Validate the complete authority fields of one managed attempt."""
    if not isinstance(work_item_token, ManagedWorkItemToken):
        raise TypeError("managed attempts require a managed work-item token")
    _validate_managed_work_item_token_value(work_item_token.value)
    if not isinstance(attempt_token, bytes) or len(attempt_token) != 16:
        raise TypeError("managed attempt tokens must contain exactly 16 bytes")


@dataclass(frozen=True, slots=True)
class ManagedUnitFrame:
    """Canonical logical-unit binding transported with one work item."""

    unit_segment: _RandomEventPath
    token: ManagedWorkItemToken
    derivation_abi: str = _MANAGED_WORK_ITEM_ABI

    def __post_init__(self) -> None:
        _validate_managed_unit_frame_fields(
            unit_segment=self.unit_segment,
            token=self.token,
            derivation_abi=self.derivation_abi,
        )


@dataclass(frozen=True, slots=True)
class ManagedWorkItem:
    """One immutable, canonically indexed workflow evaluation request."""

    index: int
    values: tuple[tuple[str, Any], ...]
    frame: ManagedUnitFrame

    def __post_init__(self) -> None:
        _validate_managed_work_item_fields(
            index=self.index,
            values=self.values,
            frame=self.frame,
        )

    def call_values(self) -> dict[str, Any]:
        """Materialize the keyword mapping used for execution."""
        return dict(self.values)


def _validate_managed_work_item_fields(
    *,
    index: object,
    values: object,
    frame: object,
) -> None:
    """Validate one complete managed call boundary without materializing kwargs."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise TypeError("managed work-item indexes must be non-negative integers")
    if not isinstance(values, tuple) or any(
        not isinstance(entry, tuple) or len(entry) != 2 or not isinstance(entry[0], str)
        for entry in values
    ):
        raise TypeError("managed work-item values must be name/value tuples")
    names = tuple(entry[0] for entry in values)
    if len(set(names)) != len(names):
        raise ValueError("managed work-item values contain duplicate parameter names")
    if not isinstance(frame, ManagedUnitFrame):
        raise TypeError("managed work items require a valid managed unit frame")
    _validate_managed_unit_frame_fields(
        unit_segment=frame.unit_segment,
        token=frame.token,
        derivation_abi=frame.derivation_abi,
    )


@dataclass(frozen=True, slots=True)
class ManagedAttemptState:
    """Operational identity for one execution attempt of a work item."""

    work_item_token: ManagedWorkItemToken
    attempt_token: bytes

    def __post_init__(self) -> None:
        _validate_managed_attempt_fields(
            work_item_token=self.work_item_token,
            attempt_token=self.attempt_token,
        )

    @classmethod
    def create(cls, work_item_token: ManagedWorkItemToken) -> ManagedAttemptState:
        """Create a fresh attempt identity for an existing work item."""
        return cls(work_item_token=work_item_token, attempt_token=uuid.uuid4().bytes)


@dataclass(frozen=True, slots=True)
class ManagedParentEnvelope:
    """Serializable root and occurrence authority for one remote managed unit."""

    root_words: tuple[int, int]
    parent_occurrence_path: _RandomEventPath
    frame: ManagedUnitFrame
    attempt: ManagedAttemptState
    replay_expected_effects: tuple[ManagedEffectClaim, ...] | None = None
    retry_effects: tuple[ManagedEffectClaim, ...] = ()

    def __post_init__(self) -> None:
        _validate_managed_parent_envelope_fields(
            root_words=self.root_words,
            parent_occurrence_path=self.parent_occurrence_path,
            frame=self.frame,
            attempt=self.attempt,
            replay_expected_effects=self.replay_expected_effects,
            retry_effects=self.retry_effects,
        )


@dataclass(frozen=True, slots=True)
class ManagedEffectClaim:
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
        _validate_managed_effect_claim_fields(
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


def _validate_managed_effect_claim_instance(
    effect: object,
) -> ManagedEffectClaim:
    """Rerun complete effect validation on an existing immutable instance."""
    if not isinstance(effect, ManagedEffectClaim):
        raise TypeError("managed reports must contain ManagedEffectClaim values")
    _validate_managed_effect_claim_fields(
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


def _validate_managed_parent_envelope_fields(
    *,
    root_words: object,
    parent_occurrence_path: object,
    frame: object,
    attempt: object,
    replay_expected_effects: object,
    retry_effects: object,
) -> None:
    """Validate parent authority and all nested transported effect snapshots."""
    if not isinstance(frame, ManagedUnitFrame) or not isinstance(
        attempt,
        ManagedAttemptState,
    ):
        raise TypeError("managed parent authority requires a frame and attempt")
    _validate_managed_unit_frame_fields(
        unit_segment=frame.unit_segment,
        token=frame.token,
        derivation_abi=frame.derivation_abi,
    )
    _validate_managed_attempt_fields(
        work_item_token=attempt.work_item_token,
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
        raise TypeError("managed parent root words must be two uint32 integers")
    if not isinstance(parent_occurrence_path, tuple):
        raise TypeError("managed parent occurrence paths must be tuples")
    _validate_random_event_value(parent_occurrence_path)
    if attempt.work_item_token != frame.token:
        raise ValueError("managed parent attempt must own its frame")
    if replay_expected_effects is not None and (
        not isinstance(replay_expected_effects, tuple)
        or any(not isinstance(effect, ManagedEffectClaim) for effect in replay_expected_effects)
    ):
        raise TypeError("managed replay expectations must be effect tuples or None")
    if not isinstance(retry_effects, tuple) or any(
        not isinstance(effect, ManagedEffectClaim) for effect in retry_effects
    ):
        raise TypeError("managed retry effects must be an effect tuple")
    if replay_expected_effects is not None:
        for effect in replay_expected_effects:
            _validate_managed_effect_claim_instance(effect)
        _unique_effects(
            replay_expected_effects,
            field_name="managed replay expectations",
        )
    for effect in retry_effects:
        _validate_managed_effect_claim_instance(effect)
    _unique_effects(
        retry_effects,
        field_name="managed retry effects",
    )


def _validate_managed_claim_report_fields(
    *,
    frame: object,
    attempt: object,
    child_count: object,
    effects: object,
    successful_effects: object,
) -> None:
    """Validate a complete remote claim report without mutating state."""
    if not isinstance(frame, ManagedUnitFrame) or not isinstance(
        attempt,
        ManagedAttemptState,
    ):
        raise TypeError("managed claim reports require a frame and attempt")
    _validate_managed_unit_frame_fields(
        unit_segment=frame.unit_segment,
        token=frame.token,
        derivation_abi=frame.derivation_abi,
    )
    _validate_managed_attempt_fields(
        work_item_token=attempt.work_item_token,
        attempt_token=attempt.attempt_token,
    )
    if attempt.work_item_token != frame.token:
        raise ValueError("managed report attempt must own its frame")
    if isinstance(child_count, bool) or not isinstance(child_count, int) or child_count < 0:
        raise TypeError("managed child counts must be non-negative integers")
    if not isinstance(effects, tuple) or any(
        not isinstance(effect, ManagedEffectClaim) for effect in effects
    ):
        raise TypeError("managed effect reports must contain a tuple of effects")
    if not isinstance(successful_effects, tuple) or any(
        not isinstance(effect, ManagedEffectClaim) for effect in successful_effects
    ):
        raise TypeError("managed successful effects must contain a tuple of effects")
    for effect in (*effects, *successful_effects):
        _validate_managed_effect_claim_instance(effect)
    effects_by_identity = _unique_effects(
        effects,
        field_name="managed effect report",
    )
    successful_by_identity = _unique_effects(
        successful_effects,
        field_name="managed successful effect report",
    )
    if any(
        effects_by_identity.get(identity) != effect
        for identity, effect in successful_by_identity.items()
    ):
        raise ValueError("managed successful effects must be claimed by the same attempt")


@dataclass(frozen=True, slots=True)
class ManagedClaimReport:
    """Serializable claim summary returned by one remote attempt."""

    frame: ManagedUnitFrame
    attempt: ManagedAttemptState
    child_count: int
    effects: tuple[ManagedEffectClaim, ...] = ()
    successful_effects: tuple[ManagedEffectClaim, ...] = ()

    def __post_init__(self) -> None:
        _validate_managed_claim_report_fields(
            frame=self.frame,
            attempt=self.attempt,
            child_count=self.child_count,
            effects=self.effects,
            successful_effects=self.successful_effects,
        )


def _validated_managed_effect_claim_snapshot(
    effect: object,
) -> ManagedEffectClaim:
    """Reconstruct one immutable effect while rerunning its full validation."""
    effect = _validate_managed_effect_claim_instance(effect)
    return ManagedEffectClaim(
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


def _validated_managed_claim_report_snapshot(
    report: object,
) -> ManagedClaimReport:
    """Deeply revalidate and freeze an untrusted transported report."""
    if not isinstance(report, ManagedClaimReport):
        raise TypeError("remote managed reports must be ManagedClaimReport values")
    _validate_managed_claim_report_fields(
        frame=report.frame,
        attempt=report.attempt,
        child_count=report.child_count,
        effects=report.effects,
        successful_effects=report.successful_effects,
    )

    frame_token = ManagedWorkItemToken(report.frame.token.value)
    frame = ManagedUnitFrame(
        unit_segment=report.frame.unit_segment,
        token=frame_token,
        derivation_abi=report.frame.derivation_abi,
    )
    attempt = ManagedAttemptState(
        work_item_token=ManagedWorkItemToken(report.attempt.work_item_token.value),
        attempt_token=report.attempt.attempt_token,
    )
    return ManagedClaimReport(
        frame=frame,
        attempt=attempt,
        child_count=report.child_count,
        effects=tuple(
            _validated_managed_effect_claim_snapshot(effect) for effect in report.effects
        ),
        successful_effects=tuple(
            _validated_managed_effect_claim_snapshot(effect) for effect in report.successful_effects
        ),
    )


@dataclass(frozen=True, slots=True)
class ManagedPrefectPayload:
    """Serializable Prefect task input for an initial or coordinated attempt."""

    item: ManagedWorkItem
    attempt: ManagedAttemptState
    provenance_mode: ProvenanceMode
    parent: ManagedParentEnvelope | None = None

    def __post_init__(self) -> None:
        _validate_managed_prefect_payload_fields(
            item=self.item,
            attempt=self.attempt,
            provenance_mode=self.provenance_mode,
            parent=self.parent,
        )


@dataclass(frozen=True, slots=True)
class ManagedExecutionOutcome:
    """Serializable Prefect result with operational claim information."""

    index: int
    value: Any = None
    error: Exception | None = None
    coordination_required: bool = False
    report: ManagedClaimReport | None = None

    def __post_init__(self) -> None:
        _validate_managed_execution_outcome_fields(
            index=self.index,
            value=self.value,
            error=self.error,
            coordination_required=self.coordination_required,
            report=self.report,
        )


def _validate_managed_prefect_payload_fields(
    *,
    item: object,
    attempt: object,
    provenance_mode: object,
    parent: object,
) -> None:
    """Validate a complete worker payload, including its nested authority."""
    if not isinstance(item, ManagedWorkItem) or not isinstance(
        attempt,
        ManagedAttemptState,
    ):
        raise TypeError("managed Prefect payloads require a work item and attempt")
    _validate_managed_work_item_fields(
        index=item.index,
        values=item.values,
        frame=item.frame,
    )
    _validate_managed_attempt_fields(
        work_item_token=attempt.work_item_token,
        attempt_token=attempt.attempt_token,
    )
    if not isinstance(provenance_mode, ProvenanceMode):
        raise TypeError("managed Prefect payloads require a provenance mode")
    if parent is not None and not isinstance(parent, ManagedParentEnvelope):
        raise TypeError("managed Prefect parent authority must be an envelope or None")
    if attempt.work_item_token != item.frame.token:
        raise ValueError("managed payload attempt must own its work item")
    if parent is not None:
        _validate_managed_parent_envelope_fields(
            root_words=parent.root_words,
            parent_occurrence_path=parent.parent_occurrence_path,
            frame=parent.frame,
            attempt=parent.attempt,
            replay_expected_effects=parent.replay_expected_effects,
            retry_effects=parent.retry_effects,
        )
        if parent.frame != item.frame or parent.attempt != attempt:
            raise ValueError("managed payload parent authority must match its item and attempt")


def _validate_managed_execution_outcome_fields(
    *,
    index: object,
    value: object,
    error: object,
    coordination_required: object,
    report: object,
) -> None:
    """Validate one worker result without trusting its dataclass annotation."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise TypeError("managed outcome indexes must be non-negative integers")
    if type(coordination_required) is not bool:
        raise TypeError("managed outcome coordination flag must be a bool")
    if error is not None and not isinstance(error, Exception):
        raise TypeError("managed outcome errors must be Exception values or None")
    if report is not None:
        if not isinstance(report, ManagedClaimReport):
            raise TypeError("managed outcome reports must be claim reports or None")
        _validate_managed_claim_report_fields(
            frame=report.frame,
            attempt=report.attempt,
            child_count=report.child_count,
            effects=report.effects,
            successful_effects=report.successful_effects,
        )
    if coordination_required and error is not None:
        raise ValueError("managed coordination outcomes cannot contain errors")
    if error is not None and value is not None:
        raise ValueError("managed error outcomes cannot contain successful values")


def _validated_managed_unit_frame_snapshot(frame: object) -> ManagedUnitFrame:
    """Deeply reconstruct one transported managed frame."""
    if not isinstance(frame, ManagedUnitFrame):
        raise TypeError("transported managed frames must be ManagedUnitFrame values")
    _validate_managed_unit_frame_fields(
        unit_segment=frame.unit_segment,
        token=frame.token,
        derivation_abi=frame.derivation_abi,
    )
    return ManagedUnitFrame(
        unit_segment=frame.unit_segment,
        token=ManagedWorkItemToken(frame.token.value),
        derivation_abi=frame.derivation_abi,
    )


def _validated_managed_attempt_snapshot(attempt: object) -> ManagedAttemptState:
    """Deeply reconstruct one transported managed attempt."""
    if not isinstance(attempt, ManagedAttemptState):
        raise TypeError("transported managed attempts must be ManagedAttemptState values")
    _validate_managed_attempt_fields(
        work_item_token=attempt.work_item_token,
        attempt_token=attempt.attempt_token,
    )
    return ManagedAttemptState(
        work_item_token=ManagedWorkItemToken(attempt.work_item_token.value),
        attempt_token=attempt.attempt_token,
    )


def _validated_managed_work_item_snapshot(item: object) -> ManagedWorkItem:
    """Deeply reconstruct one transported work item while preserving values."""
    if not isinstance(item, ManagedWorkItem):
        raise TypeError("transported work items must be ManagedWorkItem values")
    _validate_managed_work_item_fields(
        index=item.index,
        values=item.values,
        frame=item.frame,
    )
    return ManagedWorkItem(
        index=item.index,
        values=item.values,
        frame=_validated_managed_unit_frame_snapshot(item.frame),
    )


def _validated_managed_parent_envelope_snapshot(
    parent: object,
) -> ManagedParentEnvelope:
    """Deeply reconstruct one transported parent authority envelope."""
    if not isinstance(parent, ManagedParentEnvelope):
        raise TypeError("transported parent authority must be a ManagedParentEnvelope")
    _validate_managed_parent_envelope_fields(
        root_words=parent.root_words,
        parent_occurrence_path=parent.parent_occurrence_path,
        frame=parent.frame,
        attempt=parent.attempt,
        replay_expected_effects=parent.replay_expected_effects,
        retry_effects=parent.retry_effects,
    )
    replay_expected_effects = (
        None
        if parent.replay_expected_effects is None
        else tuple(
            _validated_managed_effect_claim_snapshot(effect)
            for effect in parent.replay_expected_effects
        )
    )
    return ManagedParentEnvelope(
        root_words=parent.root_words,
        parent_occurrence_path=parent.parent_occurrence_path,
        frame=_validated_managed_unit_frame_snapshot(parent.frame),
        attempt=_validated_managed_attempt_snapshot(parent.attempt),
        replay_expected_effects=replay_expected_effects,
        retry_effects=tuple(
            _validated_managed_effect_claim_snapshot(effect) for effect in parent.retry_effects
        ),
    )


def _validated_managed_prefect_payload_snapshot(
    payload: object,
) -> ManagedPrefectPayload:
    """Deeply reconstruct a payload before worker submission or execution."""
    if not isinstance(payload, ManagedPrefectPayload):
        raise TypeError("transported worker payloads must be ManagedPrefectPayload values")
    _validate_managed_prefect_payload_fields(
        item=payload.item,
        attempt=payload.attempt,
        provenance_mode=payload.provenance_mode,
        parent=payload.parent,
    )
    return ManagedPrefectPayload(
        item=_validated_managed_work_item_snapshot(payload.item),
        attempt=_validated_managed_attempt_snapshot(payload.attempt),
        provenance_mode=payload.provenance_mode,
        parent=(
            None
            if payload.parent is None
            else _validated_managed_parent_envelope_snapshot(payload.parent)
        ),
    )


def _validated_managed_execution_outcome_snapshot(
    outcome: object,
) -> ManagedExecutionOutcome:
    """Deeply reconstruct a worker outcome before parent-side admission."""
    if not isinstance(outcome, ManagedExecutionOutcome):
        raise TypeError("transported worker outcomes must be ManagedExecutionOutcome values")
    _validate_managed_execution_outcome_fields(
        index=outcome.index,
        value=outcome.value,
        error=outcome.error,
        coordination_required=outcome.coordination_required,
        report=outcome.report,
    )
    return ManagedExecutionOutcome(
        index=outcome.index,
        value=outcome.value,
        error=outcome.error,
        coordination_required=outcome.coordination_required,
        report=(
            None
            if outcome.report is None
            else _validated_managed_claim_report_snapshot(outcome.report)
        ),
    )


def point_unit_segment() -> _RandomEventPath:
    """Return the canonical segment for one plain point evaluation."""
    return ("managed-unit", _MANAGED_WORK_ITEM_ABI, "point", 0)


def sweep_unit_segment(coordinates: tuple[int, ...]) -> _RandomEventPath:
    """Return the canonical segment for one row-major sweep cell."""
    if not isinstance(coordinates, tuple):
        raise TypeError("managed sweep coordinates must be a tuple")
    if not coordinates:
        raise ValueError("managed sweep segments require at least one coordinate")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in coordinates):
        raise TypeError("managed sweep coordinates must be non-boolean integers")
    if any(item < 0 for item in coordinates):
        raise ValueError("managed sweep coordinates must be non-negative")
    return ("managed-unit", _MANAGED_WORK_ITEM_ABI, "sweep-cell", *coordinates)


def lifted_evaluation_unit_segment(
    logical_unit_id: tuple[str | int, ...],
    flat_index: int,
) -> _RandomEventPath:
    """Return the canonical segment for one evaluation within a lifted unit."""
    if not isinstance(logical_unit_id, tuple):
        raise TypeError("managed lifted logical-unit identities must be tuples")
    if not _is_logical_unit_id(logical_unit_id):
        raise ValueError("managed lifted logical-unit identities must be canonical")
    if isinstance(flat_index, bool) or not isinstance(flat_index, int):
        raise TypeError("managed lifted evaluation indexes must be non-boolean integers")
    if flat_index < 0:
        raise ValueError("managed lifted evaluation indexes must be non-negative")
    return (
        "managed-unit",
        _MANAGED_WORK_ITEM_ABI,
        "lifted-evaluation",
        logical_unit_id,
        flat_index,
    )


def _is_managed_unit_segment(segment: tuple[object, ...]) -> bool:
    """Return whether a segment follows the closed managed-unit grammar."""
    if len(segment) < 3 or segment[:2] != ("managed-unit", _MANAGED_WORK_ITEM_ABI):
        return False
    layout = segment[2]
    if layout == "point":
        return len(segment) == 4 and _is_nonnegative_int(segment[3]) and segment[3] == 0
    if layout == "sweep-cell":
        return len(segment) >= 4 and all(_is_nonnegative_int(item) for item in segment[3:])
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
        and all(_is_nonnegative_int(item) for item in value[1:])
    )


def _is_nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _unique_effects(
    effects: tuple[ManagedEffectClaim, ...],
    *,
    field_name: str,
) -> dict[tuple[object, ...], ManagedEffectClaim]:
    """Index an effect tuple while rejecting duplicate structural identities."""
    result = {}
    for effect in effects:
        identity = (
            effect.occurrence_path,
            effect.stochastic_source_id,
            effect.logical_unit_id,
        )
        if identity in result:
            raise ValueError(f"{field_name} contains a duplicate effect identity")
        result[identity] = effect
    return result


def make_managed_work_items(
    call_values: Sequence[Mapping[str, Any]],
    *,
    unit_segments: Sequence[_RandomEventPath],
) -> tuple[ManagedWorkItem, ...]:
    """Freeze ordered call mappings and their preassigned unit segments."""
    if len(call_values) != len(unit_segments):
        raise ValueError("call values and managed unit segments must have equal lengths")
    return tuple(
        ManagedWorkItem(
            index=index,
            values=tuple(values.items()),
            frame=ManagedUnitFrame(
                unit_segment=tuple(unit_segments[index]),
                token=ManagedWorkItemToken.create(),
            ),
        )
        for index, values in enumerate(call_values)
    )
