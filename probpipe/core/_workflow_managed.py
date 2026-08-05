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

_MANAGED_WORK_ITEM_ABI = "probpipe.managed_work_item/v1"

type _CanonicalDescriptorValue = (
    None | bool | int | str | bytes | tuple[_CanonicalDescriptorValue, ...]
)
type _CanonicalDescriptor = tuple[_CanonicalDescriptorValue, ...]


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


@dataclass(frozen=True, slots=True)
class ManagedWorkItemToken:
    """Opaque, serializable ownership token for one managed work item."""

    value: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.value, bytes) or len(self.value) != 16:
            raise TypeError("managed work-item tokens must contain exactly 16 bytes")

    @classmethod
    def create(cls) -> ManagedWorkItemToken:
        """Create one process-independent operational token."""
        return cls(uuid.uuid4().bytes)


@dataclass(frozen=True, slots=True)
class ManagedUnitFrame:
    """Canonical logical-unit binding transported with one work item."""

    unit_segment: _RandomEventPath
    token: ManagedWorkItemToken
    derivation_abi: str = _MANAGED_WORK_ITEM_ABI

    def __post_init__(self) -> None:
        if not isinstance(self.unit_segment, tuple):
            raise TypeError("managed unit segments must be tuples")
        if not _is_managed_unit_segment(self.unit_segment):
            raise ValueError("managed unit segments must use a canonical managed unit segment")
        if self.derivation_abi != _MANAGED_WORK_ITEM_ABI:
            raise ValueError(f"unsupported managed work-item ABI: {self.derivation_abi!r}")


@dataclass(frozen=True, slots=True)
class ManagedWorkItem:
    """One immutable, canonically indexed workflow evaluation request."""

    index: int
    values: tuple[tuple[str, Any], ...]
    frame: ManagedUnitFrame

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise TypeError("managed work-item indexes must be non-negative integers")
        if not isinstance(self.values, tuple) or any(
            not isinstance(entry, tuple) or len(entry) != 2 or not isinstance(entry[0], str)
            for entry in self.values
        ):
            raise TypeError("managed work-item values must be name/value tuples")

    def call_values(self) -> dict[str, Any]:
        """Materialize the keyword mapping used for execution."""
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class ManagedAttemptState:
    """Operational identity for one execution attempt of a work item."""

    work_item_token: ManagedWorkItemToken
    attempt_token: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_token, bytes) or len(self.attempt_token) != 16:
            raise TypeError("managed attempt tokens must contain exactly 16 bytes")

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
        if not isinstance(self.frame, ManagedUnitFrame) or not isinstance(
            self.attempt,
            ManagedAttemptState,
        ):
            raise TypeError("managed parent authority requires a frame and attempt")
        if (
            not isinstance(self.root_words, tuple)
            or len(self.root_words) != 2
            or any(
                isinstance(word, bool) or not isinstance(word, int) or not 0 <= word <= 0xFFFFFFFF
                for word in self.root_words
            )
        ):
            raise TypeError("managed parent root words must be two uint32 integers")
        if not isinstance(self.parent_occurrence_path, tuple):
            raise TypeError("managed parent occurrence paths must be tuples")
        _validate_random_event_value(self.parent_occurrence_path)
        if self.attempt.work_item_token != self.frame.token:
            raise ValueError("managed parent attempt must own its frame")
        if self.replay_expected_effects is not None and (
            not isinstance(self.replay_expected_effects, tuple)
            or any(
                not isinstance(effect, ManagedEffectClaim)
                for effect in self.replay_expected_effects
            )
        ):
            raise TypeError("managed replay expectations must be effect tuples or None")
        if not isinstance(self.retry_effects, tuple) or any(
            not isinstance(effect, ManagedEffectClaim) for effect in self.retry_effects
        ):
            raise TypeError("managed retry effects must be an effect tuple")


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
        if not isinstance(self.occurrence_path, tuple):
            raise TypeError("managed effect occurrence paths must be tuples")
        if not isinstance(self.stochastic_source_id, tuple) or not isinstance(
            self.logical_unit_id,
            tuple,
        ):
            raise TypeError("managed effect source and unit identities must be tuples")
        _validate_random_event_value(self.occurrence_path)
        _validate_random_event_value(self.stochastic_source_id)
        _validate_random_event_value(self.logical_unit_id)
        if not isinstance(self.occurrence_kind, str):
            raise TypeError("managed effect occurrence_kind must be a string")
        if self.occurrence_kind not in {"invocation", "operation"}:
            raise ValueError("managed effect occurrence_kind must be 'invocation' or 'operation'")
        _validate_stochastic_effect_fields(
            operation_kind=self.operation_kind,
            execution_mode=self.execution_mode,
            sample_shape=self.sample_shape,
            sampling_abi=self.sampling_abi,
            provider_abi=self.provider_abi,
            record_path=self.record_path,
            descendant_descriptor=self.descendant_descriptor,
        )


@dataclass(frozen=True, slots=True)
class ManagedClaimReport:
    """Serializable claim summary returned by one remote attempt."""

    frame: ManagedUnitFrame
    attempt: ManagedAttemptState
    child_count: int
    effects: tuple[ManagedEffectClaim, ...] = ()
    successful_effects: tuple[ManagedEffectClaim, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ManagedUnitFrame) or not isinstance(
            self.attempt,
            ManagedAttemptState,
        ):
            raise TypeError("managed claim reports require a frame and attempt")
        if self.attempt.work_item_token != self.frame.token:
            raise ValueError("managed report attempt must own its frame")
        if (
            isinstance(self.child_count, bool)
            or not isinstance(self.child_count, int)
            or self.child_count < 0
        ):
            raise TypeError("managed child counts must be non-negative integers")
        if not isinstance(self.effects, tuple) or any(
            not isinstance(effect, ManagedEffectClaim) for effect in self.effects
        ):
            raise TypeError("managed effect reports must contain a tuple of effects")
        if not isinstance(self.successful_effects, tuple) or any(
            not isinstance(effect, ManagedEffectClaim) for effect in self.successful_effects
        ):
            raise TypeError("managed successful effects must contain a tuple of effects")
        effects_by_identity = _unique_effects(
            self.effects,
            field_name="managed effect report",
        )
        successful_by_identity = _unique_effects(
            self.successful_effects,
            field_name="managed successful effect report",
        )
        if any(
            effects_by_identity.get(identity) != effect
            for identity, effect in successful_by_identity.items()
        ):
            raise ValueError("managed successful effects must be claimed by the same attempt")


@dataclass(frozen=True, slots=True)
class ManagedPrefectPayload:
    """Serializable Prefect task input for an initial or coordinated attempt."""

    item: ManagedWorkItem
    attempt: ManagedAttemptState
    parent: ManagedParentEnvelope | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.item, ManagedWorkItem) or not isinstance(
            self.attempt,
            ManagedAttemptState,
        ):
            raise TypeError("managed Prefect payloads require a work item and attempt")
        if self.parent is not None and not isinstance(self.parent, ManagedParentEnvelope):
            raise TypeError("managed Prefect parent authority must be an envelope or None")
        if self.attempt.work_item_token != self.item.frame.token:
            raise ValueError("managed payload attempt must own its work item")
        if self.parent is not None and (
            self.parent.frame != self.item.frame or self.parent.attempt != self.attempt
        ):
            raise ValueError("managed payload parent authority must match its item and attempt")


@dataclass(frozen=True, slots=True)
class ManagedExecutionOutcome:
    """Serializable Prefect result with operational claim information."""

    index: int
    value: Any = None
    error: Exception | None = None
    coordination_required: bool = False
    report: ManagedClaimReport | None = None


def point_unit_segment() -> _RandomEventPath:
    """Return the canonical segment for one plain point evaluation."""
    return ("managed-unit", _MANAGED_WORK_ITEM_ABI, "point", 0)


def sweep_unit_segment(coordinates: tuple[int, ...]) -> _RandomEventPath:
    """Return the canonical segment for one row-major sweep cell."""
    return ("managed-unit", _MANAGED_WORK_ITEM_ABI, "sweep-cell", *coordinates)


def lifted_evaluation_unit_segment(
    logical_unit_id: tuple[str | int, ...],
    flat_index: int,
) -> _RandomEventPath:
    """Return the canonical segment for one evaluation within a lifted unit."""
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
