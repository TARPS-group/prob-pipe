"""Immutable payloads for ProbPipe-managed workflow execution."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_MANAGED_WORK_ITEM_ABI = "probpipe.managed_work_item/v1"


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class ManagedUnitFrame:
    """Canonical logical-unit binding transported with one work item."""

    unit_segment: tuple[Any, ...]
    token: ManagedWorkItemToken
    derivation_abi: str = _MANAGED_WORK_ITEM_ABI

    def __post_init__(self) -> None:
        if not isinstance(self.unit_segment, tuple):
            raise TypeError("managed unit segments must be tuples")
        if self.derivation_abi != _MANAGED_WORK_ITEM_ABI:
            raise ValueError(f"unsupported managed work-item ABI: {self.derivation_abi!r}")


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class ManagedParentEnvelope:
    """Serializable root and occurrence authority for one remote managed unit."""

    root_words: tuple[int, int]
    parent_occurrence_path: tuple[Any, ...]
    frame: ManagedUnitFrame
    replay_expected_effects: tuple[ManagedEffectClaim, ...] | None = None
    retry_effects: tuple[ManagedEffectClaim, ...] = ()

    def __post_init__(self) -> None:
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


@dataclass(frozen=True)
class ManagedEffectClaim:
    """Serializable descriptor of one automatic stochastic effect claim."""

    occurrence_path: tuple[Any, ...]
    occurrence_kind: str
    stochastic_source_id: tuple[str | int, ...]
    logical_unit_id: tuple[str | int, ...]
    operation_kind: str
    execution_mode: str
    sample_shape: tuple[int, ...] | None
    sampling_abi: str
    provider_abi: str

    def __post_init__(self) -> None:
        if not isinstance(self.occurrence_path, tuple):
            raise TypeError("managed effect occurrence paths must be tuples")
        if not isinstance(self.stochastic_source_id, tuple) or not isinstance(
            self.logical_unit_id,
            tuple,
        ):
            raise TypeError("managed effect source and unit identities must be tuples")
        if self.sample_shape is not None and not isinstance(self.sample_shape, tuple):
            raise TypeError("managed effect sample shapes must be tuples or None")


@dataclass(frozen=True)
class ManagedClaimReport:
    """Serializable claim summary returned by one remote attempt."""

    frame: ManagedUnitFrame
    attempt: ManagedAttemptState
    child_count: int
    effects: tuple[ManagedEffectClaim, ...] = ()
    successful_effects: tuple[ManagedEffectClaim, ...] = ()

    def __post_init__(self) -> None:
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
        if any(effect not in self.effects for effect in self.successful_effects):
            raise ValueError("managed successful effects must be claimed by the same attempt")


@dataclass(frozen=True)
class ManagedPrefectPayload:
    """Serializable Prefect task input for an initial or coordinated attempt."""

    item: ManagedWorkItem
    parent: ManagedParentEnvelope | None = None


@dataclass(frozen=True)
class ManagedExecutionOutcome:
    """Serializable Prefect result with operational claim information."""

    index: int
    value: Any = None
    error: Exception | None = None
    coordination_required: bool = False
    report: ManagedClaimReport | None = None


def point_unit_segment() -> tuple[Any, ...]:
    """Return the canonical segment for one plain point evaluation."""
    return ("managed-unit", _MANAGED_WORK_ITEM_ABI, "point", 0)


def sweep_unit_segment(coordinates: tuple[int, ...]) -> tuple[Any, ...]:
    """Return the canonical segment for one row-major sweep cell."""
    return ("managed-unit", _MANAGED_WORK_ITEM_ABI, "sweep-cell", *coordinates)


def lifted_evaluation_unit_segment(
    logical_unit_id: tuple[str | int, ...],
    flat_index: int,
) -> tuple[Any, ...]:
    """Return the canonical segment for one evaluation within a lifted unit."""
    return (
        "managed-unit",
        _MANAGED_WORK_ITEM_ABI,
        "lifted-evaluation",
        logical_unit_id,
        flat_index,
    )


def make_managed_work_items(
    call_values: Sequence[Mapping[str, Any]],
    *,
    unit_segments: Sequence[tuple[Any, ...]],
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
