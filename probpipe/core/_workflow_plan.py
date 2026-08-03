"""Function broadcast-planning helpers.

This private module classifies already-normalized workflow inputs into
the broadcast regime and sweep shape that ``Function`` should
execute. Planning is intentionally side-effect-free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product as cartesian_product
from math import prod
from typing import Any, Literal

from . import _workflow_call, _workflow_distribution_normalization
from ._distribution_array import DistributionArray
from ._record_array import RecordArray
from ._record_distribution import _RecordDistributionView
from .distribution import Distribution, EmpiricalDistribution

BroadcastRegime = Literal["none", "distribution", "sweep", "nested"]
StochasticSourceKind = Literal["direct", "record_view"]
StochasticExecutionMode = Literal["exact", "sampled"]
StochasticEvaluationMode = Literal["exact", "sampled", "mixed_exact_sampled"]
LogicalUnitLayout = Literal["singleton", "canonical_sweep"]
StructuralRngId = tuple[str | int, ...]


@dataclass(frozen=True)
class ArrayBroadcastGroup:
    """One parent-identity group of array-valued sweep arguments."""

    arg_refs: tuple[_workflow_call.WorkflowInputRef, ...]
    batch_shape: tuple[int, ...]
    size: int


@dataclass(frozen=True)
class BroadcastPlan:
    """Pure broadcast classification for one resolved workflow call."""

    regime: BroadcastRegime
    dist_args: tuple[_workflow_call.WorkflowInputRef, ...]
    array_args: tuple[_workflow_call.WorkflowInputRef, ...]
    array_groups: tuple[ArrayBroadcastGroup, ...]
    sweep_batch_shape: tuple[int, ...]
    n_sweep: int


@dataclass(frozen=True)
class StochasticSourceGroup:
    """One canonically ordered stochastic source consumed by lifting."""

    index: int
    arg_refs: tuple[_workflow_call.WorkflowInputRef, ...]
    source_kind: StochasticSourceKind
    execution_mode: StochasticExecutionMode
    exact_size: int | None

    @property
    def stochastic_source_id(self) -> StructuralRngId:
        """Return the structural identity used by the workflow RNG broker."""
        return ("source-group", self.index)


@dataclass(frozen=True)
class LogicalUnit:
    """One singleton or row-major sweep cell in a lifting plan."""

    layout: LogicalUnitLayout
    flat_index: int
    coordinates: tuple[int, ...]

    @property
    def logical_unit_id(self) -> StructuralRngId:
        """Return the structural identity used by the workflow RNG broker."""
        if self.layout == "singleton":
            return ("singleton",)
        return ("cell", *self.coordinates)


@dataclass(frozen=True)
class PlannedRandomEvent:
    """Derived source/unit identity for one planned random event."""

    stochastic_source_id: StructuralRngId
    logical_unit_id: StructuralRngId


@dataclass(frozen=True)
class StochasticPlan:
    """Immutable stochastic lifting decisions for one normalized call."""

    evaluation_mode: StochasticEvaluationMode
    arg_refs: tuple[_workflow_call.WorkflowInputRef, ...]
    source_groups: tuple[StochasticSourceGroup, ...]
    logical_units: tuple[LogicalUnit, ...]
    n_broadcast_samples: int
    sample_shape: tuple[int, ...] | None
    exact_group_order: tuple[int, ...]
    exact_combination_order: tuple[tuple[int, ...], ...]
    repetitions_per_combination: int
    n_evaluations: int

    @property
    def random_events(self) -> tuple[PlannedRandomEvent, ...]:
        """Derive sampled source/unit events without storing a second table."""
        return tuple(
            PlannedRandomEvent(
                stochastic_source_id=group.stochastic_source_id,
                logical_unit_id=unit.logical_unit_id,
            )
            for unit in self.logical_units
            for group in self.source_groups
            if group.execution_mode == "sampled"
        )


def build_broadcast_plan(
    *,
    values: Mapping[str, Any],
    signature_info: _workflow_call.WorkflowSignatureInfo,
) -> BroadcastPlan:
    """Classify normalized values into a broadcast execution plan."""
    dist_args: list[_workflow_call.WorkflowInputRef] = []
    array_args: list[_workflow_call.WorkflowInputRef] = []

    for ref in _workflow_call.iter_input_refs(signature_info, values):
        value = _workflow_call.input_ref_value(values, ref)
        expected = _workflow_call.input_ref_hint(signature_info, ref)

        is_record_array = isinstance(value, RecordArray)
        is_dist_array = isinstance(value, DistributionArray)
        if (is_record_array or is_dist_array) and len(value.batch_shape) > 0:
            if (
                _is_same_array_hint(
                    expected,
                    is_record_array=is_record_array,
                    is_dist_array=is_dist_array,
                )
                or expected is Any
            ):
                continue
            array_args.append(ref)
            continue

        if isinstance(value, Distribution):
            if _workflow_distribution_normalization.is_distribution_hint(expected):
                continue
            dist_args.append(ref)

    array_groups = group_array_args_by_parent(values=values, refs=array_args)
    sweep_batch_shape = tuple(axis for group in array_groups for axis in group.batch_shape)
    n_sweep = prod(sweep_batch_shape)

    return BroadcastPlan(
        regime=_broadcast_regime(dist_args=dist_args, array_args=array_args),
        dist_args=tuple(dist_args),
        array_args=tuple(array_args),
        array_groups=tuple(array_groups),
        sweep_batch_shape=sweep_batch_shape,
        n_sweep=n_sweep,
    )


def build_stochastic_plan(
    values: Mapping[str, Any],
    broadcast_plan: BroadcastPlan,
    n_broadcast_samples: int,
) -> StochasticPlan | None:
    """Build immutable stochastic decisions without claiming random events."""
    if broadcast_plan.regime in ("none", "sweep"):
        return None

    _validate_stochastic_sample_count(n_broadcast_samples)
    arg_refs = tuple(broadcast_plan.dist_args)
    grouped_refs, source_values, source_kinds = _group_stochastic_sources(
        values=values,
        refs=arg_refs,
    )

    candidates = [
        (index, source)
        for index, (source, source_kind) in enumerate(zip(source_values, source_kinds))
        if source_kind == "direct"
        and isinstance(source, EmpiricalDistribution)
        and source.num_atoms <= n_broadcast_samples
    ]
    candidates.sort(key=lambda pair: pair[1].num_atoms)

    exact_group_indices: list[int] = []
    exact_sizes: dict[int, int] = {}
    exact_product = 1
    for index, source in candidates:
        size = source.num_atoms
        if exact_product * size <= n_broadcast_samples:
            exact_group_indices.append(index)
            exact_sizes[index] = size
            exact_product *= size

    source_groups = tuple(
        StochasticSourceGroup(
            index=index,
            arg_refs=tuple(refs),
            source_kind=source_kinds[index],
            execution_mode="exact" if index in exact_sizes else "sampled",
            exact_size=exact_sizes.get(index),
        )
        for index, refs in enumerate(grouped_refs)
    )
    logical_units = _build_logical_units(broadcast_plan)
    exact_group_order = tuple(exact_group_indices)
    exact_combination_order = tuple(
        cartesian_product(*(range(exact_sizes[index]) for index in exact_group_order))
    )

    has_sampled_groups = any(group.execution_mode == "sampled" for group in source_groups)
    if not has_sampled_groups:
        evaluation_mode: StochasticEvaluationMode = "exact"
        repetitions_per_combination = 1
        n_evaluations = exact_product
        sample_shape = None
    elif exact_group_order:
        evaluation_mode = "mixed_exact_sampled"
        repetitions_per_combination = max(1, n_broadcast_samples // exact_product)
        n_evaluations = exact_product * repetitions_per_combination
        sample_shape = (n_evaluations,)
    else:
        evaluation_mode = "sampled"
        repetitions_per_combination = n_broadcast_samples
        n_evaluations = n_broadcast_samples
        sample_shape = (n_broadcast_samples,)

    return StochasticPlan(
        evaluation_mode=evaluation_mode,
        arg_refs=arg_refs,
        source_groups=source_groups,
        logical_units=logical_units,
        n_broadcast_samples=n_broadcast_samples,
        sample_shape=sample_shape,
        exact_group_order=exact_group_order,
        exact_combination_order=exact_combination_order,
        repetitions_per_combination=repetitions_per_combination,
        n_evaluations=n_evaluations,
    )


def _group_stochastic_sources(
    *,
    values: Mapping[str, Any],
    refs: Sequence[_workflow_call.WorkflowInputRef],
) -> tuple[
    list[list[_workflow_call.WorkflowInputRef]],
    list[Distribution],
    list[StochasticSourceKind],
]:
    """Discover current-wave sources while keeping object IDs out of plans."""
    grouped_refs: list[list[_workflow_call.WorkflowInputRef]] = []
    source_values: list[Distribution] = []
    source_kinds: list[StochasticSourceKind] = []
    view_group_by_parent: dict[int, int] = {}

    for ref in refs:
        value = _workflow_call.input_ref_value(values, ref)
        if isinstance(value, _RecordDistributionView):
            parent_identity = id(value.parent)
            existing_index = view_group_by_parent.get(parent_identity)
            if existing_index is not None:
                grouped_refs[existing_index].append(ref)
                continue
            view_group_by_parent[parent_identity] = len(grouped_refs)
            grouped_refs.append([ref])
            source_values.append(value.parent)
            source_kinds.append("record_view")
            continue

        grouped_refs.append([ref])
        source_values.append(value)
        source_kinds.append("direct")

    return grouped_refs, source_values, source_kinds


def _build_logical_units(broadcast_plan: BroadcastPlan) -> tuple[LogicalUnit, ...]:
    if broadcast_plan.regime == "distribution":
        return (LogicalUnit(layout="singleton", flat_index=0, coordinates=()),)

    return tuple(
        LogicalUnit(
            layout="canonical_sweep",
            flat_index=flat_index,
            coordinates=tuple(coordinates),
        )
        for flat_index, coordinates in enumerate(
            cartesian_product(*(range(axis) for axis in broadcast_plan.sweep_batch_shape))
        )
    )


def _validate_stochastic_sample_count(n_broadcast_samples: int) -> None:
    if not isinstance(n_broadcast_samples, int):
        raise TypeError(f"n_broadcast_samples must be an integer; got {n_broadcast_samples!r}")
    if n_broadcast_samples <= 0:
        raise ValueError(
            f"n_broadcast_samples must be a positive integer; got {n_broadcast_samples!r}"
        )


def group_by_parent(
    *,
    values: Mapping[str, Any],
    refs: Sequence[_workflow_call.WorkflowInputRef],
) -> dict[int, list[_workflow_call.WorkflowInputRef]]:
    """Group input references by the identity of their source parent."""
    groups: dict[int, list[_workflow_call.WorkflowInputRef]] = {}
    for ref in refs:
        value = _workflow_call.input_ref_value(values, ref)
        parent = getattr(value, "parent", value)
        groups.setdefault(id(parent), []).append(ref)
    return groups


def group_array_args_by_parent(
    *,
    values: Mapping[str, Any],
    refs: Sequence[_workflow_call.WorkflowInputRef],
) -> tuple[ArrayBroadcastGroup, ...]:
    """Build parent-identity groups for array-valued sweep arguments."""
    groups: list[ArrayBroadcastGroup] = []
    for arg_refs in group_by_parent(values=values, refs=refs).values():
        first = _workflow_call.input_ref_value(values, arg_refs[0])
        batch_shape = tuple(first.batch_shape)
        groups.append(
            ArrayBroadcastGroup(
                arg_refs=tuple(arg_refs),
                batch_shape=batch_shape,
                size=prod(batch_shape),
            )
        )
    return tuple(groups)


def _broadcast_regime(
    *,
    dist_args: Sequence[_workflow_call.WorkflowInputRef],
    array_args: Sequence[_workflow_call.WorkflowInputRef],
) -> BroadcastRegime:
    if dist_args and array_args:
        return "nested"
    if dist_args:
        return "distribution"
    if array_args:
        return "sweep"
    return "none"


def _is_same_array_hint(
    expected: Any,
    *,
    is_record_array: bool,
    is_dist_array: bool,
) -> bool:
    try:
        return isinstance(expected, type) and (
            (is_record_array and issubclass(expected, RecordArray))
            or (is_dist_array and issubclass(expected, DistributionArray))
        )
    except TypeError:
        return False
