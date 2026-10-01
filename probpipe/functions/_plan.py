"""Function broadcast-planning helpers.

This private module classifies already-normalized workflow inputs into
the broadcast regime and sweep shape that ``Function`` should
execute. Planning is intentionally side-effect-free.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product as cartesian_product
from math import prod
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from ..core._batch import Batch
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution
from ..values import _binding
from . import _descendants, _normalization

BroadcastRegime = Literal["none", "distribution", "sweep", "nested"]
StochasticExecutionMode = Literal["exact", "sampled"]
StochasticEvaluationMode = Literal["exact", "sampled", "mixed_exact_sampled"]
LogicalUnitLayout = Literal["singleton", "canonical_sweep"]
StructuralRngId = tuple[str | int, ...]


@dataclass(frozen=True)
class ArrayBroadcastGroup:
    """One zip group of swept batches, read along the same axes.

    The batches of one group carry the same level names, since levels are how
    batches align.
    """

    arg_refs: tuple[_binding.WorkflowInputRef, ...]
    batch_shape: tuple[int, ...]
    size: int
    # What the group's axes range over, for the aggregate to mint its levels
    # under: the batches' own level names are what an output must carry to align
    # with them.
    level_names: tuple[str, ...]
    axis_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BroadcastPlan:
    """Pure broadcast classification for one resolved workflow call."""

    regime: BroadcastRegime
    dist_args: tuple[_binding.WorkflowInputRef, ...]
    array_args: tuple[_binding.WorkflowInputRef, ...]
    array_groups: tuple[ArrayBroadcastGroup, ...]
    sweep_batch_shape: tuple[int, ...]
    sweep_level_names: tuple[str, ...]
    sweep_axis_groups: tuple[tuple[int, ...], ...]
    n_sweep: int


@dataclass(frozen=True)
class StochasticConsumerPlan:
    """Canonical projection of one argument from a co-sampled root."""

    arg_ref: _binding.WorkflowInputRef
    record_path: tuple[str, ...]
    descendant_descriptor: tuple[Any, ...] | None
    _descriptor_abi_summary: _descendants._DescriptorAbiSummary = field(
        init=False,
        compare=False,
        hash=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Derive non-authoritative execution metadata from the descriptor."""
        object.__setattr__(
            self,
            "_descriptor_abi_summary",
            _descendants._summarize_descriptor_abis(self.descendant_descriptor),
        )


@dataclass(frozen=True)
class StochasticSourceGroup:
    """One recursive stochastic root and its ordered consumers."""

    index: int
    consumers: tuple[StochasticConsumerPlan, ...]
    execution_mode: StochasticExecutionMode
    exact_size: int | None

    @property
    def arg_refs(self) -> tuple[_binding.WorkflowInputRef, ...]:
        """Return consumer references in canonical argument order."""
        return tuple(consumer.arg_ref for consumer in self.consumers)

    @property
    def stochastic_source_id(self) -> StructuralRngId:
        """Return the structural identity used by the workflow RNG broker."""
        return ("source-group", self.index)


@dataclass(frozen=True)
class StochasticRuntimeBinding:
    """Live root and preflight-captured evaluators for one source group."""

    root: Distribution = field(compare=False, hash=False, repr=False)
    sample_root: Callable[[Any, tuple[int, ...]], Any] = field(
        compare=False,
        hash=False,
        repr=False,
    )
    consumer_evaluators: tuple[Callable[[Any], Any], ...] = field(
        compare=False,
        hash=False,
        repr=False,
    )


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
    arg_refs: tuple[_binding.WorkflowInputRef, ...]
    source_groups: tuple[StochasticSourceGroup, ...]
    logical_units: tuple[LogicalUnit, ...]
    n_broadcast_samples: int
    sample_shape: tuple[int, ...] | None
    exact_group_order: tuple[int, ...]
    exact_combination_order: tuple[tuple[int, ...], ...]
    repetitions_per_combination: int
    n_evaluations: int
    runtime_bindings: tuple[StochasticRuntimeBinding, ...] = field(
        compare=False,
        hash=False,
        repr=False,
    )

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


def _is_batched(value: Any) -> bool:
    """Whether *value* holds a multiplicity on at least one level.

    Any Batch is such an operand: what makes a value sweepable is that it holds
    a multiplicity on named levels, which is the Batch contract rather than
    anything specific to records.
    """
    return isinstance(value, Batch) and len(value.batch_shape) > 0


def is_swept(value: Any, expected: Any) -> bool:
    """Whether the call sweeps *value* at a parameter whose lifting annotation is *expected*.

    A batched argument is swept unless the annotation names a batched class it
    satisfies or is ``Any``, either of which passes it to the body whole.
    """
    return _is_batched(value) and not (_value_matches_hint(value, expected) or expected is Any)


def is_broadcast(value: Any, expected: Any) -> bool:
    """Whether the call samples *value* at a parameter whose lifting annotation is *expected*.

    A distribution is broadcast unless it is batched, in which case the call
    sweeps it or passes it whole, or the annotation names a distribution, which
    consumes it.
    """
    return (
        isinstance(value, Distribution)
        and not _is_batched(value)
        and not _normalization.is_distribution_hint(expected)
    )


def build_broadcast_plan(
    *,
    values: Mapping[str, Any],
    signature_info: _binding.WorkflowSignatureInfo,
) -> BroadcastPlan:
    """Classify normalized values into a broadcast execution plan."""
    dist_args: list[_binding.WorkflowInputRef] = []
    array_args: list[_binding.WorkflowInputRef] = []

    for ref in _binding.iter_input_refs(signature_info, values):
        value = _binding.input_ref_value(values, ref)
        expected = _binding.input_ref_hint(signature_info, ref)
        if is_swept(value, expected):
            array_args.append(ref)
        elif is_broadcast(value, expected):
            dist_args.append(ref)

    array_groups = build_array_zip_groups(values=values, refs=array_args)
    sweep_batch_shape = tuple(axis for group in array_groups for axis in group.batch_shape)
    sweep_level_names = tuple(n for group in array_groups for n in group.level_names)
    sweep_axis_groups = tuple(g for group in array_groups for g in group.axis_groups)
    n_sweep = prod(sweep_batch_shape)

    return BroadcastPlan(
        regime=_broadcast_regime(dist_args=dist_args, array_args=array_args),
        dist_args=tuple(dist_args),
        array_args=tuple(array_args),
        array_groups=tuple(array_groups),
        sweep_batch_shape=sweep_batch_shape,
        sweep_level_names=sweep_level_names,
        sweep_axis_groups=sweep_axis_groups,
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
    grouped_consumers, source_values, runtime_samplers, runtime_evaluators = (
        _group_stochastic_sources(
            values=values,
            refs=arg_refs,
        )
    )

    candidates = [
        (index, source)
        for index, source in enumerate(source_values)
        if isinstance(source, EmpiricalDistribution) and source.num_atoms <= n_broadcast_samples
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
            consumers=tuple(consumers),
            execution_mode="exact" if index in exact_sizes else "sampled",
            exact_size=exact_sizes.get(index),
        )
        for index, consumers in enumerate(grouped_consumers)
    )
    runtime_bindings = tuple(
        StochasticRuntimeBinding(
            root=source,
            sample_root=runtime_samplers[index],
            consumer_evaluators=tuple(runtime_evaluators[index]),
        )
        for index, source in enumerate(source_values)
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
        runtime_bindings=runtime_bindings,
    )


def _group_stochastic_sources(
    *,
    values: Mapping[str, Any],
    refs: Sequence[_binding.WorkflowInputRef],
) -> tuple[
    list[list[StochasticConsumerPlan]],
    list[Distribution],
    list[Callable[[Any, tuple[int, ...]], Any]],
    list[list[Callable[[Any], Any]]],
]:
    """Discover live roots while keeping object IDs out of canonical plans."""
    grouped_consumers: list[list[StochasticConsumerPlan]] = []
    source_values: list[Distribution] = []
    runtime_samplers: list[Callable[[Any, tuple[int, ...]], Any]] = []
    runtime_evaluators: list[list[Callable[[Any], Any]]] = []
    group_index_by_root_id: dict[int, int] = {}

    source_entries = tuple((ref, _binding.input_ref_value(values, ref)) for ref in refs)
    captured_consumers = _descendants.capture_stochastic_consumers(
        tuple(value for _ref, value in source_entries)
    )

    for (ref, _value), captured in zip(source_entries, captured_consumers, strict=True):
        root = captured.root

        root_identity = id(root)
        group_index = group_index_by_root_id.get(root_identity)
        if group_index is None:
            group_index = len(grouped_consumers)
            group_index_by_root_id[root_identity] = group_index
            grouped_consumers.append([])
            source_values.append(root)
            runtime_samplers.append(captured.sample_root)
            runtime_evaluators.append([])
        descendant_descriptor = captured.descendant_descriptor
        if descendant_descriptor is not None:
            descendant_descriptor = (
                "stochastic-descendant",
                ("base_source_slot", group_index),
                ("graph", descendant_descriptor),
            )
        grouped_consumers[group_index].append(
            StochasticConsumerPlan(
                arg_ref=ref,
                record_path=captured.record_path,
                descendant_descriptor=descendant_descriptor,
            )
        )
        runtime_evaluators[group_index].append(captured.evaluator)

    return grouped_consumers, source_values, runtime_samplers, runtime_evaluators


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
    if isinstance(n_broadcast_samples, bool) or not isinstance(n_broadcast_samples, int):
        raise TypeError(f"n_broadcast_samples must be an integer; got {n_broadcast_samples!r}")
    if n_broadcast_samples <= 0:
        raise ValueError(
            f"n_broadcast_samples must be a positive integer; got {n_broadcast_samples!r}"
        )


def group_by_alignment(
    *,
    values: Mapping[str, Any],
    refs: Sequence[_binding.WorkflowInputRef],
) -> list[tuple[Any, tuple[_binding.WorkflowInputRef, ...]]]:
    """Group the swept batches by their level names, with each group's first batch.

    Batches align by level name, so two batches carrying the same levels are
    two readings of one multiplicity and zip, while batches with no level in
    common are independent and form a product. Sibling views from one batch's
    ``select_all`` therefore zip, as do a batch and a view of it. Groups and
    their members keep argument order, so the grouping is a deterministic
    function of the call.
    """
    groups: dict[tuple[str, ...], tuple[Any, list[_binding.WorkflowInputRef]]] = {}
    for ref in refs:
        value = _binding.input_ref_value(values, ref)
        groups.setdefault(tuple(value.level_names), (value, []))[1].append(ref)
    return [(root, tuple(group_refs)) for root, group_refs in groups.values()]


def build_array_zip_groups(
    *,
    values: Mapping[str, Any],
    refs: Sequence[_binding.WorkflowInputRef],
) -> tuple[ArrayBroadcastGroup, ...]:
    """Build the zip groups of the swept batches.

    Every batch in a group is read along the same axes, so they must agree on
    what those axes are; a disagreement is a mistake about which multiplicity is
    which rather than a product to be formed silently.

    Raises
    ------
    ValueError
        If two batches carry the same levels on different axes, or two groups
        share a level without sharing all their levels.
    """
    groups: list[ArrayBroadcastGroup] = []
    for first, arg_refs in group_by_alignment(values=values, refs=refs):
        for ref in arg_refs[1:]:
            other = _binding.input_ref_value(values, ref)
            # Two operands naming the same levels claim the same axes, group by
            # group: agreeing on the flat shape alone would zip a ((2,), (3, 4))
            # partition with a ((2, 3), (4,)) one and hand the output whichever
            # arrived first.
            if tuple(other.axis_groups) != tuple(first.axis_groups):
                raise ValueError(
                    f"{arg_refs[0].label!r} and {ref.label!r} carry the same levels but "
                    f"are batched differently: {tuple(first.axis_groups)} against "
                    f"{tuple(other.axis_groups)}. Levels align by name, so operands naming "
                    f"the same levels must hold them on the same axes"
                )
        batch_shape = tuple(first.batch_shape)
        groups.append(
            ArrayBroadcastGroup(
                arg_refs=tuple(arg_refs),
                batch_shape=batch_shape,
                size=prod(batch_shape),
                level_names=tuple(first.level_names),
                axis_groups=tuple(first.axis_groups),
            )
        )
    # A level name in two groups is one multiplicity read at two geometries:
    # the groups differ, so their level tuples differ, and aligning the shared
    # level across differently-leveled operands — broadcasting the rest — is
    # not built yet. Refused rather than producted: a product would read the
    # shared name as two unrelated axes, and the aggregate would then mint the
    # same level twice.
    owners: dict[str, tuple[str, tuple[str, ...]]] = {}
    for group in groups:
        names = group.level_names
        for level_name in dict.fromkeys(names):
            prior = owners.setdefault(level_name, (group.arg_refs[0].label, names))
            if prior[1] != names or prior[0] != group.arg_refs[0].label:
                raise ValueError(
                    f"{prior[0]!r} and {group.arg_refs[0].label!r} share the level "
                    f"{level_name!r} without sharing all their levels ({prior[1]} against "
                    f"{names}). Aligning one shared level across differently-leveled operands "
                    f"is not supported yet; rename it with with_level_names to sweep them "
                    f"independently, or give both operands the same levels to zip them"
                )
    return tuple(groups)


def _broadcast_regime(
    *,
    dist_args: Sequence[_binding.WorkflowInputRef],
    array_args: Sequence[_binding.WorkflowInputRef],
) -> BroadcastRegime:
    if dist_args and array_args:
        return "nested"
    if dist_args:
        return "distribution"
    if array_args:
        return "sweep"
    return "none"


def _value_matches_hint(value: Any, expected: Any) -> bool:
    """Whether the annotation names a batched container the value satisfies.

    Both halves are load-bearing. The annotation must be a batched-container
    class, because an *element* annotation — ``p: Record`` — is how a body says
    it wants one row, and the
    sweep is what delivers rows. And the value must actually satisfy it: a
    parameter annotated with one batched-record class does not accept the other,
    so family membership alone would deliver a batch whole to a body that
    declared it takes something else, silently skipping the sweep.
    """
    origin = get_origin(expected)
    if origin in (Union, UnionType):
        # An optional container annotation still names the container: the value
        # answers whichever arm it satisfies, and ``None`` answers none.
        return any(_value_matches_hint(value, arm) for arm in get_args(expected))
    base = origin or expected
    try:
        return isinstance(base, type) and issubclass(base, Batch) and isinstance(value, base)
    except TypeError:
        return False
