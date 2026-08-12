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

from . import _workflow_call, _workflow_descendants, _workflow_distribution_normalization
from ._batch import Batch
from ._distribution_array import DistributionArray
from ._record_batch import RecordBatch
from .distribution import Distribution, EmpiricalDistribution

BroadcastRegime = Literal["none", "distribution", "sweep", "nested"]
StochasticExecutionMode = Literal["exact", "sampled"]
StochasticEvaluationMode = Literal["exact", "sampled", "mixed_exact_sampled"]
LogicalUnitLayout = Literal["singleton", "canonical_sweep"]
StructuralRngId = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class ArrayBroadcastGroup:
    """One zip group of array-valued sweep arguments — read along the same axes.

    What puts two arguments in one group depends on what they are: a batch is
    grouped by its level names, since levels are how batches align; a value with
    a parent — a distribution view — by that parent, since sibling views of one
    law have no level names to align on.
    """

    arg_refs: tuple[_workflow_call.WorkflowInputRef, ...]
    batch_shape: tuple[int, ...]
    size: int
    # What the group's axes range over, for the aggregate to mint its levels
    # under. A batched operand already says: its own level names are what an
    # output must carry to align with it. An operand with no levels of its own is
    # named for the argument it arrived as, which is a name from the call rather
    # than one invented here.
    level_names: tuple[str, ...]
    axis_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class BroadcastPlan:
    """Pure broadcast classification for one resolved workflow call."""

    regime: BroadcastRegime
    dist_args: tuple[_workflow_call.WorkflowInputRef, ...]
    array_args: tuple[_workflow_call.WorkflowInputRef, ...]
    array_groups: tuple[ArrayBroadcastGroup, ...]
    sweep_batch_shape: tuple[int, ...]
    sweep_level_names: tuple[str, ...]
    sweep_axis_groups: tuple[tuple[int, ...], ...]
    n_sweep: int


@dataclass(frozen=True, slots=True)
class StochasticConsumerPlan:
    """Canonical projection of one argument from a co-sampled root."""

    arg_ref: _workflow_call.WorkflowInputRef
    record_path: tuple[str, ...]
    descendant_descriptor: tuple[Any, ...] | None
    _descriptor_abi_summary: _workflow_descendants._DescriptorAbiSummary = field(
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
            _workflow_descendants._summarize_descriptor_abis(self.descendant_descriptor),
        )


@dataclass(frozen=True, slots=True)
class StochasticSourceGroup:
    """One recursive stochastic root and its ordered consumers."""

    index: int
    consumers: tuple[StochasticConsumerPlan, ...]
    execution_mode: StochasticExecutionMode
    exact_size: int | None

    @property
    def arg_refs(self) -> tuple[_workflow_call.WorkflowInputRef, ...]:
        """Return consumer references in canonical argument order."""
        return tuple(consumer.arg_ref for consumer in self.consumers)

    @property
    def stochastic_source_id(self) -> StructuralRngId:
        """Return the structural identity used by the workflow RNG broker."""
        return ("source-group", self.index)


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class PlannedRandomEvent:
    """Derived source/unit identity for one planned random event."""

    stochastic_source_id: StructuralRngId
    logical_unit_id: StructuralRngId


@dataclass(frozen=True, slots=True)
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

        is_batched_record = isinstance(value, RecordBatch)
        is_dist_array = isinstance(value, DistributionArray)
        if (is_batched_record or is_dist_array) and len(value.batch_shape) > 0:
            if _value_matches_hint(value, expected) or expected is Any:
                continue
            array_args.append(ref)
            continue

        if isinstance(value, Distribution):
            if _workflow_distribution_normalization.is_distribution_hint(expected):
                continue
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
    refs: Sequence[_workflow_call.WorkflowInputRef],
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

    source_entries = tuple((ref, _workflow_call.input_ref_value(values, ref)) for ref in refs)
    captured_consumers = _workflow_descendants.capture_stochastic_consumers(
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
    refs: Sequence[_workflow_call.WorkflowInputRef],
) -> list[tuple[Any, tuple[_workflow_call.WorkflowInputRef, ...]]]:
    """Group input references by what aligns them, with each group's root.

    A value with no parent is its own root, so one group holds every reference
    that denotes the same underlying random variable: the same distribution
    passed twice, sibling views of one parent, and a parent passed alongside its
    own view. References with no common root land in separate groups. Groups and
    their members keep argument order, so the grouping is a deterministic
    function of the call.

    A view with a ``parent`` is grouped by a single lookup, which is exact for
    the view types that carry one: such a view's parent is always a distribution,
    never another view. A nested view type would need this
    walked transitively.

    A batch has no parent to look up — a view over one is another batch over the
    same axes — so it is grouped by its **level names** instead. That is the level
    algebra's own rule: operands align by level name, so two batches carrying the
    same level are two readings of one multiplicity and zip, while batches with no
    level in common are independent and form a product. Sibling views from one
    batch's ``select_all`` therefore zip, as do a batch and a view of it.
    """
    groups: dict[Any, tuple[Any, list[_workflow_call.WorkflowInputRef]]] = {}
    for ref in refs:
        value = _workflow_call.input_ref_value(values, ref)
        parent = getattr(value, "parent", None)
        if parent is not None:
            key: Any = id(parent)
            root = parent
        elif isinstance(value, Batch):
            key = value.level_names
            root = value
        else:
            key = id(value)
            root = value
        groups.setdefault(key, (root, []))[1].append(ref)
    return [(root, tuple(group_refs)) for root, group_refs in groups.values()]


def build_array_zip_groups(
    *,
    values: Mapping[str, Any],
    refs: Sequence[_workflow_call.WorkflowInputRef],
) -> tuple[ArrayBroadcastGroup, ...]:
    """Build the zip groups for array-valued sweep arguments.

    Every argument in a group is read along the same axes, so they must agree on
    what those axes are; a disagreement is a mistake about which multiplicity is
    which rather than a product to be formed silently.
    """
    groups: list[ArrayBroadcastGroup] = []
    for _root, arg_refs in group_by_alignment(values=values, refs=refs):
        first = _workflow_call.input_ref_value(values, arg_refs[0])
        batch_shape = tuple(first.batch_shape)
        if isinstance(first, Batch):
            level_names = tuple(first.level_names)
            group_axes = tuple(first.axis_groups)
        else:
            level_names = (arg_refs[0].label,)
            group_axes = (batch_shape,)
        for ref in arg_refs[1:]:
            other = _workflow_call.input_ref_value(values, ref)
            if isinstance(first, Batch):
                # Two operands naming the same levels claim the same axes, group
                # by group: agreeing on the flat shape alone would zip a
                # ((2,), (3, 4)) partition with a ((2, 3), (4,)) one and hand the
                # output whichever arrived first.
                if tuple(getattr(other, "axis_groups", ())) != tuple(first.axis_groups):
                    raise ValueError(
                        f"{arg_refs[0].label!r} and {ref.label!r} carry the same levels but "
                        f"are batched differently: {tuple(first.axis_groups)} against "
                        f"{tuple(getattr(other, 'axis_groups', ()))}. Levels align by name, "
                        f"so operands naming the same levels must hold them on the same axes"
                    )
            elif tuple(other.batch_shape) != batch_shape:
                raise ValueError(
                    f"{arg_refs[0].label!r} and {ref.label!r} are zipped together but are "
                    f"batched differently: {batch_shape} against {tuple(other.batch_shape)}. "
                    f"Arguments sharing a level name are read along the same axes, so give "
                    f"one of them a level of its own to sweep them independently"
                )
        groups.append(
            ArrayBroadcastGroup(
                arg_refs=tuple(arg_refs),
                batch_shape=batch_shape,
                size=prod(batch_shape),
                level_names=level_names,
                axis_groups=group_axes,
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
        first = _workflow_call.input_ref_value(values, group.arg_refs[0])
        if not isinstance(first, Batch):
            # An operand carrying no levels of its own cannot share one: its
            # multiplicity is anonymous, so it aligns with nothing by name and
            # products with everything. Standing the parameter's name in for the
            # levels it does not have would collide with a real level of that name
            # on another operand and refuse a call whose axes are independent.
            continue
        names = tuple(first.level_names)
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
        return (
            isinstance(base, type)
            and issubclass(base, (Batch, DistributionArray))
            and isinstance(value, base)
        )
    except TypeError:
        return False
