"""Function broadcast-planning helpers.

This private module classifies already-normalized workflow inputs into
the broadcast regime and sweep shape that ``Function`` should
execute. Planning is intentionally side-effect-free.

An argument at a parameter with a role, as an operation's are, lifts by the
role: it passes whole when the role admits its kind, and is otherwise swept or
broadcast over the elements or draws the role admits (V.5, VI.0). Every other
argument lifts as its parameter's annotation states.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product as cartesian_product
from math import prod
from typing import Any, Literal

import numpy as np

from ..core._batch import Batch
from ..core._spec_base import TermSpec
from ..core._specs import OutputSpec
from ..core.config import _checked_count
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution
from ..values import _binding
from . import _descendants, _normalization
from ._call import ApplicabilityError, arrived_kind, role_admits, role_lift

BroadcastRegime = Literal["none", "distribution", "sweep", "nested"]
Lift = Literal["whole", "sweep", "broadcast"]
StochasticExecutionMode = Literal["exact", "sampled"]
StochasticEvaluationMode = Literal["exact", "sampled", "mixed_exact_sampled"]
LogicalUnitLayout = Literal["singleton", "canonical_sweep"]
StructuralRngId = tuple[str | int, ...]


@dataclass(frozen=True)
class ArrayBroadcastGroup:
    """One zip group of swept batches.

    The batches of one group carry the same level names, since levels are how
    batches align. ``axis_groups`` holds each level's broadcast shape over the
    group's batches, and ``batch_shape`` is their concatenation in level order.
    """

    arg_refs: tuple[_binding.FunctionInputRef, ...]
    batch_shape: tuple[int, ...]
    size: int
    # What the group's axes range over, for the aggregate to mint its levels
    # under: the batches' own level names are what an output must carry to align
    # with them.
    level_names: tuple[str, ...]
    axis_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BroadcastPlan:
    """Pure broadcast classification for one resolved workflow call.

    ``batch_args`` holds the broadcast arguments whose parameter's annotation
    names a batch class, as :func:`receives_batch` states. A batch-valued draw
    is passed to such a parameter as a batch, and to any other in its raw form.
    """

    regime: BroadcastRegime
    dist_args: tuple[_binding.FunctionInputRef, ...]
    array_args: tuple[_binding.FunctionInputRef, ...]
    array_groups: tuple[ArrayBroadcastGroup, ...]
    sweep_batch_shape: tuple[int, ...]
    sweep_level_names: tuple[str, ...]
    sweep_axis_groups: tuple[tuple[int, ...], ...]
    n_sweep: int
    batch_args: tuple[_binding.FunctionInputRef, ...] = ()


@dataclass(frozen=True)
class StochasticConsumerPlan:
    """Canonical projection of one argument from a co-sampled root."""

    arg_ref: _binding.FunctionInputRef
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
    def arg_refs(self) -> tuple[_binding.FunctionInputRef, ...]:
        """Return consumer references in canonical argument order."""
        return tuple(consumer.arg_ref for consumer in self.consumers)

    @property
    def stochastic_source_id(self) -> StructuralRngId:
        """Return the structural identity used by the workflow RNG broker."""
        return ("source-group", self.index)


@dataclass(frozen=True)
class StochasticRuntimeBinding:
    """Live root and preflight-captured evaluators for one source group.

    ``consumer_events`` holds the event type of each consumer, in the order of
    ``consumer_evaluators``, and ``consumer_batches`` whether each consumer's
    parameter receives a batch-valued draw as a batch rather than in its raw
    form.
    """

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
    consumer_events: tuple[TermSpec, ...] = field(
        compare=False,
        hash=False,
        repr=False,
    )
    consumer_batches: tuple[bool, ...] = field(
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
    arg_refs: tuple[_binding.FunctionInputRef, ...]
    source_groups: tuple[StochasticSourceGroup, ...]
    logical_units: tuple[LogicalUnit, ...]
    max_evaluations: int
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
    return _is_batched(value) and not (
        _value_matches_hint(value, expected) or _binding.resolved_hint(expected) is Any
    )


def is_broadcast(value: Any, expected: Any) -> bool:
    """Whether the call samples *value* at a parameter whose lifting annotation is *expected*.

    A distribution is broadcast unless it is batched, in which case the call
    sweeps it or passes it whole, or the annotation names a distribution, which
    consumes it, or is ``Any``, which accepts every kind as it arrives.
    """
    return (
        isinstance(value, Distribution)
        and not _is_batched(value)
        and _binding.resolved_hint(expected) is not Any
        and not _normalization.is_distribution_hint(expected)
    )


def lift_of(value: Any, expected: Any, role: tuple[type[TermSpec], ...] | None = None) -> Lift:
    """How a call passes *value* at a parameter annotated *expected* whose role is *role*.

    Without a role the annotation decides, as :func:`is_swept` and
    :func:`is_broadcast` state. With one, *value* passes whole when the role
    admits its kind, names no kind, or the annotation is ``Any``; otherwise a
    batch over an admitted kind is swept and a law over an admitted event kind
    is broadcast. A value no lift admits passes whole, for admission to refuse.
    """
    if role is None:
        if is_swept(value, expected):
            return "sweep"
        return "broadcast" if is_broadcast(value, expected) else "whole"
    if (
        not role
        or _binding.resolved_hint(expected) is Any
        or role_admits(role, arrived_kind(value))
    ):
        return "whole"
    return role_lift(value, role) or "whole"


def lift_at(function: Any, parameter: str | None, value: Any) -> Lift:
    """How a call of *function* passes *value* at *parameter*, as :func:`lift_of` states.

    The parameter's annotation and role are the function's; with no parameter
    the value lifts as at an unannotated one.
    """
    if parameter is None:
        return lift_of(value, None)
    hint = _binding.parameter_lifting_hint(function._signature_info, parameter)
    return lift_of(value, hint, function._roles.get(parameter))


def receives_batch(expected: Any) -> bool:
    """Whether a lifted parameter annotated *expected* receives a batch-valued draw as a batch.

    The annotation names a batch class, such as ``NumericArrayBatch``,
    ``RecordBatch``, or ``Batch``, in itself or in an arm of a union. Any
    other annotation, or none, receives the draw in its raw form: an array, or
    the nested mapping of columns of a batch of records.
    """
    for arm in _normalization._arms(expected):
        named = _normalization._hint_class(arm)
        if isinstance(named, type) and issubclass(named, Batch):
            return True
    return False


def build_broadcast_plan(
    *,
    values: Mapping[str, Any],
    signature_info: _binding.FunctionSignatureInfo,
    roles: Mapping[str, tuple[type[TermSpec], ...]] | None = None,
) -> BroadcastPlan:
    """Classify normalized values into a broadcast execution plan.

    Each argument lifts as :func:`lift_of` states, by its parameter's role in
    *roles* where it has one.
    """
    dist_args: list[_binding.FunctionInputRef] = []
    array_args: list[_binding.FunctionInputRef] = []
    batch_args: list[_binding.FunctionInputRef] = []

    for ref in _binding.iter_input_refs(signature_info, values):
        value = _binding.input_ref_value(values, ref)
        expected = _binding.input_ref_hint(signature_info, ref)
        lift = lift_of(value, expected, (roles or {}).get(ref.parameter_name))
        if lift == "sweep":
            array_args.append(ref)
        elif lift == "broadcast":
            dist_args.append(ref)
            if receives_batch(expected):
                batch_args.append(ref)

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
        batch_args=tuple(batch_args),
    )


def require_empty_sweep_output(plan: BroadcastPlan, output_spec: OutputSpec | None) -> None:
    """Require a complete output declaration when a sweep has no evaluations.

    Parameters
    ----------
    plan : BroadcastPlan
        The call's lifting regime and sweep shape.
    output_spec : OutputSpec or None
        The output declaration after dimensions shared with inputs bind.

    Raises
    ------
    ValueError
        If an empty sweep has no output type or leaves an output dimension
        unbound, since no returned value can complete its declaration.
    """
    if plan.regime in ("sweep", "nested") and plan.n_sweep == 0:
        if output_spec is None or output_spec.spec is None or not output_spec.spec.is_concrete:
            raise ValueError("An empty sweep requires a concrete output_spec")


def build_stochastic_plan(
    values: Mapping[str, Any],
    broadcast_plan: BroadcastPlan,
    max_evaluations: int,
    *,
    enumerate_every_group: bool = True,
) -> StochasticPlan | None:
    """Build immutable stochastic decisions without claiming random events.

    A co-sampling group whose root is an empirical law is enumerated when its
    atoms, multiplied by the other enumerated groups', number at most
    *max_evaluations*, smallest groups first. The other groups are sampled:
    *max_evaluations* joint draws when no group enumerates, and otherwise
    ``max_evaluations // n`` draws, at least one, for each of the ``n``
    enumerated combinations. The plan therefore evaluates the function at most
    *max_evaluations* times. Without *enumerate_every_group* a plan in which
    every group would enumerate samples every group instead, as the sampling
    lift does when it is selected by name over the exact enumeration.
    """
    if broadcast_plan.regime in ("none", "sweep"):
        return None

    _checked_count(max_evaluations, "max_evaluations")
    arg_refs = tuple(broadcast_plan.dist_args)
    grouped_consumers, source_values, runtime_samplers, runtime_evaluators, runtime_events = (
        _group_stochastic_sources(
            values=values,
            refs=arg_refs,
        )
    )
    batch_args = set(broadcast_plan.batch_args)

    candidates = [
        (index, source)
        for index, source in enumerate(source_values)
        if isinstance(source, EmpiricalDistribution) and source.num_atoms <= max_evaluations
    ]
    candidates.sort(key=lambda pair: pair[1].num_atoms)

    exact_group_indices: list[int] = []
    exact_sizes: dict[int, int] = {}
    exact_product = 1
    for index, source in candidates:
        size = source.num_atoms
        if exact_product * size <= max_evaluations:
            exact_group_indices.append(index)
            exact_sizes[index] = size
            exact_product *= size

    if not enumerate_every_group and len(exact_group_indices) == len(source_values):
        exact_group_indices, exact_sizes, exact_product = [], {}, 1

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
            consumer_events=tuple(runtime_events[index]),
            consumer_batches=tuple(
                consumer.arg_ref in batch_args for consumer in grouped_consumers[index]
            ),
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
        repetitions_per_combination = max(1, max_evaluations // exact_product)
        n_evaluations = exact_product * repetitions_per_combination
        sample_shape = (n_evaluations,)
    else:
        evaluation_mode = "sampled"
        repetitions_per_combination = max_evaluations
        n_evaluations = max_evaluations
        sample_shape = (max_evaluations,)

    return StochasticPlan(
        evaluation_mode=evaluation_mode,
        arg_refs=arg_refs,
        source_groups=source_groups,
        logical_units=logical_units,
        max_evaluations=max_evaluations,
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
    refs: Sequence[_binding.FunctionInputRef],
) -> tuple[
    list[list[StochasticConsumerPlan]],
    list[Distribution],
    list[Callable[[Any, tuple[int, ...]], Any]],
    list[list[Callable[[Any], Any]]],
    list[list[TermSpec]],
]:
    """Discover live roots while keeping object IDs out of canonical plans."""
    grouped_consumers: list[list[StochasticConsumerPlan]] = []
    source_values: list[Distribution] = []
    runtime_samplers: list[Callable[[Any, tuple[int, ...]], Any]] = []
    runtime_evaluators: list[list[Callable[[Any], Any]]] = []
    runtime_events: list[list[TermSpec]] = []
    group_index_by_root_id: dict[int, int] = {}

    source_entries = tuple((ref, _binding.input_ref_value(values, ref)) for ref in refs)
    captured_consumers = _descendants.capture_stochastic_consumers(
        tuple(value for _ref, value in source_entries)
    )

    for (ref, value), captured in zip(source_entries, captured_consumers, strict=True):
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
            runtime_events.append([])
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
        runtime_events[group_index].append(value.event_spec.spec)

    return grouped_consumers, source_values, runtime_samplers, runtime_evaluators, runtime_events


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


def group_by_alignment(
    *,
    values: Mapping[str, Any],
    refs: Sequence[_binding.FunctionInputRef],
) -> list[tuple[Any, tuple[_binding.FunctionInputRef, ...]]]:
    """Group the swept batches by their level names, with each group's first batch.

    Batches align by level name, so two batches carrying the same levels are
    two readings of one multiplicity and zip, while batches with no level in
    common are independent and form a product. Sibling views from one batch's
    ``select_all`` therefore zip, as do a batch and a view of it. Groups and
    their members keep argument order, so the grouping is a deterministic
    function of the call.
    """
    groups: dict[tuple[str, ...], tuple[Any, list[_binding.FunctionInputRef]]] = {}
    for ref in refs:
        value = _binding.input_ref_value(values, ref)
        groups.setdefault(tuple(value.level_names), (value, []))[1].append(ref)
    return [(root, tuple(group_refs)) for root, group_refs in groups.values()]


def build_array_zip_groups(
    *,
    values: Mapping[str, Any],
    refs: Sequence[_binding.FunctionInputRef],
) -> tuple[ArrayBroadcastGroup, ...]:
    """Build the zip groups of the swept batches.

    Batches with the same level names broadcast their shapes level by level by
    NumPy's rule, so within each level the shapes align from the right and a
    size-one axis expands. The group's shape concatenates the broadcast shapes
    in level order, and the sweep reads each batch at index zero along its
    size-one axes.

    Parameters
    ----------
    values : Mapping of str to Any
        The call's arguments, by parameter name.
    refs : sequence of FunctionInputRef
        The references to the swept batches, in argument order.

    Returns
    -------
    tuple of ArrayBroadcastGroup
        One group per tuple of level names, in argument order.

    Raises
    ------
    ApplicabilityError
        If matching levels have shapes that cannot broadcast, or two groups
        share a level without sharing all their levels, naming the level.
    """
    groups: list[ArrayBroadcastGroup] = []
    for first, arg_refs in group_by_alignment(values=values, refs=refs):
        batches = [_binding.input_ref_value(values, ref) for ref in arg_refs]
        broadcast_groups = []
        for level, name in enumerate(first.level_names):
            shapes = [tuple(batch.axis_groups[level]) for batch in batches]
            try:
                broadcast_groups.append(np.broadcast_shapes(*shapes))
            except ValueError as error:
                described = [
                    f"{shape} for {ref.label!r}"
                    for ref, shape in zip(arg_refs, shapes, strict=True)
                ]
                listed = (
                    " and ".join(described)
                    if len(described) == 2
                    else ", ".join(described[:-1]) + ", and " + described[-1]
                )
                raise ApplicabilityError(
                    f"level {name!r} has shapes {listed}, which do not broadcast"
                ) from error
        axis_groups = tuple(broadcast_groups)
        batch_shape = tuple(axis for shape in axis_groups for axis in shape)
        groups.append(
            ArrayBroadcastGroup(
                arg_refs=tuple(arg_refs),
                batch_shape=batch_shape,
                size=prod(batch_shape),
                level_names=tuple(first.level_names),
                axis_groups=axis_groups,
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
                raise ApplicabilityError(
                    f"{prior[0]!r} has levels {prior[1]} and {group.arg_refs[0].label!r} has "
                    f"levels {names}, but arguments that share a level must have the same "
                    f"levels. Rename {level_name!r} on one of them with with_level_names to "
                    f"sweep them independently, or give both the same levels to pair them."
                )
    return tuple(groups)


def _broadcast_regime(
    *,
    dist_args: Sequence[_binding.FunctionInputRef],
    array_args: Sequence[_binding.FunctionInputRef],
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
    # An optional container annotation still names the container, and the value
    # answers whichever arm it satisfies.
    for arm in _normalization._arms(expected):
        base = _normalization._hint_class(arm)
        try:
            if isinstance(base, type) and issubclass(base, Batch) and isinstance(value, base):
                return True
        except TypeError:
            continue
    return False
