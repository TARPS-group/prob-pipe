"""Function broadcast-planning helpers.

This private module classifies already-normalized workflow inputs into
the broadcast regime and sweep shape that ``Function`` should
execute. Planning is intentionally side-effect-free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, Literal

from . import _workflow_call, _workflow_distribution_normalization
from ._distribution_array import DistributionArray
from ._record_array import RecordArray
from .distribution import Distribution

BroadcastRegime = Literal["none", "distribution", "sweep", "nested"]


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

    A root is found by a single ``parent`` lookup, which is exact for the view
    types that exist: a view's parent is always a distribution, never another
    view. A nested view type would need this walked transitively.
    """
    groups: dict[int, tuple[Any, list[_workflow_call.WorkflowInputRef]]] = {}
    for ref in refs:
        value = _workflow_call.input_ref_value(values, ref)
        root = getattr(value, "parent", value)
        groups.setdefault(id(root), (root, []))[1].append(ref)
    return [(root, tuple(group_refs)) for root, group_refs in groups.values()]


def group_array_args_by_parent(
    *,
    values: Mapping[str, Any],
    refs: Sequence[_workflow_call.WorkflowInputRef],
) -> tuple[ArrayBroadcastGroup, ...]:
    """Build parent-identity groups for array-valued sweep arguments."""
    groups: list[ArrayBroadcastGroup] = []
    for _root, arg_refs in group_by_alignment(values=values, refs=refs):
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
