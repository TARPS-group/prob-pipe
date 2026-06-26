"""WorkflowFunction broadcast-planning helpers.

This private module classifies already-normalized workflow inputs into
the broadcast regime and sweep shape that ``WorkflowFunction`` should
execute. Planning is intentionally side-effect-free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, Literal

from . import _workflow_distribution_normalization
from ._distribution_array import DistributionArray
from ._record_array import RecordArray
from .distribution import Distribution

BroadcastRegime = Literal["none", "distribution", "sweep", "nested"]


@dataclass(frozen=True)
class ArrayBroadcastGroup:
    """One parent-identity group of array-valued sweep arguments."""

    arg_names: tuple[str, ...]
    batch_shape: tuple[int, ...]
    size: int


@dataclass(frozen=True)
class BroadcastPlan:
    """Pure broadcast classification for one resolved workflow call."""

    regime: BroadcastRegime
    dist_args: tuple[str, ...]
    array_args: tuple[str, ...]
    array_groups: tuple[ArrayBroadcastGroup, ...]
    sweep_batch_shape: tuple[int, ...]
    n_sweep: int


def build_broadcast_plan(
    *,
    values: Mapping[str, Any],
    hints: Mapping[str, Any],
) -> BroadcastPlan:
    """Classify normalized values into a broadcast execution plan."""
    dist_args: list[str] = []
    array_args: list[str] = []

    for name, value in values.items():
        expected = hints.get(name)

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
            array_args.append(name)
            continue

        if isinstance(value, Distribution):
            if _workflow_distribution_normalization.is_distribution_hint(expected):
                continue
            dist_args.append(name)

    array_groups = group_array_args_by_parent(values=values, names=array_args)
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


def group_by_parent(
    *,
    values: Mapping[str, Any],
    names: Sequence[str],
) -> dict[int, list[str]]:
    """Group argument names by the identity of their source parent."""
    groups: dict[int, list[str]] = {}
    for name in names:
        value = values[name]
        parent = getattr(value, "parent", value)
        groups.setdefault(id(parent), []).append(name)
    return groups


def group_array_args_by_parent(
    *,
    values: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[ArrayBroadcastGroup, ...]:
    """Build parent-identity groups for array-valued sweep arguments."""
    groups: list[ArrayBroadcastGroup] = []
    for arg_names in group_by_parent(values=values, names=names).values():
        batch_shape = tuple(values[arg_names[0]].batch_shape)
        groups.append(
            ArrayBroadcastGroup(
                arg_names=tuple(arg_names),
                batch_shape=batch_shape,
                size=prod(batch_shape),
            )
        )
    return tuple(groups)


def _broadcast_regime(
    *,
    dist_args: Sequence[str],
    array_args: Sequence[str],
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
