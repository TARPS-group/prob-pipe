"""Function sweep execution helpers.

This private module owns array-valued workflow sweeps after call
resolution, distribution normalization, and broadcast planning have
already classified the call. It executes pure parameter sweeps and the
outer sweep layer of nested array + distribution broadcasts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..core._batch import Batch
from ..core._evaluation_points import _requested_evaluation_points
from ..core._expression import Expression
from ..core._numeric_array import NumericArray
from ..core._numeric_array_batch import NumericArrayBatch, _MappedBatchStore
from ..core._object_batch import _ObjectBatch
from ..core._opaque import Opaque
from ..core._record_batch import RecordBatch, _MappedBatchColumns
from ..core._repr import type_name
from ..core._specs import OutputSpec, RecordSpec
from ..core.config import WorkflowKind
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..distributions._distribution import Distribution
from ..values import _binding
from ..values._function_base import _validate_stacked_output
from . import _plan, _recipe, _result
from ._execution import evaluate_requested_points, sweep_cell_point_segment
from ._result import _make_stack, _row_at_its_kind


def execute_sweep(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    plan: _plan.BroadcastPlan,
    stochastic_plan: _plan.StochasticPlan | None,
    swept_point_traceability: Callable[
        [dict[str, Any], list[_binding.FunctionInputRef]], tuple[bool, str | None]
    ],
    distribution_broadcast: Callable[
        [
            dict[str, Any],
            _plan.StochasticPlan,
            _plan.LogicalUnit,
            bool,
        ],
        Distribution,
    ],
    function_name: str,
    output_label: str | None = None,
    output_expression: Expression | None = None,
    output_spec: OutputSpec | None = None,
    include_inputs: bool = False,
    output_template: RecordSpec | None = None,
    provenance_parents: list[TrackedTerm] | None = None,
    provenance_inputs: Mapping[str, Any] | None = None,
    workflow_kind: WorkflowKind = WorkflowKind.OFF,
    route: Mapping[str, Any] | None = None,
) -> Any:
    """Execute pure or nested sweep regimes for one workflow call.

    *swept_point_traceability* says whether JAX can trace one cell of a pure
    sweep, given the values and the swept arguments, with the reason when it
    cannot. *route* is the selected route's name and exactness, which
    provenance records. The aggregate is labeled *output_label* and carries
    *output_expression*, the expression the call gives its result, when one
    is given.
    """
    if plan.regime not in ("sweep", "nested"):
        raise ValueError(f"execute_sweep requires a sweep plan; got {plan.regime!r}")

    output_label = function_name if output_label is None else output_label
    array_args = list(plan.array_args)
    dist_args = list(plan.dist_args)

    if include_inputs:
        raise NotImplementedError(
            "include_inputs=True is not supported when calling over a Batch; the inputs "
            "are recorded in the result's provenance"
        )

    if not dist_args:
        per_row = execute_sweep_rows(
            func=func,
            values=values,
            array_args=array_args,
            plan=plan,
            swept_point_traceability=swept_point_traceability,
            workflow_kind=workflow_kind,
            function_name=function_name,
            output_is_declared=output_spec is not None and output_spec.term_spec is not None,
            output_label=output_label,
        )
        if output_spec is not None:
            output_spec = _result._aggregate_output_spec(output_spec, per_row)
            output_template = _result._output_record_spec(output_spec)
        aggregate = _make_stack(
            per_row,
            batch_shape=plan.sweep_batch_shape,
            # The aggregate mints the levels the sweep ranged over, so it aligns
            # by name with the batch it swept.
            level_names=plan.sweep_level_names,
            axis_groups=plan.sweep_axis_groups,
            label=output_label,
            output_spec=output_spec,
            output_template=output_template,
        )
        try:
            _validate_stacked_output(
                function_name=function_name, output_spec=output_spec, batch=aggregate
            )
        except ValueError as error:
            raise _result.ResultSchemaError(str(error)) from error
        provenance = make_sweep_provenance(
            values=values,
            array_args=array_args,
            dist_args=dist_args,
            function_name=function_name,
            batch_shape=plan.sweep_batch_shape,
            max_evaluations=None,
            parents=provenance_parents,
            inputs=provenance_inputs,
            stochastic_plan=None,
            route=route,
        )
        return _result._coerce_output(
            aggregate,
            broadcast_mode=_result.BROADCAST_STACK,
            provenance=provenance,
            label=output_label,
            expression=output_expression,
        )

    if stochastic_plan is None:  # pragma: no cover - Function planning contract guard
        raise RuntimeError("nested sweep is missing its stochastic plan")

    per_row_laws: list[Distribution] = []
    for logical_unit in stochastic_plan.logical_units:
        row_values = slice_sweep_values(
            values=values,
            index=logical_unit.flat_index,
            array_groups=plan.array_groups,
        )
        per_row_laws.append(
            distribution_broadcast(row_values, stochastic_plan, logical_unit, False)
        )

    stacked = _make_stack(
        per_row_laws,
        batch_shape=plan.sweep_batch_shape,
        level_names=plan.sweep_level_names,
        axis_groups=plan.sweep_axis_groups,
        label=output_label,
    )
    provenance = make_sweep_provenance(
        values=values,
        array_args=array_args,
        dist_args=dist_args,
        function_name=function_name,
        batch_shape=plan.sweep_batch_shape,
        max_evaluations=stochastic_plan.max_evaluations,
        parents=provenance_parents,
        inputs=provenance_inputs,
        stochastic_plan=stochastic_plan,
        route=route,
    )
    return _result._coerce_output(
        stacked,
        broadcast_mode=_result.BROADCAST_NESTED,
        provenance=provenance,
        label=output_label,
        expression=output_expression,
    )


def slice_sweep_values(
    *,
    values: Mapping[str, Any],
    index: int,
    array_groups: tuple[_plan.ArrayBroadcastGroup, ...],
) -> dict[str, Any]:
    """Read one row-major sweep cell, expanding singleton axes within each level.

    Each input uses zero on a size-one axis and the cell's coordinate otherwise.
    Shapes align from the right within each named level, so an input with fewer
    axes in a level reads that level's trailing coordinates. Each input is
    indexed in place at its own shape.
    """
    out = dict(values)
    rem = index
    # Highest-index group varies fastest under row-major flattening of
    # the concatenated sweep shape.
    for group in reversed(array_groups):
        idx = rem % group.size
        rem = rem // group.size
        coordinates = (
            (idx,)
            if len(group.batch_shape) == 1
            else tuple(int(i) for i in np.unravel_index(idx, group.batch_shape))
        )
        replacements: dict[_binding.FunctionInputRef, Any] = {}
        for ref in group.arg_refs:
            batch = _binding.input_ref_value(values, ref)
            if batch.axis_groups == group.axis_groups:
                replacements[ref] = batch[coordinates]
                continue
            position = []
            offset = 0
            for shape, target_shape in zip(batch.axis_groups, group.axis_groups, strict=True):
                # The input's axes for this level match the trailing axes of the
                # level's broadcast shape, so an omitted leading axis broadcasts
                # like size one.
                start = offset + len(target_shape) - len(shape)
                position.extend(
                    0 if size == 1 else coordinates[start + axis] for axis, size in enumerate(shape)
                )
                offset += len(target_shape)
            replacements[ref] = batch[tuple(position)]
        out = _binding.replace_input_refs(out, replacements)
    return out


def execute_sweep_rows(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    array_args: list[_binding.FunctionInputRef],
    plan: _plan.BroadcastPlan,
    swept_point_traceability: Callable[
        [dict[str, Any], list[_binding.FunctionInputRef]], tuple[bool, str | None]
    ],
    workflow_kind: WorkflowKind = WorkflowKind.OFF,
    function_name: str,
    output_is_declared: bool = False,
    output_label: str,
) -> Any:
    """The result of each cell of a pure sweep, from one set of evaluation points, one per cell.

    A sweep that JAX can trace, of one group of batches of arrays or records,
    reads each cell from the batches' storage, through the body the probe
    traces, which ``jax`` maps over every cell at once. Any other sweep reads
    each cell by its coordinates, one cell at a time, as for a batch of
    objects or for batches on different levels. It returns the cells' results
    in row-major order, as a list, or stacked on a leading axis by ``jax``.
    """
    # Zero cells run nothing, so there is no body to trace and no per-cell
    # output to disagree over: every executor takes the same empty aggregation,
    # which keeps the output schema independent of how the cells would have run.
    if plan.n_sweep == 0:
        return []
    sweep_batch_shape = plan.sweep_batch_shape

    def point_segment(cell_index: int) -> Any:
        return sweep_cell_point_segment(
            tuple(int(axis_index) for axis_index in np.unravel_index(cell_index, sweep_batch_shape))
        )

    swept_objects = any(
        isinstance(_binding.input_ref_value(values, ref), _ObjectBatch) for ref in array_args
    )
    if swept_objects:
        traceable, untraceable_reason = False, "it sweeps a batch of objects"
    elif len(plan.array_groups) > 1:
        traceable, untraceable_reason = False, "it sweeps batches on different levels"
    else:
        traceable, untraceable_reason = swept_point_traceability(values, array_args)
    if not traceable:
        evaluation_points = _requested_evaluation_points(
            lambda cell_index: func(
                **slice_sweep_values(
                    values=values, index=int(cell_index), array_groups=plan.array_groups
                )
            ),
            varying={"cell_index": np.arange(plan.n_sweep)},
            origin=function_name,
            traceable=False,
            untraceable_reason=untraceable_reason,
            point_noun="grid point",
            point_segment=point_segment,
        )
        return list(
            evaluate_requested_points(
                evaluation_points,
                stacked=False,
                workflow_kind=workflow_kind,
                function_label=function_name,
                stochastic_plan=None,
            )
        )
    evaluation_points = _requested_evaluation_points(
        mapped_row_body(
            func=func,
            values=values,
            array_args=array_args,
            label=output_label,
            output_is_declared=output_is_declared,
        ),
        varying={
            "storage_rows": tuple(
                mapped_storage(
                    _binding.input_ref_value(values, ref),
                    plan.n_sweep,
                    axis_groups=plan.array_groups[0].axis_groups,
                )
                for ref in array_args
            )
        },
        origin=function_name,
        traceable=True,
        untraceable_reason=None,
        point_noun="grid point",
        point_segment=point_segment,
    )
    return evaluate_requested_points(
        evaluation_points,
        stacked=True,
        workflow_kind=workflow_kind,
        function_label=function_name,
        stochastic_plan=None,
    )


def mapped_storage(
    batch: Batch, n_rows: int, *, axis_groups: tuple[tuple[int, ...], ...] | None = None
) -> Any:
    """The raw storage the map reads from the swept *batch*, as *n_rows* rows on one leading axis.

    The batch axes are flattened in row-major order, which is the order of the
    sweep's cells. A batch of arrays is read as its store, and a batch of records
    as one column per field, keyed by the field's path. When *axis_groups* is
    given, each named level expands to its target shape before flattening.
    Missing leading axes align within each level, preserving its boundaries.

    Parameters
    ----------
    batch : Batch
        The swept batch.
    n_rows : int
        The number of rows, which is the product of the target batch shape.
    axis_groups : tuple of tuple of int or None
        Target shapes in the order of *batch*'s levels. Defaults to the batch's
        own shapes. Each target shape must broadcast from its level's shape.

    Returns
    -------
    jax.Array or dict of str to jax.Array
        The store or the columns by path, as JAX arrays whose leading axis of
        length *n_rows* precedes the element's axes.

    Raises
    ------
    TypeError
        If *batch* stores objects, as a batch of laws or of functions does.
    ValueError
        If the target has a different number of levels, a level cannot broadcast
        to its target shape, or *n_rows* differs from the target's total size.
    """
    n_batch = len(batch.batch_shape)
    target_groups = batch.axis_groups if axis_groups is None else axis_groups
    aligned_shape = []
    for shape, target in zip(batch.axis_groups, target_groups, strict=True):
        if len(target) < len(shape) or np.broadcast_shapes(shape, target) != target:
            raise ValueError(f"cannot broadcast level shape {shape} to {target}")
        aligned_shape.extend((1,) * (len(target) - len(shape)) + shape)
    target_shape = tuple(axis for shape in target_groups for axis in shape)
    if n_rows != prod(target_shape):
        raise ValueError(f"row count {n_rows} differs from target shape {target_shape}")

    def rows(column: Any) -> Any:
        column = jnp.asarray(column)
        event_shape = jnp.shape(column)[n_batch:]
        column = jnp.reshape(column, (*aligned_shape, *event_shape))
        column = jnp.broadcast_to(column, (*target_shape, *event_shape))
        # The row count is stated, since ``-1`` cannot be inferred over a
        # zero-width event.
        return jnp.reshape(column, (n_rows, *event_shape))

    if isinstance(batch, NumericArrayBatch):
        return rows(batch.as_jax())
    if isinstance(batch, RecordBatch):
        return {path: rows(batch._raw_column(path)) for path in batch.event_template}
    raise TypeError(f"a {type(batch).__name__} stores objects, which a mapped sweep cannot read")


def _mapped_row(batch: Batch, label: str, row: Any) -> NumericArray | Record:
    """One element of the swept *batch*, rebuilt from the raw *row* the map reads.

    The element is labeled *label*. A batch of arrays yields an array under the
    spec of the batch's elements, and a batch of records a record of the row's
    fields, in the batch's order. JAX maps a dict with its keys sorted, so the
    row's own order is not the batch's.
    """
    if isinstance(batch, NumericArrayBatch):
        return NumericArray(row, spec=batch.element_spec.term_spec, label=label)
    return Record({path: row[path] for path in batch.event_template}, label=label)


def mapped_row_body(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    array_args: Sequence[_binding.FunctionInputRef],
    label: str,
    output_is_declared: bool = False,
) -> Callable[..., Any]:
    """The body of one cell of a sweep, which every executor runs and the probe traces.

    Both callers use this rather than each building its own: the probe's job is
    to trace exactly what ``jax`` maps, which two separately maintained
    functions cannot promise.

    *values* holds each swept batch at its reference in *array_args*, and the
    body receives one row of each batch's :func:`mapped_storage`. Inside the
    traced call each row is rebuilt at the kind of its batch's elements, which
    is an array or a record, so nothing infers a batch axis on the way in. On
    the way out the row takes the kind of its own return, as a cell evaluated
    alone does, and a numeric array, record, or batch is then taken apart into
    :class:`_MappedBatchColumns` or :class:`_MappedBatchStore`: the map is about
    to add an axis that neither class's unflatten hook could name, and the
    executor names it afterwards from the sweep's levels.

    *output_is_declared* says *func* already gave the row a declared template, in
    which case that template is the row's kind and nothing here re-derives one.
    """

    def one_row(storage_rows: tuple[Any, ...]) -> Any:
        replacements = {
            ref: _mapped_row(_binding.input_ref_value(values, ref), ref.label, row)
            for ref, row in zip(array_args, storage_rows)
        }
        out = func(**_binding.replace_input_refs(values, replacements))
        if not output_is_declared:
            out = _row_at_its_kind(out, label)
            if isinstance(out, Record):
                # As a batch row is carried, but with no level of its own: the
                # axis the map adds is the only one the aggregate will have.
                return _MappedBatchColumns.of_record(out)
            if isinstance(out, Opaque):
                raise TypeError(
                    f"{label}: jax.vmap cannot stack a row that returned "
                    f"{type_name(out.raw)}; it stacks only arrays, records, and batches"
                )
        if isinstance(out, RecordBatch):
            return _MappedBatchColumns.of(out)
        if isinstance(out, (NumericArray, NumericArrayBatch)):
            return _MappedBatchStore.of(out)
        return out

    return one_row


def make_sweep_provenance(
    *,
    values: Mapping[str, Any],
    array_args: list[_binding.FunctionInputRef],
    dist_args: list[_binding.FunctionInputRef],
    function_name: str,
    batch_shape: tuple[int, ...],
    max_evaluations: int | None,
    parents: list[TrackedTerm] | None = None,
    inputs: Mapping[str, Any] | None = None,
    stochastic_plan: _plan.StochasticPlan | None = None,
    route: Mapping[str, Any] | None = None,
) -> Provenance | None:
    """Build provenance metadata for pure and nested sweep outputs.

    ``parents`` carries tracked call-level lineage; ``inputs`` carries the
    original resolved plain values rather than per-cell sweep values. A nested
    sweep's metadata records the ``max_evaluations`` of the lift in each cell,
    which a pure sweep, given ``None``, does not have.
    Returns ``None`` when :attr:`ProvenanceMode.OFF` is active.
    """
    regime = "nested" if dist_args else "stack"
    if parents is None:
        array_candidates = [_binding.input_ref_value(values, ref) for ref in array_args]
        dist_candidates = [
            _binding.input_ref_value(values, ref)
            for ref in dist_args
            if isinstance(_binding.input_ref_value(values, ref), Distribution)
        ]
        parents = array_candidates + dist_candidates
    controls, diagnostics = _recipe.provenance_recipe_fields(stochastic_plan)
    return Provenance.create(
        f"workflow.{regime}",
        parents=parents,
        metadata={
            "func": function_name,
            "batch_shape": tuple(batch_shape),
            **({} if max_evaluations is None else {"max_evaluations": max_evaluations}),
            "ra_args": [ref.label for ref in array_args],
            "dist_args": [ref.label for ref in dist_args],
            **(dict(route) if route is not None else {}),
        },
        inputs=inputs,
        controls=controls,
        diagnostics=diagnostics,
    )
