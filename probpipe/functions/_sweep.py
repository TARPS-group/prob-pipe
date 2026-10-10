"""Function sweep execution helpers.

This private module owns array-valued workflow sweeps after call
resolution, distribution normalization, and broadcast planning have
already classified the call. It executes pure parameter sweeps and the
outer sweep layer of nested array + distribution broadcasts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product as cartesian_product
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..values import _binding
from ..values._function_base import _validate_stacked_output

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from ..core._batch import Batch
from ..core._expression import Expression
from ..core._numeric_array import NumericArray
from ..core._numeric_array_batch import NumericArrayBatch, _MappedBatchStore
from ..core._object_batch import _ObjectBatch
from ..core._opaque import Opaque
from ..core._record_batch import RecordBatch, _MappedBatchColumns
from ..core._repr import type_name
from ..core._specs import OutputSpec, RecordSpec
from ..core.config import WorkflowKind, prefect_config
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..distributions._distribution import Distribution
from . import _broker, _context, _execution, _execution_contract, _plan, _recipe, _result
from ._result import _make_stack, _row_at_its_kind


def execute_sweep(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    plan: _plan.BroadcastPlan,
    stochastic_plan: _plan.StochasticPlan | None,
    make_execution_config: Callable[
        [],
        _execution.WorkflowExecutionConfig,
    ],
    requested_dispatch: str,
    resolve_dispatch: Callable[..., str],
    require_jax_traceable: Callable[[dict[str, Any], list[_binding.FunctionInputRef]], None],
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

    *route* is the selected route's name and exactness, which provenance
    records. The aggregate is labeled *output_label* and carries
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
            make_execution_config=make_execution_config,
            requested_dispatch=requested_dispatch,
            resolve_dispatch=resolve_dispatch,
            require_jax_traceable=require_jax_traceable,
            workflow_kind=workflow_kind,
            function_name=function_name,
            output_is_declared=output_spec is not None and output_spec.spec is not None,
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
            field_name=output_label,
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
            k=0,
            parents=provenance_parents,
            inputs=provenance_inputs,
            stochastic_plan=None,
            route=route,
        )
        return _result._coerce_output(
            aggregate,
            broadcast_mode=_result.BROADCAST_STACK,
            provenance=provenance,
            field_name=output_label,
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
        field_name=output_label,
    )
    provenance = make_sweep_provenance(
        values=values,
        array_args=array_args,
        dist_args=dist_args,
        function_name=function_name,
        batch_shape=plan.sweep_batch_shape,
        k=stochastic_plan.n_broadcast_samples,
        parents=provenance_parents,
        inputs=provenance_inputs,
        stochastic_plan=stochastic_plan,
        route=route,
    )
    return _result._coerce_output(
        stacked,
        broadcast_mode=_result.BROADCAST_NESTED,
        provenance=provenance,
        field_name=output_label,
        expression=output_expression,
    )


def slice_sweep_values(
    *,
    values: Mapping[str, Any],
    index: int,
    array_groups: tuple[_plan.ArrayBroadcastGroup, ...],
) -> dict[str, Any]:
    """Materialize one row-major sweep cell under the zip groups."""
    out = dict(values)
    rem = index
    # Highest-index group varies fastest under row-major flattening of
    # the concatenated sweep shape.
    for group in reversed(array_groups):
        idx = rem % group.size
        rem = rem // group.size
        # A batch spanning several axes addresses its element by position, one
        # indexer per axis; a flat index would read the leading axis alone and
        # run off its end.
        position: Any = idx
        if len(group.batch_shape) > 1:
            position = tuple(int(i) for i in np.unravel_index(idx, group.batch_shape))
        replacements: dict[_binding.FunctionInputRef, Any] = {}
        for ref in group.arg_refs:
            replacements[ref] = _binding.input_ref_value(values, ref)[position]
        out = _binding.replace_input_refs(out, replacements)
    return out


def execute_sweep_rows(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    array_args: list[_binding.FunctionInputRef],
    plan: _plan.BroadcastPlan,
    make_execution_config: Callable[
        [],
        _execution.WorkflowExecutionConfig,
    ],
    requested_dispatch: str,
    resolve_dispatch: Callable[..., str],
    require_jax_traceable: Callable[[dict[str, Any], list[_binding.FunctionInputRef]], None],
    workflow_kind: WorkflowKind = WorkflowKind.OFF,
    function_name: str,
    output_is_declared: bool = False,
    output_label: str,
) -> Any:
    """Execute pure sweep rows through JAX vmap or row-wise execution."""
    # Zero rows run nothing, so there is no body for a dispatch to trace and no
    # per-row output for the paths to disagree over: every dispatch takes the
    # same empty aggregation, which is what keeps the output schema independent
    # of how the rows would have been executed.
    if plan.n_sweep == 0:
        return []

    # A batch of stored objects has no columns for the mapped body to read.
    has_object_batch = any(
        isinstance(_binding.input_ref_value(values, ref), _ObjectBatch) for ref in array_args
    )
    jax_structure_supported = not (
        has_object_batch or len(plan.array_groups) > 1 or len(array_args) > 1
    )
    jax_contract = _execution_contract.make_execution_contract(
        evaluator="jax_vmap",
        transport=_execution_contract.transport_for_workflow_kind(workflow_kind),
        stochastic_plan=None,
    )
    jax_supported = _execution_contract.supports_execution_contract(
        jax_contract,
        None,
        jax_structure_supported=jax_structure_supported,
    )
    if requested_dispatch == "jax" and not jax_supported:
        raise ValueError(
            "dispatch='jax' can only sweep over a single Batch of records or arrays; use "
            "dispatch='auto', 'sequential', or 'thread' for this call"
        )

    dispatch = resolve_dispatch(
        values,
        array_args,
        jax_supported=jax_supported,
    )

    if dispatch == "jax":
        _broker._record_active_execution_contract(jax_contract)
        if requested_dispatch == "jax":
            require_jax_traceable(values, array_args)
        return execute_sweep_rows_jax(
            func=func,
            values=values,
            array_args=array_args,
            n_total=plan.n_sweep,
            workflow_kind=workflow_kind,
            function_name=function_name,
            output_is_declared=output_is_declared,
            output_label=output_label,
        )

    per_row_values = [
        slice_sweep_values(
            values=values,
            index=i,
            array_groups=plan.array_groups,
        )
        for i in range(plan.n_sweep)
    ]
    execution = make_execution_config()
    request = _execution.WorkflowExecutionRequest(
        func=func,
        work_items=_execution.make_managed_work_items(
            per_row_values,
            unit_segments=tuple(
                _execution.sweep_unit_segment(tuple(coordinates))
                for coordinates in cartesian_product(
                    *(range(axis) for axis in plan.sweep_batch_shape)
                )
            ),
        ),
        execution=execution,
        contract=_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport=_execution_contract.transport_for_execution_mode(execution.mode),
            stochastic_plan=None,
        ),
    )
    return _execution.execute_many(request)


def mapped_storage(batch: Batch, n_rows: int) -> Any:
    """The raw storage the map reads from the swept *batch*, as *n_rows* rows on one leading axis.

    The batch axes are flattened in row-major order, which is the order of the
    sweep's cells. A batch of arrays is read as its store, and a batch of records
    as one column per field, keyed by the field's path.

    Parameters
    ----------
    batch : Batch
        The swept batch.
    n_rows : int
        The number of rows, which is the product of *batch*'s batch shape.

    Returns
    -------
    jax.Array or dict of str to jax.Array
        The store or the columns by path, as JAX arrays whose leading axis of
        length *n_rows* precedes the element's axes.

    Raises
    ------
    TypeError
        If *batch* stores objects, as a batch of laws or of functions does.
    """
    n_batch = len(batch.batch_shape)

    def rows(column: Any) -> Any:
        column = jnp.asarray(column)
        # The row count is stated, since ``-1`` cannot be inferred over a
        # zero-width event.
        return jnp.reshape(column, (n_rows, *jnp.shape(column)[n_batch:]))

    if isinstance(batch, NumericArrayBatch):
        return rows(batch.as_jax())
    if isinstance(batch, RecordBatch):
        return {path: rows(batch._raw_column(path)) for path in batch.event_template}
    raise TypeError(f"a {type(batch).__name__} stores objects, which a mapped sweep cannot read")


def _mapped_row(batch: Batch, label: str, row: Any) -> NumericArray | Record:
    """One element of the swept *batch*, rebuilt from the raw *row* the map reads.

    The element is labeled *label*. A batch of arrays yields an array under the
    batch's element spec, and a batch of records a record of the row's fields.
    """
    if isinstance(batch, NumericArrayBatch):
        return NumericArray(label, row, spec=batch.element_spec)
    return Record(label, row)


def mapped_row_body(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    array_args: Sequence[_binding.FunctionInputRef],
    field_name: str,
    output_is_declared: bool = False,
) -> Callable[[Any], Any]:
    """The body ``jax.vmap`` runs for one sweep row, and the probe traces.

    Both callers use this rather than each building its own: the probe's job is
    to trace exactly what the executor runs, which two separately maintained
    functions cannot promise.

    *values* holds each swept batch at its reference in *array_args*, and the
    body receives one row of each batch's :func:`mapped_storage`. Inside the
    traced call each row is rebuilt at the kind of its batch's elements, which
    is an array or a record, so nothing infers a batch axis on the way in. On
    the way out the row takes the kind of its own return, as a row-wise row
    does, and a numeric array, record, or batch is then taken apart into
    :class:`_MappedBatchColumns` or :class:`_MappedBatchStore`: the map is about
    to add an axis that neither class's unflatten hook could name, and the
    executor names it afterwards from the sweep's levels.

    *output_is_declared* says *func* already gave the row a declared template, in
    which case that template is the row's kind and nothing here re-derives one.
    """

    def one_row(storage_rows):
        replacements = {
            ref: _mapped_row(_binding.input_ref_value(values, ref), ref.label, row)
            for ref, row in zip(array_args, storage_rows)
        }
        out = func(**_binding.replace_input_refs(values, replacements))
        if not output_is_declared:
            out = _row_at_its_kind(out, field_name)
            if isinstance(out, Record):
                # As a batch row is carried, but with no level of its own: the
                # axis the map adds is the only one the aggregate will have.
                return _MappedBatchColumns.of_record(out)
            if isinstance(out, Opaque):
                raise TypeError(
                    f"{field_name}: dispatch='jax' cannot stack a row that returned "
                    f"{type_name(out.value)}; it stacks only arrays, records, and batches"
                )
        if isinstance(out, RecordBatch):
            return _MappedBatchColumns.of(out)
        if isinstance(out, (NumericArray, NumericArrayBatch)):
            return _MappedBatchStore.of(out)
        return out

    return one_row


def execute_sweep_rows_jax(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    array_args: list[_binding.FunctionInputRef],
    n_total: int,
    workflow_kind: WorkflowKind = WorkflowKind.OFF,
    function_name: str,
    output_is_declared: bool = False,
    output_label: str,
) -> Any:
    """Execute the limited single-batch sweep through ``jax.vmap``."""
    single_call = mapped_row_body(
        func=func,
        values=values,
        array_args=array_args,
        field_name=output_label,
        output_is_declared=output_is_declared,
    )

    vmap_input = [
        mapped_storage(_binding.input_ref_value(values, ref), n_total) for ref in array_args
    ]

    def run_vmap():
        with _context._workflow_jax_runtime_guard():
            return jax.vmap(single_call)(tuple(vmap_input))

    if workflow_kind in (WorkflowKind.TASK, WorkflowKind.FLOW):
        if task is None or flow is None:
            raise RuntimeError(
                "Prefect task or flow execution was requested, but Prefect is not "
                "installed. Install with: pip install probpipe[prefect]"
            )
        if workflow_kind is WorkflowKind.TASK:
            run_vmap = task(name=f"{function_name}_vmap")(run_vmap)
        else:
            runner = prefect_config.resolve_task_runner()
            run_vmap = flow(
                name=f"{function_name}_vmap",
                **({"task_runner": runner} if runner is not None else {}),
            )(run_vmap)
    return run_vmap()


def make_sweep_provenance(
    *,
    values: Mapping[str, Any],
    array_args: list[_binding.FunctionInputRef],
    dist_args: list[_binding.FunctionInputRef],
    function_name: str,
    batch_shape: tuple[int, ...],
    k: int,
    parents: list[TrackedTerm] | None = None,
    inputs: Mapping[str, Any] | None = None,
    stochastic_plan: _plan.StochasticPlan | None = None,
    route: Mapping[str, Any] | None = None,
) -> Provenance | None:
    """Build provenance metadata for pure and nested sweep outputs.

    ``parents`` carries tracked call-level lineage; ``inputs`` carries the
    original resolved plain values rather than per-cell sweep values.
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
            "k": k,
            "ra_args": [ref.label for ref in array_args],
            "dist_args": [ref.label for ref in dist_args],
            **(dict(route) if route is not None else {}),
        },
        inputs=inputs,
        controls=controls,
        diagnostics=diagnostics,
    )
