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
import numpy as np

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from . import (
    _workflow_broker,
    _workflow_call,
    _workflow_context,
    _workflow_execution,
    _workflow_execution_contract,
    _workflow_plan,
    _workflow_recipe,
    _workflow_result,
)
from ._batch import Batch
from ._broadcast_distributions import _make_stack, _row_at_its_kind
from ._distribution_array import DistributionArray, _make_distribution_array
from ._numeric_array_batch import NumericArrayBatch, _MappedBatchStore
from ._record_batch import RecordBatch, _MappedBatchColumns
from .config import WorkflowKind, prefect_config
from .distribution import BroadcastDistribution, Distribution
from .event_template import EventTemplate
from .provenance import Provenance
from .record import Record
from .tracked import TrackedTerm


def execute_sweep(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    plan: _workflow_plan.BroadcastPlan,
    stochastic_plan: _workflow_plan.StochasticPlan | None,
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
    requested_dispatch: str,
    resolve_dispatch: Callable[..., str],
    require_jax_traceable: Callable[[dict[str, Any], list[_workflow_call.WorkflowInputRef]], None],
    distribution_broadcast: Callable[
        [
            dict[str, Any],
            _workflow_plan.StochasticPlan,
            _workflow_plan.LogicalUnit,
            bool,
        ],
        BroadcastDistribution | Distribution,
    ],
    workflow_name: str,
    include_inputs: bool = False,
    output_template: EventTemplate | None = None,
    provenance_parents: list[TrackedTerm] | None = None,
    provenance_inputs: Mapping[str, Any] | None = None,
    workflow_kind: WorkflowKind = WorkflowKind.OFF,
) -> Any:
    """Execute pure or nested sweep regimes for one workflow call."""
    if plan.regime not in ("sweep", "nested"):
        raise ValueError(f"execute_sweep requires a sweep plan; got {plan.regime!r}")

    array_args = list(plan.array_args)
    dist_args = list(plan.dist_args)

    if include_inputs:
        raise NotImplementedError(
            "include_inputs=True is not supported with batched-record "
            "broadcasting. The inputs are already available via "
            "provenance on the stacked output."
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
            workflow_name=workflow_name,
            output_is_declared=output_template is not None,
        )
        aggregate = _make_stack(
            per_row,
            batch_shape=plan.sweep_batch_shape,
            # The aggregate mints the levels the sweep ranged over, so it aligns
            # by name with the batch it swept.
            level_names=plan.sweep_level_names,
            axis_groups=plan.sweep_axis_groups,
            name=workflow_name,
            field_name=workflow_name,
            event_template=output_template,
        )
        provenance = make_sweep_provenance(
            values=values,
            array_args=array_args,
            dist_args=dist_args,
            workflow_name=workflow_name,
            batch_shape=plan.sweep_batch_shape,
            k=0,
            parents=provenance_parents,
            inputs=provenance_inputs,
            stochastic_plan=None,
        )
        return _workflow_result._coerce_output(
            aggregate,
            broadcast_mode=_workflow_result.BROADCAST_STACK,
            provenance=provenance,
            field_name=workflow_name,
        )

    if stochastic_plan is None:  # pragma: no cover - Function planning contract guard
        raise RuntimeError("nested sweep is missing its stochastic plan")

    per_row_marginals: list[Distribution] = []
    for logical_unit in stochastic_plan.logical_units:
        row_values = slice_sweep_values(
            values=values,
            index=logical_unit.flat_index,
            array_groups=plan.array_groups,
        )
        inner = distribution_broadcast(
            row_values,
            stochastic_plan,
            logical_unit,
            True,
        )
        if isinstance(inner, BroadcastDistribution):
            marginal = inner.marginalize()
        else:
            marginal = inner
        per_row_marginals.append(marginal)

    stacked = _make_distribution_array(
        per_row_marginals,
        batch_shape=plan.sweep_batch_shape,
        name=workflow_name or "sweep",
        name_is_auto=True,
        event_template=output_template,
    )
    provenance = make_sweep_provenance(
        values=values,
        array_args=array_args,
        dist_args=dist_args,
        workflow_name=workflow_name,
        batch_shape=plan.sweep_batch_shape,
        k=stochastic_plan.n_broadcast_samples,
        parents=provenance_parents,
        inputs=provenance_inputs,
        stochastic_plan=stochastic_plan,
    )
    return _workflow_result._coerce_output(
        stacked,
        broadcast_mode=_workflow_result.BROADCAST_NESTED,
        provenance=provenance,
        field_name=workflow_name,
    )


def slice_sweep_values(
    *,
    values: Mapping[str, Any],
    index: int,
    array_groups: tuple[_workflow_plan.ArrayBroadcastGroup, ...],
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
        # run off its end. Positional tuples are used only for Batch values;
        # other array-like operands retain their flat row-major index.
        position: Any = idx
        if len(group.batch_shape) > 1:
            position = tuple(int(i) for i in np.unravel_index(idx, group.batch_shape))
        replacements: dict[_workflow_call.WorkflowInputRef, Any] = {}
        for ref in group.arg_refs:
            source = _workflow_call.input_ref_value(values, ref)
            if isinstance(source, DistributionArray):
                replacements[ref] = source._flat_component(idx)
            elif isinstance(source, Batch):
                replacements[ref] = source[position]
            else:
                replacements[ref] = source[idx]
        out = _workflow_call.replace_input_refs(out, replacements)
    return out


def execute_sweep_rows(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    array_args: list[_workflow_call.WorkflowInputRef],
    plan: _workflow_plan.BroadcastPlan,
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
    requested_dispatch: str,
    resolve_dispatch: Callable[..., str],
    require_jax_traceable: Callable[[dict[str, Any], list[_workflow_call.WorkflowInputRef]], None],
    workflow_kind: WorkflowKind = WorkflowKind.OFF,
    workflow_name: str = "workflow",
    output_is_declared: bool = False,
) -> Any:
    """Execute pure sweep rows through JAX vmap or row-wise execution."""
    # Zero rows run nothing, so there is no body for a dispatch to trace and no
    # per-row output for the paths to disagree over: every dispatch takes the
    # same empty aggregation, which is what keeps the output schema independent
    # of how the rows would have been executed.
    if plan.n_sweep == 0:
        return []

    has_dist_array = any(
        isinstance(_workflow_call.input_ref_value(values, ref), DistributionArray)
        for ref in array_args
    )
    jax_structure_supported = not (
        has_dist_array or len(plan.array_groups) > 1 or len(array_args) > 1
    )
    jax_contract = _workflow_execution_contract.make_execution_contract(
        evaluator="jax_vmap",
        transport=_workflow_execution_contract.transport_for_workflow_kind(workflow_kind),
        stochastic_plan=None,
    )
    jax_supported = _workflow_execution_contract.supports_execution_contract(
        jax_contract,
        None,
        jax_structure_supported=jax_structure_supported,
    )
    if requested_dispatch == "jax" and not jax_supported:
        raise ValueError(
            "dispatch='jax' supports only a single plain batched-record sweep; "
            "use dispatch='auto', 'sequential', or 'thread' for this path."
        )

    dispatch = resolve_dispatch(
        values,
        array_args,
        jax_supported=jax_supported,
    )

    if dispatch == "jax":
        _workflow_broker._record_active_execution_contract(jax_contract)
        if requested_dispatch == "jax":
            require_jax_traceable(values, array_args)
        return execute_sweep_rows_jax(
            func=func,
            values=values,
            array_args=array_args,
            n_total=plan.n_sweep,
            workflow_kind=workflow_kind,
            workflow_name=workflow_name,
            output_is_declared=output_is_declared,
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
    request = _workflow_execution.WorkflowExecutionRequest(
        func=func,
        work_items=_workflow_execution.make_managed_work_items(
            per_row_values,
            unit_segments=tuple(
                _workflow_execution.sweep_unit_segment(tuple(coordinates))
                for coordinates in cartesian_product(
                    *(range(axis) for axis in plan.sweep_batch_shape)
                )
            ),
        ),
        execution=execution,
        contract=_workflow_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport=_workflow_execution_contract.transport_for_execution_mode(execution.mode),
            stochastic_plan=None,
        ),
    )
    return _workflow_execution.execute_many(request)


def mapped_row_body(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    array_args: Sequence[_workflow_call.WorkflowInputRef],
    field_name: str,
    output_is_declared: bool = False,
) -> Callable[[Any], Any]:
    """The body ``jax.vmap`` runs for one sweep row, and the probe traces.

    Both callers use this rather than each building its own: the probe's job is
    to trace exactly what the executor runs, which two separately maintained
    functions cannot promise.

    A row's batched-record argument is rebuilt from raw leaf columns inside the
    traced call, so nothing infers a batch axis on the way in. On the way out the
    row takes the kind of its own return, as a row-wise row does, and a record or
    a batch is then taken apart into :class:`_MappedBatchColumns` or
    :class:`_MappedBatchStore`: the map is about to add an axis that neither
    class's unflatten hook could name, and the executor names it afterwards from
    the sweep's levels.

    *output_is_declared* says *func* already gave the row a declared template, in
    which case that template is the row's kind and nothing here re-derives one.
    """

    def one_row(array_slice_leaves):
        replacements = {
            ref: Record(ref.label, leaves, name_is_auto=True)
            for ref, leaves in zip(array_args, array_slice_leaves)
        }
        out = func(**_workflow_call.replace_input_refs(values, replacements))
        if not output_is_declared:
            out = _row_at_its_kind(out, field_name)
            if isinstance(out, Record):
                # As a batch row is carried, but with no level of its own: the
                # axis the map adds is the only one the aggregate will have.
                return _MappedBatchColumns.of_record(out)
        if isinstance(out, RecordBatch):
            return _MappedBatchColumns.of(out)
        if isinstance(out, NumericArrayBatch):
            return _MappedBatchStore.of(out)
        return out

    return one_row


def execute_sweep_rows_jax(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    array_args: list[_workflow_call.WorkflowInputRef],
    n_total: int,
    workflow_kind: WorkflowKind = WorkflowKind.OFF,
    workflow_name: str = "workflow",
    output_is_declared: bool = False,
) -> Any:
    """Execute the limited single-batch sweep through ``jax.vmap``."""
    single_call = mapped_row_body(
        func=func,
        values=values,
        array_args=array_args,
        field_name=workflow_name,
        output_is_declared=output_is_declared,
    )

    vmap_input = []
    for ref in array_args:
        array_value = _workflow_call.input_ref_value(values, ref)
        n_batch = len(array_value.batch_shape)
        vmap_input.append(
            {
                leaf: array_value[leaf].reshape((n_total, *array_value[leaf].shape[n_batch:]))
                for leaf in array_value.event_template
            }
        )

    def run_vmap():
        with _workflow_context._workflow_jax_runtime_guard():
            return jax.vmap(single_call)(tuple(vmap_input))

    if workflow_kind in (WorkflowKind.TASK, WorkflowKind.FLOW):
        if task is None or flow is None:
            raise RuntimeError(
                "Prefect task or flow execution was requested, but Prefect is not "
                "installed. Install with: pip install probpipe[prefect]"
            )
        if workflow_kind is WorkflowKind.TASK:
            run_vmap = task(name=f"{workflow_name}_vmap")(run_vmap)
        else:
            runner = prefect_config.resolve_task_runner()
            run_vmap = flow(
                name=f"{workflow_name}_vmap",
                **({"task_runner": runner} if runner is not None else {}),
            )(run_vmap)
    return run_vmap()


def make_sweep_provenance(
    *,
    values: Mapping[str, Any],
    array_args: list[_workflow_call.WorkflowInputRef],
    dist_args: list[_workflow_call.WorkflowInputRef],
    workflow_name: str,
    batch_shape: tuple[int, ...],
    k: int,
    parents: list[TrackedTerm] | None = None,
    inputs: Mapping[str, Any] | None = None,
    stochastic_plan: _workflow_plan.StochasticPlan | None = None,
) -> Provenance | None:
    """Build provenance metadata for pure and nested sweep outputs.

    ``parents`` carries tracked call-level lineage; ``inputs`` carries the
    original resolved plain values rather than per-cell sweep values.
    Returns ``None`` when :attr:`ProvenanceMode.OFF` is active.
    """
    regime = "nested" if dist_args else "stack"
    if parents is None:
        array_candidates = [_workflow_call.input_ref_value(values, ref) for ref in array_args]
        dist_candidates = [
            _workflow_call.input_ref_value(values, ref)
            for ref in dist_args
            if isinstance(_workflow_call.input_ref_value(values, ref), Distribution)
        ]
        parents = array_candidates + dist_candidates
    controls, diagnostics = _workflow_recipe.provenance_recipe_fields(stochastic_plan)
    return Provenance.create(
        f"workflow.{regime}",
        parents=parents,
        metadata={
            "func": workflow_name,
            "batch_shape": tuple(batch_shape),
            "k": k,
            "ra_args": [ref.label for ref in array_args],
            "dist_args": [ref.label for ref in dist_args],
        },
        inputs=inputs,
        controls=controls,
        diagnostics=diagnostics,
    )
