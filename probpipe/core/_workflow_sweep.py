"""Function sweep execution helpers.

This private module owns array-valued workflow sweeps after call
resolution, distribution normalization, and broadcast planning have
already classified the call. It executes pure parameter sweeps and the
outer sweep layer of nested array + distribution broadcasts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax

from . import _workflow_call, _workflow_execution, _workflow_plan, _workflow_result
from ._broadcast_distributions import _make_stack
from ._distribution_array import DistributionArray, _make_distribution_array
from ._record_array import _RecordArrayView
from .distribution import BroadcastDistribution, Distribution
from .event_template import EventTemplate
from .provenance import Provenance
from .record import Record
from .tracked import Tracked


def execute_sweep(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    plan: _workflow_plan.BroadcastPlan,
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
    requested_dispatch: str,
    resolve_dispatch: Callable[..., str],
    require_jax_traceable: Callable[[dict[str, Any], list[_workflow_call.WorkflowInputRef]], None],
    distribution_broadcast: Callable[
        [dict[str, Any], list[_workflow_call.WorkflowInputRef], int, bool],
        BroadcastDistribution | Distribution,
    ],
    workflow_name: str,
    n_broadcast_samples: int,
    include_inputs: bool = False,
    output_template: EventTemplate | None = None,
    provenance_parents: list[Tracked] | None = None,
    provenance_inputs: Mapping[str, Any] | None = None,
) -> Any:
    """Execute pure or nested sweep regimes for one workflow call."""
    if plan.regime not in ("sweep", "nested"):
        raise ValueError(f"execute_sweep requires a sweep plan; got {plan.regime!r}")

    array_args = list(plan.array_args)
    dist_args = list(plan.dist_args)

    if include_inputs:
        raise NotImplementedError(
            "include_inputs=True is not supported with RecordArray "
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
        )
        aggregate = _make_stack(
            per_row,
            batch_shape=plan.sweep_batch_shape,
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
        )
        return _workflow_result._coerce_output(
            aggregate,
            broadcast_mode=_workflow_result.BROADCAST_STACK,
            provenance=provenance,
            field_name=workflow_name,
        )

    per_row_marginals: list[Distribution] = []
    for i in range(plan.n_sweep):
        row_values = slice_sweep_values(
            values=values,
            index=i,
            array_groups=plan.array_groups,
        )
        inner = distribution_broadcast(
            row_values,
            dist_args,
            n_broadcast_samples,
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
        k=n_broadcast_samples,
        parents=provenance_parents,
        inputs=provenance_inputs,
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
    """Materialize one row-major sweep cell under parent-grouped arrays."""
    out = dict(values)
    rem = index
    # Highest-index group varies fastest under row-major flattening of
    # the concatenated sweep shape.
    for group in reversed(array_groups):
        idx = rem % group.size
        rem = rem // group.size
        replacements: dict[_workflow_call.WorkflowInputRef, Any] = {}
        for ref in group.arg_refs:
            source = _workflow_call.input_ref_value(values, ref)
            if isinstance(source, DistributionArray):
                replacements[ref] = source._flat_component(idx)
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
) -> Any:
    """Execute pure sweep rows through JAX vmap or row-wise execution."""
    has_dist_array = any(
        isinstance(_workflow_call.input_ref_value(values, ref), DistributionArray)
        for ref in array_args
    )
    has_view = any(
        isinstance(_workflow_call.input_ref_value(values, ref), _RecordArrayView)
        for ref in array_args
    )
    jax_supported = not (
        has_dist_array or has_view or len(plan.array_groups) > 1 or len(array_args) > 1
    )
    if requested_dispatch == "jax" and not jax_supported:
        raise ValueError(
            "dispatch='jax' supports only a single plain RecordArray sweep; "
            "use dispatch='auto', 'sequential', or 'thread' for this path."
        )

    dispatch = resolve_dispatch(
        values,
        array_args,
        jax_supported=jax_supported,
    )

    if dispatch == "jax":
        if requested_dispatch == "jax":
            require_jax_traceable(values, array_args)
        return execute_sweep_rows_jax(
            func=func,
            values=values,
            array_args=array_args,
            n_total=plan.n_sweep,
        )

    per_row_values = [
        slice_sweep_values(
            values=values,
            index=i,
            array_groups=plan.array_groups,
        )
        for i in range(plan.n_sweep)
    ]
    request = _workflow_execution.WorkflowExecutionRequest(
        func=func,
        call_value_list=per_row_values,
        execution=make_execution_config(),
    )
    return _workflow_execution.execute_many(request)


def execute_sweep_rows_jax(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    array_args: list[_workflow_call.WorkflowInputRef],
    n_total: int,
) -> Any:
    """Execute the limited single-RecordArray sweep through ``jax.vmap``."""

    def single_call(array_slice_leaves):
        replacements = {
            ref: Record(ref.label, leaves, name_is_auto=True)
            for ref, leaves in zip(array_args, array_slice_leaves)
        }
        return func(**_workflow_call.replace_input_refs(values, replacements))

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
    return jax.vmap(single_call)(tuple(vmap_input))


def make_sweep_provenance(
    *,
    values: Mapping[str, Any],
    array_args: list[_workflow_call.WorkflowInputRef],
    dist_args: list[_workflow_call.WorkflowInputRef],
    workflow_name: str,
    batch_shape: tuple[int, ...],
    k: int,
    parents: list[Tracked] | None = None,
    inputs: Mapping[str, Any] | None = None,
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
    )
