"""WorkflowFunction sweep execution helpers.

This private module owns array-valued workflow sweeps after call
resolution, distribution normalization, and broadcast planning have
already classified the call. It executes pure parameter sweeps and the
outer sweep layer of nested array + distribution broadcasts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax

from . import _workflow_execution, _workflow_plan, _workflow_result
from ._broadcast_distributions import _make_stack
from ._distribution_array import DistributionArray, _make_distribution_array
from ._record_array import _RecordArrayView
from .distribution import BroadcastDistribution, Distribution
from .provenance import Provenance


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
    require_jax_traceable: Callable[[dict[str, Any], list[str]], None],
    distribution_broadcast: Callable[
        [dict[str, Any], list[str], int, bool],
        BroadcastDistribution | Distribution,
    ],
    workflow_name: str,
    n_broadcast_samples: int,
    include_inputs: bool = False,
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
        )
        provenance = make_sweep_provenance(
            values=values,
            array_args=array_args,
            dist_args=dist_args,
            workflow_name=workflow_name,
            batch_shape=plan.sweep_batch_shape,
            k=0,
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
    )
    provenance = make_sweep_provenance(
        values=values,
        array_args=array_args,
        dist_args=dist_args,
        workflow_name=workflow_name,
        batch_shape=plan.sweep_batch_shape,
        k=n_broadcast_samples,
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
        for name in group.arg_names:
            source = values[name]
            if isinstance(source, DistributionArray):
                out[name] = source._flat_component(idx)
            else:
                out[name] = source[idx]
    return out


def execute_sweep_rows(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    array_args: list[str],
    plan: _workflow_plan.BroadcastPlan,
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
    requested_dispatch: str,
    resolve_dispatch: Callable[..., str],
    require_jax_traceable: Callable[[dict[str, Any], list[str]], None],
) -> Any:
    """Execute pure sweep rows through JAX vmap or row-wise execution."""
    has_dist_array = any(
        isinstance(values[name], DistributionArray) for name in array_args
    )
    has_view = any(
        isinstance(values[name], _RecordArrayView) for name in array_args
    )
    jax_supported = not (
        has_dist_array
        or has_view
        or len(plan.array_groups) > 1
        or len(array_args) > 1
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
    array_args: list[str],
    n_total: int,
) -> Any:
    """Execute the limited single-RecordArray sweep through ``jax.vmap``."""
    static = {name: value for name, value in values.items() if name not in array_args}

    def single_call(array_slice_leaves):
        kwargs = dict(static)
        for name in array_args:
            array_value = values[name]
            kwargs[name] = array_value._record_cls(array_slice_leaves[name])
        return func(**kwargs)

    vmap_input = {}
    for name in array_args:
        array_value = values[name]
        n_batch = len(array_value.batch_shape)
        vmap_input[name] = {
            field: array_value[field].reshape(
                (n_total, *array_value[field].shape[n_batch:])
            )
            for field in array_value.fields
        }
    return jax.vmap(single_call)(vmap_input)


def make_sweep_provenance(
    *,
    values: Mapping[str, Any],
    array_args: list[str],
    dist_args: list[str],
    workflow_name: str,
    batch_shape: tuple[int, ...],
    k: int,
) -> Provenance | None:
    """Build provenance metadata for pure and nested sweep outputs.

    Returns ``None`` when :attr:`ProvenanceMode.OFF` is active.
    """
    regime = "nested" if dist_args else "stack"
    array_candidates = [values[name] for name in array_args]
    dist_candidates = [
        values[name] for name in dist_args
        if isinstance(values[name], Distribution)
    ]
    return Provenance.create(
        f"workflow.{regime}",
        parents=array_candidates + dist_candidates,
        metadata={
            "func": workflow_name,
            "batch_shape": tuple(batch_shape),
            "k": k,
            "ra_args": list(array_args),
            "dist_args": list(dist_args),
        },
    )
