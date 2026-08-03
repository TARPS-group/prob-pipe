"""Function distribution-only broadcast helpers.

This private module owns scalar distribution broadcasting after call
resolution, distribution normalization, and broadcast planning have
already identified the distribution-only regime. It handles sampling,
empirical enumeration, JAX ``vmap`` execution, and
``BroadcastDistribution`` assembly.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from ..custom_types import Array, PRNGKey
from . import (
    _workflow_broker,
    _workflow_call,
    _workflow_context,
    _workflow_execution,
    _workflow_execution_contract,
    _workflow_plan,
    _workflow_recipe,
)
from .config import WorkflowKind, prefect_config
from .distribution import BroadcastDistribution, Distribution, EmpiricalDistribution
from .event_template import EventTemplate
from .provenance import Provenance
from .tracked import TrackedTerm

MIN_BROADCAST_SAMPLES = 5


def execute_distribution_broadcast(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    stochastic_plan: _workflow_plan.StochasticPlan,
    logical_unit: _workflow_plan.LogicalUnit,
    include_inputs: bool,
    get_key: Callable[[_workflow_plan.PlannedRandomEvent], PRNGKey],
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
    requested_dispatch: str,
    resolve_dispatch: Callable[..., str],
    require_jax_traceable: Callable[[dict[str, Any], list[_workflow_call.WorkflowInputRef]], None],
    workflow_name: str,
    workflow_kind: WorkflowKind,
    output_template: EventTemplate | None = None,
    provenance_parents: Sequence[TrackedTerm] = (),
    provenance_inputs: Mapping[str, Any] | None = None,
    record_recipe: bool = True,
) -> BroadcastDistribution | Distribution:
    """Execute one distribution-only broadcasted workflow call.

    The caller has already resolved function arguments, normalized any
    distribution-valued inputs, and built a broadcast plan whose regime is
    ``"distribution"``. This function samples or enumerates the named
    distribution inputs, executes ``func`` over the resulting call rows, and
    returns a ``BroadcastDistribution`` or its output marginal.

    Parameters
    ----------
    func : callable
        Wrapped user function to execute for each sampled or enumerated call.
    values : dict[str, Any]
        Resolved workflow inputs. Entries named in ``broadcast_args`` must be
        scalar ``Distribution`` values; all other entries are passed through to
        every call row.
    stochastic_plan : StochasticPlan
        Immutable source grouping, exact/sample classification, combination
        order, and sample-shape decisions for this lifted call.
    logical_unit : LogicalUnit
        Singleton or canonical sweep cell being executed.
    include_inputs : bool
        If ``True``, return the full ``BroadcastDistribution`` containing both
        sampled inputs and outputs. If ``False``, return the marginalized output
        distribution.
    get_key : callable
        Callback that claims the raw key for one planned source/unit event.
    make_execution_config : callable
        Zero-argument callback returning row-wise execution settings for
        sequential, threaded, or Prefect dispatch.
    requested_dispatch : str
        User-requested dispatch strategy, used to preserve explicit
        ``dispatch="jax"`` error behavior.
    resolve_dispatch : callable
        Callback that maps the current values and broadcast arguments to the
        effective dispatch strategy.
    require_jax_traceable : callable
        Callback used only for explicit JAX dispatch to raise a clear tracing
        error before executing.
    workflow_name : str
        Human-readable workflow name recorded in provenance metadata.
    workflow_kind : WorkflowKind
        Effective orchestration mode for this call. The value is recorded in
        provenance and passed to the JAX path so Prefect task/flow requests can
        fail clearly when Prefect is unavailable.
    output_template : EventTemplate or None
        Concrete authoritative template for declared outputs, when present.
    provenance_parents : sequence of TrackedTerm
        Call-level tracked lineage, already ordered and deduplicated.
    provenance_inputs : mapping of str to Any or None
        Call-level resolved plain inputs. Per-row sampled values do not replace
        these original descriptors.

    Returns
    -------
    BroadcastDistribution or Distribution
        The full broadcast distribution when ``include_inputs`` is true;
        otherwise the output marginal distribution.
    """
    broadcast_args = list(stochastic_plan.arg_refs)
    n_broadcast_samples = stochastic_plan.n_broadcast_samples
    _validate_n_broadcast_samples(n_broadcast_samples)

    jax_contract = _workflow_execution_contract.make_execution_contract(
        evaluator="jax_vmap",
        transport=_workflow_execution_contract.transport_for_workflow_kind(workflow_kind),
        stochastic_plan=stochastic_plan,
    )
    jax_supported = _workflow_execution_contract.supports_execution_contract(
        jax_contract,
        stochastic_plan,
    )
    dispatch = resolve_dispatch(
        values,
        broadcast_args,
        jax_supported=jax_supported,
    )
    if requested_dispatch == "jax" and stochastic_plan.evaluation_mode != "sampled":
        raise ValueError(
            "dispatch='jax' does not support exact empirical enumeration; "
            "use dispatch='auto', 'sequential', or 'thread' for this path."
        )

    # Enumeration preserves exact empirical weights and must run in all row-wise
    # dispatch modes; otherwise cartesian-product semantics vary by dispatch.
    if stochastic_plan.evaluation_mode != "sampled":
        result = _broadcast_enumerate(
            func=func,
            values=values,
            stochastic_plan=stochastic_plan,
            logical_unit=logical_unit,
            get_key=get_key,
            make_execution_config=make_execution_config,
            output_template=output_template,
        )
    elif dispatch == "jax":
        _workflow_broker._record_active_execution_contract(jax_contract)
        if requested_dispatch == "jax":
            require_jax_traceable(values, broadcast_args)
        result = _broadcast_jax(
            func=func,
            values=values,
            stochastic_plan=stochastic_plan,
            logical_unit=logical_unit,
            get_key=get_key,
            workflow_name=workflow_name,
            workflow_kind=workflow_kind,
            output_template=output_template,
        )
    else:
        result = _broadcast_sample(
            func=func,
            values=values,
            stochastic_plan=stochastic_plan,
            logical_unit=logical_unit,
            get_key=get_key,
            make_execution_config=make_execution_config,
            output_template=output_template,
        )

    provenance = _make_broadcast_provenance(
        values=values,
        broadcast_args=broadcast_args,
        dispatch=dispatch,
        workflow_kind=workflow_kind,
        n_broadcast_samples=n_broadcast_samples,
        workflow_name=workflow_name,
        func=func,
        provenance_parents=provenance_parents,
        provenance_inputs=provenance_inputs,
        stochastic_plan=stochastic_plan if record_recipe else None,
        record_recipe=record_recipe,
    )
    result.with_provenance(provenance)

    if include_inputs:
        return result
    return result.marginalize()


def _validate_n_broadcast_samples(n_broadcast_samples: int) -> None:
    if isinstance(n_broadcast_samples, bool) or not isinstance(n_broadcast_samples, int):
        raise TypeError(f"n_broadcast_samples must be an integer; got {n_broadcast_samples!r}")

    if n_broadcast_samples <= 0:
        raise ValueError(
            f"n_broadcast_samples must be a positive integer; got {n_broadcast_samples!r}"
        )

    if n_broadcast_samples < MIN_BROADCAST_SAMPLES:
        warnings.warn(
            f"n_broadcast_samples={n_broadcast_samples} is too low; "
            "results may be unreliable. "
            f"Recommended minimum is {MIN_BROADCAST_SAMPLES}.",
            stacklevel=2,
        )


def _make_broadcast_provenance(
    *,
    values: dict[str, Any],
    broadcast_args: Sequence[_workflow_call.WorkflowInputRef],
    dispatch: str,
    workflow_kind: WorkflowKind,
    n_broadcast_samples: int,
    workflow_name: str,
    func: Callable[..., Any],
    provenance_parents: Sequence[TrackedTerm],
    provenance_inputs: Mapping[str, Any] | None,
    stochastic_plan: _workflow_plan.StochasticPlan | None,
    record_recipe: bool,
) -> Provenance | None:
    controls, diagnostics = (
        _workflow_recipe.provenance_recipe_fields(stochastic_plan) if record_recipe else ({}, {})
    )
    return Provenance.create(
        "broadcast",
        parents=list(provenance_parents),
        metadata={
            "dispatch": dispatch,
            "orchestrate": workflow_kind.value,
            "n_samples": n_broadcast_samples,
            "func": workflow_name or func.__name__,
            "broadcast_args": [ref.label for ref in broadcast_args],
        },
        inputs=provenance_inputs,
        controls=controls,
        diagnostics=diagnostics,
    )


def _sample_planned_source_groups(
    stochastic_plan: _workflow_plan.StochasticPlan,
    source_groups: Sequence[_workflow_plan.StochasticSourceGroup],
    sample_shape: tuple[int, ...],
    logical_unit: _workflow_plan.LogicalUnit,
    get_key: Callable[[_workflow_plan.PlannedRandomEvent], PRNGKey],
) -> dict[_workflow_call.WorkflowInputRef, Array]:
    """Claim and sample each planned source once in one logical unit.

    Every group uses the root and consumer evaluators captured during
    preflight, so aliases and record projections share one root draw.
    """
    sampled: dict[_workflow_call.WorkflowInputRef, Array] = {}
    for group in source_groups:
        if group.execution_mode != "sampled":
            continue
        event = _workflow_plan.PlannedRandomEvent(
            stochastic_source_id=group.stochastic_source_id,
            logical_unit_id=logical_unit.logical_unit_id,
        )
        key = get_key(event)
        binding = stochastic_plan.runtime_bindings[group.index]
        root_sample = binding.sample_root(key, sample_shape)
        for consumer, evaluate in zip(group.consumers, binding.consumer_evaluators):
            sampled[consumer.arg_ref] = evaluate(root_sample)
    return sampled


def _broadcast_jax(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    stochastic_plan: _workflow_plan.StochasticPlan,
    logical_unit: _workflow_plan.LogicalUnit,
    get_key: Callable[[_workflow_plan.PlannedRandomEvent], PRNGKey],
    workflow_name: str,
    workflow_kind: WorkflowKind,
    output_template: EventTemplate | None,
) -> BroadcastDistribution:
    """Execute distribution broadcasting through local ``jax.vmap``."""
    if workflow_kind in (WorkflowKind.TASK, WorkflowKind.FLOW) and (task is None or flow is None):
        raise RuntimeError(
            "Prefect task or flow execution was requested, but Prefect is not installed. "
            "Install with: pip install probpipe[prefect]"
        )

    sample_shape = stochastic_plan.sample_shape
    if sample_shape is None:  # pragma: no cover - planner/dispatch contract guard
        raise RuntimeError("sampled stochastic plan is missing sample_shape")
    broadcast_args = list(stochastic_plan.arg_refs)
    sampled = _sample_planned_source_groups(
        stochastic_plan,
        stochastic_plan.source_groups,
        sample_shape,
        logical_unit,
        get_key,
    )

    def single_call(broadcast_slice):
        replacements = dict(zip(broadcast_args, broadcast_slice))
        return func(**_workflow_call.replace_input_refs(values, replacements))

    batch = tuple(sampled[ref] for ref in broadcast_args)

    def run_vmap():
        with _workflow_context._workflow_jax_runtime_guard():
            return jax.vmap(single_call)(batch)

    if workflow_kind in (WorkflowKind.TASK, WorkflowKind.FLOW):
        if workflow_kind == WorkflowKind.TASK:
            run_vmap = task(name=f"{workflow_name}_vmap")(run_vmap)
        else:
            runner = prefect_config.resolve_task_runner()
            run_vmap = flow(
                name=f"{workflow_name}_vmap",
                **({"task_runner": runner} if runner is not None else {}),
            )(run_vmap)

    results = run_vmap()
    return BroadcastDistribution(
        input_samples={ref.label: sampled[ref] for ref in broadcast_args},
        output_samples=results,
        weights=None,
        broadcast_args=[ref.label for ref in broadcast_args],
        output_template=output_template,
    )


def _broadcast_enumerate(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    stochastic_plan: _workflow_plan.StochasticPlan,
    logical_unit: _workflow_plan.LogicalUnit,
    get_key: Callable[[_workflow_plan.PlannedRandomEvent], PRNGKey],
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
    output_template: EventTemplate | None,
) -> BroadcastDistribution:
    """Execute the plan's exact combinations and sampled repetitions."""
    exact_entries: list[
        tuple[
            _workflow_plan.StochasticSourceGroup,
            EmpiricalDistribution,
            tuple[Any, ...],
        ]
    ] = []
    for group_index in stochastic_plan.exact_group_order:
        group = stochastic_plan.source_groups[group_index]
        binding = stochastic_plan.runtime_bindings[group_index]
        dist = binding.root
        if not isinstance(dist, EmpiricalDistribution):  # pragma: no cover - plan contract guard
            raise RuntimeError("exact stochastic source is not an EmpiricalDistribution")
        exact_entries.append(
            (
                group,
                dist,
                tuple(evaluate(dist.samples) for evaluate in binding.consumer_evaluators),
            )
        )

    sampled_groups = tuple(
        group for group in stochastic_plan.source_groups if group.execution_mode == "sampled"
    )
    sample_arg_refs = [ref for group in sampled_groups for ref in group.arg_refs]
    if sampled_groups:
        sample_shape = stochastic_plan.sample_shape
        if sample_shape is None:  # pragma: no cover - planner contract guard
            raise RuntimeError("mixed stochastic plan is missing sample_shape")
        sampled = _sample_planned_source_groups(
            stochastic_plan,
            sampled_groups,
            sample_shape,
            logical_unit,
            get_key,
        )
    else:
        sampled = {}

    call_value_list = []
    weights = []
    sample_idx = 0
    all_broadcast_args = list(stochastic_plan.arg_refs)

    for combo in stochastic_plan.exact_combination_order:
        emp_weight = 1.0
        for (_group, dist, _consumer_batches), i in zip(exact_entries, combo):
            emp_weight *= float(dist.weights[i])

        for _ in range(stochastic_plan.repetitions_per_combination):
            replacements: dict[_workflow_call.WorkflowInputRef, Any] = {}

            for (group, _dist, consumer_batches), i in zip(exact_entries, combo):
                for consumer, consumer_batch in zip(group.consumers, consumer_batches):
                    replacements[consumer.arg_ref] = _index_sample(consumer_batch, i)

            for ref in sample_arg_refs:
                replacements[ref] = _index_sample(sampled[ref], sample_idx)

            weights.append(emp_weight / stochastic_plan.repetitions_per_combination)
            call_value_list.append(_workflow_call.replace_input_refs(values, replacements))
            sample_idx += 1

    execution = make_execution_config()
    request = _workflow_execution.WorkflowExecutionRequest(
        func=func,
        work_items=_workflow_execution.make_managed_work_items(
            call_value_list,
            unit_segments=tuple(
                _workflow_execution.lifted_evaluation_unit_segment(
                    logical_unit.logical_unit_id,
                    index,
                )
                for index in range(len(call_value_list))
            ),
        ),
        execution=execution,
        contract=_workflow_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport=_workflow_execution_contract.transport_for_execution_mode(execution.mode),
            stochastic_plan=stochastic_plan,
        ),
    )
    results = _workflow_execution.execute_many(request)

    all_input_samples = {
        ref.label: _stack_input_rows(
            [_workflow_call.input_ref_value(call_values, ref) for call_values in call_value_list]
        )
        for ref in all_broadcast_args
    }

    return BroadcastDistribution(
        input_samples=all_input_samples,
        output_samples=results,
        weights=jnp.array(weights),
        broadcast_args=[ref.label for ref in all_broadcast_args],
        output_template=output_template,
    )


def _broadcast_sample(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    stochastic_plan: _workflow_plan.StochasticPlan,
    logical_unit: _workflow_plan.LogicalUnit,
    get_key: Callable[[_workflow_plan.PlannedRandomEvent], PRNGKey],
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
    output_template: EventTemplate | None,
) -> BroadcastDistribution:
    """Sample distribution arguments and execute one function call per sample."""
    sample_shape = stochastic_plan.sample_shape
    if sample_shape is None:  # pragma: no cover - planner contract guard
        raise RuntimeError("sampled stochastic plan is missing sample_shape")
    broadcast_args = list(stochastic_plan.arg_refs)
    samples_per_arg = _sample_planned_source_groups(
        stochastic_plan,
        stochastic_plan.source_groups,
        sample_shape,
        logical_unit,
        get_key,
    )

    call_value_list = []
    for i in range(stochastic_plan.n_evaluations):
        replacements = {ref: _index_sample(samples_per_arg[ref], i) for ref in broadcast_args}
        call_value_list.append(_workflow_call.replace_input_refs(values, replacements))

    execution = make_execution_config()
    request = _workflow_execution.WorkflowExecutionRequest(
        func=func,
        work_items=_workflow_execution.make_managed_work_items(
            call_value_list,
            unit_segments=tuple(
                _workflow_execution.lifted_evaluation_unit_segment(
                    logical_unit.logical_unit_id,
                    index,
                )
                for index in range(len(call_value_list))
            ),
        ),
        execution=execution,
        contract=_workflow_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport=_workflow_execution_contract.transport_for_execution_mode(execution.mode),
            stochastic_plan=stochastic_plan,
        ),
    )
    results = _workflow_execution.execute_many(request)

    return BroadcastDistribution(
        input_samples={ref.label: samples_per_arg[ref] for ref in broadcast_args},
        output_samples=results,
        weights=None,
        broadcast_args=[ref.label for ref in broadcast_args],
        output_template=output_template,
    )


def _index_sample(s: Any, i: int) -> Any:
    """Index row ``i`` of a per-argument sample batch."""
    from .record import Record

    if isinstance(s, Record):
        # Index each leaf field's batch row; rebuild by path key so a nested
        # sample is reconstructed with its structure intact.
        leaf_paths = tuple(s.event_template.keys())
        if len(leaf_paths) == 1:
            return s[leaf_paths[0]][i]
        return Record(s.name, {p: s[p][i] for p in leaf_paths}, name_is_auto=True)
    return s[i]


def _stack_input_rows(rows: list[Any]) -> Any:
    """Stack exact input rows while preserving Record-valued roots."""
    from ._record_array import RecordArray
    from .record import Record

    if rows and all(isinstance(row, Record) for row in rows):
        try:
            return RecordArray.stack(rows)
        except TypeError:
            return jax.tree.map(lambda *leaves: jnp.stack(leaves), *rows)
    return jnp.stack(rows)
