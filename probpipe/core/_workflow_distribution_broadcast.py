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
from . import _workflow_call, _workflow_execution, _workflow_plan
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
    include_inputs: bool,
    get_key: Callable[[], PRNGKey],
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
    include_inputs : bool
        If ``True``, return the full ``BroadcastDistribution`` containing both
        sampled inputs and outputs. If ``False``, return the marginalized output
        distribution.
    get_key : callable
        Zero-argument callback that returns the next PRNG key for sampling.
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

    dispatch = resolve_dispatch(
        values,
        broadcast_args,
        jax_supported=stochastic_plan.evaluation_mode == "sampled",
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
            get_key=get_key,
            make_execution_config=make_execution_config,
            output_template=output_template,
        )
    elif dispatch == "jax":
        if requested_dispatch == "jax":
            require_jax_traceable(values, broadcast_args)
        result = _broadcast_jax(
            func=func,
            values=values,
            stochastic_plan=stochastic_plan,
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
    )
    result.with_provenance(provenance)

    if include_inputs:
        return result
    return result.marginalize()


def _validate_n_broadcast_samples(n_broadcast_samples: int) -> None:
    if not isinstance(n_broadcast_samples, int):
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
) -> Provenance | None:
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
    )


def _sample_planned_source_groups(
    values: dict[str, Any],
    source_groups: Sequence[_workflow_plan.StochasticSourceGroup],
    sample_shape: tuple[int, ...],
    key: PRNGKey,
) -> dict[_workflow_call.WorkflowInputRef, Array]:
    """Sample plan-owned source groups from one transitional parent key.

    Sibling views from the same parent distribution share one parent draw,
    preserving cross-field correlation. Plain non-view distributions are
    sampled through their separate current-wave source groups. The caller still
    supplies one parent key in this checkpoint; source/unit event claims replace
    that transitional split in the next checkpoint.
    """
    sampled: dict[_workflow_call.WorkflowInputRef, Array] = {}
    for group in source_groups:
        if group.execution_mode != "sampled":
            continue
        first = _workflow_call.input_ref_value(values, group.arg_refs[0])
        if group.source_kind == "direct":
            for ref in group.arg_refs:
                key, subkey = jax.random.split(key)
                dist = _workflow_call.input_ref_value(values, ref)
                sampled[ref] = dist._sample(subkey, sample_shape)
            continue
        key, subkey = jax.random.split(key)
        structured = first.parent._sample(subkey, sample_shape)
        for ref in group.arg_refs:
            view = _workflow_call.input_ref_value(values, ref)
            if hasattr(view, "_extract"):
                sampled[ref] = view._extract(structured)
            else:
                val = structured
                for k in getattr(view, "_key_path", (view.field,)):
                    val = val[k]
                sampled[ref] = val
    return sampled


def _broadcast_jax(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    stochastic_plan: _workflow_plan.StochasticPlan,
    get_key: Callable[[], PRNGKey],
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
        values,
        stochastic_plan.source_groups,
        sample_shape,
        get_key(),
    )

    def single_call(broadcast_slice):
        replacements = dict(zip(broadcast_args, broadcast_slice))
        return func(**_workflow_call.replace_input_refs(values, replacements))

    batch = tuple(sampled[ref] for ref in broadcast_args)

    def run_vmap():
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
    get_key: Callable[[], PRNGKey],
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
    output_template: EventTemplate | None,
) -> BroadcastDistribution:
    """Execute the plan's exact combinations and sampled repetitions."""
    exact_entries: list[tuple[_workflow_call.WorkflowInputRef, EmpiricalDistribution]] = []
    for group_index in stochastic_plan.exact_group_order:
        group = stochastic_plan.source_groups[group_index]
        ref = group.arg_refs[0]
        dist = _workflow_call.input_ref_value(values, ref)
        if not isinstance(dist, EmpiricalDistribution):  # pragma: no cover - plan contract guard
            raise RuntimeError("exact stochastic source is not an EmpiricalDistribution")
        exact_entries.append((ref, dist))

    sampled_groups = tuple(
        group for group in stochastic_plan.source_groups if group.execution_mode == "sampled"
    )
    sample_arg_refs = [ref for group in sampled_groups for ref in group.arg_refs]
    if sampled_groups:
        sample_shape = stochastic_plan.sample_shape
        if sample_shape is None:  # pragma: no cover - planner contract guard
            raise RuntimeError("mixed stochastic plan is missing sample_shape")
        sampled = _sample_planned_source_groups(
            values,
            sampled_groups,
            sample_shape,
            get_key(),
        )
    else:
        sampled = {}

    call_value_list = []
    weights = []
    sample_idx = 0
    exact_arg_refs = [ref for ref, _dist in exact_entries]
    all_broadcast_args = exact_arg_refs + sample_arg_refs

    for combo in stochastic_plan.exact_combination_order:
        emp_weight = 1.0
        for (_ref, dist), i in zip(exact_entries, combo):
            emp_weight *= float(dist.weights[i])

        for _ in range(stochastic_plan.repetitions_per_combination):
            replacements: dict[_workflow_call.WorkflowInputRef, Any] = {}

            for (ref, dist), i in zip(exact_entries, combo):
                replacements[ref] = _index_sample(dist.samples, i)

            for ref in sample_arg_refs:
                replacements[ref] = _index_sample(sampled[ref], sample_idx)

            weights.append(emp_weight / stochastic_plan.repetitions_per_combination)
            call_value_list.append(_workflow_call.replace_input_refs(values, replacements))
            sample_idx += 1

    request = _workflow_execution.WorkflowExecutionRequest(
        func=func,
        call_value_list=call_value_list,
        execution=make_execution_config(),
    )
    results = _workflow_execution.execute_many(request)

    all_input_samples = {
        ref.label: jnp.stack(
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
    get_key: Callable[[], PRNGKey],
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
        values,
        stochastic_plan.source_groups,
        sample_shape,
        get_key(),
    )

    call_value_list = []
    for i in range(stochastic_plan.n_evaluations):
        replacements = {ref: _index_sample(samples_per_arg[ref], i) for ref in broadcast_args}
        call_value_list.append(_workflow_call.replace_input_refs(values, replacements))

    request = _workflow_execution.WorkflowExecutionRequest(
        func=func,
        call_value_list=call_value_list,
        execution=make_execution_config(),
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
