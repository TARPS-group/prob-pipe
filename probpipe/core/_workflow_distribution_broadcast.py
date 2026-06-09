"""WorkflowFunction distribution-only broadcast helpers.

This private module owns scalar distribution broadcasting after call
resolution, distribution normalization, and broadcast planning have
already identified the distribution-only regime. It handles sampling,
empirical enumeration, JAX ``vmap`` execution, and
``BroadcastDistribution`` assembly.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from itertools import product as cartesian_product
from typing import Any

import jax
import jax.numpy as jnp

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from ..custom_types import Array, PRNGKey
from . import _workflow_execution, _workflow_plan
from ._record_distribution import _RecordDistributionView
from .config import WorkflowKind, prefect_config
from .distribution import BroadcastDistribution, Distribution, EmpiricalDistribution
from .provenance import Provenance

MIN_BROADCAST_SAMPLES = 5


def execute_distribution_broadcast(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    broadcast_args: Sequence[str],
    n_broadcast_samples: int,
    include_inputs: bool,
    get_key: Callable[[], PRNGKey],
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
    requested_dispatch: str,
    resolve_dispatch: Callable[..., str],
    require_jax_traceable: Callable[[dict[str, Any], list[str]], None],
    workflow_name: str,
    workflow_kind: WorkflowKind,
) -> BroadcastDistribution | Distribution:
    """Execute distribution-only workflow broadcasting."""
    broadcast_args = list(broadcast_args)
    _validate_n_broadcast_samples(n_broadcast_samples)

    empirical_args, sample_args, product_size = _split_empirical_args(
        values=values,
        broadcast_args=broadcast_args,
        n_broadcast_samples=n_broadcast_samples,
    )

    dispatch = resolve_dispatch(
        values,
        broadcast_args,
        jax_supported=not empirical_args,
    )
    if requested_dispatch == "jax" and empirical_args:
        raise ValueError(
            "dispatch='jax' does not support exact empirical enumeration; "
            "use dispatch='auto', 'sequential', or 'thread' for this path."
        )

    # Enumeration preserves exact empirical weights and must run in all row-wise
    # dispatch modes; otherwise cartesian-product semantics vary by dispatch.
    if empirical_args:
        result = _broadcast_enumerate(
            func=func,
            values=values,
            empirical_args=empirical_args,
            sample_args=sample_args,
            product_size=product_size,
            n_broadcast_samples=n_broadcast_samples,
            get_key=get_key,
            make_execution_config=make_execution_config,
        )
    elif dispatch == "jax":
        if requested_dispatch == "jax":
            require_jax_traceable(values, broadcast_args)
        result = _broadcast_jax(
            func=func,
            values=values,
            broadcast_args=broadcast_args,
            n_broadcast_samples=n_broadcast_samples,
            get_key=get_key,
            workflow_name=workflow_name,
            workflow_kind=workflow_kind,
        )
    else:
        result = _broadcast_sample(
            func=func,
            values=values,
            broadcast_args=broadcast_args,
            n_broadcast_samples=n_broadcast_samples,
            get_key=get_key,
            make_execution_config=make_execution_config,
        )

    provenance = _make_broadcast_provenance(
        values=values,
        broadcast_args=broadcast_args,
        dispatch=dispatch,
        workflow_kind=workflow_kind,
        n_broadcast_samples=n_broadcast_samples,
        workflow_name=workflow_name,
        func=func,
    )
    result.with_source(provenance)

    if include_inputs:
        return result
    return result.marginalize()


def _validate_n_broadcast_samples(n_broadcast_samples: int) -> None:
    if not isinstance(n_broadcast_samples, int):
        raise TypeError(
            f"n_broadcast_samples must be an integer; got {n_broadcast_samples!r}"
        )

    if n_broadcast_samples <= 0:
        raise ValueError(
            "n_broadcast_samples must be a positive integer; "
            f"got {n_broadcast_samples!r}"
        )

    if n_broadcast_samples < MIN_BROADCAST_SAMPLES:
        warnings.warn(
            f"n_broadcast_samples={n_broadcast_samples} is too low; "
            "results may be unreliable. "
            f"Recommended minimum is {MIN_BROADCAST_SAMPLES}.",
            stacklevel=2,
        )


def _split_empirical_args(
    *,
    values: dict[str, Any],
    broadcast_args: Sequence[str],
    n_broadcast_samples: int,
) -> tuple[dict[str, EmpiricalDistribution], dict[str, Distribution], int]:
    candidates: list[tuple[str, EmpiricalDistribution]] = []
    sample_args: dict[str, Distribution] = {}
    for name in broadcast_args:
        dist = values[name]
        if (
            isinstance(dist, EmpiricalDistribution)
            and dist.num_atoms <= n_broadcast_samples
        ):
            candidates.append((name, dist))
        else:
            sample_args[name] = dist
    candidates.sort(key=lambda pair: pair[1].num_atoms)

    empirical_args: dict[str, EmpiricalDistribution] = {}
    product_size = 1
    for name, dist in candidates:
        if product_size * dist.num_atoms <= n_broadcast_samples:
            empirical_args[name] = dist
            product_size *= dist.num_atoms
        else:
            sample_args[name] = dist

    return empirical_args, sample_args, product_size


def _make_broadcast_provenance(
    *,
    values: dict[str, Any],
    broadcast_args: Sequence[str],
    dispatch: str,
    workflow_kind: WorkflowKind,
    n_broadcast_samples: int,
    workflow_name: str,
    func: Callable[..., Any],
) -> Provenance:
    parents = tuple(
        values[name] for name in broadcast_args if isinstance(values[name], Distribution)
    )
    return Provenance(
        "broadcast",
        parents=parents,
        metadata={
            "dispatch": dispatch,
            "orchestrate": workflow_kind.value,
            "n_samples": n_broadcast_samples,
            "func": workflow_name or func.__name__,
            "broadcast_args": list(broadcast_args),
        },
    )


def _sample_broadcast_args(
    values: dict[str, Any],
    broadcast_args: Sequence[str],
    n: int,
    key: PRNGKey,
) -> dict[str, Array]:
    """Sample all broadcast arguments, handling view reconnection.

    Sibling views from the same parent distribution share one parent draw,
    preserving cross-field correlation. Plain non-view distributions are
    sampled independently per kwarg, even if the same object is passed under
    multiple names.
    """
    sampled: dict[str, Array] = {}
    for arg_names in _workflow_plan.group_by_parent(
        values=values, names=broadcast_args,
    ).values():
        first = values[arg_names[0]]
        if not isinstance(first, _RecordDistributionView):
            for arg_name in arg_names:
                key, subkey = jax.random.split(key)
                sampled[arg_name] = values[arg_name]._sample(subkey, (n,))
            continue
        key, subkey = jax.random.split(key)
        structured = first.parent._sample(subkey, (n,))
        for arg_name in arg_names:
            view = values[arg_name]
            if hasattr(view, "_extract"):
                sampled[arg_name] = view._extract(structured)
            else:
                val = structured
                for k in getattr(view, "_key_path", (view.field,)):
                    val = val[k]
                sampled[arg_name] = val
    return sampled


def _broadcast_jax(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    broadcast_args: list[str],
    n_broadcast_samples: int,
    get_key: Callable[[], PRNGKey],
    workflow_name: str,
    workflow_kind: WorkflowKind,
) -> BroadcastDistribution:
    """Execute distribution broadcasting through local ``jax.vmap``."""
    if workflow_kind in (WorkflowKind.TASK, WorkflowKind.FLOW) and (
        task is None or flow is None
    ):
        raise RuntimeError(
            "Prefect task or flow execution was requested, but Prefect is not installed. "
            "Install with: pip install probpipe[prefect]"
        )

    key = get_key()
    sampled = _sample_broadcast_args(
        values, broadcast_args, n_broadcast_samples, key,
    )
    static = {k: v for k, v in values.items() if k not in broadcast_args}

    def single_call(broadcast_slice):
        kw = dict(static)
        kw.update(broadcast_slice)
        return func(**kw)

    batch = {name: sampled[name] for name in broadcast_args}

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
        input_samples=sampled,
        output_samples=results,
        weights=None,
        broadcast_args=broadcast_args,
    )


def _broadcast_enumerate(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    empirical_args: dict[str, EmpiricalDistribution],
    sample_args: dict[str, Distribution],
    product_size: int,
    n_broadcast_samples: int,
    get_key: Callable[[], PRNGKey],
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
) -> BroadcastDistribution:
    """Enumerate empirical distributions and sample any remaining inputs."""
    key = get_key()
    emp_names = list(empirical_args.keys())
    emp_dists = [empirical_args[name] for name in emp_names]

    reps_per_combo = max(1, n_broadcast_samples // product_size) if sample_args else 1
    total = product_size * reps_per_combo

    sample_arg_names = list(sample_args.keys())
    if sample_arg_names:
        sampled = _sample_broadcast_args(values, sample_arg_names, total, key)
    else:
        sampled = {}

    call_value_list = []
    weights = []
    sample_idx = 0

    all_broadcast_args = emp_names + sample_arg_names

    for combo in cartesian_product(*(range(d.num_atoms) for d in emp_dists)):
        emp_weight = 1.0
        for _name, dist, i in zip(emp_names, emp_dists, combo):
            emp_weight *= float(dist.weights[i])

        for _ in range(reps_per_combo):
            call_values = dict(values)

            for name, dist, i in zip(emp_names, emp_dists, combo):
                call_values[name] = _index_sample(dist.samples, i)

            for name in sample_args:
                call_values[name] = _index_sample(sampled[name], sample_idx)

            weights.append(emp_weight / reps_per_combo)
            call_value_list.append(call_values)
            sample_idx += 1

    request = _workflow_execution.WorkflowExecutionRequest(
        func=func,
        call_value_list=call_value_list,
        execution=make_execution_config(),
    )
    results = _workflow_execution.execute_many(request)

    all_input_samples = {
        name: jnp.stack([cv[name] for cv in call_value_list])
        for name in all_broadcast_args
    }

    return BroadcastDistribution(
        input_samples=all_input_samples,
        output_samples=results,
        weights=jnp.array(weights),
        broadcast_args=all_broadcast_args,
    )


def _broadcast_sample(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    broadcast_args: list[str],
    n_broadcast_samples: int,
    get_key: Callable[[], PRNGKey],
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
) -> BroadcastDistribution:
    """Sample distribution arguments and execute one function call per sample."""
    key = get_key()
    samples_per_arg = _sample_broadcast_args(
        values, broadcast_args, n_broadcast_samples, key,
    )

    call_value_list = []
    for i in range(n_broadcast_samples):
        call_values = dict(values)
        for name in broadcast_args:
            call_values[name] = _index_sample(samples_per_arg[name], i)
        call_value_list.append(call_values)

    request = _workflow_execution.WorkflowExecutionRequest(
        func=func,
        call_value_list=call_value_list,
        execution=make_execution_config(),
    )
    results = _workflow_execution.execute_many(request)

    return BroadcastDistribution(
        input_samples=samples_per_arg,
        output_samples=results,
        weights=None,
        broadcast_args=broadcast_args,
    )


def _index_sample(s: Any, i: int) -> Any:
    """Index row ``i`` of a per-argument sample batch."""
    from ._numeric_record import NumericRecord
    from .record import Record

    if isinstance(s, Record):
        if len(s.fields) == 1:
            return s[s.fields[0]][i]
        return NumericRecord({f: s[f][i] for f in s.fields})
    return s[i]
