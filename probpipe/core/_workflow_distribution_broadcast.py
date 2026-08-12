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
from itertools import product as cartesian_product
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
    broadcast_args: Sequence[_workflow_call.WorkflowInputRef],
    n_broadcast_samples: int,
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
    broadcast_args : sequence of WorkflowInputRef
        Distribution-valued input slots to broadcast over.
    n_broadcast_samples : int
        Number of Monte Carlo rows to draw. Small positive values are accepted
        with a warning; non-integers and non-positive values raise.
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
    broadcast_args = list(broadcast_args)
    _validate_n_broadcast_samples(n_broadcast_samples)

    empirical_groups, sample_args, product_size = _split_empirical_args(
        values=values,
        broadcast_args=broadcast_args,
        n_broadcast_samples=n_broadcast_samples,
    )

    dispatch = resolve_dispatch(
        values,
        broadcast_args,
        jax_supported=not empirical_groups,
    )
    if requested_dispatch == "jax" and empirical_groups:
        raise ValueError(
            "dispatch='jax' does not support exact empirical enumeration; "
            "use dispatch='auto', 'sequential', or 'thread' for this path."
        )

    # Enumeration preserves exact empirical weights and must run in all row-wise
    # dispatch modes; otherwise cartesian-product semantics vary by dispatch.
    if empirical_groups:
        result = _broadcast_enumerate(
            func=func,
            values=values,
            empirical_groups=empirical_groups,
            sample_args=sample_args,
            product_size=product_size,
            n_broadcast_samples=n_broadcast_samples,
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
            broadcast_args=broadcast_args,
            n_broadcast_samples=n_broadcast_samples,
            get_key=get_key,
            workflow_name=workflow_name,
            workflow_kind=workflow_kind,
            output_template=output_template,
        )
    else:
        result = _broadcast_sample(
            func=func,
            values=values,
            broadcast_args=broadcast_args,
            n_broadcast_samples=n_broadcast_samples,
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


def _split_empirical_args(
    *,
    values: dict[str, Any],
    broadcast_args: Sequence[_workflow_call.WorkflowInputRef],
    n_broadcast_samples: int,
) -> tuple[
    tuple[tuple[EmpiricalDistribution, tuple[_workflow_call.WorkflowInputRef, ...]], ...],
    dict[_workflow_call.WorkflowInputRef, Distribution],
    int,
]:
    """Split the broadcast arguments into enumerable empirical groups and the rest.

    A **co-sampling group** is enumerated or sampled as a unit, never split, so
    the same empirical passed twice contributes one enumeration axis rather than
    a squared grid. A group is enumerable when its root is an empirical small
    enough to enumerate and every member *is* that root: a group holding a view
    goes to the sampling path whole, which keeps a parent and its view on one
    draw instead of enumerating one and sampling the other.

    Groups are enumerated smallest-first while the running product fits the
    budget; a group that does not fit falls through to sampling, where its
    members still co-sample.
    """
    candidates: list[tuple[EmpiricalDistribution, tuple[_workflow_call.WorkflowInputRef, ...]]] = []
    sample_args: dict[_workflow_call.WorkflowInputRef, Distribution] = {}
    for root, arg_refs in _workflow_plan.group_by_alignment(values=values, refs=broadcast_args):
        enumerable = (
            isinstance(root, EmpiricalDistribution)
            and root.num_atoms <= n_broadcast_samples
            and all(_workflow_call.input_ref_value(values, ref) is root for ref in arg_refs)
        )
        if enumerable:
            candidates.append((root, arg_refs))
        else:
            for ref in arg_refs:
                sample_args[ref] = _workflow_call.input_ref_value(values, ref)
    candidates.sort(key=lambda pair: pair[0].num_atoms)

    empirical_groups: list[
        tuple[EmpiricalDistribution, tuple[_workflow_call.WorkflowInputRef, ...]]
    ] = []
    product_size = 1
    for dist, arg_refs in candidates:
        if product_size * dist.num_atoms <= n_broadcast_samples:
            empirical_groups.append((dist, arg_refs))
            product_size *= dist.num_atoms
        else:
            for ref in arg_refs:
                sample_args[ref] = dist

    return tuple(empirical_groups), sample_args, product_size


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


def _sample_broadcast_args(
    values: dict[str, Any],
    broadcast_args: Sequence[_workflow_call.WorkflowInputRef],
    n: int,
    key: PRNGKey,
) -> dict[_workflow_call.WorkflowInputRef, Array]:
    """Sample all broadcast arguments, one joint draw per co-sampling group.

    Arguments are grouped by root ancestor, so the same distribution passed
    twice, sibling views of one parent, and a parent passed alongside its own
    view all fall in one group. Each group is drawn **once**, from its root, and
    every member takes its own value out of that draw — the root itself whole, a
    view through its field path. Dependence within a group therefore flows
    through the wrapped function instead of being broken by independent
    sampling, so ``f(d, d)`` approximates ``f(X, X)`` rather than ``f(X1, X2)``.

    Groups with no common root are drawn from separate subkeys, which samples the
    product of their laws.
    """
    sampled: dict[_workflow_call.WorkflowInputRef, Array] = {}
    for root, arg_refs in _workflow_plan.group_by_alignment(values=values, refs=broadcast_args):
        key, subkey = jax.random.split(key)
        drawn = root._sample(subkey, (n,))
        for ref in arg_refs:
            member = _workflow_call.input_ref_value(values, ref)
            sampled[ref] = drawn if member is root else _project_from_root(member, drawn)
    return sampled


def _project_from_root(view: Any, drawn: Any) -> Any:
    """A view's own draw, taken out of its root's joint draw.

    Not necessarily an array: a view of a field group projects the sub-record its
    path names, so this returns whatever the root's draw holds there.
    """
    if hasattr(view, "_extract"):
        return view._extract(drawn)
    projected = drawn
    for key_name in getattr(view, "_key_path", (view.field,)):
        projected = projected[key_name]
    return projected


def _broadcast_jax(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    broadcast_args: list[_workflow_call.WorkflowInputRef],
    n_broadcast_samples: int,
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

    key = get_key()
    sampled = _sample_broadcast_args(
        values,
        broadcast_args,
        n_broadcast_samples,
        key,
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
    empirical_groups: tuple[
        tuple[EmpiricalDistribution, tuple[_workflow_call.WorkflowInputRef, ...]], ...
    ],
    sample_args: dict[_workflow_call.WorkflowInputRef, Distribution],
    product_size: int,
    n_broadcast_samples: int,
    get_key: Callable[[], PRNGKey],
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
    output_template: EventTemplate | None,
) -> BroadcastDistribution:
    """Enumerate empirical distributions and sample any remaining inputs.

    One axis of the cartesian product per co-sampling group, so every reference in
    a group takes the same atom and contributes its weight once.
    """
    key = get_key()
    emp_dists = [dist for dist, _refs in empirical_groups]
    emp_refs = [arg_refs for _dist, arg_refs in empirical_groups]

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

    all_broadcast_args = [ref for arg_refs in emp_refs for ref in arg_refs] + sample_arg_names

    for combo in cartesian_product(*(range(d.num_atoms) for d in emp_dists)):
        emp_weight = 1.0
        for dist, i in zip(emp_dists, combo, strict=True):
            emp_weight *= float(dist.weights[i])

        for _ in range(reps_per_combo):
            replacements: dict[_workflow_call.WorkflowInputRef, Any] = {}

            for dist, arg_refs, i in zip(emp_dists, emp_refs, combo, strict=True):
                atom = _index_sample(dist.samples, i)
                for ref in arg_refs:
                    replacements[ref] = atom

            for ref in sample_args:
                replacements[ref] = _index_sample(sampled[ref], sample_idx)

            weights.append(emp_weight / reps_per_combo)
            call_value_list.append(_workflow_call.replace_input_refs(values, replacements))
            sample_idx += 1

    request = _workflow_execution.WorkflowExecutionRequest(
        func=func,
        call_value_list=call_value_list,
        execution=make_execution_config(),
    )
    results = _workflow_execution.execute_many(request)

    all_input_samples = {
        ref.label: _stack_rows(
            [_workflow_call.input_ref_value(call_values, ref) for call_values in call_value_list],
            arg_name=ref.label,
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
    broadcast_args: list[_workflow_call.WorkflowInputRef],
    n_broadcast_samples: int,
    get_key: Callable[[], PRNGKey],
    make_execution_config: Callable[
        [],
        _workflow_execution.WorkflowExecutionConfig,
    ],
    output_template: EventTemplate | None,
) -> BroadcastDistribution:
    """Sample distribution arguments and execute one function call per sample."""
    key = get_key()
    samples_per_arg = _sample_broadcast_args(
        values,
        broadcast_args,
        n_broadcast_samples,
        key,
    )

    call_value_list = []
    for i in range(n_broadcast_samples):
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


def _stack_rows(rows: list[Any], *, arg_name: str) -> Any:
    """Stack one argument's per-row values into a single batched value.

    ``jnp.stack`` covers array-valued rows. A record-valued row is not an array
    — a ``Record`` has fields, not a shape — so those stack through
    ``RecordArray.stack``, giving a batch whose ``batch_shape`` is ``(n,)`` and
    whose fields are the per-row leaves.
    """
    from ._record_array import RecordArray
    from .record import Record

    if rows and isinstance(rows[0], Record):
        # Checked rather than caught: ``stack`` refuses a nested template, but it
        # refuses other things too, and attributing every refusal to nesting
        # would advise flattening a record that is already flat. A leaf path
        # differs from a child name exactly when the record nests.
        if tuple(rows[0].keys()) != rows[0].fields:
            raise TypeError(
                f"lifting {arg_name!r} would batch a record with nested fields, which a record "
                f"batch does not represent yet; flatten the law's fields, or pass the nested "
                f"parts as separate arguments"
            )
        return RecordArray.stack(rows)
    return jnp.stack(rows)


def _index_sample(s: Any, i: int) -> Any:
    """Index row ``i`` of a per-argument sample batch."""
    from ._record_batch import RecordBatch
    from .record import Record

    if isinstance(s, (Record, RecordBatch)):
        # Index each leaf field's batch row; rebuild by path key so a nested
        # sample is reconstructed with its structure intact. A row of a batch is
        # a single record, so the rebuild is the same either way.
        leaf_paths = tuple(s.event_template.keys())
        if len(leaf_paths) == 1:
            return s[leaf_paths[0]][i]
        return Record(s.name, {p: s[p][i] for p in leaf_paths}, name_is_auto=True)
    return s[i]
