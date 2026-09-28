"""Function decorator and the installed workflow call engine."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import partial
from typing import Any, overload

import jax
import jax.numpy as jnp

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from ..core._batch import Batch
from ..core._numeric_record_batch import NumericRecordBatch
from ..core._record_batch import RecordBatch
from ..core._record_distribution import RecordDistribution
from ..core._specs import NumericArraySpec, NumericRecordSpec, RecordSpec
from ..core.config import ProvenanceMode, WorkflowKind, prefect_config
from ..core.node import Node
from ..core.provenance import Provenance
from ..core.tracked import TrackedTerm
from ..values._function_base import (
    Function,
    _bind_function_inputs,
    _FunctionInvocationContext,
    _validate_function_output,
    install_call_engine,
)
from . import _broadcast as _workflow_distribution_broadcast
from . import _broker as _workflow_broker
from . import _call as _workflow_call
from . import _callable as _workflow_callable
from . import _context as _workflow_context
from . import _execution as _workflow_execution
from . import _execution_contract as _workflow_execution_contract
from . import _normalization as _workflow_distribution_normalization
from . import _plan as _workflow_plan
from . import _recipe as _workflow_recipe
from . import _replay as _workflow_replay
from . import _result as _workflow_result
from . import _sweep as _workflow_sweep
from ._contract import _bind_planned_function_inputs
from ._result import _output_record_spec, _wrap_declared_function_output

logger = logging.getLogger(__name__)


class _UnvectorizableBatchSignal(Exception):
    """The probe met a batch whose rows the mapped body cannot be built from.

    Raised so the reason survives to the caller: the generic tracing message
    would say the function is not JAX-traceable, which is not what went wrong.
    """

    def __init__(self, kinds: list[str]) -> None:
        super().__init__(", ".join(kinds))
        self.kinds = kinds


@overload
def function(_func: Callable[..., Any], /, **kwargs: Any) -> Function: ...


@overload
def function(
    _func: None = None,
    /,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Function]: ...


def function(
    _func: Callable[..., Any] | None = None,
    /,
    **kwargs: Any,
) -> Function | Callable[[Callable[..., Any]], Function]:
    """Decorator to create a :class:`Function` from a plain function.

    Bare usage wraps a function with default ``Function`` controls::

        @function
        def my_func(x, y):
            return x + y

    Pass keyword arguments to configure ProbPipe controls at definition time::

        @function(n_broadcast_samples=100, dispatch="sequential")
        def my_func(x, y):
            return x + y

    Keyword arguments passed later to the workflow call itself belong to the
    wrapped function whenever they can bind to that function. Use
    ``workflow.with_options(...)(...)`` for one-call ProbPipe controls.

    Parameters
    ----------
    _func : Callable or None
        Function being decorated for bare ``@function`` usage.
        Users should not pass this argument by keyword.
    **kwargs : Any
        Construction-time ``Function`` controls and declarations such as
        ``dispatch``, ``n_broadcast_samples``, ``include_inputs``,
        ``workflow_kind``, ``input_spec``, ``output_spec``, and ``output_name``.

    Returns
    -------
    Function or Callable
        Wrapped Function for bare usage, or a decorator when called
        with parentheses.
    """

    def decorator(func: Callable[..., Any]) -> Function:
        options = dict(kwargs)
        return Function(options.pop("name", func.__name__), func, **options)

    if _func is not None:
        return decorator(_func)
    return decorator


def effective_workflow_kind(function: Function) -> WorkflowKind:
    """Resolve the effective orchestration mode for this instance.

    Per-instance ``workflow_kind`` takes precedence over the global config.
    ``DEFAULT`` means "defer"; if both levels are ``DEFAULT``, orchestration is
    disabled. Prefect orchestration is opt-in.

    If ``TASK`` or ``FLOW`` is requested but Prefect is unavailable, the mode
    falls back to ``OFF``.
    """
    raw = function.options["workflow_kind"]

    kind = raw if raw is not WorkflowKind.DEFAULT else prefect_config.workflow_kind

    if kind is WorkflowKind.DEFAULT:
        kind = WorkflowKind.OFF

    _PREFECT_KINDS = {WorkflowKind.TASK, WorkflowKind.FLOW}
    prefect_missing = task is None or flow is None
    if kind in _PREFECT_KINDS and prefect_missing:
        warnings.warn(
            f"workflow_kind={kind!r} requested but Prefect is not installed. "
            "Falling back to OFF. Install with: pip install probpipe[prefect]",
            stacklevel=2,
        )
        return WorkflowKind.OFF

    return kind


def _make_execution_config(
    function: Function,
    *,
    mode: _workflow_execution.WorkflowExecutionMode | None = None,
) -> _workflow_execution.WorkflowExecutionConfig:
    """Build resolved execution metadata for row-wise call dispatch."""
    if mode is None:
        match effective_workflow_kind(function):
            case WorkflowKind.TASK:
                mode = "prefect_task"
            case WorkflowKind.FLOW:
                mode = "prefect_flow"
            case _ if function.options["dispatch"] == "thread":
                mode = "thread"
            case _:
                mode = "sequential"

    is_prefect = mode in ("prefect_task", "prefect_flow")

    if is_prefect and (
        function.options["dispatch"] == "thread" or function.options["max_workers"] is not None
    ):
        warnings.warn(
            "dispatch='thread' and max_workers configure only local "
            "ThreadPoolExecutor dispatch; they do not control Prefect "
            "scheduling.",
            stacklevel=2,
        )

    return _workflow_execution.WorkflowExecutionConfig(
        mode=mode,
        max_workers=function.options["max_workers"] if mode == "thread" else None,
        name=function._name,
        prefect_task_runner=(prefect_config.resolve_task_runner() if is_prefect else None),
    )


def _call_with_options(
    function: Function,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    options: _workflow_call.WorkflowCallOptions,
) -> Any:
    _workflow_context._assert_workflow_admission()
    with _workflow_replay._function_replay_scope() as replay_call:
        occurrence_path = None if replay_call is None else replay_call.occurrence_path
        with (
            _workflow_context._ephemeral_workflow_run(),
            _workflow_broker._function_stochastic_scope(occurrence_path=occurrence_path) as broker,
        ):
            if (
                replay_call is not None
                or _workflow_context._active_provenance_mode() is not ProvenanceMode.OFF
            ):
                anchor = _workflow_callable.capture_function_anchor(function)
                broker.set_callable_anchor(anchor)
                if replay_call is not None:
                    replay_call.validate_callable(anchor)
            return _call_with_options_in_context(function, args, call_inputs, options)


def _call_with_options_in_context(
    function: Function,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    options: _workflow_call.WorkflowCallOptions,
) -> Any:
    call = _workflow_call.resolve_workflow_call(
        function._signature_info,
        args,
        call_inputs,
        bind=function._bind,
        module=function._module,
        dependency_type=Node,
        workflow_name=function._name,
        default_n_broadcast_samples=function.options["n_broadcast_samples"],
        default_include_inputs=function.options["include_inputs"],
        options=options,
    )

    values = _workflow_distribution_normalization.normalize_distribution_values(
        values=call.values,
        signature_info=function._signature_info,
    )
    broadcast_plan = _workflow_plan.build_broadcast_plan(
        values=values,
        signature_info=function._signature_info,
    )
    stochastic_plan = _workflow_plan.build_stochastic_plan(
        values,
        broadcast_plan,
        call.overrides.n_broadcast_samples,
    )
    stochastic_sample_shape = None if stochastic_plan is None else stochastic_plan.sample_shape

    def get_key(event: _workflow_plan.PlannedRandomEvent):
        return _workflow_broker._resolve_automatic_key(
            None,
            _workflow_broker.StochasticEffectPlan(
                operation_kind="function_lifting",
                execution_mode="sampled",
                event=event,
                sample_shape=stochastic_sample_shape,
                sampling_abi="probpipe.distribution_sampling/v1",
                provider_abi="probpipe.distribution/v1",
            ),
        )

    workflow_kind = effective_workflow_kind(function)
    _workflow_broker._record_active_requested_execution(
        function.options["dispatch"],
        workflow_kind.value,
    )
    _workflow_replay._validate_active_plan(
        _workflow_recipe.serialize_stochastic_plan(stochastic_plan)
    )
    _, invocation_bindings = _bind_planned_function_inputs(
        function_name=function._name,
        input_spec=function.input_spec,
        values=values,
        lifted_names={
            ref.parameter_name for ref in (*broadcast_plan.dist_args, *broadcast_plan.array_args)
        },
    )
    concrete_output_spec = (
        function.output_spec.with_dim_sizes(**invocation_bindings)
        if function.output_spec is not None
        else None
    )
    concrete_output_template = (
        _output_record_spec(concrete_output_spec) if concrete_output_spec is not None else None
    )
    provenance_parents: list[TrackedTerm] = [function]
    provenance_inputs: dict[str, Any] = {}
    seen_parent_ids = {id(function)}
    for ref in _workflow_call.iter_input_refs(function._signature_info, values):
        value = _workflow_call.input_ref_value(values, ref)
        if isinstance(value, TrackedTerm) and id(value) not in seen_parent_ids:
            seen_parent_ids.add(id(value))
            provenance_parents.append(value)
        elif not isinstance(value, TrackedTerm):
            provenance_inputs[ref.label] = value

    def invoke_point(**point_values: Any) -> Any:
        _, point_bindings = _bind_function_inputs(
            function_name=function._name,
            input_spec=function.input_spec,
            values=point_values,
            bindings=invocation_bindings,
        )
        context = _FunctionInvocationContext(point_bindings)
        result = function._invoke_resolved(point_values, context=context)
        point_output_spec = _validate_function_output(
            function_name=function._name,
            output_spec=function.output_spec,
            result=result,
            bindings=context.dimension_bindings,
        )
        if point_output_spec is not None:
            result = _wrap_declared_function_output(
                result,
                function_name=function.output_name,
                output_spec=point_output_spec,
            )
        return result

    resolved_dispatch: str | None = None

    def resolve_dispatch(
        dispatch_values: dict[str, Any],
        broadcast_args: list[_workflow_call.WorkflowInputRef],
        *,
        jax_supported: bool = True,
    ) -> str:
        nonlocal resolved_dispatch
        if function.options["dispatch"] != "auto" or not jax_supported:
            return _resolve_dispatch(
                function,
                dispatch_values,
                broadcast_args,
                jax_supported=jax_supported,
                func=invoke_point,
                stochastic_plan=stochastic_plan,
            )
        if resolved_dispatch is None:
            resolved_dispatch = _resolve_dispatch(
                function,
                dispatch_values,
                broadcast_args,
                jax_supported=True,
                func=invoke_point,
                stochastic_plan=stochastic_plan,
            )
        return resolved_dispatch

    def require_jax_traceable(
        dispatch_values: dict[str, Any],
        broadcast_args: list[_workflow_call.WorkflowInputRef],
    ) -> None:
        _require_jax_traceable(
            function,
            dispatch_values,
            broadcast_args,
            func=invoke_point,
            stochastic_plan=stochastic_plan,
        )

    def execute_distribution_broadcast(
        *,
        row_values: dict[str, Any],
        plan: _workflow_plan.StochasticPlan,
        logical_unit: _workflow_plan.LogicalUnit,
        include_inputs: bool = call.overrides.include_inputs,
        record_recipe: bool = True,
    ):
        return _workflow_distribution_broadcast.execute_distribution_broadcast(
            func=invoke_point,
            values=row_values,
            stochastic_plan=plan,
            logical_unit=logical_unit,
            include_inputs=include_inputs,
            get_key=get_key,
            make_execution_config=partial(_make_execution_config, function),
            requested_dispatch=function.options["dispatch"],
            resolve_dispatch=resolve_dispatch,
            require_jax_traceable=require_jax_traceable,
            workflow_name=function._name,
            output_name=function.output_name,
            output_spec=concrete_output_spec,
            workflow_kind=workflow_kind,
            output_template=concrete_output_template,
            provenance_parents=provenance_parents,
            provenance_inputs=provenance_inputs,
            record_recipe=record_recipe,
        )

    if broadcast_plan.regime == "distribution":
        if stochastic_plan is None:  # pragma: no cover - planner contract guard
            raise RuntimeError("distribution broadcast is missing its stochastic plan")
        return execute_distribution_broadcast(
            row_values=values,
            plan=stochastic_plan,
            logical_unit=stochastic_plan.logical_units[0],
        )
    if broadcast_plan.regime in ("sweep", "nested"):

        def distribution_broadcast(
            row_values: dict[str, Any],
            plan: _workflow_plan.StochasticPlan,
            logical_unit: _workflow_plan.LogicalUnit,
            include_inputs: bool,
        ):
            return execute_distribution_broadcast(
                row_values=row_values,
                plan=plan,
                logical_unit=logical_unit,
                include_inputs=include_inputs,
                record_recipe=False,
            )

        return _workflow_sweep.execute_sweep(
            func=invoke_point,
            values=values,
            plan=broadcast_plan,
            stochastic_plan=stochastic_plan,
            make_execution_config=partial(_make_execution_config, function),
            requested_dispatch=function.options["dispatch"],
            resolve_dispatch=resolve_dispatch,
            require_jax_traceable=require_jax_traceable,
            distribution_broadcast=distribution_broadcast,
            workflow_name=function._name,
            output_name=function.output_name,
            output_spec=concrete_output_spec,
            include_inputs=call.overrides.include_inputs,
            output_template=concrete_output_template,
            provenance_parents=provenance_parents,
            provenance_inputs=provenance_inputs,
            workflow_kind=workflow_kind,
        )

    # Non-broadcast call — one function invocation, then wrap. TrackedTerm
    # values form lineage parents; every other resolved parameter remains
    # separately fingerprinted in Provenance.inputs.
    # Known harmless duplication: the distribution-broadcast module builds
    # the same request shape. A later execution cleanup can centralize this
    # without reintroducing private facade wrappers.
    execution = _make_execution_config(function)
    request = _workflow_execution.WorkflowExecutionRequest(
        func=invoke_point,
        work_items=_workflow_execution.make_managed_work_items(
            [values],
            unit_segments=(_workflow_execution.point_unit_segment(),),
        ),
        execution=execution,
        contract=_workflow_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport=_workflow_execution_contract.transport_for_execution_mode(execution.mode),
            stochastic_plan=None,
        ),
    )
    result = _workflow_execution.execute_many(request)[0]
    name = function._name
    controls, diagnostics = _workflow_recipe.provenance_recipe_fields(None)
    provenance = Provenance.create(
        f"workflow.{name}",
        parents=provenance_parents,
        metadata={"func": name},
        inputs=provenance_inputs,
        controls=controls,
        diagnostics=diagnostics,
    )
    return _workflow_result._coerce_output(
        result,
        broadcast_mode=_workflow_result.BROADCAST_WRAP,
        provenance=provenance,
        field_name=function.output_name,
    )


def _jax_traceability_error(
    function: Function,
    values: dict[str, Any],
    broadcast_args: list[_workflow_call.WorkflowInputRef],
    *,
    func: Callable[..., Any],
    stochastic_plan: _workflow_plan.StochasticPlan | None,
) -> Exception | None:
    """Return the JAX trace-probe error for the current call, if any.

    The probe traces the operation the dispatch is choosing, not merely the
    body: a body can trace cleanly bare yet be impossible under the
    transform its executor applies — one that returns a batch, whose added
    axis no level can name. So every executor that maps is probed under a
    map, each argument fed the way its own executor will feed it: a
    batched-record argument over one row, a distribution over one draw. A
    probe failure means sequential dispatch, which is always able to run the
    call — the two paths agree on results by contract, so falling back costs
    speed, never correctness.
    """
    try:
        dummy_kw = dict(values)
        broadcast_refs = set(broadcast_args)
        batched_sources: dict[_workflow_call.WorkflowInputRef, Any] = {}
        unvectorized_batches: dict[Any, Any] = {}
        drawn_refs: list[_workflow_call.WorkflowInputRef] = []
        for ref in _workflow_call.iter_input_refs(function._signature_info, values):
            v = _workflow_call.input_ref_value(values, ref)
            if ref in broadcast_refs:
                # Batched-record input: take row 0 so the dummy call
                # sees what an inner sweep iteration will actually
                # receive.
                if isinstance(v, RecordBatch):
                    batched_sources[ref] = v
                    dummy_kw = _workflow_call.replace_input_ref(dummy_kw, ref, v[0])
                elif isinstance(v, Batch):
                    # A batch that is not a batch of records is still a swept
                    # source, not a draw. The probe's synthesis below builds a
                    # leaf per field, which a single-store batch does not have,
                    # so this route declines rather than mis-reading it as a
                    # law: ``auto`` runs it sequentially and gets the right
                    # answer, which an explicit ``jax`` should not silently
                    # differ from.
                    unvectorized_batches[ref] = v
                else:
                    # Nothing to synthesize: the draw itself supplies the
                    # structure and dtype below, which is what lets a
                    # multi-field law be probed at all — it has no single
                    # event shape to stand in for one.
                    drawn_refs.append(ref)
            else:
                if isinstance(v, jnp.ndarray):
                    replacement = v
                elif hasattr(v, "__array__"):
                    replacement = jnp.asarray(v)
                else:
                    replacement = v
                dummy_kw = _workflow_call.replace_input_ref(dummy_kw, ref, replacement)
        with _workflow_context._workflow_probe():
            if unvectorized_batches:
                raise _UnvectorizableBatchSignal(
                    sorted({type(b).__name__ for b in unvectorized_batches.values()})
                )
            if batched_sources:
                refs = list(batched_sources)
                # The executor's own body, not a copy maintained in the
                # probe. ``dummy_kw`` already carries non-batched inputs.
                _row_call = _workflow_sweep.mapped_row_body(
                    func=func,
                    values=dummy_kw,
                    array_args=refs,
                    field_name=function._name,
                    output_is_declared=function.output_spec is not None,
                )

                probe_leaves = []
                for source in batched_sources.values():
                    n_batch = len(source.batch_shape)
                    # The flat size is stated, exactly as the executor states it:
                    # a ``-1`` cannot be inferred over a zero-width event, which
                    # is a shape the real reshape handles.
                    n_rows = int(math.prod(source.batch_shape))
                    probe_leaves.append(
                        {
                            leaf: jnp.reshape(
                                jnp.asarray(source[leaf]),
                                (n_rows, *jnp.shape(source[leaf])[n_batch:]),
                            )[:1]
                            for leaf in source.event_template
                        }
                    )
                jax.make_jaxpr(jax.vmap(_row_call))(tuple(probe_leaves))
            elif drawn_refs:
                if stochastic_plan is None:  # pragma: no cover - planner contract guard
                    raise RuntimeError("distribution probe is missing its stochastic plan")
                refs = drawn_refs
                _draw_call = _workflow_distribution_broadcast.mapped_draw_body(
                    func=func, values=dummy_kw, broadcast_args=refs
                )
                sampled_groups = tuple(
                    group
                    for group in stochastic_plan.source_groups
                    if group.execution_mode == "sampled"
                )
                root_probes = []
                for group in sampled_groups:
                    binding = stochastic_plan.runtime_bindings[group.index]
                    root = binding.root
                    from ..core._specs import _components_record

                    template = _components_record(root.event_spec)
                    if not isinstance(template, NumericRecordSpec) or not template.is_concrete:
                        raise TypeError(
                            f"{type(root).__name__} does not declare a concrete numeric "
                            "event template for side-effect-free JAX probing"
                        )
                    try:
                        dtypes = root.dtypes
                    except (AttributeError, NotImplementedError) as error:
                        raise TypeError(
                            f"{type(root).__name__} does not declare field dtypes for "
                            "side-effect-free JAX probing"
                        ) from error
                    columns = {}
                    for path in template:
                        dtype = dtypes.get(path)
                        if dtype is None:
                            dtype = dtypes[path.split("/", 1)[0]]
                        columns[path] = jax.ShapeDtypeStruct(
                            (1, *template[path].shape),
                            dtype,
                        )
                    if isinstance(root, RecordDistribution):
                        root_probe = NumericRecordBatch(
                            root.name,
                            columns,
                            "draw",
                            element_spec=template,
                            axes_per_level=(1,),
                        )
                    else:
                        root_probe = next(iter(columns.values()))
                    root_probes.append(root_probe)

                def probe_draw(root_values):
                    sampled = {}
                    for group, root_value in zip(
                        sampled_groups,
                        root_values,
                        strict=True,
                    ):
                        binding = stochastic_plan.runtime_bindings[group.index]
                        for consumer, evaluate in zip(
                            group.consumers,
                            binding.consumer_evaluators,
                            strict=True,
                        ):
                            sampled[consumer.arg_ref] = evaluate(root_value)
                    return _draw_call(tuple(sampled[ref] for ref in refs))

                jax.make_jaxpr(jax.vmap(probe_draw))(tuple(root_probes))
            else:
                jax.make_jaxpr(lambda kw: func(**kw))(dummy_kw)
    except Exception as exc:
        return exc
    return None


def _has_output_support(spec: Any) -> bool:
    from ..core._batch import BatchSpec

    if isinstance(spec, RecordSpec):
        return any(_has_output_support(child) for child in spec.values())
    if isinstance(spec, BatchSpec):
        return _has_output_support(spec.element_spec)
    return isinstance(spec, NumericArraySpec) and spec.support is not None


def _require_jax_traceable(
    function: Function,
    values: dict[str, Any],
    broadcast_args: list[_workflow_call.WorkflowInputRef],
    *,
    func: Callable[..., Any],
    stochastic_plan: _workflow_plan.StochasticPlan | None,
) -> None:
    """Raise a clear error if explicit JAX dispatch cannot trace."""
    output = function.output_spec.spec if function.output_spec is not None else None
    if _has_output_support(output):
        raise ValueError(
            "dispatch='jax' cannot validate output_spec support constraints "
            "during JAX tracing; use dispatch='auto', 'sequential', or 'thread'."
        )
    trace_error = _jax_traceability_error(
        function, values, broadcast_args, func=func, stochastic_plan=stochastic_plan
    )
    if trace_error is None:
        return
    if isinstance(trace_error, _UnvectorizableBatchSignal):
        raise TypeError(
            f"dispatch='jax' cannot vectorize over {', '.join(trace_error.kinds)}: the "
            f"mapped row body is built from one leaf per field, which a single-store batch "
            f"does not have. Use dispatch='auto' or 'sequential', which sweep it correctly."
        ) from trace_error
    if isinstance(trace_error, _workflow_context._StochasticProbeSignal):
        raise TypeError(
            "dispatch='jax' cannot execute a wrapped function that requests "
            "workflow-owned randomness with key=None. Pass an explicit key, "
            "or use dispatch='auto', 'sequential', or 'thread'."
        ) from trace_error
    raise ValueError(
        "dispatch='jax' failed while tracing the wrapped function with JAX; "
        "ensure the function is JAX-traceable, or use dispatch='auto', "
        "'sequential', or 'thread'."
    ) from trace_error


def _resolve_dispatch(
    function: Function,
    values: dict[str, Any],
    broadcast_args: list[_workflow_call.WorkflowInputRef],
    *,
    jax_supported: bool = True,
    func: Callable[..., Any],
    stochastic_plan: _workflow_plan.StochasticPlan | None,
) -> str:
    """Resolve the dispatch strategy, caching JAX traceability detection.

    Returns ``"jax"``, ``"sequential"``, or ``"thread"``. This is
    independent of orchestration (``workflow_kind``), which wraps
    whichever strategy is chosen.
    """
    if function.options["dispatch"] != "auto":
        return function.options["dispatch"]

    if not jax_supported:
        return "sequential"

    if (
        _jax_traceability_error(
            function, values, broadcast_args, func=func, stochastic_plan=stochastic_plan
        )
        is None
    ):
        return "jax"
    else:
        logger.info(
            "Function '%s' is not JAX-traceable; using sequential dispatch.",
            function._name,
        )
        return "sequential"


def _call_engine(function: Function, *args: Any, **kwargs: Any) -> Any:
    return _call_with_options(function, args, kwargs, _workflow_call.WorkflowCallOptions())


@contextmanager
def _apply_scope() -> Generator[None, None, None]:
    """Preserve workflow admission and RNG ownership around raw evaluation."""
    _workflow_context._assert_workflow_admission()
    _workflow_replay._reject_function_apply()
    with (
        _workflow_context._ephemeral_workflow_run(),
        _workflow_broker._function_stochastic_scope(),
    ):
        yield


install_call_engine(_call_engine, apply_scope=_apply_scope)
