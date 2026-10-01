"""Function decorator and the installed workflow call engine.

The engine runs the call stack of design Part V on every call of a Function:

1. configure: the controls a Function carries, resolved by ``with_options``;
2. bind: the arguments bind to the signature (:mod:`._call`);
3. normalize: distribution arguments are converted where their parameter names
   another class (:mod:`._normalization`) and each argument is admitted by its
   role or its declaration (:func:`._call.admit_arguments`);
4. classify the lift: the lifted arguments are found and grouped
   (:func:`._plan.build_broadcast_plan`, :func:`._plan.build_stochastic_plan`);
5. plan: the input declarations unify and the output declaration binds the
   shared dimensions (:func:`._contract._bind_planned_function_inputs`), or, for
   a Function realized by routes, each point's result rule runs
   (``Function._plan_point``);
6. resolve: a lifted call resolves through the evaluation-rule registry
   (:mod:`._rules`), and each point runs the body or the route selected among
   the Function's routes (:mod:`._resolution`);
7. execute: the points run under the dispatch mode with structural keys
   (:mod:`._broadcast`, :mod:`._sweep`, :mod:`._execution`);
8. return: the result is validated, wrapped, labeled, and given provenance
   recording the selected route, or detached under ``raw=True``
   (:mod:`._result`).
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import Any, overload

import jax
import jax.numpy as jnp

from ..values import _binding

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from ..core._batch import Batch
from ..core._dispatch import MethodInfo, ResolutionError
from ..core._numeric_record_batch import NumericRecordBatch
from ..core._record_batch import RecordBatch
from ..core._spec_base import NumericSpec, TermSpec
from ..core._specs import InputSpec, NumericArraySpec, NumericRecordSpec, OutputSpec, RecordSpec
from ..core.config import ProvenanceMode, WorkflowKind, prefect_config
from ..core.node import Node
from ..core.provenance import Provenance
from ..core.tracked import TrackedTerm
from ..values._function_base import (
    Function,
    _bind_function_inputs,
    _FunctionInvocationContext,
    _refuse_unknown_controls,
    _validate_function_output,
    install_call_engine,
)
from . import (
    _broadcast,
    _broker,
    _call,
    _callable,
    _context,
    _execution,
    _execution_contract,
    _normalization,
    _plan,
    _recipe,
    _replay,
    _resolution,
    _result,
    _rules,
    _sweep,
)
from ._call import CallReport
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
    *,
    name: str | None = None,
    input_spec: InputSpec | Mapping[str, TermSpec] | None = None,
    output_spec: OutputSpec | TermSpec | None = None,
    output_name: str | None = None,
    differentiable: NumericSpec | None = None,
    bind: Mapping[str, Any] | None = None,
    module: Any | None = None,
    **controls: Any,
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
    name : str or None
        The function label, defaulting to the decorated callable's ``__name__``.
    input_spec, output_spec, output_name, differentiable, bind, module
        The declarations and construction bindings :class:`Function` takes.
    **controls : Any
        The engine's controls, which :class:`Function` lists.

    Returns
    -------
    Function or Callable
        Wrapped Function for bare usage, or a decorator when called
        with parentheses.

    Raises
    ------
    TypeError
        If a keyword is no control, before any callable is wrapped.
    """
    _refuse_unknown_controls(controls)

    def decorator(func: Callable[..., Any]) -> Function:
        return Function(
            func.__name__ if name is None else name,
            func,
            input_spec=input_spec,
            output_spec=output_spec,
            output_name=output_name,
            differentiable=differentiable,
            bind=bind,
            module=module,
            **controls,
        )

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
    mode: _execution.WorkflowExecutionMode | None = None,
) -> _execution.WorkflowExecutionConfig:
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

    return _execution.WorkflowExecutionConfig(
        mode=mode,
        max_workers=function.options["max_workers"] if mode == "thread" else None,
        name=function._name,
        prefect_task_runner=(prefect_config.resolve_task_runner() if is_prefect else None),
    )


def _call_with_options(
    function: Function,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    options: _call.WorkflowCallOptions,
) -> Any:
    _context._assert_workflow_admission()
    with _replay._function_replay_scope() as replay_call:
        occurrence_path = None if replay_call is None else replay_call.occurrence_path
        with (
            _context._ephemeral_workflow_run(),
            _broker._function_stochastic_scope(occurrence_path=occurrence_path) as broker,
        ):
            if (
                replay_call is not None
                or _context._active_provenance_mode() is not ProvenanceMode.OFF
            ):
                anchor = _callable.capture_function_anchor(function)
                broker.set_callable_anchor(anchor)
                if replay_call is not None:
                    replay_call.validate_callable(anchor)
            return _call_with_options_in_context(function, args, call_inputs, options)


def _call_with_options_in_context(
    function: Function,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    options: _call.WorkflowCallOptions,
) -> Any:
    # A call made while a check probes, as a probe that runs an operation makes
    # one, selects and runs its routes as any call does.
    token = _call._CHECKING.set(False)
    try:
        return _run_call(function, args, call_inputs, options)
    finally:
        _call._CHECKING.reset(token)


def _realized_point(
    function: Function,
    values: Mapping[str, Any],
    controls: Mapping[str, Any],
    candidates: tuple[Any, ...],
) -> Any:
    """One point of a call realized by the route selected among *candidates*, as a term.

    The point is planned, its route selected and run, and the raw result
    validated against the point's declaration and wrapped at the kind it names.

    Raises
    ------
    ApplicabilityError
        If an applicability condition fails.
    ResolutionError
        If no candidate is feasible, or the first one that is not is unresolved.
    ResultSchemaError
        If the result does not satisfy the declaration.
    """
    point, result, _ = function._plan_point(values, controls)
    candidate, report = _resolution.selected(function.name, controls, candidates, point, result)
    value = candidate.run(point, result, report)
    return _result.declared_term(value, result, function.output_name)


def _run_call(
    function: Function,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    options: _call.WorkflowCallOptions,
) -> Any:
    call = _call.resolve_workflow_call(
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

    if function.options["conversions"]:
        raise NotImplementedError("Function.__call__: the conversions control")
    values = _normalization.normalize_distribution_values(
        values=call.values,
        signature_info=function._signature_info,
    )
    _call.admit_arguments(
        function._signature_info,
        values,
        input_spec=function.input_spec,
        function_name=function._name,
        roles=function._roles,
    )
    broadcast_plan = _plan.build_broadcast_plan(
        values=values,
        signature_info=function._signature_info,
        roles=function._roles,
    )
    controls = function.options
    candidates = function._route_candidates(controls)
    selection: tuple[Any, OutputSpec | None, Any, Any] | None = None
    if candidates is not None and broadcast_plan.regime == "none":
        # The routes realize the one point, so the selection is the call's route.
        point, result, _ = function._plan_point(values, controls)
        candidate, report = _resolution.selected(function.name, controls, candidates, point, result)
        selection = (point, result, candidate, report)
        route = _Route(
            candidate.route_name,
            candidate.exactness(report),
            method=candidate.method_of(report),
        )
    else:
        # A Function's routes take the method control, so a lifted call selects
        # its evaluation rule by rank.
        rule_method = None if candidates is not None else controls["method"]
        route = _resolve_route(function, values, broadcast_plan, call.overrides, rule_method)
    stochastic_plan = _plan.build_stochastic_plan(
        values,
        broadcast_plan,
        call.overrides.n_broadcast_samples,
        enumerate_every_group=route.name != "sampling_lift",
    )
    stochastic_sample_shape = None if stochastic_plan is None else stochastic_plan.sample_shape

    def get_key(event: _plan.PlannedRandomEvent):
        return _broker._resolve_automatic_key(
            None,
            _broker.StochasticEffectPlan(
                operation_kind="function_lifting",
                execution_mode="sampled",
                event=event,
                sample_shape=stochastic_sample_shape,
                sampling_abi="probpipe.distribution_sampling/v1",
                provider_abi="probpipe.distribution/v1",
            ),
        )

    workflow_kind = effective_workflow_kind(function)
    _broker._record_active_requested_execution(
        function.options["dispatch"],
        workflow_kind.value,
    )
    _replay._validate_active_plan(_recipe.serialize_stochastic_plan(stochastic_plan))
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
    for ref in _binding.iter_input_refs(function._signature_info, values):
        value = _binding.input_ref_value(values, ref)
        if isinstance(value, TrackedTerm) and id(value) not in seen_parent_ids:
            seen_parent_ids.add(id(value))
            provenance_parents.append(value)
        elif not isinstance(value, TrackedTerm):
            provenance_inputs[ref.label] = value

    # The selected route's result, when it carries a record of its own, which the
    # call's record keeps as a parent.
    route_records: list[TrackedTerm] = []

    def invoke_point(**point_values: Any) -> Any:
        if selection is not None:
            point, result, candidate, report = selection
            value = candidate.run(point, result, report)
            if (
                isinstance(value, TrackedTerm)
                and value.provenance is not None
                and id(value) not in seen_parent_ids
            ):
                route_records.append(value)
            return _result.declared_term(value, result, function.output_name)
        if candidates is not None:
            return _realized_point(function, point_values, controls, candidates)
        try:
            _, point_bindings = _bind_function_inputs(
                function_name=function._name,
                input_spec=function.input_spec,
                values=point_values,
                bindings=invocation_bindings,
            )
        except ValueError as error:
            raise _call.ApplicabilityError(str(error)) from error
        context = _FunctionInvocationContext(point_bindings)
        result = _result._batch_from_declared_sequence(
            function._invoke_resolved(point_values, context=context),
            function_name=function.output_name,
            output_spec=function.output_spec,
        )
        try:
            point_output_spec = _validate_function_output(
                function_name=function._name,
                output_spec=function.output_spec,
                result=result,
                bindings=context.dimension_bindings,
            )
        except ValueError as error:
            raise _result.ResultSchemaError(str(error)) from error
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
        broadcast_args: list[_binding.WorkflowInputRef],
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
        broadcast_args: list[_binding.WorkflowInputRef],
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
        plan: _plan.StochasticPlan,
        logical_unit: _plan.LogicalUnit,
        include_inputs: bool = call.overrides.include_inputs,
        record_recipe: bool = True,
        route_metadata: Mapping[str, Any] = route.metadata,
    ):
        return _broadcast.execute_distribution_broadcast(
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
            route=route_metadata,
        )

    if route.rule is not None and route.name not in _rules._ENGINE_RULES:
        return _run_registered_rule(function, route, provenance_parents, provenance_inputs)
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
            plan: _plan.StochasticPlan,
            logical_unit: _plan.LogicalUnit,
            include_inputs: bool,
        ):
            # Each element's lift is realized as its own plan says.
            inner = _Route(
                "empirical_enumeration" if plan.evaluation_mode == "exact" else "sampling_lift",
                plan.evaluation_mode == "exact",
            )
            return execute_distribution_broadcast(
                row_values=row_values,
                plan=plan,
                logical_unit=logical_unit,
                include_inputs=include_inputs,
                record_recipe=False,
                route_metadata=inner.metadata,
            )

        return _sweep.execute_sweep(
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
            route=route.metadata,
        )

    # Non-broadcast call — one function invocation, then wrap. TrackedTerm
    # values form lineage parents; every other resolved parameter remains
    # separately fingerprinted in Provenance.inputs.
    # Known harmless duplication: the distribution-broadcast module builds
    # the same request shape. A later execution cleanup can centralize this
    # without reintroducing private facade wrappers.
    execution = _make_execution_config(function)
    request = _execution.WorkflowExecutionRequest(
        func=invoke_point,
        work_items=_execution.make_managed_work_items(
            [values],
            unit_segments=(_execution.point_unit_segment(),),
        ),
        execution=execution,
        contract=_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport=_execution_contract.transport_for_execution_mode(execution.mode),
            stochastic_plan=None,
        ),
    )
    result = _execution.execute_many(request)[0]
    name = function._name
    controls, diagnostics = _recipe.provenance_recipe_fields(None)
    provenance = Provenance.create(
        f"workflow.{name}",
        parents=[*provenance_parents, *route_records],
        metadata={"func": name, **route.metadata},
        inputs=provenance_inputs,
        controls=controls,
        diagnostics=diagnostics,
    )
    return _result._coerce_output(
        result,
        broadcast_mode=_result.BROADCAST_WRAP,
        provenance=provenance,
        field_name=function.output_name,
    )


def _jax_traceability_error(
    function: Function,
    values: dict[str, Any],
    broadcast_args: list[_binding.WorkflowInputRef],
    *,
    func: Callable[..., Any],
    stochastic_plan: _plan.StochasticPlan | None,
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
        batched_sources: dict[_binding.WorkflowInputRef, Any] = {}
        unvectorized_batches: dict[Any, Any] = {}
        drawn_refs: list[_binding.WorkflowInputRef] = []
        for ref in _binding.iter_input_refs(function._signature_info, values):
            v = _binding.input_ref_value(values, ref)
            if ref in broadcast_refs:
                # Batched-record input: take row 0 so the dummy call
                # sees what an inner sweep iteration will actually
                # receive.
                if isinstance(v, RecordBatch):
                    batched_sources[ref] = v
                    dummy_kw = _binding.replace_input_ref(dummy_kw, ref, v[0])
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
                dummy_kw = _binding.replace_input_ref(dummy_kw, ref, replacement)
        with _context._workflow_probe():
            if unvectorized_batches:
                raise _UnvectorizableBatchSignal(
                    sorted({type(b).__name__ for b in unvectorized_batches.values()})
                )
            if batched_sources:
                refs = list(batched_sources)
                # The executor's own body, not a copy maintained in the
                # probe. ``dummy_kw`` already carries non-batched inputs.
                _row_call = _sweep.mapped_row_body(
                    func=func,
                    values=dummy_kw,
                    array_args=refs,
                    field_name=function.output_name,
                    output_is_declared=(
                        function.output_spec is not None and function.output_spec.spec is not None
                    ),
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
                _draw_call = _broadcast.mapped_draw_body(
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
                    # The stand-in draw is at the kind the law's event declaration
                    # names, as the law's own draws are.
                    if root.event_spec.exposes_record:
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
    broadcast_args: list[_binding.WorkflowInputRef],
    *,
    func: Callable[..., Any],
    stochastic_plan: _plan.StochasticPlan | None,
) -> None:
    """Raise a clear error if explicit JAX dispatch cannot trace.

    A result that violates its declaration while the probe traces raises its
    :class:`~._result.ResultSchemaError` or :class:`~._result.ResultKindError`
    unchanged, as the other dispatch modes raise it at return.
    """
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
    if isinstance(trace_error, (_result.ResultSchemaError, _result.ResultKindError)):
        raise trace_error
    if isinstance(trace_error, _UnvectorizableBatchSignal):
        raise TypeError(
            f"dispatch='jax' cannot vectorize over {', '.join(trace_error.kinds)}: the "
            f"mapped row body is built from one leaf per field, which a single-store batch "
            f"does not have. Use dispatch='auto' or 'sequential', which sweep it correctly."
        ) from trace_error
    if isinstance(trace_error, _context._StochasticProbeSignal):
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
    broadcast_args: list[_binding.WorkflowInputRef],
    *,
    jax_supported: bool = True,
    func: Callable[..., Any],
    stochastic_plan: _plan.StochasticPlan | None,
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


@dataclass(frozen=True)
class _Route:
    """The route selected for one call (step 6).

    Attributes
    ----------
    name : str
        The route's name: ``"body"`` for a plain call of a Function, the
        selected route for a plain call of a Function realized by routes, and
        otherwise the name of the evaluation rule selected for the lifted call.
    exact : bool or None
        Whether the route's result denotes the call's mathematical result.
    method : str or None
        The registry method the selected route delegates to.
    rule : BinaryDispatchMethod or None
        The selected evaluation rule of a lifted call.
    operand : Any
        The lifted argument the rule dispatched on.
    parameter : str or None
        The parameter the operand binds.
    fixed_args : Mapping[str, Any]
        The call's other arguments, by parameter name.
    controls : Mapping[str, Any]
        The resolved controls the rule received.
    """

    name: str
    exact: bool | None
    rule: Any = None
    operand: Any = None
    parameter: str | None = None
    fixed_args: Mapping[str, Any] = field(default_factory=dict)
    controls: Mapping[str, Any] = field(default_factory=dict)
    method: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """The route's name and exactness, and its registry method, as provenance records them."""
        recorded: dict[str, Any] = {"route": self.name, "exact": self.exact}
        if self.method is not None:
            recorded["method"] = self.method
        return recorded


#: The one route of a plain call: the function's body, which is exact.
_BODY = "body"

#: The name a check gives the selection that waits on a planned conversion.
_PLANNED = "the route of the converted call"


def _resolve_route(
    function: Function,
    values: Mapping[str, Any],
    broadcast_plan: _plan.BroadcastPlan,
    overrides: _call.WorkflowCallOverrides,
    rule_method: str | None,
) -> _Route:
    """Select the route that realizes the call (step 6).

    A plain call has its body as its one candidate, which is exact, so
    ``exact_only`` admits it and a ``method`` name matches nothing. A lifted call
    resolves through the evaluation-rule registry on the function and the first
    lifted argument, a swept batch before a law, with the call's other arguments
    as the fixed arguments; *rule_method* names a rule and ``exact_only``
    excludes the approximate ones.

    Raises
    ------
    ResolutionError
        If ``method`` names no route, or no rule is feasible under the controls,
        naming each rule tried and what it lacked.
    """
    info, route = _route_report(function, values, broadcast_plan, overrides, rule_method)
    if route is None:
        if broadcast_plan.regime == "none":
            raise ResolutionError(info.description)
        ref = (broadcast_plan.array_args or broadcast_plan.dist_args)[0]
        detail = (
            f"pending: {', '.join(info.pending)}" if info.feasible is None else info.description
        )
        restriction = " with exact_only" if function.options["exact_only"] else ""
        raise ResolutionError(
            f"{function.name}: no evaluation rule realizes the call lifting {ref.label!r}"
            f"{restriction}. {detail}"
        )
    return route


def _route_report(
    function: Function,
    values: Mapping[str, Any],
    broadcast_plan: _plan.BroadcastPlan,
    overrides: _call.WorkflowCallOverrides,
    rule_method: str | None,
) -> tuple[MethodInfo, _Route | None]:
    """The report of the route that realizes the call, and the route when it is selected.

    The report probes without executing, as :func:`_resolve_route` states the
    selection; the route is ``None`` unless the report is feasible.
    """
    method, exact_only = rule_method, function.options["exact_only"]
    if broadcast_plan.regime == "none":
        if method is not None:
            return MethodInfo(
                False,
                f"{function.name}: a call that lifts nothing has its body as its one route, "
                f"so method={method!r} names no route; a method names an evaluation rule of a "
                f"call that lifts an argument",
            ), None
        return MethodInfo(True, method_name=_BODY, exact=True), _Route(_BODY, True)
    ref = (broadcast_plan.array_args or broadcast_plan.dist_args)[0]
    operand = _binding.input_ref_value(values, ref)
    parameter = ref.parameter_name
    variadic = ref.subscript is not None
    fixed_args = {name: value for name, value in values.items() if variadic or name != parameter}
    controls = {
        **function.options,
        "n_broadcast_samples": overrides.n_broadcast_samples,
        "include_inputs": overrides.include_inputs,
    }
    registry = _rules.evaluation_rule_registry
    info = registry.check(
        function,
        operand,
        method=method,
        exact_only=exact_only,
        parameter=parameter,
        fixed_args=fixed_args,
        controls=controls,
    )
    if info.feasible is not True:
        return info, None
    return info, _Route(
        info.method_name,
        info.exact,
        registry.get_method(info.method_name),
        operand,
        parameter,
        fixed_args,
        controls,
    )


def _check_call(function: Function, args: tuple[Any, ...], kwargs: dict[str, Any]) -> CallReport:
    """Probe steps 1 to 6 of a call, executing neither the body nor a conversion.

    Binding, admission, and planning raise as the call's would. A conversion is
    planned and reported, not constructed, and where the call lifts the law a
    conversion constructs, selection waits on that law and is unresolved. A
    Function realized by routes is checked at each point of the call, as the
    engine realizes it, once its lifted call has an evaluation rule.

    Raises
    ------
    TypeError
        If the arguments do not bind to the signature.
    ApplicabilityError
        If an argument's kind is not accepted, a condition fails, or the
        declarations conflict.
    ResolutionError
        If a conversion has no converter, or the ``method`` control names no
        route of a Function realized by routes.
    """
    call = _call.resolve_workflow_call(
        function._signature_info,
        args,
        kwargs,
        bind=function._bind,
        module=function._module,
        dependency_type=Node,
        workflow_name=function._name,
        default_n_broadcast_samples=function.options["n_broadcast_samples"],
        default_include_inputs=function.options["include_inputs"],
        options=_call.WorkflowCallOptions(),
    )
    if function.options["conversions"]:
        raise NotImplementedError("Function.check: the conversions control")
    values, conversions, waiting = _normalization.plan_distribution_values(
        values=call.values, signature_info=function._signature_info
    )
    _call.admit_arguments(
        function._signature_info,
        values,
        input_spec=function.input_spec,
        function_name=function._name,
        roles=function._roles,
    )
    broadcast_plan = _plan.build_broadcast_plan(
        values=values, signature_info=function._signature_info, roles=function._roles
    )
    candidates = function._route_candidates(function.options)
    if candidates is not None:
        return _check_routes(
            function, values, broadcast_plan, call, candidates, conversions, waiting
        )
    lifted_refs = (*broadcast_plan.array_args, *broadcast_plan.dist_args)
    _, bindings = _bind_planned_function_inputs(
        function_name=function._name,
        input_spec=function.input_spec,
        values=values,
        lifted_names={ref.parameter_name for ref in lifted_refs},
    )
    result = (
        function.output_spec.with_dim_sizes(**bindings)
        if function.output_spec is not None
        else None
    )
    deferred = ()
    if result is None or result.spec is None:
        deferred = ("the result's type is completed from the returned value",)
    lifted = tuple(dict.fromkeys(ref.parameter_name for ref in lifted_refs))
    if waiting:
        # The lift of a converted argument is decided by the law the conversion
        # constructs, so the selection waits on it.
        pending = MethodInfo(None, pending=waiting, method_name=_PLANNED, exact=False)
        return CallReport(
            routes=(pending,),
            deferred=deferred,
            result=result,
            lifted=lifted,
            conversions=MappingProxyType(conversions),
        )
    info, _ = _route_report(
        function, values, broadcast_plan, call.overrides, function.options["method"]
    )
    return CallReport(
        routes=(info,),
        selected=None if info.feasible is None else info,
        deferred=deferred,
        result=result,
        lifted=lifted,
        conversions=MappingProxyType(conversions),
    )


def _check_routes(
    function: Function,
    values: Mapping[str, Any],
    broadcast_plan: _plan.BroadcastPlan,
    call: _call.ResolvedWorkflowCall,
    candidates: tuple[Any, ...],
    conversions: Mapping[str, Any],
    waiting: tuple[str, ...],
) -> CallReport:
    """The check of a call that a Function's routes realize at each of its points.

    A lifted call first needs an evaluation rule, selected by rank since the
    ``method`` control names a route; its points are then checked as the
    engine realizes them, each route probed while no route runs.
    """
    lifted = tuple(
        dict.fromkeys(
            ref.parameter_name for ref in (*broadcast_plan.array_args, *broadcast_plan.dist_args)
        )
    )
    if waiting:
        pending = MethodInfo(None, pending=waiting, method_name=_PLANNED, exact=False)
        return CallReport(
            routes=(pending,), lifted=lifted, conversions=MappingProxyType(dict(conversions))
        )
    if broadcast_plan.regime != "none":
        info, _ = _route_report(function, values, broadcast_plan, call.overrides, None)
        if info.feasible is not True:
            return CallReport(
                routes=(info,),
                selected=None if info.feasible is None else info,
                lifted=lifted,
                conversions=MappingProxyType(dict(conversions)),
            )
    token = _call._CHECKING.set(True)
    try:
        point = _resolution.check_points(
            function, values, function.options, candidates, broadcast_plan
        )
    finally:
        _call._CHECKING.reset(token)
    return _resolution.call_report(point, lifted=lifted, conversions=conversions)


def _run_registered_rule(
    function: Function,
    route: _Route,
    provenance_parents: list[TrackedTerm],
    provenance_inputs: Mapping[str, Any],
) -> Any:
    """Execute a registered evaluation rule and give its result the call's identity.

    The rule returns the result over raw forms, and the return step wraps it at
    its kind under the function's output name, with provenance recording the
    function, its inputs, and the rule.
    """
    result = route.rule.execute(
        function,
        route.operand,
        parameter=route.parameter,
        fixed_args=route.fixed_args,
        controls=route.controls,
    )
    controls, diagnostics = _recipe.provenance_recipe_fields(None)
    provenance = Provenance.create(
        f"workflow.{function._name}",
        parents=provenance_parents,
        metadata={"func": function._name, **route.metadata},
        inputs=provenance_inputs,
        controls=controls,
        diagnostics=diagnostics,
    )
    return _result._coerce_output(
        result,
        broadcast_mode=_result.BROADCAST_WRAP,
        provenance=provenance,
        field_name=function.output_name,
    )


@contextmanager
def _apply_scope() -> Generator[None, None, None]:
    """Preserve workflow admission and RNG ownership around raw evaluation."""
    _context._assert_workflow_admission()
    _replay._reject_function_apply()
    with (
        _context._ephemeral_workflow_run(),
        _broker._function_stochastic_scope(),
    ):
        yield


class _CallEngine:
    """The call stack of design Part V, installed as the call path of every Function."""

    def __call__(self, function: Function, *args: Any, **kwargs: Any) -> Any:
        result = _call_with_options(function, args, kwargs, _call.WorkflowCallOptions())
        if not function.options["raw"]:
            return result
        if function._route_candidates(function.options) is not None:
            return _result.raw_form(result)
        return _result._detach(result)

    @staticmethod
    def check(function: Function, *args: Any, **kwargs: Any) -> CallReport:
        """The probe of steps 1 to 6 that :meth:`Function.check` returns."""
        return _check_call(function, args, kwargs)

    @staticmethod
    def invoke(
        function: Function, values: Mapping[str, Any], context: _FunctionInvocationContext
    ) -> Any:
        """The one point :meth:`Function.apply` evaluates, with no lifting, tracking, or provenance.

        A Function's body runs on *values*. A Function realized by routes
        admits each argument by its role, with no lifting, runs the route
        selected for the point, and returns the result's raw form.
        """
        candidates = function._route_candidates(function.options)
        if candidates is None:
            return function._implementation.invoke(
                _binding.values_to_bound_arguments(function.signature, values), context=context
            )
        _call.admit_arguments(
            function._signature_info,
            values,
            function_name=function._name,
            roles=function._roles,
            lifts=False,
        )
        token = _call._CHECKING.set(False)
        try:
            term = _realized_point(function, values, function.options, candidates)
        finally:
            _call._CHECKING.reset(token)
        return _result.raw_form(term)

    @staticmethod
    def apply_scope() -> AbstractContextManager[None]:
        """The scope plain evaluation runs in: workflow admission and RNG ownership."""
        return _apply_scope()


#: The installed engine.
_call_engine = _CallEngine()

install_call_engine(_call_engine)
