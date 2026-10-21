"""Function decorator and the installed workflow call engine.

The engine runs the call stack of design Part V on every call of a Function:

1. configure: the controls a Function carries, resolved by ``using``;
2. bind: the arguments bind to the signature (:mod:`._call`);
3. normalize: distribution arguments are converted where their parameter names
   another class (:mod:`._normalization`) and each argument is admitted by its
   role or its input spec (:func:`._call.admit_arguments`);
4. classify the lift: the lifted arguments are found and grouped
   (:func:`._plan.build_broadcast_plan`, :func:`._plan.build_stochastic_plan`);
5. plan: the input specs unify and the output spec binds the shared
   dimensions (:func:`._contract._bind_planned_function_inputs`), or, for
   a Function realized by routes, each point's result rule runs
   (``Function._plan_route_inputs``);
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
from dataclasses import dataclass, field, replace
from functools import partial
from types import MappingProxyType
from typing import Any, cast, overload

import jax
import jax.numpy as jnp

from ..core._specs import _has_record_outputs, _output_record
from ..values import _binding

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from ..core._batch import Batch, BatchSpec, _flat_batch_spec
from ..core._call_context import _NumSamplesUsed, current_call_context
from ..core._dispatch import MethodInfo, ResolutionError
from ..core._expression import (
    Applied,
    AppliedValue,
    Draw,
    Expression,
    Named,
    constant,
    draw_of,
)
from ..core._numeric_record_batch import NumericRecordBatch
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec
from ..core._specs import InputSpec, NumericRecordSpec, OutputSpec
from ..core.config import ProvenanceMode, WorkflowKind, config
from ..core.named_tree import _unflatten_paths
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._factored import _labeled_product
from ..distributions._views import FieldView
from ..values._function_base import (
    _WARNING_SKIP_PREFIXES,
    Function,
    _bind_function_inputs,
    _FunctionInvocationContext,
    _refuse_unknown_controls,
    _ResultKindMismatch,
    _validate_function_output,
    install_call_engine,
)
from . import (
    _broadcast,
    _call,
    _callable,
    _context,
    _evaluation_keys,
    _execution,
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
    label: str | None = None,
    input_spec: InputSpec | Mapping[str, TermSpec] | None = None,
    output_spec: OutputSpec | TermSpec | None = None,
    output_label: str | None = None,
    differentiable: NumericSpec | None = None,
    bind: Mapping[str, Any] | None = None,
    **controls: Any,
) -> Function | Callable[[Callable[..., Any]], Function]:
    """Decorator to create a :class:`Function` from a plain function.

    Bare usage wraps a function with default ``Function`` controls::

        @function
        def my_func(x, y):
            return x + y

    Pass keyword arguments to configure ProbPipe controls at definition time::

        @function(max_evaluations=100, dispatch="sequential")
        def my_func(x, y):
            return x + y

    Keyword arguments passed later to the workflow call itself belong to the
    wrapped function whenever they can bind to that function. Use
    ``workflow.using(...)(...)`` for one-call ProbPipe controls.

    A parameter annotated with a class of raw values, such as ``pd.Series`` or
    ``jax.Array``, receives the raw form of an argument, or of each element or
    draw that a lift passes to it, when that form is an instance of the class.
    A function written for pandas therefore runs on a record field that holds
    a ``Series``::

        @function
        def differences(c: pd.Series) -> jax.Array:
            return jnp.asarray(c.diff().dropna().to_numpy())

        differences(Record({"c": pd.Series([1.0, 2.0, 4.0])}, label="r")["c"])

    Parameters
    ----------
    _func : Callable or None
        Function being decorated for bare ``@function`` usage.
        Users should not pass this argument by keyword.
    label : str or None
        The function label, defaulting to the callable's name, or ``𝒻`` for a lambda.
        A callable with none, such as a ``functools.partial``, needs it.
    input_spec : InputSpec or Mapping[str, TermSpec] or None
        The authoritative inputs, as :class:`Function` takes them.
    output_spec : OutputSpec or TermSpec or None
        The authoritative result declaration, as :class:`Function` takes it.
    output_label : str or None
        Optional result alias. Without one, a value result describes the
        application, and a returned law or kernel takes the function's label.
    differentiable : NumericSpec or None
        The differentiability claim, as :class:`Function` takes it.
    bind : Mapping or None
        Construction-time values of the wrapped callable's arguments, which
        call arguments override.
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
        If a keyword is no control, before any callable is wrapped; or if
        *label* is omitted for a callable that has no ``__name__``, such as a
        ``functools.partial`` or a callable instance.
    """
    _refuse_unknown_controls(controls)

    def decorator(func: Callable[..., Any]) -> Function:
        return Function(
            func,
            input_spec=input_spec,
            output_spec=output_spec,
            output_label=output_label,
            differentiable=differentiable,
            bind=bind,
            **controls,
            label=label,
        )

    if _func is not None:
        return decorator(_func)
    return decorator


def effective_workflow_kind(function: Function) -> WorkflowKind:
    """The orchestration mode of *function*, as :attr:`Function.effective_workflow_kind` states it.

    The function's ``workflow_kind`` control takes precedence over the global
    configuration, and ``DEFAULT`` defers to it; when both are ``DEFAULT``,
    orchestration is ``OFF``. A requested ``TASK`` or ``FLOW`` falls back to
    ``OFF``, with a warning, when Prefect is not installed.
    """
    return _orchestration_mode(function.options["workflow_kind"])


def _orchestration_mode(workflow_kind_control: WorkflowKind) -> WorkflowKind:
    """The orchestration mode that the ``workflow_kind`` control *workflow_kind_control* selects.

    ``DEFAULT`` defers to ``config.prefect.workflow_kind``; when both are
    ``DEFAULT``, orchestration is ``OFF``. A requested ``TASK`` or ``FLOW``
    falls back to ``OFF``, with a warning, when Prefect is not installed.
    """
    kind = (
        workflow_kind_control
        if workflow_kind_control is not WorkflowKind.DEFAULT
        else config.prefect.workflow_kind
    )

    if kind is WorkflowKind.DEFAULT:
        kind = WorkflowKind.OFF

    _PREFECT_KINDS = {WorkflowKind.TASK, WorkflowKind.FLOW}
    prefect_missing = task is None or flow is None
    if kind in _PREFECT_KINDS and prefect_missing:
        warnings.warn(
            f"workflow_kind=WorkflowKind.{kind.name} requested but Prefect is not installed; "
            "running without Prefect (WorkflowKind.OFF). Install it with: "
            "pip install probpipe[prefect]",
            skip_file_prefixes=_WARNING_SKIP_PREFIXES,
        )
        return WorkflowKind.OFF

    return kind


def _make_execution_config(
    function_label: str,
    resolved_controls: Mapping[str, Any],
    *,
    mode: _execution.WorkflowExecutionMode | None = None,
) -> _execution.WorkflowExecutionConfig:
    """Build resolved execution metadata for row-wise dispatch of a call under *resolved_controls*.

    Parameters
    ----------
    function_label : str
        The label of the called ``Function``, which names the execution.
    resolved_controls : Mapping of str to Any
        The call's resolved controls, whose ``workflow_kind``, ``dispatch``,
        and ``max_workers`` select the mode.
    mode : str or None, optional
        The mode to use, or ``None`` to select it from *resolved_controls*.

    Returns
    -------
    WorkflowExecutionConfig
        The execution metadata.

    Warns
    -----
    UserWarning
        If the mode is a Prefect mode while ``dispatch="thread"`` or
        ``max_workers`` is set, which configure only local thread dispatch.
    """
    if mode is None:
        match _orchestration_mode(resolved_controls["workflow_kind"]):
            case WorkflowKind.TASK:
                mode = "prefect_task"
            case WorkflowKind.FLOW:
                mode = "prefect_flow"
            case _ if resolved_controls["dispatch"] == "thread":
                mode = "thread"
            case _:
                mode = "sequential"

    is_prefect = mode in ("prefect_task", "prefect_flow")

    if is_prefect and (
        resolved_controls["dispatch"] == "thread" or resolved_controls["max_workers"] is not None
    ):
        warnings.warn(
            "dispatch='thread' and max_workers configure only local "
            "ThreadPoolExecutor dispatch; they do not control Prefect "
            "scheduling",
            stacklevel=2,
        )

    return _execution.WorkflowExecutionConfig(
        mode=mode,
        max_workers=resolved_controls["max_workers"] if mode == "thread" else None,
        name=function_label,
        prefect_task_runner=(config.prefect.resolve_task_runner() if is_prefect else None),
    )


def _call_function(
    function: Function,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
) -> Any:
    _context._assert_workflow_admission()
    with _replay._function_replay_scope() as replay_call:
        occurrence_path = None if replay_call is None else replay_call.occurrence_path
        with (
            _context._ephemeral_workflow_run(),
            _evaluation_keys._function_stochastic_scope(occurrence_path=occurrence_path) as broker,
        ):
            if (
                replay_call is not None
                or _context._active_provenance_mode() is not ProvenanceMode.OFF
            ):
                anchor = _callable.capture_function_anchor(function)
                broker.set_callable_anchor(anchor)
                if replay_call is not None:
                    replay_call.validate_callable(anchor)
            return _call_function_in_context(function, args, call_inputs)


def _call_function_in_context(
    function: Function,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
) -> Any:
    # A call made while a check probes, as a probe that runs an operation makes
    # one, selects and runs its routes as any call does.
    token = _call._CHECKING.set(False)
    try:
        return _run_call(function, args, call_inputs)
    finally:
        _call._CHECKING.reset(token)


def _result_expression(
    function: Function,
    values: Mapping[str, Any],
    explicit_parameter_names: frozenset[str] | None = None,
) -> Expression | None:
    """The expression of the result of a call of *function* on the arguments *values* (II.4).

    A function's result carries its explicit alias or application expression, and an operation's result
    the expression its expression rule builds from the operands'. The result
    boundary labels a returned law or kernel by the function's label instead of
    the application. ``None``
    leaves the result the expression its route gave it.
    """
    derive = getattr(function, "_derived_expression", None)
    if derive is not None:
        return derive(values)
    if function._output_label is not None:
        return Named(function.output_label)
    return AppliedValue(
        function.label,
        _value_arguments(
            function,
            values,
            frozenset(values) if explicit_parameter_names is None else explicit_parameter_names,
        ),
    )


def _value_arguments(
    function: Function, values: Mapping[str, Any], explicit_parameter_names: frozenset[str]
) -> tuple[Expression, ...]:
    """Arguments in parameter order, using declared input names for unnamed arrays."""
    arguments = (
        _argument_expression(_binding.input_ref_value(values, ref), ref, explicit_parameter_names)
        for ref in _binding.iter_input_refs(function._signature_info, values)
    )
    return tuple(argument for argument in arguments if argument is not None)


def _argument_expression(
    value: Any, ref: _binding.FunctionInputRef, explicit_parameter_names: frozenset[str]
) -> Expression | None:
    """How a call's expression shows the argument *value* at *ref*, or ``None`` to omit it.

    A tracked argument shows its expression. Any other value the caller
    passed shows its value when it is a scalar, as ``2.0``, and its
    parameter's name otherwise, as ``X``. A value the caller did not pass, such
    as a parameter's default, is omitted.
    """
    if isinstance(value, TrackedTerm):
        return value._embedded_expression()
    if ref.subscript is not None or ref.parameter_name in explicit_parameter_names:
        return constant(value) if _is_scalar(value) else Named(ref.label)
    return None


def _result_identity(
    function: Function,
    values: Mapping[str, Any],
    explicit_parameter_names: frozenset[str] | None = None,
) -> tuple[str, Expression | None]:
    """Build the result expression once and derive its presentation label."""
    expression = _result_expression(function, values, explicit_parameter_names)
    if hasattr(function, "_result_label"):
        return function._result_label(values, expression), expression
    return expression.render_label(), expression


def _lifted_expression(
    function: Function,
    values: Mapping[str, Any],
    plan: _plan.StochasticPlan,
    call_values: Mapping[str, Any] | None = None,
    *,
    explicit_parameter_names: frozenset[str],
) -> Applied:
    """The expression of the law of *function* lifted over the laws among *values* (II.4).

    It is the function applied to its arguments in parameter order, as
    ``f(beta ~ model; y)``: a draw of each law the plan lifts, where arguments
    whose laws the plan draws together from one root share one draw of the
    root, as ``f((a, b) ~ model)``; each other tracked argument by its
    expression; and each other value the caller passed by its value when it is
    a scalar, as ``2.0``, and by its parameter's name otherwise, as ``X``. A
    value the caller did not pass, such as a parameter's default, is not shown,
    whatever its value. A row of a sweep shows each scalar it takes from a
    swept batch by its value, as ``f(mu ~ prior, 4.0)``.

    Parameters
    ----------
    function : Function
        The function lifted, whose output label names the call.
    values : Mapping of str to Any
        The call's arguments, by parameter name, or a row's arguments in a
        sweep.
    plan : StochasticPlan
        The plan of the lift, whose source groups say which arguments are
        drawn together.
    call_values : Mapping of str to Any or None, optional
        The arguments of the sweep *values* is a row of. An argument whose row
        value is not the call's is a row of a swept batch.
    explicit_parameter_names : frozenset of str
        The parameters the caller's arguments bind, together with those the
        Function's construction ``bind`` values bind.

    Returns
    -------
    Applied
        The applied function.
    """
    group_of = {
        consumer.arg_ref: group.index
        for group in plan.source_groups
        for consumer in group.consumers
    }
    drawn_argument_indices: set[int] = set()
    arguments: list[Expression] = []
    for ref in _binding.iter_input_refs(function._signature_info, values):
        value = _binding.input_ref_value(values, ref)
        index = group_of.get(ref)
        if index is not None:
            if index in drawn_argument_indices:
                continue
            drawn_argument_indices.add(index)
            consumers = plan.source_groups[index].consumers
            if len(consumers) == 1:
                arguments.append(draw_of(value))
                continue
            laws = [_binding.input_ref_value(values, consumer.arg_ref) for consumer in consumers]
            outputs = dict.fromkeys(name for law in laws for name in law.event_spec.outputs)
            arguments.append(
                Draw(
                    tuple(outputs),
                    _drawn_together_from(laws, plan, index)._embedded_expression(),
                )
            )
        elif _is_swept_scalar(value, ref, call_values):
            arguments.append(constant(value.raw if isinstance(value, TrackedTerm) else value))
        elif (argument := _argument_expression(value, ref, explicit_parameter_names)) is not None:
            arguments.append(argument)
    return Applied(function.output_label, tuple(arguments))


def _swept_expression(
    function: Function,
    values: Mapping[str, Any],
    expression: Expression | None,
    plan: _plan.StochasticPlan | None,
    explicit_parameter_names: frozenset[str],
) -> Any:
    """The expression of a sweep's batch: the lifted call for a sweep of broadcasts, as ``f(mu ~ d, tau)``.

    A sweep without a law keeps the call's *expression*, and ``None`` keeps
    the expression of the aggregate. *explicit_parameter_names* names the parameters the
    caller's arguments and the construction ``bind`` values bind, which the
    lifted call shows.
    """
    if plan is not None:
        return _lifted_expression(
            function, values, plan, explicit_parameter_names=explicit_parameter_names
        )
    return _result.KEEP_EXPRESSION if expression is None else expression


def _is_swept_scalar(
    value: Any, ref: _binding.FunctionInputRef, call_values: Mapping[str, Any] | None
) -> bool:
    """Whether *value*, a row's argument at *ref*, is a scalar row of a batch the call sweeps."""
    if call_values is None or not _is_scalar(value):
        return False
    return _binding.input_ref_value(call_values, ref) is not value


def _is_scalar(value: Any) -> bool:
    """Whether *value* is a scalar, which an applied function's notation shows by its value.

    A scalar is a Python number or string, or an array or NumPy scalar of shape ``()``.
    """
    if isinstance(value, (bool, int, float, complex, str)):
        return True
    # A numeric record forwards ``shape`` to its single field, and it raises
    # TypeError for a record of several fields.
    if isinstance(value, Record):
        return False
    return getattr(value, "shape", None) == ()


def _drawn_together_from(laws: list[Any], plan: _plan.StochasticPlan, index: int) -> Any:
    """The law that the arguments *laws*, which the plan draws together, are drawn from.

    It is the parent of field views of one law, as ``model`` is for
    ``model["a"]`` and ``model["b"]``, and the root of the plan's source group
    *index* otherwise.
    """
    parents = {id(law.parent): law.parent for law in laws if isinstance(law, FieldView)}
    if len(parents) == 1 and all(isinstance(law, FieldView) for law in laws):
        return next(iter(parents.values()))
    return plan.runtime_bindings[index].root


def _expressed(term: Any, expression: Expression | None) -> Any:
    """*term*, a point's result, carrying *expression*, on a copy that keeps its provenance.

    A result that is not a tracked term, an expression of ``None``, and a term
    that carries the expression already are returned as they are, so a term
    that is also an operand of the call, such as a factor that conditioning
    returns, keeps its own expression. A term carries the expression when it
    holds that same node, which the expression rule takes from its operand,
    or a label that the expression states alone.
    """
    if expression is None or not isinstance(term, TrackedTerm):
        return term
    expression = _result._prepared(term, expression)
    if term._expression is expression:
        return term
    clone = term._shallow_copy()
    clone._store_expression(expression)
    return clone


def _route_result_as_provenance_parent(term: Any, route_value: TrackedTerm) -> TrackedTerm:
    """The route's result *route_value* as the call's provenance names it among its parents.

    *term* is *route_value* after the result boundary gave it the call's
    expression, on a copy that keeps *route_value*'s provenance, so the parent
    reads under the result's label, as ``model`` for a posterior of ``model``,
    where it read under the label the route chose, such as an inference
    method's ``posterior``. A *term* that carries another provenance, or none,
    leaves *route_value* as the parent.
    """
    if isinstance(term, TrackedTerm) and term.provenance is route_value.provenance:
        return term
    return route_value


def _keeping_route_provenance(term: Any, route_value: Any) -> Any:
    """*term*, the declared form of a route's result *route_value*, with the provenance *route_value* carries.

    A route whose result records how it was produced, as an inference
    method's posterior names the method, passes that provenance on to the
    term, so each point of a lifted call keeps it.
    """
    if (
        isinstance(term, TrackedTerm)
        and isinstance(route_value, TrackedTerm)
        and term is not route_value
        and term.provenance is None
        and route_value.provenance is not None
    ):
        term.with_provenance(route_value.provenance)
    return term


def _realized_point(
    function: Function,
    argument_values: Mapping[str, Any],
    controls: Mapping[str, Any],
    candidates: tuple[Any, ...],
    *,
    explicit_parameter_names: frozenset[str] | None = None,
    kind_error: type[Exception] = _result.ResultKindError,
) -> Any:
    """One point of a call realized by the route selected among *candidates*, as a term.

    The point is planned, its route selected and run, and the raw result
    validated against the point's declaration, wrapped at the kind it names,
    and given the expression the operation builds for the point, from which
    it takes its label.

    Parameters
    ----------
    function : Function
        The Function realized by routes, which plans the point.
    argument_values : Mapping of str to Any
        The point's arguments, by parameter name.
    controls : Mapping of str to Any
        The call's resolved controls, which planning and selection read.
    candidates : tuple
        The Function's routes, in selection order.
    explicit_parameter_names : frozenset of str or None
        The parameters the call's arguments and the Function's construction
        ``bind`` values bind, which the result's expression shows; ``None``
        shows every argument in *argument_values*.
    kind_error : type of Exception
        The error for a wrong overall return kind, passed to ``declared_term``.

    Returns
    -------
    TrackedTerm
        The result term, which takes the provenance of a route result that
        records one.

    Raises
    ------
    ApplicabilityError
        If an applicability check fails.
    ResolutionError
        If no candidate is feasible, or the first one that is not is unresolved.
    ResultKindError, ResultSchemaError
        If the result violates the declaration, using *kind_error* for its kind.
    """
    route_inputs, result_declaration, _ = function._plan_route_inputs(argument_values, controls)
    candidate, report = _resolution.selected(
        function.label, controls, candidates, route_inputs, result_declaration
    )
    route_value, _ = _resolution.settled(candidate.run(route_inputs, result_declaration, report))
    label, expression = _result_identity(function, argument_values, explicit_parameter_names)
    term = _result.declared_term(route_value, result_declaration, label, kind_error=kind_error)
    return _expressed(_keeping_route_provenance(term, route_value), expression)


def _run_call(
    function: Function,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
) -> Any:
    call_context = current_call_context()
    resolved_controls = call_context.resolved_controls
    num_samples_used = cast(_NumSamplesUsed, call_context.num_samples_used)
    argument_values = _call.resolve_function_call(
        function._signature_info,
        args,
        call_inputs,
        bind=function._bind,
        function_name=function._label,
    )

    argument_values = _normalization.normalize_distribution_values(
        values=argument_values,
        signature_info=function._signature_info,
        conversions=resolved_controls["conversions"],
    )
    _call.admit_arguments(
        function._signature_info,
        argument_values,
        input_spec=function.input_spec,
        function_name=function._label,
        roles=function._roles,
    )
    broadcast_plan = _plan.build_broadcast_plan(
        values=argument_values,
        signature_info=function._signature_info,
        roles=function._roles,
    )
    explicit_parameter_names = frozenset(
        function.signature.bind_partial(*args, **call_inputs).arguments
    ) | frozenset(function._bind)
    label, expression = _result_identity(function, argument_values, explicit_parameter_names)
    candidates = function._route_candidates(resolved_controls, argument_values)
    selection: tuple[Any, OutputSpec | None, Any, Any] | None = None
    if candidates is not None and broadcast_plan.regime == "none":
        # The routes realize the one point, so the selection is the call's route.
        route_inputs, result_declaration, _ = function._plan_route_inputs(
            argument_values, resolved_controls
        )
        candidate, report = _resolution.selected(
            function.label, resolved_controls, candidates, route_inputs, result_declaration
        )
        selection = (route_inputs, result_declaration, candidate, report)
        route = _Route(
            candidate.route_name,
            candidate.exactness(report),
            method=candidate.method_of(report),
        )
        num_samples_used.collects_calls = candidate.collects_calls
    else:
        # A Function's routes take the method control, so a lifted call selects
        # its evaluation rule by rank.
        rule_method = None if candidates is not None else resolved_controls["method"]
        route = _resolve_route(
            function, argument_values, broadcast_plan, rule_method, resolved_controls
        )
    stochastic_plan = _plan.build_stochastic_plan(
        argument_values,
        broadcast_plan,
        resolved_controls["max_evaluations"],
        enumerate_every_group=route.name != "sampling_lift",
    )
    stochastic_sample_shape = None if stochastic_plan is None else stochastic_plan.sample_shape

    def get_key(event: _plan.PlannedRandomEvent):
        return _evaluation_keys._resolve_automatic_key(
            None,
            _evaluation_keys.StochasticEffectPlan(
                operation_kind="function_lifting",
                execution_mode="sampled",
                event=event,
                sample_shape=stochastic_sample_shape,
                sampling_abi="probpipe.distribution_sampling/v1",
                provider_abi="probpipe.distribution/v1",
            ),
        )

    workflow_kind = _orchestration_mode(resolved_controls["workflow_kind"])
    _evaluation_keys._record_active_requested_execution(
        resolved_controls["dispatch"],
        workflow_kind.value,
    )
    _replay._validate_active_plan(_recipe.serialize_stochastic_plan(stochastic_plan))
    _, invocation_bindings = _bind_planned_function_inputs(
        function_name=function._label,
        input_spec=function.input_spec,
        values=argument_values,
        lifted_names={
            ref.parameter_name for ref in (*broadcast_plan.dist_args, *broadcast_plan.array_args)
        },
    )
    concrete_output_spec = (
        function.output_spec.with_dim_sizes(**invocation_bindings)
        if function.output_spec is not None
        else None
    )
    if concrete_output_spec is None and candidates is not None and selection is None:
        # A Function realized by routes declares its result at each point, so the
        # lifted call declares what one point declares.
        concrete_output_spec = _resolution.lifted_declaration(
            function, argument_values, resolved_controls, broadcast_plan
        )
    _plan.require_empty_sweep_output(broadcast_plan, concrete_output_spec)
    concrete_output_template = (
        _output_record_spec(concrete_output_spec) if concrete_output_spec is not None else None
    )
    provenance_parents: list[TrackedTerm] = [function]
    provenance_inputs: dict[str, Any] = {}
    seen_parent_ids = {id(function)}
    for ref in _binding.iter_input_refs(function._signature_info, argument_values):
        argument_value = _binding.input_ref_value(argument_values, ref)
        if isinstance(argument_value, TrackedTerm) and id(argument_value) not in seen_parent_ids:
            seen_parent_ids.add(id(argument_value))
            provenance_parents.append(argument_value)
        elif not isinstance(argument_value, TrackedTerm):
            provenance_inputs[ref.label] = argument_value

    # The selected route's result, when it carries a provenance of its own, which
    # the call's provenance keeps as a parent under the call's expression.
    route_provenance_parents: list[TrackedTerm] = []
    # The exactness the selected route settled when it ran, which the call's
    # provenance takes in place of the one its check reported.
    settled_exactness: list[bool | None] = []

    def invoke_point(**point_values: Any) -> Any:
        if selection is not None:
            route_inputs, result_declaration, candidate, report = selection
            route_value, exact = _resolution.settled(
                candidate.run(route_inputs, result_declaration, report)
            )
            if exact is not None:
                settled_exactness.append(exact)
            term = _expressed(
                _result.declared_term(route_value, result_declaration, label), expression
            )
            if (
                isinstance(route_value, TrackedTerm)
                and route_value.provenance is not None
                and id(route_value) not in seen_parent_ids
            ):
                route_provenance_parents.append(
                    _route_result_as_provenance_parent(term, route_value)
                )
            return term
        if candidates is not None:
            return _realized_point(
                function,
                point_values,
                resolved_controls,
                candidates,
                explicit_parameter_names=explicit_parameter_names,
            )
        try:
            _, point_bindings = _bind_function_inputs(
                function_name=function._label,
                input_spec=function.input_spec,
                values=point_values,
                bindings=invocation_bindings,
            )
        except ValueError as error:
            raise _call.ApplicabilityError(str(error)) from error
        context = _FunctionInvocationContext(point_bindings)
        result = _result._batch_from_declared_sequence(
            function._invoke_resolved(point_values, context=context),
            function_name=function.output_label,
            output_spec=function.output_spec,
        )
        try:
            point_output_spec = _validate_function_output(
                function_name=function._label,
                output_spec=function.output_spec,
                result=result,
                bindings=context.dimension_bindings,
            )
        except _ResultKindMismatch as error:
            raise _result.ResultKindError(str(error)) from error
        except ValueError as error:
            raise _result.ResultSchemaError(str(error)) from error
        if point_output_spec is not None:
            result = _wrap_declared_function_output(
                result,
                function_name=function.output_label,
                output_spec=point_output_spec,
            )
        # A product the function returns takes its output label as a label (IV.2).
        return _labeled_product(result)

    resolved_dispatch: str | None = None
    traceability_checked = False
    # A numeric batch has one shape and dtype per column across every cell.
    # Probe its rows as dynamic inputs so a check can be shared across cells.
    dynamic_refs = tuple(
        ref
        for ref in broadcast_plan.array_args
        if isinstance(_binding.input_ref_value(argument_values, ref).element_spec, NumericSpec)
    )
    cache_traceability = broadcast_plan.regime != "nested" or len(dynamic_refs) == len(
        broadcast_plan.array_args
    )

    general_probe_pending = broadcast_plan.regime == "nested" and cache_traceability
    general_probe_failed = False

    def prepare_shared_probe(
        dispatch_values: dict[str, Any],
        broadcast_args: list[_binding.FunctionInputRef],
    ) -> None:
        nonlocal general_probe_pending, general_probe_failed
        nonlocal traceability_checked, resolved_dispatch
        if not general_probe_pending:
            return
        general_probe_pending = False
        error = _jax_traceability_error(
            function,
            dispatch_values,
            broadcast_args,
            func=invoke_point,
            stochastic_plan=stochastic_plan,
            dynamic_refs=dynamic_refs,
        )
        if error is None:
            traceability_checked = True
            resolved_dispatch = "jax"
        else:
            # A cell can trace with its swept values fixed although they cannot
            # trace as dynamic inputs, as with a Python branch on a swept value.
            # So the first cell's fixed-value check decides whether the cells
            # share a check. A failure applies to every cell, so dispatch="auto"
            # runs them all sequentially. A success leaves each later cell to its
            # own check.
            general_probe_failed = True

    def resolve_dispatch(
        dispatch_values: dict[str, Any],
        broadcast_args: list[_binding.FunctionInputRef],
        *,
        jax_supported: bool = True,
    ) -> str:
        nonlocal resolved_dispatch, cache_traceability
        if resolved_controls["dispatch"] == "auto" and jax_supported:
            prepare_shared_probe(dispatch_values, broadcast_args)
        if resolved_controls["dispatch"] != "auto" or not jax_supported or not cache_traceability:
            return _resolve_dispatch(
                function,
                dispatch_values,
                broadcast_args,
                resolved_controls=resolved_controls,
                jax_supported=jax_supported,
                func=invoke_point,
                stochastic_plan=stochastic_plan,
            )
        if resolved_dispatch is None:
            dispatch = _resolve_dispatch(
                function,
                dispatch_values,
                broadcast_args,
                resolved_controls=resolved_controls,
                jax_supported=True,
                func=invoke_point,
                stochastic_plan=stochastic_plan,
            )
            if dispatch == "jax" and general_probe_failed:
                cache_traceability = False
                return dispatch
            resolved_dispatch = dispatch
        return resolved_dispatch

    def require_jax_traceable(
        dispatch_values: dict[str, Any],
        broadcast_args: list[_binding.FunctionInputRef],
    ) -> None:
        nonlocal traceability_checked, cache_traceability
        prepare_shared_probe(dispatch_values, broadcast_args)
        if traceability_checked and cache_traceability:
            return
        _require_jax_traceable(
            function,
            dispatch_values,
            broadcast_args,
            func=invoke_point,
            stochastic_plan=stochastic_plan,
        )
        traceability_checked = True
        if general_probe_failed:
            cache_traceability = False

    def execute_distribution_broadcast(
        *,
        row_values: dict[str, Any],
        plan: _plan.StochasticPlan,
        logical_unit: _plan.LogicalUnit,
        include_inputs: bool = resolved_controls["include_inputs"],
        record_recipe: bool = True,
        route_metadata: Mapping[str, Any] = route.metadata,
    ):
        output_spec, output_template = concrete_output_spec, concrete_output_template
        if output_spec is None and candidates is not None and row_values is not argument_values:
            # A row of a sweep binds its element itself, which may declare what the
            # element's spec alone left open.
            output_spec = _resolution.lifted_declaration(
                function, row_values, resolved_controls, broadcast_plan
            )
            output_template = None if output_spec is None else _output_record_spec(output_spec)
        return _broadcast.execute_distribution_broadcast(
            func=invoke_point,
            values=row_values,
            stochastic_plan=plan,
            logical_unit=logical_unit,
            include_inputs=include_inputs,
            get_key=get_key,
            make_execution_config=partial(
                _make_execution_config, function._label, resolved_controls
            ),
            requested_dispatch=resolved_controls["dispatch"],
            resolve_dispatch=resolve_dispatch,
            require_jax_traceable=require_jax_traceable,
            function_name=function._label,
            output_label=function.output_label,
            output_name=function._output_name,
            output_expression=_lifted_expression(
                function,
                row_values,
                plan,
                argument_values,
                explicit_parameter_names=explicit_parameter_names,
            ),
            output_spec=output_spec,
            workflow_kind=workflow_kind,
            output_template=output_template,
            provenance_parents=provenance_parents,
            provenance_inputs=provenance_inputs,
            record_recipe=record_recipe,
            route=route_metadata,
        )

    if route.rule is not None and route.name not in _rules._ENGINE_RULES:
        lifted = (
            expression
            if stochastic_plan is None
            else _lifted_expression(
                function,
                argument_values,
                stochastic_plan,
                explicit_parameter_names=explicit_parameter_names,
            )
        )
        return _run_registered_rule(
            function, route, provenance_parents, provenance_inputs, label, lifted
        )
    if broadcast_plan.regime == "distribution":
        if stochastic_plan is None:  # pragma: no cover - planner contract guard
            raise RuntimeError("distribution broadcast is missing its stochastic plan")
        return execute_distribution_broadcast(
            row_values=argument_values,
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
            values=argument_values,
            plan=broadcast_plan,
            stochastic_plan=stochastic_plan,
            make_execution_config=partial(
                _make_execution_config, function._label, resolved_controls
            ),
            requested_dispatch=resolved_controls["dispatch"],
            resolve_dispatch=resolve_dispatch,
            require_jax_traceable=require_jax_traceable,
            distribution_broadcast=distribution_broadcast,
            function_name=function._label,
            output_label=label,
            output_expression=_swept_expression(
                function, argument_values, expression, stochastic_plan, explicit_parameter_names
            ),
            output_spec=concrete_output_spec,
            include_inputs=resolved_controls["include_inputs"],
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
    execution = _make_execution_config(function._label, resolved_controls)
    request = _execution.WorkflowExecutionRequest(
        func=invoke_point,
        evaluation_points=_execution.issue_evaluation_points(
            [argument_values],
            point_segments=(_execution.plain_call_point_segment(),),
        ),
        execution=execution,
        contract=_execution.make_execution_contract(
            evaluator="rowwise",
            transport=_execution.transport_for_execution_mode(execution.mode),
            stochastic_plan=None,
        ),
    )
    result = _execution.execute_many(request)[0]
    if settled_exactness:
        exact = False if False in settled_exactness else (True if all(settled_exactness) else None)
        route = replace(route, exact=exact)
    if num_samples_used.provenance_num_samples is not None:
        route = replace(route, num_samples=num_samples_used.provenance_num_samples)
    name = function._label
    controls, diagnostics = _recipe.provenance_recipe_fields(None)
    provenance = Provenance.create(
        f"workflow.{name}",
        parents=[*provenance_parents, *route_provenance_parents],
        metadata={"func": name, **route.metadata},
        inputs=provenance_inputs,
        controls=controls,
        diagnostics=diagnostics,
    )
    return _result._coerce_output(
        result,
        broadcast_mode=_result.BROADCAST_WRAP,
        provenance=provenance,
        label=label,
        expression=_result.KEEP_EXPRESSION if expression is None else expression,
    )


def _jax_traceability_error(
    function: Function,
    values: dict[str, Any],
    broadcast_args: list[_binding.FunctionInputRef],
    *,
    func: Callable[..., Any],
    stochastic_plan: _plan.StochasticPlan | None,
    dynamic_refs: tuple[_binding.FunctionInputRef, ...] = (),
) -> Exception | None:
    """Return the JAX trace-probe error for the current call, if any.

    The probe traces the operation the dispatch is choosing, not merely the
    body: a body can trace cleanly bare yet be impossible under the
    transform its executor applies, as one that returns a record holding a
    batch is. So every executor that maps is probed under a map, each
    argument fed the way its own executor will feed it: a swept batch over its
    first row, a distribution over one draw, a batch-valued draw in its raw
    form, which a parameter annotated with a batch class receives rebuilt as
    the batch, and any other argument as the call passed it. A law whose record
    event stores a batch for each draw in an object column has no stand-in, and
    the probe reports it. A probe failure
    means sequential dispatch, which is always able to run the call — the two
    paths agree on results by contract, so falling back costs speed, never
    correctness.

    *dynamic_refs* names numeric swept inputs to trace as additional dynamic
    arguments. Success then applies across their values at the same shapes and
    dtypes. After a failed generalized probe, the caller checks rows at their
    fixed values.
    """
    try:
        dummy_kw = dict(values)
        broadcast_refs = set(broadcast_args)
        batched_sources: dict[_binding.FunctionInputRef, Any] = {}
        drawn_refs: list[_binding.FunctionInputRef] = []
        array_like_refs: list[_binding.FunctionInputRef] = []
        for ref in _binding.iter_input_refs(function._signature_info, values):
            v = _binding.input_ref_value(values, ref)
            if ref in broadcast_refs:
                if isinstance(v, Batch):
                    # A swept batch stays at its reference, where the mapped
                    # body reads the kind of its elements.
                    batched_sources[ref] = v
                else:
                    # Nothing to synthesize: the draw itself supplies the
                    # structure and dtype below, which is what lets a
                    # multi-field law be probed at all — it has no single
                    # event shape to stand in for one.
                    drawn_refs.append(ref)
            elif hasattr(v, "__array__") and not isinstance(v, jnp.ndarray):
                array_like_refs.append(ref)
        with _context._workflow_probe():
            if batched_sources:
                refs = list(batched_sources)
                # The executor's own body, not a copy maintained in the
                # probe. ``dummy_kw`` carries the swept batches at their
                # references and the other inputs as the call passed them.
                _row_call = _sweep.mapped_row_body(
                    func=func,
                    values=dummy_kw,
                    array_args=refs,
                    label=function.output_label,
                    output_is_declared=(
                        function.output_spec is not None
                        and function.output_spec.term_spec is not None
                    ),
                )
                # The first row of the storage the executor maps.
                probe_rows = tuple(
                    jax.tree.map(
                        lambda column: column[:1],
                        _sweep.mapped_storage(source, int(math.prod(source.batch_shape))),
                    )
                    for source in batched_sources.values()
                )
                jax.make_jaxpr(jax.vmap(_row_call))(probe_rows)
            elif drawn_refs:
                if stochastic_plan is None:  # pragma: no cover - planner contract guard
                    raise RuntimeError("distribution probe is missing its stochastic plan")
                refs = drawn_refs
                # The executor reads an enumerated group's atoms as it reads a
                # sampled group's draws, so every group is probed with a stand-in
                # for one draw of its root.
                root_probes = []
                for group in stochastic_plan.source_groups:
                    binding = stochastic_plan.runtime_bindings[group.index]
                    root = binding.root
                    event = root.event_spec.term_spec
                    if isinstance(event, BatchSpec):
                        # A batch-valued draw is mapped in its raw form.
                        root_probes.append(_batch_stand_in(root, event))
                        continue
                    held = _broadcast.batch_field(event)
                    if held is not None:
                        path, levels = held
                        raise _broadcast.UnmappedBatchError(
                            f"dispatch='jax' cannot map over the draws of {root.label!r}, whose "
                            f"field {path!r} stores a batch on the levels {list(levels)} for "
                            f"each draw. Lift over the marginal law at that field, as "
                            f"marginal(law, {path!r}) gives it, or use dispatch='auto', "
                            f"'sequential', or 'thread'."
                        )

                    output_record = _output_record(root.event_spec)
                    if (
                        not isinstance(output_record, NumericRecordSpec)
                        or not output_record.is_concrete
                    ):
                        raise TypeError(
                            f"{type(root).__name__} must declare a concrete numeric event "
                            "to run under dispatch='jax'"
                        )
                    try:
                        dtypes = root.dtypes
                    except (AttributeError, NotImplementedError) as error:
                        raise TypeError(
                            f"{type(root).__name__} must declare its field dtypes to run "
                            "under dispatch='jax'"
                        ) from error
                    columns = {}
                    for path in output_record:
                        dtype = dtypes.get(path)
                        if dtype is None:
                            dtype = dtypes.get(path.split("/", 1)[0])
                        if dtype is None:
                            dtype = _stored_dtype(root, path)
                        columns[path] = jax.ShapeDtypeStruct(
                            (1, *output_record[path].shape),
                            dtype,
                        )
                    # The stand-in draw is at the kind the law's event spec
                    # names, as the law's own draws are.
                    if _has_record_outputs(root.event_spec):
                        root_probe = NumericRecordBatch(
                            columns,
                            "draw",
                            element_spec=output_record,
                            axes_per_level=(1,),
                            label=root.label,
                        )
                    else:
                        root_probe = next(iter(columns.values()))
                    root_probes.append(root_probe)

                def probe_draw(root_values, dynamic_values):
                    probe_values = _binding.replace_input_refs(
                        dummy_kw, dict(zip(dynamic_refs, dynamic_values, strict=True))
                    )
                    _draw_call = _broadcast.mapped_draw_body(
                        func=func,
                        values=probe_values,
                        broadcast_args=refs,
                        batch_events=_broadcast.batch_events(stochastic_plan),
                    )
                    sampled = {}
                    for group, root_value in zip(
                        stochastic_plan.source_groups,
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

                dynamic_values = tuple(
                    _binding.input_ref_value(dummy_kw, ref) for ref in dynamic_refs
                )
                jax.make_jaxpr(jax.vmap(probe_draw, in_axes=(0, None)))(
                    tuple(root_probes), dynamic_values
                )
            else:
                # The trace takes the arguments as its inputs, so each argument
                # that converts to an array enters as a JAX array.
                traced_kw = _binding.replace_input_refs(
                    dummy_kw,
                    {
                        ref: jnp.asarray(_binding.input_ref_value(values, ref))
                        for ref in array_like_refs
                    },
                )
                jax.make_jaxpr(lambda kw: func(**kw))(traced_kw)
    except Exception as exc:
        return exc
    return None


def _batch_stand_in(root: Any, event: BatchSpec) -> Any:
    """A stand-in for one raw draw of *root*, whose event is the batch *event*, led by an axis of one.

    Parameters
    ----------
    root : Distribution
        The law whose draws the probe stands in for.
    event : BatchSpec
        The law's event spec.

    Returns
    -------
    Any
        A ``jax.ShapeDtypeStruct`` for array elements, and the nested mapping of
        them for record elements. Each has the axis of one, then the event's
        batch axes, then the element's own axes.

    Raises
    ------
    TypeError
        If the event's elements are not numeric, its shapes are not concrete, or
        a leaf's dtype is neither declared nor stored.
    """
    flat = _flat_batch_spec(event)
    element = flat.element_spec
    if not isinstance(element, NumericSpec) or not flat.is_concrete:
        raise TypeError(
            f"{type(root).__name__} must declare a concrete numeric event to run under "
            f"dispatch='jax'"
        )
    lead = (1, *flat.batch_shape)
    if isinstance(element, NumericArraySpec):
        leaves = {"": element}
    else:
        leaves = {path: element[path] for path in element}
    columns = {}
    for path, leaf in leaves.items():
        dtype = leaf.dtype if leaf.dtype is not None else _stored_dtype(root, path)
        if dtype is None:
            raise TypeError(
                f"{type(root).__name__} must declare its field dtypes to run under dispatch='jax'"
            )
        columns[path] = jax.ShapeDtypeStruct((*lead, *leaf.shape), dtype)
    if isinstance(element, NumericArraySpec):
        return columns[""]
    return _unflatten_paths(columns)


def _stored_dtype(root: Any, path: str) -> Any:
    """The dtype of the atoms an empirical *root* stores at the leaf *path*, or None.

    The probe reads it where the law's event spec leaves a dtype open, since an
    empirical law holds its atoms and reading them draws nothing. Atoms that
    are objects have no dtype.
    """
    if not isinstance(root, EmpiricalDistribution):
        return None
    rows = root._rows
    column = rows.get(path) if isinstance(rows, Mapping) else rows
    dtype = getattr(column, "dtype", None)
    return None if dtype is None or dtype.kind == "O" else dtype


def _require_jax_traceable(
    function: Function,
    values: dict[str, Any],
    broadcast_args: list[_binding.FunctionInputRef],
    *,
    func: Callable[..., Any],
    stochastic_plan: _plan.StochasticPlan | None,
) -> None:
    """Raise a clear error if explicit JAX dispatch cannot trace.

    The probe's errors that the other dispatch modes raise too are raised
    unchanged: the :class:`~._call.ApplicabilityError` of a point whose
    declarations violate the call contract, which planning raises, and the
    :class:`~._result.ResultSchemaError` or :class:`~._result.ResultKindError` of
    a result that violates its declaration, which the return raises. A batch
    the map cannot carry, such as a joint law's batch-valued field, raises a
    ``ValueError`` that names it.
    """
    trace_error = _jax_traceability_error(
        function, values, broadcast_args, func=func, stochastic_plan=stochastic_plan
    )
    if trace_error is None:
        return
    if isinstance(
        trace_error,
        (_call.ApplicabilityError, _result.ResultSchemaError, _result.ResultKindError),
    ):
        raise trace_error
    if isinstance(trace_error, _broadcast.UnmappedBatchError):
        raise ValueError(str(trace_error)) from None
    if isinstance(trace_error, _context._StochasticProbeSignal):
        message = (
            _context.JAX_SUBMISSION_MESSAGE
            if trace_error.submission
            else _context.JAX_KEYLESS_DRAW_MESSAGE
        )
        raise TypeError(message) from trace_error
    raise ValueError(
        "dispatch='jax' failed while tracing the wrapped function with JAX; "
        "ensure the function is JAX-traceable, or use dispatch='auto', "
        "'sequential', or 'thread'."
    ) from trace_error


def _resolve_dispatch(
    function: Function,
    argument_values: dict[str, Any],
    broadcast_args: list[_binding.FunctionInputRef],
    *,
    resolved_controls: Mapping[str, Any],
    jax_supported: bool = True,
    func: Callable[..., Any],
    stochastic_plan: _plan.StochasticPlan | None,
) -> str:
    """Resolve the dispatch strategy of a call under *resolved_controls*, caching JAX traceability detection.

    Returns ``"jax"``, ``"sequential"``, or ``"thread"``. This is
    independent of orchestration (``workflow_kind``), which wraps
    whichever strategy is chosen.
    """
    requested_dispatch = resolved_controls["dispatch"]
    if requested_dispatch != "auto":
        return requested_dispatch

    if not jax_supported:
        return "sequential"

    if (
        _jax_traceability_error(
            function, argument_values, broadcast_args, func=func, stochastic_plan=stochastic_plan
        )
        is None
    ):
        return "jax"
    else:
        logger.info(
            "Function '%s' is not JAX-traceable; using sequential dispatch.",
            function._label,
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
    num_samples : int, tuple of int, or None
        The number of Monte Carlo draws the route took, as
        :attr:`~probpipe.core._call_context._NumSamplesUsed.provenance_num_samples` gives
        it, or ``None`` when it took none.
    """

    name: str
    exact: bool | None
    rule: Any = None
    operand: Any = None
    parameter: str | None = None
    fixed_args: Mapping[str, Any] = field(default_factory=dict)
    controls: Mapping[str, Any] = field(default_factory=dict)
    method: str | None = None
    num_samples: int | tuple[int, ...] | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """The route's name and exactness, its registry method, and its Monte Carlo draws, as provenance records them."""
        recorded: dict[str, Any] = {"route": self.name, "exact": self.exact}
        if self.method is not None:
            recorded["method"] = self.method
        if self.num_samples is not None:
            recorded["num_samples"] = self.num_samples
        return recorded


#: The one route of a plain call: the function's body, which is exact.
_BODY = "body"

#: The name a check gives the selection that waits on a planned conversion.
_PLANNED = "the route of the converted call"


def _resolve_route(
    function: Function,
    argument_values: Mapping[str, Any],
    broadcast_plan: _plan.BroadcastPlan,
    rule_method: str | None,
    resolved_controls: Mapping[str, Any],
) -> _Route:
    """Select the route that realizes the call (step 6).

    A plain call has its body as its one candidate, which is exact, so
    ``exact_only`` admits it and a ``method`` name matches nothing. A lifted call
    resolves through the evaluation-rule registry on the function and the first
    lifted argument, a swept batch before a law, with the call's other arguments
    as the fixed arguments; *rule_method* names a rule and ``exact_only``
    excludes the approximate ones.

    Parameters
    ----------
    function : Function
        The function whose call the route realizes.
    argument_values : Mapping of str to Any
        The call's admitted arguments, by parameter name.
    broadcast_plan : BroadcastPlan
        The call's lift, whose regime and first lifted argument the selection
        reads.
    rule_method : str or None
        The evaluation rule the ``method`` control names, or ``None`` to select
        the first feasible rule by rank.
    resolved_controls : Mapping of str to Any
        The call's resolved controls, which the rules receive and whose
        ``exact_only`` excludes the approximate ones.

    Returns
    -------
    _Route
        The selected route, which for a lifted call holds the rule and the
        arguments it receives.

    Raises
    ------
    ResolutionError
        If ``method`` names no route, or no rule is feasible under the controls,
        naming each rule tried and what it lacked.
    """
    info, route = _route_report(
        function, argument_values, broadcast_plan, rule_method, resolved_controls
    )
    if route is None:
        if broadcast_plan.regime == "none":
            raise ResolutionError(info.description)
        ref = (broadcast_plan.array_args or broadcast_plan.dist_args)[0]
        detail = (
            f"pending: {', '.join(info.pending)}" if info.feasible is None else info.description
        )
        restriction = " (exact_only=True)" if resolved_controls["exact_only"] else ""
        raise ResolutionError(
            f"{function.label}: no evaluation rule applies to a call over {ref.label!r}"
            f"{restriction}. {detail}"
        )
    return route


def _route_report(
    function: Function,
    argument_values: Mapping[str, Any],
    broadcast_plan: _plan.BroadcastPlan,
    rule_method: str | None,
    resolved_controls: Mapping[str, Any],
) -> tuple[MethodInfo, _Route | None]:
    """The report of the route that realizes the call, and the route when it is selected.

    The report probes without executing, as :func:`_resolve_route` states the
    selection, under the call's *resolved_controls*; the route is ``None``
    unless the report is feasible.
    """
    method, exact_only = rule_method, resolved_controls["exact_only"]
    if broadcast_plan.regime == "none":
        if method is not None:
            return MethodInfo(
                False,
                f"{function.label}: a call that lifts nothing has its body as its one route, "
                f"so method={method!r} names no route; a method names an evaluation rule of a "
                f"call that lifts an argument",
            ), None
        return MethodInfo(True, method_name=_BODY, exact=True), _Route(_BODY, True)
    ref = (broadcast_plan.array_args or broadcast_plan.dist_args)[0]
    operand = _binding.input_ref_value(argument_values, ref)
    parameter = ref.parameter_name
    variadic = ref.subscript is not None
    fixed_args = {
        name: value for name, value in argument_values.items() if variadic or name != parameter
    }
    controls = dict(resolved_controls)
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

    Parameters
    ----------
    function : Function
        The function whose call is checked.
    args : tuple
        The call's positional arguments.
    kwargs : dict of str to Any
        The call's keyword arguments.

    Returns
    -------
    CallReport
        The report :meth:`Function.check` returns.

    Raises
    ------
    TypeError
        If the arguments do not bind to the signature.
    ApplicabilityError
        If an argument's kind is not accepted, a condition fails, or the
        declarations conflict.
    ValueError
        If an empty sweep lacks a concrete output spec after input
        dimensions bind.
    ResolutionError
        If a conversion has no converter, or the ``method`` control names no
        route of a Function realized by routes.
    """
    resolved_controls = current_call_context().resolved_controls
    bound = _call.resolve_function_call(
        function._signature_info,
        args,
        kwargs,
        bind=function._bind,
        function_name=function._label,
    )
    argument_values, conversions, waiting = _normalization.plan_distribution_values(
        values=bound,
        signature_info=function._signature_info,
        conversions=resolved_controls["conversions"],
    )
    _call.admit_arguments(
        function._signature_info,
        argument_values,
        input_spec=function.input_spec,
        function_name=function._label,
        roles=function._roles,
    )
    broadcast_plan = _plan.build_broadcast_plan(
        values=argument_values, signature_info=function._signature_info, roles=function._roles
    )
    candidates = function._route_candidates(resolved_controls, argument_values)
    if candidates is not None:
        return _check_routes(
            function,
            argument_values,
            broadcast_plan,
            candidates,
            conversions,
            waiting,
            resolved_controls,
        )
    lifted_refs = (*broadcast_plan.array_args, *broadcast_plan.dist_args)
    _, bindings = _bind_planned_function_inputs(
        function_name=function._label,
        input_spec=function.input_spec,
        values=argument_values,
        lifted_names={ref.parameter_name for ref in lifted_refs},
    )
    result = (
        function.output_spec.with_dim_sizes(**bindings)
        if function.output_spec is not None
        else None
    )
    _plan.require_empty_sweep_output(broadcast_plan, result)
    deferred = ()
    if result is None or result.term_spec is None:
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
        function, argument_values, broadcast_plan, resolved_controls["method"], resolved_controls
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
    argument_values: Mapping[str, Any],
    broadcast_plan: _plan.BroadcastPlan,
    candidates: tuple[Any, ...],
    conversions: Mapping[str, Any],
    waiting: tuple[str, ...],
    resolved_controls: Mapping[str, Any],
) -> CallReport:
    """The check of a call that a Function's routes realize at each of its points, under *resolved_controls*.

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
        info, _ = _route_report(function, argument_values, broadcast_plan, None, resolved_controls)
        if info.feasible is not True:
            return CallReport(
                routes=(info,),
                selected=None if info.feasible is None else info,
                lifted=lifted,
                conversions=MappingProxyType(dict(conversions)),
            )
    token = _call._CHECKING.set(True)
    try:
        points_report = _resolution.check_points(
            function, argument_values, resolved_controls, candidates, broadcast_plan
        )
    finally:
        _call._CHECKING.reset(token)
    _plan.require_empty_sweep_output(broadcast_plan, points_report.result)
    return _resolution.call_report(
        points_report, lifted=lifted, conversions=conversions, candidates=candidates
    )


def _run_registered_rule(
    function: Function,
    route: _Route,
    provenance_parents: list[TrackedTerm],
    provenance_inputs: Mapping[str, Any],
    label: str,
    expression: Expression | None,
) -> Any:
    """Execute a registered evaluation rule and give its result the call's identity.

    The rule returns the result over raw forms, and the return step wraps it at
    its kind under the call's result label and gives it *expression*, the
    function applied to draws of its inputs, with provenance recording the
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
        f"workflow.{function._label}",
        parents=provenance_parents,
        metadata={"func": function._label, **route.metadata},
        inputs=provenance_inputs,
        controls=controls,
        diagnostics=diagnostics,
    )
    return _result._coerce_output(
        _labeled_product(result),
        broadcast_mode=_result.BROADCAST_WRAP,
        provenance=provenance,
        label=label,
        expression=expression,
    )


@contextmanager
def _apply_scope() -> Generator[None, None, None]:
    """The scope of one :meth:`Function.apply` evaluation, which runs as a plain call's body runs.

    The evaluation is the one point of a lazy invocation, in the active
    workflow scope or in an ephemeral one outside any scope. Each
    workflow-owned draw of the body is therefore its own event, in program
    order, and an evaluation that draws nothing commits no occurrence of the
    scope.

    Raises
    ------
    ReplayCompatibilityError
        Inside ``replay_run``.
    UnmanagedConcurrentWorkflowEntryError
        If the active workflow scope belongs to another process, thread, or
        asyncio task, or has exited.
    """
    _context._assert_workflow_admission()
    _replay._reject_function_apply()
    with (
        _context._ephemeral_workflow_run(),
        _evaluation_keys._function_stochastic_scope() as broker,
        _execution.in_place_evaluation_point_scope(broker),
    ):
        yield


class _CallEngine:
    """The call stack of design Part V, installed as the call path of every Function."""

    def __call__(self, function: Function, /, *args: Any, **kwargs: Any) -> Any:
        result = _call_function(function, args, kwargs)
        resolved_controls = current_call_context().resolved_controls
        if not resolved_controls["raw"]:
            return result
        if function._route_candidates(resolved_controls) is not None:
            return _result.raw_form(result)
        return _result._detach(result)

    @staticmethod
    def check(function: Function, /, *args: Any, **kwargs: Any) -> CallReport:
        """The probe of steps 1 to 6 that :meth:`Function.check` returns."""
        return _check_call(function, args, kwargs)

    @staticmethod
    def invoke(
        function: Function, values: Mapping[str, Any], context: _FunctionInvocationContext
    ) -> Any:
        """The one point :meth:`Function.apply` evaluates, with no lifting, tracking, or provenance.

        A Function's body runs on *values*, each presented as
        :func:`._call.presented_arguments` states. A Function realized by routes
        admits each argument by its role, with no lifting, runs the route
        selected for the point, and returns the result's raw form.
        """
        resolved_controls = current_call_context().resolved_controls
        candidates = function._route_candidates(resolved_controls, values)
        if candidates is None:
            arguments = _call.presented_arguments(function._signature_info, values)
            return function._implementation.invoke(
                _binding.values_to_bound_arguments(function.signature, arguments), context=context
            )
        _call.admit_arguments(
            function._signature_info,
            values,
            function_name=function._label,
            roles=function._roles,
            lifts=False,
        )
        token = _call._CHECKING.set(False)
        try:
            term = _realized_point(
                function, values, resolved_controls, candidates, kind_error=_ResultKindMismatch
            )
        finally:
            _call._CHECKING.reset(token)
        return _result.raw_form(term)

    @staticmethod
    def apply_scope() -> AbstractContextManager[None]:
        """The scope :meth:`Function.apply` runs in: one point of a lazy invocation."""
        return _apply_scope()

    @staticmethod
    def workflow_kind(function: Function) -> WorkflowKind:
        """The orchestration mode that :attr:`Function.effective_workflow_kind` reports."""
        return effective_workflow_kind(function)


#: The installed engine.
_call_engine = _CallEngine()

install_call_engine(_call_engine)
