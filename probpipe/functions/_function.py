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
7. execute: the evaluation points run on the executor the call chooses, with
   structural keys (:mod:`._broadcast`, :mod:`._sweep`, :mod:`._execution`);
8. return: the result is validated, wrapped, labeled, and given provenance
   recording the selected route, or detached under ``raw=True``
   (:mod:`._result`).
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Generator, Hashable, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from types import MappingProxyType
from typing import Any, cast, overload

import jax
import numpy as np

from ..core._specs import _has_record_outputs, _output_record
from ..values import _binding

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from ..core._batch import Batch, BatchSpec, _flat_batch_spec
from ..core._call_context import (
    _jax_tracing_active,
    _jax_transformation_active,
    _NumSamplesUsed,
    current_call_context,
)
from ..core._dispatch import MethodInfo, ResolutionError
from ..core._evaluation_points import _DescribedEvaluationPoints, _requested_evaluation_points
from ..core._execution_claims import ProbeResult, probe
from ..core._executors import ExecutionReport, _evaluation_report
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
from ..core.named_tree import _PATH_SEP, _unflatten_paths
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._factored import _labeled_product
from ..distributions._views import CoupledMarginal
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
    _boundary,
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
    traceable: bool | None = None,
    vectorized: bool = False,
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

        @function(max_evaluations=100, executor="thread")
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
    traceable : bool or None
        Whether JAX can trace the function, as :class:`Function` takes it.
    vectorized : bool
        Whether the function takes a stack of evaluation points at once, as
        :class:`Function` takes it.
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
        If a keyword is no control, before any callable is wrapped; if
        *traceable* or *vectorized* is not a bool; or if *label* is omitted
        for a callable that has no ``__name__``, such as a
        ``functools.partial`` or a callable instance.
    """
    _refuse_unknown_controls(controls)

    def decorator(func: Callable[..., Any]) -> Function:
        return Function(
            func,
            input_spec=input_spec,
            output_spec=output_spec,
            output_label=output_label,
            traceable=traceable,
            vectorized=vectorized,
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
    if _jax_tracing_active():
        return WorkflowKind.OFF
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


def _call_function(
    function: Function,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
) -> Any:
    """Call *function* on *args* and *call_inputs*, returning a raw form inside a JAX transformation.

    Outside every JAX transformation the call runs in its workflow. Inside one, a
    plain call returns its result's raw form (II.4): inside a staging trace it
    is staged as an embedded call or runs inline, and under an eager
    transformation it runs in its workflow. An operation and a sweep return
    their terms.
    """
    if _jax_transformation_active():
        argument_values = _call.resolve_function_call(
            function._signature_info,
            args,
            call_inputs,
            bind=function._bind,
            function_name=function.label,
        )
        inline_plan = _plan.build_broadcast_plan(
            values=argument_values, signature_info=function._signature_info, roles=function._roles
        )
        staged, staged_result = (
            _boundary.maybe_stage(function, argument_values)
            if inline_plan.regime == "none"
            else (False, None)
        )
        if staged:
            return staged_result
        if _jax_tracing_active() and inline_plan.regime == "none":
            resolved_controls = current_call_context().resolved_controls
            route_candidates = function._route_candidates(resolved_controls, argument_values)
            if route_candidates is not None:
                _call.admit_arguments(
                    function._signature_info,
                    argument_values,
                    function_name=function.label,
                    roles=function._roles,
                    lifts=False,
                )
                # A law keeps its ProbPipe capabilities here. Its raw backend
                # object can have weaker guards than the law it represents.
                return _realized_point(
                    function, argument_values, resolved_controls, route_candidates
                )
            _, bindings = _bind_function_inputs(
                function_name=function.label, input_spec=function.input_spec, values=argument_values
            )
            body_result = function._invoke_resolved(
                argument_values, context=_FunctionInvocationContext(bindings)
            )
            _validate_function_output(
                function_name=function.label,
                output_spec=function.output_spec,
                result=body_result,
                bindings=bindings,
            )
            return _result.raw_form(body_result)
        if (
            inline_plan.regime == "none"
            and function._route_candidates(
                current_call_context().resolved_controls, argument_values
            )
            is None
        ):
            # Under an eager transformation a plain call runs as it does
            # outside one, and its result presents as its raw form (II.4), as
            # the inline call above returns it inside a staging trace.
            return _result.raw_form(_call_in_workflow(function, args, call_inputs))
    return _call_in_workflow(function, args, call_inputs)


def _call_in_workflow(
    function: Function, args: tuple[Any, ...], call_inputs: dict[str, Any]
) -> Any:
    """The call of *function* inside its workflow scope, replay scope, and key broker."""
    _context._assert_workflow_admission()
    with _replay._function_replay_scope() as replay_call:
        occurrence_path = None if replay_call is None else replay_call.occurrence_path
        with (
            _context._ephemeral_workflow_run(),
            _evaluation_keys._function_stochastic_scope(
                occurrence_path=occurrence_path
            ) as key_broker,
        ):
            if (
                replay_call is not None
                or _context._active_provenance_mode() is not ProvenanceMode.OFF
            ):
                callable_anchor = _callable.capture_function_anchor(function)
                key_broker.set_callable_anchor(callable_anchor)
                if replay_call is not None:
                    replay_call.validate_callable(callable_anchor)
            return _call_function_in_context(function, args, call_inputs)


def _call_function_in_context(
    function: Function,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
) -> Any:
    """Run the call of *function* in its open call context, then warn if an executor it named ran nothing."""
    # A call made while a check probes, as a probe that runs an operation makes
    # one, selects and runs its routes as any call does.
    checking_flag_reset = _call._CHECKING.set(False)
    try:
        call_result = _run_call(function, args, call_inputs)
    finally:
        _call._CHECKING.reset(checking_flag_reset)
    _execution.warn_if_named_executor_ran_nothing()
    return call_result


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
    parents = {id(law.parent): law.parent for law in laws if isinstance(law, CoupledMarginal)}
    if len(parents) == 1 and all(isinstance(law, CoupledMarginal) for law in laws):
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
    stochastic_plan = _lifted_call_stochastic_plan(
        argument_values, broadcast_plan, resolved_controls, route.name
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
    _replay._validate_active_plan(_recipe.serialize_stochastic_plan(stochastic_plan))
    invocation_bindings = _lifted_call_bindings(function, argument_values, broadcast_plan)
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
    _plan.require_distinct_swept_levels(
        broadcast_plan, concrete_output_spec, output_label=function.output_label
    )
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
        return evaluate_lifted_point(**point_values)

    evaluate_lifted_point = _lifted_point_evaluator(
        function,
        resolved_controls=resolved_controls,
        candidates=candidates,
        invocation_bindings=invocation_bindings,
        explicit_parameter_names=explicit_parameter_names,
    )

    route_cache_key = _route_cache_key(function, resolved_controls, candidates)

    def probe_point(
        point_values: dict[str, Any], lifted_refs: list[_binding.FunctionInputRef]
    ) -> ProbeResult:
        return _probe_lifted_point(
            function,
            point_values,
            lifted_refs,
            point_body=invoke_point,
            stochastic_plan=stochastic_plan,
            route_cache_key=route_cache_key,
        )

    # Whether the call has logged that its function does not trace, which it
    # logs once, however many cells of a sweep read the stored probe result.
    untraceable_logged = False

    def lifted_point_traceability(
        point_values: dict[str, Any], lifted_refs: list[_binding.FunctionInputRef]
    ) -> tuple[bool, str | None]:
        nonlocal untraceable_logged
        traceable, untraceable_reason = _lifted_point_traceability(
            function, partial(probe_point, point_values, lifted_refs)
        )
        if (
            not traceable
            and function._execution_claims.traceable is None
            and not untraceable_logged
        ):
            untraceable_logged = True
            logger.info(
                "Function '%s' is not JAX-traceable; its evaluation points run one at a time.",
                function._label,
            )
        return traceable, untraceable_reason

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
            lifted_point_traceability=lifted_point_traceability,
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
            swept_point_traceability=lifted_point_traceability,
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

    if selection is not None:
        # A route coordinates the work requested by the operation. Preserve
        # its logical context so embedded points inherit its executor without
        # occupying a worker needed by those same points.
        broker = _evaluation_keys._capture_active_broker()
        if broker is None:
            raise RuntimeError("an operation route has no active call broker")
        if not _context._workflow_side_effects_forbidden():
            _execution._record_execution_contract(None)
        with _execution.in_place_evaluation_point_scope(broker):
            result = invoke_point(**argument_values)
    else:
        # A plain call is one evaluation point, which varies only in being the
        # call's own: its arguments are one row of a set of one. TrackedTerm
        # values form lineage parents; every other resolved parameter remains
        # separately fingerprinted in Provenance.inputs.
        call_arguments = np.empty(1, dtype=object)
        call_arguments[0] = argument_values
        result = _execution.evaluate_plain_call(
            _requested_evaluation_points(
                lambda call_arguments: invoke_point(**call_arguments),
                varying={"call_arguments": call_arguments},
                origin=function._label,
                traceable=False,
                untraceable_reason=None,
                point_noun="evaluation point",
                point_segment=lambda point_index: _execution.plain_call_point_segment(),
            ),
            workflow_kind=workflow_kind,
            function_label=function._label,
        )
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


def _evaluated_body_point(
    function: Function, point_values: Mapping[str, Any], invocation_bindings: Mapping[str, int]
) -> Any:
    """The body of *function* at one point, validated against its output spec and wrapped as a term.

    Parameters
    ----------
    function : Function
        The function whose body runs.
    point_values : Mapping of str to Any
        The point's arguments, by parameter name.
    invocation_bindings : Mapping of str to int
        The dimension sizes the call's arguments bound.

    Returns
    -------
    Any
        The result at the kind the output spec names.

    Raises
    ------
    ApplicabilityError
        If the point's arguments violate the input spec.
    ResultKindError, ResultSchemaError
        If the result violates the output spec.
    """
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
    with _boundary.realizing_point(function):
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


def _lifted_execution_report(
    function: Function,
    argument_values: Mapping[str, Any],
    broadcast_plan: _plan.BroadcastPlan,
    candidates: tuple[Any, ...] | None,
    resolved_controls: Mapping[str, Any],
    route_name: str | None,
) -> ExecutionReport:
    """How the lifted call or sweep that ``check`` checks would run its function's evaluation points.

    The call asks for one evaluation point per draw, combination, or grid
    point when the engine's evaluation rule realizes it. A plain call, a call
    that another evaluation rule realizes, and a sweep over no cells ask for
    none. The call
    probes a function that states nothing when its route maps the body,
    whatever executor it names, since a function JAX can trace is always
    mapped by ``jax``. So ``check`` probes under the same conditions, with the
    same stand-ins, and stores the outcome where the call reads it. A probe of
    a nested sweep reads its first cell. A probe that fails stores its outcome
    and raises nothing, as the call's would.

    Parameters
    ----------
    function : Function
        The checked function.
    argument_values : Mapping of str to Any
        The call's normalized arguments.
    broadcast_plan : BroadcastPlan
        The call's lifting regime.
    candidates : tuple or None
        The routes of a Function realized by routes, which realize each
        evaluation point; ``None`` for a Function whose body does.
    resolved_controls : Mapping of str to Any
        The checked call's resolved controls.
    route_name : str or None
        The evaluation rule selected for the lifted call.

    Returns
    -------
    ExecutionReport
        The report of the function's evaluation points, or an empty report
        when the call asks for none.

    Raises
    ------
    ResolutionError
        If the executor the call names is not registered.
    ValueError
        If the selected executor does not take one of its options.
    """
    if broadcast_plan.regime == "none" or route_name not in _rules._ENGINE_RULES:
        return ExecutionReport()
    stochastic_plan = _lifted_call_stochastic_plan(
        argument_values, broadcast_plan, resolved_controls, route_name
    )
    if stochastic_plan is None:
        num_evaluation_points, point_noun = broadcast_plan.n_sweep, "grid point"
    else:
        num_evaluation_points = stochastic_plan.n_evaluations * (
            len(stochastic_plan.logical_units) if broadcast_plan.regime == "nested" else 1
        )
        point_noun = "draw" if stochastic_plan.evaluation_mode == "sampled" else "combination"
    if num_evaluation_points == 0:
        return ExecutionReport()
    invocation_bindings = _lifted_call_bindings(function, argument_values, broadcast_plan)
    evaluate_lifted_point = _lifted_point_evaluator(
        function,
        resolved_controls=resolved_controls,
        candidates=candidates,
        invocation_bindings=invocation_bindings,
    )
    if broadcast_plan.regime == "nested":
        probe_values = _sweep.slice_sweep_values(
            values=argument_values, index=0, array_groups=broadcast_plan.array_groups
        )
        lifted_refs = list(broadcast_plan.dist_args)
    elif broadcast_plan.regime == "sweep":
        probe_values, lifted_refs = dict(argument_values), list(broadcast_plan.array_args)
    else:
        probe_values, lifted_refs = dict(argument_values), list(broadcast_plan.dist_args)
    untraceable_reason = (
        _sweep.untraceable_sweep_reason(
            argument_values, list(broadcast_plan.array_args), broadcast_plan
        )
        if broadcast_plan.regime == "sweep"
        else None
    )
    if untraceable_reason is not None:
        traceable = False
    else:
        traceable, untraceable_reason = _lifted_point_traceability(
            function,
            partial(
                _probe_lifted_point,
                function,
                probe_values,
                lifted_refs,
                point_body=evaluate_lifted_point,
                stochastic_plan=stochastic_plan,
                route_cache_key=_route_cache_key(function, resolved_controls, candidates),
            ),
        )
    described_points = _DescribedEvaluationPoints(
        function,
        num_evaluation_points=num_evaluation_points,
        shared=argument_values,
        origin=function._label,
        traceable=traceable,
        untraceable_reason=untraceable_reason,
        point_noun=point_noun,
    )
    return ExecutionReport(
        (
            _evaluation_report(
                described_points,
                current_call_context().execution_choice,
                traceability_source=(
                    "claimed" if function._execution_claims.traceable is not None else "probed"
                ),
            ),
        )
    )


def _lifted_call_bindings(
    function: Function, argument_values: Mapping[str, Any], broadcast_plan: _plan.BroadcastPlan
) -> Mapping[str, int]:
    """The dimension sizes the arguments of a call of *function* bind, its lifted arguments left out."""
    _, invocation_bindings = _bind_planned_function_inputs(
        function_name=function._label,
        input_spec=function.input_spec,
        values=argument_values,
        lifted_names={
            ref.parameter_name for ref in (*broadcast_plan.dist_args, *broadcast_plan.array_args)
        },
    )
    return invocation_bindings


def _lifted_call_stochastic_plan(
    argument_values: Mapping[str, Any],
    broadcast_plan: _plan.BroadcastPlan,
    resolved_controls: Mapping[str, Any],
    route_name: str | None,
) -> _plan.StochasticPlan | None:
    """The plan of the laws a lifted call draws from, which enumerates every group unless *route_name* samples."""
    return _plan.build_stochastic_plan(
        argument_values,
        broadcast_plan,
        resolved_controls["max_evaluations"],
        enumerate_every_group=route_name != "sampling_lift",
    )


def _lifted_point_evaluator(
    function: Function,
    *,
    resolved_controls: Mapping[str, Any],
    candidates: tuple[Any, ...] | None,
    invocation_bindings: Mapping[str, int],
    explicit_parameter_names: frozenset[str] | None = None,
) -> Callable[..., Any]:
    """The evaluation of one point of a lifted call of *function*, a function of the point's arguments by name.

    A Function realized by routes realizes each point by the route it selects
    there, and any other Function runs its body. The call and ``check`` map and
    probe this one evaluation, so the two agree.

    Parameters
    ----------
    function : Function
        The function whose call is lifted.
    resolved_controls : Mapping of str to Any
        The call's resolved controls.
    candidates : tuple or None
        The routes of a Function realized by routes, and ``None`` otherwise.
    invocation_bindings : Mapping of str to int
        The dimension sizes the call's arguments bound.
    explicit_parameter_names : frozenset of str or None
        The parameters a result's expression shows, as :func:`_realized_point`
        takes them.

    Returns
    -------
    Callable
        ``evaluate(**point_values)``, the point's result as a term.
    """
    if candidates is not None:

        def realize_point(**point_values: Any) -> Any:
            return _realized_point(
                function,
                point_values,
                resolved_controls,
                candidates,
                explicit_parameter_names=explicit_parameter_names,
            )

        return realize_point

    def evaluate_body(**point_values: Any) -> Any:
        return _evaluated_body_point(function, point_values, invocation_bindings)

    return evaluate_body


def _probe_lifted_point(
    function: Function,
    argument_values: dict[str, Any],
    lifted_refs: list[_binding.FunctionInputRef],
    *,
    point_body: Callable[..., Any],
    stochastic_plan: _plan.StochasticPlan | None,
    route_cache_key: Hashable,
) -> ProbeResult:
    """Whether one evaluation point of a lifted call traces, read from the function's probe results.

    The probe traces the operation ``jax`` maps, not merely the body: a
    body can trace on its own yet fail under the map, as one that returns a
    record holding a batch does. So the probe maps the body over stand-ins of
    its lifted arguments, each fed the way the map will feed it: a swept batch
    over its first row, a law over one draw, and a batch-valued draw in its
    raw form, which a parameter annotated with a batch class receives rebuilt
    as the batch. Every other argument is taken as data when it is numeric,
    so the result holds for any value of its shape and dtype, and as it is
    otherwise (V.13, probe rule 1).

    The result is stored in the function's probe results under the input
    structure of one evaluation point and *route_cache_key*, so every call and
    ``check`` with inputs of that structure, and every ``using`` copy of the
    function, reads it. A probe that fails means the evaluation points run one
    at a time, which always runs the call, so falling back costs speed, never
    correctness.

    Parameters
    ----------
    function : Function
        The function whose call is lifted.
    argument_values : dict of str to Any
        The call's argument values, by parameter name.
    lifted_refs : list of FunctionInputRef
        The lifted arguments: swept batches, or laws the call draws from.
    point_body : Callable
        The evaluation of one point, a function of its arguments by name.
    stochastic_plan : StochasticPlan or None
        The plan of the laws the call draws from, when it draws.
    route_cache_key : Hashable
        What one point runs besides the function's own body, such as the
        controls a Function realized by routes selects by.

    Returns
    -------
    ProbeResult
        The outcome, whose ``error`` is the error the map raised when it did
        not trace.
    """
    swept_batches: dict[_binding.FunctionInputRef, Any] = {}
    drawn_refs: list[_binding.FunctionInputRef] = []
    for lifted_ref in lifted_refs:
        lifted_value = _binding.input_ref_value(argument_values, lifted_ref)
        if isinstance(lifted_value, Batch):
            swept_batches[lifted_ref] = lifted_value
        else:
            drawn_refs.append(lifted_ref)
    shared_refs: list[_binding.FunctionInputRef] = []
    # An argument JAX cannot flatten, such as one that holds itself, is passed
    # to the probe as it is, and keeps the result from being stored.
    every_shared_argument_flattens = True
    for input_ref in _binding.iter_input_refs(function._signature_info, argument_values):
        if input_ref in lifted_refs:
            continue
        try:
            jax.tree.flatten(_binding.input_ref_value(argument_values, input_ref))
        except Exception:
            every_shared_argument_flattens = False
        else:
            shared_refs.append(input_ref)
    shared_values = tuple(
        _binding.input_ref_value(argument_values, shared_ref) for shared_ref in shared_refs
    )

    def with_shared_values(probed_shared_values: tuple[Any, ...]) -> dict[str, Any]:
        return _binding.replace_input_refs(
            argument_values, dict(zip(shared_refs, probed_shared_values, strict=True))
        )

    try:
        if swept_batches:
            swept_refs = list(swept_batches)
            # The first row of the storage the executor maps.
            lifted_stand_ins = tuple(
                jax.tree.map(
                    lambda column: column[:1],
                    _sweep.mapped_storage(batch, int(math.prod(batch.batch_shape))),
                )
                for batch in swept_batches.values()
            )
            lifted_cache_key = (
                "rows",
                tuple(
                    (swept_ref, type(batch), batch.element_spec.term_spec)
                    for swept_ref, batch in swept_batches.items()
                ),
            )

            def map_point_body(probe_inputs: tuple[Any, tuple[Any, ...]]) -> Any:
                rows, probed_shared_values = probe_inputs
                # The executor's own body, not a copy maintained in the probe.
                evaluate_row = _sweep.mapped_row_body(
                    func=point_body,
                    values=with_shared_values(probed_shared_values),
                    array_args=swept_refs,
                    label=function.output_label,
                    output_is_declared=(
                        function.output_spec is not None
                        and function.output_spec.term_spec is not None
                    ),
                )
                return jax.vmap(evaluate_row)(rows)

        elif drawn_refs:
            if stochastic_plan is None:  # pragma: no cover - planner contract guard
                raise RuntimeError("distribution probe is missing its stochastic plan")
            lifted_stand_ins = tuple(
                _root_draw_stand_in(stochastic_plan.runtime_bindings[group.index].root)
                for group in stochastic_plan.source_groups
            )
            lifted_cache_key = (
                "draws",
                tuple(
                    (
                        group.consumers,
                        stochastic_plan.runtime_bindings[group.index].consumer_batches,
                    )
                    for group in stochastic_plan.source_groups
                ),
            )

            record_draws = _broadcast.record_events(stochastic_plan)

            def map_point_body(probe_inputs: tuple[Any, tuple[Any, ...]]) -> Any:
                root_draws, probed_shared_values = probe_inputs
                evaluate_draw = _broadcast.mapped_draw_body(
                    func=point_body,
                    values=with_shared_values(probed_shared_values),
                    broadcast_args=drawn_refs,
                    batch_events=_broadcast.batch_events(stochastic_plan),
                    record_events=record_draws,
                )

                def evaluate_at_root_draws(root_values: tuple[Any, ...]) -> Any:
                    consumed_values = {}
                    for group, root_value in zip(
                        stochastic_plan.source_groups, root_values, strict=True
                    ):
                        binding = stochastic_plan.runtime_bindings[group.index]
                        for consumer, evaluate_consumer in zip(
                            group.consumers, binding.consumer_evaluators, strict=True
                        ):
                            consumed_values[consumer.arg_ref] = _broadcast.lifted_row_form(
                                evaluate_consumer(root_value), consumer.arg_ref, record_draws
                            )
                    return evaluate_draw(
                        tuple(consumed_values[drawn_ref] for drawn_ref in drawn_refs)
                    )

                return jax.vmap(evaluate_at_root_draws)(root_draws)

        else:
            lifted_stand_ins = ()
            lifted_cache_key = ("values",)

            def map_point_body(probe_inputs: tuple[Any, tuple[Any, ...]]) -> Any:
                _, probed_shared_values = probe_inputs
                return point_body(**with_shared_values(probed_shared_values))

        probe_cache_key = (
            _hashable_or_none((route_cache_key, lifted_cache_key))
            if every_shared_argument_flattens
            else None
        )
    except Exception as error:
        # A lifted argument with no stand-in, such as a law whose event
        # declares no dtype, cannot be mapped; the result depends on the law,
        # so it is not stored.
        return ProbeResult(traces=False, error=error.with_traceback(None))

    def map_point_body_without_workflow_effects(
        probe_inputs: tuple[Any, tuple[Any, ...]],
    ) -> Any:
        # A workflow draw reached while tracing fails the probe,
        # even when the body catches the refusal.
        with _context._workflow_probe():
            return map_point_body(probe_inputs)

    return probe(
        None if probe_cache_key is None else function._probe_results,
        map_point_body_without_workflow_effects,
        (lifted_stand_ins, shared_values),
        label=function.label,
        cache_key_prefix=() if probe_cache_key is None else probe_cache_key,
    )


def _hashable_or_none(cache_key: Hashable) -> Hashable | None:
    """*cache_key* when it hashes, and ``None`` when some part of it does not."""
    try:
        hash(cache_key)
    except TypeError:
        return None
    return cache_key


def _route_cache_key(
    function: Function, resolved_controls: Mapping[str, Any], candidates: tuple[Any, ...] | None
) -> Hashable:
    """What one evaluation point of a lifted call runs besides the function's own body.

    A Function's body is the same under every control, so its probe results
    are keyed by the input structure alone, and every ``using`` copy shares
    them. A Function realized by routes selects a route at each point by its
    controls, so they enter the key.
    """
    if candidates is None:
        return ("body",)
    return ("routes", _frozen_controls(resolved_controls))


def _frozen_controls(control_value: Any) -> Any:
    """*control_value* with each mapping and sequence in it made a tuple, so a hashable one hashes."""
    if isinstance(control_value, Mapping):
        return tuple(
            sorted(
                ((str(name), _frozen_controls(value)) for name, value in control_value.items()),
                key=lambda item: item[0],
            )
        )
    if isinstance(control_value, list | tuple):
        return tuple(_frozen_controls(value) for value in control_value)
    return control_value


def _root_draw_stand_in(root: Any) -> Any:
    """A stand-in for one draw of *root*, led by an axis of one, which the probe maps.

    An enumerated group's atoms are read as a sampled group's draws are, so
    every root stands in by one draw of its event.

    Parameters
    ----------
    root : Distribution
        The law the call draws from.

    Returns
    -------
    Any
        A ``jax.ShapeDtypeStruct`` for a law over an array, a
        ``NumericRecordBatch`` of them for a law over a record, and the raw
        stand-in of a batch for a batch-valued draw.

    Raises
    ------
    UnmappedBatchError
        If the law's record event stores a batch for each draw in a field.
    TypeError
        If the law's event is not numeric and concrete, or a field's dtype is
        neither declared nor stored.
    """
    event_term_spec = root.event_spec.term_spec
    if isinstance(event_term_spec, BatchSpec):
        # A batch-valued draw is mapped in its raw form.
        return _batch_stand_in(root, event_term_spec)
    batch_valued_field = _broadcast.batch_field(event_term_spec)
    if batch_valued_field is not None:
        path, levels = batch_valued_field
        raise _broadcast.UnmappedBatchError(
            f"jax.vmap cannot map over the draws of {root.label!r}, whose field {path!r} "
            f"stores a batch on the levels {list(levels)} for each draw. Lift over the "
            f"marginal law at that field, as marginal(law, {path!r}) gives it"
        )
    output_record = _output_record(root.event_spec)
    if not isinstance(output_record, NumericRecordSpec) or not output_record.is_concrete:
        raise TypeError(
            f"{type(root).__name__} must declare a concrete numeric event to be mapped by JAX"
        )
    try:
        field_dtypes = root.dtypes
    except (AttributeError, NotImplementedError) as error:
        raise TypeError(
            f"{type(root).__name__} must declare its field dtypes to be mapped by JAX"
        ) from error
    stand_in_columns = {}
    for path in output_record:
        dtype = field_dtypes.get(path)
        if dtype is None:
            dtype = field_dtypes.get(path.split(_PATH_SEP, 1)[0])
        if dtype is None:
            dtype = _stored_dtype(root, path)
        stand_in_columns[path] = jax.ShapeDtypeStruct((1, *output_record[path].shape), dtype)
    # The stand-in draw is at the kind the law's event spec names, as the
    # law's own draws are.
    if _has_record_outputs(root.event_spec):
        return NumericRecordBatch(
            stand_in_columns,
            "draw",
            element=output_record,
            axes_per_level=(1,),
            label=root.label,
        )
    return next(iter(stand_in_columns.values()))


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
    element = flat.element_spec.term_spec
    if not isinstance(element, NumericSpec) or not flat.is_concrete:
        raise TypeError(
            f"{type(root).__name__} must declare a concrete numeric event to be mapped by JAX"
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
                f"{type(root).__name__} must declare its field dtypes to be mapped by JAX"
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


def _lifted_point_traceability(
    function: Function, probe_point: Callable[[], ProbeResult]
) -> tuple[bool, str | None]:
    """Whether JAX can trace one evaluation point of a lifted call of *function*, with the reason when it cannot.

    A function's ``traceable`` claim decides with no probe. A function that
    states nothing is traceable when one evaluation point traces, which
    *probe_point* reads from the function's stored probe results or probes
    once. A point that traced but failed ProbPipe's check of its inputs or of
    its output counts as traced, since every executor refuses it the same way.

    Parameters
    ----------
    function : Function
        The function whose call is lifted.
    probe_point : Callable
        Reads or runs the probe of one evaluation point.

    Returns
    -------
    tuple of (bool, str or None)
        Whether JAX can trace an evaluation point, and why not when it cannot.
    """
    traceable_claim = function._execution_claims.traceable
    if traceable_claim is not None:
        return traceable_claim, None if traceable_claim else "it claims traceable=False"
    probe_result = probe_point()
    if probe_result.traces:
        return True, None
    if probe_result.draws_global_random:
        return False, (
            "it draws from NumPy's global random generator, whose noise would be fixed into "
            "the compiled JAX program"
        )
    trace_error = probe_result.error
    if isinstance(
        trace_error,
        (_call.ApplicabilityError, _result.ResultSchemaError, _result.ResultKindError),
    ):
        # The callable traced, and ProbPipe's check of its inputs or output
        # refused the point, which it refuses under every executor.
        return True, None
    if isinstance(trace_error, _context._StochasticProbeSignal):
        return False, "it draws random numbers without an explicit key"
    if trace_error is None:
        return False, "tracing it failed"
    return False, f"tracing it raised {type(trace_error).__name__}: {trace_error}"


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
        dimensions bind, or a swept level is also a level of the batch the
        output spec declares.
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
    output_spec = (
        function.output_spec.with_dim_sizes(**bindings)
        if function.output_spec is not None
        else None
    )
    _plan.require_empty_sweep_output(broadcast_plan, output_spec)
    deferred = ()
    if output_spec is None or output_spec.term_spec is None:
        deferred = ("the result's type is completed from the returned value",)
    lifted = tuple(dict.fromkeys(ref.parameter_name for ref in lifted_refs))
    if waiting:
        # The lift of a converted argument is decided by the law the conversion
        # constructs, so the selection, and the kind of the result, wait on it.
        pending = MethodInfo(None, pending=waiting, method_name=_PLANNED, exact=False)
        return CallReport(
            routes=(pending,),
            deferred=deferred,
            lifted=lifted,
            conversions=MappingProxyType(conversions),
        )
    planned = _planned_result(
        function, output_spec, argument_values, broadcast_plan, resolved_controls
    )
    route_info, route = _route_report(
        function, argument_values, broadcast_plan, resolved_controls["method"], resolved_controls
    )
    execution = (
        ExecutionReport()
        if route is None
        else _lifted_execution_report(
            function, argument_values, broadcast_plan, None, resolved_controls, route.name
        )
    )
    return CallReport(
        routes=(route_info,),
        selected=None if route_info.feasible is None else route_info,
        deferred=deferred,
        result=planned,
        lifted=lifted,
        conversions=MappingProxyType(conversions),
        execution=execution,
    )


def _planned_result(
    function: Function,
    output_spec: OutputSpec | None,
    argument_values: Mapping[str, Any],
    broadcast_plan: _plan.BroadcastPlan,
    resolved_controls: Mapping[str, Any],
) -> OutputSpec | None:
    """The call's planned output spec, from the evaluation's output spec *output_spec*.

    The spec is planned as :func:`._plan.planned_result` states, under the
    function's output name and result label and the call's ``include_inputs``.
    """
    return _plan.planned_result(
        output_spec,
        broadcast_plan,
        values=argument_values,
        output_name=function._output_name,
        output_label=function.output_label,
        include_inputs=resolved_controls["include_inputs"],
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
    execution = ExecutionReport()
    if broadcast_plan.regime != "none":
        route_info, route = _route_report(
            function, argument_values, broadcast_plan, None, resolved_controls
        )
        if route_info.feasible is not True:
            return CallReport(
                routes=(route_info,),
                selected=None if route_info.feasible is None else route_info,
                lifted=lifted,
                conversions=MappingProxyType(dict(conversions)),
            )
        if route is not None:
            execution = _lifted_execution_report(
                function, argument_values, broadcast_plan, candidates, resolved_controls, route.name
            )
    checking_flag_reset = _call._CHECKING.set(True)
    try:
        points_report = _resolution.check_points(
            function, argument_values, resolved_controls, candidates, broadcast_plan
        )
    finally:
        _call._CHECKING.reset(checking_flag_reset)
    _plan.require_empty_sweep_output(broadcast_plan, points_report.result)
    return _resolution.call_report(
        points_report,
        result=_planned_result(
            function, points_report.result, argument_values, broadcast_plan, resolved_controls
        ),
        lifted=lifted,
        conversions=conversions,
        candidates=candidates,
        execution=execution,
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
        staged, result = _boundary.maybe_stage(function, values)
        if staged:
            return result
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
