"""Stage opaque Function calls and run their evaluation points on the host."""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

import jax
import numpy as np
from jax._src import core as jax_core

from ..core._call_context import (
    _ACTIVE_CALLS,
    _AUTOMATIC_SELECTION,
    _TRACED_CALL_TOKEN,
    _jax_tracing_active,
    _jax_transformation_active,
    _under_call_context,
    current_call_context,
)
from ..core._compilation import (
    _EmbeddedCallTracingError,
    _install_embedded_call_inspector,
)
from ..core._evaluation_points import EvaluationPoints
from ..core._execution_claims import ExecutionClaims, spec_stand_in
from ..core._executors import _evaluated_outputs
from ..core._specs import OutputSpec
from ..values import Function, _binding
from ..values._function_base import _bind_function_inputs, _FunctionInvocationContext
from . import _call


class _EmbeddedEvaluationFailureSignal(RuntimeError):
    """Stop the JAX program; the calling thread raises the recorded host exception."""


_REALIZING_FUNCTIONS: ContextVar[frozenset[int]] = ContextVar(
    "probpipe_realizing_functions", default=frozenset()
)


@contextmanager
def realizing_point(function: Function) -> Iterator[None]:
    """Run a point whose traceability the enclosing engine already established."""
    scope = _REALIZING_FUNCTIONS.set(_REALIZING_FUNCTIONS.get() | {id(function)})
    try:
        yield
    finally:
        _REALIZING_FUNCTIONS.reset(scope)


@dataclass(frozen=True)
class _HostPoint:
    """One raw evaluation, reconstructing static and varying Function arguments."""

    function: Function
    argument_tree: Any
    static_leaves: tuple[Any, ...]
    varying_positions: tuple[int, ...]

    def __call__(self, **arguments: Any) -> Any:
        leaves = list(self.static_leaves)
        for number, position in enumerate(self.varying_positions):
            leaves[position] = arguments[f"argument_{number}"]
        values = jax.tree.unflatten(self.argument_tree, leaves)
        if self.function._execution_claims.vectorized:
            presented = _call.presented_arguments(self.function._signature_info, values)
            return self.function._implementation.invoke(
                _binding.values_to_bound_arguments(self.function.signature, presented),
                context=_FunctionInvocationContext({}),
            )
        bound = _binding.values_to_bound_arguments(self.function.signature, values)
        return self.function.apply(*bound.args, **bound.kwargs)


def maybe_stage(function: Function, values: Mapping[str, Any]) -> tuple[bool, Any]:
    """Stage an opaque call with traced arguments, or leave a traceable call inline.

    Returns whether the call was staged and its raw result. Function routes and
    ordinary calls remain in the engine. Opaque constant calls are staged too,
    so constructing a kernel never runs their bodies.
    """
    if not _jax_transformation_active() or id(function) in _REALIZING_FUNCTIONS.get():
        return False, None
    leaves, argument_tree = jax.tree.flatten(dict(values))
    context = current_call_context()
    if function._route_candidates(context.resolved_controls, values) is not None:
        return False, None
    claim = function._execution_claims.traceable
    if claim is True:
        return False, None
    _, bindings = _bind_function_inputs(
        function_name=function.label, input_spec=function.input_spec, values=values
    )
    if claim is None:
        # Reuse the ordinary plain/lifted-point probe and its structure key.
        from ._function import _probe_lifted_point

        def invoke_body(**point_values: Any) -> Any:
            presented = _call.presented_arguments(function._signature_info, point_values)
            return function._implementation.invoke(
                _binding.values_to_bound_arguments(function.signature, presented),
                context=_FunctionInvocationContext(bindings),
            )

        result = _probe_lifted_point(
            function,
            dict(values),
            [],
            point_body=invoke_body,
            stochastic_plan=None,
            route_cache_key=("body",),
        )
        if result.traces and not result.draws_global_random:
            return False, None
    output_spec = (
        function.output_spec.with_dim_sizes(**bindings)
        if function.output_spec is not None
        else None
    )
    result_shapes = _callback_output(function.label, output_spec)
    positions = tuple(
        number
        for number, leaf in enumerate(leaves)
        if isinstance(leaf, (jax_core.Tracer, jax.Array, np.ndarray, np.generic, int, float, bool))
    )
    varying = tuple(leaves[position] for position in positions)
    ranks = tuple(np.ndim(leaf) for leaf in varying)
    static_leaves = tuple(
        None if number in positions else leaf for number, leaf in enumerate(leaves)
    )
    point = _HostPoint(function, argument_tree, static_leaves, positions)
    traced_token = _TRACED_CALL_TOKEN.get()
    expects_token = traced_token is not None
    if expects_token:
        call_token = traced_token
    elif _jax_tracing_active():
        call_token = np.uint32(0)
        _warn_user_trace(function.label)
    elif context.embedded_calls is not None:
        call_token = context.embedded_calls.token()
    else:
        call_token = np.uint32(0)
    host = functools.partial(
        _host_side,
        point=point,
        ranks=ranks,
        output_spec=output_spec,
        result_shapes=result_shapes,
        expects_token=expects_token,
        staged_choice=(
            _AUTOMATIC_SELECTION
            if not expects_token and _jax_tracing_active()
            else replace(context.execution_choice, named_executor_use=None)
        ),
    )

    @jax.custom_jvp
    def embedded(token: Any, *arguments: Any) -> Any:
        return jax.pure_callback(host, result_shapes, token, *arguments, vmap_method="expand_dims")

    @embedded.defjvp
    def refuse_gradient(primals: Any, tangents: Any) -> Any:
        raise TypeError(
            f"gradients cannot pass through {function.label!r}, "
            "which is not traceable and supplies no derivative"
        )

    return True, embedded(call_token, *varying)


def _callback_output(function_name: str, output_spec: OutputSpec | None) -> Any:
    """The concrete numeric output tree a callback can return, or its declaration error."""
    spec = output_spec.term_spec if output_spec is not None else None
    shapes = spec_stand_in(spec)
    if shapes is None:
        reason = "declare concrete numeric output shapes with output_spec"
    elif not jax.tree.leaves(shapes):
        reason = "declare at least one numeric output"
    else:
        # The callback must know a width; it cannot learn it by running opaque code.
        from ..core._record_spec import RecordSpec
        from ..core._spec_base import NumericArraySpec

        def has_dtype(term_spec: Any) -> bool:
            if isinstance(term_spec, NumericArraySpec):
                return term_spec.dtype is not None
            if isinstance(term_spec, RecordSpec):
                return all(has_dtype(child) for child in term_spec.children.values())
            return False

        if has_dtype(spec):
            return shapes
        reason = "declare the dtype of every numeric output in output_spec"
    raise _EmbeddedCallTracingError(
        f"cannot call host-only Function {function_name!r} inside a JAX trace: {reason}"
    )


def _host_side(
    call_token: Any,
    *arguments: Any,
    point: _HostPoint,
    ranks: tuple[int, ...],
    output_spec: OutputSpec,
    result_shapes: Any,
    expects_token: bool,
    staged_choice: Any,
) -> Any:
    """Flatten callback lanes into one executor request and restore their batch shape."""
    token = int(np.asarray(call_token).reshape(-1)[0])
    if expects_token and token == 0:
        raise _EmbeddedCallTracingError(_captured_token_message(point.function.label))
    context = _ACTIVE_CALLS.lookup(token)
    if staged_choice != _AUTOMATIC_SELECTION:
        token_call_use = context.execution_choice.named_executor_use
        if token_call_use is not None:
            token_call_use.note_ran_elsewhere()
        context = replace(context, execution_choice=staged_choice)
    if context.execution_choice.applies_only_to is point.function:
        context = replace(
            context,
            execution_choice=replace(context.execution_choice, applies_only_to=point),
        )
    elif context.execution_choice.names_executor():
        context = replace(
            context, execution_choice=replace(context.execution_choice, source="inherited")
        )
    context = replace(context, forbids_workflow_randomness=True)
    try:
        batch_shape = np.broadcast_shapes(
            *(
                np.shape(argument)[: len(np.shape(argument)) - rank]
                for argument, rank in zip(arguments, ranks, strict=True)
            )
        )
        count = math.prod(batch_shape)
        varying = {}
        for number, (argument, rank) in enumerate(zip(arguments, ranks, strict=True)):
            event_shape = np.shape(argument)[-rank:] if rank else ()
            varying[f"argument_{number}"] = np.broadcast_to(
                argument, (*batch_shape, *event_shape)
            ).reshape((count, *event_shape))
        if not varying:
            # A constant call still has one evaluation point; the index only
            # gives the executor its cardinality and is not a body argument.
            varying["point_index"] = np.arange(count, dtype="int32")
        with _under_call_context(context):
            points = EvaluationPoints(
                point,
                varying=varying,
                origin=point.function.label,
                output_spec=output_spec,
                claims=ExecutionClaims(
                    traceable=False, vectorized=point.function._execution_claims.vectorized
                ),
            )
            outputs = _evaluated_outputs(points, stacked=True)
        return jax.tree.map(
            lambda output, shape: np.asarray(output).reshape((*batch_shape, *shape.shape)),
            outputs,
            result_shapes,
        )
    except Exception as error:
        if context.embedded_calls is None:
            raise
        context.embedded_calls.note_error(error)
        raise _EmbeddedEvaluationFailureSignal() from error


def _warn_user_trace(function_name: str) -> None:
    """Warn when a user's staging transformation cannot carry its named executor."""
    context = current_call_context()
    choice = context.execution_choice
    if not choice.names_executor():
        return
    state = context.embedded_calls
    if state is not None:
        with state._lock:
            if state.warned_about_user_trace:
                return
            state.warned_about_user_trace = True
    warnings.warn(
        f"{function_name!r} is staged by a user's JAX transformation while executor "
        f"{choice.executor_name!r} is named. Its opaque code runs under the default context; "
        "run it through a ProbPipe call to carry the executor into the JAX program",
        stacklevel=3,
    )


def _captured_token_message(name: str) -> str:
    return (
        f"an inner jax.jit {name!r} captured an embedded call's token. "
        "Remove the inner jit or call the opaque code through a ProbPipe call"
    )


def _inspect_embedded_calls(
    jaxpr: Any,
    enclosing_name: str = "helper",
    constant_inputs: frozenset[Any] = frozenset(),
    *,
    validate_tokens: bool = True,
) -> bool:
    """Find this engine's callbacks recursively, refusing a literal token operand."""
    if hasattr(jaxpr, "jaxpr"):
        jaxpr = jaxpr.jaxpr
    constants = set(jaxpr.constvars) | set(constant_inputs)

    def is_constant(variable: Any) -> bool:
        return isinstance(variable, jax_core.Literal) or variable in constants

    found = False
    for equation in jaxpr.eqns:
        callback = equation.params.get("callback")
        host = getattr(callback, "callback_func", None)
        if isinstance(host, functools.partial) and host.func is _host_side:
            if validate_tokens and is_constant(equation.invars[0]):
                raise _EmbeddedCallTracingError(_captured_token_message(enclosing_name))
            found = True
        name = equation.params.get("name", enclosing_name)
        for nested in jax_core.jaxprs_in_params(equation.params):
            child = nested.jaxpr if hasattr(nested, "jaxpr") else nested
            operands = equation.invars
            if equation.primitive.name == "cond":
                operands = operands[1:]
            child_constants = frozenset(
                inner for inner, outer in zip(child.invars, operands) if is_constant(outer)
            )
            found = (
                _inspect_embedded_calls(
                    nested, name, child_constants, validate_tokens=validate_tokens
                )
                or found
            )
        if equation.invars and all(is_constant(variable) for variable in equation.invars):
            constants.update(equation.outvars)
    return found


_install_embedded_call_inspector(_inspect_embedded_calls)
