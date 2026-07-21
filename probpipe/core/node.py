from __future__ import annotations

import inspect
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, cast, get_args, overload

import jax
import jax.numpy as jnp

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from .config import WorkflowKind, prefect_config

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

from . import (
    _workflow_call,
    _workflow_distribution_broadcast,
    _workflow_distribution_normalization,
    _workflow_execution,
    _workflow_plan,
    _workflow_result,
    _workflow_sweep,
)
from ._function_contract import (
    _bind_function_inputs,
    _bind_planned_function_inputs,
    _CallableFunctionImplementation,
    _FunctionImplementation,
    _FunctionInvocationContext,
    _validate_function_output,
    _validate_function_templates,
    _wrap_declared_function_output,
)
from ._record_array import RecordArray
from .event_template import EventTemplate, _concretize_event_template
from .provenance import Provenance
from .tracked import Annotated, Tracked, auto_name

logger = logging.getLogger(__name__)


__all__ = [
    "AbstractModule",
    "Function",
    "InputFrozenError",
    "Module",
    "Node",
    "abstract_workflow_method",
    "function",
    "workflow_method",
]

_FunctionDispatch = Literal["auto", "jax", "sequential", "thread"]
_VALID_DISPATCH_STRATEGIES: tuple[str, ...] = get_args(_FunctionDispatch)


class InputFrozenError(Exception):
    pass


def workflow_method(func: Callable):
    """Mark a method as a workflow method for :class:`Module` subclasses.

    Methods decorated with ``@workflow_method`` are automatically
    converted to :class:`Function` instances when the
    ``Module`` is instantiated.
    """
    func._is_workflow = True
    return func


def abstract_workflow_method(func: Callable):
    """Mark a method as an abstract workflow interface.

    Combines ``@abstractmethod`` with ``@workflow_method`` so that
    :class:`AbstractModule` subclasses can declare workflow-shaped
    interfaces without providing implementations.
    """
    return abstractmethod(workflow_method(func))


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
        ``dispatch``, ``seed``, ``n_broadcast_samples``, ``include_inputs``,
        ``workflow_kind``, ``input_template``, and ``output_template``.

    Returns
    -------
    Function or Callable
        Wrapped Function for bare usage, or a decorator when called
        with parentheses.
    """

    def decorator(func: Callable[..., Any]) -> Function:
        return Function(func=func, **kwargs)

    if _func is not None:
        return decorator(_func)
    return decorator


class Node(ABC):  # noqa: B024
    """
    Base unit of the ProbPipe computational dependency graph.

    Keyword arguments are automatically split by type: values that are
    ``Node`` instances become *child nodes* (dependencies on other DAG
    units), and everything else becomes *inputs* (data, configuration,
    hyperparameters).  Both collections are frozen after construction.
    """

    def __init__(self, **kwargs: Any):
        child_nodes: dict[str, Node] = {}
        inputs: dict[str, Any] = {}

        for k, v in kwargs.items():
            if isinstance(v, Node):
                child_nodes[k] = v
            else:
                inputs[k] = v

        # Freeze internal state (read-only)
        self._child_nodes = MappingProxyType(child_nodes)
        self._inputs = MappingProxyType(inputs)

    @property
    def child_nodes(self) -> Mapping[str, Node]:
        return self._child_nodes

    @property
    def inputs(self) -> Mapping[str, Any]:
        return self._inputs


class Function(Node, Tracked, Annotated):
    """
    An immutable, tracked executable DAG node wrapping one implementation.

    Captures the wrapped callable's Python signature once, independently of
    any input or output event template. Infers dependency-vs-input from that
    signature and its type hints, and optionally resolves missing values from
    an attached Module.

    **Broadcasting**: When a ``Distribution`` is passed for an argument whose
    type hint is *not* a ``Distribution`` subclass, the workflow automatically
    samples from the distribution and calls the wrapped function once per
    sample, returning an ``EmpiricalDistribution`` over the outputs (or a
    plain list when results are not numeric).

    **Dispatch and orchestration** are orthogonal concerns:

    - *Dispatch* (``dispatch``) controls **how** samples are dispatched:
      ``jax.vmap``, local sequential function calls, or local threaded
      function calls.
    - *Orchestration* (``workflow_kind``) controls **whether** the dispatch
      is wrapped in a Prefect task or flow for compute-graph tracing.

    When both are active, the JAX-dispatched computation is executed inside
    a Prefect task/flow, giving the benefits of ``vmap`` performance with
    full Prefect lineage tracking.

    Parameters
    ----------
    func : Callable
        The function to wrap.
    workflow_kind : WorkflowKind
        Prefect orchestration mode.  ``DEFAULT`` inherits from
        ``prefect_config.workflow_kind`` (shipped default: ``OFF``;
        set via the ``PROBPIPE_WORKFLOW_KIND`` environment variable
        or explicit assignment).  ``TASK`` / ``FLOW`` explicitly
        request Prefect orchestration.  ``OFF`` disables
        orchestration.
    name : str or None
        Display name; defaults to ``func.__name__``.
    bind : dict or None
        Construction-time keyword bindings (defaults / config).
    module : Module or None
        Parent module for input / dependency resolution.
    n_broadcast_samples : int
        Default number of samples drawn when broadcasting.  Override it for
        one call with ``workflow.with_options(n_broadcast_samples=...)(...)``.
    dispatch : str
        Function-call dispatch strategy for broadcasting:

        - ``"auto"`` (default): probe with ``jax.make_jaxpr``; on success
          use ``"jax"``, on failure fall back to ``"sequential"``.
        - ``"jax"``: dispatch via ``jax.vmap``. Requires the wrapped
          function and broadcast path to be JAX-traceable.
        - ``"sequential"``: local row-wise/function-call dispatch without
          threads.
        - ``"thread"``: local row-wise/function-call dispatch through
          ``ThreadPoolExecutor``.
    max_workers : int or None
        Worker count for threaded sequential execution. ``None`` lets
        ``ThreadPoolExecutor`` choose automatically; a positive integer
        sets the worker count explicitly. Only applies when ``dispatch`` is
        ``"thread"`` and execution resolves to local thread dispatch. JAX
        ``vmap``, local sequential dispatch, Prefect, Ray, and Dask do not
        use this setting.
    seed : int
        Random seed for invocation-local JAX PRNG key management during
        broadcasting. Repeated calls with the same seed use the same key
        sequence without mutating the Function.
    include_inputs : bool
        Whether distribution broadcasting includes sampled inputs in the
        returned joint distribution by default.
    input_template : EventTemplate or None
        Optional authoritative input schema. Its top-level fields must match
        the fixed signature parameters by name. Symbolic dimensions are bound
        independently for each invocation.
    output_template : EventTemplate or None
        Optional authoritative output schema. Output symbols must be declared
        by ``input_template`` and are resolved in the same invocation-local
        dimension scope.
    **kwargs : Any
        Convenience construction-time bindings merged into ``bind``.

    Notes
    -----
    Keyword arguments passed to a workflow call belong to the wrapped user
    function whenever they can bind to that function. Use
    ``with_options(...)`` for call-time ProbPipe controls such as
    ``seed``, ``n_broadcast_samples``, and ``include_inputs``.

    Raises
    ------
    TypeError
        If ``workflow_kind`` is not a ``WorkflowKind`` enum member.
    """

    DEFAULT_N_BROADCAST_SAMPLES: int = 128
    _name: str

    def __init__(
        self,
        *,
        func: Callable,
        workflow_kind: WorkflowKind = WorkflowKind.DEFAULT,
        name: str | None = None,
        bind: dict[str, Any] | None = None,  # construction-time bindings (defaults/config)
        module: Any | None = None,  # typically a Module; kept as Any to avoid import cycles
        n_broadcast_samples: int | None = None,  # default number of samples for broadcasting
        dispatch: _FunctionDispatch = "auto",  # "auto" | "jax" | "sequential" | "thread"
        max_workers: int | None = None,  # ThreadPoolExecutor worker count
        seed: int = 0,  # JAX PRNG seed for broadcasting
        include_inputs: bool = False,  # True → return BroadcastDistribution (joint over inputs+outputs)
        input_template: EventTemplate | None = None,
        output_template: EventTemplate | None = None,
        **kwargs: Any,  # convenience bindings (merged into bind)
    ):
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func).__name__}")
        signature_info = _workflow_call.make_signature_info(func)
        implementation = _CallableFunctionImplementation(func)
        resolved_name, name_is_auto = auto_name(
            name, getattr(func, "__name__", self.__class__.__name__)
        )
        self._initialize(
            implementation=implementation,
            signature_info=signature_info,
            workflow_kind=workflow_kind,
            name=resolved_name,
            name_is_auto=name_is_auto,
            bind=bind,
            module=module,
            n_broadcast_samples=n_broadcast_samples,
            dispatch=dispatch,
            max_workers=max_workers,
            seed=seed,
            include_inputs=include_inputs,
            input_template=input_template,
            output_template=output_template,
            convenience_bindings=kwargs,
            metadata_source=func,
        )

    @staticmethod
    def _from_implementation(
        implementation: _FunctionImplementation,
        *,
        signature: inspect.Signature,
        name: str,
        input_template: EventTemplate | None = None,
        output_template: EventTemplate | None = None,
        workflow_kind: WorkflowKind = WorkflowKind.DEFAULT,
        bind: dict[str, Any] | None = None,
        module: Any | None = None,
        n_broadcast_samples: int | None = None,
        dispatch: _FunctionDispatch = "auto",
        max_workers: int | None = None,
        seed: int = 0,
        include_inputs: bool = False,
    ) -> Function:
        """Construct a normal Function around a private implementation seam."""
        if not isinstance(name, str) or not name:
            raise TypeError("Function._from_implementation() requires a non-empty name")
        if not callable(getattr(implementation, "invoke", None)):
            raise TypeError(
                "implementation must provide an invoke(bound_inputs, *, context) method"
            )
        instance = object.__new__(Function)
        instance._initialize(
            implementation=implementation,
            signature_info=_workflow_call.make_signature_info_from_signature(signature),
            workflow_kind=workflow_kind,
            name=name,
            name_is_auto=False,
            bind=bind,
            module=module,
            n_broadcast_samples=n_broadcast_samples,
            dispatch=dispatch,
            max_workers=max_workers,
            seed=seed,
            include_inputs=include_inputs,
            input_template=input_template,
            output_template=output_template,
            convenience_bindings={},
            metadata_source=None,
        )
        return instance

    def _initialize(
        self,
        *,
        implementation: _FunctionImplementation,
        signature_info: _workflow_call.WorkflowSignatureInfo,
        workflow_kind: WorkflowKind,
        name: str,
        name_is_auto: bool,
        bind: Mapping[str, Any] | None,
        module: Any | None,
        n_broadcast_samples: int | None,
        dispatch: _FunctionDispatch,
        max_workers: int | None,
        seed: int,
        include_inputs: bool,
        input_template: EventTemplate | None,
        output_template: EventTemplate | None,
        convenience_bindings: Mapping[str, Any],
        metadata_source: Callable[..., Any] | None,
    ) -> None:
        # Validate arguments before setting any instance state.
        if dispatch not in _VALID_DISPATCH_STRATEGIES:
            raise ValueError(
                f"dispatch must be one of {_VALID_DISPATCH_STRATEGIES}; got {dispatch!r}"
            )
        _workflow_execution._validate_max_workers(max_workers)
        if dispatch != "thread" and max_workers is not None:
            warnings.warn(
                "max_workers configures only dispatch='thread'; ignoring it "
                f"for dispatch={dispatch!r}.",
                stacklevel=2,
            )

        if not isinstance(workflow_kind, WorkflowKind):
            raise TypeError(
                f"workflow_kind must be a WorkflowKind enum member, "
                f"got {type(workflow_kind).__name__}"
            )

        construction_bindings = dict(bind or {})
        construction_bindings.update(convenience_bindings)
        _validate_function_templates(
            function_name=name,
            signature=signature_info.signature,
            input_template=input_template,
            output_template=output_template,
            construction_bindings=construction_bindings,
        )

        object.__setattr__(self, "_initializing", True)
        self._init_tracked(name, name_is_auto=name_is_auto)
        object.__setattr__(self, "_annotations", {})
        self._implementation = implementation
        self._signature_info = signature_info
        self._workflow_kind_raw = workflow_kind

        # Expose wrapped function's metadata for introspection (help(),
        # inspect.signature(), IDE tooltips, mkdocstrings).  We skip
        # __wrapped__ to prevent inspect.unwrap() from bypassing __call__.
        self.__doc__ = getattr(metadata_source, "__doc__", None)
        self.__name__ = self._name
        self.__qualname__ = getattr(metadata_source, "__qualname__", self._name)
        self.__signature__ = self._signature_info.signature
        self.__module__ = getattr(metadata_source, "__module__", None) or type(self).__module__
        self._module = module
        self._n_broadcast_samples = (
            n_broadcast_samples
            if n_broadcast_samples is not None
            else self.DEFAULT_N_BROADCAST_SAMPLES
        )
        self._dispatch = dispatch
        self._max_workers = max_workers
        self._seed = seed
        self._include_inputs = include_inputs
        self._input_template = input_template
        self._output_template = output_template

        # bind = "construction-time inputs" (defaults/config). kwargs are also treated as bind.
        self._bind = MappingProxyType(construction_bindings)

        super().__init__()
        object.__setattr__(self, "_initializing", False)

    @property
    def signature(self) -> inspect.Signature:
        """The independently captured Python call signature."""
        return self._signature_info.signature

    @property
    def input_template(self) -> EventTemplate | None:
        """The authoritative input schema declaration, when provided."""
        return self._input_template

    @property
    def output_template(self) -> EventTemplate | None:
        """The authoritative output schema declaration, when provided."""
        return self._output_template

    def apply(self, *args: Any, **call_inputs: Any) -> Any:
        """Execute one raw point under the Function's declared contracts.

        Arguments follow :attr:`signature`, including defaults, fixed
        construction bindings, and attached-Module resolution. Authoritative
        input and output templates are validated in one call-local symbolic
        dimension scope.

        Unlike :meth:`__call__`, this method performs no distribution lifting,
        batch sweep, orchestration, result wrapping, or call-provenance
        creation.

        Parameters
        ----------
        *args : Any
            Positional arguments for the captured Python signature.
        **call_inputs : Any
            Keyword arguments for the captured Python signature.

        Returns
        -------
        Any
            The implementation's unwrapped native result.

        Raises
        ------
        TypeError
            If arguments cannot bind to the captured signature.
        ValueError
            If an authoritative input or output template is violated.
        """
        call = _workflow_call.resolve_workflow_call(
            self._signature_info,
            args,
            call_inputs,
            bind=self._bind,
            module=self._module,
            dependency_type=Node,
            workflow_name=self._name,
            default_n_broadcast_samples=self._n_broadcast_samples,
            default_include_inputs=self._include_inputs,
        )
        _, bindings = _bind_function_inputs(
            function_name=self._name,
            input_template=self._input_template,
            values=call.values,
        )
        context = _FunctionInvocationContext(bindings)
        result = self._invoke_resolved(call.values, context=context)
        _validate_function_output(
            function_name=self._name,
            output_template=self._output_template,
            result=result,
            bindings=context.dimension_bindings,
        )
        return result

    def _invoke_resolved(
        self,
        values: Mapping[str, Any],
        *,
        context: _FunctionInvocationContext,
    ) -> Any:
        bound = _workflow_call.values_to_bound_arguments(self.signature, values)
        return self._implementation.invoke(bound, context=context)

    def with_name(self, name: str) -> Function:
        """Return a renamed shallow copy with synchronized callable metadata."""
        renamed = cast(Function, Tracked.with_name(self, name))
        object.__setattr__(renamed, "__name__", name)
        object.__setattr__(renamed, "__qualname__", name)
        return renamed

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initializing", False):
            object.__setattr__(self, name, value)
            return
        raise AttributeError("Function is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Function is immutable")

    @property
    def effective_workflow_kind(self) -> WorkflowKind:
        """Resolve the effective orchestration mode for this instance.

        Per-instance ``workflow_kind`` takes precedence over the global config.
        ``DEFAULT`` means "defer"; if both levels are ``DEFAULT``, orchestration is
        disabled. Prefect orchestration is opt-in.

        If ``TASK`` or ``FLOW`` is requested but Prefect is unavailable, the mode
        falls back to ``OFF``.
        """
        raw = self._workflow_kind_raw

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
        self,
        *,
        mode: _workflow_execution.WorkflowExecutionMode | None = None,
    ) -> _workflow_execution.WorkflowExecutionConfig:
        """Build resolved execution metadata for row-wise call dispatch."""
        if mode is None:
            match self.effective_workflow_kind:
                case WorkflowKind.TASK:
                    mode = "prefect_task"
                case WorkflowKind.FLOW:
                    mode = "prefect_flow"
                case _ if self._dispatch == "thread":
                    mode = "thread"
                case _:
                    mode = "sequential"

        is_prefect = mode in ("prefect_task", "prefect_flow")

        if is_prefect and (self._dispatch == "thread" or self._max_workers is not None):
            warnings.warn(
                "dispatch='thread' and max_workers configure only local "
                "ThreadPoolExecutor dispatch; they do not control Prefect "
                "scheduling.",
                stacklevel=2,
            )

        return _workflow_execution.WorkflowExecutionConfig(
            mode=mode,
            max_workers=self._max_workers if mode == "thread" else None,
            name=self._name,
            prefect_task_runner=(prefect_config.resolve_task_runner() if is_prefect else None),
        )

    def with_options(
        self,
        *,
        n_broadcast_samples: int | None = None,
        include_inputs: bool | None = None,
        seed: int | None = None,
    ) -> _FunctionCallWithOptions:
        """Return a callable view with temporary call-time workflow options.

        Keyword arguments passed to the returned callable are always treated
        as inputs to the wrapped function. Use this method, rather than
        call-time kwargs, to override ProbPipe controls for one call.

        Parameters
        ----------
        n_broadcast_samples : int or None
            Temporary sample-count override for distribution broadcasting.
        include_inputs : bool or None
            Temporary override for returning the joint input/output broadcast
            distribution.
        seed : int or None
            Temporary PRNG seed override for one workflow call.

        Returns
        -------
        Callable
            Callable view that applies these options to exactly one call.
        """
        return _FunctionCallWithOptions(
            self,
            _workflow_call.WorkflowCallOptions(
                n_broadcast_samples=n_broadcast_samples,
                include_inputs=include_inputs,
                seed=seed,
            ),
        )

    def __call__(self, *args: Any, **call_inputs: Any) -> Any:
        return self._call_with_options(
            args,
            call_inputs,
            _workflow_call.WorkflowCallOptions(),
        )

    def _call_with_options(
        self,
        args: tuple[Any, ...],
        call_inputs: dict[str, Any],
        options: _workflow_call.WorkflowCallOptions,
    ) -> Any:
        call = _workflow_call.resolve_workflow_call(
            self._signature_info,
            args,
            call_inputs,
            bind=self._bind,
            module=self._module,
            dependency_type=Node,
            workflow_name=self._name,
            default_n_broadcast_samples=self._n_broadcast_samples,
            default_include_inputs=self._include_inputs,
            options=options,
        )
        key = jax.random.PRNGKey(self._seed if call.overrides.seed is None else call.overrides.seed)

        def get_key():
            nonlocal key
            key, subkey = jax.random.split(key)
            return subkey

        values = _workflow_distribution_normalization.normalize_distribution_values(
            values=call.values,
            signature_info=self._signature_info,
        )
        broadcast_plan = _workflow_plan.build_broadcast_plan(
            values=values,
            signature_info=self._signature_info,
        )
        _, invocation_bindings = _bind_planned_function_inputs(
            function_name=self._name,
            input_template=self._input_template,
            values=values,
            lifted_names={
                ref.parameter_name
                for ref in (*broadcast_plan.dist_args, *broadcast_plan.array_args)
            },
        )
        concrete_output_template = (
            _concretize_event_template(
                self._output_template,
                invocation_bindings,
                context=f"Function {self._name!r} output_template",
            )
            if self._output_template is not None
            else None
        )
        provenance_parents: list[Tracked] = [self]
        seen_parent_ids = {id(self)}
        for ref in _workflow_call.iter_input_refs(self._signature_info, values):
            value = _workflow_call.input_ref_value(values, ref)
            if isinstance(value, Tracked) and id(value) not in seen_parent_ids:
                seen_parent_ids.add(id(value))
                provenance_parents.append(value)

        def invoke_point(**point_values: Any) -> Any:
            _, point_bindings = _bind_function_inputs(
                function_name=self._name,
                input_template=self._input_template,
                values=point_values,
                bindings=invocation_bindings,
            )
            context = _FunctionInvocationContext(point_bindings)
            result = self._invoke_resolved(point_values, context=context)
            point_output_template = _validate_function_output(
                function_name=self._name,
                output_template=self._output_template,
                result=result,
                bindings=context.dimension_bindings,
            )
            if point_output_template is not None:
                result = _wrap_declared_function_output(
                    result,
                    function_name=self._name,
                    output_template=point_output_template,
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
            if self._dispatch != "auto" or not jax_supported:
                return self._resolve_dispatch(
                    dispatch_values,
                    broadcast_args,
                    jax_supported=jax_supported,
                    func=invoke_point,
                )
            if resolved_dispatch is None:
                resolved_dispatch = self._resolve_dispatch(
                    dispatch_values,
                    broadcast_args,
                    jax_supported=True,
                    func=invoke_point,
                )
            return resolved_dispatch

        def require_jax_traceable(
            dispatch_values: dict[str, Any],
            broadcast_args: list[_workflow_call.WorkflowInputRef],
        ) -> None:
            self._require_jax_traceable(
                dispatch_values,
                broadcast_args,
                func=invoke_point,
            )

        def execute_distribution_broadcast(
            *,
            row_values: dict[str, Any],
            dist_args: Sequence[_workflow_call.WorkflowInputRef],
            n_broadcast_samples: int = call.overrides.n_broadcast_samples,
            include_inputs: bool = call.overrides.include_inputs,
        ):
            return _workflow_distribution_broadcast.execute_distribution_broadcast(
                func=invoke_point,
                values=row_values,
                broadcast_args=dist_args,
                n_broadcast_samples=n_broadcast_samples,
                include_inputs=include_inputs,
                get_key=get_key,
                make_execution_config=self._make_execution_config,
                requested_dispatch=self._dispatch,
                resolve_dispatch=resolve_dispatch,
                require_jax_traceable=require_jax_traceable,
                workflow_name=self._name,
                workflow_kind=self.effective_workflow_kind,
                output_template=concrete_output_template,
                provenance_parents=provenance_parents,
            )

        if broadcast_plan.regime == "distribution":
            return execute_distribution_broadcast(
                row_values=values,
                dist_args=broadcast_plan.dist_args,
            )
        if broadcast_plan.regime in ("sweep", "nested"):

            def distribution_broadcast(
                row_values: dict[str, Any],
                dist_args: list[_workflow_call.WorkflowInputRef],
                n_broadcast_samples: int,
                include_inputs: bool,
            ):
                return execute_distribution_broadcast(
                    row_values=row_values,
                    dist_args=dist_args,
                    n_broadcast_samples=n_broadcast_samples,
                    include_inputs=include_inputs,
                )

            return _workflow_sweep.execute_sweep(
                func=invoke_point,
                values=values,
                plan=broadcast_plan,
                make_execution_config=self._make_execution_config,
                requested_dispatch=self._dispatch,
                resolve_dispatch=resolve_dispatch,
                require_jax_traceable=require_jax_traceable,
                distribution_broadcast=distribution_broadcast,
                workflow_name=self._name,
                n_broadcast_samples=call.overrides.n_broadcast_samples,
                include_inputs=call.overrides.include_inputs,
                output_template=concrete_output_template,
                provenance_parents=provenance_parents,
            )

        # Non-broadcast call — one function invocation, then wrap.
        # Provenance parents are the inputs that carry their own
        # ``.provenance`` slot (Distribution / Record / RecordArray).
        # Known harmless duplication: the distribution-broadcast module builds
        # the same request shape. A later execution cleanup can centralize this
        # without reintroducing private facade wrappers.
        request = _workflow_execution.WorkflowExecutionRequest(
            func=invoke_point,
            call_value_list=[values],
            execution=self._make_execution_config(),
        )
        result = _workflow_execution.execute_many(request)[0]
        name = self._name
        provenance = Provenance.create(
            f"workflow.{name}",
            parents=provenance_parents,
            metadata={"func": name},
        )
        return _workflow_result._coerce_output(
            result,
            broadcast_mode=_workflow_result.BROADCAST_WRAP,
            provenance=provenance,
            field_name=self._name,
            output_template=concrete_output_template,
        )

    def _jax_traceability_error(
        self,
        values: dict[str, Any],
        broadcast_args: list[_workflow_call.WorkflowInputRef],
        *,
        func: Callable[..., Any],
    ) -> Exception | None:
        """Return the JAX trace-probe error for the current call, if any."""
        try:
            dummy_kw = dict(values)
            broadcast_refs = set(broadcast_args)
            for ref in _workflow_call.iter_input_refs(self._signature_info, values):
                v = _workflow_call.input_ref_value(values, ref)
                if ref in broadcast_refs:
                    # RecordArray input: construct a single Record from
                    # row 0 so the dummy call sees what an inner sweep
                    # iteration will actually receive.
                    if isinstance(v, RecordArray):
                        replacement = v[0]
                    else:
                        dist = v
                        # ``event_shape`` raises on multi-field NRDs
                        # (``NotImplementedError`` on the base or
                        # ``TypeError`` via ``_single_field_name``) —
                        # those distributions don't have a single
                        # array-shaped placeholder, so the probe can't
                        # produce a dummy. Falling out to the outer
                        # ``except Exception`` triggers row-wise dispatch,
                        # which is the right default for multi-field
                        # record-valued inputs.
                        try:
                            es = dist.event_shape
                        except (TypeError, NotImplementedError) as exc:
                            raise NotImplementedError(
                                f"Cannot probe JAX traceability for "
                                f"{type(dist).__name__} broadcast arg "
                                f"{ref.label!r}: no single ``event_shape`` "
                                f"(multi-field or abstract). "
                                f"Falling back to row-wise dispatch."
                            ) from exc
                        # Match the distribution's own dtype so the probe
                        # mirrors what the inner function actually sees.
                        dt = getattr(dist, "dtype", None) or jnp.zeros((), dtype=float).dtype
                        replacement = jnp.zeros(es, dtype=dt) if es else jnp.zeros((), dtype=dt)
                    dummy_kw = _workflow_call.replace_input_ref(dummy_kw, ref, replacement)
                else:
                    if isinstance(v, jnp.ndarray):
                        replacement = v
                    elif hasattr(v, "__array__"):
                        replacement = jnp.asarray(v)
                    else:
                        replacement = v
                    dummy_kw = _workflow_call.replace_input_ref(dummy_kw, ref, replacement)
            jax.make_jaxpr(lambda kw: func(**kw))(dummy_kw)
        except Exception as exc:
            return exc
        return None

    def _require_jax_traceable(
        self,
        values: dict[str, Any],
        broadcast_args: list[_workflow_call.WorkflowInputRef],
        *,
        func: Callable[..., Any],
    ) -> None:
        """Raise a clear error if explicit JAX dispatch cannot trace."""
        trace_error = self._jax_traceability_error(values, broadcast_args, func=func)
        if trace_error is None:
            return
        raise ValueError(
            "dispatch='jax' failed while tracing the wrapped function with JAX; "
            "ensure the function is JAX-traceable, or use dispatch='auto', "
            "'sequential', or 'thread'."
        ) from trace_error

    def _resolve_dispatch(
        self,
        values: dict[str, Any],
        broadcast_args: list[_workflow_call.WorkflowInputRef],
        *,
        jax_supported: bool = True,
        func: Callable[..., Any],
    ) -> str:
        """Resolve the dispatch strategy, caching JAX traceability detection.

        Returns ``"jax"``, ``"sequential"``, or ``"thread"``. This is
        independent of orchestration (``workflow_kind``), which wraps
        whichever strategy is chosen.
        """
        if self._dispatch != "auto":
            return self._dispatch

        if not jax_supported:
            return "sequential"

        if self._jax_traceability_error(values, broadcast_args, func=func) is None:
            return "jax"
        else:
            logger.info(
                "Function '%s' is not JAX-traceable; using sequential dispatch.",
                self._name,
            )
            return "sequential"


class _FunctionCallWithOptions:
    """Callable view that applies temporary Function options."""

    def __init__(
        self,
        function: Function,
        options: _workflow_call.WorkflowCallOptions,
    ):
        self._function = function
        self._options = options
        self.__doc__ = function.__doc__
        self.__name__ = function.__name__
        self.__qualname__ = function.__qualname__
        self.__signature__ = function.__signature__
        self.__module__ = function.__module__

    def __call__(self, *args: Any, **call_inputs: Any) -> Any:
        return self._function._call_with_options(
            args,
            call_inputs,
            self._options,
        )


class Module(Node):
    """
    Container for workflow nodes with shared inputs and child nodes.

    New user-facing API:
        MyModule(data=data_node, horizon=30, alpha=0.1)

    Internally:
        - kwargs whose values are Node instances become child_nodes
        - everything else becomes inputs

    Parameters
    ----------
    workflow_kind : WorkflowKind
        Prefect orchestration mode propagated to workflow methods built
        from this module.
    **kwargs : Any
        Shared child nodes and inputs available to workflow methods.

    Raises
    ------
    TypeError
        If ``workflow_kind`` is not a ``WorkflowKind`` enum member.
    """

    def __init__(
        self,
        *,
        workflow_kind: WorkflowKind = WorkflowKind.DEFAULT,
        **kwargs: Any,
    ):
        if not isinstance(workflow_kind, WorkflowKind):
            raise TypeError(
                f"workflow_kind must be a WorkflowKind enum member, "
                f"got {type(workflow_kind).__name__}"
            )
        self._workflow_kind = workflow_kind
        super().__init__(**kwargs)
        # validate abstract workflow implementations before wrapping
        self._validate_abstract_workflow_implementations()

        self._build_workflows()

    def _build_workflows(self):
        """
        Replace @workflow_method methods with Function instances.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not callable(attr) or not getattr(attr, "_is_workflow", False):
                continue

            func = attr

            # skip abstract workflows
            if getattr(func, "__isabstractmethod__", False):
                continue

            function_instance = Function(
                func=func,
                workflow_kind=self._workflow_kind,
                name=f"{self.__class__.__name__}.{func.__name__}",
                module=self,
            )

            setattr(self, attr_name, function_instance)

    def dag(self):
        """Return a Graphviz DAG visualization of this module."""
        if Digraph is None:
            raise ImportError(
                "graphviz is required for dag visualization. "
                "Install it with: pip install probpipe[viz]"
            )
        dot = Digraph(
            name=self.__class__.__name__,
            graph_attr={
                "rankdir": "LR",
                "fontsize": "12",
                "fontname": "Helvetica",
            },
            node_attr={
                "fontname": "Helvetica",
                "fontsize": "11",
            },
        )

        # -------------------------
        # Child nodes (outside)
        # -------------------------
        for name in self._child_nodes:
            dot.node(
                name,
                label=name,
                shape="ellipse",
                style="filled",
                fillcolor="#E8E8E8",
            )

        # -------------------------
        # Module cluster
        # -------------------------
        with dot.subgraph(name=f"cluster_{self.__class__.__name__}") as cluster:
            cluster.attr(
                label=self.__class__.__name__,
                style="rounded",
                color="#4F81BD",
                fontname="Helvetica-Bold",
                fontsize="12",
            )

            # Function nodes inside the module
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if not isinstance(attr, Function):
                    continue

                function_name = attr._name  # e.g. PM25ForecastingModule.fit
                function_label = function_name.split(".")[-1]

                cluster.node(
                    function_name,
                    label=function_label,
                    shape="box",
                    style="filled",
                    fillcolor="#C6DBEF",
                )

        # -------------------------
        # Dependency edges
        # -------------------------
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not isinstance(attr, Function):
                continue

            function_name = attr._name

            # Infer dependencies from workflow signature
            # (Functions don't store child_nodes; they resolve dependencies at runtime)
            for param_name in attr._signature_info.param_names:
                is_dependency = _workflow_call.is_dependency_param(
                    attr._signature_info,
                    param_name,
                    dependency_type=Node,
                )
                if is_dependency and param_name in self._child_nodes:
                    dot.edge(param_name, function_name)

        return dot

    def _validate_abstract_workflow_implementations(self) -> None:
        """
        Ensure that any abstract workflow interfaces in the MRO are implemented
        by a concrete workflow with a compatible signature.

        This prevents a common failure mode:
          - base class declares @abstract_workflow_method interface
          - subclass defines a method with same name but forgets @workflow_method
          - or implements it with a mismatched signature
        """
        cls = self.__class__

        # Walk MRO to find abstract workflow interfaces
        for base in cls.mro():
            for name, obj in base.__dict__.items():
                if not callable(obj):
                    continue
                if not getattr(obj, "_is_workflow", False):
                    continue
                if not getattr(obj, "__isabstractmethod__", False):
                    continue

                abstract_func = obj  # unbound function

                # Get the attribute as seen on the instance (could be method override)
                impl_attr = getattr(self, name, None)
                if impl_attr is None:
                    continue  # ABCMeta will usually catch this on AbstractModule anyway

                # If still abstract, ABCMeta will also catch it; but this provides better errors
                if getattr(impl_attr, "__isabstractmethod__", False):
                    raise TypeError(
                        f"{cls.__name__} does not implement abstract workflow '{name}'."
                    )

                # Must be marked as workflow (@workflow_method)
                if not getattr(impl_attr, "_is_workflow", False):
                    raise TypeError(
                        f"{cls.__name__}.{name} implements an abstract workflow interface "
                        f"but is not marked with @workflow_method."
                    )

                # Compare signatures (use unbound function signatures to include 'self')
                impl_func = impl_attr.__func__ if hasattr(impl_attr, "__func__") else impl_attr

                self._assert_workflow_signature_compatible(
                    abstract_func=abstract_func,
                    impl_func=impl_func,
                    name=name,
                )

    @staticmethod
    def _assert_workflow_signature_compatible(
        *,
        abstract_func: Callable,
        impl_func: Callable,
        name: str,
    ) -> None:
        abs_sig = inspect.signature(abstract_func)
        impl_sig = inspect.signature(impl_func)

        def drop_self(sig: inspect.Signature):
            params = list(sig.parameters.values())
            if params and params[0].name == "self":
                params = params[1:]
            return params

        abs_params = drop_self(abs_sig)
        impl_params = drop_self(impl_sig)

        # Build dict for implementation params by name (supports keyword usage)
        impl_by_name = {p.name: p for p in impl_params}

        # Require: every abstract param exists in impl with same kind
        for ap in abs_params:
            ip = impl_by_name.get(ap.name)
            if ip is None:
                raise TypeError(
                    f"Function '{name}' implementation is missing parameter '{ap.name}'.\n"
                    f"Expected (abstract): {abs_sig}\n"
                    f"Got (impl):          {impl_sig}"
                )
            if ip.kind != ap.kind:
                raise TypeError(
                    f"Function '{name}' parameter '{ap.name}' kind mismatch.\n"
                    f"Expected (abstract): {abs_sig}\n"
                    f"Got (impl):          {impl_sig}"
                )


class AbstractModule(Module, ABC):
    """
    Base class for modules that declare workflow interfaces via @abstract_workflow_method.

    ABCMeta will prevent instantiation until all abstract workflows are implemented
    by a concrete subclass.
    """

    pass
