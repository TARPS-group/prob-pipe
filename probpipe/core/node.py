from __future__ import annotations

import inspect
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from itertools import product as cartesian_product
from types import MappingProxyType
from typing import Any, ClassVar, Literal

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

from ..custom_types import Array, PRNGKey
from . import (
    _workflow_call,
    _workflow_distribution_normalization,
    _workflow_execution,
    _workflow_plan,
    _workflow_result,
    _workflow_sweep,
)
from ._record_array import RecordArray
from ._record_distribution import _RecordDistributionView
from .distribution import (
    BroadcastDistribution,
    Distribution,
    EmpiricalDistribution,
)
from .provenance import Provenance

logger = logging.getLogger(__name__)


__all__ = [
    "AbstractModule",
    "InputFrozenError",
    "Module",
    "Node",
    "WorkflowFunction",
    "abstract_workflow_method",
    "workflow_function",
    "workflow_method",
]

type _WorkflowFunctionDispatch = Literal[
    "auto", "jax", "sequential", "thread"
]


class InputFrozenError(Exception):
    pass


def workflow_method(func: Callable):
    """Mark a method as a workflow method for :class:`Module` subclasses.

    Methods decorated with ``@workflow_method`` are automatically
    converted to :class:`WorkflowFunction` instances when the
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


def workflow_function(_func=None, /, **kwargs):
    """Decorator to create a :class:`WorkflowFunction` from a plain function.

    Can be used with or without arguments::

        @workflow_function
        def my_func(x, y):
            return x + y

        @workflow_function(n_broadcast_samples=100, dispatch="sequential")
        def my_func(x, y):
            return x + y
    """
    def decorator(func):
        return WorkflowFunction(func=func, name=func.__name__, **kwargs)

    if _func is not None:
        # Bare @workflow_function (no parentheses)
        return decorator(_func)
    # @workflow_function(...) with arguments
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


def _index_sample(s: Any, i: int) -> Any:
    """Index row ``i`` of a per-arg sample batch.

    Parameters
    ----------
    s : Any
        Either a bare array (single-field broadcast draws), a
        :class:`Record` (multi-field auto-wrap or an
        explicit Record source), or a :class:`NumericRecord` (the
        unified ``EmpiricalDistribution`` — numeric arrays auto-wrap
        as a single-field Record — returns a ``NumericRecord`` with
        per-field axes stacked along the leading axis).
    i : int
        Row index along the leading axis.

    Returns
    -------
    Any
        Same shape the inner call would see from a single
        ``sample(dist)`` draw:

        * Single-field ``Record`` → the underlying field array's row
          ``i`` (unwrapped).
        * Multi-field ``Record`` → a per-row :class:`NumericRecord`.
        * Bare array → ``s[i]``.

    Notes
    -----
    Used by both ``_broadcast_enumerate_then_sample`` (for empirical-
    enumerated rows and for the mixed sampled-rows path) and
    ``_broadcast_sample`` (for plain MC sampling). Keeping the
    implementation in one place avoids drift between the three
    callsites.
    """
    # Local imports to avoid module-load circularity with
    # ``record`` / ``_numeric_record``.
    from ._numeric_record import NumericRecord
    from .record import Record

    if isinstance(s, Record):
        if len(s.fields) == 1:
            return s[s.fields[0]][i]
        return NumericRecord({f: s[f][i] for f in s.fields})
    return s[i]


class WorkflowFunction(Node):
    """
    A single executable DAG node wrapping exactly one function.

    Infers dependency-vs-input from the function signature and type hints.
    Optionally resolves missing values from an attached Module.

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
        orchestration.  Legacy strings (``"task"``, ``"flow"``) and
        ``None`` are auto-converted.
    name : str or None
        Display name; defaults to ``func.__name__``.
    bind : dict or None
        Construction-time keyword bindings (defaults / config).
    module : Module or None
        Parent module for input / dependency resolution.
    n_broadcast_samples : int
        Default number of samples drawn when broadcasting.  Can be overridden
        at call time by passing ``n_broadcast_samples=…`` (provided the
        wrapped function does not itself declare a parameter with that name).
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
        Random seed for JAX PRNG key management during broadcasting.
    """

    DEFAULT_N_BROADCAST_SAMPLES: int = 128
    _VALID_DISPATCH_STRATEGIES: ClassVar[tuple[str, ...]] = (
        "auto",
        "jax",
        "sequential",
        "thread",
    )

    def __init__(
        self,
        *,
        func: Callable,
        workflow_kind: WorkflowKind | str | None = WorkflowKind.DEFAULT,  # TODO: remove str | None in follow-up issue
        name: str | None = None,
        bind: dict[str, Any] | None = None,         # construction-time bindings (defaults/config)
        module: Any | None = None,                  # typically a Module; kept as Any to avoid import cycles
        n_broadcast_samples: int | None = None,      # default number of samples for broadcasting
        dispatch: _WorkflowFunctionDispatch = "auto", # "auto" | "jax" | "sequential" | "thread"
        max_workers: int | None = None,             # ThreadPoolExecutor worker count
        seed: int = 0,                              # JAX PRNG seed for broadcasting
        include_inputs: bool = False,                # True → return BroadcastDistribution (joint over inputs+outputs)
        **kwargs: Any,                              # convenience bindings (merged into bind)
    ):
        removed_keywords = {"parallel", "vectorize"} & set(kwargs)
        if removed_keywords:
            names = ", ".join(sorted(removed_keywords))
            raise TypeError(
                f"WorkflowFunction no longer accepts {names}; use dispatch= instead."
            )

        if dispatch not in self._VALID_DISPATCH_STRATEGIES:
            raise ValueError(
                f"dispatch must be one of {self._VALID_DISPATCH_STRATEGIES}; got {dispatch!r}"
            )
        _workflow_execution._validate_max_workers(max_workers)
        if dispatch != "thread" and max_workers is not None:
            warnings.warn(
                "max_workers configures only dispatch='thread'; ignoring it "
                f"for dispatch={dispatch!r}.",
                stacklevel=2,
            )

        self._func = func
        self._signature_info = _workflow_call.make_signature_info(func)
        # Convert legacy string / None values to WorkflowKind enum
        # TODO: remove this legacy conversion in follow-up issue
        if workflow_kind is None:
            self._workflow_kind_raw = WorkflowKind.OFF
        elif isinstance(workflow_kind, str) and not isinstance(workflow_kind, WorkflowKind):
            self._workflow_kind_raw = WorkflowKind(workflow_kind)
        else:
            self._workflow_kind_raw = workflow_kind
        self._name = name or getattr(func, "__name__", self.__class__.__name__)

        # Expose wrapped function's metadata for introspection (help(),
        # inspect.signature(), IDE tooltips, mkdocstrings).  We skip
        # __wrapped__ to prevent inspect.unwrap() from bypassing __call__.
        self.__doc__ = func.__doc__
        self.__name__ = self._name
        self.__qualname__ = getattr(func, "__qualname__", self._name)
        self.__signature__ = self._signature_info.signature
        self.__module__ = getattr(func, "__module__", None)
        self._module = module
        self._n_broadcast_samples = n_broadcast_samples if n_broadcast_samples is not None else self.DEFAULT_N_BROADCAST_SAMPLES
        self._dispatch = dispatch
        self._max_workers = max_workers
        self._key = jax.random.PRNGKey(seed)
        self._include_inputs = include_inputs
        self._resolved_dispatch: str | None = None  # cached auto-detection result

        # bind = "construction-time inputs" (defaults/config). kwargs are also treated as bind.
        b = dict(bind or {})
        b.update(kwargs)
        self._bind = b

        super().__init__()

        _workflow_call.validate_reserved_parameter_names(
            self._signature_info, workflow_name=self._name,
        )

    @property
    def effective_workflow_kind(self) -> WorkflowKind:
        """Resolve the orchestration mode for this instance.

        Resolution order:

        1. Per-instance override (anything other than ``DEFAULT``).
        2. Global ``prefect_config.workflow_kind``.
        3. If global is also ``DEFAULT``, fall back to ``OFF``. Prefect
           orchestration is opt-in: set the global or per-instance
           ``workflow_kind`` to ``TASK`` / ``FLOW``, or export
           ``PROBPIPE_WORKFLOW_KIND=task`` in the environment.

        If Prefect is not installed but ``TASK`` or ``FLOW`` is requested
        (either per-instance or globally), a warning is emitted and the
        mode falls back to ``OFF``.
        """
        raw = self._workflow_kind_raw

        # 1. Per-instance explicit (non-DEFAULT) override
        if raw is not WorkflowKind.DEFAULT:
            if raw in (WorkflowKind.TASK, WorkflowKind.FLOW) and task is None:
                warnings.warn(
                    f"workflow_kind={raw!r} requested but Prefect is not installed. "
                    f"Falling back to OFF. Install with: pip install probpipe[prefect]",
                    stacklevel=2,
                )
                return WorkflowKind.OFF
            return raw

        # 2. Resolve global config
        global_kind = prefect_config.workflow_kind
        if global_kind is not WorkflowKind.DEFAULT:
            kind = global_kind
        else:
            # 3. DEFAULT at global level = OFF (Prefect is opt-in)
            kind = WorkflowKind.OFF

        # Graceful fallback: global TASK/FLOW but Prefect missing
        if kind in (WorkflowKind.TASK, WorkflowKind.FLOW) and task is None:
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
            prefect_task_runner=(
                prefect_config.resolve_task_runner() if is_prefect else None
            ),
        )

    def __call__(self, *args, **call_inputs):
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
        if call.overrides.seed is not None:
            self._key = jax.random.PRNGKey(call.overrides.seed)

        values = _workflow_distribution_normalization.normalize_distribution_values(
            values=call.values, hints=self._signature_info.hints,
        )
        broadcast_plan = _workflow_plan.build_broadcast_plan(
            values=values, hints=self._signature_info.hints,
        )
        if broadcast_plan.regime != "none":
            return self._broadcast(
                values,
                broadcast_plan,
                call.overrides.n_broadcast_samples,
                call.overrides.include_inputs,
            )

        # Non-broadcast call — one function invocation, then wrap.
        # Provenance parents are the inputs that carry their own
        # ``.source`` slot (Distribution / Record / RecordArray).
        request = _workflow_execution.WorkflowExecutionRequest(
            func=self._func,
            call_value_list=[values],
            execution=self._make_execution_config(),
        )
        result = _workflow_execution.execute_many(request)[0]
        parents = tuple(
            v for v in values.values() if hasattr(v, "source")
        )
        provenance = Provenance(
            operation=f"workflow.{self._name or self._func.__name__}",
            parents=parents,
            metadata={"func": self._name or self._func.__name__},
        )
        return _workflow_result._coerce_output(
            result,
            broadcast_mode=_workflow_result.BROADCAST_WRAP,
            provenance=provenance,
            field_name=self._name,
        )

    def _get_key(self):
        """Split and advance the internal PRNG key."""
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def _jax_traceability_error(
        self,
        values: dict[str, Any],
        broadcast_args: list[str],
    ) -> Exception | None:
        """Return the JAX trace-probe error for the current call, if any."""
        try:
            dummy_kw = {}
            for name in self._signature_info.signature.parameters:
                if name == "self":
                    continue
                if name in broadcast_args:
                    v = values[name]
                    # RecordArray input: construct a single Record from
                    # row 0 so the dummy call sees what an inner sweep
                    # iteration will actually receive.
                    if isinstance(v, RecordArray):
                        dummy_kw[name] = v[0]
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
                                f"{name!r}: no single ``event_shape`` "
                                f"(multi-field or abstract). "
                                f"Falling back to row-wise dispatch."
                            ) from exc
                        # Match the distribution's own dtype so the probe
                        # mirrors what the inner function actually sees.
                        dt = getattr(dist, "dtype", None) or jnp.zeros((), dtype=float).dtype
                        dummy_kw[name] = jnp.zeros(es, dtype=dt) if es else jnp.zeros((), dtype=dt)
                elif name in values:
                    v = values[name]
                    if isinstance(v, jnp.ndarray):
                        dummy_kw[name] = v
                    elif hasattr(v, "__array__"):
                        dummy_kw[name] = jnp.asarray(v)
                    else:
                        dummy_kw[name] = v
            jax.make_jaxpr(lambda kw: self._func(**kw))(dummy_kw)
        except Exception as exc:
            return exc
        return None

    def _require_jax_traceable(
        self,
        values: dict[str, Any],
        broadcast_args: list[str],
    ) -> None:
        """Raise a clear error if explicit JAX dispatch cannot trace."""
        trace_error = self._jax_traceability_error(values, broadcast_args)
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
        broadcast_args: list[str],
        *,
        jax_supported: bool = True,
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

        if self._resolved_dispatch is not None:
            return self._resolved_dispatch

        if self._jax_traceability_error(values, broadcast_args) is None:
            self._resolved_dispatch = "jax"
        else:
            logger.info(
                "Function '%s' is not JAX-traceable; using sequential dispatch.",
                self._name,
            )
            self._resolved_dispatch = "sequential"

        return self._resolved_dispatch

    def _broadcast(
        self,
        values: dict[str, Any],
        broadcast_plan: _workflow_plan.BroadcastPlan,
        n_broadcast_samples: int,
        do_include_inputs: bool = False,
    ) -> Any:
        """Dispatcher: route to distribution broadcast or sweep execution."""
        dist_args = list(broadcast_plan.dist_args)
        ra_args = list(broadcast_plan.array_args)

        if not ra_args:
            return self._broadcast_distributions_only(
                values, dist_args, n_broadcast_samples, do_include_inputs,
            )

        return _workflow_sweep.execute_sweep(
            func=self._func,
            values=values,
            plan=broadcast_plan,
            make_execution_config=self._make_execution_config,
            requested_dispatch=self._dispatch,
            resolve_dispatch=self._resolve_dispatch,
            require_jax_traceable=self._require_jax_traceable,
            distribution_broadcast=self._broadcast_distributions_only,
            workflow_name=self._name,
            n_broadcast_samples=n_broadcast_samples,
            include_inputs=do_include_inputs,
        )

    def _broadcast_distributions_only(
        self,
        values: dict[str, Any],
        broadcast_args: list[str],
        n_broadcast_samples: int,
        do_include_inputs: bool = False,
    ) -> BroadcastDistribution | Distribution:
        """Distribution-only broadcast path (Monte Carlo marginalisation).

        Samples from each ``broadcast_args`` entry (all of which are
        ``Distribution`` instances after workflow normalization),
        calls the user's function once per sample, and wraps the n
        outputs as a single marginal distribution.

        Dispatch (``"jax"`` vs row-wise calls) and orchestration
        (``workflow_kind``) are resolved independently:

        - **dispatch="jax"**: samples are dispatched via ``jax.vmap``.
        - **dispatch="sequential"**: samples are dispatched via local
          row-wise calls.
        - **dispatch="thread"**: samples are dispatched via local threaded
          row-wise calls.
        - **workflow_kind="task"/"flow"**: whichever dispatch strategy is
          chosen gets wrapped in a Prefect task or flow for compute-graph
          tracing.
        """
        MIN_BROADCAST_SAMPLES = 5  # Recommended minimum samples

        # Validate n_broadcast_samples value and warn if too small
        if not isinstance(n_broadcast_samples, int) or n_broadcast_samples < MIN_BROADCAST_SAMPLES:
            warnings.warn(
                f"n_broadcast_samples={n_broadcast_samples} is too low; "
                f"results may be unreliable. "
                f"Recommended minimum is {MIN_BROADCAST_SAMPLES}.",
                stacklevel=2
            )

        # Collect candidate empirical dists (small enough individually),
        # sorted smallest first. Greedily include them while the product
        # stays within budget.
        candidates = []
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
                # Too large to enumerate — sample from it instead.
                sample_args[name] = dist

        dispatch = self._resolve_dispatch(
            values,
            broadcast_args,
            jax_supported=not empirical_args,
        )
        if self._dispatch == "jax" and empirical_args:
            raise ValueError(
                "dispatch='jax' does not support exact empirical enumeration; "
                "use dispatch='auto', 'sequential', or 'thread' for this path."
            )

        # Enumeration preserves exact empirical weights and must run in
        # all row-wise dispatch modes — otherwise the cartesian-product
        # semantics change with dispatch= (issue surfaced in the
        # ``EmpiricalDistribution`` broadcasting example).
        if empirical_args:
            result = self._broadcast_enumerate(
                values, empirical_args, sample_args, product_size, n_broadcast_samples,
            )
        elif dispatch == "jax":
            if self._dispatch == "jax":
                self._require_jax_traceable(values, broadcast_args)
            result = self._broadcast_jax(values, broadcast_args, n_broadcast_samples)
        else:
            result = self._broadcast_sample(values, broadcast_args, n_broadcast_samples)

        # Attach provenance
        parents = tuple(
            values[name] for name in broadcast_args
            if isinstance(values[name], Distribution)
        )
        provenance = Provenance(
            "broadcast",
            parents=parents,
            metadata={
                "dispatch": dispatch,
                "orchestrate": self.effective_workflow_kind.value,
                "n_samples": n_broadcast_samples,
                "func": self._name or self._func.__name__,
                "broadcast_args": broadcast_args,
            },
        )

        if isinstance(result, Distribution):
            result.with_source(provenance)

        if do_include_inputs:
            return result

        # Default: return the output marginal only.
        # Provenance is propagated automatically by marginalize().
        if isinstance(result, BroadcastDistribution):
            return result.marginalize()

        return result

    def _sample_broadcast_args(
        self,
        values: dict[str, Any],
        broadcast_args: list[str],
        n: int,
        key: PRNGKey,
    ) -> dict[str, Array]:
        """Sample all broadcast arguments, handling view reconnection.

        Sibling views from the same parent distribution share one
        parent draw, preserving cross-field correlation. Plain
        (non-view) distributions are sampled independently per kwarg,
        even if the same object is passed under multiple names.
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
        self,
        values: dict[str, Any],
        broadcast_args: list[str],
        n_broadcast_samples: int,
    ) -> BroadcastDistribution:
        """
        Vectorised broadcasting via ``jax.vmap``.

        Samples all broadcast distributions at once (with joint reconnection),
        then calls the wrapped function over the batch dimension using
        ``jax.vmap``.  Requires the wrapped function to be JAX-traceable.

        If ``workflow_kind`` is set, the entire vmap computation is wrapped
        in a Prefect task or flow for orchestration tracing.
        """
        key = self._get_key()
        sampled = self._sample_broadcast_args(values, broadcast_args, n_broadcast_samples, key)

        static = {k: v for k, v in values.items() if k not in broadcast_args}

        func = self._func

        def single_call(broadcast_slice):
            kw = dict(static)
            kw.update(broadcast_slice)
            return func(**kw)

        # vmap over the dict of batched arrays (axis 0 for each)
        batch = {name: sampled[name] for name in broadcast_args}

        def run_vmap():
            return jax.vmap(single_call)(batch)

        # Wrap in Prefect task/flow if orchestration is requested
        kind = self.effective_workflow_kind
        if kind in (WorkflowKind.TASK, WorkflowKind.FLOW):
            if kind == WorkflowKind.TASK:
                run_vmap = task(name=f"{self._name}_vmap")(run_vmap)
            else:
                runner = prefect_config.resolve_task_runner()
                run_vmap = flow(
                    name=f"{self._name}_vmap",
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
        self,
        values: dict[str, Any],
        empirical_args: dict[str, EmpiricalDistribution],
        sample_args: dict[str, Distribution],
        product_size: int,
        n_broadcast_samples: int,
    ) -> BroadcastDistribution:
        """
        Enumerate the cartesian product of EmpiricalDistribution samples,
        propagating their weights. When non-empirical distributions are also
        present, draws as many samples as the budget allows per combination
        (n_broadcast_samples // product_size), each receiving equal weight scaled by
        the empirical product weight.
        """
        key = self._get_key()
        emp_names = list(empirical_args.keys())
        emp_dists = [empirical_args[name] for name in emp_names]

        # Use as many non-empirical samples per combo as budget allows
        reps_per_combo = max(1, n_broadcast_samples // product_size) if sample_args else 1
        total = product_size * reps_per_combo

        # Pre-sample non-empirical distributions (with DistributionView reconnection)
        sample_arg_names = list(sample_args.keys())
        if sample_arg_names:
            sampled = self._sample_broadcast_args(values, sample_arg_names, total, key)
        else:
            sampled = {}

        call_value_list = []
        weights = []
        sample_idx = 0

        all_broadcast_args = emp_names + sample_arg_names

        for combo in cartesian_product(*(range(d.num_atoms) for d in emp_dists)):
            # Compute empirical product weight
            emp_weight = 1.0
            for _name, dist, i in zip(emp_names, emp_dists, combo):
                emp_weight *= float(dist.weights[i])

            for _ in range(reps_per_combo):
                call_values = dict(values)

                # Set empirical samples. ``dist.samples`` is a Record
                # of per-field stacked arrays; pull row ``i`` per field
                # so the inner call sees the same shape it would from a
                # single ``sample(dist)`` draw.
                for name, dist, i in zip(emp_names, emp_dists, combo):
                    call_values[name] = _index_sample(dist.samples, i)

                # Set sampled values for non-empirical distributions.
                # ``sampled[name]`` may be a Record (auto-wrapped); use
                # the same per-row indexing helper.
                for name in sample_args:
                    call_values[name] = _index_sample(sampled[name], sample_idx)

                # Weight: empirical product weight divided evenly across reps
                weights.append(emp_weight / reps_per_combo)
                call_value_list.append(call_values)
                sample_idx += 1

        request = _workflow_execution.WorkflowExecutionRequest(
            func=self._func,
            call_value_list=call_value_list,
            execution=self._make_execution_config(),
        )
        results = _workflow_execution.execute_many(request)

        # Assemble input samples aligned with results
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
        self,
        values: dict[str, Any],
        broadcast_args: list[str],
        n_broadcast_samples: int,
    ) -> BroadcastDistribution:
        """
        Sample n_broadcast_samples from each Distribution argument and call the function
        once per sample (uniform weights).  Handles DistributionView reconnection.
        """
        key = self._get_key()
        samples_per_arg = self._sample_broadcast_args(values, broadcast_args, n_broadcast_samples, key)

        call_value_list = []
        for i in range(n_broadcast_samples):
            call_values = dict(values)
            for name in broadcast_args:
                call_values[name] = _index_sample(samples_per_arg[name], i)
            call_value_list.append(call_values)

        request = _workflow_execution.WorkflowExecutionRequest(
            func=self._func,
            call_value_list=call_value_list,
            execution=self._make_execution_config(),
        )
        results = _workflow_execution.execute_many(request)

        return BroadcastDistribution(
            input_samples=samples_per_arg,
            output_samples=results,
            weights=None,
            broadcast_args=broadcast_args,
        )


class Module(Node):
    """
    Container for workflow nodes with shared inputs and child nodes.

    New user-facing API:
        MyModule(data=data_node, horizon=30, alpha=0.1)

    Internally:
        - kwargs whose values are Node instances become child_nodes
        - everything else becomes inputs
    """

    def __init__(self, *, workflow_kind: WorkflowKind | str | None = WorkflowKind.DEFAULT, **kwargs: Any):
        # Convert legacy string / None values to WorkflowKind enum
        if workflow_kind is None:
            self._workflow_kind = WorkflowKind.OFF
        elif isinstance(workflow_kind, str) and not isinstance(workflow_kind, WorkflowKind):
            self._workflow_kind = WorkflowKind(workflow_kind)
        else:
            self._workflow_kind = workflow_kind
        super().__init__(**kwargs)
        # validate abstract workflow implementations before wrapping
        self._validate_abstract_workflow_implementations()

        self._build_workflows()

    def _build_workflows(self):
        """
        Replace @workflow_method methods with WorkflowFunction instances.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not callable(attr) or not getattr(attr, "_is_workflow", False):
                continue

            func = attr

            # skip abstract workflows
            if getattr(func, "__isabstractmethod__", False):
                continue

            wf_instance = WorkflowFunction(
                func=func,
                workflow_kind=self._workflow_kind,
                name=f"{self.__class__.__name__}.{func.__name__}",
                module=self,
            )

            setattr(self, attr_name, wf_instance)

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

            # WorkflowFunction nodes inside the module
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if not isinstance(attr, WorkflowFunction):
                    continue

                wf_name = attr._name  # e.g. PM25ForecastingModule.fit
                wf_label = wf_name.split(".")[-1]

                cluster.node(
                    wf_name,
                    label=wf_label,
                    shape="box",
                    style="filled",
                    fillcolor="#C6DBEF",
                )

        # -------------------------
        # Dependency edges
        # -------------------------
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not isinstance(attr, WorkflowFunction):
                continue

            wf_name = attr._name

            # Infer dependencies from workflow signature
            # (WorkflowFunctions don't store child_nodes; they resolve dependencies at runtime)
            for param_name in attr._signature_info.param_names:
                is_dependency = _workflow_call.is_dependency_param(
                    attr._signature_info,
                    param_name,
                    dependency_type=Node,
                )
                if is_dependency and param_name in self._child_nodes:
                    dot.edge(param_name, wf_name)

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
                    f"WorkflowFunction '{name}' implementation is missing parameter '{ap.name}'.\n"
                    f"Expected (abstract): {abs_sig}\n"
                    f"Got (impl):          {impl_sig}"
                )
            if ip.kind != ap.kind:
                raise TypeError(
                    f"WorkflowFunction '{name}' parameter '{ap.name}' kind mismatch.\n"
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
