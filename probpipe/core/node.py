from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, get_type_hints
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import product as cartesian_product
from types import MappingProxyType
import warnings

import jax
import jax.numpy as jnp
import numpy as np

try:
    from prefect import task, flow
except ImportError:
    task = flow = None

from .config import WorkflowKind, prefect_config

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

from ..custom_types import PRNGKey, Array
from .distribution import (
    NumericRecordDistribution,
    BroadcastDistribution,
    Distribution,
    EmpiricalDistribution,
    _make_marginal,
)
from ._broadcast_distributions import _make_stack, AUTO_WRAP_FIELD
from ._distribution_array import _make_distribution_array
from ._numeric_record import NumericRecord
from ._record_array import RecordArray
from .provenance import Provenance
from .record import Record
from .protocols import (
    SupportsConditioning,
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
)

# Protocol types that indicate a parameter expects a distribution object.
# Used by _find_broadcast_args to avoid broadcasting over such parameters.
_DISTRIBUTION_PROTOCOLS: tuple[type, ...] = (
    SupportsExpectation,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsLogProb,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsConditioning,
)
from ..converters import converter_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output-type coercion for WorkflowFunction returns (issue #130)
# ---------------------------------------------------------------------------
#
# Every ``WorkflowFunction`` output is one of three types: ``Record``,
# ``RecordArray``, or ``Distribution`` (the three types that carry a
# ``.source`` provenance slot). ``_coerce_output`` is the single entry
# point that enforces the contract and attaches a ``Provenance`` node.
#
# Bare scalars, ``jnp.ndarray``, and Python lists are wrapped as
# ``NumericRecord(result=...)`` (or, for broadcast outputs, into the
# appropriate aggregate). Opaque Python values (strings, dicts, ...)
# fall back to ``Record(result=...)``.
#
# The ergonomic shim on single-field ``NumericRecord`` (see
# ``_numeric_record.py``) keeps ``float(mean(d))`` /
# ``np.asarray(mean(d))`` working transparently.
# ---------------------------------------------------------------------------


def _wrap_as_record(value: Any) -> Any:
    """Coerce a structure-valued return toward an output-contract type.

    The wrap is deliberately narrow — **only** ``dict`` and non-empty
    ``list``/``tuple`` returns are promoted. Everything else passes
    through so that:

    - Bare numeric scalars and ``ndarray`` values keep working with
      arithmetic (``sample(dist) + shift``) and attribute access
      (``mean(dist).shape``).
    - Callables returned by ``sample`` on a ``RandomFunction`` stay
      callable (``f = sample(grf); f(X)``).
    - Any custom domain object (xarray, pandas, ...) is passed through
      unchanged — the workflow author is responsible for any further
      wrapping.

    For the pass-through types, provenance remains reachable via
    ``provenance_ancestors(input_distribution)``.

    Promoted types:

    - ``dict`` with at least one key → ``Record(**dict)``, so the
      caller's keys stay addressable without a ``"result"`` detour.
      This is the common case for pipeline helpers that produce
      structured summaries (test statistics, diagnostics, ...).
    - Non-empty ``list`` / ``tuple`` → ``_make_stack`` dispatch:
      a list of Distributions becomes a ``DistributionArray``, a list
      of Records a ``RecordArray``, a list of arrays a stacked
      ``NumericRecordArray``. This handles the ``iterate`` /
      ``condition_on_all`` family of ops that naturally produce
      sequences of distributions.
    """
    if isinstance(value, dict) and value:
        return Record(dict(value))
    if isinstance(value, (list, tuple)) and value:
        try:
            return _make_stack(list(value), n=len(value))
        except (TypeError, ValueError):
            pass
    return value


def _coerce_output(
    value: Any,
    *,
    broadcast_mode: str,
    provenance: Provenance | None,
) -> Any:
    """Enforce the Record | RecordArray | Distribution output contract.

    Parameters
    ----------
    value
        The raw output produced by the function body or a broadcast
        aggregator. For ``broadcast_mode != "wrap"`` this is always
        already one of the three contract types.
    broadcast_mode : {"wrap", "marginalise", "stack", "nested"}
        How the value was produced:

        * ``"wrap"`` — non-broadcast call; ``value`` is whatever the
          user's function returned. Wrap scalars/arrays as
          ``NumericRecord(result=...)`` / opaque Python values as
          ``Record(result=...)``. Existing Record / RecordArray /
          Distribution values pass through.
        * ``"marginalise"`` — Distribution-only broadcast; ``value`` is
          a marginal distribution from ``_make_marginal``.
        * ``"stack"`` — RecordArray-only broadcast; ``value`` is a
          stacked aggregate from ``_make_stack``
          (``NumericRecordArray`` / ``RecordArray`` / ``DistributionArray``).
        * ``"nested"`` — RecordArray + Distribution broadcast; ``value``
          is a ``DistributionArray`` of per-row marginals.
    provenance : Provenance or None
        Provenance node to attach. ``None`` skips the attachment step.

    Returns
    -------
    Record | RecordArray | Distribution
        The value, possibly wrapped, with ``.source`` attached when it
        was empty. An already-sourced value keeps its existing source
        (inner marginals produced by the broadcast layer carry their
        own provenance; ``_coerce_output`` doesn't overwrite).
    """
    if broadcast_mode == "wrap":
        value = _wrap_as_record(value)
    if provenance is not None and hasattr(value, "with_source"):
        try:
            value.with_source(provenance)
        except RuntimeError:
            # Value already has a source (e.g., an inner marginal that
            # the broadcasting layer built with its own provenance).
            # Leave the existing source in place.
            pass
    return value

__all__ = [
    "InputFrozenError",
    "workflow_method",
    "abstract_workflow_method",
    "workflow_function",
    "Node",
    "WorkflowFunction",
    "Module",
    "AbstractModule",
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

        @workflow_function(n_broadcast_samples=100, vectorize="loop")
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


class Node(ABC):
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
    def child_nodes(self) -> Mapping[str, "Node"]:
        return self._child_nodes

    @property
    def inputs(self) -> Mapping[str, Any]:
        return self._inputs
    

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

    **Vectorization and orchestration** are orthogonal concerns:

    - *Vectorization* (``vectorize``) controls **how** samples are dispatched:
      ``jax.vmap`` for JAX-traceable functions, or a Python loop otherwise.
    - *Orchestration* (``workflow_kind``) controls **whether** the dispatch
      is wrapped in a Prefect task or flow for compute-graph tracing.

    When both are active, the JAX-vectorized computation is executed inside
    a Prefect task/flow, giving the benefits of ``vmap`` performance with
    full Prefect lineage tracking.

    Parameters
    ----------
    func : Callable
        The function to wrap.
    workflow_kind : WorkflowKind
        Prefect orchestration mode.  ``DEFAULT`` inherits from
        ``prefect_config`` (auto-uses Prefect tasks when available).
        ``TASK`` / ``FLOW`` explicitly request Prefect orchestration.
        ``OFF`` disables orchestration.  Legacy strings (``"task"``,
        ``"flow"``) and ``None`` are auto-converted.
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
    vectorize : str
        Vectorization strategy for broadcasting:

        - ``"auto"`` (default): probe with ``jax.make_jaxpr``; on success
          use ``"jax"``, on failure fall back to ``"loop"``.
        - ``"jax"``: vectorise via ``jax.vmap``.  Requires the wrapped
          function to be JAX-traceable.
        - ``"loop"``: Python loop (optionally threaded via *parallel*).
    parallel : bool or int
        Controls parallel execution during broadcasting (``"loop"``
        vectorization only).  ``False`` → sequential, ``True`` →
        ``ThreadPoolExecutor`` with default workers, ``int`` → explicit
        ``max_workers``.
    seed : int
        Random seed for JAX PRNG key management during broadcasting.
    """

    DEFAULT_N_BROADCAST_SAMPLES: int = 128

    def __init__(
        self,
        *,
        func: Callable,
        workflow_kind: WorkflowKind | str | None = WorkflowKind.DEFAULT,
        name: str | None = None,
        bind: dict[str, Any] | None = None,         # construction-time bindings (defaults/config)
        module: Any | None = None,                  # typically a Module; kept as Any to avoid import cycles
        n_broadcast_samples: int | None = None,      # default number of samples for broadcasting
        vectorize: str = "auto",                     # "auto" | "jax" | "loop"
        parallel: bool | int = False,               # True/int for ThreadPoolExecutor, or Prefect .map()
        seed: int = 0,                              # JAX PRNG seed for broadcasting
        include_inputs: bool = False,                # True → return BroadcastDistribution (joint over inputs+outputs)
        **kwargs: Any,                              # convenience bindings (merged into bind)
    ):
        self._func = func
        self._sig = inspect.signature(func)
        self._hints = get_type_hints(func)
        # Convert legacy string / None values to WorkflowKind enum
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
        self.__signature__ = self._sig
        self.__module__ = getattr(func, "__module__", None)
        self._module = module
        self._n_broadcast_samples = n_broadcast_samples if n_broadcast_samples is not None else self.DEFAULT_N_BROADCAST_SAMPLES
        self._vectorize = vectorize
        self._parallel = parallel
        self._key = jax.random.PRNGKey(seed)
        self._include_inputs = include_inputs
        self._resolved_vectorize: str | None = None  # cached auto-detection result

        # bind = "construction-time inputs" (defaults/config). kwargs are also treated as bind.
        b = dict(bind or {})
        b.update(kwargs)
        self._bind = b

        super().__init__()

        # Precompute parameter metadata once
        self._param_names = [p for p in self._sig.parameters if p != "self"]
        self._has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in self._sig.parameters.values()
        )

        # Reserved names that would collide with WorkflowFunction call-time overrides
        _RESERVED = {"n_broadcast_samples", "seed", "include_inputs"}
        collision = _RESERVED & set(self._param_names)
        if collision:
            raise ValueError(
                f"Function '{self._name}' has parameter(s) {collision} which are "
                f"reserved by WorkflowFunction for call-time overrides. Rename them in "
                f"your function signature."
            )

    @property
    def effective_workflow_kind(self) -> WorkflowKind:
        """Resolve the orchestration mode for this instance.

        Resolution order:

        1. Per-instance override (anything other than ``DEFAULT``).
        2. Global ``prefect_config.workflow_kind``.
        3. If global is also ``DEFAULT``, auto-detect: ``TASK`` when
           Prefect is installed, ``OFF`` otherwise.

        ``TASK`` / ``FLOW`` with Prefect missing raises ``ImportError``
        for explicit per-instance settings, but falls back to ``OFF``
        for values inherited from global config (graceful degradation).
        """
        raw = self._workflow_kind_raw

        # 1. Per-instance explicit (non-DEFAULT) override
        if raw is not WorkflowKind.DEFAULT:
            if raw in (WorkflowKind.TASK, WorkflowKind.FLOW) and task is None:
                raise ImportError(
                    f"Prefect is required for workflow_kind={raw!r}: "
                    f"pip install probpipe[prefect]"
                )
            return raw

        # 2. Resolve global config
        global_kind = prefect_config.workflow_kind
        if global_kind is not WorkflowKind.DEFAULT:
            kind = global_kind
        else:
            # 3. DEFAULT at global level = auto-detect
            kind = WorkflowKind.TASK if task is not None else WorkflowKind.OFF

        # Graceful fallback: global/auto-detected TASK/FLOW but Prefect missing
        if kind in (WorkflowKind.TASK, WorkflowKind.FLOW) and task is None:
            return WorkflowKind.OFF

        return kind

    def _is_dependency_param(self, name: str) -> bool:
        """
        Decide whether a parameter is a dependency (Node) vs normal input.

        Rule:
          - If annotation is a Node subclass => dependency
          - Else => normal input

        This matches your current architecture (deps are Nodes).
        """
        ann = self._hints.get(name)
        try:
            return isinstance(ann, type) and issubclass(ann, Node)
        except TypeError:
            # Generic aliases (e.g. NDArray on Python 3.10) can pass
            # isinstance(ann, type) but fail in issubclass().
            return False

    def __call__(self, *args, **call_inputs):
        # Bind positional args to parameter names using the wrapped function's
        # signature so callers can use positional arguments naturally.
        if args:
            bound = self._sig.bind_partial(*args)
            for name, value in bound.arguments.items():
                if name in call_inputs:
                    raise TypeError(
                        f"{self._name}() got multiple values for argument '{name}'"
                    )
                call_inputs[name] = value

        # If the wrapped function has a **kwargs parameter, sig.bind nests
        # the extra keyword args under the parameter name.  Unpack them so
        # they reach the function as top-level keyword arguments.
        if self._has_var_keyword:
            for p in self._sig.parameters.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD and p.name in call_inputs:
                    extra = call_inputs.pop(p.name)
                    call_inputs.update(extra)

        # Extract reserved call-time overrides (collision already prevented in __init__)
        n_broadcast_samples = call_inputs.pop("n_broadcast_samples", self._n_broadcast_samples)
        do_include_inputs = call_inputs.pop("include_inputs", self._include_inputs)

        if "seed" in call_inputs:
            self._key = jax.random.PRNGKey(call_inputs.pop("seed"))

        values = self._resolve_inputs(call_inputs)
        values = self._convert_distributions(values)

        dist_args, ra_args = self._find_broadcast_args(values)
        if dist_args or ra_args:
            return self._broadcast(
                values, dist_args, ra_args, n_broadcast_samples, do_include_inputs,
            )

        # Non-broadcast call — run the function body once and wrap the
        # return so every WorkflowFunction output satisfies the
        # Record | RecordArray | Distribution contract (issue #130
        # PR 1.5). Provenance parents are the inputs that carry their
        # own ``.source`` slot (Distribution / Record / RecordArray
        # instances) — other args are data, not lineage.
        result = self._execute_many([values])[0]
        parents = tuple(
            v for v in values.values() if hasattr(v, "source")
        )
        provenance = Provenance(
            operation=f"workflow.{self._name or self._func.__name__}",
            parents=parents,
            metadata={"func": self._name or self._func.__name__},
        )
        return _coerce_output(
            result, broadcast_mode="wrap", provenance=provenance,
        )

    def _resolve_inputs(self, call_inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Build kwargs for calling the underlying function.

        Precedence (highest -> lowest):
          1) call_inputs
          2) self._bind
          3) module.child_nodes/module.inputs (if module attached)
          4) function default values
        """
        values: dict[str, Any] = {}

        # convenience access
        mod = self._module
        mod_child_nodes = getattr(mod, "child_nodes", {}) if mod is not None else {}
        mod_inputs = getattr(mod, "inputs", {}) if mod is not None else {}

        for name, param in self._sig.parameters.items():
            if name == "self":
                continue

            is_dep = self._is_dependency_param(name)

            # 1) call-time inputs
            if name in call_inputs:
                if mod is not None and is_dep and name in mod_child_nodes:
                    # Avoid accidental overriding of module-wired deps at call time
                    raise TypeError(
                        f"Dependency '{name}' for workflow '{self._name}' is provided by the module "
                        f"and cannot be overridden at call time."
                    )
                values[name] = call_inputs[name]

            # 2) construction-time bind
            elif name in self._bind:
                values[name] = self._bind[name]

            # 3) resolve from module if available
            elif mod is not None:
                if is_dep and name in mod_child_nodes:
                    values[name] = mod_child_nodes[name]
                elif (not is_dep) and name in mod_inputs:
                    values[name] = mod_inputs[name]
                else:
                    # fall through to default/missing
                    pass

            # 4) function defaults
            if name not in values:
                if param.default is not param.empty:
                    values[name] = param.default

        # Pass through extra kwargs when the function accepts **kwargs
        has_var_keyword = any(
            p.kind == p.VAR_KEYWORD for p in self._sig.parameters.values()
        )
        if has_var_keyword:
            known_params = set(self._sig.parameters.keys())
            for k, v in call_inputs.items():
                if k not in known_params:
                    values[k] = v

        # validate required params exist (after resolution)
        for name, param in self._sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param.default is param.empty and name not in values:
                raise TypeError(f"Missing required input '{name}' for workflow '{self._name}'")

        # validate dependency types
        for name in self._param_names:
            if self._is_dependency_param(name):
                v = values.get(name)
                if not isinstance(v, Node):
                    ann = self._hints.get(name)
                    raise TypeError(
                        f"WorkflowFunction '{self._name}' expects dependency '{name}: {ann}' to be a Node, "
                        f"but got {type(v)}."
                    )

        return values

    def _convert_distributions(self, values: dict[str, Any]) -> dict[str, Any]:
        """Convert distributions based on type hints.

        Handles two cases:

        1. **Concrete type hints** – if the hint is a ``Distribution``
           subclass and the value is a recognised distribution that is
           not already an instance, convert via the registry.
        2. **Protocol type hints** – if the hint is a
           ``@runtime_checkable`` protocol (e.g., ``SupportsLogProb``)
           and the value is a ``Distribution`` that does not satisfy the
           protocol, convert via protocol-based resolution.
        """
        out = dict(values)

        for name, value in values.items():
            expected = self._hints.get(name)
            if expected is None:
                continue

            try:
                is_dist_subclass = isinstance(expected, type) and issubclass(
                    expected, Distribution
                )
            except TypeError:
                is_dist_subclass = False

            if (
                is_dist_subclass
                and converter_registry.is_distribution_type(value)
                and not isinstance(value, expected)
            ):
                out[name] = converter_registry.convert(value, expected)
            elif (
                not is_dist_subclass
                and expected in _DISTRIBUTION_PROTOCOLS
                and isinstance(value, Distribution)
                and not isinstance(value, expected)
            ):
                try:
                    out[name] = converter_registry.convert(value, expected)
                except (TypeError, AttributeError):
                    pass  # Let the impl function's own check raise

        return out

    def _find_broadcast_args(
        self, values: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """Classify argument values into distribution-broadcast and
        RecordArray-broadcast groups.

        Returns
        -------
        (dist_args, ra_args) : tuple of list
            - ``dist_args``: arguments where a Distribution was passed
              but the type hint expects a concrete (non-Distribution)
              type. These are handled by the existing Monte Carlo
              marginalisation path.
            - ``ra_args``: arguments where a ``RecordArray`` with
              nonempty ``batch_shape`` was passed to a slot whose type
              hint is *not* a RecordArray (or subclass). These are
              handled by the new parameter-sweep stack path introduced
              in issue #130.

        The two groups are disjoint — a value is either a Distribution
        or a RecordArray or neither. Both groups firing on the same
        call triggers the nested regime in ``_broadcast``: outer stack
        over the RecordArray rows, inner marginalise over the
        Distribution MC draws.

        Raises
        ------
        ValueError
            If two or more RecordArray args have different
            ``batch_shape`` (lockstep broadcasting requires matching
            leading axes).
        """
        dist_broadcast: list[str] = []
        ra_broadcast: list[str] = []
        for name, value in values.items():
            expected = self._hints.get(name)

            # --- RecordArray branch ---------------------------------------
            if isinstance(value, RecordArray) and len(value.batch_shape) > 0:
                # Dispatch rules (see issue #130):
                # - Hint is RecordArray (or subclass) → caller wants the
                #   batched object as-is, don't broadcast.
                # - Hint is ``typing.Any`` → caller opted out of
                #   help ("anything goes"). Match their intent by
                #   skipping broadcast. Also the signal used by
                #   ``ops.log_prob`` / ``ops.prob`` / ``ops.expectation``
                #   where ``value: Any`` consumes the batched Record
                #   directly.
                # - Hint is a ``Record`` / ``NumericRecord`` subclass →
                #   caller wants a scalar Record; broadcast over rows.
                # - Hint is any other concrete type, or no hint at all →
                #   broadcast (the caller didn't express a batched
                #   preference and a per-row call is the friendlier
                #   default).
                import typing
                try:
                    is_ra_hint = (
                        isinstance(expected, type)
                        and issubclass(expected, RecordArray)
                    )
                except TypeError:
                    is_ra_hint = False
                is_any_hint = expected is typing.Any
                if is_ra_hint or is_any_hint:
                    continue
                ra_broadcast.append(name)
                continue

            # --- Distribution branch (existing logic) ---------------------
            if not converter_registry.is_distribution_type(value):
                continue
            # Unwrap parameterized generics (e.g. Distribution[T]).
            origin = getattr(expected, "__origin__", None)
            expected_type = origin if isinstance(origin, type) else expected
            try:
                is_dist_hint = (
                    expected_type is not None
                    and isinstance(expected_type, type)
                    and issubclass(expected_type, Distribution)
                )
            except TypeError:
                is_dist_hint = False
            if not is_dist_hint and expected in _DISTRIBUTION_PROTOCOLS:
                is_dist_hint = True
            if is_dist_hint:
                continue
            # Auto-convert external distribution types to ProbPipe
            if not isinstance(value, Distribution):
                values[name] = converter_registry.convert(
                    value, NumericRecordDistribution,
                )
                value = values[name]
            dist_broadcast.append(name)

        # Lockstep shape check: all RecordArrays must agree on batch_shape.
        if len(ra_broadcast) >= 2:
            shapes = {n: values[n].batch_shape for n in ra_broadcast}
            unique = set(shapes.values())
            if len(unique) > 1:
                raise ValueError(
                    f"Cannot broadcast RecordArray args with mismatched "
                    f"batch_shapes: {shapes}. Align them explicitly "
                    f"(e.g., via FullFactorialDesign)."
                )

        return dist_broadcast, ra_broadcast

    def _get_key(self):
        """Split and advance the internal PRNG key."""
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def _resolve_vectorize(self, values: dict[str, Any], broadcast_args: list[str]) -> str:
        """Resolve the vectorization strategy, caching the auto-detection result.

        Returns ``"jax"`` or ``"loop"``.  This is independent of orchestration
        (``workflow_kind``), which wraps whichever strategy is chosen.
        """
        if self._vectorize != "auto":
            return self._vectorize

        if self._resolved_vectorize is not None:
            return self._resolved_vectorize

        # Probe JAX traceability with dummy inputs
        try:
            dummy_kw = {}
            for name, param in self._sig.parameters.items():
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
                        es = dist.event_shape
                        dummy_kw[name] = jnp.zeros(es) if es else jnp.zeros(())
                elif name in values:
                    v = values[name]
                    if isinstance(v, jnp.ndarray):
                        dummy_kw[name] = v
                    elif hasattr(v, '__array__'):
                        dummy_kw[name] = jnp.asarray(v)
                    else:
                        dummy_kw[name] = v
            jax.make_jaxpr(lambda kw: self._func(**kw))(dummy_kw)
            self._resolved_vectorize = "jax"
        except Exception:
            logger.info(
                "Function '%s' is not JAX-traceable; using loop vectorization.",
                self._name,
            )
            self._resolved_vectorize = "loop"

        return self._resolved_vectorize

    def _broadcast(
        self,
        values: dict[str, Any],
        dist_args: list[str],
        ra_args: list[str],
        n_broadcast_samples: int,
        do_include_inputs: bool = False,
    ) -> Any:
        """Dispatcher: route to the right broadcast regime (issue #130).

        Three regimes:

        - ``dist_args`` only — existing Monte Carlo marginalisation via
          ``_broadcast_distributions_only``. Output: marginal
          distribution (``_MixtureMarginal`` / ``_ArrayMarginal`` / ...).
        - ``ra_args`` only — pure parameter sweep. One inner call per
          RecordArray row; no marginalisation. Output: stacked
          aggregate (``NumericRecordArray`` / ``RecordArray`` /
          ``DistributionArray``) via ``_make_stack``.
        - **Both** — nested regime. Outer loop over ``ra_args[0]`` rows,
          each iteration running an inner distribution-only broadcast
          for ``dist_args``. Output always a ``DistributionArray`` of
          per-row marginals (satisfies the
          Record | RecordArray | Distribution output contract).

        The ``include_inputs=True`` override still produces the
        inner-path ``BroadcastDistribution`` when ``dist_args`` fire,
        but it is **not** supported on the pure-sweep or nested paths
        (the stack has no single joint distribution to return; the
        inputs are known via provenance). Calling with
        ``include_inputs=True`` and any ``ra_args`` raises.
        """
        # ---- No RecordArray: delegate to the existing path -------------
        if not ra_args:
            return self._broadcast_distributions_only(
                values, dist_args, n_broadcast_samples, do_include_inputs,
            )

        if do_include_inputs:
            raise NotImplementedError(
                "include_inputs=True is not supported with RecordArray "
                "broadcasting. The inputs are already available via "
                "provenance on the stacked output."
            )

        # PR 1 supports 1-D sweeps only. Multi-dim batch_shapes are a
        # follow-up (would require flattening + reshaping of outputs).
        sweep_ra_shape = values[ra_args[0]].batch_shape
        if len(sweep_ra_shape) > 1:
            raise NotImplementedError(
                f"Multi-dimensional RecordArray batch_shape={sweep_ra_shape} "
                f"broadcasting is not supported in PR 1; flatten via "
                f"reshape / stack first."
            )
        n = sweep_ra_shape[0]

        # ---- Pure sweep (no Distribution args) -------------------------
        if not dist_args:
            per_row = self._execute_sweep_rows(values, ra_args, n)
            aggregate = _make_stack(per_row, n=n, name=self._name)
            provenance = self._make_sweep_provenance(
                values, ra_args, dist_args, n=n, k=0,
            )
            return _coerce_output(
                aggregate, broadcast_mode="stack", provenance=provenance,
            )

        # ---- Nested (RecordArray + Distribution) -----------------------
        # Per sweep row: run an inner distribution-only broadcast,
        # marginalise over the k MC draws, collect the n per-row
        # marginals and stack them into a DistributionArray.
        per_row_marginals: list[Distribution] = []
        for i in range(n):
            row_values = self._slice_ra_args(values, ra_args, i)
            inner = self._broadcast_distributions_only(
                row_values, dist_args, n_broadcast_samples,
                do_include_inputs=True,
            )
            # ``_broadcast_distributions_only`` with include_inputs=True
            # returns a BroadcastDistribution; call .marginalize() to
            # get the output-only marginal distribution.
            if isinstance(inner, BroadcastDistribution):
                marginal = inner.marginalize()
            else:
                # A non-broadcast inner (shouldn't happen when dist_args
                # non-empty, but handle defensively).
                marginal = inner
            per_row_marginals.append(marginal)

        stacked = _make_distribution_array(
            per_row_marginals, name=self._name or "sweep",
        )
        provenance = self._make_sweep_provenance(
            values, ra_args, dist_args, n=n, k=n_broadcast_samples,
        )
        return _coerce_output(
            stacked, broadcast_mode="nested", provenance=provenance,
        )

    # ----- Helpers for the RecordArray-broadcast path ----------------------

    def _slice_ra_args(
        self,
        values: dict[str, Any],
        ra_args: list[str],
        i: int,
    ) -> dict[str, Any]:
        """Materialise the ``i``-th sweep row by integer-indexing every
        RecordArray argument.

        Non-RecordArray arguments pass through unchanged; each indexed
        RecordArray yields a ``Record`` (or ``NumericRecord`` for
        ``NumericRecordArray``) via ``__getitem__``.
        """
        out = dict(values)
        for name in ra_args:
            out[name] = values[name][i]
        return out

    def _execute_sweep_rows(
        self,
        values: dict[str, Any],
        ra_args: list[str],
        n: int,
    ) -> Any:
        """Execute n inner calls, one per sweep row. Returns a list of
        outputs (Python-loop path) or a stacked pytree (JAX vmap path).

        Both returns are valid inputs for ``_make_stack``.
        """
        vectorize = self._resolve_vectorize(values, ra_args)

        if vectorize == "jax":
            # vmap-friendly: split the RecordArray leaves along batch
            # axis 0, rebuild a single-row Record inside the function
            # call so the body sees a plain Record of scalars.
            static = {k: v for k, v in values.items() if k not in ra_args}

            def single_call(ra_slice_leaves):
                kw = dict(static)
                for name in ra_args:
                    ra = values[name]
                    # ra_slice_leaves[name] is a dict of per-field
                    # leaves shaped like a single row; reconstruct the
                    # Record so the body sees a Record of scalars.
                    kw[name] = ra._record_cls(ra_slice_leaves[name])
                return self._func(**kw)

            # Build the vmap-over-leaves input: a dict {name: {field: batched_array}}.
            vmap_input = {
                name: {f: values[name][f] for f in values[name].fields}
                for name in ra_args
            }
            return jax.vmap(single_call)(vmap_input)

        # Loop path.
        per_row_values = [self._slice_ra_args(values, ra_args, i) for i in range(n)]
        return self._execute_many(per_row_values)

    def _make_sweep_provenance(
        self,
        values: dict[str, Any],
        ra_args: list[str],
        dist_args: list[str],
        *,
        n: int,
        k: int,
    ) -> Provenance:
        """Build the Provenance node for a sweep output.

        Parents are the RecordArray / Distribution inputs that drove
        the broadcast. Metadata records the regime and count so
        downstream tooling can report whether uncertainty was
        marginalised or stacked.
        """
        regime = "nested" if dist_args else "stack"
        parents = tuple(values[name] for name in ra_args) + tuple(
            values[name] for name in dist_args
            if isinstance(values[name], Distribution)
        )
        return Provenance(
            operation=f"workflow.{regime}",
            parents=parents,
            metadata={
                "func": self._name or self._func.__name__,
                "n": n,
                "k": k,
                "ra_args": list(ra_args),
                "dist_args": list(dist_args),
            },
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
        ``Distribution`` instances after ``_find_broadcast_args``),
        calls the user's function once per sample, and wraps the n
        outputs as a single marginal distribution.

        This method was historically named ``_broadcast``. Commit 4 of
        PR 1 (issue #130) will re-introduce ``_broadcast`` as a dispatch
        layer that routes between this Distribution-only path, the new
        RecordArray-stack path, and the nested combination — hence the
        rename.

        Vectorization (``"jax"`` vs ``"loop"``) and orchestration
        (``workflow_kind``) are resolved independently:

        - **vectorize="jax"**: samples are dispatched via ``jax.vmap``.
        - **vectorize="loop"**: samples are dispatched via a Python loop,
          with optional empirical enumeration and threading.
        - **workflow_kind="task"/"flow"**: whichever vectorization strategy
          is chosen gets wrapped in a Prefect task or flow for compute-graph
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

        vectorize = self._resolve_vectorize(values, broadcast_args)

        # JAX vectorization: vmap (no enumeration path — sample everything)
        if vectorize == "jax":
            result = self._broadcast_jax(values, broadcast_args, n_broadcast_samples)
        else:
            # Loop vectorization: supports empirical enumeration
            # Collect candidate empirical dists (small enough individually), sorted smallest first
            candidates = []
            sample_args: dict[str, Distribution] = {}
            for name in broadcast_args:
                dist = values[name]
                if isinstance(dist, EmpiricalDistribution) and dist.n <= n_broadcast_samples:
                    candidates.append((name, dist))
                else:
                    sample_args[name] = dist

            candidates.sort(key=lambda pair: pair[1].n)

            # Greedily include smallest empirical dists while product stays within budget
            empirical_args: dict[str, EmpiricalDistribution] = {}
            product_size = 1
            for name, dist in candidates:
                if product_size * dist.n <= n_broadcast_samples:
                    empirical_args[name] = dist
                    product_size *= dist.n
                else:
                    # Too large to enumerate — sample from it instead
                    sample_args[name] = dist

            if empirical_args:
                result = self._broadcast_enumerate(
                    values, empirical_args, sample_args, product_size, n_broadcast_samples
                )
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
                "vectorize": vectorize,
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
        """
        Sample all broadcast arguments, handling view reconnection.

        When multiple arguments are views from the same parent
        distribution, the parent is sampled once and component samples
        are distributed to the appropriate arguments.  This preserves
        correlation between jointly-distributed components.

        Views are detected via duck-typing (``_parent`` + ``_key_path``
        attributes).
        """
        view_groups: dict[int, dict] = {}  # id(parent) → {parent, views}
        independent: list[str] = []

        for name in broadcast_args:
            dist = values[name]
            if hasattr(dist, "_parent") and hasattr(dist, "_key_path"):
                pid = id(dist._parent)
                if pid not in view_groups:
                    view_groups[pid] = {"parent": dist._parent, "views": {}}
                view_groups[pid]["views"][name] = dist
            else:
                independent.append(name)

        sampled: dict[str, Array] = {}

        # Sample each parent once, distribute to arguments.
        for group in view_groups.values():
            key, subkey = jax.random.split(key)
            structured = group["parent"]._sample(subkey, (n,))
            for arg_name, view in group["views"].items():
                if hasattr(view, "_extract"):
                    sampled[arg_name] = view._extract(structured)
                else:
                    val = structured
                    for k in view._key_path:
                        val = val[k]
                    sampled[arg_name] = val

        # Sample independent distributions
        for name in independent:
            key, subkey = jax.random.split(key)
            sampled[name] = values[name]._sample(subkey, (n,))

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

        for combo in cartesian_product(*(range(d.n) for d in emp_dists)):
            # Compute empirical product weight
            emp_weight = 1.0
            for name, dist, i in zip(emp_names, emp_dists, combo):
                emp_weight *= float(dist.weights[i])

            for _ in range(reps_per_combo):
                call_values = dict(values)

                # Set empirical samples
                for name, dist, i in zip(emp_names, emp_dists, combo):
                    call_values[name] = dist.samples[i]

                # Set sampled values for non-empirical distributions
                for name in sample_args:
                    call_values[name] = sampled[name][sample_idx]

                # Weight: empirical product weight divided evenly across reps
                weights.append(emp_weight / reps_per_combo)
                call_value_list.append(call_values)
                sample_idx += 1

        results = self._execute_many(call_value_list)

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
                call_values[name] = samples_per_arg[name][i]
            call_value_list.append(call_values)

        results = self._execute_many(call_value_list)

        return BroadcastDistribution(
            input_samples=samples_per_arg,
            output_samples=results,
            weights=None,
            broadcast_args=broadcast_args,
        )

    def _execute_many(self, call_value_list: list[dict[str, Any]]) -> list:
        """
        Execute the wrapped function for every dict in *call_value_list* and
        return the results in the same order.

        Dispatch strategy:
          - workflow_kind="task"  → Prefect task.map() (single mapped task)
          - workflow_kind="flow"  → Prefect: one wrapping flow with task.map()
          - parallel=True/int    → concurrent.futures.ThreadPoolExecutor
          - otherwise            → sequential list comprehension
        """
        kind = self.effective_workflow_kind
        if kind in (WorkflowKind.TASK, WorkflowKind.FLOW):
            if kind == WorkflowKind.TASK:
                return self._execute_many_prefect_task(call_value_list)
            return self._execute_many_prefect_flow(call_value_list)
        if self._parallel:
            return self._execute_many_threaded(call_value_list)
        return [self._func(**v) for v in call_value_list]

    def _execute_many_threaded(self, call_value_list: list[dict[str, Any]]) -> list:
        max_workers = self._parallel if isinstance(self._parallel, int) else None
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda v: self._func(**v), call_value_list))

    def _map_task(self, call_value_list: list[dict[str, Any]], task_name: str | None = None) -> list:
        """Create a Prefect task wrapping self._func, .map() over all calls, and resolve."""
        func = self._func

        @task(name=task_name or self._name)
        def run_func(**kwargs):
            return func(**kwargs)

        keys = call_value_list[0].keys()
        kwargs_by_param = {k: [d[k] for d in call_value_list] for k in keys}
        futures = run_func.map(**kwargs_by_param)
        return [f.result() for f in futures]

    def _execute_many_prefect_task(self, call_value_list: list[dict[str, Any]]) -> list:
        """Use Prefect task.map() inside a lightweight flow.

        Prefect 3.x requires ``task.map()`` to be called within a flow
        context.  This mode creates a minimal wrapper flow so the mapped
        task runs are tracked but not grouped under a named flow.
        """
        outer = self
        runner = prefect_config.resolve_task_runner()

        @flow(name=f"{self._name}_map",
              **({"task_runner": runner} if runner is not None else {}))
        def _task_map_flow():
            return outer._map_task(call_value_list)

        return _task_map_flow()

    def _execute_many_prefect_flow(self, call_value_list: list[dict[str, Any]]) -> list:
        """Wrap a mapped task inside a named Prefect flow."""
        outer = self
        runner = prefect_config.resolve_task_runner()

        @flow(name=self._name,
              **({"task_runner": runner} if runner is not None else {}))
        def mapped_flow():
            return outer._map_task(call_value_list, task_name=f"{outer._name}_run")

        return mapped_flow()
 

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
            for param_name in attr._param_names:
                if attr._is_dependency_param(param_name) and param_name in self._child_nodes:
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
