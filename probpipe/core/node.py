from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping, get_type_hints
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import product as cartesian_product
from types import MappingProxyType
import warnings

import jax
import jax.numpy as jnp

try:
    from prefect import task, flow
except ImportError:
    task = flow = None

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

from ..custom_types import PRNGKey, Array
from .distribution import (
    ArrayDistribution,
    BroadcastDistribution,
    Distribution,
    EmpiricalDistribution,
    _make_marginal,
)
from .provenance import Provenance
from .protocols import (
    SupportsConditioning,
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsNamedComponents,
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
    SupportsNamedComponents,
)
from ..distributions.joint import DistributionView
from ..converters import converter_registry

logger = logging.getLogger(__name__)

__all__ = ["InputFrozenError", "wf", "Node", "abstractwf", "WorkflowFunction", "Module", "AbstractModule"]

class InputFrozenError(Exception):
    pass

def abstractwf(func: Callable):
    """
    Marks a method as:
      - a workflow interface (via wf)
      - abstract (enforced by ABCMeta)

    This allows abstract modules to declare workflow-shaped interfaces
    without providing implementations.
    """
    return abstractmethod(wf(func))


def wf(func: Callable):
    func._is_workflow = True
    return func


class Node(ABC):
    """
    Base unit of the ProbPipe computational dependency graph. 

    Keyword arguments are automatically split by type: values that are
    ``Node`` instances become *child nodes* (dependencies on other DAG
    units), and everything else becomes *inputs* (data, configuration,
    hyperparameters).  Both collections are frozen after construction.
    """

    def __init__(self, **kwargs: Any):
        child_nodes: Dict[str, Node] = {}
        inputs: Dict[str, Any] = {}

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
    workflow_kind : str or None
        Prefect orchestration mode: ``"task"`` (wrap in a Prefect task),
        ``"flow"`` (wrap in a Prefect flow), or ``None`` (plain Python).
        Orchestration is independent of vectorization: a JAX-vmapped
        broadcast can still be wrapped in a Prefect task for tracing.
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
        workflow_kind: str | None = None,   # "task" or "flow" or None
        name: str | None = None,
        bind: Dict[str, Any] | None = None,         # construction-time bindings (defaults/config)
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
        self._workflow_kind = workflow_kind
        self._name = name or getattr(func, "__name__", self.__class__.__name__)
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

        # Reserved names that would collide with WorkflowFunction call-time overrides
        _RESERVED = {"n_broadcast_samples", "seed", "include_inputs"}
        collision = _RESERVED & set(self._param_names)
        if collision:
            raise ValueError(
                f"Function '{self._name}' has parameter(s) {collision} which are "
                f"reserved by WorkflowFunction for call-time overrides. Rename them in "
                f"your function signature."
            )

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

    def __call__(self, **call_inputs):
        # Extract reserved call-time overrides (collision already prevented in __init__)
        n_broadcast_samples = call_inputs.pop("n_broadcast_samples", self._n_broadcast_samples)
        do_include_inputs = call_inputs.pop("include_inputs", self._include_inputs)

        if "seed" in call_inputs:
            self._key = jax.random.PRNGKey(call_inputs.pop("seed"))

        values = self._resolve_inputs(call_inputs)
        values = self._convert_distributions(values)

        broadcast_args = self._find_broadcast_args(values)
        if broadcast_args:
            return self._broadcast(values, broadcast_args, n_broadcast_samples, do_include_inputs)

        return self._execute_many([values])[0]

    def _resolve_inputs(self, call_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build kwargs for calling the underlying function.

        Precedence (highest -> lowest):
          1) call_inputs
          2) self._bind
          3) module.child_nodes/module.inputs (if module attached)
          4) function default values
        """
        values: Dict[str, Any] = {}

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

    def _convert_distributions(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert distributions based on type hints.
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

        return out

    def _find_broadcast_args(self, values: Dict[str, Any]) -> list[str]:
        """
        Identify arguments where a Distribution was passed but the type hint
        expects a concrete (non-Distribution) type.
        """
        broadcast = []
        for name, value in values.items():
            if not converter_registry.is_distribution_type(value):
                continue
            # Check if the type hint indicates a distribution/protocol parameter.
            # If so, the caller expects a distribution object — don't broadcast.
            expected = self._hints.get(name)
            try:
                is_dist_hint = expected is not None and isinstance(expected, type) and issubclass(expected, Distribution)
            except TypeError:
                is_dist_hint = False
            if not is_dist_hint and expected in _DISTRIBUTION_PROTOCOLS:
                is_dist_hint = True
            if is_dist_hint:
                continue
            # Auto-convert external distribution types to ProbPipe
            if not isinstance(value, Distribution):
                values[name] = converter_registry.convert(value, ArrayDistribution)
                value = values[name]
            broadcast.append(name)
        return broadcast

    def _get_key(self):
        """Split and advance the internal PRNG key."""
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def _resolve_vectorize(self, values: Dict[str, Any], broadcast_args: list[str]) -> str:
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
                    dist = values[name]
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
        values: Dict[str, Any],
        broadcast_args: list[str],
        n_broadcast_samples: int,
        do_include_inputs: bool = False,
    ) -> BroadcastDistribution | Distribution:
        """
        Sample from Distribution arguments and call the function once per sample.
        Returns the output marginal by default, or a ``BroadcastDistribution``
        holding the joint over inputs and outputs when ``do_include_inputs``
        is True.

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
            sample_args: Dict[str, Distribution] = {}
            for name in broadcast_args:
                dist = values[name]
                if isinstance(dist, EmpiricalDistribution) and dist.n <= n_broadcast_samples:
                    candidates.append((name, dist))
                else:
                    sample_args[name] = dist

            candidates.sort(key=lambda pair: pair[1].n)

            # Greedily include smallest empirical dists while product stays within budget
            empirical_args: Dict[str, EmpiricalDistribution] = {}
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
                "orchestrate": self._workflow_kind or "none",
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
        values: Dict[str, Any],
        broadcast_args: list[str],
        n: int,
        key: PRNGKey,
    ) -> Dict[str, Array]:
        """
        Sample all broadcast arguments, handling DistributionView reconnection.

        When multiple arguments are DistributionViews from the same parent
        JointDistribution, the parent is sampled once and component samples
        are distributed to the appropriate arguments.  This preserves
        correlation between jointly-distributed components.
        """
        # Group DistributionView args by parent
        joint_groups: Dict[int, Dict] = {}  # id(parent) → {parent, mappings}
        independent: list[str] = []

        for name in broadcast_args:
            dist = values[name]
            if isinstance(dist, DistributionView):
                pid = id(dist._parent)
                if pid not in joint_groups:
                    joint_groups[pid] = {"parent": dist._parent, "mappings": {}}
                # Store the key path (tuple of strings) for nested pytree navigation
                joint_groups[pid]["mappings"][name] = dist._key_path
            else:
                independent.append(name)

        sampled: Dict[str, Array] = {}

        # Sample each joint group once, distribute to arguments
        for group in joint_groups.values():
            key, subkey = jax.random.split(key)
            structured = group["parent"]._sample(subkey, (n,))
            for arg_name, key_path in group["mappings"].items():
                # Walk the (possibly nested) sample pytree to extract the leaf
                val = structured
                for k in key_path:
                    val = val[k]
                sampled[arg_name] = val

        # Sample independent distributions
        for name in independent:
            key, subkey = jax.random.split(key)
            sampled[name] = values[name]._sample(subkey, (n,))

        return sampled

    def _broadcast_jax(
        self,
        values: Dict[str, Any],
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
        if self._workflow_kind is not None:
            if task is None:
                raise ImportError(
                    "Prefect is required for workflow_kind='task'/'flow'. "
                    "Install it with: pip install probpipe[prefect]"
                )
            if self._workflow_kind == "task":
                run_vmap = task(name=f"{self._name}_vmap")(run_vmap)
            else:
                run_vmap = flow(name=f"{self._name}_vmap")(run_vmap)

        results = run_vmap()
        return BroadcastDistribution(
            input_samples=sampled,
            output_samples=results,
            weights=None,
            broadcast_args=broadcast_args,
        )

    def _broadcast_enumerate(
        self,
        values: Dict[str, Any],
        empirical_args: Dict[str, EmpiricalDistribution],
        sample_args: Dict[str, Distribution],
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
        values: Dict[str, Any],
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

    def _execute_many(self, call_value_list: list[Dict[str, Any]]) -> list:
        """
        Execute the wrapped function for every dict in *call_value_list* and
        return the results in the same order.

        Dispatch strategy:
          - workflow_kind="task"  → Prefect task.map() (single mapped task)
          - workflow_kind="flow"  → Prefect: one wrapping flow with task.map()
          - parallel=True/int    → concurrent.futures.ThreadPoolExecutor
          - otherwise            → sequential list comprehension
        """
        if self._workflow_kind in ("task", "flow"):
            if task is None:
                raise ImportError(
                    "Prefect is required for workflow_kind='task'/'flow'. "
                    "Install it with: pip install probpipe[prefect]"
                )
            if self._workflow_kind == "task":
                return self._execute_many_prefect_task(call_value_list)
            return self._execute_many_prefect_flow(call_value_list)
        if self._parallel:
            return self._execute_many_threaded(call_value_list)
        return [self._func(**v) for v in call_value_list]

    def _execute_many_threaded(self, call_value_list: list[Dict[str, Any]]) -> list:
        max_workers = self._parallel if isinstance(self._parallel, int) else None
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda v: self._func(**v), call_value_list))

    def _map_task(self, call_value_list: list[Dict[str, Any]], task_name: str | None = None) -> list:
        """Create a Prefect task wrapping self._func, .map() over all calls, and resolve."""
        func = self._func

        @task(name=task_name or self._name)
        def run_func(**kwargs):
            return func(**kwargs)

        keys = call_value_list[0].keys()
        kwargs_by_param = {k: [d[k] for d in call_value_list] for k in keys}
        futures = run_func.map(**kwargs_by_param)
        return [f.result() for f in futures]

    def _execute_many_prefect_task(self, call_value_list: list[Dict[str, Any]]) -> list:
        """Use Prefect task.map() inside a lightweight flow.

        Prefect 3.x requires ``task.map()`` to be called within a flow
        context.  This mode creates a minimal wrapper flow so the mapped
        task runs are tracked but not grouped under a named flow.
        """
        outer = self

        @flow(name=f"{self._name}_map")
        def _task_map_flow():
            return outer._map_task(call_value_list)

        return _task_map_flow()

    def _execute_many_prefect_flow(self, call_value_list: list[Dict[str, Any]]) -> list:
        """Wrap a mapped task inside a named Prefect flow."""
        outer = self

        @flow(name=self._name)
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

    def __init__(self, *, workflow_kind: str | None = None, **kwargs: Any):
        self._workflow_kind = workflow_kind
        super().__init__(**kwargs)
        # validate abstract workflow implementations before wrapping
        self._validate_abstract_workflow_implementations()

        self._build_workflows()

    def _build_workflows(self):
        """
        Replace @wf methods with WorkflowFunction instances.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not callable(attr) or not getattr(attr, "_is_workflow", False):
                continue

            func = attr

            # skip abstract workflows
            if getattr(func, "__isabstractmethod__", False):
                continue

            wf = WorkflowFunction(
                func=func,
                workflow_kind=self._workflow_kind,
                name=f"{self.__class__.__name__}.{func.__name__}",
                module=self,  # <--- this is the key
            )

            setattr(self, attr_name, wf)

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
          - base class declares @abstractwf interface
          - subclass defines a method with same name but forgets @wf
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

                # Must be marked as workflow (@wf)
                if not getattr(impl_attr, "_is_workflow", False):
                    raise TypeError(
                        f"{cls.__name__}.{name} implements an abstract workflow interface "
                        f"but is not marked with @wf."
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
    Base class for modules that declare workflow interfaces via @abstractwf.

    ABCMeta will prevent instantiation until all abstract workflows are implemented
    by a concrete subclass.
    """
    pass
