from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping, get_type_hints
import inspect
from concurrent.futures import ThreadPoolExecutor
from itertools import product as cartesian_product
from types import MappingProxyType
import warnings

import numpy as np
from prefect import task, flow
from graphviz import Digraph

# THIS WILL BE CHANGED; just for implementing the template of conversion logic
from ..distributions.distribution import Distribution, EmpiricalDistribution
from ..distributions.real_vector.gaussian import Gaussian
DISTRIBUTION_TYPES = (Distribution, EmpiricalDistribution, Gaussian)

__all__ = ["InputFrozenError", "wf", "Node", "abstractwf", "Workflow", "Module", "AbstractModule"]

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
    Base DAG unit.

    A Node:
    - has child nodes (other Nodes it is allowed to call)
    - has inputs (non-Node values)
    - knows nothing about execution, prefect, or workflow functions

    New convenience:
      Node(foo=SomeNode(), bar=123)  # auto-splits

    Backward compatible:
      Node(child_nodes={...}, inputs={...})
    """

    def __init__(
        self,
        *,
        child_nodes: Dict[str, "Node"] | None = None,
        inputs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        # Start from explicitly provided dicts
        child_nodes = dict(child_nodes or {})
        inputs = dict(inputs or {})

        # Auto-split kwargs into child_nodes vs inputs
        # (kwargs override nothing by default; if you want kwargs to override, swap the order)
        for k, v in kwargs.items():
            if isinstance(v, Node):
                # If user mistakenly passed a Node both in inputs and kwargs, this will correct it
                child_nodes[k] = v
                # In case it was also present in inputs, remove it (avoid inconsistent state)
                if k in inputs:
                    del inputs[k]
            else:
                # If user mistakenly passed a non-Node both in child_nodes and kwargs, correct it
                inputs[k] = v
                if k in child_nodes:
                    del child_nodes[k]

        # Validate child nodes
        for name, node in child_nodes.items():
            if not isinstance(node, Node):
                raise TypeError(f"Child node '{name}' must be a Node, got {type(node)}")

        # Validate inputs
        for name, value in inputs.items():
            if isinstance(value, Node):
                raise TypeError(
                    f"Input '{name}' is a Node; Nodes must be declared as child_nodes "
                    f"(or passed as a normal kwarg so it is auto-detected)"
                )

        # Freeze internal state (read-only)
        self._child_nodes = MappingProxyType(dict(child_nodes))
        self._inputs = MappingProxyType(dict(inputs))

    @property
    def child_nodes(self) -> Mapping[str, "Node"]:
        return self._child_nodes

    @property
    def inputs(self) -> Mapping[str, Any]:
        return self._inputs
    

class Workflow(Node):
    """
    A single executable DAG node wrapping exactly one function.

    Infers dependency-vs-input from the function signature and type hints.
    Optionally resolves missing values from an attached Module.

    **Broadcasting**: When a ``Distribution`` is passed for an argument whose
    type hint is *not* a ``Distribution`` subclass, the workflow automatically
    samples from the distribution and calls the wrapped function once per
    sample, returning an ``EmpiricalDistribution`` over the outputs (or a
    plain list when results are not numeric).

    Parameters
    ----------
    func : Callable
        The function to wrap.
    workflow_kind : str or None
        Execution backend: ``"task"`` (Prefect task), ``"flow"`` (Prefect
        flow), or ``None`` (plain Python).
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
    parallel : bool or int
        Controls parallel execution during broadcasting.
        ``False`` → sequential, ``True`` → ``ThreadPoolExecutor`` with
        default workers, ``int`` → explicit ``max_workers``.  When
        *workflow_kind* is set, Prefect ``task.map()`` is used instead.
    """

    def __init__(
        self,
        *,
        func: Callable,
        workflow_kind: str | None = None,   # "task" or "flow" or None
        name: str | None = None,
        bind: Dict[str, Any] | None = None,         # construction-time bindings (defaults/config)
        module: Any | None = None,                  # typically a Module; kept as Any to avoid import cycles
        n_broadcast_samples: int = 128,              # default number of samples for broadcasting
        parallel: bool | int = False,               # True/int for ThreadPoolExecutor, or Prefect .map()
        **kwargs: Any,                              # convenience bindings (merged into bind)
    ):
        self._func = func
        self._sig = inspect.signature(func)
        self._hints = get_type_hints(func)
        self._workflow_kind = workflow_kind
        self._name = name or getattr(func, "__name__", self.__class__.__name__)
        self._module = module
        self._n_broadcast_samples = n_broadcast_samples
        self._parallel = parallel

        # bind = "construction-time inputs" (defaults/config). kwargs are also treated as bind.
        b = dict(bind or {})
        b.update(kwargs)
        self._bind = b

        # Keep Node base class but don't use its child_nodes/inputs split here.
        # (Module is the container; this node *infers* needs from signature.)
        super().__init__(child_nodes={}, inputs={})

        # Precompute parameter metadata once
        self._param_names = [p for p in self._sig.parameters if p != "self"]

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
        # Extract n_broadcast_samples if it's not a parameter of the wrapped function
        if "n_broadcast_samples" in call_inputs and "n_broadcast_samples" not in self._param_names:
            n_broadcast_samples = call_inputs.pop("n_broadcast_samples")
        else:
            n_broadcast_samples = self._n_broadcast_samples

        values = self._resolve_inputs(call_inputs)
        values = self._convert_distributions(values)

        broadcast_args = self._find_broadcast_args(values)
        if broadcast_args:
            return self._broadcast(values, broadcast_args, n_broadcast_samples)

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

        # validate required params exist (after resolution)
        for name, param in self._sig.parameters.items():
            if name == "self":
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
                        f"Workflow '{self._name}' expects dependency '{name}: {ann}' to be a Node, "
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
                and isinstance(value, DISTRIBUTION_TYPES)
                and not isinstance(value, expected)
            ):
                out[name] = expected.from_distribution(value)

        return out

    def _find_broadcast_args(self, values: Dict[str, Any]) -> list[str]:
        """
        Identify arguments where a Distribution was passed but the type hint
        expects a concrete (non-Distribution) type.
        """
        broadcast = []
        for name, value in values.items():
            if not isinstance(value, Distribution):
                continue
            expected = self._hints.get(name)
            # If hint IS a Distribution subclass, _convert_distributions handled it
            try:
                is_dist_hint = expected is not None and isinstance(expected, type) and issubclass(expected, Distribution)
            except TypeError:
                is_dist_hint = False
            if is_dist_hint:
                continue
            broadcast.append(name)
        return broadcast

    def _broadcast(
        self,
        values: Dict[str, Any],
        broadcast_args: list[str],
        n_broadcast_samples: int,
    ) -> EmpiricalDistribution | list:
        """
        Sample from Distribution arguments and call the function once per sample.
        Returns an EmpiricalDistribution if results are numeric arrays,
        otherwise returns a list of results.

        When EmpiricalDistribution arguments have small enough support, enumerates their
        samples (cartesian product) and propagates weights instead of resampling.
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
            return self._broadcast_enumerate(
                values, empirical_args, sample_args, product_size, n_broadcast_samples
            )
        else:
            return self._broadcast_sample(values, broadcast_args, n_broadcast_samples)

    def _broadcast_enumerate(
        self,
        values: Dict[str, Any],
        empirical_args: Dict[str, EmpiricalDistribution],
        sample_args: Dict[str, Distribution],
        product_size: int,
        n_broadcast_samples: int,
    ) -> EmpiricalDistribution | list:
        """
        Enumerate the cartesian product of EmpiricalDistribution samples,
        propagating their weights. When non-empirical distributions are also
        present, draws as many samples as the budget allows per combination
        (n_broadcast_samples // product_size), each receiving equal weight scaled by
        the empirical product weight.
        """
        emp_names = list(empirical_args.keys())
        emp_dists = [empirical_args[name] for name in emp_names]

        # Use as many non-empirical samples per combo as budget allows
        reps_per_combo = max(1, n_broadcast_samples // product_size) if sample_args else 1
        total = product_size * reps_per_combo

        # Pre-sample non-empirical distributions
        sampled = {
            name: dist.sample(total)
            for name, dist in sample_args.items()
        }

        call_value_list = []
        weights = []
        sample_idx = 0

        for combo in cartesian_product(*(range(d.n) for d in emp_dists)):
            # Compute empirical product weight
            emp_weight = 1.0
            for name, dist, i in zip(emp_names, emp_dists, combo):
                emp_weight *= dist.weights[i]

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
        return self._collect_results(results, total, np.array(weights))

    def _broadcast_sample(
        self,
        values: Dict[str, Any],
        broadcast_args: list[str],
        n_broadcast_samples: int,
    ) -> EmpiricalDistribution | list:
        """
        Sample n_broadcast_samples from each Distribution argument and call the function
        once per sample (uniform weights).
        """
        samples_per_arg = {
            name: values[name].sample(n_broadcast_samples)
            for name in broadcast_args
        }

        call_value_list = []
        for i in range(n_broadcast_samples):
            call_values = dict(values)
            for name in broadcast_args:
                call_values[name] = samples_per_arg[name][i]
            call_value_list.append(call_values)

        results = self._execute_many(call_value_list)
        return self._collect_results(results, n_broadcast_samples)

    @staticmethod
    def _collect_results(
        results: list,
        n: int,
        weights: np.ndarray | None = None,
    ) -> EmpiricalDistribution | list:
        """
        Stack results into an EmpiricalDistribution if possible,
        otherwise return a plain list.
        """
        try:
            stacked = np.stack([np.asarray(r, dtype=float) for r in results], axis=0)
        except (ValueError, TypeError):
            return results

        if stacked.ndim == 1:
            stacked = stacked.reshape(-1, 1)
        elif stacked.ndim > 2:
            stacked = stacked.reshape(n, -1)

        return EmpiricalDistribution(stacked, weights=weights)

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
        if self._workflow_kind == "task":
            return self._execute_many_prefect_task(call_value_list)
        if self._workflow_kind == "flow":
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
        """Use Prefect task.map() to run all calls as a single mapped task."""
        return self._map_task(call_value_list)

    def _execute_many_prefect_flow(self, call_value_list: list[Dict[str, Any]]) -> list:
        """Wrap a mapped task inside a single Prefect flow."""
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
        Replace @wf methods with Workflow instances.

        Key change:
          - We do NOT precompute wf_child_nodes / wf_inputs here.
          - Workflow infers needs from signature and resolves from this module at call time.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not callable(attr) or not getattr(attr, "_is_workflow", False):
                continue

            func = attr

            # skip abstract workflows
            if getattr(func, "__isabstractmethod__", False):
                continue

            wf = Workflow(
                func=func,
                workflow_kind=self._workflow_kind,
                name=f"{self.__class__.__name__}.{func.__name__}",
                module=self,  # <--- this is the key
            )

            setattr(self, attr_name, wf)

    def dag(self):
        """Return a Graphviz DAG visualization of this module."""        
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

            # Workflow nodes inside the module
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if not isinstance(attr, Workflow):
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
            if not isinstance(attr, Workflow):
                continue

            wf_name = attr._name

            # Infer dependencies from workflow signature
            # (Workflows don't store child_nodes; they resolve dependencies at runtime)
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
                    f"Workflow '{name}' implementation is missing parameter '{ap.name}'.\n"
                    f"Expected (abstract): {abs_sig}\n"
                    f"Got (impl):          {impl_sig}"
                )
            if ip.kind != ap.kind:
                raise TypeError(
                    f"Workflow '{name}' parameter '{ap.name}' kind mismatch.\n"
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
