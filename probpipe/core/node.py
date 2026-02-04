from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping, get_type_hints
import inspect
from types import MappingProxyType

from prefect import task, flow
from graphviz import Digraph

# THIS WILL BE CHANGED; just for implementing the template of conversion logic
from probpipe import Distribution, EmpiricalDistribution, Gaussian
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

    Key change:
      - No longer takes child_nodes/inputs explicitly.
      - Infers dependency-vs-input from the function signature/type hints.
      - Optionally resolves missing values from an attached Module.
    """

    def __init__(
        self,
        *,
        func: Callable,
        workflow_kind: str | None = None,   # "task" or "flow" or None
        name: str | None = None,
        bind: Dict[str, Any] | None = None,         # construction-time bindings (defaults/config)
        module: Any | None = None,                  # typically a Module; kept as Any to avoid import cycles
        **kwargs: Any,                              # convenience bindings (merged into bind)
    ):
        self._func = func
        self._sig = inspect.signature(func)
        self._hints = get_type_hints(func)
        self._workflow_kind = workflow_kind
        self._name = name or getattr(func, "__name__", self.__class__.__name__)
        self._module = module

        # bind = "construction-time inputs" (defaults/config). kwargs are also treated as bind.
        b = dict(bind or {})
        b.update(kwargs)
        self._bind = b

        # Keep Node base class but don't use its child_nodes/inputs split here.
        # (Module is the container; this node *infers* needs from signature.)
        super().__init__(child_nodes={}, inputs={})

        # Precompute parameter metadata once
        self._param_names = [p for p in self._sig.parameters if p != "self"]

    # -------------------------
    # classification helpers
    # -------------------------

    def _is_dependency_param(self, name: str) -> bool:
        """
        Decide whether a parameter is a dependency (Node) vs normal input.

        Rule:
          - If annotation is a Node subclass => dependency
          - Else => normal input

        This matches your current architecture (deps are Nodes).
        """
        ann = self._hints.get(name)
        return isinstance(ann, type) and issubclass(ann, Node)

    # -------------------------
    # execution
    # -------------------------

    def __call__(self, **call_inputs):
        values = self._resolve_inputs(call_inputs)
        values = self._convert_distributions(values)

        if self._workflow_kind == "task":
            return self._run_as_task(values)
        if self._workflow_kind == "flow":
            return self._run_as_flow(values)

        return self._call_python(values)

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

            if (
                isinstance(expected, type)
                and issubclass(expected, Distribution)
                and isinstance(value, DISTRIBUTION_TYPES)
                and not isinstance(value, expected)
            ):
                out[name] = expected.from_distribution(value)

        return out

    def _call_python(self, values: Dict[str, Any]):
        return self._func(**values)

    def _run_as_task(self, values):
        @task(name=self._name)
        def wrapped():
            return self._call_python(values)

        return wrapped()

    def _run_as_flow(self, values):
        @flow(name=self._name)
        def wrapped():
            return self._call_python(values)

        return wrapped()
 
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

            for child_name in attr._child_nodes:
                if child_name not in self._child_nodes:
                    raise ValueError(
                        f"Workflow '{wf_name}' references unknown child node '{child_name}'"
                    )

                dot.edge(child_name, wf_name)

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
