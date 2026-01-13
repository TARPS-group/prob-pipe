import inspect
from abc import ABC
from typing import Any, Dict, get_type_hints, Callable

from probpipe.core.node import Node
from probpipe.core.workflow_node import WorkflowNode
from probpipe.viz.dag import visualize_module_dag

__all__ = ["ModuleNode", "AbstractModule"]


class ModuleNode(Node):
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
        Replace @wf methods with WorkflowNode instances.

        For each @wf function:
        - params annotated as Node subclasses are sourced from self.child_nodes
        - other params are sourced from self.inputs if provided
        - anything not found remains call-time (handled by WorkflowNode._bind_inputs)

        Abstract workflows (@abstractwf / @abstractmethod) are NOT wrapped.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not callable(attr) or not getattr(attr, "_is_workflow", False):
                continue

            func = attr

            # skips abstract workflows so ABCMeta can enforce implementation
            if getattr(func, "__isabstractmethod__", False):
                continue

            sig = inspect.signature(func)
            hints = get_type_hints(func)

            wf_child_nodes: Dict[str, Node] = {}
            wf_inputs: Dict[str, Any] = {}

            for name, param in sig.parameters.items():
                if name == "self":
                    continue

                ann = hints.get(name)

                if isinstance(ann, type) and issubclass(ann, Node):
                    if name not in self.child_nodes:
                        raise TypeError(
                            f"Workflow '{func.__name__}' expects child node '{name}', "
                            f"but it was not provided to {self.__class__.__name__}."
                        )
                    wf_child_nodes[name] = self.child_nodes[name]
                    continue

                if name in self.inputs:
                    wf_inputs[name] = self.inputs[name]

            wf_node = WorkflowNode(
                func=func,
                child_nodes=wf_child_nodes,
                inputs=wf_inputs,
                workflow_kind=self._workflow_kind,
                name=f"{self.__class__.__name__}.{func.__name__}",
            )

            setattr(self, attr_name, wf_node)

    def dag(self):
        """Return a Graphviz DAG visualization of this module."""
        return visualize_module_dag(self)
    
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
            
    
class AbstractModule(ModuleNode, ABC):
    """
    Base class for modules that declare workflow interfaces via @abstractwf.

    ABCMeta will prevent instantiation until all abstract workflows are implemented
    by a concrete subclass.
    """
    pass