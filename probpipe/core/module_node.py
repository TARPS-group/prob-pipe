import inspect
from typing import Any, Dict, get_type_hints

from probpipe.core.node import Node
from probpipe.core.workflow_node import WorkflowNode
from probpipe.viz.dag import visualize_module_dag

__all__ = ["ModuleNode"]


class ModuleNode(Node):
    """
    Container for workflow nodes with shared inputs and child nodes.

    New user-facing API:
        MyModule(data=data_node, horizon=30, alpha=0.1)

    Internally:
        - kwargs whose values are Node instances become child_nodes
        - everything else becomes inputs
    """

    def __init__(
        self,
        *,
        prefect_kind: str | None = None,
        # Backward-compatible optional inputs (can remove later)
        child_nodes: Dict[str, Node] | None = None,
        inputs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self._prefect_kind = prefect_kind

        # Backward compat merge: explicit dicts override kwargs if overlapping
        child_nodes = dict(child_nodes or {})
        inputs = dict(inputs or {})

        # Auto-split kwargs into child_nodes vs inputs
        for k, v in kwargs.items():
            if isinstance(v, Node):
                child_nodes[k] = v
            else:
                inputs[k] = v

        super().__init__(
            child_nodes=child_nodes,
            inputs=inputs,
        )

        self._build_workflows()

    def _build_workflows(self):
        """
        Replace @wf methods with WorkflowNode instances.

        For each @wf function:
          - params annotated as Node subclasses are sourced from self.child_nodes
          - other params are sourced from self.inputs if provided
          - anything not found remains call-time (handled by WorkflowNode._bind_inputs)
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not callable(attr) or not getattr(attr, "_is_workflow", False):
                continue

            func = attr
            sig = inspect.signature(func)
            hints = get_type_hints(func)

            wf_child_nodes: Dict[str, Node] = {}
            wf_inputs: Dict[str, Any] = {}

            for name, param in sig.parameters.items():
                if name == "self":
                    continue

                ann = hints.get(name)

                # Node parameters are injected from module wiring
                if isinstance(ann, type) and issubclass(ann, Node):
                    if name not in self.child_nodes:
                        raise TypeError(
                            f"Workflow '{func.__name__}' expects child node '{name}', "
                            f"but it was not provided to {self.__class__.__name__}."
                        )
                    wf_child_nodes[name] = self.child_nodes[name]
                    continue

                # Non-node parameters can be bound at module construction if provided
                if name in self.inputs:
                    wf_inputs[name] = self.inputs[name]

            wf_node = WorkflowNode(
                func=func,
                child_nodes=wf_child_nodes,
                inputs=wf_inputs,
                prefect_kind=self._prefect_kind,
                name=f"{self.__class__.__name__}.{func.__name__}",
            )

            setattr(self, attr_name, wf_node)

    def dag(self):
        """Return a Graphviz DAG visualization of this module."""
        return visualize_module_dag(self)