from typing import Dict, Set
from prefect import flow
from probpipe.core.workflow_node import WorkflowNode, Node
import inspect
from typing import get_type_hints
from probpipe.viz.dag import visualize_module_dag


__all__ = ["ModuleNode"]


class ModuleNode(Node):
    """
    Container for workflow nodes with shared inputs and child nodes.
    """

    def __init__(
        self,
        *,
        child_nodes: Dict[str, Node],
        inputs: Dict[str, object],
        prefect_kind: str | None = None,
    ):
        self._prefect_kind = prefect_kind

        super().__init__(
            child_nodes=child_nodes,
            inputs=inputs,
        )

        self._build_workflows()

    def _build_workflows(self):
        """
        Replace @wf methods with WorkflowNode instances.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if not callable(attr) or not getattr(attr, "_is_workflow", False):
                continue

            func = attr

            sig = inspect.signature(func)
            hints = get_type_hints(func)

            child_nodes = {}
            inputs = {}

            for name, param in sig.parameters.items():
                if name == "self":
                    continue

                ann = hints.get(name)

                if isinstance(ann, type) and issubclass(ann, Node):
                    if name not in self.child_nodes:
                        raise TypeError(
                            f"Workflow '{func.__name__}' expects child node '{name}', "
                            f"but it was not provided to ModuleNode")
                    child_nodes[name] = self.child_nodes[name]
                else:
                    if name in self.inputs:
                        inputs[name] = self.inputs[name]

            wf_node = WorkflowNode(
                func=func,
                child_nodes=child_nodes,
                inputs=inputs,
                prefect_kind=self._prefect_kind,
                name=f"{self.__class__.__name__}.{func.__name__}",)

            setattr(self, attr_name, wf_node)

    def dag(self):
        """Return a Graphviz DAG visualization of this module."""
        return visualize_module_dag(self)