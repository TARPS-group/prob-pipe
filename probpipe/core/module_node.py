from __future__ import annotations
from typing import Callable, Dict, Any, Optional

from node import Node
from workflow_node import WorkflowFunctionNode


class ModuleNode(Node):
    """
    A ModuleNode is a stateful node that:
      - contains multiple WorkflowFunctionNodes
      - exposes shared state to all functions
      - determines its dependencies as the union of all child workflow funcs
      - passes itself as the first argument to every workflow function

    Conceptually: this is the Module in the new architecture.
    """

    def __init__(
        self,
        *, 
        # Node dependencies (other modules or workflow nodes)
        dependencies: Optional[Dict[str, Node]] = None,
        # Shared module state (very flexible for now)
        state: Optional[Dict[str, Any]] = None
    ):
        super().__init__(dependencies=dependencies)

        # Shared mutable state
        self.state: Dict[str, Any] = state or {}

        # Registered workflow functions belonging to this module
        self._workflow_funcs: Dict[str, WorkflowFunctionNode] = {}

        # recomputing module dependencies as union of children workflow-func dependencies
        self._recompute_dependencies()


    # Workflow Function Registration

    def add_workflow_function(
        self,
        name: str,
        func: Callable,
        local_conversion_rules: Optional[Dict] = None,
        global_conversion_rules: Optional[Dict] = None,
    ) -> None:
        """
        Registers a workflow function as a WorkflowFunctionNode.

        The function signature must follow:
            def f(module, ...)  # module = this ModuleNode (self)
        """

        if not callable(func):
            raise TypeError("Workflow function must be callable")

        if name in self._workflow_funcs:
            raise RuntimeError(f"Workflow function '{name}' already exists in this module")

        wf = WorkflowFunctionNode(
            func=func,
            parent_module=self,
            local_conversion_rules=local_conversion_rules,
            global_conversion_rules=global_conversion_rules,
        )

        self._workflow_funcs[name] = wf

        # Updating module dependencies based on all workflow functions
        self._recompute_dependencies()


    # Running Workflow Functions

    def run(self, name: str, **kwargs):
        """
        Execute a registered workflow function by name.
        self is automatically passed as the first argument (f(module, ...)).

        Returns whatever the workflow function returns (eg, Distribution).
        """
        if name not in self._workflow_funcs:
            raise RuntimeError(f"No workflow function '{name}' registered")

        wf = self._workflow_funcs[name]

        # Calling WorkflowFunctionNode.evaluate()
        return wf.evaluate(self, **kwargs)


    # Internal Logic

    def _recompute_dependencies(self):
        """
        Combine dependencies from all workflow functions + module-level deps.
        """
        combined = {}

        # Including dependencies passed at module construction
        for k, v in self.dependencies.items():
            combined[k] = v

        # Including union of dependencies of all workflow functions
        for wf in self._workflow_funcs.values():
            for k, v in wf.dependencies.items():
                combined[k] = v

        self.dependencies = combined


    def __repr__(self):
        return f"<ModuleNode funcs={list(self._workflow_funcs.keys())} deps={list(self.dependencies.keys())}>"