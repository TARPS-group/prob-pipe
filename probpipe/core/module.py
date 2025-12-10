from typing import Callable, Optional, Dict, Any, get_type_hints, Type, ClassVar, FrozenSet, Iterable, Set
from types import MappingProxyType
import functools
from dataclasses import dataclass
import inspect

from prefect import flow

from .node import Node

__all__ = [
    "Module",
]

class Module(Node):
    """
    Group or composite of Node instances, supports DAG execution of workflow nodes.
    """
    def __init__(self, backend="python", **dependencies):
        self._backend = backend  # "python" or "prefect"

        # Creating child nodes
        self._nodes: Dict[str, Node] = self._define_nodes(**dependencies)

        # Storing module-level dependencies (names)
        self._module_dependencies: Set[str] = set(dependencies.keys())

        # Let the module act as its own module_ref
        self._module_ref = self

        # Attaching module_ref to child nodes
        for node in self._nodes.values():
            node._module_ref = self

        # Aggregating required deps from child nodes
        all_required = set()
        for node in self._nodes.values():
            all_required |= node.required_dependencies()

        missing = all_required - self._module_dependencies
        if missing:
            raise ValueError(
                f"Missing top-level dependencies {missing} "
                f"for {self.__class__.__name__}"
            )

        # Initializing as a Node (will validate and freeze dependencies)
        super().__init__(**dependencies)

        # Building dependency DAG
        self._build_dependency_graph()


    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        is_workflow = callable(attr) and getattr(attr, "_is_workflow", False)

        # If backend is python then no change
        if not is_workflow or self._backend == "python":
            return attr

        # Otherwise backend is "prefect"
        def wrapped_flow(*args, **kwargs):
            @flow(name=f"{self.__class__.__name__}.{name}")
            def prefect_flow():
                return attr(*args, **kwargs)

            return prefect_flow()

        return wrapped_flow


    # just for debugging
    def _build_dependency_graph(self):
        """
        Build a simple dependency DAG:
          - each child node mapped to its required dependencies
          - each module-level workflow function mapped to module deps
        """
        self._dep_graph: Dict[str, list] = {}

        # Child nodes
        for name, node in self._nodes.items():
            self._dep_graph[name] = list(node.required_dependencies())

        # Module-level workflow functions (e.g. fit, predict)
        for attr_name, attr in self.__class__.__dict__.items():
            if callable(attr) and getattr(attr, "_is_workflow", False):
                wf_name = f"{self.__class__.__name__}.{attr_name}"
                self._dep_graph[wf_name] = list(self._module_dependencies)

    # just for debugging
    def print_dependency_graph(self):
        print("=== Dependency DAG ===")
        for node, deps in self._dep_graph.items():
            print(f"{node}: {deps}")

    @classmethod
    def _define_nodes(cls, **dependencies) -> Dict[str, Node]:
        """
        Return dict of node instances to be included in this module.
        Override in subclasses to create nodes.
        """
        return {}

    def required_dependencies(self) -> Set[str]:
        # Aggregate dependencies from child nodes
        deps = set()
        for node in self._nodes.values():
            deps |= node.required_dependencies()
        return deps

    def run_node(self, node_name, method_name, *args, **kwargs):
        node = self.nodes[node_name]
        method = getattr(node, method_name)
        return method(*args, **kwargs)

    # DAG execution methods would go here

    