from typing import Callable, Optional, Dict, Any, get_type_hints, Type, ClassVar, FrozenSet, Iterable, Set
from types import MappingProxyType
import functools
from dataclasses import dataclass
import inspect

from prefect import flow, task

from .node import Node

__all__ = [
    "Module",
]

class Module(Node):
    """
    Group or composite of Node instances, supports DAG execution of workflow nodes.
    """
    def __init__(self, **dependencies):
        # Register child nodes: subclasses should override _define_nodes()
        self._nodes: Dict[str, Node] = self._define_nodes(**dependencies)

        # Collect dependencies from all child nodes
        all_dependencies = set()
        for node in self._nodes.values():
            all_dependencies |= node.required_dependencies()

        # Validate module-level dependencies completeness
        missing = all_dependencies - dependencies.keys()
        if missing:
            raise ValueError(f"Missing top-level dependencies {missing} for {self.__class__.__name__}")

        # Initialize Node with full dependencies
        super().__init__(**dependencies)

        # Optionally build DAG here based on node dependencies

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

    