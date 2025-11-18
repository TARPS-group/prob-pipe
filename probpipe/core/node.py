from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any


class Node(ABC):
    """
    Abstract base class representing a node in the computation DAG.

    Each node:
        - has a name
        - has dependencies (other Node instances)
        - can be evaluated (run)
    """

    def __init__(self, name: str, **dependencies: "Node"):
        self.name = name or self.__class__.__name__
        self._dependencies: Dict[str, Node] = dependencies  # key = param name

    @property
    def dependencies(self) -> Dict[str, "Node"]:
        """Return dependency mapping (name → Node)."""
        return self._dependencies

    @abstractmethod
    def compute(self, **inputs) -> Any:
        """
        Core computation performed by this node.
        Subclasses implement this.
        """
        pass

    def __call__(self, **inputs) -> Any:
        """
        Evaluate node by:
            - recursively evaluating its dependencies
            - calling its compute() with both dependency outputs + runtime inputs
        """
        # 1. compute all dependency outputs
        dep_outputs = {
            name: node()
            for name, node in self._dependencies.items()
        }

        # 2. merge with explicit user inputs
        all_inputs = {**dep_outputs, **inputs}

        # 3. compute this node's output
        return self.compute(**all_inputs)

    # Debug / Visualization helper
    def describe(self, indent=0) -> str:
        """Return an ASCII tree representation of this node."""
        pad = " " * indent
        s = f"{pad}- Node({self.name})\n"
        for dep in self._dependencies.values():
            s += dep.describe(indent + 2)
        return s
    




