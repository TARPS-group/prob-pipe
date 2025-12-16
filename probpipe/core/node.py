from abc import ABC
from functools import wraps
from typing import Any, Callable, Dict, Mapping, Union, get_type_hints
import inspect
from types import MappingProxyType

from prefect import task, flow

# THIS WILL BE CHANGED; just for implementing the template of conversion logic
from probpipe import Distribution, EmpiricalDistribution, Multivariate
DISTRIBUTION_TYPES = (Distribution, EmpiricalDistribution, Multivariate)

__all__ = ["InputFrozenError", "wf", "FreezableDict", "Node", "WorkflowNode", ]

class InputFrozenError(Exception):
    pass


def wf(func: Callable):
    func._is_workflow = True
    return func


class FreezableDict(dict):
    """Dict that raises if updated after freezing."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen = False

    def freeze(self):
        self._frozen = True

    def __setitem__(self, key, value):
        if self._frozen:
            raise InputFrozenError("Cannot modify inputs after frozen.")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        if self._frozen:
            raise InputFrozenError("Cannot delete inputs after frozen.")
        super().__delitem__(key)


class Node(ABC):
    """
    Base DAG unit.

    A Node:
    - has child nodes (other Nodes it is allowed to call)
    - has inputs (non-Node values)
    - knows nothing about execution, prefect, or workflow functions
    """

    def __init__(
        self,
        *,
        child_nodes: Dict[str, "Node"] | None = None,
        inputs: Dict[str, Any] | None = None,
    ):
        child_nodes = child_nodes or {}
        inputs = inputs or {}

        # validates child nodes 
        for name, node in child_nodes.items():
            if not isinstance(node, Node):
                raise TypeError(
                    f"Child node '{name}' must be a Node, got {type(node)}"
                )

        # validates inputs 
        for name, value in inputs.items():
            if isinstance(value, Node):
                raise TypeError(
                    f"Input '{name}' is a Node; Nodes must be declared as child_nodes"
                )

        # freezing internal state
        self._child_nodes = MappingProxyType(dict(child_nodes))
        self._inputs = MappingProxyType(dict(inputs))

    # public read-only views

    @property
    def child_nodes(self) -> Mapping[str, "Node"]:
        return self._child_nodes

    @property
    def inputs(self) -> Mapping[str, Any]:
        return self._inputs
    