from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, Mapping, Union, get_type_hints
import inspect
from types import MappingProxyType

from prefect import task, flow

# THIS WILL BE CHANGED; just for implementing the template of conversion logic
from probpipe import Distribution, EmpiricalDistribution, Multivariate
DISTRIBUTION_TYPES = (Distribution, EmpiricalDistribution, Multivariate)

__all__ = ["InputFrozenError", "wf", "FreezableDict", "Node", ]

class InputFrozenError(Exception):
    pass


def abstractwf(func: Callable):
    """
    Marks a method as:
      - a workflow interface (visible to ModuleNode)
      - abstract (enforced by ABCMeta)

    This allows abstract modules to declare workflow-shaped interfaces
    without providing implementations.
    """
    func._is_workflow = True

    # abstractmethod sets __isabstractmethod__ = True
    return abstractmethod(func)

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
    