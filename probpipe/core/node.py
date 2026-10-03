"""Shared graph-node input storage for experimental workflow containers."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

__all__ = ["InputFrozenError", "Node"]


class InputFrozenError(Exception):
    """The error for an attempt to change the frozen inputs of a node."""


class Node(ABC):  # noqa: B024
    """
    Base unit of the ProbPipe computational dependency graph.

    Keyword arguments are automatically split by type: values that are
    ``Node`` instances become *child nodes* (dependencies on other DAG
    units), and everything else becomes *inputs* (data, configuration,
    hyperparameters).  Both collections are frozen after construction.
    """

    def __init__(self, **kwargs: Any):
        child_nodes: dict[str, Node] = {}
        inputs: dict[str, Any] = {}

        for k, v in kwargs.items():
            if isinstance(v, Node):
                child_nodes[k] = v
            else:
                inputs[k] = v

        # Freeze internal state (read-only). Written through
        # ``object.__setattr__`` so this also serves an immutable subclass, whose
        # guard refuses assignment even from its own constructor.
        object.__setattr__(self, "_child_nodes", MappingProxyType(child_nodes))
        object.__setattr__(self, "_inputs", MappingProxyType(inputs))

    @property
    def child_nodes(self) -> Mapping[str, Node]:
        return self._child_nodes

    @property
    def inputs(self) -> Mapping[str, Any]:
        return self._inputs
