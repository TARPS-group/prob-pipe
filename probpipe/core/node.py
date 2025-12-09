from abc import ABC, abstractmethod
from types import MappingProxyType
from functools import wraps
from typing import Any, Callable, Dict, Set, Optional
import inspect
import numpy as np

class DependencyFrozenError(Exception):
    pass

class WorkflowFunction:
    def __init__(self, func: Callable, node: 'Node', **dependencies):
        wraps(func)(self)
        self._func = func
        self._node = node
        self._dependencies = dependencies

    def __call__(self, *args, **kwargs):
        # Get signature of the wrapped function
        sig = inspect.signature(self._func)

        # Bind the provided arguments to get which params are already provided
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        # For each parameter in signature, if not provided, try to fill from dependencies
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name not in bound_args.arguments:
                if param_name in self._dependencies:
                    bound_args.arguments[param_name] = self._dependencies[param_name]

        # Call the bound function with all arguments
        bound_func = self._func.__get__(self._node, type(self._node))
        return bound_func(*bound_args.args, **bound_args.kwargs)

def wf(func: Callable):
    """Decorator to mark a function as a workflow node function."""
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
            raise DependencyFrozenError("Cannot modify dependencies after frozen.")
        super().__setitem__(key, value)

    def __delitem__(self, key):
        if self._frozen:
            raise DependencyFrozenError("Cannot delete dependencies after frozen.")
        super().__delitem__(key)

class Node(ABC):
    def __init__(self, **dependencies):
        required = self.required_dependencies()
        missing = required - dependencies.keys()
        if missing:
            raise ValueError(f"Missing dependencies {missing} for {self.__class__.__name__}")

        self._dependencies = FreezableDict(**dependencies)
        self._dependencies.freeze()
        self._setup_workflow_functions()

    def _setup_workflow_functions(self):
        # Bind all methods decorated with @wf as WorkflowFunction instances
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and getattr(attr, "_is_workflow", False):
                wf_func = WorkflowFunction(attr, node=self, **self._dependencies)
                setattr(self, attr_name, wf_func)

    def required_dependencies(self) -> Set[str]:
        # Default to auto-infer dependencies from wf param signatures
        return self._infer_required_dependencies()

    def _infer_required_dependencies(self) -> Set[str]:
        reqs = set()
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and getattr(attr, "_is_workflow", False):
                sig = inspect.signature(attr)
                for pname, param in sig.parameters.items():
                    if pname == "self":
                        continue
                    if param.default is param.empty:
                        reqs.add(pname)
        return reqs



    





