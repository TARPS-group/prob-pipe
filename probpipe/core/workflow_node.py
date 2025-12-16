from abc import ABC
from functools import wraps
from typing import Any, Callable, Dict, Mapping, Union, get_type_hints
import inspect
from types import MappingProxyType
from probpipe import Node

from prefect import task, flow

# THIS WILL BE CHANGED; just for implementing the template of conversion logic
from probpipe import Distribution, EmpiricalDistribution, Multivariate
DISTRIBUTION_TYPES = (Distribution, EmpiricalDistribution, Multivariate)

__all__ = ["WorkflowNode", ]



class WorkflowNode(Node):
    """
    A single executable DAG node wrapping exactly one function.
    """

    def __init__(
        self,
        *,
        func: Callable,
        child_nodes: Dict[str, Node],
        inputs: Dict[str, Any],
        prefect_kind: str | None = None,   # "task" or "flow" or None
        name: str | None = None,
    ):
        self._func = func
        self._sig = inspect.signature(func)
        self._hints = get_type_hints(func)
        self._prefect_kind = prefect_kind
        self._name = name or func.__name__

        super().__init__(
            child_nodes=child_nodes,
            inputs=inputs,
        )

        self._validate_declared_inputs()

    # validation

    def _validate_declared_inputs(self):
        """
        Ensure all declared child_nodes + inputs correspond
        to parameters of the function.
        """
        param_names = {p for p in self._sig.parameters if p != "self"}

        declared = set(self.child_nodes).union(set(self.inputs))
        unknown = declared - param_names

        if unknown:
            raise TypeError(f"Workflow '{self._name}' received unknown inputs {unknown}")

    # execution

    def __call__(self, **call_inputs):
        bound = self._bind_inputs(call_inputs)
        bound = self._convert_distributions(bound)

        if self._prefect_kind == "task":
            return self._run_as_task(bound)
        if self._prefect_kind == "flow":
            return self._run_as_flow(bound)

        return self._call_python(bound)

    def _bind_inputs(self, call_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge construction-time inputs with call-time inputs.
        """
        values = {}

        # child nodes (always present)
        values.update(self.child_nodes)

        # construction-time inputs
        values.update(self.inputs)

        # call-time inputs
        for k, v in call_inputs.items():
            if k in self.child_nodes:
                raise TypeError(f"Child node '{k}' cannot be passed at call time")
            values[k] = v

        # validate missing required params
        for name, param in self._sig.parameters.items():
            if name == "self":
                continue
            if param.default is param.empty and name not in values:
                raise TypeError(f"Missing required input '{name}' for workflow '{self._name}'")

        return values

    def _convert_distributions(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert distributions based on type hints.
        """
        out = dict(values)

        for name, value in values.items():
            expected = self._hints.get(name)
            if expected is None:
                continue

            if (isinstance(expected, type) and issubclass(expected, Distribution) 
                and isinstance(value, DISTRIBUTION_TYPES) 
                and not isinstance(value, expected)
            ):
                out[name] = expected.from_distribution(value)

        return out

    def _call_python(self, values: Dict[str, Any]):
        return self._func(**values)

    def _run_as_task(self, values):
        @task(name=self._name)
        def wrapped():
            return self._call_python(values)

        return wrapped()

    def _run_as_flow(self, values):
        @flow(name=self._name)
        def wrapped():
            return self._call_python(values)

        return wrapped()
