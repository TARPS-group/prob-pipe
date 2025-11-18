from __future__ import annotations
from typing import Callable, Dict, Any, Optional, get_type_hints
import inspect

from prefect import task, flow

from node import Node
from distributions import Distribution, EmpiricalDistribution
from multivariate import Multivariate

# Reuse the same notions as in Module
_DISTR_BASE = (Distribution, Multivariate)
_DISTR_INST = (Distribution, Multivariate, EmpiricalDistribution)

class WorkflowFunctionNode(Node):
    """
    Node that wraps a single 'workflow function' (dist-in, dist-out, etc.)
    and provides automatic distribution conversion based on type hints.

    - Stateless by design: no internal state, just function + conversion rules.
    - Dependencies are other Nodes in the DAG (their outputs can be used as inputs).
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        dependencies: Dict[str, Node],
        as_task: bool = True,
        conversion_num_samples: int = 1024,
        conversion_by_KDE: bool = False,
        conversion_fit_kwargs: Optional[dict] = None,
    ):
        super().__init__(name=name or func.__name__, dependencies=dependencies)

        self.func = func
        self._sig = inspect.signature(func)
        self._hints = get_type_hints(func)
        self.as_task = as_task

        # Conversion parameters (same semantics as Module)
        self._conv_num_samples = conversion_num_samples
        self._conv_by_kde = conversion_by_KDE
        self._conv_fit_kwargs = conversion_fit_kwargs or {}

        self._prefect_object = (
            task(func=self._execute)
            if as_task
            else flow(validate_parameters=False)(self._execute)
        )

    def _execute(self, **kwargs):
        """Actual Python execution: dependency injection + type conversion + call."""
        bound = self.sig.bind_partial(**kwargs)
        bound.apply_defaults()
        arguments = bound.arguments

        # inject dependencies as node results 
        # dependencies have already produced their outputs via DAG execution layer
        for dep_name, dep_node in self.dependencies.items():
            if dep_name in self.sig.parameters:
                arguments[dep_name] = dep_node.last_result  # DAG runner will set this

        # Automatic distribution conversion 
        for pname, val in arguments.items():
            expected = self.hints.get(pname)

            if expected is None:
                continue

            if self._is_distribution_type(expected):
                arguments[pname] = self._convert_distribution(pname, val, expected)

            else:
                # plain type check (int, float, NDArray…)
                if isinstance(expected, type) and not isinstance(val, expected):
                    raise TypeError(
                        f"Argument '{pname}' must be {expected.__name__}, got {type(val).__name__}"
                    )

        # Call the user function
        result = self.func(**arguments)

        # Optionally convert output here as well (future extension)
        return result

    def _is_distribution_type(self, tp):
        return (
            isinstance(tp, type)
            and (tp in _DISTR_BASE or issubclass(tp, _DISTR_BASE))
        )

    def _convert_distribution(self, name, value, expected):
        """Convert between distribution types using .from_distribution()"""
        if isinstance(value, expected):
            return value

        # expected is a parametric distribution subclass => convert
        if expected not in _DISTR_BASE and issubclass(expected, _DISTR_BASE):
            if not isinstance(value, _DISTR_INST):
                raise TypeError(
                    f"Argument '{name}' expected a distribution, got {type(value).__name__}"
                )

            return expected.from_distribution(
                value,
                num_samples=self.num_samples,
                conversion_by_KDE=self.use_kde,
                **self.fit_kwargs,
            )

        # base class distribution => accept raw
        if not isinstance(value, _DISTR_INST):
            raise TypeError(
                f"Argument '{name}' expected distribution-like; got {type(value).__name__}"
            )

        return value


    def compute(self, **kwargs):
        """
        Delegates to Prefect layer.
        The DAG executor will call node.compute() and store node.last_result.
        """
        result = self._prefect_object(**kwargs)
        self.last_result = result
        return result
    

    

