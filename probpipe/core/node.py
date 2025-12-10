from abc import ABC, abstractmethod
from types import MappingProxyType
from functools import wraps
from typing import Any, Callable, Dict, Set, Union, get_type_hints
import inspect
from prefect import task
import inspect


#THIS WILL BE CHANGED; just for implementing the template of conversion logic
from probpipe import Distribution, EmpiricalDistribution, Multivariate
DISTRIBUTION_TYPES = (Distribution, EmpiricalDistribution, Multivariate)


class DependencyFrozenError(Exception):
    pass

class WorkflowFunction:
    def __init__(self, func: Callable, node: 'Node', **dependencies):
        wraps(func)(self)
        self._func = func
        self._node = node
        self._dependencies = dependencies

        # OPTINAL parameters for distribution conversion
        #self._num_samples = 1024
        #self._use_kde = False
        #self._fit_kwargs = {}

    def __call__(self, *args, **kwargs):
        backend = getattr(self._node._module_ref, "_backend", "python")

        if backend == "python":
            return self._call_python(*args, **kwargs)

        if backend == "prefect":
            return self._call_prefect(*args, **kwargs)

        raise ValueError(f"Unknown backend: {backend}")
    

    def _call_python(self, *args, **kwargs):
        
        # Signature and argument binding
        sig = inspect.signature(self._func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        # Dependency Injection
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            if pname not in bound.arguments:
                if pname in self._dependencies:
                    bound.arguments[pname] = self._dependencies[pname]

        # Type Hints + Distribution Conversion
        hints = get_type_hints(self._func)

        for pname, value in bound.arguments.items():
            if pname == "self":
                continue

            expected = hints.get(pname)
            if expected is None:
                continue

            # Example: expected is a Distribution subclass
            if isinstance(expected, type) and issubclass(expected, Distribution):
                if isinstance(value, DISTRIBUTION_TYPES) and not isinstance(value, expected):
                    bound.arguments[pname] = expected.from_distribution(value)

        # Call original method
        bound_method = self._func.__get__(self._node, type(self._node))
        return bound_method(*bound.args, **bound.kwargs)
    
    def _call_prefect(self, *args, **kwargs):
        @task(name=f"{self._node.__class__.__name__}.{self._func.__name__}")
        def wrapped():
            return self._call_python(*args, **kwargs)

        return wrapped()





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

        # NEW: type checking for dependencies
        self._validate_dependency_types(dependencies)

        self._dependencies = FreezableDict(**dependencies)
        self._dependencies.freeze()
        self._setup_workflow_functions()

        self._dependency_names = list(dependencies.keys())

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
    

    def _expected_dependency_types(self) -> Dict[str, type]:
        """
        Aggregate expected types for dependencies from @wf method type hints.
        """
        expected_types = {}

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and getattr(attr, "_is_workflow", False):
                hints = get_type_hints(attr)
                for name, tp in hints.items():
                    if name == "self":
                        continue
                    # Only record if param has no default → dependency
                    sig = inspect.signature(attr)
                    param = sig.parameters[name]
                    if param.default is param.empty:
                        expected_types[name] = tp

        return expected_types
    

    def _validate_dependency_types(self, dependencies: Dict[str, Any]):
        expected = self._expected_dependency_types()

        for dep_name, dep_value in dependencies.items():
            if dep_name not in expected:
                # If dependency has no type annotation: skip
                continue

            expected_type = expected[dep_name]

            # Supportign typing.Union[X, None], Optionals, etc.
            origin = getattr(expected_type, "__origin__", None)

            if origin is not None and origin is Union:
                allowed = expected_type.__args__
                if not isinstance(dep_value, allowed):
                    raise TypeError(
                        f"Dependency '{dep_name}' expected type {expected_type}, "
                        f"got {type(dep_value)}"
                    )
            else:
                # Normal type check
                if not isinstance(dep_value, expected_type):
                    raise TypeError(
                        f"Dependency '{dep_name}' expected type {expected_type.__name__}, "
                        f"got {type(dep_value).__name__}"
                    )




