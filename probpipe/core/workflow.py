from .module import Module, InputSpec
from types import SimpleNamespace, MappingProxyType
from contextlib import contextmanager
import inspect
from typing import Callable, get_type_hints

# Workflow class for users to create workflows using module classes

# TODO:
# - ADD module class constructor 
# - Add registering dependencies (other modules)
# - Implement run function to execute the workflow -> Use Module class run function
# - Logging and error handling 
# - Support for prefect tasks within the workflow


from probpipe import Module, Distribution  # Adjust imports to your structure

def is_dependency_type(tp):
    """
    Returns True if tp should be treated as a dependency module.
    By default, returns False for Distribution or basic types.
    """
    try:
        return isinstance(tp, type) and not issubclass(tp, Distribution)
    except Exception:
        return False

def wrap_as_module(obj, name, expected_type):
    """
    Wraps any dependency object as a Module, unless already a Module.
    All attribute accesses are delegated to the underlying object.
    """
    if isinstance(obj, Module):
        return obj
    # Simple wrapper
    class _AutoWrappedModule(Module):
        DEPENDENCIES = MappingProxyType({})
        def __init__(self, dep):
            super().__init__()
            self._dep = dep
        def __getattr__(self, attr):
            # Delegate to the underlying dependency
            return getattr(self._dep, attr)
        def __repr__(self):
            return f"<WrappedModule {type(self._dep).__name__}>"
    _AutoWrappedModule.__name__ = f"Wrapped{name.capitalize()}Module"
    return _AutoWrappedModule(obj)

class Workflow:
    """
    Workflow class helps user to automatically builds a pipleline module class
    """
    @staticmethod
    def create(workflow_func: Callable, as_task: bool = True):
        sig = inspect.signature(workflow_func)
        type_hints = get_type_hints(workflow_func)
        param_names = [name for name in sig.parameters if name != "self"]

        dependencies = {
            name: type_hints[name] for name in param_names
            if name in type_hints and is_dependency_type(type_hints[name])
        }
        DEPENDENCIES = MappingProxyType(dependencies)

        class _WorkflowModule(Module):
            def __init__(self, **deps):
                wrapped_deps = {
                    name: wrap_as_module(dep, name, DEPENDENCIES[name])
                    for name, dep in deps.items()
                }
                super().__init__(**wrapped_deps)
                self.run_func(workflow_func, name="run", as_task=True)
                self._sig = inspect.signature(workflow_func)

            def __call__(self, *args, **kwargs):
                bound = self._sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                return self.run(**bound.arguments)

        _WorkflowModule.DEPENDENCIES = DEPENDENCIES
        _WorkflowModule.__name__ = f"{workflow_func.__name__}Module"
        return _WorkflowModule


