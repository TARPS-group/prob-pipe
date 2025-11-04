from .module import Module, InputSpec
from types import SimpleNamespace, MappingProxyType
from contextlib import contextmanager
from typing import Dict, Callable, Optional, Type, Any

# Workflow class for users to create workflows using module classes

# TODO:
# - ADD module class constructor 
# - Add registering dependencies (other modules)
# - Implement run function to execute the workflow -> Use Module class run function
# - Logging and error handling 
# - Support for prefect tasks within the workflow

class WorkflowBuilder:
    def __init__(self, name = "UserDefinedWorkflow"):
        self._class_name = name
        self._dependencies: Dict[str, Module] = {}
        self._run_func: Callable = None
        self._workflow_instance = None

    def register_module(self, name: str, module: Module) -> None:
        if name in self._dependencies:
            raise RuntimeError(f"Dependency '{name}' already registered")
        if not isinstance(module, Module):
            raise TypeError(f"Dependency '{name}' must be a Module instance")
        self._dependencies[name] = module

    def register_modules(self, modules: Dict[str, Module]) -> None:
        for name, module in modules.items():
            self.register_module(name, module)

    def register_run(self, f: Callable[..., Any]) -> Callable[..., Any]:
        if self._run_func is not None:
            raise RuntimeError("Run function already registered")
        self._run_func = f
        return f
    

    def build(self) -> Module:
        if self._run_func is None:
            raise RuntimeError("No run function registered")
        
        dep_types = {name: type(mod) for name, mod in self._dependencies.items()}
        cls_dict = {
            "DEPENDENCIES": MappingProxyType(dep_types)
        }

        workflow_cls = type(self._class_name, (Module,), cls_dict)

        def __init__(self, **deps):
            super(workflow_cls, self).__init__(**deps)
            self.run_func(self._run_func, name="run", as_task=True)

        workflow_cls.__init__ = __init__

        self._workflow_instance = workflow_cls(**self._dependencies)
        return self._workflow_instance

    def run_workflow(self, *args, **kwargs):
        if self._workflow_instance is None:
            raise RuntimeError("Workflow not built yet")
        return self._workflow_instance.run(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        
        self.build()
        return False

