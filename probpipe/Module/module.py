from typing import Callable, Dict, Optional, Any
import inspect


class Module:
    def __init__(self, **dependencies):
        self.dependencies: Dict[str, 'Module'] = {}
        self.inputs: Dict[str, Dict[str, Any]] = {}
        self._run_funcs: Dict[str, Callable] = {} 

        # Register passed dependencies
        for name, dep in dependencies.items():
            self.add_dependency(name, dep)

    def add_dependency(self, name: str, dep: 'Module'):
        if name in self.dependencies:
            raise ValueError(f"Dependency '{name}' already registered")
        self.dependencies[name] = dep

    def show_dependencies(self):
        print("Dependencies:")
        for name, dep in self.dependencies.items():
            print(f" - {name}: {dep}")

    def instantiate(self, **input_defaults):
        """
        Set or override default input values for this module.
        Updates self.inputs metadata default values accordingly.
        """
        for key, value in input_defaults.items():
            if key not in self.inputs:
                # Add input metadata for new inputs (type unknown)
                self.inputs[key] = {'default': value}
            else:
                self.inputs[key]['default'] = value

    def show_inputs(self):
        print("Inputs:")
        for name, meta in self.inputs.items():
            print(f" - {name}: default={meta.get('default')}")

    def show_run_functions(self):
        print("Run functions:")
        for name in self._run_funcs:
            print(f" - {name}")

    def decorate_run_fun(self, func: Callable = None, *, name: Optional[str] = None):
        """
        Decorator to register a user run function.
        Extracts inputs from signature, but doesn't enforce types by default.
        """
        def decorator(f: Callable):
            run_name = name or f.__name__
            if run_name in self._run_funcs:
                raise RuntimeError(f"Run function '{run_name}' already registered")

            # Extract inputs excluding dependencies
            sig = inspect.signature(f)
            for pname, param in sig.parameters.items():
                if pname not in self.dependencies and pname not in self.inputs:
                    self.inputs[pname] = {
                        'default': param.default if param.default is not inspect.Parameter.empty else None
                    }
                elif pname in self.inputs:
                    # Optional: ensure consistent default values, types etc. here
                    pass

            self._run_funcs[run_name] = f
            return f

        if func is None:
            return decorator
        else:
            return decorator(func)

    def run(self, name: Optional[str] = None, **inputs):
        """
        Run the named registered function.
        Resolves parameters from dependencies, inputs argument, and defaults.
        """
        if not self._run_funcs:
            raise RuntimeError("No run functions registered")

        if name is None:
            if len(self._run_funcs) == 1:
                name = next(iter(self._run_funcs))
            else:
                raise ValueError("Multiple run functions registered; specify which to run")

        run_func = self._run_funcs[name]
        sig = inspect.signature(run_func)
        call_kwargs = {}

        for param in sig.parameters.values():
            pname = param.name
            if pname in self.dependencies:
                call_kwargs[pname] = self.dependencies[pname]
            elif pname in inputs:
                call_kwargs[pname] = inputs[pname]
            elif pname in self.inputs and self.inputs[pname].get('default') is not None:
                call_kwargs[pname] = self.inputs[pname]['default']
            elif param.default is not inspect.Parameter.empty:
                call_kwargs[pname] = param.default
            else:
                raise ValueError(f"Missing required argument '{pname}' for run function '{name}'")

        return run_func(**call_kwargs)

    def __repr__(self):
        return (f"<Module deps={list(self.dependencies.keys())} "
                f"inputs={list(self.inputs.keys())} "
                f"run_funcs={list(self._run_funcs.keys())}>")

    def __str__(self):
        deps = ", ".join(self.dependencies.keys()) or "None"
        inputs = ", ".join(self.inputs.keys()) or "None"
        run_funcs = ", ".join(self._run_funcs.keys()) or "None"
        return (f"Module:\n"
                f"  Dependencies: {deps}\n"
                f"  Inputs: {inputs}\n"
                f"  Run Functions: {run_funcs}")