from .module import Module, InputSpec
from types import SimpleNamespace
from contextlib import contextmanager
from typing import Dict, Callable, Optional, Iterable, Set, Tuple

class WorkFlow(Module):
    def __init__(self):
        super().__init__()
        self._collected_funcs: Dict[str, Tuple[Callable, bool]] = {}
        self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._initialized:
            for run_name, (func, as_task) in self._collected_funcs.items():
                pf = super().run_func(func, name=run_name, as_task=as_task)
                setattr(self, run_name, pf)
            self._initialized = True
        return False

    def run_func(self, f: Callable = None, *, as_task: bool = True):
        def decorator(func: Callable):
            name = func.__name__
            if name in self._collected_funcs:
                raise RuntimeError(f"Duplicate run function {name}")
            self._collected_funcs[name] = (func, as_task)
            return func
        if f is None:
            return decorator
        else:
            return decorator(f)

