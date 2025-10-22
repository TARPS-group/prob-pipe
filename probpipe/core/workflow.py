from .module import Module, InputSpec
from types import SimpleNamespace
from contextlib import contextmanager
from typing import Dict, Callable, Optional, Iterable, Set, Tuple

class WorkFlow(Module):
    """
    Context-managed module for batch registration of run functions.

    Functions decorated with `@workflow.run_func(...)` inside a 'with WorkFlow():'
    block are collected first, and registered as Prefect tasks/flows upon exit.

    Example
    -------
    with WorkFlow() as wf:
        @wf.run_func(as_task=True)
        def step1(x: int) -> int:
            return x + 1
        @wf.run_func(as_task=False)
        def step2(y: int) -> int:
            return y * 2
       
    # Both step1 and step2 are now registered as run functions.
    """
    
    def __init__(self):
        """Initialize an empty workflow with deferred run function registry."""
        
        super().__init__()
        self._collected_funcs: Dict[str, Tuple[Callable, bool]] = {}
        self._initialized = False

    def __enter__(self):
        """Enter workflow context; returns self for use in 'with' statements."""
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        On exit, register all collected functions as Prefect tasks/flows.

        Parameters
        ----------
        exc_type, exc_value, traceback
            Standard context manager exception arguments.

        Returns
        -------
        bool
            Always returns False to propagate exceptions normally.
        """
        
        if not self._initialized:
            for run_name, (func, as_task) in self._collected_funcs.items():
                pf = super().run_func(func, name=run_name, as_task=as_task)
                setattr(self, run_name, pf)
            self._initialized = True
        return False

    def run_func(self, f: Callable = None, *, as_task: bool = True):
                """
        Decorator to collect a function for later registration.

        Parameters
        ----------
        f : callable, optional
            Function to decorate. If None, returns a decorator.
        as_task : bool, default=True
            If True, register the function as a Prefect task;
            otherwise as a Prefect flow.

        Returns
        -------
        callable
            The decorated function or decorator itself.
        """
        
        def decorator(func: Callable):
            """Record function and registration mode (task or flow)."""
            
            name = func.__name__
            if name in self._collected_funcs:
                raise RuntimeError(f"Duplicate run function {name}")
            self._collected_funcs[name] = (func, as_task)
            return func
        if f is None:
            return decorator
        else:
            return decorator(f)

