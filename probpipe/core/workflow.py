from .module import Module, InputSpec
from types import SimpleNamespace
from contextlib import contextmanager
from typing import Dict, Callable, Optional, Iterable, Set, Tuple

class WorkFlow(Module):
    """Context-managed module for batch registration of Prefect run functions.

    The `WorkFlow` class allows deferred registration of multiple functions
    as Prefect tasks or flows within a context block. Functions decorated with
    `@workflow.run_func(...)` inside a `with WorkFlow():` block are collected first
    and registered only upon exiting the context.

    Example:
        with WorkFlow() as wf:
            @wf.run_func(as_task=True)
            def step1(x: int) -> int:
                return x + 1

            @wf.run_func(as_task=False)
            def step2(y: int) -> int:
                return y * 2

        # Both step1 and step2 are now registered as Prefect run functions.
    """
    
    def __init__(self):
        """Initializes an empty workflow for deferred function registration."""
        
        super().__init__()
        self._collected_funcs: Dict[str, Tuple[Callable, bool]] = {}
        self._initialized = False

    def __enter__(self):
        """Enters the workflow context.

        Returns:
            The workflow instance, enabling use in a `with` statement.
        """
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Registers all collected functions as Prefect tasks or flows on exit.

        This method finalizes the workflow by registering all collected functions
        with Prefect when leaving the context block. Each collected function
        is converted to a task or flow based on the `as_task` flag.

        Args:
            exc_type: Exception type, if raised within the block.
            exc_value: Exception value, if raised within the block.
            traceback: Traceback object for the exception.

        Returns:
            Always returns False to propagate exceptions normally.
        """
        
        if not self._initialized:
            for run_name, (func, as_task) in self._collected_funcs.items():
                pf = super().run_func(func, name=run_name, as_task=as_task)
                setattr(self, run_name, pf)
            self._initialized = True
        return False

    def run_func(self, f: Callable = None, *, as_task: bool = True):
        """Collects a function for later registration as a Prefect task or flow.

        Used as a decorator inside a `WorkFlow` context. The function is stored
        until the context exits, at which point it is automatically registered
        as a Prefect task or flow.

        Args:
            f: Function to register. If `None`, the method
                acts as a decorator.
            as_task: If True, register the function as a Prefect
                task; otherwise, as a Prefect flow. Defaults to True.

        Returns:
            The decorated function or a decorator for later use.

        Raises:
            RuntimeError: If a function with the same name is already collected.
        """
                
        
        def decorator(func: Callable):
            """Records a function and its registration mode (task or flow).

            Args:
                func: The function to be collected.

            Returns:
                The same function, unmodified.

            Raises:
                RuntimeError: If a function with the same name already exists.
            """
            
            name = func.__name__
            if name in self._collected_funcs:
                raise RuntimeError(f"Duplicate run function {name}")
            self._collected_funcs[name] = (func, as_task)
            return func
        if f is None:
            return decorator
        else:
            return decorator(f)

