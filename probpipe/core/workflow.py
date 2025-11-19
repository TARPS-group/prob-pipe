from .module import Module, InputSpec
from types import SimpleNamespace
from contextlib import contextmanager
from typing import Dict, Callable, Optional, Type, Union

class WorkFlow:
    """Factory for creating dynamically-generated Module classes with a single run function.

    Usage:

        wf = WorkFlow()

        MyModule = wf.run_func(my_func, as_task=True)  # returns a Module subclass
        mod_instance = MyModule()
        result = mod_instance.my_func(...)

        AnotherModule = wf.run_func(another_func, as_task=False)
        ...

    Each call to `run_func` generates an independent Module subclass
    that encapsulates the provided function as its only run function.
    """

    def run_func(
        self,
        f: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        as_task: bool = True,
    ) -> Union[Type[Module], Callable]:
        def decorator(func: Callable) -> Type[Module]:
            run_name = name or func.__name__

            class RunFuncModule(Module):
                def __init__(self, **dependencies):
                    super().__init__(**dependencies)
                    self.run_func(func, name=run_name, as_task=as_task)
            return RunFuncModule

        if f is None:
            # Used as @wf.run_func(...) — return decorator waiting to receive function
            return decorator
        else:
            # Used as @wf.run_func without parentheses — decorate immediately
            return decorator(f)