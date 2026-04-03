"""Factory for inference methods that delegate to dist._condition_on."""

from __future__ import annotations

import importlib
from typing import Any

from ..core._registry import MethodInfo
from ._registry import InferenceMethod


class _DelegatingMethod(InferenceMethod):
    """Inference method that delegates to ``dist._condition_on``.

    Used for model-specific backends (CmdStan, PyMC MCMC) where the
    model class already implements conditioning logic internally.
    """

    def __init__(self, method_name: str, model_type: type, method_priority: int):
        self._method_name = method_name
        self._model_type = model_type
        self._method_priority = method_priority

    @property
    def name(self) -> str:
        return self._method_name

    def supported_types(self) -> tuple[type, ...]:
        return (self._model_type,)

    @property
    def priority(self) -> int:
        return self._method_priority

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        if not isinstance(dist, self._model_type):
            return MethodInfo(feasible=False, method_name=self.name,
                              description=f"Requires {self._model_type.__name__}")
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> Any:
        return dist._condition_on(observed, **kwargs)


def make_delegating_method(
    method_name: str,
    model_path: str,
    method_priority: int,
) -> type[_DelegatingMethod]:
    """Create a delegating method class for a model type.

    Parameters
    ----------
    method_name : str
        Registry name (e.g., ``"cmdstan_nuts"``).
    model_path : str
        Dotted import path to the model class
        (e.g., ``"probpipe.modeling._stan.StanModel"``).
    method_priority : int
        Default priority.

    Returns
    -------
    type
        A callable that returns a ``_DelegatingMethod`` instance.
        The model class is imported lazily on first instantiation.
    """
    module_path, class_name = model_path.rsplit(".", 1)

    def factory() -> _DelegatingMethod:
        mod = importlib.import_module(module_path)
        model_type = getattr(mod, class_name)
        return _DelegatingMethod(method_name, model_type, method_priority)

    return factory
