"""Nutpie NUTS inference method for the registry."""

from __future__ import annotations

from typing import Any

from ..core._registry import MethodInfo
from ._registry import InferenceMethod


class NutpieNutsMethod(InferenceMethod):
    """Nutpie-backed NUTS (Rust-based) for Stan and PyMC models."""

    def __init__(self) -> None:
        # Cache supported types at construction (imports already succeeded
        # if this class was instantiated during __init__.py registration).
        types: list[type] = []
        try:
            from ..modeling._stan import StanModel
            types.append(StanModel)
        except ImportError:
            pass
        try:
            from ..modeling._pymc import PyMCModel
            types.append(PyMCModel)
        except ImportError:
            pass
        self._supported = tuple(types)

    @property
    def name(self) -> str:
        return "nutpie_nuts"

    def supported_types(self) -> tuple[type, ...]:
        return self._supported

    @property
    def priority(self) -> int:
        return 80

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        if not isinstance(dist, self._supported):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires StanModel or PyMCModel")
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> Any:
        from ._nutpie import condition_on_nutpie
        return condition_on_nutpie._func(dist, observed, **kwargs)
