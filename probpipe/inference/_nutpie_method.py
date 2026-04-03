"""Nutpie NUTS inference method for the registry."""

from __future__ import annotations

from typing import Any

from ..core._registry import MethodInfo
from ._registry import InferenceMethod


class NutpieNutsMethod(InferenceMethod):
    """Nutpie-backed NUTS (Rust-based) for Stan and PyMC models."""

    @property
    def name(self) -> str:
        return "nutpie_nuts"

    def supported_types(self) -> tuple[type, ...]:
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
        return tuple(types)

    @property
    def priority(self) -> int:
        return 80

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        try:
            import nutpie  # noqa: F401
        except ImportError:
            return MethodInfo(feasible=False, method_name=self.name,
                              description="nutpie not installed")

        has_stan = hasattr(dist, "_bridgestan_model")
        has_pymc = hasattr(dist, "_pymc_model")
        if not (has_stan or has_pymc):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires StanModel or PyMCModel")
        return MethodInfo(feasible=True, method_name=self.name,
                          description="Nutpie NUTS (Rust-based)")

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> Any:
        from ._nutpie import condition_on_nutpie
        return condition_on_nutpie._func(dist, observed, **kwargs)
