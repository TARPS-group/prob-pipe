"""CmdStan NUTS inference method for the registry."""

from __future__ import annotations

from typing import Any

from ..core._registry import MethodInfo
from ._registry import InferenceMethod


class CmdStanNutsMethod(InferenceMethod):
    """CmdStanPy-backed NUTS for Stan models."""

    @property
    def name(self) -> str:
        return "cmdstan_nuts"

    def supported_types(self) -> tuple[type, ...]:
        from ..modeling._stan import StanModel
        return (StanModel,)

    @property
    def priority(self) -> int:
        return 70

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        from ..modeling._stan import StanModel
        if not isinstance(dist, StanModel):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires StanModel")
        return MethodInfo(feasible=True, method_name=self.name,
                          description="CmdStan NUTS")

    def condition(self, dist: Any, observed: Any, **kwargs: Any) -> Any:
        return dist._condition_on(observed, **kwargs)
