"""Workflow call engine and reproducible execution scopes.

Exports load on demand while distribution implementations still import the
engine's RNG helpers during package initialization.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..values import Function as Function
    from ..values import FunctionSpec as FunctionSpec
    from ._call import ApplicabilityError as ApplicabilityError
    from ._call import CallReport as CallReport
    from ._context import workflow_run as workflow_run
    from ._function import function as function
    from ._reparameterization import bijector_for as bijector_for
    from ._reparameterization import register_bijector as register_bijector
    from ._replay import replay_run as replay_run
    from ._result import ResultKindError as ResultKindError
    from ._result import ResultSchemaError as ResultSchemaError
    from ._rules import evaluation_rule_registry as evaluation_rule_registry

_EXPORTS = {
    "Function": "probpipe.values",
    "FunctionSpec": "probpipe.values",
    "function": "probpipe.functions._function",
    "workflow_run": "probpipe.functions._context",
    "replay_run": "probpipe.functions._replay",
    "ApplicabilityError": "probpipe.functions._call",
    "CallReport": "probpipe.functions._call",
    "ResultKindError": "probpipe.functions._result",
    "ResultSchemaError": "probpipe.functions._result",
    "bijector_for": "probpipe.functions._reparameterization",
    "register_bijector": "probpipe.functions._reparameterization",
    "evaluation_rule_registry": "probpipe.functions._rules",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
