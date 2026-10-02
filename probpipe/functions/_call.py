"""Resolve domain arguments for the Function call engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..values._binding import FunctionSignatureInfo, resolve_function_values


def resolve_function_call(
    info: FunctionSignatureInfo,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    *,
    bind: Mapping[str, Any],
    module: Any | None,
    dependency_type: type,
    function_name: str,
) -> dict[str, Any]:
    """Bind the frozen signature and fill missing domain inputs."""
    bound = info.signature.bind_partial(*args, **call_inputs)
    return resolve_function_values(
        info,
        dict(bound.arguments),
        bind=bind,
        module=module,
        dependency_type=dependency_type,
        function_name=function_name,
    )
