"""Resolve Function controls and bind arguments for the call engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..values._binding import FunctionSignatureInfo, resolve_function_values


@dataclass(frozen=True)
class FunctionCallOptions:
    """Optional call-time Function controls outside user kwargs."""

    n_broadcast_samples: int | None = None
    include_inputs: bool | None = None


@dataclass(frozen=True)
class FunctionCallOverrides:
    """Resolved call-time Function settings consumed by ``Function``."""

    n_broadcast_samples: int
    include_inputs: bool


@dataclass(frozen=True)
class ResolvedFunctionCall:
    """Fully resolved signature-shaped values plus Function overrides."""

    values: dict[str, Any]
    overrides: FunctionCallOverrides


def bind_call_inputs(
    info: FunctionSignatureInfo,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    *,
    default_n_broadcast_samples: int,
    default_include_inputs: bool,
    options: FunctionCallOptions | None = None,
) -> tuple[dict[str, Any], FunctionCallOverrides]:
    """Bind user inputs and resolve Function controls.

    Call inputs bind exactly like the wrapped Python function. Function
    controls come only from explicit ``options`` or construction defaults.
    """
    explicit_options = options if options is not None else FunctionCallOptions()

    def resolve_option(name: str, default: Any = None) -> Any:
        explicit_value = getattr(explicit_options, name)
        if explicit_value is not None:
            return explicit_value

        return default

    overrides = FunctionCallOverrides(
        n_broadcast_samples=resolve_option(
            "n_broadcast_samples",
            default_n_broadcast_samples,
        ),
        include_inputs=resolve_option(
            "include_inputs",
            default_include_inputs,
        ),
    )

    bound = info.signature.bind_partial(*args, **call_inputs)
    return dict(bound.arguments), overrides


def resolve_function_call(
    info: FunctionSignatureInfo,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    *,
    bind: Mapping[str, Any],
    module: Any | None,
    dependency_type: type,
    function_name: str,
    default_n_broadcast_samples: int,
    default_include_inputs: bool,
    options: FunctionCallOptions | None = None,
) -> ResolvedFunctionCall:
    """Resolve one ``Function`` call into values plus overrides."""
    bound_inputs, overrides = bind_call_inputs(
        info,
        args,
        call_inputs,
        default_n_broadcast_samples=default_n_broadcast_samples,
        default_include_inputs=default_include_inputs,
        options=options,
    )
    values = resolve_function_values(
        info,
        bound_inputs,
        bind=bind,
        module=module,
        dependency_type=dependency_type,
        function_name=function_name,
    )
    return ResolvedFunctionCall(values=values, overrides=overrides)
