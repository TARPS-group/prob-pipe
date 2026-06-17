"""WorkflowFunction call-resolution helpers.

This private module owns function signature metadata, workflow option
resolution, argument binding, module/default resolution, and dependency
validation. It deliberately knows nothing about distribution conversion,
broadcast planning, execution, or result coercion.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, get_type_hints


@dataclass(frozen=True)
class WorkflowSignatureInfo:
    """Cached signature metadata for one wrapped workflow function."""

    signature: inspect.Signature
    hints: Mapping[str, Any]
    param_names: tuple[str, ...]
    has_var_keyword: bool


@dataclass(frozen=True)
class WorkflowCallOptions:
    """Optional call-time workflow controls outside user kwargs."""

    n_broadcast_samples: int | None = None
    include_inputs: bool | None = None
    seed: int | None = None


@dataclass(frozen=True)
class WorkflowCallOverrides:
    """Resolved call-time workflow settings consumed by ``WorkflowFunction``."""

    n_broadcast_samples: int
    include_inputs: bool
    seed: int | None


@dataclass(frozen=True)
class ResolvedWorkflowCall:
    """Fully resolved call values plus call-time workflow overrides."""

    values: dict[str, Any]
    overrides: WorkflowCallOverrides


def make_signature_info(
    func: Callable[..., Any],
) -> WorkflowSignatureInfo:
    """Build reusable signature metadata for a wrapped function."""
    signature = inspect.signature(func)
    hints = _get_type_hints(func)
    param_names = tuple(p for p in signature.parameters if p != "self")
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    return WorkflowSignatureInfo(
        signature=signature,
        hints=hints,
        param_names=param_names,
        has_var_keyword=has_var_keyword,
    )


def is_dependency_param(
    info: WorkflowSignatureInfo,
    name: str,
    *,
    dependency_type: type,
) -> bool:
    """Return whether a parameter annotation names a workflow dependency."""
    ann = info.hints.get(name)
    try:
        return isinstance(ann, type) and issubclass(ann, dependency_type)
    except TypeError:
        return False


def _flatten_bound_inputs(
    signature: inspect.Signature,
    bound: inspect.BoundArguments,
) -> dict[str, Any]:
    """Flatten bound arguments into workflow input names.

    ``**kwargs`` parameters are expanded into the returned input mapping.
    """
    inputs: dict[str, Any] = {}

    for name, value in bound.arguments.items():
        param = signature.parameters[name]

        if param.kind == inspect.Parameter.VAR_KEYWORD:
            inputs.update(value)
        else:
            inputs[name] = value

    return inputs


def bind_call_inputs(
    info: WorkflowSignatureInfo,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    *,
    default_n_broadcast_samples: int,
    default_include_inputs: bool,
    options: WorkflowCallOptions | None = None,
) -> tuple[dict[str, Any], WorkflowCallOverrides]:
    """Bind user inputs and resolve workflow controls.

    Call inputs bind exactly like the wrapped Python function. Workflow
    controls come only from explicit ``options`` or construction defaults.
    """
    explicit_options = options if options is not None else WorkflowCallOptions()

    def resolve_option(name: str, default: Any = None) -> Any:
        explicit_value = getattr(explicit_options, name)
        if explicit_value is not None:
            return explicit_value

        return default

    overrides = WorkflowCallOverrides(
        n_broadcast_samples=resolve_option(
            "n_broadcast_samples",
            default_n_broadcast_samples,
        ),
        include_inputs=resolve_option(
            "include_inputs",
            default_include_inputs,
        ),
        seed=resolve_option("seed"),
    )

    bound = info.signature.bind_partial(*args, **call_inputs)
    bound_inputs = _flatten_bound_inputs(info.signature, bound)

    return bound_inputs, overrides


def resolve_workflow_values(
    info: WorkflowSignatureInfo,
    call_inputs: dict[str, Any],
    *,
    bind: Mapping[str, Any],
    module: Any | None,
    dependency_type: type,
    workflow_name: str,
) -> dict[str, Any]:
    """Resolve final function kwargs from call, bind, module, and defaults."""
    values: dict[str, Any] = {}
    mod_child_nodes = getattr(module, "child_nodes", {}) if module is not None else {}
    mod_inputs = getattr(module, "inputs", {}) if module is not None else {}

    for name, param in info.signature.parameters.items():
        if name == "self":
            continue

        is_dep = is_dependency_param(info, name, dependency_type=dependency_type)

        if name in call_inputs:
            if module is not None and is_dep and name in mod_child_nodes:
                raise TypeError(
                    f"Dependency '{name}' for workflow '{workflow_name}' is provided "
                    f"by the module and cannot be overridden at call time."
                )
            values[name] = call_inputs[name]
        elif name in bind:
            values[name] = bind[name]
        elif module is not None:
            if is_dep and name in mod_child_nodes:
                values[name] = mod_child_nodes[name]
            elif not is_dep and name in mod_inputs:
                values[name] = mod_inputs[name]

        if name not in values and param.default is not param.empty:
            values[name] = param.default

    if info.has_var_keyword:
        known_params = set(info.signature.parameters.keys())
        for name, value in call_inputs.items():
            if name not in known_params:
                values[name] = value

    _validate_required_values(info, values, workflow_name=workflow_name)
    _validate_dependency_values(
        info,
        values,
        dependency_type=dependency_type,
        workflow_name=workflow_name,
    )
    return values


def resolve_workflow_call(
    info: WorkflowSignatureInfo,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    *,
    bind: Mapping[str, Any],
    module: Any | None,
    dependency_type: type,
    workflow_name: str,
    default_n_broadcast_samples: int,
    default_include_inputs: bool,
    options: WorkflowCallOptions | None = None,
) -> ResolvedWorkflowCall:
    """Resolve one ``WorkflowFunction`` call into values plus overrides."""
    bound_inputs, overrides = bind_call_inputs(
        info,
        args,
        call_inputs,
        default_n_broadcast_samples=default_n_broadcast_samples,
        default_include_inputs=default_include_inputs,
        options=options,
    )
    values = resolve_workflow_values(
        info,
        bound_inputs,
        bind=bind,
        module=module,
        dependency_type=dependency_type,
        workflow_name=workflow_name,
    )
    return ResolvedWorkflowCall(values=values, overrides=overrides)


def _get_type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    type_params = getattr(func, "__type_params__", ())
    if not type_params:
        return get_type_hints(func)
    localns = {param.__name__: param for param in type_params}
    return get_type_hints(func, localns=localns)


def _validate_required_values(
    info: WorkflowSignatureInfo,
    values: dict[str, Any],
    *,
    workflow_name: str,
) -> None:
    for name, param in info.signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is param.empty and name not in values:
            raise TypeError(f"Missing required input '{name}' for workflow '{workflow_name}'")


def _validate_dependency_values(
    info: WorkflowSignatureInfo,
    values: dict[str, Any],
    *,
    dependency_type: type,
    workflow_name: str,
) -> None:
    for name in info.param_names:
        if not is_dependency_param(info, name, dependency_type=dependency_type):
            continue
        value = values.get(name)
        if not isinstance(value, dependency_type):
            ann = info.hints.get(name)
            raise TypeError(
                f"WorkflowFunction '{workflow_name}' expects dependency "
                f"'{name}: {ann}' to be a Node, but got {type(value)}."
            )
