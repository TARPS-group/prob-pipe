"""WorkflowFunction call-resolution helpers.

This private module owns function signature metadata, workflow option
resolution, argument binding, module/default resolution, and dependency
validation. It deliberately knows nothing about distribution conversion,
broadcast planning, execution, or result coercion.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, get_type_hints

WORKFLOW_CALL_OPTION_NAMES = frozenset(
    {"n_broadcast_samples", "seed", "include_inputs"}
)


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


def bind_call_inputs(
    info: WorkflowSignatureInfo,
    args: tuple[Any, ...],
    call_inputs: dict[str, Any],
    *,
    default_n_broadcast_samples: int,
    default_include_inputs: bool,
    options: WorkflowCallOptions | None = None,
    warn_legacy_overrides: bool = False,
) -> tuple[dict[str, Any], WorkflowCallOverrides]:
    """Bind user inputs and resolve workflow controls.

    Legacy call-time controls are consumed from ``call_inputs`` only when
    they cannot bind to the wrapped function. Explicit ``options`` are the
    preferred control plane and take precedence.
    """
    raw_inputs = dict(call_inputs)
    legacy_options = _pop_legacy_workflow_options(
        info,
        raw_inputs,
        warn=warn_legacy_overrides,
    )
    explicit_options = options or WorkflowCallOptions()

    n_broadcast_samples = default_n_broadcast_samples
    if legacy_options.n_broadcast_samples is not None:
        n_broadcast_samples = legacy_options.n_broadcast_samples
    if explicit_options.n_broadcast_samples is not None:
        n_broadcast_samples = explicit_options.n_broadcast_samples

    include_inputs = default_include_inputs
    if legacy_options.include_inputs is not None:
        include_inputs = legacy_options.include_inputs
    if explicit_options.include_inputs is not None:
        include_inputs = explicit_options.include_inputs

    seed = legacy_options.seed
    if explicit_options.seed is not None:
        seed = explicit_options.seed

    bound = info.signature.bind_partial(*args, **raw_inputs)
    bound_inputs: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        param = info.signature.parameters[name]
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            bound_inputs.update(value)
        else:
            bound_inputs[name] = value

    return bound_inputs, WorkflowCallOverrides(
        n_broadcast_samples=n_broadcast_samples,
        include_inputs=include_inputs,
        seed=seed,
    )


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
    warn_legacy_overrides: bool = False,
) -> ResolvedWorkflowCall:
    """Resolve one ``WorkflowFunction`` call into values plus overrides."""
    bound_inputs, overrides = bind_call_inputs(
        info,
        args,
        call_inputs,
        default_n_broadcast_samples=default_n_broadcast_samples,
        default_include_inputs=default_include_inputs,
        options=options,
        warn_legacy_overrides=warn_legacy_overrides,
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


def _pop_legacy_workflow_options(
    info: WorkflowSignatureInfo,
    raw_inputs: dict[str, Any],
    *,
    warn: bool,
) -> WorkflowCallOptions:
    consumed: dict[str, Any] = {}
    for name in WORKFLOW_CALL_OPTION_NAMES:
        if name not in raw_inputs:
            continue
        if _can_bind_as_user_input(info, name):
            continue
        consumed[name] = raw_inputs.pop(name)

    if consumed and warn:
        names = ", ".join(sorted(consumed))
        warnings.warn(
            "Passing WorkflowFunction options as call kwargs is deprecated "
            f"for {names}; use workflow.with_options(...)(...) instead.",
            DeprecationWarning,
            stacklevel=6,
        )

    return WorkflowCallOptions(
        n_broadcast_samples=consumed.get("n_broadcast_samples"),
        include_inputs=consumed.get("include_inputs"),
        seed=consumed.get("seed"),
    )


def _can_bind_as_user_input(info: WorkflowSignatureInfo, name: str) -> bool:
    return name in info.signature.parameters or info.has_var_keyword


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
