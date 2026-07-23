"""Function call-resolution helpers.

This private module owns function signature metadata, workflow option
resolution, argument binding, module/default resolution, and dependency
validation. It deliberately knows nothing about distribution conversion,
broadcast planning, execution, or result coercion.
"""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, get_type_hints


@dataclass(frozen=True)
class WorkflowSignatureInfo:
    """Cached signature metadata for one wrapped Function."""

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
    """Resolved call-time workflow settings consumed by ``Function``."""

    n_broadcast_samples: int
    include_inputs: bool
    seed: int | None


@dataclass(frozen=True)
class ResolvedWorkflowCall:
    """Fully resolved signature-shaped values plus workflow overrides."""

    values: dict[str, Any]
    overrides: WorkflowCallOverrides


@dataclass(frozen=True)
class WorkflowInputRef:
    """Reference to one planner-visible value in a resolved Python call."""

    parameter_name: str
    position: int | None = None
    keyword: str | None = None

    @property
    def label(self) -> str:
        """Stable display name for provenance and broadcast metadata."""
        if self.position is not None:
            return f"*{self.parameter_name}[{self.position}]"
        if self.keyword is not None:
            return f"**{self.parameter_name}[{self.keyword!r}]"
        return self.parameter_name


def make_signature_info(
    func: Callable[..., Any],
) -> WorkflowSignatureInfo:
    """Build reusable signature metadata for a wrapped function."""
    signature = inspect.signature(func)
    hints = _get_type_hints(func)
    param_names = tuple(signature.parameters)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    return WorkflowSignatureInfo(
        signature=signature,
        hints=hints,
        param_names=param_names,
        has_var_keyword=has_var_keyword,
    )


def make_signature_info_from_signature(
    signature: inspect.Signature,
    *,
    hints: Mapping[str, Any] | None = None,
) -> WorkflowSignatureInfo:
    """Build reusable metadata from an independently supplied signature."""
    if not isinstance(signature, inspect.Signature):
        raise TypeError(f"signature must be inspect.Signature, got {type(signature).__name__}")
    resolved_hints = dict(hints or {})
    for name, parameter in signature.parameters.items():
        if parameter.annotation is not inspect.Parameter.empty:
            resolved_hints.setdefault(name, parameter.annotation)
    param_names = tuple(signature.parameters)
    has_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    return WorkflowSignatureInfo(
        signature=signature,
        hints=resolved_hints,
        param_names=param_names,
        has_var_keyword=has_var_keyword,
    )


def values_to_bound_arguments(
    signature: inspect.Signature,
    values: Mapping[str, Any],
) -> inspect.BoundArguments:
    """Reconstruct Python call semantics from resolved workflow values."""
    arguments: OrderedDict[str, Any] = OrderedDict()
    for name in signature.parameters:
        if name in values:
            arguments[name] = values[name]
    return inspect.BoundArguments(signature, arguments)


def iter_input_refs(
    info: WorkflowSignatureInfo,
    values: Mapping[str, Any],
) -> tuple[WorkflowInputRef, ...]:
    """Return planner-visible input references in Python parameter order."""
    refs: list[WorkflowInputRef] = []
    for name, parameter in info.signature.parameters.items():
        if name not in values:
            continue
        value = values[name]
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            refs.extend(WorkflowInputRef(name, position=index) for index in range(len(value)))
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            refs.extend(WorkflowInputRef(name, keyword=key) for key in value)
        else:
            refs.append(WorkflowInputRef(name))
    return tuple(refs)


def input_ref_hint(info: WorkflowSignatureInfo, ref: WorkflowInputRef) -> Any:
    """Return the annotation governing one resolved input reference."""
    return info.hints.get(ref.parameter_name)


def input_ref_value(values: Mapping[str, Any], ref: WorkflowInputRef) -> Any:
    """Read one referenced value from signature-shaped call values."""
    value = values[ref.parameter_name]
    if ref.position is not None:
        return value[ref.position]
    if ref.keyword is not None:
        return value[ref.keyword]
    return value


def replace_input_ref(
    values: Mapping[str, Any],
    ref: WorkflowInputRef,
    value: Any,
) -> dict[str, Any]:
    """Return signature-shaped values with one referenced input replaced."""
    out = dict(values)
    if ref.position is not None:
        items = list(out[ref.parameter_name])
        items[ref.position] = value
        out[ref.parameter_name] = tuple(items)
    elif ref.keyword is not None:
        extras = dict(out[ref.parameter_name])
        extras[ref.keyword] = value
        out[ref.parameter_name] = extras
    else:
        out[ref.parameter_name] = value
    return out


def replace_input_refs(
    values: Mapping[str, Any],
    replacements: Mapping[WorkflowInputRef, Any],
) -> dict[str, Any]:
    """Return signature-shaped values with referenced inputs replaced."""
    out = dict(values)
    positional: dict[str, list[Any]] = {}
    keywords: dict[str, dict[str, Any]] = {}
    for ref, value in replacements.items():
        if ref.position is not None:
            items = positional.setdefault(ref.parameter_name, list(out[ref.parameter_name]))
            items[ref.position] = value
        elif ref.keyword is not None:
            extras = keywords.setdefault(ref.parameter_name, dict(out[ref.parameter_name]))
            extras[ref.keyword] = value
        else:
            out[ref.parameter_name] = value
    out.update({name: tuple(items) for name, items in positional.items()})
    out.update(keywords)
    return out


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
    return dict(bound.arguments), overrides


def resolve_workflow_values(
    info: WorkflowSignatureInfo,
    call_inputs: dict[str, Any],
    *,
    bind: Mapping[str, Any],
    module: Any | None,
    dependency_type: type,
    workflow_name: str,
) -> dict[str, Any]:
    """Resolve final signature-shaped arguments from every value source."""
    values: dict[str, Any] = {}
    mod_child_nodes = getattr(module, "child_nodes", {}) if module is not None else {}
    mod_inputs = getattr(module, "inputs", {}) if module is not None else {}

    var_keyword_name = next(
        (
            name
            for name, parameter in info.signature.parameters.items()
            if parameter.kind == inspect.Parameter.VAR_KEYWORD
        ),
        None,
    )

    for name, param in info.signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            extras: dict[str, Any] = {}
            bound_container = bind.get(name)
            if bound_container is not None:
                if not isinstance(bound_container, Mapping):
                    raise TypeError(
                        f"Construction binding for variadic keyword parameter "
                        f"'{name}' of workflow '{workflow_name}' must be a mapping"
                    )
                extras.update(bound_container)
            known_params = set(info.signature.parameters)
            extras.update({key: value for key, value in bind.items() if key not in known_params})
            extras.update(call_inputs.get(name, {}))
            if extras:
                values[name] = extras
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

    if var_keyword_name is None:
        # Construction declarations validate this case before call time; keep
        # the guard here for direct use of the private resolver.
        unexpected = set(bind).difference(info.signature.parameters)
        if unexpected:
            raise TypeError(
                f"Unexpected construction bindings for workflow '{workflow_name}': "
                f"{sorted(unexpected)}"
            )

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
    """Resolve one ``Function`` call into values plus overrides."""
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
    for ref in iter_input_refs(info, values):
        name = ref.parameter_name
        if not is_dependency_param(info, name, dependency_type=dependency_type):
            continue
        value = input_ref_value(values, ref)
        if not isinstance(value, dependency_type):
            ann = info.hints.get(name)
            raise TypeError(
                f"Function '{workflow_name}' expects dependency "
                f"'{ref.label}: {ann}' to be a Node, but got {type(value)}."
            )
