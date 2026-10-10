"""Frozen Python signatures and argument binding for Function values."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, get_type_hints


@dataclass(frozen=True)
class FunctionSignatureInfo:
    """Cached signature metadata for one wrapped Function."""

    signature: inspect.Signature
    hints: Mapping[str, Any]
    param_names: tuple[str, ...]
    has_var_keyword: bool


@dataclass(frozen=True)
class FunctionInputRef:
    """Reference to one planner-visible value in a resolved Python call."""

    parameter_name: str
    subscript: int | str | None = None

    @property
    def label(self) -> str:
        """Stable display name for provenance and broadcast metadata."""
        if isinstance(self.subscript, int):
            return f"*{self.parameter_name}[{self.subscript}]"
        if isinstance(self.subscript, str):
            return f"**{self.parameter_name}[{self.subscript!r}]"
        return self.parameter_name


def make_signature_info(
    func: Callable[..., Any],
) -> FunctionSignatureInfo:
    """Build reusable signature metadata for a wrapped function."""
    signature = inspect.signature(func)
    hints = _get_type_hints(func)
    return make_signature_info_from_signature(signature, hints=hints)


def make_signature_info_from_signature(
    signature: inspect.Signature,
    *,
    hints: Mapping[str, Any] | None = None,
) -> FunctionSignatureInfo:
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
    return FunctionSignatureInfo(
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
    info: FunctionSignatureInfo,
    values: Mapping[str, Any],
) -> tuple[FunctionInputRef, ...]:
    """Return planner-visible input references in Python parameter order."""
    refs: list[FunctionInputRef] = []
    for name, parameter in info.signature.parameters.items():
        if name not in values:
            continue
        value = values[name]
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            refs.extend(FunctionInputRef(name, subscript=index) for index in range(len(value)))
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            refs.extend(FunctionInputRef(name, subscript=key) for key in value)
        else:
            refs.append(FunctionInputRef(name))
    return tuple(refs)


def parameter_lifting_hint(info: FunctionSignatureInfo, name: str) -> Any:
    """Return the annotation that governs lifting at parameter *name*.

    ``Any`` on a variadic parameter governs none, so the arguments it collects
    lift and sweep as they would at an unannotated parameter.
    """
    hint = info.hints.get(name)
    parameter = info.signature.parameters.get(name)
    variadic = parameter is not None and parameter.kind in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    )
    return None if variadic and hint is Any else hint


def input_ref_hint(info: FunctionSignatureInfo, ref: FunctionInputRef) -> Any:
    """Return the annotation that governs lifting at one planner input."""
    return parameter_lifting_hint(info, ref.parameter_name)


def input_ref_value(values: Mapping[str, Any], ref: FunctionInputRef) -> Any:
    """Read one referenced value from signature-shaped call values."""
    value = values[ref.parameter_name]
    return value if ref.subscript is None else value[ref.subscript]


def replace_input_ref(
    values: Mapping[str, Any],
    ref: FunctionInputRef,
    value: Any,
) -> dict[str, Any]:
    """Return signature-shaped values with one referenced input replaced."""
    out = dict(values)
    if isinstance(ref.subscript, int):
        items = list(out[ref.parameter_name])
        items[ref.subscript] = value
        out[ref.parameter_name] = tuple(items)
    elif isinstance(ref.subscript, str):
        extras = dict(out[ref.parameter_name])
        extras[ref.subscript] = value
        out[ref.parameter_name] = extras
    else:
        out[ref.parameter_name] = value
    return out


def replace_input_refs(
    values: Mapping[str, Any],
    replacements: Mapping[FunctionInputRef, Any],
) -> dict[str, Any]:
    """Return signature-shaped values with referenced inputs replaced."""
    out = dict(values)
    positional: dict[str, list[Any]] = {}
    keywords: dict[str, dict[str, Any]] = {}
    for ref, value in replacements.items():
        if isinstance(ref.subscript, int):
            items = positional.setdefault(ref.parameter_name, list(out[ref.parameter_name]))
            items[ref.subscript] = value
        elif isinstance(ref.subscript, str):
            extras = keywords.setdefault(ref.parameter_name, dict(out[ref.parameter_name]))
            extras[ref.subscript] = value
        else:
            out[ref.parameter_name] = value
    out.update({name: tuple(items) for name, items in positional.items()})
    out.update(keywords)
    return out


def is_dependency_param(
    info: FunctionSignatureInfo,
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


def resolve_function_values(
    info: FunctionSignatureInfo,
    call_inputs: dict[str, Any],
    *,
    bind: Mapping[str, Any],
    module: Any | None,
    dependency_type: type,
    function_name: str,
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
                        f"bind[{name!r}] must be a mapping because {name!r} is the **kwargs "
                        f"parameter of {function_name!r}; got {type(bound_container).__name__}"
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
                    f"{function_name}: dependency {name!r} is provided by the module and "
                    f"cannot be overridden at call time"
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
                f"bind= names {sorted(unexpected)}, which are not parameters of "
                f"Function {function_name!r}"
            )

    _validate_required_values(info, values, function_name=function_name)
    _validate_dependency_values(
        info,
        values,
        dependency_type=dependency_type,
        function_name=function_name,
    )
    return values


def _get_type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    type_params = getattr(func, "__type_params__", ())
    try:
        if not type_params:
            return get_type_hints(func)
        localns = {param.__name__: param for param in type_params}
        return get_type_hints(func, localns=localns)
    except TypeError:
        # ``typing.get_type_hints`` rejects partials and callable instances even
        # though ``inspect.signature`` supports them. Their adjusted signature
        # remains authoritative and contributes any retained annotations below.
        if getattr(func, "__annotations__", None) is None:
            return {}
        raise


def _validate_required_values(
    info: FunctionSignatureInfo,
    values: dict[str, Any],
    *,
    function_name: str,
) -> None:
    for name, param in info.signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is param.empty and name not in values:
            raise TypeError(f"{function_name}() missing required argument {name!r}")


def _validate_dependency_values(
    info: FunctionSignatureInfo,
    values: dict[str, Any],
    *,
    dependency_type: type,
    function_name: str,
) -> None:
    for ref in iter_input_refs(info, values):
        name = ref.parameter_name
        if not is_dependency_param(info, name, dependency_type=dependency_type):
            continue
        value = input_ref_value(values, ref)
        if not isinstance(value, dependency_type):
            ann = info.hints.get(name)
            annotation = getattr(ann, "__name__", None) or repr(ann)
            raise TypeError(
                f"Function {function_name!r} expects a Node for dependency {ref.label!r} "
                f"(annotated {annotation}); got {type(value).__name__}"
            )
