"""Frozen Python signatures, resolved annotations, and argument binding for Function values."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, TypeAliasType, TypeVar, get_args, get_origin, get_type_hints

try:
    import typing_extensions
except ImportError:
    _TYPE_ALIAS_CLASSES: tuple[type, ...] = (TypeAliasType,)
else:
    # Before Python 3.14, typing_extensions defines a TypeAliasType class of its own.
    _TYPE_ALIAS_CLASSES = (TypeAliasType, typing_extensions.TypeAliasType)


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


def resolved_hint(hint: Any, expanded: set[Any] | None = None) -> Any:
    """The annotation *hint* stands for, with its type aliases and ``Annotated`` wrappers resolved.

    A type alias stands for its value. A parametrized alias stands for its value
    with each type parameter replaced by the argument in its position, so with
    ``type Maybe[T] = T | None`` the annotation ``Maybe[Normal]`` stands for
    ``Normal | None``. A generic alias written without arguments stands for its
    value as written. ``Annotated[T, ...]`` stands for ``T``, because the hints
    of a partial or a callable object keep the metadata that ``get_type_hints``
    removes from a function's hints.

    Resolution adds each alias it expands to *expanded* and stops at an alias
    already in *expanded*, which ends a recursive alias. An alias also stays
    unresolved when its value raises ``NameError``, as a value that names a
    class imported only under ``TYPE_CHECKING`` does, and when
    :func:`_substituted` cannot place its arguments, as for a type parameter
    that is a ``TypeVarTuple`` or a ``ParamSpec``.
    """
    expanded = set() if expanded is None else expanded
    origin = get_origin(hint)
    if origin is Annotated:
        return resolved_hint(get_args(hint)[0], expanded)
    alias = hint if isinstance(hint, _TYPE_ALIAS_CLASSES) else origin
    if not isinstance(alias, _TYPE_ALIAS_CLASSES) or alias in expanded:
        return hint
    try:
        value = alias.__value__
        if alias is not hint:
            value = _substituted(value, alias.__type_params__, get_args(hint))
    except (NameError, TypeError):
        return hint
    expanded.add(alias)
    return resolved_hint(value, expanded)


def _substituted(value: Any, parameters: tuple[Any, ...], arguments: tuple[Any, ...]) -> Any:
    """*value* with each type variable in *parameters* replaced by the argument in its position.

    The free type variables of *value* are replaced through its own
    subscription, so each use of a parameter receives that parameter's argument
    in any order or nesting, as in ``dict[V, K]`` for ``type Pair[K, V]``. A
    class stays as it is, because a generic class in an alias's value is not
    parametrized by the alias.

    Parameters
    ----------
    value : Any
        The value of a generic alias.
    parameters : tuple
        The alias's type parameters, in declared order.
    arguments : tuple
        The arguments of the parametrized alias, in order.

    Returns
    -------
    Any
        *value* with each parameter replaced by its argument.

    Raises
    ------
    TypeError
        If a parameter is a ``TypeVarTuple`` or a ``ParamSpec``, the arguments
        differ in number from the parameters, or the subscription refuses an
        argument.
    """
    if len(arguments) != len(parameters) or not all(
        isinstance(parameter, TypeVar) for parameter in parameters
    ):
        raise TypeError(f"cannot place the arguments {arguments} in the parameters {parameters}")
    argument_of = dict(zip(parameters, arguments))
    if isinstance(value, TypeVar):
        return argument_of.get(value, value)
    free = () if isinstance(value, type) else getattr(value, "__parameters__", ())
    if not any(parameter in argument_of for parameter in free):
        return value
    return value[tuple(argument_of.get(parameter, parameter) for parameter in free)]


def parameter_lifting_hint(info: FunctionSignatureInfo, name: str) -> Any:
    """Return the annotation that governs lifting at parameter *name*.

    ``Any`` on a variadic parameter governs none, so the arguments it collects
    lift and sweep as they would at an unannotated parameter. The annotation is
    compared with ``Any`` once :func:`resolved_hint` resolves it.
    """
    hint = info.hints.get(name)
    parameter = info.signature.parameters.get(name)
    variadic = parameter is not None and parameter.kind in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    )
    return None if variadic and resolved_hint(hint) is Any else hint


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


def resolve_function_values(
    info: FunctionSignatureInfo,
    call_inputs: dict[str, Any],
    *,
    bind: Mapping[str, Any],
    function_name: str,
) -> dict[str, Any]:
    """Resolve arguments from call values, construction bindings, then defaults."""
    values: dict[str, Any] = {}

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

        if name in call_inputs:
            values[name] = call_inputs[name]
        elif name in bind:
            values[name] = bind[name]

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
