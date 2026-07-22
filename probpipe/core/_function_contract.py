"""Private implementation and schema contract for first-class Functions."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, cast

import jax.numpy as jnp

from ._distribution_base import Distribution
from .constraints import _supports_compatible
from .event_template import (
    ArraySpec,
    EventTemplate,
    ValueSpec,
    _concretize_event_template,
    _unify_event_template_with_value,
    _unify_event_templates,
)
from .record import Record


@dataclass(frozen=True)
class _FunctionInvocationContext:
    """Immutable state shared by every point evaluation in one invocation."""

    dimension_bindings: Mapping[str, int]

    def __init__(self, dimension_bindings: Mapping[str, int] | None = None):
        object.__setattr__(
            self,
            "dimension_bindings",
            MappingProxyType(dict(dimension_bindings or {})),
        )


class _FunctionImplementation(Protocol):
    """Private execution protocol implemented by Function payloads."""

    def invoke(
        self,
        bound_inputs: inspect.BoundArguments,
        *,
        context: _FunctionInvocationContext,
    ) -> Any:
        """Execute one resolved point and return its native value."""


@dataclass(frozen=True)
class _CallableFunctionImplementation:
    """Frozen implementation adapter for a plain Python callable."""

    callable: Any

    def invoke(
        self,
        bound_inputs: inspect.BoundArguments,
        *,
        context: _FunctionInvocationContext,
    ) -> Any:
        del context
        return self.callable(*bound_inputs.args, **bound_inputs.kwargs)


def _validate_function_templates(
    *,
    function_name: str,
    signature: inspect.Signature,
    input_template: EventTemplate | None,
    output_template: EventTemplate | None,
    construction_bindings: Mapping[str, Any],
) -> None:
    """Validate signature/template relationships without specializing schemas."""
    for label, template in (
        ("input_template", input_template),
        ("output_template", output_template),
    ):
        if template is not None and not isinstance(template, EventTemplate):
            raise TypeError(
                f"Function {function_name!r} {label} must be an EventTemplate or None, "
                f"got {type(template).__name__}"
            )

    parameters = list(signature.parameters.values())
    variadic = [
        parameter.name
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    fixed_names = {
        parameter.name
        for parameter in parameters
        if parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }

    if input_template is not None:
        if variadic:
            raise ValueError(
                f"Function {function_name!r} cannot use an authoritative input_template "
                f"with variadic parameters {variadic}"
            )
        template_names = set(input_template.children)
        if fixed_names != template_names:
            raise ValueError(
                f"Function {function_name!r} input_template fields "
                f"{sorted(template_names)} must exactly match signature parameters "
                f"{sorted(fixed_names)}"
            )

    parameter_names = {parameter.name for parameter in parameters}
    has_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    unexpected_bindings = set(construction_bindings).difference(parameter_names)
    if unexpected_bindings and not has_var_keyword:
        raise ValueError(
            f"Function {function_name!r} has invalid construction bindings: "
            f"unexpected names {sorted(unexpected_bindings)}"
        )

    if input_template is not None:
        for parameter in parameters:
            if parameter.default is not inspect.Parameter.empty:
                _validate_declared_input_value(
                    function_name=function_name,
                    input_template=input_template,
                    parameter_name=parameter.name,
                    value=parameter.default,
                    source="default",
                )
        for parameter_name, value in construction_bindings.items():
            _validate_declared_input_value(
                function_name=function_name,
                input_template=input_template,
                parameter_name=parameter_name,
                value=value,
                source="construction binding",
            )

        # Validate the values that can be active together under one shared
        # symbolic-dimension scope. Individual checks above keep every
        # declared default valid even when a construction binding overrides it.
        effective_bindings: dict[str, int] = {}
        for parameter in parameters:
            if parameter.name in construction_bindings:
                value = construction_bindings[parameter.name]
                source = "construction binding"
            elif parameter.default is not inspect.Parameter.empty:
                value = parameter.default
                source = "default"
            else:
                continue
            effective_bindings = _validate_declared_input_value(
                function_name=function_name,
                input_template=input_template,
                parameter_name=parameter.name,
                value=value,
                source=source,
                bindings=effective_bindings,
            )

    output_dimensions = output_template.free_dims if output_template is not None else frozenset()
    input_dimensions = input_template.free_dims if input_template is not None else frozenset()
    undeclared = output_dimensions.difference(input_dimensions)
    if undeclared:
        dimensions = ", ".join(sorted(undeclared))
        raise ValueError(
            f"Function {function_name!r} output_template uses symbolic dimensions not "
            f"declared by input_template: {dimensions}"
        )


def _validate_declared_input_value(
    *,
    function_name: str,
    input_template: EventTemplate,
    parameter_name: str,
    value: Any,
    source: str,
    bindings: Mapping[str, int] | None = None,
) -> dict[str, int]:
    child = input_template.children[parameter_name]
    single_parameter_template = EventTemplate({parameter_name: child})
    try:
        _, resolved = _unify_event_template_with_value(
            single_parameter_template,
            {parameter_name: value},
            bindings,
            context=f"Function {function_name!r} {source}",
        )
    except ValueError as error:
        raise ValueError(str(error)) from None
    return resolved


def _bind_function_inputs(
    *,
    function_name: str,
    input_template: EventTemplate | None,
    values: Mapping[str, Any],
    bindings: Mapping[str, int] | None = None,
) -> tuple[EventTemplate | None, dict[str, int]]:
    """Bind one call's raw inputs to its declaration template."""
    if input_template is None:
        return None, {}
    return _unify_event_template_with_value(
        input_template,
        values,
        bindings,
        context=f"Function {function_name!r} input",
    )


def _bind_planned_function_inputs(
    *,
    function_name: str,
    input_template: EventTemplate | None,
    values: Mapping[str, Any],
    lifted_names: set[str],
) -> tuple[EventTemplate | None, dict[str, int]]:
    """Bind pre-lifting values using event schemas for lifted inputs."""
    if input_template is None:
        return None, {}
    schema_values = dict(values)
    for name in lifted_names:
        value = values[name]
        try:
            lifted_template = value.event_template
        except (AttributeError, TypeError) as error:
            raise ValueError(
                f"Function {function_name!r} input {name!r} does not expose an "
                "authoritative event_template for lifting"
            ) from error
        if not isinstance(lifted_template, EventTemplate):
            raise ValueError(
                f"Function {function_name!r} input {name!r} does not expose an "
                "authoritative event_template for lifting"
            )
        schema_values[name] = lifted_template
    return _unify_event_template_with_value(
        input_template,
        schema_values,
        context=f"Function {function_name!r} input",
    )


def _validate_function_output(
    *,
    function_name: str,
    output_template: EventTemplate | None,
    result: Any,
    bindings: Mapping[str, int],
) -> EventTemplate | None:
    """Validate one native result and return the call's concrete output schema."""
    if output_template is None:
        return None
    concrete = _concretize_event_template(
        output_template,
        bindings,
        context=f"Function {function_name!r} output_template",
    )

    # Record and Distribution are schema-carrying result containers. Other
    # tracked terms remain leaf values under the default event-result contract
    # and are validated by their ValueSpec (for example, FunctionSpec).
    if isinstance(result, (Record, Distribution)):
        try:
            actual_template = cast(Any, result).event_template
        except (AttributeError, TypeError) as error:
            raise ValueError(
                f"Function {function_name!r} output does not expose an authoritative event_template"
            ) from error
        if not isinstance(actual_template, EventTemplate):
            raise ValueError(
                f"Function {function_name!r} output does not expose an authoritative event_template"
            )
        _unify_event_templates(
            concrete,
            actual_template,
            context=f"Function {function_name!r} output",
        )
        if isinstance(result, Distribution):
            conforming_template = _enrich_distribution_output_template(
                function_name=function_name,
                declared_template=concrete,
                actual_template=actual_template,
                result=result,
            )
        else:
            conforming_template = actual_template
        _unify_event_templates(
            concrete,
            conforming_template,
            context=f"Function {function_name!r} output",
        )
        _validate_function_output_template_supports(
            function_name=function_name,
            declared_template=concrete,
            actual_template=conforming_template,
        )
        if isinstance(result, Record):
            _unify_event_template_with_value(
                concrete,
                result,
                context=f"Function {function_name!r} output",
            )
            _validate_function_output_supports(
                function_name=function_name,
                template=concrete,
                value=result,
            )
        return concrete

    validation_value = result
    if not isinstance(result, Mapping):
        if len(concrete) != 1:
            raise ValueError(
                f"Function {function_name!r} returned a scalar/array for a multi-field "
                f"output_template with fields {list(concrete.keys())}"
            )
        only_path = next(iter(concrete.keys()))
        only_spec = concrete[only_path]
        assert isinstance(only_spec, ValueSpec)
        if not only_spec.is_valid(result):
            raise ValueError(
                f"Function {function_name!r} output at {only_path!r} does not conform "
                f"to its field spec ({only_spec!r})"
            )
        _validate_function_output_supports(
            function_name=function_name,
            template=concrete,
            value=result,
        )
        return concrete

    _unify_event_template_with_value(
        concrete,
        validation_value,
        context=f"Function {function_name!r} output",
    )
    _validate_function_output_supports(
        function_name=function_name,
        template=concrete,
        value=validation_value,
    )
    return concrete


def _enrich_distribution_output_template(
    *,
    function_name: str,
    declared_template: EventTemplate,
    actual_template: EventTemplate,
    result: Distribution,
) -> EventTemplate:
    """Fill missing declared metadata from a Distribution's canonical accessors."""
    requires_dtype = any(
        isinstance(spec, ArraySpec) and spec.dtype is not None
        for spec in declared_template.values()
    )
    requires_support = any(
        isinstance(spec, ArraySpec) and spec.support is not None
        for spec in declared_template.values()
    )
    dtypes = _distribution_output_metadata(result, "dtypes") if requires_dtype else None
    supports = _distribution_output_metadata(result, "supports") if requires_support else None

    def _enrich(
        declared: EventTemplate,
        actual: EventTemplate,
        prefix: str,
    ) -> EventTemplate:
        children: dict[str, EventTemplate | ValueSpec] = {}
        for name, declared_spec in declared.children.items():
            path = f"{prefix}/{name}" if prefix else name
            actual_spec = actual.children[name]
            if isinstance(declared_spec, EventTemplate):
                assert isinstance(actual_spec, EventTemplate)
                children[name] = _enrich(declared_spec, actual_spec, path)
                continue
            if not isinstance(declared_spec, ArraySpec) or not isinstance(actual_spec, ArraySpec):
                assert isinstance(actual_spec, ValueSpec)
                children[name] = actual_spec
                continue

            dtype = actual_spec.dtype
            if declared_spec.dtype is not None and dtype is None:
                if dtypes is None or path not in dtypes:
                    raise ValueError(
                        f"Function {function_name!r} output at {path!r} does not expose "
                        "authoritative dtype metadata"
                    )
                dtype = dtypes[path]

            support = actual_spec.support
            if declared_spec.support is not None and support is None:
                if supports is None or path not in supports:
                    raise ValueError(
                        f"Function {function_name!r} output at {path!r} does not expose "
                        "authoritative support metadata"
                    )
                support = supports[path]
            children[name] = ArraySpec(
                actual_spec.shape,
                dtype=dtype,
                support=support,
            )
        return EventTemplate(children)

    return _enrich(declared_template, actual_template, "")


def _validate_function_output_template_supports(
    *,
    function_name: str,
    declared_template: EventTemplate,
    actual_template: EventTemplate,
) -> None:
    """Validate authoritative output-support subset relationships."""
    for path, declared_spec in declared_template.items():
        actual_spec = actual_template[path]
        if (
            isinstance(declared_spec, ArraySpec)
            and isinstance(actual_spec, ArraySpec)
            and declared_spec.support is not None
            and actual_spec.support is not None
            and not _supports_compatible(actual_spec.support, declared_spec.support)
        ):
            raise ValueError(
                f"Function {function_name!r} output/{path} support {actual_spec.support!r} "
                f"does not conform to {declared_spec.support!r}"
            )


def _distribution_output_metadata(
    result: Distribution,
    attribute: str,
) -> Mapping[str, Any] | None:
    """Read one optional canonical per-field Distribution metadata mapping."""
    try:
        metadata = getattr(result, attribute)
    except (AttributeError, NotImplementedError, TypeError):
        return None
    return metadata if isinstance(metadata, Mapping) else None


def _validate_function_output_supports(
    *,
    function_name: str,
    template: EventTemplate,
    value: Any,
) -> None:
    """Validate declared ArraySpec supports outside JAX tracing."""
    children = getattr(value, "children", None)
    if not isinstance(children, Mapping):
        children = value if isinstance(value, Mapping) else None
    if children is None:
        only_path = next(iter(template.keys()))
        only_spec = template[only_path]
        assert isinstance(only_spec, ValueSpec)
        _validate_function_output_leaf_support(
            function_name=function_name,
            path=only_path,
            spec=only_spec,
            value=value,
        )
        return

    def _walk(expected: EventTemplate, actual: Mapping[str, Any], prefix: str) -> None:
        for name, spec in expected.children.items():
            path = f"{prefix}/{name}" if prefix else name
            child = actual[name]
            if isinstance(spec, EventTemplate):
                nested = getattr(child, "children", None)
                if not isinstance(nested, Mapping):
                    nested = child
                _walk(spec, nested, path)
                continue
            _validate_function_output_leaf_support(
                function_name=function_name,
                path=path,
                spec=spec,
                value=child,
            )

    _walk(template, children, "")


def _validate_function_output_leaf_support(
    *,
    function_name: str,
    path: str,
    spec: ValueSpec,
    value: Any,
) -> None:
    """Validate one declared ArraySpec support outside JAX tracing."""
    if not isinstance(spec, ArraySpec) or spec.support is None:
        return
    support = spec.support
    membership = jnp.all(support.check(value))
    if not bool(membership):
        raise ValueError(
            f"Function {function_name!r} output at {path!r} does not conform "
            f"to declared support {support!r}"
        )


def _wrap_declared_function_output(
    result: Any,
    *,
    function_name: str,
    output_template: EventTemplate,
) -> Record | Distribution:
    """Wrap a validated event result under its authoritative template.

    Schema-carrying Record and Distribution results retain their structure.
    Other tracked terms are event leaves until #369 supplies an explicit
    term-result plan.
    """
    if isinstance(result, (Record, Distribution)):
        return result
    if isinstance(result, Mapping):
        fields = result
    else:
        only_path = next(iter(output_template.keys()))
        fields = {only_path: result}
    return Record(
        function_name,
        fields,
        event_template=output_template,
        name_is_auto=True,
    )
