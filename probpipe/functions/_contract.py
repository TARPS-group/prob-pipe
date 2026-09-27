"""Unify Function input slots against planned lifted elements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core._spec_base import _unify_specs
from ..core._specs import InputSpec, RecordSpec, TermSpec


def _lifted_element_spec(
    value: Any, *, expected: TermSpec, function_name: str, name: str
) -> TermSpec:
    """What one element of a lifted operand satisfies.

    A :class:`~probpipe.core._batch.Batch` states this uniformly in
    ``element_spec``, at every kind: a batch of records answers with a
    ``RecordSpec``, one of arrays with a ``NumericArraySpec``, one of callables
    with a ``FunctionSpec``. Reading the record-only ``event_template`` view
    instead let only a batch of records be swept by a declared function, and made
    a batch of records satisfy a declaration that named a bare array.

    Temporary legacy-template adapter: live distributions still carry
    event templates. The current sampling lift passes a sole immediate field as
    a bare value when the callable declares a leaf, and as a record for a record declaration.
    Resolve that legacy packaging here, before strict spec unification. Batch
    element specs already name their actual kinds and need no adaptation.
    Remove this unwrapping once live distributions carry OutputSpec declarations.
    """
    from ..core._batch import Batch

    if isinstance(value, Batch):
        return value.element_spec
    template = getattr(value, "event_template", None)
    if isinstance(template, RecordSpec):
        if not isinstance(expected, RecordSpec) and len(template.children) == 1:
            return next(iter(template.children.values()))
        return template
    raise ValueError(
        f"Function {function_name!r} input {name!r} states no element specification for "
        f"lifting: a {type(value).__name__} reports neither an element_spec nor an "
        f"event_template"
    )


def _bind_planned_function_inputs(
    *,
    function_name: str,
    input_spec: InputSpec | None,
    values: Mapping[str, Any],
    lifted_names: set[str],
) -> tuple[InputSpec | None, dict[str, int]]:
    """Bind pre-lifting values using event schemas for lifted inputs."""
    if input_spec is None:
        return None, {}
    context = f"Function {function_name!r} input"
    if input_spec.keys() != values.keys():
        raise ValueError(
            f"{context} fields {sorted(values)} do not match template fields {sorted(input_spec)}"
        )
    bindings: dict[str, int] = {}
    for name, expected in input_spec.items():
        path = f"{context}/{name}"
        if name in lifted_names:
            actual = _lifted_element_spec(
                values[name], expected=expected, function_name=function_name, name=name
            )
            _unify_specs(expected, actual, bindings, path)
        else:
            expected._bind_dims_from_value(values[name], bindings, path)
    return input_spec.with_dim_sizes(**bindings), bindings
