"""Unify Function input slots against planned lifted elements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core._record_spec import RecordSpec
from ..core._spec_base import _unify_specs
from ..core._specs import InputSpec, TermSpec


def _lifted_element_spec(
    value: Any, *, expected: TermSpec, function_name: str, name: str
) -> TermSpec:
    """Return the declared kind of one swept element or sampled event."""
    from ..core._batch import Batch
    from ..distributions._distribution import Distribution

    if isinstance(value, Batch):
        return value.element_spec
    if isinstance(value, Distribution):
        spec = value.event_spec.spec
        if spec is None:
            raise ValueError(
                f"Function {function_name!r} input {name!r} requires a declared event term spec"
            )
        if isinstance(spec, RecordSpec):
            path = _sole_leaf_path(spec)
            if path is not None:
                return spec[path]
        return spec
    raise ValueError(
        f"Function {function_name!r} input {name!r} states no element specification for "
        f"lifting: a {type(value).__name__} reports neither an element_spec nor an "
        f"event_spec"
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
            f"{context} slots {sorted(values)} do not match declared slots {sorted(input_spec)}"
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


def _sole_leaf_path(spec: RecordSpec) -> str | None:
    """Return the sole leaf path, or None if the record has no or multiple leaves."""
    paths = list(spec.keys())
    return paths[0] if len(paths) == 1 else None
