"""Unify Function input slots against planned lifted elements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core._spec_base import _unify_specs
from ..core._specs import InputSpec, TermSpec
from ._call import ApplicabilityError


def _lifted_element_spec(
    value: Any, *, expected: TermSpec, function_name: str, name: str
) -> TermSpec:
    """Return the declared kind of one swept element or sampled event."""
    from ..core._batch import Batch
    from ..distributions._distribution import Distribution

    if isinstance(value, Batch):
        return value.element_spec
    if isinstance(value, Distribution):
        return value.event_spec.spec
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
    """Bind pre-lifting values using event schemas for lifted inputs.

    Parameters
    ----------
    function_name : str
        The function's label, which the messages name.
    input_spec : InputSpec or None
        The declared input slots; ``None`` binds nothing.
    values : Mapping of str to Any
        The arguments by parameter name, before the call lifts any of them.
    lifted_names : set of str
        The parameters whose arguments the call lifts, each of which binds
        through its law's event or its batch's element.

    Returns
    -------
    input_spec : InputSpec or None
        The declared slots with the shared dimensions bound, or ``None`` when
        no slot is declared.
    bindings : dict of str to int
        The size bound to each shared dimension.

    Raises
    ------
    ApplicabilityError
        If the arguments' slots are not the declared ones, or an argument, the
        event of a law the call lifts, or the element of a batch it sweeps does
        not unify with its slot, the dimensions shared across slots included.
    """
    if input_spec is None:
        return None, {}
    context = f"Function {function_name!r} input"
    if input_spec.keys() != values.keys():
        raise ApplicabilityError(
            f"{context} slots {sorted(values)} do not match the declared slots {sorted(input_spec)}"
        )
    bindings: dict[str, int] = {}
    try:
        for name, expected in input_spec.items():
            path = f"{context}/{name}"
            if name in lifted_names:
                actual = _lifted_element_spec(
                    values[name], expected=expected, function_name=function_name, name=name
                )
                _unify_specs(expected, actual, bindings, path)
            else:
                expected._bind_dims_from_value(values[name], bindings, path)
    except ValueError as error:
        raise ApplicabilityError(str(error)) from error
    return input_spec.with_dim_sizes(**bindings), bindings
