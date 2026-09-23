"""The callable spec, with dimension binding through its declared sides."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._record_spec import RecordSpec
from ._spec_base import TermSpec, _unify_specs


@dataclass(frozen=True, init=False)
class FunctionSpec(TermSpec):
    """A callable-kind spec, optionally declaring its input and returned kind.

    Parameters
    ----------
    input_template : RecordSpec or None
        The current named input schema; None leaves the inputs unspecified.
    output_spec : TermSpec or None
        The kind of the returned value, stored directly. An array spec declares
        an array, and a record spec declares a record regardless of field count.
        None leaves the output unspecified.

    Raises
    ------
    TypeError
        If a specified input is not a RecordSpec or an output is not a TermSpec.

    Notes
    -----
    ``is_valid`` checks callability alone, accepting ordinary Python callables
    as well as Function values. It never executes the callable or certifies
    compatibility with the declared inputs and output. ``FunctionSpec()`` leaves
    both sides unspecified; either side can also be declared on its own.

    ``bind_dims_from_value`` reads available input/output declarations from a
    callable; ``bind_dims_from_spec`` reads them from another FunctionSpec. The
    two sides bind independently. A side left unspecified by either declaration
    adds no bindings, so a bare callable leaves all dimensions symbolic.
    Shared symbols across input and output belong to one scope; incompatible
    declared kinds, structures, dtypes, or sizes raise ValueError. Binding a
    non-callable value also raises ValueError. Neither binding method infers
    callable variance or evaluates the function.

    Dimension transforms and binding return new specs, preserving the declared
    kinds and metadata. An array output stays an array declaration; a record
    output stays a record declaration, including when it has one field.

    Examples
    --------
    >>> from probpipe import FunctionSpec, NumericArraySpec, RecordSpec
    >>> declared = FunctionSpec(RecordSpec(x=("n",)), NumericArraySpec(("n",)))
    >>> actual = FunctionSpec(RecordSpec(x=(3,)), NumericArraySpec((3,)))
    >>> declared.bind_dims_from_spec(actual).output_spec.shape
    (3,)
    >>> declared.bind_dims_from_value(lambda x: x).free_dims
    frozenset({'n'})
    """

    input_template: RecordSpec | None
    output_spec: TermSpec | None

    def __init__(
        self,
        input_template: RecordSpec | None = None,
        output_spec: TermSpec | None = None,
    ) -> None:
        if input_template is not None and not isinstance(input_template, RecordSpec):
            raise TypeError(
                f"FunctionSpec.input_template must be None or a RecordSpec, "
                f"got {type(input_template).__name__}"
            )
        if output_spec is not None and not isinstance(output_spec, TermSpec):
            raise TypeError(
                f"FunctionSpec.output_spec must be None or a TermSpec, got {type(output_spec).__name__}"
            )
        object.__setattr__(self, "input_template", input_template)
        object.__setattr__(self, "output_spec", output_spec)

    @property
    def free_dims(self) -> frozenset[str]:
        """The unbound dimensions of both declared sides, in one scope.

        A name shared between the input and the output is one dimension, which
        is how ``f(x: ("n",)) -> ("n",)`` states that it preserves a length.
        """
        dimensions: frozenset[str] = frozenset()
        if self.input_template is not None:
            dimensions |= self.input_template.free_dims
        if self.output_spec is not None:
            dimensions |= self.output_spec.free_dims
        return dimensions

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> FunctionSpec:
        """This spec around both substituted sides."""
        return FunctionSpec(
            None if self.input_template is None else self.input_template._substitute_dims(bindings),
            None if self.output_spec is None else self.output_spec._substitute_dims(bindings),
        )

    def _bind_dims_from_value(self, value: Any, bindings: dict[str, int], path: str) -> None:
        """Bind each declared side from the matching side of *value*'s own declaration.

        The two sides bind **independently**, which is not a conformance check on
        the callable: a function is contravariant in its input, so declaring that
        a value *is* an acceptable function is a separate question this does not
        answer. A bare callable declares nothing, so there is nothing to bind from
        and the dimensions stay free — which a caller learns from ``free_dims``
        rather than from a refusal here.
        """
        if not self.is_valid(value):
            raise ValueError(f"{path} does not conform to its field spec ({self!r})")
        if self.input_template is not None:
            actual_input = getattr(value, "input_template", None)
            if isinstance(actual_input, RecordSpec):
                _unify_specs(self.input_template, actual_input, bindings, path)
        if self.output_spec is not None:
            actual_output = getattr(value, "output_template", None)
            if isinstance(actual_output, RecordSpec):
                self._bind_output_from(actual_output, bindings, path)

    def _bind_output_from(
        self, actual_output: RecordSpec, bindings: dict[str, int], path: str
    ) -> None:
        """Bind the declared output against the schema the callable declares.

        A record declaration meets the template as a whole. Any other declaration
        describes the one value returned, so it meets the template's sole
        immediate field. A callable declaring several output fields does not
        match one.

        Temporary legacy-template adapter (#448): remove this unwrapping once
        live Functions carry OutputSpec declarations.
        """
        if isinstance(self.output_spec, RecordSpec):
            _unify_specs(self.output_spec, actual_output, bindings, path)
            return
        if len(actual_output.children) != 1:
            raise ValueError(
                f"{path} declares one output value ({self.output_spec!r}), but the callable "
                f"declares output fields {list(actual_output.children)}"
            )
        _unify_specs(self.output_spec, next(iter(actual_output.children.values())), bindings, path)

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        """Bind each declared side from the matching side of *actual*.

        The sides bind independently, as they do from a value: a side the other
        spec leaves undeclared binds nothing and stays free.
        """
        if not isinstance(actual, FunctionSpec):
            return False
        if self.input_template is not None and actual.input_template is not None:
            _unify_specs(self.input_template, actual.input_template, bindings, path)
        if self.output_spec is not None and actual.output_spec is not None:
            _unify_specs(self.output_spec, actual.output_spec, bindings, path)
        return True

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a callable.

        The input/output structure of a bare callable cannot be inspected, so
        validity is callability alone; ``input_template`` / ``output_spec``
        document the intended signature but are not checked against the value.
        """
        return callable(value)
