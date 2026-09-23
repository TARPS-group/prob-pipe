"""Distribution and callable specs, with dimension binding through their declared sides."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._record_spec import (
    RecordSpec,
    _check_kind_of,
    _schema_carried_by,
)
from ._spec_base import TermSpec, _unify_specs


@dataclass(frozen=True, init=False)
class DistributionSpec(TermSpec):
    """A distribution-kind spec whose current draw schema is a RecordSpec.

    Parameters
    ----------
    event_spec : RecordSpec
        The record schema describing one draw.

    Raises
    ------
    TypeError
        If the draw schema is not a RecordSpec.

    Notes
    -----
    ``is_valid`` requires a Distribution whose ``event_template`` equals
    ``event_spec``, including field order, dtype, and support metadata. A missing
    template, or a getter raising AttributeError or TypeError because the schema
    is unavailable, gives False; other getter errors propagate.

    ``bind_dims_from_value`` validates a concrete declaration by that same rule.
    For a symbolic declaration it instead learns sizes from the distribution's
    schema, without sampling. An unavailable schema raises ValueError.
    ``bind_dims_from_spec`` reads another DistributionSpec using declaration
    compatibility: matching structure, compatible declared dtypes, and consistent
    sizes; field order and support metadata need not be equal.

    Binding returns a new spec retaining the declared metadata. Repeated symbols
    share one scope, including surrounding records or input slots; conflicting
    sizes raise ValueError. ``with_dim_sizes`` may leave unsupplied dimensions symbolic.

    Examples
    --------
    >>> from probpipe import DistributionSpec, RecordSpec
    >>> declared = DistributionSpec(RecordSpec(x=("n",)))
    >>> bound = declared.bind_dims_from_spec(DistributionSpec(RecordSpec(x=(3,))))
    >>> bound.event_spec["x"].shape
    (3,)
    """

    event_spec: RecordSpec

    def __init__(self, event_spec: RecordSpec) -> None:
        if not isinstance(event_spec, RecordSpec):
            raise TypeError(
                f"DistributionSpec.event_spec must be a RecordSpec, got {type(event_spec).__name__}"
            )
        object.__setattr__(self, "event_spec", event_spec)

    @property
    def free_dims(self) -> frozenset[str]:
        """The unbound dimensions of the draw this declares."""
        return self.event_spec.free_dims

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> DistributionSpec:
        """This spec around a substituted event declaration."""
        return DistributionSpec(self.event_spec._substitute_dims(bindings))

    def _bind_dims_from_value(self, value: Any, bindings: dict[str, int], path: str) -> None:
        """Validate a concrete draw schema, or bind a symbolic one from *value*."""
        if not self.free_dims:
            super()._bind_dims_from_value(value, bindings, path)
            return
        actual = _schema_carried_by(value, self, path)
        _check_kind_of(DistributionSpec(actual), value, self, path)
        _unify_specs(self.event_spec, actual, bindings, path)

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        """Bind the declared draw schema against *actual*'s own."""
        if not isinstance(actual, DistributionSpec):
            return False
        _unify_specs(self.event_spec, actual.event_spec, bindings, path)
        return True

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a ``Distribution`` matching this event declaration.

        *value* must be a :class:`~probpipe.Distribution` whose own
        ``event_template`` equals the declared record template. A distribution
        that is not one, or that legitimately exposes no template — no
        ``event_template`` attribute, or a template that cannot yet be
        derived — does not satisfy the spec and returns ``False``. These are
        the only two "schema unavailable" conditions treated as a non-match;
        any *other* error raised while reading ``event_template`` signals a
        malfunctioning distribution and is left to propagate rather than being
        masked as invalid.
        """
        from ._distribution_base import Distribution

        if not isinstance(value, Distribution):
            return False
        try:
            template = value.event_template
        except (AttributeError, TypeError):
            # The two documented "schema unavailable" signals: no
            # ``event_template`` attribute (AttributeError) or a template that
            # cannot be derived (TypeError — e.g. an un-named auto-deriving
            # distribution). Both mean the value can't be certified. A
            # narrower catch than ``Exception`` on purpose: an unexpected
            # error is a bug to surface, not a silent "invalid".
            return False
        # Normalised at construction, so the declaration is always a RecordSpec.
        return template == self.event_spec


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
