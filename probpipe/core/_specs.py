"""Shared kind specs and input/output declarations (design II.1–II.2).

An output's component names describe its public interface independently of
the label or identity of the value satisfying the declaration.
"""

from __future__ import annotations

import keyword
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from ._kind_specs import FunctionSpec
from ._record_spec import NumericRecordSpec, RecordSpec
from ._spec_base import (
    NumericArraySpec,
    NumericSpec,
    OpaqueSpec,
    TermSpec,
    _require_hashable,
    _unify_specs,
)

__all__ = [
    "FunctionSpec",
    "InputSpec",
    "NumericArraySpec",
    "NumericRecordSpec",
    "NumericSpec",
    "OpaqueSpec",
    "OutputSpec",
    "RecordSpec",
    "TermSpec",
]


def _check_component(name: str, spec: TermSpec | None, *, allow_hole: bool = False) -> None:
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"component names must be Python identifiers, got {name!r}")
    if not isinstance(spec, TermSpec) and not (allow_hole and spec is None):
        raise TypeError(f"component {name!r} must have a TermSpec, got {type(spec).__name__}")
    _require_hashable(spec, context=f"Component {name!r} spec")


@dataclass(frozen=True, init=False, eq=False)
class InputSpec(Mapping[str, TermSpec]):
    """An immutable, flat mapping of input slot names to term specs.

    Parameters
    ----------
    slots : Mapping[str, TermSpec], optional
        A positional mapping of slots, in iteration order. A nested record is
        one slot carrying a RecordSpec.
    **components : TermSpec
        Alternatively, named input slots, including RecordSpec-valued slots.
        No constructor keyword is reserved.

    Raises
    ------
    TypeError
        If a non-None positional argument is not a Mapping, mapping and keyword
        forms are combined, or a slot value is not a hashable TermSpec.
        Type holes (None slot values) are not admitted.
    ValueError
        If a slot name is not a non-keyword Python identifier, including path keys.
    """

    _slots: dict[str, TermSpec]

    def __init__(
        self, slots: Mapping[str, TermSpec] | None = None, /, **components: TermSpec
    ) -> None:
        if slots is not None:
            if not isinstance(slots, Mapping) or components:
                raise TypeError("InputSpec expects a mapping or keyword slots, not both")
        else:
            slots = components
        for name, spec in slots.items():
            _check_component(name, spec)
        object.__setattr__(self, "_slots", dict(slots))

    def __getitem__(self, key: str) -> TermSpec:
        return self._slots[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._slots)

    def __len__(self) -> int:
        return len(self._slots)

    def __hash__(self) -> int:
        return hash(frozenset(self._slots.items()))

    @property
    def free_dims(self) -> frozenset[str]:
        """The shared dimension scope of all input slots."""
        return frozenset().union(*(spec.free_dims for spec in self._slots.values()))

    @property
    def is_concrete(self) -> bool:
        """Whether all slots have concrete dimensions."""
        return not self.free_dims

    def with_dim_sizes(self, **sizes: int) -> InputSpec:
        """Return the slots with supplied sizes substituted in their shared scope."""
        return InputSpec({name: spec.with_dim_sizes(**sizes) for name, spec in self._slots.items()})

    def with_dim_names(self, **names: str) -> InputSpec:
        """Return the slots with simultaneous symbolic-dimension renaming."""
        return InputSpec({name: spec.with_dim_names(**names) for name, spec in self._slots.items()})

    def bind_dims_from_value(self, value: Mapping[str, object]) -> InputSpec:
        """Bind all slots against named values in one shared dimension scope.

        Raises ValueError for missing/extra slots or conflicting sizes.
        """
        if not isinstance(value, Mapping):
            raise TypeError("InputSpec.bind_dims_from_value expects a mapping")
        if self._slots.keys() != value.keys():
            raise ValueError(f"InputSpec slots {list(self)} do not match values {list(value)}")
        bindings: dict[str, int] = {}
        for name, spec in self._slots.items():
            spec._bind_dims_from_value(value[name], bindings, f"InputSpec/{name}")
        return InputSpec(
            {name: spec._substitute_dims(bindings) for name, spec in self._slots.items()}
        )

    def bind_dims_from_spec(self, other: InputSpec) -> InputSpec:
        """Bind against another input declaration in one shared dimension scope."""
        if not isinstance(other, InputSpec):
            raise TypeError("InputSpec.bind_dims_from_spec expects an InputSpec")
        if self._slots.keys() != other._slots.keys():
            raise ValueError(f"InputSpec slots {list(self)} do not match slots {list(other)}")
        bindings: dict[str, int] = {}
        for name, spec in self._slots.items():
            _unify_specs(spec, other[name], bindings, f"InputSpec/{name}")
        return InputSpec(
            {name: spec._substitute_dims(bindings) for name, spec in self._slots.items()}
        )


@dataclass(frozen=True, init=False)
class OutputSpec:
    """A whole returned term or the immediate components of a returned record.

    Parameters
    ----------
    *args : RecordSpec
        The positional form exposes the record's immediate fields, even when
        there is only one field or no fields. Nested records remain components.
    **components : TermSpec or None
        One keyword names the whole returned term. Its spec may be None, a type
        hole for a later declaration. Two or more keywords expose a record;
        every component must then have a spec. No keyword is reserved.

    Raises
    ------
    TypeError
        If the positional form is not exactly one RecordSpec, forms are mixed,
        or a component lacks a spec outside the single-keyword hole form.
    ValueError
        If no declaration is given or a component name is not an identifier.

    Notes
    -----
    Only the underlying declaration is stored. ``spec`` and ``components`` are
    derived views. A single named array stays an array; no single-field record
    is inserted. To fill a hole, construct a new single-keyword declaration
    with the same component name and the now-known term spec.
    """

    _component_name: str | None
    _term_spec: TermSpec | None

    def __init__(self, *args: RecordSpec, **components: TermSpec | None) -> None:
        if args:
            if len(args) != 1 or not isinstance(args[0], RecordSpec) or components:
                raise TypeError(
                    "OutputSpec expects one positional RecordSpec or keyword components"
                )
            spec = args[0]
            for name, child in spec.children.items():
                _check_component(name, child)
            name = None
        elif len(components) == 1:
            name, spec = next(iter(components.items()))
            _check_component(name, spec, allow_hole=True)
        elif components:
            for name, child in components.items():
                _check_component(name, child)
            name, spec = None, RecordSpec(components)
        else:
            raise ValueError("OutputSpec requires at least one keyword or an explicit RecordSpec")
        object.__setattr__(self, "_component_name", name)
        object.__setattr__(self, "_term_spec", spec)

    @property
    def spec(self) -> TermSpec | None:
        """The returned term's kind and structure, or None for a pending hole."""
        return self._term_spec

    @property
    def components(self) -> Mapping[str, TermSpec | None]:
        """The ordered immediate components, as a read-only derived mapping."""
        if self._component_name is not None:
            return MappingProxyType({self._component_name: self._term_spec})
        return cast(RecordSpec, self._term_spec).children

    def _with_spec(self, spec: TermSpec | None) -> OutputSpec:
        if self._component_name is not None:
            return OutputSpec(**{self._component_name: spec})
        return OutputSpec(cast(RecordSpec, spec))

    @property
    def is_concrete(self) -> bool:
        """Whether the return kind is known and all its dimensions are concrete."""
        return self.spec is not None and self.spec.is_concrete

    def with_dim_sizes(self, **sizes: int) -> OutputSpec:
        """Substitute dimensions while preserving component exposure and holes."""
        return self._with_spec(None if self.spec is None else self.spec.with_dim_sizes(**sizes))

    def with_dim_names(self, **names: str) -> OutputSpec:
        """Rename dimensions while preserving component exposure and holes."""
        return self._with_spec(None if self.spec is None else self.spec.with_dim_names(**names))


def _components_record(declaration: OutputSpec) -> RecordSpec:
    """The record of *declaration*'s components, one field per component.

    An exposed record is that record, and a whole term is a one-field record
    under its component, which is how a model or a posterior names the
    parameters of one draw.
    """
    if declaration._component_name is None:
        return cast(RecordSpec, declaration.spec)
    return RecordSpec(**{declaration._component_name: declaration.spec})
