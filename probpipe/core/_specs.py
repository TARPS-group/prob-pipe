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

from ._record_spec import NumericRecordSpec, RecordSpec
from ._spec_base import (
    NumericArraySpec,
    NumericSpec,
    OpaqueSpec,
    TermSpec,
    _require_hashable,
    _unify_specs,
)
from .constraints import _supports_compatible
from .named_tree import _PATH_SEP

__all__ = [
    "InputSpec",
    "NumericArraySpec",
    "NumericRecordSpec",
    "NumericSpec",
    "OpaqueSpec",
    "OutputSpec",
    "RecordSpec",
    "TermSpec",
]


def _check_slot(name: str, spec: TermSpec) -> None:
    # An input slot is a Python parameter, so its name is an identifier.
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"input slot names must be Python identifiers, got {name!r}")
    _check_term(name, spec, allow_hole=False)


def _check_component(name: str, spec: TermSpec | None, *, allow_hole: bool = False) -> None:
    # A component follows the rule for a record's field names.
    if not isinstance(name, str) or not name or _PATH_SEP in name:
        raise ValueError(
            f"component names must be non-empty and contain no {_PATH_SEP!r}, got {name!r}"
        )
    _check_term(name, spec, allow_hole=allow_hole)


def _check_term(name: str, spec: TermSpec | None, *, allow_hole: bool) -> None:
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
            _check_slot(name, spec)
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
    """One returned term and its packaging: a whole term, or an exposed record.

    Parameters
    ----------
    *args : RecordSpec
        The positional form exposes the record's immediate fields as the
        components.
    **term : TermSpec or None
        One keyword names the whole returned term. Its spec may be None, which
        marks a type hole that a producer fills with :meth:`with_spec`.

    Raises
    ------
    TypeError
        If the positional form is not exactly one RecordSpec, forms are mixed,
        more than one keyword is given, or a record field lacks a spec.
    ValueError
        If no declaration is given, or a component name is empty or contains ``/``.

    Notes
    -----
    Only the underlying declaration is stored, and ``spec``, ``components``, and
    ``exposes_record`` are derived views of it. The form alone decides the
    packaging.
    """

    _component_name: str | None
    _term_spec: TermSpec | None

    def __init__(self, *args: RecordSpec, **term: TermSpec | None) -> None:
        if args:
            if len(args) != 1 or not isinstance(args[0], RecordSpec) or term:
                raise TypeError("OutputSpec expects one positional RecordSpec or one keyword")
            spec = args[0]
            for name, child in spec.children.items():
                _check_component(name, child)
            name = None
        elif len(term) == 1:
            name, spec = next(iter(term.items()))
            _check_component(name, spec, allow_hole=True)
        elif term:
            raise TypeError(
                f"OutputSpec takes one keyword, which names the whole term, but got "
                f"{sorted(term)}; declare an exposed record as OutputSpec(RecordSpec(...))"
            )
        else:
            raise ValueError("OutputSpec requires one keyword or an explicit RecordSpec")
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

    @property
    def exposes_record(self) -> bool:
        """Whether the components are the fields of an exposed record."""
        return self._component_name is None

    @classmethod
    def default(cls, spec: TermSpec, *, component: str) -> OutputSpec:
        """The declaration a producer uses when it is given none.

        It is ``OutputSpec(spec)`` if *spec* is a ``RecordSpec``, whose fields
        become the components, and ``OutputSpec(**{component: spec})``
        otherwise, so *component* is the producer's default component.

        Raises
        ------
        ValueError
            If *spec* is not a record and *component* is not a valid component
            name.
        """
        if isinstance(spec, RecordSpec):
            return cls(spec)
        return cls(**{component: spec})

    def with_spec(self, spec: TermSpec) -> OutputSpec:
        """This declaration with its type set to *spec*.

        A pending hole is filled with *spec*. A declared spec must unify with
        *spec*, which then replaces it, so the result carries what the producer
        returns.

        Raises
        ------
        TypeError
            If this declaration exposes a record and *spec* is not a ``RecordSpec``.
        ValueError
            If a declared spec does not unify with *spec*.
        """
        if self.exposes_record and not isinstance(spec, RecordSpec):
            raise TypeError(
                f"an exposed record declaration needs a RecordSpec, got {type(spec).__name__}"
            )
        if self._term_spec is not None:
            label = "the exposed record" if self.exposes_record else repr(self._component_name)
            _unify_specs(self._term_spec, spec, {}, f"Declared component {label}")
        return self._with_spec(spec)

    def with_path_names(
        self, mapping: Mapping[str, str] | None = None, /, **kwargs: str
    ) -> OutputSpec:
        """Rename or move nodes of the declaration by their paths, ``old -> new``.

        A path starts with a component: an exposed record's paths are the paths
        of its record, and a whole term's are its component followed by the
        paths within its term. Each key is the exact path of a node and each
        target its new exact path, under the rule of
        :meth:`~probpipe.core.named_tree.NamedTree.with_path_names`. The result
        keeps the packaging. An exposed record stays exposed, so a move may
        create or remove a component. A whole term's component is renamed in
        place, and the term's fields stay under it, so a field moves only within
        the component.

        Raises
        ------
        KeyError
            If a key is not a path of the declaration.
        ValueError
            As :meth:`~probpipe.core.named_tree.NamedTree.with_path_names` raises
            it, or if a whole term's component moves into a group or one of its
            fields moves out of it.
        """
        spec = self._term_spec
        component = self._component_name
        if component is None:
            return OutputSpec(cast(RecordSpec, spec).with_path_names(mapping, **kwargs))
        # A whole term's paths are those of the record of its one component.
        components = RecordSpec({component: OpaqueSpec() if spec is None else spec})
        renames = components._resolve_path_renames(mapping, kwargs)
        renamed_component = renames.get(component, component)
        if _PATH_SEP in renamed_component:
            raise ValueError(
                f"with_path_names() keeps the packaging, so the whole term's component "
                f"{component!r} is renamed in place, not moved to {renamed_component!r}"
            )
        for source, target in renames.items():
            if source != component and not target.startswith(renamed_component + _PATH_SEP):
                raise ValueError(
                    f"with_path_names() keeps the packaging, so the fields of the whole term "
                    f"{renamed_component!r} stay under it; {source!r} cannot move to {target!r}"
                )
        ((name, term),) = components.with_path_names(renames).children.items()
        return OutputSpec(**{name: None if spec is None else term})

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

    Raises
    ------
    TypeError
        If *declaration* exposes a spec that is not a ``RecordSpec``.
    """
    if declaration.exposes_record:
        if not isinstance(declaration.spec, RecordSpec):
            raise TypeError(f"an exposed declaration holds a RecordSpec, got {declaration.spec!r}")
        return declaration.spec
    return RecordSpec(dict(declaration.components))


def _check_output_template(record: RecordSpec, template: RecordSpec, path: str) -> None:
    """Raise ``ValueError`` unless *record* conforms to a Function's output *template*.

    The fields and shapes must conform, a dtype the template sets admits a
    same-kind cast, and a support the template sets must hold the record's.
    Metadata the template leaves unset matches any, so a law's full declaration
    meets a template that states only shapes.
    """
    _unify_specs(template, record, {}, path)
    for leaf, declared in template.items():
        actual = record[leaf]
        if (
            isinstance(declared, NumericArraySpec)
            and isinstance(actual, NumericArraySpec)
            and declared.support is not None
            and actual.support is not None
            and not _supports_compatible(actual.support, declared.support)
        ):
            raise ValueError(
                f"{path}/{leaf} support {actual.support!r} does not conform to {declared.support!r}"
            )
