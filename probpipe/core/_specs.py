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
from ._repr import call_repr, term_repr
from ._spec_base import (
    NumericArraySpec,
    NumericSpec,
    OpaqueSpec,
    TermSpec,
    _known_type,
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


def _check_component_name(name: str, *, context: str = "component names") -> None:
    """Check that *name* follows the rule for component names: a non-empty string without ``/``.

    *context* is the subject of the error message, such as ``"level names"``.

    Raises
    ------
    ValueError
        If *name* is not a string, is empty, or contains ``/``.
    """
    # A component follows the rule for a record's field names.
    if not isinstance(name, str) or not name or _PATH_SEP in name:
        raise ValueError(f"{context} must be non-empty and contain no {_PATH_SEP!r}, got {name!r}")


def _check_component(name: str, spec: TermSpec | None, *, allow_hole: bool = False) -> None:
    _check_component_name(name)
    _check_term(name, spec, allow_hole=allow_hole)


def _check_term(name: str, spec: TermSpec | None, *, allow_hole: bool) -> None:
    if not isinstance(spec, TermSpec) and not (allow_hole and spec is None):
        raise TypeError(f"component {name!r} must have a TermSpec, got {type(spec).__name__}")
    _require_hashable(spec, context=f"Component {name!r} spec")


@dataclass(frozen=True, init=False, eq=False)
class InputSpec(Mapping[str, TermSpec]):
    """An immutable, flat mapping of input slot names to term specs.

    A slot is required or optional. A binding may omit an optional slot, and the
    map-like kind then uses its default; :meth:`with_optional` marks slots
    optional, and the optional slots are part of the declaration, so they take
    part in equality and hashing.

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
    _optional: frozenset[str]

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
        object.__setattr__(self, "_optional", frozenset())

    def __getitem__(self, key: str) -> TermSpec:
        return self._slots[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._slots)

    def __len__(self) -> int:
        return len(self._slots)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InputSpec):
            return self._slots == other._slots and self._optional == other._optional
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash((frozenset(self._slots.items()), self._optional))

    def __repr__(self) -> str:
        """Each slot as a keyword, as the constructor takes it, then the optional slots."""
        text = term_repr("InputSpec", None, [(name, repr(spec)) for name, spec in self.items()])
        if not self._optional:
            return text
        names = ", ".join(repr(name) for name in self if name in self._optional)
        return f"{text}.with_optional({names})"

    @property
    def optional(self) -> frozenset[str]:
        """The slots a binding may omit, which then take their defaults."""
        return self._optional

    @property
    def required(self) -> tuple[str, ...]:
        """The slots a binding must supply, in slot order."""
        return tuple(name for name in self._slots if name not in self._optional)

    def with_optional(self, *names: str) -> InputSpec:
        """The slots with *names* optional, in addition to the slots already optional.

        Raises
        ------
        KeyError
            If a name is not a slot.
        """
        unknown = [name for name in names if name not in self._slots]
        if unknown:
            raise KeyError(f"{unknown} are not slots of {list(self._slots)}")
        return self._with_slots(self._slots, self._optional | frozenset(names))

    def without(self, *names: str) -> InputSpec:
        """The slots other than *names*, each optional slot staying optional."""
        return self._with_slots(
            {name: spec for name, spec in self._slots.items() if name not in names}, self._optional
        )

    def _with_slots(self, slots: Mapping[str, TermSpec], optional: frozenset[str]) -> InputSpec:
        """An input declaration of *slots*, those of *optional* among them optional."""
        result = InputSpec(slots)
        object.__setattr__(result, "_optional", optional & frozenset(slots))
        return result

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
        return self._with_slots(
            {name: spec.with_dim_sizes(**sizes) for name, spec in self._slots.items()},
            self._optional,
        )

    def with_dim_names(self, **names: str) -> InputSpec:
        """Return the slots with simultaneous symbolic-dimension renaming."""
        return self._with_slots(
            {name: spec.with_dim_names(**names) for name, spec in self._slots.items()},
            self._optional,
        )

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> InputSpec:
        """The slots with *bindings* substituted in their shared scope."""
        return self._with_slots(
            {name: spec._substitute_dims(bindings) for name, spec in self._slots.items()},
            self._optional,
        )

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
        return self._substitute_dims(bindings)

    def bind_dims_from_spec(self, other: InputSpec) -> InputSpec:
        """Bind against another input declaration in one shared dimension scope."""
        if not isinstance(other, InputSpec):
            raise TypeError("InputSpec.bind_dims_from_spec expects an InputSpec")
        if self._slots.keys() != other._slots.keys():
            raise ValueError(f"InputSpec slots {list(self)} do not match slots {list(other)}")
        bindings: dict[str, int] = {}
        for name, spec in self._slots.items():
            _unify_specs(spec, other[name], bindings, f"InputSpec/{name}")
        return self._substitute_dims(bindings)


#: The spec each kind of exposed term takes, by the kind :func:`_exposed_kind` names.
_EXPOSED_SPECS = {
    "record": "RecordSpec",
    "law": "DistributionSpec or ConditionalDistributionSpec",
    "batch": "BatchSpec of a record or a law",
}


def _exposed_kind(spec: object) -> str | None:
    """The kind of term the positional form exposes for *spec*, or None when it exposes none.

    A record exposes its immediate fields, a law or a kernel its event's
    components, and a batch its element's components.
    """
    from ._batch import BatchSpec

    if isinstance(spec, RecordSpec):
        return "record"
    if isinstance(spec, TermSpec) and isinstance(getattr(spec, "event_spec", None), OutputSpec):
        return "law"
    if isinstance(spec, BatchSpec) and _exposed_kind(spec.element_spec) is not None:
        return "batch"
    return None


def _exposed_components(spec: TermSpec) -> Mapping[str, TermSpec | None]:
    """The components an exposed *spec* declares, by the rule of :func:`_exposed_kind`."""
    kind = _exposed_kind(spec)
    if kind == "record":
        return cast(RecordSpec, spec).children
    if kind == "law":
        return cast(OutputSpec, spec.event_spec).components  # type: ignore[attr-defined]
    return _exposed_components(spec.element_spec)  # type: ignore[attr-defined]


@dataclass(frozen=True, init=False)
class OutputSpec:
    """One returned term and its packaging: a whole term, or an exposed term.

    Parameters
    ----------
    *args : RecordSpec, DistributionSpec, ConditionalDistributionSpec, or BatchSpec
        The positional form exposes the components of the term it declares: a
        record's immediate fields, a law's or a kernel's event components, or
        the components of a batch's element, which is a record or a law.
    **term : TermSpec or None
        One keyword names the whole returned term. Its spec may be None, which
        marks a type hole that a producer fills with :meth:`with_spec`.

    Raises
    ------
    TypeError
        If the positional form is not exactly one spec that exposes components,
        forms are mixed, more than one keyword is given, or a record field lacks
        a spec.
    ValueError
        If no declaration is given, or a component name is empty or contains ``/``.

    Notes
    -----
    Only the underlying declaration is stored, and ``spec``, ``components``, and
    ``exposes_record`` are derived views of it. The form alone decides the
    packaging. An exposed record's components are fields of the returned value,
    and an exposed law's are the components of its draws, so the returned term
    is the law itself.
    """

    _component_name: str | None
    _term_spec: TermSpec | None

    def __init__(self, *args: TermSpec, **term: TermSpec | None) -> None:
        if args:
            if len(args) != 1 or term or _exposed_kind(args[0]) is None:
                raise TypeError(
                    "OutputSpec expects one positional RecordSpec, DistributionSpec, "
                    "ConditionalDistributionSpec, or BatchSpec of one of them, or one keyword"
                )
            spec = args[0]
            if isinstance(spec, RecordSpec):
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

    def __repr__(self) -> str:
        """The declaration as the constructor takes it: one keyword, or the exposed record."""
        if self._component_name is None:
            return call_repr("OutputSpec", [repr(self._term_spec)])
        return term_repr("OutputSpec", None, [(self._component_name, repr(self._term_spec))])

    @property
    def spec(self) -> TermSpec | None:
        """The returned term's kind and structure, or None for a pending hole."""
        return self._term_spec

    @property
    def components(self) -> Mapping[str, TermSpec | None]:
        """The ordered immediate components, as a read-only derived mapping."""
        if self._component_name is not None:
            return MappingProxyType({self._component_name: self._term_spec})
        return _exposed_components(cast(TermSpec, self._term_spec))

    @property
    def exposes_record(self) -> bool:
        """Whether the components are the fields of an exposed record."""
        return self._component_name is None and isinstance(self._term_spec, RecordSpec)

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
        *spec*, and the result stores their unification under the declared names
        and packaging. A numeric array keeps its declared dtype and support and
        takes *spec*'s where the declaration leaves them unset, and a declared
        symbolic dimension binds to the size *spec* gives. A record unifies field
        by field, and an opaque spec takes the known type and ``meta``. Any other
        kind takes *spec*.

        Raises
        ------
        TypeError
            If this declaration exposes a term and *spec* is not a spec of that
            kind, such as a ``RecordSpec`` for an exposed record.
        ValueError
            If a declared spec does not unify with *spec*.
        """
        kind = None if self._component_name is not None else _exposed_kind(self._term_spec)
        if kind is not None and _exposed_kind(spec) != kind:
            raise TypeError(
                f"an exposed {kind} declaration needs a {_EXPOSED_SPECS[kind]}, "
                f"got {type(spec).__name__}"
            )
        if self._term_spec is not None:
            bindings: dict[str, int] = {}
            root = _root_path(self._component_name, kind)
            _unify_specs(self._term_spec, spec, bindings, root)
            spec = _unification(self._term_spec, spec, bindings)
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
            return OutputSpec(_renamed_exposed(cast(TermSpec, spec), mapping, kwargs))
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
        return OutputSpec(cast(TermSpec, spec))

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


def _root_path(component: str | None, kind: str | None) -> str:
    """The path a unification error names a declaration's root by.

    A whole term's paths start at its *component*, and an exposed record's
    fields are its components, so its root is the empty path. An exposed law or
    batch has no path above its event's components, so its root is named by its
    *kind*.
    """
    if component is not None:
        return component
    return "" if kind == "record" else f"the exposed {kind}"


def _unification(declared: TermSpec, produced: TermSpec, bindings: Mapping[str, int]) -> TermSpec:
    """The unification of a declared spec with the produced one it unifies with.

    *bindings* holds the sizes the unification bound to symbolic dimensions.
    The rule is :meth:`OutputSpec.with_spec`'s, and the result is *produced*
    when the two agree.
    """
    if isinstance(declared, NumericArraySpec) and isinstance(produced, NumericArraySpec):
        unified = NumericArraySpec(
            declared._substitute_dims(bindings).shape,
            dtype=declared.dtype if declared.dtype is not None else produced.dtype,
            support=declared.support if declared.support is not None else produced.support,
        )
    elif isinstance(declared, RecordSpec) and isinstance(produced, RecordSpec):
        unified = RecordSpec(
            {
                name: _unification(field, produced.children[name], bindings)
                for name, field in declared.children.items()
            }
        )
    elif isinstance(declared, OpaqueSpec) and isinstance(produced, OpaqueSpec):
        unified = _known_type(declared, produced)
    else:
        return produced
    return produced if unified == produced else unified


def _renamed_exposed(
    spec: TermSpec, mapping: Mapping[str, str] | None, kwargs: Mapping[str, str]
) -> TermSpec:
    """The exposed *spec* with the nodes its components start renamed, as ``with_path_names`` reads them.

    A record renames its own paths, a law or a kernel its event declaration's,
    and a batch its element's, so the term keeps its kind.
    """
    from dataclasses import replace

    kind = _exposed_kind(spec)
    if kind == "record":
        return cast(RecordSpec, spec).with_path_names(mapping, **kwargs)
    if kind == "law":
        event = cast(OutputSpec, spec.event_spec)  # type: ignore[attr-defined]
        return replace(spec, event_spec=event.with_path_names(mapping, **kwargs))  # type: ignore[type-var]
    element = _renamed_exposed(spec.element_spec, mapping, kwargs)  # type: ignore[attr-defined]
    return replace(spec, element_spec=element)  # type: ignore[type-var]


def _components_record(declaration: OutputSpec) -> RecordSpec:
    """The record of *declaration*'s components, one field per component.

    An exposed record is that record, and a whole term is a one-field record
    under its component, which is how a model or a posterior names the
    parameters of one draw.

    Raises
    ------
    TypeError
        If *declaration* exposes a spec that is not a ``RecordSpec``, since a
        law's or a batch's components are no record's fields.
    """
    if declaration.exposes_record:
        if not isinstance(declaration.spec, RecordSpec):
            raise TypeError(f"an exposed declaration holds a RecordSpec, got {declaration.spec!r}")
        return declaration.spec
    if declaration._component_name is None:
        raise TypeError(
            f"an exposed declaration holds a RecordSpec, got {declaration.spec!r}, whose "
            f"components are no record's fields"
        )
    return RecordSpec(dict(declaration.components))


def _unnamed_declaration(declaration: OutputSpec) -> tuple[object, ...]:
    """*declaration*'s packaging and its components' specs in order, without their names.

    A function's fingerprint and its replay anchor record this form, so a
    rename of the components of its output keeps both.
    """
    if declaration._component_name is not None:
        return ("whole term", (declaration.spec,))
    return ("exposed", *_unnamed_exposed(cast(TermSpec, declaration.spec)))


def _unnamed_exposed(spec: TermSpec) -> tuple[object, ...]:
    """An exposed *spec* without its component names: its kind and what it holds besides them."""
    kind = _exposed_kind(spec)
    if kind == "record":
        return ("record", tuple(_exposed_components(spec).values()))
    if kind == "law":
        given = getattr(spec, "given_spec", None)
        event = cast(OutputSpec, spec.event_spec)  # type: ignore[attr-defined]
        return ("law", type(spec).__qualname__, given, _unnamed_declaration(event))
    return (
        "batch",
        spec.axis_groups,  # type: ignore[attr-defined]
        spec.level_names,  # type: ignore[attr-defined]
        _unnamed_exposed(spec.element_spec),  # type: ignore[attr-defined]
    )


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
