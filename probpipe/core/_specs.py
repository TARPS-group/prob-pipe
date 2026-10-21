"""Shared kind specs and input/output specs (design II.1–II.2).

An output's output names describe its public interface independently of
the label or identity of the value satisfying the declaration.
"""

from __future__ import annotations

import keyword
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from .._messages import unknown_names
from ._record_spec import NumericRecordSpec, RecordSpec
from ._repr import call_repr, term_repr
from ._spec_base import (
    NumericArraySpec,
    NumericSpec,
    OpaqueSpec,
    TermSpec,
    _dim_renames,
    _dim_sizes,
    _known_type,
    _name_mismatch,
    _require_hashable,
    _unify_specs,
)
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


def _check_input(name: str, spec: TermSpec) -> None:
    # An input is a Python parameter, so its name is an identifier.
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"input names must be Python identifiers, got {name!r}")
    _check_term(name, spec, allow_hole=False, noun="input")


def _check_output_name(name: str, *, context: str = "output names") -> None:
    """Check that *name* follows the rule for output names: a non-empty string without ``/``.

    Parameters
    ----------
    name : str
        The output name or level name to check.
    context : str
        The subject of the error message, such as ``"level names"``.

    Raises
    ------
    ValueError
        If *name* is not a string, is empty, or contains ``/``.
    """
    # An output follows the rule for a record's field names.
    if not isinstance(name, str) or not name or _PATH_SEP in name:
        raise ValueError(f"{context} must be non-empty and contain no {_PATH_SEP!r}, got {name!r}")


def _check_output(name: str, spec: TermSpec | None, *, allow_hole: bool = False) -> None:
    _check_output_name(name)
    _check_term(name, spec, allow_hole=allow_hole, noun="component")


def _check_term(name: str, spec: TermSpec | None, *, allow_hole: bool, noun: str) -> None:
    if not isinstance(spec, TermSpec) and not (allow_hole and spec is None):
        message = f"{noun} {name!r} needs a TermSpec, got {type(spec).__name__}"
        if isinstance(spec, tuple):
            message += f"; use NumericArraySpec({spec!r}) for an array of that shape"
        raise TypeError(message)
    _require_hashable(spec, context=f"the spec of {noun} {name!r}")


@dataclass(frozen=True, init=False, eq=False)
class InputSpec(Mapping[str, TermSpec]):
    """An immutable, flat mapping of input names to term specs.

    An input is required or optional. A binding may omit an optional input, and the
    map-like kind then uses its default; :meth:`with_optional` marks inputs
    optional, and the optional inputs are part of the declaration, so they take
    part in equality and hashing.

    Parameters
    ----------
    inputs : Mapping[str, TermSpec], optional
        A positional mapping of inputs, in iteration order. A nested record is
        one input carrying a RecordSpec.
    **outputs : TermSpec
        Alternatively, named inputs, including RecordSpec-valued inputs.
        No constructor keyword is reserved.

    Raises
    ------
    TypeError
        If a non-None positional argument is not a Mapping, mapping and keyword
        forms are combined, or an input value is not a hashable TermSpec.
        Unspecified output types (None input values) are not admitted.
    ValueError
        If an input name is not a non-keyword Python identifier, including path keys.
    """

    _inputs: dict[str, TermSpec]
    _optional: frozenset[str]

    def __init__(
        self, inputs: Mapping[str, TermSpec] | None = None, /, **inputs_by_name: TermSpec
    ) -> None:
        if inputs is not None:
            if not isinstance(inputs, Mapping):
                raise TypeError(
                    f"InputSpec expects a mapping of input names to specs, got {type(inputs).__name__}"
                )
            if inputs_by_name:
                raise TypeError("pass InputSpec inputs as a mapping or as keywords, not both")
        else:
            inputs = inputs_by_name
        for name, spec in inputs.items():
            _check_input(name, spec)
        object.__setattr__(self, "_inputs", dict(inputs))
        object.__setattr__(self, "_optional", frozenset())

    def __getitem__(self, key: str) -> TermSpec:
        return self._inputs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._inputs)

    def __len__(self) -> int:
        return len(self._inputs)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InputSpec):
            return self._inputs == other._inputs and self._optional == other._optional
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash((frozenset(self._inputs.items()), self._optional))

    def __repr__(self) -> str:
        """Each input as a keyword, as the constructor takes it, then the optional inputs."""
        text = term_repr("InputSpec", None, [(name, repr(spec)) for name, spec in self.items()])
        if not self._optional:
            return text
        names = ", ".join(repr(name) for name in self if name in self._optional)
        return f"{text}.with_optional({names})"

    @property
    def optional(self) -> frozenset[str]:
        """The inputs a binding may omit, which then take their defaults."""
        return self._optional

    @property
    def required(self) -> tuple[str, ...]:
        """The inputs a binding must supply, in input order."""
        return tuple(name for name in self._inputs if name not in self._optional)

    def with_optional(self, *names: str) -> InputSpec:
        """The inputs with *names* optional, in addition to the inputs already optional.

        Parameters
        ----------
        *names : str
            The inputs to mark optional.

        Returns
        -------
        InputSpec
            A new spec of the same inputs.

        Raises
        ------
        KeyError
            If a name is not an input.
        """
        unknown = [name for name in names if name not in self._inputs]
        if unknown:
            raise KeyError(f"with_optional(): {unknown_names('input', unknown, self._inputs)}")
        return self._with_inputs(self._inputs, self._optional | frozenset(names))

    def without(self, *names: str) -> InputSpec:
        """The inputs other than *names*, each optional input staying optional."""
        return self._with_inputs(
            {name: spec for name, spec in self._inputs.items() if name not in names}, self._optional
        )

    def _with_inputs(self, inputs: Mapping[str, TermSpec], optional: frozenset[str]) -> InputSpec:
        """An input spec of *inputs*, those of *optional* among them optional."""
        result = InputSpec(inputs)
        object.__setattr__(result, "_optional", optional & frozenset(inputs))
        return result

    @property
    def free_dims(self) -> frozenset[str]:
        """The shared dimension scope of all inputs."""
        return frozenset().union(*(spec.free_dims for spec in self._inputs.values()))

    @property
    def is_concrete(self) -> bool:
        """Whether all inputs have concrete dimensions."""
        return not self.free_dims

    def with_dim_sizes(self, **sizes: int) -> InputSpec:
        """Return the inputs with supplied sizes substituted in their shared scope."""
        _dim_sizes(sizes)
        return self._with_inputs(
            {name: spec.with_dim_sizes(**sizes) for name, spec in self._inputs.items()},
            self._optional,
        )

    def with_dim_names(self, **names: str) -> InputSpec:
        """Return the inputs with simultaneous symbolic-dimension renaming."""
        _dim_renames(names)
        return self._with_inputs(
            {name: spec.with_dim_names(**names) for name, spec in self._inputs.items()},
            self._optional,
        )

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> InputSpec:
        """The inputs with *bindings* substituted in their shared scope."""
        return self._with_inputs(
            {name: spec._substitute_dims(bindings) for name, spec in self._inputs.items()},
            self._optional,
        )

    def bind_dims_from_value(self, value: Mapping[str, object]) -> InputSpec:
        """Bind all inputs against named values in one shared dimension scope.

        Raises ValueError for missing/extra inputs or conflicting sizes.
        """
        if not isinstance(value, Mapping):
            raise TypeError(
                f"InputSpec.bind_dims_from_value() expects a mapping, got {type(value).__name__}"
            )
        if self._inputs.keys() != value.keys():
            raise ValueError(
                f"values do not match the InputSpec inputs: {_name_mismatch(value, self)}"
            )
        bindings: dict[str, int] = {}
        for name, spec in self._inputs.items():
            spec._bind_dims_from_value(value[name], bindings, f"InputSpec/{name}")
        return self._substitute_dims(bindings)

    def bind_dims_from_spec(self, other: InputSpec) -> InputSpec:
        """Bind against another input spec in one shared dimension scope."""
        if not isinstance(other, InputSpec):
            raise TypeError(
                f"InputSpec.bind_dims_from_spec() expects an InputSpec, got {type(other).__name__}"
            )
        if self._inputs.keys() != other._inputs.keys():
            raise ValueError(
                f"the other InputSpec's inputs do not match these: {_name_mismatch(other, self)}"
            )
        bindings: dict[str, int] = {}
        for name, spec in self._inputs.items():
            _unify_specs(spec, other[name], bindings, f"InputSpec/{name}")
        return self._substitute_dims(bindings)


#: The spec each kind of exposed term takes, by the kind :func:`_exposed_kind` names.
_EXPOSED_SPECS = {
    "record": "RecordSpec",
    "law": "DistributionSpec or ConditionalDistributionSpec",
    "batch": "BatchSpec of a record or a law",
}

#: What the outputs of each kind of exposed term are, as a message names them.
_EXPOSED_OF = {
    "record": "a record's fields",
    "law": "a distribution's event outputs",
    "batch": "a batch element's outputs",
}


def _exposed_kind(spec: object) -> str | None:
    """The kind of term the positional form exposes for *spec*, or None when it exposes none.

    A record exposes its immediate fields, a law or a kernel its event's
    outputs, and a batch its element's outputs.
    """
    from ._batch import BatchSpec

    if isinstance(spec, RecordSpec):
        return "record"
    if isinstance(spec, TermSpec) and isinstance(getattr(spec, "event_spec", None), OutputSpec):
        return "law"
    if isinstance(spec, BatchSpec) and _exposed_kind(spec.element_spec) is not None:
        return "batch"
    return None


def _exposed_outputs(spec: TermSpec) -> Mapping[str, TermSpec | None]:
    """The outputs an exposed *spec* declares, by the rule of :func:`_exposed_kind`."""
    kind = _exposed_kind(spec)
    if kind == "record":
        return cast(RecordSpec, spec).children
    if kind == "law":
        return cast(OutputSpec, spec.event_spec).outputs  # type: ignore[attr-defined]
    return _exposed_outputs(spec.element_spec)  # type: ignore[attr-defined]


@dataclass(frozen=True, init=False)
class OutputSpec:
    """The outputs of a function's return value, of the values a distribution is over, or of a batch's elements.

    An output is a name under which a value can be read and passed on to an
    input of the same name. An OutputSpec holds a spec, ``term_spec``, and its
    outputs follow from it, in one of two forms:

    - the positional form, ``OutputSpec(RecordSpec(mu=(), sigma=()))``, has one
      output per top-level name of a record (``mu`` and ``sigma``), the outputs
      of a distribution's or conditional distribution's event, or the outputs of
      a batch's elements;
    - the keyword form, ``OutputSpec(y=NumericArraySpec(()))``, names one output
      that holds the whole value.

    Parameters
    ----------
    *args : RecordSpec, DistributionSpec, ConditionalDistributionSpec, or BatchSpec
        The positional form: the spec whose outputs this OutputSpec has, a
        record's top-level fields and groups, a distribution's or a conditional
        distribution's event outputs, or the outputs of a batch's element, which
        is a record or a distribution.
    **term : TermSpec or None
        The keyword form: one keyword, the output's name, over the spec of the
        whole value. None means the type is not known yet; :meth:`with_term_spec`
        gives it later.

    Raises
    ------
    TypeError
        If the positional form is not exactly one spec that has outputs, the
        forms are mixed, more than one keyword is given, or a record field lacks
        a spec.
    ValueError
        If no spec is given, or an output name is empty or contains ``/``.

    Notes
    -----
    Only the form and the spec are stored; ``term_spec`` and ``outputs`` are
    derived from them. An output of a record is a field or group of the value,
    and an output of a distribution's event is a field of each of its draws.
    """

    _output_name: str | None
    _term_spec: TermSpec | None

    def __init__(self, *args: TermSpec, **term: TermSpec | None) -> None:
        if args:
            if term:
                raise TypeError("OutputSpec takes one positional spec or one keyword, not both")
            if len(args) != 1:
                raise TypeError(f"OutputSpec takes one positional spec, got {len(args)}")
            spec = args[0]
            if _exposed_kind(spec) is None:
                raise TypeError(
                    f"OutputSpec cannot take a {type(spec).__name__} positionally; name a single "
                    f"output with a keyword, such as OutputSpec(x=...). A positional spec must be "
                    f"a RecordSpec, DistributionSpec, ConditionalDistributionSpec, or a BatchSpec "
                    f"of one"
                )
            if isinstance(spec, RecordSpec):
                for name, child in spec.children.items():
                    _check_output(name, child)
            name = None
        elif len(term) == 1:
            name, spec = next(iter(term.items()))
            _check_output(name, spec, allow_hole=True)
        elif term:
            fields = ", ".join(f"{name}=..." for name in term)
            raise TypeError(
                f"OutputSpec takes one keyword, got {list(term)}; for several named outputs, "
                f"pass a record: OutputSpec(RecordSpec({fields}))"
            )
        else:
            raise ValueError("OutputSpec requires one keyword or an explicit RecordSpec")
        object.__setattr__(self, "_output_name", name)
        object.__setattr__(self, "_term_spec", spec)

    def __repr__(self) -> str:
        """The OutputSpec as the constructor takes it: one keyword, or one positional spec."""
        if self._output_name is None:
            return call_repr("OutputSpec", [repr(self._term_spec)])
        return term_repr("OutputSpec", None, [(self._output_name, repr(self._term_spec))])

    @property
    def term_spec(self) -> TermSpec | None:
        """The spec of the values described. None means the type is not known yet."""
        return self._term_spec

    @property
    def outputs(self) -> Mapping[str, TermSpec | None]:
        """Each output's name and spec, in order, as a read-only mapping."""
        if self._output_name is not None:
            return MappingProxyType({self._output_name: self._term_spec})
        return _exposed_outputs(cast(TermSpec, self._term_spec))

    @classmethod
    def default(cls, term_spec: TermSpec, *, output_name: str) -> OutputSpec:
        """The OutputSpec ProbPipe uses for values of *term_spec* when none is given.

        It is ``OutputSpec(term_spec)`` if *term_spec* is a ``RecordSpec``, whose
        fields and groups become the outputs, and
        ``OutputSpec(**{output_name: term_spec})`` otherwise.

        Parameters
        ----------
        term_spec : TermSpec
            The spec of the values.
        output_name : str
            The name of the one output when *term_spec* is not a ``RecordSpec``.

        Returns
        -------
        OutputSpec
            The positional form for a record, and the keyword form otherwise.

        Raises
        ------
        ValueError
            If *term_spec* is not a record and *output_name* is not a valid
            output name.
        """
        if isinstance(term_spec, RecordSpec):
            return cls(term_spec)
        return cls(**{output_name: term_spec})

    def with_term_spec(self, term_spec: TermSpec, /) -> OutputSpec:
        """This OutputSpec with its spec completed from *term_spec*.

        Where this OutputSpec's spec is None, which means the type is not known
        yet, it becomes *term_spec*. Otherwise the declared spec must unify with
        *term_spec*, and the result keeps the names and the form and stores their
        unification: a numeric array keeps its declared dtype and support and
        takes *term_spec*'s where the declared spec leaves them unset, and a
        declared dimension name binds to the size *term_spec* gives. A record
        unifies field by field, and an opaque spec takes the known type and
        ``meta``. Any other kind takes *term_spec*.

        Parameters
        ----------
        term_spec : TermSpec
            The spec of the actual values.

        Returns
        -------
        OutputSpec
            The completed OutputSpec.

        Raises
        ------
        TypeError
            If this OutputSpec is in the positional form and *term_spec* is not a
            spec of the same kind, such as a ``RecordSpec`` for a record.
        ValueError
            If the declared spec does not unify with *term_spec*.
        """
        kind = None if self._output_name is not None else _exposed_kind(self._term_spec)
        if kind is not None and _exposed_kind(term_spec) != kind:
            raise TypeError(
                f"with_term_spec() needs a {_EXPOSED_SPECS[kind]} for an OutputSpec of "
                f"{_EXPOSED_OF[kind]}, got {type(term_spec).__name__}"
            )
        if self._term_spec is not None:
            bindings: dict[str, int] = {}
            root = _root_path(self._output_name, kind)
            _unify_specs(self._term_spec, term_spec, bindings, root)
            term_spec = _unification(self._term_spec, term_spec, bindings)
        return self._with_term_spec(term_spec)

    def with_path_names(
        self, mapping: Mapping[str, str] | None = None, /, **kwargs: str
    ) -> OutputSpec:
        """Rename or move nodes of the declaration by their paths, ``old -> new``.

        A path starts with an output: an exposed record's paths are the paths
        of its record, and a whole term's are its output followed by the
        paths within its term. Each key is the exact path of a node and each
        target its new exact path, under the rule of
        :meth:`~probpipe.core.named_tree.NamedTree.with_path_names`. The result
        keeps the packaging. An exposed record stays exposed, so a move may
        create or remove an output. A whole term's output is renamed in
        place, and the term's fields stay under it, so a field moves only within
        the output.

        Parameters
        ----------
        mapping : Mapping of str to str, optional
            The new path of each node, keyed by the node's path.
        **kwargs : str
            The new path of each output, keyed by the output's name.

        Returns
        -------
        OutputSpec
            The renamed declaration.

        Raises
        ------
        KeyError
            If a key is not a path of the declaration.
        ValueError
            As :meth:`~probpipe.core.named_tree.NamedTree.with_path_names` raises
            it, or if a whole term's output moves into a group or one of its
            fields moves out of it.
        """
        spec = self._term_spec
        output_name = self._output_name
        if output_name is None:
            return OutputSpec(_renamed_exposed(cast(TermSpec, spec), mapping, kwargs))
        # A whole term's paths are those of the record of its one output.
        outputs = RecordSpec({output_name: OpaqueSpec() if spec is None else spec})
        renames = outputs._resolve_path_renames(mapping, kwargs)
        renamed_output_name = renames.get(output_name, output_name)
        if _PATH_SEP in renamed_output_name:
            raise ValueError(
                f"cannot rename {output_name!r} to {renamed_output_name!r}: with_path_names() can "
                f"rename a single-output but cannot move it into a group. Choose a "
                f"name without {_PATH_SEP!r}."
            )
        prefix = renamed_output_name + _PATH_SEP
        for source, target in renames.items():
            if source != output_name and not target.startswith(prefix):
                owner = (
                    repr(output_name)
                    if renamed_output_name == output_name
                    else f"{output_name!r}, renamed to {renamed_output_name!r},"
                )
                raise ValueError(
                    f"cannot move {source!r} to {target!r}: fields of {owner} must stay "
                    f"under {prefix!r}"
                )
        ((name, term),) = outputs.with_path_names(renames).children.items()
        return OutputSpec(**{name: None if spec is None else term})

    def _with_term_spec(self, term_spec: TermSpec | None) -> OutputSpec:
        """This OutputSpec's form and names over *term_spec*, with no check."""
        if self._output_name is not None:
            return OutputSpec(**{self._output_name: term_spec})
        return OutputSpec(cast(TermSpec, term_spec))

    @property
    def is_concrete(self) -> bool:
        """Whether the type is known and all its dimensions are concrete."""
        return self._term_spec is not None and self._term_spec.is_concrete

    def with_dim_sizes(self, **sizes: int) -> OutputSpec:
        """This OutputSpec with dimension names bound to sizes, keeping its form."""
        _dim_sizes(sizes)
        term_spec = self._term_spec
        return self._with_term_spec(
            None if term_spec is None else term_spec.with_dim_sizes(**sizes)
        )

    def with_dim_names(self, **names: str) -> OutputSpec:
        """This OutputSpec with dimension names renamed, keeping its form."""
        _dim_renames(names)
        term_spec = self._term_spec
        return self._with_term_spec(
            None if term_spec is None else term_spec.with_dim_names(**names)
        )


def _root_path(output_name: str | None, kind: str | None) -> str:
    """The path a unification error names a declaration's root by.

    A whole term's paths start at its *output_name*, and an exposed record's
    fields are its outputs, so its root is the empty path. An exposed law or
    batch has no path above its event's outputs, so its root is named by its
    *kind*.
    """
    if output_name is not None:
        return output_name
    return "" if kind == "record" else _ROOT_NAMES[cast(str, kind)]


#: The name a unification error gives the root of an exposed law or batch.
_ROOT_NAMES = {"law": "the distribution", "batch": "the batch"}


def _unification(declared: TermSpec, produced: TermSpec, bindings: Mapping[str, int]) -> TermSpec:
    """The unification of a declared spec with the produced one it unifies with.

    *bindings* holds the sizes the unification bound to symbolic dimensions.
    The rule is :meth:`OutputSpec.with_term_spec`'s, and the result is *produced*
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
    """The exposed *spec* with the nodes its outputs start renamed, as ``with_path_names`` reads them.

    A record renames its own paths, a law or a kernel its event spec's,
    and a batch its element's, so the term keeps its kind.
    """
    from dataclasses import replace

    kind = _exposed_kind(spec)
    if kind == "record":
        return cast(RecordSpec, spec).with_path_names(mapping, **kwargs)
    if kind == "law":
        event = cast(OutputSpec, spec.event_spec)  # type: ignore[attr-defined]
        return replace(spec, event_spec=event.with_path_names(mapping, **kwargs))  # type: ignore[type-var]
    from ._batch import BatchSpec

    batch = cast(BatchSpec, spec)
    return batch._replace(element_spec=_renamed_exposed(batch.element_spec, mapping, kwargs))


def _root_output_name(output_spec: OutputSpec) -> str | None:
    """The name of *output_spec*'s output at the root, or None in the positional form.

    The keyword form ``OutputSpec(y=...)`` names one output that holds the whole
    value; code outside this module reads the form through this helper.
    """
    return output_spec._output_name


def _has_record_outputs(output_spec: OutputSpec) -> bool:
    """Whether *output_spec* is the positional form over a ``RecordSpec``.

    Its outputs are then the record's top-level fields and groups.
    """
    return output_spec._output_name is None and isinstance(output_spec._term_spec, RecordSpec)


#: The values an output spec in the positional form describes, as the refusal of
#: :func:`_output_record` names them.
_VALUES_WITHOUT_RECORD = {
    "DistributionSpec": "distributions",
    "ConditionalDistributionSpec": "conditional distributions",
    "BatchSpec": "batches",
}


def _output_record(output_spec: OutputSpec) -> RecordSpec:
    """The record of *output_spec*'s outputs: one field per output, holding the output's spec.

    The record exists when the outputs determine the value: in the positional form
    over a ``RecordSpec`` it is that record, and in the keyword form it is the
    one-field record under the output's name, as a merge places a single value.
    The parameters of one draw of a model or a posterior are read through it.

    Parameters
    ----------
    output_spec : OutputSpec
        The output spec of the values.

    Returns
    -------
    RecordSpec
        The record of the outputs.

    Raises
    ------
    TypeError
        If *output_spec* is in the positional form over a distribution, a
        conditional distribution, or a batch, whose values are not determined by
        named outputs; the message says to wrap the values under a name.
    """
    if _has_record_outputs(output_spec):
        return cast(RecordSpec, output_spec._term_spec)
    if output_spec._output_name is None:
        values = _VALUES_WITHOUT_RECORD.get(type(output_spec._term_spec).__name__, "values")
        raise TypeError(
            f"expected the outputs of a record or of a single named value, but the output "
            f"spec describes {values}, which are not read by their outputs; wrap them under a "
            f"name, as OutputSpec(RecordSpec(params=...))"
        )
    return RecordSpec({output_spec._output_name: output_spec._term_spec})


def _single_output_name(output_spec: OutputSpec) -> str | None:
    """The name of *output_spec*'s one output, or None when its outputs are a record's fields.

    In the keyword form this is the name of the output at the root. In the
    positional form over a distribution, a conditional distribution, or a batch,
    the outputs are still those of one draw of the values described, and this
    is the one of them; the step that gives such an output spec no outputs
    removes this helper, and its callers then read :func:`_root_output_name`.

    Raises
    ------
    ValueError
        If *output_spec* is in the positional form over a distribution, a
        conditional distribution, or a batch with more than one output.
    """
    if _has_record_outputs(output_spec):
        return None
    (output_name,) = output_spec.outputs
    return output_name


def _unnamed_declaration(declaration: OutputSpec) -> tuple[object, ...]:
    """*declaration*'s packaging and its outputs' specs in order, without their names.

    A function's fingerprint and its replay anchor record this form, so a
    rename of the outputs of its output keeps both.
    """
    if declaration._output_name is not None:
        return ("whole term", (declaration.term_spec,))
    return ("exposed", *_unnamed_exposed(cast(TermSpec, declaration.term_spec)))


def _unnamed_exposed(spec: TermSpec) -> tuple[object, ...]:
    """An exposed *spec* without its output names: its kind and what it holds besides them."""
    kind = _exposed_kind(spec)
    if kind == "record":
        return ("record", tuple(_exposed_outputs(spec).values()))
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
