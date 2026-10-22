"""Shared kind specs, input specs, and output specs (design II.1–II.2).

The names of a term's outputs describe its public interface independently of
the label or identity of the value an output spec describes.
"""

from __future__ import annotations

import keyword
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NoReturn, cast

from .._messages import unknown_names
from ._record_spec import NumericRecordSpec, RecordSpec
from ._repr import call_repr, term_repr
from ._shapes import _as_path_arguments, _as_paths, _missing_path
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
from .named_tree import _PATH_SEP, _last_name

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
    _check_term(name, spec, allow_hole=allow_hole, noun="output")


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

    An input is required or optional. A binding may omit an optional input, which
    then takes the default of the function or conditional distribution that owns
    it; :meth:`with_optional` marks inputs optional, and the optional inputs are
    part of the spec, so they take part in equality and hashing.

    Parameters
    ----------
    inputs : Mapping[str, TermSpec], optional
        A positional mapping of inputs, in iteration order. A nested record is
        one input carrying a RecordSpec.
    **inputs_by_name : TermSpec
        Alternatively, the inputs by keyword, including RecordSpec-valued inputs.
        No constructor keyword is reserved.

    Raises
    ------
    TypeError
        If a non-None positional argument is not a Mapping, mapping and keyword
        forms are combined, or an input's spec is not a hashable TermSpec; an
        input's spec may not be None.
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
        """The inputs a binding must supply, in order."""
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
            spec._bind_dims_from_value(value[name], bindings, f"InputSpec{_PATH_SEP}{name}")
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
            _unify_specs(spec, other[name], bindings, f"InputSpec{_PATH_SEP}{name}")
        return self._substitute_dims(bindings)


#: The spec of each kind of term the positional form takes, by the kind :func:`_exposed_kind` names.
_EXPOSED_SPECS = {
    "record": "RecordSpec",
    "law": "DistributionSpec or ConditionalDistributionSpec",
    "batch": "BatchSpec of a record or a law",
}

#: The values an output spec of each kind of term in the positional form describes, as a message
#: names them.
_EXPOSED_OF = {
    "record": "records",
    "law": "distributions",
    "batch": "batches",
}


def _exposed_kind(spec: object) -> str | None:
    """The kind of term the positional form takes for *spec*, or None when it takes none.

    The positional form takes a record, whose outputs are its immediate fields;
    a law or a kernel, which has no outputs; and a batch of records or of laws,
    whose outputs are its element spec's outputs.
    """
    from ._batch import BatchSpec

    if isinstance(spec, RecordSpec):
        return "record"
    if isinstance(spec, TermSpec) and isinstance(getattr(spec, "event_spec", None), OutputSpec):
        return "law"
    if isinstance(spec, BatchSpec) and _exposed_kind(spec.element_spec.term_spec) is not None:
        return "batch"
    return None


def _exposed_outputs(spec: TermSpec) -> Mapping[str, TermSpec | None]:
    """The outputs *spec* has in the positional form, by the rule of :func:`_exposed_kind`.

    A record has its top-level fields and groups, and a batch its element's
    outputs. A distribution or a conditional distribution has none, since the
    distributions of its draws' fields do not determine it, so a batch of
    them has none either.
    """
    kind = _exposed_kind(spec)
    if kind == "record":
        return cast(RecordSpec, spec).children
    if kind == "law":
        return MappingProxyType({})
    return cast(OutputSpec, spec.element_spec).outputs  # type: ignore[attr-defined]


@dataclass(frozen=True, init=False)
class OutputSpec:
    """The outputs of a function's return value, of the values a distribution is over, or of a batch's elements.

    An output is a name under which a value can be read and passed on to an
    input of the same name. An OutputSpec holds a spec, ``term_spec``, and its
    outputs follow from it, in one of two forms:

    - the positional form, ``OutputSpec(RecordSpec(mu=(), sigma=()))``, has one
      output per top-level name of a record (``mu`` and ``sigma``), or the
      outputs of a batch's elements. Over a distribution's or a conditional
      distribution's spec, as for a function that returns a distribution, it
      has no outputs; to pass the distribution on by name, put it in a record,
      ``OutputSpec(RecordSpec(params=d.spec))``;
    - the keyword form, ``OutputSpec(y=NumericArraySpec(()))``, names one output
      that holds the whole value.

    An OutputSpec is indexed by path, as ``O["g/b"]``, and ``&`` merges two of
    them. It is not a mapping and not iterable; ``outputs`` lists the outputs.

    Parameters
    ----------
    *args : RecordSpec, DistributionSpec, ConditionalDistributionSpec, or BatchSpec
        The positional form: the spec of the values, whose outputs are a
        record's top-level fields and groups, or the outputs of a batch's
        element, which is a record or a distribution; a distribution's or a
        conditional distribution's spec has none.
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

    Examples
    --------
    >>> O = OutputSpec(RecordSpec(a=(), g=RecordSpec(b=(), c=())))
    >>> list(O.outputs)
    ['a', 'g']
    >>> O["g/b"]
    OutputSpec(b=NumericArraySpec(shape=()))

    Notes
    -----
    Only the form and the spec are stored; ``term_spec`` and ``outputs`` are
    derived from them. An output of a record is a field or group of the value.
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

    def __getitem__(self, paths: str | Sequence[str], /) -> OutputSpec:
        """The OutputSpec at a path, or of a list of paths.

        Parameters
        ----------
        paths : str, or list or tuple of str
            A path, such as ``"g/b"``, or a list of paths. A tuple is a list of
            paths; a nested path is written with ``/``.

        Returns
        -------
        OutputSpec
            For one path, the OutputSpec of the value there: a single array or
            opaque value has one output named by the path's last name, as
            ``O["g/b"]`` is ``OutputSpec(b=...)``, and a record has its
            top-level names as outputs. For a list, the record of the chosen
            values, each at its own path, in this spec's order, as ``O[["g/b"]]``
            is ``OutputSpec(RecordSpec(g=RecordSpec(b=...)))``. A list may name
            the one output of the keyword form, as ``OutputSpec(x=())[["x"]]``
            is ``OutputSpec(RecordSpec(x=...))``.

        Raises
        ------
        KeyError
            If a path is not a path of this OutputSpec's record.
        TypeError
            If *paths* is neither a string nor a list or tuple of strings, or
            this OutputSpec has no paths to select, as over a single array or
            a distribution.
        ValueError
            If a path is empty or has an empty name, or a list of paths is
            empty, names a path twice, or holds a path inside another.

        Notes
        -----
        As an interim implementation detail, the name of the one output of the
        keyword form gives this OutputSpec itself, as ``Normal("x", 0.0,
        1.0)["x"]`` gives the distribution itself, until an accessor of an
        output replaces both readings; a value with no paths has no path to
        select.

        Examples
        --------
        >>> O = OutputSpec(RecordSpec(a=(), g=RecordSpec(b=(), c=())))
        >>> O["g"]
        OutputSpec(NumericRecordSpec(b=(), c=()))
        >>> O[["g/b"]]
        OutputSpec(NumericRecordSpec(g=NumericRecordSpec(b=())))
        """
        read_paths = _read_paths(
            self,
            paths,
            owner="the output spec",
            values_word="values",
            action="index the output spec by",
        )
        if isinstance(read_paths, str):
            if read_paths == self._output_name:
                return self
            node_spec = cast(RecordSpec, _path_record(self))[read_paths]
            return _output_spec_at(node_spec, _last_name(read_paths))
        if self._term_spec is None:
            raise TypeError(
                f"cannot index the output spec by {list(read_paths)}, since {_type_not_known(self)}"
            )
        return OutputSpec(_output_record(self)[read_paths])

    def __contains__(self, path: object, /) -> bool:
        """Whether *path* is a string and ``self[path]`` succeeds.

        As ``[]`` does, it reads the name of the keyword form's one output as
        an interim implementation detail.
        """
        return _has_path(self, path)

    def __iter__(self) -> NoReturn:
        """Not supported: an OutputSpec is not a mapping; iterate over its ``outputs``.

        Raises
        ------
        TypeError
            Always.
        """
        raise TypeError(
            "cannot iterate over an OutputSpec. Iterate over its outputs instead, as "
            "list(spec.outputs)"
        )

    def without(self, *paths: str) -> OutputSpec:
        """This OutputSpec without the values at some paths.

        Parameters
        ----------
        *paths : str
            The paths to remove, one per argument; a group left empty goes too.

        Returns
        -------
        OutputSpec
            The rest, each value at its own path, in the same form.

        Raises
        ------
        KeyError
            If a path is not a path of this OutputSpec's record.
        TypeError
            If a path is not a string, or this OutputSpec has no paths to
            remove, as over a single array or a distribution.
        ValueError
            If nothing would remain, or the paths name a path twice or a path
            inside another.

        Examples
        --------
        >>> O = OutputSpec(RecordSpec(a=(), g=RecordSpec(b=(), c=())))
        >>> O.without("g/c")
        OutputSpec(NumericRecordSpec(a=(), g=NumericRecordSpec(b=())))
        """
        read_paths = _as_path_arguments(paths, action="call without() with")
        for path in read_paths:
            if not _has_path(self, path):
                raise _paths_error(
                    self,
                    path,
                    owner="the output spec",
                    values_word="values",
                    action="remove",
                    after_path=" from the output spec",
                )
        if self._output_name in read_paths:
            raise ValueError(
                f"cannot remove {self._output_name!r} from the output spec, since it names the "
                f"whole value and nothing would remain"
            )
        remaining_record = _output_record(self).without(*read_paths)
        if self._output_name is None:
            return OutputSpec(remaining_record)
        return OutputSpec(**{self._output_name: remaining_record.children[self._output_name]})

    def __and__(self, other: OutputSpec, /) -> OutputSpec:
        """Merge two OutputSpecs: the outputs of both, this one's first.

        Parameters
        ----------
        other : OutputSpec
            The outputs to add after this spec's.

        Returns
        -------
        OutputSpec
            The positional form over the record of both specs' outputs; the
            output ``y`` of the keyword form becomes the field ``y``.

        Raises
        ------
        TypeError
            If either spec's type is not known yet, or it is in the
            positional form over distributions, which have no outputs, or over
            batches, whose outputs are not read as one record yet. Wrap such
            values under a name to merge them.
        ValueError
            If an output name is in both.

        Examples
        --------
        >>> OutputSpec(x=NumericArraySpec(())) & OutputSpec(RecordSpec(a=()))
        OutputSpec(NumericRecordSpec(x=(), a=()))
        """
        if not isinstance(other, OutputSpec):
            return NotImplemented
        return OutputSpec(
            _merged_output_record(
                (self, other),
                owners=("the left output spec", "the right output spec"),
                values_word="values",
            )
        )

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
                f"with_term_spec() needs a {_EXPOSED_SPECS[kind]} for an OutputSpec over "
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
        """Rename or move nodes of this OutputSpec by their paths, ``old -> new``.

        A path starts with an output. In the positional form over a record, the
        paths are the record's; in the keyword form, a path is the output's name
        followed by a path within its value. Each key is the exact path of a node
        and each target its new exact path, under the rule of
        :meth:`~probpipe.core.named_tree.NamedTree.with_path_names`. The result
        keeps the form. In the positional form a move may add or remove an
        output. In the keyword form the output is renamed in place and the
        fields of its value stay under it, so a field moves only within the
        output.

        Parameters
        ----------
        mapping : Mapping of str to str, optional
            The new path of each node, keyed by the node's path.
        **kwargs : str
            The new path of each output, keyed by the output's name.

        Returns
        -------
        OutputSpec
            The renamed OutputSpec.

        Raises
        ------
        KeyError
            If a key is not a path of this OutputSpec.
        ValueError
            As :meth:`~probpipe.core.named_tree.NamedTree.with_path_names` raises
            it, or if the keyword form's output moves into a group or one of its
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
                f"rename the output of the keyword form but cannot move it into a group. "
                f"Choose a name without {_PATH_SEP!r}."
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
    """The path a unification error names an output spec's root by.

    In the keyword form the paths start at the *output_name*. In the positional
    form over a record, the record's fields are the outputs, so the root is the
    empty path; over a law or a batch, the root is named by its *kind*.
    """
    if output_name is not None:
        return output_name
    return "" if kind == "record" else _ROOT_NAMES[cast(str, kind)]


#: The name a unification error gives the root of a law or a batch in the positional form.
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
    """*spec* in the positional form, with the nodes its outputs start renamed.

    The renames are read as ``with_path_names`` reads them. A record renames
    its own paths, a law or a kernel its event spec's, and a batch its
    element spec's, so the term keeps its kind.
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
    return batch._replace(element=batch.element_spec.with_path_names(mapping, **kwargs))


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


def _no_output_record(output_spec: OutputSpec, values_word: str) -> str:
    """Why the values *output_spec* describes have no record of their outputs, with the fix.

    The output spec is in the positional form over a distribution, a
    conditional distribution, or a batch. *values_word* is what the message
    calls those values, such as ``"draws"``.
    """
    described = _VALUES_WITHOUT_RECORD.get(type(output_spec._term_spec).__name__, "values")
    reason = (
        "whose outputs are not read as one record yet"
        if described == "batches"
        else "which have no outputs to read by name"
    )
    return (
        f"its {values_word} are {described}, {reason}. Wrap them under a name, as "
        f"OutputSpec(RecordSpec(params=...))"
    )


def _type_not_known(output_spec: OutputSpec) -> str:
    """Why the keyword form *output_spec*, whose type is not known yet, cannot be read, with the fix."""
    return (
        f"the type of its output {output_spec._output_name!r} is not known yet. Complete it "
        f"first with with_term_spec"
    )


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
        raise TypeError(
            f"cannot read the outputs of the output spec as one record, since "
            f"{_no_output_record(output_spec, 'values')}"
        )
    return RecordSpec({output_spec._output_name: output_spec._term_spec})


def _path_record(output_spec: OutputSpec) -> RecordSpec | None:
    """The record whose paths are *output_spec*'s paths, or None when it has none to select.

    In the positional form over a ``RecordSpec`` it is that record. In the
    keyword form over any spec but a single array or opaque value, which
    names that whole spec under one output, it is the one-field record under
    the output's name, so the name and the paths below it are paths. A single
    array or opaque value has no paths, nor has an output whose type is not
    known yet, and the paths of distributions, conditional distributions, and
    batches are not read through an output spec yet.
    """
    if _has_record_outputs(output_spec):
        return cast(RecordSpec, output_spec._term_spec)
    term_spec = output_spec._term_spec
    if (
        output_spec._output_name is None
        or term_spec is None
        or isinstance(term_spec, (NumericArraySpec, OpaqueSpec))
    ):
        return None
    return RecordSpec({output_spec._output_name: term_spec})


def _has_path(output_spec: OutputSpec, path: object) -> bool:
    """Whether *path* is a string that selects from *output_spec*.

    The name of the keyword form's one output selects the output spec itself,
    until the accessor of an output replaces that reading.
    """
    if not isinstance(path, str):
        return False
    if path == output_spec._output_name:
        return True
    path_record = _path_record(output_spec)
    return path_record is not None and path in path_record


def _paths_error(
    output_spec: OutputSpec,
    path: str,
    *,
    owner: str,
    values_word: str,
    action: str,
    requested: tuple[str, ...] = (),
    after_path: str = "",
) -> KeyError | TypeError:
    """The error for *path*, which does not select from *output_spec*.

    Where *output_spec* has paths, it lacks this one, a ``KeyError`` listing
    the paths it has. Otherwise there is none to look up, a ``TypeError``.

    Parameters
    ----------
    output_spec : OutputSpec
        The output spec of the term the caller selects from.
    path : str
        The path that does not select.
    owner : str
        The term the messages name, such as ``"'model'"``.
    values_word : str
        What the term's values are called, such as ``"draws"``.
    action : str
        What the caller was asked to do, which the message completes with
        *path*, such as ``"index 'model' by"``.
    requested : tuple of str
        The list of paths that held *path*, or ``()`` for one path.
    after_path : str
        What the message says after *path*, such as ``" from the output
        spec"``; nothing by default.

    Returns
    -------
    KeyError or TypeError
        The error to raise.
    """
    path_record = _path_record(output_spec)
    if path_record is not None:
        return KeyError(
            _missing_path(
                path, owner=owner, available=path_record._node_paths(), requested=requested
            )
        )
    failed_action = f"cannot {action} {path!r}{after_path}"
    if output_spec._output_name is not None and output_spec._term_spec is None:
        return TypeError(f"{failed_action}, since {_type_not_known(output_spec)}")
    values_description = _VALUES_WITHOUT_RECORD.get(type(output_spec._term_spec).__name__)
    if values_description is None:
        return TypeError(
            f"{failed_action}, since its {values_word} are single values with no paths"
        )
    return TypeError(
        f"{failed_action}, since its {values_word} are {values_description}, and indexing into them "
        f"by path is not supported yet"
    )


def _read_paths(
    output_spec: OutputSpec,
    paths: Any,
    *,
    owner: str,
    values_word: str,
    action: str,
    tuple_given: bool | None = None,
) -> str | tuple[str, ...]:
    """*paths*, read as paths that select from *output_spec*.

    One path is returned as it is, and a list of paths in the canonical order
    of *output_spec*'s record, whatever order it was given in.

    Parameters
    ----------
    output_spec : OutputSpec
        The output spec of the term the caller selects from.
    paths : Any
        One path, or a list or tuple of paths, as the caller gave them.
    owner : str
        The term the messages name, such as ``"'model'"``.
    values_word : str
        What the term's values are called, such as ``"draws"``.
    action : str
        What the caller was asked to do, which the messages complete.
    tuple_given : bool or None
        Whether the caller was given its list of paths as a tuple, which a
        missing path's message reads; ``None`` reads it from *paths*.

    Returns
    -------
    str or tuple of str
        The one path, or the list of paths in canonical order.

    Raises
    ------
    KeyError
        If *output_spec* has paths and a path is not one of them.
    TypeError
        If *paths* is neither a string nor a list or tuple of strings, or
        *output_spec* has no paths to select.
    ValueError
        If a path is empty or has an empty name, or a list of paths is empty,
        names a path twice, or holds a path inside another.
    """
    read_paths = _as_paths(paths, action=action)
    listed_paths = read_paths if isinstance(read_paths, tuple) else (read_paths,)
    tuple_was_given = isinstance(paths, tuple) if tuple_given is None else tuple_given
    for path in listed_paths:
        if not _has_path(output_spec, path):
            requested = listed_paths if tuple_was_given and isinstance(read_paths, tuple) else ()
            raise _paths_error(
                output_spec,
                path,
                owner=owner,
                values_word=values_word,
                action=action,
                requested=requested,
            )
    if isinstance(read_paths, str) or len(read_paths) == 1:
        return read_paths
    return cast(RecordSpec, _path_record(output_spec))._ordered_paths(read_paths)


def _output_spec_at(node_spec: TermSpec, last_name: str) -> OutputSpec:
    """The output spec of a node of spec *node_spec* selected at a path whose last name is *last_name*.

    A spec the positional form takes, such as a record or a distribution's
    spec, gives the positional form; any other, such as a single array, is
    named by the path's last name.
    """
    if _exposed_kind(node_spec) is not None:
        return OutputSpec(node_spec)
    return OutputSpec(**{last_name: node_spec})


def _merged_output_record(
    output_specs: Sequence[OutputSpec], *, owners: Sequence[str], values_word: str
) -> RecordSpec:
    """The record of the outputs of every one of *output_specs*, in order: their merge.

    Each output spec contributes the record of its outputs, so the output
    ``y`` of the keyword form becomes the field ``y``. A merge reads values by
    their outputs, so it refuses an output spec whose type is not known yet or
    whose values have no outputs, and one source per output name.

    Parameters
    ----------
    output_specs : sequence of OutputSpec
        The output specs to merge, in order.
    owners : sequence of str
        The term each output spec describes, as the messages name it, such as
        ``"'prior'"``.
    values_word : str
        What the terms' values are called, such as ``"draws"``.

    Returns
    -------
    RecordSpec
        The record of every output, each output spec's outputs in its order.

    Raises
    ------
    TypeError
        If an output spec's type is not known yet, or it is in the positional
        form over a distribution, a conditional distribution, or a batch,
        whose values have no outputs to merge by name.
    ValueError
        If two output specs have an output of one name.
    """
    merged_outputs: dict[str, TermSpec] = {}
    owner_of_output: dict[str, str] = {}
    for output_spec, owner in zip(output_specs, owners, strict=True):
        output_name = output_spec._output_name
        if output_name is not None and output_spec._term_spec is None:
            raise TypeError(f"cannot merge {owner}, since {_type_not_known(output_spec)}")
        if output_name is None and not _has_record_outputs(output_spec):
            raise TypeError(
                f"cannot merge {owner}, since {_no_output_record(output_spec, values_word)}"
            )
        for name, output_term_spec in _output_record(output_spec).children.items():
            if name in merged_outputs:
                raise ValueError(
                    f"cannot merge {owner_of_output[name]} and {owner}, since both have the output "
                    f"{name!r}. Rename it in one of them, as with_path_names({{{name!r}: "
                    f"{name + '_2'!r}}})"
                )
            merged_outputs[name] = output_term_spec
            owner_of_output[name] = owner
    return RecordSpec(merged_outputs)


def _unnamed_declaration(declaration: OutputSpec) -> tuple[object, ...]:
    """*declaration*'s form and its outputs' specs in order, without their names.

    A function's fingerprint and its replay anchor record this, so renaming
    the function's outputs keeps both.
    """
    if declaration._output_name is not None:
        return ("whole term", (declaration.term_spec,))
    return ("exposed", *_unnamed_exposed(cast(TermSpec, declaration.term_spec)))


def _unnamed_exposed(spec: TermSpec) -> tuple[object, ...]:
    """*spec* in the positional form without its output names: its kind and what else it holds."""
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
        _unnamed_exposed(spec.element_spec.term_spec),  # type: ignore[attr-defined]
    )
