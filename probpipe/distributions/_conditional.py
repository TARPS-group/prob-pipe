"""The conditional distribution: a probability kernel, its term spec, and its markers.

Provides:
  - ``ConditionalDistribution`` – a kernel ``K : S → P(T)``, which a value for
    its inputs turns into a ``Distribution``.
  - ``ConditionalDistributionSpec`` – the term spec of the conditional kind.
  - ``ConditionalNumericDistribution``, ``NumericConditionalDistribution``, and
    ``FullyNumericConditionalDistribution`` – the markers of a kernel whose
    event side, given side, or both are numeric.
  - ``conditional_distribution`` – the kernel of a function of its given values
    that returns a law.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

import jax
import jax.numpy as jnp

from .._messages import label_given_first, unknown_names
from ..core._dispatch import Feasibility
from ..core._expression import ELLIPSIS, Signature
from ..core._record_spec import RecordSpec
from ..core._repr import format_default, format_names, public_class_name, term_repr
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec, _unify_specs
from ..core._specs import InputSpec, OutputSpec, _has_record_outputs
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import Annotated, TrackedTerm, _TrackedTermMeta
from ._capabilities import (
    SupportsConditionalLogProb,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    _capability_guard,
    _capability_subclass,
    _check_guards,
)
from ._distribution import (
    _DECLARATION_MARKERS,
    DEFAULT_LABEL,
    Distribution,
    DistributionSpec,
    _cached_raw,
    _check_marker_claims,
    _complete_event_spec,
    _compose_operands,
    _constructor_label,
    _detached_term,
    _event_repr_parts,
    _given_label,
    _law_repr,
    _no_free_dims,
    _record_default_expression,
    _unify_declarations,
)

if TYPE_CHECKING:
    from ._factored import FactoredConditionalDistribution

__all__ = [
    "ConditionalDistribution",
    "ConditionalDistributionSpec",
    "ConditionalNumericDistribution",
    "FullyNumericConditionalDistribution",
    "NumericConditionalDistribution",
    "conditional_distribution",
]


# ---------------------------------------------------------------------------
# The declarations of a kernel
# ---------------------------------------------------------------------------


def _complete_given_spec(given_spec: Any) -> InputSpec:
    """*given_spec* as a non-empty ``InputSpec``.

    Parameters
    ----------
    given_spec : InputSpec or Mapping[str, TermSpec]
        The term spec of each input's value, keyed by input name.

    Returns
    -------
    InputSpec
        *given_spec* itself when it is an ``InputSpec``, and an ``InputSpec`` built
        from the mapping otherwise.

    Raises
    ------
    TypeError
        If *given_spec* is not a mapping of term specs.
    ValueError
        If it has no inputs, since binding the last given returns a
        ``Distribution``, or an input name is not an identifier.
    """
    if not isinstance(given_spec, InputSpec):
        if not isinstance(given_spec, Mapping):
            raise TypeError(
                f"given_spec must be an InputSpec or a mapping of term specs, got "
                f"{type(given_spec).__name__}"
            )
        given_spec = InputSpec(given_spec)
    if not len(given_spec):
        raise ValueError(
            "given_spec is empty, but a ConditionalDistribution needs at least one input; "
            "use a Distribution for a law with no inputs"
        )
    return given_spec


def _unknown_inputs(label: str, unknown: list[str], inputs: Iterable[str]) -> str:
    """The message that *unknown* name no input of the kernel *label*."""
    return f"cannot bind {label!r}: {unknown_names('input', unknown, list(inputs))}"


def _missing_inputs(label: str, missing: list[str]) -> str:
    """The message that a binding of every input of the kernel *label* leaves *missing* out."""
    inputs = f"input {missing[0]!r}" if len(missing) == 1 else f"inputs {missing}"
    return f"cannot bind {label!r}: missing a value for the required given {inputs}"


def _check_disjoint_sides(given_spec: InputSpec, event_spec: OutputSpec, owner: str) -> None:
    """Raise ``ValueError`` if an input shares a name with an output."""
    shared = set(given_spec) & set(event_spec.outputs)
    if shared:
        raise ValueError(
            f"{owner} names {sorted(shared)} both as an input and as an output field; "
            f"rename one of them"
        )


def _given_is_numeric(given_spec: InputSpec) -> bool:
    """Whether every input declares a numeric term."""
    return all(isinstance(spec, NumericSpec) for spec in given_spec.values())


# ---------------------------------------------------------------------------
# Renaming the given side
# ---------------------------------------------------------------------------

_PATH_SEP = "/"


def _given_leaf_specs(given_spec: InputSpec) -> dict[str, TermSpec]:
    """The spec of each leaf of the given side, keyed by its path, in canonical order.

    A leaf is an input, or a field of a structured input, whose path starts with
    the input.
    """
    leaves: dict[str, TermSpec] = {}
    for input_name, spec in given_spec.items():
        if isinstance(spec, RecordSpec):
            leaves.update({f"{input_name}{_PATH_SEP}{key}": spec[key] for key in spec})
        else:
            leaves[input_name] = spec
    return leaves


def _moved_inputs(
    given_spec: InputSpec, moves: Mapping[str, str]
) -> tuple[InputSpec, dict[str, str]]:
    """The given side with each node at a key of *moves* moved to its new exact path.

    A key is the path of an input or of a node within a structured input, and its
    target is the node's new path, so a move may split a field out of an input or
    group inputs into a structured one. The moves apply simultaneously, a node
    below a moved node moves with it unless it is moved itself, a node renamed
    within its parent keeps its position, a moved node is appended to its new
    parent, and a structured input that the moves empty dissolves.

    Parameters
    ----------
    given_spec : InputSpec
        The kernel's inputs, whose nodes the moves address.
    moves : Mapping[str, str]
        The new exact path of each node to move, keyed by the node's current path.

    Returns
    -------
    tuple of InputSpec and dict
        The moved inputs, and the path of each of their leaves' original leaf,
        keyed by the leaf's new path.

    Raises
    ------
    KeyError
        If a key is not the path of a node of the given side.
    ValueError
        If a target is empty or has an empty segment, lies inside its own node,
        or equals or contains another target, two nodes land on one path, or a
        new input name is not an identifier.
    """
    leaves = _given_leaf_specs(given_spec)
    nodes = {
        _PATH_SEP.join(segments[:end])
        for segments in (path.split(_PATH_SEP) for path in leaves)
        for end in range(1, len(segments) + 1)
    }
    for old, new in moves.items():
        if old not in nodes:
            raise KeyError(
                f"with_path_names(): {unknown_names('input path', [old], sorted(nodes))}"
            )
        if not isinstance(new, str):
            raise ValueError(
                f"with_path_names(): the new path of {old!r} must be a string; got "
                f"{type(new).__name__}"
            )
        if not new or not all(new.split(_PATH_SEP)):
            raise ValueError(
                f"with_path_names(): invalid new path {new!r} for {old!r}; each part between "
                f"'/' must be non-empty"
            )
        if new.startswith(old + _PATH_SEP):
            raise ValueError(f"with_path_names(): cannot move {old!r} inside itself, to {new!r}")
    targets = list(moves.values())
    for first, target in enumerate(targets):
        for other in targets[first + 1 :]:
            if (
                target == other
                or other.startswith(target + _PATH_SEP)
                or target.startswith(other + _PATH_SEP)
            ):
                raise ValueError(
                    f"with_path_names(): the new paths {target!r} and {other!r} overlap"
                )

    def parent(path: str) -> str:
        return path.rpartition(_PATH_SEP)[0]

    kept: list[tuple[str, str]] = []
    moved: dict[str, list[tuple[str, str]]] = {old: [] for old in moves}
    for path in leaves:
        sources = [old for old in moves if path == old or path.startswith(old + _PATH_SEP)]
        if not sources:
            kept.append((path, path))
            continue
        source = max(sources, key=len)
        target = moves[source] + path[len(source) :]
        if parent(source) == parent(moves[source]):
            kept.append((target, path))
        else:
            moved[source].append((target, path))
    tree: dict[str, Any] = {}
    origins: dict[str, str] = {}
    for target, path in [*kept, *(entry for old in moves for entry in moved[old])]:
        *groups, name = target.split(_PATH_SEP)
        node = tree
        for group in groups:
            node = node.setdefault(group, {})
            if not isinstance(node, dict):
                raise ValueError(
                    f"with_path_names(): cannot move to {target!r} because {group!r} is a "
                    f"field, not a group"
                )
        if name in node:
            raise ValueError(f"with_path_names(): two input paths would be at {target!r}")
        node[name] = leaves[path]
        origins[target] = path

    def spec_of(node: Any) -> TermSpec:
        if isinstance(node, dict):
            return RecordSpec({name: spec_of(child) for name, child in node.items()})
        return node

    moved_spec = InputSpec({input_name: spec_of(node) for input_name, node in tree.items()})
    # An optional input stays optional when it moves whole to an input of its own;
    # grouped into a structured input, it is part of a value the binding supplies.
    optional = []
    for input_name in given_spec.optional:
        target = moves.get(input_name, input_name)
        split = any(old.startswith(input_name + _PATH_SEP) for old in moves)
        joined = any(new.startswith(target + _PATH_SEP) for new in moves.values())
        if target in moved_spec and not split and not joined:
            optional.append(target)
    return moved_spec.with_optional(*optional), origins


#: The kernel that ``with_path_names`` returns, installed by the views module at import.
_renamed_kernel_factory: Callable[..., Any] | None = None


def _install_renamed_kernel(factory: Callable[..., Any]) -> None:
    """Install the factory of the kernel that renames at its boundary for ``with_path_names``.

    Called once, by the views module at import, so this module never imports the
    module that imports it.
    """
    global _renamed_kernel_factory
    _renamed_kernel_factory = factory


# ---------------------------------------------------------------------------
# ConditionalDistributionSpec — the term spec of the conditional kind
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class ConditionalDistributionSpec(TermSpec):
    """The conditional kind's term spec: the inputs and the event spec.

    Parameters
    ----------
    given_spec : InputSpec or Mapping[str, TermSpec]
        The named inputs a kernel conditions on, at least one.
    event_spec : OutputSpec or RecordSpec
        The declaration of one produced draw, as for ``DistributionSpec``. A
        bare ``RecordSpec`` completes to the exposed form.

    Raises
    ------
    TypeError
        If *event_spec* is neither an ``OutputSpec`` nor a ``RecordSpec``, or
        *given_spec* is not a mapping of term specs.
    ValueError
        If *given_spec* has no inputs, the event spec has an unspecified output type, or
        an input shares a name with an output.

    Notes
    -----
    ``is_valid`` accepts a ``ConditionalDistribution`` whose inputs have
    the same names and specs that unify with these, and whose event spec
    unifies with this one, both sides in one dimension scope.
    """

    given_spec: InputSpec
    event_spec: OutputSpec

    def __init__(
        self, given_spec: InputSpec | Mapping[str, TermSpec], event_spec: OutputSpec | RecordSpec
    ) -> None:
        if isinstance(event_spec, RecordSpec):
            event_spec = OutputSpec(event_spec)
        elif not isinstance(event_spec, OutputSpec):
            raise TypeError(
                f"ConditionalDistributionSpec.event_spec must be an OutputSpec or a RecordSpec, "
                f"got {type(event_spec).__name__}"
            )
        if event_spec.term_spec is None:
            raise ValueError("ConditionalDistributionSpec.event_spec does not declare a type")
        given_spec = _complete_given_spec(given_spec)
        _check_disjoint_sides(given_spec, event_spec, "ConditionalDistributionSpec")
        object.__setattr__(self, "given_spec", given_spec)
        object.__setattr__(self, "event_spec", event_spec)

    @property
    def free_dims(self) -> frozenset[str]:
        """The unbound dimensions of both sides, which share one scope."""
        return self.given_spec.free_dims | self.event_spec.term_spec.free_dims

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> ConditionalDistributionSpec:
        """This spec with *bindings* substituted on both sides."""
        declaration = self.event_spec
        return ConditionalDistributionSpec(
            self.given_spec._substitute_dims(bindings),
            declaration._with_term_spec(declaration.term_spec._substitute_dims(bindings)),
        )

    def _unify_with(
        self, given_spec: InputSpec, event_spec: OutputSpec, bindings: dict[str, int], path: str
    ) -> None:
        """Unify both sides against another kernel's, in the scope *bindings*.

        Parameters
        ----------
        given_spec : InputSpec
            The other kernel's inputs.
        event_spec : OutputSpec
            The other kernel's event spec.
        bindings : dict[str, int]
            The size of each symbolic dimension bound so far, keyed by its name,
            which unification extends in place.
        path : str
            The location of the declaration that error messages name, such as
            ``"the declaration"``.

        Raises
        ------
        ValueError
            If the inputs differ in name or their specs do not unify, or the
            event specs do not.
        """
        if set(self.given_spec) != set(given_spec):
            raise ValueError(
                f"{path} declares the inputs {sorted(self.given_spec)}, but the kernel "
                f"conditions on {sorted(given_spec)}"
            )
        for name, spec in self.given_spec.items():
            _unify_specs(spec, given_spec[name], bindings, f"{path}/given/{name}")
        _unify_declarations(self.event_spec, event_spec, bindings, path)

    def _bind_dims_from_value(self, value: Any, bindings: dict[str, int], path: str) -> None:
        """Bind both sides against the declarations *value* carries."""
        if not isinstance(value, ConditionalDistribution):
            raise ValueError(f"{path} does not conform to its field spec ({self!r})")
        self._unify_with(value.given_spec, value.event_spec, bindings, path)

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        """Bind both sides against *actual*'s own."""
        if not isinstance(actual, ConditionalDistributionSpec):
            return False
        self._unify_with(actual.given_spec, actual.event_spec, bindings, path)
        return True

    def __repr__(self) -> str:
        """The inputs and the event spec, as the constructor takes them."""
        return term_repr(
            "ConditionalDistributionSpec",
            None,
            [("given_spec", repr(self.given_spec)), ("event_spec", repr(self.event_spec))],
        )

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a ``ConditionalDistribution`` whose declarations match these."""
        if not isinstance(value, ConditionalDistribution):
            return False
        try:
            self._unify_with(value.given_spec, value.event_spec, {}, "the declaration")
        except ValueError:
            return False
        return True


# ---------------------------------------------------------------------------
# ConditionalDistribution — the kernel
# ---------------------------------------------------------------------------


class _ConditionalDistributionMeta(_TrackedTermMeta):
    """The metaclass of every conditional distribution.

    Construction checks that the instance holds its declarations, and a numeric
    marker's membership is read from the declarations whatever the class, as
    for ``NumericDistribution``. Creating a class checks each capability guard
    it defines, as for ``Distribution``.
    """

    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        # The check runs once the class is complete, since a class that fails
        # while type.__new__ builds it has no ABC caches of its own and would
        # write into its base's.
        super().__init__(*args, **kwargs)
        _check_guards(cls)

    def __instancecheck__(cls, instance: Any) -> bool:
        marker = _DECLARATION_MARKERS.get(cls)
        if marker is not None:
            return marker[0](instance)
        return super().__instancecheck__(instance)

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        if not isinstance(getattr(instance, "_spec", None), ConditionalDistributionSpec):
            raise TypeError(
                f"{type(instance).__name__}.__init__ left the kernel undeclared; pass given_spec "
                f"and event_spec to ConditionalDistribution.__init__, or call "
                f"_init_declaration when bypassing it"
            )
        _check_marker_claims(instance)
        return instance


class ConditionalDistribution(TrackedTerm, Annotated, ABC, metaclass=_ConditionalDistributionMeta):
    """A probability kernel ``K : S → P(T)``: a law over its event for each given value.

    Supplying a value for the inputs yields an ordinary ``Distribution``
    over what the kernel produces. A kernel always conditions on at least one
    input, since binding the last one returns a ``Distribution``, and the two
    kinds are distinct: neither inherits from the other.

    The kernel stores one ``ConditionalDistributionSpec``, its :attr:`spec`;
    :attr:`given_spec` and :attr:`event_spec` are views on it. The event
    declaration is read as a ``Distribution``'s is: a bare ``RecordSpec``
    exposes its fields, and an ``OutputSpec`` names the output of a whole
    term. The inputs and the produced
    outputs are distinct roles, so their names are disjoint even when the
    two spaces coincide, as in a Markov kernel ``state → next_state``. Symbolic
    dimensions are scoped over both sides jointly.

    Users call operations rather than methods: ``condition_on(K, s)`` binds the
    inputs, and ``sample(K, given=s)``, ``log_prob(K, y, given=s)``, and
    ``mean(K, given=s)`` are the fused conditional calls, equal to the same
    operation on ``condition_on(K, s)``. A subclass implements
    :meth:`_condition_on`, and it may claim the conditional capabilities.

    The conditional methods, such as ``_conditional_sample``, must be pure:
    their result is determined by the given values and the key or the event
    they take. A joint that draws or scores the kernel at many given values
    compiles the method, vectorized over those values, and keeps the compiled
    program for later calls with the same constants and input shapes. A call
    that reuses the program computes with the values that the globals the
    method reads had when the program was compiled. A call that compiles a new
    program, or that runs uncompiled, reads their current values. A Python side
    effect of the method, such as a warning, happens when a program is
    compiled. A family skips its checks of a traced parameter value, such as
    that mixture weights are nonnegative, so inside the program an invalid
    parameter computed from an unmet given or from a constant of the method
    gives draws or densities such as ``nan`` rather than an error.

    ``str(K)`` returns the kernel's :attr:`notation`, its label followed by its
    signature, as ``glm(y | beta)``, and the repr reads as a call of the
    constructor, with ``label=`` where a caller gave a label other than the
    default.

    Parameters
    ----------
    given_spec : InputSpec or Mapping[str, TermSpec]
        The named inputs the kernel conditions on, at least one; the keys are
        Python identifiers.
    event_spec : OutputSpec or RecordSpec
        The declaration of one produced draw, completed as above.
    label : str, optional
        The kernel's label, a non-empty string, ``p`` by default. A subclass's
        constructor passes the label its caller gave, or its own default.
    _provenance : Provenance, optional
        The provenance of the kernel that a reconstruction rebuilds. By default the
        provenance stays unset until ``with_provenance`` attaches one.
    _annotations : Mapping[str, Any], optional
        The annotations of the kernel that a reconstruction rebuilds, copied into
        the kernel's own store. By default :attr:`annotations` is ``None``.

    Raises
    ------
    TypeError
        If *label* is not a non-empty string, *given_spec* is not a mapping of
        term specs, or *event_spec* is neither an ``OutputSpec`` nor a
        ``RecordSpec``.
    ValueError
        If *given_spec* has no inputs, *event_spec* has an unspecified output type, or a given
        input shares a name with an output.
    """

    #: The label the constructor gives when it is given none, which the repr leaves out;
    #: ``None`` for a class whose constructor requires a label.
    _default_label: ClassVar[str | None] = None

    #: The memo of values computed from the kernel on demand, such as the
    #: compiled programs of a joint that maps it over draws, which a copy or a
    #: pickle leaves out. A program calls the kernel it was built for, so a copy
    #: that holds other bound values must build its own.
    _transient_state = ("_memo",)

    #: Where the constructor takes the output of a whole-term event, which the
    #: repr follows, as for a law.
    _repr_output_name: ClassVar[str | None] = "first"

    def __init__(
        self,
        given_spec: InputSpec | Mapping[str, TermSpec],
        event_spec: OutputSpec | TermSpec,
        *,
        label: str | None = None,
        _provenance: Provenance | None = None,
        _annotations: Mapping[str, Any] | None = None,
    ) -> None:
        label = _constructor_label(self, label, DEFAULT_LABEL) if label is None else label
        _given_label(label, owner=public_class_name(type(self)))
        self._init_tracked(label, provenance=_provenance)
        _record_default_expression(self)
        self._init_annotations(_annotations)
        self._init_declaration(given_spec, event_spec)

    def _init_declaration(
        self, given_spec: InputSpec | Mapping[str, TermSpec], event_spec: OutputSpec | TermSpec
    ) -> None:
        """Complete both declarations and store them as this kernel's spec.

        The constructor calls this after setting the label; a class that bypasses
        the constructor calls it itself.
        """
        object.__setattr__(
            self,
            "_spec",
            ConditionalDistributionSpec(
                _complete_given_spec(given_spec), _complete_event_spec(event_spec)
            ),
        )

    # -- the representation -------------------------------------------------

    @property
    def raw(self) -> ConditionalDistribution:
        """This kernel detached from the workflow, under its label and declarations.

        A kernel is represented by itself, so its raw form is a copy that shares
        its representation and carries no provenance, no annotations, and no
        reference to a batch it was an element of.

        Every read returns the same copy, which refuses provenance, since every
        reader shares it. ``with_label`` gives a copy of it that can carry
        provenance.
        """
        return _cached_raw(self, lambda: _detached_term(self))

    # -- the declarations ---------------------------------------------------

    @property
    def spec(self) -> ConditionalDistributionSpec:
        """The kernel's term spec, the one stored source of both declarations."""
        return self._spec

    @property
    def given_spec(self) -> InputSpec:
        """The named inputs the kernel conditions on, read from :attr:`spec`."""
        return self.spec.given_spec

    @property
    def event_spec(self) -> OutputSpec:
        """The output spec of one produced draw, read from :attr:`spec`."""
        return self.spec.event_spec

    # -- dimension and name transforms --------------------------------------

    def with_dim_sizes(self, **sizes: int) -> Self:
        """Bind named symbolic dimensions on both sides.

        Parameters
        ----------
        **sizes : int
            Sizes for free dimensions of either side.

        Returns
        -------
        Self
            A copy of the same class and label with the sizes substituted on
            both sides; the original is unchanged.

        Raises
        ------
        ValueError
            If a name is not a free dimension of either side, or a size is
            negative.
        TypeError
            If a size is not an integer.
        """
        unbound = set(sizes) - self.spec.free_dims
        if unbound:
            raise ValueError(_no_free_dims(self, unbound, self.spec.free_dims))
        return self._with_declarations(
            self.given_spec.with_dim_sizes(**sizes),
            self.event_spec.with_dim_sizes(**sizes),
            "with_dim_sizes",
            sizes,
        )

    def with_dim_names(self, **names: str) -> Self:
        """Rename symbolic dimensions on both sides, simultaneously.

        Parameters
        ----------
        **names : str
            New names keyed by old; names that are not free are ignored.

        Returns
        -------
        Self
            A copy of the same class and label with the dimensions renamed on
            both sides; the original is unchanged.
        """
        return self._with_declarations(
            self.given_spec.with_dim_names(**names),
            self.event_spec.with_dim_names(**names),
            "with_dim_names",
            names,
        )

    def with_path_names(
        self, mapping: Mapping[str, str] | None = None, /, **kwargs: str
    ) -> ConditionalDistribution:
        """Rename or move inputs and event paths, ``old -> new``.

        A key that starts with an input addresses the given side, and one
        that starts with an output the event side, which behaves as
        :meth:`Distribution.with_path_names`. On the given side a target is the
        node's new exact path, so it may split or group inputs, since a kernel has
        no signature that fixes its top level: ``{"a": "theta/a"}`` moves the
        input ``a`` into a structured input ``theta``, and ``{"theta/a": "a"}``
        moves the field back out as an input. A node renamed within its parent
        keeps its position, a moved node is appended to its new parent, and a
        structured input that the moves empty dissolves. The kernel is unchanged:
        the result translates a given value to this kernel's inputs before
        binding it, and renames what the binding returns.

        Parameters
        ----------
        mapping : Mapping[str, str], optional
            New names or paths keyed by the exact paths of the nodes they rename.
        **kwargs : str
            Further renames, keyed by paths that are identifiers.

        Returns
        -------
        ConditionalDistribution
            The renamed kernel under the same label; the original is unchanged.

        Raises
        ------
        KeyError
            If a key is neither a path of the given side nor an event path.
        ValueError
            If a target is empty or malformed, a node is renamed twice, no
            renames are given, a given target lies inside its own node or
            overlaps another, two nodes land on one path, an input name is not an
            identifier, or the renamed sides share a name.
        NotImplementedError
            If this kernel is factored, since a joint renames through its factors.
        """
        pairs: dict[str, str] = {}
        for source in (mapping or {}), kwargs:
            for old, new in source.items():
                if old in pairs:
                    raise ValueError(f"with_path_names(): {old!r} is renamed more than once")
                pairs[old] = new
        if not pairs:
            raise ValueError("with_path_names() requires at least one rename")
        moves: dict[str, str] = {}
        renames: dict[str, str] = {}
        for old, new in pairs.items():
            head = old.split(_PATH_SEP, 1)[0]
            if head in self.given_spec:
                moves[old] = new
            elif head in self.event_spec.outputs:
                renames[old] = new
            else:
                raise KeyError(
                    f"with_path_names(): {old!r} is neither an input nor a field of "
                    f"{self.label!r}; inputs: {list(self.given_spec)}, fields: "
                    f"{list(self.event_spec.outputs)}"
                )
        given_spec, origins = _moved_inputs(self.given_spec, moves)
        event_spec = self.event_spec.with_path_names(renames) if renames else self.event_spec
        _check_disjoint_sides(given_spec, event_spec, repr(self.label))
        if _renamed_kernel_factory is None:
            raise RuntimeError("the renamed kernel is not installed; import probpipe")
        return _renamed_kernel_factory(self, given_spec, event_spec, origins, renames, pairs)

    def _with_declarations(
        self,
        given_spec: InputSpec,
        event_spec: OutputSpec,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Self:
        """A copy of this kernel holding the declarations, with provenance recording *operation*."""
        clone = self._shallow_copy()
        object.__setattr__(clone, "_spec", ConditionalDistributionSpec(given_spec, event_spec))
        object.__setattr__(clone, "_provenance", None)
        clone.with_provenance(
            Provenance.create(operation, parents=[self], metadata=dict(arguments))
        )
        return clone

    # -- the primitive --------------------------------------------------------

    @abstractmethod
    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **options: Any
    ) -> Distribution | ConditionalDistribution:
        """The law ``K(given, ·)``, or the kernel over the rest for a partial given.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values for some or all of the inputs, by input name; every given
            value arrives here.
        **options : Any
            Options that configure the kernel or the method that evaluates it,
            such as a budget.

        Returns
        -------
        Distribution or ConditionalDistribution
            A ``Distribution`` when every input is bound, and otherwise the
            kernel over the remaining inputs.
        """

    # -- composition ------------------------------------------------------------

    def __mul__(
        self, other: Distribution | ConditionalDistribution
    ) -> FactoredConditionalDistribution | Distribution:
        """The joint of this kernel and *other*, composed conditional-first.

        This kernel may condition on what *other* produces. The result is a
        ``FactoredDistribution`` when *other* meets every given, and a
        ``FactoredConditionalDistribution`` over the unmet givens otherwise;
        see :meth:`Distribution.__mul__`.
        """
        return _compose_operands(self, other)

    # -- notation ---------------------------------------------------------------

    @property
    def notation(self) -> str:
        """The kernel's label followed by its signature, as ``glm(y | beta)``, which ``str()`` returns.

        The signature lists the event outputs in declaration order, then
        `` | `` and every input in declaration order, then ``;`` and the
        paths the kernel holds fixed at given values when it holds any, as
        ``glm(y | sigma; beta)``. Outputs, inputs, and paths are each joined by
        ``", "``. A product without a label reads factor by factor, as
        ``lik(y | beta)·prior(beta | tau)``. The notation is a rendering of the
        kernel's expression, made each time it is shown, which shows at most
        the current ``config.notation.max_depth`` nested levels. No operation
        reads the notation.
        """
        return self._expression.render_notation(self._own_signature(), warn=True)

    def _own_signature(self) -> Signature:
        """The signature the declaration states: the event outputs, then the inputs.

        An optional input shows its default, as ``n0=50.0``, or ``n0=…`` where
        the default is not a scalar or the kernel does not know its value.
        """
        known = self._given_defaults()
        defaults = tuple(
            (input_name, format_default(known[input_name]) if input_name in known else ELLIPSIS)
            for input_name in self.given_spec
            if input_name in self.given_spec.optional
        )
        return Signature(tuple(self.event_spec.outputs), tuple(self.given_spec), defaults)

    def _given_defaults(self) -> Mapping[str, Any]:
        """The value each optional input takes when a binding omits it, where the kernel knows it."""
        return {}

    def __str__(self) -> str:
        """The kernel's :attr:`notation`, as ``glm(y | beta)``."""
        return self.notation

    def __repr__(self) -> str:
        """A call of the kernel's public constructor, followed by the paths it holds fixed.

        The output, the family parameters, and the inputs come first,
        as ``given=('mu',)``, then ``label=`` and ``event_spec=`` where they
        differ from the constructor's defaults, and the inputs that conditioning
        bound follow as ``fixed=(...)``, as for a law.
        """
        arguments = [*self._repr_arguments(), ("given", format_names(self.given_spec))]
        return _law_repr(self, arguments)

    def _repr_class_name(self) -> str:
        """The first public class in this kernel's method-resolution order, which the repr names."""
        return public_class_name(type(self))

    def _derived_label(self) -> str | None:
        """The label this kernel's constructor gives it when it is given no label, as for a law."""
        return self._default_label

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The family parameters the repr shows, each by name and formatted value; none here."""
        return []

    def _event_repr_parts(self) -> tuple[str | None, str | None]:
        """The output of the event and the declaration's repr, as for a law."""
        return _event_repr_parts(self.event_spec)


# ---------------------------------------------------------------------------
# The numeric markers
# ---------------------------------------------------------------------------


def _event_is_numeric(value: Any) -> bool:
    declared = getattr(value, "_spec", None)
    return isinstance(declared, ConditionalDistributionSpec) and isinstance(
        declared.event_spec.term_spec, NumericSpec
    )


def _given_side_is_numeric(value: Any) -> bool:
    declared = getattr(value, "_spec", None)
    return isinstance(declared, ConditionalDistributionSpec) and _given_is_numeric(
        declared.given_spec
    )


class ConditionalNumericDistribution(ConditionalDistribution):
    """The marker of a kernel whose event side is numeric.

    Every ``K(s, ·)`` is then a ``NumericDistribution``. Membership is read from
    the event spec, and a class that inherits the marker claims it for
    every instance, which construction checks. The marker adds no operations.
    """

    _membership_follows_declaration = True


class NumericConditionalDistribution(ConditionalDistribution):
    """The marker of a kernel whose given side is numeric: every input declares a numeric term.

    Membership is read from the inputs, as for
    :class:`ConditionalNumericDistribution`.
    """

    _membership_follows_declaration = True


class FullyNumericConditionalDistribution(
    NumericConditionalDistribution, ConditionalNumericDistribution
):
    """The marker of a kernel whose given and event sides are both numeric."""

    _membership_follows_declaration = True


_DECLARATION_MARKERS[ConditionalNumericDistribution] = (_event_is_numeric, "a numeric event")
_DECLARATION_MARKERS[NumericConditionalDistribution] = (
    _given_side_is_numeric,
    "numeric inputs",
)
_DECLARATION_MARKERS[FullyNumericConditionalDistribution] = (
    lambda value: _event_is_numeric(value) and _given_side_is_numeric(value),
    "numeric inputs and a numeric event",
)


# ---------------------------------------------------------------------------
# The kernel of a function of its given values
# ---------------------------------------------------------------------------

#: Each capability of a law that the kernel of a function derives from the law
#: its function returns, with the conditional twin the kernel then claims and
#: the law's method that the twin calls.
_DERIVED_TWINS: tuple[tuple[type, type, str], ...] = (
    (SupportsSampling, SupportsConditionalSampling, "_sample"),
    (SupportsLogProb, SupportsConditionalLogProb, "_log_prob"),
    (SupportsUnnormalizedLogProb, SupportsConditionalUnnormalizedLogProb, "_unnormalized_log_prob"),
)


def _probed_guard(method: str, condition: str) -> Callable[..., Feasibility]:
    """The guard of a derived twin: the report of the law's guard of *method*.

    The report is the one the law the function returned at construction gave,
    and *condition* is the guard's condition in words.
    """

    def guard(self: Any, *arguments: Any, **keywords: Any) -> Feasibility:
        return self._guards[method]

    guard.__doc__ = condition
    return guard


def _kernel_sample(self: Any, given: Any, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
    """Draws of the law the function returns at a value of every input."""
    return self._law(given)._sample(key, sample_shape)


def _kernel_log_prob(self: Any, given: Any, value: Any) -> Any:
    """The log-density of *value* under the law the function returns at a value of every input."""
    return self._law(given)._log_prob(value)


def _kernel_unnormalized_log_prob(self: Any, given: Any, value: Any) -> Any:
    """The unnormalized log-density of *value* under the law the function returns there."""
    return self._law(given)._unnormalized_log_prob(value)


_FUNCTION_KERNEL_CAPABILITIES: dict[type, Mapping[str, Callable[..., Any]]] = {
    SupportsConditionalSampling: {
        "_conditional_sample": _kernel_sample,
        "_conditional_sample_guard": _probed_guard(
            "_sample", "The law the function returns admits the draw, as its guard reports."
        ),
    },
    SupportsConditionalLogProb: {
        "_conditional_log_prob": _kernel_log_prob,
        "_conditional_log_prob_guard": _probed_guard(
            "_log_prob", "The law the function returns admits the density, as its guard reports."
        ),
    },
    SupportsConditionalUnnormalizedLogProb: {
        "_conditional_unnormalized_log_prob": _kernel_unnormalized_log_prob,
        "_conditional_unnormalized_log_prob_guard": _probed_guard(
            "_unnormalized_log_prob",
            "The law the function returns admits the unnormalized density, as its guard reports.",
        ),
    },
}


def _argument(spec: TermSpec, input_name: str, value: Any) -> Any:
    """*value* at the kind its input *spec* declares, as the function receives it.

    A record input's value is a ``Record`` and an array input's an array, as a
    function's body receives a draw; a value of another kind is passed as it is.
    """
    if isinstance(spec, RecordSpec):
        return (
            value
            if isinstance(value, Record)
            else Record(
                value,
                label=input_name,
            )
        )
    if isinstance(spec, NumericArraySpec):
        return jnp.asarray(value)
    return value


def _declared_law(law: Any, event_spec: OutputSpec, label: str) -> Distribution:
    """*law*, the function's result, checked against the kernel's event spec.

    Parameters
    ----------
    law : Any
        The value the kernel's function returned, which must be a ``Distribution``.
    event_spec : OutputSpec
        The kernel's event spec, which the declaration of *law* must unify
        with.
    label : str
        The kernel's label, which error messages name.

    Returns
    -------
    Distribution
        *law* itself, once both checks pass.

    Raises
    ------
    TypeError
        If *law* is not a ``Distribution``.
    ValueError
        If its event spec does not agree with *event_spec*.
    """
    if not isinstance(law, Distribution):
        raise TypeError(
            f"the function of {label!r} must return a Distribution; got {type(law).__name__}"
        )
    if not DistributionSpec(event_spec).is_valid(law):
        raise ValueError(
            f"the function of {label!r} returned a law with event {law.event_spec!r}, but "
            f"{label!r} declares {event_spec!r}"
        )
    return law


class _FunctionKernel(ConditionalDistribution):
    """The kernel of a function of its given values: its law at a value is the one the function returns.

    :func:`conditional_distribution` builds it. Binding every required input
    calls the function with each value as the argument of that name, at the
    kind its input declares, and with its default for each optional input left
    unbound, and returns the law the call returns; binding fewer inputs returns
    the kernel over the others. The kernel claims the conditional
    twin of sampling, of the normalized density, and of the unnormalized
    density exactly when the law its function returned at construction claims
    the capability, and each twin's guard reports what that law's guard did.

    Parameters
    ----------
    label : str
        The kernel's label.
    fn : callable
        The function, called with every input's value by name.
    given_spec : InputSpec
        The inputs, one per parameter of *fn*.
    event_spec : OutputSpec
        The declaration of the law *fn* returns.
    guards : Mapping[str, Feasibility]
        The report of the law's guard of each capability it claims, keyed by
        the capability's method.
    """

    _capability_table: ClassVar = _FUNCTION_KERNEL_CAPABILITIES
    _default_label: ClassVar[str | None] = DEFAULT_LABEL

    def __new__(
        cls,
        label: str,
        fn: Callable[..., Distribution],
        given_spec: InputSpec,
        event_spec: OutputSpec,
        guards: Mapping[str, Feasibility],
    ) -> _FunctionKernel:
        claimed = [twin for _, twin, method in _DERIVED_TWINS if method in guards]
        return object.__new__(_capability_subclass(_FunctionKernel, claimed))

    def __init__(
        self,
        label: str,
        fn: Callable[..., Distribution],
        given_spec: InputSpec,
        event_spec: OutputSpec,
        guards: Mapping[str, Feasibility],
    ) -> None:
        super().__init__(
            given_spec,
            event_spec,
            label=label,
        )
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_inputs", given_spec)
        object.__setattr__(self, "_bound", {})
        object.__setattr__(self, "_guards", dict(guards))

    def _given_defaults(self) -> Mapping[str, Any]:
        """The defaults of the function's parameters, which its optional inputs take."""
        try:
            parameters = inspect.signature(self._fn).parameters
        except (TypeError, ValueError):
            return {}
        return {
            name: parameter.default
            for name, parameter in parameters.items()
            if parameter.default is not parameter.empty
        }

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **options: Any
    ) -> Distribution | ConditionalDistribution:
        """The law the function returns at a value of every input, or the kernel over the rest.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values of some or all inputs, keyed by input name.
        **options : Any
            The options of the call, which must be empty.

        Returns
        -------
        Distribution or ConditionalDistribution
            The returned law once every required input is bound. Otherwise a copy of
            this kernel that holds the bound values and conditions on the other
            inputs.

        Raises
        ------
        TypeError
            If an option is passed, since evaluating the function reads none.
        KeyError
            If *given* names a key that is not an input.
        ValueError
            If a value does not conform to its input, or the returned law departs
            from the kernel's event spec.
        """
        if options:
            raise TypeError(
                f"{self.label!r} evaluates its function exactly and takes no method options; "
                f"got {sorted(options)}"
            )
        values = self._given_values(given)
        if set(self.given_spec.required) <= set(values):
            return self._law(values)
        self._check_conformance(values)
        rest = self._shallow_copy()
        object.__setattr__(rest, "_provenance", None)
        object.__setattr__(rest, "_bound", {**self._bound, **self._arguments(values)})
        left = self.given_spec.without(*values)
        object.__setattr__(rest, "_spec", ConditionalDistributionSpec(left, self.event_spec))
        return rest

    def _given_values(self, given: Any) -> dict[str, Any]:
        """The values *given* holds, by input.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values of some or all inputs, keyed by input name.

        Returns
        -------
        dict[str, Any]
            A new dict of the values, read from the children of a ``Record`` or the
            items of a mapping.

        Raises
        ------
        KeyError
            If *given* names a key that is not an input.
        """
        values = dict((given.children if isinstance(given, Record) else given).items())
        unknown = sorted(set(values) - set(self.given_spec))
        if unknown:
            raise KeyError(_unknown_inputs(self.label, unknown, self.given_spec))
        return values

    def _check_conformance(self, values: Mapping[str, Any]) -> None:
        """Raise ``ValueError`` if a value does not conform to its input."""
        InputSpec(
            {input_name: self.given_spec[input_name] for input_name in values}
        ).bind_dims_from_value(dict(values))

    def _arguments(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """*values* at the kinds their inputs declare, as the function receives them."""
        return {
            input_name: _argument(self._inputs[input_name], input_name, value)
            for input_name, value in values.items()
        }

    def _law(self, given: Any) -> Distribution:
        """The law the function returns at *given*, a value of every required input.

        An optional input that *given* leaves out takes the function's default.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values of the inputs this kernel conditions on, keyed by input name.

        Returns
        -------
        Distribution
            The law the function returns at the values a partly bound kernel holds and
            at *given*, checked against the kernel's event spec.

        Raises
        ------
        KeyError
            If a required input has no value, or *given* names another key.
        ValueError
            If a value does not conform to its input, or the law departs from the
            kernel's event spec.
        """
        values = self._given_values(given)
        missing = [
            input_name for input_name in self.given_spec.required if input_name not in values
        ]
        if missing:
            raise KeyError(_missing_inputs(self.label, missing))
        self._check_conformance(values)
        law = self._fn(**self._bound, **self._arguments(values))
        return _declared_law(law, self.event_spec, self.label)


@dataclass(frozen=True)
class _Probe:
    """What the law a function returns at stand-ins of its inputs declares and claims.

    Attributes
    ----------
    event_spec : OutputSpec
        The law's event spec, a support that depends on the given values
        left undeclared.
    guards : Mapping[str, Feasibility]
        The report of the law's guard of each derived capability it claims,
        keyed by the capability's method.
    """

    event_spec: OutputSpec
    guards: Mapping[str, Feasibility]


def _stand_in(spec: TermSpec, path: str, label: str) -> Any:
    """The abstract stand-in of a value of *spec*: an array's shape and dtype, or a mapping of them.

    Parameters
    ----------
    spec : TermSpec
        The term spec of an input, or of a field within a record input.
    path : str
        The path of that input or field within the given side, which error messages
        name.
    label : str
        The kernel's label, which error messages name.

    Returns
    -------
    jax.ShapeDtypeStruct or dict
        A ``jax.ShapeDtypeStruct`` for an array spec, of the default float dtype
        when *spec* sets none, and a dict of each field's stand-in, keyed by field
        name, for a record spec.

    Raises
    ------
    TypeError
        If *spec* is neither an array nor a record of them, or declares free
        dimensions.
    """
    if isinstance(spec, NumericArraySpec):
        if spec.free_dims:
            raise TypeError(
                f"input {path!r} of {label!r} has unbound dimensions "
                f"{sorted(spec.free_dims)}, but conditional_distribution needs a fixed shape for "
                f"each required input; give a concrete shape, such as NumericArraySpec((3,))"
            )
        dtype = spec.dtype if spec.dtype is not None else jnp.result_type(float)
        return jax.ShapeDtypeStruct(tuple(spec.shape), dtype)
    if isinstance(spec, RecordSpec):
        return {
            key: _stand_in(child, f"{path}/{key}", label) for key, child in spec.children.items()
        }
    raise TypeError(
        f"input {path!r} of {label!r} must be declared as an array or a record of arrays; "
        f"got {type(spec).__name__}"
    )


def _holds_tracer(constraint: Any) -> bool:
    """Whether the support *constraint* holds a traced value, as a bound set by a given does."""
    held = getattr(constraint, "__dict__", {})
    return any(isinstance(value, jax.core.Tracer) for value in held.values())


def _value_free(spec: TermSpec) -> TermSpec:
    """*spec* with each support that holds a traced value left undeclared."""
    if isinstance(spec, NumericArraySpec) and _holds_tracer(spec.support):
        return NumericArraySpec(spec.shape, spec.dtype, None)
    if isinstance(spec, RecordSpec):
        return RecordSpec({key: _value_free(child) for key, child in spec.children.items()})
    return spec


def _probe(label: str, fn: Callable[..., Distribution], inputs: InputSpec) -> _Probe:
    """The declaration and the guards of the law *fn* returns, evaluated abstractly.

    *fn* runs once under ``jax.eval_shape``, with a traced stand-in of each
    required input's type at the kind the input declares, so no value is assumed,
    and with its default for each optional input; a support that a given value
    sets holds a traced bound, and is left undeclared.

    Parameters
    ----------
    label : str
        The kernel's label, which error messages name.
    fn : callable
        The function of the given values that returns a law.
    inputs : InputSpec
        The kernel's inputs, which name the parameters of *fn*.

    Returns
    -------
    _Probe
        The event spec of the returned law, and the report of the guard of
        each derived capability it claims.

    Raises
    ------
    TypeError
        If an input has no stand-in, or *fn* returns something other than a law.
    """
    stand_ins = {
        input_name: _stand_in(spec, input_name, label)
        for input_name, spec in inputs.items()
        if input_name not in inputs.optional
    }
    found: dict[str, Any] = {}

    def evaluate(values: Mapping[str, Any]) -> Any:
        law = fn(
            **{
                input_name: _argument(inputs[input_name], input_name, value)
                for input_name, value in values.items()
            }
        )
        if not isinstance(law, Distribution):
            raise TypeError(
                f"the function of {label!r} must return a Distribution; got {type(law).__name__}"
            )
        found["event_spec"] = law.event_spec
        found["guards"] = {
            method: _capability_guard(law, method)
            for capability, _, method in _DERIVED_TWINS
            if isinstance(law, capability)
        }
        return jnp.zeros(())

    jax.eval_shape(evaluate, stand_ins)
    declaration = found["event_spec"]
    return _Probe(declaration._with_term_spec(_value_free(declaration.term_spec)), found["guards"])


def _inputs_of(
    label: str, fn: Callable[..., Any], given_spec: InputSpec | Mapping[str, TermSpec] | None
) -> InputSpec:
    """The inputs of the kernel of *fn*: one per parameter, declared by *given_spec* or its annotation.

    A parameter with a default is an optional input, which a binding may omit,
    and its default's value declares it when neither *given_spec* nor its
    annotation does.

    Parameters
    ----------
    label : str
        The kernel's label, which error messages name.
    fn : callable
        The function whose parameters are the inputs.
    given_spec : InputSpec or Mapping[str, TermSpec] or None
        The specs of some or all parameters, keyed by parameter name. ``None``
        declares every input by its annotation or its default.

    Returns
    -------
    InputSpec
        One input per parameter of *fn*, in signature order, with each parameter
        that has a default marked optional.

    Raises
    ------
    TypeError
        If *given_spec* names a key that is not a parameter, a parameter is
        variadic or positional-only, or a parameter without a default is
        declared by neither.
    """
    declared = dict((given_spec or {}).items())
    try:
        signature = inspect.signature(fn, eval_str=True)
    except NameError:
        signature = inspect.signature(fn)
    unknown = sorted(set(declared) - set(signature.parameters))
    if unknown:
        raise TypeError(
            f"given_spec of {label!r}: "
            f"{unknown_names('parameter', unknown, list(signature.parameters))}"
        )
    inputs: dict[str, TermSpec] = {}
    optional: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise TypeError(
                f"the parameter {parameter.name!r} of {label!r} is variadic, but each input "
                f"must be a named parameter"
            )
        if parameter.kind is parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"the parameter {parameter.name!r} of {label!r} is positional-only, but given "
                f"input values are passed by name; make it a regular parameter"
            )
        spec = declared.get(parameter.name, parameter.annotation)
        if parameter.default is not parameter.empty:
            optional.append(parameter.name)
            if not isinstance(spec, TermSpec):
                spec = RecordSpec.infer_from({"default": parameter.default}).children["default"]
        if not isinstance(spec, TermSpec):
            raise TypeError(
                f"the input {parameter.name!r} of {label!r} declares no term spec. Pass "
                f"given_spec={{{parameter.name!r}: NumericArraySpec(())}} for a real scalar, or "
                f"NumericArraySpec((n,)) for a vector of length n, or annotate the parameter "
                f"with the spec"
            )
        inputs[parameter.name] = spec
    return InputSpec(inputs).with_optional(*optional)


def _agreed_event_spec(label: str, declared: OutputSpec | TermSpec, law: OutputSpec) -> OutputSpec:
    """*declared*, completed from *law*, the declaration of the law the function returns.

    Parameters
    ----------
    label : str
        The kernel's label, which error messages name.
    declared : OutputSpec or TermSpec
        The event spec the caller passed. A bare ``RecordSpec`` exposes
        its fields, and any other bare term spec declares a whole term under the
        output of *law*.
    law : OutputSpec
        The event spec of the law the function returns at the stand-ins.

    Returns
    -------
    OutputSpec
        *declared* under its own outputs and packaging, holding the
        unification of its type with the type of *law*.

    Raises
    ------
    ValueError
        If *declared* names other outputs or another packaging than *law*,
        or its type does not unify with the law's.
    """
    if isinstance(declared, OutputSpec):
        declaration = declared
    elif isinstance(declared, RecordSpec):
        declaration = OutputSpec(declared)
    elif not _has_record_outputs(law) and len(law.outputs) == 1:
        declaration = OutputSpec.default(declared, output_name=next(iter(law.outputs)))
    else:
        raise ValueError(
            f"event_spec of {label!r} needs an explicit OutputSpec for the law's outputs "
            f"{list(law.outputs)}"
        )
    if _has_record_outputs(declaration) != _has_record_outputs(law) or tuple(
        declaration.outputs
    ) != tuple(law.outputs):
        raise ValueError(
            f"event_spec of {label!r} declares the fields {list(declaration.outputs)}, but its "
            f"function returns a law with {list(law.outputs)}"
        )
    return declaration.with_term_spec(law.spec)


def _function_kernel(
    label: str | None,
    fn: Any,
    given_spec: InputSpec | Mapping[str, TermSpec] | None,
    event_spec: OutputSpec | TermSpec | None,
) -> ConditionalDistribution:
    """The kernel of *fn*, labeled *label* or after the function.

    Parameters
    ----------
    label : str or None
        The kernel's label. ``None`` takes the label from ``fn.__name__``, and
        ``ℙ`` for a lambda or a callable without a name.
    fn : callable
        The function of the given values that returns a law.
    given_spec : InputSpec or Mapping[str, TermSpec] or None
        The specs of some or all inputs, keyed by parameter name, which
        :func:`_inputs_of` completes.
    event_spec : OutputSpec or TermSpec or None
        The declared event, which must agree with the declaration of the law *fn*
        returns. ``None`` takes the returned law's declaration.

    Returns
    -------
    ConditionalDistribution
        The kernel, which claims the conditional twin of each derived capability
        that the returned law claims.

    Raises
    ------
    TypeError
        If *fn* is not callable, or as :func:`_inputs_of` and :func:`_probe` raise.
    ValueError
        As :func:`_agreed_event_spec` raises.
    """
    if not callable(fn):
        raise TypeError(f"conditional_distribution: fn must be callable; got {type(fn).__name__}")
    if label is None:
        name = getattr(fn, "__name__", None)
        label = name if isinstance(name, str) and name.isidentifier() else DEFAULT_LABEL
    inputs = _inputs_of(label, fn, given_spec)
    probe = _probe(label, fn, inputs)
    declaration = (
        probe.event_spec
        if event_spec is None
        else _agreed_event_spec(label, event_spec, probe.event_spec)
    )
    return _FunctionKernel(label, fn, inputs, declaration, probe.guards)


def conditional_distribution(
    fn: Callable[..., Distribution] | None = None,
    /,
    *,
    label: str | None = None,
    given_spec: InputSpec | Mapping[str, TermSpec] | None = None,
    event_spec: OutputSpec | TermSpec | None = None,
) -> Any:
    """Build a ``ConditionalDistribution`` from a function of its given values that returns a law.

    Each parameter of the function is an input, declared by its entry in
    *given_spec* or else by its annotation, which is then a term spec. A
    parameter with a default is an optional input, which its default's value
    declares when neither does. Construction evaluates the function once,
    abstractly, at stand-ins of the required inputs' types and at the defaults
    of the optional ones, and reads the law it returns: its event spec
    is the kernel's, unless *event_spec* declares one that agrees with it, and
    the kernel claims conditional sampling and the conditional density exactly
    when that law claims sampling and a density, each with the law's guard.
    Binding every required input calls the function with each value as the
    argument of that name, at the kind its input declares, and with its default
    for each optional input left unbound, and returns the law the call returns;
    binding fewer inputs returns the kernel over the rest.

    The kernel is labeled *label*, or after the function's ``__name__``, and
    ``ℙ`` for a lambda. The call form takes the function first::

        likelihood = conditional_distribution(
            lambda mu, tau: Normal("y", mu, tau), label="lik", given_spec={"mu": real, "tau": scale}
        )

    An optional input holds a constant of the model. In a joint, a factor that
    produces an output of its name meets it, so one kernel serves a model
    with the default and a model with a prior on the input::

        counts = conditional_distribution(
            lambda r, n0=50.0: Poisson("y", n0 * jnp.exp(r)),
            label="counts",
            given_spec={"r": real},
        )
        fixed = counts * Normal("r", 0.0, 1.0)                           # n0 is 50
        uncertain = counts * (Normal("r", 0.0, 1.0) * LogNormal("n0", 4.0, 0.3))

    and the decorator form labels the kernel after the function, or as given::

        @conditional_distribution(given_spec=prior.event_spec.outputs)
        def y(population, groups, sigma=1.0):
            return Normal("y", population["mu"] + population["tau"] * groups["theta"], sigma)

    Parameters
    ----------
    fn : callable, optional
        The function of the given values that returns a law; without it, the
        result is a decorator.
    label : str, optional
        The kernel's label. Defaults to the function's ``__name__``, and to
        ``ℙ`` for a lambda or a callable without a name.
    given_spec : InputSpec or Mapping[str, TermSpec], optional
        The specs of some or all inputs, keyed by parameter.
    event_spec : OutputSpec or TermSpec, optional
        The event spec, which must name the outputs of the returned
        law and unify with its type. A bare term spec other than a record
        declares the returned law's one output.

    Returns
    -------
    ConditionalDistribution or callable
        The kernel, or a decorator that builds it from a function.

    Raises
    ------
    TypeError
        If *fn* is a string, since the label is the keyword *label*; if *label*
        is not a non-empty string; if a parameter without a default declares no
        term spec, a parameter is variadic or positional-only, or *given_spec*
        names a key that is not a parameter; if a required input is
        neither an array nor a record of arrays, or declares free dimensions;
        or if the function returns something other than a law.
    ValueError
        If *event_spec* is a bare term spec other than a ``RecordSpec`` while the
        returned law exposes a record or has more than one output, if
        *event_spec* names other outputs or another packaging than the
        returned law, or if its type does not unify with the law's.

    Notes
    -----
    The function computes the kernel's conditional draws and densities, so it
    must be pure, as ``ConditionalDistribution`` states. A value that changes
    between calls belongs in an input, as ``sigma`` is in the decorator
    example, and not in a global. A function that does not compile with its
    unmet givens traced, such as one that converts a given to a Python number,
    runs uncompiled.
    """
    if isinstance(fn, str):
        raise TypeError(label_given_first("conditional_distribution", "function", fn))
    if label is not None:
        _given_label(label, owner="conditional_distribution")
    if fn is not None:
        return _function_kernel(label, fn, given_spec, event_spec)

    def decorate(function: Callable[..., Distribution]) -> ConditionalDistribution:
        return _function_kernel(label, function, given_spec, event_spec)

    return decorate
