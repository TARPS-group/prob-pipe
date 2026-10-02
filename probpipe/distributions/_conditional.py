"""The conditional distribution: a probability kernel, its term spec, and its markers.

Provides:
  - ``ConditionalDistribution`` – a kernel ``K : S → P(T)``, which a value for
    its given slots turns into a ``Distribution``.
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

import jax
import jax.numpy as jnp

from ..core._dispatch import Feasibility
from ..core._record_spec import RecordSpec
from ..core._repr import format_names, public_class_name, term_repr
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec, _unify_specs
from ..core._specs import InputSpec, OutputSpec
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
    Distribution,
    DistributionSpec,
    _check_marker_claims,
    _complete_event_spec,
    _compose_operands,
    _detached_term,
    _is_default_declaration,
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

    Raises
    ------
    TypeError
        If *given_spec* is not a mapping of term specs.
    ValueError
        If it has no slots, since binding the last given returns a
        ``Distribution``, or a slot name is not an identifier.
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
            "a ConditionalDistribution conditions on at least one given slot; with none, "
            "it is a Distribution"
        )
    return given_spec


def _check_disjoint_sides(given_spec: InputSpec, event_spec: OutputSpec, owner: str) -> None:
    """Raise ``ValueError`` if a given slot shares a name with a produced component."""
    shared = set(given_spec) & set(event_spec.components)
    if shared:
        raise ValueError(
            f"{owner} names {sorted(shared)} both as a given slot and as a produced "
            f"component; a kernel's given and event are distinct roles, so rename one side"
        )


def _given_is_numeric(given_spec: InputSpec) -> bool:
    """Whether every given slot declares a numeric term."""
    return all(isinstance(spec, NumericSpec) for spec in given_spec.values())


# ---------------------------------------------------------------------------
# Renaming the given side
# ---------------------------------------------------------------------------

_PATH_SEP = "/"


def _given_leaf_specs(given_spec: InputSpec) -> dict[str, TermSpec]:
    """The spec of each leaf of the given side, keyed by its path, in canonical order.

    A leaf is a slot, or a field of a structured slot, whose path starts with
    the slot.
    """
    leaves: dict[str, TermSpec] = {}
    for slot, spec in given_spec.items():
        if isinstance(spec, RecordSpec):
            leaves.update({f"{slot}{_PATH_SEP}{key}": spec[key] for key in spec})
        else:
            leaves[slot] = spec
    return leaves


def _moved_slots(
    given_spec: InputSpec, moves: Mapping[str, str]
) -> tuple[InputSpec, dict[str, str]]:
    """The given side with each node at a key of *moves* moved to its new exact path.

    A key is the path of a slot or of a node within a structured slot, and its
    target is the node's new path, so a move may split a field out of a slot or
    group slots into a structured one. The moves apply simultaneously, a node
    below a moved node moves with it unless it is moved itself, a node renamed
    within its parent keeps its position, a moved node is appended to its new
    parent, and a structured slot that the moves empty dissolves.

    Returns
    -------
    tuple of InputSpec and dict
        The moved slots, and the path of each of their leaves' original leaf,
        keyed by the leaf's new path.

    Raises
    ------
    KeyError
        If a key is not the path of a node of the given side.
    ValueError
        If a target is empty or has an empty segment, lies inside its own node,
        or equals or contains another target, two nodes land on one path, or a
        new slot name is not an identifier.
    """
    leaves = _given_leaf_specs(given_spec)
    nodes = {
        _PATH_SEP.join(segments[:end])
        for segments in (path.split(_PATH_SEP) for path in leaves)
        for end in range(1, len(segments) + 1)
    }
    for old, new in moves.items():
        if old not in nodes:
            raise KeyError(old)
        if not isinstance(new, str) or not new or not all(new.split(_PATH_SEP)):
            raise ValueError(f"the new path of {old!r} has an empty segment: {new!r}")
        if new.startswith(old + _PATH_SEP):
            raise ValueError(f"{old!r} cannot move into its own node, to {new!r}")
    targets = list(moves.values())
    for first, target in enumerate(targets):
        for other in targets[first + 1 :]:
            if (
                target == other
                or other.startswith(target + _PATH_SEP)
                or target.startswith(other + _PATH_SEP)
            ):
                raise ValueError(f"the targets {target!r} and {other!r} overlap")

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
                raise ValueError(f"{target!r} lands inside the field {group!r}")
        if name in node:
            raise ValueError(f"two nodes land on {target!r}")
        node[name] = leaves[path]
        origins[target] = path

    def spec_of(node: Any) -> TermSpec:
        if isinstance(node, dict):
            return RecordSpec({name: spec_of(child) for name, child in node.items()})
        return node

    return InputSpec({slot: spec_of(node) for slot, node in tree.items()}), origins


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
    """The conditional kind's term spec: the given slots and the event declaration.

    Parameters
    ----------
    given_spec : InputSpec or Mapping[str, TermSpec]
        The named slots a kernel conditions on, at least one.
    event_spec : OutputSpec or RecordSpec
        The declaration of one produced draw, as for ``DistributionSpec``. A
        bare ``RecordSpec`` completes to the exposed form.

    Raises
    ------
    TypeError
        If *event_spec* is neither an ``OutputSpec`` nor a ``RecordSpec``, or
        *given_spec* is not a mapping of term specs.
    ValueError
        If *given_spec* has no slots, the event declaration has a type hole, or
        a given slot shares a name with a produced component.

    Notes
    -----
    ``is_valid`` accepts a ``ConditionalDistribution`` whose given slots have
    the same names and specs that unify with these, and whose event declaration
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
        if event_spec.spec is None:
            raise ValueError("ConditionalDistributionSpec.event_spec has a type hole")
        given_spec = _complete_given_spec(given_spec)
        _check_disjoint_sides(given_spec, event_spec, "ConditionalDistributionSpec")
        object.__setattr__(self, "given_spec", given_spec)
        object.__setattr__(self, "event_spec", event_spec)

    @property
    def free_dims(self) -> frozenset[str]:
        """The unbound dimensions of both sides, which share one scope."""
        return self.given_spec.free_dims | self.event_spec.spec.free_dims

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> ConditionalDistributionSpec:
        """This spec with *bindings* substituted on both sides."""
        declaration = self.event_spec
        return ConditionalDistributionSpec(
            InputSpec(
                {name: spec._substitute_dims(bindings) for name, spec in self.given_spec.items()}
            ),
            declaration._with_spec(declaration.spec._substitute_dims(bindings)),
        )

    def _unify_with(
        self, given_spec: InputSpec, event_spec: OutputSpec, bindings: dict[str, int], path: str
    ) -> None:
        """Unify both sides against another kernel's, in the scope *bindings*.

        Raises
        ------
        ValueError
            If the given slots differ in name or their specs do not unify, or the
            event declarations do not.
        """
        if set(self.given_spec) != set(given_spec):
            raise ValueError(
                f"{path} declares the given slots {sorted(self.given_spec)}, but the kernel "
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
        """The given slots and the event declaration, as the constructor takes them."""
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

    Supplying a value for the given slots yields an ordinary ``Distribution``
    over what the kernel produces. A kernel always conditions on at least one
    slot, since binding the last one returns a ``Distribution``, and the two
    kinds are distinct: neither inherits from the other.

    The kernel stores one ``ConditionalDistributionSpec``, its :attr:`spec`;
    :attr:`given_spec` and :attr:`event_spec` are views on it. The event
    declaration is read as a ``Distribution``'s is: a bare ``RecordSpec``
    exposes its fields, and any other term spec is a whole term whose component
    defaults to the kernel's ``name``. The given slots and the produced
    components are distinct roles, so their names are disjoint even when the
    two spaces coincide, as in a Markov kernel ``state → next_state``. Symbolic
    dimensions are scoped over both sides jointly.

    Users call operations rather than methods: ``condition_on(K, s)`` binds the
    given slots, and ``sample(K, given=s)``, ``log_prob(K, y, given=s)``, and
    ``mean(K, given=s)`` are the fused conditional paths, equal to the same
    operation on ``condition_on(K, s)``. A subclass implements
    :meth:`_condition_on`, and it may claim the conditional capabilities.

    Parameters
    ----------
    name : str
        Non-empty name for this kernel.
    given_spec : InputSpec or Mapping[str, TermSpec]
        The named slots the kernel conditions on, at least one; the keys are
        Python identifiers.
    event_spec : OutputSpec or TermSpec
        The declaration of one produced draw, completed as above.

    Raises
    ------
    TypeError
        If *name* is not a non-empty string, *given_spec* is not a mapping of
        term specs, or *event_spec* is not a spec.
    ValueError
        If *given_spec* has no slots, *event_spec* has a type hole, a given slot
        shares a name with a produced component, or *event_spec* is a bare term
        spec other than a record and *name* is not a valid component name.
    """

    def __init__(
        self,
        name: str,
        given_spec: InputSpec | Mapping[str, TermSpec],
        event_spec: OutputSpec | TermSpec,
        *,
        _provenance: Provenance | None = None,
        _annotations: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"{type(self).__name__} requires a non-empty label as its first argument"
            )
        self._init_tracked(name, provenance=_provenance)
        self._init_annotations(_annotations)
        self._init_declaration(given_spec, event_spec)

    def _init_declaration(
        self, given_spec: InputSpec | Mapping[str, TermSpec], event_spec: OutputSpec | TermSpec
    ) -> None:
        """Complete both declarations and store them as this kernel's spec.

        The constructor calls this after setting the name; a class that bypasses
        the constructor calls it itself.
        """
        object.__setattr__(
            self,
            "_spec",
            ConditionalDistributionSpec(
                _complete_given_spec(given_spec), _complete_event_spec(event_spec, self._label)
            ),
        )

    # -- the representation -------------------------------------------------

    def raw(self) -> ConditionalDistribution:
        """This kernel detached from the workflow, under its name and declarations.

        A kernel is represented by itself, so its raw form is a copy that shares
        its representation and carries no provenance, no annotations, and no
        reference to a batch it was an element of.
        """
        return _detached_term(self)

    # -- the declarations ---------------------------------------------------

    @property
    def spec(self) -> ConditionalDistributionSpec:
        """The kernel's term spec, the one stored source of both declarations."""
        return self._spec

    @property
    def given_spec(self) -> InputSpec:
        """The named slots the kernel conditions on, read from :attr:`spec`."""
        return self.spec.given_spec

    @property
    def event_spec(self) -> OutputSpec:
        """The output declaration of one produced draw, read from :attr:`spec`."""
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
            A copy of the same class and name with the sizes substituted on
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
            raise ValueError(
                f"{type(self).__name__} {self.label!r} has no free dimensions "
                f"{sorted(unbound)} to bind"
            )
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
            A copy of the same class and name with the dimensions renamed on
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
        """Rename or move given slots and event paths, ``old -> new``.

        A key that starts with a given slot addresses the given side, and one
        that starts with a produced component the event side, which behaves as
        :meth:`Distribution.with_path_names`. On the given side a target is the
        node's new exact path, so it may split or group slots, since a kernel has
        no signature that fixes its top level: ``{"a": "theta/a"}`` moves the
        slot ``a`` into a structured slot ``theta``, and ``{"theta/a": "a"}``
        moves the field back out as a slot. A node renamed within its parent
        keeps its position, a moved node is appended to its new parent, and a
        structured slot that the moves empty dissolves. The kernel is unchanged:
        the result translates a given value to this kernel's slots before
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
            The renamed kernel under the same name; the original is unchanged.

        Raises
        ------
        KeyError
            If a key is neither a path of the given side nor an event path.
        ValueError
            If a target is empty or malformed, a node is renamed twice, no
            renames are given, a given target lies inside its own node or
            overlaps another, two nodes land on one path, a slot name is not an
            identifier, or the renamed sides share a name.
        NotImplementedError
            If this kernel is factored, since a joint renames through its factors.
        """
        pairs: dict[str, str] = {}
        for source in (mapping or {}), kwargs:
            for old, new in source.items():
                if old in pairs:
                    raise ValueError(f"node {old!r} is renamed more than once")
                pairs[old] = new
        if not pairs:
            raise ValueError("with_path_names() requires at least one rename")
        moves: dict[str, str] = {}
        renames: dict[str, str] = {}
        for old, new in pairs.items():
            head = old.split(_PATH_SEP, 1)[0]
            if head in self.given_spec:
                moves[old] = new
            elif head in self.event_spec.components:
                renames[old] = new
            else:
                raise KeyError(old)
        given_spec, origins = _moved_slots(self.given_spec, moves)
        event_spec = self.event_spec.with_path_names(renames) if renames else self.event_spec
        _check_disjoint_sides(given_spec, event_spec, type(self).__name__)
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
        """The law ``K(given, ·)``, or a curried kernel for a partial given.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values for some or all of the given slots, by slot name; every given
            value arrives here.
        **options : Any
            Options that configure the kernel or the method that evaluates it,
            such as a budget.

        Returns
        -------
        Distribution or ConditionalDistribution
            A ``Distribution`` when every slot is bound, and otherwise the
            kernel over the remaining slots.
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

    def __repr__(self) -> str:
        """The public class, the label, the family parameters, the given slots, and the declaration.

        The event declaration is shown when it differs from the default for the
        kernel's label, as for a law.
        """
        fields = [
            *self._repr_arguments(),
            ("given", format_names(self.given_spec)),
            *self._event_repr_arguments(),
        ]
        return term_repr(self._repr_class_name(), self.label, fields)

    def _repr_class_name(self) -> str:
        """The first public class in this kernel's method-resolution order, which the repr names."""
        return public_class_name(type(self))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The family parameters the repr shows, each by name and formatted value; none here."""
        return []

    def _event_repr_arguments(self) -> list[tuple[str, str]]:
        """The event declaration, unless it is the default for this kernel's label."""
        if _is_default_declaration(self.event_spec, self.label):
            return []
        return [("event_spec", repr(self.event_spec))]


# ---------------------------------------------------------------------------
# The numeric markers
# ---------------------------------------------------------------------------


def _event_is_numeric(value: Any) -> bool:
    declared = getattr(value, "_spec", None)
    return isinstance(declared, ConditionalDistributionSpec) and isinstance(
        declared.event_spec.spec, NumericSpec
    )


def _given_side_is_numeric(value: Any) -> bool:
    declared = getattr(value, "_spec", None)
    return isinstance(declared, ConditionalDistributionSpec) and _given_is_numeric(
        declared.given_spec
    )


class ConditionalNumericDistribution(ConditionalDistribution):
    """The marker of a kernel whose event side is numeric.

    Every ``K(s, ·)`` is then a ``NumericDistribution``. Membership is read from
    the event declaration, and a class that inherits the marker claims it for
    every instance, which construction checks. The marker adds no operations.
    """

    _membership_follows_declaration = True


class NumericConditionalDistribution(ConditionalDistribution):
    """The marker of a kernel whose given side is numeric: every slot declares a numeric term.

    Membership is read from the given slots, as for
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
    "numeric given slots",
)
_DECLARATION_MARKERS[FullyNumericConditionalDistribution] = (
    lambda value: _event_is_numeric(value) and _given_side_is_numeric(value),
    "numeric given slots and a numeric event",
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


def _probed_guard(method: str, condition: str) -> Callable[[Any], Feasibility]:
    """The guard of a derived twin: the report of the law's guard of *method*.

    The report is the one the law the function returned at construction gave,
    and *condition* is the guard's condition in words.
    """

    def guard(self: Any) -> Feasibility:
        return self._guards[method]

    guard.__doc__ = condition
    return guard


def _kernel_sample(self: Any, given: Any, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
    """Draws of the law the function returns at a value of every given slot."""
    return self._law(given)._sample(key, sample_shape)


def _kernel_log_prob(self: Any, given: Any, value: Any) -> Any:
    """The log-density of *value* under the law the function returns at a value of every slot."""
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


def _argument(spec: TermSpec, slot: str, value: Any) -> Any:
    """*value* at the kind its slot *spec* declares, as the function receives it.

    A record slot's value is a ``Record`` and an array slot's an array, as a
    function's body receives a draw; a value of another kind is passed as it is.
    """
    if isinstance(spec, RecordSpec):
        return value if isinstance(value, Record) else Record(slot, value)
    if isinstance(spec, NumericArraySpec):
        return jnp.asarray(value)
    return value


def _declared_law(law: Any, event_spec: OutputSpec, name: str) -> Distribution:
    """*law*, the function's result, checked against the kernel's event declaration.

    Raises
    ------
    TypeError
        If *law* is not a ``Distribution``.
    ValueError
        If its event declaration does not agree with *event_spec*.
    """
    if not isinstance(law, Distribution):
        raise TypeError(f"the function of {name!r} returned a {type(law).__name__}, not a law")
    if not DistributionSpec(event_spec).is_valid(law):
        raise ValueError(
            f"the function of {name!r} returned a law that declares {law.event_spec!r}, and the "
            f"kernel declares {event_spec!r}"
        )
    return law


class _FunctionKernel(ConditionalDistribution):
    """The kernel of a function of its given values: its law at a value is the one the function returns.

    :func:`conditional_distribution` builds it. Binding every given slot calls
    the function with each value as the argument of that name, at the kind its
    slot declares, and returns the law the call returns; binding some slots
    returns the kernel over the others. The kernel claims the conditional
    twin of sampling, of the normalized density, and of the unnormalized
    density exactly when the law its function returned at construction claims
    the capability, and each twin's guard reports what that law's guard did.

    Parameters
    ----------
    name : str
        The kernel's label.
    fn : callable
        The function, called with every given slot's value by name.
    given_spec : InputSpec
        The given slots, one per parameter of *fn*.
    event_spec : OutputSpec
        The declaration of the law *fn* returns.
    guards : Mapping[str, Feasibility]
        The report of the law's guard of each capability it claims, keyed by
        the capability's method.
    """

    _capability_table: ClassVar = _FUNCTION_KERNEL_CAPABILITIES

    def __new__(
        cls,
        name: str,
        fn: Callable[..., Distribution],
        given_spec: InputSpec,
        event_spec: OutputSpec,
        guards: Mapping[str, Feasibility],
    ) -> _FunctionKernel:
        claimed = [twin for _, twin, method in _DERIVED_TWINS if method in guards]
        return object.__new__(_capability_subclass(_FunctionKernel, claimed))

    def __init__(
        self,
        name: str,
        fn: Callable[..., Distribution],
        given_spec: InputSpec,
        event_spec: OutputSpec,
        guards: Mapping[str, Feasibility],
    ) -> None:
        super().__init__(name, given_spec, event_spec)
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_slots", given_spec)
        object.__setattr__(self, "_bound", {})
        object.__setattr__(self, "_guards", dict(guards))

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **options: Any
    ) -> Distribution | ConditionalDistribution:
        """The law the function returns at a value of every given slot, or the curried kernel.

        Raises
        ------
        TypeError
            If an option is passed, since evaluating the function reads none.
        KeyError
            If *given* names a key that is not a given slot.
        ValueError
            If a value does not conform to its slot, or the returned law departs
            from the kernel's event declaration.
        """
        if options:
            raise TypeError(
                f"the kernel {self.label!r} evaluates its function exactly and reads no options; "
                f"got {sorted(options)}"
            )
        values = self._given_values(given)
        if set(values) == set(self.given_spec):
            return self._law(values)
        self._check_conformance(values)
        curried = self._shallow_copy()
        object.__setattr__(curried, "_provenance", None)
        object.__setattr__(curried, "_bound", {**self._bound, **self._arguments(values)})
        left = InputSpec(
            {slot: spec for slot, spec in self.given_spec.items() if slot not in values}
        )
        object.__setattr__(curried, "_spec", ConditionalDistributionSpec(left, self.event_spec))
        return curried

    def _given_values(self, given: Any) -> dict[str, Any]:
        """The values *given* holds, by given slot.

        Raises
        ------
        KeyError
            If *given* names a key that is not a given slot.
        """
        values = dict((given.children if isinstance(given, Record) else given).items())
        unknown = sorted(set(values) - set(self.given_spec))
        if unknown:
            raise KeyError(
                f"{unknown} are not given slots of {self.label!r}, whose slots are "
                f"{list(self.given_spec)}"
            )
        return values

    def _check_conformance(self, values: Mapping[str, Any]) -> None:
        """Raise ``ValueError`` if a value does not conform to its slot."""
        InputSpec({slot: self.given_spec[slot] for slot in values}).bind_dims_from_value(
            dict(values)
        )

    def _arguments(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """*values* at the kinds their slots declare, as the function receives them."""
        return {slot: _argument(self._slots[slot], slot, value) for slot, value in values.items()}

    def _law(self, given: Any) -> Distribution:
        """The law the function returns at *given*, a value of every given slot.

        Raises
        ------
        KeyError
            If a given slot has no value, or *given* names another key.
        ValueError
            If a value does not conform to its slot, or the law departs from the
            kernel's event declaration.
        """
        values = self._given_values(given)
        missing = [slot for slot in self.given_spec if slot not in values]
        if missing:
            raise KeyError(f"{self.label!r} needs a value of every given slot; {missing} have none")
        self._check_conformance(values)
        law = self._fn(**self._bound, **self._arguments(values))
        return _declared_law(law, self.event_spec, self.label)


@dataclass(frozen=True)
class _Probe:
    """What the law a function returns at stand-ins of its slots declares and claims.

    Attributes
    ----------
    event_spec : OutputSpec
        The law's event declaration, a support that depends on the given values
        left undeclared.
    guards : Mapping[str, Feasibility]
        The report of the law's guard of each derived capability it claims,
        keyed by the capability's method.
    """

    event_spec: OutputSpec
    guards: Mapping[str, Feasibility]


def _stand_in(spec: TermSpec, path: str, name: str) -> Any:
    """The abstract stand-in of a value of *spec*: an array's shape and dtype, or a mapping of them.

    Raises
    ------
    TypeError
        If *spec* is neither an array nor a record of them, or declares free
        dimensions.
    """
    if isinstance(spec, NumericArraySpec):
        if spec.free_dims:
            raise TypeError(
                f"the given slot {path!r} of {name!r} declares the free dimensions "
                f"{sorted(spec.free_dims)}; the kernel reads its law at a stand-in of each slot, "
                f"which needs a concrete shape"
            )
        dtype = spec.dtype if spec.dtype is not None else jnp.result_type(float)
        return jax.ShapeDtypeStruct(tuple(spec.shape), dtype)
    if isinstance(spec, RecordSpec):
        return {
            key: _stand_in(child, f"{path}/{key}", name) for key, child in spec.children.items()
        }
    raise TypeError(
        f"the given slot {path!r} of {name!r} declares a {type(spec).__name__}; the kernel reads "
        f"its law at a stand-in of each slot, which needs arrays or records of arrays"
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


def _probe(name: str, fn: Callable[..., Distribution], slots: InputSpec) -> _Probe:
    """The declaration and the guards of the law *fn* returns, evaluated abstractly.

    *fn* runs once under ``jax.eval_shape``, with a traced stand-in of each
    slot's type at the kind the slot declares, so no value is assumed; a
    support that a given value sets holds a traced bound, and is left
    undeclared.

    Raises
    ------
    TypeError
        If a slot has no stand-in, or *fn* returns something other than a law.
    """
    stand_ins = {slot: _stand_in(spec, slot, name) for slot, spec in slots.items()}
    found: dict[str, Any] = {}

    def evaluate(values: Mapping[str, Any]) -> Any:
        law = fn(**{slot: _argument(slots[slot], slot, value) for slot, value in values.items()})
        if not isinstance(law, Distribution):
            raise TypeError(f"the function of {name!r} returned a {type(law).__name__}, not a law")
        found["event_spec"] = law.event_spec
        found["guards"] = {
            method: _capability_guard(law, method)
            for capability, _, method in _DERIVED_TWINS
            if isinstance(law, capability)
        }
        return jnp.zeros(())

    jax.eval_shape(evaluate, stand_ins)
    declaration = found["event_spec"]
    return _Probe(declaration._with_spec(_value_free(declaration.spec)), found["guards"])


def _slots_of(
    name: str, fn: Callable[..., Any], given_spec: InputSpec | Mapping[str, TermSpec] | None
) -> InputSpec:
    """The given slots of the kernel of *fn*: one per parameter, declared by *given_spec* or its annotation.

    Raises
    ------
    TypeError
        If *given_spec* names a key that is not a parameter, a parameter is
        variadic or positional-only, or a parameter is declared by neither.
    """
    declared = dict((given_spec or {}).items())
    try:
        signature = inspect.signature(fn, eval_str=True)
    except NameError:
        signature = inspect.signature(fn)
    unknown = sorted(set(declared) - set(signature.parameters))
    if unknown:
        raise TypeError(f"given_spec names {unknown}, which are not parameters of {name!r}")
    slots: dict[str, TermSpec] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise TypeError(
                f"the parameter {parameter.name!r} of {name!r} is variadic; each given slot is "
                f"a named parameter"
            )
        if parameter.kind is parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"the parameter {parameter.name!r} of {name!r} is positional-only; each given "
                f"slot's value is passed by name"
            )
        spec = declared.get(parameter.name, parameter.annotation)
        if not isinstance(spec, TermSpec):
            raise TypeError(
                f"the given slot {parameter.name!r} of {name!r} declares no term spec; annotate "
                f"the parameter with one, or name it in given_spec"
            )
        slots[parameter.name] = spec
    return InputSpec(slots)


def _agreed_event_spec(name: str, declared: OutputSpec | TermSpec, law: OutputSpec) -> OutputSpec:
    """*declared*, completed from *law*, the declaration of the law the function returns.

    Raises
    ------
    ValueError
        If *declared* names other components or another packaging than *law*,
        or its type does not unify with the law's.
    """
    declaration = (
        declared
        if isinstance(declared, OutputSpec)
        else OutputSpec.default(declared, component=name)
    )
    if declaration.exposes_record != law.exposes_record or tuple(declaration.components) != tuple(
        law.components
    ):
        raise ValueError(
            f"the event_spec of {name!r} declares the components {list(declaration.components)}, "
            f"and the law its function returns declares {list(law.components)}"
        )
    return declaration.with_spec(law.spec)


def _function_kernel(
    name: str | None,
    fn: Any,
    given_spec: InputSpec | Mapping[str, TermSpec] | None,
    event_spec: OutputSpec | TermSpec | None,
) -> ConditionalDistribution:
    """The kernel of *fn*, labeled *name* or after the function.

    Raises
    ------
    TypeError
        If *fn* is not callable or there is no label, or as :func:`_slots_of`
        and :func:`_probe` raise.
    ValueError
        As :func:`_agreed_event_spec` raises.
    """
    if not callable(fn):
        raise TypeError(f"conditional_distribution takes a function, got {type(fn).__name__}")
    label = getattr(fn, "__name__", None) if name is None else name
    if not isinstance(label, str) or not label:
        raise TypeError(
            f"conditional_distribution needs a name for a {type(fn).__name__}, which has no "
            f"__name__ to take it from"
        )
    slots = _slots_of(label, fn, given_spec)
    probe = _probe(label, fn, slots)
    declaration = (
        probe.event_spec
        if event_spec is None
        else _agreed_event_spec(label, event_spec, probe.event_spec)
    )
    return _FunctionKernel(label, fn, slots, declaration, probe.guards)


def conditional_distribution(
    name: str | Callable[..., Distribution] | None = None,
    fn: Callable[..., Distribution] | None = None,
    /,
    *,
    given_spec: InputSpec | Mapping[str, TermSpec] | None = None,
    event_spec: OutputSpec | TermSpec | None = None,
) -> Any:
    """Build a ``ConditionalDistribution`` from a function of its given values that returns a law.

    Each parameter of the function is a given slot, declared by its entry in
    *given_spec* or else by its annotation, which is then a term spec.
    Construction evaluates the function once, abstractly, at stand-ins of the
    slots' types, and reads the law it returns: its event declaration is the
    kernel's, unless *event_spec* declares one that agrees with it, and the
    kernel claims conditional sampling and the conditional density exactly
    when that law claims sampling and a density, each with the law's guard.
    Binding every given slot calls the function with each value as the
    argument of that name, at the kind its slot declares, and returns the law
    the call returns; binding some slots curries the kernel over the rest.

    The call form takes the name first and the function second::

        likelihood = conditional_distribution(
            "y", lambda mu, tau: Normal("y", mu, tau), given_spec={"mu": real, "tau": scale}
        )

    and the decorator form names the kernel after the function, or as given::

        @conditional_distribution(given_spec=prior.event_spec.components)
        def y(population, groups):
            return Normal("y", population["mu"] + population["tau"] * groups["theta"], sigma)

    Parameters
    ----------
    name : str or callable, optional
        The kernel's label; a function passed alone is the function, labeled
        after its ``__name__``.
    fn : callable, optional
        The function of the given values that returns a law; without it, the
        result is a decorator.
    given_spec : InputSpec or Mapping[str, TermSpec], optional
        The specs of some or all given slots, keyed by parameter.
    event_spec : OutputSpec or TermSpec, optional
        The event declaration, which must name the components of the returned
        law and unify with its type.

    Returns
    -------
    ConditionalDistribution or callable
        The kernel, or a decorator that builds it from a function.

    Raises
    ------
    TypeError
        If a parameter declares no term spec, is variadic or positional-only,
        or *given_spec* names a key that is not a parameter; if a slot is
        neither an array nor a record of arrays, or declares free dimensions;
        or if the function returns something other than a law.
    ValueError
        If *event_spec* departs from the declaration of the returned law.
    """
    if callable(name) and fn is None:
        return _function_kernel(None, name, given_spec, event_spec)
    if name is not None and not isinstance(name, str):
        raise TypeError(f"conditional_distribution takes a name first, got {type(name).__name__}")
    if fn is not None:
        return _function_kernel(name, fn, given_spec, event_spec)

    def decorate(function: Callable[..., Distribution]) -> ConditionalDistribution:
        return _function_kernel(name, function, given_spec, event_spec)

    return decorate
