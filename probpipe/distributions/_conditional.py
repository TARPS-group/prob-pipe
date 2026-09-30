"""The conditional distribution: a probability kernel, its term spec, and its markers.

Provides:
  - ``ConditionalDistribution`` – a kernel ``K : S → P(T)``, which a value for
    its given slots turns into a ``Distribution``.
  - ``ConditionalDistributionSpec`` – the term spec of the conditional kind.
  - ``ConditionalNumericDistribution``, ``NumericConditionalDistribution``, and
    ``FullyNumericConditionalDistribution`` – the markers of a kernel whose
    event side, given side, or both are numeric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericSpec, TermSpec, _unify_specs
from ..core._specs import InputSpec, OutputSpec
from ..core.provenance import Provenance
from ..core.tracked import Annotated, TrackedTerm, _TrackedTermMeta
from ._capabilities import _check_guards
from ._distribution import (
    _DECLARATION_MARKERS,
    _check_marker_claims,
    _complete_event_spec,
    _compose_operands,
    _unify_declarations,
)

if TYPE_CHECKING:
    from ..core.record import Record
    from ._distribution import Distribution
    from ._factored import FactoredConditionalDistribution

__all__ = [
    "ConditionalDistribution",
    "ConditionalDistributionSpec",
    "ConditionalNumericDistribution",
    "FullyNumericConditionalDistribution",
    "NumericConditionalDistribution",
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
                f"{type(self).__name__} requires a non-empty name as its first argument"
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
                _complete_given_spec(given_spec), _complete_event_spec(event_spec, self._name)
            ),
        )

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
                f"{type(self).__name__} {self.name!r} has no free dimensions "
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

    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self:
        """Rename or move given slots and event paths, ``old -> new``.

        The event side behaves as :meth:`Distribution.with_path_names`. On the
        given side a path-valued target may split or group slots, since a kernel
        has no signature that fixes its top level. The kernel is unchanged: the
        result conditions on the renamed slots and produces the renamed draws.

        Raises
        ------
        KeyError
            If a key is neither a given slot nor an event path.
        ValueError
            If a new name is empty or contains ``/`` where a slot is meant, a
            node is renamed twice, no renames are given, or the renamed sides
            share a name.
        NotImplementedError
            Always, until renaming reaches the subclass's primitive.
        """
        raise NotImplementedError("ConditionalDistribution.with_path_names")

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
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The law ``K(given, ·)``, or a curried kernel for a partial given.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values for some or all of the given slots, by slot name.
        **kwargs : Any
            Further given values, by slot name.

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
        return f"{type(self).__name__}(name={self.name!r})"


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
