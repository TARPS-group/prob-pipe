"""Factored distributions: joints built from an ordered list of factors.

Provides:
  - ``SupportsFactors`` – the capability of carrying an explicit factorization.
  - ``FactoredDistribution`` – a joint ``Distribution`` of its factors.
  - ``FactoredConditionalDistribution`` – a joint ``ConditionalDistribution``,
    whose factors leave some givens unmet.
  - the numeric markers of the two factored kinds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from ..core._dispatch import Feasibility
from ..core._record_spec import RecordSpec
from ..core._spec_base import TermSpec, _unify_specs
from ..core._specs import InputSpec, OutputSpec
from ._capabilities import (
    _CONDITIONAL_TWINS,
    SupportsCovariance,
    SupportsLogProb,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
    _capability_guard,
    _capability_subclass,
)
from ._conditional import (
    ConditionalDistribution,
    _event_is_numeric,
    _given_side_is_numeric,
)
from ._distribution import _DECLARATION_MARKERS, Distribution, _declares_numeric_event

if TYPE_CHECKING:
    from ..core.record import Record

__all__ = [
    "FactoredConditionalDistribution",
    "FactoredConditionalNumericDistribution",
    "FactoredDistribution",
    "FactoredFullyNumericConditionalDistribution",
    "FactoredNumericConditionalDistribution",
    "FactoredNumericDistribution",
    "SupportsFactors",
]

_PATH_SEP = "/"

type Factor = Distribution | ConditionalDistribution


@runtime_checkable
class SupportsFactors(Protocol):
    """A distribution built from an ordered list of sub-distributions, its factors.

    Each factor is a ``Distribution`` or a ``ConditionalDistribution``. The
    dependence graph is derived by matching each factor's given slots against
    the components the factors to its right produce, so it is not stored.
    """

    @property
    def factors(self) -> tuple[Distribution | ConditionalDistribution, ...]: ...


# ---------------------------------------------------------------------------
# The factor graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FactorGraph:
    """The validated structure of an ordered list of factors.

    Attributes
    ----------
    factors : tuple
        The factors, with dimensions bound in the joint's one scope.
    producers : Mapping[str, int]
        The index of the factor producing each component.
    edges : tuple of (int, int, str)
        Each dependency as the consuming factor's index, the producing factor's
        index, and the matched name.
    unmet : InputSpec or None
        The givens no factor produces, or ``None`` when every given is met.
    event_spec : OutputSpec
        The exposed record of every factor's components, in factor order.
    """

    factors: tuple[Factor, ...]
    producers: Mapping[str, int]
    edges: tuple[tuple[int, int, str], ...]
    unmet: InputSpec | None
    event_spec: OutputSpec

    def parents(self, index: int) -> set[int]:
        """The indices of the factors that factor *index* conditions on."""
        return {producer for consumer, producer, _ in self.edges if consumer == index}

    def ancestors(self, indices: set[int]) -> set[int]:
        """*indices* with every factor they condition on, transitively."""
        closure = set(indices)
        frontier = list(indices)
        while frontier:
            for parent in self.parents(frontier.pop()):
                if parent not in closure:
                    closure.add(parent)
                    frontier.append(parent)
        return closure


def _given_of(factor: Factor) -> InputSpec | None:
    return factor.given_spec if isinstance(factor, ConditionalDistribution) else None


def _free_dims(factor: Factor) -> frozenset[str]:
    if isinstance(factor, ConditionalDistribution):
        return factor.spec.free_dims
    return factor.event_spec.spec.free_dims


def _factor_graph(factors: Sequence[Factor]) -> _FactorGraph:
    """Validate the ordered *factors* by the composition rules and derive their graph.

    The order is conditional-first: a factor may condition on a component that
    a factor to its right produces, and never on one a factor to its left
    produces. Every component is produced once. A matched component's spec must
    unify with the consuming slot's, and same-named unmet givens unify into one
    slot, all in one dimension scope whose bindings are applied to the factors.

    Raises
    ------
    TypeError
        If a factor is neither distribution kind.
    ValueError
        If there are no factors, a component is produced twice, a factor
        consumes a component a factor to its left produces, or matched specs do
        not unify.
    """
    factors = tuple(factors)
    if not factors:
        raise ValueError("a factored distribution has at least one factor")
    producers: dict[str, int] = {}
    component_specs: dict[str, TermSpec] = {}
    for index, factor in enumerate(factors):
        if not isinstance(factor, (Distribution, ConditionalDistribution)):
            raise TypeError(
                f"a factor is a Distribution or a ConditionalDistribution, got "
                f"{type(factor).__name__}"
            )
        for component, spec in factor.event_spec.components.items():
            if component in producers:
                raise ValueError(
                    f"the component {component!r} is produced by both "
                    f"{factors[producers[component]].name!r} and {factor.name!r}; each "
                    f"component is produced once, so rename one with with_path_names"
                )
            producers[component] = index
            component_specs[component] = spec
    bindings: dict[str, int] = {}
    edges: list[tuple[int, int, str]] = []
    unmet: dict[str, TermSpec] = {}
    for index, factor in enumerate(factors):
        given = _given_of(factor)
        if given is None:
            continue
        for slot, slot_spec in given.items():
            producer = producers.get(slot)
            if producer is None:
                if slot in unmet:
                    _unify_specs(unmet[slot], slot_spec, bindings, f"the given {slot!r}")
                else:
                    unmet[slot] = slot_spec
                continue
            if producer < index:
                raise ValueError(
                    f"{factor.name!r} conditions on {slot!r}, which {factors[producer].name!r} "
                    f"produces to its left; composition is conditional-first, so put the "
                    f"producer on the right"
                )
            _unify_specs(slot_spec, component_specs[slot], bindings, f"the given {slot!r}")
            edges.append((index, producer, slot))
    if bindings:
        factors = tuple(
            factor.with_dim_sizes(**subset)
            if (subset := {d: bindings[d] for d in _free_dims(factor) if d in bindings})
            else factor
            for factor in factors
        )
        unmet = {slot: spec._substitute_dims(bindings) for slot, spec in unmet.items()}
    event_spec = OutputSpec(
        RecordSpec(
            {
                component: spec
                for factor in factors
                for component, spec in factor.event_spec.components.items()
            }
        )
    )
    return _FactorGraph(
        factors=factors,
        producers=producers,
        edges=tuple(edges),
        unmet=InputSpec(unmet) if unmet else None,
        event_spec=event_spec,
    )


# ---------------------------------------------------------------------------
# The derived capabilities
# ---------------------------------------------------------------------------


def _all_claim(factors: Sequence[Factor], protocol: type) -> bool:
    """Whether every factor claims *protocol*, a conditional factor through its twin."""
    twin = _CONDITIONAL_TWINS[protocol]
    return all(
        isinstance(factor, twin if isinstance(factor, ConditionalDistribution) else protocol)
        for factor in factors
    )


def _stub(name: str) -> Callable[..., Any]:
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(name)

    method.__name__ = name.rsplit(".", 1)[-1]
    method.__qualname__ = name
    return method


def _marginal_guard(self: Any, path: str | tuple[str, ...]) -> Feasibility:
    """Whether the marginal at *path* is exact, by the factor graph.

    Exact when the target is ancestrally closed, so the sub-joint of whole
    factors is the marginal, or when the target's ancestor closure is a single
    factor whose own marginal is exact at the path within it.
    """
    graph: _FactorGraph = self._graph
    paths = (path,) if isinstance(path, str) else tuple(path)
    heads = {p.split(_PATH_SEP, 1)[0] for p in paths}
    unknown = heads - set(graph.producers)
    if unknown:
        return Feasibility(False, f"no component {sorted(unknown)} in the joint")
    targets = {graph.producers[head] for head in heads}
    closure = graph.ancestors(targets)
    wanted = set(paths)
    whole = {
        component for component, index in graph.producers.items() if index in closure
    } == wanted
    if whole:
        return Feasibility(True)
    if len(closure) == 1:
        (index,) = closure
        factor = graph.factors[index]
        if isinstance(factor, SupportsMarginals):
            return _capability_guard(factor, "_marginal", path)
        return Feasibility(False, f"the factor {factor.name!r} has no marginals")
    return Feasibility(
        False,
        "the marginal integrates out ancestor fields of more than one factor, which has no "
        "closed form here",
    )


#: The method each unconditional capability of a joint names.
_JOINT_METHODS: dict[type, str] = {
    SupportsSampling: "_sample",
    SupportsLogProb: "_log_prob",
    SupportsUnnormalizedLogProb: "_unnormalized_log_prob",
    SupportsMean: "_mean",
    SupportsVariance: "_variance",
    SupportsCovariance: "_cov",
    SupportsQuantile: "_quantile",
}


def _joint_table(owner: str, *, conditional: bool) -> dict[type, Mapping[str, Callable[..., Any]]]:
    """Each capability a joint of the *owner* kind may claim, with its methods.

    A conditional joint claims each capability's conditional twin, whose method
    is ``_conditional`` followed by the unconditional method's name.
    """
    table: dict[type, Mapping[str, Callable[..., Any]]] = {}
    for protocol, method in _JOINT_METHODS.items():
        if conditional:
            protocol, method = _CONDITIONAL_TWINS[protocol], f"_conditional{method}"
        table[protocol] = {method: _stub(f"{owner}.{method}")}
    if not conditional:
        table[SupportsMarginals] = {
            "_marginal": _stub(f"{owner}._marginal"),
            "_marginal_guard": _marginal_guard,
        }
    return table


def _joint_protocols(graph: _FactorGraph, *, conditional: bool) -> set[type]:
    """The capabilities a joint of *graph* claims, decided at construction.

    Sampling and the densities are the intersection of the factors'. The moments
    are claimed by an edge-free joint whose every factor has them; a dependent
    joint claims no moment, since factor-wise conditional moments do not compose.
    The unconditional joint claims marginals, guarded per path.
    """
    factors = graph.factors
    claimed: set[type] = set()
    if _all_claim(factors, SupportsSampling):
        claimed.add(SupportsSampling)
    if _all_claim(factors, SupportsLogProb):
        claimed.add(SupportsLogProb)
    elif _all_claim(factors, SupportsUnnormalizedLogProb):
        claimed.add(SupportsUnnormalizedLogProb)
    if not graph.edges:
        claimed.update(
            moment
            for moment in (SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile)
            if _all_claim(factors, moment)
        )
    if conditional:
        return {_CONDITIONAL_TWINS[protocol] for protocol in claimed}
    return claimed | {SupportsMarginals}


def _each_factor(
    factors: Sequence[Factor], method: str, mapping: Mapping[str, Any]
) -> list[Factor]:
    """Each factor with *method* applied to the entries of *mapping* it declares."""
    result: list[Factor] = []
    for factor in factors:
        subset = {dim: value for dim, value in mapping.items() if dim in _free_dims(factor)}
        result.append(getattr(factor, method)(**subset) if subset else factor)
    return result


# ---------------------------------------------------------------------------
# The factored kinds
# ---------------------------------------------------------------------------


class FactoredDistribution(Distribution, SupportsFactors):
    """A joint distribution built from an ordered list of factors, every given met.

    The factors are ``Distribution``s and ``ConditionalDistribution``s in
    conditional-first order: a factor may condition on components that factors
    to its right produce. The joint's event declaration is an exposed record of
    every factor's components in factor order, each factor's own order kept, and
    extraction and reconstruction keep each factor's packaging. Each component
    is produced by exactly one factor, while factor labels may repeat.

    Sampling and the log-density capabilities are the intersection of the
    factors'. The moment capabilities are decided at construction: an edge-free
    joint has a moment exactly when every factor does, and a dependent joint has
    none. The marginal is resolved per path, and its guard reports whether the
    marginal at a path is exact.

    Parameters
    ----------
    name : str
        The joint's label.
    factors : Sequence[Distribution | ConditionalDistribution]
        The factors, in conditional-first order.

    Raises
    ------
    TypeError
        If a factor is neither distribution kind.
    ValueError
        If there are no factors, a component is produced twice, a factor
        conditions on a component a factor to its left produces, matched specs
        do not unify, or a given is left unmet, which makes the joint a
        ``FactoredConditionalDistribution``.
    """

    _capability_table = _joint_table("FactoredDistribution", conditional=False)

    def __new__(cls, name: str, factors: Sequence[Factor]) -> FactoredDistribution:
        protocols = _joint_protocols(_factor_graph(factors), conditional=False)
        return object.__new__(_capability_subclass(FactoredDistribution, protocols))

    def __init__(self, name: str, factors: Sequence[Factor]) -> None:
        graph = _factor_graph(factors)
        if graph.unmet is not None:
            raise ValueError(
                f"the factors of {name!r} leave the givens {sorted(graph.unmet)} unmet, so "
                f"the joint is a FactoredConditionalDistribution"
            )
        super().__init__(name, graph.event_spec)
        object.__setattr__(self, "_graph", graph)

    @property
    def factors(self) -> tuple[Factor, ...]:
        """The factors, in conditional-first order."""
        return self._graph.factors

    def with_dim_sizes(self, **sizes: int) -> Self:
        """Bind named symbolic dimensions in every factor that declares them.

        Raises
        ------
        ValueError
            If a name is not a free dimension of the joint.
        """
        unbound = set(sizes) - self.event_spec.spec.free_dims
        if unbound:
            raise ValueError(
                f"{type(self).__name__} {self.name!r} has no free dimensions "
                f"{sorted(unbound)} to bind"
            )
        return FactoredDistribution(self.name, _each_factor(self.factors, "with_dim_sizes", sizes))

    def with_dim_names(self, **names: str) -> Self:
        """Rename symbolic dimensions in every factor, simultaneously."""
        return FactoredDistribution(self.name, _each_factor(self.factors, "with_dim_names", names))


class FactoredConditionalDistribution(ConditionalDistribution, SupportsFactors):
    """A joint conditional distribution: factors that leave some givens unmet.

    Its ``given_spec`` is exactly the unmet givens, same-named ones unified into
    one slot, and its event declaration is read as for
    :class:`FactoredDistribution`. Conditioning on every given yields a
    ``FactoredDistribution``, and conditioning on some curries to a smaller
    ``FactoredConditionalDistribution``.

    Parameters
    ----------
    name : str
        The joint's label.
    factors : Sequence[Distribution | ConditionalDistribution]
        The factors, in conditional-first order.

    Raises
    ------
    TypeError
        If a factor is neither distribution kind.
    ValueError
        If the factors break a composition rule, as for
        :class:`FactoredDistribution`, or meet every given, which makes the
        joint a ``FactoredDistribution``.
    """

    _capability_table = _joint_table("FactoredConditionalDistribution", conditional=True)

    def __new__(cls, name: str, factors: Sequence[Factor]) -> FactoredConditionalDistribution:
        protocols = _joint_protocols(_factor_graph(factors), conditional=True)
        return object.__new__(_capability_subclass(FactoredConditionalDistribution, protocols))

    def __init__(self, name: str, factors: Sequence[Factor]) -> None:
        graph = _factor_graph(factors)
        if graph.unmet is None:
            raise ValueError(
                f"the factors of {name!r} meet every given, so the joint is a FactoredDistribution"
            )
        super().__init__(name, graph.unmet, graph.event_spec)
        object.__setattr__(self, "_graph", graph)

    @property
    def factors(self) -> tuple[Factor, ...]:
        """The factors, in conditional-first order."""
        return self._graph.factors

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """Bind given slots in every factor that names them, and rebuild the joint.

        Returns
        -------
        FactoredDistribution or FactoredConditionalDistribution
            The joint over the bound factors, conditional while a given remains.

        Raises
        ------
        KeyError
            If a bound name is not a given slot of the joint.
        """
        values = {**dict(given.items()), **kwargs}
        unknown = set(values) - set(self.given_spec)
        if unknown:
            raise KeyError(f"{sorted(unknown)} are not given slots of {self.name!r}")
        factors: list[Factor] = []
        for factor in self.factors:
            if isinstance(factor, ConditionalDistribution):
                bound = {slot: value for slot, value in values.items() if slot in factor.given_spec}
                if bound:
                    factor = factor._condition_on(bound)
            factors.append(factor)
        if set(values) == set(self.given_spec):
            return FactoredDistribution(self.name, factors)
        return FactoredConditionalDistribution(self.name, factors)


# ---------------------------------------------------------------------------
# The numeric markers
# ---------------------------------------------------------------------------


class FactoredNumericDistribution(FactoredDistribution):
    """The marker of an unconditional joint whose event is numeric."""

    _membership_follows_declaration = True


class FactoredConditionalNumericDistribution(FactoredConditionalDistribution):
    """The marker of a conditional joint whose event is numeric."""

    _membership_follows_declaration = True


class FactoredNumericConditionalDistribution(FactoredConditionalDistribution):
    """The marker of a conditional joint whose given slots are numeric."""

    _membership_follows_declaration = True


class FactoredFullyNumericConditionalDistribution(
    FactoredNumericConditionalDistribution, FactoredConditionalNumericDistribution
):
    """The marker of a conditional joint whose given and event sides are both numeric."""

    _membership_follows_declaration = True


def _is_factored_conditional(value: Any) -> bool:
    return isinstance(value, FactoredConditionalDistribution)


_DECLARATION_MARKERS[FactoredNumericDistribution] = (
    lambda value: isinstance(value, FactoredDistribution) and _declares_numeric_event(value),
    "a numeric event",
)
_DECLARATION_MARKERS[FactoredConditionalNumericDistribution] = (
    lambda value: _is_factored_conditional(value) and _event_is_numeric(value),
    "a numeric event",
)
_DECLARATION_MARKERS[FactoredNumericConditionalDistribution] = (
    lambda value: _is_factored_conditional(value) and _given_side_is_numeric(value),
    "numeric given slots",
)
_DECLARATION_MARKERS[FactoredFullyNumericConditionalDistribution] = (
    lambda value: (
        _is_factored_conditional(value)
        and _event_is_numeric(value)
        and _given_side_is_numeric(value)
    ),
    "numeric given slots and a numeric event",
)
