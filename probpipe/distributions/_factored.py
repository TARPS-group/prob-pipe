"""Factored distributions: joints built from an ordered list of factors.

Provides:
  - ``SupportsFactors`` – the capability of carrying an explicit factorization.
  - ``FactoredDistribution`` – a joint ``Distribution`` of its factors.
  - ``FactoredConditionalDistribution`` – a joint ``ConditionalDistribution``,
    whose factors leave some givens unmet.
  - the numeric markers of the two factored kinds.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

from ..core._dispatch import Feasibility, ResolutionError
from ..core._object_batch import _is_object_array
from ..core._record_batch import RecordBatch
from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec, _unify_specs
from ..core._specs import InputSpec, OutputSpec
from ..core.named_tree import _unflatten_paths
from ..core.provenance import Provenance
from ..core.record import Record
from ..linalg import DenseLinOp, to_dense
from ._capabilities import (
    _CONDITIONAL_TWINS,
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
    _capability_guard,
    _capability_subclass,
    _claimed,
    _claims,
    _conjunction,
    _kernel_claims,
    _marginal_claims,
)
from ._conditional import (
    ConditionalDistribution,
    ConditionalDistributionSpec,
    _event_is_numeric,
    _given_side_is_numeric,
)
from ._distribution import (
    _DECLARATION_MARKERS,
    Distribution,
    DistributionSpec,
    _declares_numeric_event,
)

if TYPE_CHECKING:
    from ..custom_types import Array, ArrayLike, PRNGKey

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

#: The separator a joint's label places between the labels it joins.
_LABEL_SEP = "·"

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
    scope : Mapping[str, int]
        The joint's dimension scope: each dimension name bound while the factors
        were unified, so a later composition binds the same name to the same
        size, whatever the grouping.
    """

    factors: tuple[Factor, ...]
    producers: Mapping[str, int]
    edges: tuple[tuple[int, int, str], ...]
    unmet: InputSpec | None
    event_spec: OutputSpec
    scope: Mapping[str, int]

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


def _flattened(
    factors: Sequence[Factor], scope: Mapping[str, int] | None
) -> tuple[tuple[Factor, ...], dict[str, int]]:
    """*factors* with each factored factor replaced by its own, and the merged scope.

    Raises
    ------
    ValueError
        If two scopes bind one dimension name to different sizes.
    """
    bindings = dict(scope or {})
    flat: list[Factor] = []
    for factor in factors:
        graph = getattr(factor, "_graph", None)
        if isinstance(factor, SupportsFactors) and isinstance(graph, _FactorGraph):
            flat.extend(graph.factors)
            for name, size in graph.scope.items():
                if bindings.setdefault(name, size) != size:
                    raise ValueError(
                        f"the dimension {name!r} is bound to {bindings[name]} and to {size}; "
                        f"rename one apart with with_dim_names"
                    )
        else:
            flat.append(factor)
    return tuple(flat), bindings


def _unify_either_way(
    first: TermSpec, second: TermSpec, bindings: dict[str, int], path: str
) -> None:
    """Unify two specs that meet as equals, whichever direction admits the other.

    Raises
    ------
    ValueError
        If neither direction unifies, with the error of the first.
    """
    trial = dict(bindings)
    try:
        _unify_specs(first, second, trial, path)
    except ValueError as forward:
        trial = dict(bindings)
        try:
            _unify_specs(second, first, trial, path)
        except ValueError:
            raise forward from None
    bindings.update(trial)


def _factor_graph(
    factors: Sequence[Factor], scope: Mapping[str, int] | None = None
) -> _FactorGraph:
    """Validate the ordered *factors* by the composition rules and derive their graph.

    A factored factor enters as its own factors, with its scope. The order is
    conditional-first: a factor may condition on a component that a factor to
    its right produces, and never on one a factor to its left produces. Every
    component is produced once. A matched component's spec must unify with the
    consuming slot's, and same-named unmet givens unify into one slot, all in one
    dimension scope, starting from *scope*, whose bindings are applied to the
    factors and recorded with the graph.

    Raises
    ------
    TypeError
        If a factor is neither distribution kind.
    ValueError
        If there are no factors, a component is produced twice, a factor
        consumes a component a factor to its left produces, matched specs do
        not unify, or two scopes bind one dimension to different sizes.
    """
    factors, bindings = _flattened(tuple(factors), scope)
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
                    _unify_either_way(unmet[slot], slot_spec, bindings, f"the given {slot!r}")
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
        scope=bindings,
    )


# ---------------------------------------------------------------------------
# The raw forms of a joint
# ---------------------------------------------------------------------------
#
# A factor contributes its components to a joint's value by its own packaging:
# a whole term's component is the factor's value itself, and an exposed record's
# components are the value's immediate children. Scoring reverses the
# extraction, so every factor receives a value of the kind it declares.


def _raw_record(value: Any) -> Any:
    """*value* in the raw form of a record-valued result: the nested mapping of its raw leaves.

    A ``Record`` gives the nested mapping of its leaves, and a batch of records
    the nested mapping of its columns with the batch axes leading. A mapping is
    converted value by value, and any other value is returned as it is.
    """
    if isinstance(value, RecordBatch):
        return _unflatten_paths(value._raw_columns())
    if isinstance(value, Record):
        return value.to_nested_dict()
    if isinstance(value, Mapping):
        return {key: _raw_record(entry) for key, entry in value.items()}
    return value


def _children(value: Any) -> Mapping[str, Any]:
    """The immediate children of *value*, the raw form of a record, keyed by name.

    A raw record is a mapping of its fields. A ``Record`` is read through its
    one-level view, and a batch of records through its raw columns.

    Raises
    ------
    TypeError
        If *value* is none of these.
    """
    if isinstance(value, Mapping):
        return value
    if isinstance(value, RecordBatch):
        return _raw_record(value)
    children = getattr(value, "children", None)
    if isinstance(children, Mapping):
        return children
    raise TypeError(f"a record value is a mapping of its fields, got {type(value).__name__}")


def _components_of(declaration: OutputSpec, value: Any) -> dict[str, Any]:
    """Each component *declaration* declares, extracted from *value*.

    *value* is a draw of the declared event, or an event-typed result such as a
    mean, and the leading axes of a batch stay on each component.
    """
    if not declaration.exposes_record:
        (component,) = declaration.components
        return {component: value}
    children = _children(value)
    return {component: children[component] for component in declaration.components}


def _event_of(declaration: OutputSpec, components: Mapping[str, Any]) -> Any:
    """The value *declaration* declares, reconstructed from a joint's *components*.

    It inverts :func:`_components_of`: a whole term is its component's value,
    and an exposed record is the mapping of its components.
    """
    if not declaration.exposes_record:
        (component,) = declaration.components
        return components[component]
    return {component: components[component] for component in declaration.components}


def _given_values(joint: Any, given: Record | Mapping[str, Any]) -> dict[str, Any]:
    """The value of each given slot of the conditional *joint* in *given*, by slot name.

    Raises
    ------
    KeyError
        If *given* names a slot the joint does not have, or omits one it has,
        since a fused conditional path binds every given slot.
    """
    top = given.children if hasattr(given, "children") else given
    values = dict(top.items())
    unknown = set(values) - set(joint.given_spec)
    if unknown:
        raise KeyError(f"{sorted(unknown)} are not given slots of {joint.name!r}")
    missing = [slot for slot in joint.given_spec if slot not in values]
    if missing:
        raise KeyError(f"the given of {joint.name!r} omits the given slots {missing}")
    return values


def _leading_axes(value: Any, spec: TermSpec) -> tuple[int, ...] | None:
    """The axes of *value* before the ones *spec* declares, read at its first array leaf.

    Returns ``None`` when *spec* declares no array leaf to read them at.
    """
    if isinstance(spec, NumericArraySpec):
        shape = tuple(jnp.shape(value))
        return shape[: max(len(shape) - len(spec.shape), 0)]
    if isinstance(spec, RecordSpec):
        children = _children(value)
        for name, child in spec.children.items():
            axes = _leading_axes(children[name], child)
            if axes is not None:
                return axes
    return None


def _flatten_draws(tree: Any, batch_shape: tuple[int, ...]) -> Any:
    """*tree* with the leading *batch_shape* axes of each leaf merged into one axis.

    An object array holds a column of values that are not arrays, and it is
    merged too.
    """
    rank, count = len(batch_shape), math.prod(batch_shape)

    def merged(leaf: Any) -> Any:
        if _is_object_array(leaf):
            return leaf.reshape((count, *leaf.shape[rank:]))
        return jnp.reshape(leaf, (count, *jnp.shape(leaf)[rank:]))

    return jax.tree.map(merged, tree)


def _unflatten_draws(tree: Any, batch_shape: tuple[int, ...]) -> Any:
    """*tree* with the leading axis of each leaf split into *batch_shape*."""
    return jax.tree.map(lambda leaf: jnp.reshape(leaf, (*batch_shape, *jnp.shape(leaf)[1:])), tree)


def _stacked(values: Sequence[Any]) -> Any:
    """The pytrees in *values*, stacked leaf by leaf along a new leading axis."""
    return jax.tree.map(lambda *leaves: jnp.stack([jnp.asarray(leaf) for leaf in leaves]), *values)


def _numeric_givens(factor: ConditionalDistribution, values: Mapping[str, Any]) -> bool:
    """Whether every given slot of *factor* that *values* binds is numeric, so a call traces."""
    return all(isinstance(factor.given_spec[slot], NumericSpec) for slot in values)


def _each_value(function: Callable[..., Any], count: int, *trees: Any, traceable: bool) -> Any:
    """*function* at each of the *count* values along the leading axis of *trees*, stacked.

    A traceable call is vectorized with ``jax.vmap``. Otherwise *function* is
    called on one value at a time, since a value that is not an array cannot be
    traced, and the results are stacked leaf by leaf.
    """
    if traceable:
        return jax.vmap(function)(*trees)
    return _stacked(
        [
            function(*(jax.tree.map(lambda leaf, at=index: leaf[at], tree) for tree in trees))
            for index in range(count)
        ]
    )


def _batch_axes(
    factor: ConditionalDistribution, inner: Mapping[str, Any], event: Any
) -> tuple[int, ...]:
    """The batch axes of *factor*'s part of a joint's value, or ``()`` for one value.

    They are read at the first array leaf of the components that *inner* binds
    and, when no bound component has one, at the factor's own event. With no
    bound component the factor scores its part in one call.
    """
    for slot, value in inner.items():
        axes = _leading_axes(value, factor.given_spec[slot])
        if axes is not None:
            return axes
    if inner:
        axes = _leading_axes(event, factor.event_spec.spec)
        if axes is not None:
            return axes
    return ()


# ---------------------------------------------------------------------------
# Sampling and scoring
# ---------------------------------------------------------------------------


def _ancestral_sample(
    graph: _FactorGraph, fixed: Mapping[str, Any], key: PRNGKey, sample_shape: tuple[int, ...]
) -> dict[str, Any]:
    """The draw of the joint of *graph*, each unmet given set by *fixed*.

    Conditional-first order makes right to left an ancestral order, so every
    component a factor conditions on is drawn before the factor draws. Each
    factor's draw is read in its raw form, so the joint's draw is the nested
    mapping of raw leaves whatever form a factor returns.
    """
    keys = jax.random.split(key, len(graph.factors))
    drawn: dict[str, Any] = {}
    for index in reversed(range(len(graph.factors))):
        factor = graph.factors[index]
        if isinstance(factor, ConditionalDistribution):
            draw = _conditional_draw(factor, drawn, fixed, keys[index], sample_shape)
        else:
            draw = factor._sample(keys[index], sample_shape)
        drawn.update(_components_of(factor.event_spec, _raw_record(draw)))
    return {component: drawn[component] for component in graph.event_spec.components}


def _conditional_draw(
    factor: ConditionalDistribution,
    drawn: Mapping[str, Any],
    fixed: Mapping[str, Any],
    key: PRNGKey,
    sample_shape: tuple[int, ...],
) -> Any:
    """A draw of *factor* at the components in *drawn* it names and its unmet givens in *fixed*.

    Under a non-empty *sample_shape* the factor is mapped over the draws of
    those components, one given value per call, each call with its own key.
    """
    inner = {slot: drawn[slot] for slot in factor.given_spec if slot in drawn}

    def given(values: Mapping[str, Any]) -> dict[str, Any]:
        return {slot: values[slot] if slot in values else fixed[slot] for slot in factor.given_spec}

    if not inner or not sample_shape:
        return factor._conditional_sample(given(inner), key, sample_shape)
    count = math.prod(sample_shape)
    draws = _each_value(
        lambda values, subkey: factor._conditional_sample(given(values), subkey, ()),
        count,
        _flatten_draws(inner, sample_shape),
        jax.random.split(key, count),
        traceable=_numeric_givens(factor, inner),
    )
    return _unflatten_draws(draws, sample_shape)


def _score(graph: _FactorGraph, fixed: Mapping[str, Any], value: Any, method: str) -> Array:
    """The sum of each factor's density *method* at its part of *value*, unmet givens in *fixed*."""
    components = _children(value)
    total: Any = None
    for factor in graph.factors:
        event = _event_of(factor.event_spec, components)
        if isinstance(factor, ConditionalDistribution):
            score = _conditional_score(
                factor, f"_conditional{method}", graph, components, fixed, event
            )
        else:
            score = getattr(factor, method)(event)
        total = score if total is None else total + score
    return total


def _conditional_score(
    factor: ConditionalDistribution,
    method: str,
    graph: _FactorGraph,
    components: Mapping[str, Any],
    fixed: Mapping[str, Any],
    event: Any,
) -> Array:
    """*factor*'s density *method* at *event*, at the given the joint's components supply."""
    inner = {slot: components[slot] for slot in factor.given_spec if slot in graph.producers}

    def given(values: Mapping[str, Any]) -> dict[str, Any]:
        return {slot: values[slot] if slot in values else fixed[slot] for slot in factor.given_spec}

    density = getattr(factor, method)
    batch = _batch_axes(factor, inner, event)
    if not batch:
        return density(given(inner), event)
    scores = _each_value(
        lambda values, one: density(given(values), one),
        math.prod(batch),
        *_flatten_draws((inner, event), batch),
        traceable=_numeric_givens(factor, inner),
    )
    return _unflatten_draws(scores, batch)


# ---------------------------------------------------------------------------
# The moments of an edge-free joint
# ---------------------------------------------------------------------------


def _factor_results(
    graph: _FactorGraph, fixed: Mapping[str, Any], method: str, *arguments: Any
) -> Iterator[tuple[Factor, Any]]:
    """Each factor with the result of its *method*, a conditional factor's at its givens in *fixed*.

    An edge-free joint calls this, so every given of a conditional factor is unmet.
    """
    for factor in graph.factors:
        if isinstance(factor, ConditionalDistribution):
            given = {slot: fixed[slot] for slot in factor.given_spec}
            yield factor, getattr(factor, f"_conditional{method}")(given, *arguments)
        else:
            yield factor, getattr(factor, method)(*arguments)


def _componentwise(
    graph: _FactorGraph, fixed: Mapping[str, Any], method: str, *arguments: Any
) -> dict[str, Any]:
    """Each factor's event-typed *method* result in its raw form, assembled per component."""
    assembled: dict[str, Any] = {}
    for factor, result in _factor_results(graph, fixed, method, *arguments):
        assembled.update(_components_of(factor.event_spec, _raw_record(result)))
    return {component: assembled[component] for component in graph.event_spec.components}


def _block_diagonal(graph: _FactorGraph, fixed: Mapping[str, Any]) -> DenseLinOp:
    """The factors' covariances as the diagonal blocks of one operator, in factor order.

    Each factor's components are contiguous in canonical factor order and keep
    the factor's own order, so factor order is the order of the joint's
    flattened coordinates.
    """
    blocks = [to_dense(cov) for _, cov in _factor_results(graph, fixed, "_cov")]
    return DenseLinOp(block_diag(*blocks))


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------


def _requests(graph: _FactorGraph, paths: tuple[str, ...]) -> dict[int, tuple[str, ...]]:
    """The requested paths within each factor that produces one, keyed by index in factor order."""
    requested: dict[int, list[str]] = {}
    for path in paths:
        requested.setdefault(graph.producers[path.split(_PATH_SEP, 1)[0]], []).append(path)
    return {index: tuple(requested[index]) for index in sorted(requested)}


def _kept_whole(factor: Factor, requested: tuple[str, ...]) -> bool:
    """Whether a marginal keeps *factor* whole: every component of it is requested."""
    return set(requested) == set(factor.event_spec.components)


def _factor_request(requested: tuple[str, ...]) -> str | tuple[str, ...]:
    """The path argument of a factor's marginal: one path, or the selection of several."""
    return requested[0] if len(requested) == 1 else requested


def _sole_field_projection(method: str) -> Callable[..., Any]:
    """The capability of a :class:`_SoleField` that takes the one field of the law's *method*."""

    def projected(self: _SoleField, *arguments: Any) -> Any:
        return _children(_raw_record(getattr(self._law, method)(*arguments)))[self._component]

    projected.__name__ = method
    projected.__qualname__ = f"_SoleField.{method}"
    projected.__doc__ = f"The one field of the record law's ``{method}``, in its raw form."
    return projected


def _sole_field_density(method: str) -> Callable[..., Any]:
    """The density *method* of a :class:`_SoleField`, the law's at the record of the value."""

    def density(self: _SoleField, value: Any) -> Array:
        return getattr(self._law, method)({self._component: value})

    density.__name__ = method
    density.__qualname__ = f"_SoleField.{method}"
    density.__doc__ = f"The record law's ``{method}`` at the record whose one field is *value*."
    return density


def _sole_field_guard(method: str) -> Callable[..., Feasibility]:
    """The guard of a :class:`_SoleField`'s *method*, which is the law's guard of *method*."""

    def guard(self: _SoleField, *arguments: Any) -> Feasibility:
        return _capability_guard(self._law, method, *arguments)

    guard.__name__ = f"{method}_guard"
    guard.__qualname__ = f"_SoleField.{method}_guard"
    guard.__doc__ = f"The record law's guard of ``{method}``."
    return guard


def _sole_field_cov(self: _SoleField) -> Any:
    """The record law's covariance, since the field and the record flatten alike."""
    return self._law._cov()


def _sole_field_expectation(self: _SoleField, f: Callable[[Any], Array]) -> Array:
    """The record law's expectation of *f* at the one field of each draw, in its raw form."""
    return self._law._expectation(lambda record: f(_children(_raw_record(record))[self._component]))


def _sole_field_marginal(self: _SoleField, path: str | tuple[str, ...]) -> Distribution:
    """This law at its component, and the record law's marginal at any other path.

    The field and the record have the same event paths.
    """
    return self if path == self._component else self._law._marginal(path)


def _sole_field_marginal_guard(self: _SoleField, path: str | tuple[str, ...]) -> Feasibility:
    """Exact at the component, and the record law's guard at any other path."""
    if path == self._component:
        return Feasibility(True)
    return _capability_guard(self._law, "_marginal", path)


def _sole_field_marginal_capabilities(
    self: _SoleField, path: str | tuple[str, ...]
) -> frozenset[type]:
    """This law's claims at its component, and the record law's report at any other path."""
    if path == self._component:
        return _claims(self)
    return _marginal_claims(self._law, path)


#: Each capability a :class:`_SoleField` takes from its record law, with its methods.
_SOLE_FIELD_CAPABILITIES: dict[type, Mapping[str, Callable[..., Any]]] = {
    SupportsSampling: {
        "_sample": _sole_field_projection("_sample"),
        "_sample_guard": _sole_field_guard("_sample"),
    },
    SupportsLogProb: {
        "_log_prob": _sole_field_density("_log_prob"),
        "_log_prob_guard": _sole_field_guard("_log_prob"),
    },
    SupportsUnnormalizedLogProb: {
        "_unnormalized_log_prob": _sole_field_density("_unnormalized_log_prob"),
        "_unnormalized_log_prob_guard": _sole_field_guard("_unnormalized_log_prob"),
    },
    SupportsMean: {
        "_mean": _sole_field_projection("_mean"),
        "_mean_guard": _sole_field_guard("_mean"),
    },
    SupportsVariance: {
        "_variance": _sole_field_projection("_variance"),
        "_variance_guard": _sole_field_guard("_variance"),
    },
    SupportsCovariance: {"_cov": _sole_field_cov, "_cov_guard": _sole_field_guard("_cov")},
    SupportsQuantile: {
        "_quantile": _sole_field_projection("_quantile"),
        "_quantile_guard": _sole_field_guard("_quantile"),
    },
    SupportsExpectation: {
        "_expectation": _sole_field_expectation,
        "_expectation_guard": _sole_field_guard("_expectation"),
    },
    SupportsMarginals: {
        "_marginal": _sole_field_marginal,
        "_marginal_guard": _sole_field_marginal_guard,
        "_marginal_capabilities": _sole_field_marginal_capabilities,
    },
}


class _SoleField(Distribution):
    """The law of the one field of a one-field record law, declared as a whole term.

    The field determines the record, so the two are one law in two packagings,
    and each capability is the record law's through that correspondence. A
    joint's marginal returns it for a projection onto the one component of a
    factor that exposes a record of it, since a projection is a whole term.

    Parameters
    ----------
    law : Distribution
        A law whose event exposes a record of one field.
    """

    _capability_table = _SOLE_FIELD_CAPABILITIES

    def __new__(cls, law: Distribution) -> _SoleField:
        protocols = _claimed(law, _SOLE_FIELD_CAPABILITIES)
        return object.__new__(_capability_subclass(_SoleField, protocols))

    def __init__(self, law: Distribution) -> None:
        ((component, spec),) = law.event_spec.components.items()
        super().__init__(law.name, OutputSpec(**{component: spec}))
        object.__setattr__(self, "_law", law)
        object.__setattr__(self, "_component", component)


def _requested_paths(joint: Any, path: str | tuple[str, ...]) -> tuple[str, ...]:
    """The event paths of *joint* that *path* requests: one path, or a selection of several.

    Raises
    ------
    KeyError
        If a path is not an event path of the joint.
    TypeError
        If a path is not a string.
    ValueError
        If a selection names no path, or two of its paths share a final
        segment.
    """
    paths = (path,) if isinstance(path, str) else tuple(path)
    if not paths:
        raise ValueError("a selection of event paths names at least one path")
    record = joint._graph.event_spec.spec
    for requested in paths:
        if not isinstance(requested, str):
            raise TypeError(f"an event path is a string, got {type(requested).__name__}")
        try:
            record.at_path(*requested.split(_PATH_SEP))
        except KeyError:
            raise KeyError(
                f"{requested!r} is not an event path of {joint.name!r}, whose components are "
                f"{list(joint.event_spec.components)}"
            ) from None
    finals = [requested.rsplit(_PATH_SEP, 1)[-1] for requested in paths]
    shared = sorted({final for final in finals if finals.count(final) > 1})
    if shared:
        raise ValueError(
            f"the selected paths {list(paths)} share the final segments {shared}, which would "
            f"name two fields of the selected record alike"
        )
    return paths


def _marginal_guard(self: Any, path: str | tuple[str, ...]) -> Feasibility:
    """Whether the marginal at *path* is exact, by the factor graph.

    A path must be an event path of the joint, and the paths of a selection
    must end in distinct segments. The target's ancestor closure must add no
    factor, since integrating out a field that a kept factor conditions on has
    no closed form here. Within the target, a factor requested whole is kept
    whole, and a factor requested in part delegates to its own marginal guard,
    provided no other requested factor conditions on what that reduction
    integrates out.
    """
    graph: _FactorGraph = self._graph
    try:
        paths = _requested_paths(self, path)
    except (KeyError, TypeError, ValueError) as error:
        return Feasibility(False, str(error.args[0]) if error.args else repr(error))
    heads = {p.split(_PATH_SEP, 1)[0] for p in paths}
    targets = {graph.producers[head] for head in heads}
    ancestors = graph.ancestors(targets) - targets
    if ancestors:
        names = sorted(graph.factors[index].name for index in ancestors)
        return Feasibility(
            False, f"the marginal integrates out {names}, which the requested fields condition on"
        )
    whole = {p for p in paths if _PATH_SEP not in p}
    for consumer, producer, name in graph.edges:
        if consumer in targets and producer in targets and name not in whole:
            return Feasibility(
                False,
                f"the marginal reduces {name!r}, which {graph.factors[consumer].name!r} "
                f"conditions on",
            )
    for index, requested in _requests(graph, paths).items():
        factor = graph.factors[index]
        if _kept_whole(factor, requested):
            continue
        if not isinstance(factor, SupportsMarginals):
            return Feasibility(False, f"the factor {factor.name!r} has no marginals")
        report = _capability_guard(factor, "_marginal", _factor_request(requested))
        if report.feasible is not True:
            return report
    return Feasibility(True)


# ---------------------------------------------------------------------------
# The derived capabilities
# ---------------------------------------------------------------------------
#
# Each function below is installed under the method name it realizes, in the
# capability subclass of a joint that claims it.


def _joint_sample(self: Any, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> dict[str, Any]:
    """Draw from the joint by ancestral sampling over its factors.

    The factors draw from right to left, and factor ``i`` draws with the
    ``i``-th key of a split of *key* into one key per factor, so one key
    reproduces the draw. A conditional factor conditions on the components the
    factors to its right drew. Under a non-empty *sample_shape* such a factor
    is mapped over the draws, each with its own key from a split of the
    factor's key, so it receives one given value per call.

    Parameters
    ----------
    key : PRNGKey
        The key of the draw.
    sample_shape : tuple of int, optional
        Batch axes prepended to every component; ``()`` draws once.

    Returns
    -------
    dict
        Each component's raw value, keyed by component in canonical factor order.
    """
    return _ancestral_sample(self._graph, {}, key, tuple(sample_shape))


def _joint_log_prob(self: Any, value: Any) -> Array:
    """The normalized log-density of the joint: the sum of its factors' log-densities.

    Each factor scores its own event, reconstructed from the components of
    *value* by the factor's packaging, and a conditional factor is scored at the
    given the components it names supply. A batch of values keeps its leading
    axes, and a conditional factor is mapped over the batch, so it receives one
    given value per call.

    Parameters
    ----------
    value : Mapping[str, Any] or Record
        A value of the joint's event, or a batch of them, keyed by component.

    Returns
    -------
    Array
        The log-density, shaped as the batch axes.
    """
    return _score(self._graph, {}, value, "_log_prob")


def _joint_unnormalized_log_prob(self: Any, value: Any) -> Array:
    """The log-density of the joint up to an additive constant.

    It is the sum of the factors' unnormalized log-densities, each factor scored
    as for the normalized log-density.

    Parameters
    ----------
    value : Mapping[str, Any] or Record
        A value of the joint's event, or a batch of them, keyed by component.

    Returns
    -------
    Array
        The unnormalized log-density, shaped as the batch axes.
    """
    return _score(self._graph, {}, value, "_unnormalized_log_prob")


def _joint_mean(self: Any) -> dict[str, Any]:
    """The mean of an edge-free joint, each component's from the factor that produces it.

    Returns
    -------
    dict
        Each component's mean, keyed by component in canonical factor order.
    """
    return _componentwise(self._graph, {}, "_mean")


def _joint_variance(self: Any) -> dict[str, Any]:
    """The variance of an edge-free joint, each component's from the factor that produces it.

    Returns
    -------
    dict
        Each component's variance, keyed by component in canonical factor order.
    """
    return _componentwise(self._graph, {}, "_variance")


def _joint_cov(self: Any) -> DenseLinOp:
    """The covariance of an edge-free joint over its flattened draw.

    The factors are independent, so the covariance is block diagonal, with each
    factor's covariance as a block in factor order, the order of the joint's
    flattened coordinates.

    Returns
    -------
    DenseLinOp
        The ``(d, d)`` covariance, where ``d`` is the size of the flattened draw.
    """
    return _block_diagonal(self._graph, {})


def _joint_quantile(self: Any, q: ArrayLike) -> dict[str, Any]:
    """The quantiles of an edge-free joint at the levels *q*, assembled per component.

    A whole-term factor's quantiles are its component's, and an exposed record's
    are the children of the factor's result. Each factor's result is its
    event's raw form with the level axes leading in each leaf, so the joint's is
    the same form of its own event.

    Returns
    -------
    dict
        Each component's quantiles, keyed by component in canonical factor order.
    """
    return _componentwise(self._graph, {}, "_quantile", q)


def _joint_conditional_sample(
    self: Any,
    given: Record | Mapping[str, Any],
    key: PRNGKey,
    sample_shape: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Draw from the joint at *given* by ancestral sampling over its factors.

    Each unmet given takes its value from *given*, and the factors draw as for
    the unconditional joint's ``_sample``, keys included.

    Returns
    -------
    dict
        Each component's raw value, keyed by component in canonical factor order.

    Raises
    ------
    KeyError
        If *given* names a slot the joint does not have, or omits one it has.
    """
    return _ancestral_sample(self._graph, _given_values(self, given), key, tuple(sample_shape))


def _joint_conditional_log_prob(self: Any, given: Record | Mapping[str, Any], value: Any) -> Array:
    """The normalized log-density of the joint at *given*: the sum of its factors'.

    Each unmet given takes its value from *given*, and the factors are scored as
    the unconditional joint's are.

    Raises
    ------
    KeyError
        If *given* names a slot the joint does not have, or omits one it has.
    """
    return _score(self._graph, _given_values(self, given), value, "_log_prob")


def _joint_conditional_unnormalized_log_prob(
    self: Any, given: Record | Mapping[str, Any], value: Any
) -> Array:
    """The log-density of the joint at *given*, up to an additive constant.

    Raises
    ------
    KeyError
        If *given* names a slot the joint does not have, or omits one it has.
    """
    return _score(self._graph, _given_values(self, given), value, "_unnormalized_log_prob")


def _joint_conditional_mean(self: Any, given: Record | Mapping[str, Any]) -> dict[str, Any]:
    """The mean of an edge-free joint at *given*, each component's from its factor.

    Raises
    ------
    KeyError
        If *given* names a slot the joint does not have, or omits one it has.
    """
    return _componentwise(self._graph, _given_values(self, given), "_mean")


def _joint_conditional_variance(self: Any, given: Record | Mapping[str, Any]) -> dict[str, Any]:
    """The variance of an edge-free joint at *given*, each component's from its factor.

    Raises
    ------
    KeyError
        If *given* names a slot the joint does not have, or omits one it has.
    """
    return _componentwise(self._graph, _given_values(self, given), "_variance")


def _joint_conditional_cov(self: Any, given: Record | Mapping[str, Any]) -> DenseLinOp:
    """The block-diagonal covariance of an edge-free joint at *given*.

    Raises
    ------
    KeyError
        If *given* names a slot the joint does not have, or omits one it has.
    """
    return _block_diagonal(self._graph, _given_values(self, given))


def _joint_conditional_quantile(
    self: Any, given: Record | Mapping[str, Any], q: ArrayLike
) -> dict[str, Any]:
    """The quantiles of an edge-free joint at *given* and the levels *q*, per component.

    Raises
    ------
    KeyError
        If *given* names a slot the joint does not have, or omits one it has.
    """
    return _componentwise(self._graph, _given_values(self, given), "_quantile", q)


def _joint_marginal(self: Any, path: str | tuple[str, ...]) -> Distribution:
    """The exact marginal of the joint at *path*, detached from the joint.

    A factor whose every component is requested is kept as it is, and a factor
    requested in part is reduced to its own marginal at its requested paths. A
    projection onto one path returns the one factor it keeps or reduces, as a
    whole term: a kept factor that exposes a record of that one component is
    returned as the law of its field. A selection of several paths returns an
    exposed record: the one factor itself when it exposes a record, and
    otherwise the joint of the kept factors in factor order, repackaged when
    the paths name the fields in another order, so its fields follow the
    order of the paths (III.8). The marginal is labeled as the view at *path*
    is: by the path, or by the paths of a selection joined with ``", "``.

    Parameters
    ----------
    path : str or tuple of str
        An event path of the joint, or a selection of several.

    Returns
    -------
    Distribution
        The marginal, which shares the joint's factors.

    Raises
    ------
    KeyError
        If a path is not an event path of the joint.
    TypeError
        If a path is not a string.
    ValueError
        If a selection names no path, or two of its paths share a final
        segment.
    ResolutionError
        If the marginal guard does not accept *path*, so no exact marginal is
        available there.
    """
    paths = _requested_paths(self, path)
    report = _marginal_guard(self, path)
    if report.feasible is not True:
        reason = report.description or "; ".join(report.pending)
        raise ResolutionError(f"{self.name!r} has no exact marginal at {path!r}: {reason}")
    graph: _FactorGraph = self._graph
    projection = isinstance(path, str)
    label = path if projection else ", ".join(paths)
    kept: list[Distribution] = []
    for index, requested in _requests(graph, paths).items():
        factor = graph.factors[index]
        if not _kept_whole(factor, requested):
            kept.append(factor._marginal(_factor_request(requested)))
        elif projection and factor.event_spec.exposes_record:
            kept.append(_SoleField(factor))
        else:
            kept.append(factor)
    if len(kept) == 1 and (projection or kept[0].event_spec.exposes_record):
        (marginal,) = kept
        marginal = marginal if marginal.name == label else marginal.with_name(label)
    else:
        marginal = FactoredDistribution(label, kept)
    return marginal if projection else _in_requested_order(marginal, paths)


def _in_requested_order(marginal: Distribution, paths: tuple[str, ...]) -> Distribution:
    """*marginal*, the marginal at the selection *paths*, with its fields in the order of the paths.

    The kept factors declare their fields in factor order, so a selection that
    names them in another order is repackaged, each field keeping its path.
    """
    from ._views import _leaf_paths, _renamed_by_leaves

    order = [requested.rsplit(_PATH_SEP, 1)[-1] for requested in paths]
    components = marginal.event_spec.components
    if list(components) == order:
        return marginal
    target = OutputSpec(RecordSpec({name: components[name] for name in order}))
    leaves = {leaf: leaf for leaf in _leaf_paths(marginal.event_spec)}
    return _renamed_by_leaves(marginal, target, leaves)


def _kept_claims(factor: Factor, requested: tuple[str, ...]) -> frozenset[type]:
    """The claims of *factor* as a marginal keeps it: whole, or reduced to *requested*.

    A conditional factor's claims are those of the law it yields at a given
    value, and a factor reduced to part of its event reports its own marginal
    there.
    """
    if isinstance(factor, ConditionalDistribution):
        return _kernel_claims(factor)
    if _kept_whole(factor, requested):
        return _claims(factor)
    return _marginal_claims(factor, _factor_request(requested))


def _joint_marginal_capabilities(self: Any, path: str | tuple[str, ...]) -> frozenset[type]:
    """The claims of the factors that the marginal at *path* keeps, read from their declarations.

    One kept factor reports its own claims, since the marginal is that factor
    or its reduction. Several report what the joint of them claims: sampling
    and each density when every kept factor has it, a moment when no kept
    factor conditions on another and every one has the moment, and marginals.

    Raises
    ------
    KeyError
        If a path is not an event path of the joint.
    TypeError
        If a path is not a string.
    ValueError
        If a selection names no path, or two of its paths share a final segment.
    """
    graph: _FactorGraph = self._graph
    requests = _requests(graph, _requested_paths(self, path))
    reports = [
        _kept_claims(graph.factors[index], requested) for index, requested in requests.items()
    ]
    if len(reports) == 1:
        return reports[0]

    def every(protocol: type) -> bool:
        return all(protocol in report for report in reports)

    claims = {SupportsMarginals}
    if every(SupportsSampling):
        claims.add(SupportsSampling)
    if every(SupportsLogProb):
        claims |= {SupportsLogProb, SupportsUnnormalizedLogProb}
    elif every(SupportsUnnormalizedLogProb):
        claims.add(SupportsUnnormalizedLogProb)
    if not any(
        consumer in requests and producer in requests for consumer, producer, _ in graph.edges
    ):
        claims.update(
            moment
            for moment in (SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile)
            if every(moment)
        )
    return frozenset(claims)


def _all_claim(factors: Sequence[Factor], protocol: type) -> bool:
    """Whether every factor claims *protocol*, a conditional factor through its twin."""
    twin = _CONDITIONAL_TWINS[protocol]
    return all(
        isinstance(factor, twin if isinstance(factor, ConditionalDistribution) else protocol)
        for factor in factors
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

#: The function that realizes each method a joint may claim, keyed by method name.
_JOINT_IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {
    "_sample": _joint_sample,
    "_log_prob": _joint_log_prob,
    "_unnormalized_log_prob": _joint_unnormalized_log_prob,
    "_mean": _joint_mean,
    "_variance": _joint_variance,
    "_cov": _joint_cov,
    "_quantile": _joint_quantile,
    "_conditional_sample": _joint_conditional_sample,
    "_conditional_log_prob": _joint_conditional_log_prob,
    "_conditional_unnormalized_log_prob": _joint_conditional_unnormalized_log_prob,
    "_conditional_mean": _joint_conditional_mean,
    "_conditional_variance": _joint_conditional_variance,
    "_conditional_cov": _joint_conditional_cov,
    "_conditional_quantile": _joint_conditional_quantile,
    "_marginal": _joint_marginal,
}


def _factors_guard(owner: str, joint_method: str, method: str) -> Callable[[Any], Feasibility]:
    """The guard of a joint's *joint_method*, which calls each factor's *method*.

    A conditional factor is called through its twin, ``_conditional`` followed by
    *method*, and the joint's call is feasible when every factor's is.
    """

    def guard(self: Any) -> Feasibility:
        return _conjunction(
            _capability_guard(
                factor,
                f"_conditional{method}" if isinstance(factor, ConditionalDistribution) else method,
            )
            for factor in self._graph.factors
        )

    guard.__name__ = f"{joint_method}_guard"
    guard.__qualname__ = f"{owner}.{joint_method}_guard"
    guard.__doc__ = f"Every factor's guard of the ``{method}`` the joint calls on it."
    return guard


def _joint_table(owner: str, *, conditional: bool) -> dict[type, Mapping[str, Callable[..., Any]]]:
    """Each capability a joint of the *owner* kind may claim, with its methods and guards.

    A conditional joint claims each capability's conditional twin, whose method
    is ``_conditional`` followed by the unconditional method's name. Each derived
    method carries the conjunction of the factors' guards.
    """
    table: dict[type, Mapping[str, Callable[..., Any]]] = {}
    for protocol, method in _JOINT_METHODS.items():
        joint_protocol, joint_method = protocol, method
        if conditional:
            joint_protocol, joint_method = _CONDITIONAL_TWINS[protocol], f"_conditional{method}"
        table[joint_protocol] = {
            joint_method: _JOINT_IMPLEMENTATIONS[joint_method],
            f"{joint_method}_guard": _factors_guard(owner, joint_method, method),
        }
    if not conditional:
        table[SupportsMarginals] = {
            "_marginal": _JOINT_IMPLEMENTATIONS["_marginal"],
            "_marginal_guard": _marginal_guard,
            "_marginal_capabilities": _joint_marginal_capabilities,
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


def _rebuilt(joint: Any, method: str, mapping: Mapping[str, Any], *, free: Any = None) -> Any:
    """*joint* rebuilt from its factors with *method* applied, recording the transform.

    Raises
    ------
    ValueError
        If *free* is given and *mapping* names a dimension outside it.
    """
    if free is not None:
        unbound = set(mapping) - set(free)
        if unbound:
            raise ValueError(
                f"{type(joint).__name__} {joint.name!r} has no free dimensions "
                f"{sorted(unbound)} to bind"
            )
    base = vars(type(joint)).get("_capability_base", type(joint))
    scope = dict(joint._graph.scope)
    if method == "with_dim_sizes":
        scope.update(mapping)
    rebuilt = base(joint.name, _each_factor(joint.factors, method, mapping), _scope=scope)
    return rebuilt.with_provenance(
        Provenance.create(method, parents=[joint], metadata=dict(mapping))
    )


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
# Refinements
# ---------------------------------------------------------------------------

#: Each registered refinement of the factored law: a more specific class and the
#: predicate over the flattened factors that admits it, in registration order.
_REFINEMENTS: list[tuple[type, Callable[[tuple[Factor, ...]], bool]]] = []


def _register_refinement(cls: type, predicate: Callable[[tuple[Factor, ...]], bool]) -> None:
    """Register *cls* as the class of an unconditional joint whose factors satisfy *predicate*.

    A family registers its factored class at import. ``*``, a joint rebuilt by a
    transform, and a conditional joint bound at its givens construct the most
    specific registered class whose predicate holds for the flattened factors.

    Raises
    ------
    TypeError
        If *cls* is not a subclass of :class:`FactoredDistribution`.
    """
    if not (isinstance(cls, type) and issubclass(cls, FactoredDistribution)):
        raise TypeError(f"a refinement is a subclass of FactoredDistribution, got {cls!r}")
    _REFINEMENTS.append((cls, predicate))


def _refined_class(factors: tuple[Factor, ...]) -> type:
    """The most specific registered class whose predicate holds for *factors*.

    A candidate replaces the class chosen so far when it subclasses it, so of
    two unrelated candidates the earlier registration stands; with none, the
    class is :class:`FactoredDistribution`.
    """
    chosen: type = FactoredDistribution
    for cls, holds in _REFINEMENTS:
        if issubclass(cls, chosen) and holds(factors):
            chosen = cls
    return chosen


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
    marginal at a path is exact. Constructing the class itself gives the most
    specific registered refinement whose predicate the factors satisfy, as a
    family registers its factored class.

    A draw is a mapping from each component to its raw value, in canonical
    factor order, and so is each event-typed moment. Sampling is ancestral: the
    factors draw from right to left, each conditional factor at the components
    it names. Scoring sums the factors' densities, each at its own event
    reconstructed from the components.

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

    def __new__(
        cls, name: str, factors: Sequence[Factor], *, _scope: Mapping[str, int] | None = None
    ) -> FactoredDistribution:
        graph = _factor_graph(factors, _scope)
        base = vars(cls).get("_capability_base", cls)
        if base is FactoredDistribution:
            base = _refined_class(graph.factors)
        return object.__new__(
            _capability_subclass(base, _joint_protocols(graph, conditional=False))
        )

    def __init__(
        self, name: str, factors: Sequence[Factor], *, _scope: Mapping[str, int] | None = None
    ) -> None:
        graph = _factor_graph(factors, _scope)
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

        Parameters
        ----------
        **sizes : int
            Sizes for free dimensions of the joint.

        Returns
        -------
        Self
            The joint of the bound factors, under the same name.

        Raises
        ------
        ValueError
            If a name is not a free dimension of the joint.
        """
        return _rebuilt(self, "with_dim_sizes", sizes, free=self.event_spec.spec.free_dims)

    def with_dim_names(self, **names: str) -> Self:
        """Rename symbolic dimensions in every factor, simultaneously.

        Parameters
        ----------
        **names : str
            New names keyed by old; names that are not free are ignored.

        Returns
        -------
        Self
            The joint of the renamed factors, under the same name.
        """
        return _rebuilt(self, "with_dim_names", names)


class FactoredConditionalDistribution(ConditionalDistribution, SupportsFactors):
    """A joint conditional distribution: factors that leave some givens unmet.

    Its ``given_spec`` is exactly the unmet givens, same-named ones unified into
    one slot, and its event declaration is read as for
    :class:`FactoredDistribution`. Conditioning on every given yields a
    ``FactoredDistribution``, and conditioning on some curries to a smaller
    ``FactoredConditionalDistribution``.

    Each conditional capability is its unconditional counterpart's on
    :class:`FactoredDistribution`, with every unmet given set by the ``given``
    of the call, and each factor is called through its own capability at that
    value.

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

    def __new__(
        cls, name: str, factors: Sequence[Factor], *, _scope: Mapping[str, int] | None = None
    ) -> FactoredConditionalDistribution:
        protocols = _joint_protocols(_factor_graph(factors, _scope), conditional=True)
        base = vars(cls).get("_capability_base", cls)
        return object.__new__(_capability_subclass(base, protocols))

    def __init__(
        self, name: str, factors: Sequence[Factor], *, _scope: Mapping[str, int] | None = None
    ) -> None:
        graph = _factor_graph(factors, _scope)
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

    def with_dim_sizes(self, **sizes: int) -> Self:
        """Bind named symbolic dimensions in every factor that declares them.

        Parameters
        ----------
        **sizes : int
            Sizes for free dimensions of either side.

        Returns
        -------
        Self
            The joint of the bound factors, under the same name.

        Raises
        ------
        ValueError
            If a name is not a free dimension of the joint.
        """
        return _rebuilt(self, "with_dim_sizes", sizes, free=self.spec.free_dims)

    def with_dim_names(self, **names: str) -> Self:
        """Rename symbolic dimensions in every factor, simultaneously.

        Parameters
        ----------
        **names : str
            New names keyed by old; names that are not free are ignored.

        Returns
        -------
        Self
            The joint of the renamed factors, under the same name.
        """
        return _rebuilt(self, "with_dim_names", names)

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **options: Any
    ) -> Distribution | ConditionalDistribution:
        """Bind given slots in every factor that names them, and rebuild the joint.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values for some or all of the joint's given slots, by slot name;
            every given value arrives here.
        **options : Any
            Options for the primitive of each factor that a value binds.

        Returns
        -------
        FactoredDistribution or FactoredConditionalDistribution
            The joint over the bound factors, conditional while a given remains.

        Raises
        ------
        KeyError
            If a bound name is not a given slot of the joint.
        ValueError
            If a factor's primitive returns a law or kernel whose declarations do
            not match the factor's.
        """
        top = given.children if hasattr(given, "children") else given
        values = dict(top.items())
        unknown = set(values) - set(self.given_spec)
        if unknown:
            raise KeyError(f"{sorted(unknown)} are not given slots of {self.name!r}")
        factors: list[Factor] = []
        for factor in self.factors:
            if isinstance(factor, ConditionalDistribution):
                bound = {slot: value for slot, value in values.items() if slot in factor.given_spec}
                if bound:
                    factor = _bound_factor(factor, bound, options)
            factors.append(factor)
        if set(values) == set(self.given_spec):
            return FactoredDistribution(self.name, factors)
        return FactoredConditionalDistribution(self.name, factors)


def _bound_factor(
    factor: ConditionalDistribution, bound: Mapping[str, Any], options: Mapping[str, Any]
) -> Factor:
    """*factor* conditioned on *bound* under *options*, checked to keep its declarations.

    Raises
    ------
    ValueError
        If the primitive returns a law or kernel whose event declaration, or whose
        remaining given slots, differ from the factor's.
    """
    result = factor._condition_on(bound, **options)
    remaining = {slot: spec for slot, spec in factor.given_spec.items() if slot not in bound}
    expected = (
        ConditionalDistributionSpec(remaining, factor.event_spec)
        if remaining
        else DistributionSpec(factor.event_spec)
    )
    if not expected.is_valid(result):
        raise ValueError(
            f"{factor.name!r} conditioned on {sorted(bound)} returned "
            f"{type(result).__name__} {getattr(result, 'name', '')!r}, whose declarations do not "
            f"match the factor's"
        )
    return result


# ---------------------------------------------------------------------------
# The numeric markers
# ---------------------------------------------------------------------------


class FactoredNumericDistribution(FactoredDistribution):
    """The marker of an unconditional joint whose event is numeric.

    Membership is read from the declaration, as for ``NumericDistribution``, and
    a class that inherits the marker claims it for every instance, which
    construction checks.
    """

    _membership_follows_declaration = True


class FactoredConditionalNumericDistribution(FactoredConditionalDistribution):
    """The marker of a conditional joint whose event is numeric, read from its declaration."""

    _membership_follows_declaration = True


class FactoredNumericConditionalDistribution(FactoredConditionalDistribution):
    """The marker of a conditional joint whose given slots are numeric, by its declaration."""

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
