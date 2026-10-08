"""The mixture family: a convex combination of laws over one event declaration.

Provides:
  - ``MixtureDistribution`` – the finite mixture of its components, what
    ``mixture`` returns for a finite mixing distribution and what the Monte
    Carlo mean of a law over laws returns.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from .. import _messages
from ..core._record_spec import RecordSpec
from ..core._repr import format_value, sequence_repr
from ..core._spec_base import NumericArraySpec, TermSpec
from ..core._specs import OutputSpec
from ..core.constraints import _known_equal
from ..distributions._capabilities import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
    _capability_guard,
    _capability_subclass,
    _conjunction,
)
from ..distributions._conversion import _event_difference, _term_difference
from ..distributions._distribution import Distribution
from ..distributions._factored import _raw_record
from ..linalg import DenseLinOp, LinOp
from ..operations import _moments

if TYPE_CHECKING:
    from ..core._dispatch import Feasibility
    from ..custom_types import Array, ArrayLike, PRNGKey

__all__ = ["MixtureDistribution"]


# ---------------------------------------------------------------------------
# The components' values in raw form
# ---------------------------------------------------------------------------


def _leaves(value: Any, spec: Any) -> list[Array]:
    """The array leaves of *value*, a raw value of the event *spec*, in the declaration's order.

    An array value is its one leaf, and a record value, a ``Record`` or the
    nested mapping of its leaves, has the leaves of its record's paths.
    """
    if not isinstance(spec, RecordSpec):
        return [jnp.asarray(value)]
    raw = _raw_record(value)
    leaves = []
    for path in spec:
        leaf = raw
        for segment in path.split("/"):
            leaf = leaf[segment]
        leaves.append(jnp.asarray(leaf))
    return leaves


def _combined(values: Sequence[Any], spec: Any, combine: Callable[..., Array]) -> Any:
    """The raw value whose leaves are *combine* of the components' leaves at each path.

    Each of *values* is a raw value of the event *spec*: an array for an array
    event, and a ``Record`` or the nested mapping of its leaves for a record
    event, which the result returns as the nested mapping.
    """
    columns = zip(*(_leaves(value, spec) for value in values), strict=True)
    combined = [combine(*column) for column in columns]
    if not isinstance(spec, RecordSpec):
        return combined[0]
    nested: dict[str, Any] = {}
    for path, leaf in zip(spec, combined, strict=True):
        *groups, field = path.split("/")
        node = nested
        for group in groups:
            node = node.setdefault(group, {})
        node[field] = leaf
    return nested


def _weighted_sum(weights: Array) -> Callable[..., Array]:
    """The function summing one leaf per component, each scaled by its component's weight."""

    def total(*leaves: Array) -> Array:
        stacked = jnp.stack([jnp.asarray(leaf) for leaf in leaves])
        scale = jnp.reshape(weights, (-1,) + (1,) * (stacked.ndim - 1))
        return jnp.sum(scale * stacked, axis=0)

    return total


def _flat(value: Any, spec: Any) -> Array:
    """The flat coordinates of one raw value of *spec*: its leaves raveled and concatenated."""
    return jnp.concatenate([jnp.ravel(leaf) for leaf in _leaves(value, spec)])


# ---------------------------------------------------------------------------
# The capabilities
# ---------------------------------------------------------------------------


def _mixture_sample(
    self: MixtureDistribution, key: PRNGKey, sample_shape: tuple[int, ...] = ()
) -> Any:
    """Draws whose component is chosen by the weights, each from its chosen component.

    Each draw's component is chosen first, and each component then draws as
    many times as it was chosen, so the draws are independent with the
    mixture's law. Under tracing, where the counts are unknown, every
    component draws at *sample_shape* and each draw keeps its chosen
    component's.
    """
    shape = tuple(sample_shape)
    choice_key, *component_keys = jax.random.split(key, len(self._components) + 1)
    choice = jax.random.categorical(choice_key, jnp.log(self._weights), shape=shape)
    if isinstance(choice, jax.core.Tracer) or not math.prod(shape):
        return _drawn_from_every_component(self, choice, component_keys, shape)
    chosen = np.asarray(choice).reshape(-1)
    positions = [np.flatnonzero(chosen == index) for index in range(len(self._components))]
    drawn = [
        (component._sample(component_key, (len(where),)), where)
        for component, component_key, where in zip(
            self._components, component_keys, positions, strict=True
        )
        if len(where)
    ]
    # The draws come grouped by component; this order puts each at its position.
    order = np.argsort(np.concatenate([where for _, where in drawn]), kind="stable")

    def assembled(*leaves: Array) -> Array:
        rows = jnp.concatenate([jnp.asarray(leaf) for leaf in leaves])[order]
        return jnp.reshape(rows, (*shape, *rows.shape[1:]))

    return _combined([draws for draws, _ in drawn], self.event_spec.spec, assembled)


def _drawn_from_every_component(
    self: MixtureDistribution, choice: Array, keys: Sequence[PRNGKey], shape: tuple[int, ...]
) -> Any:
    """Draws at the components *choice* holds, every component drawing at *shape*."""
    draws = [
        component._sample(component_key, shape)
        for component, component_key in zip(self._components, keys, strict=True)
    ]

    def chosen(*leaves: Array) -> Array:
        selected = jnp.asarray(leaves[0])
        for index, leaf in enumerate(leaves[1:], start=1):
            leaf = jnp.asarray(leaf)
            mask = jnp.reshape(choice == index, choice.shape + (1,) * (leaf.ndim - choice.ndim))
            selected = jnp.where(mask, leaf, selected)
        return selected

    return _combined(draws, self.event_spec.spec, chosen)


def _mixture_log_prob(self: MixtureDistribution, value: ArrayLike) -> Array:
    """The weighted log-sum-exp ``log Σ wᵢ pᵢ(x)`` of the components' log-densities.

    The leading axes of a batch of values are kept, as each component keeps them.
    """
    scores = jnp.stack([component._log_prob(value) for component in self._components])
    log_weights = jnp.reshape(jnp.log(self._weights), (-1,) + (1,) * (scores.ndim - 1))
    return jax.scipy.special.logsumexp(log_weights + scores, axis=0)


def _mixture_mean(self: MixtureDistribution) -> Any:
    """The weighted mean ``Σ wᵢ mᵢ`` of the components' means, shaped like one draw."""
    means = [component._mean() for component in self._components]
    return _combined(means, self.event_spec.spec, _weighted_sum(self._weights))


def _mixture_variance(self: MixtureDistribution) -> Any:
    """The variance ``Σ wᵢ (vᵢ + mᵢ²) − m²`` of each coordinate, shaped like one draw."""
    spec = self.event_spec.spec
    means = [component._mean() for component in self._components]
    second_moments = [
        _combined([mean, component._variance()], spec, lambda m, v: v + m**2)
        for component, mean in zip(self._components, means, strict=True)
    ]
    mean = _combined(means, spec, _weighted_sum(self._weights))
    second = _combined(second_moments, spec, _weighted_sum(self._weights))
    return _combined([second, mean], spec, lambda s, m: s - m**2)


def _mixture_cov(self: MixtureDistribution) -> LinOp:
    """The covariance ``Σ wᵢ (Σᵢ + mᵢ mᵢᵀ) − m mᵀ`` of the flat coordinates, as a dense operator."""
    spec = self.event_spec.spec
    means = jnp.stack([_flat(component._mean(), spec) for component in self._components])
    covariances = jnp.stack(
        [jnp.asarray(component._cov().to_dense()) for component in self._components]
    )
    mean = self._weights @ means
    second = jnp.einsum("k,kij->ij", self._weights, covariances) + jnp.einsum(
        "k,ki,kj->ij", self._weights, means, means
    )
    return DenseLinOp(second - jnp.outer(mean, mean))


def _components_guard(capability: str, methods: tuple[str, ...]) -> Callable[[Any], Feasibility]:
    """The guard of the mixture's *capability*, which calls each component's *methods*."""

    def guard(self: MixtureDistribution) -> Feasibility:
        return _conjunction(
            _capability_guard(component, method)
            for component in self._components
            for method in methods
        )

    guard.__name__ = f"{capability}_guard"
    guard.__qualname__ = f"MixtureDistribution.{capability}_guard"
    guard.__doc__ = f"Every component's guard of the {', '.join(methods)} the mixture calls."
    return guard


#: Each capability a mixture may claim, with the capabilities every component must claim for it.
_REQUIRED: dict[type, tuple[type, ...]] = {
    SupportsSampling: (SupportsSampling,),
    SupportsLogProb: (SupportsLogProb,),
    SupportsMean: (SupportsMean,),
    SupportsVariance: (SupportsMean, SupportsVariance),
    SupportsCovariance: (SupportsMean, SupportsCovariance),
}

#: The capabilities a mixture may claim, with the methods and guards realizing each.
_MIXTURE_CAPABILITIES: dict[type, Mapping[str, Callable[..., Any]]] = {
    SupportsSampling: {
        "_sample": _mixture_sample,
        "_sample_guard": _components_guard("_sample", ("_sample",)),
    },
    SupportsLogProb: {
        "_log_prob": _mixture_log_prob,
        "_log_prob_guard": _components_guard("_log_prob", ("_log_prob",)),
    },
    SupportsMean: {
        "_mean": _mixture_mean,
        "_mean_guard": _components_guard("_mean", ("_mean",)),
    },
    SupportsVariance: {
        "_variance": _mixture_variance,
        "_variance_guard": _components_guard("_variance", ("_mean", "_variance")),
    },
    SupportsCovariance: {
        "_cov": _mixture_cov,
        "_cov_guard": _components_guard("_cov", ("_mean", "_cov")),
    },
}


def _claims(components: Sequence[Distribution]) -> tuple[type, ...]:
    """The capabilities a mixture of *components* claims: those every component supports."""
    return tuple(
        protocol
        for protocol, required in _REQUIRED.items()
        if all(isinstance(component, need) for component in components for need in required)
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _components(components: Sequence[Distribution]) -> tuple[Distribution, ...]:
    """*components* as a tuple, checked to be at least one law sharing one event declaration.

    Parameters
    ----------
    components : Sequence[Distribution]
        The component laws in mixture order, as the constructor received them.

    Returns
    -------
    tuple of Distribution
        The components in the order of *components*.

    Raises
    ------
    TypeError
        If a component is not a ``Distribution``.
    ValueError
        If there is no component, or two components declare different events.
    """
    laws = tuple(components)
    if not laws:
        raise ValueError("MixtureDistribution needs at least one component")
    for index, law in enumerate(laws):
        if not isinstance(law, Distribution):
            raise TypeError(
                f"MixtureDistribution components must be distributions, but component {index} "
                f"is {type(law).__name__}"
            )
    first = laws[0]
    for index, law in enumerate(laws[1:], start=1):
        sides = (f"component 0 ({first.label!r})", f"component {index} ({law.label!r})")
        if _event_difference(first.event_spec, law.event_spec, sides, dtypes=False) is not None:
            raise ValueError(_component_mismatch(first, law, index))
    return laws


def _component_mismatch(first: Distribution, other: Distribution, index: int) -> str:
    """The message for component *index*, *other*, drawing a different event than component 0."""
    head = "MixtureDistribution components must draw the same event"
    expected, actual = first.event_spec, other.event_spec
    if expected.exposes_record != actual.exposes_record:
        return (
            f"{head}: component 0 ({first.label!r}) draws {_drawn(expected)} but "
            f"component {index} ({other.label!r}) draws {_drawn(actual)}"
        )
    if tuple(expected.components) != tuple(actual.components):
        return (
            f"{head}: component 0 ({first.label!r}) has components "
            f"{list(expected.components)} but component {index} ({other.label!r}) has "
            f"{list(actual.components)}. Rename them with with_path_names() so they match."
        )
    for name, spec in expected.components.items():
        mismatch = _first_mismatch(spec, actual.components[name], name)
        if mismatch is not None:
            path, left, right = mismatch
            return (
                f"{head}: {path!r} is {_term(left)} in component 0 but {_term(right)} in "
                f"component {index}"
            )
    return head  # pragma: no cover - _event_difference found a difference above


def _drawn(declaration: OutputSpec) -> str:
    """What a law with *declaration* draws, in a mixture's mismatch message."""
    if declaration.exposes_record:
        return f"a record with fields {list(declaration.components)}"
    (name,) = declaration.components
    return f"a single value {name!r}"


def _first_mismatch(
    left: TermSpec | None, right: TermSpec | None, path: str
) -> tuple[str, TermSpec | None, TermSpec | None] | None:
    """The first path at which the terms *left* and *right* differ, with the two terms there."""
    if _term_difference(left, right, path, dtypes=False) is None:
        return None
    if (
        isinstance(left, RecordSpec)
        and isinstance(right, RecordSpec)
        and tuple(left.children) == tuple(right.children)
    ):
        for name, child in left.children.items():
            found = _first_mismatch(child, right.children[name], f"{path}/{name}")
            if found is not None:
                return found
    return path, left, right


def _term(spec: TermSpec | None) -> str:
    """The term *spec* in plain words, for a mixture's mismatch message."""
    if isinstance(spec, NumericArraySpec):
        return f"an array of shape {spec.shape}"
    if isinstance(spec, RecordSpec):
        return f"a record with fields {list(spec.children)}"
    return f"a {type(spec).__name__}" if spec is not None else "undeclared"


def _joined(specs: Sequence[TermSpec]) -> TermSpec:
    """The term spec the components' *specs* share: each leaf's dtype and support where all agree.

    The specs share their kinds and shapes, and a leaf whose components declare
    different dtypes or supports leaves that metadata undeclared.
    """
    first = specs[0]
    if all(spec == first for spec in specs[1:]):
        return first
    if isinstance(first, NumericArraySpec):
        dtype = first.dtype if all(spec.dtype == first.dtype for spec in specs) else None
        same_support = all(_known_equal(spec.support, first.support) for spec in specs)
        return NumericArraySpec(first.shape, dtype, first.support if same_support else None)
    if isinstance(first, RecordSpec):
        return RecordSpec(
            {name: _joined([spec.children[name] for spec in specs]) for name in first.children}
        )
    return first


def _declaration(laws: Sequence[Distribution]) -> OutputSpec:
    """The event declaration of the mixture of *laws*, which share their packaging and kinds."""
    first = laws[0].event_spec
    return first._with_spec(_joined([law.event_spec.spec for law in laws]))


def _weights(weights: ArrayLike, count: int) -> Array:
    """*weights* as a floating array, one nonnegative weight per component, summing to one.

    A traced array is not checked, since its values are unknown while tracing.

    Parameters
    ----------
    weights : ArrayLike
        The mixture weights, as the constructor received them.
    count : int
        The number of components, which fixes the shape ``(count,)``.

    Returns
    -------
    Array
        The weights, promoted to a floating dtype of at least single precision.

    Raises
    ------
    ValueError
        If *weights* does not have shape ``(count,)``, a weight is negative, or the
        weights do not sum to one.
    """
    array = jnp.asarray(weights)
    array = array.astype(jnp.result_type(array.dtype, jnp.float32))
    if array.shape != (count,):
        raise ValueError(
            f"MixtureDistribution has {_messages.count(count, 'component')} but got weights of "
            f"shape {array.shape}; pass one weight per component"
        )
    if isinstance(array, jax.core.Tracer):
        return array
    if bool(jnp.any(array < 0)):
        raise ValueError(f"mixture weights must be nonnegative, got {array}")
    if not bool(jnp.isclose(jnp.sum(array), 1.0, atol=1e-5)):
        raise ValueError(
            f"mixture weights must sum to 1, got {array} (sum {float(jnp.sum(array)):.6g})"
        )
    return array


#: How many components a mixture's repr shows before it gives their count instead.
_SHOWN_COMPONENTS = 4


class MixtureDistribution(Distribution):
    """A convex combination of component laws that share one event declaration.

    The components share one event declaration, which includes the component
    names, the kind, and the packaging, while their labels may differ; a leaf
    whose components declare different dtypes or supports leaves them
    undeclared in the mixture's declaration. The
    mixture samples when every component samples, choosing a component by the
    weights and then drawing from it, and it has a normalized log-density, the
    weighted log-sum-exp of the components', when every component has one. Its
    moments combine componentwise when every component provides them: the
    mean is ``Σ wᵢ mᵢ``, the variance ``Σ wᵢ (vᵢ + mᵢ²) − m²``, and the
    covariance of the flat coordinates ``Σ wᵢ (Σᵢ + mᵢ mᵢᵀ) − m mᵀ``. Each of
    these capabilities carries the components' guards of the methods it calls.

    Parameters
    ----------
    label : str
        The mixture's label.
    components : Sequence[Distribution]
        The component laws, at least one, sharing one event declaration.
    weights : Array
        One nonnegative weight per component, summing to one.

    Raises
    ------
    TypeError
        If a component is not a ``Distribution``.
    ValueError
        If there is no component, two components declare different events, or
        the weights are not one nonnegative weight per component summing to one.
    """

    _capability_table: ClassVar = _MIXTURE_CAPABILITIES

    _components: tuple[Distribution, ...]
    _weights: Array

    def __new__(
        cls, label: str, components: Sequence[Distribution], weights: ArrayLike
    ) -> MixtureDistribution:
        base = vars(cls).get("_capability_base", cls)
        laws = _components(components)
        return object.__new__(_capability_subclass(base, _claims(laws)))

    def __init__(self, label: str, components: Sequence[Distribution], weights: ArrayLike) -> None:
        laws = _components(components)
        object.__setattr__(self, "_components", laws)
        object.__setattr__(self, "_weights", _weights(weights, len(laws)))
        super().__init__(label, _declaration(laws))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The components, by their count when there are more than four, and the weights."""
        components = self._components
        shown = (
            sequence_repr(repr(law) for law in components)
            if len(components) <= _SHOWN_COMPONENTS
            else repr(len(components))
        )
        return [("components", shown), ("weights", format_value(self._weights))]


# The Monte Carlo mean of a law over laws is the finite mixture of its draws.
_moments._install_mixture(MixtureDistribution)
