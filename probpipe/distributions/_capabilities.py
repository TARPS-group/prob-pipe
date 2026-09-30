"""The capabilities of the distribution kinds.

A **capability** is a protocol that names one underscore implementation, such as
``_sample`` for :class:`SupportsSampling`, which the matching operation calls
through its route on raw forms. Protocol membership establishes that the
implementation exists. Where support is partial, the capability carries a
**guard**: a companion method ``_<method>_guard`` that takes the call's
arguments other than raw values and returns a
:class:`~probpipe.core._dispatch.Feasibility` for that call. A capability with
no guard is total on its declared domain, and :func:`_capability_guard` reads a
guard in either case.

Provides:
  - the unconditional capabilities of a ``Distribution``, including
    :class:`SupportsMarginals`;
  - their conditional twins for a ``ConditionalDistribution``, whose methods
    prepend the conditioning value ``given`` to the unconditional signature;
  - :func:`_capability_subclass`, the subclass of a base that claims exactly a
    chosen set of capabilities, for a class whose instances differ in what
    they support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..core._dispatch import Feasibility

if TYPE_CHECKING:
    from ..core.record import Record
    from ..custom_types import Array, ArrayLike, PRNGKey
    from ..linalg import LinOp
    from ._distribution import Distribution

__all__ = [
    "SupportsApproximateConditioning",
    "SupportsConditionalCovariance",
    "SupportsConditionalExpectation",
    "SupportsConditionalLogProb",
    "SupportsConditionalMarginals",
    "SupportsConditionalMean",
    "SupportsConditionalQuantile",
    "SupportsConditionalRandomLogProb",
    "SupportsConditionalRandomUnnormalizedLogProb",
    "SupportsConditionalSampling",
    "SupportsConditionalUnnormalizedLogProb",
    "SupportsConditionalVariance",
    "SupportsCovariance",
    "SupportsExactConditioning",
    "SupportsExpectation",
    "SupportsLogProb",
    "SupportsMarginals",
    "SupportsMean",
    "SupportsQuantile",
    "SupportsRandomLogProb",
    "SupportsRandomUnnormalizedLogProb",
    "SupportsSampling",
    "SupportsUnnormalizedLogProb",
    "SupportsVariance",
]


# ---------------------------------------------------------------------------
# The unconditional capabilities
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsSampling(Protocol):
    """A distribution that draws via ``_sample(key, sample_shape)``.

    ``sample_shape=()`` returns one draw in its raw form, and a non-empty shape
    prepends independent-draw axes to it: an array for an array-drawing law, a
    record of stacked columns for a record-drawing one. The ``sample``
    operation names the axes it adds, since a law cannot know what a caller's
    ``sample_shape`` means.
    """

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Any: ...


@runtime_checkable
class SupportsUnnormalizedLogProb(Protocol):
    """A distribution with a log-density up to an additive constant.

    ``_unnormalized_log_prob(value)`` scores one draw, or a batch of draws
    whose leading axes the result keeps.
    """

    def _unnormalized_log_prob(self, value: Any) -> Array: ...


@runtime_checkable
class SupportsLogProb(SupportsUnnormalizedLogProb, Protocol):
    """A distribution with the normalized log-density ``_log_prob(value)``.

    It refines :class:`SupportsUnnormalizedLogProb`, so the unnormalized
    density defaults to the normalized one.
    """

    def _log_prob(self, value: Any) -> Array: ...

    def _unnormalized_log_prob(self, value: Any) -> Array:
        """The normalized log-density, which is also an unnormalized one."""
        return self._log_prob(value)


@runtime_checkable
class SupportsRandomUnnormalizedLogProb(Protocol):
    """A random measure with a random unnormalized log-density.

    For a random measure ``M``, ``_random_unnormalized_log_prob()`` returns the
    law of ``x ↦ log D̃(x)`` with ``D̃`` the unnormalized density of a draw
    ``D ~ M``, itself a random function.
    """

    def _random_unnormalized_log_prob(self) -> Distribution: ...


@runtime_checkable
class SupportsRandomLogProb(Protocol):
    """A random measure with a random normalized log-density, ``_random_log_prob()``.

    It returns the law of ``x ↦ log D(x)`` with ``D ~ M``, as
    :class:`SupportsRandomUnnormalizedLogProb` does for the unnormalized one.
    """

    def _random_log_prob(self) -> Distribution: ...


@runtime_checkable
class SupportsMean(Protocol):
    """A distribution with an exact mean, ``_mean()``, a value shaped like one draw.

    A random function's mean is its mean function, and a random measure's is
    the marginalized law.
    """

    def _mean(self) -> Any: ...


@runtime_checkable
class SupportsVariance(Protocol):
    """A distribution with an exact variance, ``_variance()``, a value shaped like one draw."""

    def _variance(self) -> Any: ...


@runtime_checkable
class SupportsCovariance(Protocol):
    """A numeric distribution with an exact covariance, ``_cov()``.

    The result is a ``(d, d)`` linear operator over the flattened draw, whose
    size is ``d``.
    """

    def _cov(self) -> LinOp: ...


@runtime_checkable
class SupportsQuantile(Protocol):
    """A numeric distribution with quantiles, ``_quantile(q)``, per coordinate at each level."""

    def _quantile(self, q: ArrayLike) -> Array: ...


@runtime_checkable
class SupportsExpectation(Protocol):
    """A distribution with the exact expectation ``E[f(X)]`` of an arbitrary ``f``.

    ``_expectation(f)`` integrates any function exactly, which in practice
    means finite support. Its argument is an opaque callable that no guard can
    inspect, so a law that is exact only for special maps does not claim the
    capability; the ``expectation`` operation estimates such a law's
    expectation through a registered method instead.
    """

    def _expectation(self, f: Callable[[Any], Array]) -> Array: ...


class SupportsExactConditioning(ABC):
    """A distribution whose ``_condition_on`` returns the conditional law.

    Inherit this to claim exact conditioning, such as a conjugate update or
    reweighting an empirical joint; ``condition_on`` prefers it over the
    inference registry and keeps it when the caller asks for exactness. The
    capability is claimed by inheriting rather than by defining
    ``_condition_on``: exactness is a claim about the result, which no
    structural check can read, and two protocols declaring the same method
    would match the same classes.
    """

    @abstractmethod
    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any: ...


class SupportsApproximateConditioning(ABC):
    """A distribution whose ``_condition_on`` returns a stand-in for the conditional law.

    Inherit this to claim a built-in conditioning path that does not return
    the conditional law itself, such as an amortized posterior that runs one
    forward pass. ``condition_on`` prefers it over the inference registry and
    excludes it when the caller asks for exactness. A model whose conditioning
    requires MCMC or variational inference claims neither capability, so the
    inference registry selects an algorithm for it.
    """

    @abstractmethod
    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# The marginal capability
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsMarginals(Protocol):
    """A distribution that returns the detached marginal of a field or field group.

    ``_marginal(path)`` returns a standalone ``Distribution`` over the node at
    ``path``, an event path of this law, with no reference back to it. A path
    that names an interior node selects the group of fields under it, and a
    tuple of paths selects several nodes, returned as an exposed record of
    them. Support may depend on the path, so a class whose marginal is exact
    only at some paths defines the guard ``_marginal_guard(path)``.
    """

    def _marginal(self, path: str | tuple[str, ...]) -> Distribution: ...


# ---------------------------------------------------------------------------
# The conditional twins
# ---------------------------------------------------------------------------
#
# Every unconditional capability has a conditional counterpart whose method is
# named ``_conditional_<name>`` and prepends ``given`` to the unconditional
# signature. The names differ because a ``@runtime_checkable`` check matches on
# method name alone.


@runtime_checkable
class SupportsConditionalSampling(Protocol):
    """A kernel that draws from ``K(given, ·)``, the twin of :class:`SupportsSampling`."""

    def _conditional_sample(
        self,
        given: Record | Mapping[str, Any],
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Any: ...


@runtime_checkable
class SupportsConditionalUnnormalizedLogProb(Protocol):
    """A kernel with an unnormalized log-density of ``K(given, ·)``."""

    def _conditional_unnormalized_log_prob(
        self, given: Record | Mapping[str, Any], value: Any
    ) -> Array: ...


@runtime_checkable
class SupportsConditionalLogProb(SupportsConditionalUnnormalizedLogProb, Protocol):
    """A kernel with the normalized log-density of ``K(given, ·)``.

    It refines :class:`SupportsConditionalUnnormalizedLogProb`, as
    :class:`SupportsLogProb` refines its unnormalized counterpart, so the
    unnormalized density defaults to the normalized one.
    """

    def _conditional_log_prob(self, given: Record | Mapping[str, Any], value: Any) -> Array: ...

    def _conditional_unnormalized_log_prob(
        self, given: Record | Mapping[str, Any], value: Any
    ) -> Array:
        """The normalized log-density, which is also an unnormalized one."""
        return self._conditional_log_prob(given, value)


@runtime_checkable
class SupportsConditionalRandomUnnormalizedLogProb(Protocol):
    """A random-measure kernel: the law of ``x ↦ log D̃(x)`` with ``D ~ K(given, ·)``."""

    def _conditional_random_unnormalized_log_prob(
        self, given: Record | Mapping[str, Any]
    ) -> Distribution: ...


@runtime_checkable
class SupportsConditionalRandomLogProb(Protocol):
    """Like :class:`SupportsConditionalRandomUnnormalizedLogProb`, with the normalized density."""

    def _conditional_random_log_prob(self, given: Record | Mapping[str, Any]) -> Distribution: ...


@runtime_checkable
class SupportsConditionalMean(Protocol):
    """A kernel with the mean of ``K(given, ·)``, a value shaped like one draw."""

    def _conditional_mean(self, given: Record | Mapping[str, Any]) -> Any: ...


@runtime_checkable
class SupportsConditionalVariance(Protocol):
    """A kernel with the variance of ``K(given, ·)``, a value shaped like one draw."""

    def _conditional_variance(self, given: Record | Mapping[str, Any]) -> Any: ...


@runtime_checkable
class SupportsConditionalCovariance(Protocol):
    """A kernel with the covariance of ``K(given, ·)``, a ``(d, d)`` operator."""

    def _conditional_cov(self, given: Record | Mapping[str, Any]) -> LinOp: ...


@runtime_checkable
class SupportsConditionalQuantile(Protocol):
    """A numeric kernel with the quantiles of ``K(given, ·)`` at the levels ``q``."""

    def _conditional_quantile(self, given: Record | Mapping[str, Any], q: ArrayLike) -> Array: ...


@runtime_checkable
class SupportsConditionalExpectation(Protocol):
    """A kernel with the exact expectation ``E[f(Y)]`` for ``Y ~ K(given, ·)``."""

    def _conditional_expectation(
        self, given: Record | Mapping[str, Any], f: Callable[[Any], Array]
    ) -> Array: ...


@runtime_checkable
class SupportsConditionalMarginals(Protocol):
    """A kernel with the detached marginal of ``K(given, ·)`` at an event path."""

    def _conditional_marginal(
        self, given: Record | Mapping[str, Any], path: str | tuple[str, ...]
    ) -> Distribution: ...


#: Each unconditional capability and its conditional twin.
_CONDITIONAL_TWINS: dict[type, type] = {
    SupportsSampling: SupportsConditionalSampling,
    SupportsUnnormalizedLogProb: SupportsConditionalUnnormalizedLogProb,
    SupportsLogProb: SupportsConditionalLogProb,
    SupportsRandomUnnormalizedLogProb: SupportsConditionalRandomUnnormalizedLogProb,
    SupportsRandomLogProb: SupportsConditionalRandomLogProb,
    SupportsMean: SupportsConditionalMean,
    SupportsVariance: SupportsConditionalVariance,
    SupportsCovariance: SupportsConditionalCovariance,
    SupportsQuantile: SupportsConditionalQuantile,
    SupportsExpectation: SupportsConditionalExpectation,
    SupportsMarginals: SupportsConditionalMarginals,
}


# ---------------------------------------------------------------------------
# Guards and per-instance capability sets
# ---------------------------------------------------------------------------


def _capability_guard(term: Any, method: str, *arguments: Any, **keywords: Any) -> Feasibility:
    """The guard of *term*'s capability *method* for one call.

    Parameters
    ----------
    term : Any
        The object claiming the capability.
    method : str
        The capability's method name, such as ``"_marginal"``.
    *arguments, **keywords
        The call's arguments other than raw values, such as a marginal's path.

    Returns
    -------
    Feasibility
        The report of ``term._<method>_guard(*arguments, **keywords)`` when the
        class defines that guard, and a feasible report otherwise, since a
        capability with no guard is total on its declared domain.

    Raises
    ------
    AttributeError
        If *term* does not implement *method*.
    """
    if not callable(getattr(term, method, None)):
        raise AttributeError(f"{type(term).__name__} does not implement {method}")
    guard = getattr(term, f"{method}_guard", None)
    if guard is None:
        return Feasibility(True)
    return guard(*arguments, **keywords)


_CAPABILITY_SUBCLASSES: dict[tuple[type, frozenset[type]], type] = {}


def _capability_subclass(base: type, protocols: Iterable[type]) -> type:
    """The subclass of *base* that claims exactly *protocols*.

    Membership in a capability is decided by the class, so a class whose
    instances support different operations constructs each instance as the
    subclass claiming that instance's set. *base* declares
    ``_capability_table``, which maps each protocol an instance may claim to the
    methods realizing it, keyed by method name. The subclass is created once per
    base and set of protocols, keeps the base's name, and pickles and copies by
    reference to the base and the set, so it round-trips in a fresh process.

    Returns
    -------
    type
        *base* itself when *protocols* is empty, and the cached subclass
        otherwise, which inherits *base* and every claimed protocol.

    Raises
    ------
    KeyError
        If a protocol has no entry in the base's ``_capability_table``.
    """
    claimed = frozenset(protocols)
    if not claimed:
        return base
    key = (base, claimed)
    cached = _CAPABILITY_SUBCLASSES.get(key)
    if cached is not None:
        return cached
    table: Mapping[type, Mapping[str, Callable[..., Any]]] = base._capability_table
    ordered = tuple(sorted(claimed, key=lambda protocol: protocol.__name__))
    namespace: dict[str, Any] = {
        "__module__": base.__module__,
        "__qualname__": base.__qualname__,
        "__reduce_ex__": _reduce_capability_instance,
        "_capability_base": base,
        "_claimed_capabilities": ordered,
    }
    for protocol in ordered:
        namespace.update(table[protocol])
    subclass = type(base)(base.__name__, (base, *ordered), namespace)
    _CAPABILITY_SUBCLASSES[key] = subclass
    return subclass


def _new_capability_instance(base: type, protocols: tuple[type, ...]) -> Any:
    """An uninitialized instance of the capability subclass, for unpickling."""
    return object.__new__(_capability_subclass(base, protocols))


def _reduce_capability_instance(self: Any, protocol: int) -> tuple[Any, ...]:
    """Pickle an instance of a capability subclass by its base and claimed set."""
    cls = type(self)
    return (
        _new_capability_instance,
        (cls._capability_base, cls._claimed_capabilities),
        self.__getstate__(),
    )


def _claimed(term: Any, protocols: Iterable[type]) -> set[type]:
    """The protocols among *protocols* that *term* claims."""
    return {protocol for protocol in protocols if isinstance(term, protocol)}
