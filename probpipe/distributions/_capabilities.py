"""The capabilities of the distribution kinds.

A **capability** is a protocol that names one underscore implementation, such as
``_sample`` for :class:`SupportsSampling`, which the matching operation calls
through its route on raw forms. Protocol membership establishes that the
implementation exists. Where support depends on the call's arguments, the
capability carries a **guard**: a companion method ``_<method>_guard`` that
takes the call's arguments other than raw values. It returns ``True`` or
``False``, or ``None`` when the answer depends on values not yet known, and a
:class:`~probpipe.core._dispatch.Feasibility` only to give its own reason. The
first paragraph of the guard's docstring is its condition, which the reports
quote. A capability with no guard is total on its declared domain, and
:func:`_capability_guard` reads a guard in either case. Support that depends
only on the instance needs no guard, since the class claims the capability for
those instances alone (:func:`_capability_subclass`).

Provides:
  - the unconditional capabilities of a ``Distribution``, including
    :class:`SupportsMarginals`;
  - their conditional twins for a ``ConditionalDistribution``, whose methods
    prepend the conditioning value ``given`` to the unconditional signature;
  - :func:`_is_normalized` and :func:`_kernel_is_normalized`, which classify a
    law, or a kernel's laws, as normalized from the capabilities claimed;
  - :func:`_capability_subclass`, the subclass of a base that claims exactly a
    chosen set of capabilities, for a class whose instances differ in what
    they support.
"""

from __future__ import annotations

import difflib
import inspect
import reprlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..core._dispatch import Feasibility
from ..core._repr import public_class_name

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
    """A numeric distribution with quantiles, ``_quantile(q)``, per coordinate at each level.

    The result is the event's raw form with the level axes leading in each
    leaf: an array of shape ``(*q.shape, *event_shape)`` for an array event,
    and the nested mapping of such arrays for a record event. The ``quantile``
    operation wraps it: one level as a value of the event's kind, and several
    levels as the matching batch on a level of their own.
    """

    def _quantile(self, q: ArrayLike) -> Array | Mapping[str, Any]: ...


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
    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **options: Any
    ) -> Distribution: ...


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
    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **options: Any
    ) -> Distribution: ...


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
    them. The marginal is labeled as the ``marginal`` operation labels it, so
    a law without factors gives its marginals its own label. Support may
    depend on the path, so a class whose marginal is exact only at some paths
    defines the guard ``_marginal_guard(path)``.

    A class may define the companion ``_marginal_capabilities(path)``, which
    returns the capabilities its exact marginal at ``path`` claims, read from
    its declarations without building the marginal; :func:`_marginal_claims`
    reads it. The marginals of a law that defines none claim what the law
    claims itself.
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


#: The capabilities a law may claim, which a report of a marginal's claims ranges over.
_LAW_CAPABILITIES: tuple[type, ...] = (
    *_CONDITIONAL_TWINS,
    SupportsExactConditioning,
    SupportsApproximateConditioning,
)


def _claims(law: Any) -> frozenset[type]:
    """The capabilities among :data:`_LAW_CAPABILITIES` that *law* claims."""
    return frozenset(protocol for protocol in _LAW_CAPABILITIES if isinstance(law, protocol))


def _kernel_claims(kernel: Any) -> frozenset[type]:
    """Each unconditional capability whose conditional twin *kernel* claims.

    They are the claims of the law the kernel yields at a given value.
    """
    return frozenset(
        protocol for protocol, twin in _CONDITIONAL_TWINS.items() if isinstance(kernel, twin)
    )


def _marginal_claims(law: Any, path: str | tuple[str, ...]) -> frozenset[type]:
    """The capabilities *law*'s exact marginal at *path* claims, as *law* reports them.

    A law that defines ``_marginal_capabilities(path)`` reports them from its
    declarations, and the marginals of any other law claim what the law claims
    itself.
    """
    report = getattr(law, "_marginal_capabilities", None)
    if report is None:
        return _claims(law)
    return frozenset(report(path))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

#: The capabilities whose answer presupposes a probability law: a density that
#: integrates to one, draws, which determine the law, and integrals against it.
_NORMALIZING_CAPABILITIES: tuple[type, ...] = (
    SupportsLogProb,
    SupportsSampling,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsQuantile,
    SupportsExpectation,
)


def _is_normalized(law: Any) -> bool:
    """Whether *law* is normalized: it claims a capability that presupposes a probability law.

    The capabilities are ``SupportsLogProb``, ``SupportsSampling``, and the
    moment, quantile, and expectation capabilities. A law that claims none of
    them is unnormalized, since no capability it claims fixes its normalizing
    constant. The classification reads protocol membership alone, so it calls
    no capability and reads no guard.
    """
    return any(isinstance(law, capability) for capability in _NORMALIZING_CAPABILITIES)


def _kernel_is_normalized(kernel: Any) -> bool:
    """Whether *kernel*'s laws are normalized: it claims the twin of a normalizing capability.

    The twins are those of the capabilities :func:`_is_normalized` reads, and
    the classification reads protocol membership alone, as that one does.
    """
    return any(
        isinstance(kernel, _CONDITIONAL_TWINS[capability])
        for capability in _NORMALIZING_CAPABILITIES
    )


# ---------------------------------------------------------------------------
# Guards and per-instance capability sets
# ---------------------------------------------------------------------------


#: The protocol defaults that call another capability, whose guard applies to them.
_DELEGATING_DEFAULTS: dict[Callable[..., Any], str] = {
    SupportsLogProb._unnormalized_log_prob: "_log_prob",
    SupportsConditionalLogProb._conditional_unnormalized_log_prob: "_conditional_log_prob",
}

_GUARD_SUFFIX = "_guard"


def _capability_guard(term: Any, method: str, *arguments: Any, **keywords: Any) -> Feasibility:
    """The guard of *term*'s capability *method* for one call.

    Parameters
    ----------
    term : Any
        The object claiming the capability.
    method : str
        The capability's method name, such as ``"_marginal"``.
    *arguments : Any
        The call's positional arguments other than raw values, such as a marginal's
        path, which the guard receives in order.
    **keywords : Any
        The call's keyword arguments other than raw values, which the guard receives
        by name.

    Returns
    -------
    Feasibility
        The report of ``term._<method>_guard(*arguments, **keywords)``. A
        ``Feasibility`` is returned as the guard gave it, and a ``bool`` or
        ``None`` becomes a report whose reason names the capability, its
        arguments, *term*'s class, and the guard's condition. Without a guard the report is feasible, since the
        capability is total on its declared domain, unless *method* is a
        protocol default that calls another capability, whose guard then applies.

    Raises
    ------
    AttributeError
        If *term* does not implement *method*.
    TypeError
        If the guard returns anything but a ``bool``, ``None``, or a ``Feasibility``.
    """
    if not callable(getattr(term, method, None)):
        raise AttributeError(f"{type(term).__name__} does not implement {method}")
    guard = getattr(term, f"{method}{_GUARD_SUFFIX}", None)
    if guard is None:
        delegate = _DELEGATING_DEFAULTS.get(getattr(type(term), method, None))
        if delegate is not None:
            return _capability_guard(term, delegate, *arguments, **keywords)
        return Feasibility(True)
    report = guard(*arguments, **keywords)
    if isinstance(report, Feasibility):
        return report
    if report is True:
        return Feasibility(True)
    rendered = ", ".join(
        [reprlib.repr(argument) for argument in arguments]
        + [f"{name}={reprlib.repr(value)}" for name, value in keywords.items()]
    )
    requested = f"{method.lstrip('_')}({rendered})"
    subject = public_class_name(type(term))
    condition = _requirement(guard)
    suffix = f"; it requires: {condition}" if condition else ""
    if report is False:
        return Feasibility(False, f"{requested} is not available for {subject}{suffix}")
    if report is None:
        return Feasibility(
            None,
            pending=(
                f"whether {requested} is available for {subject} depends on values not yet "
                f"known{suffix}",
            ),
        )
    call = f"{type(term).__name__}.{method}{_GUARD_SUFFIX}({rendered})"
    raise TypeError(f"{call} returned {report!r}; a guard returns a bool, None, or a Feasibility")


def _guard_condition(guard: Callable[..., Any]) -> str:
    """The condition *guard* states: its docstring's first paragraph on one line, or ``""``."""
    doc = inspect.getdoc(guard)
    if not doc:
        return ""
    return " ".join(doc.split("\n\n", 1)[0].split())


def _requirement(check: Callable[..., Any]) -> str:
    """The condition *check* states, as a message quotes it, or ``""``.

    It is :func:`_guard_condition` with its first letter in lowercase, unless
    the condition starts with an identifier, and without a final period.
    """
    condition = _guard_condition(check).rstrip(".")
    word = condition.split(" ", 1)[0]
    if word[:1].isupper() and word[1:] == word[1:].lower():
        condition = condition[0].lower() + condition[1:]
    return condition


def _conjunction(reports: Iterable[Feasibility]) -> Feasibility:
    """The report of a call that needs every one of *reports* to be feasible.

    The first infeasible report is returned. Otherwise the result is unresolved
    with every pending entry when any report is unresolved, and feasible when
    none is.
    """
    pending: list[str] = []
    for report in reports:
        if report.feasible is False:
            return report
        pending.extend(report.pending)
    return Feasibility(None, pending=tuple(pending)) if pending else Feasibility(True)


def _implements(cls: type, method: str) -> bool:
    """Whether *cls* implements *method*, not only declares it.

    The first class in the MRO that defines *method* decides. A protocol's
    declaration and an abstract method are not implementations, while a
    protocol default that calls another capability is.
    """
    for klass in cls.__mro__:
        value = vars(klass).get(method)
        if value is None:
            continue
        if getattr(value, "__isabstractmethod__", False):
            return False
        if vars(klass).get("_is_protocol", False) and value not in _DELEGATING_DEFAULTS:
            return False
        return callable(value) or isinstance(value, (staticmethod, classmethod))
    return False


def _check_guards(cls: type) -> None:
    """Check that each guard in the body of *cls* guards a method *cls* implements.

    The metaclasses of ``Distribution`` and ``ConditionalDistribution`` run this
    check on each class they create, so a misspelled guard raises rather than
    going unread. A guard set to ``None`` removes an inherited one.

    Parameters
    ----------
    cls : type
        The class just created, whose own namespace ``vars(cls)`` holds the guards
        to check.

    Raises
    ------
    TypeError
        If a ``_<method>_guard`` in the class body is not callable, or *cls*
        does not implement ``_<method>``.
    """
    for name, guard in vars(cls).items():
        method = name.removesuffix(_GUARD_SUFFIX)
        if method == name or len(method) < 2 or not method.startswith("_"):
            continue
        if method.startswith("__") or guard is None:
            continue
        if not callable(guard) and not isinstance(guard, (staticmethod, classmethod)):
            raise TypeError(f"{cls.__name__}.{name} must be a method that guards {method}")
        if _implements(cls, method):
            continue
        implemented = [
            member
            for member in dir(cls)
            if member.startswith("_")
            and not member.startswith("__")
            and not member.endswith(_GUARD_SUFFIX)
            and _implements(cls, member)
        ]
        close = difflib.get_close_matches(method, implemented, n=1)
        hint = f"; did you mean {close[0]}{_GUARD_SUFFIX}?" if close else ""
        raise TypeError(
            f"{cls.__name__}.{name} guards {method}, which {cls.__name__} does not implement{hint}"
        )


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

    Parameters
    ----------
    base : type
        The class to subclass, whose instances differ in the capabilities they
        claim.
    protocols : iterable of type
        The capability protocols to claim, which are keys of the base's
        ``_capability_table``.

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
