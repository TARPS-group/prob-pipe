"""The density operations: log-densities, densities, and random log-densities.

``log_prob(d, value)`` requires ``SupportsLogProb`` and returns the normalized
log-density, and ``unnormalized_log_prob`` requires only
``SupportsUnnormalizedLogProb`` and returns it up to an additive constant. The
value conforms to the law's event declaration, its packaging included, and a
batch of values is scored at once, keeping its levels. ``prob`` and
``unnormalized_prob`` are derived operations, defined by exponentiating the
matching log-density. ``random_log_prob(M)`` and
``random_unnormalized_log_prob(M)`` return the law of a random measure's
log-density function.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..core._batch import BatchSpec
from ..core._spec_base import NumericArraySpec, TermSpec, _unify_specs
from ..core._specs import OutputSpec
from ..core.constraints import non_negative
from ..distributions._capabilities import (
    SupportsLogProb,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsUnnormalizedLogProb,
)
from ..distributions._distribution import Distribution, DistributionSpec
from ._operation import ApplicabilityError, CallCheck, operation

__all__ = [
    "log_prob",
    "prob",
    "random_log_prob",
    "random_unnormalized_log_prob",
    "unnormalized_log_prob",
    "unnormalized_prob",
]


def _score_declaration(
    d: DistributionSpec, value: TermSpec, component: str, spec: NumericArraySpec
) -> OutputSpec:
    """The declaration of a score of *value*: *spec*, or a batch of it at a batch value's levels.

    Raises
    ------
    ApplicabilityError
        If the value, or a batch value's element, does not conform to the law's
        event declaration, packaging included.
    """
    scored = value.element_spec if isinstance(value, BatchSpec) else value
    try:
        # The bindings are local to this call, so a polymorphic law scores
        # values of any size.
        _unify_specs(d.event_spec.spec, scored, {}, f"{component} value")
    except (TypeError, ValueError) as error:
        raise ApplicabilityError(
            f"{component}: the value does not conform to the event declaration: {error}"
        ) from None
    if isinstance(value, BatchSpec):
        return OutputSpec(**{component: BatchSpec(spec, value.axis_groups, value.level_names)})
    return OutputSpec(**{component: spec})


def _log_prob_result(d: DistributionSpec, value: TermSpec) -> OutputSpec:
    """A scalar log-density per value, under the component ``log_prob``.

    Raises
    ------
    ApplicabilityError
        If the value does not conform to the event declaration.
    """
    return _score_declaration(d, value, "log_prob", NumericArraySpec(()))


def _unnormalized_log_prob_result(d: DistributionSpec, value: TermSpec) -> OutputSpec:
    """A scalar unnormalized log-density per value, under ``unnormalized_log_prob``.

    Raises
    ------
    ApplicabilityError
        If the value does not conform to the event declaration.
    """
    return _score_declaration(d, value, "unnormalized_log_prob", NumericArraySpec(()))


def _prob_result(d: DistributionSpec, value: TermSpec) -> OutputSpec:
    """A non-negative scalar density per value, under the component ``prob``.

    Raises
    ------
    ApplicabilityError
        If the value does not conform to the event declaration.
    """
    return _score_declaration(d, value, "prob", NumericArraySpec((), support=non_negative))


def _unnormalized_prob_result(d: DistributionSpec, value: TermSpec) -> OutputSpec:
    """A non-negative scalar unnormalized density per value, under ``unnormalized_prob``.

    Raises
    ------
    ApplicabilityError
        If the value does not conform to the event declaration.
    """
    return _score_declaration(
        d, value, "unnormalized_prob", NumericArraySpec((), support=non_negative)
    )


def _random_log_prob_result(M: DistributionSpec) -> OutputSpec:
    """A law over log-density functions, whose declaration the returned law fills."""
    return OutputSpec(random_log_prob=None)


def _random_unnormalized_log_prob_result(M: DistributionSpec) -> OutputSpec:
    """A law over unnormalized log-density functions, whose declaration the returned law fills."""
    return OutputSpec(random_unnormalized_log_prob=None)


@operation(result=_log_prob_result)
def log_prob(d: Distribution, value):
    """The normalized log-density of *value* under *d*.

    Parameters
    ----------
    d : Distribution
        A law claiming ``SupportsLogProb``.
    value : Any
        One value conforming to ``d.event_spec.spec``, or a batch of them; a
        scored value binds the law's symbolic dimensions for this call only. A
        law over such values lifts the call, which then returns the law of the
        log-density at its draws.

    Returns
    -------
    NumericArray or NumericArrayBatch
        The log-density, or one per element of a batch value, at its levels.

    Raises
    ------
    ApplicabilityError
        If *value* does not conform to the event declaration.
    ResolutionError
        If *d* does not claim ``SupportsLogProb``.
    """


log_prob.capability_route(
    "exact", operand="d", protocol=SupportsLogProb, method="_log_prob", exact=True
)


@operation(result=_unnormalized_log_prob_result)
def unnormalized_log_prob(d: Distribution, value):
    """The log-density of *value* under *d* up to an additive constant.

    Parameters
    ----------
    d : Distribution
        A law claiming ``SupportsUnnormalizedLogProb``.
    value : Any
        One value conforming to ``d.event_spec.spec``, or a batch of them. A law
        over such values lifts the call, which then returns the law of the
        log-density at its draws.

    Returns
    -------
    NumericArray or NumericArrayBatch
        The unnormalized log-density, or one per element of a batch value.

    Raises
    ------
    ApplicabilityError
        If *value* does not conform to the event declaration.
    ResolutionError
        If *d* does not claim ``SupportsUnnormalizedLogProb``.
    """


unnormalized_log_prob.capability_route(
    "exact",
    operand="d",
    protocol=SupportsUnnormalizedLogProb,
    method="_unnormalized_log_prob",
    exact=True,
)


def _log_prob_applies(d: Any, value: Any) -> CallCheck:
    """``log_prob`` has a route for the law and the value."""
    return log_prob.check(d, value)


def _unnormalized_log_prob_applies(d: Any, value: Any) -> CallCheck:
    """``unnormalized_log_prob`` has a route for the law and the value."""
    return unnormalized_log_prob.check(d, value)


@operation(result=_prob_result, identity_check=_log_prob_applies)
def prob(d: Distribution, value):
    """The density of *value* under *d*, defined as ``exp ∘ log_prob``.

    Returns
    -------
    NumericArray or NumericArrayBatch
        The density, or one per element of a batch value.

    Raises
    ------
    ApplicabilityError
        If *value* does not conform to the event declaration.
    ResolutionError
        If ``log_prob`` has no route for *d*.
    """
    return jnp.exp(log_prob.with_options(raw=True)(d, value))


@operation(result=_unnormalized_prob_result, identity_check=_unnormalized_log_prob_applies)
def unnormalized_prob(d: Distribution, value):
    """The density of *value* under *d* up to a factor, defined as ``exp ∘ unnormalized_log_prob``.

    Returns
    -------
    NumericArray or NumericArrayBatch
        The unnormalized density, or one per element of a batch value.

    Raises
    ------
    ApplicabilityError
        If *value* does not conform to the event declaration.
    ResolutionError
        If ``unnormalized_log_prob`` has no route for *d*.
    """
    return jnp.exp(unnormalized_log_prob.with_options(raw=True)(d, value))


@operation(result=_random_log_prob_result)
def random_log_prob(M: Distribution):
    """The law of ``x ↦ log D(x)`` for ``D ~ M``, a random function.

    The density at a point is that random function called at the point, so the
    operation takes no value.

    Returns
    -------
    Distribution
        A random function over log-densities.

    Raises
    ------
    ResolutionError
        If *M* does not claim ``SupportsRandomLogProb``.
    """


random_log_prob.capability_route(
    "exact", operand="M", protocol=SupportsRandomLogProb, method="_random_log_prob", exact=True
)


@operation(result=_random_unnormalized_log_prob_result)
def random_unnormalized_log_prob(M: Distribution):
    """The law of ``x ↦ log D̃(x)`` for ``D ~ M``, with ``D̃`` the unnormalized density of ``D``.

    Returns
    -------
    Distribution
        A random function over unnormalized log-densities.

    Raises
    ------
    ResolutionError
        If *M* does not claim ``SupportsRandomUnnormalizedLogProb``.
    """


random_unnormalized_log_prob.capability_route(
    "exact",
    operand="M",
    protocol=SupportsRandomUnnormalizedLogProb,
    method="_random_unnormalized_log_prob",
    exact=True,
)
