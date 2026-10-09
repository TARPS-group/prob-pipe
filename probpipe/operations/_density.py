"""The density operations: log-densities, densities, and random log-densities.

``log_prob(d, value)`` requires ``SupportsLogProb`` and returns the normalized
log-density, and ``unnormalized_log_prob`` requires only
``SupportsUnnormalizedLogProb`` and returns it up to an additive constant. The
value conforms to the law's event declaration, its packaging included. A batch
of values is swept, and the scores keep the batch's levels: the law scores every
element in one vectorized call when its density traces, and one element at a
time otherwise. ``prob`` and ``unnormalized_prob`` are derived operations,
defined by exponentiating the matching log-density. ``random_log_prob(M)`` and
``random_unnormalized_log_prob(M)`` return the law of a random measure's
log-density function.

A score of a law ``d`` is labeled ``log`` followed by ``d``'s notation, as
``log prior(mu)`` or ``log model(mu; y)``, and its component is the
operation's call over ``d``'s components, as ``log_prob(mu)`` or
``log_prob(y, mu)``. A density is labeled by ``d``'s notation, as
``prior(mu)``, since that notation denotes the density, and its component is
``prob(mu)``. A batch of scores takes the label of one score.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..core._expression import DENSITY, SCORE, Expression, Summary, embedded
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
from ..functions._call import ApplicabilityError, CallReport
from ._operation import _install_expression_rule, operation

__all__ = [
    "log_prob",
    "prob",
    "random_log_prob",
    "random_unnormalized_log_prob",
    "unnormalized_log_prob",
    "unnormalized_prob",
]


def _score_declaration(
    d: DistributionSpec, value: TermSpec, operation_name: str, spec: NumericArraySpec
) -> OutputSpec:
    """The declaration of the score of one *value*: *spec* under the score's call over the components.

    The component is the operation's call over the law's event components, as
    ``log_prob(mu)`` or ``log_prob(y, mu)``, as VI.0 names a summary of
    several components. A batch of values is swept, so the rule reads one
    element's spec.

    Parameters
    ----------
    d : DistributionSpec
        The law's spec, against whose event declaration *value* is unified.
    value : TermSpec
        The spec of the value scored, or of one draw where a law at the value
        lifts the call.
    operation_name : str
        The operation's name, which the component calls and each error message
        opens with.
    spec : NumericArraySpec
        The term spec of one score.

    Returns
    -------
    OutputSpec
        A declaration that names the score as one whole term.

    Raises
    ------
    ApplicabilityError
        If the value does not conform to the law's event declaration,
        packaging included.
    """
    try:
        # The bindings are local to this call, so a polymorphic law scores
        # values of any size.
        _unify_specs(d.event_spec.spec, value, {}, f"{operation_name} value")
    except (TypeError, ValueError) as error:
        raise ApplicabilityError(
            f"{operation_name}: the value does not conform to the event declaration: {error}"
        ) from None
    component = f"{operation_name}({', '.join(d.event_spec.components)})"
    return OutputSpec(**{component: spec})


def _score_expression(d: Any) -> Expression:
    """The expression of a score of *d*: ``log`` and *d*'s notation, as ``log prior(mu)`` (II.4)."""
    return Summary(SCORE, embedded(d))


def _density_expression(d: Any) -> Expression:
    """The expression of a density of *d*, which reads as *d*'s notation, as ``prior(mu)`` (II.4)."""
    return Summary(DENSITY, embedded(d))


def _random_score_expression(M: Any) -> Expression:
    """The expression of the law of a random measure's log-density: ``log`` and *M*'s notation."""
    return Summary(SCORE, embedded(M))


def _log_prob_result(d: DistributionSpec, value: TermSpec) -> OutputSpec:
    """A scalar log-density per value, under the component ``log_prob(...)`` of the law's components.

    Parameters
    ----------
    d : DistributionSpec
        The law's spec, whose event declaration the value conforms to.
    value : TermSpec
        The spec of one value: for a lifted argument, the spec of one element
        of the batch or one draw of the law.

    Returns
    -------
    OutputSpec
        The declaration of one scalar under the component that calls
        ``log_prob`` over the law's components, as ``log_prob(mu)``.

    Raises
    ------
    ApplicabilityError
        If the value does not conform to the event declaration.
    """
    return _score_declaration(d, value, "log_prob", NumericArraySpec(()))


def _unnormalized_log_prob_result(d: DistributionSpec, value: TermSpec) -> OutputSpec:
    """A scalar unnormalized log-density per value, under ``unnormalized_log_prob(...)``.

    Parameters
    ----------
    d : DistributionSpec
        The law's spec, whose event declaration the value conforms to.
    value : TermSpec
        The spec of one value: for a lifted argument, the spec of one element
        of the batch or one draw of the law.

    Returns
    -------
    OutputSpec
        The declaration of one scalar under the component that calls
        ``unnormalized_log_prob`` over the law's components, as
        ``unnormalized_log_prob(mu)``.

    Raises
    ------
    ApplicabilityError
        If the value does not conform to the event declaration.
    """
    return _score_declaration(d, value, "unnormalized_log_prob", NumericArraySpec(()))


def _prob_result(d: DistributionSpec, value: TermSpec) -> OutputSpec:
    """A non-negative scalar density per value, under the component ``prob(...)``.

    Parameters
    ----------
    d : DistributionSpec
        The law's spec, whose event declaration the value conforms to.
    value : TermSpec
        The spec of one value: for a lifted argument, the spec of one element
        of the batch or one draw of the law.

    Returns
    -------
    OutputSpec
        The declaration of one non-negative scalar under the component that
        calls ``prob`` over the law's components, as ``prob(mu)``.

    Raises
    ------
    ApplicabilityError
        If the value does not conform to the event declaration.
    """
    return _score_declaration(d, value, "prob", NumericArraySpec((), support=non_negative))


def _unnormalized_prob_result(d: DistributionSpec, value: TermSpec) -> OutputSpec:
    """A non-negative scalar unnormalized density per value, under ``unnormalized_prob(...)``.

    Parameters
    ----------
    d : DistributionSpec
        The law's spec, whose event declaration the value conforms to.
    value : TermSpec
        The spec of one value: for a lifted argument, the spec of one element
        of the batch or one draw of the law.

    Returns
    -------
    OutputSpec
        The declaration of one non-negative scalar under the component that
        calls ``unnormalized_prob`` over the law's components, as
        ``unnormalized_prob(mu)``.

    Raises
    ------
    ApplicabilityError
        If the value does not conform to the event declaration.
    """
    return _score_declaration(
        d, value, "unnormalized_prob", NumericArraySpec((), support=non_negative)
    )


def _random_log_prob_result(M: DistributionSpec) -> None:
    """None: the returned law over log-density functions carries its declaration."""
    return None


def _random_unnormalized_log_prob_result(M: DistributionSpec) -> None:
    """None: the returned law over unnormalized log-density functions carries its declaration."""
    return None


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
        The log-density, or one per element of a batch value, at its levels,
        labeled ``log`` and *d*'s notation, as ``log prior(mu)``, under the
        component ``log_prob(mu)``.

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
        The unnormalized log-density, or one per element of a batch value,
        labeled ``log`` and *d*'s notation, as ``log prior(mu)``, under the
        component ``unnormalized_log_prob(mu)``.

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


def _log_prob_applies(d: Any, value: Any) -> CallReport:
    """``log_prob`` has a route for the law and the value."""
    return log_prob.check(d, value)


def _unnormalized_log_prob_applies(d: Any, value: Any) -> CallReport:
    """``unnormalized_log_prob`` has a route for the law and the value."""
    return unnormalized_log_prob.check(d, value)


@operation(result=_prob_result, identity_check=_log_prob_applies)
def prob(d: Distribution, value):
    """The density of *value* under *d*, defined as ``exp ∘ log_prob``.

    Parameters
    ----------
    d : Distribution
        A law claiming ``SupportsLogProb``, or any other law that ``log_prob``
        has a route for.
    value : Any
        One value conforming to ``d.event_spec.spec``, or a batch of them. A law
        over such values lifts the call, which then returns the law of the
        density at its draws.

    Returns
    -------
    NumericArray or NumericArrayBatch
        The density, or one per element of a batch value, labeled by *d*'s
        notation, as ``prior(mu)``, under the component ``prob(mu)``.

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

    Parameters
    ----------
    d : Distribution
        A law claiming ``SupportsUnnormalizedLogProb``, or any other law that
        ``unnormalized_log_prob`` has a route for.
    value : Any
        One value conforming to ``d.event_spec.spec``, or a batch of them. A law
        over such values lifts the call, which then returns the law of the
        unnormalized density at its draws.

    Returns
    -------
    NumericArray or NumericArrayBatch
        The unnormalized density, or one per element of a batch value, labeled
        by *d*'s notation, as ``prior(mu)``, under the component
        ``unnormalized_prob(mu)``.

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

    Parameters
    ----------
    M : Distribution
        A random measure, which is a law whose draws ``D`` are distributions.

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

    Parameters
    ----------
    M : Distribution
        A random measure, which is a law whose draws ``D`` are distributions.

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


_install_expression_rule(log_prob, _score_expression)
_install_expression_rule(unnormalized_log_prob, _score_expression)
_install_expression_rule(prob, _density_expression)
_install_expression_rule(unnormalized_prob, _density_expression)
_install_expression_rule(random_log_prob, _random_score_expression)
_install_expression_rule(random_unnormalized_log_prob, _random_score_expression)
