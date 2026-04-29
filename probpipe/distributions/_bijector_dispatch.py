"""Constraint → Bijector dispatch for unconstrained reparameterization.

Provides :func:`bijector_for`, a registry-backed lookup that returns a
canonical :class:`tfb.Bijector` mapping ℝⁿ to the support described by a
:class:`~probpipe.core.constraints.Constraint`, and
:func:`register_bijector` for plugging in custom factories.

Usage::

    from probpipe import bijector_for, interval

    bij = bijector_for(interval(2.0, 5.0))
    y = bij.forward(jnp.linspace(-3, 3, 11))
    assert jnp.all((y > 2.0) & (y < 5.0))

External code can register custom factories::

    register_bijector(MyConstraint, lambda c: MyBijector(c.param))
    register_bijector(my_singleton_constraint, lambda c: AlternateBijector())

Instance registrations override type registrations.

This module intentionally lives at the ``distributions/`` layer rather
than on :class:`Constraint` itself: ``probpipe/core/`` is
backend-agnostic and has no TFP imports today, while
``probpipe/distributions/`` already pairs Constraints with TFP
bijectors (see :data:`_BIJECTOR_SUPPORT_MAP` in
:mod:`probpipe.distributions.transformed`).

The forward direction (bijector → support) in ``_BIJECTOR_SUPPORT_MAP``
and the inverse direction (support → bijector) implemented here are
**not** strict inverses of each other.  In particular,
``TransformedDistribution(base, bijector_for(c)).support == c`` holds
only for ``real``, ``positive``, and ``unit_interval`` (the cases where
the canonical bijector is unparameterized and is in the forward map).
For ``non_negative`` (Softplus → ``positive``), ``interval(low, high)``
(parameterized Sigmoid → ``unit_interval``), ``simplex`` and
``positive_definite`` (Chain → not in forward map → ``real``), and
``greater_than`` (Chain → ``real``), the round-trip drifts to a
coarser support.  The two maps answer different questions with
different reliability tiers; see the issue tracker for proposals to
unify them.
"""

from __future__ import annotations

from collections.abc import Callable

import tensorflow_probability.substrates.jax.bijectors as tfb

from ..core.constraints import (
    Constraint,
    _Boolean,
    _GreaterThan,
    _IntegerInterval,
    _Interval,
    _NonNegative,
    _NonNegativeInteger,
    _Positive,
    _PositiveDefinite,
    _Real,
    _Simplex,
    _Sphere,
    _UnitInterval,
)

__all__ = ["bijector_for", "register_bijector"]


# Factory takes the (parameterized) constraint instance and returns a bijector.
BijectorFactory = Callable[[Constraint], tfb.Bijector]

# Registry. Keys may be Constraint subclasses (type-keyed default) or
# specific Constraint instances (instance-keyed override).  Instance
# lookup wins over type lookup; this matches PyTorch's
# ``constraint_registry`` precedence.
_CONSTRAINT_BIJECTOR_REGISTRY: dict[type | Constraint, BijectorFactory] = {}


def register_bijector(
    key: type | Constraint,
    factory: BijectorFactory,
) -> None:
    """Register a bijector factory for a Constraint type or singleton.

    Parameters
    ----------
    key : type or Constraint
        Either a :class:`Constraint` subclass (applies to all instances of
        that type, including parameterized ones) or a specific
        :class:`Constraint` value (applies to constraints equal to it).
        Instance keys take precedence over type keys at lookup time.
    factory : callable
        ``factory(constraint) -> tfb.Bijector``.  The constraint instance
        is passed in so the factory can read parameters (e.g., ``low`` /
        ``high`` from an :class:`_Interval`).

    Notes
    -----
    Re-registering an existing key silently overwrites the previous
    factory.

    Avoid registering against the base :class:`Constraint` class itself:
    every constraint shares it in their MRO, so a base-class registration
    would catch every unmatched constraint.
    """
    _CONSTRAINT_BIJECTOR_REGISTRY[key] = factory


def bijector_for(constraint: Constraint) -> tfb.Bijector:
    """Return a canonical bijector mapping ℝⁿ to *constraint*'s support.

    Lookup precedence:

    1. Exact instance match (e.g., the singleton ``positive``).
    2. Type match, walking the Constraint MRO (most-specific first).
    3. :class:`NotImplementedError` if nothing matches.

    Parameters
    ----------
    constraint : Constraint
        The target support.

    Returns
    -------
    tfb.Bijector
        A bijector whose forward image lies in *constraint*'s support.

    Raises
    ------
    NotImplementedError
        If no factory is registered for *constraint*'s type, or if the
        constraint is one for which no smooth bijector exists (discrete
        constraints, the unit sphere).
    """
    # 1. Instance match.  ``Constraint.__hash__`` hashes
    # ``(type, sorted-__dict__-items)``, which raises ``TypeError`` when
    # ``__dict__`` contains an unhashable value (e.g., a JAX array as
    # ``low``); catch that and fall through to type lookup.
    try:
        if constraint in _CONSTRAINT_BIJECTOR_REGISTRY:
            return _CONSTRAINT_BIJECTOR_REGISTRY[constraint](constraint)
    except TypeError:
        pass

    # 2. Type match via MRO.
    for cls in type(constraint).__mro__:
        if cls in _CONSTRAINT_BIJECTOR_REGISTRY:
            return _CONSTRAINT_BIJECTOR_REGISTRY[cls](constraint)

    raise NotImplementedError(
        f"No bijector registered for {constraint!r}. "
        f"Use ``probpipe.register_bijector`` to add one."
    )


# ---------------------------------------------------------------------------
# Default registrations
# ---------------------------------------------------------------------------

# Continuous, smooth, well-defined.  Choices follow Pyro / NumPyro
# conventions: ``Exp`` for ``positive`` (matches ProbPipe's existing
# ``_BIJECTOR_SUPPORT_MAP[Exp] = positive`` round-trip),
# ``Softplus`` for ``non_negative`` (well-defined at zero),
# ``Sigmoid`` for bounded intervals.
register_bijector(_Real, lambda c: tfb.Identity())
register_bijector(_Positive, lambda c: tfb.Exp())
register_bijector(_NonNegative, lambda c: tfb.Softplus())
register_bijector(_UnitInterval, lambda c: tfb.Sigmoid())
register_bijector(_Interval, lambda c: tfb.Sigmoid(low=c.low, high=c.high))
register_bijector(
    _GreaterThan,
    lambda c: tfb.Chain([tfb.Shift(c.lower_bound), tfb.Exp()]),
)
register_bijector(_Simplex, lambda c: tfb.SoftmaxCentered())
# x ∈ ℝ^{n(n+1)/2} → L lower-triangular → L Lᵀ ∈ SPD.
# TFP ``Chain`` applies ``bijectors[-1]`` first and ``bijectors[0]`` last.
register_bijector(
    _PositiveDefinite,
    lambda c: tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()]),
)


# Constraints with no canonical smooth bijector.  Register an explicit
# raiser so the error message names the reason rather than a generic
# "not registered".
def _no_smooth_bijector(reason: str) -> BijectorFactory:
    def _raise(c: Constraint) -> tfb.Bijector:
        raise NotImplementedError(
            f"{c!r} has no canonical bijector: {reason}"
        )
    return _raise


register_bijector(
    _Sphere,
    _no_smooth_bijector(
        "no smooth global chart from ℝⁿ to the unit sphere; "
        "consider a stereographic projection or restrict to a single chart"
    ),
)
register_bijector(
    _Boolean,
    _no_smooth_bijector(
        "discrete support; consider a continuous relaxation "
        "(e.g., Gumbel-Sigmoid) for gradient-based optimization"
    ),
)
register_bijector(
    _NonNegativeInteger,
    _no_smooth_bijector(
        "discrete support; no continuous bijector exists"
    ),
)
register_bijector(
    _IntegerInterval,
    _no_smooth_bijector(
        "discrete support; consider a continuous relaxation "
        "(e.g., Gumbel-Softmax) or rounding"
    ),
)
