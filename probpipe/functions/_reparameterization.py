"""The constraint-to-bijector factory: a map from ℝⁿ onto each constrained support.

Gradient-based inference runs on an unconstrained space, so a constrained
support is reparameterized by a **bijector**: a ``Function`` that takes ℝⁿ onto
the support and claims ``SupportsInverse`` and ``SupportsLogDetJacobian``.
:func:`bijector_for` returns the canonical one for a constraint, and
:func:`register_bijector` plugs in a factory for a constraint type or for one
constraint, an instance registration taking precedence over its type's.

A factory returns a bijector ``Function``, or a backend bijector, which enters
as a ``Function`` whose forward map is the backend's and which claims the
inverse and the log-determinant of the Jacobian the backend computes.

Provides:
  - :func:`bijector_for` – the canonical bijector onto a constraint's support.
  - :func:`register_bijector` – a factory for a constraint type or instance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb

from ..core._dispatch import MathematicalDomainError, ResolutionError
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
    greater_than,
    interval,
    positive,
    real,
    unit_interval,
)
from ..custom_types import Array, ArrayLike
from ..values import Function, SupportsInverse, SupportsLogDetJacobian, is_invertible

__all__ = ["bijector_for", "register_bijector"]

#: A factory takes the constraint, whose parameters it may read, and returns a bijector.
BijectorFactory = Callable[[Constraint], "Function | tfb.Bijector"]

#: The registered factories, keyed by a constraint type or by one constraint.
_CONSTRAINT_BIJECTOR_REGISTRY: dict[type | Constraint, BijectorFactory] = {}

#: The support each backend bijector without parameters maps the real line onto.
_BACKEND_IMAGES: dict[type, Constraint] = {
    tfb.Exp: positive,
    tfb.Square: positive,
}


def _backend_image(bijector: tfb.Bijector) -> Constraint | None:
    """The support *bijector* maps the real line onto, or ``None`` unless it is known exactly.

    A chain applies its first bijector last. Its image is that bijector's image
    when every bijector applied before it is affine, and so maps the line onto
    itself; otherwise the image is a subset this module does not compute.
    """
    if isinstance(bijector, tfb.Chain):
        if not bijector.bijectors:
            return real
        outermost, *inner = bijector.bijectors
        if all(_backend_is_affine(part) for part in inner):
            return _backend_image(outermost)
        return None
    if isinstance(bijector, tfb.Sigmoid):
        if bijector.low is None and bijector.high is None:
            return unit_interval
        return interval(bijector.low, bijector.high)
    if isinstance(bijector, tfb.Softplus):
        return positive if bijector.low is None else greater_than(bijector.low)
    if _backend_is_affine(bijector):
        return real
    return _BACKEND_IMAGES.get(type(bijector))


#: The backend bijectors whose forward map is affine, ``x ↦ A x + b``.
_AFFINE_BACKENDS: tuple[type, ...] = (
    tfb.Identity,
    tfb.Shift,
    tfb.Scale,
    tfb.ScaleMatvecDiag,
    tfb.ScaleMatvecTriL,
    tfb.ScaleMatvecLU,
    tfb.ScaleMatvecLinearOperator,
)


def _backend_is_affine(bijector: tfb.Bijector) -> bool:
    """Whether *bijector*'s forward map is affine: an affine backend, or a chain of them."""
    if isinstance(bijector, tfb.Chain):
        return all(_backend_is_affine(part) for part in bijector.bijectors)
    return isinstance(bijector, _AFFINE_BACKENDS)


def _image(bijector: Function) -> Constraint | None:
    """The support onto which *bijector* maps the real line, when it is recorded."""
    return getattr(bijector, "_image", None)


def _is_affine(bijector: Function) -> bool:
    """Whether *bijector*'s forward map is recorded as affine, so moments push through it."""
    return getattr(bijector, "_affine", False)


class _ForwardMap:
    """The forward map of a backend bijector at one point."""

    def __init__(self, bijector: tfb.Bijector) -> None:
        self._bijector = bijector

    def __call__(self, x: ArrayLike) -> Array:
        return self._bijector.forward(jnp.asarray(x))


class _BackendBijector(Function, SupportsInverse, SupportsLogDetJacobian):
    """A backend bijector entered as a ``Function``.

    Its forward map is the backend's forward transform, its inverse the
    backend's inverse, and its log-Jacobian the backend's log-determinant at
    one point, the whole array being the point. It records the support it maps
    onto when that is known, and whether its forward map is affine.
    """

    def __init__(self, bijector: tfb.Bijector, image: Constraint | None = None) -> None:
        super().__init__(bijector.name, _ForwardMap(bijector))
        object.__setattr__(self, "_bijector", bijector)
        object.__setattr__(self, "_image", image)
        object.__setattr__(self, "_affine", _backend_is_affine(bijector))

    def _inverse(self, y: ArrayLike) -> Array:
        """The point the forward map sends to *y*."""
        return self._bijector.inverse(jnp.asarray(y))

    def _log_det_jacobian(self, x: ArrayLike) -> Array:
        """``log |det J(x)|`` of the forward map at the point *x*."""
        x = jnp.asarray(x)
        return self._bijector.forward_log_det_jacobian(x, event_ndims=x.ndim)

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild from the backend bijector, whose state pickles where a Function's controls do not."""
        return (_rebuilt_backend_bijector, (self._bijector, self._image, self.label))


def _rebuilt_backend_bijector(
    bijector: tfb.Bijector, image: Constraint | None, name: str
) -> _BackendBijector:
    """The backend-bijector Function of *bijector* under the label *name*, for unpickling."""
    rebuilt = _BackendBijector(bijector, image)
    return rebuilt if rebuilt.label == name else rebuilt.with_label(name)


def _as_bijector(value: Any, image: Constraint | None = None) -> Function:
    """*value* as a bijector ``Function``: a backend bijector enters through the adapter.

    Parameters
    ----------
    value : Function or tfb.Bijector
        The bijector a factory returned.
    image : Constraint or None
        The support a backend bijector maps the real line onto, which the
        adapter records; ``None`` takes the support the backend bijector is
        known to map onto, when there is one.

    Returns
    -------
    Function
        *value* itself when it is a ``Function``, and otherwise the adapter.

    Raises
    ------
    TypeError
        If *value* is neither a ``Function`` nor a backend bijector.
    """
    if isinstance(value, Function):
        return value
    if isinstance(value, tfb.Bijector):
        return _BackendBijector(value, _backend_image(value) if image is None else image)
    raise TypeError(f"a bijector is a Function or a backend bijector, got {type(value).__name__}")


def register_bijector(key: type[Constraint] | Constraint, factory: BijectorFactory) -> None:
    """Register a bijector factory for a constraint type or one constraint.

    Parameters
    ----------
    key : type[Constraint] or Constraint
        A ``Constraint`` subclass, which covers each of its instances, or one
        constraint, which covers the constraints equal to it. An instance key
        takes precedence over a type key.
    factory : callable
        ``factory(constraint)`` returns a bijector ``Function`` claiming the
        inverse and the log-determinant of its Jacobian, or a backend bijector,
        which enters as such a ``Function``. It receives the constraint, so it
        can read the constraint's parameters.

    Notes
    -----
    Registering a key again replaces its factory. A registration for
    ``Constraint`` itself would cover every constraint without a registration
    of its own.
    """
    _CONSTRAINT_BIJECTOR_REGISTRY[key] = factory


def bijector_for(constraint: Constraint) -> Function:
    """The canonical bijector from ℝⁿ onto *constraint*'s support.

    The factory registered for *constraint* itself applies first, and then the
    one registered for the nearest type in its method-resolution order.

    Parameters
    ----------
    constraint : Constraint
        The support to map onto.

    Returns
    -------
    Function
        A map onto the support that claims ``SupportsInverse`` and
        ``SupportsLogDetJacobian``.

    Raises
    ------
    ResolutionError
        If no factory is registered for *constraint* or its types, or the
        factory's map does not claim both capabilities.
    MathematicalDomainError
        If *constraint* is a support onto which no smooth bijector exists, such
        as a discrete support or the unit sphere.
    """
    factory = _factory(constraint)
    bijector = _as_bijector(factory(constraint), image=constraint)
    missing = [
        claim
        for claim, holds in (
            ("SupportsInverse", is_invertible(bijector)),
            ("SupportsLogDetJacobian", isinstance(bijector, SupportsLogDetJacobian)),
        )
        if not holds
    ]
    if missing:
        raise ResolutionError(
            f"the bijector {bijector.label!r} registered for {constraint!r} does not claim "
            f"{' and '.join(missing)}"
        )
    return bijector


def _factory(constraint: Constraint) -> BijectorFactory:
    """The factory registered for *constraint*, by instance and then by type.

    Parameters
    ----------
    constraint : Constraint
        The support whose factory is looked up.

    Returns
    -------
    callable
        The factory, which takes *constraint* and returns a bijector.

    Raises
    ------
    ResolutionError
        If no factory is registered for *constraint* or any of its types.
    """
    # A constraint with an array parameter does not hash, so only its type keys it.
    try:
        factory = _CONSTRAINT_BIJECTOR_REGISTRY.get(constraint)
    except TypeError:
        factory = None
    if factory is not None:
        return factory
    for cls in type(constraint).__mro__:
        if cls in _CONSTRAINT_BIJECTOR_REGISTRY:
            return _CONSTRAINT_BIJECTOR_REGISTRY[cls]
    raise ResolutionError(
        f"No bijector registered for {constraint!r}; register one with probpipe.register_bijector."
    )


# ---------------------------------------------------------------------------
# The canonical bijectors
# ---------------------------------------------------------------------------


class _SimplexBijector(tfb.SoftmaxCentered):
    """The backend's map from ``R^(K-1)`` onto the ``K``-simplex, with its Jacobian taken in the simplex's first ``K - 1`` coordinates.

    A density on the simplex, a Dirichlet's for one, is stated in those
    coordinates. The backend's log-determinant measures the simplex by its
    surface area in ``R^K``, which adds ``log(K) / 2`` to it.
    """

    def _forward_log_det_jacobian(self, x: Any) -> Any:
        return super()._forward_log_det_jacobian(x) - 0.5 * jnp.log(x.shape[-1] + 1.0)

    def _inverse_log_det_jacobian(self, y: Any) -> Any:
        return super()._inverse_log_det_jacobian(y) + 0.5 * jnp.log(float(y.shape[-1]))


register_bijector(_Real, lambda c: tfb.Identity())
register_bijector(_Positive, lambda c: tfb.Exp())
# Softplus is defined at zero, which the boundary of the non-negative half-line includes.
register_bijector(_NonNegative, lambda c: tfb.Softplus())
register_bijector(_UnitInterval, lambda c: tfb.Sigmoid())
register_bijector(_Interval, lambda c: tfb.Sigmoid(low=c.low, high=c.high))
register_bijector(_GreaterThan, lambda c: tfb.Chain([tfb.Shift(c.lower_bound), tfb.Exp()]))
register_bijector(_Simplex, lambda c: _SimplexBijector())
# ℝ^{n(n+1)/2} to a lower-triangular L, then to L Lᵀ; Chain applies its last bijector first.
register_bijector(
    _PositiveDefinite, lambda c: tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])
)


def _no_smooth_bijector(reason: str) -> BijectorFactory:
    """A factory that raises for a support onto which no smooth bijector exists."""

    def _raise(constraint: Constraint) -> Function:
        raise MathematicalDomainError(f"{constraint!r} has no smooth bijector: {reason}")

    return _raise


register_bijector(
    _Sphere,
    _no_smooth_bijector(
        "no smooth global chart maps ℝⁿ onto the unit sphere; a stereographic projection "
        "covers all but one point"
    ),
)
register_bijector(
    _Boolean,
    _no_smooth_bijector("the support is discrete; a continuous relaxation can stand in"),
)
register_bijector(_NonNegativeInteger, _no_smooth_bijector("the support is discrete"))
register_bijector(
    _IntegerInterval,
    _no_smooth_bijector("the support is discrete; a continuous relaxation can stand in"),
)
