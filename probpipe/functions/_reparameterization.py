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

from .._messages import unknown_names
from ..core._catalog import EntrySummary, registry_catalog
from ..core._dispatch import MathematicalDomainError, ResolutionError
from ..core._repr import type_name
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

#: The one-line description of each registered key, which the registry catalog shows.
_BIJECTOR_DESCRIPTIONS: dict[type | Constraint, str] = {}

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
    """The forward map of a backend bijector at one point, named as the backend names it.

    The name makes the bijector's label and output component the backend's
    name, as ``exp``, by the rule that names a Function and its output after
    its callable.
    """

    def __init__(self, bijector: tfb.Bijector) -> None:
        self._bijector = bijector
        self.__name__ = bijector.name

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
        super().__init__(_ForwardMap(bijector))
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
    bijector: tfb.Bijector, image: Constraint | None, label: str
) -> _BackendBijector:
    """The backend-bijector Function of *bijector* under *label*, for unpickling."""
    rebuilt = _BackendBijector(bijector, image)
    return rebuilt if rebuilt.label == label else rebuilt.with_label(label)


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
    raise TypeError(f"bijector must be a Function or a TFP bijector; got {type_name(value)}")


def register_bijector(
    key: type[Constraint] | Constraint, factory: BijectorFactory, *, description: str = ""
) -> None:
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
    description : str, optional
        A one-line description of the bijector, which the registry catalog
        shows under ``"bijectors"``; empty by default.

    Raises
    ------
    TypeError
        If *description* is not a ``str``.

    Notes
    -----
    Registering a key again replaces its factory and its description. A
    registration for ``Constraint`` itself would cover every constraint
    without a registration of its own.
    """
    if not isinstance(description, str):
        raise TypeError(f"register_bijector's description must be a str, got {description!r}")
    _CONSTRAINT_BIJECTOR_REGISTRY[key] = factory
    _BIJECTOR_DESCRIPTIONS[key] = description


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
    TypeError
        If the factory returns neither a ``Function`` nor a backend bijector.
    MathematicalDomainError
        If *constraint* is a support onto which no smooth bijector exists, such
        as a discrete support or the unit sphere.
    """
    factory = _factory(constraint)
    returned = factory(constraint)
    if not isinstance(returned, (Function, tfb.Bijector)):
        raise TypeError(
            f"the bijector factory registered for {constraint!r} returned "
            f"{type_name(returned)}; it must return a Function or a TFP bijector"
        )
    bijector = _as_bijector(returned, image=constraint)
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
            f"the bijector {bijector.label!r} registered for {constraint!r} must support "
            f"SupportsInverse and SupportsLogDetJacobian, but lacks {' and '.join(missing)}"
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
        f"no bijector registered for {constraint!r}; register one with probpipe.register_bijector"
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


register_bijector(_Real, lambda c: tfb.Identity(), description="The identity on the real line.")
register_bijector(_Positive, lambda c: tfb.Exp(), description="exp, onto the positive half-line.")
# Softplus is defined at zero, which the boundary of the non-negative half-line includes.
register_bijector(
    _NonNegative, lambda c: tfb.Softplus(), description="softplus, onto the non-negative half-line."
)
register_bijector(
    _UnitInterval, lambda c: tfb.Sigmoid(), description="The sigmoid, onto the unit interval."
)
register_bijector(
    _Interval,
    lambda c: tfb.Sigmoid(low=c.low, high=c.high),
    description="A sigmoid scaled onto the interval's bounds.",
)
register_bijector(
    _GreaterThan,
    lambda c: tfb.Chain([tfb.Shift(c.lower_bound), tfb.Exp()]),
    description="exp shifted to the lower bound.",
)
register_bijector(
    _Simplex,
    lambda c: _SimplexBijector(),
    description="The centered softmax from R^(K-1) onto the K-simplex.",
)
# ℝ^{n(n+1)/2} to a lower-triangular L, then to L Lᵀ; Chain applies its last bijector first.
register_bijector(
    _PositiveDefinite,
    lambda c: tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()]),
    description="A lower-triangular L filled from a vector, then L Lᵀ.",
)


def _register_no_smooth_bijector(key: type[Constraint], reason: str) -> None:
    """Register for *key* a factory that raises, since no smooth bijector maps onto its support.

    The catalog describes the entry by the same *reason* the factory raises
    with, so the listing does not suggest that :func:`bijector_for` returns a
    bijector for it.
    """

    def _raise(constraint: Constraint) -> Function:
        raise MathematicalDomainError(f"{constraint!r} has no smooth bijector: {reason}")

    register_bijector(key, _raise, description=f"No smooth bijector: {reason}.")


_register_no_smooth_bijector(
    _Sphere,
    "no smooth global chart maps ℝⁿ onto the unit sphere; a stereographic projection "
    "covers all but one point",
)
_register_no_smooth_bijector(
    _Boolean, "the support is discrete; a continuous relaxation can stand in"
)
_register_no_smooth_bijector(_NonNegativeInteger, "the support is discrete")
_register_no_smooth_bijector(
    _IntegerInterval, "the support is discrete; a continuous relaxation can stand in"
)


# ---------------------------------------------------------------------------
# The factory in the registry catalog
# ---------------------------------------------------------------------------


def _short_name(key: type | Constraint) -> str:
    """A key's short catalog name: a type's name without leading underscores, or an instance's ``repr``."""
    return key.__name__.lstrip("_") if isinstance(key, type) else repr(key)


def _qualified_name(key: type | Constraint) -> str:
    """A key's catalog name when its short name is shared: qualified by its module."""
    if isinstance(key, type):
        return f"{key.__module__}.{key.__qualname__}"
    return f"{key!r} ({type(key).__module__}.{type(key).__qualname__})"


def _entry_names(keys: list[type | Constraint]) -> dict[type | Constraint, str]:
    """The catalog name of each registered key, unique among *keys*.

    A key is named by its short name, as ``_Positive`` is named ``Positive``
    and a constraint instance by its ``repr``. Where keys share a short name,
    each key that ProbPipe does not define is named by its qualified name in
    its place, so a third-party ``Positive`` is named
    ``"mypackage.Positive"`` and the built-in keeps ``Positive``.
    """
    counts: dict[str, int] = {}
    for key in keys:
        counts[_short_name(key)] = counts.get(_short_name(key), 0) + 1
    names = {}
    for key in keys:
        short = _short_name(key)
        module = key.__module__ if isinstance(key, type) else type(key).__module__
        builtin = module == "probpipe" or module.startswith("probpipe.")
        names[key] = short if counts[short] == 1 or builtin else _qualified_name(key)
    return names


class _BijectorRegistryView:
    """The constraint-to-bijector factory as the registry catalog lists it.

    The factory maps a constraint, by instance and then by the nearest type in
    its method-resolution order, to a function returning a bijector;
    :func:`register_bijector` populates it and :func:`bijector_for` reads it.
    This object implements
    :class:`~probpipe.core._catalog.SupportsRegistryCataloging` over those
    registrations, with one entry per registered key in name order, each
    named as :func:`_entry_names` names it, and it is cataloged as
    ``"bijectors"``. Each entry's description is the one
    :func:`register_bijector` received. The factory does not rank its entries,
    so each entry's ``priority`` and ``exact`` are ``None``.
    """

    name = "bijectors"
    description = (
        "Bijector factories onto each constrained support, by constraint instance and then type."
    )
    kind = "factory"

    def entry_summaries(self) -> list[EntrySummary]:
        """One :class:`~probpipe.core._catalog.EntrySummary` per registered key, in name order."""
        names = _entry_names(list(_CONSTRAINT_BIJECTOR_REGISTRY))
        summaries = [
            EntrySummary(
                name=names[key],
                priority=None,
                supported_types=(key,),
                description=_BIJECTOR_DESCRIPTIONS.get(key, ""),
                module_path=key.__module__ if isinstance(key, type) else type(key).__module__,
            )
            for key in _CONSTRAINT_BIJECTOR_REGISTRY
        ]
        return sorted(summaries, key=lambda summary: summary.name)

    def describe_entry(self, name: str) -> EntrySummary:
        """The :class:`~probpipe.core._catalog.EntrySummary` of the key named *name*.

        Parameters
        ----------
        name : str
            A registered key's catalog name, such as ``"Positive"``.

        Returns
        -------
        EntrySummary
            The key's summary, as :meth:`entry_summaries` reports it.

        Raises
        ------
        KeyError
            If no registered key has that name.
        """
        summaries = self.entry_summaries()
        for summary in summaries:
            if summary.name == name:
                return summary
        raise KeyError(
            unknown_names(
                "bijector entry",
                [name],
                [summary.name for summary in summaries],
                plural="bijector entries",
            )
        )


registry_catalog.register(_BijectorRegistryView())
