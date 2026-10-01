"""The evaluation-result families: the lazy linear pushforward and the bijector transform.

Each evaluation rule returns a family of the catalog. The generic linear rule
returns a ``LinearPushforwardDistribution`` when no family-specific rule
applies, and the change-of-variables rule returns a
``BijectorTransformedDistribution``.

Provides:
  - ``LinearPushforwardDistribution`` – the law of ``op @ X`` for ``X ~ base``.
  - ``BijectorTransformedDistribution`` – the law of ``f(X)`` for an invertible
    ``f`` with a tractable Jacobian determinant.
"""

from __future__ import annotations

from math import prod
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from ..core._dispatch import ResolutionError
from ..core._spec_base import NumericArraySpec
from ..core.provenance import Provenance
from ..custom_types import Array, ArrayLike, PRNGKey
from ..distributions._capabilities import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
    _capability_subclass,
)
from ..distributions._distribution import Distribution
from ..functions._reparameterization import _as_bijector, _image, _is_affine
from ..linalg import DenseLinOp, LinOp
from ..values import Function, SupportsLogDetJacobian, is_invertible

__all__ = ["BijectorTransformedDistribution", "LinearPushforwardDistribution"]


class LinearPushforwardDistribution(Distribution):
    """The law of ``op @ X`` for ``X ~ base``, represented lazily.

    The event type is ``op``'s output type, under the pushforward's own
    component. Sampling pushes the base's draws through ``op``. The mean and
    the covariance delegate exactly, ``E[A X] = A E[X]`` and ``Cov(A X) = A
    Cov(X) Aᵀ``, lazily through the operator algebra. The log-density exists
    only when ``op`` is invertible, by change of variables.

    Parameters
    ----------
    name : str
        The pushforward's label, and the component of its event.
    base : Distribution
        The law of ``X``, whose numeric event is ``op``'s input.
    op : LinOp
        The linear map.

    Raises
    ------
    NotImplementedError
        Always, until the linear evaluation rule is implemented.
    """

    def __init__(self, name: str, base: Distribution, op: LinOp) -> None:
        raise NotImplementedError("LinearPushforwardDistribution.__init__")


# ---------------------------------------------------------------------------
# The bijector transform
# ---------------------------------------------------------------------------


def _event_rank(law: Distribution) -> int:
    return len(law.event_spec.spec.shape)


def _per_point(function: Any, value: Array, rank: int) -> Array:
    """*function* of one point, mapped over the leading axes of *value* beyond *rank*."""
    batch = value.shape[: value.ndim - rank]
    if not batch:
        return function(value)
    flat = jnp.reshape(value, (prod(batch), *value.shape[value.ndim - rank :]))
    result = jax.vmap(function)(flat)
    return jnp.reshape(result, (*batch, *result.shape[1:]))


def _transformed_sample(
    self: BijectorTransformedDistribution, key: PRNGKey, sample_shape: tuple[int, ...] = ()
) -> Array:
    """Draws of the base pushed through the bijector, with *sample_shape* leading."""
    draws = jnp.asarray(self._base._sample(key, sample_shape))
    return _per_point(self._bijector.apply, draws, _event_rank(self._base))


def _change_of_variables(
    self: BijectorTransformedDistribution, value: ArrayLike, density: str
) -> Array:
    """The base's *density* at the preimage of *value* minus the log-Jacobian there."""
    y = jnp.asarray(value)

    def one(point: Array) -> Array:
        x = self._bijector._inverse(point)
        return getattr(self._base, density)(x) - self._bijector._log_det_jacobian(x)

    return _per_point(one, y, _event_rank(self))


def _transformed_log_prob(self: BijectorTransformedDistribution, value: ArrayLike) -> Array:
    """``log p(f⁻¹(y)) − log |det J_f(f⁻¹(y))|``, the change of variables."""
    return _change_of_variables(self, value, "_log_prob")


def _transformed_unnormalized_log_prob(
    self: BijectorTransformedDistribution, value: ArrayLike
) -> Array:
    """The base's unnormalized log-density at the preimage, less the log-Jacobian."""
    return _change_of_variables(self, value, "_unnormalized_log_prob")


def _affine_jacobian(self: BijectorTransformedDistribution) -> Array:
    """The constant Jacobian of the affine forward map over the flattened events."""
    base_shape = self._base.event_spec.spec.shape
    point = jnp.reshape(jnp.asarray(self._base._mean()), (-1,))

    def flat_map(vector: Array) -> Array:
        return jnp.reshape(self._bijector.apply(jnp.reshape(vector, base_shape)), (-1,))

    return jax.jacfwd(flat_map)(point)


def _transformed_mean(self: BijectorTransformedDistribution) -> Array:
    """``f(E[X])``, which is ``E[f(X)]`` for an affine ``f``."""
    return self._bijector.apply(jnp.asarray(self._base._mean()))


def _transformed_cov(self: BijectorTransformedDistribution) -> LinOp:
    """``J Cov(X) Jᵀ`` over the flattened event, with ``J`` the affine map's Jacobian."""
    jacobian = _affine_jacobian(self)
    return DenseLinOp(jacobian @ self._base._cov().to_dense() @ jacobian.T)


def _transformed_variance(self: BijectorTransformedDistribution) -> Array:
    """The diagonal of the pushed covariance, shaped like one draw."""
    diagonal = jnp.diagonal(_transformed_cov(self).to_dense())
    return jnp.reshape(diagonal, self.event_spec.spec.shape)


def _claimed(base: Distribution, bijector: Function) -> list[type]:
    """The capabilities the law of ``f(X)`` claims for this base and bijector."""
    claimed: list[type] = []
    if isinstance(base, SupportsSampling):
        claimed.append(SupportsSampling)
    if isinstance(base, SupportsLogProb):
        claimed.append(SupportsLogProb)
    elif isinstance(base, SupportsUnnormalizedLogProb):
        claimed.append(SupportsUnnormalizedLogProb)
    if _is_affine(bijector) and isinstance(base, SupportsMean):
        claimed.append(SupportsMean)
        if isinstance(base, SupportsCovariance):
            claimed.extend((SupportsCovariance, SupportsVariance))
    return claimed


class BijectorTransformedDistribution(Distribution):
    """The law of ``f(X)`` for ``X ~ base`` and an invertible map ``f`` with a Jacobian.

    Construction checks that ``bijector`` is invertible and claims the
    log-determinant of its Jacobian. Sampling pushes the base's draws through
    the bijector, and the log-density at ``y`` is the base's log-density at the
    preimage minus the log-determinant of the Jacobian there. A backend
    bijector enters as a ``Function`` that claims the inverse and the
    log-Jacobian the backend computes.

    The law claims sampling when the base does and the density the base has.
    It claims a moment only where the bijector gives it in closed form: when
    the forward map is affine, ``E[f(X)] = f(E[X])`` and ``Cov(f(X)) = J
    Cov(X) Jᵀ`` with ``J`` the map's constant Jacobian. The moment operations
    estimate the others by their Monte Carlo fallback.

    One draw is an array whose shape and dtype are those of the bijector's
    output at a draw of the base, declared as a whole term under the law's
    name; its support is the one the bijector maps onto when that is known.

    Parameters
    ----------
    name : str
        The transformed law's label.
    base : Distribution
        The law of ``X``, whose draws are arrays.
    bijector : Function
        The invertible map ``f``, claiming the inverse and the log-determinant
        of its Jacobian; a backend bijector enters as such a ``Function``.

    Raises
    ------
    TypeError
        If *base* is not a ``Distribution`` whose draws are arrays, or
        *bijector* is neither a ``Function`` nor a backend bijector.
    ResolutionError
        If *bijector* does not claim ``SupportsInverse``, or its guard
        declines, or it does not claim ``SupportsLogDetJacobian``.
    """

    _capability_table: ClassVar = {
        SupportsSampling: {"_sample": _transformed_sample},
        SupportsLogProb: {"_log_prob": _transformed_log_prob},
        SupportsUnnormalizedLogProb: {"_unnormalized_log_prob": _transformed_unnormalized_log_prob},
        SupportsMean: {"_mean": _transformed_mean},
        SupportsVariance: {"_variance": _transformed_variance},
        SupportsCovariance: {"_cov": _transformed_cov},
    }

    def __new__(
        cls, name: str, base: Distribution, bijector: Function | Any
    ) -> BijectorTransformedDistribution:
        claimed = _claimed(base, _as_bijector(bijector)) if isinstance(base, Distribution) else ()
        return object.__new__(_capability_subclass(cls, claimed))

    def __init__(self, name: str, base: Distribution, bijector: Function | Any) -> None:
        if not isinstance(base, Distribution) or not isinstance(
            base.event_spec.spec, NumericArraySpec
        ):
            raise TypeError(
                f"BijectorTransformedDistribution takes a base whose draws are arrays, got "
                f"{type(base).__name__}"
            )
        bijector = _as_bijector(bijector)
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
                f"the bijector {bijector.name!r} of {name!r} does not claim "
                f"{' and '.join(missing)}, which a change of variables needs"
            )
        base_spec = base.event_spec.spec
        point = jax.ShapeDtypeStruct(tuple(base_spec.shape), base_spec.dtype or jnp.float32)
        image = jax.eval_shape(bijector.apply, point)
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_bijector", bijector)
        super().__init__(name, NumericArraySpec(tuple(image.shape), image.dtype, _image(bijector)))
        self.with_provenance(
            Provenance.create("transform", parents=[base], metadata={"bijector": bijector.name})
        )

    @property
    def base(self) -> Distribution:
        """The law of ``X``."""
        return self._base

    @property
    def bijector(self) -> Function:
        """The invertible map ``f``."""
        return self._bijector

    def __repr__(self) -> str:
        return (
            f"BijectorTransformedDistribution(name={self.name!r}, base={type(self._base).__name__}, "
            f"bijector={self._bijector.name!r}, event_shape={self.event_shape})"
        )
