"""
Transformed distributions via TFP bijectors.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from ..core.distribution import (
    ArrayDistribution,
    TFPDistribution,
    Constraint,
    Provenance,
    real,
    positive,
    unit_interval,
)
from ..custom_types import Array, ArrayLike, PRNGKey

__all__ = ["TransformedDistribution"]

# Map bijector class names to the output support they produce when applied
# to a base distribution with ``real`` support.
_BIJECTOR_SUPPORT_MAP: dict[str, Constraint] = {
    "Exp": positive,
    "Softplus": positive,
    "Sigmoid": unit_interval,
    "Square": positive,  # real → non-negative, but TFP Square maps R→R+
}


class TransformedDistribution(ArrayDistribution):
    """
    Distribution formed by applying a TFP bijector to a base distribution.

    When *base* is a :class:`TFPDistribution`, sampling and density
    evaluation delegate to ``tfd.TransformedDistribution`` for maximum
    efficiency.  Otherwise (e.g. :class:`EmpiricalDistribution`), the
    bijector's ``forward`` / ``inverse`` / ``inverse_log_det_jacobian``
    are applied manually.

    Parameters
    ----------
    base : Distribution
        The untransformed base distribution.
    bijector : tfb.Bijector
        A TFP bijector (e.g. ``tfb.Exp()``, ``tfb.Sigmoid()``).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        base: ArrayDistribution,
        bijector: tfb.Bijector,
        *,
        name: str | None = None,
    ):
        self._base = base
        self._bijector = bijector
        self._name = name

        if isinstance(base, TFPDistribution):
            self._tfp_transformed = tfd.TransformedDistribution(
                distribution=base._tfp_dist,
                bijector=bijector,
            )
        else:
            self._tfp_transformed = None

        self._approximate = base.is_approximate

        self.with_source(Provenance(
            "transform",
            parents=(base,),
            metadata={"bijector": type(bijector).__name__},
        ))

    # -- convenient accessors -----------------------------------------------

    @property
    def base(self) -> ArrayDistribution:
        """The untransformed base distribution."""
        return self._base

    @property
    def bijector(self) -> tfb.Bijector:
        """The TFP bijector applied to *base*."""
        return self._bijector

    # -- shape delegation ---------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        if self._tfp_transformed is not None:
            return tuple(self._tfp_transformed.event_shape)
        return tuple(
            self._bijector.forward_event_shape(self._base.event_shape)
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        if self._tfp_transformed is not None:
            return tuple(self._tfp_transformed.batch_shape)
        return self._base.batch_shape

    @property
    def dtype(self) -> jnp.dtype:
        if self._tfp_transformed is not None:
            return self._tfp_transformed.dtype
        return self._base.dtype

    # -- support ------------------------------------------------------------

    @property
    def support(self) -> Constraint:
        """Derive the output support from the bijector when possible."""
        bij_name = type(self._bijector).__name__
        if bij_name in _BIJECTOR_SUPPORT_MAP:
            return _BIJECTOR_SUPPORT_MAP[bij_name]
        # For Chain bijectors, check the outermost (last-applied) bijector
        if bij_name == "Chain" and hasattr(self._bijector, "bijectors"):
            # TFP Chain applies bijectors in reverse order: bijectors[0] is
            # applied last (outermost).
            outermost = type(self._bijector.bijectors[0]).__name__
            if outermost in _BIJECTOR_SUPPORT_MAP:
                return _BIJECTOR_SUPPORT_MAP[outermost]
        return real

    # -- sampling & density -------------------------------------------------

    def _sample(self, key: PRNGKey) -> Array:
        """Draw a single sample by transforming a base sample."""
        if self._tfp_transformed is not None:
            return self._tfp_transformed.sample(seed=key)
        raw = self._base.sample(key)
        return self._bijector.forward(raw)

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw samples, delegating to TFP when available for efficiency."""
        if key is None:
            from ..core.distribution import _auto_key
            key = _auto_key()
        if self._tfp_transformed is not None:
            return self._tfp_transformed.sample(seed=key, sample_shape=sample_shape)
        raw = self._base.sample(key, sample_shape)
        return self._bijector.forward(raw)

    def _log_prob(self, x: ArrayLike) -> Array:
        x = jnp.asarray(x)
        if self._tfp_transformed is not None:
            return self._tfp_transformed.log_prob(x)
        raw = self._bijector.inverse(x)
        return (
            self._base._log_prob(raw)
            + self._bijector.inverse_log_det_jacobian(
                x, event_ndims=len(self.event_shape)
            )
        )

    def _unnormalized_log_prob(self, x: ArrayLike) -> Array:
        x = jnp.asarray(x)
        if self._tfp_transformed is not None:
            return self._tfp_transformed.unnormalized_log_prob(x)
        raw = self._bijector.inverse(x)
        return (
            self._base._unnormalized_log_prob(raw)
            + self._bijector.inverse_log_det_jacobian(
                x, event_ndims=len(self.event_shape)
            )
        )

    def _prob(self, x: ArrayLike) -> Array:
        x = jnp.asarray(x)
        if self._tfp_transformed is not None:
            return self._tfp_transformed.prob(x)
        return jnp.exp(self._log_prob(x))

    # -- moments (delegate to TFP when available) ---------------------------

    def mean(self) -> Array:
        if self._tfp_transformed is not None:
            return self._tfp_transformed.mean()
        raise NotImplementedError(
            "mean() is not available for TransformedDistribution "
            "with a non-TFP base distribution."
        )

    def variance(self) -> Array:
        if self._tfp_transformed is not None:
            return self._tfp_transformed.variance()
        raise NotImplementedError(
            "variance() is not available for TransformedDistribution "
            "with a non-TFP base distribution."
        )

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        bij_name = type(self._bijector).__name__
        base_name = type(self._base).__name__
        parts = [f"TransformedDistribution({base_name}, {bij_name}"]
        if self.name:
            parts[0] += f", name={self.name!r}"
        parts[0] += f", event_shape={self.event_shape})"
        return parts[0]
