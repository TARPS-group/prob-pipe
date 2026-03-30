"""BijectorTransformedDistribution: distribution formed by applying a Bijector."""

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ..core.distribution import (
    ArrayDistribution,
    TFPDistribution,
    Provenance,
    real,
    _mc_expectation,
)
from ..core.protocols import (
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from ..custom_types import Array, ArrayLike, PRNGKey
from .bijector import Bijector, TFPBijector

if TYPE_CHECKING:
    from ..core.distribution import Constraint

__all__ = ["BijectorTransformedDistribution"]


class BijectorTransformedDistribution(
    ArrayDistribution,
    SupportsSampling,
    SupportsLogProb,
    SupportsMean,
    SupportsVariance,
):
    """Distribution formed by applying a :class:`Bijector` to a base
    distribution.

    When both the base and bijector are TFP-backed
    (:class:`TFPDistribution` and :class:`TFPBijector`), sampling and
    density evaluation delegate to ``tfd.TransformedDistribution`` for
    efficiency.  Otherwise the bijector's ``forward`` / ``inverse`` /
    ``inverse_log_det_jacobian`` are applied manually.

    Parameters
    ----------
    base : ArrayDistribution
        The untransformed base distribution.
    bijector : Bijector
        A probpipe bijector (e.g. ``TFPBijector(tfb.Exp())``).
    name : str, optional
        Distribution name for provenance / display.
    """

    _sampling_cost: ClassVar[str] = "low"
    _preferred_orchestration: ClassVar[str | None] = None

    def __init__(
        self,
        base: ArrayDistribution,
        bijector: Bijector,
        *,
        name: str | None = None,
    ) -> None:
        self._base = base
        self._bijector = bijector
        self._name = name

        # TFP fast path
        if isinstance(base, TFPDistribution) and isinstance(bijector, TFPBijector):
            self._tfp_transformed = tfd.TransformedDistribution(
                distribution=base._tfp_dist,
                bijector=bijector.tfp_bijector,
            )
        else:
            self._tfp_transformed = None

        self._approximate = base.is_approximate

        self.with_source(
            Provenance(
                "pushforward",
                parents=(base,),
                metadata={"bijector": repr(bijector)},
            )
        )

    # -- accessors ----------------------------------------------------------

    @property
    def base(self) -> ArrayDistribution:
        """The untransformed base distribution."""
        return self._base

    @property
    def bijector(self) -> Bijector:
        """The bijector applied to *base*."""
        return self._bijector

    # -- shape delegation ---------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        if self._tfp_transformed is not None:
            return tuple(self._tfp_transformed.event_shape)
        # For non-TFP bijectors, assume shape is preserved.
        return self._base.event_shape

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
        """Derive output support from the bijector when possible."""
        constraint = self._bijector.output_constraint
        if constraint is not None:
            return constraint
        return real

    # -- sampling & density -------------------------------------------------

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        if self._tfp_transformed is not None:
            return self._tfp_transformed.sample(
                seed=key, sample_shape=sample_shape
            )
        raw = self._base._sample(key, sample_shape)
        return self._bijector.forward(raw)

    def _log_prob(self, x: ArrayLike) -> Array:
        x = jnp.asarray(x)
        if self._tfp_transformed is not None:
            return self._tfp_transformed.log_prob(x)
        raw = self._bijector.inverse(x)
        return self._base._log_prob(raw) + self._bijector.inverse_log_det_jacobian(
            x, event_ndims=len(self.event_shape)
        )

    # -- moments (delegate to TFP when available, else MC fallback) ----------

    def _mean(self) -> Array:
        if self._tfp_transformed is not None:
            return self._tfp_transformed.mean()
        return self._expectation(lambda x: x, return_dist=False)

    def _variance(self) -> Array:
        if self._tfp_transformed is not None:
            return self._tfp_transformed.variance()
        mu = self._mean()
        return self._expectation(lambda x: (x - mu) ** 2, return_dist=False)

    def _expectation(
        self,
        f,
        *,
        key=None,
        num_evaluations=None,
        return_dist=None,
    ):
        return _mc_expectation(
            self, f, key=key, num_evaluations=num_evaluations,
            return_dist=return_dist,
        )

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        bij_name = repr(self._bijector)
        base_name = type(self._base).__name__
        parts = f"BijectorTransformedDistribution({base_name}, {bij_name}"
        if self.name:
            parts += f", name={self.name!r}"
        parts += f", event_shape={self.event_shape})"
        return parts
