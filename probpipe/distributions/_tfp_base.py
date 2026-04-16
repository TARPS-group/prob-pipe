"""TFPDistribution base class for distributions backed by TFP instances.

Factored out of ``core/distribution.py`` because no ``core/`` module
imports ``TFPDistribution`` – it is only used by the concrete
distribution modules in ``distributions/``.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ..core.distribution import (
    NumericRecordDistribution,
    _mc_expectation,
)
from ..core.protocols import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from ..core.record import Record, RecordTemplate
from ..custom_types import Array, ArrayLike, PRNGKey


class TFPDistribution(
    NumericRecordDistribution,
    SupportsSampling,
    SupportsLogProb,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
):
    """
    Base class for distributions backed by a ``tfd.Distribution`` instance.

    Subclasses set ``self._tfp_dist`` in ``__init__``.  The private
    protocol methods ``_sample``, ``_expectation``, ``_log_prob``,
    ``_mean``, and ``_variance`` all delegate to TFP (or use MC
    fallback for expectations).

    Inherits from :class:`SupportsSampling`, :class:`SupportsExpectation`,
    :class:`SupportsLogProb` (provides ``_prob``,
    ``_unnormalized_log_prob``, ``_unnormalized_prob`` defaults),
    :class:`SupportsMean`, and :class:`SupportsVariance`.
    """

    _tfp_dist: tfd.Distribution
    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    # -- record_template auto-generation ------------------------------------

    @property
    def record_template(self):
        """Auto-build record_template from name + event_shape when named."""
        tpl = getattr(self, "_record_template", None)
        if tpl is not None:
            return tpl
        name = getattr(self, "_name", None)
        if name is not None:
            tpl = RecordTemplate(**{name: self.event_shape})
            object.__setattr__(self, "_record_template", tpl)
            return tpl
        return None

    # -- shape delegation ---------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self._tfp_dist.event_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self._tfp_dist.batch_shape)

    @property
    def dtype(self) -> jnp.dtype:
        return self._tfp_dist.dtype

    @property
    def dtypes(self) -> dict[str, jnp.dtype]:
        """Per-field dtypes (single field for TFPDistribution)."""
        tpl = self.record_template
        if tpl is not None:
            return {name: self.dtype for name in tpl.fields}
        return {}

    @property
    def support(self):
        """The support of this distribution.  Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.support")

    @property
    def supports(self) -> dict[str, any]:
        """Per-field support constraints (single field for TFPDistribution)."""
        tpl = self.record_template
        if tpl is not None:
            return {name: self.support for name in tpl.fields}
        return {}

    # -- sampling & density -------------------------------------------------

    def _sample_one(self, key: PRNGKey) -> Array:
        """Draw a single sample from the TFP distribution."""
        return self._tfp_dist.sample(seed=key)

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw samples using TFP's efficient batched sampling."""
        return self._tfp_dist.sample(seed=key, sample_shape=sample_shape)

    def _log_prob(self, x: ArrayLike) -> Array:
        return self._tfp_dist.log_prob(jnp.asarray(x))

    def _mean(self) -> Array:
        return self._tfp_dist.mean()

    def _variance(self) -> Array:
        return self._tfp_dist.variance()

    def _cov(self) -> Array:
        if self.event_shape == () or self.event_shape == (1,):
            return self._tfp_dist.variance()
        return self._tfp_dist.covariance()

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Any:
        return _mc_expectation(
            self, f, key=key, num_evaluations=num_evaluations,
            return_dist=return_dist,
        )
