"""
Transformed distributions via TFP bijectors.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from ._tfp_base import TFPDistribution
from ..core.distribution import (
    NumericRecordDistribution,
)
from ..core.provenance import Provenance
from ..core.constraints import (
    Constraint,
    real,
    positive,
    unit_interval,
)
from ..core.distribution import _mc_expectation
from ..core.protocols import SupportsLogProb, SupportsMean, SupportsSampling, SupportsVariance
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


# ---------------------------------------------------------------------------
# Dynamic protocol factory for TransformedDistribution
# ---------------------------------------------------------------------------

_TRANSFORMED_CLASS_CACHE: dict[frozenset[str], type] = {}


def _transformed_class_for_base(base: NumericRecordDistribution) -> type:
    """Return a TransformedDistribution subclass whose protocol bases
    match what the base distribution supports.

    ``_sample``, ``_log_prob``, ``_mean``, ``_variance`` are installed
    on the dynamic subclass (not the base class) so that ``isinstance``
    checks against the ``SupportsFoo`` protocols are accurate: an
    instance only has the method when its base actually supports the
    corresponding protocol. ``_mean`` / ``_variance`` use a Monte Carlo
    fallback, which requires sampling.
    """
    supports_sample = isinstance(base, SupportsSampling)
    supports_log_prob = isinstance(base, SupportsLogProb)
    supports_mean = isinstance(base, SupportsMean) or supports_sample
    supports_variance = isinstance(base, SupportsVariance) or supports_sample

    signature = (supports_sample, supports_log_prob, supports_mean, supports_variance)
    key = frozenset(
        name for name, flag in zip(
            ("sample", "log_prob", "mean", "variance"), signature,
        ) if flag
    )
    if key in _TRANSFORMED_CLASS_CACHE:
        return _TRANSFORMED_CLASS_CACHE[key]

    extra_bases: list[type] = []
    extra_methods: dict[str, object] = {}

    if supports_sample:
        extra_bases.append(SupportsSampling)

        def _sample(
            self,
            key: PRNGKey,
            sample_shape: tuple[int, ...] = (),
        ) -> Array:
            if self._tfp_transformed is not None:
                return self._tfp_transformed.sample(seed=key, sample_shape=sample_shape)
            raw = self._base._sample(key, sample_shape)
            return self._bijector.forward(raw)

        extra_methods["_sample"] = _sample

    if supports_log_prob:
        extra_bases.append(SupportsLogProb)

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

        extra_methods["_log_prob"] = _log_prob

    if supports_mean:
        extra_bases.append(SupportsMean)

        def _mean(self) -> Array:
            if self._tfp_transformed is not None:
                return self._tfp_transformed.mean()
            return self._expectation(lambda x: x, return_dist=False)

        extra_methods["_mean"] = _mean

    if supports_variance:
        extra_bases.append(SupportsVariance)

        def _variance(self) -> Array:
            if self._tfp_transformed is not None:
                return self._tfp_transformed.variance()
            mu = self._mean()
            return self._expectation(lambda x: (x - mu) ** 2, return_dist=False)

        extra_methods["_variance"] = _variance

    if not extra_bases:
        _TRANSFORMED_CLASS_CACHE[key] = TransformedDistribution
        return TransformedDistribution

    cls = type(
        "TransformedDistribution",
        (TransformedDistribution, *extra_bases),
        extra_methods,
    )
    _TRANSFORMED_CLASS_CACHE[key] = cls
    return cls


class TransformedDistribution(NumericRecordDistribution):
    """
    Distribution formed by applying a TFP bijector to a base distribution.

    When *base* is a :class:`TFPDistribution`, sampling and density
    evaluation delegate to ``tfd.TransformedDistribution`` for maximum
    efficiency.  Otherwise (e.g. :class:`EmpiricalDistribution`), the
    bijector's ``forward`` / ``inverse`` / ``inverse_log_det_jacobian``
    are applied manually.

    **Dynamic protocol support:** ``SupportsLogProb``, ``SupportsMean``,
    and ``SupportsVariance`` are included only when the base distribution
    supports them.

    Parameters
    ----------
    base : Distribution
        The untransformed base distribution.
    bijector : tfb.Bijector
        A TFP bijector (e.g. ``tfb.Exp()``, ``tfb.Sigmoid()``).
    name : str, optional
        Distribution name for provenance.
    """

    def __new__(
        cls,
        base: NumericRecordDistribution,
        bijector: tfb.Bijector,
        *,
        name: str | None = None,
    ):
        actual_cls = _transformed_class_for_base(base)
        return object.__new__(actual_cls)

    def __init__(
        self,
        base: NumericRecordDistribution,
        bijector: tfb.Bijector,
        *,
        name: str | None = None,
    ):
        self._base = base
        self._bijector = bijector
        if name is None:
            name = f"transformed({base.name})"
        super().__init__(name=name)

        if isinstance(base, TFPDistribution):
            self._tfp_transformed = tfd.TransformedDistribution(
                distribution=base._tfp_dist,
                bijector=bijector,
                name=name or "TransformedDistribution",
            )
        else:
            self._tfp_transformed = None

        self._approximate = base.is_approximate

        self.with_source(Provenance(
            "transform",
            parents=(base,),
            metadata={"bijector": type(bijector).__name__},
        ))

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    # -- convenient accessors -----------------------------------------------

    @property
    def base(self) -> NumericRecordDistribution:
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

    # -- sampling, density, moments are installed dynamically in
    # -- _transformed_class_for_base based on what ``base`` supports.

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
        bij_name = type(self._bijector).__name__
        base_name = type(self._base).__name__
        parts = [f"TransformedDistribution({base_name}, {bij_name}"]
        if self.name:
            parts[0] += f", name={self.name!r}"
        parts[0] += f", event_shape={self.event_shape})"
        return parts[0]
