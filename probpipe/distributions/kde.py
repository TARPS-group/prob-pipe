"""Kernel density estimation distribution backed by TFP.

Provides :class:`KDEDistribution`, a Gaussian KDE that satisfies both
:class:`~probpipe.core.protocols.SupportsLogProb` and
:class:`~probpipe.core.protocols.SupportsSampling`.  Useful for
converting a sampling-only distribution (e.g., MCMC output) into one
that supports density evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from .._dtype import _as_float_array
from .._weights import Weights
from ..core._empirical import RecordEmpiricalDistribution
from ..core._numeric_record import NumericRecord
from ..core._numeric_record_distribution import NumericRecordDistribution
from ..core._record_array import NumericRecordArray
from ..core.constraints import Constraint, real
from ..core.record import NumericRecordTemplate, Record
from ..custom_types import Array, ArrayLike
from ._tfp_base import TFPDistribution

if TYPE_CHECKING:
    from ..core.record import RecordTemplate

__all__ = ["KDEDistribution"]


class KDEDistribution(TFPDistribution):
    """Gaussian kernel density estimate as a ProbPipe distribution.

    Wraps a TFP ``MixtureSameFamily(Categorical, MultivariateNormalDiag)``
    to provide a smooth density approximation from a set of weighted
    samples.  Inherits all protocol implementations from
    :class:`TFPDistribution`.

    Parameters
    ----------
    samples : array-like
        Sample matrix of shape ``(n,)`` or ``(n, d)``.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Non-negative weights.  A pre-built :class:`~probpipe.Weights`
        object is also accepted.  Mutually exclusive with
        *log_weights*.  When neither is given, uniform weights are used.
    log_weights : array-like, :class:`~probpipe.Weights`, or None
        Log-unnormalized weights.  A pre-built :class:`~probpipe.Weights`
        object is also accepted.  Mutually exclusive with *weights*.
    bandwidth : array-like or None
        Per-dimension bandwidth (standard deviation of each Gaussian
        kernel), shape ``(d,)`` or scalar.  If ``None``, Silverman's
        rule is used: ``n^{-1/(d+4)} * std_j`` for each dimension *j*.
    record_template : RecordTemplate or None
        Structural template for the KDE's value type. When ``None`` (the
        default) a single-field template keyed by ``name`` is auto-built,
        matching the historical behavior. When supplied with multiple
        fields, the template defines how the flat ``(n, d)`` sample matrix
        maps back to a structured ``NumericRecord`` / ``NumericRecordArray``
        — preserving named fields end-to-end across e.g. an MCMC posterior
        being routed through KDE as the new prior in
        :class:`~probpipe.modeling.IncrementalConditioner`. The template's
        ``flat_size`` must equal ``samples.shape[1]``.
    name : str or None
        Distribution name for provenance.
    """

    def __init__(
        self,
        samples: ArrayLike,
        weights: ArrayLike | Weights | None = None,
        *,
        log_weights: ArrayLike | Weights | None = None,
        bandwidth: ArrayLike | None = None,
        record_template: "RecordTemplate | None" = None,
        name: str | None = None,
    ):
        samples = _as_float_array(samples)
        if samples.ndim == 0:
            raise ValueError("samples must have at least 1 dimension.")
        if samples.ndim == 1:
            samples = samples[:, None]  # (n,) -> (n, 1)
            self._scalar = True
        else:
            self._scalar = False

        n, d = samples.shape
        self._samples = samples
        self._d = d
        if name is None:
            name = "kde"

        # Multi-field template support: when the caller supplies a template
        # with more than one field, preset ``_record_template`` so that
        # ``NumericRecordDistribution.record_template`` (parent) skips its
        # single-field auto-build keyed by ``name``. Validate that the
        # template's flat width matches the samples' trailing dimension.
        if record_template is not None and len(record_template.fields) > 1:
            if isinstance(record_template, NumericRecordTemplate):
                expected = record_template.flat_size
            else:
                expected = sum(
                    int(jnp.prod(jnp.array(shape))) if shape else 1
                    for shape in record_template.leaf_shapes.values()
                )
            if expected != d:
                raise ValueError(
                    f"record_template flat_size ({expected}) does not match "
                    f"samples flat dimension ({d}); template fields="
                    f"{record_template.fields}"
                )
            object.__setattr__(self, "_record_template", record_template)

        super().__init__(name=name)

        # Weights
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)
        w = self._w.normalized

        # Bandwidth (Silverman's rule default)
        if bandwidth is not None:
            bw = jnp.broadcast_to(jnp.asarray(bandwidth, dtype=samples.dtype), (d,))
        else:
            std = jnp.sqrt(self._w.variance(samples))
            # Silverman's rule: n^{-1/(d+4)} * std
            silverman_factor = n ** (-1.0 / (d + 4))
            bw = silverman_factor * jnp.maximum(std, 1e-8)
        self._bandwidth = bw

        # Build the TFP mixture distribution
        if d == 1:
            components = tfd.Normal(
                loc=samples[:, 0],
                scale=bw[0],
            )
        else:
            components = tfd.MultivariateNormalDiag(
                loc=samples,
                scale_diag=jnp.broadcast_to(bw, (n, d)),
            )
        self._tfp_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=w),
            components_distribution=components,
        )

    # -- KDE-specific properties -----------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def num_atoms(self) -> int:
        """Number of kernel centres (atoms) backing the KDE."""
        return self._samples.shape[0]

    @property
    def support(self) -> Constraint:
        return real

    # -- sampling & density (template-aware overrides) ------------------------
    #
    # When ``_record_template`` is multi-field, sample output is unflattened
    # back into ``NumericRecord`` / ``NumericRecordArray`` keyed by the
    # template, and log_prob accepts both structured and flat inputs. Single-
    # field auto-templates fall through to the TFP base class behaviour, so
    # existing call sites are unchanged.

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        flat = self._tfp_dist.sample(seed=key, sample_shape=sample_shape)
        tpl = getattr(self, "_record_template", None)
        if tpl is None or len(tpl.fields) <= 1:
            return flat
        return NumericRecordDistribution.unflatten_value(flat, template=tpl)

    def _log_prob(self, value: Any) -> Array:
        tpl = getattr(self, "_record_template", None)
        if tpl is not None and len(tpl.fields) > 1:
            if isinstance(value, (Record, NumericRecord, NumericRecordArray)):
                value = NumericRecordDistribution.flatten_value(value)
        return self._tfp_dist.log_prob(jnp.asarray(value))

    # -- factories ------------------------------------------------------------

    @classmethod
    def from_empirical(
        cls,
        source: Any,
        *,
        bandwidth: ArrayLike | None = None,
        name: str | None = None,
    ) -> "KDEDistribution":
        """Build a KDE from a :class:`RecordEmpiricalDistribution` source.

        Reuses the source's stored samples, weights, and record template,
        so the resulting KDE preserves the source's named-field structure
        end-to-end. Works for any subclass (notably
        :class:`~probpipe.inference.ApproximateDistribution`).

        Parameters
        ----------
        source : RecordEmpiricalDistribution
            Empirical or approximate distribution with stored samples.
        bandwidth : array-like or None
            Per-dimension bandwidth (see :class:`KDEDistribution`).
        name : str or None
            Distribution name; defaults to ``source.name``.
        """
        if not isinstance(source, RecordEmpiricalDistribution):
            raise TypeError(
                f"from_empirical requires a RecordEmpiricalDistribution "
                f"(or subclass); got {type(source).__name__}"
            )
        name = name or source.name
        tpl = source.record_template
        if len(tpl.fields) == 1:
            field = tpl.fields[0]
            arr = source.samples[field]
            return cls(arr, weights=source._w, bandwidth=bandwidth, name=name)
        return cls(
            source.flat_samples,
            weights=source._w,
            bandwidth=bandwidth,
            record_template=tpl,
            name=name,
        )

    def __repr__(self) -> str:
        return (
            f"KDEDistribution(num_atoms={self.num_atoms}, "
            f"event_shape={self.event_shape})"
        )
