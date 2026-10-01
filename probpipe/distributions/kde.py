"""Kernel density estimation distribution backed by TFP.

Provides :class:`KDEDistribution`, a Gaussian KDE that satisfies both
:class:`~probpipe.SupportsLogProb` and
:class:`~probpipe.SupportsSampling`.  Useful for
converting a sampling-only distribution (e.g., MCMC output) into one
that supports density evaluation.
"""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from .._dtype import _as_float_array
from .._weights import Weights
from ..core._empirical import RecordEmpiricalDistribution
from ..core._numeric_record import NumericRecord
from ..core._numeric_record_batch import NumericRecordBatch
from ..core._numeric_record_distribution import NumericRecordDistribution
from ..core._record_distribution import _record_with_leaves
from ..core._specs import NumericArraySpec, NumericRecordSpec, OutputSpec, RecordSpec
from ..core.constraints import Constraint, real
from ..core.record import Record
from ..custom_types import Array, ArrayLike
from ..families._backend import TFPDistribution
from ._capabilities import SupportsCovariance, SupportsMean, SupportsVariance

__all__ = ["KDEDistribution"]


class KDEDistribution(TFPDistribution):
    """Gaussian kernel density estimate as a ProbPipe distribution.

    Wraps a TFP ``MixtureSameFamily(Categorical, MultivariateNormalDiag)``
    to provide a smooth density approximation from a set of weighted
    samples.  Inherits all protocol implementations from
    :class:`TFPDistribution`.

    Parameters
    ----------
    name : str
        Distribution name for provenance.
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
    event_spec : OutputSpec, RecordSpec, or None
        The declaration of one draw, which the KDE completes with the term it
        draws. When ``None`` (the default), one draw is an array under
        ``name``. A record defines how the flat ``(n, d)`` sample matrix maps
        back to a structured ``NumericRecord`` / ``NumericRecordBatch``, and
        one draw is declared as that record, each array leaf declaring the
        samples' dtype and the real line as its support. Any other declaration
        is completed with the flat array one draw is.

    Raises
    ------
    ValueError
        If *event_spec* is a record whose flat width is not ``samples.shape[1]``,
        or any other declaration whose type does not unify with the flat array
        one draw is.
    """

    _backend_capabilities = frozenset({SupportsMean, SupportsVariance, SupportsCovariance})

    def __init__(
        self,
        name: str,
        samples: ArrayLike,
        weights: ArrayLike | Weights | None = None,
        *,
        log_weights: ArrayLike | Weights | None = None,
        bandwidth: ArrayLike | None = None,
        event_spec: OutputSpec | RecordSpec | None = None,
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

        # The KDE completes its declaration with the term it draws. A record
        # gives the draws its structure, after a check that its flat width
        # matches the samples' trailing dimension, and any other declaration
        # takes the flat array, which a one-column KDE draws as scalars.
        if event_spec is not None and not isinstance(event_spec, OutputSpec):
            event_spec = OutputSpec.default(event_spec, component=name)
        if event_spec is not None and event_spec.exposes_record:
            record = cast(RecordSpec, event_spec.spec)
            if isinstance(record, NumericRecordSpec):
                expected = record.vector_size
            else:
                expected = sum(
                    int(jnp.prod(jnp.array(shape))) if shape else 1
                    for shape in record.leaf_shapes.values()
                )
            if expected != d:
                raise ValueError(
                    f"event_spec vector_size ({expected}) does not match "
                    f"samples flat dimension ({d}); record fields={record.fields}"
                )
            declaration = event_spec.with_spec(_record_with_leaves(record, samples.dtype, real))
        else:
            array = NumericArraySpec((d,) if d > 1 else (), samples.dtype, real)
            declaration = array if event_spec is None else event_spec.with_spec(array)

        super().__init__(name, None, event_spec=declaration)

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

    def _event_support(self) -> Constraint:
        """The support of one draw: a Gaussian kernel density is positive everywhere."""
        return real

    # -- sampling & density ---------------------------------------------------
    #
    # A KDE that declares a record unflattens its draws into ``NumericRecord``
    # / ``NumericRecordBatch``, and its log_prob accepts structured and flat
    # inputs alike. A KDE that draws one array behaves as the TFP base class.

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        flat = self._tfp_dist.sample(seed=key, sample_shape=sample_shape)
        spec = self.event_spec.spec
        if isinstance(spec, NumericArraySpec):
            return flat
        return NumericRecordDistribution.unflatten_value(flat, template=spec)

    def _log_prob(self, value: Any) -> Array:
        if not isinstance(self.event_spec.spec, NumericArraySpec):
            if isinstance(value, (Record, NumericRecord, NumericRecordBatch)):
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
    ) -> KDEDistribution:
        """Build a KDE from a :class:`RecordEmpiricalDistribution` source.

        Reuses the source's stored samples, weights, and declared record, so
        the resulting KDE keeps the source's named fields, and a posterior's
        nested ones. Works for any subclass, such as
        :class:`~probpipe.inference.ApproximateDistribution`.

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
        # A posterior's target record keeps the nesting its stored chunks
        # flatten, an interim reader until the posterior declares the nesting.
        tpl = getattr(source, "_target_record", None)
        if tpl is None:
            tpl = source.event_spec.spec
        if len(tpl.fields) == 1:
            field = tpl.fields[0]
            arr = source.samples[field]
            return cls(name, arr, weights=source._w, bandwidth=bandwidth)
        return cls(
            name,
            source.flat_samples,
            weights=source._w,
            bandwidth=bandwidth,
            event_spec=tpl,
        )

    def __repr__(self) -> str:
        # The declaration, not the flat TFP backend, gives the shape of one draw.
        spec = self.event_spec.spec
        if isinstance(spec, NumericArraySpec):
            shape = f"event_shape={spec.shape}"
        else:
            shape = f"event_shapes={dict(spec.leaf_shapes)}"
        return f"KDEDistribution(num_atoms={self.num_atoms}, {shape})"
