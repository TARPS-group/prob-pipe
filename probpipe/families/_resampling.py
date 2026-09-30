"""The resampling families: the smoothing kernels of a kernel density estimate.

A kernel density estimate smooths its atoms with a **smoothing kernel**: a
mean-zero density ``K`` recentered at each atom and scaled by the bandwidth, so
its law is the weighted mixture ``Σᵢ wᵢ h⁻ᵈ K((x − xᵢ)/h)``. A kernel class
builds the bank of placed copies through one uniform constructor, so the
estimate holds the kernel class and never reads kernel-specific parameters.

Provides:
  - ``SmoothingKernel`` – a bank of mean-zero kernel copies, one per center.
  - ``GaussianKernel`` – the standard normal kernel.
  - ``EpanechnikovKernel`` – the product Epanechnikov kernel, compactly
    supported on ``[-1, 1]`` in each coordinate.

The bootstrap replicate is defined in :mod:`probpipe.core._empirical`, and the
kernel density estimate in :mod:`probpipe.distributions.kde`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from ..core._numeric_record import NumericRecord
from ..core._numeric_record_batch import NumericRecordBatch

if TYPE_CHECKING:
    from ..core._specs import NumericRecordSpec
    from ..custom_types import Array, ArrayLike, PRNGKey

__all__ = ["EpanechnikovKernel", "GaussianKernel", "SmoothingKernel"]


def _flat_centers(centers: ArrayLike | NumericRecordBatch) -> Array:
    """The centers as an array ``(n, *event)``, with a record batch flattened to ``(n, d)``.

    Raises
    ------
    ValueError
        If the centers have no leading atom axis or no atoms.
    """
    if isinstance(centers, NumericRecordBatch):
        if len(centers.batch_shape) != 1:
            raise ValueError(
                f"the centers of a kernel bank form one batch axis of atoms, got a batch of "
                f"shape {centers.batch_shape}"
            )
        array = jnp.asarray(centers.to_vector())
    else:
        array = jnp.asarray(centers)
    if array.ndim == 0:
        raise ValueError("the centers of a kernel bank need a leading axis of atoms")
    if array.shape[0] == 0:
        raise ValueError("a kernel bank has at least one center")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.result_type(float))
    return array


def _record_scales(scales: NumericRecord, fields: NumericRecordSpec | None) -> Array:
    """A record of scales as one scale per coordinate, in the order of the centers' fields.

    Each field of *scales* is matched to the centers' field at the same leaf
    path and broadcast over that field's coordinates.

    Raises
    ------
    ValueError
        If the centers are not records, the leaf paths of *scales* are not
        those of the centers, or a field's scale does not broadcast over that
        field's coordinates.
    """
    if fields is None:
        raise ValueError(
            "a record of scales matches the fields of record centers, but the centers are an "
            "array; pass the scales as an array"
        )
    expected, given = list(fields), list(scales)
    if set(given) != set(expected):
        raise ValueError(
            f"the scales' fields {given} are not the centers' fields {expected}: "
            f"missing {sorted(set(expected) - set(given))}, "
            f"unexpected {sorted(set(given) - set(expected))}"
        )
    shapes = fields.leaf_shapes
    blocks = []
    for path in expected:
        scale = jnp.asarray(scales[path])
        try:
            block = jnp.broadcast_to(scale, shapes[path])
        except ValueError:
            raise ValueError(
                f"the scale of field {path!r}, of shape {scale.shape}, does not broadcast over "
                f"the field's shape {shapes[path]}"
            ) from None
        blocks.append(jnp.reshape(block, -1))
    return jnp.concatenate(blocks)


def _flat_scales(
    scales: ArrayLike | NumericRecord, centers: Array, fields: NumericRecordSpec | None
) -> Array:
    """The scales broadcast to the centers' shape ``(n, *event)``.

    *fields* is the element spec of record centers, and ``None`` for array
    centers. A record of scales is matched to it by :func:`_record_scales`.

    Raises
    ------
    ValueError
        If a record of scales does not match the centers' fields, the scales do
        not broadcast against the centers, or a concrete scale is not positive.
    """
    if isinstance(scales, NumericRecord):
        array = _record_scales(scales, fields)
    else:
        array = jnp.asarray(scales)
    try:
        array = jnp.broadcast_to(array, centers.shape).astype(centers.dtype)
    except ValueError:
        raise ValueError(
            f"scales of shape {array.shape} do not broadcast against centers of shape "
            f"{centers.shape}; pass one scale, one per coordinate, or one per center and "
            f"coordinate"
        ) from None
    if not isinstance(array, jax.core.Tracer) and not bool(np.all(np.asarray(array) > 0)):
        raise ValueError("every scale of a kernel bank must be positive")
    return array


class SmoothingKernel(ABC):
    """A bank of mean-zero kernel copies, one placed at each center.

    A smoothing kernel is a mean-zero density ``K`` over the coordinates of one
    atom. The copy placed at center ``c`` with scales ``h`` has the density
    ``x ↦ K((x − c) / h) / ∏ⱼ hⱼ``, where the product runs over the coordinates.
    :meth:`build_kernels` is the uniform constructor: it takes the centers and
    the scales and returns the bank, whatever the concrete kernel, so a kernel
    density estimate holds the kernel class and never reads kernel-specific
    parameters. The bank draws from indexed copies with :meth:`_sample` and
    scores a point under every copy with :meth:`_log_density`.

    Parameters
    ----------
    centers : ArrayLike or NumericRecordBatch
        One center per atom along the leading axis, ``(n, *event)``. A batch of
        numeric records is flattened to its coordinates, ``(n, d)``.
    scales : ArrayLike or NumericRecord
        The scales, broadcast against the centers: one scale, one per
        coordinate, or one per center and coordinate. A numeric record of
        scales requires record centers with the same fields, matched by leaf
        path. Each field's scale broadcasts over that field's coordinates, and
        the scales are flattened in the centers' field order.

    Attributes
    ----------
    variance : float
        The variance of ``K`` in each coordinate, so the copy with scale ``h``
        has the variance ``h² * variance`` in that coordinate.

    Raises
    ------
    ValueError
        If the centers have no atom axis, a record of scales does not have the
        centers' fields, or the scales do not broadcast against the centers or
        are not positive.
    """

    variance: ClassVar[float]

    def __init__(
        self, centers: ArrayLike | NumericRecordBatch, scales: ArrayLike | NumericRecord
    ) -> None:
        fields = centers.event_template if isinstance(centers, NumericRecordBatch) else None
        self._centers = _flat_centers(centers)
        self._scales = _flat_scales(scales, self._centers, fields)

    @classmethod
    @abstractmethod
    def build_kernels(
        cls, centers: ArrayLike | NumericRecordBatch, scales: ArrayLike | NumericRecord
    ) -> SmoothingKernel:
        """The bank of copies placed at *centers* with *scales*, one copy per center.

        Returns
        -------
        SmoothingKernel
            An instance of this kernel class holding the placed copies.

        Raises
        ------
        ValueError
            If the centers have no atom axis, a record of scales does not have
            the centers' fields, or the scales do not broadcast against the
            centers or are not positive.
        """

    @abstractmethod
    def _sample(self, key: PRNGKey, index: Array) -> Array:
        """One draw from each indexed copy.

        Parameters
        ----------
        key : PRNGKey
            The key of the draws.
        index : Array
            Integer indices of copies, of any shape.

        Returns
        -------
        Array
            An array ``(*index.shape, *event)`` that holds the draw from each indexed copy.
        """

    @abstractmethod
    def _log_density(self, x: Array) -> Array:
        """The log-density of each copy at *x*, the scale Jacobian included.

        Parameters
        ----------
        x : Array
            A point ``(*event)``, or points ``(*batch, *event)``.

        Returns
        -------
        Array
            An array ``(*batch, n)``: the log-density of every copy at each point.
        """

    # -- shared by the concrete kernels -----------------------------------------

    def _standardized(self, x: Array) -> Array:
        """``(x − c) / h`` for every copy, an array ``(*batch, n, *event)``."""
        x = jnp.asarray(x, dtype=self._centers.dtype)
        event_ndim = self._centers.ndim - 1
        return (jnp.expand_dims(x, x.ndim - event_ndim) - self._centers) / self._scales

    def _summed_over_event(self, per_coordinate: Array) -> Array:
        """*per_coordinate* ``(*batch, n, *event)`` summed over the event axes."""
        event_ndim = self._centers.ndim - 1
        if event_ndim == 0:
            return per_coordinate
        return jnp.sum(per_coordinate, axis=tuple(range(-event_ndim, 0)))

    def _placed(self, index: Array, standard: Array) -> Array:
        """Standard draws *standard* ``(*index.shape, *event)`` moved to the indexed copies."""
        index = jnp.asarray(index)
        return self._centers[index] + self._scales[index] * standard

    def _log_scale(self) -> Array:
        """``Σⱼ log hⱼ`` of every copy, an array ``(n,)``."""
        log_scales = jnp.log(self._scales)
        return log_scales.reshape(log_scales.shape[0], -1).sum(axis=-1)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_copies={self._centers.shape[0]})"


class GaussianKernel(SmoothingKernel):
    """The Gaussian smoothing kernel: a standard normal density in each coordinate.

    Its variance is one, so the copy with scale ``h`` is the normal law with the
    center as its mean and ``h`` as its standard deviation in each coordinate.
    Parameters and errors are those of :class:`SmoothingKernel`.
    """

    variance: ClassVar[float] = 1.0

    @classmethod
    def build_kernels(
        cls, centers: ArrayLike | NumericRecordBatch, scales: ArrayLike | NumericRecord
    ) -> GaussianKernel:
        """The bank of normal copies placed at *centers* with the standard deviations *scales*."""
        return cls(centers, scales)

    def _sample(self, key: PRNGKey, index: Array) -> Array:
        """One normal draw from each indexed copy, ``(*index.shape, *event)``."""
        index = jnp.asarray(index)
        shape = (*index.shape, *self._centers.shape[1:])
        return self._placed(index, jax.random.normal(key, shape, dtype=self._centers.dtype))

    def _log_density(self, x: Array) -> Array:
        """The normal log-density of each copy at *x*, ``(*batch, n)``."""
        u = self._standardized(x)
        per_coordinate = -0.5 * u**2 - 0.5 * math.log(2.0 * math.pi)
        return self._summed_over_event(per_coordinate) - self._log_scale()


class EpanechnikovKernel(SmoothingKernel):
    """The product Epanechnikov kernel: ``K(u) = ¾ (1 − u²)`` on ``[-1, 1]`` per coordinate.

    Its variance is one fifth in each coordinate. A copy has compact support, so
    its log-density is ``-inf`` beyond one scale from its center in any
    coordinate. Parameters and errors are those of :class:`SmoothingKernel`.
    """

    variance: ClassVar[float] = 0.2

    @classmethod
    def build_kernels(
        cls, centers: ArrayLike | NumericRecordBatch, scales: ArrayLike | NumericRecord
    ) -> EpanechnikovKernel:
        """The bank of Epanechnikov copies placed at *centers* with the half-widths *scales*."""
        return cls(centers, scales)

    def _sample(self, key: PRNGKey, index: Array) -> Array:
        """One draw from each indexed copy, ``(*index.shape, *event)``.

        A standard draw is ``2 sin(arcsin(2p − 1) / 3)`` for a uniform ``p``, which
        inverts the kernel's distribution function ``(2 + 3u − u³) / 4``.
        """
        index = jnp.asarray(index)
        shape = (*index.shape, *self._centers.shape[1:])
        p = jax.random.uniform(key, shape, dtype=self._centers.dtype)
        return self._placed(index, 2.0 * jnp.sin(jnp.arcsin(2.0 * p - 1.0) / 3.0))

    def _log_density(self, x: Array) -> Array:
        """The Epanechnikov log-density of each copy at *x*, ``(*batch, n)``."""
        u = self._standardized(x)
        inside = jnp.abs(u) < 1.0
        per_coordinate = jnp.where(
            inside, math.log(0.75) + jnp.log1p(-(jnp.where(inside, u, 0.0) ** 2)), -jnp.inf
        )
        return self._summed_over_event(per_coordinate) - self._log_scale()
