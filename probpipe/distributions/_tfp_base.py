"""TFPDistribution base class for distributions backed by TFP instances.

Factored out of ``core/distribution.py`` because no ``core/`` module
imports ``TFPDistribution`` – it is only used by the concrete
distribution modules in ``distributions/``.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from .._utils import prod
from ..core._distribution_base import Distribution
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
    _DistributionArrayBackend,
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

    # -- SupportsArrayBackend (fused storage for DistributionArray) ----------

    @classmethod
    def _make_array_backend(
        cls,
        *,
        name: str,
        batch_shape: tuple[int, ...],
        **batched_params: Any,
    ) -> "_TFPArrayBackend":
        """Construct a fused TFP-batched backend for ``DistributionArray``.

        Inherited automatically by every concrete TFP-backed distribution
        (``Normal``, ``Beta``, ``Gamma``, ``MultivariateNormal``, …); the
        same code path covers the whole family because the wrapped TFP
        constructor handles the per-class param-name mapping.

        See :class:`probpipe.core.protocols.SupportsArrayBackend` for the
        protocol contract. Phase 1 of PR-C — additive only; nothing yet
        forces users through this path.
        """
        return _TFPArrayBackend(
            dist_cls=cls,
            name=name,
            batch_shape=tuple(batch_shape),
            batched_params=dict(batched_params),
        )


# ---------------------------------------------------------------------------
# Fused storage backend for DistributionArray
# ---------------------------------------------------------------------------


class _TFPArrayBackend:
    """Fused TFP-batched backend for ``DistributionArray``.

    Owns one ``tfd.Distribution`` instance with TFP's native
    ``batch_shape != ()`` plus the constructor params used to make it,
    so per-cell materialisation (``cell(i)``) can construct a fresh
    *scalar* :class:`Distribution` with the row-``i`` slice of each
    param.

    Implementation strategy: the backend wraps a *single* ProbPipe
    ``Distribution`` instance constructed with the batched params
    (legal in Phase 1 — Phase 2 will need a bypass for the un-batched
    assertion this class introduces). Vectorised ops forward to that
    wrapped instance's TFP backend; ``cell(i)`` slices the params and
    runs the ordinary scalar constructor with a suffixed name.

    Not a :class:`Distribution` itself — the backend exists only as
    the contract between :meth:`TFPDistribution._make_array_backend`
    and :class:`~probpipe.DistributionArray`. See
    :class:`probpipe.core.protocols._DistributionArrayBackend`.

    Parameters
    ----------
    dist_cls : type
        The concrete ``TFPDistribution`` subclass (e.g., ``Normal``).
        Used to materialise per-cell scalars.
    name : str
        Base name. Per-cell scalars auto-suffix as ``f"{name}_{flat}"``
        where ``flat`` is the row-major flat index over ``batch_shape``.
    batch_shape : tuple of int
        Leading shape of the batched parameters.
    batched_params : dict[str, Any]
        Constructor kwargs for ``dist_cls`` with leading ``batch_shape``
        already applied. Scalars are passed through unchanged in
        ``cell(i)`` (broadcast across all cells).
    """

    def __init__(
        self,
        *,
        dist_cls: type,
        name: str,
        batch_shape: tuple[int, ...],
        batched_params: dict[str, Any],
    ) -> None:
        self._dist_cls = dist_cls
        self._name = name
        self._batch_shape = tuple(batch_shape)
        self._batched_params = batched_params
        # Construct the fused ProbPipe Distribution with the batched
        # params. This is legal in Phase 1; Phase 2's un-batched
        # assertion will need a private bypass here (the backend's
        # whole purpose is to hold a batched form).
        self._batched_dist: TFPDistribution = dist_cls(
            **batched_params,
            name=f"{name}__array_backend",
        )
        # Sanity check: the constructed TFP dist's batch_shape should
        # match what the caller said. Mismatches usually mean the
        # caller's params don't actually broadcast to ``batch_shape``.
        actual = tuple(self._batched_dist._tfp_dist.batch_shape)
        if actual != self._batch_shape:
            raise ValueError(
                f"_TFPArrayBackend: declared batch_shape={self._batch_shape} "
                f"but {dist_cls.__name__} with the given batched_params "
                f"produced TFP batch_shape={actual}. Check that every "
                f"batched parameter broadcasts to batch_shape."
            )

    # -- shape ---------------------------------------------------------------

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self._batched_dist._tfp_dist.event_shape)

    @property
    def dtype(self) -> jnp.dtype:
        return self._batched_dist._tfp_dist.dtype

    # -- per-cell materialisation -------------------------------------------

    def cell(self, index: int | tuple[int, ...]) -> Distribution:
        """Fabricate a fresh scalar :class:`Distribution` for cell ``index``.

        ``index`` may be a flat ``int`` (interpreted row-major over
        ``batch_shape``) or a ``tuple[int, ...]`` of axis-aligned
        indices. The returned distribution is fully scalar
        (``batch_shape == ()``) — no caching; each call re-runs the
        ordinary ``dist_cls(**scalar_params, name=...)`` constructor.
        """
        multi = self._normalize_index(index)
        flat = int(np.ravel_multi_index(multi, self._batch_shape)) if self._batch_shape else 0
        scalar_params = {
            key: _slice_param_at(value, multi)
            for key, value in self._batched_params.items()
        }
        return self._dist_cls(
            **scalar_params,
            name=f"{self._name}_{flat}",
        )

    def _normalize_index(
        self, index: int | tuple[int, ...]
    ) -> tuple[int, ...]:
        bshape = self._batch_shape
        if isinstance(index, (int, np.integer)) or hasattr(index, "__index__"):
            i = int(index)
            if not bshape:
                if i != 0:
                    raise IndexError(
                        f"_TFPArrayBackend.cell: scalar backend has only one "
                        f"cell; got index={i}."
                    )
                return ()
            if len(bshape) == 1:
                if not 0 <= i < bshape[0]:
                    raise IndexError(
                        f"_TFPArrayBackend.cell: index {i} out of range for "
                        f"batch_shape={bshape}."
                    )
                return (i,)
            return tuple(int(x) for x in np.unravel_index(i, bshape))
        idx = tuple(int(x) for x in index)
        if len(idx) != len(bshape):
            raise IndexError(
                f"_TFPArrayBackend.cell: index {idx} has rank {len(idx)} "
                f"but batch_shape={bshape} has rank {len(bshape)}."
            )
        for i, dim in zip(idx, bshape):
            if not 0 <= i < dim:
                raise IndexError(
                    f"_TFPArrayBackend.cell: index {idx} out of range for "
                    f"batch_shape={bshape}."
                )
        return idx

    # -- vectorised ops (forward to the wrapped batched distribution) -------

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        return self._batched_dist._sample(key, sample_shape)

    def _log_prob(self, value: ArrayLike) -> Array:
        return self._batched_dist._log_prob(value)

    def _mean(self) -> Array:
        return self._batched_dist._mean()

    def _variance(self) -> Array:
        return self._batched_dist._variance()

    def _cov(self) -> Array:
        return self._batched_dist._cov()

    def __repr__(self) -> str:
        return (
            f"_TFPArrayBackend({self._dist_cls.__name__}, "
            f"batch_shape={self._batch_shape}, name={self._name!r})"
        )


def _slice_param_at(
    value: Any, multi_index: tuple[int, ...]
) -> Any:
    """Index the leading ``len(multi_index)`` axes of ``value`` at ``multi_index``.

    A scalar / lower-rank value (broadcast across every cell of the
    batch) is passed through unchanged. The trailing event axes (if
    any) are preserved in the slice — e.g., for ``MultivariateNormal``
    with ``loc`` of shape ``(batch, d)`` and ``multi_index=(i,)``,
    returns ``loc[i]`` of shape ``(d,)``.
    """
    if not multi_index:
        return value
    arr = jnp.asarray(value)
    if arr.ndim < len(multi_index):
        # Lower-rank — value is broadcast across the batch axes.
        return value
    return arr[multi_index]
