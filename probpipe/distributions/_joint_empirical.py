"""JointEmpirical --- weighted joint samples distribution.

Stores per-component sample arrays (all with the same number of rows)
and optional weights.  Sampling resamples rows jointly, preserving
correlation between components.

Two concrete classes:

* :class:`JointEmpirical` — generic base. Accepts numeric or object
  samples; claims only ``SupportsSampling`` and ``SupportsConditioning``.
* :class:`NumericJointEmpirical` — all fields numeric. Additionally
  claims ``SupportsLogProb`` (Gaussian approximation),
  ``SupportsMean``, ``SupportsVariance``.

Construct via ``JointEmpirical(...)`` — when every field is a numeric
array, the class dispatches in ``__new__`` to ``NumericJointEmpirical``
(same pattern as ``EmpiricalDistribution`` → ``NumericEmpiricalDistribution``).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np
from .._utils import prod, _is_numeric_array

from ..custom_types import Array, ArrayLike, PRNGKey
from .._weights import Weights
from ..core.distribution import (
    NumericRecordDistribution,
    NumericEmpiricalDistribution,
    _mc_expectation,
)
from ..core._record_distribution import RecordDistribution, _build_record_template
from ..core.record import Record
from ..core.provenance import Provenance
from ..core.protocols import (
    SupportsConditioning,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from ._joint_utils import (
    KeyPath,
    _parse_condition_args,
)

__all__ = ["JointEmpirical", "NumericJointEmpirical"]


class JointEmpirical(RecordDistribution, SupportsSampling, SupportsConditioning):
    """
    Joint distribution from weighted joint samples.

    Stores per-component sample arrays (all with the same number of rows)
    and optional weights. Sampling resamples rows jointly, preserving
    correlation between components.

    **Dynamic dispatch via ``__new__``:** when every field is a numeric
    array (numpy, JAX, or numeric scalar), constructing ``JointEmpirical``
    returns a :class:`NumericJointEmpirical` instance, which additionally
    supports log-prob (Gaussian approximation), mean, and variance. Fall
    through to this base class for mixed / opaque data (e.g. object-dtype
    arrays of labels).

    When used in broadcasting enumeration, the joint is treated as a single
    unit with ``n`` samples (no cartesian decomposition).

    Parameters
    ----------
    weights : array-like, :class:`~probpipe.Weights`, or None
        Non-negative sample weights (normalized internally). A pre-built
        :class:`~probpipe.Weights` object is also accepted. Mutually
        exclusive with *log_weights*.
    log_weights : array-like, :class:`~probpipe.Weights`, or None
        Log-unnormalized sample weights. A pre-built
        :class:`~probpipe.Weights` object is also accepted. Mutually
        exclusive with *weights*.
    name : str, optional
        Distribution name.
    **samples : array-like
        Named component sample arrays. Each must have the same number of
        rows (first dimension = ``n``).
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __new__(
        cls,
        *,
        weights: ArrayLike | Weights | None = None,
        log_weights: ArrayLike | Weights | None = None,
        name: str | None = None,
        **samples: ArrayLike,
    ):
        # Only auto-dispatch when someone calls ``JointEmpirical(...)``
        # directly; explicit ``NumericJointEmpirical(...)`` skips the
        # sniff and goes straight to the numeric subclass.
        if cls is JointEmpirical and samples:
            if all(_is_numeric_array(v) for v in samples.values()):
                return object.__new__(NumericJointEmpirical)
        return object.__new__(cls)

    def __init__(
        self,
        *,
        weights: ArrayLike | Weights | None = None,
        log_weights: ArrayLike | Weights | None = None,
        name: str | None = None,
        **samples: ArrayLike,
    ):
        if not samples:
            raise ValueError("JointEmpirical requires at least one component.")

        # Generic path: store samples as-is (numpy or jax arrays). Validate
        # that all components have the same leading row count. Numeric
        # coercion happens in NumericJointEmpirical.
        stored: dict[str, Any] = {}
        n: int | None = None
        for cname, arr in samples.items():
            if not hasattr(arr, "shape") or len(arr.shape) == 0:
                raise ValueError(
                    f"Component '{cname}' must have at least 1 dimension "
                    f"(first dim = number of samples)."
                )
            if n is None:
                n = arr.shape[0]
            elif arr.shape[0] != n:
                raise ValueError(
                    f"All components must have the same number of samples. "
                    f"First component has {n}, but '{cname}' has {arr.shape[0]}."
                )
            stored[cname] = arr

        self._joint_samples = stored
        self._n = n
        if name is None:
            name = "joint_empirical(" + ",".join(sorted(samples.keys())) + ")"
        super().__init__(name=name)
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)
        self._components = self._build_component_dists()
        self._record_template = (
            _build_record_template(self._components)
            if self._components is not None
            else None
        )

    # Hook for NumericJointEmpirical to override; base class returns None
    # because generic joint samples can't be expressed as per-component
    # NumericRecordDistribution leaves without numeric coercion.
    def _build_component_dists(self) -> dict[str, NumericRecordDistribution] | None:
        return None

    # -- Properties ---------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of joint samples."""
        return self._n

    @property
    def is_uniform(self) -> bool:
        return self._w.is_uniform

    @property
    def weights(self) -> Array:
        """Normalised weights, shape ``(n,)``."""
        return self._w.normalized

    @property
    def fields(self) -> tuple[str, ...]:
        """Component names in insertion order."""
        return tuple(self._joint_samples.keys())

    @property
    def components(self):
        """Read-only view of the component distributions (numeric case)."""
        if self._components is None:
            return None
        return MappingProxyType(self._components)

    # -- Sampling -----------------------------------------------------------

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ):
        return self._sample_joint_rows(key, sample_shape)

    def _sample_joint_rows(self, key: PRNGKey, sample_shape: tuple[int, ...]):
        """Resample rows jointly, preserving per-row correlation.

        The generic base returns a ``Record`` regardless of
        ``sample_shape`` (with batched fields for non-empty shapes).
        Subclasses override to return a typed batched container (e.g.
        ``NumericRecordArray``) when the leaves are numeric.
        """
        return Record(self._resample_rows(key, sample_shape))

    def _resample_rows(
        self, key: PRNGKey, sample_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        """Draw ``prod(sample_shape)`` row indices and slice each field.

        Used by both :class:`JointEmpirical` and
        :class:`NumericJointEmpirical` so that the index-and-slice logic
        lives in one place.
        """
        n_draws = prod(sample_shape)
        indices = self._w.choice(key, shape=(n_draws,))
        result: dict[str, Any] = {}
        for cname, arr in self._joint_samples.items():
            drawn = arr[indices]
            if sample_shape:
                try:
                    result[cname] = drawn.reshape(sample_shape + arr.shape[1:])
                except (TypeError, ValueError):
                    # Non-numeric object arrays don't always support reshape.
                    result[cname] = drawn
            else:
                result[cname] = (
                    drawn[0] if drawn.shape and drawn.shape[0] == 1 else drawn
                )
        return result

    # -- Conditioning -------------------------------------------------------

    def _condition_on(self, observed=None, /, **kwargs):
        observed_leaves = _parse_condition_args(self, observed, kwargs)
        return self._condition_on_impl(observed_leaves)

    def _condition_on_impl(self, observed_leaves: dict[KeyPath, ArrayLike]) -> "JointEmpirical":
        """Remove conditioned components and return a new JointEmpirical.

        Since ``JointEmpirical`` stores raw sample arrays keyed by name,
        conditioning simply drops those components from the joint sample
        matrix (preserving row-wise correlation among the remaining
        components).

        .. note::

            ``JointEmpirical`` only supports **flat dicts** (no nesting).
            All key paths must be length-1 (top-level component names).
        """
        for path in observed_leaves:
            if len(path) != 1:
                raise TypeError(
                    f"JointEmpirical only supports flat (non-nested) "
                    f"components.  Cannot condition on nested key path "
                    f"{path!r}."
                )
        observed_names = {path[0] for path in observed_leaves}

        remaining_samples = {
            cname: arr
            for cname, arr in self._joint_samples.items()
            if cname not in observed_names
        }
        if not remaining_samples:
            raise ValueError(
                "Cannot condition on all component distributions --- "
                "at least one must remain unconditioned."
            )

        result = type(self)(
            **remaining_samples,
            weights=self._w,
            name=self._name,
        )
        result.with_source(Provenance(
            "condition_on", parents=(self,),
            metadata={"conditioned": list(observed_names)},
        ))
        return result


# ---------------------------------------------------------------------------
# NumericJointEmpirical — all-numeric case, adds log_prob / mean / variance
# ---------------------------------------------------------------------------


class NumericJointEmpirical(JointEmpirical, SupportsLogProb, SupportsMean, SupportsVariance):
    """Joint empirical where every field is a numeric array.

    Subclass of :class:`JointEmpirical` that additionally implements
    :class:`~probpipe.core.protocols.SupportsLogProb` (via a diagonal
    Gaussian approximation), :class:`~probpipe.core.protocols.SupportsMean`,
    and :class:`~probpipe.core.protocols.SupportsVariance`.

    Construction coerces every field to ``jnp.float32``; fields that
    aren't numeric arrays raise ``TypeError``. Typically constructed via
    :class:`JointEmpirical`, which dispatches here automatically when all
    fields are numeric.
    """

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(
        self,
        *,
        weights: ArrayLike | Weights | None = None,
        log_weights: ArrayLike | Weights | None = None,
        name: str | None = None,
        **samples: ArrayLike,
    ):
        if not samples:
            raise ValueError("NumericJointEmpirical requires at least one component.")

        # Coerce every field to jnp.float32 up front. Non-numeric inputs
        # raise ``TypeError`` with a clear message before any bookkeeping.
        coerced: dict[str, Array] = {}
        for cname, arr in samples.items():
            if not _is_numeric_array(arr):
                raise TypeError(
                    f"NumericJointEmpirical: field {cname!r} must be a "
                    f"numeric array, got {type(arr).__name__}. Use "
                    f"JointEmpirical directly for non-numeric components."
                )
            coerced[cname] = jnp.asarray(arr, dtype=jnp.float32)

        super().__init__(
            weights=weights, log_weights=log_weights, name=name, **coerced,
        )

    def _build_component_dists(self) -> dict[str, NumericRecordDistribution]:
        return {
            cname: NumericEmpiricalDistribution(arr, weights=self._w, name=cname)
            for cname, arr in self._joint_samples.items()
        }

    # -- Sampling: return NumericRecordArray for batched draws --------------

    def _sample_joint_rows(self, key: PRNGKey, sample_shape: tuple[int, ...]):
        from ..core._record_array import NumericRecordArray
        result = self._resample_rows(key, sample_shape)
        if sample_shape:
            return NumericRecordArray(
                result, batch_shape=sample_shape,
                template=self.record_template,
            )
        return Record(result)

    # -- event_shapes (used by the record template) ------------------------

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-component event shapes."""
        return {k: v.event_shape for k, v in self._components.items()}

    # -- Log-prob (Gaussian approximation) ---------------------------------

    def _log_prob(self, value) -> Array:
        """Gaussian-approximation log-density (same as :class:`~probpipe.NumericEmpiricalDistribution`).

        Evaluates a diagonal Gaussian approximation in the flat space.
        """
        if not isinstance(value, Record):
            value = Record(value)
        flat = self.flatten_value(value)
        mu = self._flat_mean()
        var = self._flat_variance()
        var = jnp.maximum(var, 1e-12)
        log_norm = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * var))
        diff = flat - mu
        return log_norm - 0.5 * jnp.sum(diff**2 / var, axis=-1)

    def _flat_mean(self) -> Array:
        """Flat mean vector (internal helper for log_prob)."""
        parts = []
        for _, arr in self._joint_samples.items():
            parts.append(self._w.mean(arr).reshape(-1))
        return jnp.concatenate(parts)

    def _flat_variance(self) -> Array:
        """Flat variance vector (internal helper for log_prob)."""
        parts = []
        for _, arr in self._joint_samples.items():
            arr_flat = arr.reshape(self._n, -1)
            parts.append(self._w.variance(arr_flat))
        return jnp.concatenate(parts)

    # -- Moments -----------------------------------------------------------

    def _mean(self) -> Record:
        """Per-component weighted means."""
        return Record({
            cname: self._w.mean(arr)
            for cname, arr in self._joint_samples.items()
        })

    def _variance(self) -> Record:
        """Per-component weighted variances."""
        return Record({
            cname: self._w.variance(arr)
            for cname, arr in self._joint_samples.items()
        })

    def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
        return _mc_expectation(
            self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist,
        )
