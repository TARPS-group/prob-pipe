"""JointEmpirical --- weighted joint samples distribution.

Stores per-component sample arrays (all with the same number of rows)
and optional weights.  Sampling resamples rows jointly, preserving
correlation between components.
"""

from __future__ import annotations

from types import MappingProxyType

import jax
import jax.numpy as jnp
from .._utils import prod

from ..custom_types import Array, ArrayLike, PRNGKey
from .._weights import Weights
from ..core.distribution import (
    ArrayDistribution,
    ArrayEmpiricalDistribution,
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


class JointEmpirical(RecordDistribution, SupportsSampling, SupportsLogProb, SupportsMean, SupportsVariance, SupportsConditioning):
    """
    Joint distribution from weighted joint samples.

    Stores per-component sample arrays (all with the same number of rows)
    and optional weights.  Sampling resamples rows jointly, preserving
    correlation between components.

    When used in broadcasting enumeration, the joint is treated as a single
    unit with ``n`` samples (no cartesian decomposition).

    Parameters
    ----------
    weights : array-like, :class:`~probpipe.Weights`, or None
        Non-negative sample weights (normalized internally).  A pre-built
        :class:`~probpipe.Weights` object is also accepted.  Mutually
        exclusive with *log_weights*.
    log_weights : array-like, :class:`~probpipe.Weights`, or None
        Log-unnormalized sample weights.  A pre-built
        :class:`~probpipe.Weights` object is also accepted.  Mutually
        exclusive with *weights*.
    name : str, optional
        Distribution name.
    **samples : array-like
        Named component sample arrays.  Each must have the same number of
        rows (first dimension = ``n``).
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

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

        # Convert and validate shapes
        converted: dict[str, Array] = {}
        n: int | None = None
        for cname, arr in samples.items():
            arr = jnp.asarray(arr, dtype=jnp.float32)
            if arr.ndim == 0:
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
            converted[cname] = arr

        self._joint_samples = converted
        self._n = n
        if name is None:
            name = "joint_empirical(" + ",".join(sorted(samples.keys())) + ")"
        super().__init__(name=name)
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)

        # Build _components as ArrayEmpiricalDistribution per component
        # (JointDistribution requires ArrayDistribution leaves for shape introspection)
        comp_dists: dict[str, ArrayEmpiricalDistribution] = {}
        for cname, arr in self._joint_samples.items():
            comp_dists[cname] = ArrayEmpiricalDistribution(
                arr, weights=self._w, name=cname,
            )
        self._components = comp_dists
        self._record_template = _build_record_template(self._components)

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
    def component_names(self) -> tuple[str, ...]:
        """Component names in insertion order."""
        return tuple(self._components.keys())

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-component event shapes from component distributions."""
        return {k: v.event_shape for k, v in self._components.items()}

    # flatten_value / unflatten_value inherited from RecordDistribution

    @property
    def components(self):
        """Read-only view of the component distributions."""
        return MappingProxyType(self._components)

    def _sample_one(self, key: PRNGKey) -> Record:
        return self._sample_joint_rows(key, ())

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ):
        return self._sample_joint_rows(key, sample_shape)

    def _sample_joint_rows(
        self, key: PRNGKey, sample_shape: tuple[int, ...]
    ):
        """Resample rows jointly, preserving correlation."""
        from ..core._record_array import NumericRecordArray
        n_draws = prod(sample_shape)
        indices = self._w.choice(key, shape=(n_draws,))
        result = {}
        for cname, arr in self._joint_samples.items():
            drawn = arr[indices]
            if sample_shape:
                result[cname] = drawn.reshape(sample_shape + arr.shape[1:])
            else:
                result[cname] = drawn.squeeze(axis=0)
        if sample_shape:
            return NumericRecordArray(
                result, batch_shape=sample_shape,
                template=self.record_template,
            )
        return Record(result)

    def _log_prob(self, value) -> Array:
        """Gaussian-approximation log-density (same as EmpiricalDistribution).

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
        """Flat mean vector (for internal use by log_prob)."""
        parts = []
        for cname, arr in self._joint_samples.items():
            parts.append(self._w.mean(arr).reshape(-1))
        return jnp.concatenate(parts)

    def _flat_variance(self) -> Array:
        """Flat variance vector (for internal use by log_prob)."""
        parts = []
        for cname, arr in self._joint_samples.items():
            arr_flat = arr.reshape(self._n, -1)
            parts.append(self._w.variance(arr_flat))
        return jnp.concatenate(parts)

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
        return _mc_expectation(self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist)

    def _condition_on(self, observed=None, /, **kwargs):
        observed_leaves = _parse_condition_args(self, observed, kwargs)
        return self._condition_on_impl(observed_leaves)

    def _condition_on_impl(
        self, observed_leaves: dict[KeyPath, ArrayLike],
    ) -> "JointEmpirical":
        """Remove conditioned components and return a new JointEmpirical.

        Since ``JointEmpirical`` stores raw sample arrays keyed by name,
        conditioning simply drops those components from the joint sample
        matrix (preserving the row-wise correlation among the remaining
        components).

        .. note::

            ``JointEmpirical`` only supports **flat dicts** (no nesting).
            All key paths must be length-1 (top-level component names).
        """
        # Enforce flat-only
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

        result = JointEmpirical(
            **remaining_samples,
            weights=self._w,
            name=self._name,
        )
        result.with_source(Provenance(
            "condition_on", parents=(self,),
            metadata={"conditioned": list(observed_names)},
        ))
        return result
