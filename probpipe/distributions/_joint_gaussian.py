"""JointGaussian --- analytical joint Gaussian with cross-covariance.

Supports exact analytical conditioning via :meth:`condition_on`.
"""

from __future__ import annotations

from types import MappingProxyType

import jax
import jax.numpy as jnp
from .._utils import prod

from ..custom_types import Array, ArrayLike, PRNGKey
from ..core.distribution import (
    ArrayDistribution,
    _mc_expectation,
)
from ..core._values_distribution import ValuesDistribution, _build_values_template
from ..core.values import Values
from ..core.provenance import Provenance
from ..core.protocols import (
    SupportsConditioning,
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from ._joint_utils import (
    KeyPath,
    _parse_condition_args,
    _flatten_values_batched,
    _unflatten_values_batched,
)


class JointGaussian(ValuesDistribution, SupportsSampling, SupportsLogProb, SupportsMean, SupportsVariance, SupportsCovariance, SupportsConditioning):
    """
    Joint Gaussian distribution with named components and cross-covariance.

    Supports exact analytical conditioning via :meth:`condition_on`.

    Parameters
    ----------
    mean : array-like, shape ``(d,)``
        Full (flat) mean vector.
    cov : array-like, shape ``(d, d)``
        Full (flat) covariance matrix.
    name : str, optional
        Distribution name.
    **component_shapes : int
        Named components with their dimensionality.  The sum of all
        dimensions must equal ``d``.

    Examples
    --------
    >>> joint = JointGaussian(
    ...     mean=jnp.array([0.0, 0.0, 1.0, 2.0]),
    ...     cov=jnp.eye(4),
    ...     x=1,    # x is 1-dimensional
    ...     yz=3,   # yz is 3-dimensional
    ... )
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __init__(
        self,
        *,
        mean: ArrayLike,
        cov: ArrayLike,
        name: str | None = None,
        **component_shapes: int,
    ):
        if not component_shapes:
            raise ValueError("JointGaussian requires at least one component.")

        mean = jnp.asarray(mean, dtype=jnp.float32)
        cov = jnp.asarray(cov, dtype=jnp.float32)

        total_dim = sum(component_shapes.values())
        if mean.shape != (total_dim,):
            raise ValueError(
                f"mean shape {mean.shape} does not match total dimension "
                f"({total_dim},) from component shapes {component_shapes}."
            )
        if cov.shape != (total_dim, total_dim):
            raise ValueError(
                f"cov shape {cov.shape} does not match ({total_dim}, {total_dim})."
            )

        self._mean_vec = mean
        self._cov_mat = cov
        self._name = name
        self._component_shapes = dict(component_shapes)

        # Build slices and component MultivariateNormal distributions
        from .multivariate import MultivariateNormal as MVN

        slices = {}
        components = {}
        offset = 0
        for cname, dim in self._component_shapes.items():
            sl = slice(offset, offset + dim)
            slices[cname] = sl
            components[cname] = MVN(
                loc=mean[sl],
                cov=cov[sl, sl],
                name=cname,
            )
            offset += dim

        self._components = components
        self._component_slices = slices  # still needed for Gaussian conditioning
        self._values_template = _build_values_template(self._components)
        self._total_dim = total_dim  # still needed for Gaussian conditioning

    @property
    def mean_vector(self) -> Array:
        """Full mean vector."""
        return self._mean_vec

    @property
    def covariance(self) -> Array:
        """Full covariance matrix."""
        return self._cov_mat

    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names in insertion order."""
        return tuple(self._component_shapes.keys())

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-component event shapes."""
        return {k: (v,) for k, v in self._component_shapes.items()}

    def flatten_value(self, value: Values) -> Array:
        """Flatten a (possibly batched) Values to ``(*leading, event_size)``."""
        return _flatten_values_batched(value, self.event_shapes)

    def unflatten_value(self, flat: Array) -> Values:
        """Reconstruct a Values from a flat ``(*leading, event_size)`` array."""
        return _unflatten_values_batched(flat, self.event_shapes)

    @property
    def components(self):
        """Read-only view of the component distributions."""
        return MappingProxyType(self._components)

    def _sample_one(self, key: PRNGKey) -> Values:
        from .multivariate import MultivariateNormal as MVN
        full_mvn = MVN(loc=self._mean_vec, cov=self._cov_mat)
        flat = full_mvn._sample(key)
        return self._unflatten_flat_vec(flat)

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Values:
        from .multivariate import MultivariateNormal as MVN
        full_mvn = MVN(loc=self._mean_vec, cov=self._cov_mat)
        flat = full_mvn._sample(key, sample_shape)
        return self._unflatten_flat_vec(flat)

    def _unflatten_flat_vec(self, flat: Array) -> Values:
        """Split a flat Gaussian sample vector into per-component arrays."""
        result = {}
        for cname in self._component_shapes:
            sl = self._component_slices[cname]
            result[cname] = flat[..., sl]
        return Values(result)

    def _log_prob(self, value) -> Array:
        if not isinstance(value, Values):
            value = Values(value)
        from .multivariate import MultivariateNormal as MVN
        full_mvn = MVN(loc=self._mean_vec, cov=self._cov_mat)
        flat = self.flatten_value(value)
        return full_mvn._log_prob(flat)

    def _mean(self) -> Values:
        result = {}
        for cname in self._component_shapes:
            sl = self._component_slices[cname]
            result[cname] = self._mean_vec[sl]
        return Values(result)

    def _variance(self) -> Values:
        diag = jnp.diag(self._cov_mat)
        result = {}
        for cname in self._component_shapes:
            sl = self._component_slices[cname]
            result[cname] = diag[sl]
        return Values(result)

    def _cov(self) -> Array:
        """Full covariance matrix."""
        return self._cov_mat

    def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
        return _mc_expectation(self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist)

    def _condition_on(self, observed=None, /, **kwargs):
        observed_leaves = _parse_condition_args(self, observed, kwargs)
        return self._condition_on_impl(observed_leaves)

    def _condition_on_impl(
        self, observed_leaves: dict[KeyPath, ArrayLike],
    ) -> "JointGaussian":
        """Condition on observed component values using exact Gaussian formulas.

        Returns a new :class:`JointGaussian` over the remaining (unobserved)
        components.

        .. note::

            ``JointGaussian`` only supports **flat dicts** (no nesting)
            because the conditioning math operates on flat index slices
            of the joint mean vector and covariance matrix.
        """
        # Enforce flat-only: all key paths must be length 1
        for path in observed_leaves:
            if len(path) != 1:
                raise TypeError(
                    f"JointGaussian only supports flat (non-nested) "
                    f"components.  Cannot condition on nested key path "
                    f"{path!r}."
                )
        # Convert {("x",): val} -> {"x": val}
        observed = {path[0]: val for path, val in observed_leaves.items()}

        # Partition into observed (o) and unobserved (u) indices
        o_indices = []
        u_indices = []
        u_shapes: dict[str, int] = {}
        for cname, dim in self._component_shapes.items():
            sl = self._component_slices[cname]
            idx = list(range(sl.start, sl.stop))
            if cname in observed:
                o_indices.extend(idx)
            else:
                u_indices.extend(idx)
                u_shapes[cname] = dim

        if not u_shapes:
            raise ValueError(
                "Cannot condition on all component distributions --- "
                "at least one must remain unconditioned."
            )

        o_idx = jnp.array(o_indices)
        u_idx = jnp.array(u_indices)

        # Collect observed values into a single vector
        o_vals_parts = []
        for cname in self._component_shapes:
            if cname in observed:
                o_vals_parts.append(
                    jnp.asarray(observed[cname], dtype=jnp.float32).reshape(-1)
                )
        o_vals = jnp.concatenate(o_vals_parts)

        # Extract block matrices
        mu_u = self._mean_vec[u_idx]
        mu_o = self._mean_vec[o_idx]
        Sigma_uu = self._cov_mat[jnp.ix_(u_idx, u_idx)]
        Sigma_uo = self._cov_mat[jnp.ix_(u_idx, o_idx)]
        Sigma_oo = self._cov_mat[jnp.ix_(o_idx, o_idx)]

        # Gaussian conditioning: mu_u|o = mu_u + Sigma_uo @ Sigma_oo^{-1} @ (o - mu_o)
        Sigma_oo_inv = jnp.linalg.inv(Sigma_oo)
        cond_mean = mu_u + Sigma_uo @ Sigma_oo_inv @ (o_vals - mu_o)
        cond_cov = Sigma_uu - Sigma_uo @ Sigma_oo_inv @ Sigma_uo.T
        # Symmetrise for numerical stability
        cond_cov = (cond_cov + cond_cov.T) / 2

        result = JointGaussian(
            mean=cond_mean,
            cov=cond_cov,
            name=self._name,
            **u_shapes,
        )
        result.with_source(Provenance(
            "condition_on",
            parents=(self,),
            metadata={"conditioned": list(observed.keys())},
        ))
        return result

    def __repr__(self) -> str:
        comp_str = ", ".join(
            f"{k}={v}" for k, v in self._component_shapes.items()
        )
        name_str = f", name='{self._name}'" if self._name else ""
        return f"JointGaussian({comp_str}{name_str})"
