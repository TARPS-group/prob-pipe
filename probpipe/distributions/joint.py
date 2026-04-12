"""Joint distributions built on ValuesDistribution.

All joint distributions in this module inherit from
:class:`~probpipe.core._values_distribution.ValuesDistribution` and
return :class:`~probpipe.core.values.Values` from ``_sample()``.
Component access uses ``_ValuesDistributionView`` via
``dist["name"]`` or ``dist.select("x", "y")``.

**Concrete classes:**

*  :class:`ProductDistribution` — independent leaf components.
*  :class:`SequentialJointDistribution` — autoregressive dependencies.
*  :class:`JointEmpirical` — joint empirical distribution from samples.
*  :class:`JointGaussian` — multivariate Gaussian with named components
   and closed-form conditioning.

**Dynamic protocol support:**

``ProductDistribution`` and ``TransformedDistribution`` dynamically
include ``SupportsLogProb``, ``SupportsMean``, and ``SupportsVariance``
only when all leaf components support them.

**Flat-dict limitation:**

:class:`SequentialJointDistribution`, :class:`JointEmpirical`, and
:class:`JointGaussian` support flat dicts only (no nesting).
:class:`ProductDistribution` supports nested dicts.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

import jax
import jax.numpy as jnp
from .._utils import prod, _auto_key

from ..custom_types import Array, ArrayLike, PRNGKey
from .._weights import Weights, weighted_mean, weighted_variance
from ..core.distribution import (
    ArrayDistribution,
    ArrayEmpiricalDistribution,
    EmpiricalDistribution,
    _vmap_sample,
    _mc_expectation,
)
from ..core._joint import (
    ProductDistribution,
    KeyPath,
    _build_values_template,
    _normalize_key,
    _walk_pytree,
    _component_key_paths,
    _collect_observed_leaves,
    _parse_condition_args,
)
from ..core._values_distribution import ValuesDistribution
from ..core.values import Values
from ..core.provenance import Provenance
from ..core.constraints import Constraint, real
from ..core.protocols import (
    SupportsConditioning,
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)

__all__ = [
    "ProductDistribution",
    "SequentialJointDistribution",
    "JointEmpirical",
    "JointGaussian",
]


def _flatten_values_batched(value: Values, event_shapes: dict[str, tuple[int, ...]]) -> Array:
    """Flatten a (possibly batched) Values into ``(*leading, event_size)``."""
    parts = []
    for name in sorted(event_shapes.keys()):
        arr = jnp.asarray(value[name])
        es = event_shapes[name]
        n_event = prod(es) if es else 1
        n_event_dims = len(es)
        if n_event_dims:
            leading = arr.shape[:arr.ndim - n_event_dims]
        else:
            leading = arr.shape
        parts.append(arr.reshape(*leading, n_event))
    return jnp.concatenate(parts, axis=-1)


def _unflatten_values_batched(flat: Array, event_shapes: dict[str, tuple[int, ...]]) -> Values:
    """Unflatten ``(*leading, event_size)`` back into a Values."""
    fields: dict[str, Array] = {}
    offset = 0
    for name in sorted(event_shapes.keys()):
        es = event_shapes[name]
        n_event = prod(es) if es else 1
        chunk = flat[..., offset:offset + n_event]
        if es:
            leading = flat.shape[:-1]
            fields[name] = chunk.reshape(*leading, *es)
        else:
            fields[name] = chunk.squeeze(axis=-1)
        offset += n_event
    return Values(fields)

class SequentialJointDistribution(ValuesDistribution, SupportsSampling, SupportsLogProb, SupportsConditioning):
    """
    Joint distribution with autoregressive (sequential) dependence.

    Components can be :class:`Distribution` instances (roots) or callables
    that receive previously-sampled values and return a ``Distribution``
    (conditionals).

    Example::

        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0),
            x=lambda z: Normal(loc=z, scale=0.5),
            y=lambda z, x: Normal(loc=z + x, scale=0.1),
        )

    Callable signatures are inspected: parameter names must match earlier
    component names.

    Parameters
    ----------
    name : str, optional
        Distribution name.
    **components : Distribution or Callable[..., Distribution]
        Named components in topological (dependency) order.
    """

    _sampling_cost = "medium"
    _preferred_orchestration = None

    def __init__(
        self,
        *,
        name: str | None = None,
        **components: ArrayDistribution | Callable[..., ArrayDistribution],
    ):
        if not components:
            raise ValueError("SequentialJointDistribution requires at least one component.")

        self._raw_components: dict[str, ArrayDistribution | Callable] = dict(components)
        self._name = name
        self._conditioned_names: frozenset[str] = frozenset()
        self._conditioned_values: dict[str, Array] = {}
        self._sampleable_error: str | None = None
        # Map callable component names to their dependency parameter names
        self._callable_parents: dict[str, tuple[str, ...]] = {}

        # Validate ordering: callable args must reference earlier names
        seen: list[str] = []
        for cname, comp in self._raw_components.items():
            if callable(comp) and not isinstance(comp, ArrayDistribution):
                params = list(inspect.signature(comp).parameters.keys())
                for p in params:
                    if p not in seen:
                        raise ValueError(
                            f"Component '{cname}' depends on '{p}', which "
                            f"is not defined before it. "
                            f"Available: {seen}"
                        )
                self._callable_parents[cname] = tuple(params)
            seen.append(cname)

        # Do a prototype forward pass to determine component distributions
        # and compute event shapes / slices
        proto_key = jax.random.PRNGKey(0)
        proto_structured = self._sample_sequential(proto_key, ())
        self._proto_components: dict[str, ArrayDistribution] = {}

        resolved: dict[str, ArrayDistribution] = {}
        for cname, comp in self._raw_components.items():
            if isinstance(comp, ArrayDistribution):
                resolved[cname] = comp
            else:
                # Resolve the callable with zero-valued parents to get shape info
                parent_vals = {}
                for prev_name in list(self._raw_components.keys()):
                    if prev_name == cname:
                        break
                    parent_vals[prev_name] = proto_structured[prev_name]
                sig = inspect.signature(comp)
                call_kw = {p: parent_vals[p] for p in sig.parameters if p in parent_vals}
                resolved[cname] = comp(**call_kw)
        self._proto_components = resolved

        # Build _components dict from resolved prototypes (for shape introspection)
        self._components = resolved
        self._values_template = _build_values_template(self._components)

    @staticmethod
    def _compute_sampleable_error(
        conditioned_names: frozenset[str],
        callable_parents: dict[str, tuple[str, ...]],
    ) -> str | None:
        """Return an error message if sampling is impossible, else None.

        A conditioned non-root component is sampleable only if all of its
        parents are also conditioned.  Root components have no parents, so
        conditioning on them is always sampleable.
        """
        bad = []
        for cname in conditioned_names:
            parents = callable_parents.get(cname, ())
            unconditioned = [p for p in parents if p not in conditioned_names]
            if unconditioned:
                bad.append((cname, unconditioned))
        if not bad:
            return None
        details = "; ".join(
            f"'{c}' has unconditioned parent(s) {ps}" for c, ps in bad
        )
        return (
            f"Cannot sample from this SequentialJointDistribution: "
            f"{details}. Forward sampling would draw these ancestors "
            f"from the prior, not the posterior. Condition on the "
            f"parent(s) as well, or use log_prob() to evaluate the "
            f"density."
        )

    def _check_sampleable(self) -> None:
        """Raise if sampling is not possible due to non-root conditioning."""
        if self._sampleable_error is not None:
            raise NotImplementedError(self._sampleable_error)

    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names in topological (insertion) order."""
        return tuple(self._components.keys())

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-component event shapes from component distributions."""
        return {k: v.event_shape for k, v in self._components.items()}

    def flatten_value(self, value: Values) -> Array:
        """Flatten a (possibly batched) Values to ``(*leading, event_size)``."""
        return _flatten_values_batched(value, self.event_shapes)

    def unflatten_value(self, flat: Array) -> Values:
        """Reconstruct a Values from a flat ``(*leading, event_size)`` array."""
        return _unflatten_values_batched(flat, self.event_shapes)

    @property
    def components(self):
        """Read-only view of the component distributions."""
        from types import MappingProxyType
        return MappingProxyType(self._components)

    def _sample_sequential(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...],
    ) -> dict[str, Array]:
        """Sample components sequentially, feeding earlier samples to later callables.

        Returns a dict of *all* components (including conditioned ones).
        Callers that expose results externally should filter to unconditioned.
        """
        self._check_sampleable()
        keys = jax.random.split(key, len(self._raw_components))
        sampled: dict[str, Array] = {}

        for subkey, (cname, comp) in zip(keys, self._raw_components.items()):
            if cname in self._conditioned_values:
                # Conditioned component: broadcast fixed value to sample_shape
                val = self._conditioned_values[cname]
                sampled[cname] = jnp.broadcast_to(val, sample_shape + val.shape)
            elif isinstance(comp, ArrayDistribution):
                # Root distribution: sample with sample_shape
                sampled[cname] = comp._sample(subkey, sample_shape)
            else:
                # Conditional: callable receives batched parent samples,
                # returning a batched distribution.  Sample with () since
                # the batch is already in the distribution's batch_shape.
                sig = inspect.signature(comp)
                call_kw = {p: sampled[p] for p in sig.parameters if p in sampled}
                dist = comp(**call_kw)
                sampled[cname] = dist._sample(subkey)

        return sampled

    def _sample_one(self, key: PRNGKey) -> Values:
        full = self._sample_sequential(key, ())
        return Values({k: v for k, v in full.items() if k not in self._conditioned_names})

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Values:
        full = self._sample_sequential(key, sample_shape)
        return Values({k: v for k, v in full.items() if k not in self._conditioned_names})

    def _eval_log_prob(self, value, *, components: str) -> Array:
        """Evaluate log-density over selected components.

        Parameters
        ----------
        value : dict[str, ArrayLike] or Values
            Dict of unconditioned component values.
        components : ``"all"`` or ``"unconditioned"``
            Which components to include in the sum.  ``"all"`` sums over
            every component (conditioned ones evaluated at their observed
            values), giving the unnormalized conditional.
            ``"unconditioned"`` sums only over unconditioned components
            (with conditioned values plugged in as parents), giving the
            normalized conditional when the Markov structure permits it.
        """
        if isinstance(value, Values):
            value = value.to_dict()
        structured = {k: jnp.asarray(v) for k, v in value.items()}

        # Add conditioned values so callables can receive them
        for cname, val in self._conditioned_values.items():
            structured[cname] = val

        total = None
        for cname, comp in self._raw_components.items():
            if components == "unconditioned" and cname in self._conditioned_names:
                continue
            val = structured[cname]
            if isinstance(comp, ArrayDistribution):
                lp = comp._log_prob(val)
            else:
                sig = inspect.signature(comp)
                call_kw = {p: structured[p] for p in sig.parameters if p in structured}
                cond_dist = comp(**call_kw)
                lp = cond_dist._log_prob(val)
            total = lp if total is None else total + lp

        return total

    def _log_prob(self, value: dict[str, ArrayLike]) -> Array:
        """Evaluate the normalized log-density.

        For an unconditioned joint, this is the full joint log p(x).

        After conditioning on a set of components whose parents are all
        also conditioned (i.e., the conditioned set forms a root
        sub-graph), the Markov structure makes the normalized conditional
        log p(unconditioned | conditioned) computable by summing only
        the unconditioned components' log-densities with conditioned
        values substituted for their parents.

        Raises ``NotImplementedError`` when conditioning makes the
        normalizing constant intractable (e.g., conditioning on a leaf
        whose parents are unconditioned).
        """
        if self._sampleable_error is not None:
            raise NotImplementedError(
                "log_prob is not available for this conditioned "
                "SequentialJointDistribution because the normalizing "
                "constant is intractable.  Use unnormalized_log_prob "
                "instead."
            )
        return self._eval_log_prob(value, components="unconditioned")

    def _unnormalized_log_prob(self, value: dict[str, ArrayLike]) -> Array:
        """Evaluate the (possibly unnormalized) log-density.

        For an unconditioned joint, this equals the full joint log p(x).
        After conditioning, this sums log-densities of *all* components
        (conditioned ones evaluated at their observed values), giving
        log p(unconditioned, conditioned=values), which is proportional
        to log p(unconditioned | conditioned=values).
        """
        return self._eval_log_prob(value, components="all")

    def _mean(self) -> Values:
        """Per-component means (approximate — uses prototype components).

        For sequential joints, the true marginal mean is not simply the
        per-component mean because later components depend on earlier
        samples.  This returns the prototype (prior-evaluated) means
        as an approximation.
        """
        return Values({k: v._mean() for k, v in self._proto_components.items()})

    def _variance(self) -> Values:
        """Per-component variances (approximate — uses prototype components)."""
        return Values({k: v._variance() for k, v in self._proto_components.items()})

    def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
        return _mc_expectation(self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist)

    def _condition_on(self, observed=None, /, **kwargs):
        observed_leaves = _parse_condition_args(self, observed, kwargs)
        return self._condition_on_impl(observed_leaves)

    def _condition_on_impl(
        self, observed_leaves: dict[KeyPath, ArrayLike],
    ) -> "SequentialJointDistribution":
        """Condition on observed component values.

        The resulting distribution is sampleable as long as every conditioned
        non-root component has all of its parents also conditioned (so that no
        unconditioned ancestor would be drawn from the prior instead of the
        posterior).  Root components have no parents, so conditioning on them
        is always sampleable.  If the sampleability condition is violated, a
        :class:`NotImplementedError` is raised at sample time.
        ``log_prob()`` always works regardless (returns the unnormalized
        conditional log-density).

        .. note::

            ``SequentialJointDistribution`` only supports **flat dicts**
            (no nesting).  All key paths must be length-1 (top-level
            component names).
        """
        # Enforce flat-only: all key paths must be length 1
        for path in observed_leaves:
            if len(path) != 1:
                raise TypeError(
                    f"SequentialJointDistribution only supports flat "
                    f"(non-nested) components.  Cannot condition on "
                    f"nested key path {path!r}."
                )
        # Convert {("x",): val} → {"x": val}
        observed = {path[0]: val for path, val in observed_leaves.items()}

        all_conditioned = self._conditioned_names | frozenset(observed)
        if len(all_conditioned) >= len(self._raw_components):
            raise ValueError(
                "Cannot condition on all component distributions — "
                "at least one must remain unconditioned."
            )

        result = SequentialJointDistribution.__new__(SequentialJointDistribution)
        result._raw_components = dict(self._raw_components)  # originals unchanged
        result._name = self._name
        result._proto_components = dict(self._proto_components)
        result._callable_parents = self._callable_parents
        result._conditioned_names = all_conditioned
        result._conditioned_values = {
            **self._conditioned_values,
            **{k: jnp.asarray(v, dtype=jnp.float32) for k, v in observed.items()},
        }
        result._sampleable_error = self._compute_sampleable_error(
            result._conditioned_names, result._callable_parents,
        )

        # Expose only unconditioned components
        unconditioned = {
            k: v for k, v in self._proto_components.items()
            if k not in result._conditioned_names
        }
        result._components = unconditioned
        result._values_template = _build_values_template(unconditioned)

        result.with_source(Provenance(
            "condition_on", parents=(self,),
            metadata={"conditioned": list(observed)},
        ))
        return result

    def __repr__(self) -> str:
        parts = []
        for k, v in self._raw_components.items():
            if isinstance(v, ArrayDistribution):
                parts.append(f"{k}={type(v).__name__}")
            else:
                parts.append(f"{k}=<callable>")
        comp_str = ", ".join(parts)
        name_str = f", name='{self._name}'" if self._name else ""
        return f"SequentialJointDistribution({comp_str}{name_str})"


# ---------------------------------------------------------------------------
# JointEmpirical — weighted joint samples
# ---------------------------------------------------------------------------

class JointEmpirical(ValuesDistribution, SupportsSampling, SupportsLogProb, SupportsMean, SupportsVariance, SupportsConditioning):
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
        self._name = name
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)

        # Build _components as ArrayEmpiricalDistribution per component
        # (JointDistribution requires ArrayDistribution leaves for shape introspection)
        comp_dists: dict[str, ArrayEmpiricalDistribution] = {}
        for cname, arr in self._joint_samples.items():
            comp_dists[cname] = ArrayEmpiricalDistribution(
                arr, weights=self._w, name=cname,
            )
        self._components = comp_dists
        self._values_template = _build_values_template(self._components)

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

    def flatten_value(self, value: Values) -> Array:
        """Flatten a (possibly batched) Values to ``(*leading, event_size)``."""
        return _flatten_values_batched(value, self.event_shapes)

    def unflatten_value(self, flat: Array) -> Values:
        """Reconstruct a Values from a flat ``(*leading, event_size)`` array."""
        return _unflatten_values_batched(flat, self.event_shapes)

    @property
    def components(self):
        """Read-only view of the component distributions."""
        from types import MappingProxyType
        return MappingProxyType(self._components)

    def _sample_one(self, key: PRNGKey) -> Values:
        return self._sample_joint_rows(key, ())

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Values:
        return self._sample_joint_rows(key, sample_shape)

    def _sample_joint_rows(
        self, key: PRNGKey, sample_shape: tuple[int, ...]
    ) -> Values:
        """Resample rows jointly, preserving correlation."""
        n_draws = prod(sample_shape)
        indices = self._w.choice(key, shape=(n_draws,))
        result = {}
        for cname, arr in self._joint_samples.items():
            drawn = arr[indices]
            if sample_shape:
                result[cname] = drawn.reshape(sample_shape + arr.shape[1:])
            else:
                result[cname] = drawn.squeeze(axis=0)
        return Values(result)

    def _log_prob(self, value) -> Array:
        """Gaussian-approximation log-density (same as EmpiricalDistribution).

        Evaluates a diagonal Gaussian approximation in the flat space.
        """
        if not isinstance(value, Values):
            value = Values(value)
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

    def _mean(self) -> Values:
        """Per-component weighted means."""
        return Values({
            cname: self._w.mean(arr)
            for cname, arr in self._joint_samples.items()
        })

    def _variance(self) -> Values:
        """Per-component weighted variances."""
        return Values({
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
                "Cannot condition on all component distributions — "
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


# ---------------------------------------------------------------------------
# JointGaussian — analytical joint Gaussian with cross-covariance
# ---------------------------------------------------------------------------

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
        from types import MappingProxyType
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
        # Convert {("x",): val} → {"x": val}
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
                "Cannot condition on all component distributions — "
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


# ---------------------------------------------------------------------------
# Pytree registration
# ---------------------------------------------------------------------------

def _product_flatten(dist):
    """Flatten a ProductDistribution for JAX pytree registration.

    Stores the leaf ArrayDistributions as children and the component
    pytree structure (treedef) + name as auxiliary data.  This handles
    both flat and nested component dicts.
    """
    leaves = jax.tree.leaves(dist._components)
    comp_treedef = jax.tree.structure(dist._components)
    aux = (comp_treedef, dist._name)
    return leaves, aux


def _product_unflatten(aux, children):
    """Unflatten a ProductDistribution from JAX pytree data.

    Reconstructs the component pytree from the stored treedef and
    then passes it to the constructor as keyword arguments.
    """
    comp_treedef, name = aux
    components = jax.tree.unflatten(comp_treedef, children)
    return ProductDistribution(**components, name=name)


jax.tree_util.register_pytree_node(
    ProductDistribution,
    _product_flatten,
    _product_unflatten,
)
