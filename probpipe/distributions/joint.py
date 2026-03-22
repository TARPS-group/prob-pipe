"""Joint distributions and component views for correlated broadcasting."""
from __future__ import annotations

import inspect
from types import MappingProxyType
from typing import Any, Callable

import jax
import jax.numpy as jnp
import math

from ..custom_types import Array, ArrayLike, PRNGKey
from .distribution import Distribution, EmpiricalDistribution, Provenance, Constraint, real

__all__ = [
    "JointDistribution",
    "ProductDistribution",
    "SequentialJointDistribution",
    "JointEmpirical",
    "JointGaussian",
    "DistributionView",
    "ConditionedComponent",
]


# ---------------------------------------------------------------------------
# DistributionView — lightweight reference to a JointDistribution component
# ---------------------------------------------------------------------------

class DistributionView(Distribution):
    """
    A lightweight reference to a named component of a :class:`JointDistribution`.

    Broadcasting logic in :class:`~probpipe.core.node.Workflow` detects
    ``DistributionView`` instances and groups those sharing the same
    ``_parent`` so they are sampled jointly (preserving correlation).

    For standalone use (outside broadcasting), sampling draws from the
    parent and extracts this component.
    """

    def __init__(self, parent: JointDistribution, component_name: str):
        if component_name not in parent.component_names:
            raise KeyError(
                f"Component '{component_name}' not found in "
                f"{type(parent).__name__}. Available: {parent.component_names}"
            )
        self._parent = parent
        self._component_name = component_name
        self._component = parent.components[component_name]
        self._name = component_name

    # -- Distribution ABC --------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._component.event_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._component.batch_shape

    @property
    def dtype(self):
        return self._component.dtype

    @property
    def support(self) -> Constraint:
        return self._component.support

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        structured = self._parent.sample_structured(key, sample_shape)
        return structured[self._component_name]

    def log_prob(self, x: ArrayLike) -> Array:
        return self._component.log_prob(x)

    def mean(self) -> Array:
        return self._component.mean()

    def variance(self) -> Array:
        return self._component.variance()

    @classmethod
    def _from_distribution(cls, other, *, key, **kwargs):
        raise NotImplementedError(
            "Cannot convert to DistributionView; it is a structural reference."
        )

    @classmethod
    def _default_support(cls) -> Constraint:
        return real

    def __repr__(self) -> str:
        return (
            f"DistributionView(parent={type(self._parent).__name__}, "
            f"component='{self._component_name}')"
        )


# ---------------------------------------------------------------------------
# ConditionedComponent — a distribution fixed to an observed value
# ---------------------------------------------------------------------------

class ConditionedComponent(Distribution):
    """
    A distribution whose value is fixed (observed / conditioned on).

    Used internally by :meth:`JointDistribution.condition_on` to replace a
    component with a constant.  Sampling always returns the conditioned value;
    ``log_prob`` evaluates the base distribution at the conditioned value.
    """

    def __init__(self, base: Distribution, value: ArrayLike, *, name: str | None = None):
        self._base = base
        self._value = jnp.asarray(value, dtype=jnp.float32)
        self._name = name or getattr(base, "name", None)
        if self._value.shape != base.event_shape:
            raise ValueError(
                f"Pinned value shape {self._value.shape} does not match "
                f"base event_shape {base.event_shape}"
            )

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._base.event_shape

    @property
    def support(self) -> Constraint:
        return self._base.support

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        return jnp.broadcast_to(self._value, sample_shape + self.event_shape)

    def log_prob(self, x: ArrayLike) -> Array:
        # Log-prob is constant: the base distribution evaluated at the pinned value
        return self._base.log_prob(self._value)

    def mean(self) -> Array:
        return self._value

    def variance(self) -> Array:
        return jnp.zeros(self.event_shape, dtype=jnp.float32)

    @classmethod
    def _from_distribution(cls, other, *, key, **kwargs):
        raise NotImplementedError("Cannot convert to ConditionedComponent.")

    @classmethod
    def _default_support(cls) -> Constraint:
        return real

    def __repr__(self) -> str:
        return f"ConditionedComponent(base={type(self._base).__name__}, value={self._value})"


# ---------------------------------------------------------------------------
# JointDistribution — base class for named multi-component distributions
# ---------------------------------------------------------------------------

class JointDistribution(Distribution):
    """
    Base class for named multi-component distributions.

    Components are stored as an ordered mapping ``{name: Distribution}``.
    Access individual components via ``joint['name']`` (returns a
    :class:`DistributionView`) or ``joint.bind(a='x', b='y')`` for name
    remapping.

    The flat representation concatenates component samples along the last
    axis: shape ``(*sample_shape, total_dim)`` where
    ``total_dim = sum(prod(c.event_shape) for c in components)``.
    """

    def __init__(self, *, name: str | None = None, **components: Distribution):
        if not components:
            raise ValueError("JointDistribution requires at least one component.")
        for k, v in components.items():
            if not isinstance(v, Distribution):
                raise TypeError(
                    f"Component '{k}' must be a Distribution, got {type(v).__name__}"
                )
        self._components = dict(components)
        self._name = name

        # Precompute component slices for flat ↔ structured conversion
        slices = {}
        offset = 0
        for cname, cdist in self._components.items():
            dim = 1
            for s in cdist.event_shape:
                dim *= s
            slices[cname] = slice(offset, offset + dim)
            offset += dim
        self._component_slices = slices
        self._total_dim = offset

    @classmethod
    def from_distributions(cls, distributions: list[Distribution], **kwargs):
        """
        Construct from a list of named distributions.

        Each distribution must have a non-None ``name`` attribute, and names
        must be unique.
        """
        components = {}
        for dist in distributions:
            if dist.name is None:
                raise ValueError(
                    f"All distributions must have a name; "
                    f"{type(dist).__name__} has name=None"
                )
            if dist.name in components:
                raise ValueError(f"Duplicate component name: '{dist.name}'")
            components[dist.name] = dist
        return cls(**components, **kwargs)

    # -- Properties --------------------------------------------------------

    @property
    def components(self) -> dict[str, Distribution]:
        return MappingProxyType(self._components)

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(self._components.keys())

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._total_dim,)

    @property
    def support(self) -> Constraint:
        return real  # conservative default for the flat representation

    # -- Structured access -------------------------------------------------

    def __getitem__(self, name: str) -> DistributionView:
        return DistributionView(self, name)

    def bind(self, **mapping: str) -> dict[str, DistributionView]:
        """
        Create a dict of :class:`DistributionView` instances with remapped names.

        Example::

            joint = ProductDistribution(x=Normal(0, 1), y=Normal(1, 2))
            views = joint.bind(a='x', b='y')
            # views == {'a': DistributionView(x), 'b': DistributionView(y)}
        """
        return {arg_name: self[comp_name] for arg_name, comp_name in mapping.items()}

    # -- Flat ↔ structured conversion -------------------------------------

    def unflatten(self, flat: ArrayLike) -> dict[str, Array]:
        """Split a flat array into per-component arrays."""
        flat = jnp.asarray(flat)
        result = {}
        for cname, cdist in self._components.items():
            sl = self._component_slices[cname]
            chunk = flat[..., sl]
            # Reshape back to component event_shape if needed
            target_shape = flat.shape[:-1] + cdist.event_shape
            result[cname] = chunk.reshape(target_shape)
        return result

    def flatten_structured(self, structured: dict[str, ArrayLike]) -> Array:
        """Concatenate per-component arrays into a flat array."""
        parts = []
        for cname, cdist in self._components.items():
            arr = jnp.asarray(structured[cname])
            if cdist.event_shape:
                batch_shape = arr.shape[:-len(cdist.event_shape)]
                parts.append(arr.reshape(batch_shape + (-1,)))
            else:
                parts.append(arr[..., None])  # scalar → (*, 1)
        return jnp.concatenate(parts, axis=-1)

    # -- Sampling (abstract — subclasses must implement) -------------------

    def sample_structured(
        self, key: PRNGKey | None = None, sample_shape: tuple[int, ...] = ()
    ) -> dict[str, Array]:
        """
        Draw samples and return a dict of per-component arrays.

        Subclasses should override this to implement their sampling strategy.
        The default calls :meth:`sample` (flat) and unflattens.
        """
        from .distribution import _auto_key
        if key is None:
            key = _auto_key()
        flat = self._sample(key, sample_shape)
        return self.unflatten(flat)

    # -- Conditioning ------------------------------------------------------

    def condition_on(self, **observed: ArrayLike) -> "JointDistribution":
        """
        Return the conditional distribution over the remaining (unconditioned)
        components, representing p(unconditioned | conditioned = values).

        Parameters
        ----------
        **observed
            Keyword arguments mapping component names to their observed values.

        Returns
        -------
        JointDistribution
            A new instance of the same type with only the unconditioned
            components.
        """
        unknown = set(observed) - set(self._components)
        if unknown:
            raise KeyError(
                f"Unknown component(s): {unknown}. "
                f"Available: {self.component_names}"
            )
        if len(observed) >= len(self._components):
            raise ValueError("Cannot condition on all components.")
        new_components = {
            cname: cdist
            for cname, cdist in self._components.items()
            if cname not in observed
        }
        result = type(self)(**new_components, name=self._name)
        result.with_source(Provenance(
            "condition_on", parents=(self,),
            metadata={"conditioned": list(observed)},
        ))
        return result

    def log_prob(self, x: ArrayLike) -> Array:
        raise NotImplementedError(
            f"{type(self).__name__}.log_prob() must be implemented by subclasses."
        )

    @classmethod
    def _from_distribution(cls, other, *, key, **kwargs):
        raise NotImplementedError("Cannot convert to JointDistribution.")

    @classmethod
    def _default_support(cls) -> Constraint:
        return real

    def __repr__(self) -> str:
        comp_str = ", ".join(
            f"{k}={type(v).__name__}" for k, v in self._components.items()
        )
        name_str = f", name='{self._name}'" if self._name else ""
        return f"{type(self).__name__}({comp_str}{name_str})"


# ---------------------------------------------------------------------------
# ProductDistribution — independent components
# ---------------------------------------------------------------------------

class ProductDistribution(JointDistribution):
    """
    Joint distribution with independent components.

    Sampling draws from each component independently and concatenates.
    ``log_prob`` is the sum of component log-probs.
    """

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        keys = jax.random.split(key, len(self._components))
        parts = []
        for subkey, (cname, cdist) in zip(keys, self._components.items()):
            s = cdist.sample(subkey, sample_shape)
            # Flatten event dims to (..., dim_i)
            if cdist.event_shape:
                flat_shape = s.shape[:len(sample_shape)] + (-1,)
                parts.append(s.reshape(flat_shape))
            else:
                parts.append(s[..., None])  # scalar → (..., 1)
        return jnp.concatenate(parts, axis=-1)

    def sample_structured(
        self, key: PRNGKey | None = None, sample_shape: tuple[int, ...] = ()
    ) -> dict[str, Array]:
        from .distribution import _auto_key
        if key is None:
            key = _auto_key()
        keys = jax.random.split(key, len(self._components))
        result = {}
        for subkey, (cname, cdist) in zip(keys, self._components.items()):
            result[cname] = cdist.sample(subkey, sample_shape)
        return result

    def log_prob(self, x: ArrayLike) -> Array:
        x = jnp.asarray(x)
        structured = self.unflatten(x)
        total = jnp.zeros(x.shape[:-1])
        for cname, cdist in self._components.items():
            total = total + cdist.log_prob(structured[cname])
        return total

    def mean(self) -> Array:
        parts = []
        for cname, cdist in self._components.items():
            m = cdist.mean()
            if cdist.event_shape:
                parts.append(m.reshape(-1))
            else:
                parts.append(m[None])
        return jnp.concatenate(parts)

    def variance(self) -> Array:
        parts = []
        for cname, cdist in self._components.items():
            v = cdist.variance()
            if cdist.event_shape:
                parts.append(v.reshape(-1))
            else:
                parts.append(v[None])
        return jnp.concatenate(parts)


# ---------------------------------------------------------------------------
# SequentialJointDistribution — autoregressive dependence
# ---------------------------------------------------------------------------

class SequentialJointDistribution(JointDistribution):
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

    def __init__(
        self,
        *,
        name: str | None = None,
        **components: Distribution | Callable[..., Distribution],
    ):
        if not components:
            raise ValueError("SequentialJointDistribution requires at least one component.")

        self._raw_components: dict[str, Distribution | Callable] = dict(components)
        self._name = name
        self._conditioned_names: frozenset[str] = frozenset()
        self._conditioned_values: dict[str, Array] = {}
        self._sampleable_error: str | None = None
        # Map callable component names to their dependency parameter names
        self._callable_parents: dict[str, tuple[str, ...]] = {}

        # Validate ordering: callable args must reference earlier names
        seen: list[str] = []
        for cname, comp in self._raw_components.items():
            if callable(comp) and not isinstance(comp, Distribution):
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
        self._proto_components: dict[str, Distribution] = {}

        resolved: dict[str, Distribution] = {}
        for cname, comp in self._raw_components.items():
            if isinstance(comp, Distribution):
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

        # Build _components dict from resolved prototypes (for slices, event_shape, etc.)
        # Note: we store resolved prototypes for shape introspection only
        self._components = resolved

        # Compute slices from resolved components
        slices = {}
        offset = 0
        for cname, cdist in self._components.items():
            dim = int(math.prod(cdist.event_shape)) if cdist.event_shape else 1
            slices[cname] = slice(offset, offset + dim)
            offset += dim
        self._component_slices = slices
        self._total_dim = offset

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
            elif isinstance(comp, Distribution):
                # Root distribution: sample with sample_shape
                sampled[cname] = comp.sample(subkey, sample_shape)
            else:
                # Conditional: callable receives batched parent samples,
                # returning a batched distribution.  Sample with () since
                # the batch is already in the distribution's batch_shape.
                sig = inspect.signature(comp)
                call_kw = {p: sampled[p] for p in sig.parameters if p in sampled}
                dist = comp(**call_kw)
                sampled[cname] = dist.sample(subkey)

        return sampled

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        structured = self._sample_sequential(key, sample_shape)
        return self.flatten_structured(structured)

    def sample_structured(
        self, key: PRNGKey | None = None, sample_shape: tuple[int, ...] = ()
    ) -> dict[str, Array]:
        from .distribution import _auto_key
        if key is None:
            key = _auto_key()
        full = self._sample_sequential(key, sample_shape)
        # Return only unconditioned components
        return {k: v for k, v in full.items() if k not in self._conditioned_names}

    def log_prob(self, x: ArrayLike) -> Array:
        """Evaluate the (unnormalized) conditional log-density.

        For an unconditioned joint, this is the full joint log p(x).
        After conditioning, this is log p(unconditioned, conditioned=values),
        which is proportional to log p(unconditioned | conditioned=values).
        """
        x = jnp.asarray(x)
        # unflatten gives only unconditioned components
        structured = self.unflatten(x)

        # Add conditioned values so callables can receive them
        for cname, val in self._conditioned_values.items():
            structured[cname] = val

        # Sum log-probs over all components (conditioned ones contribute
        # the likelihood term, making this proportional to the conditional)
        total = jnp.zeros(x.shape[:-1])
        for cname, comp in self._raw_components.items():
            val = structured[cname]
            if isinstance(comp, Distribution):
                total = total + comp.log_prob(val)
            else:
                # Build conditional distribution from parent values
                sig = inspect.signature(comp)
                call_kw = {p: structured[p] for p in sig.parameters if p in structured}
                cond_dist = comp(**call_kw)
                total = total + cond_dist.log_prob(val)

        return total

    def mean(self) -> Array:
        # For sequential joints, the marginal mean is not simply the concatenation
        # of component means (because components depend on earlier samples).
        # We use the prototype components' means as an approximation.
        parts = []
        for cname, cdist in self._proto_components.items():
            m = cdist.mean()
            if cdist.event_shape:
                parts.append(m.reshape(-1))
            else:
                parts.append(m[None])
        return jnp.concatenate(parts)

    def variance(self) -> Array:
        # Same caveat as mean() — uses prototype components
        parts = []
        for cname, cdist in self._proto_components.items():
            v = cdist.variance()
            if cdist.event_shape:
                parts.append(v.reshape(-1))
            else:
                parts.append(v[None])
        return jnp.concatenate(parts)

    def condition_on(self, **observed: ArrayLike) -> "SequentialJointDistribution":
        """
        Return the conditional distribution over the remaining (unconditioned)
        components, representing p(unconditioned | conditioned = values).

        The resulting distribution is sampleable as long as every conditioned
        non-root component has all of its parents also conditioned (so that no
        unconditioned ancestor would be drawn from the prior instead of the
        posterior).  Root components have no parents, so conditioning on them
        is always sampleable.  If the sampleability condition is violated, a
        :class:`NotImplementedError` is raised at sample time.
        ``log_prob()`` always works regardless (returns the unnormalized
        conditional log-density).

        Parameters
        ----------
        **observed
            Component names mapped to their observed values.
        """
        unknown = set(observed) - set(self._raw_components)
        if unknown:
            raise KeyError(
                f"Unknown component(s): {unknown}. "
                f"Available: {self.component_names}"
            )

        all_conditioned = self._conditioned_names | frozenset(observed)
        if len(all_conditioned) >= len(self._raw_components):
            raise ValueError("Cannot condition on all components.")

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

        # Recompute slices for unconditioned components only
        slices = {}
        offset = 0
        for cname, cdist in unconditioned.items():
            dim = int(math.prod(cdist.event_shape)) if cdist.event_shape else 1
            slices[cname] = slice(offset, offset + dim)
            offset += dim
        result._component_slices = slices
        result._total_dim = offset

        result.with_source(Provenance(
            "condition_on", parents=(self,),
            metadata={"conditioned": list(observed)},
        ))
        return result

    def __repr__(self) -> str:
        parts = []
        for k, v in self._raw_components.items():
            if isinstance(v, Distribution):
                parts.append(f"{k}={type(v).__name__}")
            else:
                parts.append(f"{k}=<callable>")
        comp_str = ", ".join(parts)
        name_str = f", name='{self._name}'" if self._name else ""
        return f"SequentialJointDistribution({comp_str}{name_str})"


# ---------------------------------------------------------------------------
# JointEmpirical — weighted joint samples
# ---------------------------------------------------------------------------

class JointEmpirical(JointDistribution):
    """
    Joint distribution from weighted joint samples.

    Stores per-component sample arrays (all with the same number of rows)
    and optional weights.  Sampling resamples rows jointly, preserving
    correlation between components.

    When used in broadcasting enumeration, the joint is treated as a single
    unit with ``n`` samples (no cartesian decomposition).

    Parameters
    ----------
    weights : array-like, shape ``(n,)``, optional
        Non-negative sample weights (normalised internally).
    log_weights : array-like, shape ``(n,)``, optional
        Log-unnormalised weights.  Mutually exclusive with *weights*.
    name : str, optional
        Distribution name.
    **samples : array-like
        Named component sample arrays.  Each must have the same number of
        rows (first dimension = ``n``).
    """

    def __init__(
        self,
        *,
        weights: ArrayLike | None = None,
        log_weights: ArrayLike | None = None,
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

        # Handle weights (same logic as EmpiricalDistribution)
        if weights is not None and log_weights is not None:
            raise ValueError("Provide either weights or log_weights, not both.")

        if weights is not None:
            weights = jnp.asarray(weights, dtype=jnp.float32)
            if weights.shape != (n,):
                raise ValueError(f"weights shape {weights.shape} != ({n},)")
            if jnp.any(weights < 0):
                raise ValueError("weights must be non-negative.")
            self._log_weights = jnp.log(weights)
            self._is_uniform = False
        elif log_weights is not None:
            log_weights = jnp.asarray(log_weights, dtype=jnp.float32)
            if log_weights.shape != (n,):
                raise ValueError(f"log_weights shape {log_weights.shape} != ({n},)")
            self._log_weights = log_weights
            self._is_uniform = False
        else:
            self._log_weights = None
            self._is_uniform = True

        self._weights_cache: Array | None = None

        # Build _components as EmpiricalDistribution per component (for shape introspection)
        comp_dists: dict[str, EmpiricalDistribution] = {}
        for cname, arr in self._joint_samples.items():
            if self._is_uniform:
                comp_dists[cname] = EmpiricalDistribution(arr, name=cname)
            else:
                comp_dists[cname] = EmpiricalDistribution(
                    arr, log_weights=self._log_weights, name=cname
                )
        self._components = comp_dists

        # Compute slices
        slices = {}
        offset = 0
        for cname, cdist in self._components.items():
            dim = int(math.prod(cdist.event_shape)) if cdist.event_shape else 1
            slices[cname] = slice(offset, offset + dim)
            offset += dim
        self._component_slices = slices
        self._total_dim = offset

    @property
    def n(self) -> int:
        """Number of joint samples."""
        return self._n

    @property
    def is_uniform(self) -> bool:
        return self._is_uniform

    @property
    def weights(self) -> Array:
        """Normalised weights, shape ``(n,)``."""
        if self._is_uniform:
            return jnp.ones(self._n, dtype=jnp.float32) / self._n
        if self._weights_cache is None:
            self._weights_cache = jax.nn.softmax(self._log_weights)
        return self._weights_cache

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        structured = self._sample_joint_rows(key, sample_shape)
        return self.flatten_structured(structured)

    def _sample_joint_rows(
        self, key: PRNGKey, sample_shape: tuple[int, ...]
    ) -> dict[str, Array]:
        """Resample rows jointly, preserving correlation."""
        n_draws = int(math.prod(sample_shape)) if sample_shape else 1
        if self._is_uniform:
            indices = jax.random.randint(key, shape=(n_draws,), minval=0, maxval=self._n)
        else:
            indices = jax.random.choice(
                key, self._n, shape=(n_draws,), p=self.weights, replace=True,
            )
        result = {}
        for cname, arr in self._joint_samples.items():
            drawn = arr[indices]
            if sample_shape:
                result[cname] = drawn.reshape(sample_shape + arr.shape[1:])
            else:
                result[cname] = drawn.squeeze(axis=0)
        return result

    def sample_structured(
        self, key: PRNGKey | None = None, sample_shape: tuple[int, ...] = ()
    ) -> dict[str, Array]:
        from .distribution import _auto_key
        if key is None:
            key = _auto_key()
        return self._sample_joint_rows(key, sample_shape)

    def log_prob(self, x: ArrayLike) -> Array:
        """Gaussian-approximation log-density (same as EmpiricalDistribution)."""
        x = jnp.asarray(x)
        mu = self.mean()
        var = self.variance()
        var = jnp.maximum(var, 1e-12)
        log_norm = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * var))
        diff = x - mu
        return log_norm - 0.5 * jnp.sum(diff**2 / var, axis=-1)

    def mean(self) -> Array:
        parts = []
        for cname, arr in self._joint_samples.items():
            if self._is_uniform:
                m = jnp.mean(arr, axis=0)
            else:
                m = jnp.einsum("n,n...->...", self.weights, arr)
            parts.append(m.reshape(-1))
        return jnp.concatenate(parts)

    def variance(self) -> Array:
        mu_flat = self.mean()
        # Compute per-component variances
        parts = []
        offset = 0
        for cname, arr in self._joint_samples.items():
            flat_dim = int(math.prod(arr.shape[1:])) if arr.ndim > 1 else 1
            mu_comp = mu_flat[offset:offset + flat_dim]
            arr_flat = arr.reshape(self._n, -1)
            diff = arr_flat - mu_comp
            if self._is_uniform:
                v = jnp.mean(diff**2, axis=0)
            else:
                v = jnp.einsum("n,nd->d", self.weights, diff**2)
            parts.append(v)
            offset += flat_dim
        return jnp.concatenate(parts)


# ---------------------------------------------------------------------------
# JointGaussian — analytical joint Gaussian with cross-covariance
# ---------------------------------------------------------------------------

class JointGaussian(JointDistribution):
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
        self._component_slices = slices
        self._total_dim = total_dim

    @property
    def mean_vector(self) -> Array:
        """Full mean vector."""
        return self._mean_vec

    @property
    def covariance(self) -> Array:
        """Full covariance matrix."""
        return self._cov_mat

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        from .multivariate import MultivariateNormal as MVN
        full_mvn = MVN(loc=self._mean_vec, cov=self._cov_mat)
        return full_mvn.sample(key, sample_shape)

    def sample_structured(
        self, key: PRNGKey | None = None, sample_shape: tuple[int, ...] = ()
    ) -> dict[str, Array]:
        from .distribution import _auto_key
        if key is None:
            key = _auto_key()
        flat = self._sample(key, sample_shape)
        return self.unflatten(flat)

    def log_prob(self, x: ArrayLike) -> Array:
        from .multivariate import MultivariateNormal as MVN
        full_mvn = MVN(loc=self._mean_vec, cov=self._cov_mat)
        return full_mvn.log_prob(x)

    def mean(self) -> Array:
        return self._mean_vec

    def variance(self) -> Array:
        return jnp.diag(self._cov_mat)

    def condition_on(self, **observed: ArrayLike) -> "JointGaussian":
        """
        Condition on observed component values using exact Gaussian formulas.

        Returns a new :class:`JointGaussian` over the remaining (unobserved)
        components.

        Parameters
        ----------
        **observed
            Component names mapped to their observed values.

        Returns
        -------
        JointGaussian
            Conditional distribution over the remaining components.
        """
        unknown = set(observed) - set(self._component_shapes)
        if unknown:
            raise KeyError(
                f"Unknown component(s): {unknown}. "
                f"Available: {tuple(self._component_shapes.keys())}"
            )

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
            raise ValueError("Cannot condition on all components.")

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
    children = tuple(dist._components.values())
    aux = (tuple(dist._components.keys()), dist._name)
    return children, aux


def _product_unflatten(aux, children):
    keys, name = aux
    return ProductDistribution(**dict(zip(keys, children)), name=name)


jax.tree_util.register_pytree_node(
    ProductDistribution,
    _product_flatten,
    _product_unflatten,
)
