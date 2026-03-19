"""Joint distributions and component views for correlated broadcasting."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp

from ..custom_types import Array, ArrayLike, PRNGKey
from .distribution import Distribution, EmpiricalDistribution, Provenance, Constraint, real

__all__ = [
    "JointDistribution",
    "ProductDistribution",
    "DistributionView",
    "PinnedComponent",
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
# PinnedComponent — a distribution fixed to a specific value
# ---------------------------------------------------------------------------

class PinnedComponent(Distribution):
    """
    A distribution whose value is fixed (observed / conditioned).

    Used internally by :meth:`JointDistribution.pin` to replace a component
    with a constant.  Sampling always returns the pinned value;
    ``log_prob`` evaluates the base distribution at the pinned value.
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
        raise NotImplementedError("Cannot convert to PinnedComponent.")

    @classmethod
    def _default_support(cls) -> Constraint:
        return real

    def __repr__(self) -> str:
        return f"PinnedComponent(base={type(self._base).__name__}, value={self._value})"


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
        for cname in self._components:
            arr = jnp.asarray(structured[cname])
            # Flatten event dims
            flat_shape = arr.shape[:-len(self._components[cname].event_shape) or None]
            if self._components[cname].event_shape:
                flat_shape = arr.shape[:-len(self._components[cname].event_shape)]
                parts.append(arr.reshape(flat_shape + (-1,)))
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

    # -- Pinning -----------------------------------------------------------

    def pin(self, **pinned_values: ArrayLike) -> "JointDistribution":
        """
        Return a new joint distribution with specified components pinned
        (fixed to concrete values).

        Parameters
        ----------
        **pinned_values
            Keyword arguments mapping component names to their fixed values.

        Returns
        -------
        JointDistribution
            A new instance of the same type with pinned components replaced
            by :class:`PinnedComponent` instances.
        """
        new_components = {}
        for cname, cdist in self._components.items():
            if cname in pinned_values:
                new_components[cname] = PinnedComponent(
                    base=cdist,
                    value=pinned_values[cname],
                    name=cname,
                )
            else:
                new_components[cname] = cdist
        unknown = set(pinned_values) - set(self._components)
        if unknown:
            raise KeyError(
                f"Unknown component(s) to pin: {unknown}. "
                f"Available: {self.component_names}"
            )
        result = type(self)(**new_components, name=self._name)
        result.with_source(Provenance("pin", parents=(self,), metadata={"pinned": list(pinned_values)}))
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
