"""Joint distributions and component views for correlated broadcasting.

This module implements joint distributions whose samples are **dictionaries**
mapping component names to arrays.  The base class :class:`JointDistribution`
inherits from :class:`PyTreeArrayDistribution[dict]`, so all pytree shape
semantics (``event_shapes``, ``event_size``, ``flatten_value`` /
``unflatten_value``, ``as_flat_distribution()``) are available.

**Structural contract:**

*  Components are identified by string keys in a flat ``dict``.
*  ``sample()`` returns ``dict[str, Array]`` where each value has shape
   ``(*sample_shape, *batch_shape, *event_shape_for_that_component)``.
*  ``log_prob()`` accepts a ``dict[str, Array]`` with the same structure.
*  ``event_shapes`` returns ``{name: event_shape}`` for each component.
*  ``batch_shape`` is shared across all components (an empty tuple unless
   the component distributions themselves are batched identically).
*  ``flatten_value`` / ``unflatten_value`` (inherited from
   ``PyTreeArrayDistribution``) convert between the dict representation
   and a flat ``(*leading_dims, event_size)`` array.  Leaf ordering
   follows Python's canonical dict iteration order (insertion order,
   which for these dicts is the order components were passed to the
   constructor).
*  ``as_flat_distribution()`` returns a :class:`FlattenedView` with
   ``event_shape = (event_size,)`` for use with algorithms expecting
   flat vectors.

**Independence assumptions:**

*  :class:`JointDistribution` (base) makes **no** independence assumption.
   Subclasses define the factorization structure.
*  :class:`ProductDistribution` assumes **all components are independent**.
   ``log_prob`` is the sum of per-component log-probs with no coupling.
"""
from __future__ import annotations

import inspect
from types import MappingProxyType
from typing import Any, Callable

import jax
import jax.numpy as jnp
import math

from ..custom_types import Array, ArrayLike, PRNGKey
from .distribution import (
    ArrayDistribution,
    PyTreeArrayDistribution,
    EmpiricalDistribution,
    Provenance,
    Constraint,
    real,
    _auto_key,
)

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

class DistributionView(ArrayDistribution):
    """A lightweight reference to a named component of a :class:`JointDistribution`.

    A ``DistributionView`` acts as an ``ArrayDistribution`` for a single
    component of a joint distribution.  It is the object returned by
    ``joint["component_name"]``.

    **Broadcasting contract:** The broadcasting logic in
    :class:`~probpipe.core.node.WorkflowFunction` detects
    ``DistributionView`` instances and groups those sharing the same
    ``_parent`` so they are sampled jointly (preserving correlation
    between components).

    **Standalone sampling:** When sampled outside of broadcasting, the
    view draws a full joint sample from its parent and extracts this
    component.

    **Key path (current implementation):** The view stores a single
    string ``component_name`` that addresses a leaf in the parent's
    flat-dict structure.  A future extension will generalize this to
    tuple key paths for nested pytree joints.

    Parameters
    ----------
    parent : JointDistribution
        The joint distribution this view belongs to.
    component_name : str
        The name of the component to extract.
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

    def _sample(self, key: PRNGKey) -> Array:
        structured = self._parent.sample(key)
        return structured[self._component_name]

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        if key is None:
            key = _auto_key()
        structured = self._parent.sample(key, sample_shape)
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

class ConditionedComponent(ArrayDistribution):
    """
    A distribution whose value is fixed (observed / conditioned on).

    Used internally by :meth:`JointDistribution.condition_on` to replace a
    component with a constant.  Sampling always returns the conditioned value;
    ``log_prob`` evaluates the base distribution at the conditioned value.
    """

    def __init__(self, base: ArrayDistribution, value: ArrayLike, *, name: str | None = None):
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

    def _sample(self, key: PRNGKey) -> Array:
        return self._value

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        if sample_shape == ():
            return self._value
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

class JointDistribution(PyTreeArrayDistribution[dict]):
    """Base class for named multi-component distributions.

    A ``JointDistribution`` is a :class:`PyTreeArrayDistribution` whose
    samples are **dictionaries** mapping component names (strings) to
    arrays.  Each component is backed by an ``ArrayDistribution`` that
    defines the marginal (or conditional) distribution for that
    component.

    **Structural requirements:**

    *  Components are passed as keyword arguments to ``__init__`` and
       stored in insertion order.
    *  Each component must be an ``ArrayDistribution``.
    *  Component names are strings (dict keys); the pytree structure
       (``treedef``) is the corresponding ``dict`` treedef.
    *  ``batch_shape`` is assumed to be shared across all components
       and defaults to ``()``.

    **Sample type:**

    ``sample()`` returns ``dict[str, Array]`` where each value has shape
    ``(*sample_shape, *batch_shape, *component_event_shape)``.

    **Log-prob contract:**

    ``log_prob()`` accepts a ``dict[str, Array]`` with the same
    structure and returns a scalar (or batch-shaped array).  The base
    class raises ``NotImplementedError``; subclasses define the
    factorization (e.g., product of marginals for
    :class:`ProductDistribution`).

    **Flat-vector interop:**

    The inherited ``flatten_value`` / ``unflatten_value`` methods convert
    between the dict representation and flat ``(*leading, event_size)``
    arrays.  ``as_flat_distribution()`` returns a :class:`FlattenedView`
    with ``event_shape = (event_size,)`` for algorithms expecting flat
    vectors.

    **Component access:**

    ``joint["name"]`` returns a :class:`DistributionView` — an
    ``ArrayDistribution`` that extracts the named component from joint
    samples.  ``joint.bind(a="x", b="y")`` creates a dict of views
    with remapped names for use in broadcasting.

    Parameters
    ----------
    name : str, optional
        Distribution name for provenance / display.
    **components : ArrayDistribution
        Named component distributions.  At least one required.
    """

    def __init__(self, *, name: str | None = None, **components: ArrayDistribution):
        if not components:
            raise ValueError("JointDistribution requires at least one component.")
        for k, v in components.items():
            if not isinstance(v, ArrayDistribution):
                raise TypeError(
                    f"Component '{k}' must be an ArrayDistribution, got {type(v).__name__}"
                )
        self._components = dict(components)
        self._name = name

    @classmethod
    def from_distributions(cls, distributions: list[ArrayDistribution], **kwargs):
        """Construct from a list of named distributions.

        Each distribution must have a non-``None`` ``name`` attribute,
        and names must be unique.

        Parameters
        ----------
        distributions : list[ArrayDistribution]
            Distributions with unique, non-None ``.name`` attributes.
        **kwargs
            Passed to the constructor (e.g., ``name``).
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

    # -- PyTreeArrayDistribution interface ---------------------------------

    @property
    def treedef(self) -> jax.tree_util.PyTreeDef:
        """Pytree structure of a single sample (a dict with component keys)."""
        prototype = {k: 0 for k in self._components}
        return jax.tree.structure(prototype)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape, shared across all components.

        Defaults to ``()``.  If the component distributions themselves
        have non-trivial batch shapes, they must all agree.
        """
        return ()

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-component event shapes.

        Returns a dict ``{name: event_shape}`` with the same key
        ordering as the components.
        """
        return {k: v.event_shape for k, v in self._components.items()}

    # -- Properties --------------------------------------------------------

    @property
    def components(self) -> dict[str, ArrayDistribution]:
        """Read-only view of the component distributions."""
        return MappingProxyType(self._components)

    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names in insertion order."""
        return tuple(self._components.keys())

    # -- Structured access -------------------------------------------------

    def __getitem__(self, name: str) -> DistributionView:
        """Return a :class:`DistributionView` for the named component.

        The returned view is an ``ArrayDistribution`` whose samples
        are the marginal values of this component.  When sampled
        standalone, it draws a full joint sample and extracts the
        relevant component.

        Parameters
        ----------
        name : str
            Component name (must exist in ``component_names``).
        """
        return DistributionView(self, name)

    def bind(self, **mapping: str) -> dict[str, DistributionView]:
        """Create a dict of views with remapped names.

        Example::

            joint = ProductDistribution(x=Normal(0, 1), y=Normal(1, 2))
            views = joint.bind(a='x', b='y')
            # views == {'a': DistributionView(x), 'b': DistributionView(y)}
        """
        return {arg_name: self[comp_name] for arg_name, comp_name in mapping.items()}

    # -- Component-level log_prob ------------------------------------------

    def component_log_prob(self, value: dict[str, ArrayLike]) -> dict[str, Array]:
        """Per-component log-density contributions.

        Returns a dict ``{name: scalar_or_batch}`` with the same
        structure as the components.  The total ``log_prob`` is the
        sum of these values (plus any cross-component coupling terms
        defined by the subclass).

        The base implementation evaluates each component's ``log_prob``
        independently.  Subclasses with cross-component dependencies
        should override this method.
        """
        return {
            k: self._components[k].log_prob(jnp.asarray(value[k]))
            for k in self._components
        }

    # -- Conditioning ------------------------------------------------------

    def condition_on(self, **observed: ArrayLike) -> "JointDistribution":
        """Return the conditional distribution given observed component values.

        Parameters
        ----------
        **observed
            Component names mapped to their observed values.

        Returns
        -------
        JointDistribution
            A new joint over the unconditioned components.

        Raises
        ------
        KeyError
            If an observed name is not a valid component.
        ValueError
            If all components are conditioned on.
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

    # -- log_prob (abstract) -----------------------------------------------

    def log_prob(self, value: dict[str, ArrayLike]) -> Array:
        """Log-density of the joint distribution.

        Parameters
        ----------
        value : dict[str, ArrayLike]
            A dict mapping component names to arrays, each with shape
            ``(*batch_dims, *component_event_shape)``.

        Returns
        -------
        Array
            Scalar (or batch-shaped array) of log-densities.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.log_prob() must be implemented by subclasses."
        )

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
    """Joint distribution with **independent** components.

    All components are sampled independently.  The joint ``log_prob``
    is the sum of per-component log-probs (no coupling terms).

    **Sample type:** ``dict[str, Array]`` — same as
    :class:`JointDistribution`.

    **Independence assumption:** This class assumes statistical
    independence across all components.  For sequential/autoregressive
    dependence, use :class:`SequentialJointDistribution`.  For
    arbitrary dependence with a known joint density, subclass
    :class:`JointDistribution` directly.

    **Flat-vector interop:** Use ``as_flat_distribution()`` or
    ``flatten_value()`` to obtain flat representations compatible
    with algorithms expecting ``ArrayDistribution``.

    Parameters
    ----------
    name : str, optional
        Distribution name.
    **components : ArrayDistribution
        Named independent component distributions.

    Examples
    --------
    >>> joint = ProductDistribution(
    ...     x=Normal(loc=0.0, scale=1.0),
    ...     y=Normal(loc=3.0, scale=2.0),
    ... )
    >>> s = joint.sample(jax.random.PRNGKey(0))
    >>> s.keys()
    dict_keys(['x', 'y'])
    """

    def _sample(self, key: PRNGKey) -> dict[str, Array]:
        keys = jax.random.split(key, len(self._components))
        return {
            cname: cdist.sample(subkey)
            for subkey, (cname, cdist) in zip(keys, self._components.items())
        }

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> dict[str, Array]:
        """Draw independent samples from each component.

        Parameters
        ----------
        key : PRNGKey, optional
            JAX PRNG key.  Auto-generated if ``None``.
        sample_shape : tuple[int, ...], optional
            Leading dimensions for the samples.

        Returns
        -------
        dict[str, Array]
            Per-component samples.  Each value has shape
            ``(*sample_shape, *batch_shape, *component_event_shape)``.
        """
        if key is None:
            key = _auto_key()
        keys = jax.random.split(key, len(self._components))
        return {
            cname: cdist.sample(subkey, sample_shape)
            for subkey, (cname, cdist) in zip(keys, self._components.items())
        }

    def log_prob(self, value: dict[str, ArrayLike]) -> Array:
        """Sum of independent component log-probs.

        Parameters
        ----------
        value : dict[str, ArrayLike]
            A dict mapping component names to arrays.

        Returns
        -------
        Array
            Scalar (or batch-shaped) log-density.
        """
        total = None
        for cname, cdist in self._components.items():
            lp = cdist.log_prob(jnp.asarray(value[cname]))
            total = lp if total is None else total + lp
        return total

    def mean(self, **kwargs) -> dict[str, Array]:
        """Per-component means (exact, no MC needed for independent components)."""
        return {k: v.mean() for k, v in self._components.items()}

    def variance(self, **kwargs) -> dict[str, Array]:
        """Per-component variances (exact, no MC needed for independent components)."""
        return {k: v.variance() for k, v in self._components.items()}


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
            elif isinstance(comp, ArrayDistribution):
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

    def _sample(self, key: PRNGKey) -> dict[str, Array]:
        full = self._sample_sequential(key, ())
        return {k: v for k, v in full.items() if k not in self._conditioned_names}

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> dict[str, Array]:
        if key is None:
            key = _auto_key()
        full = self._sample_sequential(key, sample_shape)
        return {k: v for k, v in full.items() if k not in self._conditioned_names}

    def _eval_log_prob(self, value: dict[str, ArrayLike], *, components: str) -> Array:
        """Evaluate log-density over selected components.

        Parameters
        ----------
        value : dict[str, ArrayLike]
            Dict of unconditioned component values.
        components : ``"all"`` or ``"unconditioned"``
            Which components to include in the sum.  ``"all"`` sums over
            every component (conditioned ones evaluated at their observed
            values), giving the unnormalized conditional.
            ``"unconditioned"`` sums only over unconditioned components
            (with conditioned values plugged in as parents), giving the
            normalized conditional when the Markov structure permits it.
        """
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
                lp = comp.log_prob(val)
            else:
                sig = inspect.signature(comp)
                call_kw = {p: structured[p] for p in sig.parameters if p in structured}
                cond_dist = comp(**call_kw)
                lp = cond_dist.log_prob(val)
            total = lp if total is None else total + lp

        return total

    def log_prob(self, value: dict[str, ArrayLike]) -> Array:
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

    def unnormalized_log_prob(self, value: dict[str, ArrayLike]) -> Array:
        """Evaluate the (possibly unnormalized) log-density.

        For an unconditioned joint, this equals the full joint log p(x).
        After conditioning, this sums log-densities of *all* components
        (conditioned ones evaluated at their observed values), giving
        log p(unconditioned, conditioned=values), which is proportional
        to log p(unconditioned | conditioned=values).
        """
        return self._eval_log_prob(value, components="all")

    def mean(self, **kwargs) -> dict[str, Array]:
        """Per-component means (approximate — uses prototype components).

        For sequential joints, the true marginal mean is not simply the
        per-component mean because later components depend on earlier
        samples.  This returns the prototype (prior-evaluated) means
        as an approximation.
        """
        return {k: v.mean() for k, v in self._proto_components.items()}

    def variance(self, **kwargs) -> dict[str, Array]:
        """Per-component variances (approximate — uses prototype components)."""
        return {k: v.variance() for k, v in self._proto_components.items()}

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

    def _sample(self, key: PRNGKey) -> dict[str, Array]:
        return self._sample_joint_rows(key, ())

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> dict[str, Array]:
        if key is None:
            key = _auto_key()
        return self._sample_joint_rows(key, sample_shape)

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

    def log_prob(self, value: dict[str, ArrayLike]) -> Array:
        """Gaussian-approximation log-density (same as EmpiricalDistribution).

        Evaluates a diagonal Gaussian approximation in the flat space.
        """
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
            if self._is_uniform:
                m = jnp.mean(arr, axis=0)
            else:
                m = jnp.einsum("n,n...->...", self.weights, arr)
            parts.append(m.reshape(-1))
        return jnp.concatenate(parts)

    def _flat_variance(self) -> Array:
        """Flat variance vector (for internal use by log_prob)."""
        mu_flat = self._flat_mean()
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

    def mean(self, **kwargs) -> dict[str, Array]:
        """Per-component weighted means."""
        result = {}
        for cname, arr in self._joint_samples.items():
            if self._is_uniform:
                result[cname] = jnp.mean(arr, axis=0)
            else:
                result[cname] = jnp.einsum("n,n...->...", self.weights, arr)
        return result

    def variance(self, **kwargs) -> dict[str, Array]:
        """Per-component weighted variances."""
        means = self.mean()
        result = {}
        for cname, arr in self._joint_samples.items():
            diff = arr - means[cname]
            if self._is_uniform:
                result[cname] = jnp.mean(diff**2, axis=0)
            else:
                result[cname] = jnp.einsum("n,n...->...", self.weights, diff**2)
        return result


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
        self._component_slices = slices  # still needed for Gaussian conditioning
        self._total_dim = total_dim  # still needed for Gaussian conditioning

    @property
    def mean_vector(self) -> Array:
        """Full mean vector."""
        return self._mean_vec

    @property
    def covariance(self) -> Array:
        """Full covariance matrix."""
        return self._cov_mat

    def _sample(self, key: PRNGKey) -> dict[str, Array]:
        from .multivariate import MultivariateNormal as MVN
        full_mvn = MVN(loc=self._mean_vec, cov=self._cov_mat)
        flat = full_mvn.sample(key)
        return self._unflatten_flat_vec(flat)

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> dict[str, Array]:
        from .multivariate import MultivariateNormal as MVN
        if key is None:
            key = _auto_key()
        full_mvn = MVN(loc=self._mean_vec, cov=self._cov_mat)
        flat = full_mvn.sample(key, sample_shape)
        return self._unflatten_flat_vec(flat)

    def _unflatten_flat_vec(self, flat: Array) -> dict[str, Array]:
        """Split a flat Gaussian sample vector into per-component arrays."""
        result = {}
        for cname in self._component_shapes:
            sl = self._component_slices[cname]
            result[cname] = flat[..., sl]
        return result

    def log_prob(self, value: dict[str, ArrayLike]) -> Array:
        from .multivariate import MultivariateNormal as MVN
        full_mvn = MVN(loc=self._mean_vec, cov=self._cov_mat)
        flat = self.flatten_value(value)
        return full_mvn.log_prob(flat)

    def mean(self, **kwargs) -> dict[str, Array]:
        result = {}
        for cname in self._component_shapes:
            sl = self._component_slices[cname]
            result[cname] = self._mean_vec[sl]
        return result

    def variance(self, **kwargs) -> dict[str, Array]:
        diag = jnp.diag(self._cov_mat)
        result = {}
        for cname in self._component_shapes:
            sl = self._component_slices[cname]
            result[cname] = diag[sl]
        return result

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
