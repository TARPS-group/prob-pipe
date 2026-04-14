"""ProductDistribution --- independent-component joint distribution.

Provides:
  - ``ProductDistribution``  -- Independent-component joint distribution
    (inherits from :class:`RecordDistribution`).
  - ``TFPProductDistribution`` -- Subclass that exposes a combined TFP
    distribution (``_tfp_dist``) when all leaf components are TFP-backed.
  - Dynamic protocol factory for automatic protocol support.
  - Helpers for nested component sampling and mapping.
  - JAX pytree registration.
"""

from __future__ import annotations

from types import MappingProxyType

import jax
import jax.numpy as jnp

from ..custom_types import Array, ArrayLike, PRNGKey
from ..core._array_distributions import (
    ArrayDistribution,
    _mc_expectation,
)
from ..core.provenance import Provenance
from ..core.record import Record
from ..core._record_distribution import (
    RecordDistribution,
    _register_dynamic_subclass,
    _build_record_template,
)
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
    _prune_leaves,
)


# ---------------------------------------------------------------------------
# Dynamic protocol factory for ProductDistribution
# ---------------------------------------------------------------------------

_PRODUCT_CLASS_CACHE: dict[frozenset[str], type] = {}


def _product_class_for_components(components: dict) -> type:
    """Return a ProductDistribution subclass whose protocol bases match
    what ALL leaf components support.

    SupportsSampling and SupportsConditioning are always included.
    SupportsLogProb, SupportsMean, SupportsVariance are included only
    when every leaf component supports them.  When all leaf components
    are TFP-backed, ``TFPProductDistribution`` is used as the base
    to expose a combined ``_tfp_dist``.
    """
    leaves = jax.tree.leaves(components)

    protocols: set[str] = set()
    if all(isinstance(l, SupportsLogProb) for l in leaves):
        protocols.add("log_prob")
    if all(isinstance(l, SupportsMean) for l in leaves):
        protocols.add("mean")
    if all(isinstance(l, SupportsVariance) for l in leaves):
        protocols.add("variance")
    all_tfp = all(hasattr(l, "_tfp_dist") for l in leaves)
    if all_tfp:
        protocols.add("tfp")

    key = frozenset(protocols)
    if key in _PRODUCT_CLASS_CACHE:
        return _PRODUCT_CLASS_CACHE[key]

    base = TFPProductDistribution if all_tfp else ProductDistribution

    extra_bases: list[type] = []
    if "log_prob" in protocols:
        extra_bases.append(SupportsLogProb)
    if "mean" in protocols:
        extra_bases.append(SupportsMean)
    if "variance" in protocols:
        extra_bases.append(SupportsVariance)

    if not extra_bases:
        _PRODUCT_CLASS_CACHE[key] = base
        return base

    cls = type("ProductDistribution", (base, *extra_bases), {})
    _register_dynamic_subclass(cls)
    _PRODUCT_CLASS_CACHE[key] = cls
    return cls


def _validate_nested_names(parent_key: str, d: dict) -> None:
    """Recursively validate that leaf distribution names match their dict keys."""
    for key, val in d.items():
        if isinstance(val, dict):
            _validate_nested_names(key, val)
        elif hasattr(val, "name") and val.name != key:
            raise ValueError(
                f"Component name mismatch inside '{parent_key}': "
                f"keyword '{key}' but distribution has name='{val.name}'"
            )


# ---------------------------------------------------------------------------
# ProductDistribution
# ---------------------------------------------------------------------------


class ProductDistribution(
    RecordDistribution,
    SupportsSampling, SupportsConditioning,
):
    """Joint distribution with **independent** leaf components.

    Inherits from :class:`RecordDistribution`.  All leaf components are
    sampled independently.  ``_sample()`` returns :class:`Record`.

    **Dynamic protocol support:** ``SupportsLogProb``, ``SupportsMean``,
    and ``SupportsVariance`` are included only when ALL leaf components
    support them.  ``isinstance(product, SupportsLogProb)`` is ``True``
    only when every component has ``_log_prob``.

    Parameters
    ----------
    name : str, optional
        Distribution name.
    **components : ArrayDistribution or dict
        Named independent component distributions.  Record may be
        ``ArrayDistribution`` instances (leaves) or nested dicts
        whose leaves are ``ArrayDistribution`` instances.
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __new__(cls, *, name: str | None = None, **components):
        if not components:
            return object.__new__(cls)
        actual_cls = _product_class_for_components(components)
        return object.__new__(actual_cls)

    def __init__(self, *, name: str | None = None, **components):
        if not components:
            raise ValueError("ProductDistribution requires at least one component.")
        for key, comp in components.items():
            if isinstance(comp, dict):
                _validate_nested_names(key, comp)
            elif comp.name != key:
                raise ValueError(
                    f"Component name mismatch: keyword '{key}' but "
                    f"distribution has name='{comp.name}'"
                )
        for leaf in jax.tree.leaves(components):
            if not isinstance(leaf, ArrayDistribution):
                raise TypeError(
                    f"All leaf components must be ArrayDistribution, "
                    f"got {type(leaf).__name__}"
                )
        self._components = dict(components)
        if name is None:
            name = "product(" + ",".join(sorted(components.keys())) + ")"
        super().__init__(name=name)
        self._record_template = _build_record_template(self._components)

    # -- Sampling (returns Record) ------------------------------------------

    def _sample_one(self, key: PRNGKey) -> Record:
        return self._sample(key)

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Record:
        """Draw independent samples from each component, returning Record."""
        sorted_names = sorted(self._components.keys())
        keys = jax.random.split(key, len(sorted_names))
        fields: dict[str, jnp.ndarray | Record] = {}
        for subkey, name in zip(keys, sorted_names):
            comp = self._components[name]
            if isinstance(comp, dict):
                fields[name] = _sample_nested(comp, subkey, sample_shape)
            else:
                fields[name] = comp._sample(subkey, sample_shape)
        return Record(fields)

    # -- Log-prob -----------------------------------------------------------

    def _log_prob(self, value) -> Array:
        """Sum of independent leaf log-probs.

        Accepts Record, dict, or flat array (auto-unflattened via template).
        """
        if isinstance(value, jnp.ndarray) and self._record_template is not None:
            value = Record.unflatten(value, template=self._record_template)
        if isinstance(value, Record):
            value = value.to_dict()
        lp_tree = jax.tree.map(
            lambda dist, val: dist._log_prob(jnp.asarray(val)),
            self._components,
            value,
        )
        lp_leaves = jax.tree.leaves(lp_tree)
        total = lp_leaves[0]
        for lp in lp_leaves[1:]:
            total = total + lp
        return total

    # -- Moments (return Record) --------------------------------------------

    def _mean(self) -> Record:
        return _map_components(self._components, lambda d: d._mean())

    def _variance(self) -> Record:
        return _map_components(self._components, lambda d: d._variance())

    def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
        return _mc_expectation(self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist)

    # -- Component access (for backward compat) ----------------------------

    @property
    def components(self):
        """Read-only view of the component distributions."""
        if all(isinstance(v, ArrayDistribution) for v in self._components.values()):
            return MappingProxyType(self._components)
        return self._components

    # -- Conditioning -------------------------------------------------------

    def _condition_on(self, observed=None, /, **kwargs):
        observed_leaves = _parse_condition_args(self, observed, kwargs)
        return self._condition_on_impl(observed_leaves)

    def _condition_on_impl(
        self, observed_leaves: dict[KeyPath, ArrayLike],
    ) -> ProductDistribution:
        new_components = _prune_leaves(self._components, set(observed_leaves.keys()))
        result = ProductDistribution(**new_components, name=self._name)
        conditioned_names = [" > ".join(path) for path in observed_leaves]
        result.with_source(Provenance(
            "condition_on", parents=(self,),
            metadata={"conditioned": conditioned_names},
        ))
        return result

    def __repr__(self) -> str:
        comp_str = ", ".join(
            f"{k}={type(v).__name__}" if isinstance(v, ArrayDistribution)
            else f"{k}={{...}}"
            for k, v in self._components.items()
        )
        name_str = f", name='{self._name}'" if self._name else ""
        return f"ProductDistribution({comp_str}{name_str})"


# ---------------------------------------------------------------------------
# TFPProductDistribution — TFP-backed subclass
# ---------------------------------------------------------------------------

# Metadata parameter names that should not be stacked into arrays.
_TFP_PARAM_SKIP = frozenset({"name", "validate_args", "allow_nan_stats"})


class TFPProductDistribution(ProductDistribution):
    """ProductDistribution subclass that exposes a combined TFP distribution.

    Instantiated automatically by ``ProductDistribution.__new__`` when all
    leaf components are TFP-backed (i.e., have a ``_tfp_dist`` attribute).
    Provides ``event_shape``, ``batch_shape``, ``dtype``, and ``_tfp_dist``
    for interop with SBI and other TFP-dependent subsystems.
    """

    def __init__(self, *, name: str | None = None, **components):
        super().__init__(name=name, **components)
        self._build_tfp_dist()

    def _build_tfp_dist(self):
        """Construct a combined TFP distribution from the components.

        Collects component TFP distributions in sorted field order
        (matching the ``Record`` layout).  For the common case of
        same-family scalar distributions, stacks parameters into a
        single ``tfd.Independent`` (bijector-compatible with sbijax).
        Falls back to ``tfd.Blockwise`` for mixed or vector components.
        """
        from tensorflow_probability.substrates.jax import distributions as tfd

        tfp_dists = []
        for comp_name in sorted(self._components.keys()):
            comp = self._components[comp_name]
            if isinstance(comp, dict):
                for sub_leaf in jax.tree.leaves(comp):
                    tfp_dists.append(sub_leaf._tfp_dist)
            else:
                tfp_dists.append(comp._tfp_dist)

        all_same_type = len(set(type(d) for d in tfp_dists)) == 1
        all_scalar = all(d.event_shape.rank == 0 for d in tfp_dists)
        if all_same_type and all_scalar:
            exemplar = tfp_dists[0]
            stacked_params = {}
            for pname, pval in exemplar.parameters.items():
                if pname in _TFP_PARAM_SKIP:
                    continue
                vals = [d.parameters[pname] for d in tfp_dists]
                if all(v is not None for v in vals):
                    stacked_params[pname] = jnp.stack(vals)
            self._tfp_dist = tfd.Independent(
                type(exemplar)(**stacked_params),
                reinterpreted_batch_ndims=1,
            )
        else:
            self._tfp_dist = tfd.Blockwise(tfp_dists)

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self._tfp_dist.event_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self._tfp_dist.batch_shape)

    @property
    def dtype(self):
        return self._tfp_dist.dtype


# -- Helpers for nested component pytrees ----------------------------------


def _sample_nested(components: dict, key, sample_shape) -> Record:
    """Recursively sample from nested component dicts, returning nested Record."""
    sorted_names = sorted(components.keys())
    keys = jax.random.split(key, len(sorted_names))
    fields: dict = {}
    for subkey, name in zip(keys, sorted_names):
        comp = components[name]
        if isinstance(comp, dict):
            fields[name] = _sample_nested(comp, subkey, sample_shape)
        else:
            fields[name] = comp._sample(subkey, sample_shape)
    return Record(fields)


def _map_components(components: dict, fn) -> Record:
    """Apply fn to each leaf distribution, returning nested Record."""
    fields: dict = {}
    for name, comp in components.items():
        if isinstance(comp, dict):
            fields[name] = _map_components(comp, fn)
        else:
            fields[name] = fn(comp)
    return Record(fields)


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
