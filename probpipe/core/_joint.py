"""ProductDistribution and supporting helpers.

Provides:
  - ``ProductDistribution``  – Independent-component joint distribution
    (inherits from :class:`ValuesDistribution`).
  - Key-path helpers for navigating nested component pytrees.
  - Conditioning argument parser.
"""

from __future__ import annotations

from types import MappingProxyType

import jax
import jax.numpy as jnp

from ..custom_types import Array, ArrayLike, PRNGKey
from ._array_distributions import (
    ArrayDistribution,
    _mc_expectation,
)
from .provenance import Provenance
from .values import Values
from ._values_distribution import ValuesDistribution, _register_dynamic_subclass
from .protocols import (
    SupportsConditioning,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)


# ---------------------------------------------------------------------------
# Key-path helpers
# ---------------------------------------------------------------------------

# Type alias for key paths used to address components in nested pytrees.
# A KeyPath is a tuple of strings, e.g. ("physics", "force").
KeyPath = tuple[str, ...]


def _normalize_key(key) -> KeyPath:
    """Normalize a component key to a KeyPath tuple.

    Accepts a single string (``"x"``), a tuple of strings
    (``("physics", "force")``), or a sequence passed as ``__getitem__``
    positional args.

    Returns
    -------
    KeyPath
        A tuple of strings, e.g. ``("x",)`` or ``("physics", "force")``.
    """
    if isinstance(key, str):
        return (key,)
    if isinstance(key, tuple) and all(isinstance(k, str) for k in key):
        return key
    raise TypeError(
        f"Component key must be a string or tuple of strings, got {key!r}"
    )


def _walk_pytree(tree, key_path: KeyPath):
    """Navigate a nested dict/pytree by following a key path.

    Parameters
    ----------
    tree : dict or leaf
        The pytree to navigate.
    key_path : KeyPath
        Sequence of string keys, e.g. ``("physics", "force")``.

    Returns
    -------
    object
        The leaf (or sub-tree) at the given path.

    Raises
    ------
    KeyError
        If any key in the path is not found.
    """
    node = tree
    for k in key_path:
        if not isinstance(node, dict):
            raise KeyError(
                f"Cannot navigate further into non-dict node at "
                f"key path prefix; remaining key: {k!r}"
            )
        if k not in node:
            raise KeyError(
                f"Key {k!r} not found. Available: {tuple(node.keys())}"
            )
        node = node[k]
    return node


def _component_key_paths(components) -> tuple:
    """Extract leaf key paths from a pytree of ArrayDistributions.

    For a flat dict ``{"x": Normal, "y": MVN}``, returns
    ``("x", "y")`` — plain strings for backward compatibility.

    For a nested dict ``{"g": {"a": Normal}, "b": MVN}``, returns
    ``(("b",), ("g", "a"))`` — tuples of strings in JAX's canonical
    traversal order (sorted dict keys, depth-first).

    Parameters
    ----------
    components : dict
        A (possibly nested) dict whose leaves are ArrayDistribution.

    Returns
    -------
    tuple[str, ...] or tuple[KeyPath, ...]
        Leaf identifiers.  Plain strings if the dict is flat; tuples
        of strings if any nesting is present.
    """
    # Check if the dict is flat (all values are ArrayDistribution leaves)
    is_flat = isinstance(components, dict) and all(
        isinstance(v, ArrayDistribution) for v in components.values()
    )
    if is_flat:
        return tuple(components.keys())

    # Nested: extract full key paths from JAX's traversal
    paths_and_leaves = jax.tree_util.tree_leaves_with_path(components)
    key_paths = []
    for path, _leaf in paths_and_leaves:
        str_path = tuple(
            k.key if hasattr(k, "key") else str(k) for k in path
        )
        key_paths.append(str_path)
    return tuple(key_paths)


# ---------------------------------------------------------------------------
# Values template builder
# ---------------------------------------------------------------------------


def _build_values_template(components: dict) -> Values:
    """Build a Values template from a component pytree.

    Each ``ArrayDistribution`` leaf becomes a ``jnp.zeros(event_shape)``
    placeholder.  Nested dicts become nested ``Values``.
    """
    fields: dict[str, jnp.ndarray | Values] = {}
    for name, comp in components.items():
        if isinstance(comp, dict):
            fields[name] = _build_values_template(comp)
        elif isinstance(comp, ArrayDistribution):
            fields[name] = jnp.zeros(comp.event_shape)
        else:
            raise TypeError(f"Unexpected component type: {type(comp).__name__}")
    return Values(fields)


# ---------------------------------------------------------------------------
# Conditioning argument parser (module-level helper)
# ---------------------------------------------------------------------------

def _collect_observed_leaves(
    obs_tree: dict,
    comp_tree: dict,
    prefix: KeyPath,
    out: dict[KeyPath, ArrayLike],
) -> None:
    """Recursively walk an observed-value tree, validating against the component tree.

    For each leaf (non-dict) value in *obs_tree*, verifies that the
    corresponding node in *comp_tree* is an ``ArrayDistribution``
    (not an internal dict node), and records the ``(key_path, value)``
    pair in *out*.

    Parameters
    ----------
    obs_tree : dict
        The user-provided observed values (possibly nested).
    comp_tree : dict
        The corresponding level of the component pytree.
    prefix : KeyPath
        Key path accumulated so far.
    out : dict
        Accumulator for ``{key_path: value}`` pairs.

    Raises
    ------
    KeyError
        If a key in *obs_tree* is not present in *comp_tree*.
    TypeError
        If the user provides a scalar/array for a key that maps to
        an intermediate dict node (must condition on individual
        component distributions), or provides a dict for a key that
        maps to a component distribution.
    """
    for key, obs_val in obs_tree.items():
        path = prefix + (key,)
        path_str = " > ".join(path)

        # Validate key exists
        if not isinstance(comp_tree, dict) or key not in comp_tree:
            available = tuple(comp_tree.keys()) if isinstance(comp_tree, dict) else ()
            raise KeyError(
                f"Component key {key!r} not found at level "
                f"'{' > '.join(prefix)}'. "
                f"Available keys: {available}"
            )

        comp_node = comp_tree[key]

        if isinstance(obs_val, dict):
            # User provided a nested dict — component must also be a dict
            if isinstance(comp_node, ArrayDistribution):
                raise TypeError(
                    f"Key path '{path_str}' resolves to a component "
                    f"distribution ({type(comp_node).__name__}), but a "
                    f"dict of values was provided.  Pass a single array "
                    f"value to condition on this component."
                )
            if not isinstance(comp_node, dict):
                raise TypeError(
                    f"Key path '{path_str}' resolves to "
                    f"{type(comp_node).__name__}, which is neither a "
                    f"component distribution nor a dict."
                )
            # Recurse
            _collect_observed_leaves(obs_val, comp_node, path, out)
        else:
            # User provided a value — must be a component distribution
            if isinstance(comp_node, dict):
                leaf_names = list(comp_node.keys())
                raise TypeError(
                    f"Cannot condition on '{path_str}' with a single "
                    f"value — it contains component distributions "
                    f"{leaf_names}.  Provide values for individual "
                    f"component distributions, e.g.: "
                    f"condition_on({key}={{'{leaf_names[0]}': ...}})"
                )
            if not isinstance(comp_node, ArrayDistribution):
                raise TypeError(
                    f"Key path '{path_str}' resolves to "
                    f"{type(comp_node).__name__}, not a component "
                    f"distribution."
                )
            out[path] = obs_val


def _parse_condition_args(
    joint: JointDistribution,
    observed: dict | None,
    kwargs: dict,
) -> dict[KeyPath, ArrayLike]:
    """Parse and validate conditioning arguments for a joint distribution.

    Normalizes both calling conventions (positional dict and kwargs)
    into a flat mapping ``{key_path: observed_value}`` where each
    key path addresses a leaf ``ArrayDistribution`` in the component
    pytree.

    Parameters
    ----------
    joint : JointDistribution
        The joint distribution whose components are being conditioned.
    observed : dict or None
        Positional dict argument from ``_condition_on``.
    kwargs : dict
        Keyword arguments from ``_condition_on``.

    Returns
    -------
    dict[KeyPath, ArrayLike]
        Mapping from leaf key paths to observed values.

    Raises
    ------
    TypeError
        If both ``observed`` and ``kwargs`` are provided, or if a
        key path resolves to an internal node.
    KeyError
        If a key is not found in the component pytree.
    ValueError
        If no leaves are specified, or if all leaves are conditioned.
    """
    if observed is not None and kwargs:
        raise TypeError(
            "condition_on() accepts either a positional dict or keyword "
            "arguments, not both."
        )
    tree = observed if observed is not None else kwargs
    if not tree:
        raise ValueError("condition_on() requires at least one observed value.")

    # Walk the provided tree to extract (key_path, value) for each leaf
    observed_leaves: dict[KeyPath, ArrayLike] = {}
    _collect_observed_leaves(tree, joint._components, prefix=(), out=observed_leaves)

    # Check we're not conditioning on everything
    all_leaf_paths = set(
        p if isinstance(p, tuple) else (p,)
        for p in _component_key_paths(joint._components)
    )
    conditioned_paths = set(observed_leaves.keys())
    if conditioned_paths >= all_leaf_paths:
        raise ValueError(
            "Cannot condition on all component distributions — "
            "at least one must remain unconditioned."
        )
    return observed_leaves


# ---------------------------------------------------------------------------
# ProductDistribution — independent components
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dynamic protocol factory for ProductDistribution
# ---------------------------------------------------------------------------

_PRODUCT_CLASS_CACHE: dict[frozenset[str], type] = {}


def _product_class_for_components(components: dict) -> type:
    """Return a ProductDistribution subclass whose protocol bases match
    what ALL leaf components support.

    SupportsSampling and SupportsConditioning are always included.
    SupportsLogProb, SupportsMean, SupportsVariance are included only
    when every leaf component supports them.
    """
    leaves = jax.tree.leaves(components)

    protocols: set[str] = set()
    if all(isinstance(l, SupportsLogProb) for l in leaves):
        protocols.add("log_prob")
    if all(isinstance(l, SupportsMean) for l in leaves):
        protocols.add("mean")
    if all(isinstance(l, SupportsVariance) for l in leaves):
        protocols.add("variance")

    key = frozenset(protocols)
    if key in _PRODUCT_CLASS_CACHE:
        return _PRODUCT_CLASS_CACHE[key]

    extra_bases: list[type] = []
    if "log_prob" in protocols:
        extra_bases.append(SupportsLogProb)
    if "mean" in protocols:
        extra_bases.append(SupportsMean)
    if "variance" in protocols:
        extra_bases.append(SupportsVariance)

    if not extra_bases:
        _PRODUCT_CLASS_CACHE[key] = ProductDistribution
        return ProductDistribution

    cls = type("ProductDistribution", (ProductDistribution, *extra_bases), {})
    _register_dynamic_subclass(cls)
    _PRODUCT_CLASS_CACHE[key] = cls
    return cls


# ---------------------------------------------------------------------------
# ProductDistribution
# ---------------------------------------------------------------------------


class ProductDistribution(
    ValuesDistribution,
    SupportsSampling, SupportsConditioning,
):
    """Joint distribution with **independent** leaf components.

    Inherits from :class:`ValuesDistribution`.  All leaf components are
    sampled independently.  ``_sample()`` returns :class:`Values`.

    **Dynamic protocol support:** ``SupportsLogProb``, ``SupportsMean``,
    and ``SupportsVariance`` are included only when ALL leaf components
    support them.  ``isinstance(product, SupportsLogProb)`` is ``True``
    only when every component has ``_log_prob``.

    Parameters
    ----------
    name : str, optional
        Distribution name.
    **components : ArrayDistribution or dict
        Named independent component distributions.  Values may be
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
        for leaf in jax.tree.leaves(components):
            if not isinstance(leaf, ArrayDistribution):
                raise TypeError(
                    f"All leaf components must be ArrayDistribution, "
                    f"got {type(leaf).__name__}"
                )
        self._components = dict(components)
        self._name = name
        self._values_template = _build_values_template(self._components)

    # -- Sampling (returns Values) ------------------------------------------

    def _sample_one(self, key: PRNGKey) -> Values:
        return self._sample(key)

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Values:
        """Draw independent samples from each component, returning Values."""
        sorted_names = sorted(self._components.keys())
        keys = jax.random.split(key, len(sorted_names))
        fields: dict[str, jnp.ndarray | Values] = {}
        for subkey, name in zip(keys, sorted_names):
            comp = self._components[name]
            if isinstance(comp, dict):
                fields[name] = _sample_nested(comp, subkey, sample_shape)
            else:
                fields[name] = comp._sample(subkey, sample_shape)
        return Values(fields)

    # -- Log-prob -----------------------------------------------------------

    def _log_prob(self, value) -> Array:
        """Sum of independent leaf log-probs."""
        if isinstance(value, Values):
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

    # -- Moments (return Values) --------------------------------------------

    def _mean(self) -> Values:
        return _map_components(self._components, lambda d: d._mean())

    def _variance(self) -> Values:
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


# -- Helpers for nested component pytrees ----------------------------------


def _sample_nested(components: dict, key, sample_shape) -> Values:
    """Recursively sample from nested component dicts, returning nested Values."""
    sorted_names = sorted(components.keys())
    keys = jax.random.split(key, len(sorted_names))
    fields: dict = {}
    for subkey, name in zip(keys, sorted_names):
        comp = components[name]
        if isinstance(comp, dict):
            fields[name] = _sample_nested(comp, subkey, sample_shape)
        else:
            fields[name] = comp._sample(subkey, sample_shape)
    return Values(fields)


def _map_components(components: dict, fn) -> Values:
    """Apply fn to each leaf distribution, returning nested Values."""
    fields: dict = {}
    for name, comp in components.items():
        if isinstance(comp, dict):
            fields[name] = _map_components(comp, fn)
        else:
            fields[name] = fn(comp)
    return Values(fields)


def _prune_leaves(tree: dict, remove_paths: set[KeyPath], prefix: tuple = ()) -> dict:
    """Remove specified leaves from a nested dict and prune empty sub-dicts.

    Parameters
    ----------
    tree : dict
        The component pytree (nested dict of ``ArrayDistribution`` leaves).
    remove_paths : set[KeyPath]
        Set of key paths to remove.
    prefix : tuple
        Current path prefix (used in recursion).

    Returns
    -------
    dict
        A new dict with the specified leaves removed.  Empty intermediate
        dicts are also removed.
    """
    result = {}
    for key, value in tree.items():
        path = prefix + (key,)
        if path in remove_paths:
            continue
        if isinstance(value, dict):
            pruned = _prune_leaves(value, remove_paths, path)
            if pruned:
                result[key] = pruned
        else:
            result[key] = value
    return result
