"""Shared helpers for joint distributions.

Key-path navigation, conditioning argument parsing, and leaf pruning
used by :class:`ProductDistribution`, :class:`SequentialJointDistribution`,
:class:`JointEmpirical`, and :class:`JointGaussian`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..custom_types import ArrayLike
from ..core._array_distributions import ArrayDistribution
from ..core.values import Values
from ..core._values_distribution import ValuesDistribution


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
    ``("x", "y")`` --- plain strings for backward compatibility.

    For a nested dict ``{"g": {"a": Normal}, "b": MVN}``, returns
    ``(("b",), ("g", "a"))`` --- tuples of strings in JAX's canonical
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
    joint: ValuesDistribution,
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
    joint : ValuesDistribution
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


# ---------------------------------------------------------------------------
# Batched flatten / unflatten for Values with known event_shapes
# ---------------------------------------------------------------------------


def _flatten_values_batched(value: Values, event_shapes: dict[str, tuple[int, ...]]) -> jnp.ndarray:
    """Flatten a (possibly batched) Values into ``(*leading, event_size)``."""
    from .._utils import prod

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


def _unflatten_values_batched(flat: jnp.ndarray, event_shapes: dict[str, tuple[int, ...]]) -> Values:
    """Unflatten ``(*leading, event_size)`` back into a Values."""
    from .._utils import prod

    fields: dict[str, jnp.ndarray] = {}
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
