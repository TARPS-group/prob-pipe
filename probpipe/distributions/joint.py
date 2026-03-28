"""Joint distributions and component views for correlated broadcasting.

This module implements joint distributions whose samples are **pytrees of
arrays** (typically nested dicts).  The base class
:class:`JointDistribution` inherits from
:class:`PyTreeArrayDistribution[T]`, so all pytree shape semantics
(``event_shapes``, ``event_size``, ``flatten_value`` /
``unflatten_value``, ``as_flat_distribution()``) are available.

**Component structure:**

Components are stored as a pytree whose leaves are
``ArrayDistribution`` instances.  The simplest case is a flat dict::

    ProductDistribution(x=Normal(0, 1), y=Normal(1, 2))

but components may also be nested dicts::

    ProductDistribution(
        physics={"force": Normal(0, 1), "mass": Gamma(2, 1)},
        observation=Normal(0, 0.1),
    )

The pytree structure of the components determines the structure of
samples, ``event_shapes``, ``_log_prob`` inputs, etc.

**Structural contract:**

*  ``_sample(key, sample_shape)`` returns a pytree with the same
   structure as the components, but with ``ArrayDistribution`` leaves
   replaced by arrays of shape
   ``(*sample_shape, *batch_shape, *leaf_event_shape)``.
*  ``_log_prob(value)`` accepts a pytree with the same structure.
*  ``event_shapes`` returns a pytree with the same structure, where
   each leaf is the ``event_shape`` tuple of the corresponding
   component distribution.
*  ``batch_shape`` is shared across all components (an empty tuple unless
   the component distributions themselves are batched identically).
*  ``flatten_value`` / ``unflatten_value`` (inherited from
   ``PyTreeArrayDistribution``) convert between the pytree
   representation and a flat ``(*leading_dims, event_size)`` array.
   Leaf ordering follows JAX's canonical pytree traversal order
   (sorted dict keys, depth-first).
*  ``as_flat_distribution()`` returns a :class:`FlattenedView` with
   ``event_shape = (event_size,)`` for use with algorithms expecting
   flat vectors.

**Component access:**

*  ``joint["name"]`` returns a :class:`DistributionView` when the
   key resolves to a component distribution (leaf).
*  ``joint["physics", "force"]`` returns a ``DistributionView`` for a
   nested component, navigating through intermediate dict levels.
*  ``joint["physics"]`` returns a new :class:`ProductDistribution`
   when the key resolves to an intermediate dict node (a sub-tree).
   This sub-joint is the **marginal** distribution over the
   components in that sub-tree — sampling it draws only from those
   components.
*  ``component_names`` returns a tuple of key paths — plain strings
   for flat dicts, or tuples of strings for nested components.

**Independence assumptions:**

*  :class:`JointDistribution` (base) makes **no** independence assumption.
   Subclasses define the factorization structure.
*  :class:`ProductDistribution` assumes **all leaf components are
   independent**.  ``log_prob`` is the sum of per-leaf log-probs with
   no coupling.

**Subclass structure limitations:**

*  :class:`SequentialJointDistribution`, :class:`JointEmpirical`, and
   :class:`JointGaussian` currently support **flat dicts only** (no
   nesting).  This is because sequential dependencies use function
   parameter names (inherently flat), empirical distributions store
   sample arrays keyed by name, and Gaussian conditioning operates on
   flat index slices.
"""
from __future__ import annotations

import inspect
from types import MappingProxyType
from typing import Any, Callable

import jax
import jax.numpy as jnp
from .._utils import prod

from ..custom_types import Array, ArrayLike, PRNGKey
from ..core.distribution import (
    ArrayDistribution,
    PyTreeArrayDistribution,
    EmpiricalDistribution,
    Provenance,
    Constraint,
    real,
    _auto_key,
    _vmap_sample,
    _mc_expectation,
)
from ..core.protocols import (
    SupportsConditioning,
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsNamedComponents,
    SupportsSampling,
    SupportsVariance,
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

# Type alias for key paths used to address components in nested pytrees.
# A KeyPath is a tuple of strings, e.g. ("physics", "force").
# For flat dicts, component_names returns plain strings for convenience,
# but DistributionView and __getitem__ always normalize to tuples internally.
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
# DistributionView — lightweight reference to a JointDistribution component
# ---------------------------------------------------------------------------

class DistributionView(ArrayDistribution, SupportsSampling, SupportsLogProb, SupportsMean, SupportsVariance):
    """A lightweight reference to a leaf component of a :class:`JointDistribution`.

    .. note:: Sampling and expectation are provided via ``SupportsSampling``.

    A ``DistributionView`` acts as an ``ArrayDistribution`` for a single
    component distribution (leaf) of a joint distribution's pytree.  It
    is the object returned by ``joint["component_name"]`` or
    ``joint["physics", "force"]``.

    **Broadcasting contract:** The broadcasting logic in
    :class:`~probpipe.core.node.WorkflowFunction` detects
    ``DistributionView`` instances and groups those sharing the same
    ``_parent`` so they are sampled jointly (preserving correlation
    between components).

    **Standalone sampling:** When sampled outside of broadcasting, the
    view draws a full joint sample from its parent and extracts this
    component by walking the sample pytree using the stored key path.

    **Key path:** The view stores a :data:`KeyPath` — a tuple of
    strings that addresses a leaf in the parent's (possibly nested)
    pytree structure.  For flat joints ``("x",)`` navigates to
    ``sample["x"]``; for nested joints ``("physics", "force")``
    navigates to ``sample["physics"]["force"]``.

    **Backward compatibility:** The ``_component_name`` property
    returns the last element of the key path (the leaf name) for
    code that reads it as a simple string.

    Parameters
    ----------
    parent : JointDistribution
        The joint distribution this view belongs to.
    key_path : str or KeyPath
        A string (for flat access) or tuple of strings (for nested
        access) identifying the leaf component.
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __init__(self, parent: JointDistribution, key_path: str | KeyPath):
        self._key_path: KeyPath = _normalize_key(key_path)
        # Validate: the key path must resolve to an ArrayDistribution leaf
        leaf = _walk_pytree(parent._components, self._key_path)
        if not isinstance(leaf, ArrayDistribution):
            raise KeyError(
                f"Key path {self._key_path} resolves to "
                f"{type(leaf).__name__}, not an ArrayDistribution leaf. "
                f"Use a longer key path to reach a leaf."
            )
        self._parent = parent
        self._component = leaf
        self._name = self._key_path[-1]

    @property
    def _component_name(self) -> str:
        """Leaf name (last element of key path).

        Provided for backward compatibility with code that reads
        ``view._component_name`` as a simple string — including
        the broadcasting logic in
        :class:`~probpipe.core.node.WorkflowFunction`.
        """
        return self._key_path[-1] if len(self._key_path) == 1 else None

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

    def _sample_one(self, key: PRNGKey) -> Array:
        structured = self._parent._sample(key)
        return _walk_pytree(structured, self._key_path)

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        structured = self._parent._sample(key, sample_shape)
        return _walk_pytree(structured, self._key_path)

    def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
        return _mc_expectation(self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist)

    def _log_prob(self, x: ArrayLike) -> Array:
        return self._component._log_prob(x)

    def _mean(self) -> Array:
        return self._component._mean()

    def _variance(self) -> Array:
        return self._component._variance()

    @classmethod
    def _from_distribution(cls, other, *, key, **kwargs):
        raise NotImplementedError(
            "Cannot convert to DistributionView; it is a structural reference."
        )

    @classmethod
    def _default_support(cls) -> Constraint:
        return real

    def __repr__(self) -> str:
        path_str = " > ".join(self._key_path)
        return (
            f"DistributionView(parent={type(self._parent).__name__}, "
            f"path='{path_str}')"
        )


# ---------------------------------------------------------------------------
# ConditionedComponent — a distribution fixed to an observed value
# ---------------------------------------------------------------------------

class ConditionedComponent(ArrayDistribution, SupportsSampling, SupportsLogProb, SupportsMean, SupportsVariance):
    """
    A distribution whose value is fixed (observed / conditioned on).

    Used internally by conditioning to replace a component with a constant.
    Sampling always returns the conditioned value; ``log_prob`` evaluates the
    base distribution at the conditioned value.
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

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

    def _sample_one(self, key: PRNGKey) -> Array:
        return self._value

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        if sample_shape == ():
            return self._value
        return jnp.broadcast_to(self._value, sample_shape + self.event_shape)

    def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
        return _mc_expectation(self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist)

    def _log_prob(self, x: ArrayLike) -> Array:
        # Log-prob is constant: the base distribution evaluated at the pinned value
        return self._base._log_prob(self._value)

    def _mean(self) -> Array:
        return self._value

    def _variance(self) -> Array:
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

class JointDistribution(PyTreeArrayDistribution, SupportsNamedComponents):
    """Base class for named multi-component distributions.

    A ``JointDistribution`` is a :class:`PyTreeArrayDistribution` whose
    samples are **pytrees** (typically dicts, possibly nested) mapping
    component names to arrays.  Each leaf of the component pytree is
    an ``ArrayDistribution`` that defines the marginal (or conditional)
    distribution for that component.

    **Component structure:**

    Components are stored in ``self._components`` as a pytree whose
    leaves are ``ArrayDistribution`` instances.  For flat dicts this is
    ``{"x": Normal, "y": MVN}``.  For nested dicts this is e.g.
    ``{"physics": {"force": Normal, "mass": Gamma}, "obs": Normal}``.

    The pytree structure determines:

    *  The ``treedef`` — derived from the component pytree by replacing
       each ``ArrayDistribution`` leaf with a scalar placeholder, then
       calling ``jax.tree.structure()``.
    *  ``event_shapes`` — a pytree of the same structure where each
       leaf is the component's ``event_shape`` tuple.
    *  ``component_names`` — a tuple of key paths.  For flat dicts these
       are plain strings ``("x", "y")``.  For nested dicts these are
       tuples ``(("physics", "force"), ("physics", "mass"), ("obs",))``,
       in JAX's canonical traversal order (sorted dict keys, depth-first).

    **Flat-dict backward compatibility:**

    When all top-level values are ``ArrayDistribution`` leaves (no
    nesting), the API behaves identically to a flat-dict joint:
    ``component_names`` returns plain strings, ``__getitem__`` accepts
    a single string, and ``_sample()`` returns a flat dict.

    **Sample type:**

    ``_sample(key, sample_shape)`` returns a pytree with the same
    structure as the components, but with ``ArrayDistribution`` leaves
    replaced by arrays of shape
    ``(*sample_shape, *batch_shape, *leaf_event_shape)``.

    **Log-prob contract:**

    ``_log_prob(value)`` accepts a pytree with the same structure and
    returns a scalar (or batch-shaped array).  The base class raises
    ``NotImplementedError``; subclasses define the factorization.

    **Flat-vector interop:**

    The inherited ``flatten_value`` / ``unflatten_value`` methods convert
    between the pytree representation and flat
    ``(*leading, event_size)`` arrays.  ``as_flat_distribution()``
    returns a :class:`FlattenedView` with ``event_shape = (event_size,)``
    for algorithms expecting flat vectors.

    **Component access:**

    ``joint["name"]`` returns a :class:`DistributionView` — an
    ``ArrayDistribution`` that extracts the named component from
    joint samples.  For nested dicts, use a key-path tuple like
    ``joint["physics", "force"]`` to reach a nested component.
    ``joint.bind(a="x", b="y")`` creates a dict of views with
    remapped names for use in broadcasting.

    **Subclass note:**

    :class:`SequentialJointDistribution`, :class:`JointEmpirical`, and
    :class:`JointGaussian` override ``__init__`` and only support flat
    dicts.  Nested pytree support is available in
    :class:`ProductDistribution` and the base ``JointDistribution``.

    Parameters
    ----------
    name : str, optional
        Distribution name for provenance / display.
    **components : ArrayDistribution or dict
        Named component distributions.  Values may be
        ``ArrayDistribution`` instances (leaves) or nested dicts
        whose leaves are ``ArrayDistribution`` instances.
        At least one leaf component is required.
    """

    def __init__(self, *, name: str | None = None, **components):
        if not components:
            raise ValueError("JointDistribution requires at least one component.")
        # Validate: all leaves must be ArrayDistribution
        leaves = jax.tree.leaves(components)
        if not leaves:
            raise ValueError("JointDistribution requires at least one component.")
        for leaf in leaves:
            if not isinstance(leaf, ArrayDistribution):
                raise TypeError(
                    f"All leaf components must be ArrayDistribution, "
                    f"got {type(leaf).__name__}"
                )
        self._components = dict(components)
        self._name = name

    # -- PyTreeArrayDistribution interface ---------------------------------

    @property
    def treedef(self) -> jax.tree_util.PyTreeDef:
        """Pytree structure of a single sample.

        Derived from the component pytree by replacing each
        ``ArrayDistribution`` leaf with a scalar placeholder.
        For flat dicts this produces a flat-dict ``PyTreeDef``;
        for nested dicts it produces a nested-dict ``PyTreeDef``.
        """
        prototype = jax.tree.map(lambda _: 0, self._components)
        return jax.tree.structure(prototype)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape, shared across all components.

        Defaults to ``()``.  If the component distributions themselves
        have non-trivial batch shapes, they must all agree.
        """
        return ()

    @property
    def event_shapes(self):
        """Per-component event shapes as a pytree.

        Returns a pytree with the same structure as the components,
        where each leaf is the ``event_shape`` tuple of the
        corresponding ``ArrayDistribution``.

        For a flat dict ``{"x": Normal, "y": MVN(3)}`` this returns
        ``{"x": (), "y": (3,)}``.  For a nested dict it returns
        a nested dict of shapes.
        """
        return jax.tree.map(lambda d: d.event_shape, self._components)

    # -- Properties --------------------------------------------------------

    @property
    def components(self):
        """The component pytree (read-only for flat dicts).

        For flat dicts, returns a ``MappingProxyType`` for read-only
        access.  For nested dicts, returns the dict directly (Python
        does not support nested read-only views without deep copying).
        """
        if self._is_flat:
            return MappingProxyType(self._components)
        return self._components

    @property
    def _is_flat(self) -> bool:
        """True if all top-level dict values are ArrayDistribution leaves."""
        return isinstance(self._components, dict) and all(
            isinstance(v, ArrayDistribution) for v in self._components.values()
        )

    @property
    def component_names(self) -> tuple:
        """Leaf component identifiers.

        For flat dicts, returns a tuple of plain strings::

            ("x", "y")

        For nested dicts, returns a tuple of key-path tuples in JAX's
        canonical traversal order (sorted dict keys, depth-first)::

            (("physics", "force"), ("physics", "mass"), ("obs",))

        These key paths can be passed to ``joint[key_path]`` to get
        a :class:`DistributionView` for the leaf.
        """
        return _component_key_paths(self._components)

    # -- Structured access -------------------------------------------------

    def __getitem__(self, key):
        """Access a component by key path.

        The return type depends on what the key path resolves to in the
        component pytree:

        *  **Component distribution** (``ArrayDistribution`` leaf):
           returns a :class:`DistributionView` — a lightweight
           ``ArrayDistribution`` whose samples are the marginal values
           of that component.
        *  **Intermediate dict node**: returns a new
           :class:`ProductDistribution` wrapping the sub-tree.  This
           sub-joint is the **marginal** distribution over the
           component distributions in that sub-tree.

        Accepts a single string for top-level access, or a tuple of
        strings for nested access::

            joint["x"]               # component → DistributionView
            joint["physics", "force"]  # nested component → DistributionView
            joint["physics"]          # intermediate node → ProductDistribution

        Parameters
        ----------
        key : str or tuple[str, ...]
            Component key path.

        Returns
        -------
        DistributionView or ProductDistribution
            A view for a component distribution, or a sub-joint for an
            intermediate dict node.

        Raises
        ------
        KeyError
            If the key path does not resolve to a valid node in the
            component pytree.

        Notes
        -----
        The sub-joint returned for an intermediate dict node is an
        **independent copy** — it does not preserve correlations with
        sibling sub-trees that may exist in subclasses like
        :class:`SequentialJointDistribution`.  For
        :class:`ProductDistribution` (where all components are
        independent), the marginal sub-joint is exact.
        """
        key_path = _normalize_key(key)
        node = _walk_pytree(self._components, key_path)
        if isinstance(node, ArrayDistribution):
            return DistributionView(self, key)
        if isinstance(node, dict):
            # Internal node → return a sub-joint over the sub-tree.
            return ProductDistribution(**node)
        raise KeyError(
            f"Key path {key_path} resolves to {type(node).__name__}, "
            f"which is neither an ArrayDistribution leaf nor a dict node."
        )

    def bind(self, **mapping) -> dict[str, DistributionView]:
        """Create a dict of views with remapped names.

        Values may be strings (for flat access) or tuples of strings
        (for nested access)::

            joint.bind(a='x', b='y')                     # flat
            joint.bind(f=('physics', 'force'), m='obs')   # nested

        Parameters
        ----------
        **mapping
            ``{workflow_arg_name: component_key_or_path}``.
        """
        return {arg_name: self[comp_key] for arg_name, comp_key in mapping.items()}

    # -- Component-level log_prob ------------------------------------------

    def component_log_prob(self, value) -> dict:
        """Per-leaf log-density contributions.

        Returns a pytree with the same structure as the components,
        where each leaf is the scalar (or batch-shaped) log-density
        from the corresponding component evaluated at the matching
        value.

        The base implementation evaluates each leaf component's
        ``log_prob`` independently.  Subclasses with cross-component
        dependencies should override this method.

        Parameters
        ----------
        value : pytree
            A pytree with the same structure as the components, where
            each leaf is an array.
        """
        return jax.tree.map(
            lambda dist, val: dist._log_prob(jnp.asarray(val)),
            self._components,
            value,
        )

    # -- log_prob (abstract) -----------------------------------------------

    def _log_prob(self, value) -> Array:
        """Log-density of the joint distribution.

        Parameters
        ----------
        value : pytree
            A pytree with the same structure as the components, where
            each leaf is an array of shape
            ``(*batch_dims, *component_event_shape)``.

        Returns
        -------
        Array
            Scalar (or batch-shaped array) of log-densities.
        """
        raise NotImplementedError(
            f"{type(self).__name__}._log_prob() must be implemented by subclasses."
        )

    def __repr__(self) -> str:
        if self._is_flat:
            comp_str = ", ".join(
                f"{k}={type(v).__name__}" for k, v in self._components.items()
            )
        else:
            # For nested, show the tree structure compactly
            def _leaf_repr(d):
                return type(d).__name__
            comp_str = str(jax.tree.map(_leaf_repr, self._components))
        name_str = f", name='{self._name}'" if self._name else ""
        return f"{type(self).__name__}({comp_str}{name_str})"


# ---------------------------------------------------------------------------
# ProductDistribution — independent components
# ---------------------------------------------------------------------------

class ProductDistribution(JointDistribution, SupportsSampling, SupportsLogProb, SupportsMean, SupportsVariance, SupportsConditioning):
    """Joint distribution with **independent** leaf components.

    All leaf components are sampled independently.  The joint
    ``log_prob`` is the sum of per-leaf log-probs (no coupling terms).

    **Sample type:** A pytree with the same structure as the components,
    where each ``ArrayDistribution`` leaf is replaced by a sample array.
    For flat dicts this is ``dict[str, Array]``; for nested dicts it is
    a nested dict of arrays.

    **Independence assumption:** This class assumes statistical
    independence across **all** component distributions.  Organizing
    components into nested dicts is purely structural — it does not
    introduce any dependence between components.

    For sequential/autoregressive dependence, use
    :class:`SequentialJointDistribution`.  For arbitrary dependence
    with a known joint density, subclass :class:`JointDistribution`
    directly.

    **Flat-vector interop:** Use ``as_flat_distribution()`` or
    ``flatten_value()`` to obtain flat representations compatible
    with algorithms expecting ``ArrayDistribution``.

    Parameters
    ----------
    name : str, optional
        Distribution name.
    **components : ArrayDistribution or dict
        Named independent component distributions.  Values may be
        ``ArrayDistribution`` instances (leaves) or nested dicts
        whose leaves are ``ArrayDistribution`` instances.

    Examples
    --------
    Flat dict (most common)::

        >>> joint = ProductDistribution(
        ...     x=Normal(loc=0.0, scale=1.0),
        ...     y=Normal(loc=3.0, scale=2.0),
        ... )
        >>> s = joint._sample(jax.random.PRNGKey(0))
        >>> s.keys()
        dict_keys(['x', 'y'])

    Nested dict (grouped parameters)::

        >>> joint = ProductDistribution(
        ...     physics={"force": Normal(0, 1), "mass": Gamma(2, 1)},
        ...     observation=Normal(0, 0.1),
        ... )
        >>> s = joint._sample(jax.random.PRNGKey(0))
        >>> s["physics"]["force"].shape  # scalar leaf
        ()
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def _sample_one(self, key: PRNGKey):
        leaves = jax.tree.leaves(self._components)
        keys = jax.random.split(key, len(leaves))
        sampled_leaves = [
            dist._sample(subkey) for subkey, dist in zip(keys, leaves)
        ]
        return jax.tree.unflatten(self.treedef, sampled_leaves)

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ):
        """Draw independent samples from each leaf component.

        Parameters
        ----------
        key : PRNGKey
            JAX PRNG key.
        sample_shape : tuple[int, ...], optional
            Leading dimensions for the samples.

        Returns
        -------
        pytree
            A pytree with the same structure as the components.
            Each leaf is an array of shape
            ``(*sample_shape, *batch_shape, *leaf_event_shape)``.
        """
        leaves = jax.tree.leaves(self._components)
        keys = jax.random.split(key, len(leaves))
        sampled_leaves = [
            dist._sample(subkey, sample_shape) for subkey, dist in zip(keys, leaves)
        ]
        return jax.tree.unflatten(self.treedef, sampled_leaves)

    def _log_prob(self, value) -> Array:
        """Sum of independent leaf log-probs.

        Parameters
        ----------
        value : pytree
            A pytree with the same structure as the components,
            where each leaf is an array.

        Returns
        -------
        Array
            Scalar (or batch-shaped) log-density: sum of all leaf
            log-probs.
        """
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

    def _mean(self):
        """Per-leaf means (exact for independent components).

        Returns a pytree with the same structure as the components.
        """
        return jax.tree.map(lambda d: d._mean(), self._components)

    def _variance(self):
        """Per-leaf variances (exact for independent components).

        Returns a pytree with the same structure as the components.
        """
        return jax.tree.map(lambda d: d._variance(), self._components)

    def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
        return _mc_expectation(self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist)

    # -- Conditioning (full nested support) --------------------------------

    def _condition_on(self, observed=None, /, **kwargs):
        observed_leaves = _parse_condition_args(self, observed, kwargs)
        return self._condition_on_impl(observed_leaves)

    def _condition_on_impl(
        self, observed_leaves: dict[KeyPath, ArrayLike],
    ) -> "ProductDistribution":
        """Remove conditioned component distributions and return a new
        ProductDistribution.

        Since all component distributions are independent, conditioning
        simply removes the specified components from the pytree.  If
        removing components causes an intermediate dict to become empty,
        that dict is pruned from the tree.

        Parameters
        ----------
        observed_leaves : dict[KeyPath, ArrayLike]
            Validated mapping from leaf key paths to observed values.
        """
        new_components = _prune_leaves(self._components, set(observed_leaves.keys()))
        result = ProductDistribution(**new_components, name=self._name)
        # Record provenance
        conditioned_names = [
            " > ".join(path) for path in observed_leaves
        ]
        result.with_source(Provenance(
            "condition_on", parents=(self,),
            metadata={"conditioned": conditioned_names},
        ))
        return result


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
            # Skip this leaf — it's being conditioned on
            continue
        if isinstance(value, dict):
            # Recurse into sub-dict
            pruned = _prune_leaves(value, remove_paths, path)
            if pruned:  # Only keep non-empty sub-dicts
                result[key] = pruned
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# SequentialJointDistribution — autoregressive dependence
# ---------------------------------------------------------------------------

class SequentialJointDistribution(JointDistribution, SupportsSampling, SupportsLogProb, SupportsMean, SupportsVariance, SupportsConditioning):
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

    def _sample_one(self, key: PRNGKey) -> dict[str, Array]:
        full = self._sample_sequential(key, ())
        return {k: v for k, v in full.items() if k not in self._conditioned_names}

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> dict[str, Array]:
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

    def _mean(self) -> dict[str, Array]:
        """Per-component means (approximate — uses prototype components).

        For sequential joints, the true marginal mean is not simply the
        per-component mean because later components depend on earlier
        samples.  This returns the prototype (prior-evaluated) means
        as an approximation.
        """
        return {k: v._mean() for k, v in self._proto_components.items()}

    def _variance(self) -> dict[str, Array]:
        """Per-component variances (approximate — uses prototype components)."""
        return {k: v._variance() for k, v in self._proto_components.items()}

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

class JointEmpirical(JointDistribution, SupportsSampling, SupportsMean, SupportsVariance, SupportsConditioning):
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

    _sampling_cost = "low"
    _preferred_orchestration = None

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

    def _sample_one(self, key: PRNGKey) -> dict[str, Array]:
        return self._sample_joint_rows(key, ())

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> dict[str, Array]:
        return self._sample_joint_rows(key, sample_shape)

    def _sample_joint_rows(
        self, key: PRNGKey, sample_shape: tuple[int, ...]
    ) -> dict[str, Array]:
        """Resample rows jointly, preserving correlation."""
        n_draws = prod(sample_shape)
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

    def _log_prob(self, value: dict[str, ArrayLike]) -> Array:
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
            flat_dim = prod(arr.shape[1:])
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

    def _mean(self) -> dict[str, Array]:
        """Per-component weighted means."""
        result = {}
        for cname, arr in self._joint_samples.items():
            if self._is_uniform:
                result[cname] = jnp.mean(arr, axis=0)
            else:
                result[cname] = jnp.einsum("n,n...->...", self.weights, arr)
        return result

    def _variance(self) -> dict[str, Array]:
        """Per-component weighted variances."""
        means = self._mean()
        result = {}
        for cname, arr in self._joint_samples.items():
            diff = arr - means[cname]
            if self._is_uniform:
                result[cname] = jnp.mean(diff**2, axis=0)
            else:
                result[cname] = jnp.einsum("n,n...->...", self.weights, diff**2)
        return result

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

        if self._is_uniform:
            result = JointEmpirical(**remaining_samples, name=self._name)
        else:
            result = JointEmpirical(
                **remaining_samples,
                log_weights=self._log_weights,
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

class JointGaussian(JointDistribution, SupportsSampling, SupportsLogProb, SupportsMean, SupportsVariance, SupportsCovariance, SupportsConditioning):
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
        self._total_dim = total_dim  # still needed for Gaussian conditioning

    @property
    def mean_vector(self) -> Array:
        """Full mean vector."""
        return self._mean_vec

    @property
    def covariance(self) -> Array:
        """Full covariance matrix."""
        return self._cov_mat

    def _sample_one(self, key: PRNGKey) -> dict[str, Array]:
        from .multivariate import MultivariateNormal as MVN
        full_mvn = MVN(loc=self._mean_vec, cov=self._cov_mat)
        flat = full_mvn._sample(key)
        return self._unflatten_flat_vec(flat)

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> dict[str, Array]:
        from .multivariate import MultivariateNormal as MVN
        full_mvn = MVN(loc=self._mean_vec, cov=self._cov_mat)
        flat = full_mvn._sample(key, sample_shape)
        return self._unflatten_flat_vec(flat)

    def _unflatten_flat_vec(self, flat: Array) -> dict[str, Array]:
        """Split a flat Gaussian sample vector into per-component arrays."""
        result = {}
        for cname in self._component_shapes:
            sl = self._component_slices[cname]
            result[cname] = flat[..., sl]
        return result

    def _log_prob(self, value: dict[str, ArrayLike]) -> Array:
        from .multivariate import MultivariateNormal as MVN
        full_mvn = MVN(loc=self._mean_vec, cov=self._cov_mat)
        flat = self.flatten_value(value)
        return full_mvn._log_prob(flat)

    def _mean(self) -> dict[str, Array]:
        result = {}
        for cname in self._component_shapes:
            sl = self._component_slices[cname]
            result[cname] = self._mean_vec[sl]
        return result

    def _variance(self) -> dict[str, Array]:
        diag = jnp.diag(self._cov_mat)
        result = {}
        for cname in self._component_shapes:
            sl = self._component_slices[cname]
            result[cname] = diag[sl]
        return result

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
