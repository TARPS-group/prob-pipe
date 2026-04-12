"""Joint distribution base class and supporting types.

Provides:
  - ``JointDistribution``    – Base class for named multi-component distributions.
  - ``ProductDistribution``  – Independent-component joint distribution.
  - ``DistributionView``     – Lightweight reference to a joint component.
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
    PyTreeArrayDistribution,
    _mc_expectation,
)
from .constraints import Constraint, real
from .provenance import Provenance
from .values import Values
from ._values_distribution import ValuesDistribution
from .protocols import (
    SupportsConditioning,
    SupportsLogProb,
    SupportsMean,
    SupportsNamedComponents,
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
    def _default_support(cls) -> Constraint:
        return real

    def __repr__(self) -> str:
        path_str = " > ".join(self._key_path)
        return (
            f"DistributionView(parent={type(self._parent).__name__}, "
            f"path='{path_str}')"
        )


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
    ``joint.select("x", "y")`` creates a dict of views for
    workflow function broadcasting.

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
        self._values_template = _build_values_template(self._components)

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
            return ProductDistribution(**node)
        raise KeyError(
            f"Key path {key_path} resolves to {type(node).__name__}, "
            f"which is neither an ArrayDistribution leaf nor a dict node."
        )

    def select(self, *fields: str, **mapping: str) -> dict[str, DistributionView]:
        """Select named components as views for workflow function broadcasting.

        Positional args use the component name as the argument name.
        Keyword args remap: ``select(x="component_name")``.
        For nested access, use keyword args with tuple values::

            joint.select("x", "y")                           # flat
            joint.select(f=("physics", "force"), m="obs")    # nested / remapped

        Usage::

            predict(**joint.select("x", "y"), grid=x_grid)
        """
        result: dict[str, DistributionView] = {}
        for f in fields:
            result[f] = self[f]
        for arg_name, comp_key in mapping.items():
            result[arg_name] = self[comp_key]
        return result

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

class ProductDistribution(
    ValuesDistribution,
    SupportsSampling, SupportsLogProb, SupportsMean, SupportsVariance, SupportsConditioning,
):
    """Joint distribution with **independent** leaf components.

    Inherits from :class:`ValuesDistribution`.  All leaf components are
    sampled independently; the joint ``log_prob`` is the sum of per-leaf
    log-probs.  ``_sample()`` returns :class:`Values`.

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
