"""Joint distribution base class and supporting types.

Provides:
  - ``JointDistribution``    – Generic base class.
  - ``JointArrayDistribution``– JAX-backed base.
  - ``ProductDistribution``  – Generic independent-component joint.
  - ``ProductArrayDistribution`` – JAX-backed independent-component joint.                         
  - ``DistributionView``     – Lightweight reference to a joint component.
  - Key-path helpers for navigating nested component pytrees.
  - Conditioning argument parser.

"""

from __future__ import annotations

from abc import abstractmethod
from types import MappingProxyType
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from ..custom_types import Array, ArrayLike, PRNGKey
from .._utils import _auto_key
from ._array_distributions import (
    ArrayDistribution,
    PyTreeArrayDistribution,
    _mc_expectation,
    _vmap_sample,
)
from ._distribution_base import Distribution
from .constraints import Constraint, real
from .provenance import Provenance
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

KeyPath = tuple[str, ...]


def _normalize_key(key) -> KeyPath:
    """Normalize a component key to a KeyPath tuple."""
    if isinstance(key, str):
        return (key,)
    if isinstance(key, tuple) and all(isinstance(k, str) for k in key):
        return key
    raise TypeError(
        f"Component key must be a string or tuple of strings, got {key!r}"
    )


def _walk_pytree(tree, key_path: KeyPath):
    """Navigate a nested dict/pytree by following a key path."""
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
    """Extract leaf key paths from a pytree of ArrayDistributions."""
    is_flat = isinstance(components, dict) and all(
        isinstance(v, ArrayDistribution) for v in components.values()
    )
    if is_flat:
        return tuple(components.keys())

    paths_and_leaves = jax.tree_util.tree_leaves_with_path(components)
    key_paths = []
    for path, _leaf in paths_and_leaves:
        str_path = tuple(
            k.key if hasattr(k, "key") else str(k) for k in path
        )
        key_paths.append(str_path)
    return tuple(key_paths)


# ---------------------------------------------------------------------------
# Conditioning argument parser  
# ---------------------------------------------------------------------------

def _collect_observed_leaves(
    obs_tree: dict,
    comp_tree: dict,
    prefix: KeyPath,
    out: dict[KeyPath, ArrayLike],
) -> None:
    """Recursively walk an observed-value tree, validating against the component tree."""
    for key, obs_val in obs_tree.items():
        path = prefix + (key,)
        path_str = " > ".join(path)

        if not isinstance(comp_tree, dict) or key not in comp_tree:
            available = tuple(comp_tree.keys()) if isinstance(comp_tree, dict) else ()
            raise KeyError(
                f"Component key {key!r} not found at level "
                f"'{' > '.join(prefix)}'. "
                f"Available keys: {available}"
            )

        comp_node = comp_tree[key]

        if isinstance(obs_val, dict):
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
            _collect_observed_leaves(obs_val, comp_node, path, out)
        else:
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
    joint: "JointDistribution",
    observed: dict | None,
    kwargs: dict,
) -> dict[KeyPath, ArrayLike]:
    """Parse and validate conditioning arguments for a joint distribution."""
    if observed is not None and kwargs:
        raise TypeError(
            "condition_on() accepts either a positional dict or keyword "
            "arguments, not both."
        )
    tree = observed if observed is not None else kwargs
    if not tree:
        raise ValueError("condition_on() requires at least one observed value.")

    observed_leaves: dict[KeyPath, ArrayLike] = {}
    _collect_observed_leaves(tree, joint._components, prefix=(), out=observed_leaves)

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
# JointDistribution — Generic base
# ---------------------------------------------------------------------------

class JointDistribution(Distribution, SupportsNamedComponents):
    """Generic base class for named multi-component distributions.

    This class is **backend-agnostic**: it defines the structural interface
    for joint distributions (named components, component access, conditioning
    argument parsing) without importing JAX or inheriting from
    ``PyTreeArrayDistribution``.

    **What this class provides:**

    *  ``_components`` — a dict (possibly nested) mapping names to component
       distribution objects.  Leaves do *not* have to be ``ArrayDistribution``
       instances; subclasses may hold any distribution type.
    *  ``component_names`` — ordered tuple of leaf key identifiers.
    *  ``__getitem__`` — component access by name or key-path.
    *  ``bind()`` — remapped dict of component views.

    **What this class does NOT provide:**

    *  Sampling, log-prob, mean, variance — these are protocol-guarded and
       must be declared by concrete subclasses.
    *  Shape semantics (``event_shape``, ``batch_shape``, ``treedef``,
       ``flatten_value``) — these live in ``JointArrayDistribution``, which
       additionally inherits ``PyTreeArrayDistribution``.
    *  JAX operations of any kind.

    **When to inherit directly from JointDistribution:**

    Subclass this class when your joint distribution does not use JAX arrays
    internally (e.g., a joint over scipy distributions, or a pure-Python
    joint that holds heterogeneous backends).

    **When to inherit from JointArrayDistribution instead:**

    For all JAX-backed joint distributions — which is every concrete type
    currently in ``distributions/joint.py`` — inherit from
    ``JointArrayDistribution``.

    Parameters
    ----------
    name : str, optional
        Distribution name.
    **components
        Named component distributions (leaves or nested dicts).
        At least one component is required.
    """

    def __init__(self, *, name: str | None = None, **components: Any) -> None:
        if not components:
            raise ValueError(
                f"{type(self).__name__} requires at least one component."
            )
        self._components: dict[str, Any] = dict(components)
        self._name = name

    # -- SupportsNamedComponents -------------------------------------------

    @property
    def component_names(self) -> tuple:
        """Leaf component identifiers (strings for flat, KeyPath tuples for nested)."""
        return _component_key_paths(self._components)

    def __getitem__(self, key: Any) -> Any:
        """Access a component distribution by name or key path.

        Returns the component distribution directly (not a ``DistributionView``,
        since that requires ``ArrayDistribution`` leaves).  Subclasses that hold
        ``ArrayDistribution`` leaves should override this to return
        ``DistributionView`` instances as ``JointArrayDistribution`` does.
        """
        key_path = _normalize_key(key)
        return _walk_pytree(self._components, key_path)

    def bind(self, **mapping: Any) -> dict[str, Any]:
        """Create a dict of component references with remapped names."""
        return {arg_name: self[comp_key] for arg_name, comp_key in mapping.items()}

    # -- Internal helpers used by subclasses and _parse_condition_args ------

    @property
    def _is_flat(self) -> bool:
        """True if all top-level values are leaf (non-dict) distributions."""
        return isinstance(self._components, dict) and all(
            not isinstance(v, dict) for v in self._components.values()
        )

    @property
    def components(self) -> Mapping[str, Any]:
        """Read-only view of the top-level component mapping."""
        if self._is_flat:
            return MappingProxyType(self._components)
        return self._components

    def __repr__(self) -> str:
        comp_str = ", ".join(
            f"{k}={type(v).__name__}" for k, v in self._components.items()
        )
        name_str = f", name='{self._name}'" if self._name else ""
        return f"{type(self).__name__}({comp_str}{name_str})"


# ---------------------------------------------------------------------------
# JointArrayDistribution — JAX-backed base
# ---------------------------------------------------------------------------

class JointArrayDistribution(JointDistribution, PyTreeArrayDistribution, SupportsNamedComponents):
    """JAX-backed base class for named multi-component distributions.

    A ``JointArrayDistribution`` is a :class:`PyTreeArrayDistribution`
    whose samples are **pytrees** (typically dicts, possibly nested)
    mapping component names to JAX arrays.  Each leaf of the component
    pytree is an ``ArrayDistribution``.

    **Component structure:**

    Components are stored in ``self._components`` as a pytree whose
    leaves are ``ArrayDistribution`` instances.  For flat dicts this is
    ``{"x": Normal, "y": MVN}``.  For nested dicts this is e.g.
    ``{"physics": {"force": Normal, "mass": Gamma}, "obs": Normal}``.

    The pytree structure determines:

    *  The ``treedef`` — derived from the component pytree by replacing
       each ``ArrayDistribution`` leaf with a scalar placeholder.
    *  ``event_shapes`` — a pytree of the same structure where each
       leaf is the component's ``event_shape`` tuple.
    *  ``component_names`` — a tuple of key paths.

    **Flat-vector interop:**

    The inherited ``flatten_value`` / ``unflatten_value`` methods convert
    between the pytree representation and flat
    ``(*leading, event_size)`` arrays.  ``as_flat_distribution()``
    returns a :class:`FlattenedView` with ``event_shape = (event_size,)``
    for algorithms expecting flat vectors.

    **Component access:**

    ``joint["name"]`` returns a :class:`DistributionView` — an
    ``ArrayDistribution`` that extracts the named component from
    joint samples.

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

    def __init__(self, *, name: str | None = None, **components: Any) -> None:
        if not components:
            raise ValueError(
                f"{type(self).__name__} requires at least one component."
            )
        # Validate: all leaves must be ArrayDistribution
        leaves = jax.tree.leaves(components)
        if not leaves:
            raise ValueError(
                f"{type(self).__name__} requires at least one component."
            )
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
        """Pytree structure of a single sample."""
        prototype = jax.tree.map(lambda _: 0, self._components)
        return jax.tree.structure(prototype)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape, shared across all components."""
        return ()

    @property
    def event_shapes(self):
        """Per-component event shapes as a pytree."""
        return jax.tree.map(lambda d: d.event_shape, self._components)

    # -- Properties --------------------------------------------------------

    @property
    def _is_flat(self) -> bool:
        """True if all top-level dict values are ArrayDistribution leaves."""
        return isinstance(self._components, dict) and all(
            isinstance(v, ArrayDistribution) for v in self._components.values()
        )

    @property
    def components(self):
        """The component pytree (read-only for flat dicts)."""
        if self._is_flat:
            return MappingProxyType(self._components)
        return self._components

    @property
    def component_names(self) -> tuple:
        """Leaf component identifiers."""
        return _component_key_paths(self._components)

    # -- Structured access -------------------------------------------------

    def __getitem__(self, key):
        """Access a component by key path.

        Returns a :class:`DistributionView` for leaf components, or a new
        :class:`ProductArrayDistribution` for intermediate dict nodes.
        """
        key_path = _normalize_key(key)
        node = _walk_pytree(self._components, key_path)
        if isinstance(node, ArrayDistribution):
            return DistributionView(self, key)
        if isinstance(node, dict):
            return ProductArrayDistribution(**node)
        raise KeyError(
            f"Key path {key_path} resolves to {type(node).__name__}, "
            f"which is neither an ArrayDistribution leaf nor a dict node."
        )

    def bind(self, **mapping) -> dict[str, "DistributionView"]:
        """Create a dict of views with remapped names."""
        return {arg_name: self[comp_key] for arg_name, comp_key in mapping.items()}

    # -- Component-level log_prob ------------------------------------------

    def component_log_prob(self, value) -> dict:
        """Per-leaf log-density contributions."""
        return jax.tree.map(
            lambda dist, val: dist._log_prob(jnp.asarray(val)),
            self._components,
            value,
        )

    # -- log_prob (abstract) -----------------------------------------------

    def _log_prob(self, value) -> Array:
        """Log-density of the joint distribution (must be overridden)."""
        raise NotImplementedError(
            f"{type(self).__name__}._log_prob() must be implemented by subclasses."
        )

    def __repr__(self) -> str:
        if self._is_flat:
            comp_str = ", ".join(
                f"{k}={type(v).__name__}" for k, v in self._components.items()
            )
        else:
            def _leaf_repr(d):
                return type(d).__name__
            comp_str = str(jax.tree.map(_leaf_repr, self._components))
        name_str = f", name='{self._name}'" if self._name else ""
        return f"{type(self).__name__}({comp_str}{name_str})"


# ---------------------------------------------------------------------------
# DistributionView  
# ---------------------------------------------------------------------------

class DistributionView(ArrayDistribution, SupportsSampling, SupportsLogProb, SupportsMean, SupportsVariance):
    """A lightweight reference to a leaf component of a :class:`JointArrayDistribution`.

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

    Parameters
    ----------
    parent : JointArrayDistribution
        The joint distribution this view belongs to.
    key_path : str or KeyPath
        A string (for flat access) or tuple of strings (for nested
        access) identifying the leaf component.
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __init__(self, parent: JointArrayDistribution, key_path: str | KeyPath):
        self._key_path: KeyPath = _normalize_key(key_path)
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
        """Leaf name (last element of key path)."""
        return self._key_path[-1] if len(self._key_path) == 1 else None

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
# ProductDistribution — NEW generic base (no JAX)
# ---------------------------------------------------------------------------

class ProductDistribution(JointDistribution):
    """Generic base for a joint distribution of **independent** components.

    This class is **backend-agnostic**: it declares the independence
    semantics (joint log-prob = sum of marginal log-probs) but does not
    import JAX or impose array types on the component distributions.

    The JAX-backed version — :class:`ProductArrayDistribution` — inherits
    from this class and additionally from :class:`JointArrayDistribution`.
    All current concrete usage in ProbPipe should use
    ``ProductArrayDistribution``.

    Subclass this class when your components are backed by a non-JAX
    backend (e.g., scipy).  Override ``_log_prob`` and ``_sample``
    accordingly.

    Parameters
    ----------
    name : str, optional
        Distribution name.
    **components
        Named independent component distributions.
    """
    # This class is intentionally sparse — it is a semantic marker for
    # "independent components" and a superclass for isinstance checks.
    # The full implementation lives in ProductArrayDistribution.
    pass


# ---------------------------------------------------------------------------
# ProductArrayDistribution — JAX-backed independent joint 
# ---------------------------------------------------------------------------

class ProductArrayDistribution(ProductDistribution, JointArrayDistribution, SupportsSampling, SupportsLogProb, SupportsMean, SupportsVariance, SupportsConditioning):
    """Joint distribution with **independent** leaf components, backed by JAX.

    All leaf components are sampled independently.  The joint
    ``log_prob`` is the sum of per-leaf log-probs (no coupling terms).

    **Sample type:** A pytree with the same structure as the components,
    where each ``ArrayDistribution`` leaf is replaced by a sample array.
    For flat dicts this is ``dict[str, Array]``; for nested dicts it is
    a nested dict of arrays.

    **Independence assumption:** This class assumes statistical
    independence across **all** component distributions.  Organizing
    components into nested dicts is purely structural.

    **Flat-vector interop:** Use ``as_flat_distribution()`` or
    ``flatten_value()`` to obtain flat representations compatible
    with algorithms expecting ``ArrayDistribution``.

    Parameters
    ----------
    name : str, optional
        Distribution name.
    **components : ArrayDistribution or dict
        Named independent component distributions.

    Examples
    --------
    Flat dict (most common)::

        >>> joint = ProductArrayDistribution(
        ...     x=Normal(loc=0.0, scale=1.0),
        ...     y=Normal(loc=3.0, scale=2.0),
        ... )

    Nested dict::

        >>> joint = ProductArrayDistribution(
        ...     physics={"force": Normal(0, 1), "mass": Gamma(2, 1)},
        ...     observation=Normal(0, 0.1),
        ... )
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
        leaves = jax.tree.leaves(self._components)
        keys = jax.random.split(key, len(leaves))
        sampled_leaves = [
            dist._sample(subkey, sample_shape) for subkey, dist in zip(keys, leaves)
        ]
        return jax.tree.unflatten(self.treedef, sampled_leaves)

    def _log_prob(self, value) -> Array:
        """Sum of independent leaf log-probs."""
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
        """Per-leaf means (exact for independent components)."""
        return jax.tree.map(lambda d: d._mean(), self._components)

    def _variance(self):
        """Per-leaf variances (exact for independent components)."""
        return jax.tree.map(lambda d: d._variance(), self._components)

    def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
        return _mc_expectation(self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist)

    # -- Conditioning (full nested support) --------------------------------

    def _condition_on(self, observed=None, /, **kwargs):
        observed_leaves = _parse_condition_args(self, observed, kwargs)
        return self._condition_on_impl(observed_leaves)

    def _condition_on_impl(
        self, observed_leaves: dict[KeyPath, ArrayLike],
    ) -> "ProductArrayDistribution":
        """Remove conditioned component distributions and return a new
        ProductArrayDistribution."""
        new_components = _prune_leaves(self._components, set(observed_leaves.keys()))
        result = ProductArrayDistribution(**new_components, name=self._name)
        conditioned_names = [
            " > ".join(path) for path in observed_leaves
        ]
        result.with_source(Provenance(
            "condition_on", parents=(self,),
            metadata={"conditioned": conditioned_names},
        ))
        return result


def _prune_leaves(tree: dict, remove_paths: set[KeyPath], prefix: tuple = ()) -> dict:
    """Remove specified leaves from a nested dict and prune empty sub-dicts."""
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
