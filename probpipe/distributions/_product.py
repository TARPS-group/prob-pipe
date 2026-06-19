"""ProductDistribution --- independent-component joint distribution.

Provides:
  - ``ProductDistribution``  -- Independent-component joint distribution
    (inherits from :class:`NumericRecordDistribution`).
  - ``TFPProductDistribution`` -- Subclass that exposes a combined TFP
    distribution (``_tfp_dist``) when all leaf components are TFP-backed.
  - Dynamic protocol factory for automatic protocol support.
  - Helpers for nested component sampling and mapping.
  - JAX pytree registration.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp

from ..custom_types import Array, ArrayLike, PRNGKey
from ..core._numeric_record_distribution import (
    NumericRecordDistribution,
    _mc_expectation,
)
from ..core.provenance import Provenance
from ..core.record import Record
from ..core._distribution_base import Distribution
from ..core._record_distribution import (
    RecordDistribution,
    _register_dynamic_subclass,
    _build_record_template,
)
from ..core.protocols import (
    protocols_supported_by_all,
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

_PRODUCT_CLASS_CACHE: dict[tuple[frozenset[type], bool, bool], type] = {}


def _product_class_for_components(components: dict) -> type:
    """Return a ProductDistribution subclass whose bases match what ALL
    leaf components support.

    The base class is always :class:`ProductDistribution` (rooted at
    :class:`RecordDistribution`); the numeric API
    (:class:`NumericRecordDistribution`) and per-protocol mixins are
    added dynamically:

    - ``NumericRecordDistribution`` is added when every leaf is itself
      a :class:`NumericRecordDistribution`. The joint's content is
      then numeric end-to-end and the numeric-only methods
      (``event_size``, ``flatten_value`` / ``unflatten_value``,
      ``as_flat_distribution``) become available. With mixed or
      non-numeric leaves the joint stays on the generic
      :class:`RecordDistribution` surface — sampling, conditioning,
      and named-component access still work; the numeric methods are
      simply absent.
    - ``SupportsSampling`` and ``SupportsConditioning`` are always
      included.
    - ``SupportsLogProb``, ``SupportsMean``, ``SupportsVariance`` are
      added only when every leaf supports them.
    - When every leaf is TFP-backed, ``TFPProductDistribution``
      replaces the generic base to expose a combined ``_tfp_dist``.
    """
    leaves = jax.tree.leaves(components)

    extra_bases = protocols_supported_by_all(
        leaves, (SupportsLogProb, SupportsMean, SupportsVariance),
    )
    all_tfp = all(hasattr(l, "_tfp_dist") for l in leaves)
    all_numeric = all(isinstance(l, NumericRecordDistribution) for l in leaves)

    key = (frozenset(extra_bases), all_tfp, all_numeric)
    if key in _PRODUCT_CLASS_CACHE:
        return _PRODUCT_CLASS_CACHE[key]

    base = TFPProductDistribution if all_tfp else ProductDistribution

    # Order matters: ``NumericRecordDistribution`` mixes in the numeric
    # API on top of the generic ``ProductDistribution`` base. Listing
    # it before the protocol mixins keeps the MRO consistent with
    # standalone NRDs (numeric API → protocols → object).
    numeric_mixin: tuple[type, ...] = (
        (NumericRecordDistribution,) if all_numeric else ()
    )
    bases = (base, *numeric_mixin, *extra_bases)

    if bases == (base,):
        _PRODUCT_CLASS_CACHE[key] = base
        return base

    cls = type("ProductDistribution", bases, {})
    _register_dynamic_subclass(cls)
    _PRODUCT_CLASS_CACHE[key] = cls
    return cls


def _resolve_nested_names(parent_key: str, d: dict) -> dict:
    """Recursively auto-rename nested leaf distributions to match their dict keys."""
    result = {}
    for key, val in d.items():
        if isinstance(val, dict):
            result[key] = _resolve_nested_names(key, val)
        elif hasattr(val, "name") and val.name != key:
            result[key] = val.renamed(key)
        else:
            result[key] = val
    return result


def _merge_positional_and_keyword(
    positional: tuple, keyword: dict,
) -> dict:
    """Merge positional distributions (keyed by .name) with keyword components."""
    components = {}
    for dist in positional:
        if not hasattr(dist, "name") or not dist.name:
            raise ValueError(
                "Positional arguments to ProductDistribution must be "
                "named distributions (have a non-empty .name attribute)"
            )
        key = dist.name
        if key in components or key in keyword:
            raise ValueError(
                f"Duplicate component name {key!r}: appears in both "
                "positional and keyword arguments"
            )
        components[key] = dist
    components.update(keyword)
    return components


# ---------------------------------------------------------------------------
# ProductDistribution
# ---------------------------------------------------------------------------


class ProductDistribution(
    RecordDistribution,
    SupportsSampling, SupportsConditioning,
):
    """Joint distribution with **independent** leaf components.

    Inherits from :class:`RecordDistribution` (the general
    named-fields base); leaves can be any :class:`Distribution`.
    The product is well-defined for numeric and non-numeric leaves
    alike — sampling produces a :class:`Record` keyed by component
    name, conditioning works on any named subset, and named-component
    access (``dist[field]``, ``dist.fields``, ``dist.event_shapes``)
    is always available.

    **When every leaf is a :class:`NumericRecordDistribution`** the
    dynamic class factory mixes in :class:`NumericRecordDistribution`
    too, so the joint also exposes the numeric API (``event_size``,
    ``flatten_value`` / ``unflatten_value``, ``as_flat_distribution``,
    ``dtypes``, ``supports``). For mixed or non-numeric leaves those
    methods are simply absent on the instance — the joint stays on
    the generic :class:`RecordDistribution` surface. See
    :func:`_product_class_for_components` for the dispatch.

    All leaf components are sampled independently. ``_sample()``
    returns :class:`NumericRecord` when all leaves are numeric, and
    a plain :class:`Record` otherwise.

    **Dynamic protocol support:** ``SupportsLogProb``, ``SupportsMean``,
    and ``SupportsVariance`` are included only when ALL leaf components
    support them.  ``isinstance(product, SupportsLogProb)`` is ``True``
    only when every component has ``_log_prob``.

    Parameters
    ----------
    *positional : NumericRecordDistribution
        Named distributions.  Each distribution's ``.name`` is used as
        the component key.
    name : str, optional
        Distribution name for the joint.
    **components : NumericRecordDistribution or dict
        Named independent component distributions.  Values may be
        ``NumericRecordDistribution`` instances (leaves) or nested dicts
        whose leaves are ``NumericRecordDistribution`` instances.
        When a keyword key differs from the distribution's name, the
        distribution is automatically renamed (via ``renamed()``) to
        match the key.

    Examples
    --------
    ::

        # Positional — uses each distribution's name as the key:
        ProductDistribution(Normal(0, 1, name="x"), Gamma(2, 1, name="y"))

        # Keyword — auto-renames if the key differs:
        ProductDistribution(growth_rate=Normal(0, 1, name="x"))

        # Mixed:
        ProductDistribution(Normal(0, 1, name="x"), scale=Gamma(2, 1, name="y"))
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __new__(cls, *positional, name: str | None = None, **components):
        components = _merge_positional_and_keyword(positional, components)
        if not components:
            return object.__new__(cls)
        actual_cls = _product_class_for_components(components)
        return object.__new__(actual_cls)

    def __init__(self, *positional, name: str | None = None, **components):
        components = _merge_positional_and_keyword(positional, components)
        if not components:
            raise ValueError("ProductDistribution requires at least one component.")
        resolved: dict[str, Any] = {}
        for key, comp in components.items():
            if isinstance(comp, dict):
                resolved[key] = _resolve_nested_names(key, comp)
            elif comp.name != key:
                resolved[key] = comp.renamed(key)
            else:
                resolved[key] = comp
        # Leaves can be any ``Distribution``. When every leaf is a
        # ``NumericRecordDistribution`` the dynamic class factory
        # additionally mixes in the numeric API; otherwise the joint
        # stays on the generic ``RecordDistribution`` surface.
        for leaf in jax.tree.leaves(resolved):
            if not isinstance(leaf, Distribution):
                raise TypeError(
                    f"All leaf components must be Distribution instances, "
                    f"got {type(leaf).__name__}"
                )
        self._components = resolved
        if name is None:
            name = "product(" + ",".join(resolved.keys()) + ")"
        super().__init__(name=name)
        self._record_template = _build_record_template(self._components)

    def __reduce__(self):
        return (_unpickle_product_distribution, (dict(self._components), self._name))

    # -- Sampling (returns Record) ------------------------------------------

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()):
        """Draw independent samples from each component.

        Parameters
        ----------
        key : PRNGKey
            JAX PRNG key for sampling.
        sample_shape : tuple of int, optional
            Leading shape for independent draws. ``()`` draws a single sample.

        Returns
        -------
        Record or NumericRecordArray or RecordArray
            ``Record`` when ``sample_shape == ()``. With a non-empty
            ``sample_shape``: ``NumericRecordArray`` when every leaf is
            a :class:`NumericRecordDistribution` (the dynamic mixin
            case), otherwise a plain :class:`RecordArray`.
        """
        from ..core._record_array import NumericRecordArray, RecordArray
        names = list(self._components.keys())
        keys = jax.random.split(key, len(names))
        numeric = isinstance(self, NumericRecordDistribution)
        fields: dict[str, jnp.ndarray | Record] = {}
        for subkey, name in zip(keys, names):
            comp = self._components[name]
            if isinstance(comp, dict):
                # Pass the sub-template so a batched nested draw is a nested
                # record-array (canonical, flattenable), not a plain Record.
                sub_template = self.record_template[name] if sample_shape else None
                fields[name] = _sample_nested(
                    comp, subkey, sample_shape,
                    template=sub_template, numeric=numeric)
            else:
                fields[name] = comp._sample(subkey, sample_shape)
        if sample_shape:
            # NRD mixin → numeric batched container; otherwise the
            # plain RecordArray which doesn't require numeric leaves.
            if isinstance(self, NumericRecordDistribution):
                return NumericRecordArray(
                    fields, batch_shape=sample_shape,
                    template=self.record_template,
                )
            return RecordArray(
                fields, batch_shape=sample_shape,
                template=self.record_template,
            )
        return Record(fields)

    # -- Log-prob -----------------------------------------------------------

    def _log_prob(self, value: Any) -> Array:
        """Sum of independent leaf log-probs.

        Accepts Record, dict, or — when the joint is the all-numeric
        case (i.e., :class:`NumericRecordDistribution` is mixed in by
        the dynamic factory) — a flat array, which is auto-unflattened
        via the template. Flat-array input is rejected for the
        general (non-numeric) case because ``unflatten_value`` isn't
        available there.
        """
        from ..core._record_array import RecordArray
        if isinstance(value, jnp.ndarray):
            if not isinstance(self, NumericRecordDistribution):
                raise TypeError(
                    "Flat-array input to log_prob requires every leaf "
                    "to be a NumericRecordDistribution; this joint has "
                    "non-numeric leaves. Pass a Record / dict instead."
                )
            # Ensure the input has a trailing event axis ``(*batch,
            # event_size)`` so the static ``unflatten_value`` can
            # reshape it. Single-component RWMH / NUTS callers pass a
            # scalar or 1-D vector (``flat.shape == (event_size,)``);
            # ``unflatten_value`` needs that as the trailing axis.
            flat = jnp.asarray(value)
            if flat.ndim == 0:
                flat = flat[None]
            value = self.unflatten_value(flat, template=self.record_template)
            # Single-field templates return a raw array (preserving the
            # "single-leaf returns raw" contract on the static method);
            # the tree-map below expects a per-field structure, so
            # re-key it under the lone field name.
            if isinstance(value, jnp.ndarray):
                (field_name,) = self.record_template.fields
                value = {field_name: value}
        if isinstance(value, RecordArray):
            value = {k: v for k, v in value.items()}
        if isinstance(value, Record):
            value = value.to_dict()
        # Recursively convert nested Record values to dicts
        def _to_dict(v):
            if isinstance(v, Record):
                return v.to_dict()
            return v
        if isinstance(value, dict):
            value = {k: _to_dict(v) for k, v in value.items()}
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
        if all(isinstance(v, NumericRecordDistribution) for v in self._components.values()):
            return MappingProxyType(self._components)
        return self._components

    @property
    def supports(self):
        """Per-leaf support constraints -- each leaf component's ``support``.

        Nested components are keyed by slash-delimited paths
        (``"outer/a"``), matching ``RecordTemplate.leaf_shapes``, so every
        value is a ``Constraint``."""
        out: dict = {}

        def _walk(components: dict, prefix: str) -> None:
            for name, comp in components.items():
                if isinstance(comp, dict):
                    _walk(comp, f"{prefix}{name}/")
                else:
                    out[f"{prefix}{name}"] = comp.support

        _walk(self._components, "")
        return out

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
        # Nested-dict groups print as ``{...}``; concrete ``Distribution``
        # leaves print their class name regardless of whether they're
        # numeric (the previous ``isinstance(v, NumericRecordDistribution)``
        # check hid non-numeric ``RecordDistribution`` leaves like
        # ``JointEmpirical``).
        comp_str = ", ".join(
            f"{k}={{...}}" if isinstance(v, dict) else f"{k}={type(v).__name__}"
            for k, v in self._components.items()
        )
        name_str = f", name='{self._name}'" if self._name else ""
        return f"ProductDistribution({comp_str}{name_str})"


def _unpickle_product_distribution(components, name):
    """Reconstruct a ProductDistribution (or dynamic subclass) from its components."""
    return ProductDistribution(**components, name=name)


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

    def __init__(self, *positional, name: str | None = None, **components):
        super().__init__(*positional, name=name, **components)
        self._build_tfp_dist()

    def _build_tfp_dist(self):
        """Construct a combined TFP distribution from the components.

        Collects component TFP distributions in field-insertion order
        (matching the ``Record`` layout).  For the common case of
        same-family scalar distributions, stacks parameters into a
        single ``tfd.Independent``.  Falls back to ``tfd.Blockwise``
        for mixed or vector components.
        """
        from tensorflow_probability.substrates.jax import distributions as tfd

        tfp_dists = []
        for comp_name in self._components.keys():
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
    def dtypes(self) -> dict[str, "jnp.dtype"]:
        """Per-field dtype — the TFP Blockwise's dtype spread
        across the auto-built single-field template."""
        return self._per_field_dict(self._tfp_dist.dtype)


# -- Helpers for nested component pytrees ----------------------------------


def _sample_nested(components: dict, key, sample_shape, template=None, numeric=False):
    """Recursively sample from nested component dicts.

    For an **unbatched** draw (``sample_shape == ()``) returns a plain nested
    ``Record``. For a **batched** draw returns a nested record-array
    (``NumericRecordArray`` when ``numeric`` else ``RecordArray``) carrying the
    sub-``template`` and ``batch_shape``, so the result is canonical and
    flattenable rather than a plain ``Record`` with batch-shaped leaves.
    """
    names = list(components.keys())
    keys = jax.random.split(key, len(names))
    fields: dict = {}
    for subkey, name in zip(keys, names):
        comp = components[name]
        if isinstance(comp, dict):
            sub_template = template[name] if template is not None else None
            fields[name] = _sample_nested(
                comp, subkey, sample_shape, template=sub_template, numeric=numeric)
        else:
            fields[name] = comp._sample(subkey, sample_shape)
    if sample_shape and template is not None:
        from ..core._record_array import NumericRecordArray, RecordArray
        cls = NumericRecordArray if numeric else RecordArray
        return cls(fields, batch_shape=sample_shape, template=template)
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
    pytree structure (treedef) + name + top-level key order as
    auxiliary data. The explicit key order matters: JAX's default
    dict traversal sorts keys, but ``ProductDistribution`` preserves
    insertion order (per the Record-family convention), so we restore
    it manually on unflatten.
    """
    leaves = jax.tree.leaves(dist._components)
    comp_treedef = jax.tree.structure(dist._components)
    aux = (comp_treedef, dist._name, tuple(dist._components.keys()))
    return leaves, aux


def _product_unflatten(aux, children):
    """Unflatten a ProductDistribution from JAX pytree data.

    Reconstructs the component pytree from the stored treedef, then
    re-orders the top-level dict to match the original insertion
    order before passing to the constructor.
    """
    comp_treedef, name, key_order = aux
    components = jax.tree.unflatten(comp_treedef, children)
    # Restore insertion order at the top level (JAX returns dict keys
    # sorted; we re-key in the original order).
    ordered = {k: components[k] for k in key_order}
    return ProductDistribution(**ordered, name=name)


jax.tree_util.register_pytree_node(
    ProductDistribution,
    _product_flatten,
    _product_unflatten,
)
