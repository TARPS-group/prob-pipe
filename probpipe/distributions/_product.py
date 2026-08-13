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

from ..core._distribution_base import Distribution
from ..core._numeric_record_distribution import (
    NumericRecordDistribution,
    _mc_expectation,
)
from ..core._record_distribution import (
    RecordDistribution,
    _build_event_template,
    _register_dynamic_subclass,
)
from ..core.named_tree import _PATH_SEP
from ..core.protocols import (
    SupportsConditioning,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
    protocols_supported_by_all,
)
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import auto_name
from ..custom_types import Array, ArrayLike, PRNGKey
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
        leaves,
        (SupportsLogProb, SupportsMean, SupportsVariance),
    )
    all_tfp = all(hasattr(leaf, "_tfp_dist") for leaf in leaves)
    all_numeric = all(isinstance(leaf, NumericRecordDistribution) for leaf in leaves)

    key = (frozenset(extra_bases), all_tfp, all_numeric)
    if key in _PRODUCT_CLASS_CACHE:
        return _PRODUCT_CLASS_CACHE[key]

    base = TFPProductDistribution if all_tfp else ProductDistribution

    # Order matters: ``NumericRecordDistribution`` mixes in the numeric
    # API on top of the generic ``ProductDistribution`` base. Listing
    # it before the protocol mixins keeps the MRO consistent with
    # standalone NRDs (numeric API → protocols → object).
    numeric_mixin: tuple[type, ...] = (NumericRecordDistribution,) if all_numeric else ()
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
            result[key] = val.with_name(key)
        else:
            result[key] = val
    return result


def _merge_positional_and_keyword(
    positional: tuple,
    keyword: dict,
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
    SupportsSampling,
    SupportsConditioning,
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
        distribution is automatically renamed (via ``with_name()``) to
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
                resolved[key] = comp.with_name(key)
            else:
                resolved[key] = comp
        # Leaves can be any ``Distribution``. When every leaf is a
        # ``NumericRecordDistribution`` the dynamic class factory
        # additionally mixes in the numeric API; otherwise the joint
        # stays on the generic ``RecordDistribution`` surface.
        for leaf in jax.tree.leaves(resolved):
            if not isinstance(leaf, Distribution):
                raise TypeError(
                    f"All leaf components must be Distribution instances, got {type(leaf).__name__}"
                )
        self._components = resolved
        name, name_is_auto = auto_name(name, "product(" + ",".join(resolved.keys()) + ")")
        super().__init__(name=name, name_is_auto=name_is_auto)
        self._event_template = _build_event_template(self._components)

    def __reduce__(self):
        return (
            _unpickle_product_distribution,
            (
                dict(self._components),
                self._name,
                self._name_is_auto,
                self._provenance,
                self.annotations,
            ),
        )

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
        Record or NumericRecordBatch or RecordBatch
            ``Record`` when ``sample_shape == ()``. With a non-empty
            ``sample_shape``: a batch of draws over one ``draw`` level,
            :class:`NumericRecordBatch` when every leaf is a
            :class:`NumericRecordDistribution` (the dynamic mixin case),
            otherwise a plain :class:`RecordBatch`.
        """
        from ..core._numeric_record_batch import NumericRecordBatch
        from ..core._record_batch import RecordBatch

        if sample_shape:
            # A batch stores one column per *field*, so a nested product's draw
            # is one flat batch over leaf paths rather than a batch per subtree.
            # NRD mixin → the numeric batch; otherwise the plain one, which does
            # not require numeric leaves.
            cls = NumericRecordBatch if isinstance(self, NumericRecordDistribution) else RecordBatch
            return cls(
                _sample_columns(self._components, key, sample_shape),
                "draw",
                element_spec=self.event_template,
                axis_groups=(sample_shape,),
                name=self.name,
                name_is_auto=True,
            )

        names = list(self._components.keys())
        keys = jax.random.split(key, len(names))
        fields: dict[str, jnp.ndarray | Record] = {}
        for subkey, name in zip(keys, names):
            comp = self._components[name]
            fields[name] = (
                _sample_nested(name, comp, subkey)
                if isinstance(comp, dict)
                else comp._sample(subkey, ())
            )
        return Record(self.name, fields, name_is_auto=True)

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
        from ..core._record_batch import RecordBatch
        from ..core.named_tree import _unflatten_paths

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
            value = self.unflatten_value(flat, template=self.event_template)
            # Single-field templates return a raw array (preserving the
            # "single-leaf returns raw" contract on the static method);
            # the tree-map below expects a per-field structure, so
            # re-key it under the lone field name.
            if isinstance(value, jnp.ndarray):
                (field_name,) = self.event_template.fields
                value = {field_name: value}
        if isinstance(value, RecordBatch):
            # Leaf-keyed columns, re-nested, so the tree map below pairs each
            # column with the component that declared it.
            value = _unflatten_paths({path: value[path] for path in value.event_template})
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
        return _map_components(self.name, self._components, lambda d: d._mean())

    def _variance(self) -> Record:
        return _map_components(self.name, self._components, lambda d: d._variance())

    def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
        return _mc_expectation(
            self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist
        )

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
        (``"outer/a"``), matching ``EventTemplate.leaf_shapes``, so every
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
        self,
        observed_leaves: dict[KeyPath, ArrayLike],
    ) -> ProductDistribution:
        new_components = _prune_leaves(self._components, set(observed_leaves.keys()))
        result = ProductDistribution(**new_components, name=self._name)
        # The result inherits this joint's name, so it mirrors this joint's
        # auto flag; the constructor would otherwise treat the inherited
        # (possibly auto-derived) name as user-given. Set directly — the
        # **components signature leaves no room for a name_is_auto keyword.
        object.__setattr__(result, "_name_is_auto", self._name_is_auto)
        conditioned_names = [" > ".join(path) for path in observed_leaves]
        result.with_provenance(
            Provenance.create(
                "condition_on",
                parents=[self],
                metadata={"conditioned": conditioned_names},
            )
        )
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


def _unpickle_product_distribution(components, name, name_is_auto, provenance, annotations=None):
    """Reconstruct a ProductDistribution (or dynamic subclass) from its components."""
    p = ProductDistribution(**components, name=name)
    p._restore_identity(name_is_auto=name_is_auto, provenance=provenance)
    if annotations is not None:
        p._annotations = annotations
    return p


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
        for comp_name in self._components:
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
            for pname, _pval in exemplar.parameters.items():
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
    def dtypes(self) -> dict[str, jnp.dtype]:
        """Per-field dtype — the TFP Blockwise's dtype spread
        across the auto-built single-field template."""
        return self._per_field_dict(self._tfp_dist.dtype)


# -- Helpers for nested component pytrees ----------------------------------


def _sample_columns(components: dict, key, sample_shape) -> dict:
    """One column per leaf of *components*, keyed by its leaf path.

    A batch stores a column per field rather than a value per element, so a
    nested product draws into one flat mapping over ``/``-paths — the keying the
    batch constructor takes. A component that draws a structured value of its own
    contributes its leaves under its own path, so a sub-product nests by path
    rather than by containment.

    Keys are split per level, exactly as the unbatched recursion splits them, so
    a batched draw and a single draw derive their subkeys the same way.
    """
    from ..core._record_batch import RecordBatch

    names = list(components.keys())
    keys = jax.random.split(key, len(names))
    columns: dict = {}
    for subkey, field_name in zip(keys, names):
        comp = components[field_name]
        if isinstance(comp, dict):
            columns.update(
                {
                    f"{field_name}{_PATH_SEP}{path}": column
                    for path, column in _sample_columns(comp, subkey, sample_shape).items()
                }
            )
            continue
        drawn = comp._sample(subkey, sample_shape)
        if isinstance(drawn, RecordBatch):
            # Raw columns: a field that is not an array presents as its own object
            # batch, and what belongs in this batch's storage is the column.
            columns.update(
                {
                    f"{field_name}{_PATH_SEP}{path}": column
                    for path, column in drawn._raw_columns().items()
                }
            )
        elif isinstance(drawn, Record):
            columns.update(
                {f"{field_name}{_PATH_SEP}{path}": drawn[path] for path in drawn.event_template}
            )
        else:
            columns[field_name] = drawn
    return columns


def _sample_nested(name: str, components: dict, key) -> Record:
    """One single draw from nested component dicts, as a nested ``Record``."""
    names = list(components.keys())
    keys = jax.random.split(key, len(names))
    fields: dict = {}
    for subkey, field_name in zip(keys, names):
        comp = components[field_name]
        fields[field_name] = (
            _sample_nested(field_name, comp, subkey)
            if isinstance(comp, dict)
            else comp._sample(subkey, ())
        )
    return Record(name, fields, name_is_auto=True)


def _map_components(name: str, components: dict, fn) -> Record:
    """Apply fn to each leaf distribution, returning nested Record."""
    fields: dict = {}
    for field_name, comp in components.items():
        if isinstance(comp, dict):
            fields[field_name] = _map_components(field_name, comp, fn)
        else:
            fields[field_name] = fn(comp)
    return Record(name, fields, name_is_auto=True)


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
