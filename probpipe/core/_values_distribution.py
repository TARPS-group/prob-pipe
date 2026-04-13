"""ValuesDistribution — generic Values-based distribution base.

Provides the named-component layer (``component_names``, ``__getitem__``,
``select()``) and Values-aware flatten/unflatten, without imposing TFP
shape conventions (dtype, support, batch_shape).  Those live on
``TFPShapeMixin`` and its consumers.

``_ValuesDistributionView`` is the lightweight component reference,
analogous to :class:`~probpipe.core._joint.DistributionView` but for
any distribution whose ``values_template`` is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ._distribution_base import Distribution
from .protocols import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from .values import Values
from ..custom_types import Array, PRNGKey

if TYPE_CHECKING:
    from ._array_distributions import FlattenedView


__all__ = ["ValuesDistribution", "_ValuesDistributionView"]


# ---------------------------------------------------------------------------
# Dynamic view class factory
# ---------------------------------------------------------------------------

_VIEW_CLASS_CACHE: dict[frozenset[str], type] = {}


def _view_class_for_parent(parent: Distribution) -> type[_ValuesDistributionView]:
    """Return a ``_ValuesDistributionView`` subclass whose protocol bases
    match the capabilities of *parent*.

    Dynamic subclasses are cached by protocol signature so each unique
    combination is created only once.  The subclass has concrete
    implementations of the delegated protocol methods at the **class**
    level, which means ``isinstance(view, SupportsLogProb)`` works
    correctly with ``@runtime_checkable`` protocols.
    """
    protocols: set[str] = set()
    if isinstance(parent, SupportsLogProb):
        protocols.add("log_prob")
    if isinstance(parent, SupportsCovariance):
        protocols.add("cov")

    key = frozenset(protocols)
    if key in _VIEW_CLASS_CACHE:
        return _VIEW_CLASS_CACHE[key]

    extra_bases: list[type] = []
    extra_methods: dict[str, object] = {}

    if "log_prob" in protocols:
        extra_bases.append(SupportsLogProb)

        def _log_prob(self, value):
            components = getattr(self._parent, "_components", None)
            if components is not None and self._key in components:
                return components[self._key]._log_prob(value)
            raise NotImplementedError(
                f"_log_prob not available for view {self._key!r}"
            )
        extra_methods["_log_prob"] = _log_prob

    if "cov" in protocols:
        extra_bases.append(SupportsCovariance)

        def _cov(self):
            c = self._parent._cov()
            if isinstance(c, Values):
                return c[self._key]
            raise NotImplementedError(
                f"_cov not available for view {self._key!r}"
            )
        extra_methods["_cov"] = _cov

    if not extra_bases:
        _VIEW_CLASS_CACHE[key] = _ValuesDistributionView
        return _ValuesDistributionView

    cls_name = "_ValuesDistributionView"
    new_cls = type(cls_name, (_ValuesDistributionView, *extra_bases), extra_methods)
    _VIEW_CLASS_CACHE[key] = new_cls
    return new_cls


# ---------------------------------------------------------------------------
# _ValuesDistributionView
# ---------------------------------------------------------------------------


class _ValuesDistributionView(Distribution, SupportsSampling, SupportsMean, SupportsVariance):
    """Lightweight reference to a single named field of a Values-based distribution.

    The Values-world analog of
    :class:`~probpipe.core._joint.DistributionView`.  Preserves
    correlation when multiple views from the same parent are used in
    :class:`~probpipe.core.node.WorkflowFunction` broadcasting.

    **Dynamic protocol support:** The view's ``isinstance`` protocol
    compliance matches the parent's capabilities.  When the parent
    supports ``SupportsLogProb``, the view does too (via a cached
    dynamic subclass).  Use :func:`_view_class_for_parent` or call
    ``_ValuesDistributionView(parent, key)`` — the ``__new__`` method
    selects the right subclass automatically.

    Parameters
    ----------
    parent : Distribution
        A distribution with ``values_template`` set.
    key : str
        Field name in the parent's ``values_template``.
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __new__(cls, parent: Distribution, key: str):
        actual_cls = _view_class_for_parent(parent)
        return object.__new__(actual_cls)

    def __init__(self, parent: Distribution, key: str):
        template = parent.values_template
        if template is None or key not in template:
            raise KeyError(
                f"No field {key!r} in values_template "
                f"(available: {template.fields() if template else ()})"
            )
        self._parent = parent
        self._key = key
        self._key_path = (key,)
        self._template_field = template[key]

    # -- Shape info ---------------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        f = self._template_field
        return f.shape if not isinstance(f, Values) else ()

    # -- SupportsSampling ---------------------------------------------------

    def _sample_one(self, key: PRNGKey) -> Array:
        structured = self._parent._sample(key)
        return self._extract(structured)

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        structured = self._parent._sample(key, sample_shape)
        return self._extract(structured)

    # -- SupportsMean / SupportsVariance ------------------------------------

    def _mean(self) -> Array:
        if isinstance(self._parent, SupportsMean):
            m = self._parent._mean()
            if isinstance(m, Values):
                return m[self._key]
        return self._field_draws().mean(axis=0)

    def _variance(self) -> Array:
        if isinstance(self._parent, SupportsVariance):
            v = self._parent._variance()
            if isinstance(v, Values):
                return v[self._key]
        return self._field_draws().var(axis=0)

    # -- Internals ----------------------------------------------------------

    def _extract(self, structured: Array) -> Array:
        """Extract this field from a parent sample (flat array or Values)."""
        if isinstance(structured, Values):
            return structured[self._key]
        template = self._parent.values_template
        if structured.ndim == 1:
            return Values.unflatten(structured, template=template)[self._key]
        return _unflatten_batched(structured, template)[self._key]

    def _field_draws(self) -> Array:
        """All draws for this field (requires parent to have a ``draws()`` method)."""
        draws = self._parent.draws()
        if isinstance(draws, Values):
            return jnp.asarray(draws[self._key])
        return jnp.asarray(
            _unflatten_batched(draws, self._parent.values_template)[self._key]
        )

    def __repr__(self) -> str:
        return (
            f"_ValuesDistributionView(parent={type(self._parent).__name__}, "
            f"field={self._key!r})"
        )


# ---------------------------------------------------------------------------
# Batched unflatten helper
# ---------------------------------------------------------------------------


def _unflatten_batched(flat_draws: Array, template: Values) -> Values:
    """Unflatten a ``(num_draws, flat_size)`` array into a Values with a draw axis.

    Each field in the returned Values has shape ``(num_draws, *event_shape)``
    where ``event_shape`` comes from the corresponding field in *template*.
    """
    if jnp.ndim(flat_draws) < 2:
        raise ValueError(
            f"_unflatten_batched expects at least 2-D input, "
            f"got shape {jnp.shape(flat_draws)}"
        )
    fields: dict[str, jnp.ndarray | Values] = {}
    offset = 0
    for name in template.fields():
        tval = template[name]
        if isinstance(tval, Values):
            size = tval.flat_size
            child_flat = flat_draws[:, offset:offset + size]
            fields[name] = _unflatten_batched(child_flat, tval)
            offset += size
        else:
            size = tval.size
            event_shape = tval.shape
            chunk = flat_draws[:, offset:offset + size]
            fields[name] = chunk.reshape(flat_draws.shape[0], *event_shape)
            offset += size
    return Values(fields)


# ---------------------------------------------------------------------------
# Values template builder
# ---------------------------------------------------------------------------


def _build_values_template(components: dict) -> Values:
    """Build a Values template from a component pytree.

    Each ``ArrayDistribution`` leaf becomes a ``jnp.zeros(event_shape)``
    placeholder.  Nested dicts become nested ``Values``.
    """
    from ._array_distributions import ArrayDistribution

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
# ValuesDistribution
# ---------------------------------------------------------------------------


class ValuesDistribution(Distribution[Values]):
    """Generic Values-based distribution.

    Provides named component access (``component_names``, ``__getitem__``,
    ``select()``) and Values-aware flatten / unflatten.  Does NOT impose
    TFP shape conventions (dtype, support, batch_shape) — those belong
    on ``TFPShapeMixin`` and its consumers.

    Concrete subclasses must set ``_values_template`` (a :class:`Values`
    giving the named structure) and implement the relevant sampling /
    log-prob protocols.
    """

    # -- Named component access ---------------------------------------------

    @property
    def component_names(self) -> tuple[str, ...]:
        """Field names from the values_template, or empty tuple."""
        tpl = self.values_template
        return tpl.fields() if tpl is not None else ()

    def __getitem__(self, key: str) -> _ValuesDistributionView:
        return _ValuesDistributionView(self, key)

    def __getattr__(self, name: str):
        """Attribute access for named fields: ``dist.field_name`` → view."""
        # Skip private attrs to avoid infinite recursion during __dict__ lookups
        if name.startswith('_'):
            raise AttributeError(name)
        tpl = self.__dict__.get('_values_template')
        if tpl is not None and name in tpl:
            return self[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute {name!r}"
        )

    def select(self, *fields: str, **mapping: str) -> dict[str, _ValuesDistributionView]:
        """Select named fields as views for workflow function broadcasting.

        Positional args use the field name as the argument name.
        Keyword args remap: ``select(x="field_name")``.

        Usage::

            predict(**posterior.select("r", "K", "phi"), x=x_grid)
        """
        result: dict[str, _ValuesDistributionView] = {}
        for f in fields:
            result[f] = self[f]
        for arg_name, field_name in mapping.items():
            result[arg_name] = self[field_name]
        return result

    # -- Flatten / unflatten ------------------------------------------------

    def flatten_value(self, value: Values) -> Array:
        """Flatten a Values sample to a 1-D array."""
        return value.flatten()

    def unflatten_value(self, flat: Array) -> Values:
        """Reconstruct a Values from a flat array using the template."""
        tpl = self.values_template
        if tpl is None:
            raise RuntimeError("Cannot unflatten without values_template")
        return Values.unflatten(flat, template=tpl)

    def as_flat_distribution(self) -> FlattenedView:
        """View this distribution as a flat ``ArrayDistribution``.

        Returns a :class:`~probpipe.core._array_distributions.FlattenedView`
        with ``event_shape = (event_size,)`` for algorithms expecting
        flat vectors (MCMC, optimizers, VI methods).
        """
        from ._array_distributions import FlattenedView
        return FlattenedView(self)

    @property
    def event_size(self) -> int:
        """Total number of scalar elements in one sample."""
        tpl = self.values_template
        return tpl.flat_size if tpl is not None else 0

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-field event shapes derived from values_template."""
        tpl = self.values_template
        if tpl is None:
            return {}
        result: dict[str, tuple[int, ...]] = {}
        for name in tpl.fields():
            val = tpl[name]
            result[name] = val.shape if not isinstance(val, Values) else ()
        return result



# ---------------------------------------------------------------------------
# JAX PyTree registration helpers
# ---------------------------------------------------------------------------


def _register_dynamic_subclass(cls: type) -> type:
    """Register a dynamically-created ValuesDistribution subclass as a JAX
    pytree node, reusing the flatten/unflatten from the existing
    ``ProductDistribution`` registration in ``distributions/joint.py``.
    """
    # Avoid double-registration (the base ProductDistribution is
    # already registered in distributions/joint.py).
    try:
        jax.tree_util.tree_flatten(cls)  # probe
    except Exception:
        pass

    # Dynamic subclasses of ProductDistribution share the same
    # flatten/unflatten logic.  Delegate to _components + _name.
    def _flatten(dist):
        leaves = jax.tree.leaves(dist._components)
        treedef = jax.tree.structure(dist._components)
        return leaves, (treedef, dist._name)

    def _unflatten(aux, children):
        treedef, name = aux
        components = jax.tree.unflatten(treedef, children)
        return cls(**components, name=name)

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls
