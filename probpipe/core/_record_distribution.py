"""RecordDistribution — generic Record-based distribution base.

Provides the named-component layer (``component_names``, ``__getitem__``,
``select()``) and Record-aware flatten/unflatten, without imposing TFP
shape conventions (dtype, support, batch_shape).  Those live on
``TFPShapeMixin`` and its consumers.

``_RecordDistributionView`` is the lightweight component reference,
analogous to the former ``DistributionView`` but for any distribution
whose ``record_template`` is set.
"""

from __future__ import annotations

from collections.abc import Iterator
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
from .record import Record, RecordTemplate
from ..custom_types import Array, PRNGKey

if TYPE_CHECKING:
    from ._array_distributions import FlattenedView


__all__ = ["RecordDistribution", "_RecordDistributionView"]


# ---------------------------------------------------------------------------
# Dynamic view class factory
# ---------------------------------------------------------------------------

_VIEW_CLASS_CACHE: dict[frozenset[str], type] = {}


def _view_class_for_parent(parent: Distribution) -> type[_RecordDistributionView]:
    """Return a ``_RecordDistributionView`` subclass whose protocol bases
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
            if isinstance(c, Record):
                return c[self._key]
            raise NotImplementedError(
                f"_cov not available for view {self._key!r}"
            )
        extra_methods["_cov"] = _cov

    if not extra_bases:
        _VIEW_CLASS_CACHE[key] = _RecordDistributionView
        return _RecordDistributionView

    cls_name = "_RecordDistributionView"
    new_cls = type(cls_name, (_RecordDistributionView, *extra_bases), extra_methods)
    _VIEW_CLASS_CACHE[key] = new_cls
    return new_cls


# ---------------------------------------------------------------------------
# _RecordDistributionView
# ---------------------------------------------------------------------------


class _RecordDistributionView(Distribution, SupportsSampling, SupportsMean, SupportsVariance):
    """Lightweight reference to a single named field of a Record-based distribution.

    The Record-world analog of
    :class:`~probpipe.core._joint.DistributionView`.  Preserves
    correlation when multiple views from the same parent are used in
    :class:`~probpipe.core.node.WorkflowFunction` broadcasting.

    **Dynamic protocol support:** The view's ``isinstance`` protocol
    compliance matches the parent's capabilities.  When the parent
    supports ``SupportsLogProb``, the view does too (via a cached
    dynamic subclass).  Use :func:`_view_class_for_parent` or call
    ``_RecordDistributionView(parent, key)`` — the ``__new__`` method
    selects the right subclass automatically.

    Parameters
    ----------
    parent : Distribution
        A distribution with ``record_template`` set.
    key : str
        Field name in the parent's ``record_template``.
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __new__(cls, parent: Distribution, key: str):
        actual_cls = _view_class_for_parent(parent)
        return object.__new__(actual_cls)

    def __init__(self, parent: Distribution, key: str):
        template = parent.record_template
        if template is None or key not in template:
            raise KeyError(
                f"No field {key!r} in record_template "
                f"(available: {template.fields if template is not None else ()})"
            )
        # Bypass Distribution.__init__ validation (view name comes from
        # the field key, not a user-supplied argument).
        self._name = key
        self._parent = parent
        self._key = key
        self._key_path = (key,)
        self._template_field = template[key]

    # -- Shape info ---------------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        f = self._template_field
        if isinstance(f, tuple):
            # RecordTemplate: field spec is a shape tuple
            return f
        if isinstance(f, RecordTemplate):
            return ()
        # Legacy Record template: field is an array
        return f.shape if not isinstance(f, Record) else ()

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
            if isinstance(m, Record):
                return m[self._key]
        return self._field_draws().mean(axis=0)

    def _variance(self) -> Array:
        if isinstance(self._parent, SupportsVariance):
            v = self._parent._variance()
            if isinstance(v, Record):
                return v[self._key]
        return self._field_draws().var(axis=0)

    # -- Internals ----------------------------------------------------------

    def _extract(self, structured) -> Array:
        """Extract this field from a parent sample (Record, NumericRecordArray, or flat array)."""
        from ._record_array import RecordArray
        if isinstance(structured, (Record, RecordArray)):
            return structured[self._key]
        # Flat array — unflatten via parent's unflatten_value
        result = self._parent.unflatten_value(jnp.asarray(structured))
        if isinstance(result, (Record, RecordArray)):
            return result[self._key]
        return result

    def _field_draws(self) -> Array:
        """All draws for this field (requires parent to have a ``draws()`` method)."""
        from ._record_array import RecordArray
        draws = self._parent.draws()
        if isinstance(draws, (Record, RecordArray)):
            return jnp.asarray(draws[self._key])
        # Flat array draws — unflatten
        from ._record_array import NumericRecordArray
        result = NumericRecordArray.unflatten(
            jnp.asarray(draws), template=self._parent.record_template
        )
        return jnp.asarray(result[self._key])

    def __repr__(self) -> str:
        return (
            f"_RecordDistributionView(parent={type(self._parent).__name__}, "
            f"field={self._key!r})"
        )


# ---------------------------------------------------------------------------
# Record template builder
# ---------------------------------------------------------------------------


def _build_record_template(components: dict) -> RecordTemplate:
    """Build a RecordTemplate from a component pytree.

    Each ``ArrayDistribution`` leaf contributes its ``event_shape``.
    Nested dicts become nested ``RecordTemplate``.
    """
    from ._array_distributions import ArrayDistribution

    specs: dict[str, tuple[int, ...] | RecordTemplate] = {}
    for name, comp in components.items():
        if isinstance(comp, dict):
            specs[name] = _build_record_template(comp)
        elif isinstance(comp, ArrayDistribution):
            specs[name] = comp.event_shape
        else:
            raise TypeError(f"Unexpected component type: {type(comp).__name__}")
    return RecordTemplate(specs)


# ---------------------------------------------------------------------------
# RecordDistribution
# ---------------------------------------------------------------------------


class RecordDistribution(Distribution[Record]):
    """Generic Record-based distribution.

    Provides named component access (``component_names``, ``__getitem__``,
    ``select()``) and Record-aware flatten / unflatten.  Does NOT impose
    TFP shape conventions (dtype, support, batch_shape) — those belong
    on ``NumericRecordDistribution`` and its consumers.

    Concrete subclasses must set ``_record_template`` (a
    :class:`~probpipe.core.record.RecordTemplate` describing the named
    structure) and implement the relevant sampling / log-prob protocols.
    """

    # -- Record template (owned here, NOT on Distribution base) -------------

    @property
    def record_template(self) -> RecordTemplate | None:
        """Structural template describing this distribution's samples.

        Returns a :class:`~probpipe.core.record.RecordTemplate` with
        field names and per-field shapes, or ``None`` if no template
        is set.
        """
        return getattr(self, "_record_template", None)

    # -- Named component access ---------------------------------------------

    @property
    def component_names(self) -> tuple[str, ...]:
        """Field names from the record_template, or empty tuple."""
        tpl = self.record_template
        return tpl.fields if tpl is not None else ()

    def __getitem__(self, key: str) -> _RecordDistributionView:
        return _RecordDistributionView(self, key)

    def __getattr__(self, name: str):
        """Attribute access for named fields: ``dist.field_name`` → view."""
        # Skip private attrs to avoid infinite recursion during __dict__ lookups
        if name.startswith('_'):
            raise AttributeError(name)
        tpl = self.__dict__.get('_record_template')
        if tpl is not None and name in tpl:
            return self[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute {name!r}"
        )

    def select(self, *fields: str, **mapping: str) -> dict[str, _RecordDistributionView]:
        """Select named fields as views for workflow function broadcasting.

        Positional args use the field name as the argument name.
        Keyword args remap: ``select(x="field_name")``.

        Usage::

            predict(**posterior.select("r", "K", "phi"), x=x_grid)
        """
        result: dict[str, _RecordDistributionView] = {}
        for f in fields:
            result[f] = self[f]
        for arg_name, field_name in mapping.items():
            result[arg_name] = self[field_name]
        return result

    # -- Dict-like interface (mirrors Record) ---------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        """Field names, matching ``Record.fields``."""
        return self.component_names

    def __contains__(self, name: str) -> bool:
        return name in self.component_names

    # NOTE: __iter__ and __len__ are intentionally NOT implemented.
    # Adding them causes JAX/numpy to treat distributions as sequences,
    # making jnp.asarray(dist) silently convert to an empty array.

    def keys(self) -> Iterator[str]:
        """Iterate over component names."""
        return iter(self.component_names)

    def values(self) -> Iterator[_RecordDistributionView]:
        """Iterate over component views."""
        for name in self.component_names:
            yield self[name]

    def items(self) -> Iterator[tuple[str, _RecordDistributionView]]:
        """Iterate over (name, view) pairs."""
        for name in self.component_names:
            yield name, self[name]

    # -- Flatten / unflatten ------------------------------------------------

    def flatten_value(self, value) -> Array:
        """Flatten a Record or NumericRecordArray sample to a flat array."""
        from ._record_array import NumericRecordArray
        if isinstance(value, NumericRecordArray):
            return value.flatten()
        if isinstance(value, Record):
            return value.flatten()
        return value

    def unflatten_value(self, flat: Array):
        """Reconstruct a NumericRecord or NumericRecordArray from a flat array."""
        from ._numeric_record import NumericRecord
        from ._record_array import NumericRecordArray
        tpl = self.record_template
        if tpl is None:
            raise RuntimeError("Cannot unflatten without record_template")
        flat = jnp.asarray(flat)
        if flat.ndim < 2:
            return NumericRecord.unflatten(flat, template=tpl)
        return NumericRecordArray.unflatten(flat, template=tpl)

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
        tpl = self.record_template
        if tpl is None:
            return 0
        if isinstance(tpl, RecordTemplate):
            from .._utils import prod
            return sum(
                prod(shape) if shape else 1
                for shape in tpl.numeric_leaf_shapes.values()
            )
        # Legacy Record template fallback
        return tpl.flat_size

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-field event shapes derived from record_template."""
        tpl = self.record_template
        if tpl is None:
            return {}
        if isinstance(tpl, RecordTemplate):
            # Return top-level field shapes only; nested RecordTemplate → ()
            result: dict[str, tuple[int, ...]] = {}
            for name in tpl.fields:
                spec = tpl[name]
                if isinstance(spec, RecordTemplate):
                    result[name] = ()
                elif spec is not None:
                    result[name] = spec
            return result
        # Legacy Record template fallback
        result: dict[str, tuple[int, ...]] = {}
        for name in tpl.fields:
            val = tpl[name]
            result[name] = val.shape if not isinstance(val, Record) else ()
        return result



# ---------------------------------------------------------------------------
# JAX PyTree registration helpers
# ---------------------------------------------------------------------------


def _register_dynamic_subclass(cls: type) -> type:
    """Register a dynamically-created RecordDistribution subclass as a JAX
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
