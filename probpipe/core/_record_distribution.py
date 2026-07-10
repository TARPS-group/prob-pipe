"""RecordDistribution — generic Record-based distribution base.

Provides the named-component layer (``fields``, ``__getitem__``,
``select()``) and Record-aware flatten/unflatten, without imposing
the numeric shape / dtype conventions (``dtype``, ``support``,
``event_shape``).  Those live on ``NumericRecordDistribution`` and
its consumers.

``_RecordDistributionView`` is the lightweight component reference,
analogous to the former ``DistributionView`` but for any distribution
whose ``event_template`` is set.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp

from ..custom_types import Array, PRNGKey
from ._distribution_base import Distribution
from .event_template import ArraySpec, EventTemplate
from .protocols import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from .record import Record
from .tracked import _TrackedMeta

__all__ = ["RecordDistribution", "_RecordDistributionView"]


def _field_event_shape(template: EventTemplate, name: str) -> tuple[int, ...]:
    """Event shape of one top-level field of *template*.

    An :class:`ArraySpec` field returns its array ``shape``; a nested
    sub-structure or non-array (opaque / distribution / function) field has no
    single event shape and returns ``()``. This is a *distribution-side* view —
    "what is the per-field event shape of one draw?" — kept here rather than on
    :class:`EventTemplate`, whose own shape surface is leaf-level
    (:attr:`~probpipe.NumericEventTemplate.leaf_shapes`).
    """
    spec = template.children[name]
    return spec.shape if isinstance(spec, ArraySpec) else ()


# ---------------------------------------------------------------------------
# Dynamic view class factory
# ---------------------------------------------------------------------------

_VIEW_CLASS_CACHE: dict[frozenset[str], type] = {}


def _view_class_for_parent(parent: Distribution) -> type[_RecordDistributionView]:
    """Return a ``_RecordDistributionView`` subclass whose protocol bases
    match the capabilities of *parent*.

    The view only claims to implement a protocol when the underlying
    parent implements it — otherwise ``isinstance(view, SupportsFoo)``
    would lie, and dispatch code that checks the protocol would hand
    the view work the parent can't satisfy.  All protocol methods
    (sampling, mean, variance, log-prob, covariance) are therefore
    added dynamically here, not on the base class. The cache key is
    the set of supported protocol names, so each unique combination
    produces one subclass.
    """
    protocols: set[str] = set()
    if isinstance(parent, SupportsSampling):
        protocols.add("sample")
    if isinstance(parent, SupportsMean):
        protocols.add("mean")
    if isinstance(parent, SupportsVariance):
        protocols.add("variance")
    if isinstance(parent, SupportsLogProb):
        protocols.add("log_prob")
    if isinstance(parent, SupportsCovariance):
        protocols.add("cov")

    key = frozenset(protocols)
    if key in _VIEW_CLASS_CACHE:
        return _VIEW_CLASS_CACHE[key]

    extra_bases: list[type] = []
    extra_methods: dict[str, object] = {}

    if "sample" in protocols:
        extra_bases.append(SupportsSampling)

        def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
            structured = self._parent._sample(key, sample_shape)
            return self._extract(structured)

        extra_methods["_sample"] = _sample

    if "mean" in protocols:
        extra_bases.append(SupportsMean)

        def _mean(self) -> Array:
            m = self._parent._mean()
            if isinstance(m, Record):
                return m[self._key]
            # Parent returned a flat array — fall back to empirical mean
            # over draws for just this field. Requires the parent to
            # expose ``draws()`` (all ApproximateDistribution subclasses do).
            return self._field_draws().mean(axis=0)

        extra_methods["_mean"] = _mean

    if "variance" in protocols:
        extra_bases.append(SupportsVariance)

        def _variance(self) -> Array:
            v = self._parent._variance()
            if isinstance(v, Record):
                return v[self._key]
            return self._field_draws().var(axis=0)

        extra_methods["_variance"] = _variance

    if "log_prob" in protocols:
        extra_bases.append(SupportsLogProb)

        def _log_prob(self, value):
            components = getattr(self._parent, "_components", None)
            if components is not None and self._key in components:
                return components[self._key]._log_prob(value)
            raise NotImplementedError(f"_log_prob not available for view {self._key!r}")

        extra_methods["_log_prob"] = _log_prob

    if "cov" in protocols:
        extra_bases.append(SupportsCovariance)

        def _cov(self):
            c = self._parent._cov()
            if isinstance(c, Record):
                return c[self._key]
            raise NotImplementedError(f"_cov not available for view {self._key!r}")

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


class _RecordDistributionView(Distribution):
    """Lightweight reference to a single named field of a Record-based distribution.

    The Record-world analog of
    :class:`~probpipe.core._joint.DistributionView`. Preserves
    correlation when multiple views from the same parent are used in
    :class:`~probpipe.core.node.WorkflowFunction` broadcasting.

    **Dynamic protocol support:** this base class intentionally does
    not inherit any ``SupportsFoo`` protocols. Each concrete instance
    is a cached subclass built by :func:`_view_class_for_parent`, which
    mixes in only the protocols the parent actually implements. Calling
    ``_RecordDistributionView(parent, key)`` routes through ``__new__``
    and picks the right subclass automatically, so
    ``isinstance(view, SupportsSampling)`` is True iff the parent is.

    Parameters
    ----------
    parent : Distribution
        A distribution with ``event_template`` set.
    key : str
        Field name in the parent's ``event_template``.
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __new__(cls, parent: RecordDistribution, key: str) -> _RecordDistributionView:
        actual_cls = _view_class_for_parent(parent)
        return object.__new__(actual_cls)

    def __init__(self, parent: RecordDistribution, key: str) -> None:
        # ``parent.event_template`` is contractually non-``None``
        # (metaclass-enforced on every ``RecordDistribution`` instance).
        template = parent.event_template
        if key not in template.children:
            raise KeyError(
                f"No field {key!r} in event_template (available: {tuple(template.children)})"
            )
        # Bypass Distribution.__init__ validation; the view's name is
        # derived from the field key, not user-supplied, so it is auto.
        self._init_tracked(key, name_is_auto=True)
        self._parent = parent
        self._key = key
        self._key_path = (key,)
        self._template_field = template.children[key]

    # -- Parent identity (mirrors ``_RecordArrayView``) --------------------

    @property
    def parent(self) -> Distribution:
        """The :class:`RecordDistribution` this view points at.

        Shared-identity signal for the ``WorkflowFunction`` sweep layer:
        views with the same ``parent`` co-sample (preserve correlation)
        when passed as sibling broadcast args to a workflow function.
        Matches the ``_RecordArrayView.parent`` surface.
        """
        return self._parent

    @property
    def field(self) -> str:
        """Name of the viewed field (the top-level key into the parent)."""
        return self._key

    # -- Shape info ---------------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        f = self._template_field
        if isinstance(f, ArraySpec):
            # Numeric array leaf — its event shape is the spec shape.
            return f.shape
        # Nested template / opaque / distribution / function leaf.
        return ()

    # -- Single-field array-like shims (mirrors ``_RecordArrayView``) ------

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of one draw from this view — equals ``event_shape``."""
        return self.event_shape

    @property
    def dtype(self) -> jnp.dtype | None:
        """Dtype of a single draw, if the parent exposes ``dtypes``."""
        dtypes = getattr(self._parent, "dtypes", None)
        if isinstance(dtypes, dict):
            return dtypes.get(self._key)
        return None

    @property
    def ndim(self) -> int:
        """Number of axes in a single draw (``len(event_shape)``)."""
        return len(self.shape)

    # -- Internals ----------------------------------------------------------

    def _extract(self, structured: Any) -> Array:
        """Extract this field from a parent sample (Record, NumericRecordArray, or flat array)."""
        from ._record_array import RecordArray

        if isinstance(structured, (Record, RecordArray)):
            return structured[self._key]
        # Flat array — unflatten via the parent's static unflatten_value.
        # Only numeric parents define unflatten_value; non-numeric Record
        # parents never reach this branch (their samples are Records).
        unflatten = getattr(type(self._parent), "unflatten_value", None)
        if unflatten is None:
            raise TypeError(
                f"Cannot extract field {self._key!r} from a flat array "
                f"on {type(self._parent).__name__}: parent does not "
                f"implement unflatten_value."
            )
        result = unflatten(
            jnp.asarray(structured),
            template=self._parent.event_template,
        )
        if isinstance(result, (Record, RecordArray)):
            return result[self._key]
        return result

    def _field_draws(self) -> Array:
        """All draws for this field (requires parent to have a ``draws()`` method).

        Used by the dynamically-installed ``_mean`` / ``_variance`` when
        the parent's own ``_mean()`` / ``_variance()`` returns a flat
        array rather than a ``Record``. Only reachable when the parent
        is ``SupportsMean`` / ``SupportsVariance`` (so the method is
        present on the dynamic subclass at all), and such parents in
        practice are ``ApproximateDistribution`` subclasses that do
        expose ``draws()``.
        """
        from ._record_array import RecordArray

        draws = self._parent.draws()
        if isinstance(draws, (Record, RecordArray)):
            return jnp.asarray(draws[self._key])
        result = self._parent.event_template.from_vector(jnp.asarray(draws))
        return jnp.asarray(result[self._key])

    def __repr__(self) -> str:
        return f"_RecordDistributionView(parent={type(self._parent).__name__}, field={self._key!r})"


# ---------------------------------------------------------------------------
# Record template builder
# ---------------------------------------------------------------------------


def _build_event_template(
    components: dict[str, Any],
) -> EventTemplate:
    """Build an EventTemplate from a component pytree.

    Each leaf contributes a spec for the parent template:

    - Nested ``dict`` → recursively built nested ``EventTemplate``.
    - :class:`NumericRecordDistribution` → the leaf's ``event_shape``
      (numeric shape tuple).
    - Any other :class:`RecordDistribution` → the leaf's
      ``event_template`` (embedded as a nested structural template).
    - Any other :class:`Distribution` → ``None`` (opaque leaf — the
      template records the field name but not a shape).
    """
    from ._distribution_base import Distribution
    from ._numeric_record_distribution import NumericRecordDistribution

    specs: dict[str, Any] = {}
    for name, comp in components.items():
        if isinstance(comp, dict):
            specs[name] = _build_event_template(comp)
        elif isinstance(comp, NumericRecordDistribution):
            specs[name] = comp.event_shape
        elif isinstance(comp, RecordDistribution):
            specs[name] = comp.event_template
        elif isinstance(comp, Distribution):
            specs[name] = None
        else:
            raise TypeError(f"Unexpected component type: {type(comp).__name__}")
    return EventTemplate(specs)


# ---------------------------------------------------------------------------
# RecordDistribution
# ---------------------------------------------------------------------------


class _RecordDistributionMeta(_TrackedMeta):
    """Metaclass adding the ``event_template`` set-or-derivable check
    on top of the base ``name`` check.

    After ``__init__`` returns, accesses ``instance.event_template``
    once. Either path is fine: ``_event_template`` was set directly
    (multi-leaf joints), or the auto-build path on
    :class:`~probpipe.core._numeric_record_distribution.NumericRecordDistribution`
    derives a single-field template from ``name`` + ``event_shape``.
    Both paths must yield a non-``None`` ``EventTemplate``.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        try:
            tpl = instance.event_template
        except (TypeError, NotImplementedError) as exc:
            raise TypeError(
                f"{cls.__name__}.__init__ left event_template unresolved: {exc}"
            ) from exc
        if tpl is None:
            raise TypeError(
                f"{cls.__name__}.__init__ must leave event_template "
                f"set — either assign self._event_template = ..., "
                f"or declare event_shape so the auto-build path can "
                f"derive a single-field template."
            )
        return instance


class RecordDistribution(Distribution[Record], metaclass=_RecordDistributionMeta):
    """Generic Record-based distribution.

    Provides named component access (``fields``, ``__getitem__``,
    ``select()``) and Record-aware flatten / unflatten.  Does NOT impose
    numeric shape / dtype conventions (``dtype``, ``support``,
    ``event_shape``) — those belong on ``NumericRecordDistribution``
    and its consumers.

    Concrete subclasses must set ``_event_template`` (a
    :class:`~probpipe.core.record.EventTemplate` describing the named
    structure) and implement the relevant sampling / log-prob protocols.
    """

    # -- Record template (owned here, NOT on Distribution base) -------------

    @property
    def event_template(self) -> EventTemplate | None:
        """Structural template describing this distribution's samples.

        Returns a :class:`~probpipe.core.record.EventTemplate` with
        field names and per-field shapes, or ``None`` if no template
        is set.
        """
        return getattr(self, "_event_template", None)

    # -- Named component access ---------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        """Field names from the event_template.

        The metaclass guarantees ``event_template`` is non-``None`` on
        every :class:`RecordDistribution` instance, so this is a direct
        delegate (no ``None`` fallback).
        """
        return self.event_template.fields

    def __getitem__(self, key: str) -> _RecordDistributionView:
        return _RecordDistributionView(self, key)

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

    def select_all(self) -> dict[str, _RecordDistributionView]:
        """Return every component as a view, for splatting into function calls.

        Sugar for ``select(*self.fields)``. Matches
        :meth:`Record.select_all` / :meth:`RecordArray.select_all` so
        the splat-all pattern works uniformly across the three field-
        bearing container types. Preserves cross-field correlation via
        the parent-identity machinery in the ``WorkflowFunction`` sweep
        layer.
        """
        return self.select(*self.fields)

    # -- Dict-like interface (mirrors Record) ---------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self.fields

    # NOTE: __iter__ and __len__ are intentionally NOT implemented.
    # Adding them causes JAX/numpy to treat distributions as sequences,
    # making jnp.asarray(dist) silently convert to an empty array.

    def keys(self) -> Iterator[str]:
        """Iterate over component names."""
        return iter(self.fields)

    def values(self) -> Iterator[_RecordDistributionView]:
        """Iterate over component views."""
        for name in self.fields:
            yield self[name]

    def items(self) -> Iterator[tuple[str, _RecordDistributionView]]:
        """Iterate over (name, view) pairs."""
        for name in self.fields:
            yield name, self[name]

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-field event shapes (top-level fields only).

        An array-valued field reports its array shape; a nested sub-structure or
        non-array (opaque / distribution / function) field reports ``()``. The
        metaclass guarantees ``event_template`` is non-``None``.
        """
        tpl = self.event_template
        return {name: _field_event_shape(tpl, name) for name in tpl.fields}

    # -- Single-field array-like shims --------------------------------------
    # On a single-field distribution, ``.shape`` / ``.ndim`` delegate to
    # the sole field's event shape. Multi-field distributions raise
    # ``TypeError``; use ``.event_shapes`` for a per-field dict or
    # index into a view (``dist[field]``).

    def _single_field_name(self) -> str:
        fields = self.fields
        if len(fields) != 1:
            raise TypeError(
                f"{type(self).__name__} with {len(fields)} fields is not "
                f"array-like; index a specific field via dist[field] or "
                f"use .event_shapes dict."
            )
        return fields[0]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of one draw (equals the sole field's event_shape).

        Raises ``TypeError`` via :meth:`_single_field_name` on
        multi-field distributions.
        """
        name = self._single_field_name()
        return _field_event_shape(self.event_template, name)

    @property
    def ndim(self) -> int:
        """Number of axes in one draw."""
        return len(self.shape)


# ---------------------------------------------------------------------------
# JAX PyTree registration helpers
# ---------------------------------------------------------------------------


def _register_dynamic_subclass(cls: type) -> type:
    """Register a dynamically-created RecordDistribution subclass as a JAX
    pytree node, reusing the flatten/unflatten from the existing
    ``ProductDistribution`` registration in ``distributions/joint.py``.

    Dynamic subclasses of ``ProductDistribution`` share the same
    flatten/unflatten logic. Delegate to ``_components`` + ``_name``,
    capturing the top-level key order explicitly so insertion order
    survives the round-trip (JAX's dict treedef is sorted).
    """

    def _flatten(dist: Any) -> tuple[list[Any], tuple[Any, str, tuple[str, ...]]]:
        leaves = jax.tree.leaves(dist._components)
        treedef = jax.tree.structure(dist._components)
        return leaves, (treedef, dist._name, tuple(dist._components.keys()))

    def _unflatten(
        aux: tuple[Any, str, tuple[str, ...]],
        children: list[Any],
    ) -> Any:
        treedef, name, key_order = aux
        components = jax.tree.unflatten(treedef, children)
        ordered = {k: components[k] for k in key_order}
        return cls(**ordered, name=name)

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls
