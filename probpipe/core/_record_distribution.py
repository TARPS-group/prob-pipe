"""RecordDistribution — generic Record-based distribution base.

Provides the named-component layer (``fields``, ``select()``) and
Record-aware flatten/unflatten over the event declaration. ``event_shape`` and
``d[name]`` belong to :class:`~probpipe.Distribution`, and ``dtypes``,
``supports``, ``dtype``, and ``support`` to :class:`~probpipe.NumericDistribution`.

``_RecordDistributionView`` is the lightweight component reference that
``d[name]`` returns for a law exposing a record.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp

from ..custom_types import Array, PRNGKey
from ..distributions._distribution import Distribution
from ._specs import NumericArraySpec, OutputSpec, RecordSpec, TermSpec, _components_record
from .named_tree import _PATH_SEP
from .protocols import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from .record import Record

__all__ = ["RecordDistribution", "_RecordDistributionView"]


def _interim_template(declaration: OutputSpec) -> RecordSpec:
    """The record template a declared law presents, until its readers move to the declaration.

    An interim implementation detail. An exposed record is its own template. A
    whole term presents as a one-field record under its component, holding only
    an array's shape, as the template built from a name and a shape did.
    """
    component = declaration._component_name
    if component is None:
        return declaration.spec
    spec = declaration.spec
    return RecordSpec(**{component: spec.shape if isinstance(spec, NumericArraySpec) else spec})


def _field_event_shape(template: RecordSpec, name: str) -> tuple[int, ...]:
    """Event shape of one top-level field of *template*.

    An :class:`NumericArraySpec` field returns its array ``shape``; a nested
    sub-structure or non-array (opaque / distribution / function) field has no
    single event shape and returns ``()``. This is a *distribution-side* view —
    "what is the per-field event shape of one draw?" — kept here rather than on
    :class:`RecordSpec`, whose own shape surface is leaf-level
    (:attr:`~probpipe.NumericRecordSpec.leaf_shapes`).
    """
    spec = template.children[name]
    return spec.shape if isinstance(spec, NumericArraySpec) else ()


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
                return self._extract(m)
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
                return self._extract(v)
            return self._field_draws().var(axis=0)

        extra_methods["_variance"] = _variance

    if "log_prob" in protocols:
        extra_bases.append(SupportsLogProb)

        def _log_prob(self, value):
            components = getattr(self._parent, "_components", None)
            for key in self._key_path:
                if not isinstance(components, dict) or key not in components:
                    break
                components = components[key]
            else:
                return components._log_prob(value)
            raise NotImplementedError(f"_log_prob not available for view {self._key_path!r}")

        extra_methods["_log_prob"] = _log_prob

    if "cov" in protocols:
        extra_bases.append(SupportsCovariance)

        def _cov(self):
            c = self._parent._cov()
            if isinstance(c, Record):
                return self._extract(c)
            raise NotImplementedError(f"_cov not available for view {self._key_path!r}")

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
    :class:`~probpipe.core.node.Function` broadcasting.

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
        A distribution whose declaration has the field.
    key : str
        Field name among the parent's declared components.
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __new__(
        cls,
        parent: RecordDistribution,
        key: str | tuple[str, ...],
    ) -> _RecordDistributionView:
        actual_cls = _view_class_for_parent(parent)
        return object.__new__(actual_cls)

    def __init__(self, parent: RecordDistribution, key: str | tuple[str, ...]) -> None:
        # The record the parent's declared components form.
        template = _components_record(parent.event_spec)
        # A string key is a slash path, as a tuple key is.
        key_path = tuple(key.split(_PATH_SEP)) if isinstance(key, str) else tuple(key)
        if not key_path:
            raise KeyError("Record distribution view path must not be empty")
        try:
            template_field = template.at_path(key_path)
        except KeyError as exc:
            raise KeyError(
                f"No field path {key_path!r} in the declaration "
                f"(available: {tuple(template.keys())})"
            ) from exc
        # Bypass Distribution.__init__ validation; the view's name is
        # derived from the field key, not user-supplied, so it is auto.
        self._init_tracked("/".join(key_path))
        self._parent = parent
        self._key = key_path[-1]
        self._key_path = key_path
        self._template_field = template_field
        # The parent's declared term at the path, a whole term under the
        # path's last segment (III.7).
        self._init_declaration(OutputSpec(**{self._key: template_field}))

    # -- Parent identity ---------------------------------------------------

    @property
    def parent(self) -> Distribution:
        """The :class:`RecordDistribution` this view points at.

        Shared-identity signal for the ``Function`` sweep layer: views with
        the same ``parent`` co-sample (preserve correlation) when passed as
        sibling broadcast args to a Function.

        A *value* batch needs no such pointer — a field selection off a
        ``RecordBatch`` is an ordinary batch, and sibling selections align by
        their shared level names. A distribution view has no level names to align
        on, so identity is what says two views draw from one law.
        """
        return self._parent

    @property
    def field(self) -> str:
        """Name of the viewed field (the final segment of its parent path)."""
        return self._key

    def __getitem__(self, key: str) -> _RecordDistributionView:
        """Return a view of one child below a structured record field."""
        if not isinstance(key, str):
            raise TypeError(f"key must be str, got {type(key).__name__}")
        if not isinstance(self._template_field, RecordSpec):
            raise KeyError(f"{self._key_path!r} is a field, not a nested record")
        if key not in self._template_field.children:
            raise KeyError(
                f"No field {key!r} below {self._key_path!r} "
                f"(available: {tuple(self._template_field.children)})"
            )
        return _RecordDistributionView(self._parent, (*self._key_path, key))

    # -- Single-field array-like shims -------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of one draw from this view — equals ``event_shape``."""
        return self.event_shape

    @property
    def ndim(self) -> int:
        """Number of axes in a single draw (``len(event_shape)``)."""
        return len(self.shape)

    # -- Internals ----------------------------------------------------------

    def _extract(self, structured: Any) -> Any:
        """Extract this field from a parent record, record batch, or flat array."""
        from ._record_batch import RecordBatch

        if isinstance(structured, Record):
            return structured.at_path(self._key_path)
        if isinstance(structured, RecordBatch):
            return structured[self._key_path]
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
        result = unflatten(jnp.asarray(structured), template=self._parent.event_spec.spec)
        if isinstance(result, Record):
            return result.at_path(self._key_path)
        if isinstance(result, RecordBatch):
            return result[self._key_path]
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
        from ._record_batch import RecordBatch

        draws = self._parent.draws()
        if isinstance(draws, (Record, RecordBatch)):
            return jnp.asarray(self._extract(draws))
        from ._numeric_record import _reconstruct_from_vector

        result = _reconstruct_from_vector(
            self._parent.name, _components_record(self._parent.event_spec), jnp.asarray(draws)
        )
        return jnp.asarray(self._extract(result))

    def __repr__(self) -> str:
        return (
            f"_RecordDistributionView(parent={type(self._parent).__name__}, "
            f"path={self._key_path!r})"
        )


# ---------------------------------------------------------------------------
# Record template builder
# ---------------------------------------------------------------------------


def _build_event_template(
    components: dict[str, Any],
) -> RecordSpec:
    """Build a RecordSpec from a component pytree.

    Each leaf contributes a spec for the parent template:

    - Nested ``dict`` → recursively built nested ``RecordSpec``.
    - :class:`NumericRecordDistribution` → the leaf's ``event_shape``, or,
      for a leaf that draws a record, as a nested joint does, its
      ``event_template``.
    - Any other :class:`RecordDistribution` → the leaf's
      ``event_template`` (embedded as a nested structural template).
    - Any other :class:`Distribution` → ``None`` (opaque leaf — the
      template records the field name but not a shape).
    """
    from ..distributions._distribution import Distribution
    from ._numeric_record_distribution import NumericRecordDistribution

    specs: dict[str, Any] = {}
    for name, comp in components.items():
        if isinstance(comp, dict):
            specs[name] = _build_event_template(comp)
        elif isinstance(comp, NumericRecordDistribution):
            try:
                specs[name] = comp.event_shape
            except TypeError:
                specs[name] = comp.event_template
        elif isinstance(comp, RecordDistribution):
            specs[name] = comp.event_template
        elif isinstance(comp, Distribution):
            specs[name] = None
        else:
            raise TypeError(f"Unexpected component type: {type(comp).__name__}")
    return RecordSpec(specs)


def _record_with_leaves(template: RecordSpec, dtype: Any, support: Any) -> RecordSpec:
    """*template* with every array leaf declaring *dtype* and *support*."""

    def leaf(spec: TermSpec) -> TermSpec:
        if isinstance(spec, RecordSpec):
            return _record_with_leaves(spec, dtype, support)
        if isinstance(spec, NumericArraySpec):
            return NumericArraySpec(spec.shape, dtype, support)
        return spec

    return RecordSpec({field: leaf(spec) for field, spec in template.children.items()})


def _joint_event_spec(components: dict[str, Any]) -> RecordSpec:
    """The record a joint draws: each component's declared term, a nested dict as a record.

    The declaration keeps each component's dtype and support.
    """
    return RecordSpec(
        {
            name: _joint_event_spec(comp) if isinstance(comp, dict) else comp.event_spec.spec
            for name, comp in components.items()
        }
    )


# ---------------------------------------------------------------------------
# RecordDistribution
# ---------------------------------------------------------------------------


class RecordDistribution(Distribution):
    """Generic Record-based distribution.

    Provides named component access (``fields``, ``select()``) and
    Record-aware flatten / unflatten over the event declaration, an interim
    implementation detail of a class the design retires.
    """

    # -- Record template --------------------------------------------------------

    @property
    def event_template(self) -> RecordSpec:
        """The declaration presented as a record template.

        An interim implementation detail, until its readers move to
        :attr:`event_spec`: a class that stores a template presents it, and any
        other law presents its declaration, a whole term as a one-field record
        under its component.
        """
        stored = getattr(self, "_event_template", None)
        return stored if stored is not None else _interim_template(self.event_spec)

    # -- Named component access ---------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        """Top-level field names of one draw, the components of its declaration."""
        return tuple(self.event_spec.components)

    def select(self, *fields: str, **mapping: str) -> dict[str, _RecordDistributionView]:
        """Select named fields as views for Function broadcasting.

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
        :meth:`Record.select_all` / :meth:`RecordBatch.select_all` so
        the splat-all pattern works uniformly across the three field-
        bearing container types. Preserves cross-field correlation via
        the parent-identity machinery in the ``Function`` sweep
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
        non-array (opaque / distribution / function) field reports ``()``.
        """
        return {
            name: spec.shape if isinstance(spec, NumericArraySpec) else ()
            for name, spec in self.event_spec.components.items()
        }

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
        return _field_event_shape(_components_record(self.event_spec), name)

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
