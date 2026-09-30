"""Field views: the law of one event path of a parent distribution.

Provides:
  - ``FieldView`` – the ``Distribution`` that ``d[path]`` returns for a node
    below a law's whole term, or for a selection of several nodes, holding a
    reference to its parent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from ..core._dispatch import Feasibility
from ..core._record_batch import RecordBatch
from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericSpec, TermSpec
from ..core._specs import OutputSpec
from ..core.provenance import Provenance
from ..core.record import Record
from ..linalg import DenseLinOp, LinOp
from ._capabilities import (
    SupportsApproximateConditioning,
    SupportsCovariance,
    SupportsExactConditioning,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsVariance,
    _capability_guard,
    _capability_subclass,
)
from ._distribution import Distribution, _whole_term_component

if TYPE_CHECKING:
    from ..custom_types import Array, ArrayLike, PRNGKey

__all__ = ["FieldView"]

_PATH_SEP = "/"


# ---------------------------------------------------------------------------
# Paths, nodes, and coordinates of a declaration
# ---------------------------------------------------------------------------


def _node_at(declaration: OutputSpec, path: str) -> TermSpec:
    """The term spec of the node at *path* in *declaration*.

    A path starts with a component: an exposed record's paths are its record's,
    and a whole term's are its component followed by the paths within its term.

    Raises
    ------
    KeyError
        If *path* is not a path of *declaration*.
    """
    segments = tuple(path.split(_PATH_SEP))
    if not path or not all(segments):
        raise KeyError(path)
    component = _whole_term_component(declaration)
    spec = declaration.spec
    if component is None:
        return spec.at_path(*segments)
    head, *rest = segments
    if head != component:
        raise KeyError(path)
    if not rest:
        return spec
    if not isinstance(spec, RecordSpec):
        raise KeyError(path)
    return spec.at_path(*rest)


def _final_segment(path: str) -> str:
    return path.rsplit(_PATH_SEP, 1)[-1]


def _view_declaration(declaration: OutputSpec, path: str | tuple[str, ...]) -> OutputSpec:
    """The declaration of the view at *path* of a law declared by *declaration*.

    One path gives the node whole under a component named by its final segment,
    and a tuple of paths gives an exposed record of the nodes, in order.

    Raises
    ------
    TypeError
        If *path* is neither a string nor a tuple of strings.
    KeyError
        If a path is not a path of *declaration*, or the tuple is empty.
    ValueError
        If two selected paths share their final segment, so their components
        would collide.
    """
    if isinstance(path, str):
        return OutputSpec(**{_final_segment(path): _node_at(declaration, path)})
    if not isinstance(path, tuple) or not all(isinstance(each, str) for each in path):
        raise TypeError(f"a view's path is a string or a tuple of strings, got {path!r}")
    if not path:
        raise KeyError(path)
    nodes: dict[str, TermSpec] = {}
    for each in path:
        node = _node_at(declaration, each)
        component = _final_segment(each)
        if component in nodes:
            raise ValueError(
                f"the selected paths {list(path)} share the final segment {component!r}, "
                f"so their components would collide"
            )
        nodes[component] = node
    return OutputSpec(RecordSpec(nodes))


def _draw_segments(declaration: OutputSpec, path: str) -> tuple[str, ...]:
    """The segments of *path* below the top of a draw of *declaration*.

    An exposed record's draw is its record, which every segment addresses, and a
    whole term's draw is the term, so its component is dropped.
    """
    segments = tuple(path.split(_PATH_SEP))
    return segments if _whole_term_component(declaration) is None else segments[1:]


def _coordinate_range(declaration: OutputSpec, path: str) -> range:
    """The coordinates of the node at *path* in the flat vector of a draw of *declaration*.

    The flat vector lays out the array leaves in canonical order, depth first
    and in insertion order, so a node's coordinates are one contiguous range.

    Raises
    ------
    TypeError
        If the declaration is not numeric, so that a draw has no flat vector.
    ValueError
        If a dimension that the range depends on is unbound.
    """
    spec = declaration.spec
    if not isinstance(spec, NumericSpec):
        raise TypeError(
            f"a draw of {declaration!r} has no flat vector, since its declaration is not numeric"
        )
    start = 0
    for segment in _draw_segments(declaration, path):
        for name, child in spec.children.items():
            if name == segment:
                spec = child
                break
            start += child.vector_size
    return range(start, start + spec.vector_size)


def _leaf_paths(declaration: OutputSpec) -> list[str]:
    """The paths of *declaration* that address its leaves, each starting with a component."""
    leaves: list[str] = []
    for component, spec in declaration.components.items():
        if isinstance(spec, RecordSpec):
            leaves.extend(f"{component}{_PATH_SEP}{key}" for key in spec)
        else:
            leaves.append(component)
    return leaves


def _is_within(path: str, ancestor: str) -> bool:
    """Whether *path* is *ancestor* or a path below it."""
    return path == ancestor or path.startswith(ancestor + _PATH_SEP)


# ---------------------------------------------------------------------------
# The extraction of a node from a raw value
# ---------------------------------------------------------------------------


def _extract(value: Any, segments: tuple[str, ...]) -> Any:
    """The node at *segments* below the top of *value*, a raw draw or a value shaped like one.

    A record gives its field or sub-record, a batch of records the field's
    column or the sub-batch, and a nested mapping its entry.

    Raises
    ------
    TypeError
        If *segments* is not empty and *value* is none of these.
    """
    if not segments:
        return value
    if isinstance(value, Record):
        return value.at_path(*segments)
    if isinstance(value, RecordBatch):
        return value[segments]
    if isinstance(value, Mapping):
        for segment in segments:
            value = value[segment]
        return value
    raise TypeError(
        f"the node at {_PATH_SEP.join(segments)!r} cannot be read from a "
        f"{type(value).__name__}; a value with paths is a record, a batch of records, "
        f"or a mapping"
    )


def _assemble(value: Any, parts: Sequence[tuple[str, tuple[str, ...]]], name: str) -> Any:
    """The record of the nodes of *value* at each part's segments, keyed by its component.

    The result keeps the raw form of *value*: a batch of records gives a batch
    at the same levels, a record a record, and a mapping a mapping. The result's
    schema is read from the schema *value* carries. A value of any other kind
    is a whole term, which every part takes whole, and gives a record named
    *name*.
    """
    if isinstance(value, RecordBatch):
        template = value.element_spec
        stored = value._raw_columns()
        columns: dict[str, Any] = {}
        for component, segments in parts:
            prefix = _PATH_SEP.join(segments)
            for key, column in stored.items():
                if not segments:
                    columns[f"{component}{_PATH_SEP}{key}"] = column
                elif key == prefix:
                    columns[component] = column
                elif key.startswith(prefix + _PATH_SEP):
                    columns[component + key[len(prefix) :]] = column
        schema = RecordSpec(
            {
                component: template.at_path(*segments) if segments else template
                for component, segments in parts
            }
        )
        return value._rebuilt(columns, schema)
    if isinstance(value, Record):
        template = value.event_template
        return Record(
            value.name,
            {component: _extract(value, segments) for component, segments in parts},
            event_template=RecordSpec(
                {
                    component: template.at_path(*segments) if segments else template
                    for component, segments in parts
                }
            ),
        )
    if isinstance(value, Mapping):
        return {component: _extract(value, segments) for component, segments in parts}
    return Record(name, {component: value for component, _ in parts})


def _named_as(law: Distribution, components: Sequence[str]) -> Distribution:
    """*law* with its components renamed, in order, to *components*."""
    renames = {
        found: wanted
        for found, wanted in zip(law.event_spec.components, components)
        if found != wanted
    }
    return law.with_path_names(renames) if renames else law


# ---------------------------------------------------------------------------
# The derived capabilities
# ---------------------------------------------------------------------------
#
# Each function realizes one row of the derivation from a parent's capability,
# with pi the extraction of the view's node from a parent draw, or of the record
# of the selected nodes.


def _view_sample(self: FieldView, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Any:
    """Co-sample: draw ``X`` from the parent with the same key and return ``pi(X)``.

    The key and the sample shape pass to the parent unchanged, so views of one
    parent drawn with one key project one parent draw.
    """
    return self._project(self._parent._sample(key, sample_shape))


def _view_mean(self: FieldView) -> Any:
    """Projection: the parent's mean at the view's path, since ``E[pi X] = pi E[X]``."""
    return self._project(self._parent._mean())


def _view_variance(self: FieldView) -> Any:
    """Restriction of the parent's variance to the coordinates of the view's path."""
    return self._project(self._parent._variance())


def _view_cov(self: FieldView) -> LinOp:
    """The sub-block ``P Σ Pᵀ`` of the parent's covariance, ``P`` selecting the path.

    ``P`` selects the view's coordinates of the parent's flat vector, in the
    parent's order, and the product stays lazy.

    Raises
    ------
    TypeError
        If the parent's declaration is not numeric.
    """
    cov = self._parent._cov()
    rows = jnp.asarray(self._coordinates())
    selection = DenseLinOp(jnp.eye(cov.shape[1], dtype=cov.dtype)[rows])
    return selection @ cov @ selection.T


def _view_quantile(self: FieldView, q: ArrayLike) -> Array:
    """Restriction of the parent's per-coordinate quantiles to the path.

    The parent's quantiles carry the level axes first and the parent's flat
    coordinates last, and the view keeps its own coordinates of the last axis.

    Raises
    ------
    TypeError
        If the parent's declaration is not numeric.
    """
    return jnp.asarray(self._parent._quantile(q))[..., jnp.asarray(self._coordinates())]


def _view_expectation(self: FieldView, f: Callable[[Any], Array]) -> Array:
    """Composition: the parent's exact expectation of ``f ∘ pi``."""
    return self._parent._expectation(lambda value: f(self._project(value)))


def _view_log_prob(self: FieldView, value: Any) -> Array:
    """The log-density of the parent's detached marginal at the path.

    Raises
    ------
    TypeError
        If that marginal has no normalized density.
    """
    marginal = self._parent._marginal(self._path)
    if not isinstance(marginal, SupportsLogProb):
        raise TypeError(
            f"the marginal of {self._parent.name!r} at {self._path!r} has no normalized density"
        )
    return marginal._log_prob(value)


def _view_log_prob_guard(self: FieldView) -> Feasibility:
    """The parent's marginal guard at the view's path.

    Whether that marginal scores is decided when the marginal is built.
    """
    return _capability_guard(self._parent, "_marginal", self._path)


def _view_marginal(self: FieldView, path: str | tuple[str, ...]) -> Distribution:
    """Path composition: the parent's marginal at the parent's paths for *path*.

    *path* is an event path of the view, or a tuple of them, and the result is
    named by the paths of the view.

    Raises
    ------
    KeyError
        If a path is not an event path of the view.
    """
    paths = (path,) if isinstance(path, str) else tuple(path)
    parent_paths = self._parent_paths(paths)
    marginal = self._parent._marginal(
        parent_paths[0] if isinstance(path, str) else tuple(parent_paths)
    )
    return _named_as(marginal, [_final_segment(each) for each in paths])


def _view_marginal_guard(self: FieldView, path: str | tuple[str, ...]) -> Feasibility:
    """The parent's marginal guard at the parent's paths for *path*, paths of the view.

    A path of the view starts with its component, and a tuple of paths selects
    several nodes, whose final segments must differ.
    """
    paths = (path,) if isinstance(path, str) else tuple(path)
    if not paths:
        return Feasibility(False, "no path was requested")
    parent_paths = []
    for each in paths:
        parent_path = self._parent_path(each)
        if parent_path is None:
            return Feasibility(False, f"{each!r} is not an event path of the view")
        parent_paths.append(parent_path)
    components = [_final_segment(each) for each in paths]
    if len(set(components)) < len(components):
        return Feasibility(False, f"the paths {list(paths)} share a final segment")
    return _capability_guard(
        self._parent, "_marginal", parent_paths[0] if isinstance(path, str) else tuple(parent_paths)
    )


def _view_condition_on(self: FieldView, given: Any, /, **kwargs: Any) -> Distribution:
    """Conditioning commutes with marginalization: the parent conditioned, then viewed.

    *given* is keyed by event paths of the view. The parent is conditioned at
    the parent's paths for them, and the result is the view of the conditioned
    parent at each of the view's nodes that keeps a field unconditioned.

    Raises
    ------
    KeyError
        If a key of *given* is not an event path of the view.
    ValueError
        If *given* covers every field of the view.
    """
    items = list(given.items())
    parent_paths = self._parent_paths([path for path, _ in items])
    kept = self._kept_components([path for path, _ in items])
    if not kept:
        raise ValueError(f"the given covers every field of {self.name!r}, so no law remains")
    conditioned = self._parent._condition_on(
        {parent_path: value for parent_path, (_, value) in zip(parent_paths, items)}, **kwargs
    )
    if isinstance(self._path, str):
        return _named_as(FieldView(conditioned, self._path), kept)
    nodes = self._component_paths()
    return _named_as(FieldView(conditioned, tuple(nodes[component] for component in kept)), kept)


def _view_condition_on_guard(self: FieldView, paths: tuple[str, ...]) -> Feasibility:
    """The parent's conditioning guard at the parent's paths for *paths*, paths of the view.

    Some field of the view must remain unconditioned.
    """
    parent_paths = []
    for path in paths:
        parent_path = self._parent_path(path)
        if parent_path is None:
            return Feasibility(False, f"{path!r} is not an event path of the view")
        parent_paths.append(parent_path)
    if not self._kept_components(paths):
        return Feasibility(False, f"the paths {list(paths)} cover every field of the view")
    return _capability_guard(self._parent, "_condition_on", tuple(parent_paths))


def _parent_guard(method: str) -> Callable[..., Feasibility]:
    """The guard of a view's *method*, which calls the parent's *method* and takes its guard."""

    def guard(self: FieldView, *arguments: Any, **keywords: Any) -> Feasibility:
        return _capability_guard(self._parent, method, *arguments, **keywords)

    guard.__name__ = f"{method}_guard"
    guard.__qualname__ = f"FieldView.{method}_guard"
    guard.__doc__ = f"The parent's guard of ``{method}``, for the same arguments."
    return guard


#: Each capability a view may derive, with the methods that realize it and their
#: guards. The density's protocol default ``_unnormalized_log_prob`` takes the
#: guard of ``_log_prob``.
_VIEW_CAPABILITIES: dict[type, Mapping[str, Callable[..., Any]]] = {
    SupportsSampling: {"_sample": _view_sample, "_sample_guard": _parent_guard("_sample")},
    SupportsMean: {"_mean": _view_mean, "_mean_guard": _parent_guard("_mean")},
    SupportsVariance: {"_variance": _view_variance, "_variance_guard": _parent_guard("_variance")},
    SupportsCovariance: {"_cov": _view_cov, "_cov_guard": _parent_guard("_cov")},
    SupportsQuantile: {"_quantile": _view_quantile, "_quantile_guard": _parent_guard("_quantile")},
    SupportsExpectation: {
        "_expectation": _view_expectation,
        "_expectation_guard": _parent_guard("_expectation"),
    },
    SupportsLogProb: {"_log_prob": _view_log_prob, "_log_prob_guard": _view_log_prob_guard},
    SupportsMarginals: {"_marginal": _view_marginal, "_marginal_guard": _view_marginal_guard},
    SupportsExactConditioning: {
        "_condition_on": _view_condition_on,
        "_condition_on_guard": _view_condition_on_guard,
    },
    SupportsApproximateConditioning: {
        "_condition_on": _view_condition_on,
        "_condition_on_guard": _view_condition_on_guard,
    },
}


def _derived_protocols(parent: Distribution, node: TermSpec) -> set[type]:
    """The capabilities a view of *parent* derives, over the event declared by *node*."""
    numeric = isinstance(node, NumericSpec)
    derived: set[type] = set()
    for protocol in (SupportsSampling, SupportsMean, SupportsVariance, SupportsExpectation):
        if isinstance(parent, protocol):
            derived.add(protocol)
    for protocol in (SupportsCovariance, SupportsQuantile):
        if isinstance(parent, protocol) and numeric:
            derived.add(protocol)
    if isinstance(parent, SupportsMarginals):
        derived |= {SupportsLogProb, SupportsMarginals}
    for conditioning in (SupportsExactConditioning, SupportsApproximateConditioning):
        if isinstance(parent, conditioning):
            derived.add(conditioning)
    return derived


# ---------------------------------------------------------------------------
# FieldView
# ---------------------------------------------------------------------------


class FieldView(Distribution):
    """The law of the field or field group at one event path of a parent law.

    ``d[path]`` returns a ``FieldView`` for a node below the law's whole term;
    it is never constructed by hand. The view holds a reference to its parent
    rather than a detached law, so sibling views co-sample from one parent
    draw and the correlation between them is preserved. Its declaration is the
    parent's schema at the path, the leaf or subtree whole, under a component
    named by the path's final segment, and its label is the path. A tuple of
    paths selects several nodes: the view declares an exposed record of them,
    in order, and is labeled by the paths joined with ``", "``.

    Its capabilities are derived from the parent's, one by one:

    ======================================  ========================================
    capability on the view                  available when
    ======================================  ========================================
    ``_sample``                             the parent samples
    ``_mean``, ``_variance``                the parent has the moment
    ``_cov``, ``_quantile``                 the parent has it and the node is numeric
    ``_expectation``                        the parent has it
    ``_log_prob``                           the parent has marginals; its guard asks
                                            that the marginal at the path be exact
    ``_marginal`` at a sub-path             the parent has marginals
    ``_condition_on`` a sub-field           the parent's conditioning capability
    ======================================  ========================================

    Each derived capability carries the parent's guard for the call it makes.
    The projection rows are exact whenever the parent's answer is, and only
    sampling requires the parent to sample.

    Parameters
    ----------
    parent : Distribution
        The law whose event the view reads.
    path : str or tuple of str
        An event path of *parent*, a slash path that starts with a component,
        or a tuple of them.

    Raises
    ------
    KeyError
        If a path is not an event path of *parent*, or the tuple is empty.
    TypeError
        If *parent* is not a ``Distribution``, or *path* is neither a string
        nor a tuple of strings.
    ValueError
        If two selected paths share their final segment.
    """

    _capability_table = _VIEW_CAPABILITIES

    def __new__(cls, parent: Distribution, path: str | tuple[str, ...]) -> FieldView:
        if not isinstance(parent, Distribution):
            raise TypeError(f"a field view reads a Distribution, got {type(parent).__name__}")
        declaration = _view_declaration(parent.event_spec, path)
        return object.__new__(
            _capability_subclass(FieldView, _derived_protocols(parent, declaration.spec))
        )

    def __init__(self, parent: Distribution, path: str | tuple[str, ...]) -> None:
        declaration = _view_declaration(parent.event_spec, path)
        self._init_tracked(path if isinstance(path, str) else ", ".join(path))
        self._init_annotations(None)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_path", path)
        self._init_declaration(declaration)
        self.with_provenance(
            Provenance.create("__getitem__", parents=[parent], metadata={"path": path})
        )

    @property
    def parent(self) -> Distribution:
        """The law this view reads, which sibling views share."""
        return self._parent

    @property
    def path(self) -> str | tuple[str, ...]:
        """The event path of the parent that this view reads, or the tuple of them."""
        return self._path

    def _component_paths(self) -> dict[str, str]:
        """Each component of the view, with the parent's path of its node."""
        if isinstance(self._path, str):
            return {_whole_term_component(self.event_spec): self._path}
        return dict(zip(self.event_spec.components, self._path))

    def _parent_path(self, path: Any) -> str | None:
        """The parent's path for *path*, an event path of this view, or None if it is not one.

        The view's paths start with a component, which stands for the node the
        view reads at that component.
        """
        if not isinstance(path, str):
            return None
        try:
            _node_at(self.event_spec, path)
        except KeyError:
            return None
        head, _, rest = path.partition(_PATH_SEP)
        node = self._component_paths()[head]
        return f"{node}{_PATH_SEP}{rest}" if rest else node

    def _parent_paths(self, paths: Sequence[str]) -> list[str]:
        """The parent's path for each of *paths*, event paths of this view.

        Raises
        ------
        KeyError
            If a path is not an event path of the view.
        """
        parent_paths = []
        for path in paths:
            parent_path = self._parent_path(path)
            if parent_path is None:
                raise KeyError(path)
            parent_paths.append(parent_path)
        return parent_paths

    def _kept_components(self, paths: Sequence[str]) -> list[str]:
        """The view's components with a field that no path in *paths* covers."""
        leaves = [
            leaf
            for leaf in _leaf_paths(self.event_spec)
            if not any(_is_within(leaf, path) for path in paths)
        ]
        return [
            component
            for component in self.event_spec.components
            if any(_is_within(leaf, component) for leaf in leaves)
        ]

    def _project(self, value: Any) -> Any:
        """pi: the view's node of *value*, a raw parent draw or a value shaped like one.

        A selection gives the record of its nodes, in the raw form of *value*.
        """
        declaration = self._parent.event_spec
        if isinstance(self._path, str):
            return _extract(value, _draw_segments(declaration, self._path))
        parts = [
            (component, _draw_segments(declaration, path))
            for component, path in self._component_paths().items()
        ]
        return _assemble(value, parts, self.name)

    def _coordinates(self) -> list[int]:
        """The view's coordinates of the parent's flat vector, in the view's order.

        Raises
        ------
        TypeError
            If the parent's declaration is not numeric.
        """
        return [
            index
            for path in self._component_paths().values()
            for index in _coordinate_range(self._parent.event_spec, path)
        ]

    def __getitem__(self, key: str | tuple[str, ...]) -> Distribution:
        """This view under its component, the parent's view at a path within it, or a selection.

        A path of the view starts with its component, so ``d["a"]["a/b/c"]`` is
        ``d["a/b/c"]``. A tuple of paths of the view selects several nodes, as
        the parent's view at the tuple of their paths, named by the view's paths.

        Raises
        ------
        KeyError
            If *key* is not an event path of the view, or a tuple of them.
        ValueError
            If two selected paths share their final segment.
        """
        if isinstance(key, str):
            parent_path = self._parent_path(key)
            if parent_path is None:
                raise KeyError(key)
            if parent_path == self._path:
                return self
            return FieldView(self._parent, parent_path)
        if not isinstance(key, tuple) or not key or not all(isinstance(each, str) for each in key):
            raise KeyError(key)
        selection = FieldView(self._parent, tuple(self._parent_paths(key)))
        return _named_as(selection, [_final_segment(each) for each in key])

    def __repr__(self) -> str:
        return f"FieldView(parent={self._parent.name!r}, path={self._path!r})"
