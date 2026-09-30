"""Field views and renamed laws: laws that read a parent distribution's values.

Provides:
  - ``FieldView`` – the ``Distribution`` that ``d[path]`` returns for a node
    below a law's whole term, or for a selection of several nodes, holding a
    reference to its parent.
  - the renamed law and the renamed kernel that ``with_path_names`` returns
    when the values must carry new names, which this module installs on the
    two distribution kinds at import.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from ..core._dispatch import Feasibility
from ..core._record_batch import RecordBatch
from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericSpec, TermSpec
from ..core._specs import InputSpec, OutputSpec
from ..core.provenance import Provenance
from ..core.record import Record
from ..linalg import DenseLinOp, LinOp
from ._capabilities import (
    SupportsApproximateConditioning,
    SupportsConditionalCovariance,
    SupportsConditionalExpectation,
    SupportsConditionalLogProb,
    SupportsConditionalMarginals,
    SupportsConditionalMean,
    SupportsConditionalQuantile,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsConditionalVariance,
    SupportsCovariance,
    SupportsExactConditioning,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
    _capability_guard,
    _capability_subclass,
    _claimed,
)
from ._conditional import (
    ConditionalDistribution,
    _given_leaf_specs,
    _install_renamed_kernel,
)
from ._distribution import Distribution, _install_renamed_law, _whole_term_component
from ._factored import SupportsFactors, _raw_record

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
    """The node at *segments* below the top of *value*, a raw value shaped like a draw.

    *value* is in raw form, so a value with paths is a nested mapping of raw
    leaves, which each segment indexes. With no segments *value* is returned
    whole.

    Raises
    ------
    TypeError
        If *segments* is not empty and *value* is not a mapping.
    """
    for segment in segments:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"the node at {_PATH_SEP.join(segments)!r} cannot be read from a "
                f"{type(value).__name__}; a raw value with paths is a mapping"
            )
        value = value[segment]
    return value


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


def _parent_guard(method: str, owner: str = "FieldView") -> Callable[..., Feasibility]:
    """The guard of *owner*'s *method*, which calls the parent's *method* and takes its guard."""

    def guard(self: Any, *arguments: Any, **keywords: Any) -> Feasibility:
        return _capability_guard(self._parent, method, *arguments, **keywords)

    guard.__name__ = f"{method}_guard"
    guard.__qualname__ = f"{owner}.{method}_guard"
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

        The node is in raw form: a leaf's raw value, or the nested mapping of a
        group's raw leaves, and a selection gives the mapping of its nodes keyed
        by component. Leading batch axes stay on every leaf. A ``Record`` or a
        batch of records is read through that raw form.
        """
        raw = _raw_record(value)
        declaration = self._parent.event_spec
        if isinstance(self._path, str):
            return _extract(raw, _draw_segments(declaration, self._path))
        return {
            component: _extract(raw, _draw_segments(declaration, path))
            for component, path in self._component_paths().items()
        }

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


# ---------------------------------------------------------------------------
# Renames of paths and of the values that carry them
# ---------------------------------------------------------------------------


def _has_path(declaration: OutputSpec, path: Any) -> bool:
    """Whether *path* is a path of *declaration*."""
    if not isinstance(path, str):
        return False
    try:
        _node_at(declaration, path)
    except KeyError:
        return False
    return True


def _renamed_path(path: str, renames: Mapping[str, str]) -> str:
    """*path* with each of its prefixes that *renames* keys renamed to the name given."""
    segments = path.split(_PATH_SEP)
    return _PATH_SEP.join(
        renames.get(_PATH_SEP.join(segments[: end + 1]), segment)
        for end, segment in enumerate(segments)
    )


def _inverse_renames(renames: Mapping[str, str]) -> dict[str, str]:
    """The original name of each node that *renames* renames, keyed by the node's new path."""
    return {_renamed_path(path, renames): _final_segment(path) for path in renames}


def _renames_below(renames: Mapping[str, str], path: str) -> dict[str, str]:
    """The entries of *renames* strictly below *path*, keyed relative to it."""
    prefix = path + _PATH_SEP
    return {key[len(prefix) :]: name for key, name in renames.items() if key.startswith(prefix)}


def _renamed_keys(
    value: Mapping[str, Any], renames: Mapping[str, str], prefix: tuple[str, ...]
) -> dict[str, Any]:
    """The mapping *value*, found at *prefix*, with its keys renamed by *renames*.

    A key may itself be a path, and a nested mapping is renamed in turn.
    """
    result: dict[str, Any] = {}
    for key, entry in value.items():
        path = (*prefix, *key.split(_PATH_SEP))
        new = [
            renames.get(_PATH_SEP.join(path[: end + 1]), segment)
            for end, segment in enumerate(path)
        ]
        result[_PATH_SEP.join(new[len(prefix) :])] = (
            _renamed_keys(entry, renames, path) if isinstance(entry, Mapping) else entry
        )
    return result


def _renamed_value(value: Any, renames: Mapping[str, str]) -> Any:
    """*value* with the node at each key of *renames* renamed in place.

    A record or a batch of records renames its fields and a mapping its keys;
    a value of any other kind has no paths and is returned as it is.
    """
    if not renames:
        return value
    if isinstance(value, (Record, RecordBatch)):
        return value.with_path_names(dict(renames))
    if isinstance(value, Mapping):
        return _renamed_keys(value, renames, ())
    return value


@dataclass(frozen=True)
class _EventRenames:
    """In-place renames of nodes of an event declaration, keyed by their exact paths.

    A draw of an exposed record carries the paths of its record, and a draw of
    a whole term the paths below its component, so renaming the component alone
    renames nothing a draw carries.

    Attributes
    ----------
    declaration : OutputSpec
        The original declaration.
    renames : Mapping[str, str]
        The new name of each renamed node, keyed by its original path.
    """

    declaration: OutputSpec
    renames: Mapping[str, str]

    @property
    def _draw_renames(self) -> dict[str, str]:
        component = _whole_term_component(self.declaration)
        if component is None:
            return dict(self.renames)
        return _renames_below(self.renames, component)

    def draw(self, value: Any) -> Any:
        """*value*, a raw value of the original declaration, under the new names."""
        return _renamed_value(value, self._draw_renames)

    def undraw(self, value: Any) -> Any:
        """*value*, a raw value of the renamed declaration, under the original names."""
        return _renamed_value(value, _inverse_renames(self._draw_renames))

    def original(self, path: str) -> str:
        """The original path of *path*, a path of the renamed declaration."""
        return _renamed_path(path, _inverse_renames(self.renames))

    def undraw_at(self, path: str, value: Any) -> Any:
        """*value*, the node at *path* of the renamed declaration, under the original names."""
        return _renamed_value(value, _renames_below(_inverse_renames(self.renames), path))

    def law(self, law: Distribution) -> Distribution:
        """*law*, over the original declaration or the part of it that remains, renamed.

        Each rename of a node that *law* still declares applies, so a law
        conditioned on some fields takes the renames of the fields that remain.
        """
        present = {
            path: name for path, name in self.renames.items() if _has_path(law.event_spec, path)
        }
        return law.with_path_names(present) if present else law

    def marginal(
        self, law: Distribution, paths: Sequence[str], originals: Sequence[str]
    ) -> Distribution:
        """*law*, the marginal at *originals*, under the names of *paths*, the renamed paths.

        The marginal names each node by the final segment of its original path,
        which takes the final segment of the renamed path and the renames below
        the node.
        """
        renames: dict[str, str] = {}
        for path, original in zip(paths, originals):
            component = _final_segment(original)
            if _final_segment(path) != component:
                renames[component] = _final_segment(path)
            renames.update(
                {
                    f"{component}{_PATH_SEP}{key}": name
                    for key, name in _renames_below(self.renames, original).items()
                }
            )
        return law.with_path_names(renames) if renames else law


# ---------------------------------------------------------------------------
# The renamed law
# ---------------------------------------------------------------------------


def _renamed_sample(
    self: _RenamedDistribution, key: PRNGKey, sample_shape: tuple[int, ...] = ()
) -> Any:
    """The parent's draw, with the key and sample shape unchanged, under the new names."""
    return self._event.draw(self._parent._sample(key, sample_shape))


def _renamed_mean(self: _RenamedDistribution) -> Any:
    """The parent's mean under the new names."""
    return self._event.draw(self._parent._mean())


def _renamed_variance(self: _RenamedDistribution) -> Any:
    """The parent's variance under the new names."""
    return self._event.draw(self._parent._variance())


def _renamed_cov(self: _RenamedDistribution) -> LinOp:
    """The parent's covariance, since an in-place rename keeps the flat coordinates' order."""
    return self._parent._cov()


def _renamed_quantile(self: _RenamedDistribution, q: ArrayLike) -> Array:
    """The parent's quantiles, since an in-place rename keeps the flat coordinates' order."""
    return self._parent._quantile(q)


def _renamed_expectation(self: _RenamedDistribution, f: Callable[[Any], Array]) -> Array:
    """The parent's exact expectation of ``f`` at the renamed value."""
    return self._parent._expectation(lambda value: f(self._event.draw(value)))


def _renamed_log_prob(self: _RenamedDistribution, value: Any) -> Array:
    """The parent's log-density of *value* under the original names."""
    return self._parent._log_prob(self._event.undraw(value))


def _renamed_unnormalized_log_prob(self: _RenamedDistribution, value: Any) -> Array:
    """The parent's unnormalized log-density of *value* under the original names."""
    return self._parent._unnormalized_log_prob(self._event.undraw(value))


def _renamed_marginal(self: _RenamedDistribution, path: str | tuple[str, ...]) -> Distribution:
    """The parent's marginal at the original paths for *path*, named by *path*.

    Raises
    ------
    KeyError
        If a path is not an event path of this law.
    """
    paths = (path,) if isinstance(path, str) else tuple(path)
    originals = self._originals(paths)
    marginal = self._parent._marginal(originals[0] if isinstance(path, str) else tuple(originals))
    return self._event.marginal(marginal, paths, originals)


def _renamed_marginal_guard(self: _RenamedDistribution, path: str | tuple[str, ...]) -> Feasibility:
    """The parent's marginal guard at the original paths for *path*, paths of this law."""
    paths = (path,) if isinstance(path, str) else tuple(path)
    for each in paths:
        if not _has_path(self.event_spec, each):
            return Feasibility(False, f"{each!r} is not an event path of {self.name!r}")
    originals = [self._event.original(each) for each in paths]
    return _capability_guard(
        self._parent, "_marginal", originals[0] if isinstance(path, str) else tuple(originals)
    )


def _renamed_condition_on(self: _RenamedDistribution, given: Any, /, **kwargs: Any) -> Distribution:
    """The parent conditioned on *given* under the original names, then renamed.

    Raises
    ------
    KeyError
        If a key of *given* is not an event path of this law.
    """
    items = list(given.items())
    originals = self._originals([path for path, _ in items])
    translated = {
        original: self._event.undraw_at(path, value)
        for original, (path, value) in zip(originals, items)
    }
    return self._event.law(self._parent._condition_on(translated, **kwargs))


def _renamed_condition_on_guard(self: _RenamedDistribution, paths: tuple[str, ...]) -> Feasibility:
    """The parent's conditioning guard at the original paths for *paths*, paths of this law."""
    for path in paths:
        if not _has_path(self.event_spec, path):
            return Feasibility(False, f"{path!r} is not an event path of {self.name!r}")
    return _capability_guard(
        self._parent, "_condition_on", tuple(self._event.original(path) for path in paths)
    )


_RENAMED = "_RenamedDistribution"

#: Each capability a renamed law takes from its parent, with its methods and guards.
_RENAMED_CAPABILITIES: dict[type, Mapping[str, Callable[..., Any]]] = {
    SupportsSampling: {
        "_sample": _renamed_sample,
        "_sample_guard": _parent_guard("_sample", _RENAMED),
    },
    SupportsMean: {"_mean": _renamed_mean, "_mean_guard": _parent_guard("_mean", _RENAMED)},
    SupportsVariance: {
        "_variance": _renamed_variance,
        "_variance_guard": _parent_guard("_variance", _RENAMED),
    },
    SupportsCovariance: {"_cov": _renamed_cov, "_cov_guard": _parent_guard("_cov", _RENAMED)},
    SupportsQuantile: {
        "_quantile": _renamed_quantile,
        "_quantile_guard": _parent_guard("_quantile", _RENAMED),
    },
    SupportsExpectation: {
        "_expectation": _renamed_expectation,
        "_expectation_guard": _parent_guard("_expectation", _RENAMED),
    },
    SupportsLogProb: {
        "_log_prob": _renamed_log_prob,
        "_log_prob_guard": _parent_guard("_log_prob", _RENAMED),
    },
    SupportsUnnormalizedLogProb: {
        "_unnormalized_log_prob": _renamed_unnormalized_log_prob,
        "_unnormalized_log_prob_guard": _parent_guard("_unnormalized_log_prob", _RENAMED),
    },
    SupportsMarginals: {"_marginal": _renamed_marginal, "_marginal_guard": _renamed_marginal_guard},
    SupportsExactConditioning: {
        "_condition_on": _renamed_condition_on,
        "_condition_on_guard": _renamed_condition_on_guard,
    },
    SupportsApproximateConditioning: {
        "_condition_on": _renamed_condition_on,
        "_condition_on_guard": _renamed_condition_on_guard,
    },
}


class _RenamedDistribution(Distribution):
    """A parent law under renamed event paths, which renames values at its boundary.

    ``Distribution.with_path_names`` returns it for a rename that reaches a field
    of a record draw, whose values must then carry the new names. It claims each
    of the parent's capabilities that a rename carries over: a draw, a moment,
    or a marginal is renamed on the way out, and a scored value, a given, or a
    path on the way in. The renames are in place, so the flat coordinates keep
    their order and the covariance and quantiles are the parent's. Each
    capability carries the parent's guard at the original paths.

    Parameters
    ----------
    parent : Distribution
        The law whose values are renamed.
    event_spec : OutputSpec
        The renamed declaration.
    renames : Mapping[str, str]
        The new name of each renamed node, keyed by its path in the parent's
        declaration.
    """

    _capability_table = _RENAMED_CAPABILITIES

    def __new__(
        cls, parent: Distribution, event_spec: OutputSpec, renames: Mapping[str, str]
    ) -> _RenamedDistribution:
        return object.__new__(
            _capability_subclass(_RenamedDistribution, _claimed(parent, _RENAMED_CAPABILITIES))
        )

    def __init__(
        self, parent: Distribution, event_spec: OutputSpec, renames: Mapping[str, str]
    ) -> None:
        self._init_tracked(parent.name)
        self._init_annotations(None)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_event", _EventRenames(parent.event_spec, dict(renames)))
        self._init_declaration(event_spec)
        self.with_provenance(
            Provenance.create("with_path_names", parents=[parent], metadata=dict(renames))
        )

    def _originals(self, paths: Sequence[str]) -> list[str]:
        """The parent's path for each of *paths*, event paths of this law.

        Raises
        ------
        KeyError
            If a path is not an event path of this law.
        """
        for path in paths:
            if not _has_path(self.event_spec, path):
                raise KeyError(path)
        return [self._event.original(path) for path in paths]

    def __repr__(self) -> str:
        return f"_RenamedDistribution(name={self.name!r}, parent={self._parent.name!r})"


def _renamed_law(
    parent: Distribution, event_spec: OutputSpec, renames: Mapping[str, str]
) -> Distribution:
    """The law ``with_path_names`` returns for *parent* when a rename reaches a record field.

    Raises
    ------
    NotImplementedError
        If *parent* is factored, since a joint renames through its factors.
    """
    if isinstance(parent, SupportsFactors):
        raise NotImplementedError("FactoredDistribution.with_path_names")
    return _RenamedDistribution(parent, event_spec, renames)


# ---------------------------------------------------------------------------
# The renamed kernel
# ---------------------------------------------------------------------------


def _leaf_values(given_spec: InputSpec, given: Any) -> dict[str, Any]:
    """The value *given* binds at each leaf of the given side, keyed by the leaf's path.

    A key of *given* is a slot or a path into a structured slot, and the value
    of a structured node is a record or a mapping of its fields.

    Raises
    ------
    KeyError
        If a key is not a path of the given side.
    TypeError
        If the value of a structured node is neither a record nor a mapping.
    """
    values: dict[str, Any] = {}

    def visit(path: str, value: Any) -> None:
        head, _, rest = path.partition(_PATH_SEP)
        spec = given_spec[head]
        if rest:
            if not isinstance(spec, RecordSpec):
                raise KeyError(path)
            spec = spec.at_path(rest)
        if not isinstance(spec, RecordSpec):
            values[path] = value
            return
        if not isinstance(value, (Record, Mapping)):
            raise TypeError(
                f"the value of the structured node {path!r} is a record or a mapping of its "
                f"fields, got {type(value).__name__}"
            )
        for key, entry in value.items():
            visit(f"{path}{_PATH_SEP}{key}", entry)

    for key, value in given.items():
        visit(key, value)
    return values


def _slot_value(slot: str, spec: TermSpec, leaves: Mapping[str, Any]) -> Any:
    """The value of the slot *slot*, declared by *spec*, from the values at its leaves."""
    if not isinstance(spec, RecordSpec):
        return leaves[slot]
    return Record(slot, {leaf[len(slot) + 1 :]: value for leaf, value in leaves.items()})


def _renamed_conditional_sample(
    self: _RenamedConditionalDistribution,
    given: Any,
    key: PRNGKey,
    sample_shape: tuple[int, ...] = (),
) -> Any:
    """The parent's draw at the translated given, under the new names."""
    return self._event.draw(
        self._parent._conditional_sample(self._parent_given(given), key, sample_shape)
    )


def _renamed_conditional_log_prob(
    self: _RenamedConditionalDistribution, given: Any, value: Any
) -> Array:
    """The parent's log-density at the translated given of *value* under the original names."""
    return self._parent._conditional_log_prob(self._parent_given(given), self._event.undraw(value))


def _renamed_conditional_unnormalized_log_prob(
    self: _RenamedConditionalDistribution, given: Any, value: Any
) -> Array:
    """The parent's unnormalized log-density at the translated given, as for the density."""
    return self._parent._conditional_unnormalized_log_prob(
        self._parent_given(given), self._event.undraw(value)
    )


def _renamed_conditional_mean(self: _RenamedConditionalDistribution, given: Any) -> Any:
    """The parent's mean at the translated given, under the new names."""
    return self._event.draw(self._parent._conditional_mean(self._parent_given(given)))


def _renamed_conditional_variance(self: _RenamedConditionalDistribution, given: Any) -> Any:
    """The parent's variance at the translated given, under the new names."""
    return self._event.draw(self._parent._conditional_variance(self._parent_given(given)))


def _renamed_conditional_cov(self: _RenamedConditionalDistribution, given: Any) -> LinOp:
    """The parent's covariance at the translated given."""
    return self._parent._conditional_cov(self._parent_given(given))


def _renamed_conditional_quantile(
    self: _RenamedConditionalDistribution, given: Any, q: ArrayLike
) -> Array:
    """The parent's quantiles at the translated given."""
    return self._parent._conditional_quantile(self._parent_given(given), q)


def _renamed_conditional_expectation(
    self: _RenamedConditionalDistribution, given: Any, f: Callable[[Any], Array]
) -> Array:
    """The parent's exact expectation at the translated given of ``f`` at the renamed value."""
    return self._parent._conditional_expectation(
        self._parent_given(given), lambda value: f(self._event.draw(value))
    )


def _renamed_conditional_marginal(
    self: _RenamedConditionalDistribution, given: Any, path: str | tuple[str, ...]
) -> Distribution:
    """The parent's marginal at the translated given and the original paths, named by *path*.

    Raises
    ------
    KeyError
        If a path is not an event path of this kernel.
    """
    paths = (path,) if isinstance(path, str) else tuple(path)
    for each in paths:
        if not _has_path(self.event_spec, each):
            raise KeyError(each)
    originals = [self._event.original(each) for each in paths]
    marginal = self._parent._conditional_marginal(
        self._parent_given(given), originals[0] if isinstance(path, str) else tuple(originals)
    )
    return self._event.marginal(marginal, paths, originals)


def _renamed_conditional_marginal_guard(
    self: _RenamedConditionalDistribution, path: str | tuple[str, ...]
) -> Feasibility:
    """The parent's conditional marginal guard at the original paths for *path*."""
    paths = (path,) if isinstance(path, str) else tuple(path)
    for each in paths:
        if not _has_path(self.event_spec, each):
            return Feasibility(False, f"{each!r} is not an event path of {self.name!r}")
    originals = [self._event.original(each) for each in paths]
    return _capability_guard(
        self._parent,
        "_conditional_marginal",
        originals[0] if isinstance(path, str) else tuple(originals),
    )


_RENAMED_KERNEL = "_RenamedConditionalDistribution"


def _forwarded(method: str, body: Callable[..., Any]) -> dict[str, Callable[..., Any]]:
    """The entry of a kernel capability whose guard is the parent's for the same arguments."""
    return {method: body, f"{method}_guard": _parent_guard(method, _RENAMED_KERNEL)}


#: Each conditional capability a renamed kernel takes from its parent, with its
#: methods and guards; the conditioning capabilities add nothing to the kernel's
#: own ``_condition_on``.
_RENAMED_KERNEL_CAPABILITIES: dict[type, Mapping[str, Callable[..., Any]]] = {
    SupportsConditionalSampling: _forwarded("_conditional_sample", _renamed_conditional_sample),
    SupportsConditionalLogProb: _forwarded("_conditional_log_prob", _renamed_conditional_log_prob),
    SupportsConditionalUnnormalizedLogProb: _forwarded(
        "_conditional_unnormalized_log_prob", _renamed_conditional_unnormalized_log_prob
    ),
    SupportsConditionalMean: _forwarded("_conditional_mean", _renamed_conditional_mean),
    SupportsConditionalVariance: _forwarded("_conditional_variance", _renamed_conditional_variance),
    SupportsConditionalCovariance: _forwarded("_conditional_cov", _renamed_conditional_cov),
    SupportsConditionalQuantile: _forwarded("_conditional_quantile", _renamed_conditional_quantile),
    SupportsConditionalExpectation: _forwarded(
        "_conditional_expectation", _renamed_conditional_expectation
    ),
    SupportsConditionalMarginals: {
        "_conditional_marginal": _renamed_conditional_marginal,
        "_conditional_marginal_guard": _renamed_conditional_marginal_guard,
    },
    SupportsExactConditioning: {},
    SupportsApproximateConditioning: {},
}


class _RenamedConditionalDistribution(ConditionalDistribution):
    """A parent kernel under renamed given slots and event paths, renaming at its boundary.

    ``ConditionalDistribution.with_path_names`` returns it. Binding translates a
    given value to the parent's slots, moving the fields that a split or a
    grouping moved, and renames the law or the curried kernel the parent
    returns. A binding that completes only part of a parent slot, as binding
    one of the slots a split made does, holds the bound fields until the rest
    of the slot is bound, so every slot of this kernel binds on its own. Each
    conditional capability of the parent is claimed, translated the same way,
    and carries the parent's guard.

    Parameters
    ----------
    parent : ConditionalDistribution
        The kernel whose values are renamed.
    given_spec : InputSpec
        The renamed slots.
    event_spec : OutputSpec
        The renamed event declaration.
    origins : Mapping[str, str]
        The parent's leaf for each leaf of the renamed slots, keyed by its path.
    renames : Mapping[str, str]
        The new name of each renamed event node, keyed by its path in the
        parent's event declaration.
    pending : Mapping[str, Any], optional
        The values already bound at leaves of parent slots that are not yet
        complete, keyed by the parent's leaf paths.
    """

    _capability_table = _RENAMED_KERNEL_CAPABILITIES

    def __new__(
        cls,
        parent: ConditionalDistribution,
        given_spec: InputSpec,
        event_spec: OutputSpec,
        origins: Mapping[str, str],
        renames: Mapping[str, str],
        *,
        pending: Mapping[str, Any] | None = None,
    ) -> _RenamedConditionalDistribution:
        return object.__new__(
            _capability_subclass(
                _RenamedConditionalDistribution, _claimed(parent, _RENAMED_KERNEL_CAPABILITIES)
            )
        )

    def __init__(
        self,
        parent: ConditionalDistribution,
        given_spec: InputSpec,
        event_spec: OutputSpec,
        origins: Mapping[str, str],
        renames: Mapping[str, str],
        *,
        pending: Mapping[str, Any] | None = None,
    ) -> None:
        self._init_tracked(parent.name)
        self._init_annotations(None)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_origins", dict(origins))
        object.__setattr__(self, "_pending", dict(pending or {}))
        object.__setattr__(self, "_event", _EventRenames(parent.event_spec, dict(renames)))
        self._init_declaration(given_spec, event_spec)

    def _translated(self, given: Any) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
        """The parent slots *given* completes, the parent leaves left pending, and the slots bound.

        Raises
        ------
        KeyError
            If a key of *given* is not a path of the given side.
        ValueError
            If *given* binds part of a slot of this kernel.
        """
        values = _leaf_values(self.given_spec, given)
        slots = {path.partition(_PATH_SEP)[0] for path in values}
        missing = [
            path
            for path in _given_leaf_specs(self.given_spec)
            if path.partition(_PATH_SEP)[0] in slots and path not in values
        ]
        if missing:
            raise ValueError(f"the given binds part of a slot of {self.name!r}, without {missing}")
        bound = {**self._pending, **{self._origins[path]: value for path, value in values.items()}}
        by_slot: dict[str, list[str]] = {}
        for leaf in _given_leaf_specs(self._parent.given_spec):
            by_slot.setdefault(leaf.partition(_PATH_SEP)[0], []).append(leaf)
        complete = {
            slot: _slot_value(
                slot, self._parent.given_spec[slot], {leaf: bound[leaf] for leaf in leaves}
            )
            for slot, leaves in by_slot.items()
            if all(leaf in bound for leaf in leaves)
        }
        pending = {
            leaf: value
            for leaf, value in bound.items()
            if leaf.partition(_PATH_SEP)[0] not in complete
        }
        return complete, pending, slots

    def _parent_given(self, given: Any) -> dict[str, Any]:
        """*given*, a value for every slot of this kernel, translated to the parent's slots.

        Raises
        ------
        ValueError
            If *given* leaves a slot of this kernel unbound.
        """
        complete, pending, _ = self._translated(given)
        if pending or len(complete) != len(self._parent.given_spec):
            raise ValueError(f"{self.name!r} needs a value for every slot {list(self.given_spec)}")
        return complete

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The parent bound at *given*, translated to its slots, under the new names.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values for some or all of this kernel's slots, by slot name.
        **kwargs : Any
            Options for the parent's primitive.

        Returns
        -------
        Distribution or ConditionalDistribution
            The renamed law when every slot is bound, and otherwise the renamed
            kernel over the remaining slots.

        Raises
        ------
        KeyError
            If a key of *given* is not a path of the given side.
        ValueError
            If *given* binds part of a slot.
        """
        complete, pending, slots = self._translated(given)
        result = self._parent._condition_on(complete, **kwargs) if complete else self._parent
        if not isinstance(result, ConditionalDistribution):
            return self._event.law(result)
        remaining = InputSpec(
            {slot: spec for slot, spec in self.given_spec.items() if slot not in slots}
        )
        origins = {
            path: leaf
            for path, leaf in self._origins.items()
            if path.partition(_PATH_SEP)[0] not in slots
        }
        return _RenamedConditionalDistribution(
            result, remaining, self.event_spec, origins, self._event.renames, pending=pending
        )


def _renamed_kernel(
    parent: ConditionalDistribution,
    given_spec: InputSpec,
    event_spec: OutputSpec,
    origins: Mapping[str, str],
    renames: Mapping[str, str],
    pairs: Mapping[str, str],
) -> ConditionalDistribution:
    """The kernel ``with_path_names`` returns for *parent*, with the rename's provenance.

    Raises
    ------
    NotImplementedError
        If *parent* is factored, since a joint renames through its factors.
    """
    if isinstance(parent, SupportsFactors):
        raise NotImplementedError("FactoredConditionalDistribution.with_path_names")
    kernel = _RenamedConditionalDistribution(parent, given_spec, event_spec, origins, renames)
    kernel.with_provenance(
        Provenance.create("with_path_names", parents=[parent], metadata=dict(pairs))
    )
    return kernel


_install_renamed_law(_renamed_law)
_install_renamed_kernel(_renamed_kernel)
