"""Field views and renamed laws: laws that read a parent distribution's values.

Provides:
  - ``FieldView`` – the ``Distribution`` that ``d[path]`` returns for a node
    below a law's whole term, or for a selection of several nodes, holding a
    reference to its parent.
  - the renamed law and the renamed kernel that ``with_path_names`` returns
    when the values must carry new names and the family does not rebuild
    itself under them, which this module installs on the two distribution
    kinds at import.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from ..core._dispatch import Feasibility, ResolutionError
from ..core._record_batch import RecordBatch
from ..core._record_spec import RecordSpec
from ..core._repr import WIDTH, call_repr, mapping_repr, term_repr
from ..core._spec_base import NumericSpec, TermSpec
from ..core._specs import InputSpec, OutputSpec, _components_record
from ..core.named_tree import _unflatten_paths
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
    _marginal_claims,
)
from ._conditional import (
    ConditionalDistribution,
    _given_leaf_specs,
    _install_renamed_kernel,
)
from ._distribution import (
    _RENAME_SOURCE,
    Distribution,
    _install_field_view,
    _install_renamed_law,
    _whole_term_component,
)
from ._factored import (
    FactoredConditionalDistribution,
    FactoredDistribution,
    SupportsFactors,
    _factor_graph,
    _FactorGraph,
    _joined_label,
    _raw_record,
)

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
        If a path is not a path of *declaration*.
    ValueError
        If the tuple is empty, since an empty selection is malformed, or two
        selected paths share their final segment, so their components would
        collide.
    """
    if isinstance(path, str):
        return OutputSpec(**{_final_segment(path): _node_at(declaration, path)})
    if not isinstance(path, tuple) or not all(isinstance(each, str) for each in path):
        raise TypeError(f"a view's path is a string or a tuple of strings, got {path!r}")
    if not path:
        raise ValueError("a selection of event paths names at least one path")
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


def _projector(declaration: OutputSpec, path: str | tuple[str, ...]) -> Callable[[Any], Any]:
    """pi for a law declared by *declaration*: the extraction of its node at *path*.

    A tuple of paths extracts the record of the selected nodes, keyed by
    component. The paths are read now, so the extraction does not follow a later
    change to whatever supplied them. A ``Record`` holding draws, as the lift
    holds a record law's draws, gives its node at the path, or a ``Record`` of
    the selected nodes, and any other value is read in its raw form and gives
    the raw node, or the mapping of the selected nodes. Leading batch axes stay
    on every leaf.
    """
    single = isinstance(path, str)
    paths = (path,) if single else tuple(path)
    segments = [_draw_segments(declaration, each) for each in paths]
    components = [_final_segment(each) for each in paths]

    def project(value: Any) -> Any:
        if isinstance(value, Record):
            nodes = [value.at_path(*each) if each else value for each in segments]
            if single:
                return nodes[0]
            return Record(
                value.label,
                {component: _raw_record(node) for component, node in zip(components, nodes)},
            )
        raw = _raw_record(value)
        nodes = [_extract(raw, each) for each in segments]
        return nodes[0] if single else dict(zip(components, nodes))

    return project


def _detached(law: Distribution, name: str) -> Distribution:
    """*law* detached from the workflow under *name*: no provenance and no annotations."""
    clone = law._shallow_copy()
    object.__setattr__(clone, "_label", name)
    object.__setattr__(clone, "_provenance", None)
    object.__setattr__(clone, "_annotations", None)
    return clone


def _labeled(law: Distribution, name: str) -> Distribution:
    """*law* under the label *name*, which a marginal takes from the law it is a marginal of."""
    return law if law.label == name else law.with_label(name)


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

#: The method of each moment row. A view projects the moment of a parent that
#: claims it, and otherwise computes it from the parent's exact marginal at the
#: view's path, when the parent reports that marginal claims it.
_MOMENT_METHODS: dict[type, str] = {
    SupportsMean: "_mean",
    SupportsVariance: "_variance",
    SupportsCovariance: "_cov",
    SupportsQuantile: "_quantile",
}

#: The moment rows that need a numeric node.
_NUMERIC_MOMENTS = (SupportsCovariance, SupportsQuantile)


def _marginal_moment(self: FieldView, method: str, *arguments: Any) -> Any:
    """The moment *method* of the parent's exact marginal at the view's path."""
    return getattr(self._parent._marginal(self._path), method)(*arguments)


def _view_sample(self: FieldView, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Any:
    """Co-sample: draw ``X`` from the parent with the same key and return ``pi(X)``.

    The key and the sample shape pass to the parent unchanged, so views of one
    parent drawn with one key project one parent draw.
    """
    return self._project(self._parent._sample(key, sample_shape))


def _view_mean(self: FieldView) -> Any:
    """Projection: the parent's mean at the view's path, since ``E[pi X] = pi E[X]``.

    A parent without a mean gives the mean of its exact marginal at the path.
    """
    if not isinstance(self._parent, SupportsMean):
        return _marginal_moment(self, "_mean")
    return self._project(self._parent._mean())


def _view_variance(self: FieldView) -> Any:
    """Restriction of the parent's variance to the coordinates of the view's path.

    A parent without a variance gives the variance of its exact marginal at the
    path.
    """
    if not isinstance(self._parent, SupportsVariance):
        return _marginal_moment(self, "_variance")
    return self._project(self._parent._variance())


def _view_cov(self: FieldView) -> LinOp:
    """The sub-block ``P Σ Pᵀ`` of the parent's covariance, ``P`` selecting the path.

    ``P`` selects the view's coordinates of the parent's flat vector, in the
    parent's order, and the product stays lazy. A parent without a covariance
    gives the covariance of its exact marginal at the path.

    Raises
    ------
    TypeError
        If the parent's declaration is not numeric.
    """
    if not isinstance(self._parent, SupportsCovariance):
        return _marginal_moment(self, "_cov")
    cov = self._parent._cov()
    rows = jnp.asarray(self._coordinates())
    selection = DenseLinOp(jnp.eye(cov.shape[1], dtype=cov.dtype)[rows])
    return selection @ cov @ selection.T


def _view_quantile(self: FieldView, q: ArrayLike) -> Any:
    """Restriction of the parent's per-coordinate quantiles to the view's node.

    The parent's quantiles are its event's raw form with the level axes
    leading in each leaf, so the view's quantiles are the parent's at the
    view's node, projected as a draw is, with the level axes leading in each
    leaf. A parent without quantiles gives those of its exact marginal at the
    path.
    """
    if not isinstance(self._parent, SupportsQuantile):
        return _marginal_moment(self, "_quantile", q)
    return self._project(self._parent._quantile(q))


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
            f"the marginal of {self._parent.label!r} at {self._path!r} has no normalized density"
        )
    return marginal._log_prob(value)


def _view_unnormalized_log_prob(self: FieldView, value: Any) -> Array:
    """The unnormalized log-density of the parent's detached marginal at the path.

    Raises
    ------
    TypeError
        If that marginal has no density.
    """
    marginal = self._parent._marginal(self._path)
    if not isinstance(marginal, SupportsUnnormalizedLogProb):
        raise TypeError(f"the marginal of {self._parent.label!r} at {self._path!r} has no density")
    return marginal._unnormalized_log_prob(value)


def _view_log_prob_guard(self: FieldView) -> Feasibility:
    """The parent's marginal guard at the view's path.

    The parent's report of that marginal's capabilities, read when the view was
    constructed, decides whether the view claims the density.
    """
    return _capability_guard(self._parent, "_marginal", self._path)


def _view_marginal(self: FieldView, path: str | tuple[str, ...]) -> Distribution:
    """Path composition: the parent's marginal at the parent's paths for *path*.

    *path* is an event path of the view, or a tuple of them, and the result's
    components are named by the paths of the view. The result keeps the view's
    label.

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
    return _labeled(_named_as(marginal, [_final_segment(each) for each in paths]), self.label)


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


def _view_marginal_capabilities(self: FieldView, path: str | tuple[str, ...]) -> frozenset[type]:
    """The parent's report of its marginal at the parent's paths for *path*, paths of the view.

    Raises
    ------
    KeyError
        If a path is not an event path of the view.
    """
    paths = (path,) if isinstance(path, str) else tuple(path)
    parent_paths = self._parent_paths(paths)
    return _marginal_claims(
        self._parent, parent_paths[0] if isinstance(path, str) else tuple(parent_paths)
    )


def _view_condition_on(self: FieldView, given: Any, /, **options: Any) -> Distribution:
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
        raise ValueError(f"the given covers every field of {self.label!r}, so no law remains")
    conditioned = self._parent._condition_on(
        {parent_path: value for parent_path, (_, value) in zip(parent_paths, items)}, **options
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


def _moment_guard(protocol: type) -> Callable[..., Feasibility]:
    """The guard of the moment row of *protocol*, for the source that computes the moment.

    A projected moment carries the parent's guard of its method. A moment from
    the marginal needs the parent's marginal guard at the path to accept, and
    then carries the marginal's guard of its method.
    """
    method = _MOMENT_METHODS[protocol]

    def guard(self: FieldView, *arguments: Any) -> Feasibility:
        if isinstance(self._parent, protocol):
            return _capability_guard(self._parent, method, *arguments)
        exact = _capability_guard(self._parent, "_marginal", self._path)
        if exact.feasible is not True:
            return exact
        marginal = self._parent._marginal(self._path)
        if not isinstance(marginal, protocol):
            return Feasibility(
                False,
                f"the marginal of {self._parent.label!r} at {self._path!r} claims no "
                f"{protocol.__name__}",
            )
        return _capability_guard(marginal, method, *arguments)

    guard.__name__ = f"{method}_guard"
    guard.__qualname__ = f"FieldView.{method}_guard"
    guard.__doc__ = (
        f"The parent's guard of ``{method}``, or the parent's marginal guard at the path "
        f"followed by that marginal's guard of ``{method}``."
    )
    return guard


#: Each capability a view may derive, with the methods that realize it and their
#: guards. The density's protocol default ``_unnormalized_log_prob`` takes the
#: guard of ``_log_prob``.
_VIEW_CAPABILITIES: dict[type, Mapping[str, Callable[..., Any]]] = {
    SupportsSampling: {"_sample": _view_sample, "_sample_guard": _parent_guard("_sample")},
    SupportsMean: {"_mean": _view_mean, "_mean_guard": _moment_guard(SupportsMean)},
    SupportsVariance: {
        "_variance": _view_variance,
        "_variance_guard": _moment_guard(SupportsVariance),
    },
    SupportsCovariance: {"_cov": _view_cov, "_cov_guard": _moment_guard(SupportsCovariance)},
    SupportsQuantile: {
        "_quantile": _view_quantile,
        "_quantile_guard": _moment_guard(SupportsQuantile),
    },
    SupportsExpectation: {
        "_expectation": _view_expectation,
        "_expectation_guard": _parent_guard("_expectation"),
    },
    SupportsLogProb: {"_log_prob": _view_log_prob, "_log_prob_guard": _view_log_prob_guard},
    SupportsUnnormalizedLogProb: {
        "_unnormalized_log_prob": _view_unnormalized_log_prob,
        "_unnormalized_log_prob_guard": _view_log_prob_guard,
    },
    SupportsMarginals: {
        "_marginal": _view_marginal,
        "_marginal_guard": _view_marginal_guard,
        "_marginal_capabilities": _view_marginal_capabilities,
    },
    SupportsExactConditioning: {
        "_condition_on": _view_condition_on,
        "_condition_on_guard": _view_condition_on_guard,
    },
    SupportsApproximateConditioning: {
        "_condition_on": _view_condition_on,
        "_condition_on_guard": _view_condition_on_guard,
    },
}


def _derived_protocols(
    parent: Distribution, node: TermSpec, path: str | tuple[str, ...]
) -> set[type]:
    """The capabilities a view of *parent* at *path* derives, over the event declared by *node*.

    The projection rows derive from the parent's own capabilities. The density
    rows, and the moment rows the parent does not claim, derive from the
    parent's report of its marginal at *path*, read once, here: the view claims
    the normalized density when the report includes it, and otherwise the
    unnormalized one when the report includes that, and it claims each moment
    the report includes. The covariance and quantile rows need a numeric node
    from either source.
    """
    numeric = isinstance(node, NumericSpec)
    derived: set[type] = set()
    for protocol in (SupportsSampling, SupportsExpectation):
        if isinstance(parent, protocol):
            derived.add(protocol)
    moments = {
        protocol for protocol in _MOMENT_METHODS if numeric or protocol not in _NUMERIC_MOMENTS
    }
    derived.update(protocol for protocol in moments if isinstance(parent, protocol))
    if isinstance(parent, SupportsMarginals):
        derived.add(SupportsMarginals)
        report = _marginal_claims(parent, path)
        if SupportsLogProb in report:
            derived.add(SupportsLogProb)
        elif SupportsUnnormalizedLogProb in report:
            derived.add(SupportsUnnormalizedLogProb)
        derived.update(protocol for protocol in moments if protocol in report)
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
    named by the path's final segment, and it keeps its parent's label. A
    tuple of paths selects several nodes: the view declares an exposed record
    of them, in order, and keeps its parent's label as well.

    Its capabilities are derived from the parent's, one by one:

    ======================================  ========================================
    capability on the view                  available when
    ======================================  ========================================
    ``_sample``                             the parent samples
    ``_mean``, ``_variance``                the parent has the moment, or has marginals
                                            and reports it for its marginal at the path
    ``_cov``, ``_quantile``                 as for the mean, and the node is numeric
    ``_expectation``                        the parent has it
    ``_log_prob``, ``_unnormalized_log_prob``  the parent has marginals and reports the
                                            density for its marginal at the path; the
                                            guard asks that marginal be exact
    ``_marginal`` at a sub-path             the parent has marginals
    ``_condition_on`` a sub-field           the parent's conditioning capability
    ======================================  ========================================

    Each derived capability carries the parent's guard for the call it makes.
    A moment the parent does not claim is the moment of the parent's exact
    marginal at the path, under the parent's marginal guard and then the
    marginal's guard of the moment, so a view of a dependent joint at a root
    factor takes the factor's moments. The projection rows are exact whenever
    the parent's answer is, and only sampling requires the parent to sample.
    The parent reports what its marginal at the path claims through
    ``_marginal_capabilities``, and a parent that defines none reports its own
    claims; the view reads the report once, at construction.

    The view's ``raw()`` is the parent's detached marginal at the path, and
    ``with_dim_sizes`` and ``with_dim_names`` apply to the parent and return
    the view of the result at the same path.

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
        If a path is not an event path of *parent*.
    TypeError
        If *parent* is not a ``Distribution``, or *path* is neither a string
        nor a tuple of strings.
    ValueError
        If the tuple is empty, or two selected paths share their final segment.
    """

    _capability_table = _VIEW_CAPABILITIES

    def __new__(cls, parent: Distribution, path: str | tuple[str, ...]) -> FieldView:
        if not isinstance(parent, Distribution):
            raise TypeError(f"a field view reads a Distribution, got {type(parent).__name__}")
        declaration = _view_declaration(parent.event_spec, path)
        return object.__new__(
            _capability_subclass(FieldView, _derived_protocols(parent, declaration.spec, path))
        )

    def __init__(self, parent: Distribution, path: str | tuple[str, ...]) -> None:
        declaration = _view_declaration(parent.event_spec, path)
        self._init_tracked(parent.label)
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
        return _projector(self._parent.event_spec, self._path)(_raw_record(value))

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
            If *key* is an empty tuple, or two selected paths share their final
            segment.
        """
        if isinstance(key, str):
            parent_path = self._parent_path(key)
            if parent_path is None:
                raise KeyError(key)
            if parent_path == self._path:
                return self
            return FieldView(self._parent, parent_path)
        if not isinstance(key, tuple) or not all(isinstance(each, str) for each in key):
            raise KeyError(key)
        if not key:
            raise ValueError("a selection of event paths names at least one path")
        selection = FieldView(self._parent, tuple(self._parent_paths(key)))
        return _named_as(selection, [_final_segment(each) for each in key])

    def raw(self) -> Distribution:
        """The parent's detached marginal at the path, under the view's name.

        Returns
        -------
        Distribution
            A standalone law with no reference to the parent, and no provenance
            or annotations.

        Raises
        ------
        ResolutionError
            If the parent has no exact marginal at the path.
        """
        parent = self._parent
        if not isinstance(parent, SupportsMarginals):
            raise ResolutionError(
                f"{parent.label!r} has no marginals, so the view {self.label!r} has no detached law"
            )
        report = _capability_guard(parent, "_marginal", self._path)
        if report.feasible is not True:
            reason = report.description or "; ".join(report.pending)
            raise ResolutionError(
                f"{parent.label!r} has no exact marginal at {self._path!r}: {reason}"
            )
        return _detached(parent._marginal(self._path), self.label)

    def with_dim_sizes(self, **sizes: int) -> FieldView:
        """Bind named symbolic dimensions in the parent, and view the result at the same path.

        The view's declaration is the parent's schema at the path, and a schema
        is one dimension scope, so a dimension binds wherever the parent
        declares it.

        Parameters
        ----------
        **sizes : int
            Sizes for free dimensions of the view's declaration.

        Returns
        -------
        FieldView
            The view, under the same name, of the parent with the sizes bound.

        Raises
        ------
        ValueError
            If a name is not a free dimension of the view's declaration.
        """
        unbound = set(sizes) - self.event_spec.spec.free_dims
        if unbound:
            raise ValueError(
                f"the view {self.label!r} has no free dimensions {sorted(unbound)} to bind"
            )
        return self._viewed(self._parent.with_dim_sizes(**sizes))

    def with_dim_names(self, **names: str) -> FieldView:
        """Rename symbolic dimensions in the parent, and view the result at the same path.

        Parameters
        ----------
        **names : str
            New names keyed by old; names that are not free are ignored.

        Returns
        -------
        FieldView
            The view, under the same name, of the parent with the dimensions
            renamed.
        """
        return self._viewed(self._parent.with_dim_names(**names))

    def _viewed(self, parent: Distribution) -> FieldView:
        """The view of *parent* at this view's path, under this view's name."""
        view = FieldView(parent, self._path)
        return view if view.label == self.label else view.with_label(self.label)

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The parent's path that the view reads."""
        return [("path", repr(self._path))]

    def _event_repr_arguments(self) -> list[tuple[str, str]]:
        """A view's declaration is its parent's schema at its path, which the path states."""
        return []


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


def _draw_path(declaration: OutputSpec, path: str) -> str:
    """*path*, a path of *declaration*, as the path of its node within a draw."""
    return _PATH_SEP.join(_draw_segments(declaration, path))


def _leaves_at(declaration: OutputSpec, path: str) -> list[str]:
    """The paths of the leaves of *declaration* at or below *path*, in canonical order."""
    return [leaf for leaf in _leaf_paths(declaration) if _is_within(leaf, path)]


def _below(path: str, node: str) -> str:
    """*path*, a path at or below *node*, relative to *node*."""
    return path[len(node) + 1 :]


def _shared_final_segment(paths: Sequence[str]) -> bool:
    """Whether two of *paths* share their final segment, so that their components collide."""
    components = [_final_segment(path) for path in paths]
    return len(set(components)) < len(components)


def _field_moves(
    pairs: Sequence[tuple[str, str]], source_order: Sequence[str]
) -> dict[str, str] | None:
    """The source field that each target field takes, keyed by the target field in its order.

    *pairs* lists each target field with its source field, in the target's
    order, and *source_order* lists the source fields in the source's order. The
    result is None when every field keeps its path and its position.
    """
    moves = dict(pairs)
    if list(moves) == list(moves.values()) == list(source_order):
        return None
    return moves


def _fields_of(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """The fields of the nested mapping *value*, keyed by their paths below it.

    A key may itself be a path, and a record within the mapping gives its fields.
    """
    fields: dict[str, Any] = {}
    for key, entry in value.items():
        path = f"{prefix}{key}"
        if isinstance(entry, Mapping):
            fields.update(_fields_of(entry, path + _PATH_SEP))
        elif isinstance(entry, Record):
            fields.update({f"{path}{_PATH_SEP}{leaf}": field for leaf, field in entry.items()})
        else:
            fields[path] = entry
    return fields


def _moved_value(value: Any, moves: Mapping[str, str] | None) -> Any:
    """*value* with the field at each value of *moves* placed at its key, in the order of *moves*.

    A record or a batch of records is rebuilt with its fields' specs, and a
    mapping as a nested mapping, in which a field that *moves* does not name
    keeps its path. A value of any other kind has no fields and is returned as
    it is, as is every value when *moves* is None.
    """
    if moves is None:
        return value
    if isinstance(value, RecordBatch):
        columns, template = value._raw_columns(), value.element_spec
        return value._rebuilt(
            {new: columns[old] for new, old in moves.items()},
            RecordSpec({new: template[old] for new, old in moves.items()}),
        )
    if isinstance(value, Record):
        template = value.event_template
        return Record(
            value.label,
            {new: value[old] for new, old in moves.items()},
            event_template=RecordSpec({new: template[old] for new, old in moves.items()}),
        )
    if isinstance(value, Mapping):
        fields = _fields_of(value)
        moved = {new: fields[old] for new, old in moves.items() if old in fields}
        taken = set(moves.values())
        moved.update({path: field for path, field in fields.items() if path not in taken})
        return _unflatten_paths(moved)
    return value


def _node_leaves(declaration: OutputSpec) -> dict[str, frozenset[str]]:
    """The leaves at or below each node of *declaration*, keyed by the node's path.

    The nodes come in canonical order, so a group comes before the nodes below it.
    """
    held: dict[str, set[str]] = {}
    for leaf in _leaf_paths(declaration):
        segments = leaf.split(_PATH_SEP)
        for end in range(1, len(segments) + 1):
            held.setdefault(_PATH_SEP.join(segments[:end]), set()).add(leaf)
    return {node: frozenset(leaves) for node, leaves in held.items()}


def _held_nodes(
    source: OutputSpec, target: OutputSpec, leaves: Mapping[str, str]
) -> dict[str, str]:
    """The node of *source* that each node of *target* holds, keyed by the target node's path.

    A target node holds a source node when its fields are the images under
    *leaves* of exactly that node's fields. Nodes that hold the same fields form
    a chain of only children, and the two chains pair up from their deepest
    nodes, so that a field holds a field.
    """
    chains: dict[frozenset[str], list[str]] = {}
    for node, held in _node_leaves(source).items():
        if all(leaf in leaves for leaf in held):
            chains.setdefault(frozenset(leaves[leaf] for leaf in held), []).append(node)
    targets: dict[frozenset[str], list[str]] = {}
    for node, held in _node_leaves(target).items():
        targets.setdefault(held, []).append(node)
    nodes: dict[str, str] = {}
    for held, chain in targets.items():
        nodes.update(zip(reversed(chain), reversed(chains.get(held, [])), strict=False))
    return nodes


def _declared_like(like: OutputSpec, specs: Mapping[str, TermSpec]) -> OutputSpec:
    """The declaration of the leaves *specs*, keyed by their paths, packaged as *like* allows.

    An exposed record stays exposed, and a whole term stays whole while its
    leaves lie under one component.
    """
    record = RecordSpec(dict(specs))
    if like.exposes_record or len(record.children) != 1:
        return OutputSpec(record)
    ((component, term),) = record.children.items()
    return OutputSpec(**{component: term})


@dataclass(frozen=True)
class _EventRenames:
    """The renames and moves that take an event declaration to a renamed one.

    The renamed declaration holds each field of the original at a new path, and
    maybe in a new order, so the values of a law over the original translate to
    it field by field: a draw of an exposed record carries the paths of its
    record, and a draw of a whole term the paths below its component. A node of
    the renamed declaration holds a node of the original when its fields are
    exactly that node's, and a path of the renamed law addresses the original's
    marginals and conditioning through the node it holds.

    Attributes
    ----------
    declaration : OutputSpec
        The original declaration.
    renamed : OutputSpec
        The renamed declaration.
    leaves : Mapping[str, str]
        The path in *renamed* of each leaf of *declaration*, keyed by its path in
        *declaration*.
    """

    declaration: OutputSpec
    renamed: OutputSpec
    leaves: Mapping[str, str]

    @classmethod
    def of(
        cls, declaration: OutputSpec, renamed: OutputSpec, renames: Mapping[str, str]
    ) -> _EventRenames:
        """The renames that take *declaration* to *renamed*, which *renames* gives."""
        if not renames:
            return cls(declaration, renamed, {leaf: leaf for leaf in _leaf_paths(declaration)})
        paths = _components_record(declaration)
        moved = paths._moved_leaf_paths(paths._resolve_path_renames(renames, {}))
        return cls(declaration, renamed, moved)

    def followed_by(self, then: _EventRenames) -> _EventRenames:
        """These renames followed by *then*, which starts from this renamed declaration."""
        leaves = {old: then.leaves[new] for old, new in self.leaves.items() if new in then.leaves}
        return _EventRenames(self.declaration, then.renamed, leaves)

    @cached_property
    def _draw_moves(self) -> dict[str, str] | None:
        inverse = {new: old for old, new in self.leaves.items()}
        return _field_moves(
            [
                (_draw_path(self.renamed, new), _draw_path(self.declaration, inverse[new]))
                for new in _leaf_paths(self.renamed)
                if new in inverse
            ],
            [
                _draw_path(self.declaration, old)
                for old in _leaf_paths(self.declaration)
                if old in self.leaves
            ],
        )

    @cached_property
    def _undraw_moves(self) -> dict[str, str] | None:
        images = set(self.leaves.values())
        return _field_moves(
            [
                (_draw_path(self.declaration, old), _draw_path(self.renamed, self.leaves[old]))
                for old in _leaf_paths(self.declaration)
                if old in self.leaves
            ],
            [_draw_path(self.renamed, new) for new in _leaf_paths(self.renamed) if new in images],
        )

    @cached_property
    def _nodes(self) -> dict[str, str]:
        return _held_nodes(self.declaration, self.renamed, self.leaves)

    @cached_property
    def _coordinate_order(self) -> list[int] | None:
        """The original flat coordinate at each coordinate of a renamed draw, or None if equal."""
        inverse = {new: old for old, new in self.leaves.items()}
        order = [
            index
            for new in _leaf_paths(self.renamed)
            for index in _coordinate_range(self.declaration, inverse[new])
        ]
        return None if order == list(range(len(order))) else order

    def draw(self, value: Any) -> Any:
        """*value*, a raw value of the original declaration or a batch of them, under the new paths.

        The fields of a batch of records take the new paths in the renamed
        declaration's order, and the batch keeps its label and levels.
        """
        return _moved_value(value, self._draw_moves)

    def undraw(self, value: Any) -> Any:
        """*value*, a raw value of the renamed declaration, under the original paths."""
        return _moved_value(value, self._undraw_moves)

    def covariance(self, cov: LinOp) -> LinOp:
        """*cov*, the original's covariance, over the flat coordinates in the renamed order.

        A move reorders the flat coordinates, and the product ``P Σ Pᵀ``, with
        ``P`` permuting them, stays lazy.
        """
        order = self._coordinate_order
        if order is None:
            return cov
        permutation = DenseLinOp(jnp.eye(cov.shape[1], dtype=cov.dtype)[jnp.asarray(order)])
        return permutation @ cov @ permutation.T

    def quantiles(self, quantiles: Any) -> Any:
        """*quantiles*, the original's, under the new paths.

        Quantiles in the event's raw form, per-field arrays with the level axes
        leading, move with their fields, and quantiles over the flat
        coordinates, which follow the level axes, are permuted into the renamed
        order.
        """
        if isinstance(quantiles, (Record, RecordBatch, Mapping)):
            return self.draw(quantiles)
        order = self._coordinate_order
        return quantiles if order is None else jnp.asarray(quantiles)[..., jnp.asarray(order)]

    def original(self, path: str) -> str | None:
        """The original node that *path*, a path of the renamed declaration, holds, or None."""
        return self._nodes.get(path)

    def undraw_at(self, path: str, value: Any) -> Any:
        """*value*, the node at *path* of the renamed declaration, as the original node it holds."""
        original = self._nodes[path]
        moves = _field_moves(
            [
                (_below(old, original), _below(self.leaves[old], path))
                for old in _leaves_at(self.declaration, original)
            ],
            [_below(new, path) for new in _leaves_at(self.renamed, path)],
        )
        return _moved_value(value, moves)

    def law(self, law: Distribution) -> Distribution:
        """*law*, over the original declaration or the part of it that remains, renamed.

        Each field that *law* still declares takes its new path, in the renamed
        declaration's order, so a law conditioned on some fields keeps the paths
        of the fields that remain.
        """
        fields = _leaf_paths(law.event_spec)
        if not any(field in self.leaves for field in fields):
            return law
        moved = {field: self.leaves.get(field, field) for field in fields}
        rank = {new: index for index, new in enumerate(_leaf_paths(self.renamed))}
        order = sorted(fields, key=lambda field: rank.get(moved[field], len(rank)))
        specs = {moved[field]: _node_at(law.event_spec, field) for field in order}
        return _renamed_by_leaves(law, _declared_like(law.event_spec, specs), moved)

    def marginal(
        self, law: Distribution, paths: Sequence[str], originals: Sequence[str]
    ) -> Distribution:
        """*law*, the marginal at the original nodes *originals*, as the marginal at *paths*.

        A marginal names each node by the final segment of its path, so each
        field of an original node takes the final segment of the renamed path
        followed by the field's path below the renamed node, in the renamed
        order.
        """
        moved: dict[str, str] = {}
        specs: dict[str, TermSpec] = {}
        for path, original in zip(paths, originals, strict=True):
            held = {self.leaves[old]: old for old in _leaves_at(self.declaration, original)}
            for new in _leaves_at(self.renamed, path):
                field = _final_segment(original) + held[new][len(original) :]
                moved[field] = _final_segment(path) + new[len(path) :]
                specs[moved[field]] = _node_at(law.event_spec, field)
        return _renamed_by_leaves(law, _declared_like(law.event_spec, specs), moved)


def _original_nodes(
    event: _EventRenames, declaration: OutputSpec, paths: Sequence[str], parent: str
) -> list[str]:
    """The node of the parent *parent* that each of *paths*, paths of *declaration*, holds.

    Raises
    ------
    KeyError
        If a path is not an event path of *declaration*.
    ValueError
        If a path holds no single node of the parent, since a move gathered or
        regrouped its fields.
    """
    originals = []
    for path in paths:
        if not _has_path(declaration, path):
            raise KeyError(path)
        original = event.original(path)
        if original is None:
            raise ValueError(
                f"{path!r} holds no single node of {parent!r}, since a move gathered or "
                f"regrouped its fields"
            )
        originals.append(original)
    return originals


def _unreached(
    event: _EventRenames, declaration: OutputSpec, paths: Sequence[str], owner: str
) -> Feasibility | None:
    """Why *paths*, of the law *owner* declared by *declaration*, miss the parent, or None."""
    for path in paths:
        if not _has_path(declaration, path):
            return Feasibility(False, f"{path!r} is not an event path of {owner!r}")
        if event.original(path) is None:
            return Feasibility(
                False, f"{path!r} holds no single node of the parent, since a move regrouped it"
            )
    return None


def _renamed_by_leaves(
    law: Distribution, target: OutputSpec, leaves: Mapping[str, str]
) -> Distribution:
    """*law* declared by *target*, each of its leaves at the path that *leaves* gives it.

    A whole term whose draws keep their fields is renamed by its component,
    which keeps the law's class.
    """
    source = law.event_spec
    if target == source:
        return law
    event = _EventRenames(source, target, dict(leaves))
    if event._draw_moves is None and not (source.exposes_record or target.exposes_record):
        (component,), (renamed,) = source.components, target.components
        return law.with_path_names({component: renamed})
    return _renamed(law, event, leaves)


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
    """The parent's covariance, over the flat coordinates in the renamed declaration's order."""
    return self._event.covariance(self._parent._cov())


def _renamed_quantile(self: _RenamedDistribution, q: ArrayLike) -> Any:
    """The parent's quantiles at the levels *q*, under the new paths."""
    return self._event.quantiles(self._parent._quantile(q))


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
    """The parent's marginal at the original nodes for *path*, named and arranged by *path*.

    The marginal keeps this law's label.

    Raises
    ------
    KeyError
        If a path is not an event path of this law.
    ValueError
        If a path holds no single node of the parent, or two selected paths
        share their final segment.
    """
    paths = (path,) if isinstance(path, str) else tuple(path)
    originals = self._originals(paths)
    if _shared_final_segment(paths):
        raise ValueError(
            f"the selected paths {list(paths)} share a final segment, so their components "
            f"would collide"
        )
    marginal = self._parent._marginal(originals[0] if isinstance(path, str) else tuple(originals))
    return _labeled(self._event.marginal(marginal, paths, originals), self.label)


def _renamed_marginal_guard(self: _RenamedDistribution, path: str | tuple[str, ...]) -> Feasibility:
    """The parent's marginal guard at the original nodes for *path*, paths of this law.

    Each path must hold one node of the parent, and a tuple of paths selects
    several nodes, whose final segments must differ.
    """
    paths = (path,) if isinstance(path, str) else tuple(path)
    unreached = _unreached(self._event, self.event_spec, paths, self.label)
    if unreached is not None:
        return unreached
    if _shared_final_segment(paths):
        return Feasibility(False, f"the paths {list(paths)} share a final segment")
    originals = self._originals(paths)
    return _capability_guard(
        self._parent, "_marginal", originals[0] if isinstance(path, str) else tuple(originals)
    )


def _renamed_marginal_capabilities(
    self: _RenamedDistribution, path: str | tuple[str, ...]
) -> frozenset[type]:
    """The parent's report of its marginal at the original nodes for *path*.

    A path that holds no single node of the parent, as a regrouping node does,
    has no exact marginal here, which the marginal guard reports, so its report
    claims nothing and a view there claims no density.

    Raises
    ------
    KeyError
        If a path is not an event path of this law.
    """
    paths = (path,) if isinstance(path, str) else tuple(path)
    for each in paths:
        if not _has_path(self.event_spec, each):
            raise KeyError(each)
    if any(self._event.original(each) is None for each in paths):
        return frozenset()
    originals = self._originals(paths)
    return _marginal_claims(
        self._parent, originals[0] if isinstance(path, str) else tuple(originals)
    )


def _renamed_condition_on(
    self: _RenamedDistribution, given: Any, /, **options: Any
) -> Distribution:
    """The parent conditioned on *given* under the original names, then renamed.

    Raises
    ------
    KeyError
        If a key of *given* is not an event path of this law.
    ValueError
        If a key holds no single node of the parent.
    """
    items = list(given.items())
    originals = self._originals([path for path, _ in items])
    translated = {
        original: self._event.undraw_at(path, value)
        for original, (path, value) in zip(originals, items, strict=True)
    }
    return self._event.law(self._parent._condition_on(translated, **options))


def _renamed_condition_on_guard(self: _RenamedDistribution, paths: tuple[str, ...]) -> Feasibility:
    """The parent's conditioning guard at the original nodes for *paths*, paths of this law."""
    unreached = _unreached(self._event, self.event_spec, paths, self.label)
    if unreached is not None:
        return unreached
    return _capability_guard(self._parent, "_condition_on", tuple(self._originals(paths)))


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
    SupportsMarginals: {
        "_marginal": _renamed_marginal,
        "_marginal_guard": _renamed_marginal_guard,
        "_marginal_capabilities": _renamed_marginal_capabilities,
    },
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
    """A parent law under renamed or moved event paths, which translates values at its boundary.

    ``Distribution.with_path_names`` returns it for a rename that changes the
    path of a field of a record draw when the parent's family does not rebuild
    itself under the new paths. It claims each of the parent's capabilities
    that a rename carries over: a draw, a moment, a quantile, or a marginal is
    translated on the way out, and a scored value, a given, or a path on the
    way in. A move reorders the flat coordinates, so the covariance, and
    quantiles over the flat coordinates, are permuted into the renamed
    declaration's order. Each capability carries the parent's guard at the
    original nodes. A further rename of this law renames its parent by both
    renames at once.

    Parameters
    ----------
    parent : Distribution
        The law whose values are translated.
    event : _EventRenames
        The renames from the parent's declaration to this law's.
    renames : tuple of Mapping[str, str]
        The renames applied to the parent, in order, each as its caller gave
        it, which the repr shows.
    """

    _capability_table = _RENAMED_CAPABILITIES

    def __new__(
        cls,
        parent: Distribution,
        event: _EventRenames,
        renames: tuple[Mapping[str, str], ...],
    ) -> _RenamedDistribution:
        return object.__new__(
            _capability_subclass(_RenamedDistribution, _claimed(parent, _RENAMED_CAPABILITIES))
        )

    def __init__(
        self,
        parent: Distribution,
        event: _EventRenames,
        renames: tuple[Mapping[str, str], ...],
    ) -> None:
        self._init_tracked(parent.label)
        self._init_annotations(None)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_event", event)
        object.__setattr__(self, "_renames", tuple(dict(step) for step in renames))
        self._init_declaration(event.renamed)

    def with_path_names(
        self, mapping: Mapping[str, str] | None = None, /, **kwargs: str
    ) -> Distribution:
        """Rename or move nodes of the event declaration, as :meth:`Distribution.with_path_names`.

        The result renames this law's parent by this law's renames followed by
        the new ones, so a rename of the component alone keeps the renames of
        the fields below it.
        """
        renamed = self.event_spec.with_path_names(mapping, **kwargs)
        renames = {**dict(mapping or {}), **kwargs}
        return _renamed(self, _EventRenames.of(self.event_spec, renamed, renames), renames)

    def _original_given(self, given: Any) -> Record | None:
        """*given*, keyed by event paths of this law, as the record of the parent's nodes it binds.

        ``None`` when a path holds no single node of the parent, as at a node a
        rename gathered, or is not an event path of this law.
        """
        items = list(given.items())
        if any(self._event.original(path) is None for path, _ in items):
            return None
        return Record(
            "given",
            {
                self._event.original(path): self._event.undraw_at(path, value)
                for path, value in items
            },
        )

    def _originals(self, paths: Sequence[str]) -> list[str]:
        """The parent's node for each of *paths*, event paths of this law.

        Raises
        ------
        KeyError
            If a path is not an event path of this law.
        ValueError
            If a path holds no single node of the parent.
        """
        return _original_nodes(self._event, self.event_spec, paths, self._parent.label)

    def __repr__(self) -> str:
        """The parent's repr followed by ``.with_path_names({...})`` for each rename.

        Each rename shows the paths it changed, and a label that differs from
        the parent's follows as ``.with_label(...)``. A reorder that changes no
        path, which no rename can state, reads as a call of the parent's class
        with the parent's arguments and this law's declaration.
        """
        renames = [{old: new for old, new in step.items() if old != new} for step in self._renames]
        parent = self._parent
        if not all(renames):
            return term_repr(
                self._repr_class_name(),
                self.label,
                [*parent._repr_arguments(), ("event_spec", repr(self.event_spec))],
            )
        text = repr(parent)
        for step in renames:
            mapping = mapping_repr({old: repr(new) for old, new in step.items()})
            text = _method_call(text, "with_path_names", mapping)
        if self.label != parent.label:
            text = _method_call(text, "with_label", repr(self.label))
        return text

    def _repr_class_name(self) -> str:
        """The class of the law this one renames, which a law presenting this one names."""
        return self._parent._repr_class_name()


def _method_call(receiver: str, method: str, argument: str) -> str:
    """The repr ``receiver.method(argument)``, on the receiver's last line where the call fits.

    A call that would pass the repr width lays its argument out on a line of its
    own, as :func:`~probpipe.core._repr.call_repr` lays out a constructor call.
    """
    call = call_repr(f".{method}", [argument])
    if "\n" not in call and len(receiver.rsplit("\n", 1)[-1]) + len(call) <= WIDTH:
        return receiver + call
    return call_repr(f"{receiver}.{method}", [argument])


def _renamed(law: Distribution, event: _EventRenames, arguments: Mapping[str, str]) -> Distribution:
    """The law that translates the values of *law* by *event*, with the rename's provenance.

    A renamed law's parent is translated by both renames at once, and the result
    keeps each rename's *arguments* for its repr.
    """
    if isinstance(law, _RenamedDistribution):
        renamed = _RenamedDistribution(
            law._parent, law._event.followed_by(event), (*law._renames, arguments)
        )
    else:
        renamed = _RenamedDistribution(law, event, (arguments,))
    renamed.with_provenance(
        Provenance.create("with_path_names", parents=[law], metadata=dict(arguments))
    )
    return renamed


def _rename_source(law: Distribution) -> tuple[Distribution, _EventRenames] | None:
    """The law *law* renames, with the renames from its declaration to *law*'s, or None.

    A law that translates its parent's values at its boundary holds its parent,
    and a member of a family that ``with_path_names`` rebuilt records the law it
    renames. Either one reads that law's draws in a lift (V.5).
    """
    if isinstance(law, _RenamedDistribution):
        return law._parent, law._event
    return getattr(law, _RENAME_SOURCE, None)


def _renamed_law(
    parent: Distribution, event_spec: OutputSpec, renames: Mapping[str, str]
) -> Distribution:
    """The law ``with_path_names`` returns for *parent* when a rename changes the path of a field.

    A factored law renames through its factors where they carry the rename, and
    regroups them where the rename gathers their components under new nodes. A
    family that rebuilds itself under the new paths returns its member, which
    records *parent* as the law it renames, so a lift draws the two together.
    Any other law, or a rename its factors cannot carry, is translated at the
    boundary of the law that holds it.
    """
    if isinstance(parent, SupportsFactors):
        joint = _renamed_through_factors(parent, renames, event_spec)
        if joint is None:
            joint = _regrouped(parent, renames, event_spec)
        if joint is not None:
            return joint
    event = _EventRenames.of(parent.event_spec, event_spec, renames)
    member = parent._renamed_in_family(event)
    if member is None:
        return _renamed(parent, event, renames)
    object.__setattr__(member, _RENAME_SOURCE, (parent, event))
    return member.with_provenance(
        Provenance.create("with_path_names", parents=[parent], metadata=dict(renames))
    )


# ---------------------------------------------------------------------------
# Renaming a factored law through its factors
# ---------------------------------------------------------------------------


def _leaf_specs(declaration: OutputSpec) -> dict[str, TermSpec]:
    """The spec of each leaf of *declaration*, keyed by its path."""
    return {path: _node_at(declaration, path) for path in _leaf_paths(declaration)}


def _factor_renames(graph: _FactorGraph, pairs: Mapping[str, str]) -> list[dict[str, str]]:
    """The renames each factor of *graph* applies for the joint's renames *pairs*.

    The factor that produces a renamed component renames the event path, each
    factor that conditions on the component renames the matching given path,
    and a renamed unmet given is renamed in every factor that names it. A
    joint's path that starts with a factor's component is the same path in the
    factor's own declaration, and so is a path into a given slot.
    """
    renames: list[dict[str, str]] = [{} for _ in graph.factors]
    for old, new in pairs.items():
        head = old.split(_PATH_SEP, 1)[0]
        producer = graph.producers.get(head)
        if producer is not None:
            renames[producer][old] = new
        for index, factor in enumerate(graph.factors):
            if isinstance(factor, ConditionalDistribution) and head in factor.given_spec:
                renames[index][old] = new
    return renames


def _renamed_through_factors(
    joint: Any, pairs: Mapping[str, str], event_spec: OutputSpec
) -> Distribution | ConditionalDistribution | None:
    """*joint* renamed by *pairs* through its factors, or None when the factors cannot carry it.

    The result is the factored joint of the renamed factors over the same
    graph, under the joint's name. Its components follow the factors, so a
    moved node joins its factor's components rather than the end of the joint's.
    The factors cannot carry a rename that changes a factor's packaging, as
    moving a whole term's component into a group does, that places components of
    two factors under one node, or that changes which factor conditions on
    which. A packaged joint renames the paths below its component through its
    factors, and keeps its packaging.

    Parameters
    ----------
    joint : FactoredDistribution or FactoredConditionalDistribution
        The joint renamed.
    pairs : Mapping[str, str]
        The new exact path of each renamed node, keyed by its exact path; a key
        starts with a component or, for a conditional joint, a given slot.
    event_spec : OutputSpec
        The joint's event declaration with the renames applied, whose leaves
        the result must declare.
    """
    graph: _FactorGraph = joint._graph
    packaging: dict[str, str] = {}
    component = _whole_term_component(joint.event_spec)
    if component is not None:
        inner = _within_package(joint, pairs, component)
        target = _whole_term_component(event_spec)
        if inner is None or target is None:
            return None
        pairs, packaging = inner, {"_component": target}
    try:
        factors = [
            factor.with_path_names(renames) if renames else factor
            for factor, renames in zip(graph.factors, _factor_renames(graph, pairs), strict=True)
        ]
        kind = (
            FactoredConditionalDistribution
            if isinstance(joint, ConditionalDistribution)
            else FactoredDistribution
        )
        renamed = kind(joint.label, factors, _scope=graph.scope, **packaging)
    except (KeyError, ValueError):
        return None
    if _leaf_specs(renamed.event_spec) != _leaf_specs(event_spec):
        return None
    if {edge[:2] for edge in renamed._graph.edges} != {edge[:2] for edge in graph.edges}:
        return None
    return renamed.with_provenance(
        Provenance.create("with_path_names", parents=[joint], metadata=dict(pairs))
    )


def _within_package(joint: Any, pairs: Mapping[str, str], component: str) -> dict[str, str] | None:
    """*pairs* as paths of the factor record of *joint*, packaged under *component*.

    A key of the event side and its target each start with the component, and
    a key of a given slot is kept. None when an event-side rename leaves the
    component or renames it.
    """
    inner: dict[str, str] = {}
    prefix = f"{component}{_PATH_SEP}"
    for old, new in pairs.items():
        if isinstance(joint, ConditionalDistribution) and old.split(_PATH_SEP, 1)[0] in (
            joint.given_spec
        ):
            inner[old] = new
        elif old.startswith(prefix) and new.startswith(prefix):
            inner[old[len(prefix) :]] = new[len(prefix) :]
        else:
            return None
    return inner


def _gathered(graph: _FactorGraph, pairs: Mapping[str, str]) -> dict[str, str] | None:
    """The gathering node each gathered component joins, keyed by the component.

    A rename gathers a component when it moves the whole component into a new
    top-level node. None when no rename gathers, or when another rename reaches
    a gathering node or a gathered component.
    """
    nodes: dict[str, str] = {}
    for old, new in pairs.items():
        head, _, rest = new.partition(_PATH_SEP)
        if old in graph.producers and rest and head not in graph.producers:
            nodes[old] = head
    if not nodes:
        return None
    for old, new in pairs.items():
        if old in nodes:
            continue
        if old.split(_PATH_SEP, 1)[0] in nodes or new.split(_PATH_SEP, 1)[0] in nodes.values():
            return None
    return nodes


def _inner_path(path: str, node: str) -> str:
    """*path*, which starts with *node*, as the path below it."""
    return path[len(node) + 1 :]


def _regrouped_renames(
    graph: _FactorGraph,
    pairs: Mapping[str, str],
    nodes: Mapping[str, str],
    groups: Mapping[int, str | None],
) -> list[dict[str, str]]:
    """The renames each factor of *graph* applies when the rename *pairs* regroups it.

    A factor in a group renames its components to their paths below the
    group's node, and a factor left in place renames them as *pairs* does. A
    consumer of a component renames its given slot to the component's path
    below the node when both factors are in one group, and to the component's
    new path otherwise, which moves the slot into a structured slot named by
    the node.
    """
    renames: list[dict[str, str]] = [{} for _ in graph.factors]
    for old, new in pairs.items():
        head = old.split(_PATH_SEP, 1)[0]
        producer = graph.producers.get(head)
        node = nodes.get(head)
        if producer is not None:
            target = new if node is None else _inner_path(new, node)
            if target != old:
                renames[producer][old] = target
        for index, factor in enumerate(graph.factors):
            if not (isinstance(factor, ConditionalDistribution) and head in factor.given_spec):
                continue
            target = new
            if node is not None and groups[index] == node:
                target = _inner_path(new, node)
            if target != old:
                renames[index][old] = target
    return renames


def _widened(
    kernel: ConditionalDistribution, slot: str, record: RecordSpec
) -> ConditionalDistribution | None:
    """*kernel* with its structured slot *slot* declared as *record*, which holds its fields.

    The result reads the kernel's own fields of a value of *record* and drops
    the others, so a consumer of part of a gathered node conditions on the
    node's whole record. None when *record* lacks a field of the slot or
    declares it otherwise.
    """
    declared = kernel.given_spec[slot]
    if not isinstance(declared, RecordSpec):
        return None
    if any(record.children.get(name) != spec for name, spec in declared.children.items()):
        return None
    given_spec = InputSpec(
        {name: (record if name == slot else spec) for name, spec in kernel.given_spec.items()}
    ).with_optional(*(kernel.given_spec.optional - {slot}))
    own = set(_given_leaf_specs(kernel.given_spec))
    origins = {leaf: (leaf if leaf in own else None) for leaf in _given_leaf_specs(given_spec)}
    event = _EventRenames.of(kernel.event_spec, kernel.event_spec, {})
    widened = _RenamedConditionalDistribution(kernel, given_spec, kernel.event_spec, origins, event)
    return widened.with_provenance(
        Provenance.create("with_path_names", parents=[kernel], metadata={"widened": slot})
    )


def _regrouped(
    joint: Any, pairs: Mapping[str, str], event_spec: OutputSpec
) -> Distribution | ConditionalDistribution | None:
    """*joint* renamed by *pairs* with the factors under each gathering node packaged as one.

    The factors whose components the rename moves into a new node form a
    packaged sub-joint, labeled by joining their labels, whose event is the
    node as one component holding their components' record; the other factors
    rename in place. A consumer of a gathered component conditions on the
    sub-joint through its renamed given slot, widened to the node's whole
    record. The result orders its factors conditional-first, each one at its
    first factor's position where the dependencies allow.

    None when no rename gathers whole components, a factor's components land
    in two places, the groups condition on one another in a cycle, or the
    result does not declare the leaves of *event_spec*.
    """
    if _whole_term_component(joint.event_spec) is not None:
        return None
    graph: _FactorGraph = joint._graph
    nodes = _gathered(graph, pairs)
    if nodes is None:
        return None
    groups: dict[int, str | None] = {}
    for index, factor in enumerate(graph.factors):
        joined = {nodes.get(component) for component in factor.event_spec.components}
        if len(joined) > 1:
            return None
        groups[index] = next(iter(joined), None)
    try:
        renamed = [
            factor.with_path_names(renames) if renames else factor
            for factor, renames in zip(
                graph.factors, _regrouped_renames(graph, pairs, nodes, groups), strict=True
            )
        ]
    except (KeyError, ValueError):
        return None
    members: dict[str, list[int]] = {}
    for index, node in groups.items():
        if node is not None:
            members.setdefault(node, []).append(index)
    records = {
        node: RecordSpec(
            {
                component: spec
                for index in indices
                for component, spec in renamed[index].event_spec.components.items()
            }
        )
        for node, indices in members.items()
    }
    for index, factor in enumerate(renamed):
        if not isinstance(factor, ConditionalDistribution):
            continue
        for slot in list(factor.given_spec):
            if slot in records and factor.given_spec[slot] != records[slot]:
                factor = _widened(factor, slot, records[slot])
                if factor is None:
                    return None
        renamed[index] = factor
    units: list[tuple[int, Any]] = []
    try:
        for node, indices in members.items():
            parts = [renamed[index] for index in indices]
            kind = (
                FactoredDistribution
                if _factor_graph(parts, graph.scope).unmet is None
                else FactoredConditionalDistribution
            )
            label = _joined_label(part.label for part in parts)
            units.append((indices[0], kind(label, parts, _scope=graph.scope, _component=node)))
    except (KeyError, TypeError, ValueError):
        return None
    units.extend((index, renamed[index]) for index, node in groups.items() if node is None)
    order = _conditional_first(units)
    if order is None:
        return None
    kind = (
        FactoredConditionalDistribution
        if isinstance(joint, ConditionalDistribution)
        else FactoredDistribution
    )
    try:
        result = kind(joint.label, [units[position][1] for position in order], _scope=graph.scope)
    except (KeyError, TypeError, ValueError):
        return None
    if _leaf_specs(result.event_spec) != _leaf_specs(event_spec):
        return None
    return result.with_provenance(
        Provenance.create("with_path_names", parents=[joint], metadata=dict(pairs))
    )


def _conditional_first(units: Sequence[tuple[int, Any]]) -> list[int] | None:
    """An order of *units* in which each consumer precedes the factors it conditions on.

    Each unit is a factor with the position of its first original factor, which
    breaks ties, so an order with no regrouping keeps the original. None when
    the units condition on one another in a cycle.
    """
    producers = {
        component: position
        for position, (_, unit) in enumerate(units)
        for component in unit.event_spec.components
    }
    consumes = {
        position: {
            producers[slot]
            for slot in (unit.given_spec if isinstance(unit, ConditionalDistribution) else ())
            if slot in producers
        }
        - {position}
        for position, (_, unit) in enumerate(units)
    }
    order: list[int] = []
    remaining = set(range(len(units)))
    while remaining:
        ready = [
            position
            for position in remaining
            if not any(position in consumes[other] for other in remaining if other != position)
        ]
        if not ready:
            return None
        chosen = min(ready, key=lambda position: units[position][0])
        order.append(chosen)
        remaining.remove(chosen)
    return order


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
    """The parent's covariance at the translated given, over the coordinates in the renamed order."""
    return self._event.covariance(self._parent._conditional_cov(self._parent_given(given)))


def _renamed_conditional_quantile(
    self: _RenamedConditionalDistribution, given: Any, q: ArrayLike
) -> Any:
    """The parent's quantiles at the translated given, under the new paths."""
    return self._event.quantiles(self._parent._conditional_quantile(self._parent_given(given), q))


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
    """The parent's marginal at the translated given and the original nodes, arranged by *path*.

    Raises
    ------
    KeyError
        If a path is not an event path of this kernel.
    ValueError
        If a path holds no single node of the parent, or two selected paths
        share their final segment.
    """
    paths = (path,) if isinstance(path, str) else tuple(path)
    originals = _original_nodes(self._event, self.event_spec, paths, self._parent.label)
    if _shared_final_segment(paths):
        raise ValueError(
            f"the selected paths {list(paths)} share a final segment, so their components "
            f"would collide"
        )
    marginal = self._parent._conditional_marginal(
        self._parent_given(given), originals[0] if isinstance(path, str) else tuple(originals)
    )
    return self._event.marginal(marginal, paths, originals)


def _renamed_conditional_marginal_guard(
    self: _RenamedConditionalDistribution, path: str | tuple[str, ...]
) -> Feasibility:
    """The parent's conditional marginal guard at the original nodes for *path*.

    Each path must hold one node of the parent, and a tuple of paths selects
    several nodes, whose final segments must differ.
    """
    paths = (path,) if isinstance(path, str) else tuple(path)
    unreached = _unreached(self._event, self.event_spec, paths, self.name)
    if unreached is not None:
        return unreached
    if _shared_final_segment(paths):
        return Feasibility(False, f"the paths {list(paths)} share a final segment")
    originals = _original_nodes(self._event, self.event_spec, paths, self._parent.label)
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
    origins : Mapping[str, str or None]
        The parent's leaf for each leaf of the renamed slots, keyed by its path;
        a leaf that a widened slot adds has none, and its value is dropped.
    event : _EventRenames
        The renames from the parent's event declaration to this kernel's.
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
        event: _EventRenames,
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
        event: _EventRenames,
        *,
        pending: Mapping[str, Any] | None = None,
    ) -> None:
        self._init_tracked(parent.label)
        self._init_annotations(None)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_origins", dict(origins))
        object.__setattr__(self, "_pending", dict(pending or {}))
        object.__setattr__(self, "_event", event)
        self._init_declaration(given_spec, event_spec)

    def _repr_class_name(self) -> str:
        """The class of the kernel this one renames, which it presents under new names."""
        return self._parent._repr_class_name()

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The family parameters of the kernel this one renames."""
        return self._parent._repr_arguments()

    def _event_repr_arguments(self) -> list[tuple[str, str]]:
        """The renamed declaration, by which this kernel differs from the one it renames."""
        return [("event_spec", repr(self.event_spec))]

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
            raise ValueError(f"the given binds part of a slot of {self.label!r}, without {missing}")
        bound = {
            **self._pending,
            **{
                self._origins[path]: value
                for path, value in values.items()
                if self._origins[path] is not None
            },
        }
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
            raise ValueError(f"{self.label!r} needs a value for every slot {list(self.given_spec)}")
        return complete

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **options: Any
    ) -> Distribution | ConditionalDistribution:
        """The parent bound at *given*, translated to its slots, under the new names.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values for some or all of this kernel's slots, by slot name.
        **options : Any
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
        remaining = self.given_spec.without(*slots)
        # A binding that leaves no required slot evaluates the parent, whose
        # optional slots left unbound take their defaults.
        evaluate = bool(complete) or not remaining.required
        result = self._parent._condition_on(complete, **options) if evaluate else self._parent
        if not isinstance(result, ConditionalDistribution):
            return self._event.law(result)
        origins = {
            path: leaf
            for path, leaf in self._origins.items()
            if path.partition(_PATH_SEP)[0] not in slots
        }
        return _RenamedConditionalDistribution(
            result, remaining, self.event_spec, origins, self._event, pending=pending
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

    A factored kernel renames through its factors where they carry the rename,
    and regroups them where the rename gathers their components under new
    nodes. Any other kernel, or a rename its factors cannot carry, renames at
    the boundary of the kernel that holds it.
    """
    if isinstance(parent, SupportsFactors):
        for attempt in (_renamed_through_factors, _regrouped):
            joint = attempt(parent, pairs, event_spec)
            if joint is not None and _given_leaf_specs(joint.given_spec) == _given_leaf_specs(
                given_spec
            ):
                return joint
    event = _EventRenames.of(parent.event_spec, event_spec, renames)
    kernel = _RenamedConditionalDistribution(parent, given_spec, event_spec, origins, event)
    kernel.with_provenance(
        Provenance.create("with_path_names", parents=[parent], metadata=dict(pairs))
    )
    return kernel


_install_field_view(FieldView)
_install_renamed_law(_renamed_law)
_install_renamed_kernel(_renamed_kernel)
