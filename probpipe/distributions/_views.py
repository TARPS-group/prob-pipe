"""Field views: the law of one event path of a parent distribution.

Provides:
  - ``FieldView`` – the ``Distribution`` that ``d[path]`` returns for a node
    below a law's whole term, holding a reference to its parent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericSpec, TermSpec
from ..core._specs import OutputSpec
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
    from ..core._dispatch import Feasibility
    from ..custom_types import Array, ArrayLike, PRNGKey

__all__ = ["FieldView"]

_PATH_SEP = "/"


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


# ---------------------------------------------------------------------------
# The derived capabilities
# ---------------------------------------------------------------------------
#
# Each function realizes one row of the derivation from a parent's capability,
# with pi the extraction of the view's node from a parent draw.


def _view_sample(self: FieldView, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Any:
    """Co-sample: draw ``X`` from the parent and return ``pi(X)``."""
    raise NotImplementedError("FieldView._sample")


def _view_mean(self: FieldView) -> Any:
    """Projection: the parent's mean at the view's path, since ``E[pi X] = pi E[X]``."""
    raise NotImplementedError("FieldView._mean")


def _view_variance(self: FieldView) -> Any:
    """Restriction of the parent's variance to the coordinates of the view's path."""
    raise NotImplementedError("FieldView._variance")


def _view_cov(self: FieldView) -> Any:
    """The sub-block ``P Σ Pᵀ`` of the parent's covariance, ``P`` selecting the path."""
    raise NotImplementedError("FieldView._cov")


def _view_quantile(self: FieldView, q: ArrayLike) -> Array:
    """Restriction of the parent's per-coordinate quantiles to the path."""
    raise NotImplementedError("FieldView._quantile")


def _view_expectation(self: FieldView, f: Callable[[Any], Array]) -> Array:
    """Composition: the parent's expectation of ``f ∘ pi``."""
    raise NotImplementedError("FieldView._expectation")


def _view_log_prob(self: FieldView, value: Any) -> Array:
    """The log-density of the parent's detached marginal at the path."""
    raise NotImplementedError("FieldView._log_prob")


def _view_log_prob_guard(self: FieldView) -> Feasibility:
    """Feasible where the parent's marginal at the path is exact and scores."""
    return _capability_guard(self._parent, "_marginal", self._path)


def _view_marginal(self: FieldView, path: str | tuple[str, ...]) -> Distribution:
    """Path composition: the parent's marginal at the view's path joined with *path*."""
    raise NotImplementedError("FieldView._marginal")


def _view_marginal_guard(self: FieldView, path: str | tuple[str, ...]) -> Feasibility:
    """The parent's guard at the joined path."""
    if not isinstance(path, str):
        raise NotImplementedError("FieldView._marginal_guard: a selection of several paths")
    return _capability_guard(self._parent, "_marginal", f"{self._path}{_PATH_SEP}{path}")


def _view_condition_on(self: FieldView, given: Any, /, **kwargs: Any) -> Distribution:
    """Conditioning commutes with marginalization: the parent conditioned, then viewed."""
    raise NotImplementedError("FieldView._condition_on")


#: Each capability a view may derive, with the methods that realize it.
_VIEW_CAPABILITIES: dict[type, Mapping[str, Callable[..., Any]]] = {
    SupportsSampling: {"_sample": _view_sample},
    SupportsMean: {"_mean": _view_mean},
    SupportsVariance: {"_variance": _view_variance},
    SupportsCovariance: {"_cov": _view_cov},
    SupportsQuantile: {"_quantile": _view_quantile},
    SupportsExpectation: {"_expectation": _view_expectation},
    SupportsLogProb: {
        "_log_prob": _view_log_prob,
        "_log_prob_guard": _view_log_prob_guard,
        "_unnormalized_log_prob_guard": _view_log_prob_guard,
    },
    SupportsMarginals: {"_marginal": _view_marginal, "_marginal_guard": _view_marginal_guard},
    SupportsExactConditioning: {"_condition_on": _view_condition_on},
    SupportsApproximateConditioning: {"_condition_on": _view_condition_on},
}


def _derived_protocols(parent: Distribution, node: TermSpec) -> set[type]:
    """The capabilities a view of *parent* at a node declared by *node* derives."""
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
    named by the path's final segment, and its label is the path.

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

    The projection rows are exact whenever the parent's answer is, and only
    sampling requires the parent to sample.

    Parameters
    ----------
    parent : Distribution
        The law whose event the view reads.
    path : str
        An event path of *parent*, a slash path that starts with a component.

    Raises
    ------
    KeyError
        If *path* is not an event path of *parent*.
    TypeError
        If *parent* is not a ``Distribution``.
    """

    _capability_table = _VIEW_CAPABILITIES

    def __new__(cls, parent: Distribution, path: str) -> FieldView:
        if not isinstance(parent, Distribution):
            raise TypeError(f"a field view reads a Distribution, got {type(parent).__name__}")
        node = _node_at(parent.event_spec, path)
        return object.__new__(_capability_subclass(FieldView, _derived_protocols(parent, node)))

    def __init__(self, parent: Distribution, path: str) -> None:
        node = _node_at(parent.event_spec, path)
        self._init_tracked(path)
        self._init_annotations(None)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_path", path)
        self._init_declaration(OutputSpec(**{path.split(_PATH_SEP)[-1]: node}))

    @property
    def parent(self) -> Distribution:
        """The law this view reads, which sibling views share."""
        return self._parent

    @property
    def path(self) -> str:
        """The event path of the parent that this view reads."""
        return self._path

    def __getitem__(self, key: str | tuple[str, ...]) -> Distribution:
        """This view under its component, or the parent's view at a path within it.

        A path within the node this view covers joins the view's own path, so
        ``d["a"]["b/c"]`` is ``d["a/b/c"]``.

        Raises
        ------
        KeyError
            If the joined path is not an event path of the parent.
        """
        if not isinstance(key, str):
            raise NotImplementedError("FieldView.__getitem__: a selection of several paths")
        component = self._path.split(_PATH_SEP)[-1]
        if key == component:
            return self
        head, _, rest = key.partition(_PATH_SEP)
        if head != component or not rest:
            raise KeyError(key)
        return FieldView(self._parent, f"{self._path}{_PATH_SEP}{rest}")

    def __repr__(self) -> str:
        return f"FieldView(parent={self._parent.name!r}, path={self._path!r})"
