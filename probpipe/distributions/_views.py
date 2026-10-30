"""Coupled marginals and renamed laws: laws that read a parent distribution's values.

Provides:
  - ``CoupledMarginal`` – the ``Distribution`` that ``d[path]`` returns for a node
    below a law's whole term, or for a list of paths, holding a reference to
    its parent.
  - the renamed law and the renamed kernel that ``rename`` returns
    when the values must carry new names and the family does not rebuild
    itself under them, which this module installs on the two distribution
    kinds at import.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
import numpy as np

from ..core._batch import Batch, BatchSpec
from ..core._dispatch import Feasibility, ResolutionError
from ..core._execution_claims import Traceability
from ..core._expression import (
    Expression,
    Product,
    Selected,
)
from ..core._object_batch import _is_object_array
from ..core._record_batch import RecordBatch, _batch_over, _stacked_batches
from ..core._record_spec import RecordSpec
from ..core._repr import WIDTH, call_repr, format_names, mapping_repr
from ..core._shapes import PathsLike
from ..core._spec_base import NumericSpec, TermSpec
from ..core._specs import (
    InputSpec,
    OutputSpec,
    _exposed_kind,
    _has_record_outputs,
    _output_record,
    _root_output_name,
)
from ..core.named_tree import _PATH_SEP, _last_name, _unflatten_paths
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import TrackedTerm
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
    _conditional_capability_guard,
    _marginal_claims,
)
from ._conditional import (
    ConditionalDistribution,
    _given_leaf_specs,
    _install_renamed_kernel,
)
from ._distribution import (
    _COPY_SOURCE,
    Distribution,
    _cached_raw,
    _fixed_paths,
    _fixes_every_field,
    _install_copy_source,
    _install_coupled_marginal,
    _install_renamed_law,
    _keeps_fixed_paths,
    _law_repr,
    _no_free_dims,
    _not_an_event_path,
    _read_event_paths,
)
from ._factored import (
    FactoredConditionalDistribution,
    FactoredDistribution,
    SupportsFactors,
    _closed_factors,
    _derived_product,
    _extract,
    _factor_graph,
    _FactorGraph,
    _is_named,
    _joined_label,
    _product_of,
    _raw_record,
    _SoleField,
    _with_named,
)

if TYPE_CHECKING:
    from ..custom_types import Array, ArrayLike, PRNGKey

__all__ = ["CoupledMarginal"]


# ---------------------------------------------------------------------------
# Paths, nodes, and coordinates of an event spec
# ---------------------------------------------------------------------------


def _node_at(declaration: OutputSpec, path: str) -> TermSpec:
    """The term spec of the node at *path* in *declaration*.

    A path starts with an output's name: the paths of the positional form over
    a record are its record's, and a whole term's are its output's name
    followed by the paths within its term.

    Parameters
    ----------
    declaration : OutputSpec
        The output spec to read, such as a law's event spec.
    path : str
        A path of *declaration*.

    Returns
    -------
    TermSpec
        The spec of a leaf, or the ``RecordSpec`` of an interior node.

    Raises
    ------
    KeyError
        If *path* is not a path of *declaration*.
    """
    try:
        paths = _output_record(declaration)
    except TypeError:
        raise KeyError(path) from None
    return paths._node_at(path)


def _view_declaration(parent_event_spec: OutputSpec, paths: str | tuple[str, ...]) -> OutputSpec:
    """The event spec of the coupled marginal at *paths* of a law whose event spec is *parent_event_spec*.

    It is ``parent_event_spec[paths]``: one path to a single value gives the value
    under an output named by its last name, and a list of paths gives the
    positional form over the record that keeps each node at its own path, in
    the canonical order of *parent_event_spec*. As an interim implementation
    detail, one path to a group, or to a field that holds a distribution or a
    batch, gives the node whole under an output named by its last name, until a
    coupled marginal of a group has the group's children as outputs and a coupled marginal at such a
    field is in the positional form.

    Parameters
    ----------
    parent_event_spec : OutputSpec
        The parent law's event spec.
    paths : str or tuple of str
        A path of *parent_event_spec*, or a list of paths to select jointly, as
        :func:`~probpipe.core._shapes._as_paths` reads them.

    Returns
    -------
    OutputSpec
        The keyword form for one path, and the positional form over a record for
        a list.

    Raises
    ------
    KeyError
        If a path is not a path of *parent_event_spec*.
    """
    if isinstance(paths, str):
        node_spec = _node_at(parent_event_spec, paths)
        if _exposed_kind(node_spec) is not None:
            return OutputSpec(**{_last_name(paths): node_spec})
    return parent_event_spec[paths]


def _draw_segments(event_spec: OutputSpec, path: str) -> tuple[str, ...]:
    """The segments of *path* below the top of a draw of *event_spec*.

    A draw of the positional form over a record is the record, which every
    segment addresses, and a whole term's draw is the term, so its output's
    name is dropped.
    """
    segments = tuple(path.split(_PATH_SEP))
    return segments if _root_output_name(event_spec) is None else segments[1:]


def _coordinate_range(declaration: OutputSpec, path: str) -> range:
    """The coordinates of the node at *path* in the flat vector of a draw of *declaration*.

    The flat vector lays out the array leaves in canonical order, depth first
    and in insertion order, so a node's coordinates are one contiguous range.

    Parameters
    ----------
    declaration : OutputSpec
        A numeric event spec.
    path : str
        A path of *declaration*.

    Returns
    -------
    range
        The indices from the node's first coordinate to one past its last.

    Raises
    ------
    TypeError
        If *declaration* is not numeric, so that a draw has no flat vector.
    ValueError
        If a dimension that the range depends on is unbound.
    """
    spec = declaration.term_spec
    if not isinstance(spec, NumericSpec):
        raise TypeError(
            f"a draw of {declaration!r} has no flat vector, since the event spec is not numeric"
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
    """The paths of *declaration* that address its leaves, each starting with an output's name."""
    leaves: list[str] = []
    for output_name, spec in declaration.outputs.items():
        if isinstance(spec, RecordSpec):
            leaves.extend(f"{output_name}{_PATH_SEP}{key}" for key in spec)
        else:
            leaves.append(output_name)
    return leaves


def _is_within(path: str, ancestor: str) -> bool:
    """Whether *path* is *ancestor* or a path below it."""
    return path == ancestor or path.startswith(ancestor + _PATH_SEP)


# ---------------------------------------------------------------------------
# The extraction of a node from a raw value
# ---------------------------------------------------------------------------


def _projector(declaration: OutputSpec, paths: str | tuple[str, ...]) -> Callable[[Any], Any]:
    """pi for a law declared by *declaration*: the extraction of its node at *paths*.

    A list of paths, in the canonical order of *declaration*, extracts the
    record that keeps each selected node at its own path. The paths are read
    now, so the extraction does not follow a later change to whatever supplied
    them. A ``Record`` holding draws, as the lift holds a record law's draws,
    gives its node at the path, or a ``Record`` of the selected nodes, and any
    other value is read in its raw form and gives the raw node, or the nested
    mapping of the selected nodes. Leading batch axes stay on every leaf.
    """
    single = isinstance(paths, str)
    listed_paths = (paths,) if single else tuple(paths)
    segments = [_draw_segments(declaration, selected_path) for selected_path in listed_paths]

    def project(value: Any) -> Any:
        if isinstance(value, Record):
            nodes = [
                value[_PATH_SEP.join(selected_path)] if selected_path else value
                for selected_path in segments
            ]
            if single:
                return nodes[0]
            return Record(
                {
                    selected_path: _raw_record(node)
                    for selected_path, node in zip(listed_paths, nodes)
                },
                label=value.label,
            )
        raw = _raw_record(value)
        nodes = [_extract(raw, selected_path) for selected_path in segments]
        return nodes[0] if single else _unflatten_paths(dict(zip(listed_paths, nodes)))

    return project


def _path_tuple(path: str | tuple[str, ...]) -> tuple[str, ...]:
    """*path* as a tuple of paths: a list of paths as it is, and one path as a tuple of one."""
    return tuple(path) if isinstance(path, tuple) else (path,)


def _marginal_expression_at(law: Distribution, paths: str | tuple[str, ...]) -> Expression:
    """The expression of the marginal of *law* at *paths*, which the coupled marginal of *law* at *paths* carries too.

    A marginal over the whole events of some factors, none of which conditions
    on an output outside them, is those factors (VI.8): one factor gives its
    own expression, and several their product without a label, in the law's
    order, which reads factor by factor, as ``a(a)·b(b)``. The factors of a
    field coupled marginal are its parent's, at the parent's paths. The marginal of a coupled marginal
    at its whole event is the coupled marginal, and any other marginal is *law* selected
    at *paths*, which keeps *law*'s label. The marginal holds the paths its
    factors hold fixed, followed by those *law* holds.

    Parameters
    ----------
    law : Distribution
        The law, which may be a field coupled marginal.
    paths : str or tuple of str
        An event path of *law*, or a list of them in *law*'s canonical order.

    Returns
    -------
    Expression
        The expression of the marginal.
    """
    requested = _path_tuple(paths)
    owner, listed_paths = law, requested
    if isinstance(law, CoupledMarginal):
        parent_paths = tuple(law._parent_path(selected_path) for selected_path in requested)
        if parent_paths == _path_tuple(law.path):
            return law._expression
        if None not in parent_paths:
            owner, listed_paths = law.parent, cast("tuple[str, ...]", parent_paths)
    closed = _closed_factors(owner, listed_paths)
    held = _fixed_paths(owner) + _fixed_paths(law)
    if closed is None:
        return Selected(law._embedded_expression(), requested, listed=isinstance(paths, tuple))
    if len(closed) == 1:
        return closed[0]._embedded_expression().with_fixed(held)
    return _product_of(closed).with_fixed(held)


def _carrying(law: Any, expression: Expression) -> Any:
    """*law* carrying *expression*: *law* itself when it carries it, and otherwise a copy that does.

    Two expressions are the same when they read alike under *law*'s own
    signature, so a factor that is its own marginal is returned as it is. The
    copy keeps *law*'s provenance.
    """
    own = law._own_signature()
    if law._expression.signed(own) == expression.signed(own):
        return law
    clone = law._shallow_copy()
    clone._store_expression(expression)
    return clone


def _keeps_expression(term: Any, source: Any) -> Any:
    """*term*, a law or kernel just built to rename *source*, carrying *source*'s expression.

    The rename keeps the label and the derivation, and the signature follows
    *term*'s own declaration. A product without a label, which reads factor
    by factor, reads by its joined label and *term*'s outputs instead, as
    ``(lik·prior)(beta, y)``, since its factors keep their own names.
    """
    expression = source._expression
    if isinstance(expression, Product):
        expression = Selected(expression, tuple(term.event_spec.outputs))
    term._store_expression(expression)
    return term


# ---------------------------------------------------------------------------
# The derived capabilities
# ---------------------------------------------------------------------------
#
# Each function realizes one row of the derivation from a parent's capability,
# with pi the extraction of the coupled marginal's node from a parent draw, or of the record
# of the selected nodes.

#: The method of each moment row. A coupled marginal projects the moment of a parent that
#: claims it, and otherwise computes it from the parent's exact marginal at the
#: coupled marginal's path, when the parent reports that marginal claims it.
_MOMENT_METHODS: dict[type, str] = {
    SupportsMean: "_mean",
    SupportsVariance: "_variance",
    SupportsCovariance: "_cov",
    SupportsQuantile: "_quantile",
}

#: The moment rows that need a numeric node.
_NUMERIC_MOMENTS = (SupportsCovariance, SupportsQuantile)


def _marginal_moment(self: CoupledMarginal, method: str, *arguments: Any) -> Any:
    """The moment *method* of the parent's exact marginal at the coupled marginal's path."""
    return getattr(self._parent._marginal(self._path), method)(*arguments)


def _view_sample(self: CoupledMarginal, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Any:
    """Co-sample: draw ``X`` from the parent with the same key and return ``pi(X)``.

    The key and the sample shape pass to the parent unchanged, so coupled marginals of one
    parent drawn with one key project one parent draw. The draws take the raw
    form of the coupled marginal's own event, which is the raw form of its standalone
    marginal's draws (see :func:`_drawn_form`).
    """
    drawn = self._project(self._parent._sample(key, sample_shape))
    return _drawn_form(drawn, self.event_spec.term_spec, tuple(sample_shape))


def _drawn_form(value: Any, spec: TermSpec, sample_shape: tuple[int, ...]) -> Any:
    """The raw draws *value* of an event that *spec* declares, in the raw form of its draws.

    A batch-valued node has two raw forms. A whole term that is a batch is
    drawn as its raw values, with the sample axes before the batch axes. A field
    of a record that is a batch is drawn as one batch per draw: the batch itself
    for one draw, and an object array of batches over the sample axes otherwise.
    A field coupled marginal's node can have the other form in its parent, so the coupled marginal's
    draws are converted to the form its own event declares.

    Parameters
    ----------
    value : Any
        The projected draws, with the sample axes leading.
    spec : TermSpec
        The coupled marginal's event type.
    sample_shape : tuple of int
        The sample axes that lead *value*.

    Returns
    -------
    Any
        *value* with each batch-valued node in the form its place in *spec*
        declares, and *value* itself when *spec* holds no batch.
    """
    if isinstance(spec, BatchSpec):
        return _batch_values(value, spec, sample_shape)
    if not (_holds_batch(spec) and _is_raw_mapping(value)):
        return value
    return _with_batch_fields(value, cast(RecordSpec, spec), sample_shape)


def _is_raw_mapping(value: Any) -> bool:
    """Whether *value* is the raw form of record draws: a mapping that is not a tracked term."""
    return isinstance(value, Mapping) and not isinstance(value, TrackedTerm)


def _holds_batch(spec: TermSpec) -> bool:
    """Whether *spec* is a batch or a record with a batch at some path."""
    if isinstance(spec, BatchSpec):
        return True
    return isinstance(spec, RecordSpec) and any(
        _holds_batch(child) for child in spec.children.values()
    )


def _with_batch_fields(value: Any, spec: RecordSpec, sample_shape: tuple[int, ...]) -> Any:
    """The record draws *value* that *spec* declares, with one batch per draw per batch field."""
    fields = {}
    for name, child in spec.children.items():
        node = value[name]
        if isinstance(child, BatchSpec):
            node = _batch_per_draw(node, child, sample_shape, name)
        elif _holds_batch(child) and _is_raw_mapping(node):
            node = _with_batch_fields(node, cast(RecordSpec, child), sample_shape)
        fields[name] = node
    return fields


def _is_batch_per_draw(value: Any, sample_shape: tuple[int, ...]) -> bool:
    """Whether *value* is one batch per draw: a batch, or an object array of batches."""
    if not sample_shape:
        return isinstance(value, Batch)
    return _is_object_array(value) and value.shape == sample_shape


def _batch_values(value: Any, spec: BatchSpec, sample_shape: tuple[int, ...]) -> Any:
    """The draws *value* of a batch that *spec* declares, as raw values after the sample axes."""
    if not _is_batch_per_draw(value, sample_shape):
        return value
    if isinstance(value, Batch):
        return _raw_record(value) if isinstance(value, RecordBatch) else value.raw
    stacked = _stacked_batches(value, spec)
    return _unflatten_paths(stacked) if isinstance(stacked, Mapping) else stacked


def _batch_per_draw(value: Any, spec: BatchSpec, sample_shape: tuple[int, ...], label: str) -> Any:
    """The draws *value* of a batch that *spec* declares, as one batch per draw labeled *label*."""
    if _is_batch_per_draw(value, sample_shape):
        return value
    if not sample_shape:
        return _batch_over(value, spec, label)
    batches = np.empty(sample_shape, dtype=object)
    for index in np.ndindex(*sample_shape):
        batches[index] = _batch_over(_entry_at(value, index), spec, label)
    return batches


def _entry_at(value: Any, index: tuple[int, ...]) -> Any:
    """The entry at *index* of the leading axes of an array or a nested mapping of arrays."""
    if isinstance(value, Mapping):
        return {key: _entry_at(entry, index) for key, entry in value.items()}
    return value[index]


def _view_mean(self: CoupledMarginal) -> Any:
    """Projection: the parent's mean at the coupled marginal's path, since ``𝔼[pi X] = pi 𝔼[X]``.

    A parent without a mean gives the mean of its exact marginal at the path.
    """
    if not isinstance(self._parent, SupportsMean):
        return _marginal_moment(self, "_mean")
    return self._project(self._parent._mean())


def _view_variance(self: CoupledMarginal) -> Any:
    """Restriction of the parent's variance to the coordinates of the coupled marginal's path.

    A parent without a variance gives the variance of its exact marginal at the
    path.
    """
    if not isinstance(self._parent, SupportsVariance):
        return _marginal_moment(self, "_variance")
    return self._project(self._parent._variance())


def _view_cov(self: CoupledMarginal) -> LinOp:
    """The sub-block ``P Σ Pᵀ`` of the parent's covariance, ``P`` selecting the path.

    ``P`` selects the coupled marginal's coordinates of the parent's flat vector, in the
    parent's order, and the product stays lazy. A parent without a covariance
    gives the covariance of its exact marginal at the path.

    Parameters
    ----------
    self : CoupledMarginal
        The coupled marginal whose capability subclass installs this function as ``_cov``.

    Returns
    -------
    LinOp
        A ``(k, k)`` operator, where ``k`` is the size of the coupled marginal's flat vector.

    Raises
    ------
    TypeError
        If the parent's event spec is not numeric.
    """
    if not isinstance(self._parent, SupportsCovariance):
        return _marginal_moment(self, "_cov")
    cov = self._parent._cov()
    rows = jnp.asarray(self._coordinates())
    selection = DenseLinOp(jnp.eye(cov.shape[1], dtype=cov.dtype)[rows])
    return selection @ cov @ selection.T


def _view_quantile(self: CoupledMarginal, q: ArrayLike) -> Any:
    """Restriction of the parent's per-coordinate quantiles to the coupled marginal's node.

    The parent's quantiles are its event's raw form with the level axes
    leading in each leaf, so the coupled marginal's quantiles are the parent's at the
    coupled marginal's node, projected as a draw is, with the level axes leading in each
    leaf. A parent without quantiles gives those of its exact marginal at the
    path.
    """
    if not isinstance(self._parent, SupportsQuantile):
        return _marginal_moment(self, "_quantile", q)
    return self._project(self._parent._quantile(q))


def _view_expectation(self: CoupledMarginal, f: Callable[[Any], Array]) -> Array:
    """Composition: the parent's exact expectation of ``f ∘ pi``."""
    return self._parent._expectation(lambda value: f(self._project(value)))


def _view_log_prob(self: CoupledMarginal, value: Any) -> Array:
    """The log-density of the parent's standalone marginal at the path.

    Parameters
    ----------
    self : CoupledMarginal
        The coupled marginal whose capability subclass installs this function as
        ``_log_prob``.
    value : Any
        A value of the coupled marginal's event, or a batch of them.

    Returns
    -------
    Array
        The marginal's log-density at *value*.

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


def _view_unnormalized_log_prob(self: CoupledMarginal, value: Any) -> Array:
    """The unnormalized log-density of the parent's standalone marginal at the path.

    Parameters
    ----------
    self : CoupledMarginal
        The coupled marginal whose capability subclass installs this function as
        ``_unnormalized_log_prob``.
    value : Any
        A value of the coupled marginal's event, or a batch of them.

    Returns
    -------
    Array
        The marginal's unnormalized log-density at *value*.

    Raises
    ------
    TypeError
        If that marginal has no density.
    """
    marginal = self._parent._marginal(self._path)
    if not isinstance(marginal, SupportsUnnormalizedLogProb):
        raise TypeError(f"the marginal of {self._parent.label!r} at {self._path!r} has no density")
    return marginal._unnormalized_log_prob(value)


def _view_log_prob_guard(self: CoupledMarginal) -> Feasibility:
    """The parent's marginal guard at the coupled marginal's path.

    The parent's report of that marginal's capabilities, read when the coupled marginal was
    constructed, decides whether the coupled marginal claims the density.
    """
    return _capability_guard(self._parent, "_marginal", self._path)


def _view_marginal(self: CoupledMarginal, paths: str | tuple[str, ...]) -> Distribution:
    """Path composition: the parent's marginal at the parent's paths for *paths*.

    *paths* is an event path of the coupled marginal, or a list of them, and the result is
    indexed by the paths of the coupled marginal. The result carries the expression
    ``marginal`` gives it: the parent's factors whose whole events the paths
    select, and otherwise the coupled marginal selected at *paths*.

    Parameters
    ----------
    self : CoupledMarginal
        The coupled marginal whose capability subclass installs this function as
        ``_marginal``.
    paths : str or tuple of str
        An event path of the coupled marginal, or a list of them.

    Returns
    -------
    Distribution
        The marginal, which holds no reference to the parent.

    Raises
    ------
    KeyError
        If a path is not an event path of the coupled marginal.
    """
    request = _ordered_paths_of(self, paths)
    parent_request = self._parent_request(request)
    marginal = self._parent._marginal(parent_request.paths)
    if parent_request.group is not None:
        marginal = _law_of_group(marginal, parent_request.group)
    if parent_request.renames:
        marginal = marginal.rename(parent_request.renames)
    return _carrying(marginal, _marginal_expression_at(self, request))


def _view_marginal_guard(self: CoupledMarginal, paths: str | tuple[str, ...]) -> Feasibility:
    """The parent's marginal guard at the parent's paths for *paths*, paths of the coupled marginal.

    For a group that the coupled marginal's list of paths cut down, the parent's paths are
    the selected paths below it.
    """
    try:
        parent_request = self._parent_request(_ordered_paths_of(self, paths))
    except (KeyError, TypeError) as error:
        return Feasibility(False, str(error.args[0]))
    return _capability_guard(self._parent, "_marginal", parent_request.paths)


def _view_marginal_capabilities(
    self: CoupledMarginal, paths: str | tuple[str, ...]
) -> frozenset[type]:
    """The parent's report of its marginal at the parent's paths for *paths*, paths of the coupled marginal.

    Parameters
    ----------
    self : CoupledMarginal
        The coupled marginal whose capability subclass installs this function as
        ``_marginal_capabilities``.
    paths : str or tuple of str
        An event path of the coupled marginal, or a list of them.

    Returns
    -------
    frozenset of type
        The capability protocols the parent reports for that marginal. For a
        group that the coupled marginal's list of paths cut down, the parent's paths are
        the selected paths below it.

    Raises
    ------
    KeyError
        If a path is not an event path of the coupled marginal.
    """
    parent_request = self._parent_request(_ordered_paths_of(self, paths))
    return _marginal_claims(self._parent, parent_request.paths)


def _law_of_group(law: Distribution, group: str) -> Distribution:
    """The law of the group at *group*, whole under its last name, from *law*.

    *law* exposes the record that keeps a list of paths at their paths, each
    below *group*, so *group* is a chain of groups above them. The group moves
    to the top, and the law of the record's one field is that group whole.
    """
    name = _last_name(group)
    return _SoleField(law if group == name else law.rename({group: name}))


def _view_condition_on(self: CoupledMarginal, given: Any, /, **options: Any) -> Distribution:
    """Conditioning commutes with marginalization: the parent conditioned, then viewed.

    *given* is keyed by event paths of the coupled marginal. The parent is conditioned at
    the parent's paths for them, and the result is the coupled marginal of the conditioned
    parent at each of the coupled marginal's nodes that keeps a field unconditioned. A
    value given at a group that the coupled marginal's list of paths cut down conditions
    the parent at each selected path below the group.

    Parameters
    ----------
    self : CoupledMarginal
        The coupled marginal whose capability subclass installs this function as
        ``_condition_on``.
    given : Record or Mapping[str, Any]
        The observed values, keyed by event paths of the coupled marginal.
    **options : Any
        Options passed to the parent's ``_condition_on``.

    Returns
    -------
    Distribution
        The coupled marginal of the conditioned parent, at the coupled marginal's paths.

    Raises
    ------
    KeyError
        If a key of *given* is not an event path of the coupled marginal.
    ValueError
        If *given* covers every field of the coupled marginal.
    """
    items = list(given.items())
    kept = self._kept_nodes([path for path, _ in items])
    if not kept:
        raise ValueError(_fixes_every_field(self.label))
    parent_given: dict[str, Any] = {}
    for path, value in items:
        parent_request = self._parent_request(path)
        if parent_request.group is None:
            parent_given[cast(str, parent_request.paths)] = value
            continue
        raw = _raw_record(value)
        for selected in parent_request.paths:
            parent_given[selected] = _extract(raw, tuple(_below(selected, path).split(_PATH_SEP)))
    conditioned = self._parent._condition_on(parent_given, **options)
    return CoupledMarginal(conditioned, self._path if isinstance(self._path, str) else kept)


def _view_condition_on_guard(self: CoupledMarginal, paths: tuple[str, ...]) -> Feasibility:
    """The parent's conditioning guard at the parent's paths for *paths*, paths of the coupled marginal.

    Some field of the coupled marginal must remain unconditioned.
    """
    parent_paths: list[str] = []
    for path in paths:
        try:
            parent_request = self._parent_request(path)
        except (KeyError, TypeError) as error:
            return Feasibility(False, str(error.args[0]))
        parent_paths.extend(_path_tuple(parent_request.paths))
    if not self._kept_nodes(paths):
        return Feasibility(False, _fixes_every_field(self.label))
    return _capability_guard(self._parent, "_condition_on", tuple(parent_paths))


def _ordered_paths_of(
    law: Distribution | ConditionalDistribution, paths: str | tuple[str, ...]
) -> str | tuple[str, ...]:
    """*paths*, one path or a list of paths of *law*'s event, a list in its canonical order.

    Parameters
    ----------
    law : Distribution or ConditionalDistribution
        The term whose event the paths address.
    paths : str or tuple of str
        One path, or a list of paths.

    Returns
    -------
    str or tuple of str
        The one path, or the list of paths in canonical order.

    Raises
    ------
    KeyError
        If *law* is over a record and a path is not one of its paths.
    TypeError
        If *law* is over a value with no paths, such as an array.
    """
    return _read_event_paths(
        law.event_spec, law.label, paths, action=f"take a marginal of {law.label!r} at"
    )


@dataclass(frozen=True)
class _ParentRequest:
    """A request of a field coupled marginal, a path or a list of paths of its event, stated on its parent.

    Attributes
    ----------
    paths : str or tuple of str
        The parent's path, or its list of paths in the parent's canonical
        order, at which the parent's marginal or coupled marginal answers the request.
    group : str or None
        For one path of a coupled marginal over a list of paths that names a group the list
        cut down, that path: the parent's law at *paths* keeps the group at its
        path, and the request is the law of that group (:func:`_law_of_group`).
        Otherwise None.
    renames : Mapping[str, str] or None
        For a coupled marginal over one path whose output differs from the parent's
        node path, the rename the parent's law at *paths* takes to be indexed
        by the coupled marginal's paths: for a list of paths, the move of the parent's node
        to the output, and for the output itself, the rename of the
        whole term to it. Otherwise None.
    """

    paths: str | tuple[str, ...]
    group: str | None = None
    renames: Mapping[str, str] | None = None


def _parent_guard(method: str, owner: str = "CoupledMarginal") -> Callable[..., Feasibility]:
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

    def guard(self: CoupledMarginal, *arguments: Any) -> Feasibility:
        if isinstance(self._parent, protocol):
            return _capability_guard(self._parent, method, *arguments)
        exact = _capability_guard(self._parent, "_marginal", self._path)
        if exact.feasible is not True:
            return exact
        marginal = self._parent._marginal(self._path)
        if not isinstance(marginal, protocol):
            return Feasibility(
                False,
                f"the marginal of {self._parent.label!r} at {self._path!r} does not support "
                f"{protocol.__name__}",
            )
        return _capability_guard(marginal, method, *arguments)

    guard.__name__ = f"{method}_guard"
    guard.__qualname__ = f"CoupledMarginal.{method}_guard"
    guard.__doc__ = (
        f"The parent's guard of ``{method}``, or the parent's marginal guard at the path "
        f"followed by that marginal's guard of ``{method}``."
    )
    return guard


#: Each capability a coupled marginal may derive, with the methods that realize it and their
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
    parent: Distribution, node: TermSpec, paths: str | tuple[str, ...]
) -> set[type]:
    """The capabilities a coupled marginal of *parent* at *paths* derives, over the event declared by *node*.

    The projection rows derive from the parent's own capabilities. The density
    rows, and the moment rows the parent does not claim, derive from the
    parent's report of its marginal at *paths*, read once, here: the coupled marginal claims
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
        report = _marginal_claims(parent, paths)
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
# CoupledMarginal
# ---------------------------------------------------------------------------


class CoupledMarginal(Distribution):
    """The law of the node at one event path of a parent law, or of the nodes at a list of paths.

    ``d[path]`` returns a ``CoupledMarginal`` for a node below the law's whole term,
    and ``d[paths]`` for a list or a tuple of paths; it is never constructed by
    hand. The coupled marginal holds a reference to its parent rather than a standalone law,
    so sibling coupled marginals co-sample from one parent draw and the correlation between
    them is preserved. For one path, its event spec is the parent's schema at
    the path, the leaf or group whole, under an output named by the path's
    last name. For a list of paths, its event spec is the positional form over
    the record that keeps each node at its own path, in the parent's order
    whatever the order of the list, so ``grouped["g/e", "a"]`` draws records
    ``{a, g: {e}}``.

    The coupled marginal displays as the standalone marginal at its path does, and its repr
    names the class and the path, which distinguishes it from that marginal.
    It is labeled as ``marginal`` labels the marginal there (VI.8), so
    ``model["y"]`` reads as ``model(y)``, and ``model["mu"]`` at the factor
    ``prior`` reads as ``prior(mu)``. A coupled marginal of the whole events of several
    factors reads factor by factor, as ``a(a)·b(b)`` (:attr:`notation`). The
    coupled marginal holds the paths its parent holds fixed, after those of the one factor
    it is when it is one.

    Its capabilities are derived from the parent's, one by one:

    ======================================  ========================================
    capability on the coupled marginal                  available when
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
    marginal's guard of the moment, so a coupled marginal of a dependent joint at a root
    factor takes the factor's moments. The projection rows are exact whenever
    the parent's answer is, and only sampling requires the parent to sample.
    The parent reports what its marginal at the path claims through
    ``_marginal_capabilities``, and a parent that defines none reports its own
    claims; the coupled marginal reads the report once, at construction.

    The coupled marginal's ``raw`` is the raw form of the parent's marginal at the path,
    ``parent._marginal(path).raw``: the backend object where the marginal
    has one, such as the TFP distribution of a TFP family, and the standalone
    law otherwise. ``with_dim_sizes`` and ``with_dim_names`` apply to the
    parent and return the coupled marginal of the result at the same path.

    Parameters
    ----------
    parent : Distribution
        The law whose event the coupled marginal reads.
    path : str, or list or tuple of str
        An event path of *parent*, a slash path that starts with an output's
        name, or a list of them.

    Raises
    ------
    KeyError
        If *parent* is over a record and a path is not one of its paths.
    TypeError
        If *parent* is not a ``Distribution``, *path* is neither a string nor
        a list or tuple of strings, or *parent* is over a value with no paths.
    ValueError
        If a path is empty or has an empty name, or a list of paths is empty,
        names a path twice, or holds a path inside another.
    """

    _capability_table = _VIEW_CAPABILITIES

    def __new__(cls, parent: Distribution, path: PathsLike) -> CoupledMarginal:
        if not isinstance(parent, Distribution):
            raise TypeError(
                f"CoupledMarginal: parent must be a Distribution, got {type(parent).__name__}"
            )
        path = _read_event_paths(
            parent.event_spec, parent.label, path, action=f"index {parent.label!r} by"
        )
        declaration = _view_declaration(parent.event_spec, path)
        return object.__new__(
            _capability_subclass(
                CoupledMarginal, _derived_protocols(parent, declaration.term_spec, path)
            )
        )

    def __init__(self, parent: Distribution, path: PathsLike) -> None:
        path = _read_event_paths(
            parent.event_spec, parent.label, path, action=f"index {parent.label!r} by"
        )
        declaration = _view_declaration(parent.event_spec, path)
        self._init_tracked(parent.label)
        self._init_annotations(None)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_path", path)
        self._init_declaration(declaration)
        self._store_expression(_marginal_expression_at(parent, path))
        self.with_provenance(
            Provenance.create("__getitem__", parents=[parent], metadata={"path": path})
        )

    def _derived_traceability(self) -> Traceability:
        """The parent's traceability, since every method of the coupled marginal reads the parent."""
        return self._parent._traceability()

    @property
    def parent(self) -> Distribution:
        """The law this coupled marginal reads, which sibling coupled marginals share."""
        return self._parent

    @property
    def path(self) -> str | tuple[str, ...]:
        """The event path of the parent that this coupled marginal reads, or the tuple of them in the parent's order."""
        return self._path

    def _selected_paths(self) -> tuple[str, ...]:
        """The parent's paths of the nodes this coupled marginal reads, in the parent's order."""
        return _path_tuple(self._path)

    def _parent_request(self, request: str | tuple[str, ...]) -> _ParentRequest:
        """*request*, a path or a list of paths of this coupled marginal in its order, stated on the parent.

        A coupled marginal over one path names its node by the path's last name, so a path
        of the coupled marginal is the parent's path with the output in place of the
        node's path, and a list of paths moves the parent's node back to the
        output. A coupled marginal over a list of paths keeps the parent's paths, so a
        path at or below a selected node is the parent's own, and a group that
        the list cut down stands for the selected paths below it.

        Parameters
        ----------
        request : str or tuple of str
            One path of this coupled marginal, or a list of them in its order.

        Returns
        -------
        _ParentRequest
            The request stated on the parent.

        Raises
        ------
        KeyError
            If this coupled marginal is over a record and a path is not one of its paths.
        TypeError
            If this coupled marginal is over a value with no paths, such as an array.
        """
        requested = _path_tuple(request)
        for path in requested:
            if path not in self.event_spec:
                listed = requested if isinstance(request, tuple) else ()
                raise _not_an_event_path(self.event_spec, self.label, path, listed)
        if isinstance(self._path, str):
            output_name = cast(str, _root_output_name(self.event_spec))
            parent_paths = tuple(self._path + path[len(output_name) :] for path in requested)
            if isinstance(request, str):
                # A coupled marginal renamed by its output names the node otherwise than
                # the parent's path does.
                node_name = _last_name(self._path)
                if request != output_name or node_name == output_name:
                    return _ParentRequest(parent_paths[0])
                return _ParentRequest(parent_paths[0], renames={node_name: output_name})
            renames = None if self._path == output_name else {self._path: output_name}
            return _ParentRequest(parent_paths, renames=renames)
        if isinstance(request, str):
            if self._is_selected(request):
                return _ParentRequest(request)
            return _ParentRequest(self._selected_below(request), group=request)
        parent_paths: list[str] = []
        for path in requested:
            parent_paths.extend((path,) if self._is_selected(path) else self._selected_below(path))
        return _ParentRequest(tuple(parent_paths))

    def _is_selected(self, path: str) -> bool:
        """Whether *path*, a path of this coupled marginal, is at or below a node the coupled marginal reads."""
        return any(_is_within(path, selected) for selected in self._selected_paths())

    def _selected_below(self, group: str) -> tuple[str, ...]:
        """The paths of the nodes this coupled marginal reads below *group*, a group the list of paths cut down."""
        return tuple(selected for selected in self._selected_paths() if _is_within(selected, group))

    def _parent_path(self, path: Any) -> str | None:
        """The parent's path for *path*, an event path of this coupled marginal, or None if it has none.

        A path has none when it is not an event path of the coupled marginal, or names a
        group that the coupled marginal's list of paths cut down.
        """
        if not isinstance(path, str):
            return None
        try:
            request = self._parent_request(path)
        except (KeyError, TypeError):
            return None
        return None if request.group is not None else cast(str, request.paths)

    def _kept_nodes(self, paths: Sequence[str]) -> tuple[str, ...]:
        """The parent's paths of the nodes this coupled marginal reads that keep a field no path of *paths* covers.

        *paths* are paths of the coupled marginal.
        """
        leaves = [
            leaf
            for leaf in _leaf_paths(self.event_spec)
            if not any(_is_within(leaf, path) for path in paths)
        ]
        if isinstance(self._path, str):
            return (self._path,) if leaves else ()
        return tuple(
            selected
            for selected in self._path
            if any(_is_within(leaf, selected) for leaf in leaves)
        )

    def _project(self, value: Any) -> Any:
        """pi: the coupled marginal's node of *value*, a raw parent draw or a value shaped like one.

        The node is in raw form: a leaf's raw value, or the nested mapping of a
        group's raw leaves, and a list of paths gives the nested mapping that
        keeps each node at its path. Leading batch axes stay on every leaf. A
        ``Record`` or a batch of records is read through that raw form.
        """
        return _projector(self._parent.event_spec, self._path)(_raw_record(value))

    def _coordinates(self) -> list[int]:
        """The coupled marginal's coordinates of the parent's flat vector, in the coupled marginal's order.

        Returns
        -------
        list of int
            The indices into the parent's flat vector, node by node.

        Raises
        ------
        TypeError
            If the parent's event spec is not numeric.
        """
        return [
            index
            for path in self._selected_paths()
            for index in _coordinate_range(self._parent.event_spec, path)
        ]

    def __getitem__(self, paths: PathsLike, /) -> Distribution:
        """This coupled marginal itself, or the law at a path or a list of paths of the coupled marginal.

        A tuple is a list of paths; a nested path is written with ``/``.
        A path of a coupled marginal over one path starts with its output's name, so
        ``d["a"]["a/b/c"]`` reads the node that ``d["a/b/c"]`` reads. A list of
        paths of the coupled marginal gives the law whose event keeps each node at its path
        of the coupled marginal. A coupled marginal over a list of paths keeps its parent's paths, so
        ``d["a", "g/c"]["g/c"]`` reads the node that ``d["g/c"]`` reads. The
        result is labeled as the marginal of this coupled marginal at *paths* is labeled
        (:func:`marginal`), and it holds the paths this coupled marginal holds fixed.

        Parameters
        ----------
        paths : str, or list or tuple of str
            An event path of the coupled marginal, or a list of them.

        Returns
        -------
        Distribution
            This coupled marginal itself, or a law that reads the parent.

        Raises
        ------
        KeyError
            If the coupled marginal is over a record and a path is not one of its paths.
        TypeError
            If *paths* is neither a string nor a list or tuple of strings, or
            the coupled marginal is over a value with no paths, such as an array.
        ValueError
            If a path is empty or has an empty name, or a list of paths is
            empty, names a path twice, or holds a path inside another.
        """
        request = _read_event_paths(
            self.event_spec,
            self.label,
            paths,
            action=f"index {self.label!r} by",
        )
        parent_request = self._parent_request(request)
        if parent_request.group is not None:
            return CoupledMarginal(self, request)
        if parent_request.paths == self._path:
            return self
        coupled_marginal = self._sub_view(request, parent_request.paths)
        return (
            coupled_marginal.rename(parent_request.renames)
            if parent_request.renames
            else coupled_marginal
        )

    def _sub_view(
        self, key: str | tuple[str, ...], parent_path: str | tuple[str, ...]
    ) -> CoupledMarginal:
        """The parent's coupled marginal at *parent_path*, for *key*, labeled as this coupled marginal's marginal at *key*."""
        coupled_marginal = CoupledMarginal(self._parent, parent_path)
        coupled_marginal._store_expression(_marginal_expression_at(self, key))
        return coupled_marginal

    @property
    def raw(self) -> Any:
        """The raw form of the parent's marginal at the path, ``parent._marginal(path).raw``.

        Every read returns the same object, resolved on the first read. A
        standalone law refuses provenance, since every reader shares it.

        Returns
        -------
        Any
            The marginal's backend object where it has one, such as the TFP
            distribution of a TFP family. Otherwise the marginal standalone: a
            standalone law with no reference to the parent, and no provenance
            or annotations, which carries the coupled marginal's expression, so it is
            labeled as ``marginal`` labels it and holds the paths the parent
            holds fixed.

        Raises
        ------
        ResolutionError
            If the parent has no exact marginal at the path. The error is not
            an ``AttributeError``, so ``hasattr(coupled marginal, "raw")`` raises it rather
            than returning ``False``, and each read tries the marginal again.
        """
        return _cached_raw(self, self._resolved_raw)

    def _resolved_raw(self) -> Any:
        """Resolve the raw form of the parent's marginal at the path, as :attr:`raw` describes."""
        parent = self._parent
        if not isinstance(parent, SupportsMarginals):
            raise ResolutionError(
                f"{parent.label!r} has no marginals, so the coupled_marginal {self.label!r} has no standalone law"
            )
        report = _capability_guard(parent, "_marginal", self._path)
        if report.feasible is not True:
            reason = report.description or "; ".join(report.pending)
            raise ResolutionError(
                f"{parent.label!r} has no exact marginal at {self._path!r}: {reason}"
            )
        raw = parent._marginal(self._path).raw
        if not isinstance(raw, Distribution):
            return raw
        return _carrying(raw, self._expression)

    def with_dim_sizes(self, **sizes: int) -> CoupledMarginal:
        """Bind named symbolic dimensions in the parent, and coupled marginal the result at the same path.

        The coupled marginal's event spec is the parent's schema at the path, and a schema
        is one dimension scope, so a dimension binds wherever the parent
        declares it.

        Parameters
        ----------
        **sizes : int
            Sizes for free dimensions of the coupled marginal's declaration.

        Returns
        -------
        CoupledMarginal
            The coupled marginal, under the same label, of the parent with the sizes bound.

        Raises
        ------
        ValueError
            If a name is not a free dimension of the coupled marginal's declaration.
        """
        unbound = set(sizes) - self.event_spec.term_spec.free_dims
        if unbound:
            raise ValueError(_no_free_dims(self, unbound, self.event_spec.term_spec.free_dims))
        return self._viewed(self._parent.with_dim_sizes(**sizes))

    def with_dim_names(self, **names: str) -> CoupledMarginal:
        """Rename symbolic dimensions in the parent, and coupled marginal the result at the same path.

        Parameters
        ----------
        **names : str
            New names keyed by old; names that are not free are ignored.

        Returns
        -------
        CoupledMarginal
            The coupled marginal, under the same label, of the parent with the dimensions
            renamed.
        """
        return self._viewed(self._parent.with_dim_names(**names))

    def _viewed(self, parent: Distribution) -> CoupledMarginal:
        """The coupled marginal of *parent* at this coupled marginal's path, carrying this coupled marginal's expression."""
        return _carrying(CoupledMarginal(parent, self._path), self._expression)

    def __repr__(self) -> str:
        """``CoupledMarginal(parent, path=...)``, followed by ``.with_label(...)`` for a coupled marginal under another label.

        The constructor gives a coupled marginal the label of its parent's marginal at its
        path, so a coupled marginal that ``with_label`` relabeled reads as the constructor
        call relabeled. The parent's repr shows any paths it holds fixed.
        """
        text = call_repr(
            self._repr_class_name(), [repr(self._parent)], [("path", repr(self._path))]
        )
        if self.label != _marginal_expression_at(self._parent, self._path).render_label():
            text = _method_call(text, "with_label", repr(self._displayed_label()))
        return text


# ---------------------------------------------------------------------------
# Renames of paths and of the values that carry them
# ---------------------------------------------------------------------------


def _draw_path(declaration: OutputSpec, path: str) -> str:
    """*path*, a path of *declaration*, as the path of its node within a draw."""
    return _PATH_SEP.join(_draw_segments(declaration, path))


def _leaves_at(declaration: OutputSpec, path: str) -> list[str]:
    """The paths of the leaves of *declaration* at or below *path*, in canonical order."""
    return [leaf for leaf in _leaf_paths(declaration) if _is_within(leaf, path)]


def _below(path: str, node: str) -> str:
    """*path*, a path at or below *node*, relative to *node*."""
    return path[len(node) + 1 :]


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
        columns, template = value._raw_columns(), value.element_spec.term_spec
        return value._rebuilt(
            {new: columns[old] for new, old in moves.items()},
            RecordSpec({new: template[old] for new, old in moves.items()}),
        )
    if isinstance(value, Record):
        template = value.event_template
        return Record(
            {new: value[old] for new, old in moves.items()},
            event_template=RecordSpec({new: template[old] for new, old in moves.items()}),
            label=value.label,
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
    """The event spec of the leaves *specs*, keyed by their paths, in the form *like* allows.

    The positional form over a record keeps that form, and a whole term stays
    whole while its leaves lie under one output.
    """
    record = RecordSpec(dict(specs))
    if _has_record_outputs(like) or len(record.children) != 1:
        return OutputSpec(record)
    ((output_name, term),) = record.children.items()
    return OutputSpec(**{output_name: term})


@dataclass(frozen=True)
class _EventRenames:
    """The renames and moves that take an event spec to a renamed one.

    The renamed event spec holds each field of the original at a new path, and
    maybe in a new order, so the values of a law over the original translate to
    it field by field: a draw of the positional form over a record carries the
    paths of its record, and a draw of a whole term the paths below its output.
    A node of the renamed event spec holds a node of the original when its
    fields are exactly that node's, and a path of the renamed law addresses the original's
    marginals and conditioning through the node it holds.

    Attributes
    ----------
    declaration : OutputSpec
        The original event spec.
    renamed : OutputSpec
        The renamed event spec.
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
        paths = _output_record(declaration)
        moved = paths._moved_leaf_paths(paths._resolve_path_renames(renames, {}))
        return cls(declaration, renamed, moved)

    def followed_by(self, then: _EventRenames) -> _EventRenames:
        """These renames followed by *then*, which starts from this renamed event spec."""
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

    @property
    def keeps_paths(self) -> bool:
        """Whether every leaf keeps its path and every field of a draw its position.

        The renames of a relabeling or a dimension transform keep the paths, and
        their ``draw`` returns its value.
        """
        return self._draw_moves is None and all(old == new for old, new in self.leaves.items())

    def draw(self, value: Any) -> Any:
        """*value*, a raw value of the original event spec or a batch of them, under the new paths.

        The fields of a batch of records take the new paths in the renamed
        event spec's order, and the batch keeps its label and levels.
        """
        return _moved_value(value, self._draw_moves)

    def undraw(self, value: Any) -> Any:
        """*value*, a raw value of the renamed event spec, under the original paths."""
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
        """The original node that *path*, a path of the renamed event spec, holds, or None."""
        return self._nodes.get(path)

    def undraw_at(self, path: str, value: Any) -> Any:
        """*value*, the node at *path* of the renamed event spec, as the original node it holds."""
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
        """*law*, over the original event spec or the part of it that remains, renamed.

        Each field that *law* still declares takes its new path, in the renamed
        event spec's order, so a law conditioned on some fields keeps the paths
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
        self,
        law: Distribution,
        paths: Sequence[str],
        originals: Sequence[str],
        *,
        list_of_paths: bool,
    ) -> Distribution:
        """*law*, the marginal at the original nodes *originals*, as the marginal at *paths*.

        A marginal at one path names its node by the path's last name, so each
        field of the original node takes the last name of the renamed path
        followed by the field's path below the renamed node. A marginal at a
        list of paths keeps each node at its path, so each field of an
        original node, which *law* keeps at its original path, takes its
        renamed path. Either way the fields come in the renamed order.

        Parameters
        ----------
        law : Distribution
            The parent's marginal at *originals*.
        paths : sequence of str
            Paths of the renamed declaration, in its canonical order.
        originals : sequence of str
            The original node each of *paths* holds.
        list_of_paths : bool
            Whether the marginal is at a list of paths rather than at one path.

        Returns
        -------
        Distribution
            *law* under the renamed paths.
        """
        moved: dict[str, str] = {}
        specs: dict[str, TermSpec] = {}
        for path, original in zip(paths, originals, strict=True):
            held = {self.leaves[old]: old for old in _leaves_at(self.declaration, original)}
            for new in _leaves_at(self.renamed, path):
                if list_of_paths:
                    field, target = held[new], new
                else:
                    field = _last_name(original) + held[new][len(original) :]
                    target = _last_name(path) + new[len(path) :]
                moved[field] = target
                specs[target] = _node_at(law.event_spec, field)
        if list_of_paths:
            return _renamed_by_leaves(law, OutputSpec(RecordSpec(specs)), moved)
        return _renamed_by_leaves(law, _declared_like(law.event_spec, specs), moved)


def _original_nodes(
    event: _EventRenames, event_spec: OutputSpec, paths: Sequence[str], parent: str
) -> list[str]:
    """The node of the parent *parent* that each of *paths*, paths of *event_spec*, holds.

    Parameters
    ----------
    event : _EventRenames
        The renames from the parent's event spec to *event_spec*.
    event_spec : OutputSpec
        The renamed event spec.
    paths : sequence of str
        Paths of *event_spec*.
    parent : str
        The parent's label, which error messages name.

    Returns
    -------
    list of str
        The parent's path of each held node, in the order of *paths*.

    Raises
    ------
    KeyError
        If a path is not an event path of *event_spec*.
    ValueError
        If a path holds no single node of the parent, since a move gathered or
        regrouped its fields.
    """
    originals = []
    for path in paths:
        if path not in event_spec:
            raise KeyError(path)
        original = event.original(path)
        if original is None:
            raise ValueError(_regrouped_path(path, parent))
        originals.append(original)
    return originals


def _regrouped_path(path: str, owner: str) -> str:
    """The message that *path* gathers fields that ``rename`` moved from several places."""
    return (
        f"{path!r} does not correspond to a single field of {owner!r} because "
        f"rename() regrouped fields into it; select its fields individually"
    )


def _unreached(
    event: _EventRenames, event_spec: OutputSpec, paths: Sequence[str], owner: str
) -> Feasibility | None:
    """Why *paths*, of the law *owner* over *event_spec*, miss the parent, or None."""
    for path in paths:
        if path not in event_spec:
            return Feasibility(False, f"{path!r} is not an event path of {owner!r}")
        if event.original(path) is None:
            return Feasibility(False, _regrouped_path(path, owner))
    return None


def _exposed_under(law: Distribution, output_name: str) -> Distribution:
    """*law* declared as the exposed record that holds its event under *output_name*.

    A law whose event is the whole term *output_name* keeps its paths, and a law
    that exposes a record has its paths moved below *output_name*, so a list of
    paths of a packaged joint keeps the joint's output above the kept nodes.
    """
    source = law.event_spec
    whole = _root_output_name(source) is not None
    leaves = {
        leaf: leaf if whole else f"{output_name}{_PATH_SEP}{leaf}" for leaf in _leaf_paths(source)
    }
    return _renamed_by_leaves(law, OutputSpec(RecordSpec({output_name: source.term_spec})), leaves)


def _renamed_by_leaves(
    law: Distribution, target: OutputSpec, leaves: Mapping[str, str]
) -> Distribution:
    """*law* declared by *target*, each of its leaves at the path that *leaves* gives it.

    A whole term whose draws keep their fields is renamed by its output's name,
    which keeps the law's class.
    """
    source = law.event_spec
    if target == source:
        return law
    event = _EventRenames(source, target, dict(leaves))
    if event._draw_moves is None and not (
        _has_record_outputs(source) or _has_record_outputs(target)
    ):
        (output_name,), (renamed,) = source.outputs, target.outputs
        return law.rename({output_name: renamed})
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
    """The parent's covariance, over the flat coordinates in the renamed event spec's order."""
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


def _renamed_marginal(self: _RenamedDistribution, paths: str | tuple[str, ...]) -> Distribution:
    """The parent's marginal at the original nodes for *paths*, named and arranged by *paths*.

    The marginal is this law selected at *paths*, so it keeps this law's label.

    Parameters
    ----------
    self : _RenamedDistribution
        The renamed law whose capability subclass installs this function as
        ``_marginal``.
    paths : str or tuple of str
        An event path of this law, or a tuple of them.

    Returns
    -------
    Distribution
        The marginal, which holds no reference to the parent.

    Raises
    ------
    KeyError
        If a path is not an event path of this law.
    ValueError
        If a path holds no single node of the parent.
    """
    request = _ordered_paths_of(self, paths)
    listed_paths = _path_tuple(request)
    originals = self._originals(listed_paths)
    marginal = self._parent._marginal(
        originals[0] if isinstance(request, str) else tuple(originals)
    )
    return _carrying(
        self._event.marginal(
            marginal, listed_paths, originals, list_of_paths=isinstance(request, tuple)
        ),
        _marginal_expression_at(self, request),
    )


def _renamed_marginal_guard(
    self: _RenamedDistribution, paths: str | tuple[str, ...]
) -> Feasibility:
    """The parent's marginal guard at the original nodes for *paths*, paths of this law.

    Each path must hold one node of the parent.
    """
    listed_paths = (paths,) if isinstance(paths, str) else tuple(paths)
    unreached = _unreached(self._event, self.event_spec, listed_paths, self.label)
    if unreached is not None:
        return unreached
    originals = self._originals(listed_paths)
    return _capability_guard(
        self._parent, "_marginal", originals[0] if isinstance(paths, str) else tuple(originals)
    )


def _renamed_marginal_capabilities(
    self: _RenamedDistribution, paths: str | tuple[str, ...]
) -> frozenset[type]:
    """The parent's report of its marginal at the original nodes for *paths*.

    A path that holds no single node of the parent, as a regrouping node does,
    has no exact marginal here, which the marginal guard reports, so its report
    claims nothing and a coupled marginal there claims no density.

    Parameters
    ----------
    self : _RenamedDistribution
        The renamed law whose capability subclass installs this function as
        ``_marginal_capabilities``.
    paths : str or tuple of str
        An event path of this law, or a tuple of them.

    Returns
    -------
    frozenset of type
        The capability protocols the parent reports for that marginal.

    Raises
    ------
    KeyError
        If a path is not an event path of this law.
    """
    listed_paths = (paths,) if isinstance(paths, str) else tuple(paths)
    for each in listed_paths:
        if each not in self.event_spec:
            raise KeyError(each)
    if any(self._event.original(each) is None for each in listed_paths):
        return frozenset()
    originals = self._originals(listed_paths)
    return _marginal_claims(
        self._parent, originals[0] if isinstance(paths, str) else tuple(originals)
    )


def _renamed_condition_on(
    self: _RenamedDistribution, given: Any, /, **options: Any
) -> Distribution:
    """The parent conditioned on *given* under the original names, then renamed.

    Parameters
    ----------
    self : _RenamedDistribution
        The renamed law whose capability subclass installs this function as
        ``_condition_on``.
    given : Record or Mapping[str, Any]
        The observed values, keyed by event paths of this law.
    **options : Any
        Options passed to the parent's ``_condition_on``.

    Returns
    -------
    Distribution
        The conditioned parent, with each remaining field at its new path.

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

    ``Distribution.rename`` returns it for a rename that changes the
    path of a field of a record draw when the parent's family does not rebuild
    itself under the new paths. It claims each of the parent's capabilities
    that a rename carries over: a draw, a moment, a quantile, or a marginal is
    translated on the way out, and a scored value, a given, or a path on the
    way in. A move reorders the flat coordinates, so the covariance, and
    quantiles over the flat coordinates, are permuted into the renamed
    event spec's order. Each capability carries the parent's guard at the
    original nodes. A further rename of this law renames its parent by both
    renames at once.

    Parameters
    ----------
    parent : Distribution
        The law whose values are translated.
    event : _EventRenames
        The renames from the parent's event spec to this law's.
    renames : tuple of Mapping[str, str]
        The renames applied to the parent, in order, each as its caller gave
        it, which the repr shows.
    """

    _capability_table = _RENAMED_CAPABILITIES

    def _derived_traceability(self) -> Traceability:
        """The parent's traceability, since a rename only translates values at the boundary."""
        return self._parent._traceability()

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

    def rename(
        self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str
    ) -> Distribution:
        """Rename or move nodes of the event spec, as :meth:`Distribution.rename`.

        The result renames this law's parent by this law's renames followed by
        the new ones, so a rename of the output alone keeps the renames of
        the fields below it.
        """
        renamed_event_spec = self.event_spec.rename(renames, **keyword_renames)
        all_renames = {**dict(renames or {}), **keyword_renames}
        return _renamed(
            self, _EventRenames.of(self.event_spec, renamed_event_spec, all_renames), all_renames
        )

    def _original_given(self, given: Any) -> Record | None:
        """*given*, keyed by event paths of this law, as the record of the parent's nodes it binds.

        ``None`` when a path holds no single node of the parent, as at a node a
        rename gathered, or is not an event path of this law.
        """
        items = list(given.items())
        if any(self._event.original(path) is None for path, _ in items):
            return None
        return Record(
            {
                self._event.original(path): self._event.undraw_at(path, value)
                for path, value in items
            },
            label="given",
        )

    def _originals(self, paths: Sequence[str]) -> list[str]:
        """The parent's node for each of *paths*, event paths of this law.

        Parameters
        ----------
        paths : sequence of str
            Paths that each start with the name of an output of this law.

        Returns
        -------
        list of str
            One parent path per entry of *paths*, in the same order.

        Raises
        ------
        KeyError
            If a path is not an event path of this law.
        ValueError
            If a path holds no single node of the parent.
        """
        return _original_nodes(self._event, self.event_spec, paths, self._parent.label)

    def __repr__(self) -> str:
        """The parent's repr followed by ``.rename({...})`` for each rename.

        Each rename shows the paths it changed, and a label that differs from
        the parent's follows as ``.with_label(...)``. A reorder that changes no
        path, which no rename can state, reads as a call of the parent's
        constructor with the parent's arguments and this law's event spec as
        ``event_spec=``.
        """
        renames = [{old: new for old, new in step.items() if old != new} for step in self._renames]
        parent = self._parent
        if not all(renames):
            return _law_repr(self, parent._repr_arguments(), parent)
        text = repr(parent)
        for step in renames:
            mapping = mapping_repr({old: repr(new) for old, new in step.items()})
            text = _method_call(text, "rename", mapping)
        if self.label != parent.label:
            text = _method_call(text, "with_label", repr(self._displayed_label()))
        return text

    def _repr_class_name(self) -> str:
        """The class of the law this one renames, which a law presenting this one names."""
        return self._parent._repr_class_name()

    def _event_repr_parts(self) -> tuple[str | None, str | None]:
        """The output name and event spec, which set this law apart from the law it renames."""
        return _root_output_name(self.event_spec), repr(self.event_spec)


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
    _keeps_expression(renamed, law).with_provenance(
        Provenance.create("rename", parents=[law], metadata=dict(arguments))
    )
    return renamed


def _copy_source(law: Distribution) -> tuple[Distribution, _EventRenames] | None:
    """The law *law* is made from, with the renames from its event spec to *law*'s, or None.

    A law that translates its parent's values at its boundary holds its parent.
    Every other law that ``rename``, ``with_label``, ``with_dim_names``,
    or ``with_dim_sizes`` returns records the law it is made from, and the
    renames of a relabeling or a dimension transform are the identity. Each one
    reads that law's draws in a lift (V.5).
    """
    if isinstance(law, _RenamedDistribution):
        return law._parent, law._event
    return getattr(law, _COPY_SOURCE, None)


def _with_copy_source(
    law: Distribution, parent: Distribution, renames: Mapping[str, str]
) -> Distribution:
    """*law*, a copy or a rename of *parent*, recording *parent* as its source.

    The record is *parent* with the renames from its event spec to *law*'s,
    which :func:`_copy_source` reads. A factored joint orders its outputs
    by its factors, so the renames end at *law*'s own declaration. *law* also
    holds the paths *parent* holds fixed.
    """
    event = _EventRenames.of(parent.event_spec, law.event_spec, renames)
    object.__setattr__(law, _COPY_SOURCE, (parent, event))
    return _keeps_fixed_paths(law, parent)


def _renamed_law(
    parent: Distribution, event_spec: OutputSpec, renames: Mapping[str, str]
) -> Distribution:
    """The law ``rename`` returns for *parent*, which reads the draw of *parent*.

    A rename of a whole term's output alone changes only the event spec, so
    the result is a copy of *parent*. A factored law renames through its factors
    where they carry the rename, and regroups them where the rename gathers
    their outputs under new nodes. A law whose family rebuilds itself under
    the new paths returns the member that :meth:`Distribution._renamed_in_family`
    gives. Any other law, or a rename its factors cannot carry, is translated at
    the boundary of a law that holds *parent*. Each other result records
    *parent* and the renames from its event spec to the result's, which a lift
    reads to draw the result together with *parent* (V.5).
    """
    if event_spec.term_spec == parent.event_spec.term_spec:
        copy = parent._with_declaration(event_spec, "rename", renames)
        return _with_copy_source(_keeps_expression(copy, parent), parent, renames)
    if isinstance(parent, SupportsFactors):
        joint = _renamed_through_factors(parent, renames, event_spec)
        if joint is None:
            joint = _regrouped(parent, renames, event_spec)
        if joint is not None:
            return _with_copy_source(joint, parent, renames)
    event = _EventRenames.of(parent.event_spec, event_spec, renames)
    member = parent._renamed_in_family(event)
    if member is None:
        return _renamed(parent, event, renames)
    return _with_copy_source(member, parent, renames).with_provenance(
        Provenance.create("rename", parents=[parent], metadata=dict(renames))
    )


# ---------------------------------------------------------------------------
# Renaming a factored law through its factors
# ---------------------------------------------------------------------------


def _leaf_specs(declaration: OutputSpec) -> dict[str, TermSpec]:
    """The spec of each leaf of *declaration*, keyed by its path."""
    return {path: _node_at(declaration, path) for path in _leaf_paths(declaration)}


def _factor_renames(graph: _FactorGraph, pairs: Mapping[str, str]) -> list[dict[str, str]]:
    """The renames each factor of *graph* applies for the joint's renames *pairs*.

    The factor that produces a renamed output renames the event path, each
    factor that conditions on the output renames the matching given path,
    and a renamed unmet given is renamed in every factor that names it. A
    joint's path that starts with a factor's output is the same path in the
    factor's own event spec, and so is a path into an input.
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
    graph, under the joint's label. Its outputs follow the factors, so a
    moved node joins its factor's outputs rather than the end of the joint's.
    The factors cannot carry a rename that changes a factor's form, as
    moving a whole term's output into a group does, that places outputs of
    two factors under one node, or that changes which factor conditions on
    which. A packaged joint renames the paths below its output through its
    factors, and keeps its form.

    Parameters
    ----------
    joint : FactoredDistribution or FactoredConditionalDistribution
        The joint renamed.
    pairs : Mapping[str, str]
        The new exact path of each renamed node, keyed by its exact path; a key
        starts with an output's name or, for a conditional joint, an input's.
    event_spec : OutputSpec
        The joint's event spec with the renames applied, whose leaves
        the result must declare.

    Returns
    -------
    FactoredDistribution or FactoredConditionalDistribution or None
        The joint of the renamed factors, whose provenance records *pairs*, or
        ``None``.
    """
    graph: _FactorGraph = joint._graph
    form_keywords: dict[str, str] = {}
    output_name = _root_output_name(joint.event_spec)
    if output_name is not None:
        inner = _within_package(joint, pairs, output_name)
        target = _root_output_name(event_spec)
        if inner is None or target is None:
            return None
        pairs, form_keywords = inner, {"_output_name": target}
    try:
        factors = [
            factor.rename(renames) if renames else factor
            for factor, renames in zip(graph.factors, _factor_renames(graph, pairs), strict=True)
        ]
        kind = (
            FactoredConditionalDistribution
            if isinstance(joint, ConditionalDistribution)
            else FactoredDistribution
        )
        renamed = _derived_product(
            kind(factors, label=joint.label, _scope=graph.scope, **form_keywords), joint
        )
    except (KeyError, ValueError):
        return None
    if _leaf_specs(renamed.event_spec) != _leaf_specs(event_spec):
        return None
    if {edge[:2] for edge in renamed._graph.edges} != {edge[:2] for edge in graph.edges}:
        return None
    return renamed.with_provenance(
        Provenance.create("rename", parents=[joint], metadata=dict(pairs))
    )


def _within_package(
    joint: Any, pairs: Mapping[str, str], output_name: str
) -> dict[str, str] | None:
    """*pairs* as paths of the factor record of *joint*, packaged under *output_name*.

    A key of the event side and its target each start with the output's name, and
    a key of an input is kept. None when an event-side rename leaves the
    output or renames it.
    """
    inner: dict[str, str] = {}
    prefix = f"{output_name}{_PATH_SEP}"
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
    """The gathering node each gathered output joins, keyed by the output's name.

    A rename gathers an output when it moves the whole output into a new
    top-level node. None when no rename gathers, or when another rename reaches
    a gathering node or a gathered output.
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

    A factor in a group renames its outputs to their paths below the
    group's node, and a factor left in place renames them as *pairs* does. A
    consumer of an output renames its input to the output's path
    below the node when both factors are in one group, and to the output's
    new path otherwise, which moves the input into a structured input named by
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
    kernel: ConditionalDistribution, input_name: str, record: RecordSpec
) -> ConditionalDistribution | None:
    """*kernel* with its structured input *input_name* declared as *record*, which holds its fields.

    The result reads the kernel's own fields of a value of *record* and drops
    the others, so a consumer of part of a gathered node conditions on the
    node's whole record. None when *record* lacks a field of the input or
    declares it otherwise.
    """
    declared = kernel.given_spec[input_name]
    if not isinstance(declared, RecordSpec):
        return None
    if any(record.children.get(name) != spec for name, spec in declared.children.items()):
        return None
    given_spec = InputSpec(
        {name: (record if name == input_name else spec) for name, spec in kernel.given_spec.items()}
    ).with_optional(*(kernel.given_spec.optional - {input_name}))
    own = set(_given_leaf_specs(kernel.given_spec))
    origins = {leaf: (leaf if leaf in own else None) for leaf in _given_leaf_specs(given_spec)}
    event = _EventRenames.of(kernel.event_spec, kernel.event_spec, {})
    widened = _RenamedConditionalDistribution(kernel, given_spec, kernel.event_spec, origins, event)
    return widened.with_provenance(
        Provenance.create("rename", parents=[kernel], metadata={"widened": input_name})
    )


def _regrouped(
    joint: Any, pairs: Mapping[str, str], event_spec: OutputSpec
) -> Distribution | ConditionalDistribution | None:
    """*joint* renamed by *pairs* with the factors under each gathering node packaged as one.

    The factors whose outputs the rename moves into a new node form a
    packaged sub-joint, labeled by joining their labels, whose event is the
    node as one output holding their outputs' record; the other factors
    rename in place. A consumer of a gathered output conditions on the
    sub-joint through its renamed input, widened to the node's whole
    record. The result orders its factors conditional-first, each one at its
    first factor's position where the dependencies allow.

    None when no rename gathers whole outputs, a factor's outputs land
    in two places, the groups condition on one another in a cycle, or the
    result does not declare the leaves of *event_spec*.
    """
    if _root_output_name(joint.event_spec) is not None:
        return None
    graph: _FactorGraph = joint._graph
    nodes = _gathered(graph, pairs)
    if nodes is None:
        return None
    groups: dict[int, str | None] = {}
    for index, factor in enumerate(graph.factors):
        joined = {nodes.get(output_name) for output_name in factor.event_spec.outputs}
        if len(joined) > 1:
            return None
        groups[index] = next(iter(joined), None)
    try:
        renamed = [
            factor.rename(renames) if renames else factor
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
                output_name: spec
                for index in indices
                for output_name, spec in renamed[index].event_spec.outputs.items()
            }
        )
        for node, indices in members.items()
    }
    for index, factor in enumerate(renamed):
        if not isinstance(factor, ConditionalDistribution):
            continue
        for input_name in list(factor.given_spec):
            if input_name in records and factor.given_spec[input_name] != records[input_name]:
                factor = _widened(factor, input_name, records[input_name])
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
            unit = kind(parts, label=label, _scope=graph.scope, _output_name=node)
            units.append((indices[0], _with_named(unit, _is_named(joint))))
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
        result = kind(
            [units[position][1] for position in order], label=joint.label, _scope=graph.scope
        )
    except (KeyError, TypeError, ValueError):
        return None
    if _leaf_specs(result.event_spec) != _leaf_specs(event_spec):
        return None
    return _derived_product(result, joint).with_provenance(
        Provenance.create("rename", parents=[joint], metadata=dict(pairs))
    )


def _conditional_first(units: Sequence[tuple[int, Any]]) -> list[int] | None:
    """An order of *units* in which each consumer precedes the factors it conditions on.

    Each unit is a factor with the position of its first original factor, which
    breaks ties, so an order with no regrouping keeps the original. None when
    the units condition on one another in a cycle.
    """
    producers = {
        output_name: position
        for position, (_, unit) in enumerate(units)
        for output_name in unit.event_spec.outputs
    }
    consumes = {
        position: {
            producers[input_name]
            for input_name in (unit.given_spec if isinstance(unit, ConditionalDistribution) else ())
            if input_name in producers
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

    A key of *given* is an input's name or a path into a structured input, and the value
    of a structured node is a record or a mapping of its fields.

    Parameters
    ----------
    given_spec : InputSpec
        The given side whose paths the keys of *given* are.
    given : Record or Mapping[str, Any]
        Values keyed by input name or by a path into a structured input.

    Returns
    -------
    dict[str, Any]
        One entry per leaf that *given* binds, in the order *given* lists them.

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
                f"the value for {path!r} must be a Record or a mapping of its fields; got "
                f"{type(value).__name__}"
            )
        for key, entry in value.items():
            visit(f"{path}{_PATH_SEP}{key}", entry)

    for key, value in given.items():
        visit(key, value)
    return values


def _input_value(input_name: str, spec: TermSpec, leaves: Mapping[str, Any]) -> Any:
    """The value of the input *input_name*, declared by *spec*, from the values at its leaves."""
    if not isinstance(spec, RecordSpec):
        return leaves[input_name]
    return Record(
        {leaf[len(input_name) + 1 :]: value for leaf, value in leaves.items()},
        label=input_name,
    )


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
    self: _RenamedConditionalDistribution, given: Any, paths: str | tuple[str, ...]
) -> Distribution:
    """The parent's marginal at the translated given and the original nodes, arranged by *paths*.

    Parameters
    ----------
    self : _RenamedConditionalDistribution
        The renamed kernel whose capability subclass installs this function as
        ``_conditional_marginal``.
    given : Record or Mapping[str, Any]
        A value of every input of this kernel, keyed by input name or by a path
        into a structured input.
    paths : str or tuple of str
        An event path of this kernel, or a tuple of them.

    Returns
    -------
    Distribution
        The marginal, which holds no reference to the parent.

    Raises
    ------
    KeyError
        If a path is not an event path of this kernel.
    ValueError
        If a path holds no single node of the parent.
    """
    request = _ordered_paths_of(self, paths)
    listed_paths = _path_tuple(request)
    originals = _original_nodes(self._event, self.event_spec, listed_paths, self._parent.label)
    marginal = self._parent._conditional_marginal(
        self._parent_given(given), originals[0] if isinstance(request, str) else tuple(originals)
    )
    return self._event.marginal(
        marginal, listed_paths, originals, list_of_paths=isinstance(request, tuple)
    )


def _renamed_conditional_marginal_guard(
    self: _RenamedConditionalDistribution, paths: str | tuple[str, ...]
) -> Feasibility:
    """The parent's conditional marginal guard at the original nodes for *paths*.

    Each path must hold one node of the parent.
    """
    listed_paths = (paths,) if isinstance(paths, str) else tuple(paths)
    unreached = _unreached(self._event, self.event_spec, listed_paths, self.label)
    if unreached is not None:
        return unreached
    originals = _original_nodes(self._event, self.event_spec, listed_paths, self._parent.label)
    return _capability_guard(
        self._parent,
        "_conditional_marginal",
        originals[0] if isinstance(paths, str) else tuple(originals),
    )


def _forwarded(method: str, body: Callable[..., Any]) -> dict[str, Callable[..., Any]]:
    """The entry of a kernel capability whose guard receives the translated inputs."""

    def guard(self: Any, given: Any = None, *arguments: Any, **keywords: Any) -> Feasibility:
        """The parent's guard at the inputs translated through this boundary."""
        translated = None if given is None else self._translated(given)[0]
        return _conditional_capability_guard(
            self._parent, method, translated, *arguments, **keywords
        )

    return {method: body, f"{method}_guard": guard}


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
    """A parent kernel under renamed inputs and event paths, renaming at its boundary.

    ``ConditionalDistribution.rename`` returns it. Binding translates a
    given value to the parent's inputs, moving the fields that a split or a
    grouping moved, and renames the law or the kernel over the rest the parent
    returns. A binding that completes only part of a parent input, as binding
    one of the inputs a split made does, holds the bound fields until the rest
    of the input is bound, so every input of this kernel binds on its own. Each
    conditional capability of the parent is claimed, translated the same way,
    and carries the parent's guard.

    Parameters
    ----------
    parent : ConditionalDistribution
        The kernel whose values are renamed.
    given_spec : InputSpec
        The renamed inputs.
    event_spec : OutputSpec
        The renamed event spec.
    origins : Mapping[str, str or None]
        The parent's leaf for each leaf of the renamed inputs, keyed by its path;
        a leaf that a widened input adds has none, and its value is dropped.
    event : _EventRenames
        The renames from the parent's event spec to this kernel's.
    pending : Mapping[str, Any], optional
        The values already bound at leaves of parent inputs that are not yet
        complete, keyed by the parent's leaf paths.
    """

    _parent: Any
    _capability_table = _RENAMED_KERNEL_CAPABILITIES

    def _derived_traceability(self) -> Traceability:
        """The parent's traceability, since a rename only translates values at the boundary."""
        return self._parent._traceability()

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

    def __repr__(self) -> str:
        """A call of the renamed kernel's constructor, with this kernel's inputs and event spec."""
        arguments = [*self._repr_arguments(), ("given", format_names(self.given_spec))]
        return _law_repr(self, arguments, self._parent)

    def _repr_class_name(self) -> str:
        """The class of the kernel this one renames, which it presents under new names."""
        return self._parent._repr_class_name()

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The family parameters of the kernel this one renames."""
        return self._parent._repr_arguments()

    def _event_repr_parts(self) -> tuple[str | None, str | None]:
        """The output name and the renamed event spec of this kernel.

        They set this kernel apart from the one it renames.
        """
        return _root_output_name(self.event_spec), repr(self.event_spec)

    def _translated(self, given: Any) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
        """The parent inputs *given* completes, parent leaves left pending, and the inputs bound.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values of some inputs of this kernel, keyed by input name or by a path
            into a structured input.

        Returns
        -------
        complete : dict[str, Any]
            The value of each parent input whose leaves are all bound, keyed by
            input name.
        pending : dict[str, Any]
            The values bound so far at leaves of the other parent inputs, keyed by
            the parent's leaf paths.
        inputs : set of str
            The inputs of this kernel that *given* binds.

        Raises
        ------
        KeyError
            If a key of *given* is not a path of the given side.
        ValueError
            If *given* binds part of an input of this kernel.
        """
        values = _leaf_values(self.given_spec, given)
        inputs = {path.partition(_PATH_SEP)[0] for path in values}
        missing = [
            path
            for path in _given_leaf_specs(self.given_spec)
            if path.partition(_PATH_SEP)[0] in inputs and path not in values
        ]
        if missing:
            raise ValueError(
                f"cannot bind part of an input of {self.label!r}: missing {missing}. Bind "
                f"every field of a structured input together"
            )
        bound = {
            **self._pending,
            **{
                self._origins[path]: value
                for path, value in values.items()
                if self._origins[path] is not None
            },
        }
        by_input: dict[str, list[str]] = {}
        for leaf in _given_leaf_specs(self._parent.given_spec):
            by_input.setdefault(leaf.partition(_PATH_SEP)[0], []).append(leaf)
        complete = {
            input_name: _input_value(
                input_name,
                self._parent.given_spec[input_name],
                {leaf: bound[leaf] for leaf in leaves},
            )
            for input_name, leaves in by_input.items()
            if all(leaf in bound for leaf in leaves)
        }
        pending = {
            leaf: value
            for leaf, value in bound.items()
            if leaf.partition(_PATH_SEP)[0] not in complete
        }
        return complete, pending, inputs

    def _parent_given(self, given: Any) -> dict[str, Any]:
        """*given*, values for this kernel's required inputs, translated to the parent's inputs.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values keyed by input name or by a path into a structured input.

        Returns
        -------
        dict[str, Any]
            The supplied parent inputs, keyed by the parent's input name. Omitted
            optional inputs retain the parent's defaults.

        Raises
        ------
        ValueError
            If *given* leaves a required input unbound or supplies only part of an input.
        """
        complete, pending, _ = self._translated(given)
        if pending or not set(self._parent.given_spec.required) <= set(complete):
            raise ValueError(
                f"cannot bind {self.label!r}: it needs a value for every required input "
                f"{list(self.given_spec.required)}"
            )
        return complete

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **options: Any
    ) -> Distribution | ConditionalDistribution:
        """The parent bound at *given*, translated to its inputs, under the new names.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values for some or all of this kernel's inputs, by input name.
        **options : Any
            Options for the parent's primitive.

        Returns
        -------
        Distribution or ConditionalDistribution
            The renamed law when every input is bound, and otherwise the renamed
            kernel over the remaining inputs.

        Raises
        ------
        KeyError
            If a key of *given* is not a path of the given side.
        ValueError
            If *given* binds part of an input.
        """
        complete, pending, inputs = self._translated(given)
        remaining = self.given_spec.without(*(name for name in inputs if name in self.given_spec))
        # A binding that leaves no required input evaluates the parent, whose
        # optional inputs left unbound take their defaults.
        evaluate = bool(complete) or not remaining.required
        result = self._parent._condition_on(complete, **options) if evaluate else self._parent
        if not isinstance(result, ConditionalDistribution):
            return self._event.law(result)
        origins = {
            path: leaf
            for path, leaf in self._origins.items()
            if path.partition(_PATH_SEP)[0] not in inputs
        }
        return _RenamedConditionalDistribution(
            result, remaining, self.event_spec, origins, self._event, pending=pending
        )


def _hold_inputs(
    kernel: ConditionalDistribution, given: Mapping[str, Any]
) -> ConditionalDistribution:
    """Hold complete inputs while leaving optional inputs available to upstream factors."""
    values = _unflatten_paths(dict(given))
    if kernel.spec.free_dims:
        bindings: dict[str, int] = {}
        for name, spec in kernel.given_spec.items():
            if name in values:
                spec._bind_dims_from_value(values[name], bindings, name)
        kernel = kernel.with_dim_sizes(**bindings)
    remaining = kernel.given_spec.without(*values)
    return _RenamedConditionalDistribution(
        kernel,
        remaining,
        kernel.event_spec,
        {path: path for path in _given_leaf_specs(remaining)},
        _EventRenames.of(kernel.event_spec, kernel.event_spec, {}),
        pending=_leaf_values(kernel.given_spec, values),
    )


def _renamed_kernel(
    parent: ConditionalDistribution,
    given_spec: InputSpec,
    event_spec: OutputSpec,
    origins: Mapping[str, str],
    renames: Mapping[str, str],
    pairs: Mapping[str, str],
) -> ConditionalDistribution:
    """The kernel ``rename`` returns for *parent*, with the rename's provenance.

    A factored kernel renames through its factors where they carry the rename,
    and regroups them where the rename gathers their outputs under new
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
    _keeps_expression(kernel, parent).with_provenance(
        Provenance.create("rename", parents=[parent], metadata=dict(pairs))
    )
    return kernel


_install_copy_source(_with_copy_source)
_install_coupled_marginal(CoupledMarginal)
_install_renamed_law(_renamed_law)
_install_renamed_kernel(_renamed_kernel)
