"""The empirical law: a finite, possibly weighted set of atoms of any event type.

Provides:
  - ``EmpiricalDistribution`` – the law of a finite set of weighted atoms, which
    is the closure family that sampling-based operations construct.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from .._array_utils import _is_numeric_array
from .._weights import (
    Weights,
    weighted_choice,
    weighted_covariance,
    weighted_mean,
    weighted_variance,
)
from ..core._array_backend import _to_jax_array
from ..core._batch import Batch
from ..core._dispatch import Feasibility
from ..core._kinds import batch_class_for_spec
from ..core._numeric_array_batch import NumericArrayBatch
from ..core._numeric_record_batch import NumericRecordBatch
from ..core._object_batch import _is_object_array, _ObjectBatch
from ..core._record_batch import RecordBatch, _batch_class_for
from ..core._record_spec import NumericRecordSpec, RecordSpec
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec
from ..core._specs import OutputSpec
from ..core.named_tree import _unflatten_paths
from ..linalg import DenseLinOp
from ._capabilities import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsVariance,
    _capability_subclass,
)
from ._distribution import Distribution, _whole_term_component
from ._factored import _raw_record, _stacked
from ._views import _node_at

if TYPE_CHECKING:
    from ..custom_types import Array, ArrayLike, PRNGKey
    from ..linalg import LinOp

__all__ = ["EmpiricalDistribution"]

_PATH_SEP = "/"

#: The moments an instance claims when its event is numeric.
_NUMERIC_MOMENTS = (SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile)

#: A requested event path, its segments within one atom, and its node's spec.
type _Selected = tuple[str, tuple[str, ...], TermSpec]


# ---------------------------------------------------------------------------
# The atoms and their weights
# ---------------------------------------------------------------------------


def _numeric_atoms(atoms: Any) -> bool:
    """Whether one atom of *atoms* is numeric, read without validating *atoms*."""
    if isinstance(atoms, Batch):
        return isinstance(atoms.element_spec, NumericSpec)
    return _is_numeric_array(atoms)


def _atom_spec(atoms: Any) -> TermSpec:
    """The term spec of one atom of *atoms*.

    Raises
    ------
    TypeError
        If *atoms* is neither a batch of a stored kind nor a numeric array.
    ValueError
        If *atoms* holds no atom, or is an array without a leading axis.
    """
    if isinstance(atoms, Batch):
        if not isinstance(atoms, (NumericArrayBatch, RecordBatch, _ObjectBatch)):
            raise TypeError(
                f"an empirical law stores array, record, or object batches, got "
                f"{type(atoms).__name__}"
            )
        if atoms.batch_size == 0:
            raise ValueError(f"an empirical law has at least one atom, and {atoms.name!r} is empty")
        return atoms.element_spec
    if _is_numeric_array(atoms):
        if atoms.ndim == 0:
            raise ValueError(
                "the leading axis of an array of atoms indexes the atoms, and a 0-d array has none"
            )
        if atoms.shape[0] == 0:
            raise ValueError("an empirical law has at least one atom, and the array holds none")
        return NumericArraySpec(atoms.shape[1:], atoms.dtype)
    raise TypeError(
        f"atoms are given in the event's batch form, or as an array whose leading axis "
        f"indexes array atoms; got {type(atoms).__name__}"
    )


def _atom_weights(weights: Any, atoms: Batch) -> Weights:
    """The weights of *atoms*, uniform when *weights* is ``None``.

    An array is flat, one entry per atom in the row-major order of the batch
    axes, or shaped like the batch axes.

    Raises
    ------
    ValueError
        If the weights do not number one per atom, one is negative, or they do
        not sum to a positive value.
    """
    count = atoms.batch_size
    if weights is None:
        return Weights.uniform(count)
    if isinstance(weights, Weights):
        return Weights(n=count, weights=weights)
    values = jnp.asarray(weights)
    if values.shape == tuple(atoms.batch_shape):
        values = jnp.reshape(values, (count,))
    return Weights(n=count, weights=values)


def _flattened(column: Any, count: int, rank: int) -> Any:
    """*column*, stored with *rank* batch axes leading, with those axes merged into one."""
    if _is_object_array(column):
        return column.reshape((count, *column.shape[rank:]))
    values = _to_jax_array(column)
    return jnp.reshape(values, (count, *values.shape[rank:]))


def _flat_rows(atoms: Batch) -> Any:
    """The raw atoms of *atoms* along one leading axis, in row-major order of the batch axes.

    Returns
    -------
    Any
        An array of shape ``(n, *event_shape)`` for array atoms, the columns of
        that form keyed by leaf path for record atoms, and an object array of
        shape ``(n,)`` otherwise.
    """
    count, rank = atoms.batch_size, len(atoms.batch_shape)
    if isinstance(atoms, NumericArrayBatch):
        return _flattened(atoms.as_jax(), count, rank)
    if isinstance(atoms, RecordBatch):
        columns = atoms._raw_columns()
        return {path: _flattened(columns[path], count, rank) for path in atoms.element_spec}
    return _flattened(atoms._store, count, rank)


def _stored_values(atoms: Batch) -> Any:
    """The storage of array or object atoms, with the batch axes leading."""
    if isinstance(atoms, NumericArrayBatch):
        return atoms.values
    return atoms._store


def _taken(column: Any, index: Any) -> Any:
    """The rows of *column* at *index*, an integer array whose axes lead the result."""
    if _is_object_array(column):
        return column[np.asarray(index)]
    return column[index]


def _ranks(atoms: Batch) -> tuple[int, ...]:
    """How many axes each level of *atoms* holds, outermost first."""
    return tuple(len(group) for group in atoms.axis_groups)


def _batch_form(name: str, raw: Any, level: str, spec: TermSpec) -> Batch:
    """*raw*, values of *spec* in raw form along one leading axis, as their batch on *level*.

    The raw form is an array of array values, the nested mapping of columns, or
    a ``Record`` of them, for record values, and an object array of any other
    values.

    Raises
    ------
    TypeError
        If *spec* has no batch form.
    """
    if isinstance(spec, RecordSpec):
        return _batch_class_for(spec)(name, _raw_record(raw), level, element_spec=spec)
    batch_class = batch_class_for_spec(spec)
    if batch_class is None:
        raise TypeError(f"a value declared as {type(spec).__name__} has no batch form")
    return batch_class(name, raw, level, element_spec=spec)


# ---------------------------------------------------------------------------
# The moments of a numeric event
# ---------------------------------------------------------------------------


def _leafwise(law: EmpiricalDistribution, reduce: Callable[[Array | None, Array], Array]) -> Any:
    """*reduce* of the weights and each leaf's atoms, in the raw form of one draw.

    A record event gives the nested mapping of each leaf's result.
    """
    rows, weights = law._rows, law._p
    if isinstance(rows, dict):
        return _unflatten_paths({path: reduce(weights, column) for path, column in rows.items()})
    return reduce(weights, rows)


def _coordinates(law: EmpiricalDistribution) -> Array:
    """The flat coordinates of the atoms, an ``(n, d)`` array in the event's flat layout.

    An array atom is raveled in row-major order, and a record atom's leaves are
    raveled and concatenated in the record's canonical order.
    """
    rows, count = law._rows, law.num_atoms
    if isinstance(rows, dict):
        return jnp.concatenate([jnp.reshape(column, (count, -1)) for column in rows.values()], 1)
    return jnp.reshape(rows, (count, -1))


def _empirical_mean(self: EmpiricalDistribution) -> Any:
    """The weighted mean of the atoms, a value shaped like one draw."""
    return _leafwise(self, weighted_mean)


def _empirical_variance(self: EmpiricalDistribution) -> Any:
    """The weighted variance ``Σᵢ wᵢ (xᵢ − x̄)²`` of each coordinate, shaped like one draw."""
    return _leafwise(self, weighted_variance)


def _empirical_cov(self: EmpiricalDistribution) -> LinOp:
    """The weighted covariance of the flat coordinates, a ``(d, d)`` dense operator."""
    return DenseLinOp(weighted_covariance(self._p, _coordinates(self)))


def _inverse_cdf(values: Array, weights: Array | None, q: ArrayLike) -> Array:
    """The generalized inverse ``inf{x : F(x) >= q}`` of each coordinate's weighted CDF.

    For each coordinate of the atoms and each level ``q`` in ``[0, 1]``, the
    quantile is the smallest atom at which the cumulative weight reaches ``q``
    of the total. A zero-weight atom therefore has no effect, and the level
    ``0`` gives the smallest atom of positive weight, the limit of the levels
    above it. Every quantile is an atom, so it keeps the atoms' dtype.

    Parameters
    ----------
    values : Array
        The atoms along the leading axis, shaped ``(n, *event_shape)``.
    weights : Array or None
        The atoms' weights, shaped ``(n,)``; ``None`` for uniform weights.
    q : ArrayLike
        The levels, of any shape.

    Returns
    -------
    Array
        The quantiles, shaped ``(*q.shape, *event_shape)``.
    """
    levels = jnp.asarray(q)
    count = values.shape[0]
    flat = jnp.reshape(values, (count, -1))
    order = jnp.argsort(flat, axis=0)
    ordered = jnp.take_along_axis(flat, order, axis=0)
    mass = jnp.ones(count) if weights is None else jnp.asarray(weights)
    cumulative = jnp.cumsum(mass[order], axis=0)
    targets = jnp.reshape(levels, (-1,))

    def column(cdf: Array, atoms: Array) -> Array:
        reached = jnp.searchsorted(cdf, targets * cdf[-1], side="left")
        first_positive = jnp.searchsorted(cdf, 0.0, side="right")
        index = jnp.where(targets > 0, reached, first_positive)
        return atoms[jnp.minimum(index, count - 1)]

    quantiles = jax.vmap(column, in_axes=1, out_axes=1)(cumulative, ordered)
    return jnp.reshape(quantiles, (*levels.shape, *values.shape[1:]))


def _empirical_quantile(self: EmpiricalDistribution, q: ArrayLike) -> Array | dict[str, Any]:
    """The weighted quantiles of each coordinate at the levels *q* in ``[0, 1]``.

    The quantile at a level ``q`` is the generalized inverse of the
    coordinate's CDF, ``inf{x : F(x) >= q}``: the smallest atom at which the
    cumulative weight reaches ``q`` (see :func:`_inverse_cdf`). So the atoms 1,
    2, 3, 4 with uniform weights have median 2, and a zero-weight atom has no
    effect.

    Returns
    -------
    Array or dict
        The event's raw form with the level axes leading in each leaf: an
        array of shape ``(*q.shape, *event_shape)`` for an array event, and the
        nested mapping of such arrays for a record event.
    """
    return _leafwise(self, lambda weights, column: _inverse_cdf(column, weights, q))


#: The capabilities an instance claims when its event is numeric, with their methods.
_MOMENT_CAPABILITIES: dict[type, dict[str, Callable[..., Any]]] = {
    SupportsMean: {"_mean": _empirical_mean},
    SupportsVariance: {"_variance": _empirical_variance},
    SupportsCovariance: {"_cov": _empirical_cov},
    SupportsQuantile: {"_quantile": _empirical_quantile},
}


# ---------------------------------------------------------------------------
# EmpiricalDistribution
# ---------------------------------------------------------------------------


class EmpiricalDistribution(Distribution, SupportsSampling, SupportsExpectation, SupportsMarginals):
    """The law of a finite, possibly weighted set of atoms of any event type.

    The atoms are given in the event's batch form, such as a ``RecordBatch`` or
    an ``OpaqueBatch``, or as an array whose leading axis indexes array atoms.
    Every batch axis indexes atoms, in the row-major order positional indexing
    reads. A batch is stored as given and keeps its own levels, which
    ``with_level_names`` renames. An array is stored as a ``NumericArrayBatch``
    on one level named by *level*, which defaults to the law's component. The
    weights are normalized and default to uniform.

    **The event declaration.** Without *event_spec*, record atoms expose their
    fields, and any other atoms form a whole-term event whose component defaults
    to the law's *name*. An *event_spec* names the components and the packaging.
    ``OutputSpec.with_spec`` completes it with the atoms' spec: a type hole
    takes that spec, and a declared type must unify with it.

    **Capabilities.**

    ==========================  ===================================================
    capability                  realized by
    ==========================  ===================================================
    ``_sample``                 weighted resampling of the atoms
    ``_expectation``            the weighted mean of ``f`` over the atoms, exact
    ``_marginal``               the empirical law of the projected atoms under the
                                same weights, exact at every event path
    ``_mean``, ``_variance``    the weighted moments of a numeric event
    ``_cov``                    the weighted covariance of a numeric event as a
                                ``DenseLinOp`` over its flat coordinates
    ``_quantile``               the generalized inverse ``inf{x : F(x) >= q}`` of
                                each coordinate's weighted CDF, for a numeric event
    ==========================  ===================================================

    An instance whose event is not numeric claims no moment. The law claims no
    density, since an empirical measure has none in general. Every result that
    is shaped like a draw is in the draw's raw form, so a record event's draw,
    moment, or quantiles are a nested mapping of raw leaves.

    Parameters
    ----------
    name : str
        The law's label, which is also the default component of a whole-term
        event.
    atoms : Batch or Array
        The atoms in the event's batch form, or an array of array atoms along
        its leading axis.
    weights : Array or Weights, optional
        Nonnegative weights with a positive sum, normalized at construction:
        flat, one per atom in the row-major order of the batch axes, or shaped
        like the batch axes. A ``Weights`` object, which can be built from log
        weights, is adopted as it is.
    level : str, optional
        The name of the one level a plain array's atoms lie on. It defaults to
        the law's component.
    event_spec : OutputSpec, optional
        The declaration of one draw, completed with the atoms' spec.

    Raises
    ------
    TypeError
        If *name* is not a non-empty string, *atoms* is neither a batch of a
        stored kind nor a numeric array, *level* is not a string or is given
        with a batch of atoms, *event_spec* is not an ``OutputSpec``, or
        *event_spec* exposes a record for atoms that are not records.
    ValueError
        If *atoms* holds no atom or is a 0-d array, the weights do not number
        one per atom or are negative or sum to zero, *level* is not a valid level
        name, *event_spec* declares a type that does not unify with the atoms'
        spec, or the default component *name* is not a valid component name.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> law = EmpiricalDistribution("theta", jnp.array([0.0, 1.0, 3.0]), jnp.array([1.0, 1.0, 2.0]))
    >>> list(law.event_spec.components)
    ['theta']
    >>> law.atoms.level_names
    ('theta',)
    >>> float(law._mean())
    1.75
    """

    _capability_table = _MOMENT_CAPABILITIES

    #: Derived from the stored atoms rather than transported.
    _transient_state = ("_rows_cache",)

    def __new__(
        cls,
        name: str,
        atoms: Batch | Array,
        weights: Array | Weights | None = None,
        *,
        level: str | None = None,
        event_spec: OutputSpec | None = None,
    ) -> EmpiricalDistribution:
        base = vars(cls).get("_capability_base", cls)
        protocols = _NUMERIC_MOMENTS if _numeric_atoms(atoms) else ()
        return object.__new__(_capability_subclass(base, protocols))

    def __init__(
        self,
        name: str,
        atoms: Batch | Array,
        weights: Array | Weights | None = None,
        *,
        level: str | None = None,
        event_spec: OutputSpec | None = None,
    ) -> None:
        atom_spec = _atom_spec(atoms)
        if level is not None and not isinstance(level, str):
            raise TypeError(f"level is the name of a level, got {type(level).__name__}")
        if level is not None and isinstance(atoms, Batch):
            raise TypeError(
                f"level names the level of a plain array's atoms; the batch {atoms.name!r} keeps "
                f"its own levels {list(atoms.level_names)}, which with_level_names renames"
            )
        if event_spec is None:
            declared: OutputSpec | TermSpec = atom_spec
        elif isinstance(event_spec, OutputSpec):
            declared = event_spec.with_spec(atom_spec)
        else:
            raise TypeError(f"event_spec must be an OutputSpec, got {type(event_spec).__name__}")
        super().__init__(name, declared)
        if isinstance(atoms, Batch):
            stored = atoms
        else:
            # An array's atoms form a whole-term event, whose component names their level.
            on_level = _whole_term_component(self.event_spec) if level is None else level
            stored = NumericArrayBatch(name, atoms, on_level, element_spec=atom_spec)
        atom_weights = _atom_weights(weights, stored)
        object.__setattr__(self, "_atoms", stored)
        object.__setattr__(self, "_w", atom_weights)
        # The weights the weighted helpers take, None when uniform. Read here, outside
        # any trace, so that a draw or a moment traced later reads a concrete array.
        probabilities = None if atom_weights.is_uniform else atom_weights.normalized
        object.__setattr__(self, "_p", probabilities)

    # -- the atoms and their weights ------------------------------------------

    @property
    def atoms(self) -> Batch:
        """The atoms in the event's batch form, as stored."""
        return self._atoms

    @property
    def weights(self) -> Array:
        """The normalized weights, one per atom in the row-major order of the batch axes."""
        return self._w.normalized if self._p is None else self._p

    @property
    def num_atoms(self) -> int:
        """The number of atoms."""
        return self._atoms.batch_size

    @property
    def _rows(self) -> Any:
        """The raw atoms along one leading axis (see :func:`_flat_rows`).

        Built on first use and kept when concrete; rows built under a trace stay
        within it.
        """
        cached = getattr(self, "_rows_cache", None)
        if cached is None:
            cached = _flat_rows(self._atoms)
            if not any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(cached)):
                object.__setattr__(self, "_rows_cache", cached)
        return cached

    def _atoms_at(self, index: Any) -> Any:
        """The atoms at *index*, an integer array, in their raw form with its axes leading.

        A record atom is the nested mapping of its raw leaves. With leading axes
        each leaf is the stacked column of the indexed atoms, which is an object
        array for a leaf that is not numeric.
        """
        rows = self._rows
        if not isinstance(self.event_spec.spec, RecordSpec):
            return _taken(rows, index)
        return _unflatten_paths({path: _taken(column, index) for path, column in rows.items()})

    # -- sampling ---------------------------------------------------------------

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Any:
        """Draw atoms independently, each with probability its weight.

        Returns
        -------
        Any
            One atom in its raw form for ``sample_shape=()``: an array for array
            atoms, the nested mapping of raw leaves for record atoms, and the
            stored object otherwise. A non-empty shape prepends its axes to the
            array, to each leaf of the mapping, or as the axes of an object array.
        """
        index = weighted_choice(key, self.num_atoms, weights=self._p, shape=tuple(sample_shape))
        return self._atoms_at(index)

    # -- expectation ------------------------------------------------------------

    def _expectation(self, f: Callable[[Any], Array]) -> Array:
        """The exact ``E[f(X)]``: the weighted mean of ``f`` over the atoms.

        ``f`` receives each atom in its raw form, as a draw is returned, so a
        record atom is the nested mapping of its raw leaves, and returns an
        array or a pytree of arrays. Over a numeric event ``f`` is evaluated at
        every atom at once with ``jax.vmap``, so it must be traceable; over any
        other event it is called on each atom in turn.

        Returns
        -------
        Array or pytree of arrays
            ``Σᵢ wᵢ f(xᵢ)``, which has the shape of the output of ``f``.
        """
        spec = self.event_spec.spec
        rows = self._rows
        if isinstance(spec, NumericArraySpec):
            values = jax.vmap(f)(rows)
        elif isinstance(spec, NumericRecordSpec):
            values = jax.vmap(f)(_unflatten_paths(rows))
        else:
            values = _stacked([f(self._atoms_at(index)) for index in range(self.num_atoms)])
        return jax.tree.map(lambda value: weighted_mean(self._p, value), values)

    # -- marginals --------------------------------------------------------------

    def _marginal(self, path: str | tuple[str, ...]) -> EmpiricalDistribution:
        """The empirical law of the atoms projected onto *path*, under the same weights.

        A single event path yields the leaf or subtree at that path whole, under
        a component named by the path's final segment. A tuple of event paths
        yields an exposed record of the selected nodes keyed by their final
        segments. Either way the result keeps this law's label, the projected
        atoms keep the stored atoms' levels, and the result holds no reference
        to this law.

        Raises
        ------
        KeyError
            If a path is not an event path of this law.
        TypeError
            If a path is not a string.
        ValueError
            If a selection names no path, or two of its paths share a final
            segment.
        """
        selected = self._selection(path)
        if isinstance(path, str):
            ((requested, segments, node),) = selected
            atoms = self._projection(segments, node)
            declaration = OutputSpec(**{requested.rsplit(_PATH_SEP, 1)[-1]: node})
        else:
            atoms = self._selection_batch(selected)
            declaration = OutputSpec(atoms.element_spec)
        return EmpiricalDistribution(self.name, atoms, self._w, event_spec=declaration)

    def _marginal_guard(self, path: str | tuple[str, ...]) -> Feasibility | bool:
        """Whether *path* is an event path, or a selection of them with distinct final segments.

        The marginal is exact at every such path.
        """
        try:
            self._selection(path)
        except (KeyError, TypeError, ValueError) as error:
            return Feasibility(False, str(error.args[0]) if error.args else repr(error))
        return True

    def _selection(self, path: str | tuple[str, ...]) -> tuple[_Selected, ...]:
        """Each requested event path with its segments within one atom and its node's spec.

        Raises
        ------
        KeyError
            If a path is not an event path of this law.
        TypeError
            If a path is not a string.
        ValueError
            If a selection names no path, or two of its paths share a final
            segment.
        """
        paths = (path,) if isinstance(path, str) else tuple(path)
        if not paths:
            raise ValueError("a selection of event paths names at least one path")
        component = _whole_term_component(self.event_spec)
        selected = []
        for requested in paths:
            if not isinstance(requested, str):
                raise TypeError(f"an event path is a string, got {type(requested).__name__}")
            try:
                node = _node_at(self.event_spec, requested)
            except KeyError:
                raise KeyError(
                    f"{requested!r} is not an event path of {self.name!r}, whose components "
                    f"are {list(self.event_spec.components)}"
                ) from None
            segments = tuple(requested.split(_PATH_SEP))
            selected.append((requested, segments if component is None else segments[1:], node))
        if not isinstance(path, str):
            finals = [requested.rsplit(_PATH_SEP, 1)[-1] for requested in paths]
            shared = sorted({final for final in finals if finals.count(final) > 1})
            if shared:
                raise ValueError(
                    f"the selected paths {list(paths)} share the final segments {shared}, which "
                    f"would name two fields of the selected record alike"
                )
        return tuple(selected)

    def _projection(self, segments: tuple[str, ...], node: TermSpec) -> Batch:
        """The atoms projected onto the node at *segments* within one atom, in its batch form."""
        atoms = self._atoms
        if not segments:
            return atoms
        key = _PATH_SEP.join(segments)
        column = atoms[key]
        if isinstance(node, NumericArraySpec):
            return NumericArrayBatch(
                f"{atoms.name}[{key!r}]",
                column,
                atoms.level_names,
                element_spec=node,
                axes_per_level=_ranks(atoms),
            )
        return column

    def _selection_batch(self, selected: tuple[_Selected, ...]) -> RecordBatch:
        """The atoms projected onto the selected nodes, one field per node under its final segment.

        The batch is named by the stored atoms' name indexed by the selected paths.
        """
        atoms = self._atoms
        is_record = isinstance(self.event_spec.spec, RecordSpec)
        stored = atoms._raw_columns() if is_record else {}
        columns: dict[str, Any] = {}
        fields: dict[str, TermSpec] = {}
        for requested, segments, node in selected:
            final = requested.rsplit(_PATH_SEP, 1)[-1]
            fields[final] = node
            if not is_record:
                columns[final] = _stored_values(atoms)
                continue
            prefix = _PATH_SEP.join(segments)
            for leaf, column in stored.items():
                if not prefix:
                    columns[f"{final}{_PATH_SEP}{leaf}"] = column
                elif leaf == prefix:
                    columns[final] = column
                elif leaf.startswith(prefix + _PATH_SEP):
                    columns[f"{final}{_PATH_SEP}{leaf[len(prefix) + 1 :]}"] = column
        element = RecordSpec(fields)
        batch_class = NumericRecordBatch if isinstance(element, NumericRecordSpec) else RecordBatch
        return batch_class(
            f"{atoms.name}[{tuple(requested for requested, _, _ in selected)!r}]",
            columns,
            atoms.level_names,
            element_spec=element,
            axes_per_level=_ranks(atoms),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, num_atoms={self.num_atoms})"
