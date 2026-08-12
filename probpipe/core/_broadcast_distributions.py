"""Broadcast distribution and marginal types.

Provides:
  - ``_RecordMarginal``                    – Record-shaped output marginal.
  - ``_MixtureMarginal[T]``                – Distribution output marginal (mixture).
  - ``_ListMarginal[T]``                   – Non-stackable output marginal.
  - ``MarginalizedBroadcastDistribution``  – Union type alias.
  - ``_make_marginal()``                   – Factory for marginal construction.
  - ``BroadcastDistribution``              – Joint over broadcast inputs and output.
"""

from __future__ import annotations

from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .._weights import Weights
from ..custom_types import Array
from ._array_backend import _to_jax_array
from ._distribution_base import Distribution
from ._empirical import (
    EmpiricalDistribution,
    RecordEmpiricalDistribution,
)
from ._numeric_record_batch import NumericRecordBatch
from ._object_batch import _from_iterable, _is_object_array
from ._record_batch import RecordBatch, _batch_class_for
from .event_template import (
    ArraySpec,
    EventTemplate,
    NumericEventTemplate,
    _full_array_shape_or_none,
)
from .protocols import (
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from .record import Record
from .tracked import auto_name

# ---------------------------------------------------------------------------
# MarginalizedBroadcastDistribution — output marginal of a broadcast
# ---------------------------------------------------------------------------
#
# Protocol support is determined dynamically via a factory function that
# picks the right concrete subclass, so ``isinstance`` checks are truthful.
# ---------------------------------------------------------------------------


class _RecordMarginal(RecordEmpiricalDistribution):
    """Record-shaped output marginal of a :class:`BroadcastDistribution`.

    Wraps the broadcast outputs as a Record-valued empirical
    distribution with per-field weighted resampling and moments. Bare
    array outputs auto-wrap as a single-field Record keyed by ``name``
    (defaulting to ``"marginal"`` since the WF-output context doesn't
    carry a more meaningful name).
    """

    def __init__(
        self,
        samples: Record | RecordBatch | Array,
        weights: Array | Weights | None = None,
        *,
        log_weights: Array | Weights | None = None,
        name: str | None = None,
        event_template: EventTemplate | None = None,
    ):
        # A batch of records holds its rows axis in the batch, and the merged
        # constructor wants one row per batch index, so peel it: the leaves keep
        # the rows axis and the record above them loses it. Leaf-keyed, so a
        # nested batch peels correctly; path-keyed construction rebuilds nesting.
        if isinstance(samples, RecordBatch):
            template = samples.event_template
            # Raw columns: a non-array field presents as its own object batch,
            # which is not what belongs in a record of batched leaves.
            samples = Record(samples.name, samples._raw_columns(), name_is_auto=True)
        else:
            template = None
        # Default field name for bare-array outputs (the WF marginal
        # context doesn't carry a more meaningful name).
        if not isinstance(samples, Record) and not name:
            name = "marginal"
        super().__init__(samples, weights=weights, log_weights=log_weights, name=name)
        if event_template is not None:
            self._event_template = event_template
        elif template is not None:
            # Preserve the exact template the batch carried.
            self._event_template = template

    def __repr__(self):
        return (
            f"MarginalizedBroadcastDistribution(num_atoms={self.num_atoms}, "
            f"fields=({', '.join(self._record_data.fields)}))"
        )


class _MixtureMarginal[T](Distribution[T]):
    """Output marginal when broadcast outputs are Distribution objects.

    Acts as a finite mixture: ``p(y) = Σ_i w_i p_i(y)``.  Protocol support
    depends on what the component distributions support.

    This base class provides no protocol methods.  The factory
    :func:`_make_mixture_marginal` dynamically constructs a subclass that
    inherits the appropriate protocol mixins.
    """

    def __init__(
        self,
        components: list,
        weights: Array | Weights | None = None,
        *,
        log_weights: Array | Weights | None = None,
        name: str | None = None,
        event_template: EventTemplate | None = None,
    ):
        n = len(components)
        self._components = components
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)
        name, name_is_auto = auto_name(name, "mixture_marginal")
        super().__init__(name=name, name_is_auto=name_is_auto)
        self._approximate = True
        self._event_template = event_template

    @property
    def num_atoms(self) -> int:
        return len(self._components)

    @property
    def components(self) -> list:
        return self._components

    @property
    def weights(self) -> Array:
        return self._w.normalized

    @property
    def event_template(self) -> EventTemplate | None:
        """Authoritative template shared by the mixture components."""
        return self._event_template

    def __repr__(self):
        return f"MarginalizedBroadcastDistribution(mixture, num_atoms={self.num_atoms})"


# -- Mixture protocol mixins (combined dynamically) -------------------------


class _MixtureSampling:
    """SupportsSampling mixin for mixture marginals.

    Returns a raw ``Array`` when component samples are arrays (the
    common case — broadcasting a numeric function over a distribution
    of inputs), and a batch of records when component samples are
    ``Record``-valued (e.g., broadcasting a ``Record``-returning
    ``Function``). Opaque / non-stackable component outputs
    raise a ``TypeError`` with the component types listed.
    """

    _sampling_cost: str = "medium"
    _preferred_orchestration: str | None = None

    def _sample(self, key, sample_shape=()):
        from .record import Record

        n_draws = prod(sample_shape) if sample_shape else 1
        key1, key2 = jax.random.split(key)
        indices = self._w.choice(key1, shape=(n_draws,))
        keys = jax.random.split(key2, n_draws)

        results = [self._components[int(indices[i])]._sample(keys[i], ()) for i in range(n_draws)]

        # Dispatch on result type so a mixture of Record-returning components
        # stacks into a batch of draws rather than crashing in jnp.stack. Scalars
        # and arrays stay on the numeric path. A component that draws a batch of
        # its own is not a Record, so it takes neither path — stacking batches is
        # not supported here.
        if all(isinstance(r, Record) for r in results):
            template = results[0].event_template
            cls = _batch_class_for(template)
            stacked = cls.stack(results, level_name=DRAW_LEVEL)
            if sample_shape == ():
                return stacked[0]
            # Reshape the leading axis to sample_shape: one draw level over
            # however many axes the shape spans.
            columns = {
                path: stacked[path].reshape(sample_shape + stacked[path].shape[1:])
                for path in stacked.event_template
            }
            return cls(
                columns,
                DRAW_LEVEL,
                element_spec=stacked.element_spec,
                axis_groups=(sample_shape,),
            )

        try:
            stacked = jnp.stack(results, axis=0)
        except (TypeError, ValueError) as exc:
            types_seen = sorted({type(r).__name__ for r in results})
            raise TypeError(
                f"_MixtureSampling cannot stack component samples of "
                f"types {types_seen}; mixture marginals support numeric "
                f"arrays and Record values only."
            ) from exc
        if sample_shape == ():
            return stacked[0]
        return stacked.reshape(sample_shape + stacked.shape[1:])


class _MixtureMean:
    """SupportsMean mixin for mixture marginals."""

    def _mean(self):
        means = jnp.stack([c._mean() for c in self._components], axis=0)
        return self._w.mean(means)


class _MixtureVariance:
    """SupportsVariance mixin for mixture marginals (law of total variance)."""

    def _variance(self):
        means = jnp.stack([c._mean() for c in self._components], axis=0)
        variances = jnp.stack([c._variance() for c in self._components], axis=0)
        overall_mean = self._w.mean(means)
        # Law of total variance: E[Var(Y|X)] + Var(E[Y|X])
        e_var = self._w.mean(variances)
        diff = means - overall_mean
        var_e = self._w.mean(diff**2)
        return e_var + var_e


class _MixtureLogProb:
    """SupportsLogProb mixin for mixture marginals."""

    def _log_prob(self, value):
        log_w = self._w.log_normalized
        component_lps = jnp.stack([c._log_prob(value) for c in self._components], axis=0)
        return jax.scipy.special.logsumexp(log_w + component_lps)


# Map protocol → (mixin class, required component protocols)
_MIXTURE_PROTOCOL_MAP: list[tuple[type, type, tuple[type, ...]]] = [
    (SupportsSampling, _MixtureSampling, (SupportsSampling,)),
    (SupportsMean, _MixtureMean, (SupportsMean,)),
    (SupportsVariance, _MixtureVariance, (SupportsMean, SupportsVariance)),
    (SupportsLogProb, _MixtureLogProb, (SupportsLogProb,)),
]

# Cache dynamically created classes to avoid repeated type() calls
_mixture_class_cache: dict[tuple[type, ...], type] = {}


def _make_mixture_marginal(
    components: list,
    weights: Array | Weights | None = None,
    *,
    name: str | None = None,
    event_template: EventTemplate | None = None,
) -> _MixtureMarginal:
    """Factory that builds a mixture marginal with dynamic protocol support.

    Inspects the component distributions to determine which protocols they
    all support, then creates (and caches) a concrete subclass that inherits
    the corresponding mixin classes.
    """
    # Determine which protocols all components support
    active_protocols: list[type] = []
    active_mixins: list[type] = []
    for protocol, mixin, required in _MIXTURE_PROTOCOL_MAP:
        if all(isinstance(c, req) for c in components for req in required):
            active_protocols.append(protocol)
            active_mixins.append(mixin)

    cache_key = tuple(active_protocols)
    if cache_key not in _mixture_class_cache:
        bases = (*tuple(active_mixins), _MixtureMarginal, *tuple(active_protocols))
        cls_name = "_DynMixtureMarginal"
        _mixture_class_cache[cache_key] = type(cls_name, bases, {})

    cls = _mixture_class_cache[cache_key]
    obj = object.__new__(cls)
    _MixtureMarginal.__init__(
        obj,
        components,
        weights,
        name=name,
        event_template=event_template,
    )
    return obj


class _ListMarginal[T](Distribution[T]):
    """Output marginal when broadcast outputs are non-stackable (e.g., strings).

    No protocol support — outputs cannot be resampled or summarised.
    """

    def __init__(
        self,
        items: list,
        weights: Array | Weights | None = None,
        *,
        log_weights: Array | Weights | None = None,
        name: str | None = None,
    ):
        self._items = items
        self._w = Weights(n=len(items), weights=weights, log_weights=log_weights)
        name, name_is_auto = auto_name(name, "list_marginal")
        super().__init__(name=name, name_is_auto=name_is_auto)

    @property
    def num_atoms(self) -> int:
        return len(self._items)

    @property
    def items(self) -> list:
        return self._items

    @property
    def weights(self) -> Array:
        return self._w.normalized

    def __repr__(self):
        return f"MarginalizedBroadcastDistribution(list, num_atoms={self.num_atoms})"


# Public alias for type checking / isinstance
MarginalizedBroadcastDistribution = _RecordMarginal | _MixtureMarginal | _ListMarginal
"""Union type for the output marginal of a :class:`BroadcastDistribution`.

Concrete subtype depends on output kind:

- :class:`_RecordMarginal` — stackable array or Record outputs
  (numeric arrays auto-wrap as single-field Records)
- :class:`_MixtureMarginal` — distribution outputs (mixture)
- :class:`_ListMarginal` — non-stackable outputs
"""


def _packed_object_column(values: list) -> np.ndarray:
    """*values* as a frozen object column, one entry per row.

    A field the declaration does not call an array holds one value per element
    whatever those values look like, so they are packed rather than stacked. An
    opaque field whose rows are arrays is the case that matters: stacking them
    numerically would turn each row's own axes into batch axes the levels never
    named.
    """
    column = _from_iterable(values, kind="declared output")
    column.setflags(write=False)
    return column


def _stack_declared_columns(
    records: list[Record] | Record,
    *,
    batch_shape: tuple[int, ...],
    axis_groups: tuple[tuple[int, ...], ...],
    level_names: tuple[str, ...],
    template: EventTemplate,
    name: str,
) -> RecordBatch:
    """Build one batch for validated authoritative Function outputs.

    Columns are keyed by leaf path, so a nested declared output costs the stacking
    nothing: every leaf is one column whatever depth it sits at, and there is no
    per-subtree container to build.
    """
    n_total = prod(batch_shape)
    if isinstance(records, list) and len(records) != n_total:
        raise ValueError(
            f"Expected {n_total} declared outputs for batch_shape={batch_shape}, got {len(records)}"
        )

    columns: dict[str, Any] = {}
    for path in template:
        if isinstance(records, list):
            values = [record[path] for record in records]
            # The declared *kind* decides the storage, not what the values happen
            # to look like. An opaque field holding one array per row is a column
            # of two objects, not a numeric column whose second axis is another
            # multiplicity — reading it off the runtime shape states a batch
            # geometry the levels never described.
            if isinstance(template[path], ArraySpec):
                batched = jnp.stack(
                    [
                        _to_jax_array(value)
                        if _full_array_shape_or_none(value) is not None
                        else value
                        for value in values
                    ],
                    axis=0,
                )
            else:
                batched = _packed_object_column(values)
        else:
            # A batched ``Record`` arrives from the JAX paths, where ``vmap``
            # stacked the rows itself and a declared-opaque field is whatever
            # shape its values happened to have. The declared kind decides here
            # too, or that shape is read as a second multiplicity.
            batched = records[path]
            if not isinstance(template[path], ArraySpec):
                batched = _packed_object_column(list(batched))

        shape = getattr(batched, "shape", ())
        if tuple(shape[:1]) != (n_total,):
            raise ValueError(
                f"Declared output field {path!r} has batched shape {tuple(shape)}, "
                f"expected a leading axis of length {n_total}"
            )
        columns[path] = batched.reshape(batch_shape + tuple(shape[1:]))

    cls = _batch_class_for(template)
    return cls(
        columns,
        level_names,
        element_spec=template,
        axis_groups=axis_groups,
        name=name,
    )


def _empty_declared_stack(
    batch_shape: tuple[int, ...],
    *,
    template: EventTemplate,
    name: str,
    level_names: tuple[str, ...],
    axis_groups: tuple[tuple[int, ...], ...],
) -> Any:
    """The declared aggregate at zero rows: every field present, every axis empty.

    Built exactly as :func:`_stack_declared_columns` would have built it, so an
    empty sweep and a one-row sweep hand back the same type with the same fields
    and the same levels — a numeric leaf is an empty array of its declared shape
    and dtype, and a non-array field an empty object column. Columns are keyed by
    leaf path, so a nested template needs no per-subtree container.
    """
    columns: dict[str, Any] = {}
    for path, spec in template.items():
        if isinstance(spec, ArraySpec) and all(isinstance(size, int) for size in spec.shape):
            dtype = spec.dtype if spec.dtype is not None else jnp.zeros(()).dtype
            columns[path] = jnp.zeros((*batch_shape, *spec.shape), dtype=dtype)
        else:
            columns[path] = np.empty(batch_shape, dtype=object)
    cls = _batch_class_for(template)
    return cls(columns, level_names, element_spec=template, axis_groups=axis_groups, name=name)


def _make_marginal(
    output_samples: Any,
    weights: Array | Weights | None = None,
    *,
    output_distributions: list | None = None,
    name: str | None = None,
    event_template: EventTemplate | None = None,
) -> MarginalizedBroadcastDistribution:
    """Factory to construct the appropriate marginal subtype."""
    if output_distributions is not None:
        return _make_mixture_marginal(
            output_distributions,
            weights,
            name=name,
            event_template=event_template,
        )

    if event_template is not None and isinstance(output_samples, list):
        from ._function_contract import _wrap_declared_function_output

        output_samples = [
            _wrap_declared_function_output(
                output,
                function_name=name or "marginal",
                output_template=event_template,
            )
            for output in output_samples
        ]

    if event_template is not None and isinstance(output_samples, jnp.ndarray):
        if len(event_template) != 1:
            raise ValueError(
                "bare array aggregation requires a single-leaf event_template; "
                "authoritative Function outputs must be wrapped before aggregation"
            )
        only_path = next(iter(event_template.keys()))
        output_samples = Record(
            name or "marginal",
            {only_path: output_samples},
            name_is_auto=True,
        )

    if isinstance(output_samples, RecordBatch) and not isinstance(
        output_samples.event_template, NumericEventTemplate
    ):
        # The record marginal is empirical over numeric leaves — its reductions
        # and resampling convert columns to arrays — so a batch holding a
        # non-numeric field takes the general list marginal over its elements.
        rows = [
            output_samples[tuple(int(i) for i in position)]
            for position in np.ndindex(*output_samples.batch_shape)
        ]
        return _ListMarginal(rows, weights, name=name)

    if isinstance(output_samples, RecordBatch):
        return _RecordMarginal(
            output_samples,
            weights,
            name=name,
            event_template=event_template,
        )

    # Record with batched leaves (e.g., from jax.vmap over a Record-returning fn).
    # All fields must be arrays with a consistent leading batch dimension.
    if isinstance(output_samples, Record) and len(output_samples):
        # Leaf values (values() descends into nested Records), so a nested
        # sample inspects its arrays rather than tripping on an interior node.
        resolved = list(output_samples.values())
        if all(hasattr(v, "ndim") and v.ndim > 0 for v in resolved):
            n = resolved[0].shape[0]
            if all(v.shape[0] == n for v in resolved):
                return _RecordMarginal(
                    output_samples,
                    weights,
                    name=name,
                    event_template=event_template,
                )

    if isinstance(output_samples, jnp.ndarray):
        return _RecordMarginal(
            output_samples,
            weights,
            name=name or "marginal",
            event_template=event_template,
        )

    if isinstance(output_samples, list):
        if output_samples and all(isinstance(r, Record) for r in output_samples):
            try:
                if event_template is not None:
                    ra = _stack_declared_columns(
                        output_samples,
                        batch_shape=(len(output_samples),),
                        axis_groups=((len(output_samples),),),
                        level_names=(DRAW_LEVEL,),
                        template=event_template,
                        name=name or "marginal",
                    )
                else:
                    ra = RecordBatch.stack(output_samples, level_name=DRAW_LEVEL)
                return _RecordMarginal(
                    ra,
                    weights,
                    name=name,
                    event_template=event_template,
                )
            except (ValueError, TypeError):
                pass
        try:
            stacked = jnp.stack([jnp.asarray(r) for r in output_samples], axis=0)
            return _RecordMarginal(
                stacked,
                weights,
                name=name or "marginal",
                event_template=event_template,
            )
        except (ValueError, TypeError):
            pass
        if output_samples and all(isinstance(r, Distribution) for r in output_samples):
            return _make_mixture_marginal(
                output_samples,
                weights,
                name=name,
                event_template=event_template,
            )
        return _ListMarginal(output_samples, weights, name=name)

    # Single array result (e.g., from vmap); ensure at least 1D for the sample axis
    arr = jnp.atleast_1d(jnp.asarray(output_samples))
    return _RecordMarginal(
        arr,
        weights,
        name=name or "marginal",
        event_template=event_template,
    )


# ---------------------------------------------------------------------------
# _make_stack — stacked sibling of _make_marginal for batched broadcasts
# ---------------------------------------------------------------------------
#
# When a Function broadcasts over a batch of records (parameter
# sweep), the n inner outputs are independent scenarios indexed by
# input row — *not* MC draws. The wrapper must preserve row identity:
#
#   numeric → NumericRecordBatch(result=..., one sweep level of (n,))
#   Record → RecordBatch.stack (NumericRecordBatch when all leaves numeric)
#   Distribution → DistributionArray
#   a batch per row (each (m,)) → one batch, levels (sweep, …) over (n, m)
#
# Opaque Python values (e.g. strings) that can't be stacked fall
# through to a plain-list wrapping with a clear error if even that
# fails.
#
# Caller attaches ``.with_provenance(...)`` externally via ``_coerce_output``.
# ---------------------------------------------------------------------------


# The level a draw mints, which the design names (05-operations, ``sample``). A
# broadcast has no name of its own to give: the level it mints is the one it
# swept, so the caller supplies that name rather than this module inventing one.
DRAW_LEVEL = "draw"


def _make_stack(
    inner_outputs: Any,
    *,
    batch_shape: tuple[int, ...] | None = None,
    n: int | None = None,
    level_names: tuple[str, ...],
    axis_groups: tuple[tuple[int, ...], ...] | None = None,
    name: str | None = None,
    field_name: str,
    event_template: EventTemplate | None = None,
) -> Any:
    """Wrap inner Function outputs as a shape-``batch_shape``
    aggregate.

    Internally the outputs are aggregated along a single leading axis
    of length ``prod(batch_shape)``; the final aggregate reshapes that
    axis to ``batch_shape`` so multi-d sweeps produce multi-d output
    shapes.

    Dispatch on ``inner_outputs`` — either a Python ``list`` of length
    ``prod(batch_shape)`` (Python-loop execution path) or a pytree with
    a leading axis of length ``prod(batch_shape)`` (``jax.vmap``
    execution path).

    Parameters
    ----------
    inner_outputs : list or pytree
        Either a list of inner-function results, or a single stackable
        pytree with a leading axis equal to ``prod(batch_shape)``.
    batch_shape : tuple of int, optional
        Shape of the output aggregate's leading axes. Pass either
        ``batch_shape`` or ``n`` (the 1-D shortcut); exactly one.
    n : int, optional
        Shortcut for ``batch_shape=(n,)``.
    level_names : tuple of str
        Names the levels this aggregation mints, one per group of
        ``batch_shape``'s axes, and required for the reason
        :meth:`Batch.with_level_names` gives: an operation that mints a level
        takes the name to give it, since operands align by level name and only
        the caller knows what the axes range over. A sweep passes the names of
        the levels it swept, so the aggregate aligns with the input it came from.
    name : str, optional
        Name for the resulting aggregate.

    Returns
    -------
    NumericRecordBatch | RecordBatch | DistributionArray
        Output type depends on the inner-return type; see module
        docstring for the dispatch table.

    Raises
    ------
    TypeError
        If the inner outputs can't be coerced into any of the three
        aggregate types. The error lists the observed types.
    """
    from ._distribution_array import _make_distribution_array
    from .record import Record

    # Resolve batch_shape vs. n. Exactly one must be provided.
    if batch_shape is None and n is None:
        raise TypeError("_make_stack requires either batch_shape or n")
    if batch_shape is not None and n is not None:
        raise TypeError("_make_stack: pass batch_shape OR n, not both")
    if batch_shape is None:
        assert n is not None
        batch_shape = (n,)
    batch_shape = tuple(batch_shape)
    n_total = int(prod(batch_shape)) if batch_shape else 1
    # One level per swept group, tiling the aggregate's leading axes. A single
    # name takes them all, which is what a one-group sweep and the ``n``
    # shortcut both are.
    sweep_groups = tuple(axis_groups) if axis_groups is not None else (batch_shape,)
    if len(sweep_groups) != len(level_names):
        raise ValueError(
            f"_make_stack mints one level per group of axes: {len(sweep_groups)} groups "
            f"{sweep_groups} against {len(level_names)} names {list(level_names)}"
        )

    # --- List-of-X path (Python-loop execution) -------------------------
    if isinstance(inner_outputs, list):
        # With no rows there is no output to read a type off, so the declared
        # template is the only honest source; without one, the generic handlers
        # below would name a single opaque field after the function. Only a
        # sweep that *expects* zero rows takes this path — an empty list where
        # rows were expected is a missing-output error, and fabricating the
        # declared fields would hide it.
        if not inner_outputs and n_total == 0 and event_template is not None:
            return _empty_declared_stack(
                batch_shape,
                template=event_template,
                name=name or field_name,
                level_names=level_names,
                axis_groups=sweep_groups,
            )
        if len(inner_outputs) != n_total:
            raise ValueError(
                f"_make_stack got {len(inner_outputs)} outputs but "
                f"expected prod(batch_shape)={n_total} "
                f"(batch_shape={batch_shape})."
            )
        outs: Any = inner_outputs
        if event_template is not None:
            from ._function_contract import _wrap_declared_function_output

            outs = [
                _wrap_declared_function_output(
                    output,
                    function_name=field_name,
                    output_template=event_template,
                )
                for output in outs
            ]

        # A batch per row stacks into one batch with the sweep in front of the
        # rows' own levels. Checked before the Record branch below, which would
        # otherwise claim a batch of records and collapse its
        # inner batch axis.
        if outs and any(isinstance(o, RecordBatch) for o in outs):
            # A batch row is all-or-nothing. Falling through on a mixture, or on
            # rows that disagree, reaches the generic handlers — which would take a
            # single-field numeric batch for an array, read its inner batch axis as
            # an event axis, and discard the level names with it. There is no
            # aggregate to build from rows that do not agree on what they hold, so
            # this says so where the disagreement is visible.
            if not all(isinstance(o, RecordBatch) for o in outs):
                kinds = sorted({type(o).__name__ for o in outs})
                raise TypeError(
                    f"{field_name}: some rows returned a batch of records and some did not "
                    f"({', '.join(kinds)}). A swept body returns one kind for every row, since "
                    f"the aggregate has one schema; return a batch from every row or from none"
                )
            first = outs[0]
            # Every row must agree on what it holds and on which axes hold it.
            # Matching the shape alone would take the first row's schema and level
            # names for all of them, dropping a field the others have and
            # misnaming their axes — a batch whose spec is a false statement about
            # its own columns.
            for other in outs[1:]:
                if (
                    other.element_spec == first.element_spec
                    and other.batch_shape == first.batch_shape
                    and other.level_names == first.level_names
                    and other.axis_groups == first.axis_groups
                ):
                    continue
                raise ValueError(
                    f"{field_name}: the rows returned batches that disagree — "
                    f"{first.level_names} over {first.batch_shape} against "
                    f"{other.level_names} over {other.batch_shape}. Rows stack into one batch, "
                    f"which states one element spec and one multiplicity for all of them, so "
                    f"every row must return the same schema on the same levels"
                )
            # Columns are leaf-keyed, so a nested element needs no special
            # case — and they are read raw: a field that is not an array
            # presents as its own object batch, and what stacks is the
            # column, through numpy so the objects are taken as they are.
            columns = {}
            for path in first.event_template:
                cols = [o._raw_column(path) for o in outs]
                if any(_is_object_array(c) for c in cols):
                    stacked = np.stack(cols, axis=0)
                else:
                    stacked = jnp.stack(cols, axis=0)
                columns[path] = stacked.reshape(batch_shape + stacked.shape[1:])
            return _batch_class_for(first.element_spec)(
                columns,
                (*level_names, *first.level_names),
                element_spec=first.element_spec,
                axis_groups=(*sweep_groups, *first.axis_groups),
                name=name or field_name,
                name_is_auto=True,
            )

        # All (scalar) Records → stack into one batch. NumericRecordBatch if
        # every leaf is numeric; otherwise the permissive RecordBatch, building the
        # columns manually so non-numeric leaves (strings, xarray objects, ...)
        # survive.
        if outs and all(isinstance(o, Record) for o in outs):
            if event_template is not None:
                return _stack_declared_columns(
                    outs,
                    batch_shape=batch_shape,
                    axis_groups=sweep_groups,
                    level_names=level_names,
                    template=event_template,
                    name=name or field_name,
                )
            # Stack flat, then reshape the leading axis to batch_shape.
            try:
                flat = NumericRecordBatch.stack(list(outs), level_name=level_names[0])
            except (TypeError, ValueError):
                flat = None
            if flat is not None:
                if batch_shape == (n_total,):
                    return flat
                n_cur = len(flat.batch_shape)
                return NumericRecordBatch(
                    {
                        path: flat[path].reshape(batch_shape + flat[path].shape[n_cur:])
                        for path in flat.event_template
                    },
                    level_names,
                    element_spec=flat.element_spec,
                    axis_groups=sweep_groups,
                )
            # No declared template, so the element structure is inferred from the
            # rows. ``RecordBatch.stack`` is what infers it: columns are keyed by
            # leaf path, so a nested element is columns like any other — the
            # nesting needs no special case here, and neither does a field whose
            # values are opaque, which stacks into an object column.
            first = outs[0]
            if any(tuple(o.children) != tuple(first.children) for o in outs):
                raise TypeError("_make_stack: Records in list have inconsistent fields.")
            # One level over all the rows, then re-cut to the sweep's own
            # geometry: the rows arrive flat and the grid is what they came from.
            flat = RecordBatch.stack(outs, level_name=level_names[0])
            columns = {
                path: column.reshape(batch_shape + column.shape[1:])
                for path, column in flat._raw_columns().items()
            }
            return _batch_class_for(flat.element_spec)(
                columns,
                level_names,
                element_spec=flat.element_spec,
                axis_groups=sweep_groups,
            )

        # All Distributions → stacked DistributionArray, shaped to
        # batch_shape.
        if outs and all(isinstance(o, Distribution) for o in outs):
            return _make_distribution_array(
                outs,
                batch_shape=batch_shape,
                name=name,
                name_is_auto=True,
                event_template=event_template,
            )

        # Numeric scalars / arrays → wrap in a NumericRecordBatch with
        # the single "result" field carrying the stacked values,
        # reshape leading axis to batch_shape.
        try:
            stacked = jnp.stack(
                [jnp.asarray(o) for o in outs],
                axis=0,
            )
        except (TypeError, ValueError):
            stacked = None

        if stacked is not None:
            event_shape = tuple(stacked.shape[1:])
            reshaped = stacked.reshape(batch_shape + event_shape)
            return NumericRecordBatch(
                {field_name: reshaped},
                level_names,
                element_spec=EventTemplate(**{field_name: event_shape}),
                axis_groups=sweep_groups,
            )

        # Last-ditch: wrap as a RecordBatch whose single field holds a
        # numpy object-dtype array of the opaque outputs.
        try:
            object_array = np.asarray(outs, dtype=object).reshape(batch_shape)
            return RecordBatch(
                {field_name: object_array},
                level_names,
                element_spec=EventTemplate(**{field_name: None}),
                axis_groups=sweep_groups,
            )
        except (TypeError, ValueError) as exc:
            types_seen = sorted({type(o).__name__ for o in outs})
            raise TypeError(
                f"_make_stack cannot aggregate outputs of types "
                f"{types_seen}; supported: numeric arrays, Record, "
                f"a batch of records, Distribution."
            ) from exc

    # --- Single-pytree path (jax.vmap execution) ------------------------

    # vmap of a numeric-returning function produces a jnp.ndarray with
    # leading axis of length n_total. Reshape to batch_shape.
    if isinstance(inner_outputs, jnp.ndarray):
        if inner_outputs.shape[:1] != (n_total,):
            raise ValueError(
                f"_make_stack got array of shape {inner_outputs.shape} but "
                f"expected leading axis of length {n_total} "
                f"(batch_shape={batch_shape})."
            )
        event_shape = tuple(inner_outputs.shape[1:])
        if event_template is not None:
            if len(event_template) != 1:
                raise ValueError(
                    "bare array aggregation requires a single-leaf event_template; "
                    "authoritative Function outputs must be wrapped before aggregation"
                )
            output_field = next(iter(event_template.keys()))
            batched_record = Record(
                name or field_name,
                {output_field: inner_outputs},
                name_is_auto=True,
            )
            return _stack_declared_columns(
                batched_record,
                batch_shape=batch_shape,
                axis_groups=sweep_groups,
                level_names=level_names,
                template=event_template,
                name=name or field_name,
            )
        return NumericRecordBatch(
            {field_name: inner_outputs.reshape(batch_shape + event_shape)},
            level_names,
            element_spec=EventTemplate(**{field_name: event_shape}),
            axis_groups=sweep_groups,
        )

    # vmap of a Record-returning function produces a Record with batched leaves
    # (each leaf has leading axis n_total). Promote it to a batch — numeric when
    # every leaf is — with the leading axis reshaped to batch_shape.
    if isinstance(inner_outputs, Record) and inner_outputs.children:
        if event_template is not None:
            return _stack_declared_columns(
                inner_outputs,
                batch_shape=batch_shape,
                axis_groups=sweep_groups,
                level_names=level_names,
                template=event_template,
                name=name or field_name,
            )
        # Leaf-keyed, so a nested output is one column per leaf and needs no
        # flattening by the caller.
        paths = list(inner_outputs.event_template)
        resolved = [inner_outputs[path] for path in paths]
        if all(hasattr(v, "shape") and v.shape[:1] == (n_total,) for v in resolved):
            tpl = event_template or EventTemplate(
                dict(zip(paths, (v.shape[1:] for v in resolved), strict=True))
            )
            columns = {
                path: v.reshape(batch_shape + v.shape[1:])
                for path, v in zip(paths, resolved, strict=True)
            }
            shared = {
                "element_spec": tpl,
                "axis_groups": sweep_groups,
            }
            try:
                return NumericRecordBatch(columns, level_names, **shared)
            except (TypeError, ValueError):
                return RecordBatch(columns, level_names, **shared)

    # Fallback — shouldn't reach here with well-formed vmap output; if
    # we do, raise with the type info.
    raise TypeError(
        f"_make_stack cannot aggregate output of type "
        f"{type(inner_outputs).__name__}; expected a list, jnp.ndarray, "
        f"or batched Record."
    )


# ---------------------------------------------------------------------------
# BroadcastDistribution — joint over broadcast inputs and function output
# ---------------------------------------------------------------------------


def _record_rows(record: Record, rows: Any) -> Record:
    """*record* with every leaf reduced to *rows*, its specification re-derived.

    A record's pytree aux data carries its event template, whose shapes describe
    the leaves it was built from, so ``jax.tree.map`` would rebuild a record
    claiming the shapes it started with while holding gathered ones. Rebuilding by
    path lets the template be re-derived from the leaves that survived.
    """
    paths = tuple(record.event_template.keys())
    return type(record)(
        record.name, {path: record[path][rows] for path in paths}, name_is_auto=True
    )


def _row_count(component: Any) -> int:
    """How many rows one batched component holds.

    Every component is batched along its leading axis, but what reports that
    axis differs. An array has a ``shape`` and a list a ``len``. A record batch
    has neither that means the right thing — its ``len`` is the *field* count and
    its ``shape`` raises unless it holds a single leaf — so it reports
    ``batch_shape``. A plain record batched on its leaves has no ``batch_shape``
    either, which is the form ``RecordEmpiricalDistribution._sample`` returns, so
    its rows are read off a leaf.
    """
    batch_shape = getattr(component, "batch_shape", None)
    if batch_shape:
        return batch_shape[0]
    if isinstance(component, Record):
        leaf = next(iter(component.values()))
        return leaf.shape[0]
    return component.shape[0] if hasattr(component, "shape") else len(component)


def _gather_column(column: Any, indices: Array) -> Any:
    """One column's rows at *indices*, keeping the column's own kind.

    A numeric column gathers as an array. An object column holds one Python value
    per row and gathers through numpy, which takes the positions without trying to
    make an array of the values themselves.
    """
    if _is_object_array(column):
        return column[np.asarray(indices)]
    return column[indices]


def _take_rows(component: Any, indices: Array) -> Any:
    """Gather the rows *indices* of one batched component, keeping its container.

    A component of a broadcast is batched along its leading axis, but what carries
    that axis depends on what the component holds. An array is indexed directly. A
    list holds one object per row and is gathered positionally. A record has fields
    rather than a shape, so its leaves are gathered and the record rebuilt around
    them — a batch stating the row count it now holds, since that count is stored
    rather than read off the leaves.
    """
    if isinstance(component, list):
        return [component[int(i)] for i in indices]
    if isinstance(component, RecordBatch):
        # Only the leading axis's size changes, so the levels carry over with
        # that one size rewritten; which level holds the axis is unchanged.
        leading, *rest = component.axis_groups
        return type(component)(
            {
                path: _gather_column(component._raw_column(path), indices)
                for path in component.event_template
            },
            component.level_names,
            element_spec=component.element_spec,
            axis_groups=((indices.shape[0], *leading[1:]), *rest),
            name=component.name,
            name_is_auto=True,
        )
    if isinstance(component, Record):
        return _record_rows(component, indices)
    return component[indices]


def _one_row(component: Any) -> Any:
    """One drawn row of a component, presented on its own.

    ``sample_shape=()`` asks for a single draw rather than a batch of one, so the
    row is unwrapped: a record of that row rather than a one-row batch of records,
    and the object itself rather than a one-element list. Field names survive —
    one draw of a record-valued component is a record, whatever its field count.
    """
    if isinstance(component, RecordBatch):
        return component[0]
    if isinstance(component, Record):
        return _record_rows(component, 0)
    if isinstance(component, list):
        return component[0]
    return component[0] if hasattr(component, "__getitem__") else component


class BroadcastDistribution(Distribution[dict], SupportsSampling):
    """Joint distribution over broadcast inputs and function output.

    Stores the paired input–output samples from a
    :class:`~probpipe.core.node.Function` broadcast.  Supports
    joint sampling (resampling paired input–output tuples) and named
    component access.

    Call :meth:`marginalize` to obtain the output-only marginal, which
    supports moment protocols (mean, variance, etc.) when the output
    data permits.

    .. note::

       ``BroadcastDistribution`` does **not** inherit from
       :class:`~probpipe.distributions.joint.JointDistribution`.
       ``JointDistribution`` requires all leaves to be
       ``NumericRecordDistribution`` instances with TFP shape semantics
       (``batch_shape``, ``event_shape``), but a broadcast output can be
       any type — arrays, distributions, strings, etc. — and input
       samples are plain arrays without distribution metadata.  The two
       hierarchies serve different roles: ``JointDistribution`` models
       structured probabilistic variables; ``BroadcastDistribution``
       records the empirical input–output mapping of a function
       evaluation.

    Parameters
    ----------
    input_samples : dict[str, Array or RecordBatch or list]
        ``{arg_name: rows}`` for each broadcast argument, every value batched
        over the same leading axis of length ``n``: an array of shape
        ``(n, *event_shape)``, a record batch of ``batch_shape == (n,)`` for a
        record-valued argument, or a list of ``n`` objects.
    output_samples : Array, Record, or list
        The outputs, batched over the same axis: ``(n, *event_shape)`` for array
        outputs, a record whose leaves carry that axis for record-returning
        functions, or a list of length *n*.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Non-negative weights (normalized internally).  A pre-built
        :class:`~probpipe.Weights` object is also accepted.  Mutually
        exclusive with *log_weights*.  ``None`` for uniform.
    log_weights : array-like, :class:`~probpipe.Weights`, or None
        Log-unnormalized weights.  A pre-built :class:`~probpipe.Weights`
        object is also accepted.  Mutually exclusive with *weights*.
    output_distributions : list of Distribution or None
        When each function evaluation returns a ``Distribution``, these
        are the *n* component distributions for the mixture marginal.
    broadcast_args : list of str
        Ordered names of the broadcast arguments.
    name : str or None
        Distribution name for provenance.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        input_samples: dict[str, Any],
        output_samples: Any,
        weights: Array | Weights | None = None,
        *,
        log_weights: Array | Weights | None = None,
        output_distributions: list | None = None,
        broadcast_args: list[str],
        name: str | None = None,
        output_template: EventTemplate | None = None,
    ):
        self._input_samples = input_samples
        self._output_samples = output_samples
        self._output_distributions = output_distributions
        self._output_template = output_template

        # The row count, taken from the first broadcast arg.
        n = _row_count(input_samples[next(iter(broadcast_args))])
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)
        self._broadcast_args = list(broadcast_args)
        name, name_is_auto = auto_name(name, "broadcast")
        super().__init__(name=name, name_is_auto=name_is_auto)
        self._approximate = True
        self._marginal_cache: MarginalizedBroadcastDistribution | None = None

    # -- basic properties ---------------------------------------------------

    @property
    def num_atoms(self) -> int:
        """Number of input–output pairs."""
        return self._w.n

    @property
    def weights(self) -> Array:
        """Normalised weights, shape ``(n,)``."""
        return self._w.normalized

    @property
    def input_samples(self) -> dict[str, Any]:
        """Broadcast input samples: ``{arg_name: (n, *event_shape)}``."""
        return self._input_samples

    @property
    def samples(self) -> Any:
        """Output samples (forwarded to output marginal for backward compat)."""
        m = self.marginalize()
        return m.samples if hasattr(m, "samples") else m.items

    # -- Named components ----------------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        return (*tuple(self._broadcast_args), "_output")

    def __getitem__(self, key: str):
        if key == "_output":
            return self.marginalize()
        if key in self._input_samples:
            arr = self._input_samples[key]
            return EmpiricalDistribution(arr, weights=self._w, name=key)
        raise KeyError(f"Unknown component {key!r}; available: {self.fields}")

    # -- joint sampling -----------------------------------------------------

    def _sample(self, key, sample_shape=()):
        """Resample paired input–output tuples.

        Every component is batched along the same leading axis, so one set of
        drawn rows is gathered from each — the inputs and the output alike, which
        is what keeps a drawn tuple paired.
        """
        n_draws = prod(sample_shape) if sample_shape else 1
        indices = self._w.choice(key, shape=(n_draws,))

        result = {
            name: _take_rows(self._input_samples[name], indices) for name in self._broadcast_args
        }
        result["_output"] = _take_rows(self._output_samples, indices)

        if sample_shape == ():
            return {name: _one_row(component) for name, component in result.items()}
        return result

    # -- marginalization ----------------------------------------------------

    def marginalize(self) -> MarginalizedBroadcastDistribution:
        """Return the output marginal distribution.

        Lazy — the marginal is constructed on first call and cached.
        The marginal inherits this distribution's provenance (if any)
        so the lineage is preserved without a direct reference to the
        ``BroadcastDistribution``.
        """
        if self._marginal_cache is None:
            self._marginal_cache = _make_marginal(
                self._output_samples,
                self._w,
                output_distributions=self._output_distributions,
                event_template=self._output_template,
            )
            if self.provenance is not None and isinstance(self._marginal_cache, Distribution):
                self._marginal_cache.with_provenance(self.provenance)
        return self._marginal_cache

    @property
    def output(self) -> MarginalizedBroadcastDistribution:
        """Alias for :meth:`marginalize`."""
        return self.marginalize()

    def __repr__(self):
        return (
            f"BroadcastDistribution(num_atoms={self.num_atoms}, "
            f"broadcast_args={self._broadcast_args})"
        )
