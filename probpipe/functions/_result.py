"""Wrap, label, and attach provenance to Function results at their own kind.

This is the return step of the call stack: it validates the produced terms
against the completed declaration, wraps a raw host into the kind its spec
names, labels the result, and attaches its provenance, or detaches the result
when the call asks for its raw form. A result that violates its declaration is
a defect of the function, reported as :class:`ResultKindError` or
:class:`ResultSchemaError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import prod
from typing import Any, Literal, cast

import jax.numpy as jnp
import numpy as np

from ..core._array_backend import _event_shape_of, _numpy_dtype_of, _to_jax_array
from ..core._batch import Batch, BatchSpec, _ranks_of
from ..core._function_batch import FunctionBatch
from ..core._kinds import batch_class_for_spec
from ..core._numeric_array import NumericArray
from ..core._numeric_array_batch import NumericArrayBatch, _MappedBatchStore
from ..core._numeric_record import _is_numeric_leaf
from ..core._numeric_record_batch import NumericRecordBatch
from ..core._object_batch import _from_iterable, _is_object_array, _ObjectBatch
from ..core._opaque import Opaque
from ..core._opaque_batch import OpaqueBatch
from ..core._record_batch import RecordBatch, _batch_class_for, _MappedBatchColumns
from ..core._record_spec import _reshaped_template
from ..core._spec_base import _full_array_shape_or_none, _known_type, _unify_specs
from ..core._specs import NumericArraySpec, OpaqueSpec, OutputSpec, RecordSpec
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..distributions._batches import DistributionBatch
from ..distributions._distribution import Distribution
from ..values._function_base import _validate_function_output

BroadcastMode = Literal["wrap", "stack", "nested"]
BROADCAST_WRAP: BroadcastMode = "wrap"
BROADCAST_STACK: BroadcastMode = "stack"
BROADCAST_NESTED: BroadcastMode = "nested"

#: The hosts an undeclared return reads as one ``Opaque`` rather than a batch.
_SEQUENCES = (list, tuple, set, frozenset)


class ResultKindError(TypeError):
    """A function returned a term of another kind than its declaration names.

    This is a defect of the function's return contract, not a failure to admit
    the caller's arguments.
    """


class ResultSchemaError(ValueError):
    """A function's result is incompatible with its completed declaration.

    Raised for incompatible structure, dimensions, dtype, or support, and for a
    violated declared output interface. This is a defect of the function's
    return contract, not a failure to admit the caller's arguments.
    """


def _detach(result: Any) -> Any:
    """The result detached from the workflow, as its ``raw()`` returns it."""
    return raw_form(result)


def _wrap_declared_function_output(
    result: Any, *, function_name: str, output_spec: OutputSpec
) -> Any:
    """Wrap a validated result at its declared kind, preserving component exposure."""
    spec = output_spec.spec
    if isinstance(result, TrackedTerm):
        return _copy_result_term(result, output_spec=output_spec)
    if isinstance(spec, RecordSpec):
        return Record(function_name, result, event_template=spec)
    if isinstance(spec, NumericArraySpec):
        from ..core._numeric_array import NumericArray

        return NumericArray(function_name, result, spec=spec)
    if isinstance(spec, OpaqueSpec):
        return Opaque(function_name, result, spec=spec)
    from ..values import Function, FunctionSpec

    if isinstance(spec, FunctionSpec):
        return Function(
            function_name, result, input_spec=spec.input_spec, output_spec=spec.output_spec
        )
    return _wrap_as_term(result, function_name)


def _aggregate_output_spec(output_spec: OutputSpec, outputs: Any) -> OutputSpec:
    """Complete a declaration from validated rows, retaining its component names.

    Mapped outputs carry the row's spec as static metadata, independently of
    the leading axis JAX added. Empty outputs leave the declaration unchanged.
    Row-wise results share dimension bindings so incompatible rows are refused.

    Parameters
    ----------
    output_spec : OutputSpec
        The declaration to complete.
    outputs : Any
        The validated rows: a list of row-wise results, or one mapped output.

    Returns
    -------
    OutputSpec
        *output_spec* with its spec completed by the rows and its shared
        dimensions bound.

    Raises
    ------
    ResultSchemaError
        If a row does not unify with the declaration or with the other rows.
    """
    from ..core._batch import BatchSpec
    from ..core._numeric_array_batch import _MappedBatchStore
    from ..core._record_batch import _MappedBatchColumns
    from ..core._spec_base import _unify_specs
    from ..values._function_base import _complete_output_metadata, _produced_spec

    spec = output_spec.spec
    bindings: dict[str, int] = {}
    rows = outputs if isinstance(outputs, list) else (outputs,)
    for row in rows:
        if isinstance(row, (_MappedBatchColumns, _MappedBatchStore)):
            actual = (
                BatchSpec(row.element_spec, row.axis_groups, row.level_names)
                if row.axis_groups
                else row.element_spec
            )
        else:
            actual = _produced_spec(row)
        if spec is None:
            spec = actual
        try:
            _unify_specs(spec, actual, bindings, "Function aggregate output")
        except ValueError as error:
            raise ResultSchemaError(str(error)) from error
        spec = _complete_output_metadata(spec, actual)
    return output_spec._with_spec(None if spec is None else spec._substitute_dims(bindings))


def _output_record_spec(output_spec: OutputSpec) -> RecordSpec | None:
    """Return a record-valued output's schema for the existing aggregate builders.

    Returned laws retain their own event declarations. Their declarations are
    matched by DistributionSpec, without projecting them onto a record template.
    """
    spec = output_spec.spec
    if isinstance(spec, RecordSpec):
        return spec
    return None


def _wrap_as_term(value: Any, result_name: str) -> Any:
    """Wrap a raw host as its tracked kind under the result label *result_name*.

    Mappings become records, numeric values become NumericArray, callables
    become Function, and other values, a list, a tuple, or a set among them,
    become Opaque, since a batch is declared through ``output_spec``. Existing
    tracked terms are retained here and copied by the public result boundary.
    """
    match value:
        case TrackedTerm():
            return value
        case Mapping():
            return Record(result_name, dict(value))
        case _ if _is_numeric_leaf(value):
            from ..core._numeric_array import NumericArray

            return NumericArray(result_name, value)
        case _ if callable(value):
            from ..values import Function

            return Function(result_name, value)
        case _:
            from ..core._opaque import Opaque

            return Opaque(result_name, value)


def _coerce_output(
    value: Any,
    *,
    broadcast_mode: BroadcastMode,
    provenance: Provenance | None,
    field_name: str,
) -> Any:
    """Return an independently labeled term with this call's provenance.

    ``field_name`` is the Function's output_label, separate from the function
    label in provenance and the declared output components. A tracked
    return is shallow-copied, sharing value data while owning its metadata.
    """
    if broadcast_mode == BROADCAST_WRAP and not isinstance(value, TrackedTerm):
        value = _wrap_as_term(value, field_name)
    elif isinstance(value, TrackedTerm):
        value = value._with_label(field_name)
    if isinstance(value, TrackedTerm) and provenance is not None:
        value.with_provenance(provenance)
    return value


def _copy_result_term(value: TrackedTerm, *, output_spec: OutputSpec | None = None) -> TrackedTerm:
    """Copy a tracked result while retaining its value and declaration metadata."""
    clone = value._shallow_copy()
    spec = output_spec.spec if output_spec is not None else None
    if isinstance(spec, RecordSpec) and isinstance(clone, Record):
        object.__setattr__(clone, "_spec", spec)
    elif isinstance(spec, NumericArraySpec):
        from ..core._numeric_array import NumericArray

        if isinstance(clone, NumericArray):
            object.__setattr__(clone, "_spec", spec)
    elif isinstance(spec, OpaqueSpec):
        if isinstance(clone, Opaque):
            # The declaration and the term's own spec unified at completion, so the
            # term takes the known type and the set meta of the two.
            object.__setattr__(clone, "_spec", _known_type(spec, clone.spec))
    elif spec is not None:
        from ..core._batch import Batch, BatchSpec
        from ..values import Function, FunctionSpec

        if isinstance(spec, FunctionSpec) and isinstance(clone, Function):
            from ..values._function_base import _validate_function_declarations

            # A side the declaration leaves unspecified keeps the returned function's own.
            spec = FunctionSpec(
                input_spec=clone.input_spec if spec.input_spec is None else spec.input_spec,
                output_spec=clone.output_spec if spec.output_spec is None else spec.output_spec,
            )
            _validate_function_declarations(
                function_name=clone.label,
                signature=clone.signature,
                input_spec=spec.input_spec,
                construction_bindings=clone._bind,
            )
            object.__setattr__(clone, "_spec", spec)
        elif isinstance(spec, BatchSpec) and isinstance(clone, Batch):
            object.__setattr__(clone, "_spec", spec)
            if isinstance(clone, RecordBatch) and isinstance(spec.element_spec, RecordSpec):
                columns = clone._raw_columns()
                object.__setattr__(
                    clone, "_columns", {path: columns[path] for path in spec.element_spec}
                )
    object.__setattr__(clone, "_provenance", None)
    return clone


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
    name: str,
    records: list[Record] | Record,
    /,
    *,
    batch_shape: tuple[int, ...],
    axes_per_level: tuple[int, ...],
    level_names: tuple[str, ...],
    template: RecordSpec,
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
            values = [
                record.raw(path) if isinstance(record, Record) else record[path]
                for record in records
            ]
            # The declared *kind* decides the storage, not what the values happen
            # to look like. An opaque field holding one array per row is a column
            # of two objects, not a numeric column whose second axis is another
            # multiplicity — reading it off the runtime shape states a batch
            # geometry the levels never described.
            if isinstance(template[path], NumericArraySpec):
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
            batched = records.raw(path)
            if not isinstance(template[path], NumericArraySpec):
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
        name,
        columns,
        level_names,
        element_spec=template,
        axes_per_level=axes_per_level,
    )


def _empty_declared_stack(
    name: str,
    batch_shape: tuple[int, ...],
    /,
    *,
    template: RecordSpec,
    level_names: tuple[str, ...],
    axes_per_level: tuple[int, ...],
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
        if isinstance(spec, NumericArraySpec) and all(isinstance(size, int) for size in spec.shape):
            dtype = spec.dtype if spec.dtype is not None else jnp.zeros(()).dtype
            columns[path] = jnp.zeros((*batch_shape, *spec.shape), dtype=dtype)
        else:
            columns[path] = np.empty(batch_shape, dtype=object)
    cls = _batch_class_for(template)
    return cls(
        name,
        columns,
        level_names,
        element_spec=template,
        axes_per_level=axes_per_level,
    )


# ---------------------------------------------------------------------------
# _make_stack — the row aggregator of a sweep, a lift, and a batch of draws
# ---------------------------------------------------------------------------
#
# A sweep's rows are independent scenarios indexed by input row, so the
# aggregate keeps row identity. Each row is first given the kind of the host it
# is, then the rows aggregate at that kind:
#
#   numeric   → NumericArrayBatch          Record       → RecordBatch
#   opaque    → OpaqueBatch                Distribution → DistributionBatch
#   callable  → FunctionBatch
#
# A row that is itself a batch stacks into one batch, the sweep's levels in
# front of the rows' own. Provenance is attached by ``_coerce_output``.
# ---------------------------------------------------------------------------


#: The level the draws of a law lie on: an operation names the level it mints
#: after itself (II.5), and ``sample`` mints this one.
SAMPLE_LEVEL = "sample"

#: The level the atoms of a lifted call's empirical result lie on, one atom per
#: evaluation of the function, whether drawn or enumerated.
DRAW_LEVEL = "draw"


def _batch_over_swept_columns(
    name: str,
    columns: dict[str, Any],
    /,
    *,
    batch_shape: tuple[int, ...],
    sweep_level_names: tuple[str, ...],
    sweep_groups: tuple[tuple[int, ...], ...],
    element_spec: Any,
    inner_level_names: tuple[str, ...],
    inner_axis_groups: tuple[tuple[int, ...], ...],
) -> RecordBatch:
    """Build the aggregate for a sweep whose rows each held a batch.

    *columns* arrive stacked on a single leading axis of ``prod(batch_shape)``,
    which the sweep's own axes replace; the levels are the sweep's followed by
    the rows'. Both dispatches land here — the row-wise one after stacking the
    rows itself, the mapped one with the transform's output axis already in
    place — so the two agree by construction rather than by being maintained in
    parallel.
    """
    return _batch_class_for(element_spec)(
        name,
        {path: column.reshape(batch_shape + column.shape[1:]) for path, column in columns.items()},
        (*sweep_level_names, *inner_level_names),
        element_spec=element_spec,
        axes_per_level=_ranks_of((*sweep_groups, *inner_axis_groups)),
    )


def _agreeing_batch_rows(outs: list, *, field_name: str) -> Any:
    """The first row, once every row is a batch that agrees with it.

    A batch row is all-or-nothing. Falling through on a mixture, or on rows that
    disagree, reaches the generic handlers — which would read a row's own batch
    axis as an event axis and discard its level names with it. Matching the
    shape alone would take the first row's schema for all of them, dropping a
    field the others have and misnaming their axes.
    """
    first = outs[0]
    family = None
    # The family, not the exact class: a RecordBatch and a NumericRecordBatch row
    # hold the same thing, and only one of them says so in its name. An
    # OpaqueBatch and a FunctionBatch share their storage but not their element
    # spec, so the spec comparison below is what separates them.
    for candidate in (NumericArrayBatch, _ObjectBatch, RecordBatch):
        if isinstance(first, candidate):
            family = candidate
            break
    if family is None or not all(isinstance(o, family) for o in outs):
        kinds = sorted({type(o).__name__ for o in outs})
        raise TypeError(
            f"{field_name}: some rows returned a batch and some did not "
            f"({', '.join(kinds)}). A swept body returns one kind for every row, since "
            f"the aggregate has one schema; return a batch from every row or from none"
        )
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
    return first


def _row_at_its_kind(row: Any, field_name: str) -> Any:
    """One swept row as the tracked term of its own kind.

    Only the two hosts the branches below would misread are converted here. A
    numeric or callable row already reaches a branch that batches it at its
    kind, and converting it early would cost the vectorized path for no gain.

    - a ``Mapping`` is a tree, so it becomes a ``Record`` and the rows stack into
      a batch of records;
    - a list, a tuple, or a set is an ``Opaque``, as ``_wrap_as_term`` reads one
      returned on its own, so the rows store one opaque element each rather than
      stacking the sequence's items as an event axis.
    """
    if isinstance(row, Mapping):
        return Record(field_name, dict(row))
    if isinstance(row, _SEQUENCES):
        return Opaque(field_name, row)
    return row


def _batch_from_declared_sequence(
    result: Any, *, function_name: str, output_spec: OutputSpec | None
) -> Any:
    """A raw sequence returned under a batch declaration, as the declared batch.

    A batch is declared through ``output_spec``, so a list or tuple returned
    under a ``BatchSpec`` holds the batch's elements in row-major order over its
    declared axes. A declared axis of symbolic size takes the sequence's length
    when it is the only axis. Any other result is returned as it is.
    """
    spec = None if output_spec is None else output_spec.spec
    if not isinstance(spec, BatchSpec) or not isinstance(result, (list, tuple)):
        return result
    sizes = [size for group in spec.axis_groups for size in group]
    if len(sizes) == 1:
        batch_shape: tuple[int, ...] = (len(result),)
        axis_groups: tuple[tuple[int, ...], ...] = (batch_shape,)
    elif all(isinstance(size, int) for size in sizes):
        batch_shape = tuple(sizes)
        axis_groups = tuple(tuple(group) for group in spec.axis_groups)
    else:
        return result
    element = spec.element_spec
    is_record = isinstance(element, RecordSpec)
    return _make_stack(
        list(result),
        batch_shape=batch_shape,
        level_names=tuple(spec.level_names),
        axis_groups=axis_groups,
        name=function_name,
        field_name=function_name,
        output_template=element if is_record else None,
        # The aggregator reads only the declared element spec, so the component
        # name of this declaration is never read.
        output_spec=None if is_record else OutputSpec.default(element, component="element"),
    )


def _make_stack(
    inner_outputs: Any,
    *,
    batch_shape: tuple[int, ...] | None = None,
    n: int | None = None,
    level_names: tuple[str, ...],
    axis_groups: tuple[tuple[int, ...], ...] | None = None,
    name: str | None = None,
    field_name: str,
    output_template: RecordSpec | None = None,
    output_spec: OutputSpec | None = None,
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
    axis_groups : tuple of tuple of int, optional
        The sizes of the axes each level spans, which partition
        ``batch_shape``; ``None`` gives every axis to one level.
    name : str, optional
        The resulting aggregate's label.
    field_name : str
        The label a wrapped row takes, and the aggregate's label when *name* is
        ``None``.
    output_template : RecordSpec, optional
        The declared record of a row, by which each row is wrapped and the
        columns are laid out.
    output_spec : OutputSpec, optional
        The completed output declaration, whose spec is the element
        declaration of an empty sweep's batch and of a batch of objects.

    Returns
    -------
    Batch
        The batch form of the rows' kind, on the given levels, whose element
        declaration records the dtypes its store holds (see
        :func:`_record_stored_dtypes`): a
        ``NumericArrayBatch``, a ``RecordBatch`` or ``NumericRecordBatch``, a
        ``DistributionBatch``, a ``FunctionBatch``, or an ``OpaqueBatch``.

    Raises
    ------
    TypeError
        If the inner outputs can't be coerced into any of the three
        aggregate types. The error lists the observed types.
    ValueError
        If rows have incompatible declarations, shapes, or batch levels, or
        the output shape does not match the requested batch shape and grouping.
    """
    return _record_stored_dtypes(
        _stack_rows(
            inner_outputs,
            batch_shape=batch_shape,
            n=n,
            level_names=level_names,
            axis_groups=axis_groups,
            name=name,
            field_name=field_name,
            output_template=output_template,
            output_spec=output_spec,
        )
    )


def _record_stored_dtypes(aggregate: Any) -> Any:
    """*aggregate* with its element declaration recording the dtypes its store holds.

    The rows of a sweep or a lift are stacked into one JAX array, which holds
    their canonical dtype, so 64-bit rows are stored as 32-bit ones while JAX's
    64-bit mode is off. A declared, inferred, or tracked row dtype may therefore
    differ from the stored one, and the declaration takes the stored dtype:
    an array element records it always, and a record's array field records it
    where its declaration states a dtype. Any other aggregate is returned as it is.
    """
    if isinstance(aggregate, NumericArrayBatch):
        element = _with_stored_dtypes(aggregate.element_spec, aggregate.values, fill=True)
    elif isinstance(aggregate, RecordBatch):
        element = _with_stored_dtypes(aggregate.element_spec, aggregate._raw_columns(), fill=False)
    else:
        return aggregate
    if element != aggregate.element_spec:
        object.__setattr__(aggregate, "_spec", replace(aggregate.spec, element_spec=element))
    return aggregate


def _with_stored_dtypes(spec: Any, store: Any, *, fill: bool, path: str = "") -> Any:
    """*spec* with each array leaf's dtype set to the dtype its column of *store* holds.

    *store* is one array for an array spec and the columns keyed by leaf path
    for a record spec. A leaf that declares no dtype takes the stored one only
    under *fill*, and a column with no single dtype leaves its leaf as it is.
    """
    if isinstance(spec, RecordSpec):
        return RecordSpec(
            {
                key: _with_stored_dtypes(
                    child, store, fill=fill, path=f"{path}/{key}" if path else key
                )
                for key, child in spec.children.items()
            }
        )
    if not isinstance(spec, NumericArraySpec):
        return spec
    column = store[path] if path else store
    stored = _numpy_dtype_of(column)
    if stored is None or (spec.dtype is None and not fill) or spec.dtype == stored:
        return spec
    return replace(spec, dtype=stored)


def _with_dtypes_of(spec: Any, reference: Any) -> Any:
    """*spec* with each array leaf's dtype taken from the matching leaf of *reference*.

    A leaf whose match in *reference* states no dtype keeps its own, so a law's
    event declaration can take the dtypes its stored atoms hold.
    """
    if isinstance(spec, RecordSpec) and isinstance(reference, RecordSpec):
        return RecordSpec(
            {
                key: _with_dtypes_of(child, reference.children[key])
                if key in reference.children
                else child
                for key, child in spec.children.items()
            }
        )
    if (
        isinstance(spec, NumericArraySpec)
        and isinstance(reference, NumericArraySpec)
        and reference.dtype is not None
        and spec.dtype != reference.dtype
    ):
        return replace(spec, dtype=reference.dtype)
    return spec


def _stack_rows(
    inner_outputs: Any,
    *,
    batch_shape: tuple[int, ...] | None = None,
    n: int | None = None,
    level_names: tuple[str, ...],
    axis_groups: tuple[tuple[int, ...], ...] | None = None,
    name: str | None = None,
    field_name: str,
    output_template: RecordSpec | None = None,
    output_spec: OutputSpec | None = None,
) -> Any:
    """The aggregate :func:`_make_stack` returns, before it records the stored dtypes."""
    result_name = field_name if name is None else name

    # Resolve batch_shape vs. n. Exactly one must be provided.
    if batch_shape is None:
        if n is None:
            raise TypeError("_make_stack requires either batch_shape or n")
        batch_shape = (n,)
    elif n is not None:
        raise TypeError("_make_stack: pass batch_shape OR n, not both")
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

    # --- Mapped batch-returning body -----------------------------------
    # Before the generic pytree handling below, which would read the inner batch
    # axis as an event axis and drop the level names with it — the same reason
    # the batch-of-batches case precedes the Record case on the list path.
    if isinstance(inner_outputs, _MappedBatchStore):
        # One store rather than columns: the sweep's axes replace the leading one
        # the transform produced, and the levels are the sweep's then the rows'.
        store = inner_outputs.store
        return NumericArrayBatch(
            result_name,
            store.reshape(batch_shape + store.shape[1:]),
            (*level_names, *inner_outputs.level_names),
            element_spec=inner_outputs.element_spec,
            axes_per_level=_ranks_of((*sweep_groups, *inner_outputs.axis_groups)),
        )
    if isinstance(inner_outputs, _MappedBatchColumns):
        return _batch_over_swept_columns(
            result_name,
            inner_outputs.columns,
            batch_shape=batch_shape,
            sweep_level_names=level_names,
            sweep_groups=sweep_groups,
            element_spec=inner_outputs.element_spec,
            inner_level_names=inner_outputs.level_names,
            inner_axis_groups=inner_outputs.axis_groups,
        )

    if (
        isinstance(inner_outputs, list)
        and not inner_outputs
        and n_total == 0
        and output_spec is not None
    ):
        spec = output_spec.spec
        if spec is None or not spec.is_concrete:
            raise ValueError("An empty sweep requires a concrete output_spec")
        if isinstance(spec, NumericArraySpec):
            dtype = spec.dtype if spec.dtype is not None else jnp.zeros(()).dtype
            return NumericArrayBatch(
                result_name,
                jnp.empty((*batch_shape, *spec.shape), dtype=dtype),
                level_names,
                element_spec=spec,
                axes_per_level=_ranks_of(sweep_groups),
            )
        if not isinstance(spec, RecordSpec):
            batch_class = batch_class_for_spec(spec)
            if batch_class is None:
                raise ValueError(f"An empty sweep has no batch form for {spec!r}")
            return batch_class(
                result_name,
                np.empty(batch_shape, dtype=object),
                level_names,
                element_spec=spec,
                axes_per_level=_ranks_of(sweep_groups),
            )

    # --- List-of-X path (Python-loop execution) -------------------------
    if isinstance(inner_outputs, list):
        # With no rows there is no output to read a type off, so the declared
        # template is the only honest source; without one, the generic handlers
        # below would name a single opaque field after the function. Only a
        # sweep that *expects* zero rows takes this path — an empty list where
        # rows were expected is a missing-output error, and fabricating the
        # declared fields would hide it.
        if not inner_outputs and n_total == 0 and output_template is not None:
            return _empty_declared_stack(
                result_name,
                batch_shape,
                template=output_template,
                level_names=level_names,
                axes_per_level=_ranks_of(sweep_groups),
            )
        if len(inner_outputs) != n_total:
            raise ValueError(
                f"_make_stack got {len(inner_outputs)} outputs but "
                f"expected prod(batch_shape)={n_total} "
                f"(batch_shape={batch_shape})."
            )
        outs: Any = inner_outputs
        if output_template is not None:
            outs = [
                _wrap_declared_function_output(
                    output,
                    function_name=field_name,
                    output_spec=OutputSpec(output_template),
                )
                for output in outs
            ]
        else:
            # Each row takes the kind of the host it is, before the branches
            # below read what the rows are. A row is one call's return, and the
            # boundary that names a *single* return's kind (V.0) is the same rule
            # — reading a mapping row as an unstackable object, or a sequence row
            # as event shape, states something the row never said.
            outs = [_row_at_its_kind(output, field_name) for output in outs]

        # A batch per row stacks into one batch with the sweep in front of the
        # rows' own levels. Checked before the Record branch below, which would
        # otherwise claim a batch of records and collapse its
        # inner batch axis.
        # Both batch kinds a swept body can return; each keeps its own kind
        # through the sweep, as a scalar row does.
        stackable = (RecordBatch, NumericArrayBatch, _ObjectBatch)
        if outs and any(isinstance(o, stackable) for o in outs):
            first = _agreeing_batch_rows(outs, field_name=field_name)
            if isinstance(first, _ObjectBatch):
                # The elements are stored, not stacked, so the aggregate is one
                # object array over the sweep's axes then the rows' own.
                store = np.stack([o._store for o in outs], axis=0)
                return type(first)(
                    result_name,
                    store.reshape(batch_shape + store.shape[1:]),
                    (*level_names, *first.level_names),
                    element_spec=first.element_spec,
                    axes_per_level=_ranks_of((*sweep_groups, *first.axis_groups)),
                )
            if isinstance(first, NumericArrayBatch):
                # One store rather than columns, so the rows stack directly. Each
                # row converts through ``as_jax``, not through the backend on its
                # raw store: the batch reuses concrete conversions and keeps
                # traced conversions within their transform.
                store = jnp.stack([o.as_jax() for o in outs], axis=0)
                return NumericArrayBatch(
                    result_name,
                    store.reshape(batch_shape + store.shape[1:]),
                    (*level_names, *first.level_names),
                    element_spec=first.element_spec,
                    axes_per_level=_ranks_of((*sweep_groups, *first.axis_groups)),
                )
            # Columns are leaf-keyed, so a nested element needs no special
            # case — and they are read raw: a field that is not an array
            # presents as its own object batch, and what stacks is the
            # column, through numpy so the objects are taken as they are.
            columns = {}
            for path in first.event_template:
                cols = [o._raw_column(path) for o in outs]
                if any(_is_object_array(c) for c in cols):
                    columns[path] = np.stack(cols, axis=0)
                else:
                    columns[path] = jnp.stack(cols, axis=0)
            return _batch_over_swept_columns(
                result_name,
                columns,
                batch_shape=batch_shape,
                sweep_level_names=level_names,
                sweep_groups=sweep_groups,
                element_spec=first.element_spec,
                inner_level_names=tuple(first.level_names),
                inner_axis_groups=tuple(first.axis_groups),
            )

        # All (scalar) Records → stack into one batch. NumericRecordBatch if
        # every leaf is numeric; otherwise the permissive RecordBatch, building the
        # columns manually so non-numeric leaves (strings, xarray objects, ...)
        # survive.
        if outs and all(isinstance(o, Record) for o in outs):
            if output_template is not None:
                return _stack_declared_columns(
                    result_name,
                    outs,
                    batch_shape=batch_shape,
                    axes_per_level=_ranks_of(sweep_groups),
                    level_names=level_names,
                    template=output_template,
                )
            # Stack flat, then reshape the leading axis to batch_shape.
            try:
                flat = NumericRecordBatch.stack(list(outs), level_name=level_names[0])
            except (TypeError, ValueError):
                flat = None
            if flat is not None:
                # No early return for the one-level case: the reshape below is
                # an identity there, and ``stack`` labeled the batch after its own
                # class, where every aggregation labels it for the function that
                # produced the rows.
                n_cur = len(flat.batch_shape)
                return NumericRecordBatch(
                    result_name,
                    {
                        path: flat._raw_column(path).reshape(
                            batch_shape + flat._raw_column(path).shape[n_cur:]
                        )
                        for path in flat.event_template
                    },
                    level_names,
                    element_spec=flat.element_spec,
                    axes_per_level=_ranks_of(sweep_groups),
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
                result_name,
                columns,
                level_names,
                element_spec=flat.element_spec,
                axes_per_level=_ranks_of(sweep_groups),
            )

        # All Distributions → a DistributionBatch over the sweep's levels, whose
        # rows share the first row's declaration.
        if outs and all(isinstance(o, Distribution) for o in outs):
            return DistributionBatch(
                result_name,
                _from_iterable(outs, kind="_make_stack").reshape(batch_shape),
                level_names,
                axes_per_level=_ranks_of(sweep_groups),
            )

        # Numeric scalars / arrays → the batch form of their own kind, with the
        # leading axis re-cut to batch_shape.
        try:
            stacked = jnp.stack(
                [jnp.asarray(o) for o in outs],
                axis=0,
            )
        except (TypeError, ValueError):
            stacked = None

        if stacked is not None:
            event_shape = tuple(stacked.shape[1:])
            element_spec = NumericArraySpec(event_shape, dtype=stacked.dtype)
            if all(isinstance(output, NumericArray) for output in outs):
                specs = []
                for output in outs:
                    spec = output.spec
                    if spec.free_dims:
                        bindings: dict[str, int] = {}
                        spec._bind_dims_from_value(output, bindings, field_name)
                        spec = spec._substitute_dims(bindings)
                    specs.append(spec)
                element_spec = specs[0]
                for spec in specs[1:]:
                    if replace(spec, dtype=element_spec.dtype) != element_spec:
                        raise ValueError(
                            f"{field_name}: numeric rows returned declarations that disagree "
                            f"({element_spec!r} and {spec!r}); return numeric rows with "
                            "one shared event shape and support"
                        )
                # Promote the complete set: pairwise NumPy promotion can depend
                # on row order. JAX handles extended dtypes such as bfloat16.
                dtypes = [spec.dtype for spec in specs]
                if any(dtype is None for dtype in dtypes):
                    dtype = None
                else:
                    try:
                        dtype = np.result_type(*dtypes)
                    except np.exceptions.DTypePromotionError:
                        dtype = jnp.result_type(*dtypes)
                element_spec = replace(element_spec, dtype=dtype)
            return NumericArrayBatch(
                result_name,
                stacked.reshape(batch_shape + event_shape),
                level_names,
                element_spec=element_spec,
                axes_per_level=_ranks_of(sweep_groups),
            )

        # Numeric rows that do not stack disagree on their shape, and an object
        # column would record that disagreement as though it were the answer. The
        # rows say what they are; the aggregate says so too.
        if outs and all(_is_numeric_leaf(o) for o in outs):
            shapes = sorted({tuple(_event_shape_of(o)) for o in outs})
            raise ValueError(
                f"{field_name}: the rows returned numeric values of differing shapes "
                f"{shapes}, so they do not stack into one batch. Every row of a sweep "
                f"contributes one element of one shape; pad the rows, or return a batch "
                f"from each and let the levels record the difference"
            )

        # Rows that do not stack take the batch form of their own kind, which is
        # the multiplicity side of the table ``_wrap_as_term`` states for one
        # value: a callable batches as a FunctionBatch and anything else as an
        # OpaqueBatch. Both store their elements, so nothing has to stack.
        try:
            object_array = _from_iterable(outs, kind="_make_stack").reshape(batch_shape)
            shared = {
                "axes_per_level": _ranks_of(sweep_groups),
            }
            if output_spec is not None:
                shared["element_spec"] = output_spec.spec
            # ``outs`` first: every row of none is vacuously callable, and no row
            # is a reason to claim the function kind over the fallback.
            if outs and all(callable(o) for o in outs):
                return FunctionBatch(result_name, object_array, level_names, **shared)
            return OpaqueBatch(result_name, object_array, level_names, **shared)
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
        if output_template is not None:
            if len(output_template) != 1:
                raise ValueError(
                    "bare array aggregation requires a single-leaf output_template; "
                    "authoritative Function outputs must be wrapped before aggregation"
                )
            output_field = next(iter(output_template.keys()))
            batched_record = Record(
                result_name,
                {output_field: inner_outputs},
            )
            return _stack_declared_columns(
                result_name,
                batched_record,
                batch_shape=batch_shape,
                axes_per_level=_ranks_of(sweep_groups),
                level_names=level_names,
                template=output_template,
            )
        return NumericArrayBatch(
            result_name,
            inner_outputs.reshape(batch_shape + event_shape),
            level_names,
            element_spec=NumericArraySpec(event_shape, dtype=inner_outputs.dtype),
            axes_per_level=_ranks_of(sweep_groups),
        )

    # vmap of a Record-returning function produces a Record with batched leaves
    # (each leaf has leading axis n_total). Promote it to a batch — numeric when
    # every leaf is — with the leading axis reshaped to batch_shape.
    if isinstance(inner_outputs, Record) and inner_outputs.children:
        if output_template is not None:
            return _stack_declared_columns(
                result_name,
                inner_outputs,
                batch_shape=batch_shape,
                axes_per_level=_ranks_of(sweep_groups),
                level_names=level_names,
                template=output_template,
            )
        # Leaf-keyed, so a nested output is one column per leaf and needs no
        # flattening by the caller.
        paths = list(inner_outputs.event_template)
        resolved = [inner_outputs.raw(path) for path in paths]
        if all(hasattr(v, "shape") and v.shape[:1] == (n_total,) for v in resolved):
            tpl = output_template or RecordSpec(
                dict(zip(paths, (v.shape[1:] for v in resolved), strict=True))
            )
            columns = {
                path: v.reshape(batch_shape + v.shape[1:])
                for path, v in zip(paths, resolved, strict=True)
            }
            shared = {
                "element_spec": tpl,
                "axes_per_level": _ranks_of(sweep_groups),
            }
            try:
                return NumericRecordBatch(result_name, columns, level_names, **shared)
            except (TypeError, ValueError):
                return RecordBatch(result_name, columns, level_names, **shared)

    # Fallback — shouldn't reach here with well-formed vmap output; if
    # we do, raise with the type info.
    raise TypeError(
        f"_make_stack cannot aggregate output of type "
        f"{type(inner_outputs).__name__}; expected a list, jnp.ndarray, "
        f"or batched Record."
    )


# ---------------------------------------------------------------------------
# The return of a Function realized by routes
# ---------------------------------------------------------------------------


def declared_term(value: Any, declared: OutputSpec | None, label: str) -> Any:
    """*value* validated against *declared* and wrapped at the kind it names.

    This is the return step of one point of a call that a route realized
    (V.10): a type hole is completed from *value*, a batch declaration reads
    *value*'s leading axes as its levels, and an undeclared result wraps by
    the kind-directed table.

    Parameters
    ----------
    value : Any
        The result a route returned for the point.
    declared : OutputSpec or None
        The point's result declaration, or ``None`` when the point declares none.
    label : str
        The call's result label, which names each term built from *value*. A
        tracked *value* keeps its own label unless a batch declaration builds a
        batch from it.

    Returns
    -------
    TrackedTerm
        The result term, which under a batch declaration is a batch on its levels.

    Raises
    ------
    ResultSchemaError
        If *value* does not satisfy the declaration.
    """
    try:
        if declared is not None and isinstance(declared.spec, BatchSpec):
            return _batch_at(value, declared.spec, label)
        if declared is None:
            return _wrap_as_term(value, label)
        completed = _validate_function_output(
            function_name=label, output_spec=declared, result=value, bindings={}
        )
    except ResultSchemaError:
        raise
    except ValueError as error:
        raise ResultSchemaError(str(error)) from error
    return _wrap_declared_function_output(
        value, function_name=label, output_spec=cast("OutputSpec", completed)
    )


def _batch_at(value: Any, spec: BatchSpec, label: str) -> Any:
    """*value*, whose leading axes range over *spec*'s levels, as the batch *spec* declares.

    A batch is kept as it is. Record columns and stacked arrays become the
    batch form of their element kind, whose element declaration is read from
    the value and unified with the declared one, and an object array becomes
    the batch form the kind table records for the declared element.

    Parameters
    ----------
    value : Any
        The route's result, such as a batch or a ``Record`` of stacked columns.
    spec : BatchSpec
        The batch declaration, whose levels and element declaration a batch
        built from *value* takes.
    label : str
        The label of a batch built from *value*, which the messages also name.

    Returns
    -------
    Batch
        A batch whose leading axes are *spec*'s batch shape.

    Raises
    ------
    ValueError
        If the leading axes are not the declared batch shape, the element does
        not unify with the declared element, or *value* has no batch form.
    """
    batch_shape = tuple(spec.batch_shape)
    ranks = _ranks_of(spec.axis_groups)
    levels = tuple(spec.level_names)
    n_axes = len(batch_shape)

    def require_leading(shape: tuple[Any, ...]) -> None:
        if tuple(shape[:n_axes]) != batch_shape:
            raise ValueError(
                f"{label}: the result's leading axes {tuple(shape[:n_axes])} are not the "
                f"declared batch shape {batch_shape}"
            )

    if isinstance(value, Batch):
        require_leading(tuple(value.batch_shape))
        return value
    if isinstance(value, Mapping) and not isinstance(value, Record):
        # A record-valued route returns the nested mapping of its stacked columns.
        value = Record(label, **value)
    if isinstance(value, Record):
        template = value.event_template
        columns = {path: value.raw(path) for path in template}
        for column in columns.values():
            require_leading(tuple(_event_shape_of(column)))
        element = _reshaped_template(template, lambda shape: shape[n_axes:])
        _unify_specs(spec.element_spec, element, {}, f"{label} element")
        return _batch_class_for(element)(
            label, columns, levels, element_spec=element, axes_per_level=ranks
        )
    if _is_object_array(value):
        require_leading(value.shape)
        batch_class = batch_class_for_spec(spec.element_spec)
        if batch_class is not None and batch_class is not NumericArrayBatch:
            return batch_class(
                label, value, levels, element_spec=spec.element_spec, axes_per_level=ranks
            )
        return _make_stack(
            list(value.reshape((prod(batch_shape), *value.shape[n_axes:]))),
            batch_shape=batch_shape,
            axis_groups=tuple(spec.axis_groups),
            level_names=levels,
            field_name=label,
            name=label,
        )
    if _is_numeric_leaf(value):
        shape = tuple(_event_shape_of(value))
        require_leading(shape)
        declared = spec.element_spec
        support = declared.support if isinstance(declared, NumericArraySpec) else None
        element = NumericArraySpec(shape[n_axes:], _numpy_dtype_of(value), support)
        _unify_specs(declared, element, {}, f"{label} element")
        return NumericArrayBatch(label, value, levels, element_spec=element, axes_per_level=ranks)
    raise ValueError(f"{label}: a {type(value).__name__} has no form as the declared batch")


def raw_form(term: Any) -> Any:
    """*term*'s representation, detached from the workflow, as its kind's ``raw()`` gives it.

    A value that is not a tracked term is already raw and is returned as it is.
    """
    return term.raw() if isinstance(term, TrackedTerm) else term
