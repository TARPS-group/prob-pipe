"""Wrap, label, and attach provenance to Function results at their own kind."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from ..core._broadcast_distributions import _make_stack
from ..core._numeric_record import _is_numeric_leaf
from ..core._record_batch import RecordBatch
from ..core._specs import NumericArraySpec, OutputSpec, RecordSpec
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import TrackedTerm

BroadcastMode = Literal["wrap", "stack", "nested"]
BROADCAST_WRAP: BroadcastMode = "wrap"
BROADCAST_STACK: BroadcastMode = "stack"
BROADCAST_NESTED: BroadcastMode = "nested"


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
    from ..core._opaque import Opaque
    from ..core._specs import OpaqueSpec

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
    """
    from ..core._batch import BatchSpec
    from ..core._numeric_array_batch import _MappedBatchStore
    from ..core._record_batch import _MappedBatchColumns
    from ..core._spec_base import _unify_specs

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
            actual = RecordSpec.infer_from({"result": row}).children["result"]
        if spec is None:
            spec = actual
        _unify_specs(spec, actual, bindings, "Function aggregate output")
    return output_spec._with_spec(None if spec is None else spec._substitute_dims(bindings))


def _output_record_spec(output_spec: OutputSpec) -> RecordSpec | None:
    """Adapt the result declaration to the legacy aggregate's record template."""
    from ..distributions._distribution import DistributionSpec

    spec = output_spec.spec
    if isinstance(spec, RecordSpec):
        return spec
    if isinstance(spec, DistributionSpec):
        return spec.event_spec
    return None


def _wrap_as_term(
    value: Any, field_name: str, output_spec: OutputSpec | None = None, *, name: str | None = None
) -> Any:
    """Wrap a raw host as its tracked kind under the caller's result label.

    Mappings become records, numeric values become NumericArray, callables
    become Function, and other values become Opaque. Existing tracked terms
    are retained here and copied by the public result boundary.

    Interim implementation detail: undeclared sequences use the existing batch
    assembler. An explicitly declared OpaqueSpec keeps a sequence atomic.
    """
    result_name = field_name if name is None else name
    if output_spec is not None and output_spec.spec is not None:
        return _wrap_declared_function_output(
            value, function_name=result_name, output_spec=output_spec
        )
    match value:
        case TrackedTerm():
            return value
        case Mapping():
            return Record(result_name, dict(value))
        case list() | tuple():
            if not value:
                from ..core._opaque_batch import OpaqueBatch

                return OpaqueBatch(result_name, [], field_name)
            return _make_stack(
                list(value),
                n=len(value),
                level_names=(field_name,),
                field_name=field_name,
                name=result_name,
            )
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

    ``field_name`` is the Function's output_name, separate from the function
    label in provenance and the declared output components. A tracked
    return is shallow-copied, sharing value data while owning its metadata.
    """
    raw_value = value
    if broadcast_mode == BROADCAST_WRAP:
        value = _wrap_as_term(value, field_name)
    if isinstance(value, TrackedTerm):
        if value is raw_value or value.provenance is not None:
            value = _copy_result_term(value)
        object.__setattr__(value, "_name", field_name)
        from ..values import Function

        if isinstance(value, Function):
            object.__setattr__(value, "__name__", field_name)
            object.__setattr__(value, "__qualname__", field_name)
        if provenance is not None:
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
    elif spec is not None:
        from ..core._batch import Batch, BatchSpec
        from ..values import Function, FunctionSpec

        if isinstance(spec, FunctionSpec) and isinstance(clone, Function):
            from ..values._function_base import _validate_function_declarations

            _validate_function_declarations(
                function_name=clone.name,
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
