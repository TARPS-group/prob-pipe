"""The sampling lift and the empirical enumeration of a lifted call.

This private module executes a call whose lifted arguments are laws, after
call resolution, normalization, planning, and route selection have identified
the distribution regime. It draws or enumerates the lifted arguments, runs the
function at one evaluation point per draw on the executor the call chooses,
and returns the pushforward as an :class:`~probpipe.EmpiricalDistribution` over the outputs,
whose event spec is the function's completed output spec (V.6, V.10). With
``include_inputs`` the law is the joint empirical law of the lifted
inputs and the outputs.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..core._batch import Batch, BatchSpec, _flat_batch_spec
from ..core._evaluation_points import _requested_evaluation_points
from ..core._executors import _holds_tracer
from ..core._expression import Expression
from ..core._numeric_array import NumericArray
from ..core._numeric_array_batch import NumericArrayBatch, _MappedBatchStore
from ..core._object_batch import _from_iterable, _is_object_array, _ObjectBatch
from ..core._record_batch import (
    RecordBatch,
    _batch_class_for,
    _MappedBatchColumns,
    _stacked_batches,
)
from ..core._record_spec import RecordSpec, _layout_record_mapping
from ..core._spec_base import NumericArraySpec
from ..core._specs import OutputSpec, _has_record_outputs, _root_output_name
from ..core.config import WorkflowKind, _checked_count
from ..core.named_tree import _PATH_SEP, _unflatten_paths
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..custom_types import Array, PRNGKey
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._factored import _raw_record
from ..values._binding import FunctionInputRef, input_ref_value, replace_input_refs
from ..values._function_base import _NAMELESS_OUTPUT, _validate_stacked_output
from . import _plan, _recipe
from ._call import ApplicabilityError
from ._execution import (
    evaluate_requested_points,
    lifted_evaluation_point_segment,
    preflight_prefect_route,
    recorded_evaluations,
)
from ._result import (
    ResultSchemaError,
    _aggregate_output_spec,
    _batch_at,
    _make_stack,
    _output_record_spec,
    _record_stored_dtypes,
    _with_dtypes_of,
    raw_form,
)

#: The fewest evaluations a lift draws without a warning that its result may be unreliable.
MIN_EVALUATIONS = 5


@dataclass(frozen=True)
class _LiftDraws:
    """What one lifted call evaluated: the drawn inputs, the outputs, and their weights.

    Attributes
    ----------
    inputs : Mapping[FunctionInputRef, Any]
        Each lifted argument's values along one leading axis of ``count``
        evaluations, in their raw form: an array, a ``Record`` of columns, a
        batch of records, or a sequence of objects.
    outputs : Any
        The function's results, as a list of ``count`` results or as one
        pytree stacked on a leading axis of ``count``.
    weights : Array or None
        One weight per evaluation, or ``None`` for uniform weights.
    count : int
        The number of evaluations.
    """

    inputs: Mapping[FunctionInputRef, Any]
    outputs: Any
    weights: Array | None
    count: int


def execute_distribution_broadcast(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    stochastic_plan: _plan.StochasticPlan,
    logical_unit: _plan.LogicalUnit,
    include_inputs: bool,
    get_key: Callable[[_plan.PlannedRandomEvent], PRNGKey],
    lifted_point_traceability: Callable[
        [dict[str, Any], list[FunctionInputRef]], tuple[bool, str | None]
    ],
    function_name: str,
    output_label: str | None = None,
    output_name: str = _NAMELESS_OUTPUT,
    output_expression: Expression | None = None,
    output_spec: OutputSpec | None = None,
    workflow_kind: WorkflowKind,
    output_template: RecordSpec | None = None,
    provenance_parents: Sequence[TrackedTerm] = (),
    provenance_inputs: Mapping[str, Any] | None = None,
    record_recipe: bool = True,
    route: Mapping[str, Any] | None = None,
) -> EmpiricalDistribution:
    """Execute one lifted call whose lifted arguments are laws, returning its pushforward.

    The caller has already resolved the function's arguments, normalized the
    distribution-valued inputs, planned the lift, and selected its route. A plan
    whose every lifted group enumerates evaluates each combination of atoms once
    with the product of their weights, which is the exact pushforward; any other
    plan enumerates the groups the plan enumerates and samples the others, at
    most ``max_evaluations`` evaluations in all, and is a stand-in for it. The
    evaluations form one set of evaluation points, one per draw or
    combination, which runs on the executor the call chooses, so a function
    JAX can trace is mapped with one ``jax.vmap``.

    Parameters
    ----------
    func : callable
        Wrapped user function to execute for each sampled or enumerated call.
    values : dict[str, Any]
        Resolved workflow inputs. Entries the plan lifts are laws; all other
        entries are passed through to every evaluation.
    stochastic_plan : StochasticPlan
        Immutable source grouping, exact/sample classification, combination
        order, and sample-shape decisions for this lifted call.
    logical_unit : LogicalUnit
        Singleton or canonical sweep cell being executed.
    include_inputs : bool
        If ``True``, return the joint empirical law of the lifted inputs and the
        outputs. If ``False``, return the law of the outputs.
    get_key : callable
        Callback that claims the raw key for one planned source/unit event.
    lifted_point_traceability : callable
        Whether JAX can trace one evaluation point of the lift, given the
        values and the lifted arguments, with the reason when it cannot, as
        the function's claim or its stored probe result says.
    function_name : str
        The function's label, which provenance metadata records.
    output_label : str or None
        The result's display alias; the function's label by default.
    output_name : str
        The name of the default output, independent of display aliases.
    output_expression : Expression or None
        The result's expression, the function applied to draws of its
        inputs, which gives the result its label; *output_label* labels the
        result when it is ``None``.
    output_spec : OutputSpec or None
        The function's output spec with the call's shared dimensions
        bound, which the outputs complete.
    workflow_kind : WorkflowKind
        Effective orchestration mode for this call, which provenance records
        and which selects Prefect to run the evaluation points.
    output_template : RecordSpec or None
        Concrete authoritative template for declared record outputs, when present.
    provenance_parents : sequence of TrackedTerm
        Call-level tracked lineage, already ordered and deduplicated.
    provenance_inputs : mapping of str to Any or None
        Call-level resolved plain inputs. Per-row sampled values do not replace
        these original descriptors.
    record_recipe : bool
        Whether provenance records the call's replay recipe.
    route : mapping or None
        The selected route's name and exactness, recorded in provenance.

    Returns
    -------
    EmpiricalDistribution
        Atoms on the level the output names, one per evaluation. The event spec is
        the completed output spec: a declared output completed by the
        returned values, or else, for an undeclared record return, the
        positional form over the record.
        An undeclared whole-term result uses the callable's original name.
        Under *include_inputs* the event exposes one field per lifted
        parameter, holding its complete draw, followed by the function's
        outputs.

    Raises
    ------
    ApplicabilityError
        If *include_inputs* is set and a lifted parameter shares its name with
        one of the outputs.
    ResultSchemaError
        If the outputs do not satisfy the output spec or do not agree on
        one spec.
    """
    broadcast_args = list(stochastic_plan.arg_refs)
    max_evaluations = stochastic_plan.max_evaluations
    _validate_max_evaluations(max_evaluations)
    traceable, untraceable_reason = lifted_point_traceability(values, broadcast_args)
    preflight_prefect_route(workflow_kind, function_name, maps_with_jax=traceable)
    weights: Array | None
    if stochastic_plan.evaluation_mode != "sampled":
        inputs, weights = _enumerated_draws(stochastic_plan, logical_unit, get_key)
        count = len(weights)
    else:
        sampled = _sample_planned_source_groups(
            stochastic_plan,
            stochastic_plan.source_groups,
            _required_sample_shape(stochastic_plan),
            logical_unit,
            get_key,
        )
        inputs = {ref: sampled[ref] for ref in broadcast_args}
        weights = None
        count = stochastic_plan.n_evaluations
    record_draws = record_events(stochastic_plan)
    evaluation_points = _requested_evaluation_points(
        mapped_draw_body(
            func=func,
            values=values,
            broadcast_args=broadcast_args,
            batch_events=batch_events(stochastic_plan),
            record_events=record_draws,
        ),
        varying={
            "lifted_rows": tuple(
                lifted_row_form(inputs[ref], ref, record_draws) for ref in broadcast_args
            )
        },
        origin=function_name,
        traceable=traceable,
        untraceable_reason=untraceable_reason,
        point_noun="draw" if stochastic_plan.evaluation_mode == "sampled" else "combination",
        point_segment=partial(lifted_evaluation_point_segment, logical_unit.logical_unit_id),
    )
    outputs = evaluate_requested_points(
        evaluation_points,
        stacked=True,
        workflow_kind=workflow_kind,
        function_label=function_name,
        stochastic_plan=stochastic_plan,
    )
    draws = _LiftDraws(inputs=inputs, outputs=outputs, weights=weights, count=count)

    result = _lift_result(
        draws,
        values=values,
        broadcast_args=broadcast_args,
        output_label=output_label or function_name,
        output_spec=output_spec,
        output_name=output_name,
        include_inputs=include_inputs,
    )
    if output_expression is not None:
        result._store_expression(output_expression)
    provenance = _make_broadcast_provenance(
        values=values,
        broadcast_args=broadcast_args,
        evaluations=recorded_evaluations(
            origin=evaluation_points.origin, set_ordinal=evaluation_points._set_ordinal
        ),
        workflow_kind=workflow_kind,
        max_evaluations=max_evaluations,
        function_name=function_name,
        func=func,
        provenance_parents=provenance_parents,
        provenance_inputs=provenance_inputs,
        stochastic_plan=stochastic_plan if record_recipe else None,
        record_recipe=record_recipe,
        route=route,
    )
    return result.with_provenance(provenance)


def _lift_result(
    draws: _LiftDraws,
    *,
    values: Mapping[str, Any],
    broadcast_args: Sequence[FunctionInputRef],
    output_label: str,
    output_spec: OutputSpec | None,
    output_name: str,
    include_inputs: bool,
) -> EmpiricalDistribution:
    """The empirical law of a lifted call's evaluations, under their weights.

    The atoms are the outputs on the level the output names, and the event spec
    is the completed output spec. An output spec in the positional form
    over a returned law holds each law whole under *output_name* instead,
    since each atom is a law. Under *include_inputs* each atom joins
    the draw of every lifted argument, under its parameter's label, to the
    function's outputs.

    Parameters
    ----------
    draws : _LiftDraws
        The lifted call's drawn inputs, outputs, and weights.
    values : mapping of str to Any
        The call's resolved arguments, from which each lifted law's event
        spec is read under *include_inputs*.
    broadcast_args : sequence of FunctionInputRef
        The references to the lifted arguments, in the order of their fields
        in a joint atom.
    output_label : str
        The result's display alias.
    output_spec : OutputSpec or None
        The function's output spec with the call's shared dimensions
        bound, or ``None`` when the function declares no output.
    output_name : str
        The name of the default output, independent of display aliases.
    include_inputs : bool
        Whether the law is the joint law of the lifted inputs and the outputs.

    Returns
    -------
    EmpiricalDistribution
        The law labeled *output_label*, with one atom per evaluation.

    Raises
    ------
    ApplicabilityError
        If a lifted parameter shares its name with an output.
    ResultSchemaError
        If the outputs do not satisfy the output spec.
    """
    atoms, declaration = _output_atoms(
        draws.outputs, draws.count, output_label, output_spec, output_name
    )
    if _root_output_name(declaration) is None and not _has_record_outputs(declaration):
        # A law's event is a value, so the law of the evaluations holds each
        # returned law whole under the default output, as an undeclared
        # output places its result.
        declaration = OutputSpec(**{output_name: declaration.term_spec})
    if not include_inputs:
        return EmpiricalDistribution(
            atoms, draws.weights, label=output_label, event_spec=declaration
        )
    joint = _joint_atoms(atoms, declaration, draws, values, broadcast_args, output_label)
    return EmpiricalDistribution(joint, draws.weights, label=output_label)


def _output_atoms(
    outputs: Any,
    count: int,
    output_label: str,
    output_spec: OutputSpec | None,
    output_name: str,
) -> tuple[Batch, OutputSpec]:
    """The outputs as a batch on the level the output names, and the spec they complete.

    A declared output completes to its output spec with the shared dimensions
    bound by the outputs and any unspecified type filled in. An undeclared
    record exposes its fields as outputs; an undeclared whole-term output
    uses *output_name*.

    Parameters
    ----------
    outputs : Any
        The function's results, as a list of *count* results or as one
        pytree stacked on a leading axis of *count*.
    count : int
        The number of evaluations.
    output_label : str
        The batch's display alias, independent of its output names.
    output_spec : OutputSpec or None
        The function's output spec with the call's shared dimensions
        bound, or ``None`` when the function declares no output.
    output_name : str
        The name of the default output, independent of display aliases.

    Returns
    -------
    atoms : Batch
        The outputs, with one atom per evaluation.
    declaration : OutputSpec
        The completed output spec, which records the dtypes the atoms hold.

    Raises
    ------
    ResultSchemaError
        If the outputs do not satisfy the output spec or do not agree on one spec,
        or an evaluation returns a batch with a level of the output's name,
        which names the atom level.
    """
    atom_level = (
        _root_output_name(output_spec) if output_spec is not None else None
    ) or output_name
    if isinstance(outputs, NumericArray):
        # Stacked outputs can be one array term led by the draw axis. A map
        # that returns its operand returns the stacked draws' own term, whose spec
        # covers every draw, so the elements' spec is read from one point.
        element = outputs.spec
        point = tuple(jnp.shape(outputs.raw)[1:])
        if tuple(element.shape) != point:
            element = NumericArraySpec(point, element.dtype, element.support)
        atoms: Batch = _record_stored_dtypes(
            NumericArrayBatch(outputs.raw, atom_level, element=element, label=output_label)
        )
    else:
        rows = _rows_of(outputs, output_label)
        returned = _returned_levels(rows)
        if atom_level in returned:
            raise ResultSchemaError(
                f"the atoms of {output_label!r} lie on the level {atom_level!r} named by the "
                f"output, but the evaluations return batches on the levels "
                f"{list(returned)}, which include it. Pass an explicit OutputSpec with a distinct "
                f"output name, such as "
                f"output_spec=OutputSpec({_free_output(dict.fromkeys(returned))}=None)."
            )
        completed = None if output_spec is None else _aggregate_output_spec(output_spec, rows)
        try:
            atoms = _make_stack(
                rows,
                n=count,
                level_names=(atom_level,),
                label=output_label,
                output_spec=completed,
                output_template=None if completed is None else _output_record_spec(completed),
            )
        except (TypeError, ValueError) as error:
            raise ResultSchemaError(str(error)) from error
    actual = (
        BatchSpec._from_groups(
            atoms.element_spec.term_spec, atoms.axis_groups[1:], atoms.level_names[1:]
        )
        if len(atoms.level_names) > 1
        else atoms.element_spec.term_spec
    )
    if output_spec is None:
        return atoms, OutputSpec.default(actual, output_name=output_name)
    try:
        _validate_stacked_output(function_name=output_label, output_spec=output_spec, batch=atoms)
    except ValueError as error:
        raise ResultSchemaError(str(error)) from error
    declaration = output_spec.with_term_spec(actual)
    return atoms, declaration._with_term_spec(_with_dtypes_of(declaration.term_spec, actual))


def _returned_levels(rows: Any) -> tuple[str, ...]:
    """The levels of the batches that the evaluations in *rows* returned, in first-seen order."""
    entries = rows if isinstance(rows, list) else [rows]
    levels: dict[str, None] = {}
    for entry in entries:
        if isinstance(entry, (Batch, _MappedBatchStore, _MappedBatchColumns)):
            levels.update(dict.fromkeys(entry.level_names))
    return tuple(levels)


def _rows_of(outputs: Any, output_label: str) -> Any:
    """The outputs in a form the row aggregator reads.

    Outputs that do not stack arrive as an object array of one result per
    evaluation, which the aggregator reads as the list of the results.
    Stacked outputs arrive as one array, record, or mapping, which is read as
    a record of columns, or as a returned batch's stacked store, which the
    aggregator reads as it is.
    """
    if _is_object_array(outputs):
        return list(outputs)
    if isinstance(outputs, Mapping) and not isinstance(outputs, TrackedTerm):
        return Record(dict(outputs), label=output_label)
    return outputs


def _joint_atoms(
    atoms: Batch,
    declaration: OutputSpec,
    draws: _LiftDraws,
    values: Mapping[str, Any],
    broadcast_args: Sequence[FunctionInputRef],
    output_label: str,
) -> RecordBatch:
    """The batch of joint atoms: each lifted argument's draw, then the returned values' outputs.

    Each lifted parameter contributes one field, under its label, holding its
    complete draw at the kind its law's event spec names, so a record
    draw stays nested even when it has one field. The returned values
    contribute the outputs of *declaration*.

    Parameters
    ----------
    atoms : Batch
        The outputs on the level the output names, as :func:`_output_atoms`
        returns them.
    declaration : OutputSpec
        The completed output spec, as :func:`_output_atoms` returns it.
    draws : _LiftDraws
        The lifted call's evaluations, whose ``inputs`` hold each lifted
        argument's draws.
    values : mapping of str to Any
        The call's resolved arguments, from which each lifted law's event
        spec is read.
    broadcast_args : sequence of FunctionInputRef
        The references to the lifted arguments, in the order of their fields.
    output_label : str
        The batch's label. The function's outputs come from *declaration*.

    Returns
    -------
    RecordBatch
        The joint atoms on the level the output names, labeled *output_label*.

    Raises
    ------
    ApplicabilityError
        If a lifted parameter's label is also an output.
    """
    fields: dict[str, Any] = {}
    columns: dict[str, Any] = {}
    for ref in broadcast_args:
        event = input_ref_value(values, ref).event_spec.term_spec
        if isinstance(event, BatchSpec):
            event = _flat_batch_spec(event)
        fields[ref.label] = event
        inputs = draws.inputs[ref]
        if isinstance(event, BatchSpec):
            # A batch-valued field holds one batch per draw in its object column.
            inputs = _record_columns(inputs, ref.label, event, as_batch=True)
        columns.update(_draw_columns(ref.label, inputs))
    clash = sorted(set(fields) & set(declaration.outputs))
    if clash:
        if len(clash) == 1:
            named, held = f"parameter {clash[0]!r}", "an output with that name"
        else:
            named, held = f"parameters {clash}", "outputs with those names"
        raise ApplicabilityError(
            f"include_inputs=True cannot add {named} to the output of {output_label!r}, which "
            f"already has {held}; rename the {'parameter' if len(clash) == 1 else 'parameters'}, "
            f"or pass output_spec=OutputSpec({_free_output(fields)}=None) to hold the output "
            "under a name no parameter has"
        )
    if isinstance(declaration.term_spec, BatchSpec):
        output_name = cast(str, _root_output_name(declaration))
        fields[output_name] = declaration.term_spec
        columns[output_name] = _from_iterable(
            [atoms[index] for index in range(draws.count)], kind="batch-valued output atoms"
        )
        element = RecordSpec(fields)
        return _batch_class_for(element)(
            columns, atoms.level_names[0], element=element, label=output_label
        )
    stored = _batch_columns(atoms)
    if _has_record_outputs(declaration):
        fields.update(declaration.term_spec.children)
        columns.update(stored)
    else:
        output_name = cast(str, _root_output_name(declaration))
        fields[output_name] = declaration.term_spec
        columns.update(_prefixed(output_name, stored))
    element = RecordSpec(fields)
    return _batch_class_for(element)(
        columns, atoms.level_names[0], element=element, label=output_label
    )


def _free_output(taken: Mapping[str, Any]) -> str:
    """An output name that no key of *taken* is: ``output``, extended by ``_`` until it is free."""
    name = "output"
    while name in taken:
        name += "_"
    return name


def _draw_columns(label: str, draws: Any) -> dict[str, Any]:
    """One lifted argument's draws as columns led by the draw axis, keyed under *label*."""
    if isinstance(draws, RecordBatch):
        return _prefixed(label, draws._raw_columns())
    if isinstance(draws, Record):
        return _prefixed(label, {path: draws._raw_at(path) for path in draws.event_template})
    if isinstance(draws, Mapping):
        return _prefixed(label, dict(_raw_record(draws)))
    if isinstance(draws, list):
        return {label: _from_iterable(draws, kind="the draws of a lifted argument")}
    return {label: draws}


def _batch_columns(atoms: Batch) -> dict[str, Any]:
    """The columns of a batch led by its axes: by leaf path for records, else one unnamed column."""
    if isinstance(atoms, RecordBatch):
        return dict(atoms._raw_columns())
    if isinstance(atoms, NumericArrayBatch | _ObjectBatch):
        return {"": atoms.raw}
    raise TypeError(f"a lifted output of kind {type(atoms).__name__} has no columns")


def _prefixed(prefix: str, columns: Mapping[str, Any]) -> dict[str, Any]:
    """*columns* keyed under *prefix*, an unnamed column taking *prefix* itself."""
    return {
        f"{prefix}{_PATH_SEP}{path}" if path else prefix: column for path, column in columns.items()
    }


def _validate_max_evaluations(max_evaluations: int) -> None:
    """Refuse a count that is not a positive integer, and warn below ``MIN_EVALUATIONS``."""
    _checked_count(max_evaluations, "max_evaluations")
    if max_evaluations < MIN_EVALUATIONS:
        warnings.warn(
            f"max_evaluations={max_evaluations} is too low; "
            "results may be unreliable. "
            f"Recommended minimum is {MIN_EVALUATIONS}.",
            stacklevel=2,
        )


def _make_broadcast_provenance(
    *,
    values: dict[str, Any],
    broadcast_args: Sequence[FunctionInputRef],
    evaluations: list[dict[str, Any]],
    workflow_kind: WorkflowKind,
    max_evaluations: int,
    function_name: str,
    func: Callable[..., Any],
    provenance_parents: Sequence[TrackedTerm],
    provenance_inputs: Mapping[str, Any] | None,
    stochastic_plan: _plan.StochasticPlan | None,
    record_recipe: bool,
    route: Mapping[str, Any] | None = None,
) -> Provenance | None:
    controls, diagnostics = (
        _recipe.provenance_recipe_fields(stochastic_plan) if record_recipe else ({}, {})
    )
    return Provenance.create(
        "broadcast",
        parents=list(provenance_parents),
        metadata={
            "evaluations": evaluations,
            "orchestrate": workflow_kind.value,
            "max_evaluations": max_evaluations,
            "func": function_name or func.__name__,
            "broadcast_args": [ref.label for ref in broadcast_args],
            **(dict(route) if route is not None else {}),
        },
        inputs=provenance_inputs,
        controls=controls,
        diagnostics=diagnostics,
    )


def _sample_planned_source_groups(
    stochastic_plan: _plan.StochasticPlan,
    source_groups: Sequence[_plan.StochasticSourceGroup],
    sample_shape: tuple[int, ...],
    logical_unit: _plan.LogicalUnit,
    get_key: Callable[[_plan.PlannedRandomEvent], PRNGKey],
) -> dict[FunctionInputRef, Array]:
    """Claim and sample each planned source once in one logical unit.

    Every group uses the root and consumer evaluators captured during
    preflight, so aliases and record projections share one root draw. The
    draws of a batch-valued consumer are its raw values, which the body of one
    draw rebuilds as a batch (see :func:`mapped_draw_body`).
    """
    sampled: dict[FunctionInputRef, Array] = {}
    for group in source_groups:
        if group.execution_mode != "sampled":
            continue
        event = _plan.PlannedRandomEvent(
            stochastic_source_id=group.stochastic_source_id,
            logical_unit_id=logical_unit.logical_unit_id,
        )
        key = get_key(event)
        binding = stochastic_plan.runtime_bindings[group.index]
        root_sample = _record_columns(
            binding.sample_root(key, sample_shape),
            binding.root.label,
            binding.root.event_spec.term_spec,
        )
        for consumer, evaluate, event in zip(
            group.consumers, binding.consumer_evaluators, binding.consumer_events, strict=True
        ):
            sampled[consumer.arg_ref] = _record_columns(
                evaluate(root_sample), consumer.arg_ref.label, event
            )
    return sampled


def _record_columns(draws: Any, label: str, event: Any = None, *, as_batch: bool = False) -> Any:
    """*draws*, raw draws along a leading axis, with record draws held in a ``Record`` of columns.

    A record-valued law's raw draws are the nested mapping of its columns, and
    the lift reads each row of a record argument as a ``Record``. Any other
    draws are returned as they are.

    The draws of a batch-valued *event* arrive either as the batches' raw values
    or as one batch per draw, which a field of a ``Record`` of columns holds in
    its object column. They are returned as their raw values with the draw axis
    leading (see :func:`_stacked_draws`), and under *as_batch* as an object
    array of one batch per draw, labeled *label*, for a parameter whose
    annotation names a batch class.
    """
    if isinstance(event, BatchSpec):
        raw = _stacked_draws(draws, event)
        if not as_batch:
            return raw
        count = len(jax.tree.leaves(raw)[0])
        return _from_iterable(
            [_batch_at(_row_at(raw, index), event, label) for index in range(count)],
            kind="batch-valued input atoms",
        )
    if isinstance(draws, NumericArray):
        return draws.raw
    if isinstance(draws, Mapping) and not isinstance(draws, TrackedTerm):
        return Record(_raw_record(draws), label=label)
    return draws


def _stacked_draws(draws: Any, event: BatchSpec) -> Any:
    """The draws of a batch-valued *event* as their raw values, with the draw axis leading.

    Parameters
    ----------
    draws : Any
        The draws along one leading axis: their raw values, a ``Record`` or a
        mapping of raw columns, or an object array of one batch per draw,
        possibly held in a tracked term.
    event : BatchSpec
        The spec of one draw.

    Returns
    -------
    Any
        An array for array elements, the nested mapping of raw columns for
        record elements, and an object array for any other elements. Each leaf
        has the draw axis, then the event's batch axes, then the element's own
        axes.
    """
    if isinstance(draws, (Record, RecordBatch, Mapping)):
        return _raw_record(draws)
    raw = raw_form(draws)
    if _is_object_array(raw) and raw.ndim == 1 and all(isinstance(entry, Batch) for entry in raw):
        stacked = _stacked_batches(raw, event)
        return _unflatten_paths(stacked) if isinstance(stacked, Mapping) else stacked
    return raw


def _row_at(raw: Any, index: int) -> Any:
    """Row *index* of *raw*, an array or a nested mapping of columns led by the draw axis."""
    return jax.tree.map(lambda column: column[index], raw)


def _required_sample_shape(stochastic_plan: _plan.StochasticPlan) -> tuple[int, ...]:
    """The sample shape of a plan that samples a group."""
    sample_shape = stochastic_plan.sample_shape
    if sample_shape is None:  # pragma: no cover - planner contract guard
        raise RuntimeError("sampled stochastic plan is missing sample_shape")
    return sample_shape


def _enumerated_draws(
    stochastic_plan: _plan.StochasticPlan,
    logical_unit: _plan.LogicalUnit,
    get_key: Callable[[_plan.PlannedRandomEvent], PRNGKey],
) -> tuple[dict[FunctionInputRef, Any], Array]:
    """Each lifted argument's value at the plan's exact combinations of atoms and sampled repetitions.

    It returns each lifted argument's value at every evaluation, along one
    leading axis, and the weight of each evaluation. Each combination of the
    enumerated groups' atoms carries the product of their weights, shared
    equally among its repetitions.
    """
    roots = tuple(
        _enumerated_root(stochastic_plan, group_index)
        for group_index in stochastic_plan.exact_group_order
    )
    # The atom of each enumerated group at every evaluation, one column per group
    # in the plan's exact group order: each combination, once per repetition.
    atom_indices = np.repeat(
        np.asarray(stochastic_plan.exact_combination_order, dtype=np.int64).reshape(-1, len(roots)),
        stochastic_plan.repetitions_per_combination,
        axis=0,
    )
    weights = np.ones(len(atom_indices))
    for column, root in enumerate(roots):
        weights = weights * np.asarray(root.weights, dtype=np.float64)[atom_indices[:, column]]
    weights = jnp.array(weights / stochastic_plan.repetitions_per_combination)

    sampled_groups = tuple(
        group for group in stochastic_plan.source_groups if group.execution_mode == "sampled"
    )
    sampled = (
        _sample_planned_source_groups(
            stochastic_plan,
            sampled_groups,
            _required_sample_shape(stochastic_plan),
            logical_unit,
            get_key,
        )
        if sampled_groups
        else {}
    )
    return _enumerated_inputs(stochastic_plan, roots, atom_indices, sampled), weights


def _enumerated_root(
    stochastic_plan: _plan.StochasticPlan, group_index: int
) -> EmpiricalDistribution:
    """The empirical law at the root of the enumerated group *group_index*.

    Parameters
    ----------
    stochastic_plan : StochasticPlan
        The lift's plan, whose runtime bindings hold each group's root.
    group_index : int
        The group's index in the plan's ``source_groups``.

    Returns
    -------
    EmpiricalDistribution
        The root the plan captured for the group.

    Raises
    ------
    RuntimeError
        If the root is not an empirical law, or its atom count differs from the plan's.
    """
    group = stochastic_plan.source_groups[group_index]
    root = stochastic_plan.runtime_bindings[group_index].root
    if not isinstance(root, EmpiricalDistribution):  # pragma: no cover - plan contract guard
        raise RuntimeError("exact stochastic source is not an EmpiricalDistribution")
    if group.exact_size is None or root.num_atoms != group.exact_size:
        raise RuntimeError(
            "exact empirical size changed after planning: "
            f"planned {group.exact_size}, found {root.num_atoms}"
        )
    return root


def _enumerated_inputs(
    stochastic_plan: _plan.StochasticPlan,
    roots: Sequence[EmpiricalDistribution],
    atom_indices: np.ndarray,
    sampled: Mapping[FunctionInputRef, Any],
) -> dict[FunctionInputRef, Any]:
    """Each lifted argument's value at every evaluation, along one leading axis.

    An enumerated group reads the atoms of its root that its column of
    *atom_indices* names, in their raw form, and each consumer reads its
    argument from them as it reads one from a draw. A batch-valued argument
    keeps its raw values, which the mapped body rebuilds row by row. A sampled
    group's arguments are its draws in *sampled*.
    """
    inputs = dict(sampled)
    for column, (group_index, root) in enumerate(
        zip(stochastic_plan.exact_group_order, roots, strict=True)
    ):
        group = stochastic_plan.source_groups[group_index]
        binding = stochastic_plan.runtime_bindings[group_index]
        atoms = _record_columns(
            root._atoms_at(jnp.asarray(atom_indices[:, column])),
            root.label,
            root.event_spec.term_spec,
        )
        for consumer, evaluate, event in zip(
            group.consumers, binding.consumer_evaluators, binding.consumer_events, strict=True
        ):
            inputs[consumer.arg_ref] = _record_columns(
                evaluate(atoms), consumer.arg_ref.label, event
            )
    return {ref: inputs[ref] for ref in stochastic_plan.arg_refs}


def batch_events(stochastic_plan: _plan.StochasticPlan) -> dict[FunctionInputRef, BatchSpec]:
    """The event of each lifted argument whose parameter receives a batch-valued draw as a batch."""
    events: dict[FunctionInputRef, BatchSpec] = {}
    for group in stochastic_plan.source_groups:
        binding = stochastic_plan.runtime_bindings[group.index]
        for consumer, event, as_batch in zip(
            group.consumers, binding.consumer_events, binding.consumer_batches, strict=True
        ):
            if as_batch and isinstance(event, BatchSpec):
                events[consumer.arg_ref] = event
    return events


def record_events(stochastic_plan: _plan.StochasticPlan) -> dict[FunctionInputRef, RecordSpec]:
    """The event of each lifted argument whose draws are records."""
    events: dict[FunctionInputRef, RecordSpec] = {}
    for group in stochastic_plan.source_groups:
        binding = stochastic_plan.runtime_bindings[group.index]
        for consumer, event in zip(group.consumers, binding.consumer_events, strict=True):
            if isinstance(event, RecordSpec):
                events[consumer.arg_ref] = event
    return events


def lifted_row_form(
    draws: Any, ref: FunctionInputRef, record_draws: Mapping[FunctionInputRef, RecordSpec]
) -> Any:
    """A lifted argument's draws, or one row of them, in the form the body of one draw receives.

    The draws of a record argument are given as the nested mapping of their
    raw columns, which ``jax.vmap`` and the row of every other executor cut
    into the raw fields of one draw; the body rebuilds the record (see
    :func:`mapped_draw_body`). A record held whole across the map would keep
    the spec of every draw at once. Any other draws are returned as they are.
    """
    if ref not in record_draws:
        return draws
    if isinstance(draws, Record):
        return _unflatten_paths({path: draws._raw_at(path) for path in draws.event_template})
    return _raw_record(draws)


def mapped_draw_body(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    broadcast_args: Sequence[FunctionInputRef],
    batch_events: Mapping[FunctionInputRef, BatchSpec] | None = None,
    record_events: Mapping[FunctionInputRef, RecordSpec] | None = None,
) -> Callable[..., Any]:
    """The body of one evaluation point of a lift, which every executor runs and the probe traces.

    One body serves the lift under every executor, and the probe maps it, so
    the probe traces exactly what ``jax`` maps. The body receives one row of
    each lifted argument's draws, in the form :func:`lifted_row_form` gives:
    the raw fields of one draw of a law over records, which the body rebuilds
    as a record labeled by the argument, and the raw row of a batch-valued
    draw, which the body rebuilds as the batch its event declares, labeled by
    the argument. Under ``jax.vmap``, a function that returns a numeric batch
    returns it as a :class:`_MappedBatchStore` or a
    :class:`_MappedBatchColumns`, whose leaves stack with the draw axis first,
    and the lift's result keeps the batch's levels after the atom level; a
    draw evaluated alone returns its result as it is.

    Parameters
    ----------
    func : callable
        The function the lift evaluates at each draw.
    values : dict of str to Any
        The call's resolved arguments, whose lifted entries each draw replaces.
    broadcast_args : sequence of FunctionInputRef
        The references to the lifted arguments, in the order of the rows the
        body receives.
    batch_events : mapping of FunctionInputRef to BatchSpec, optional
        The event of each lifted argument whose parameter receives a
        batch-valued draw as a batch, as :func:`batch_events` gives them.
    record_events : mapping of FunctionInputRef to RecordSpec, optional
        The event of each lifted argument whose draws are records, as
        :func:`record_events` gives them.

    Returns
    -------
    callable
        ``one_draw(lifted_rows)``, the result at the draw whose rows of the
        lifted arguments are *lifted_rows*. Under ``jax.vmap`` it raises
        :class:`UnmappedBatchError` if the function returns a record that
        holds a batch, whose rows the map cannot stack.
    """
    events = dict(batch_events or {})
    record_draws = dict(record_events or {})

    def argument_at_draw(ref: FunctionInputRef, row: Any) -> Any:
        if ref in events:
            return _batch_at(row, events[ref], ref.label)
        if ref in record_draws:
            # JAX maps a dict with its keys sorted, so the draw is laid out in
            # the event's order again before it is a record.
            return Record(_layout_record_mapping(record_draws[ref], row), label=ref.label)
        return row

    def one_draw(lifted_rows: tuple[Any, ...]) -> Any:
        replacements = {
            ref: argument_at_draw(ref, row)
            for ref, row in zip(broadcast_args, lifted_rows, strict=True)
        }
        draw_output = func(**replace_input_refs(values, replacements))
        if isinstance(draw_output, Mapping):
            # JAX maps and stacks a dict with its keys sorted, so a returned
            # mapping becomes a record first, which keeps the function's order.
            draw_output = Record(dict(draw_output))
        if not _holds_tracer(lifted_rows):
            # One draw evaluated alone returns its result as it is, which the
            # aggregate stacks with the other draws' results.
            return draw_output
        if isinstance(draw_output, RecordBatch):
            return _MappedBatchColumns.of(draw_output)
        if isinstance(draw_output, NumericArrayBatch):
            return _MappedBatchStore.of(draw_output)
        if isinstance(draw_output, Record):
            held = batch_field(draw_output.spec)
            if held is not None:
                path, levels = held
                raise UnmappedBatchError(
                    f"jax.vmap cannot stack a returned record whose field {path!r} is a batch on "
                    f"the levels {list(levels)}. Return the batch alone, or claim "
                    "traceable=False so the function runs one draw at a time"
                )
        return draw_output

    return one_draw


def batch_field(spec: Any, prefix: str = "") -> tuple[str, tuple[str, ...]] | None:
    """The path and levels of the first batch-valued field of a record *spec*, or None."""
    if not isinstance(spec, RecordSpec):
        return None
    for name, child in spec.children.items():
        path = f"{prefix}{name}"
        if isinstance(child, BatchSpec):
            return path, tuple(child.level_names)
        held = batch_field(child, f"{path}{_PATH_SEP}")
        if held is not None:
            return held
    return None


class UnmappedBatchError(ValueError):
    """A batch that a lift cannot carry through ``jax.vmap``.

    The probe records it as the reason JAX cannot trace the lift, so the lift
    runs one draw at a time, and an executor ``jax`` named for the function
    gives its message.
    """
