"""The sampling lift and the empirical enumeration of a lifted call.

This private module executes a call whose lifted arguments are laws, after
call resolution, normalization, planning, and route selection have identified
the distribution regime. It draws or enumerates the lifted arguments, runs the
function once per evaluation under the dispatch mode, and returns the
pushforward as an :class:`~probpipe.EmpiricalDistribution` over the outputs,
whose event declaration is the function's completed output declaration (V.6,
V.10). With ``include_inputs`` the law is the joint empirical law of the lifted
inputs and the outputs.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from ..core._array_backend import _is_numeric_leaf
from ..core._batch import Batch, BatchSpec, _flat_batch_spec
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
from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericArraySpec
from ..core._specs import OutputSpec
from ..core.config import WorkflowKind, _checked_count, config
from ..core.named_tree import _unflatten_paths
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..custom_types import Array, PRNGKey
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._factored import _raw_record
from ..values._binding import FunctionInputRef, input_ref_value, replace_input_refs
from ..values._function_base import _NAMELESS_COMPONENT, _validate_stacked_output
from . import _execution, _plan, _recipe
from ._broker import _record_active_execution_contract
from ._call import ApplicabilityError
from ._context import _workflow_jax_runtime_guard
from ._execution_contract import (
    make_execution_contract,
    supports_execution_contract,
    transport_for_execution_mode,
    transport_for_workflow_kind,
)
from ._result import (
    DRAW_LEVEL,
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
        The function's results, as a list of ``count`` results or as the one
        stacked pytree the mapped dispatch returns.
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
    make_execution_config: Callable[
        [],
        _execution.WorkflowExecutionConfig,
    ],
    requested_dispatch: str,
    resolve_dispatch: Callable[..., str],
    require_jax_traceable: Callable[[dict[str, Any], list[FunctionInputRef]], None],
    function_name: str,
    output_label: str | None = None,
    output_component: str = _NAMELESS_COMPONENT,
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
    most ``max_evaluations`` evaluations in all, and is a stand-in for it. Under
    the ``"jax"`` dispatch every evaluation runs in one ``jax.vmap``, with the
    evaluations and weights of the row-wise dispatch modes.

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
    make_execution_config : callable
        Zero-argument callback returning row-wise execution settings for
        sequential, threaded, or Prefect dispatch.
    requested_dispatch : str
        User-requested dispatch strategy, used to preserve explicit
        ``dispatch="jax"`` error behavior.
    resolve_dispatch : callable
        Callback that maps the current values and broadcast arguments to the
        effective dispatch strategy.
    require_jax_traceable : callable
        Callback used only for explicit JAX dispatch to raise a clear tracing
        error before executing.
    function_name : str
        The function's label, which provenance metadata records.
    output_label : str or None
        The result's display alias; the function's label by default.
    output_component : str
        Default mathematical component, independent of display aliases.
    output_expression : Expression or None
        The result's expression, the function applied to draws of its
        inputs, which gives the result its label; *output_label* labels the
        result when it is ``None``.
    output_spec : OutputSpec or None
        The function's output declaration with the call's shared dimensions
        bound, which the outputs complete.
    workflow_kind : WorkflowKind
        Effective orchestration mode for this call. The value is recorded in
        provenance and passed to the JAX path so Prefect task/flow requests can
        fail clearly when Prefect is unavailable.
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
        Atoms on the output component level, one per evaluation. The event declaration is
        the completed output declaration: a declared output completed by the
        returned values, or else an exposed record for an undeclared record return.
        An undeclared whole-term result uses the callable's original name. Under *include_inputs* the
        event exposes one field per lifted parameter, holding its complete draw,
        followed by the output's components.

    Raises
    ------
    ApplicabilityError
        If *include_inputs* is set and a lifted parameter shares its name with a
        component of the output.
    ResultSchemaError
        If the outputs do not satisfy the output declaration or do not agree on
        one declaration.
    """
    broadcast_args = list(stochastic_plan.arg_refs)
    max_evaluations = stochastic_plan.max_evaluations
    _validate_max_evaluations(max_evaluations)

    jax_contract = make_execution_contract(
        evaluator="jax_vmap",
        transport=transport_for_workflow_kind(workflow_kind),
        stochastic_plan=stochastic_plan,
    )
    jax_supported = supports_execution_contract(
        jax_contract,
        stochastic_plan,
    )
    dispatch = resolve_dispatch(
        values,
        broadcast_args,
        jax_supported=jax_supported,
    )
    if dispatch == "jax":
        _record_active_execution_contract(jax_contract)
        if requested_dispatch == "jax":
            require_jax_traceable(values, broadcast_args)

    if stochastic_plan.evaluation_mode != "sampled":
        draws = _broadcast_enumerate(
            func=func,
            values=values,
            stochastic_plan=stochastic_plan,
            logical_unit=logical_unit,
            get_key=get_key,
            make_execution_config=make_execution_config,
            dispatch=dispatch,
            function_name=function_name,
            workflow_kind=workflow_kind,
        )
    elif dispatch == "jax":
        draws = _broadcast_jax(
            func=func,
            values=values,
            stochastic_plan=stochastic_plan,
            logical_unit=logical_unit,
            get_key=get_key,
            function_name=function_name,
            workflow_kind=workflow_kind,
        )
    else:
        draws = _broadcast_sample(
            func=func,
            values=values,
            stochastic_plan=stochastic_plan,
            logical_unit=logical_unit,
            get_key=get_key,
            make_execution_config=make_execution_config,
        )

    result = _lift_result(
        draws,
        values=values,
        broadcast_args=broadcast_args,
        output_label=output_label or function_name,
        output_spec=output_spec,
        output_component=output_component,
        include_inputs=include_inputs,
    )
    if output_expression is not None:
        result._store_expression(output_expression)
    provenance = _make_broadcast_provenance(
        values=values,
        broadcast_args=broadcast_args,
        dispatch=dispatch,
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
    output_component: str,
    include_inputs: bool,
) -> EmpiricalDistribution:
    """The empirical law of a lifted call's evaluations, under their weights.

    The atoms are the outputs on the output component level, and the event declaration
    is the completed output declaration. A declaration that exposes a returned
    law's components holds each law whole under *output_component* instead,
    since each atom is a law. Under *include_inputs* each atom joins
    the draw of every lifted argument, under its parameter's label, to the
    output's components.

    Parameters
    ----------
    draws : _LiftDraws
        The lifted call's drawn inputs, outputs, and weights.
    values : mapping of str to Any
        The call's resolved arguments, from which each lifted law's event
        declaration is read under *include_inputs*.
    broadcast_args : sequence of FunctionInputRef
        The references to the lifted arguments, in the order of their fields
        in a joint atom.
    output_label : str
        The result's display alias.
    output_spec : OutputSpec or None
        The function's output declaration with the call's shared dimensions
        bound, or ``None`` when the function declares no output.
    output_component : str
        Default mathematical component, independent of display aliases.
    include_inputs : bool
        Whether the law is the joint law of the lifted inputs and the outputs.

    Returns
    -------
    EmpiricalDistribution
        The law labeled *output_label*, with one atom per evaluation.

    Raises
    ------
    ApplicabilityError
        If a lifted parameter shares its name with an output component.
    ResultSchemaError
        If the outputs do not satisfy the output declaration.
    """
    atoms, declaration = _output_atoms(
        draws.outputs, draws.count, output_label, output_spec, output_component
    )
    if declaration._component_name is None and not declaration.exposes_record:
        # A law's event is a value, so the law of the evaluations holds each
        # returned law whole under the default component, as an undeclared
        # output places its result.
        declaration = OutputSpec(**{output_component: declaration.spec})
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
    output_component: str,
) -> tuple[Batch, OutputSpec]:
    """The outputs as a batch on the output component level, and the declaration they complete.

    A declared output completes to its declaration with the shared dimensions
    bound by the outputs and any type hole filled. An undeclared record exposes
    its fields; an undeclared whole-term output uses *output_component*.

    Parameters
    ----------
    outputs : Any
        The function's results, as a list of *count* results or as the one
        stacked pytree the mapped dispatch returns.
    count : int
        The number of evaluations.
    output_label : str
        The batch's display alias, independent of its output components.
    output_spec : OutputSpec or None
        The function's output declaration with the call's shared dimensions
        bound, or ``None`` when the function declares no output.
    output_component : str
        Default mathematical component, independent of display aliases.

    Returns
    -------
    atoms : Batch
        The outputs, with one atom per evaluation.
    declaration : OutputSpec
        The completed output declaration, which records the dtypes the atoms hold.

    Raises
    ------
    ResultSchemaError
        If the outputs do not satisfy the declaration or do not agree on one spec,
        or an evaluation returns a batch with a level named like the output
        component, which names the atom level.
    """
    atom_level = (
        output_spec._component_name if output_spec is not None else None
    ) or output_component
    if isinstance(outputs, NumericArray):
        # The mapped dispatch returns one array term led by the draw axis. A map
        # that returns its operand returns the stacked draws' own term, whose spec
        # covers every draw, so the element spec is read from one point.
        element = outputs.spec
        point = tuple(jnp.shape(outputs.value)[1:])
        if tuple(element.shape) != point:
            element = NumericArraySpec(point, element.dtype, element.support)
        atoms: Batch = _record_stored_dtypes(
            NumericArrayBatch(outputs.value, atom_level, element_spec=element, label=output_label)
        )
    else:
        rows = _rows_of(outputs, output_label)
        returned = _returned_levels(rows)
        if atom_level in returned:
            raise ResultSchemaError(
                f"the atoms of {output_label!r} lie on the level {atom_level!r} that the output "
                f"component names, but the evaluations return batches on the levels "
                f"{list(returned)}, which include it. Pass an explicit OutputSpec with a distinct "
                f"output component, such as "
                f"output_spec=OutputSpec({_free_component(dict.fromkeys(returned))}=None)."
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
        BatchSpec._from_groups(atoms.element_spec, atoms.axis_groups[1:], atoms.level_names[1:])
        if len(atoms.level_names) > 1
        else atoms.element_spec
    )
    if output_spec is None:
        return atoms, OutputSpec.default(actual, component=output_component)
    try:
        _validate_stacked_output(function_name=output_label, output_spec=output_spec, batch=atoms)
    except ValueError as error:
        raise ResultSchemaError(str(error)) from error
    declaration = output_spec.with_spec(actual)
    return atoms, declaration._with_spec(_with_dtypes_of(declaration.spec, actual))


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

    Row-wise dispatch returns a list of results, and the mapped dispatch one
    stacked array, record, or mapping, which is read as a record of columns,
    or a returned batch's stacked store, which the aggregator reads as it is.
    """
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
    """The batch of joint atoms: each lifted argument's draw, then the output's components.

    Each lifted parameter contributes one field, under its label, holding its
    complete draw at the kind its law's event declaration names, so a record
    draw stays nested even when it has one field. The output contributes the
    components *declaration* exposes.

    Parameters
    ----------
    atoms : Batch
        The outputs on the output component level, as :func:`_output_atoms` returns them.
    declaration : OutputSpec
        The completed output declaration, as :func:`_output_atoms` returns it.
    draws : _LiftDraws
        The lifted call's evaluations, whose ``inputs`` hold each lifted
        argument's draws.
    values : mapping of str to Any
        The call's resolved arguments, from which each lifted law's event
        declaration is read.
    broadcast_args : sequence of FunctionInputRef
        The references to the lifted arguments, in the order of their fields.
    output_label : str
        The batch's label. The output's components come from *declaration*.

    Returns
    -------
    RecordBatch
        The joint atoms on the output component level, labeled *output_label*.

    Raises
    ------
    ApplicabilityError
        If a lifted parameter's label is also an output component.
    """
    fields: dict[str, Any] = {}
    columns: dict[str, Any] = {}
    for ref in broadcast_args:
        event = input_ref_value(values, ref).event_spec.spec
        if isinstance(event, BatchSpec):
            event = _flat_batch_spec(event)
        fields[ref.label] = event
        inputs = draws.inputs[ref]
        if isinstance(event, BatchSpec):
            # A batch-valued field holds one batch per draw in its object column.
            inputs = _record_columns(inputs, ref.label, event, as_batch=True)
        columns.update(_draw_columns(ref.label, inputs))
    clash = sorted(set(fields) & set(declaration.components))
    if clash:
        if len(clash) == 1:
            named, held = f"parameter {clash[0]!r}", "a component with that name"
        else:
            named, held = f"parameters {clash}", "components with those names"
        raise ApplicabilityError(
            f"include_inputs=True cannot add {named} to the output of {output_label!r}, which "
            f"already has {held}; rename the {'parameter' if len(clash) == 1 else 'parameters'}, "
            f"or pass output_spec=OutputSpec({_free_component(fields)}=None) to hold the output "
            "under a name no parameter has"
        )
    if isinstance(declaration.spec, BatchSpec):
        component = cast(str, declaration._component_name)
        fields[component] = declaration.spec
        columns[component] = _from_iterable(
            [atoms[index] for index in range(draws.count)], kind="batch-valued output atoms"
        )
        element = RecordSpec(fields)
        return _batch_class_for(element)(
            columns, atoms.level_names[0], element_spec=element, label=output_label
        )
    stored = _batch_columns(atoms)
    if declaration.exposes_record:
        fields.update(declaration.spec.children)
        columns.update(stored)
    else:
        component = cast(str, declaration._component_name)
        fields[component] = declaration.spec
        columns.update(_prefixed(component, stored))
    element = RecordSpec(fields)
    return _batch_class_for(element)(
        columns, atoms.level_names[0], element_spec=element, label=output_label
    )


def _free_component(taken: Mapping[str, Any]) -> str:
    """A component name that no key of *taken* is: ``output``, extended by ``_`` until it is free."""
    name = "output"
    while name in taken:
        name += "_"
    return name


def _draw_columns(label: str, draws: Any) -> dict[str, Any]:
    """One lifted argument's draws as columns led by the draw axis, keyed under *label*."""
    if isinstance(draws, RecordBatch):
        return _prefixed(label, draws._raw_columns())
    if isinstance(draws, Record):
        return _prefixed(label, {path: draws.raw(path) for path in draws.event_template})
    if isinstance(draws, Mapping):
        return _prefixed(label, dict(_raw_record(draws)))
    if isinstance(draws, list):
        return {label: _from_iterable(draws, kind="the draws of a lifted argument")}
    return {label: draws}


def _batch_columns(atoms: Batch) -> dict[str, Any]:
    """The columns of a batch led by its axes: by leaf path for records, else one unnamed column."""
    if isinstance(atoms, RecordBatch):
        return dict(atoms._raw_columns())
    if isinstance(atoms, NumericArrayBatch):
        return {"": atoms.values}
    if isinstance(atoms, _ObjectBatch):
        return {"": atoms._store}
    raise TypeError(f"a lifted output of kind {type(atoms).__name__} has no columns")


def _prefixed(prefix: str, columns: Mapping[str, Any]) -> dict[str, Any]:
    """*columns* keyed under *prefix*, an unnamed column taking *prefix* itself."""
    return {f"{prefix}/{path}" if path else prefix: column for path, column in columns.items()}


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
    dispatch: str,
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
            "dispatch": dispatch,
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
    *,
    mapped: bool = False,
) -> dict[FunctionInputRef, Array]:
    """Claim and sample each planned source once in one logical unit.

    Every group uses the root and consumer evaluators captured during
    preflight, so aliases and record projections share one root draw. The
    draws of a batch-valued consumer are its raw values when *mapped*, for the
    mapped body to rebuild row by row, and otherwise take the form its
    parameter receives (see :func:`_record_columns`).
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
            binding.sample_root(key, sample_shape), binding.root.label, binding.root.event_spec.spec
        )
        for consumer, evaluate, event, as_batch in zip(
            group.consumers,
            binding.consumer_evaluators,
            binding.consumer_events,
            binding.consumer_batches,
            strict=True,
        ):
            sampled[consumer.arg_ref] = _record_columns(
                evaluate(root_sample),
                consumer.arg_ref.label,
                event,
                as_batch=as_batch and not mapped,
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
        return draws.raw()
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
        The declaration of one draw.

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


def _broadcast_jax(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    stochastic_plan: _plan.StochasticPlan,
    logical_unit: _plan.LogicalUnit,
    get_key: Callable[[_plan.PlannedRandomEvent], PRNGKey],
    function_name: str,
    workflow_kind: WorkflowKind,
) -> _LiftDraws:
    """Draw the lifted arguments and evaluate the function on every draw with one ``jax.vmap``."""
    sample_shape = stochastic_plan.sample_shape
    if sample_shape is None:  # pragma: no cover - planner/dispatch contract guard
        raise RuntimeError("sampled stochastic plan is missing sample_shape")
    broadcast_args = list(stochastic_plan.arg_refs)
    evaluate = _mapped_evaluator(
        func=func,
        values=values,
        broadcast_args=broadcast_args,
        batch_events=batch_events(stochastic_plan),
        function_name=function_name,
        workflow_kind=workflow_kind,
    )
    sampled = _sample_planned_source_groups(
        stochastic_plan,
        stochastic_plan.source_groups,
        sample_shape,
        logical_unit,
        get_key,
        mapped=True,
    )
    inputs = {ref: sampled[ref] for ref in broadcast_args}
    return _LiftDraws(
        inputs=inputs,
        outputs=evaluate(inputs),
        weights=None,
        count=stochastic_plan.n_evaluations,
    )


def _mapped_evaluator(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    broadcast_args: Sequence[FunctionInputRef],
    batch_events: Mapping[FunctionInputRef, BatchSpec],
    function_name: str,
    workflow_kind: WorkflowKind,
) -> Callable[[Mapping[FunctionInputRef, Any]], Any]:
    """The callable that evaluates *func* on every row of its inputs with one ``jax.vmap``.

    The callable takes each argument of *broadcast_args* along one leading axis,
    keyed by its reference, and returns the stacked results; *values* holds the
    other arguments. A Prefect *workflow_kind* runs the map as one task or flow,
    which is built here, so a caller builds the evaluator before it draws the
    lifted arguments and a route that cannot be built fails before any draw.

    Parameters
    ----------
    func : callable
        The function the map evaluates at each row.
    values : dict of str to Any
        The call's resolved arguments, whose lifted entries each row replaces.
    broadcast_args : sequence of FunctionInputRef
        The references to the lifted arguments.
    batch_events : mapping of FunctionInputRef to BatchSpec
        The event of each lifted argument whose parameter receives a
        batch-valued draw as a batch, as :func:`batch_events` gives them.
    function_name : str
        The function's label, from which a Prefect task or flow takes its name.
    workflow_kind : WorkflowKind
        The call's orchestration mode.

    Returns
    -------
    callable
        The evaluator, which runs one map per call.

    Raises
    ------
    RuntimeError
        If *workflow_kind* asks for a Prefect task or flow and Prefect is not
        installed.
    """
    if workflow_kind in (WorkflowKind.TASK, WorkflowKind.FLOW) and (task is None or flow is None):
        raise RuntimeError(
            "Prefect task or flow execution was requested, but Prefect is not installed. "
            "Install with: pip install probpipe[prefect]"
        )
    single_call = mapped_draw_body(
        func=func, values=values, broadcast_args=broadcast_args, batch_events=batch_events
    )
    batch: list[tuple[Any, ...]] = []

    def run_vmap():
        with _workflow_jax_runtime_guard():
            return jax.vmap(single_call)(batch[-1])

    if workflow_kind == WorkflowKind.TASK:
        run_vmap = task(name=f"{function_name}_vmap")(run_vmap)
    elif workflow_kind == WorkflowKind.FLOW:
        runner = config.prefect.resolve_task_runner()
        run_vmap = flow(
            name=f"{function_name}_vmap",
            **({"task_runner": runner} if runner is not None else {}),
        )(run_vmap)

    def evaluate(inputs: Mapping[FunctionInputRef, Any]) -> Any:
        batch.append(tuple(inputs[ref] for ref in broadcast_args))
        return run_vmap()

    return evaluate


def _broadcast_enumerate(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    stochastic_plan: _plan.StochasticPlan,
    logical_unit: _plan.LogicalUnit,
    get_key: Callable[[_plan.PlannedRandomEvent], PRNGKey],
    make_execution_config: Callable[
        [],
        _execution.WorkflowExecutionConfig,
    ],
    dispatch: str,
    function_name: str,
    workflow_kind: WorkflowKind,
) -> _LiftDraws:
    """Evaluate the plan's exact combinations of atoms and its sampled repetitions.

    Each combination of the enumerated groups' atoms carries the product of
    their weights, shared equally among its repetitions. Under the ``"jax"``
    dispatch every evaluation runs in one ``jax.vmap``; under any other the
    evaluations run row by row under the execution settings.
    """
    if dispatch == "jax":
        mapped = _mapped_evaluator(
            func=func,
            values=values,
            broadcast_args=stochastic_plan.arg_refs,
            batch_events=batch_events(stochastic_plan),
            function_name=function_name,
            workflow_kind=workflow_kind,
        )
    else:
        mapped = None
        execution = make_execution_config()
        _execution._preflight_execution_config(execution)
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
    if sampled_groups:
        sample_shape = stochastic_plan.sample_shape
        if sample_shape is None:  # pragma: no cover - planner contract guard
            raise RuntimeError("mixed stochastic plan is missing sample_shape")
        sampled = _sample_planned_source_groups(
            stochastic_plan,
            sampled_groups,
            sample_shape,
            logical_unit,
            get_key,
            mapped=mapped is not None,
        )
    else:
        sampled = {}

    if mapped is not None:
        inputs = _enumerated_inputs(stochastic_plan, roots, atom_indices, sampled)
        return _LiftDraws(
            inputs=inputs, outputs=mapped(inputs), weights=weights, count=len(atom_indices)
        )

    exact_entries: list[tuple[_plan.StochasticSourceGroup, tuple[Any, ...]]] = []
    for group_index, root in zip(stochastic_plan.exact_group_order, roots, strict=True):
        binding = stochastic_plan.runtime_bindings[group_index]
        # Every atom along one leading axis, in its raw form.
        atoms = _record_columns(
            root._atoms_at(jnp.arange(root.num_atoms)), root.label, root.event_spec.spec
        )
        group = stochastic_plan.source_groups[group_index]
        exact_entries.append(
            (
                group,
                tuple(
                    _record_columns(
                        evaluate(atoms), consumer.arg_ref.label, event, as_batch=as_batch
                    )
                    for consumer, evaluate, event, as_batch in zip(
                        group.consumers,
                        binding.consumer_evaluators,
                        binding.consumer_events,
                        binding.consumer_batches,
                        strict=True,
                    )
                ),
            )
        )
    sample_arg_refs = [ref for group in sampled_groups for ref in group.arg_refs]

    call_value_list = []
    for sample_idx, combo in enumerate(atom_indices.tolist()):
        replacements: dict[FunctionInputRef, Any] = {}
        for (group, consumer_batches), i in zip(exact_entries, combo):
            for consumer, consumer_batch in zip(group.consumers, consumer_batches):
                replacements[consumer.arg_ref] = _index_sample(consumer_batch, i)
        for ref in sample_arg_refs:
            replacements[ref] = _index_sample(sampled[ref], sample_idx)
        call_value_list.append(replace_input_refs(values, replacements))

    request = _execution.WorkflowExecutionRequest(
        func=func,
        work_items=_execution.make_managed_work_items(
            call_value_list,
            unit_segments=tuple(
                _execution.lifted_evaluation_unit_segment(
                    logical_unit.logical_unit_id,
                    index,
                )
                for index in range(len(call_value_list))
            ),
        ),
        execution=execution,
        contract=make_execution_contract(
            evaluator="rowwise",
            transport=transport_for_execution_mode(execution.mode),
            stochastic_plan=stochastic_plan,
        ),
        stochastic_plan=stochastic_plan,
    )
    results = _execution.execute_many(request)

    all_input_samples = {
        ref: _stack_rows([input_ref_value(call_values, ref) for call_values in call_value_list])
        for ref in stochastic_plan.arg_refs
    }

    return _LiftDraws(
        inputs=all_input_samples,
        outputs=results,
        weights=weights,
        count=len(call_value_list),
    )


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
            root._atoms_at(jnp.asarray(atom_indices[:, column])), root.label, root.event_spec.spec
        )
        for consumer, evaluate, event in zip(
            group.consumers, binding.consumer_evaluators, binding.consumer_events, strict=True
        ):
            inputs[consumer.arg_ref] = _record_columns(
                evaluate(atoms), consumer.arg_ref.label, event
            )
    return {ref: inputs[ref] for ref in stochastic_plan.arg_refs}


def _broadcast_sample(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    stochastic_plan: _plan.StochasticPlan,
    logical_unit: _plan.LogicalUnit,
    get_key: Callable[[_plan.PlannedRandomEvent], PRNGKey],
    make_execution_config: Callable[
        [],
        _execution.WorkflowExecutionConfig,
    ],
) -> _LiftDraws:
    """Draw the lifted arguments and evaluate the function once per draw, row by row."""
    execution = make_execution_config()
    _execution._preflight_execution_config(execution)
    sample_shape = stochastic_plan.sample_shape
    if sample_shape is None:  # pragma: no cover - planner contract guard
        raise RuntimeError("sampled stochastic plan is missing sample_shape")
    broadcast_args = list(stochastic_plan.arg_refs)
    samples_per_arg = _sample_planned_source_groups(
        stochastic_plan,
        stochastic_plan.source_groups,
        sample_shape,
        logical_unit,
        get_key,
    )

    call_value_list = []
    for i in range(stochastic_plan.n_evaluations):
        replacements = {ref: _index_sample(samples_per_arg[ref], i) for ref in broadcast_args}
        call_value_list.append(replace_input_refs(values, replacements))

    request = _execution.WorkflowExecutionRequest(
        func=func,
        work_items=_execution.make_managed_work_items(
            call_value_list,
            unit_segments=tuple(
                _execution.lifted_evaluation_unit_segment(
                    logical_unit.logical_unit_id,
                    index,
                )
                for index in range(len(call_value_list))
            ),
        ),
        execution=execution,
        contract=make_execution_contract(
            evaluator="rowwise",
            transport=transport_for_execution_mode(execution.mode),
            stochastic_plan=stochastic_plan,
        ),
        stochastic_plan=stochastic_plan,
    )
    results = _execution.execute_many(request)

    return _LiftDraws(
        inputs={ref: samples_per_arg[ref] for ref in broadcast_args},
        outputs=results,
        weights=None,
        count=len(call_value_list),
    )


def _stack_rows(rows: list[Any]) -> Any:
    """Stack one argument's per-evaluation values along one leading axis.

    Record values stack through ``RecordBatch.stack``, on one level ``draw``
    over the rows, since a ``Record`` has fields rather than a shape. Nested
    mappings of columns stack column by column. Array values stack with
    ``jnp.stack``, and any other values form an object array.
    """
    if rows and isinstance(rows[0], Batch):
        return _from_iterable(rows, kind="batch-valued input atoms")
    if rows and isinstance(rows[0], Record):
        return RecordBatch.stack(rows, level_name=DRAW_LEVEL)
    if rows and isinstance(rows[0], Mapping) and not isinstance(rows[0], TrackedTerm):
        # The raw form of a batch of records: the nested mapping of its columns.
        return jax.tree.map(lambda *columns: _stack_rows(list(columns)), *rows)
    if all(_is_numeric_leaf(row) for row in rows):
        return jnp.stack([jnp.asarray(row) for row in rows])
    return _from_iterable(rows, kind="the draws of a lifted argument")


def _index_sample(s: Any, i: int) -> Any:
    """Row ``i`` of a per-argument sample batch, at the kind one draw is.

    A record draw stays a record whatever its number of fields, an array draw
    is an array, and the raw draw of a batch of records is the nested mapping
    of its columns.
    """
    if isinstance(s, RecordBatch):
        # The raw columns, so a field that is not an array reaches the body as
        # the value it holds rather than as a view of its column.
        return Record({p: s._raw_column(p)[i] for p in s.event_template}, label=s.label)
    if isinstance(s, Record):
        # Index each leaf field's batch row; rebuild by path key so a nested
        # sample is reconstructed with its structure intact.
        return Record({p: s.raw(p)[i] for p in s.event_template}, label=s.label)
    if isinstance(s, Mapping) and not isinstance(s, TrackedTerm):
        # The raw draws of a batch of records: the nested mapping of their columns.
        return _row_at(s, i)
    return s[i]


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


def mapped_draw_body(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    broadcast_args: Sequence[FunctionInputRef],
    batch_events: Mapping[FunctionInputRef, BatchSpec] | None = None,
) -> Callable[[Any], Any]:
    """The body ``jax.vmap`` runs for one draw, and the probe traces.

    Shared so the probe traces exactly what the executor runs, which two
    separately maintained functions cannot promise. Mapping a batch of records
    yields a record per draw, which the body receives as the row-wise paths
    present it.

    Each argument in *batch_events* arrives as the raw row of a batch-valued
    draw, and the body rebuilds it as the batch its event declares, labeled by
    the argument, as the row-wise paths build it. A function that returns a
    numeric batch returns it as a :class:`_MappedBatchStore` or a
    :class:`_MappedBatchColumns`, whose leaves the map stacks with the draw
    axis first, and the lift's result keeps the batch's levels after the atom
    level.

    Parameters
    ----------
    func : callable
        The function the map evaluates at each draw.
    values : dict of str to Any
        The call's resolved arguments, whose lifted entries each draw replaces.
    broadcast_args : sequence of FunctionInputRef
        The references to the lifted arguments, in the order of the slice the
        body receives.
    batch_events : mapping of FunctionInputRef to BatchSpec, optional
        The event of each lifted argument whose parameter receives a
        batch-valued draw as a batch, as :func:`batch_events` gives them.

    Returns
    -------
    callable
        The body of one draw, which takes the tuple of the lifted arguments'
        rows and raises :class:`UnmappedBatchError` if the function returns a
        record that holds a batch, whose rows the map cannot stack.
    """
    events = dict(batch_events or {})

    def one_draw(broadcast_slice):
        replacements = {
            ref: row if ref not in events else _batch_at(row, events[ref], ref.label)
            for ref, row in zip(broadcast_args, broadcast_slice, strict=True)
        }
        out = func(**replace_input_refs(values, replacements))
        if isinstance(out, RecordBatch):
            return _MappedBatchColumns.of(out)
        if isinstance(out, NumericArrayBatch):
            return _MappedBatchStore.of(out)
        if isinstance(out, Record):
            held = batch_field(out.spec)
            if held is not None:
                path, levels = held
                raise UnmappedBatchError(
                    f"dispatch='jax' cannot stack a returned record whose field {path!r} is a "
                    f"batch on the levels {list(levels)}. Return the batch alone, or use "
                    f"dispatch='auto', 'sequential', or 'thread'."
                )
        return out

    return one_draw


def batch_field(spec: Any, prefix: str = "") -> tuple[str, tuple[str, ...]] | None:
    """The path and levels of the first batch-valued field of a record *spec*, or None."""
    if not isinstance(spec, RecordSpec):
        return None
    for name, child in spec.children.items():
        path = f"{prefix}{name}"
        if isinstance(child, BatchSpec):
            return path, tuple(child.level_names)
        held = batch_field(child, f"{path}/")
        if held is not None:
            return held
    return None


class UnmappedBatchError(ValueError):
    """A batch that the mapped dispatch cannot carry through ``jax.vmap``.

    ``dispatch='jax'`` raises its message as it stands, in place of the generic
    tracing failure, and ``dispatch='auto'`` falls back to the row-wise dispatch.
    """
