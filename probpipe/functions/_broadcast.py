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
from ..core._batch import Batch
from ..core._expression import Expression
from ..core._numeric_array import NumericArray
from ..core._numeric_array_batch import NumericArrayBatch
from ..core._object_batch import _from_iterable, _ObjectBatch
from ..core._record_batch import RecordBatch, _batch_class_for
from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericArraySpec
from ..core._specs import OutputSpec
from ..core.config import WorkflowKind, prefect_config
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..custom_types import Array, PRNGKey
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._factored import _raw_record
from ..values._binding import FunctionInputRef, input_ref_value, replace_input_refs
from ..values._function_base import _validate_stacked_output
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
    _make_stack,
    _output_record_spec,
    _record_stored_dtypes,
    _with_dtypes_of,
)

MIN_BROADCAST_SAMPLES = 5


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
    output_component: str = "result",
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
    plan draws ``n_broadcast_samples`` joint draws per co-sampling group,
    enumerating the groups the plan enumerates, and is a stand-in for it. Under
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
        Atoms on the level ``draw``, one per evaluation. The event declaration is
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
    n_broadcast_samples = stochastic_plan.n_broadcast_samples
    _validate_n_broadcast_samples(n_broadcast_samples)

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
        n_broadcast_samples=n_broadcast_samples,
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

    The atoms are the outputs on the level ``draw``, and the event declaration
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
    """The outputs as a batch on the level ``draw``, and the declaration they complete.

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
        If the outputs do not satisfy the declaration or do not agree on one spec.
    NotImplementedError
        If each evaluation returned a batch, whose empirical law is not built yet.
    """
    if isinstance(outputs, NumericArray):
        # The mapped dispatch returns one array term led by the draw axis. A map
        # that returns its operand returns the stacked draws' own term, whose spec
        # covers every draw, so the element spec is read from one point.
        element = outputs.spec
        point = tuple(jnp.shape(outputs.value)[1:])
        if tuple(element.shape) != point:
            element = NumericArraySpec(point, element.dtype, element.support)
        atoms: Batch = _record_stored_dtypes(
            NumericArrayBatch(
                outputs.value,
                DRAW_LEVEL,
                element_spec=element,
                label=output_label,
            )
        )
    else:
        rows = _rows_of(outputs, output_label)
        completed = None if output_spec is None else _aggregate_output_spec(output_spec, rows)
        try:
            atoms = _make_stack(
                rows,
                n=count,
                level_names=(DRAW_LEVEL,),
                label=output_label,
                output_spec=completed,
                output_template=None if completed is None else _output_record_spec(completed),
            )
        except (TypeError, ValueError) as error:
            raise ResultSchemaError(str(error)) from error
    if len(atoms.level_names) > 1:
        raise NotImplementedError(
            f"{output_label!r} returned a Batch while broadcasting over a distribution "
            f"argument, which is not supported yet"
        )
    if output_spec is None:
        return atoms, OutputSpec.default(atoms.element_spec, component=output_component)
    try:
        _validate_stacked_output(function_name=output_label, output_spec=output_spec, batch=atoms)
    except ValueError as error:
        raise ResultSchemaError(str(error)) from error
    # The declaration records the dtypes the atoms hold, as the atoms' own does.
    declaration = output_spec.with_spec(atoms.element_spec)
    return atoms, declaration._with_spec(_with_dtypes_of(declaration.spec, atoms.element_spec))


def _rows_of(outputs: Any, output_label: str) -> Any:
    """The outputs in a form the row aggregator reads.

    Row-wise dispatch returns a list of results, and the mapped dispatch one
    stacked array, record, or mapping, which is read as a record of columns.
    """
    if isinstance(outputs, Mapping) and not isinstance(outputs, TrackedTerm):
        return Record(
            dict(outputs),
            label=output_label,
        )
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
        The outputs on the level ``draw``, as :func:`_output_atoms` returns them.
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
        The batch's label, and the component of a whole-term output whose
        declaration names none.

    Returns
    -------
    RecordBatch
        The joint atoms on the level ``draw``, labeled *output_label*.

    Raises
    ------
    ApplicabilityError
        If a lifted parameter's label is also an output component.
    """
    fields: dict[str, Any] = {}
    columns: dict[str, Any] = {}
    for ref in broadcast_args:
        law = input_ref_value(values, ref)
        fields[ref.label] = law.event_spec.spec
        columns.update(_draw_columns(ref.label, draws.inputs[ref]))
    clash = sorted(set(fields) & set(declaration.components))
    if clash:
        if len(clash) == 1:
            named, held = f"parameter {clash[0]!r}", "a component with that name"
        else:
            named, held = f"parameters {clash}", "components with those names"
        raise ApplicabilityError(
            f"include_inputs=True cannot add {named} to the output of {output_label!r}, which "
            f"already has {held}. Rename the parameters or the output components."
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
        columns,
        DRAW_LEVEL,
        element_spec=element,
        label=output_label,
    )


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


def _validate_n_broadcast_samples(n_broadcast_samples: int) -> None:
    if isinstance(n_broadcast_samples, bool) or not isinstance(n_broadcast_samples, int):
        raise TypeError(f"n_broadcast_samples must be an integer; got {n_broadcast_samples!r}")

    if n_broadcast_samples <= 0:
        raise ValueError(
            f"n_broadcast_samples must be a positive integer; got {n_broadcast_samples!r}"
        )

    if n_broadcast_samples < MIN_BROADCAST_SAMPLES:
        warnings.warn(
            f"n_broadcast_samples={n_broadcast_samples} is too low; "
            "results may be unreliable. "
            f"Recommended minimum is {MIN_BROADCAST_SAMPLES}.",
            stacklevel=2,
        )


def _make_broadcast_provenance(
    *,
    values: dict[str, Any],
    broadcast_args: Sequence[FunctionInputRef],
    dispatch: str,
    workflow_kind: WorkflowKind,
    n_broadcast_samples: int,
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
            "n_samples": n_broadcast_samples,
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
    preflight, so aliases and record projections share one root draw.
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
        root_sample = _record_columns(binding.sample_root(key, sample_shape), binding.root.label)
        for consumer, evaluate in zip(group.consumers, binding.consumer_evaluators):
            sampled[consumer.arg_ref] = _record_columns(
                evaluate(root_sample), consumer.arg_ref.label
            )
    return sampled


def _record_columns(draws: Any, label: str) -> Any:
    """*draws*, raw draws along a leading axis, with record draws held in a ``Record`` of columns.

    A record-valued law's raw draws are the nested mapping of its columns, and
    the lift reads each row of a record argument as a ``Record``. Any other
    draws are returned as they are.
    """
    if isinstance(draws, NumericArray):
        return draws.raw()
    if isinstance(draws, Mapping) and not isinstance(draws, TrackedTerm):
        return Record(
            _raw_record(draws),
            label=label,
        )
    return draws


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
        function_name=function_name,
        workflow_kind=workflow_kind,
    )
    sampled = _sample_planned_source_groups(
        stochastic_plan,
        stochastic_plan.source_groups,
        sample_shape,
        logical_unit,
        get_key,
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
    single_call = mapped_draw_body(func=func, values=values, broadcast_args=broadcast_args)
    batch: list[tuple[Any, ...]] = []

    def run_vmap():
        with _workflow_jax_runtime_guard():
            return jax.vmap(single_call)(batch[-1])

    if workflow_kind == WorkflowKind.TASK:
        run_vmap = task(name=f"{function_name}_vmap")(run_vmap)
    elif workflow_kind == WorkflowKind.FLOW:
        runner = prefect_config.resolve_task_runner()
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
        atoms = _record_columns(root._atoms_at(jnp.arange(root.num_atoms)), root.label)
        exact_entries.append(
            (
                stochastic_plan.source_groups[group_index],
                tuple(
                    _record_columns(evaluate(atoms), root.label)
                    for evaluate in binding.consumer_evaluators
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
    argument from them as it reads one from a draw. A sampled group's arguments
    are its draws in *sampled*.
    """
    inputs = dict(sampled)
    for column, (group_index, root) in enumerate(
        zip(stochastic_plan.exact_group_order, roots, strict=True)
    ):
        group = stochastic_plan.source_groups[group_index]
        binding = stochastic_plan.runtime_bindings[group_index]
        atoms = _record_columns(root._atoms_at(jnp.asarray(atom_indices[:, column])), root.label)
        for consumer, evaluate in zip(group.consumers, binding.consumer_evaluators, strict=True):
            inputs[consumer.arg_ref] = _record_columns(evaluate(atoms), consumer.arg_ref.label)
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
    over the rows, since a ``Record`` has fields rather than a shape. Array
    values stack with ``jnp.stack``, and any other values form an object array.
    """
    if rows and isinstance(rows[0], Record):
        return RecordBatch.stack(rows, level_name=DRAW_LEVEL)
    if all(_is_numeric_leaf(row) for row in rows):
        return jnp.stack([jnp.asarray(row) for row in rows])
    return _from_iterable(rows, kind="the draws of a lifted argument")


def _index_sample(s: Any, i: int) -> Any:
    """Row ``i`` of a per-argument sample batch, at the kind one draw is.

    A record draw stays a record whatever its number of fields, and an array
    draw is an array.
    """
    if isinstance(s, RecordBatch):
        # The raw columns, so a field that is not an array reaches the body as
        # the value it holds rather than as a view of its column.
        return Record(
            {p: s._raw_column(p)[i] for p in s.event_template},
            label=s.label,
        )
    if isinstance(s, Record):
        # Index each leaf field's batch row; rebuild by path key so a nested
        # sample is reconstructed with its structure intact.
        return Record(
            {p: s.raw(p)[i] for p in s.event_template},
            label=s.label,
        )
    return s[i]


def mapped_draw_body(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    broadcast_args: Sequence[FunctionInputRef],
) -> Callable[[Any], Any]:
    """The body ``jax.vmap`` runs for one draw, and the probe traces.

    Shared so the probe traces exactly what the executor runs, which two
    separately maintained functions cannot promise. Mapping a batch of records
    yields a record per draw, which the body receives as the row-wise paths
    present it.
    """

    def one_draw(broadcast_slice):
        replacements = dict(zip(broadcast_args, broadcast_slice, strict=True))
        return func(**replace_input_refs(values, replacements))

    return one_draw
