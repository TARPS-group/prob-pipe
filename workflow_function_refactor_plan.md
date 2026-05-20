# WorkflowFunction Refactor Plan

## Summary

`WorkflowFunction` is the central execution facade for `ProbPipe` operations and
user-defined workflow functions. It should remain the public facade, but it
should no longer be the implementation hub for call resolution, distribution
normalization, broadcast planning, sampling, sweep execution, JAX vectorization,
execution dispatch, output coercion, and provenance assembly. In a word, this class is turning into a big ball of mud.

This plan proposes a staged internal refactor that keeps the public API stable
while moving distinct responsibilities into private core modules. The immediate
goal is maintainability and testability. The longer-term goal is to expose a
backend-neutral internal execution interface: execution adapters should receive
plain functions, plain call dictionaries, and resolved execution configuration
rather than a `WorkflowFunction` instance.

To be clarified, the refactor should not change current `Ray` behavior. Ray support remains the
existing short-term "Ray via Prefect" path until a separate native Ray design is
approved.

## Status Quo of `WorkflowFunction`

`WorkflowFunction` currently does much more than wrap and call a user function.
To put it more intuitively, it has exceeded 1,200 lines. At construction time, it stores the wrapped function, signature, type hints,
workflow kind, display name, module binding, broadcast defaults, vectorization
mode, threading configuration, PRNG key, include-inputs default, and cached
JAX-vectorization decision. It also precomputes parameter names, detects
`**kwargs`, and rejects user function parameters that collide with reserved
call-time override names such as `n_broadcast_samples`, `seed`, and
`include_inputs`.

At call time, `__call__` is effectively a mini compiler and executor:

1. It binds positional arguments into keyword arguments using the wrapped
   signature.
2. It unpacks nested `**kwargs` when the wrapped function accepts arbitrary
   keyword arguments.
3. It extracts call-time overrides for broadcast sample count, seed, and
   `include_inputs`.
4. It resolves missing inputs from call-time values, construction-time binds,
   attached `Module` child nodes or module inputs, and function defaults.
5. It validates dependency-typed parameters and missing required inputs.
6. It converts distribution-valued arguments according to concrete distribution
   hints or distribution protocol hints.
7. It classifies inputs into scalar-distribution broadcast arguments and
   array-valued sweep arguments.
8. It either executes one non-broadcast call or routes into broadcast execution.
9. It coerces the raw result back into the public
   `Record | RecordArray | Distribution` contract and attaches provenance.

The broadcast path currently has three regimes:

- distribution-only Monte Carlo or empirical broadcast;
- pure array sweep over `RecordArray` and non-scalar `DistributionArray` inputs;
- nested sweep, where each array sweep cell runs an inner distribution broadcast.

The distribution-only path also chooses among empirical enumeration, JAX `vmap`,
and loop-based sampling. Loop paths eventually build a list of per-call keyword
argument dictionaries and call `_execute_many`. The JAX path remains local and
vectorized.

The sweep path groups array-valued arguments by parent identity so sibling views
from the same parent array zip together, while values from different parents
combine by product rule. It then slices row-major sweep cells, optionally uses a
limited JAX `vmap` path for a simple single-`RecordArray` case, or executes loop
rows through `_execute_many`.

Execution dispatch is also inside `WorkflowFunction`. `_execute_many` selects
among sequential execution, `ThreadPoolExecutor`, Prefect task mapping, and a
Prefect flow wrapper. The Prefect helpers currently close over the
`WorkflowFunction` instance in some paths, which is acceptable for the current
implementation but is not the right boundary for future distributed backends.

Finally, `node.py` also contains adjacent public workflow primitives:
`Node`, `workflow_function`, `workflow_method`, `abstract_workflow_method`,
`Module`, and `AbstractModule`, plus DAG rendering and abstract workflow
signature validation. These are related public workflow constructs, but they are
not part of the inner execution engine of `WorkflowFunction`.

The problem is not that any one feature is wrong. The problem is that one class
currently owns all of these concerns, so even small behavioral changes require a
maintainer to reason about function signatures, module dependency resolution,
distribution conversion, broadcast shape algebra, random sampling, JAX tracing,
Prefect orchestration, threading, output wrapping, and provenance at the same
time.

## Motivation

`probpipe/core/node.py` is over the repository's rough module-size threshold and
contains several independent concerns in one file:

- public workflow primitives: `Node`, `WorkflowFunction`, `Module`, decorators;
- call input resolution from positional args, binds, defaults, and `Module`
  dependencies;
- call-time override extraction for broadcast sample count, seed, and
  `include_inputs`;
- distribution argument normalization through the converter registry;
- broadcast classification for scalar `Distribution`, `RecordArray`, and
  `DistributionArray` inputs;
- distribution-only Monte Carlo broadcast, empirical enumeration, and JAX
  `vmap`;
- array sweep execution over `RecordArray` and `DistributionArray` product
  shapes;
- sampling helpers, sibling-view reconnection, and PRNG-key management;
- execution dispatch through sequential Python, `ThreadPoolExecutor`, Prefect
  task mapping, and Prefect flows;
- output coercion and provenance attachment;
- module-level workflow construction, DAG rendering, and abstract workflow
  validation.

The current shape makes small changes high-risk because unrelated concerns are
interleaved. Recent work on Prefect-Ray also highlighted that execution dispatch
is the natural place for scheduler integration, but today that logic is embedded
inside `WorkflowFunction` methods and can capture more state than a distributed
scheduler should need.

The refactor should preserve the core ProbPipe design:

- `WorkflowFunction` remains the public facade for operations and user workflow
  functions.
- ProbPipe outputs still cross the workflow boundary as
  `Record | RecordArray | Distribution`.
- Distribution and record semantics remain unchanged.
- Prefect orchestration remains opt-in.
- Current Ray support remains the existing Prefect-Ray path.

## Design Principles

1. Preserve public API compatibility.

   Existing imports such as `from probpipe import WorkflowFunction` and
   `from probpipe.core.node import WorkflowFunction` must continue to work.
   Public decorators and `Module` behavior must remain stable.

2. Move implementation detail into private modules only.

   New modules should live under `probpipe/core/_workflow_*.py`. They are
   implementation details and should not be documented as public import paths.

3. Prefer deep modules over shallow forwarding.

   Each extracted module must hide a meaningful implementation behind a small
   interface. A module that only forwards one method call back into
   `WorkflowFunction` should not be created.

4. Separate call resolution, planning, execution, and result assembly.

   `WorkflowFunction.__call__` should become a coordinator. It should resolve a
   call, normalize values, build a plan, execute the plan, and assemble the
   result. Each step should have a direct test surface.

5. Execution adapters should not know probabilistic semantics.

   Execution adapters should receive a callable, ordered call dictionaries, and
   resolved execution configuration. They should not know about
   `Distribution`, `RecordArray`, sampling, broadcast weights, or provenance.

6. Keep dependency direction from getting worse.

   The current `node.py` path already depends on the converter registry. This
   plan should either keep that dependency in `node.py` temporarily or isolate it
   in one normalization module rather than spreading it across planning,
   execution, or result modules. This refactor should not claim to solve the
   broader dependency-direction issue.

7. Keep scheduler-specific behavior out of this refactor.

   This refactor should make future execution backends easier to add later, but
   it should not add `WorkflowKind.RAY`, import `ray`, create a public
   `probpipe.ray` module, or change any orchestration defaults. Native Ray
   support remains a separate future design.

8. Preserve JAX semantics.

   JAX `vmap` is local vectorization. Future distributed backends should enter
   through loop-style execution adapters unless a separate design explicitly
   changes vectorized execution semantics.

## Target Internal Pipeline

The desired internal shape is:

```text
WorkflowFunction.__call__
  -> resolve call inputs and overrides
  -> normalize distribution-valued inputs
  -> build BroadcastPlan
  -> choose non-broadcast / sweep / distribution broadcast / nested path
  -> produce ordered logical call rows when loop execution is needed
  -> execute rows through an execution adapter
  -> assemble Record | RecordArray | Distribution result
  -> attach provenance
```

Future execution backends should enter only at the execution adapter step:

```text
ordered call rows
  -> backend-neutral execution adapter
  -> ordered raw results
  -> ProbPipe result assembly
```

Backend-specific features such as object-store placement, resource hints,
actors, or job submission require separate designs. They are not part of this
behavior-preserving refactor.

## Proposed Module Split

### 1. Call resolution module

Add `probpipe/core/_workflow_call.py`.

Move or introduce:

- signature metadata caching;
- positional-argument binding;
- `**kwargs` unpacking;
- reserved call-time override extraction;
- input resolution from call-time inputs, construction-time binds, attached
  `Module` child nodes or inputs, and function defaults;
- required-parameter validation;
- dependency-typed parameter validation.

Proposed internal data shapes:

```python
@dataclass(frozen=True)
class WorkflowSignatureInfo:
    signature: inspect.Signature
    hints: Mapping[str, Any]
    param_names: tuple[str, ...]
    has_var_keyword: bool
    reserved_names: frozenset[str]


@dataclass(frozen=True)
class WorkflowCallOverrides:
    n_broadcast_samples: int
    include_inputs: bool
    seed: int | None


@dataclass(frozen=True)
class ResolvedWorkflowCall:
    values: dict[str, Any]
    overrides: WorkflowCallOverrides
```

Reasoning:

- Input resolution is neither broadcast planning nor execution.
- A future execution adapter should not depend on `WorkflowFunction._sig`,
  `WorkflowFunction._bind`, or `WorkflowFunction._module`.
- This module gives direct tests for function-call behavior that is currently
  hidden inside `__call__`.

### 2. Result contract module

Add `probpipe/core/_workflow_result.py`.

Move:

- `BroadcastMode`
- `BROADCAST_WRAP`
- `BROADCAST_STACK`
- `BROADCAST_NESTED`
- `_wrap_as_record`
- `_coerce_output`

Interface:

```python
def coerce_workflow_output(
    value: Any,
    *,
    broadcast_mode: BroadcastMode,
    provenance: Provenance | None,
    field_name: str,
) -> Any:
    ...
```

The exact function name may stay private (`_coerce_output`) during the first
implementation PR to minimize churn. `probpipe.core.node` can re-export the old
private helper temporarily for existing tests.

Reasoning:

- The output contract is a stable domain rule and is already mostly isolated.
- It is low-risk and provides a safe early refactor.
- Remote or distributed workers should return raw function results; result
  coercion should stay in the coordinator path after ordered results are
  collected.

### 3. Execution module

Add `probpipe/core/_workflow_execution.py`.

Move:

- `_execute_many`
- `_execute_many_threaded`
- `_map_task`
- `_execute_many_prefect_task`
- `_execute_many_prefect_flow`

Proposed internal data shapes:

```python
@dataclass(frozen=True)
class WorkflowExecutionConfig:
    mode: Literal[
        "sequential",
        "thread",
        "prefect_task",
        "prefect_flow",
    ]
    parallel: bool | int = False
    name: str = "workflow"
    prefect_task_runner: Any | None = None


@dataclass(frozen=True)
class WorkflowExecutionRequest:
    func: Callable[..., Any]
    call_value_list: list[dict[str, Any]]
    execution: WorkflowExecutionConfig
```

Recommended implementation style:

```python
class WorkflowExecutor(Protocol):
    def execute_many(self, request: WorkflowExecutionRequest) -> list[Any]:
        ...


class SequentialExecutor: ...
class ThreadExecutor: ...
class PrefectTaskExecutor: ...
class PrefectFlowExecutor: ...


def execute_many(request: WorkflowExecutionRequest) -> list[Any]:
    return resolve_executor(request.execution).execute_many(request)
```

The public internal entry point can remain function-like, but adapter classes
keep the module extensible.

Important constraints:

- The execution module must not accept or close over a `WorkflowFunction`
  instance.
- The execution module should not import JAX, `Distribution`, `Record`,
  `RecordArray`, or scheduler-specific packages such as Ray.
- The request should contain resolved execution mode, not raw
  `WorkflowKind.DEFAULT`.
- Prefect helper closures should capture only the wrapped function, call rows,
  names, and task runner metadata.

Reasoning:

- This is the highest-leverage split for backend-neutral execution.
- Sequential, thread, Prefect task, and Prefect flow dispatch are execution
  adapters over the same conceptual interface: ordered execution of many
  independent calls.
- Future execution adapters can be added behind the same interface without
  touching broadcast planning, sampling, output coercion, or provenance.
- Avoiding `self` capture improves distributed serialization behavior.

Non-goals for this step:

- Do not add a Ray adapter.
- Do not add new public execution configuration.
- Do not change `WorkflowKind`.
- Do not make Prefect or Ray auto-enable.

### 4. Distribution normalization module

Add `probpipe/core/_workflow_normalize.py`.

Move or introduce:

- scalar distribution conversion based on concrete `Distribution` hints;
- protocol-based distribution conversion;
- `DistributionArray` skip behavior for non-scalar arrays;
- 0-d `DistributionArray` unwrapping behavior;
- external distribution auto-conversion that currently happens inside broadcast
  argument classification.

Interface sketch:

```python
def normalize_workflow_values(
    *,
    values: dict[str, Any],
    hints: Mapping[str, Any],
) -> dict[str, Any]:
    ...
```

Reasoning:

- Current broadcast classification is partly impure because it can mutate values
  by converting external distributions to `NumericRecordDistribution`.
- Planning should be a pure classification step. Normalization should own value
  conversion.
- Centralizing `converter_registry` here preserves the existing dependency in
  one place and avoids spreading it across planning and execution modules. This
  does not attempt to resolve the broader `core -> converters` dependency
  direction.

### 5. Broadcast plan module

Add `probpipe/core/_workflow_plan.py`.

Move or introduce:

- broadcast argument classification currently handled by `_find_broadcast_args`;
- parent grouping for array-valued sweep arguments;
- sweep batch shape and total sweep size calculation;
- regime selection.

Proposed internal data shapes:

```python
@dataclass(frozen=True)
class ArrayBroadcastGroup:
    arg_names: tuple[str, ...]
    batch_shape: tuple[int, ...]
    size: int


@dataclass(frozen=True)
class BroadcastPlan:
    regime: Literal["none", "distribution", "sweep", "nested"]
    dist_args: tuple[str, ...]
    array_args: tuple[str, ...]
    array_groups: tuple[ArrayBroadcastGroup, ...]
    sweep_batch_shape: tuple[int, ...]
    n_sweep: int
    output_order: Literal["row_major"] = "row_major"
```

Reasoning:

- `WorkflowFunction` should first decide which regime applies, then execute that
  regime.
- An explicit plan makes product-rule sweep behavior easier to test directly.
- The planner should not mutate `values` and should not import the converter
  registry.

### 6. Sweep execution module

Add `probpipe/core/_workflow_sweep.py`.

Move:

- `_slice_ra_args`
- `_execute_sweep_rows`
- `_make_sweep_provenance`
- pure sweep execution from `_broadcast`
- nested outer sweep execution from `_broadcast`

Interface sketch:

```python
def execute_sweep(
    *,
    func: Callable[..., Any],
    values: dict[str, Any],
    plan: BroadcastPlan,
    execution: WorkflowExecutionConfig,
    vectorize: str,
    field_name: str,
) -> Any:
    ...
```

The exact interface should be refined during implementation. The important
constraint is that row execution uses the execution module rather than calling
the wrapped function directly, except for the existing JAX `vmap` path.

Reasoning:

- Sweep semantics are distinct from distribution marginalization semantics.
- `RecordArray` and `DistributionArray` product-shape logic should be tested as
  a coherent implementation.
- Future distributed backends may operate at this layer for parameter sweeps or
  bootstrap-like workloads, possibly with chunked row execution. Chunking itself
  is out of scope for this refactor.

### 7. Distribution broadcast module

Add `probpipe/core/_workflow_distribution_broadcast.py`.

Move:

- `_broadcast_distributions_only`
- `_sample_broadcast_args`
- `_broadcast_jax`
- `_broadcast_enumerate`
- `_broadcast_sample`
- `_index_sample`
- JAX vectorization resolution used by distribution-only broadcast
- `BroadcastDistribution` assembly for distribution-only paths

Reasoning:

- This is the largest and riskiest extraction, so it should happen after call
  resolution, result assembly, execution, normalization, and planning are
  stable.
- The module can own Monte Carlo marginalization, empirical enumeration, JAX
  vmap dispatch, sampling alignment, and `BroadcastDistribution` construction.
- Loop-based distribution broadcast should use the execution module for actual
  call dispatch. JAX `vmap` should remain a local vectorized execution path.

### 8. Optional later split: vectorization helpers

Consider adding `probpipe/core/_workflow_vectorize.py` only if vectorization
logic proves too large or too widely shared after the distribution-broadcast and
sweep modules are extracted.

Potential move:

- `_resolve_vectorize`;
- JAX traceability probing;
- cached auto-detection update protocol;
- small helpers for JAX-vmap eligibility checks.

Proposed internal data shape:

```python
@dataclass(frozen=True)
class VectorizationDecision:
    mode: Literal["jax", "loop"]
    reason: str | None = None
```

Reasoning:

- Vectorization is a planner decision, not an execution backend.
- JAX `vmap` should remain explicitly separate from loop execution adapters.
- This split is optional because extracting it too early may create a shallow
  module coupled to distribution broadcast and sweep details.

### 9. Optional later split: sampling helpers

Consider adding `probpipe/core/_workflow_sampling.py` only if sampling logic
needs an independent test surface after distribution-broadcast extraction.

Potential move:

- `_index_sample`;
- `_sample_broadcast_args`;
- sibling distribution-view reconnection logic;
- pure PRNG helper functions.

Potential data shape:

```python
@dataclass(frozen=True)
class SamplingConfig:
    base_key: PRNGKey
    n_samples: int
    include_inputs: bool
```

Reasoning:

- Sampling has its own correctness concerns: sibling-view correlation,
  empirical sample alignment, and deterministic PRNG handling.
- The current `_get_key` mutates `self._key`. This refactor may keep that outer
  state for compatibility, but sampling helpers should be made as pure as
  practical.
- A later distributed design may bind randomness to logical row indices rather
  than execution order, but that is out of scope for this refactor.

### 10. Optional later split: Module and DAG

Consider moving `Module` and `AbstractModule` into
`probpipe/core/_workflow_module.py` after the main `WorkflowFunction` refactor.

Reasoning:

- `Module` and DAG rendering are public workflow primitives but not directly
  part of function broadcast execution.
- Moving them too early would make the first refactor larger and riskier.
- If moved, `probpipe.core.node` must continue to re-export `Module` and
  `AbstractModule` to preserve compatibility.

## Staged Implementation Plan

### PR 0: Characterization tests before moving behavior

Add or strengthen behavior-preserving tests before refactoring implementation.

Suggested coverage:

- positional argument binding;
- duplicate positional/keyword error;
- `**kwargs` expansion;
- reserved overrides: `n_broadcast_samples`, `seed`, `include_inputs`;
- construction-time bind precedence;
- module child dependency and module input precedence;
- dependency override errors;
- missing required input errors;
- concrete distribution conversion;
- protocol-based distribution conversion;
- 0-d `DistributionArray` unwrap;
- size-1 `DistributionArray` remains a sweep;
- scalar `Distribution` broadcast;
- concrete `Distribution` slots do not broadcast;
- `RecordArray` sweep;
- `DistributionArray` sweep;
- sibling views from the same parent zip together;
- different parent arrays use product rule;
- nested array plus scalar-distribution broadcast;
- empirical enumeration weight exactness;
- sample alignment between input samples and output samples;
- `vectorize="loop"`, `vectorize="jax"`, and `vectorize="auto"` behavior;
- Prefect task-map result ordering;
- empty execution call list;
- thread execution result ordering.

Expected tests:

```bash
pytest tests/test_broadcasting.py tests/test_record_array_view.py tests/test_design.py -x -v
pytest tests/test_product.py tests/test_provenance.py tests/test_transition.py -x -v
pytest tests/test_prefect_config.py tests/test_prefect_orchestration.py -x -v
```

`tests/test_prefect_orchestration.py` may be skipped when Prefect is not
installed.

### PR 1: Result contract extraction

- Add `_workflow_result.py`.
- Move output coercion helpers.
- Keep compatibility imports from `probpipe.core.node` if tests currently import
  private helpers from there.

Expected tests:

```bash
pytest tests/test_broadcast_distribution.py tests/test_broadcasting.py -x -v
```

### PR 2: Execution extraction

- Add `_workflow_execution.py`.
- Move sequential, thread, Prefect task, and Prefect flow dispatch.
- Introduce `WorkflowExecutionConfig` and `WorkflowExecutionRequest`.
- Use internal private executor classes behind a small `execute_many(request)`
  entry point.
- Ensure execution helpers do not close over `WorkflowFunction`.
- Ensure `WorkflowExecutionRequest` contains only the wrapped function,
  call-value list, and resolved execution metadata.

Focused tests:

- sequential result ordering;
- `parallel=True`;
- `parallel=<positive int>`;
- `parallel=False`;
- `parallel=0`, negative int, and non-bool/non-int invalid values;
- Prefect `.map()` keyword expansion;
- empty call lists;
- Prefect missing fallback or error behavior, as currently specified.

Expected tests:

```bash
pytest tests/test_broadcasting.py tests/test_prefect_config.py -x -v
pytest tests/test_prefect_orchestration.py -x -v
```

### PR 3: Call resolution extraction

- Add `_workflow_call.py`.
- Move signature metadata, positional binding, `**kwargs` expansion, reserved
  override extraction, input resolution, and dependency validation.
- Keep converter-registry logic out of this PR.
- Preserve all existing call-time behavior.

Expected tests:

```bash
pytest tests/test_broadcasting.py tests/test_validation.py tests/test_transition.py -x -v
```

### PR 4: Distribution normalization and BroadcastPlan introduction

- Add `_workflow_normalize.py`.
- Add `_workflow_plan.py`.
- Move conversion behavior into normalization.
- Replace `_find_broadcast_args` with explicit `BroadcastPlan` construction.
- Keep behavior unchanged for scalar distribution broadcast, concrete
  distribution slots, protocol slots, `RecordArray` sweeps,
  `DistributionArray` sweeps, sibling-view zipping, and product-rule shape.

Expected tests:

```bash
pytest tests/test_broadcasting.py tests/test_record_array_view.py tests/test_design.py -x -v
pytest tests/test_product.py -x -v
```

### PR 5: Sweep execution extraction

- Add `_workflow_sweep.py`.
- Move pure sweep and nested sweep execution.
- Keep output shapes, provenance metadata, and row-major product ordering
  unchanged.
- Route loop row execution through `_workflow_execution.execute_many`.
- Keep the limited JAX-vmap sweep path behavior unchanged.

Expected tests:

```bash
pytest tests/test_record_array_view.py tests/test_design.py tests/test_transition.py -x -v
pytest tests/test_provenance.py -x -v
```

### PR 6: Distribution broadcast extraction

- Add `_workflow_distribution_broadcast.py`.
- Move distribution-only sampling, enumeration, JAX vmap, and
  `BroadcastDistribution` assembly.
- Keep sampling and vectorization helpers inside this module unless they prove
  broadly reusable or too large.
- Keep empirical enumeration exactness and sample alignment unchanged.
- Keep provenance metadata unchanged.
- Ensure loop-based call dispatch goes through `_workflow_execution.execute_many`.

Expected tests:

```bash
pytest tests/test_broadcasting.py tests/test_product.py tests/test_provenance.py -x -v
```

### PR 7: Optional sampling/vectorization extraction

- Add `_workflow_sampling.py` and/or `_workflow_vectorize.py` only if the
  distribution-broadcast extraction leaves a module that is still too broad.
- Keep this as a separate cleanup decision rather than a prerequisite for the
  main refactor.

Expected tests:

```bash
pytest tests/test_broadcasting.py tests/test_product.py -x -v
```

### PR 8: Optional Module/DAG extraction

- Move `Module` and `AbstractModule` only if the previous PRs leave `node.py`
  still too broad.
- Preserve all public imports from `probpipe` and `probpipe.core.node`.

Expected tests:

```bash
pytest tests/test_prefect_config.py tests/test_transition.py tests/test_validation.py -x -v
```

## Expected `WorkflowFunction.__call__` End State

A healthy end state is not an empty facade, but a readable coordinator:

```python
def __call__(self, *args, **kwargs):
    call = resolve_workflow_call(...)

    if call.overrides.seed is not None:
        self._key = jax.random.PRNGKey(call.overrides.seed)

    values = normalize_workflow_values(...)
    plan = build_broadcast_plan(...)
    execution = self._make_execution_config()

    if plan.regime == "none":
        result = execute_many(
            WorkflowExecutionRequest(
                func=self._func,
                call_value_list=[values],
                execution=execution,
            )
        )[0]
        provenance = make_non_broadcast_provenance(...)
        return coerce_workflow_output(...)

    return execute_broadcast_plan(
        func=self._func,
        values=values,
        plan=plan,
        execution=execution,
        sampling=...,
        vectorization=...,
        field_name=self._name,
    )
```

The goal is not to make `WorkflowFunction` trivial. The goal is to make it
obvious which phase is responsible for each concern.

## Future Execution Backend Vision

This refactor does not implement any new execution backend, but it prepares the
execution path for backend-neutral scheduling.

The future shape should be:

```text
WorkflowFunction call
  -> resolve inputs
  -> normalize distribution arguments
  -> build BroadcastPlan
  -> produce per-call kwargs or vectorized batch
  -> execute through an adapter
  -> coerce ordered results into ProbPipe outputs
```

Future adapters should enter at the adapter step, not in the domain logic. They
must preserve ordered raw results so ProbPipe can assemble provenance-tracked
outputs consistently. Backend-specific features such as resource hints, data
placement, persistent workers, job submission, or dataset-style execution need
separate designs.

A later native Ray design can use this execution interface, but this refactor
does not decide which Ray mechanisms should be used. That choice belongs in a
separate native Ray plan.

Short-term Prefect-Ray support remains unchanged. Users who want Ray today
should continue to use Prefect-Ray through `prefect_config.task_runner`.

## Non-Goals

- Do not change public imports or the public `WorkflowFunction` constructor.
- Do not change the `Record | RecordArray | Distribution` output contract.
- Do not add `WorkflowKind.RAY`.
- Do not add a public `probpipe.ray` module.
- Do not import `ray` from `core/`.
- Do not add new optional dependencies.
- Do not change Prefect off-by-default behavior.
- Do not change distribution conversion semantics.
- Do not change `RecordArray` or `DistributionArray` broadcasting semantics.
- Do not introduce backend-specific features such as resource hints, placement
  groups, actors, dataset execution, job submission, or object-store APIs in
  this refactor.
- Do not make Ray a dependency of core ProbPipe.

## Risks And Mitigations

### Risk: private helper imports in tests or downstream code

Some tests may import private helpers from `probpipe.core.node`. Implementation
PRs can temporarily re-export moved helpers from `node.py` and migrate tests
gradually. Private helper re-exports should not be treated as permanent public
API unless explicitly approved.

### Risk: behavior drift in call resolution

Function-call behavior has many edge cases: positional binding, duplicate
arguments, `**kwargs`, module dependency resolution, construction-time binds, and
reserved override extraction. Add characterization tests before moving this
logic.

### Risk: behavior drift in broadcast semantics

Broadcasting has many edge cases: empirical enumeration, sibling view
reconnection, array product shape, nested distribution plus array behavior, and
JAX vectorization. Each extraction PR should be behavior-preserving and should
run focused characterization tests before cleanup.

### Risk: normalization and planning get mixed again

Current broadcast classification can mutate values by converting external
distribution objects. The refactor should split normalization from planning so
`BroadcastPlan` construction is pure.

### Risk: dependency direction gets worse

The converter registry dependency should either stay in `node.py` temporarily or
be concentrated in `_workflow_normalize.py`. It should not be copied into
execution, planning, sampling, or result modules. This refactor should document
that it preserves the existing dependency-direction issue rather than resolving
it.

### Risk: execution adapter interface becomes too backend-specific

The execution module should first model the behavior that already exists:
ordered execution of many independent call dictionaries. Backend-specific
concerns such as object-store placement, resource hints, actors, job submission,
or dataset execution should be left for separate plans.

### Risk: RNG behavior becomes less reproducible

The current implementation mutates `self._key`. This may remain for public
behavior compatibility, but moved sampling helpers should be made deterministic
given an explicit key. A later distributed design should bind random keys to
logical row indices, not worker execution order.

### Risk: too many small modules reduce locality

Use the deletion test. If deleting a proposed module merely moves the same
complexity back into one caller without duplication, the module is probably too
shallow. Keep tightly coupled helpers together.

## Acceptance Criteria

The full refactor is complete when:

- `WorkflowFunction` remains the public facade and preserves all existing public
  imports.
- `node.py` no longer contains the implementation details for call resolution,
  distribution normalization, output coercion, execution dispatch, broadcast
  planning, sweep execution, sampling, vectorization, and distribution-only
  broadcast.
- Existing user-visible behavior is unchanged.
- Prefect orchestration remains opt-in.
- Short-term Ray via Prefect continues to work through `prefect_config`.
- No new external dependencies are added.
- Focused tests for broadcasting, Prefect orchestration, provenance,
  `RecordArray` views, designs, transition helpers, validation, and output
  coercion pass.
- New execution tests cover result ordering, empty call lists, thread dispatch,
  invalid `parallel` values, and Prefect `.map()` argument expansion.
- New call-resolution tests cover positional arguments, `**kwargs`, reserved
  overrides, bind/module/default precedence, and dependency validation.
- New planning tests cover broadcast regime selection, sibling-view grouping,
  product-rule sweep shape, and non-mutating plan construction.
- Execution request objects do not contain a `WorkflowFunction` instance.
- Loop-based broadcast and sweep execution go through the execution module.
- JAX-vmap paths remain local vectorized paths and do not go through the loop
  execution adapter.
- The implementation leaves a clear internal execution adapter point suitable
  for future execution backends, including a separately designed native Ray
  adapter.

## Review Questions

- Should private helper re-exports from `probpipe.core.node` be removed after
  one migration release, or kept indefinitely for test compatibility?
- Should `WorkflowExecutionConfig` remain strictly private throughout this
  refactor series?
- How much PRNG cleanup should be included in this refactor versus deferred to a
  separate reproducibility-focused change?
- Should `Module` and DAG rendering move in this refactor series, or remain in
  `node.py` until a separate cleanup?
- Should the converter registry dependency be fully isolated in
  `_workflow_normalize.py` now, or temporarily kept in `node.py` during the
  first planning PR?
