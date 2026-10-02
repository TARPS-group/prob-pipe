# Workflows and orchestration

`Function` wraps every [op](operations.md) and every user-written
`@function`. `Module` is the stateful container with
`@workflow_method` children.

`Function` is an immutable, tracked and annotated ProbPipe object. Its
`signature` is captured from the wrapped Python callable once at construction;
optional input and output declarations describe values, but do not replace or derive that
Python calling contract. Function calls record the Function itself as the first
provenance parent, followed by tracked inputs in parameter order. Every resolved
non-tracked parameter is recorded separately in `provenance.inputs`, including
defaults, construction bindings, and Module-provided values. Plain parameters
use their names; variadic slots use stable labels such as `*items[0]` and
`**extras['scale']`.

Prefect orchestration is **off by default**. Set
`prefect_config.workflow_kind = WorkflowKind.TASK` (or `FLOW`) globally, or
export `PROBPIPE_WORKFLOW_KIND=task` in the environment.

## Options namespace

Use bare `@function` when no ProbPipe controls are needed:

```python
@function
def score(x, seed):
    return x + seed
```

Use `@function(...)` for definition-time controls:

```python
@function(dispatch="jax", n_broadcast_samples=1_000)
def score(x, seed):
    return x + seed
```

Use `workflow.with_options(...)` to create a reusable Function copy with revised
controls. The original is unchanged; the copy keeps its controls on every call:

```python
configured = score.with_options(n_broadcast_samples=2_000)
result = configured(x, seed=7)
```

All engine controls are accepted: `workflow_kind`, `n_broadcast_samples`,
`dispatch`, `max_workers`, and `include_inputs`. A value of `None` keeps the
existing setting. Construction metadata (`name`, `output_name`, declarations,
and bindings) and domain arguments are not controls.

Keyword arguments in the final workflow call belong to the wrapped user
function whenever they can bind to that function. This keeps common names
such as `seed`, `name`, `dispatch`, `n_broadcast_samples`, and
`include_inputs` available for user APIs.

Randomness belongs to a workflow execution rather than to a `Function`. Use an
explicit run for reproducible lifted calls:

```python
from probpipe import workflow_run

with workflow_run(seed=42):
    result = score(dist, seed=7)
```

Legacy `Function(..., seed=...)` warns and ignores that argument;
`with_options(seed=...)` raises `TypeError`. A wrapped function's own `seed`
parameter remains an ordinary input.

## Workflow RNG scopes

`workflow_run` owns ProbPipe randomness for one structural execution:

```python
from probpipe import Normal, sample, workflow_run

dist = Normal("value", 0.0, 1.0)

with workflow_run(seed=42):
    first = sample(dist)
    second = sample(dist)
```

`first` and `second` use distinct occurrence paths. Repeating the complete
block with seed 42 reproduces both positions. Changing the seed changes every
workflow-owned event reached by the same structure.

With `seed=None`, a root scope reads eight bytes of OS entropy only when its
first automatic-random operation commits. A bare omitted-key call follows the
same rule in a private ephemeral scope, so two bare calls normally use distinct
roots. Empty scopes, deterministic work, caller-keyed work, wholly exact
enumeration, and preflight failures do not materialize a root or consume a
stochastic position.

Nested scopes are structural isolation boundaries. An unseeded nested scope
retains its parent's root; an integer seed replaces the root for that scope.
Both forms add a scope segment only if stochastic work commits, so edits inside
one nested scope do not renumber later outer events. Same-seed sibling scopes
remain distinct because their paths contain different scope segments.

### Automatic and caller-owned keys

Omitting a key delegates ownership to the active workflow run. Distribution
lifting, direct `sample`, Monte Carlo expectations, sampled conversions,
validation, and diagnostics use the same private broker. Each planned source
and logical unit receives one batched random event; the number of Monte Carlo
draws does not create per-draw events. Exact and otherwise non-consuming
branches request no key.

Passing an explicit key keeps ownership with the caller. ProbPipe passes the
key object to the existing provider unchanged, creates no workflow RNG recipe,
and does not shift the next automatic event. Explicit inference
`random_seed` values likewise remain algorithm inputs rather than workflow
roots. Arbitrary randomness inside user code or third-party services is
outside this contract.

When several transformations must share one exact realization, put them in one
joint `Function` invocation or reuse materialized samples. RNG recipe version 1
has no call-local common-random-number control.

### Co-sampling and execution routes

Within one lifted call, repeated references to the same distribution root,
record projections, and supported deterministic transformed descendants share
one planned root realization. Equal-parameter but distinct distribution
objects remain independent. Unsupported descendant graphs fail during
preflight instead of silently sampling independently.

Sequential, threaded, supported JAX, and Prefect execution consume the same
canonical source and logical-unit identities. Worker start order, completion
order, and retry attempt number do not enter key derivation; results are
restored to canonical plan order. This is a random-event and pairing contract,
not a promise of bit-identical floating-point output across execution routes.

A run is owned by the thread and asyncio task that entered it. ProbPipe-managed
work-item frames may participate in that run. A passively copied context that
enters from another thread or task raises
`UnmanagedConcurrentWorkflowEntryError` before preflight or randomness. A new
thread that receives no copied context instead performs an independent bare
call.

`dispatch="auto"` and `dispatch="jax"` require an observationally pure Python
body because route probing may trace it. A nested omitted-key ProbPipe effect
aborts the probe without consuming RNG or provenance state: automatic dispatch
falls back to row-wise execution, while explicit JAX dispatch raises `TypeError`.
A trace-compatible caller-keyed operation remains eligible. Use sequential or
thread dispatch when tracing the body would itself be inappropriate.

For artifact-driven reproduction and its compatibility limits, see
[Identity & provenance](provenance.md).

## Function declarations and result names

Construct a function with `Function(name, fn, *, input_spec=None,
output_spec=None, output_name=None, ...)`. The name is required and comes first.
The `@function` decorator defaults it to the Python callable's `__name__`.
`input_spec` accepts an `InputSpec` or a mapping from parameter names to term
specs. The captured Python signature remains the calling contract.

A function has three independent names: its own label, its result's
`output_name`, and the components in its output declaration. `output_name`
defaults to the initial function name. `with_name` changes only the function
label; results keep their output name, and provenance identifies the function
actually called. Function labels do not participate in `FunctionSpec` matching.
This naming rule applies to every operation implemented as a Function:

| Call | Result label |
| --- | --- |
| `sample(law)` | `sample` |
| `condition_on(model, ...)` | `condition_on` |
| `from_distribution(law, Target)` | `from_distribution` |
| `mean(law)` | `mean` |

These are object labels: component names, record fields, and existing batch
levels are retained. For example, conditioning a model named `params` produces
a law labeled `condition_on`, whose unobserved components keep their names.
Use `result.with_name("draws")` to label a result explicitly, or `apply` to
preserve the raw implementation's name and identity.

```python
import jax.numpy as jnp

from probpipe import Function, InputSpec, NumericArraySpec, OutputSpec, function


@function(
    input_spec=InputSpec(x=NumericArraySpec(("obs",)), scale=NumericArraySpec(())),
    output_spec=OutputSpec(value=NumericArraySpec(("obs",))),
    output_name="standardized",
)
def standardize(x, scale=1.0):
    return x / scale


values = jnp.array([1.0, 2.0])
raw = standardize.apply(values, scale=2.0)  # underlying array, unchanged
wrapped = standardize(values, scale=2.0)  # NumericArray named "standardized"
renamed = standardize.with_name("rescale")  # same declaration and output_name
identity = Function("identity", lambda x: x)
```

`output_spec` accepts an `OutputSpec` or a bare `TermSpec`:

- `OutputSpec(value=NumericArraySpec(()))` declares a whole array under the
  component `value`; the array is not wrapped in a single-field record.
- `OutputSpec(RecordSpec(x=(), y=()))` exposes the returned record's fields.
- `OutputSpec(bundle=RecordSpec(x=(), y=()))` exposes the whole record as one
  component. The returned term remains a record in either case.
- `OutputSpec(value=None)` leaves a type hole filled from each return value.
- A bare `RecordSpec` exposes its fields. Any other bare spec declares the
  whole term under `output_name`. With no declaration, the return kind is inferred.

`FunctionSpec` stores `InputSpec | None` and `OutputSpec | None`. Legacy
constructor keywords emit `FutureWarning`: `input_template`, `output_template`,
and `seed` are ignored; `func` overrides `fn`. Both `name` and `fn` remain
required constructor arguments. The old template properties are unavailable.
Each warning identifies the supplied option at the caller's location. This is
not full legacy compatibility: ignored templates install no validation, and
`Function(name="n", func=f)` still fails because `fn` is required.
Use `input_spec` and `output_spec` for declarations, `workflow_run(seed=...)`
for workflow randomness, and `bind={"seed": ...}` for a callable's seed parameter.
`with_options(seed=...)` remains an error.
Use `DistributionSpec` for a returned distribution and
`BatchSpec` for a returned batch, rather than its element schema alone.

Declarations are authoritative. Named input slots must match fixed signature
parameters; variadic signatures remain supported when the input side is
undeclared. Structures, kinds, shapes, same-kind dtypes, and declared output
supports are checked. Symbolic dimensions are bound per call, shared across
inputs and outputs, and never written back into the function declaration.
Output-only dimensions and type holes are resolved from the returned value.
Unset array dtype and support metadata are retained from the validated result,
including inside records and batches. A type hole uses the same return-kind
inference as an undeclared call; an explicit `OpaqueSpec` keeps a sequence atomic.
An existing distribution retains its own matching event declaration.
Broadcast marginals preserve named whole-record, Function, opaque, and batch
components. Numeric arrays still use the existing one-field record marginal,
and returned distributions still form mixtures; those result families are
unchanged by the Function migration.
For a returned Function, an unspecified side of its declared `FunctionSpec`
retains that Function's own input or output declaration.

Variadic Functions classify, lift, sample, or sweep each `*args` element and
`**kwargs` entry independently. Their annotations apply to each expanded slot;
`Any` supplies no pass-through guarantee. The original Python call is
reconstructed before execution. Provenance labels retain the slot, such as
`*items[0]` or `**extras['scale']`.

## Raw application and lifted calls

`apply` performs one raw evaluation with Python binding, defaults, construction
bindings, and declaration checks. It preserves the returned object's identity,
annotations, and provenance. It does not lift distributions, sweep batches,
wrap results, orchestrate execution, or create call provenance.
A returned callable's declared input slots and defaults are checked against its
signature without executing it; its future output is checked when the wrapped
Function is called.

`__call__` adds those operations. An existing tracked return is shallow-copied,
shares value data, receives `output_name`, and owns independent annotations and
current-call provenance. This also applies when provenance is disabled and
when the returned value is itself a Function. Arrays stay arrays; single-field
records stay records. Declared nested mappings are packed consistently before
sequential, threaded, or JAX aggregation. Sweeps use the result kind's batch
family, including a declared kind for an empty sweep.

Output-support checks require concrete data. Automatic dispatch falls back to
row-wise execution for support-bearing declarations; explicit JAX dispatch
reports the limitation. Direct `jax.jit(function.apply)` preserves JAX's tracer
error. Support-free declarations retain the existing JAX paths.

Broadcasts return marginal, joint, or distribution-array families carrying the
result label and output declaration. An empty
distribution-valued sweep remains unavailable because `DistributionArray`
requires at least one component.

## Wrappers and decorators

Import `Function` and the decorators from `probpipe`. `probpipe.values` also
exports `Function` and `FunctionSpec`; `probpipe.functions` exports the decorator
and experimental Module interfaces. The former exports from `probpipe.core.node`
have moved without import aliases; that module retains `Node` and `InputFrozenError`.
Configure invocation logging under `probpipe.functions._function`.

::: probpipe.Function

`Module`, `AbstractModule`, `workflow_method`, and `abstract_workflow_method`
are experimental. Their shared-input and dependency behavior remains available,
but their API may change. Using them emits no experimental runtime warning.
Module methods use `Class.method` as their function label and `method` as their
output name, with the same undeclared return inference as an ordinary Function.

::: probpipe.Module

::: probpipe.AbstractModule

::: probpipe.function

::: probpipe.workflow_method

::: probpipe.abstract_workflow_method

## Workflow RNG API

::: probpipe.workflow_run

::: probpipe.UnmanagedConcurrentWorkflowEntryError

## Orchestration configuration

`fn.options["workflow_kind"]` is the stored setting;
`fn.effective_workflow_kind` is the read-only mode used for execution. The latter
resolves the instance override, then the current global configuration when the
instance uses `DEFAULT`. If both are `DEFAULT`, the effective mode is `OFF`.
Resolution runs on every access, including after global configuration changes
or `with_options`. A requested `TASK` or `FLOW` warns and falls back to `OFF`
when Prefect is unavailable. Before the call engine is installed, the property
returns `OFF`, matching plain evaluation.

::: probpipe.WorkflowKind

::: probpipe.prefect_config

### `PROBPIPE_WORKFLOW_KIND` environment variable

`PROBPIPE_WORKFLOW_KIND` (case-insensitive: `off` / `task` / `flow` /
`default`) sets the initial `prefect_config.workflow_kind` at import time.
Unknown values raise `ValueError`. `prefect_config.reset()` re-reads the
variable.
