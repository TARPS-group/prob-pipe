# Workflows and orchestration

`Function` wraps every [op](operations.md) and every user-written
`@function`. `Module` is the stateful container with
`@workflow_method` children.

`Function` is an immutable, tracked and annotated ProbPipe object. Its
`signature` is captured from the wrapped Python callable once at construction;
optional event templates describe values, but do not replace or derive that
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

Use `workflow.with_options(...)(...)` for one-call overrides:

```python
result = score.with_options(n_broadcast_samples=2_000)(x, seed=7)
```

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

`Function(..., seed=...)` and `with_options(seed=...)` are not supported. A
wrapped function's own `seed` parameter remains an ordinary input.

## Workflow RNG scopes

`workflow_run` owns ProbPipe randomness for one structural execution:

```python
from probpipe import Normal, sample, workflow_run

dist = Normal(loc=0.0, scale=1.0, name="value")

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

## Raw application and authoritative templates

Calling a `Function` through `__call__` enables ProbPipe lifting, sweeps,
orchestration, result wrapping, and call provenance. Use `apply` when a caller
needs exactly one raw evaluation under the same signature, binding, default,
and schema checks:

```python
import jax.numpy as jnp

from probpipe import EventTemplate, function


@function(
    input_template=EventTemplate(x=("obs",), scale=()),
    output_template=EventTemplate(y=("obs",)),
)
def standardize(x, scale=1.0):
    return x / scale


values = jnp.array([1.0, 2.0])
raw = standardize.apply(values, scale=2.0)  # underlying array value
wrapped = standardize(values, scale=2.0)  # Record with field "y"
```

String dimensions such as `"obs"` are symbolic. They are bound separately for
each call, shared between the input and output templates, and never written
back into the declaration. Repeating a symbol requires equal sizes, including
across nested fields. `EventTemplate.free_dims` lists unresolved symbols and
`EventTemplate.is_concrete` reports whether none remain. A polymorphic numeric
template has no `vector_size` until its symbols are bound.

When supplied, templates are authoritative:

- input-template top-level fields and fixed signature parameters must match by
  name; variadic signatures can still be used when no input template is set;
- every symbolic output dimension must be declared by the input template;
- mappings must match the declared output structure, while a scalar or array
  result can satisfy only a single-leaf output template;
- existing `Record` results must conform to the same field tree and concrete
  shapes. Dtypes use same-kind conformance, just like bare values;
- an existing `Distribution` must expose an `event_template` exactly equal to
  the concrete declaration. Function does not reconcile separate `dtypes` or
  `supports` accessors, so a metadata-bearing declaration requires the
  Distribution's own template to be schema-complete;
- every declared output support is checked against concrete scalar, array,
  mapping, or Record data.

Authoritative mapping outputs are normalized to the declared `Record` pytree
before dispatch aggregation. Flat and nested output structures therefore have
the same value type, data, and concrete template under sequential, threaded,
Prefect, and JAX execution. This recursive packing is private to the Function
planner; it does not broaden the public `RecordBatch.stack` contract.

Variadic Functions participate fully when no authoritative input template is
declared. Each `*args` element and `**kwargs` entry is classified, lifted,
sampled, or swept independently. Informative variadic annotations apply to
each expanded slot; `Any` supplies no pass-through guarantee, so generic
`**kwargs: Any` APIs retain lifting and sweep behavior. The original Python
call is reconstructed before execution. Provenance and `include_inputs` labels
are stable, for example `*items[0]` and `**extras['scale']`. Tracked-term slots form
deduplicated lineage parents; ordinary slots remain distinct in
`Provenance.inputs` even when multiple parameters refer to the same object.

`apply` deliberately performs no distribution lifting, batch sweep, result
wrapping, orchestration, or call-provenance creation. It is therefore also the
raw execution boundary used by inference integrations. If the implementation
returns an existing `Record`, batch of records, or `Distribution`, `apply`
preserves that object's identity, annotations, and provenance. `__call__`
instead creates a shallow independent result item: value data and templates are
shared by default, the annotations container is copied, prior provenance is
cleared, and the current Function and tracked inputs become the new direct
parents. With an authoritative output declaration, this public result copy
carries the concrete declared template for `Record` and `RecordBatch` results
even when the raw implementation result had a weaker inferred template.
Distribution results instead retain their intrinsic, already-matching
`event_template`; Function never rewrites it. Value data remains shared and
`apply` leaves every raw object's template unchanged. This copy is still made
when provenance tracking is disabled.

Output-support checks are data-dependent and cannot execute under JAX tracing.
For broadcast and sweep calls, `dispatch="auto"` therefore detects a
support-bearing output and falls back to row-wise execution. An explicit
`dispatch="jax"` reports that output support validation requires eager
execution, while a direct `jax.jit(function.apply)` preserves JAX's native
tracer error. Neither path silently omits the support guarantee. Output
templates without support constraints retain their existing JAX path.

When a sweep returns distributions, its `DistributionArray.event_template`
records the concrete authoritative output template. Ordinary
`DistributionArray` construction exposes a common component template when all
components agree, and otherwise returns `None`. Broadcast marginals and nested
sweeps preserve the same concrete template rather than re-inferring it from an
arbitrary result cell.

## Wrappers and decorators

::: probpipe.Function

::: probpipe.Module

::: probpipe.function

::: probpipe.workflow_method

::: probpipe.abstract_workflow_method

## Workflow RNG API

::: probpipe.workflow_run

::: probpipe.UnmanagedConcurrentWorkflowEntryError

## Orchestration configuration

::: probpipe.WorkflowKind

::: probpipe.prefect_config

### `PROBPIPE_WORKFLOW_KIND` environment variable

`PROBPIPE_WORKFLOW_KIND` (case-insensitive: `off` / `task` / `flow` /
`default`) sets the initial `prefect_config.workflow_kind` at import time.
Unknown values raise `ValueError`. `prefect_config.reset()` re-reads the
variable.
