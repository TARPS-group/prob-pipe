# Workflows and orchestration

`Function` wraps every [op](operations.md) and every user-written
`@function`. `Module` is the stateful container with
`@workflow_method` children.

`Function` is an immutable, tracked and annotated ProbPipe object. Its
`signature` is captured from the wrapped Python callable once at construction;
optional event templates describe values, but do not replace or derive that
Python calling contract. Function calls record the Function itself as the first
provenance parent, followed by tracked inputs in parameter order.

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
@function(dispatch="jax", n_broadcast_samples=1_000, seed=0)
def score(x, seed):
    return x + seed
```

Use `workflow.with_options(...)(...)` for one-call overrides:

```python
result = score.with_options(seed=42, n_broadcast_samples=2_000)(x, seed=7)
```

Keyword arguments in the final workflow call belong to the wrapped user
function whenever they can bind to that function. This keeps common names
such as `seed`, `name`, `dispatch`, `n_broadcast_samples`, and
`include_inputs` available for user APIs.

Seeds are invocation-local. Repeating a call with the same construction seed,
or the same `with_options(seed=...)` override, produces the same sampling key
sequence without mutating the `Function`; concurrent calls do not share RNG or
automatic-dispatch state.

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
- existing `Record` and `Distribution` results must carry the same concrete
  event template.

Authoritative mapping outputs are normalized to the declared `Record` pytree
before dispatch aggregation. Flat and nested output structures therefore have
the same value type, data, and concrete template under sequential, threaded,
Prefect, and JAX execution. This recursive packing is private to the Function
planner; it does not broaden the public `RecordArray.stack` contract.

Variadic Functions participate fully when no authoritative input template is
declared. Each `*args` element and `**kwargs` entry is classified, lifted,
sampled, or swept independently, using the variadic parameter's annotation.
The original Python call is reconstructed before execution. Provenance and
`include_inputs` labels are stable, for example `*items[0]` and
`**extras['scale']`.

`apply` deliberately performs no distribution lifting, batch sweep, result
wrapping, orchestration, or call-provenance creation. It is therefore also the
raw execution boundary used by inference integrations. If the implementation
returns an existing `Record`, `RecordArray`, or `Distribution`, `apply`
preserves that object's identity, annotations, and provenance. `__call__`
instead creates a shallow independent result item: value data and templates are
shared, the annotations container is copied, prior provenance is cleared, and
the current Function and tracked inputs become the new direct parents. This
copy is still made when provenance tracking is disabled.

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

## Orchestration configuration

::: probpipe.WorkflowKind

::: probpipe.prefect_config

### `PROBPIPE_WORKFLOW_KIND` environment variable

`PROBPIPE_WORKFLOW_KIND` (case-insensitive: `off` / `task` / `flow` /
`default`) sets the initial `prefect_config.workflow_kind` at import time.
Unknown values raise `ValueError`. `prefect_config.reset()` re-reads the
variable.
