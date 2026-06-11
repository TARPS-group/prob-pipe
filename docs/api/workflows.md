# Workflows and orchestration

`WorkflowFunction` wraps every [op](operations.md) and every user-written
`@workflow_function`. `Module` is the stateful container with
`@workflow_method` children.

Prefect orchestration is **off by default**. Set
`prefect_config.workflow_kind = WorkflowKind.TASK` (or `FLOW`) globally, or
export `PROBPIPE_WORKFLOW_KIND=task` in the environment.

## Options namespace

Use bare `@workflow_function` when no ProbPipe controls are needed:

```python
@workflow_function
def score(x, seed):
    return x + seed
```

Use `@workflow_function.options(...)` for definition-time controls:

```python
@workflow_function.options(dispatch="jax", n_broadcast_samples=1_000, seed=0)
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

## Wrappers and decorators

::: probpipe.WorkflowFunction

::: probpipe.Module

::: probpipe.workflow_function

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
