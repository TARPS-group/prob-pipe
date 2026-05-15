# Workflows and orchestration

`WorkflowFunction` wraps every [op](operations.md) and every user-written
`@workflow_function`. `Module` is the stateful container with
`@workflow_method` children.

Prefect orchestration is **off by default**. Set
`prefect_config.workflow_kind = WorkflowKind.TASK` (or `FLOW`) globally, or
export `PROBPIPE_WORKFLOW_KIND=task` in the environment.

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
