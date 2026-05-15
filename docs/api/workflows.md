# Workflows and orchestration

`WorkflowFunction` is the orchestration-aware function wrapper that backs
every [op](operations.md) and every user-written `@workflow_function`.
`Module` is the stateful container with `@workflow_method` children.

Prefect orchestration is **off by default**. Opt in globally by setting
`prefect_config.workflow_kind = WorkflowKind.TASK` (or
`WorkflowKind.FLOW`), or by exporting `PROBPIPE_WORKFLOW_KIND=task` in the
environment.

## Wrappers and decorators

::: probpipe.WorkflowFunction

::: probpipe.Module

::: probpipe.workflow_function

::: probpipe.workflow_method

::: probpipe.abstract_workflow_method

## Orchestration configuration

`WorkflowKind` is the enum that selects the orchestration mode; the
`prefect_config` singleton holds the global default plus an optional Prefect
task runner. Per-instance overrides (e.g.
`@workflow_function(workflow_kind="task")`) bypass the global.

::: probpipe.WorkflowKind

::: probpipe.prefect_config

### `PROBPIPE_WORKFLOW_KIND` environment variable

Setting `PROBPIPE_WORKFLOW_KIND` (case-insensitive: `off` / `task` / `flow` /
`default`) configures the initial `prefect_config.workflow_kind` at import
time. Unknown values raise `ValueError` so deployment-config typos surface
loudly. `prefect_config.reset()` re-reads the variable.
