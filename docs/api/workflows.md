> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Workflows and reproducibility

A workflow scope derives the key of each random draw from its seed and the structure of the calls, so entering a scope again with the same seed and the same calls reproduces its draws.
`replay_run` replays one recorded call from its provenance.
This page also documents the orchestration modes, and the settings `config.prefect` and `config.sampling` are on [Settings](config.md).

## Workflow scopes

::: probpipe.workflow_run

::: probpipe.UnmanagedConcurrentWorkflowEntryError

## Replay

::: probpipe.replay_run

::: probpipe.ReplayCompatibilityError

::: probpipe.ReplayUnsupportedCallableError

## Orchestration

`config.prefect.workflow_kind` takes one of these modes, as [Settings](config.md) states.

::: probpipe.WorkflowKind
