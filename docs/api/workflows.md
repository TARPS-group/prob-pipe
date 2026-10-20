> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Workflows and reproducibility

A workflow scope derives the key of each random draw from its seed and the structure of the calls, so entering a scope again with the same seed and the same calls reproduces its draws.
`replay_run` replays one recorded call from its provenance.
This page also documents the orchestration settings and the session's defaults of the two count controls, and the provenance settings are on [Labels and provenance](provenance.md).

## Workflow scopes

::: probpipe.workflow_run

::: probpipe.UnmanagedConcurrentWorkflowEntryError

## Replay

::: probpipe.replay_run

::: probpipe.ReplayCompatibilityError

::: probpipe.ReplayUnsupportedCallableError

## Orchestration

::: probpipe.WorkflowKind

::: probpipe.prefect_config

::: probpipe.core.config.PrefectConfig
    options:
      show_root_full_path: true
      docstring_options:
        warn_unknown_params: false

## The default counts

`config.sampling` holds the session's defaults of the controls `max_evaluations` and `num_samples`.

::: probpipe.config

::: probpipe.core.config.Config
    options:
      show_root_full_path: true

::: probpipe.core.config.SamplingConfig
    options:
      show_root_full_path: true
