> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Workflows and reproducibility

A workflow scope derives the key of each random draw from its seed and the structure of the calls, so entering a scope again with the same seed and the same calls reproduces its draws.
`replay_run` replays one recorded call from its provenance.
This page also documents the orchestration modes, and the settings `config.prefect` and `config.sampling` are on [Settings](config.md).

## Workflow scopes

`workflow_run.set_default(seed=...)` starts a continuing random stream for bare calls.
Setting it again restarts that stream. `clear_default()` restores independent
anonymous scopes for bare calls. Both methods apply to the current context.

```python
from probpipe import Normal, sample, workflow_run

law = Normal("x", loc=0.0, scale=1.0)
workflow_run.set_default(seed=114514)
first = sample(law)
second = sample(law)
with workflow_run(seed=42):
    separate = sample(law)
third = sample(law)
workflow_run.clear_default()
```

The first, second, and third draws use consecutive positions of the default
stream. The explicit scope has its own positions. `set_default()` without a
seed creates one continuing stream rooted in fresh entropy on its first draw.
A default scope fixes the provenance mode separately for each top-level call,
so the mode can change between calls. An explicit scope fixes it for the whole
block. Setting or clearing the default inside either running calculation raises
`RuntimeError`. A failed call keeps the random positions it consumed.

The default belongs to the process, thread, and asyncio task that set it.
In an IPython notebook, it continues across cells, including cells with top-level
`await`. Each cell uses the same stream without setting the seed again. Tasks
created with `asyncio.create_task` do not acquire the cell's permission to use
that stream. A child task can set its own default or open an explicit scope
without changing the notebook's default. Clearing the default in a cell also
clears it for later cells.

A copied context in another owner must set or clear its own default before a
bare call. Workers dispatched by ProbPipe inherit their calling workflow's
random authority. `config.reset()` leaves the default intact.

::: probpipe.workflow_run
    options:
      show_attribute_values: false

::: probpipe.UnmanagedConcurrentWorkflowEntryError

## Replay

::: probpipe.replay_run

::: probpipe.ReplayCompatibilityError

::: probpipe.ReplayUnsupportedCallableError

## Orchestration

`config.prefect.workflow_kind` takes one of these modes, as [Settings](config.md) states.

::: probpipe.WorkflowKind
