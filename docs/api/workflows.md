> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Workflows and reproducibility

A workflow scope derives the key of each random draw from its seed and the structure of the calls, so entering a scope again with the same seed and the same calls reproduces its draws.
`replay_run` replays one recorded call from its provenance.
This page also documents the orchestration modes, and the settings `config.prefect` and `config.sampling` are on [Settings](config.md).

## Workflow scopes

[Set a default](#probpipe.workflow_run--workflow-run-set-default) once to continue a random stream
across bare calls in a script or notebook. Setting it again restarts the stream.
[Clear the default](#probpipe.workflow_run--workflow-run-clear-default) to restore independent anonymous
scopes for bare calls.

```python
from probpipe import Normal, sample, workflow_run

law = Normal("x", loc=0.0, scale=1.0)
workflow_run.set_default(seed=42)
first = sample(law)
second = sample(law)
with workflow_run(seed=7):
    separate = sample(law)
third = sample(law)
workflow_run.clear_default()
print("Default stream draws:", first, second, third)
print("Independent scope draw:", separate)
```

The explicit scope leaves the default stream's next position unchanged.
Notebook cells share the configured stream, including cells with top-level
`await`. Each bare call fixes its own provenance mode, while an explicit scope
fixes the mode for its whole block.

::: probpipe.workflow_run
    options:
      show_attribute_values: false
      show_root_heading: false
      heading_level: 3

::: probpipe.UnmanagedConcurrentWorkflowEntryError

## Async tasks and threads

A notebook's top-level `await` continues in that cell's execution task.
`asyncio.create_task`, `asyncio.run`, and `asyncio.to_thread` can instead copy
settings into another task or thread. Bare calls there cannot use an inherited
default stream: they raise `UnmanagedConcurrentWorkflowEntryError`.
Set a default inside the new task, clear its inherited default to use fresh
anonymous scopes, or open an explicit scope inside the worker.

```python
import asyncio
from probpipe import Normal, sample, workflow_run

law = Normal("x", loc=0.0, scale=1.0)

def worker():
    with workflow_run(seed=7):
        return sample(law)

async def main():
    workflow_run.set_default(seed=42)
    try:
        first = sample(law)
        separate = await asyncio.to_thread(worker)
        second = sample(law)
        print("Task's default stream draws:", first, second)
        print("Worker's independent draw:", separate)
    finally:
        workflow_run.clear_default()

asyncio.run(main())  # In a notebook, use await main() instead.
```

## Replay

::: probpipe.replay_run

::: probpipe.ReplayCompatibilityError

::: probpipe.ReplayUnsupportedCallableError

## Orchestration

`config.prefect.workflow_kind` takes one of these modes, as [Settings](config.md) states.

::: probpipe.WorkflowKind
