> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Settings

`probpipe.config` holds the global settings of ProbPipe in four sections:

1. `config.sampling`: the default evaluation and draw counts of a call, the controls `max_evaluations` and `num_samples`;
2. `config.notation`: how many nested levels a label or a notation shows;
3. `config.provenance`: how much lineage history a provenance record keeps, a `ProvenanceMode` documented on [Labels and provenance](provenance.md);
4. `config.prefect`: the orchestration of a `Function`, a `WorkflowKind` documented on [Workflows and reproducibility](workflows.md), and the Prefect task runner.

Assigning a setting sets the session's value.
A `using` block sets temporary values, for one section with `config.sampling.using(num_samples=4096)` or for several with `config.using(sampling={"num_samples": 4096}, notation={"max_depth": 12})`.
A view's own control, which `f.using(...)` sets, wins over a block, and a block wins over the session's value.
`f.using(...)` returns a reusable configured copy of `f`, while `config.using(...)` is a context manager.
The work a call dispatches, such as a function's body run in a worker thread, reads the session's values as they were when the outermost call opened, except those of `config.prefect`, so a session assignment made during a call applies from the next call on.
The provenance mode is fixed for a `workflow_run`: inside the run, a block, an assignment, or a `reset` that would change it raises `RuntimeError`, so set the mode before the run opens.

```python
import probpipe

probpipe.config.notation.max_depth = 12
with probpipe.config.sampling.using(num_samples=4096):
    ...
probpipe.config.reset()
```

## The container

::: probpipe.config

::: probpipe.core.config.Config
    options:
      show_root_full_path: true

## The sections

::: probpipe.core.config.SamplingConfig
    options:
      show_root_full_path: true

::: probpipe.core.config.NotationConfig
    options:
      show_root_full_path: true

::: probpipe.core.config.ProvenanceConfig
    options:
      show_root_full_path: true

::: probpipe.core.config.PrefectConfig
    options:
      show_root_full_path: true
      docstring_options:
        warn_unknown_params: false

## Settings in worker processes

Work that a call runs in another process, as the `process` executor does, carries the call's settings.
The `prefect` section stays behind, so a call in a worker process runs without Prefect orchestration.
A setting whose value cannot be pickled travels as a placeholder, and reading it in the worker raises `UnshippableSettingError`.

::: probpipe.UnshippableSettingError
