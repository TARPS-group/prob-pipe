> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Settings

`probpipe.config` holds the global settings of ProbPipe in four sections:

1. `config.sampling`: the default evaluation and draw counts of a call, the controls `max_evaluations` and `num_samples`;
2. `config.notation`: how many nested levels a label or a notation shows;
3. `config.provenance`: the data retained by provenance records, a `ProvenanceMode` documented on [Labels and provenance](provenance.md);
4. `config.prefect`: the orchestration of a `Function`, a `WorkflowKind` documented on [Workflows and reproducibility](workflows.md), and the Prefect task runner.

Assigning a setting sets the session's value.
A `using` block sets temporary values, for one section with `config.sampling.using(num_samples=4096)` or for several with `config.using(sampling={"num_samples": 4096}, notation={"max_depth": 12})`.
A view's own control, which `f.using(...)` sets, wins over a block, and a block wins over the session's value.
`f.using(...)` returns a reusable configured copy of `f`, while `config.using(...)` is a context manager.
The work a call dispatches, such as a function's body run in a worker thread, reads the session's values as they were when the outermost call opened, except those of `config.prefect`, so a session assignment made during a call applies from the next call on.
[Workflow scopes](workflows.md#workflow-scopes) determine when each call's provenance mode is fixed.

```python
import probpipe

probpipe.config.notation.max_depth = 12
with probpipe.config.sampling.using(num_samples=4096):
    ...
probpipe.config.reset()
```

## Static and inert settings

Each setting is one of two kinds, which its docstring states.

1. A **static** setting can change what a compiled JAX program computes, because ProbPipe code reads it while JAX traces the program, as a kernel inside a joint may read `num_samples`. A JAX program that ProbPipe compiles for a call is compiled for the values of the static settings in effect, so a call under other values compiles its own and computes with its own values. The static settings are `sampling.max_evaluations` and `sampling.num_samples`.
2. An **inert** setting changes no computed value, such as how a label renders or whether provenance is recorded, so no compiled JAX program depends on it, and changing it never compiles anything again. Every setting of `config.notation`, `config.provenance`, and `config.prefect` is inert.

```python
from probpipe import Normal, NumericArraySpec, conditional_distribution, sample, workflow_run


@conditional_distribution(given_spec={"mu": NumericArraySpec(())})
def likelihood(mu):
    return Normal("y", mu, 1.0)


model = (likelihood & Normal("mu", 0.0, 1.0))
with workflow_run(seed=0):
    sample(model, 100)  # compiles the joint's JAX program for the session's num_samples, 1024
    with probpipe.config.sampling.using(num_samples=2000):
        sample(model, 100)  # compiles another, for 2000
    with probpipe.config.notation.using(max_depth=12):
        sample(model, 100)  # an inert setting: reuses the JAX program compiled for 1024
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
