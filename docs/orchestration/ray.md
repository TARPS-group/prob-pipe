# Ray via Prefect

Current support: **Ray through Prefect-Ray**.

This is not a native Ray backend. ProbPipe does not expose `ray.remote`,
`ray.put`, Ray actors, placement groups, or Ray resource hints as public
ProbPipe APIs. Ray is used through Prefect's task runner interface, so existing
`WorkflowFunction` calls can dispatch mapped tasks to Ray worker processes when
Prefect orchestration is explicitly enabled.

## Execution Model

The current execution path is:

1. A `WorkflowFunction` receives distribution-valued or array-valued inputs.
2. ProbPipe expands the call into mapped work with `Prefect task.map()` inside a
   lightweight flow.
3. `probpipe.prefect_config.task_runner` supplies the Prefect task runner.
4. `RayTaskRunner` submits the mapped tasks to Ray.
5. Ray runs each task in worker processes and returns the results to Prefect.

This keeps Ray at the orchestration layer. The probabilistic model, records,
distributions, inference methods, and public operations stay unchanged.

Ray via Prefect is a good fit for coarse-grained independent work:

- bagged posteriors
- bootstrap replicates
- parameter sweeps
- multiple independent MCMC fits

It is usually not a good fit for very small operations where scheduling
overhead dominates, workflows that depend on shared mutable state, or workloads
that require direct Ray actors, direct object-store control, or fine-grained
CPU/GPU resource hints.

## Install

For an installed ProbPipe package:

```bash
pip install "probpipe[prefect]"
pip install "prefect[ray]"
```

For local development from a checkout:

```bash
pip install -e ".[prefect]"
pip install "prefect[ray]"
```

There is intentionally no `probpipe[ray]` extra yet. Ray support currently
enters through Prefect-Ray, and adding a first-class Ray extra would expand the
packaging surface.

## Local Demo

The canonical local demo is `example_scripts/run_ray_demo.py`. Run it with a
persistent local Ray head:

```bash
prefect server start
ray start --head
python example_scripts/run_ray_demo.py
```

The script configures:

```python
from prefect_ray import RayTaskRunner

import probpipe
from probpipe import WorkflowKind

probpipe.prefect_config.workflow_kind = WorkflowKind.TASK
probpipe.prefect_config.task_runner = RayTaskRunner(address="auto")
```

`WorkflowKind.TASK` is still a Prefect orchestration mode. Provenance records the
orchestration value as `"task"`, not `"ray"`.

## Choosing A Ray Address

Use `RayTaskRunner(address="auto")` for the short-term supported pattern. It
attaches the Prefect driver process to a Ray raylet on the same machine. With a
local demo, starting `ray start --head` first gives the driver a persistent Ray
head to attach to.

Use `RayTaskRunner(address="ray://<head>:10001")` when the driver runs outside
the Ray cluster and connects through Ray Client. This is useful for some remote
setups, but it has stricter environment requirements: the Ray head and workers
must be able to import the packages that appear in the pickled task graph.

In both modes, the task function, arguments, closed-over state, and return value
must be serializable. ProbPipe's `Record` family and several composite
distributions include pickle support for this reason, and the serialization
tests cover those contracts without requiring a live Ray cluster.

## Troubleshooting

If the flow cannot find a Ray cluster, make sure `ray start --head` is running
before launching the demo, or pass the correct remote Ray address.

If workers raise `ModuleNotFoundError`, install the same task dependencies on
the process running the Ray driver and on the Ray cluster environment.

If a task fails during serialization, simplify what the task receives: prefer
Records, numeric arrays, distributions with tested pickle support, and
top-level functions over closures that capture large or non-serializable state.

If the workload is slower under Ray, increase the grain size. Ray via Prefect is
intended for expensive independent tasks such as bootstrap replicates and MCMC
fits, not tiny scalar operations.

## Future Directions

The current Ray via Prefect path leaves room for a later native or
multi-backend execution design. That future design may address:

- native execution backend selection
- `ray.put` and object-store data locality
- resource hints for CPU, GPU, memory, or custom resources
- actors for persistent state
- Ray Jobs or cluster submission workflows

Those capabilities are not part of the current public API.
