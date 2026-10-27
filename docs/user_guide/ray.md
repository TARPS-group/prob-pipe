> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Ray via Prefect

Use Ray through Prefect when a function has expensive independent evaluations that can run on separate workers.
Start with [Functions on values and distributions](05_functions_and_lifting.ipynb) to understand how a function applies to distributions and batches.
The configuration below changes where evaluations run while preserving the mathematical request.

## Choose what to distribute

ProbPipe runs functions in the current process by default.
Enable Prefect orchestration explicitly to use a task runner such as `RayTaskRunner`.
ProbPipe chooses the work it submits to Prefect from the callable's traceability:

| Callable | Work submitted with `WorkflowKind.TASK` |
|---|---|
| Traceable by JAX | The vectorized computation runs as one Prefect task. |
| Opaque to JAX | Prefect maps separate evaluation points as tasks through the configured runner. |

A traceable callable can be compiled and mapped by JAX for its input shapes.
An opaque callable needs ordinary Python evaluation, such as a call to an external program.
Prefect owns execution when orchestration is enabled; naming an `executor` then does not select a different task-mapping pattern.

Examples of work that can benefit from separate workers include:

- expensive parameter sweeps;
- independent model fits;
- bootstrap replicates with expensive statistics.

Measure the runtime of your workload with and without distribution.
For short scalar computations, task scheduling can cost more than evaluation.

## Install

Install the `prefect` extra from the source branch used by this documentation, then install Prefect-Ray:

```bash
python -m pip install "probpipe-core[prefect] @ git+https://github.com/TARPS-group/prob-pipe.git@main"
python -m pip install prefect-ray
```

For an existing checkout, the first command can instead be:

```bash
python -m pip install -e ".[prefect]"
```

[Installation](../get_started/installation.md) explains the Python environment and other optional backends.
Workers need the packages that the submitted function imports.
The [Prefect-Ray guide](https://docs.prefect.io/integrations/prefect-ray/index) covers its installation and deployment options.

## Start local services

Start a Prefect server in one terminal and leave it running:

```bash
prefect server start
```

In another terminal, start a Ray head and configure the Prefect API address:

```bash
ray start --head
export PREFECT_API_URL="http://127.0.0.1:4200/api"
```

Run your script or notebook from that environment.
On Windows, set `PREFECT_API_URL` using your shell's environment-variable syntax.

## Configure and run a mapped function

Set the runner explicitly so its address is part of the configuration you can inspect.
This example declares an opaque function so the batch's elements are submitted as separate tasks:

<!-- docs-test: skip, this example needs Prefect-Ray and running Prefect and Ray services -->
```python
import jax.numpy as jnp
from prefect_ray import RayTaskRunner

import probpipe
from probpipe import NumericArrayBatch, WorkflowKind, function

probpipe.config.prefect.task_runner = RayTaskRunner(address="auto")


@function(traceable=False, workflow_kind=WorkflowKind.TASK)
def square(x: float) -> float:
    return float(x) ** 2


values = NumericArrayBatch(jnp.arange(4.0), "case", label="values")
result = square(values)
print("squared values:", result.raw)
```

The squared values are `[0., 1., 4., 9.]`.
The scalar calculation illustrates task mapping; choose a more expensive function when measuring a benefit from Ray.

For process-wide orchestration, set `probpipe.config.prefect.workflow_kind = WorkflowKind.TASK`.
An explicit `workflow_kind` on a function overrides that default.
When `task_runner` is unset, ProbPipe tries the following runners in order:

1. `RayTaskRunner()` when `prefect_ray` is importable.
2. A Dask runner when `prefect_dask` is importable.
3. Prefect's default runner.

Installing Prefect-Ray can therefore change the automatically selected runner once orchestration is enabled.

Provenance records the orchestration mode as `"task"`.
Ray is the configured execution service; it is not a separate `WorkflowKind`.

## Connect workers and diagnose failures

`RayTaskRunner(address="auto")` attaches to an existing Ray cluster discovered in the local environment.
A remote driver can use a Ray Client address such as `ray://<head>:10001` when the cluster supports that connection.
Follow your cluster's networking and environment configuration for remote deployment.

The submitted function, its arguments, and its result must be serializable.
Prefer top-level functions with explicit inputs over closures that capture large mutable objects.
The Python setup block above is excluded from the ordinary documentation tests because it needs running services.
ProbPipe's serialization tests exercise records and supported composite distributions without a Ray cluster; they do not verify cluster connectivity.

| Symptom | What to check |
|---|---|
| Prefect cannot connect | The server is running and `PREFECT_API_URL` points to it. |
| Ray cannot find a cluster | The Ray head is running and the runner's address matches the deployment. |
| A worker raises `ModuleNotFoundError` | The driver and workers have the required task dependencies. |
| Serialization fails | The function and its inputs can be serialized without local process state. |
| The workload is slower | Each task has enough computation to justify its scheduling and transfer costs. |
