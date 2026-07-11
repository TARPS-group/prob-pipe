"""Distributed bagged posteriors with Ray via Prefect.

Runs the same bagged posterior workflow as ``run_prefect_demo.py``, but
uses ``RayTaskRunner`` so each bootstrap MCMC fit runs in a separate Ray
worker process — true parallelism across CPU cores.

This is the current Ray support path for ProbPipe: Prefect orchestrates
``WorkflowFunction`` tasks, and Prefect-Ray submits those tasks to Ray. It is
not a native Ray backend and does not expose ``ray.remote`` or ``ray.put``
through ProbPipe.

This works because ``Record.__reduce__`` was added to make Records
pickle-serializable, which is required for Ray to ship arguments to workers.
The task function, arguments, closed-over state, and return value must all be
serializable.

Usage
-----
1. Start the Prefect server in a separate terminal:

       prefect server start

2. Start a persistent local Ray head in another terminal:

       ray start --head

   This is required, not optional. ``RayTaskRunner.__exit__`` calls
   ``ray.shutdown()`` after every flow run, which tears down any
   cluster the driver process owns. Starting Ray externally first means
   ``ray.shutdown()`` only disconnects the driver — the head survives,
   and subsequent flows reuse it instead of bootstrapping their own.
   See https://docs.prefect.io/integrations/prefect-ray for the full
   deployment topology guide.

3. Open the Prefect dashboard in your browser:

       http://127.0.0.1:4200

4. Run this script (assuming you're in the ``prob-pipe`` root directory):

       python example_scripts/run_ray_demo.py

You will see each bootstrap MCMC fit appear as a task in the Prefect
dashboard, dispatched across Ray worker processes.

Prerequisites
-------------
    pip install probpipe[prefect]
    pip install "prefect[ray]"

For production deployments, read the Prefect-Ray driver-placement guidance and
Ray's serialization and dependency documentation before choosing between
``address="auto"`` and ``ray://<head>:10001``.

"""

import warnings

warnings.filterwarnings("ignore", message=r"Explicitly requested dtype.*float64.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import time

import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax.glm as tfp_glm

# ---------------------------------------------------------------------------
# 1. Configure Prefect to use RayTaskRunner
# ---------------------------------------------------------------------------
from prefect_ray import RayTaskRunner

import probpipe
from probpipe import (
    BootstrapReplicateDistribution,
    EmpiricalDistribution,
    GLMLikelihood,
    Normal,
    ProductDistribution,
    Record,
    SimpleModel,
    WorkflowKind,
    condition_on,
    mean,
    workflow_function,
)

# ``address="auto"`` attaches to the persistent Ray head you launched with
# ``ray start --head`` (see Usage above). Without that head, each flow would
# bootstrap a fresh local cluster — and tear it down — because
# ``RayTaskRunner.__exit__`` calls ``ray.shutdown()`` unconditionally.
#
# For a production setup with a remote head node, use the Ray Client URL
# instead — e.g. ``RayTaskRunner(address="ray://<head>:10001")``. That path
# has different tradeoffs (30s Client-disconnect timeouts, head-side
# dependency requirements); see
# https://docs.prefect.io/integrations/prefect-ray for guidance.
probpipe.prefect_config.workflow_kind = WorkflowKind.TASK
probpipe.prefect_config.task_runner = RayTaskRunner(address="auto")

print(f"Workflow kind: {probpipe.prefect_config.workflow_kind}")
print(f"Task runner:   {type(probpipe.prefect_config.task_runner).__name__}")
print()


# ---------------------------------------------------------------------------
# 2. Load data and build models (same as Getting Started tutorial)
# ---------------------------------------------------------------------------

df = pd.read_csv("./docs/tutorials/data/horseshoe_crabs.csv")


@workflow_function
def prep_data(width, satellites) -> Record:
    width = np.asarray(width, dtype=np.float32)
    width_z = (width - np.mean(width)) / np.std(width)
    X = np.column_stack([np.ones(len(width)), width_z]).astype(np.float32)
    return Record(X=X, y=np.asarray(satellites, dtype=np.float32))


data = prep_data(df["width_cm"], df["satellites"])

prior = ProductDistribution(
    intercept=Normal(loc=0.0, scale=jnp.sqrt(5.0), name="intercept"),
    slope=Normal(loc=0.0, scale=jnp.sqrt(5.0), name="slope"),
)

model_poisson = SimpleModel(prior, GLMLikelihood(tfp_glm.Poisson(), data["X"]), name="poisson")
model_nb = SimpleModel(prior, GLMLikelihood(tfp_glm.NegativeBinomial(), data["X"]), name="negbin")

print(f"Observations: {len(data['y'])}")
print(f"Models: {model_poisson.name}, {model_nb.name}")
print()


# ---------------------------------------------------------------------------
# 3. Run bagged posteriors — dispatched to Ray workers via Prefect
# ---------------------------------------------------------------------------

N_REPLICATES = 16

bootstrap = BootstrapReplicateDistribution(EmpiricalDistribution(data))

print(f"Running {N_REPLICATES} bootstrap replicates per model via Ray...")
print("Check the Prefect dashboard at http://127.0.0.1:4200 to see tasks.\n")

t0 = time.perf_counter()
bagged_poisson = condition_on(
    model_poisson,
    X=bootstrap["X"],
    y=bootstrap["y"],
    n_broadcast_samples=N_REPLICATES,
)
t_poisson = time.perf_counter() - t0
print(f"  Poisson: {t_poisson:.1f}s")

t0 = time.perf_counter()
bagged_nb = condition_on(
    model_nb,
    X=bootstrap["X"],
    y=bootstrap["y"],
    n_broadcast_samples=N_REPLICATES,
)
t_nb = time.perf_counter() - t0
print(f"  NegBin:  {t_nb:.1f}s")
print()


# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------

# Turn off Prefect for post-processing
probpipe.prefect_config.workflow_kind = WorkflowKind.OFF

for label, bagged in [("Poisson", bagged_poisson), ("NegBin", bagged_nb)]:
    ind_means = np.array([np.array(mean(p).flatten()) for p in bagged.components])
    all_draws = np.concatenate([np.asarray(p.draws().flatten()) for p in bagged.components])
    ratio = np.var(ind_means, axis=0) / np.var(all_draws, axis=0)
    print(f"{label}:")
    print(f"  Posterior mean: {np.array2string(np.mean(ind_means, axis=0), precision=3)}")
    print(f"  Sampling variability ratio: {np.array2string(ratio, precision=3)}")
    print(f"  Misspecified: {'YES' if np.any(ratio > 0.5) else 'no'}")
    print()


# ---------------------------------------------------------------------------
# 5. Provenance
# ---------------------------------------------------------------------------

src = bagged_nb.provenance
print("Provenance (NegBin bagged posterior):")
print(f"  Operation:   {src.operation}")
print(f"  Orchestrate: {src.metadata['orchestrate']}")
print(f"  N replicates: {src.metadata['n_samples']}")
print()
print("Done! Each replicate ran in a separate Ray worker process.")
print("Check the Prefect dashboard for the full task history.")
