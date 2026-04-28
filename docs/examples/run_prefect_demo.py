"""Distributed bagged posteriors with a real Prefect server.

Usage
-----
1. Start the Prefect server in a separate terminal:

       prefect server start

2. Open the Prefect dashboard in your browser:

       http://127.0.0.1:4200

3. Run this script (assuming you're in the ``prob-pipe`` root directory):

       python docs/examples/run_prefect_demo.py

You will see each bootstrap MCMC fit appear as a task in the Prefect
dashboard.

Prerequisites
-------------
    pip install probpipe[prefect]

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

import probpipe
from probpipe import (
    Record, Normal, ProductDistribution,
    EmpiricalDistribution, BootstrapReplicateDistribution,
    GLMLikelihood, SimpleModel, WorkflowKind,
    condition_on, mean, workflow_function,
)


# ---------------------------------------------------------------------------
# 1. Configure Prefect globally
# ---------------------------------------------------------------------------

# Use Prefect tasks with the default ThreadPool runner.
# Ray auto-detection is disabled because Record is not yet
# pickle-serializable (see docstring above).
from prefect.task_runners import ThreadPoolTaskRunner

probpipe.prefect_config.workflow_kind = WorkflowKind.TASK
probpipe.prefect_config.task_runner = ThreadPoolTaskRunner()

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

model_poisson = SimpleModel(
    prior, GLMLikelihood(tfp_glm.Poisson(), data["X"]), name="poisson"
)
model_nb = SimpleModel(
    prior, GLMLikelihood(tfp_glm.NegativeBinomial(), data["X"]), name="negbin"
)

print(f"Observations: {len(data['y'])}")
print(f"Models: {model_poisson.name}, {model_nb.name}")
print()


# ---------------------------------------------------------------------------
# 3. Run bagged posteriors — tracked by Prefect
# ---------------------------------------------------------------------------

N_REPLICATES = 16

bootstrap = BootstrapReplicateDistribution(EmpiricalDistribution(data))

print(f"Running {N_REPLICATES} bootstrap replicates per model...")
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
    ind_means = np.array([np.array(mean(p)) for p in bagged.components])
    all_draws = np.concatenate(
        [np.asarray(p.draws().flatten()) for p in bagged.components]
    )
    ratio = np.var(ind_means, axis=0) / np.var(all_draws, axis=0)
    print(f"{label}:")
    print(f"  Posterior mean: {np.array2string(np.mean(ind_means, axis=0), precision=3)}")
    print(f"  Sampling variability ratio: {np.array2string(ratio, precision=3)}")
    print(f"  Misspecified: {'YES' if np.any(ratio > 0.5) else 'no'}")
    print()


# ---------------------------------------------------------------------------
# 5. Provenance
# ---------------------------------------------------------------------------

src = bagged_nb.source
print("Provenance (NegBin bagged posterior):")
print(f"  Operation:   {src.operation}")
print(f"  Orchestrate: {src.metadata['orchestrate']}")
print(f"  N replicates: {src.metadata['n_samples']}")
print()
print("Done! Check the Prefect dashboard for the full task history.")
