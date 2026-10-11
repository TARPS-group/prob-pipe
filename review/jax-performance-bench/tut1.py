# ruff: noqa: E402  (the patches must apply before probpipe is imported)
"""Time the Tutorial 1 calls and count their compilations.

Usage: uv run python review/jax-performance-bench/tut1.py MODE
MODE is baseline, sampler, or kernel (see patches.py). USERJIT=1 jits the Ricker function.
"""

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import jax
import jax.numpy as jnp
from compilecount import fmt, measure

mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
if mode != "baseline":
    import patches

    patches.apply(mode)
from probpipe import (
    Beta,
    LogNormal,
    Poisson,
    condition_on,
    conditional_distribution,
    mean,
    sample,
)

n_years = 22
K = LogNormal("K", loc=jnp.log(350.0), scale=0.4, label="prior")
r = LogNormal("r", jnp.log(0.4), 0.4, label="prior")
phi = Beta("phi", 2.0, 2.0, label="prior")
prior = K * r * phi


def ricker(r, K, N0, years):
    def step(N, _):
        N_next = N * jnp.exp(r * (1.0 - N / K))
        return N_next, N_next

    _, N = jax.lax.scan(step, jnp.asarray(N0, dtype=jnp.float32), None, length=years)
    return N


import os

if os.environ.get("USERJIT"):
    ricker = jax.jit(ricker, static_argnames="years")


@conditional_distribution(given_spec=prior.event_spec.spec, label="likelihood")
def likelihood(K, r, phi, n0=50.0):
    return Poisson("y", phi * ricker(r, K, n0, n_years))


model = likelihood * prior
at = condition_on(model, {"K": 350.0, "r": 0.4, "phi": 0.6})
calls = {
    "sample(K, 1000)": lambda: sample(K, 1000),
    "sample(prior, 100)": lambda: sample(prior, 100),
    'sample(model["r"], 10)': lambda: sample(model["r"], 10),
    "sample(model, 50)": lambda: sample(model, 50),
    "sample(condition_on(model,...), 50)": lambda: sample(at, 50),
    "mean(model)": lambda: mean(model),
}
print(f"mode={mode}   per call: wall, c=#XLA compiles(compile secs), t=#jaxpr traces")
for name, call in calls.items():
    print(fmt(name, measure(call)), flush=True)
