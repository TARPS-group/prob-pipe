# ruff: noqa: E402  (the patches must apply before probpipe is imported)
"""Kernels that convert a given to a Python value, and a kernel that calls sample.

Usage: uv run python review/jax-performance-bench/concrete.py MODE
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax.numpy as jnp

mode = sys.argv[1]
if mode != "baseline":
    import patches

    patches.apply(mode)
from probpipe import LogNormal, Normal, Poisson, condition_on, conditional_distribution, sample

prior = LogNormal("r", 0.0, 0.3) * LogNormal("n0", jnp.log(50.0), 0.1)


@conditional_distribution(given_spec=prior.event_spec.spec)
def lik(r, n0, years=5):
    steps = int(years)  # host conversion of a given that is never drawn
    return Poisson("y", n0 * jnp.exp(r * jnp.arange(steps, dtype=jnp.float32)))


model = lik * prior
for name, f in [
    ("sample(model, 8)", lambda: sample(model, 8)),
    ("sample(condition_on(model, n0=40), 8)", lambda: sample(condition_on(model, {"n0": 40.0}), 8)),
]:
    try:
        f()
        print(mode, name, "OK")
    except Exception as e:
        print(mode, name, "ERROR", type(e).__name__, str(e)[:160])
# a kernel that calls a workflow-owned random operation (issue question 3)
try:

    @conditional_distribution(given_spec=prior.event_spec.spec)
    def bad(r, n0):
        jitter = sample(Normal("e", 0.0, 1.0), 1).raw()
        return Poisson("y", n0 * jnp.exp(r) + jitter)

    print(mode, "kernel calling sample: built")
except Exception as e:
    print(mode, "kernel calling sample: ERROR at construction", type(e).__name__, str(e)[:240])
