# ruff: noqa: E402  (the patches must apply before probpipe is imported)
"""A kernel that calls int() on a given that condition_on fixes.

Usage: uv run python review/jax-performance-bench/concrete2.py MODE
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
from probpipe import LogNormal, Poisson, condition_on, conditional_distribution, sample

prior = LogNormal("r", 0.0, 0.3) * LogNormal("n0", jnp.log(50.0), 0.1)


@conditional_distribution(given_spec=prior.event_spec.spec)
def lik(r, n0):
    steps = int(n0) // 10  # host conversion of a given
    return Poisson("y", n0 * jnp.exp(r * jnp.arange(steps, dtype=jnp.float32)))


model = lik * prior
try:
    sample(condition_on(model, {"n0": 40.0}), 8)
    print(mode, "fixed n0, int(n0): OK")
except Exception as e:
    print(mode, "fixed n0, int(n0): ERROR", type(e).__name__, str(e)[:200])
