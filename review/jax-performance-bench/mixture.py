# ruff: noqa: E402  (the patches must apply before probpipe is imported)
"""Recompilation of a mixture's sample, and its eager against its jitted draws.

Usage: uv run python review/jax-performance-bench/mixture.py MODE
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp
from compilecount import fmt, measure

mode = sys.argv[1]
if mode != "baseline":
    import patches

    patches.apply(mode)
from probpipe import Beta, sample
from probpipe.families._mixture import MixtureDistribution

m = MixtureDistribution([Beta("x", 2.0, 5.0), Beta("x", 5.0, 2.0)], jnp.array([0.3, 0.7]))
print(mode)
print(fmt("sample(mixture of Betas, 100)", measure(lambda: sample(m, 100), n=4)))
key = jax.random.key(0)
eager = m._sample(key, (100,))
traced = jax.jit(lambda k: m._sample(k, (100,)))(key)
print(
    "eager == jitted draws at one key:",
    bool(jnp.all(eager == traced)),
    "max abs diff",
    float(jnp.max(jnp.abs(eager - traced))),
)
