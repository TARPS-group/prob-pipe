"""A jitted chain runner reuses its compiled program when the target is a Partial over a stable function.

Usage: uv run python review/jax-performance-bench/partial_cache.py
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp
from compilecount import _stats
from jax.tree_util import Partial


def density(theta, data):
    return -0.5 * jnp.sum((data - theta) ** 2)


@partial(jax.jit, static_argnames="n")
def runner(target, key, n):
    def step(x, k):
        x = x + 0.1 * jax.random.normal(k) + 0.01 * jax.grad(target)(x)
        return x, x

    return jax.lax.scan(step, 0.0, jax.random.split(key, n))[1]


for label, make in [
    ("Partial(stable fn, data)", lambda d: Partial(density, data=d)),
    ("fresh closure each call ", lambda d: Partial(lambda t: density(t, d))),
]:
    counts = []
    for i in range(3):
        _stats.clear()
        runner(make(jnp.arange(5.0) + i), jax.random.key(i), 200).block_until_ready()
        counts.append(_stats["compiles"])
    print(label, "compiles per call:", counts)
