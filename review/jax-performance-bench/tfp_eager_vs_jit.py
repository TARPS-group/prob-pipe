"""TFP sampling, eager and jitted: the time of repeated calls, and whether the draws agree.

Usage: uv run python review/jax-performance-bench/tfp_eager_vs_jit.py
"""

import time
import warnings
from functools import partial

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

warnings.filterwarnings("ignore")
tfd = tfp.distributions


@partial(jax.jit, static_argnames="sample_shape")
def jitted_sample(backend, key, sample_shape):
    return backend.sample(seed=key, sample_shape=sample_shape)


laws = {
    "LogNormal": tfd.LogNormal(jnp.log(350.0), 0.4),
    "Beta": tfd.Beta(2.0, 2.0),
    "Gamma": tfd.Gamma(2.0, 1.0),
    "Poisson": tfd.Poisson(jnp.linspace(10.0, 300.0, 22)),
}

print("seconds for calls 1, 2, 3, each with a new key")
for name, law in laws.items():
    for label, draw in [
        ("eager", lambda key, law=law: law.sample(100, seed=key)),
        ("jit", lambda key, law=law: jitted_sample(law, key, (100,))),
    ]:
        times = []
        for i in range(3):
            start = time.perf_counter()
            jax.block_until_ready(draw(jax.random.key(i)))
            times.append(time.perf_counter() - start)
        print(f"{name:10s}{label:6s}" + "".join(f"{t:10.3f}" for t in times))

print("\neager and jitted draws at one key")
for name, law in laws.items():
    key = jax.random.key(3)
    eager, jitted = law.sample(100, seed=key), jitted_sample(law, key, (100,))
    difference = float(jnp.max(jnp.abs(eager - jitted) / jnp.maximum(jnp.abs(eager), 1e-30)))
    print(
        f"{name:10s} identical={bool((eager == jitted).all())}  max relative difference={difference:.2g}"
    )
