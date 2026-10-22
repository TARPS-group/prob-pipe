"""Compare minibatch sampling with the former full-population permutation.

Run from the repository root with
``uv run python -m scripts.benchmarks.minibatch_indices --repeats 10``.
Use ``.venv-gpu/bin/python`` and ``--device gpu`` for CUDA measurements.
JSON output includes the first call and synchronized warm calls, with backend
compilation counts. Key creation, model construction, and device placement
precede timing. The first call includes tracing and compilation, but may load
an executable from JAX's persistent cache. Warm calls use distinct keys.
The gradient case uses the actual SGMCMC gradient estimator, with only index
sampling replaced for the reference measurement. No inference runner is timed.
"""

from __future__ import annotations

import argparse
import json
import platform
from functools import partial
from statistics import median
from time import perf_counter
from unittest.mock import patch

import jax
import jax.numpy as jnp

from probpipe import MultivariateNormal
from probpipe.families import BernoulliFamily, glm_likelihood
from probpipe.inference._blackjax_sgmcmc import _build_grad_estimator
from probpipe.inference._minibatch import MinibatchedDistribution, _draw_indices
from tests._compilation import count_compilations


def _permutation_indices(key, n, batch_size, *, with_replacement):
    if with_replacement:
        return jax.random.randint(key, (batch_size,), 0, n)
    return jax.random.permutation(key, n)[:batch_size]


def _measure(function, keys):
    with count_compilations() as cold_compilations:
        start = perf_counter()
        jax.block_until_ready(function(keys[0]))
        cold_ms = (perf_counter() - start) * 1000
    durations = []
    with count_compilations() as warm_compilations:
        for key in keys[1:]:
            start = perf_counter()
            jax.block_until_ready(function(key))
            durations.append((perf_counter() - start) * 1000)
    return {
        "first_call_ms": cold_ms,
        "first_call_compilations": cold_compilations.count,
        "warm_median_ms": median(durations),
        "warm_min_ms": min(durations),
        "warm_max_ms": max(durations),
        "warm_total_compilations": warm_compilations.count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    with jax.default_device(jax.devices(args.device)[0]):
        _run(args)


def _run(args):
    keys = list(jax.random.split(jax.random.key(2026), args.repeats + 1))
    jax.block_until_ready(keys)
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "jax": jax.__version__,
                "device": jax.devices(args.device)[0].device_kind,
                "x64": jax.config.jax_enable_x64,
                "repeats": args.repeats,
                "key_preparation_timed": False,
            }
        ),
        flush=True,
    )
    cases = [(n, b) for n in (1000, 10_000, 100_000, 1_000_000) for b in (32, 256)]
    cases += [
        (65536, 256),
        (66049, 257),
        (1_000_000, 4096),
        (10_000, 5000),
        (10_000, 5001),
        (1_000_000, 900_000),
        (1_000_000, 999_968),
        (1024, 1024),
    ]
    for n, size in cases:
        for name, sampler in [("permutation", _permutation_indices), ("current", _draw_indices)]:
            draw = jax.jit(partial(sampler, n=n, batch_size=size, with_replacement=False))
            result = _measure(draw, keys)
            print(
                json.dumps(
                    {"case": "indices", "n": n, "batch_size": size, "sampler": name, **result}
                ),
                flush=True,
            )

    # Mapped calls expose retry tails across lanes. Keys for all lanes are
    # prepared before timing; each timed call receives a different key batch.
    mapped_keys = [jax.random.split(key, 64) for key in keys]
    jax.block_until_ready(mapped_keys)
    for n, size in [(10_000, 256), (65536, 256)]:
        for name, sampler in [("permutation", _permutation_indices), ("current", _draw_indices)]:
            draw = jax.jit(jax.vmap(partial(sampler, n=n, batch_size=size, with_replacement=False)))
            result = _measure(draw, mapped_keys)
            print(
                json.dumps(
                    {"case": "vmap64", "n": n, "batch_size": size, "sampler": name, **result}
                ),
                flush=True,
            )

    n = 1_000_000
    x = jnp.linspace(-2, 2, n)
    design = jnp.stack([jnp.ones(n), x], axis=1)
    response = (jnp.arange(n) % 3 == 0).astype(jnp.float32)
    position = jnp.array([0.3, -0.2])
    prior = MultivariateNormal("beta", loc=jnp.zeros(2), cov=jnp.eye(2))
    likelihood = glm_likelihood("y", BernoulliFamily(), X=design)
    jax.block_until_ready((design, response, position))
    for size in (32, 256):
        measure = MinibatchedDistribution(prior, likelihood, response, size)
        estimator = _build_grad_estimator(measure)
        for name, sampler in [("permutation", _permutation_indices), ("current", _draw_indices)]:
            with patch("probpipe.inference._minibatch._draw_indices", sampler):
                gradient = jax.jit(
                    lambda position, key, estimator=estimator: estimator(position, key)
                )
                result = _measure(partial(gradient, position), keys)
                print(
                    json.dumps(
                        {"case": "gradient", "n": n, "batch_size": size, "sampler": name, **result}
                    ),
                    flush=True,
                )

                # An outer compiled loop includes key splitting and consumes
                # each gradient in a checksum. Report time for all 25 steps.
                def scan(position, key, estimator=estimator):
                    def step(state, unused):
                        key, total = state
                        key, subkey = jax.random.split(key)
                        return (key, total + estimator(position, subkey)), None

                    return jax.lax.scan(step, (key, jnp.zeros_like(position)), None, length=25)

                # Keep the full-population CPU reference to single calls.
                if args.device == "gpu" or name == "current":
                    result = _measure(partial(jax.jit(scan), position), keys)
                    print(
                        json.dumps(
                            {
                                "case": "gradient_scan",
                                "steps": 25,
                                "n": n,
                                "batch_size": size,
                                "sampler": name,
                                **result,
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
