# ruff: noqa: E402  (the patches must apply before probpipe is imported)
"""User-side choices: a Python loop or lax.scan, a user jit or not, and a model rebuilt per call.

Usage: uv run python review/jax-performance-bench/userstyle.py MODE YEARS
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp
from compilecount import fmt, measure

mode, years = sys.argv[1], int(sys.argv[2])
if mode != "baseline":
    import patches

    patches.apply(mode)
from probpipe import Beta, LogNormal, Poisson, conditional_distribution, sample

prior = (
    LogNormal("K", jnp.log(350.0), 0.4) * LogNormal("r", jnp.log(0.4), 0.4) * Beta("phi", 2.0, 2.0)
)


def ricker_scan(r, K, n0, years):
    def step(N, _):
        N = N * jnp.exp(r * (1.0 - N / K))
        return N, N

    return jax.lax.scan(step, jnp.asarray(n0, jnp.float32), None, length=years)[1]


def ricker_loop(r, K, n0, years):
    N, out = jnp.asarray(n0, jnp.float32), []
    for _ in range(years):
        N = N * jnp.exp(r * (1.0 - N / K))
        out.append(N)
    return jnp.stack(out)


ricker_scan_jit = jax.jit(ricker_scan, static_argnames="years")


def model_with(traj):
    @conditional_distribution(given_spec=prior.event_spec.spec)
    def lik(K, r, phi):
        return Poisson("y", phi * traj(r, K, 50.0, years))

    return lik * prior


models = {
    name: model_with(f)
    for name, f in [
        ("scan", ricker_scan),
        ("python loop", ricker_loop),
        ("scan, user jit", ricker_scan_jit),
    ]
}
print(f"mode={mode} years={years}")
for name, m in models.items():
    print(fmt(f"sample(model[{name}], 100)", measure(lambda m=m: sample(m, 100))), flush=True)
print(
    fmt("model rebuilt every call (scan)", measure(lambda: sample(model_with(ricker_scan), 100))),
    flush=True,
)
