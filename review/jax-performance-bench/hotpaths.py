# ruff: noqa: E402  (the patches must apply before probpipe is imported)
"""Count the compilations of warm calls on paths beyond the sampler and kernel.

Usage: uv run python review/jax-performance-bench/hotpaths.py MODE [cheap|nuts|ppc|all]
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
group = sys.argv[2] if len(sys.argv) > 2 else "all"
if mode != "baseline":
    import patches

    patches.apply(mode)
from probpipe import (
    Beta,
    LogNormal,
    Poisson,
    condition_on,
    conditional_distribution,
    function,
    log_prob,
    predictive_check,
    quantile,
    sample,
    variance,
    workflow_run,
)

n_years = 22
K = LogNormal("K", loc=jnp.log(350.0), scale=0.4)
r = LogNormal("r", jnp.log(0.4), 0.4)
phi = Beta("phi", 2.0, 2.0)
prior = (K * r * phi).with_label("prior")


def ricker(r, K, N0, years):
    def step(N, _):
        N_next = N * jnp.exp(r * (1.0 - N / K))
        return N_next, N_next

    _, N = jax.lax.scan(step, jnp.asarray(N0, dtype=jnp.float32), None, length=years)
    return N


@conditional_distribution(given_spec=prior.event_spec.spec, label="likelihood")
def likelihood(K, r, phi, n0=50.0):
    return Poisson("y", phi * ricker(r, K, n0, n_years))


model = likelihood * prior
with workflow_run(seed=0):
    counts = sample(condition_on(model, {"K": 350.0, "r": 0.4, "phi": 0.6}), 1).raw()[0]
observed = {"y": counts}
levels = jnp.array([0.05, 0.95])


@function
def capacity_count(K: jax.Array, phi: jax.Array) -> jax.Array:
    return phi * K


draws = sample(prior, 64)


def change_spread(y):
    return jnp.std(jnp.diff(y))


small = condition_on.with_options(
    method_options={"num_chains": 2, "num_results": 200, "num_warmup": 200}
)
calls = {}
if group in ("all", "cheap"):
    calls.update(
        {
            "quantile(Beta, levels)": lambda: quantile(phi, levels),
            "quantile(LogNormal, levels)": lambda: quantile(K, levels),
            "variance(phi) exact": lambda: variance(phi),
            "capacity_count(prior[K], prior[phi])": lambda: capacity_count(
                prior["K"], prior["phi"]
            ),
            "log_prob(prior, 64 draws)": lambda: log_prob(prior, draws),
            "condition_on.check(model, obs)": lambda: condition_on.check(model, observed).route,
        }
    )
if group in ("all", "nuts"):
    calls["condition_on NUTS 2x(200+200)"] = lambda: small(model, observed)
if group in ("all", "ppc"):
    post = small(model, observed)
    calls["predictive_check(lik, post, f)"] = lambda: predictive_check(
        likelihood, post, [change_spread], counts
    )
print(f"mode={mode}")
for name, call in calls.items():
    try:
        print(
            fmt(
                name,
                measure(
                    lambda call=call: (call(), jnp.zeros(()))[1], n=3 if "NUTS" not in name else 2
                ),
            ),
            flush=True,
        )
    except Exception as e:
        print(f"{name}: ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)
