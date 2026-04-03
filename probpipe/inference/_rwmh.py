"""Random-walk Metropolis-Hastings: standalone function + registry method."""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp

from ..core._registry import MethodInfo
from ..core.distribution import Distribution
from ..core.provenance import Provenance
from ..core.node import workflow_function
from ..core.protocols import SupportsLogProb, SupportsMean
from ..custom_types import Array, ArrayLike, PRNGKey
from ._diagnostics import InferenceDiagnostics
from ._mcmc_distribution import MCMCApproximateDistribution
from ._registry import InferenceMethod
from ._tfp_mcmc import _get_init_state, _get_prior

logger = logging.getLogger(__name__)

__all__ = ["rwmh", "TFPRWMHMethod"]


# ---------------------------------------------------------------------------
# Standalone WorkflowFunction
# ---------------------------------------------------------------------------


@workflow_function
def rwmh(
    dist: SupportsLogProb,
    data: ArrayLike | None = None,
    *,
    log_prob_fn: Any | None = None,
    num_results: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 1,
    step_size: float = 0.1,
    init: ArrayLike | None = None,
    random_seed: int = 0,
) -> MCMCApproximateDistribution:
    """Gradient-free random-walk Metropolis-Hastings.

    Parameters
    ----------
    dist : SupportsLogProb
        Distribution providing ``_log_prob``.
    data : array-like or None
        Observed data.
    log_prob_fn : callable or None
        ``log_prob_fn(params, data) -> float``.  Combined with
        ``dist._log_prob(params)`` to form the target density.
    num_results, num_warmup, num_chains, step_size, random_seed
        MCMC tuning parameters.
    init : array-like or None
        Initial chain state.  Tries ``dist._mean()``, then zeros.

    Returns
    -------
    MCMCApproximateDistribution
    """
    if not isinstance(dist, SupportsLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support log_prob "
            f"(does not implement SupportsLogProb)"
        )

    if log_prob_fn is not None and data is not None:
        data_jnp = jnp.asarray(data)
        def target_log_prob(params):
            return dist._log_prob(params) + log_prob_fn(params, data_jnp)
    else:
        def target_log_prob(params):
            return dist._log_prob(params)

    if init is not None:
        init_state = jnp.atleast_1d(jnp.asarray(init, dtype=jnp.float32))
    elif isinstance(dist, SupportsMean):
        try:
            init_state = jnp.atleast_1d(jnp.asarray(dist._mean(), dtype=jnp.float32))
        except Exception:
            init_state = jnp.zeros(dist.event_shape, dtype=jnp.float32)
    else:
        init_state = jnp.zeros(dist.event_shape, dtype=jnp.float32)

    d = init_state.shape[0]
    key = jax.random.PRNGKey(random_seed)

    chains = []
    warmup_chains = []
    total_accepts = 0
    total_steps = 0

    for _ in range(num_chains):
        key, chain_key = jax.random.split(key)
        mu_curr = jnp.array(init_state)
        logp_curr = float(target_log_prob(mu_curr))

        warmup_samples: list[jnp.ndarray] = []
        kept: list[jnp.ndarray] = []
        chain_accepts = 0
        chain_total = num_warmup + num_results

        for t in range(chain_total):
            chain_key, subkey_prop, subkey_accept = jax.random.split(chain_key, 3)
            noise = jax.random.normal(subkey_prop, shape=(d,), dtype=mu_curr.dtype)
            mu_prop = mu_curr + step_size * noise
            logp_prop = float(target_log_prob(mu_prop))

            u = jax.random.uniform(subkey_accept, dtype=mu_curr.dtype)
            if jnp.log(u) < min(0.0, logp_prop - logp_curr):
                mu_curr = mu_prop
                logp_curr = logp_prop
                chain_accepts += 1

            if t < num_warmup:
                warmup_samples.append(mu_curr)
            else:
                kept.append(mu_curr)

        chains.append(jnp.stack(kept))
        warmup_chains.append(jnp.stack(warmup_samples) if warmup_samples else None)
        total_accepts += chain_accepts
        total_steps += chain_total

    accept_rate = total_accepts / total_steps

    diagnostics = InferenceDiagnostics(
        algorithm="rwmh",
        log_accept_ratio=jnp.zeros(num_results * num_chains),
        step_size=step_size,
    )
    diagnostics["_accept_rate_override"] = accept_rate

    warmup = warmup_chains if all(w is not None for w in warmup_chains) else None

    result = MCMCApproximateDistribution(
        chains, diagnostics=diagnostics, warmup_samples=warmup, name="posterior",
    )
    result.with_source(Provenance(
        "rwmh", parents=(dist,),
        metadata={"num_results": num_results, "num_warmup": num_warmup,
                  "num_chains": num_chains, "step_size": step_size,
                  "accept_rate": accept_rate},
    ))
    return result


# ---------------------------------------------------------------------------
# Registry method
# ---------------------------------------------------------------------------


class TFPRWMHMethod(InferenceMethod):
    """Registry method for gradient-free RWMH.

    Feasible when the distribution (or its prior) supports
    ``SupportsLogProb``.
    """

    @property
    def name(self) -> str:
        return "tfp_rwmh"

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    @property
    def priority(self) -> int:
        return 50

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        prior = _get_prior(dist)
        if not isinstance(prior, SupportsLogProb):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires SupportsLogProb")
        if observed is not None and isinstance(observed, dict):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Does not support dict-based conditioning")
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> MCMCApproximateDistribution:
        prior = _get_prior(dist)
        log_prob_fn = None
        if hasattr(dist, "_likelihood"):
            lik = dist._likelihood
            log_prob_fn = lambda params, d: lik.log_likelihood(params=params, data=d)

        init = kwargs.get("init")
        if init is None:
            init = _get_init_state(prior, None, observed)

        return rwmh._func(
            prior, observed,
            log_prob_fn=log_prob_fn,
            num_results=kwargs.get("num_results", 1000),
            num_warmup=kwargs.get("num_warmup", 500),
            num_chains=kwargs.get("num_chains", 1),
            step_size=kwargs.get("step_size", 0.1),
            init=init,
            random_seed=kwargs.get("random_seed", 0),
        )
