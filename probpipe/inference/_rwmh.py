"""Random-walk Metropolis-Hastings as a standalone workflow function."""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp

from ..core.distribution import Provenance
from ..core.node import WorkflowFunction
from ..core.protocols import SupportsLogProb, SupportsMean
from ..custom_types import Array, ArrayLike, PRNGKey
from ._diagnostics import MCMCDiagnostics
from ._mcmc_distribution import MCMCApproximateDistribution

logger = logging.getLogger(__name__)

__all__ = ["rwmh"]


def _rwmh_impl(
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
        Distribution (or model) providing ``_log_prob`` or
        ``_unnormalized_log_prob``.  When *data* is provided, the
        target log-density is ``dist._log_prob(params) + log_prob_fn(params, data)``.
    data : array-like or None
        Observed data.  If provided, *log_prob_fn* must also be given
        (or *dist* must be a model whose ``_log_prob`` already
        incorporates the likelihood).
    log_prob_fn : callable or None
        ``log_prob_fn(params, data) -> float``.  Combined with
        ``dist._log_prob(params)`` to form the target density.
        If ``None``, the target is just ``dist._log_prob(params)``.
    num_results : int
        Number of post-warmup samples per chain (default 1000).
    num_warmup : int
        Number of warmup (burn-in) steps (default 500).
    num_chains : int
        Number of independent chains (default 1).
    step_size : float
        Random-walk proposal scale (default 0.1).
    init : array-like or None
        Initial chain state.  If ``None``, tries ``dist._mean()``,
        then zeros.
    random_seed : int
        Random seed (default 0).

    Returns
    -------
    MCMCApproximateDistribution
        Posterior samples with chain structure and diagnostics.
    """
    if not isinstance(dist, SupportsLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support log_prob "
            f"(does not implement SupportsLogProb)"
        )

    # Build target log-density
    if log_prob_fn is not None and data is not None:
        data_jnp = jnp.asarray(data)

        def target_log_prob(params):
            return dist._log_prob(params) + log_prob_fn(params, data_jnp)
    else:

        def target_log_prob(params):
            return dist._log_prob(params)

    # Determine initial state
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

    diagnostics = MCMCDiagnostics(
        log_accept_ratio=jnp.zeros(num_results * num_chains),
        step_size=step_size,
        is_accepted=None,
        algorithm="rwmh",
    )
    diagnostics._numpy_accept_rate = accept_rate

    # Only include warmup if we actually have warmup samples
    warmup = warmup_chains if all(w is not None for w in warmup_chains) else None

    result = MCMCApproximateDistribution(
        chains,
        diagnostics=diagnostics,
        warmup_samples=warmup,
        name="posterior",
    )
    result.with_source(
        Provenance(
            "rwmh",
            parents=(dist,),
            metadata={
                "num_results": num_results,
                "num_warmup": num_warmup,
                "num_chains": num_chains,
                "step_size": step_size,
                "accept_rate": accept_rate,
            },
        )
    )
    return result


rwmh = WorkflowFunction(func=_rwmh_impl, name="rwmh")
