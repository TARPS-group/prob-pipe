"""TFP-backed inference methods: NUTS and HMC."""

from __future__ import annotations

from typing import Any, Callable

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.mcmc as tfp_mcmc

from ..core._registry import MethodInfo
from ..core.distribution import Distribution
from ..core.protocols import SupportsLogProb, SupportsMean
from ..custom_types import Array, ArrayLike
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._registry import InferenceMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_jax_traceable(fn: Callable, init_state: jnp.ndarray) -> bool:
    """Probe whether *fn* can be traced by JAX."""
    try:
        jax.make_jaxpr(fn)(init_state)
        return True
    except Exception:
        return False


def _get_init_state(
    dist: Any, init: ArrayLike | None, data: ArrayLike | None,
) -> jnp.ndarray:
    """Determine an initial chain state from the distribution or data."""
    if init is not None:
        return jnp.atleast_1d(jnp.asarray(init, dtype=jnp.float32))

    if isinstance(dist, SupportsMean):
        try:
            m = dist._mean()
            return jnp.atleast_1d(jnp.asarray(m, dtype=jnp.float32))
        except Exception:
            pass

    if data is not None:
        return jnp.atleast_1d(jnp.mean(jnp.asarray(data), axis=0))

    if hasattr(dist, "event_shape"):
        return jnp.zeros(dist.event_shape, dtype=jnp.float32)

    raise ValueError(
        "Cannot determine initial state: provide init=, "
        "a distribution with SupportsMean, or observed data."
    )


def _is_simple_model(dist: Any) -> bool:
    """Check if *dist* is a SimpleModel (lazy import to avoid circularity)."""
    from ..modeling._simple import SimpleModel
    return isinstance(dist, SimpleModel)


def _build_target_log_prob(dist: Any, observed: Any) -> Callable[[jnp.ndarray], Array]:
    """Build a target_log_prob_fn(params) from *dist* and *observed*.

    Handles three cases:

    1. **SimpleModel** (has prior + likelihood):
       ``prior._log_prob(params) + likelihood.log_likelihood(params, data)``
    2. **Bare SupportsLogProb with data**: ``dist._log_prob(params)``
       (data is assumed already folded in, e.g., via closure).
    3. **Joint over (params, data)**: ``dist._log_prob((params, data))``
    """
    # SimpleModel: separate prior + likelihood
    if _is_simple_model(dist):
        data = jnp.asarray(observed) if observed is not None else None

        def target_log_prob_fn(params):
            lp = dist._prior._log_prob(params)
            if data is not None:
                lp = lp + dist._likelihood.log_likelihood(params=params, data=data)
            return lp

        return target_log_prob_fn

    # Joint _log_prob((params, data))
    if observed is not None:
        data = jnp.asarray(observed)
        return lambda params: dist._log_prob((params, data))

    # Bare SupportsLogProb (no data)
    return dist._log_prob


def _run_tfp_chains(
    target_log_prob_fn: Callable,
    init_state: jnp.ndarray,
    *,
    algorithm: str,
    num_results: int,
    num_warmup: int,
    num_chains: int,
    step_size: float,
    random_seed: int,
) -> tuple[list[Array], dict[str, Any]]:
    """Run TFP-backed MCMC chains.

    Returns (chains, sample_stats_dict) where sample_stats_dict contains
    arrays shaped (num_chains, num_results) for building InferenceData.
    """
    if algorithm == "nuts":
        inner_kernel = tfp_mcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
        )
    elif algorithm == "hmc":
        inner_kernel = tfp_mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=10,
        )
    else:
        raise ValueError(f"algorithm must be 'nuts' or 'hmc', got {algorithm!r}")

    num_adapt = int(0.8 * num_warmup) if num_warmup > 0 else 0
    if num_adapt > 0:
        kernel = tfp_mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner_kernel,
            num_adaptation_steps=num_adapt,
            target_accept_prob=0.75,
        )
    else:
        kernel = inner_kernel

    key = jax.random.PRNGKey(random_seed)
    chain_keys = jax.random.split(key, num_chains)

    def _run_one_chain(chain_key):
        return tfp_mcmc.sample_chain(
            num_results=num_results,
            current_state=init_state,
            kernel=kernel,
            num_burnin_steps=num_warmup,
            seed=chain_key,
            trace_fn=lambda _, kr: kr,
        )

    all_samples, all_traces = jax.vmap(_run_one_chain)(chain_keys)
    # all_samples: (num_chains, num_results, *event_shape)
    chains = [all_samples[c] for c in range(num_chains)]

    # Extract sample_stats from traces for InferenceData
    sample_stats = _extract_sample_stats(all_traces, num_chains)
    return chains, sample_stats


def _extract_sample_stats(traces: Any, num_chains: int) -> dict[str, Any]:
    """Extract sample stats arrays from TFP traces.

    Returns dict of numpy arrays shaped (num_chains, num_draws).
    """
    results = traces
    stats: dict[str, Any] = {}

    if hasattr(results, "new_step_size"):
        stats["step_size"] = np.asarray(results.new_step_size)
        results = results.inner_results
    elif hasattr(results, "step_size"):
        stats["step_size"] = np.asarray(results.step_size)

    log_ar = getattr(results, "log_accept_ratio", None)
    if log_ar is not None:
        ar = np.asarray(jnp.exp(jnp.minimum(log_ar, 0.0)))
        stats["acceptance_rate"] = ar

    is_accepted = getattr(results, "is_accepted", None)
    if is_accepted is not None:
        stats["is_accepted"] = np.asarray(is_accepted)

    return stats


def _build_tfp_inference_data(
    chains: list[Array],
    sample_stats: dict[str, Any],
):
    """Build an ArviZ ``DataTree`` from TFP chains and sample stats."""
    # Stack chains: (num_chains, num_draws, *event_shape)
    posterior_array = np.stack([np.asarray(c) for c in chains], axis=0)
    groups: dict[str, Any] = {"posterior": {"params": posterior_array}}
    if sample_stats:
        groups["sample_stats"] = sample_stats
    return az.from_dict(groups)


# ---------------------------------------------------------------------------
# Helpers for prior extraction
# ---------------------------------------------------------------------------

def _get_prior(dist: Distribution) -> Distribution:
    """Extract the prior from a model, or return dist itself."""
    return dist._prior if _is_simple_model(dist) else dist


# ---------------------------------------------------------------------------
# Inference methods
# ---------------------------------------------------------------------------


class _TFPGradientMethod(InferenceMethod):
    """Base for TFP gradient-based MCMC methods (NUTS, HMC)."""

    def __init__(self, algorithm: str, method_name: str, method_priority: int):
        self._algorithm = algorithm
        self._method_name = method_name
        self._method_priority = method_priority

    @property
    def name(self) -> str:
        return self._method_name

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    @property
    def priority(self) -> int:
        return self._method_priority

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        # Intentionally probes JAX traceability (via jax.make_jaxpr) to avoid
        # selecting a gradient-based method that would fail at execute() time.
        # The cost is ~one JAX trace, cached by JAX on subsequent calls.
        if not isinstance(dist, SupportsLogProb):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires SupportsLogProb")
        try:
            target = _build_target_log_prob(dist, observed)
            init = _get_init_state(_get_prior(dist), kwargs.get("init"), observed)
            if not _is_jax_traceable(target, init):
                return MethodInfo(feasible=False, method_name=self.name,
                                  description="Log-prob is not JAX-traceable")
        except Exception as e:
            return MethodInfo(feasible=False, method_name=self.name,
                              description=str(e))
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        target = _build_target_log_prob(dist, observed)
        prior = _get_prior(dist)
        init = _get_init_state(prior, kwargs.get("init"), observed)

        num_results = kwargs.get("num_results", 1000)
        num_warmup = kwargs.get("num_warmup", 500)
        num_chains = kwargs.get("num_chains", 1)

        chains, sample_stats = _run_tfp_chains(
            target, init,
            algorithm=self._algorithm,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
            step_size=kwargs.get("step_size", 0.1),
            random_seed=kwargs.get("random_seed", 0),
        )
        inference_data = _build_tfp_inference_data(chains, sample_stats)
        return make_posterior(
            chains, parents=(prior,), algorithm=self._method_name,
            inference_data=inference_data,
            num_results=num_results, num_warmup=num_warmup, num_chains=num_chains,
        )


def TFPNutsMethod() -> _TFPGradientMethod:
    """TFP No-U-Turn Sampler (gradient-based MCMC)."""
    return _TFPGradientMethod("nuts", "tfp_nuts", 100)


def TFPHmcMethod() -> _TFPGradientMethod:
    """TFP Hamiltonian Monte Carlo."""
    return _TFPGradientMethod("hmc", "tfp_hmc", 90)
