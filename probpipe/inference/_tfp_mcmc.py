"""TFP-backed inference methods: NUTS and HMC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from xarray import DataTree

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.mcmc as tfp_mcmc

from ..core._registry import MethodInfo
from ..core.distribution import Distribution
from ..core.protocols import SupportsUnnormalizedLogProb
from ..custom_types import Array
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._inference_utils import (
    as_prng_key,
    build_mcmc_datatree,
    build_target_log_prob,
    extract_event_template,
    get_init_state,
    get_prior,
    is_jax_traceable,
)
from ._registry import InferenceMethod


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
) -> tuple[list[Array], dict[str, np.ndarray]]:
    """Run TFP-backed MCMC chains.

    Returns (chains, sample_stats_dict) where sample_stats_dict contains
    arrays shaped (num_chains, num_results) for building DataTree.
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

    chain_keys = jax.random.split(as_prng_key(random_seed), num_chains)

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

    sample_stats = _extract_sample_stats(all_traces, num_chains)
    return chains, sample_stats


def _extract_sample_stats(traces: Any, num_chains: int) -> dict[str, np.ndarray]:
    """Extract sample stats arrays from TFP traces.

    Returns dict of numpy arrays shaped (num_chains, num_draws).
    """
    results = traces
    stats: dict[str, np.ndarray] = {}

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
        if not isinstance(dist, SupportsUnnormalizedLogProb):
            return MethodInfo(feasible=False, method_name=self.name,
                              description="Requires SupportsUnnormalizedLogProb")
        try:
            target = build_target_log_prob(dist, observed)
            init = get_init_state(
                dist, kwargs.get("init"),
                random_seed=kwargs.get("random_seed", 0),
            )
            if not is_jax_traceable(target, init):
                return MethodInfo(feasible=False, method_name=self.name,
                                  description="Log-prob is not JAX-traceable")
        except Exception as e:
            return MethodInfo(feasible=False, method_name=self.name,
                              description=str(e))
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        random_seed = kwargs.get("random_seed", 0)
        target = build_target_log_prob(dist, observed)
        prior = get_prior(dist)
        init = get_init_state(
            dist, kwargs.get("init"), random_seed=random_seed,
        )
        event_template = extract_event_template(dist)

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
            random_seed=random_seed,
        )
        auxiliary = build_mcmc_datatree(chains, sample_stats)
        return make_posterior(
            chains, parents=(prior,), algorithm=self._method_name,
            auxiliary=auxiliary, event_template=event_template,
            num_results=num_results, num_warmup=num_warmup, num_chains=num_chains,
        )


def TFPNutsMethod() -> _TFPGradientMethod:
    """TFP No-U-Turn Sampler (gradient-based MCMC).

    Opt-in only (``priority=0``); reach via ``method="tfp_nuts"``.
    ``blackjax_nuts`` is the auto-dispatch default for any
    ``SupportsLogProb`` + JAX-traceable target. This method stays
    available for bit-pattern regression or side-by-side backend
    comparison.
    """
    return _TFPGradientMethod("nuts", "tfp_nuts", 0)


def TFPHmcMethod() -> _TFPGradientMethod:
    """TFP Hamiltonian Monte Carlo.

    Opt-in only (``priority=0``); reach via ``method="tfp_hmc"``.
    Both HMC kernels (``tfp_hmc`` and ``blackjax_hmc``) sit at
    ``priority=0`` — they share their respective NUTS sibling's
    ``check()`` and so are structurally unreachable in auto-dispatch.
    """
    return _TFPGradientMethod("hmc", "tfp_hmc", 0)
