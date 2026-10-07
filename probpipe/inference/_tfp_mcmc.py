"""The TFP-backed inference method, NUTS."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.mcmc as tfp_mcmc

from ..core._dispatch import Feasibility
from ..custom_types import Array
from ..distributions._capabilities import SupportsUnnormalizedLogProb
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution
from ..operations._condition import InferenceMethod
from ._approximate_distribution import make_posterior
from ._inference_utils import (
    as_prng_key,
    build_mcmc_datatree,
    build_target_log_prob,
    build_target_log_prob_flat,
    extract_event_spec,
    flat_record,
    get_init_state,
    is_jax_traceable,
    observed_parts,
    run_seed,
    unconstrained_chain,
)


def _run_tfp_chains(
    target_log_prob_fn: Callable,
    init_state: jnp.ndarray,
    *,
    algorithm: str,
    num_results: int,
    num_warmup: int,
    num_chains: int,
    step_size: float,
    target_accept_prob: float,
    random_seed: int,
) -> tuple[list[Array], dict[str, np.ndarray]]:
    """Run TFP-backed MCMC chains.

    Returns (chains, sample_stats_dict) where sample_stats_dict contains
    arrays shaped (num_chains, num_results) for building DataTree.
    """
    if algorithm != "nuts":
        raise ValueError(f"algorithm must be 'nuts', got {algorithm!r}")
    inner_kernel = tfp_mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
    )

    num_adapt = int(0.8 * num_warmup) if num_warmup > 0 else 0
    if num_adapt > 0:
        kernel = tfp_mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner_kernel,
            num_adaptation_steps=num_adapt,
            target_accept_prob=target_accept_prob,
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

    has_divergence = getattr(results, "has_divergence", None)
    if has_divergence is not None:
        stats["diverging"] = np.asarray(has_divergence)

    return stats


def _chain_target(
    model: Any, observed: Any, *, init: Any, random_seed: int
) -> tuple[Callable[[Array], Array], Array, Any]:
    """The log-density at a chain state, the initial state, and the posterior's declaration.

    TFP runs on the flat state :func:`get_init_state` gives. A target that
    declares a numeric record it has no flat view of is scored at that state
    unflattened, and any other at the state as it is.
    """
    if flat_record(model) is not None:
        return build_target_log_prob_flat(model, observed, init=init, random_seed=random_seed)
    return (
        build_target_log_prob(model, observed),
        get_init_state(model, init, random_seed=random_seed),
        extract_event_spec(model),
    )


# ---------------------------------------------------------------------------
# Inference methods
# ---------------------------------------------------------------------------


class _TFPGradientMethod(InferenceMethod):
    """Base for TFP gradient-based MCMC methods (NUTS, HMC)."""

    _method_options = (
        "init",
        "num_chains",
        "num_results",
        "num_warmup",
        "step_size",
        "target_accept_prob",
    )

    def __init__(self, algorithm: str, method_name: str, method_priority: int | None):
        self._algorithm = algorithm
        self._method_name = method_name
        self._method_priority = method_priority

    @property
    def name(self) -> str:
        return self._method_name

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    @property
    def priority(self) -> int | None:
        return self._method_priority

    def check(self, target: Any, /, **kwargs: Any) -> Feasibility:
        """Whether the target has an unnormalized density that JAX traces at its initial state."""
        # Intentionally probes JAX traceability (via jax.make_jaxpr) to avoid
        # selecting a gradient-based method that would fail at execute() time.
        # The cost is ~one JAX trace, cached by JAX on subsequent calls.
        model, observed = observed_parts(target)
        if not isinstance(model, SupportsUnnormalizedLogProb):
            return Feasibility(
                feasible=False,
                description="Requires SupportsUnnormalizedLogProb",
            )
        try:
            density, init, _ = _chain_target(
                model, observed, init=kwargs.get("init"), random_seed=0
            )
            density, init, _ = unconstrained_chain(density, init, model)
            if not is_jax_traceable(density, init):
                return Feasibility(
                    feasible=False,
                    description="Log-prob is not JAX-traceable",
                )
        except Exception as e:
            return Feasibility(feasible=False, description=str(e))
        return Feasibility(feasible=True)

    def execute(self, target: Any, /, **kwargs: Any) -> EmpiricalDistribution:
        """Chains of the TFP kernel on the target's unnormalized density.

        Raises
        ------
        ValueError
            If ``target_accept_prob`` is not strictly between 0 and 1.
        """
        self._check_options(kwargs)
        target_accept_prob = kwargs.get("target_accept_prob", 0.75)
        if not 0.0 < target_accept_prob < 1.0:
            raise ValueError(
                f"target_accept_prob must be strictly between 0 and 1, got {target_accept_prob!r}"
            )
        random_seed = run_seed(self.name)
        model, observed = observed_parts(target)
        density, init, event_spec = _chain_target(
            model, observed, init=kwargs.get("init"), random_seed=random_seed
        )
        density, init, constrain = unconstrained_chain(density, init, model)

        num_results = kwargs.get("num_results", 1000)
        num_warmup = kwargs.get("num_warmup", 500)
        num_chains = kwargs.get("num_chains", 4)

        chains, sample_stats = _run_tfp_chains(
            density,
            init,
            algorithm=self._algorithm,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
            step_size=kwargs.get("step_size", 0.1),
            target_accept_prob=target_accept_prob,
            random_seed=random_seed,
        )
        chains = [constrain(chain) for chain in chains]
        annotations = build_mcmc_datatree(chains, sample_stats)
        return make_posterior(
            chains,
            parents=(target,),
            method=self._method_name,
            annotations=annotations,
            event_spec=event_spec,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
        )


def TFPNutsMethod() -> _TFPGradientMethod:
    """TFP No-U-Turn Sampler, registered as ``tfp_nuts``, opt-in-only.

    Runs only when the caller pins ``method="tfp_nuts"``; ``blackjax_nuts``
    is what automatic selection picks for the same targets.

    Its ``method_options`` are the draw, warmup, and chain counts, ``init``,
    the initial ``step_size``, and ``target_accept_prob``, the
    acceptance probability that warmup's step-size adaptation targets, 0.75
    unless set. A higher target adapts a smaller step, which removes the
    divergent transitions of a posterior with regions of high curvature at the
    cost of longer trajectories.

    Notes
    -----
    Kept for bit-pattern regression and side-by-side backend comparison.
    """
    return _TFPGradientMethod("nuts", "tfp_nuts", None)
