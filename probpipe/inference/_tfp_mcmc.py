"""The TFP-backed inference method, NUTS."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.mcmc as tfp_mcmc

from ..core._dispatch import Feasibility
from ..custom_types import Array, PRNGKey
from ..distributions._capabilities import SupportsUnnormalizedLogProb
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution
from ..operations._condition import InferenceMethod
from ..operations._inference_target import InferenceTarget, _flat_record_spec
from ._approximate_distribution import make_posterior
from ._inference_utils import (
    UNTRACEABLE_DENSITY,
    as_prng_key,
    build_mcmc_datatree,
    build_target_log_prob,
    build_target_log_prob_flat,
    chain_initial_states,
    get_init_state,
    no_density_reason,
    nuts_controls,
    observed_parts,
    traces_in_jax,
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
    max_tree_depth: int = 10,
    random_seed: int | PRNGKey,
    per_chain_init: bool = False,
) -> tuple[list[Array], dict[str, np.ndarray]]:
    """Run TFP-backed MCMC chains.

    Returns (chains, sample_stats_dict) where sample_stats_dict contains
    arrays shaped (num_chains, num_results) for building DataTree.
    ``init_state`` is one state, or one per chain on its leading axis when
    ``per_chain_init`` is set.
    """
    if algorithm != "nuts":
        raise ValueError(f"algorithm must be 'nuts'; got {algorithm!r}")
    inner_kernel = tfp_mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        max_tree_depth=max_tree_depth,
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

    def _run_one_chain(chain_key, chain_init):
        return tfp_mcmc.sample_chain(
            num_results=num_results,
            current_state=chain_init,
            kernel=kernel,
            num_burnin_steps=num_warmup,
            seed=chain_key,
            trace_fn=lambda _, kr: kr,
        )

    init_axis = 0 if per_chain_init else None
    all_samples, all_traces = jax.vmap(_run_one_chain, in_axes=(0, init_axis))(
        chain_keys, init_state
    )
    # all_samples: (num_chains, num_results, *event_shape)
    chains = [all_samples[c] for c in range(num_chains)]

    sample_stats = _extract_sample_stats(all_traces, num_chains)
    return chains, sample_stats


def _extract_sample_stats(traces: Any, num_chains: int) -> dict[str, np.ndarray]:
    """Extract sample stats arrays from TFP traces.

    Returns dict of numpy arrays shaped (num_chains, num_draws). The NUTS
    kernel's ``leapfrogs_taken`` is recorded as ``num_integration_steps``, the
    name ``blackjax_nuts`` records it under, the depth its tree reached,
    ``ceil(log2(leapfrogs_taken + 1))``, as ArviZ's ``tree_depth``, and its
    ``reach_max_depth`` as ArviZ's ``reached_max_treedepth``, which marks each
    draw whose trajectory reached the maximum depth without turning.
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

    leapfrogs = getattr(results, "leapfrogs_taken", None)
    if leapfrogs is not None:
        steps = np.asarray(leapfrogs).astype(np.int64)
        stats["num_integration_steps"] = steps
        stats["tree_depth"] = np.ceil(np.log2(steps + 1)).astype(np.int64)

    reach_max_depth = getattr(results, "reach_max_depth", None)
    if reach_max_depth is not None:
        stats["reached_max_treedepth"] = np.asarray(reach_max_depth)

    return stats


def _chain_target(
    model: Any, observed: Any, *, init: Any, random_seed: int | PRNGKey
) -> tuple[Callable[[Array], Array], Array]:
    """The log-density at a chain state, and the initial state.

    TFP runs on the flat state :func:`get_init_state` gives. A target that
    declares a numeric record it has no flat view of is scored at that state
    unflattened, and any other at the state as it is.
    """
    if _flat_record_spec(model) is not None:
        return build_target_log_prob_flat(model, observed, init=init, random_seed=random_seed)
    return (
        build_target_log_prob(model, observed),
        get_init_state(model, init, random_seed=random_seed),
    )


# ---------------------------------------------------------------------------
# Inference methods
# ---------------------------------------------------------------------------


class _TFPGradientMethod(InferenceMethod):
    """Base for TFP gradient-based MCMC methods (NUTS, HMC)."""

    _method_options = (
        "init",
        "max_tree_depth",
        "num_chains",
        "num_results",
        "num_warmup",
        "step_size",
        "target_accept_prob",
    )

    def __init__(
        self,
        algorithm: str,
        method_name: str,
        method_priority: int | None,
        method_description: str,
    ):
        self._algorithm = algorithm
        self._method_name = method_name
        self._method_priority = method_priority
        self._method_description = method_description

    @property
    def name(self) -> str:
        return self._method_name

    @property
    def description(self) -> str:
        return self._method_description

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    @property
    def priority(self) -> int | None:
        return self._method_priority

    def check(self, target: Any, /, **kwargs: Any) -> Feasibility:
        """Whether the target has an unnormalized density that JAX traces."""
        model, observed = observed_parts(target)
        if not isinstance(model, SupportsUnnormalizedLogProb):
            return Feasibility(feasible=False, description=no_density_reason(model))
        if not traces_in_jax(model):
            return Feasibility(feasible=False, description=UNTRACEABLE_DENSITY)
        init = kwargs.get("init")
        if isinstance(init, str) or np.ndim(init) >= 2:
            # A starting point for each chain is resolved and checked when the method runs.
            init = None
        try:
            density, init = _chain_target(model, observed, init=init, random_seed=0)
            unconstrained_chain(density, init, model)
        except Exception as e:
            return Feasibility(feasible=False, description=str(e))
        return Feasibility(feasible=True)

    def execute(
        self, target: InferenceTarget, /, *, key: PRNGKey, **options: Any
    ) -> EmpiricalDistribution:
        """Chains of the TFP kernel on the target's unnormalized density.

        Parameters
        ----------
        target : InferenceTarget
            The law to sample: a law with an unnormalized density, or an
            unnormalized conditional at data its joint does not declare as fields.
        key : PRNGKey
            The run's random key.
        **options : Any
            The call's ``method_options``, which :func:`TFPNutsMethod` lists.

        Returns
        -------
        EmpiricalDistribution
            The posterior, with its atoms on the levels ``chain`` and ``draw``.

        Raises
        ------
        TypeError
            If an option is one the method does not read, or
            ``target_accept_prob`` or ``max_tree_depth`` has the wrong type.
        ValueError
            If ``target_accept_prob`` is not strictly between 0 and 1,
            ``max_tree_depth`` is less than 1, or ``init`` is neither a state,
            one state per chain, nor ``"per_chain"``.
        """
        self._check_options(options)
        target_accept_prob, max_tree_depth = nuts_controls(
            self._method_name, options, default_target_accept_prob=0.75
        )
        num_results = options.get("num_results", 1000)
        num_warmup = options.get("num_warmup", 500)
        num_chains = options.get("num_chains", 4)
        model, observed = observed_parts(target)
        init, per_chain_init = chain_initial_states(
            model,
            options.get("init"),
            num_chains=num_chains,
            key=key,
            method=self._method_name,
            default_per_chain=True,
        )
        density, init = _chain_target(model, observed, init=init, random_seed=key)
        density, init, constrain = unconstrained_chain(
            density, init, model, per_chain=per_chain_init
        )

        chains, sample_stats = _run_tfp_chains(
            density,
            init,
            algorithm=self._algorithm,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
            step_size=options.get("step_size", 0.1),
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            random_seed=key,
            per_chain_init=per_chain_init,
        )
        chains = [constrain(chain) for chain in chains]
        annotations = build_mcmc_datatree(chains, sample_stats)
        return make_posterior(
            target,
            chains,
            method=self._method_name,
            annotations=annotations,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
        )


def TFPNutsMethod() -> _TFPGradientMethod:
    """TFP No-U-Turn Sampler, registered as ``tfp_nuts``, opt-in-only.

    Runs only when the caller pins ``method="tfp_nuts"``; ``blackjax_nuts``
    is what automatic selection picks for the same targets.

    Its ``method_options`` are:

    - ``num_results``, ``num_warmup``, and ``num_chains``: the draw, warmup,
      and chain counts, which are 1000, 500, and 4 unless set. Step-size
      adaptation runs over the first 80% of the warmup steps;
    - ``step_size``: the initial step size, which is 0.1 unless set;
    - ``target_accept_prob``: the acceptance probability that warmup's
      step-size adaptation targets, a number strictly between 0 and 1, which
      is 0.75 unless set. A higher target adapts a smaller step, which removes
      the divergent transitions of a posterior with regions of high curvature
      at the cost of longer trajectories;
    - ``max_tree_depth``: the maximum depth of the NUTS tree, a positive
      integer, which is TFP's 10 unless set. A trajectory takes at most
      ``2**max_tree_depth - 1`` leapfrog steps;
    - ``init``: where the chains start, in the four forms that
      ``BlackJAXNutsMethod`` states for ``blackjax_nuts``.

    The run's sample statistics hold ``num_integration_steps``, the leapfrog
    steps of each draw, and, under ArviZ's names, ``tree_depth``, the depth
    each draw's tree reached, and ``reached_max_treedepth``, which marks each
    draw whose trajectory reached ``max_tree_depth`` without turning. They hold
    no ``energy``, so ``add_mcmc_diagnostics`` computes no E-BFMI for this
    method.

    Notes
    -----
    Kept for bit-pattern regression and side-by-side backend comparison.
    """
    return _TFPGradientMethod(
        "nuts",
        "tfp_nuts",
        None,
        "NUTS through TensorFlow Probability, for a JAX-traceable log-density.",
    )
