"""BlackJAX-backed gradient MCMC methods: NUTS and HMC.

Two :class:`~probpipe.core._registry.Method` subclasses registered with
:data:`~probpipe.inference.inference_method_registry`:

* ``blackjax_nuts`` — No-U-Turn Sampler with window-adapted step size and
  mass matrix.
* ``blackjax_hmc`` — Hamiltonian Monte Carlo with window-adapted step
  size and a *randomized* trajectory length: production samples draw the
  number of leapfrog steps from a low-discrepancy Halton sequence with
  mean ``num_integration_steps`` (a user-tunable kwarg, default ``10``),
  breaking the fixed-``L`` resonance that can stall a static-HMC chain.

Both methods consume any :class:`~probpipe.core.protocols.SupportsUnnormalizedLogProb`
target whose log-density is JAX-traceable. They run on the flat-vector
form of the target produced by
:func:`~probpipe.inference._inference_utils.build_target_log_prob_flat`,
then lift the resulting chain back through the prior's
``record_template`` so the posterior preserves the structured
parameterisation.

The per-draw diagnostics (``acceptance_rate``, ``is_divergent``,
``energy``, ``num_integration_steps``) are read off the BlackJAX
``info`` objects and packed into a dict consumed by
:func:`build_mcmc_datatree`, the same ArviZ converter the TFP path
uses. The adapted ``step_size`` is not carried on those ``info``
objects, so it is threaded out of the warmup separately and injected
into the same dict (broadcast across draws).
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Literal

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.mcmc.dynamic_hmc import (
    build_kernel as _dynamic_hmc_build_kernel,
    halton_trajectory_length,
    init as _dynamic_hmc_init,
)

from ..core._registry import MethodInfo
from ..core.distribution import Distribution
from ..core.protocols import SupportsUnnormalizedLogProb
from ..custom_types import Array
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._inference_utils import (
    as_prng_key,
    build_mcmc_datatree,
    build_target_log_prob_flat,
    get_prior,
    is_jax_traceable,
    parallel_chain_map,
    run_chain_scan,
)
from ._registry import InferenceMethod

__all__ = ["BlackJAXNutsMethod", "BlackJAXHmcMethod"]

Algorithm = Literal["nuts", "hmc"]

_KERNEL_FACTORY: dict[Algorithm, Callable[..., Any]] = {
    "nuts": blackjax.nuts,
    "hmc": blackjax.hmc,
}

_EXTRA_KWARGS: dict[Algorithm, Callable[[int], dict[str, Any]]] = {
    "nuts": lambda _num_integration_steps: {},
    "hmc": lambda num_integration_steps: {
        "num_integration_steps": num_integration_steps,
    },
}


# ---------------------------------------------------------------------------
# Randomized HMC trajectory length (shared by warmup and production)
# ---------------------------------------------------------------------------


def _next_random_arg(i: Array) -> Array:
    """Advance the Halton counter one step (``DynamicHMCState`` carries it)."""
    return i + 1


def _halton_steps_fn(num_integration_steps: int) -> Callable[[Array], Array]:
    """Quasi-random leapfrog-step count with mean ``num_integration_steps``.

    BlackJAX's ``halton_trajectory_length`` spans ``[0, 2L-1]`` and so
    occasionally returns ``0`` (~0.1% of draws at ``L=10``). A zero-step
    trajectory is a no-op leapfrog — the proposal equals the current
    position and is always "accepted" — which wastes the draw and
    inflates the acceptance rate. Floor at ``1`` so every iteration makes
    at least one leapfrog step. The mean shift from clamping the rare
    zeros is negligible (< 0.001 at ``L=10``); the floor is applied in
    this one shared helper so warmup and production stay calibrated to
    the identical step-count distribution.
    """
    return lambda i: jnp.maximum(halton_trajectory_length(i, num_integration_steps), 1)


def _halton_warmup_algorithm(num_integration_steps: int) -> Any:
    """A ``window_adaptation``-compatible *algorithm* wrapping ``dynamic_hmc``
    with the same Halton trajectory length used at production.

    ``window_adaptation`` only touches ``algorithm.init(position,
    logdensity_fn, random_generator_arg)`` and
    ``algorithm.build_kernel(integrator)`` (a module-like interface), so a
    tiny namespace shim suffices. ``dynamic_hmc``'s ``init`` takes the
    Halton-counter seed as a third argument; the shim defaults it to ``0``
    so ``window_adaptation`` (which calls ``init`` with two positional
    args) works unchanged. Tuning against the randomized-``L`` kernel —
    rather than a fixed-``L`` stand-in — keeps dual-averaging's acceptance
    target calibrated to the kernel that actually runs, since acceptance
    is nonlinear in the trajectory length.
    """
    steps_fn = _halton_steps_fn(num_integration_steps)

    def init(position: Any, logdensity_fn: Callable, random_generator_arg: Array | None = None):
        if random_generator_arg is None:
            random_generator_arg = jnp.asarray(0)
        return _dynamic_hmc_init(position, logdensity_fn, random_generator_arg)

    def build_kernel(integrator: Callable) -> Callable:
        return _dynamic_hmc_build_kernel(
            integrator=integrator,
            next_random_arg_fn=_next_random_arg,
            integration_steps_fn=steps_fn,
        )

    return SimpleNamespace(init=init, build_kernel=build_kernel)


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def _run_blackjax_chains(
    target_log_prob_fn: Callable[[Array], Array],
    init_state: Array,
    *,
    algorithm: Algorithm,
    num_results: int,
    num_warmup: int,
    num_chains: int,
    step_size: float,
    random_seed: int,
    num_integration_steps: int = 10,
) -> tuple[list[Array], dict[str, np.ndarray]]:
    """Run BlackJAX MCMC chains. Returns ``(chains, sample_stats_dict)``.

    Uses :func:`blackjax.window_adaptation` for NUTS / HMC step-size and
    mass-matrix adaptation during the warmup window, then samples via
    :func:`jax.lax.scan` over the adapted kernel. ``num_integration_steps``
    applies to HMC only; window adaptation does not tune it, and at
    production time HMC draws a Halton-quasi-random trajectory length with
    this value as the *mean* (see :func:`run_one_chain`).
    """
    kernel_factory = _KERNEL_FACTORY[algorithm]
    extra_kwargs = _EXTRA_KWARGS[algorithm](num_integration_steps)

    chain_keys = jax.random.split(as_prng_key(random_seed), num_chains)

    def _adapt(warmup_key: Array) -> tuple[Any, dict[str, Any]]:
        """Either run window-adaptation or fall back to user-provided params.

        The non-positive-``num_warmup`` branch builds the kernel
        directly from ``step_size`` plus an identity mass matrix
        (BlackJAX NUTS / HMC both require ``inverse_mass_matrix`` as a
        constructor argument — window-adaptation normally supplies it).

        For HMC, window-adaptation tunes against the *same* Halton
        randomized-trajectory-length kernel used at production (via
        :func:`_halton_warmup_algorithm`), so dual-averaging's acceptance
        target is calibrated to the kernel that actually runs rather than
        a fixed-``L`` stand-in.
        """
        if num_warmup <= 0:
            params = {
                "step_size": step_size,
                "inverse_mass_matrix": jnp.ones_like(init_state),
                **extra_kwargs,
            }
            state = kernel_factory(target_log_prob_fn, **params).init(init_state)
            return state, params
        if algorithm == "hmc":
            warmup_algorithm = _halton_warmup_algorithm(num_integration_steps)
            warmup = blackjax.window_adaptation(
                warmup_algorithm, target_log_prob_fn, initial_step_size=step_size,
            )
        else:
            warmup = blackjax.window_adaptation(
                kernel_factory, target_log_prob_fn,
                initial_step_size=step_size, **extra_kwargs,
            )
        (state, params), _ = warmup.run(warmup_key, init_state, num_steps=num_warmup)
        return state, params

    def run_one_chain(chain_key: Array) -> tuple[Array, Any, Array]:
        warmup_key, sample_key = jax.random.split(chain_key)
        state, adapted_params = _adapt(warmup_key)
        if algorithm == "hmc":
            # Randomize the trajectory length around ``num_integration_steps``
            # (its mean) with a low-discrepancy Halton sequence. A *fixed*
            # number of leapfrog steps can land the proposal back near the
            # start for near-periodic (e.g. near-Gaussian) targets — high
            # acceptance, no divergences, yet poor mixing. Jittering L breaks
            # that resonance (Neal 2011, "MCMC using Hamiltonian dynamics",
            # sec. 4.2). Warmup (above) tunes step size + mass matrix on the
            # same randomized-L kernel.
            kernel = blackjax.dynamic_hmc(
                target_log_prob_fn,
                step_size=adapted_params["step_size"],
                inverse_mass_matrix=adapted_params["inverse_mass_matrix"],
                next_random_arg_fn=_next_random_arg,
                integration_steps_fn=_halton_steps_fn(num_integration_steps),
            )
            # ``dynamic_hmc`` state carries the Halton counter; re-init from
            # the (possibly dynamic) warmup state's position.
            state = kernel.init(state.position, jnp.asarray(0))
        else:
            kernel = kernel_factory(target_log_prob_fn, **adapted_params)
        positions, infos = run_chain_scan(kernel, state, num_results, sample_key)
        # BlackJAX NUTSInfo / HMCInfo carry no ``step_size`` field, so the
        # adapted (or, for the zero-warmup branch, user-supplied) step size
        # has to be threaded out of ``_adapt`` explicitly to land in the
        # sample-stats dict below. Wrap in ``jnp.asarray``: the zero-warmup
        # branch supplies a plain Python float, but ``parallel_chain_map``'s
        # single-chain path maps ``a[None]`` over the returned pytree and so
        # requires every leaf to be an array.
        return positions, infos, jnp.asarray(adapted_params["step_size"])

    all_positions, all_infos, adapted_step_sizes = parallel_chain_map(
        run_one_chain, chain_keys,
    )
    chains = [all_positions[c] for c in range(num_chains)]
    sample_stats = _extract_blackjax_sample_stats(all_infos, adapted_step_sizes, num_results)
    return chains, sample_stats


def _extract_blackjax_sample_stats(
    infos: Any, step_sizes: Array, num_results: int,
) -> dict[str, np.ndarray]:
    """Pack BlackJAX per-step ``info`` into a TFP-shaped sample-stats dict.

    Mirrors the ArviZ-compatible keys produced by
    :func:`probpipe.inference._tfp_mcmc._extract_sample_stats` (each array
    shaped ``(num_chains, num_results)``) so the downstream
    :func:`build_mcmc_datatree` consumes both backends uniformly.

    ``step_sizes`` is the per-chain adapted (or user-supplied) step size,
    shape ``(num_chains,)``; BlackJAX ``NUTSInfo`` / ``HMCInfo`` do not
    expose ``step_size`` on the per-step info, so it is injected here and
    broadcast across draws to match the per-draw diagnostics. Info fields
    not present on a given algorithm are silently skipped.
    """
    stats: dict[str, np.ndarray] = {}
    for key in ("acceptance_rate", "is_divergent", "num_integration_steps", "energy"):
        value = getattr(infos, key, None)
        if value is not None:
            stats[key] = np.asarray(value)
    step_sizes = np.asarray(step_sizes)
    stats["step_size"] = np.broadcast_to(
        step_sizes[:, None], (step_sizes.shape[0], num_results),
    ).copy()
    return stats


# ---------------------------------------------------------------------------
# Method classes
# ---------------------------------------------------------------------------


class _BlackJAXMCMCMethod(InferenceMethod):
    """Base for BlackJAX gradient MCMC methods (NUTS, HMC)."""

    def __init__(self, algorithm: Algorithm, method_name: str, method_priority: int):
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
        if not isinstance(dist, SupportsUnnormalizedLogProb):
            return MethodInfo(
                feasible=False, method_name=self.name,
                description="Requires SupportsUnnormalizedLogProb",
            )
        try:
            target_flat, flat_init, _ = build_target_log_prob_flat(dist, observed)
            if not is_jax_traceable(target_flat, flat_init):
                return MethodInfo(
                    feasible=False, method_name=self.name,
                    description="Log-prob is not JAX-traceable",
                )
        except Exception as e:
            return MethodInfo(
                feasible=False, method_name=self.name, description=str(e),
            )
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        random_seed: int = kwargs.get("random_seed", 0)
        target_flat, flat_init, record_template = build_target_log_prob_flat(
            dist, observed, init=kwargs.get("init"), random_seed=random_seed,
        )
        num_results: int = kwargs.get("num_results", 1000)
        num_warmup: int = kwargs.get("num_warmup", 500)
        num_chains: int = kwargs.get("num_chains", 1)
        step_size: float = kwargs.get("step_size", 0.1)
        num_integration_steps: int = kwargs.get("num_integration_steps", 10)

        chains, sample_stats = _run_blackjax_chains(
            target_flat, flat_init,
            algorithm=self._algorithm,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
            step_size=step_size,
            random_seed=random_seed,
            num_integration_steps=num_integration_steps,
        )
        auxiliary = build_mcmc_datatree(chains, sample_stats)
        prior = get_prior(dist)
        return make_posterior(
            chains, parents=(prior,), algorithm=self._method_name,
            auxiliary=auxiliary, record_template=record_template,
            num_results=num_results, num_warmup=num_warmup, num_chains=num_chains,
        )


def BlackJAXNutsMethod() -> _BlackJAXMCMCMethod:
    """BlackJAX No-U-Turn Sampler.

    Tier 81-90 (optimised JAX-native backend; the primary auto-dispatch
    winner for any JAX-traceable ``SupportsLogProb`` target — the
    canonical ProbPipe model class). Priority 85. Sits below
    ``nutpie_nuts`` (88; Rust gradients win the constant-factor race for
    Stan / PyMC models) and at the same tier as ``cmdstan_nuts`` /
    ``pymc_nuts`` (82), which apply to disjoint model classes.
    """
    return _BlackJAXMCMCMethod("nuts", "blackjax_nuts", 85)


def BlackJAXHmcMethod() -> _BlackJAXMCMCMethod:
    """BlackJAX Hamiltonian Monte Carlo.

    Tier 61-70 by algorithm category (well-understood, hand-tuned step
    size; trajectory length randomized around a hand-set mean), but
    registered at the opt-in-only sentinel ``priority=0``. Reasoning:
    HMC's ``check()`` is identical to ``blackjax_nuts`` (same
    ``SupportsUnnormalizedLogProb`` + JAX-traceability gate), so with
    NUTS at 85, HMC is structurally unreachable in auto-dispatch. Keeping
    it at 0 makes that explicit; callers who specifically want HMC pin
    ``method="blackjax_hmc"``. The ``num_integration_steps`` kwarg
    (default ``10``) is the *mean* trajectory length: production draws a
    Halton-quasi-random number of leapfrog steps so a fixed-``L``
    resonance cannot silently stall mixing.
    """
    return _BlackJAXMCMCMethod("hmc", "blackjax_hmc", 0)
