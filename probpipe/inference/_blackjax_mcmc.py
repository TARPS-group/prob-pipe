"""BlackJAX-backed gradient MCMC methods: NUTS and HMC.

Two :class:`~probpipe.core._dispatch.UnaryDispatchMethod` subclasses registered with
:data:`~probpipe.inference.inference_method_registry`:

* ``blackjax_nuts`` — No-U-Turn Sampler with window-adapted step size and
  mass matrix, whose ``target_accept_prob`` and ``max_tree_depth`` options
  set the adaptation's target acceptance rate and the trajectory's maximum
  number of doublings.
* ``blackjax_hmc`` — Hamiltonian Monte Carlo with window-adapted step
  size and a *randomized* trajectory length: production samples draw the
  number of leapfrog steps from a low-discrepancy Halton sequence with
  mean ``num_integration_steps`` (a user-tunable kwarg, default ``10``),
  breaking the fixed-``L`` resonance that can stall a static-HMC chain.

Both methods consume any :class:`~probpipe.SupportsUnnormalizedLogProb`
target whose log-density is JAX-traceable. They run on the flat-vector
form of the target produced by
:func:`~probpipe.inference._inference_utils.build_target_log_prob_flat`,
then lift the resulting chain back through the prior's event spec,
so the posterior names its fields by the prior's outputs.

The per-draw diagnostics (``acceptance_rate``, ``diverging``,
``energy``, ``num_integration_steps``, and for NUTS ``tree_depth`` and
``reached_max_treedepth``) are read off the BlackJAX ``info`` objects,
under ArviZ's names, and packed into a dict consumed by
:func:`build_mcmc_datatree`, the same ArviZ converter the TFP path
uses. The adapted ``step_size`` is not carried on those ``info``
objects, so it is threaded out of the warmup separately and injected
into the same dict (broadcast across draws).
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, ClassVar, Literal

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.mcmc.dynamic_hmc import (
    build_kernel as _dynamic_hmc_build_kernel,
)
from blackjax.mcmc.dynamic_hmc import (
    halton_trajectory_length,
)
from blackjax.mcmc.dynamic_hmc import (
    init as _dynamic_hmc_init,
)

from ..core._dispatch import Feasibility
from ..custom_types import Array, PRNGKey
from ..distributions._capabilities import SupportsUnnormalizedLogProb
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution
from ..operations._condition import InferenceMethod
from ..operations._inference_target import InferenceTarget
from ._approximate_distribution import make_posterior
from ._inference_utils import (
    UNTRACEABLE_DENSITY,
    as_prng_key,
    build_mcmc_datatree,
    build_target_log_prob_flat,
    chain_initial_states,
    no_density_reason,
    nuts_controls,
    observed_parts,
    parallel_chain_map,
    run_chain_scan,
    traces_in_jax,
    unconstrained_chain,
)

__all__ = ["BlackJAXHmcMethod", "BlackJAXNutsMethod"]

Algorithm = Literal["nuts", "hmc"]

_KERNEL_FACTORY: dict[Algorithm, Callable[..., Any]] = {
    "nuts": blackjax.nuts,
    "hmc": blackjax.hmc,
}

#: BlackJAX's defaults of the target acceptance rate and the maximum number of doublings.
_DEFAULT_TARGET_ACCEPT_PROB = 0.8
_DEFAULT_MAX_TREE_DEPTH = 10


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
    target_accept_prob: float = _DEFAULT_TARGET_ACCEPT_PROB,
    max_tree_depth: int = _DEFAULT_MAX_TREE_DEPTH,
    per_chain_init: bool = False,
) -> tuple[list[Array], dict[str, np.ndarray]]:
    """Run BlackJAX MCMC chains. Returns ``(chains, sample_stats_dict)``.

    Uses :func:`blackjax.window_adaptation` for NUTS / HMC step-size and
    mass-matrix adaptation during the warmup window, then samples via
    :func:`jax.lax.scan` over the adapted kernel. ``num_integration_steps``
    applies to HMC only; window adaptation does not tune it, and at
    production time HMC draws a Halton-quasi-random trajectory length with
    this value as the *mean* (see :func:`run_one_chain`).
    ``target_accept_prob`` is the acceptance rate that NUTS's step-size
    adaptation targets, and ``max_tree_depth`` is NUTS's maximum number of
    trajectory doublings. ``init_state`` is one state, or one per chain on its
    leading axis when ``per_chain_init`` is set.
    """
    kernel_factory = _KERNEL_FACTORY[algorithm]
    if algorithm == "nuts":
        extra_kwargs: dict[str, Any] = {"max_num_doublings": max_tree_depth}
    else:
        extra_kwargs = {"num_integration_steps": num_integration_steps}

    chain_keys = jax.random.split(as_prng_key(random_seed), num_chains)

    def _adapt(warmup_key: Array, chain_init: Array) -> tuple[Any, dict[str, Any]]:
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
                "inverse_mass_matrix": jnp.ones_like(chain_init),
                **extra_kwargs,
            }
            state = kernel_factory(target_log_prob_fn, **params).init(chain_init)
            return state, params
        if algorithm == "hmc":
            warmup_algorithm = _halton_warmup_algorithm(num_integration_steps)
            warmup = blackjax.window_adaptation(
                warmup_algorithm,
                target_log_prob_fn,
                initial_step_size=step_size,
            )
        else:
            warmup = blackjax.window_adaptation(
                kernel_factory,
                target_log_prob_fn,
                initial_step_size=step_size,
                target_acceptance_rate=target_accept_prob,
                **extra_kwargs,
            )
        (state, params), _ = warmup.run(warmup_key, chain_init, num_steps=num_warmup)
        return state, params

    def run_one_chain(chain_key: Array, chain_init: Array) -> tuple[Array, Any, Array]:
        warmup_key, sample_key = jax.random.split(chain_key)
        state, adapted_params = _adapt(warmup_key, chain_init)
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

    if per_chain_init:
        all_positions, all_infos, adapted_step_sizes = parallel_chain_map(
            lambda chain: run_one_chain(*chain), (chain_keys, init_state)
        )
    else:
        all_positions, all_infos, adapted_step_sizes = parallel_chain_map(
            lambda chain_key: run_one_chain(chain_key, init_state), chain_keys
        )
    chains = [all_positions[c] for c in range(num_chains)]
    sample_stats = _extract_blackjax_sample_stats(
        all_infos, adapted_step_sizes, num_results, max_tree_depth
    )
    return chains, sample_stats


def _extract_blackjax_sample_stats(
    infos: Any,
    step_sizes: Array,
    num_results: int,
    max_tree_depth: int = _DEFAULT_MAX_TREE_DEPTH,
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

    NUTS's ``num_trajectory_expansions`` is recorded as ArviZ's
    ``tree_depth``, and ``reached_max_treedepth`` marks each draw whose
    trajectory reached ``max_tree_depth`` doublings without turning or
    diverging, which is the draw the depth limit cut short.
    """
    stats: dict[str, np.ndarray] = {}
    # Each info field under the name ArviZ gives it, so a divergence is "diverging".
    for field, key in (
        ("acceptance_rate", "acceptance_rate"),
        ("is_divergent", "diverging"),
        ("num_integration_steps", "num_integration_steps"),
        ("energy", "energy"),
    ):
        value = getattr(infos, field, None)
        if value is not None:
            stats[key] = np.asarray(value)
    expansions = getattr(infos, "num_trajectory_expansions", None)
    if expansions is not None:
        tree_depth = np.asarray(expansions).astype(np.int64)
        stats["tree_depth"] = tree_depth
        stats["reached_max_treedepth"] = (
            (tree_depth >= max_tree_depth)
            & ~np.asarray(infos.is_turning)
            & ~np.asarray(infos.is_divergent)
        )
    step_sizes = np.asarray(step_sizes)
    stats["step_size"] = np.broadcast_to(
        step_sizes[:, None],
        (step_sizes.shape[0], num_results),
    ).copy()
    return stats


# ---------------------------------------------------------------------------
# Method classes
# ---------------------------------------------------------------------------


class _BlackJAXMCMCMethod(InferenceMethod):
    """Base for BlackJAX gradient MCMC methods (NUTS, HMC)."""

    _method_options: ClassVar[tuple[str, ...] | None] = (
        "init",
        "num_chains",
        "num_integration_steps",
        "num_results",
        "num_warmup",
        "step_size",
    )

    def __init__(
        self,
        algorithm: Algorithm,
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
        try:
            target_flat, flat_init = build_target_log_prob_flat(model, observed)
            unconstrained_chain(target_flat, flat_init, model)
        except Exception as e:
            return Feasibility(
                feasible=False,
                description=str(e),
            )
        return Feasibility(feasible=True)

    def execute(
        self, target: InferenceTarget, /, *, key: PRNGKey, **options: Any
    ) -> EmpiricalDistribution:
        """Chains of the BlackJAX kernel on the flat form of the target's unnormalized density.

        Parameters
        ----------
        target : InferenceTarget
            The law to sample: a law with an unnormalized density, or an
            unnormalized conditional at data its joint does not declare as fields.
        key : PRNGKey
            The run's random key.
        **options : Any
            The call's ``method_options``, which :func:`BlackJAXNutsMethod`
            and :func:`BlackJAXHmcMethod` list.

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
        model, observed = observed_parts(target)
        num_results: int = options.get("num_results", 1000)
        num_warmup: int = options.get("num_warmup", 500)
        num_chains: int = options.get("num_chains", 4)
        step_size: float = options.get("step_size", 0.1)
        num_integration_steps: int = options.get("num_integration_steps", 10)
        if self._algorithm == "nuts":
            target_accept_prob, max_tree_depth = nuts_controls(self._method_name, options)
        else:
            target_accept_prob, max_tree_depth = (
                _DEFAULT_TARGET_ACCEPT_PROB,
                _DEFAULT_MAX_TREE_DEPTH,
            )
        init, per_chain_init = chain_initial_states(
            model,
            options.get("init"),
            num_chains=num_chains,
            key=key,
            method=self._method_name,
            default_per_chain=self._algorithm == "nuts",
        )
        target_flat, flat_init = build_target_log_prob_flat(
            model,
            observed,
            init=init,
            random_seed=key,
        )

        density, init, constrain = unconstrained_chain(
            target_flat, flat_init, model, per_chain=per_chain_init
        )
        chains, sample_stats = _run_blackjax_chains(
            density,
            init,
            algorithm=self._algorithm,
            num_results=num_results,
            num_warmup=num_warmup,
            num_chains=num_chains,
            step_size=step_size,
            random_seed=key,
            num_integration_steps=num_integration_steps,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
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


class _BlackJAXNutsMethod(_BlackJAXMCMCMethod):
    """The BlackJAX NUTS method, which reads the NUTS options and no trajectory length."""

    _method_options = (
        "init",
        "max_tree_depth",
        "num_chains",
        "num_results",
        "num_warmup",
        "step_size",
        "target_accept_prob",
    )


def BlackJAXNutsMethod() -> _BlackJAXMCMCMethod:
    """BlackJAX No-U-Turn Sampler, registered as ``blackjax_nuts`` at priority 85.

    Applies to any ``SupportsUnnormalizedLogProb`` target with a
    JAX-traceable log-density, the canonical ProbPipe model class, for which
    it is the method automatic selection picks. Its ``method_options`` are:

    - ``num_results``: the number of draws of each chain, which is 1000 unless
      set;
    - ``num_warmup``: the number of warmup steps of each chain, in which
      BlackJAX's window adaptation tunes the step size and a diagonal mass
      matrix, which is 500 unless set;
    - ``num_chains``: the number of chains, which is 4 unless set;
    - ``step_size``: the initial step size of the adaptation, or the step size
      of a run without warmup, which is 0.1 unless set;
    - ``target_accept_prob``: the acceptance rate that the step-size
      adaptation targets, a number strictly between 0 and 1, which is
      BlackJAX's 0.8 unless set. A higher target adapts a smaller step, which
      removes the divergent transitions of a posterior with regions of high
      curvature at the cost of longer trajectories;
    - ``max_tree_depth``: the maximum number of times a trajectory doubles, a
      positive integer, which is BlackJAX's 10 unless set. A trajectory takes
      at most ``2**max_tree_depth - 1`` integration steps;
    - ``init``: where the chains start, in the four forms below.

    ``init`` takes four forms, for this method, ``blackjax_hmc``, and ``tfp_nuts``:

    1. unset or ``None``: for ``blackjax_nuts`` and ``tfp_nuts``, each chain
       starts from its own default initial state, as for ``"per_chain"``, and
       for ``blackjax_hmc``, every chain starts from one default initial
       state. A default initial state is a draw of the prior when the prior
       can be sampled. A target whose parameters are neither one array nor a
       record of arrays starts every chain from one state;
    2. one state, a flat vector of the target's parameters in the order of
       its event spec, or the array of a target with one array parameter:
       every chain starts from it;
    3. one state per chain, an array of shape ``(num_chains, *state_shape)``:
       each chain starts from its row;
    4. ``"per_chain"``: each chain starts from its own default initial state,
       drawn with a key derived from the chain's key.

    Chains from distinct starting points let R-hat detect a chain that has not
    left the mode it started in.

    The run's sample statistics hold, under ArviZ's names, ``tree_depth``, the
    number of doublings of each draw's trajectory, and
    ``reached_max_treedepth``, which marks each draw whose trajectory reached
    ``max_tree_depth`` doublings without turning or diverging.

    Notes
    -----
    An optimised JAX-native backend. Below ``nutpie_nuts`` (88), whose Rust
    gradients win the constant-factor race for Stan and PyMC models, and
    above ``cmdstan_nuts`` and ``pymc_nuts`` (82), which apply to disjoint
    model classes.
    """
    return _BlackJAXNutsMethod(
        "nuts", "blackjax_nuts", 85, "NUTS through BlackJAX, for a JAX-traceable log-density."
    )


def BlackJAXHmcMethod() -> _BlackJAXMCMCMethod:
    """BlackJAX Hamiltonian Monte Carlo, registered as ``blackjax_hmc``, opt-in-only.

    Same feasibility class as ``blackjax_nuts``; runs only when the caller
    pins ``method="blackjax_hmc"``. The ``num_integration_steps`` kwarg
    (default ``10``) is the *mean* trajectory length: each draw uses a
    Halton-quasi-random number of leapfrog steps so a fixed-``L`` resonance
    cannot silently stall mixing. Its other ``method_options`` are
    ``num_results``, ``num_warmup``, ``num_chains``, ``step_size``, and
    ``init``, as for ``blackjax_nuts``.

    Notes
    -----
    Well understood but hand-tuned: a hand-set step size and a trajectory
    length randomized around a hand-set mean. Its ``check()`` is identical to
    that of ``blackjax_nuts``, so with NUTS ranked, HMC would never be
    selected automatically; ``priority=None`` makes that explicit.
    """
    return _BlackJAXMCMCMethod(
        "hmc",
        "blackjax_hmc",
        None,
        "HMC through BlackJAX with a hand-set step size, for a JAX-traceable log-density.",
    )
