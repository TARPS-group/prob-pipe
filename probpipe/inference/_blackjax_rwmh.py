"""BlackJAX-backed gradient-free MCMC: random-walk Metropolis-Hastings.

Two execution paths share the same BlackJAX kernel:

* **Fast path** — ``jax.lax.scan`` for the inner step loop, ``jax.vmap``
  across chains. Used when the target log-density is JAX-traceable.
* **Eager fallback** — a Python ``for`` loop over ``sampler.step``,
  used when the target is *not* JAX-traceable (BridgeStan / scipy /
  external-simulator likelihoods). BlackJAX's ``sampler.step`` accepts
  concrete arrays and runs the user's log-density host-side.

The default warmup is a Stan-style window adaptation: ``n_windows``
geometrically-growing windows each sample with the current proposal
Cholesky and accumulate Welford statistics on positions, refreshing
the proposal at window boundaries. Production samples with
``proposal = chol(Sigma_hat) * 2.38 / sqrt(d)``, the
Roberts-Gelman-Gilks scaling
([Roberts, Gelman & Gilks 1997](https://projecteuclid.org/journals/annals-of-applied-probability/volume-7/issue-1/Weak-convergence-and-optimal-scaling-of-random-walk-Metropolis-algorithms/10.1214/aoap/1034625254.full)).
The ``adapt=False`` path uses ``sigma = step_size * I`` throughout.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from blackjax.adaptation.mass_matrix import welford_algorithm

from ..core._registry import MethodInfo
from ..core.distribution import Distribution
from ..core.node import workflow_function
from ..core.protocols import SupportsUnnormalizedLogProb
from ..custom_types import Array, ArrayLike
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._inference_utils import (
    build_mcmc_datatree,
    extract_record_template,
    get_init_state,
    get_prior,
    is_jax_traceable,
    is_simple_model,
)
from ._registry import InferenceMethod

logger = logging.getLogger(__name__)

__all__ = ["rwmh", "BlackJAXRWMHMethod", "TFPRWMHMethod"]


# ---------------------------------------------------------------------------
# Adaptive warmup
# ---------------------------------------------------------------------------

# Roberts-Gelman-Gilks 1997 asymptotic optimal scaling for RWMH on a
# d-dimensional target: ``proposal_cov = (2.38^2 / d) * Sigma_target``.
# We use ``2.38 / sqrt(d)`` as the scalar multiplier on the proposal
# Cholesky factor (squared, this matches the variance scaling above).
def _rgg_scale(d: int) -> float:
    return 2.38 / float(np.sqrt(d))


def _initial_sigma(d: int) -> Array:
    """Default warmup proposal: ``2.38 / sqrt(d) * I``."""
    return jnp.eye(d) * _rgg_scale(d)


def _production_sigma(cov: Array, d: int) -> Array:
    """RGG-scaled Cholesky of an empirical covariance estimate.

    Returns a lower-triangular ``(d, d)`` matrix usable as the BlackJAX
    ``normal_random_walk`` ``sigma`` argument. Falls back to a diagonal
    of the per-dimension empirical standard deviations when ``cov`` is
    not positive-definite (rare but possible early in warmup).
    """
    try:
        chol = jsl.cholesky(cov, lower=True)
    except Exception:
        diag = jnp.sqrt(jnp.clip(jnp.diag(cov), a_min=1e-8))
        chol = jnp.diag(diag)
    return chol * _rgg_scale(d)


# ---------------------------------------------------------------------------
# Window scheduling
# ---------------------------------------------------------------------------


_MIN_STEPS_PER_WINDOW = 25  # minimum steps for Welford to settle


def _window_sizes(num_warmup: int, n_windows: int, ratio: float = 2.0) -> list[int]:
    """Geometric window sizes summing to ``num_warmup``.

    Stan-style window adaptation uses growing windows so the first
    (badly-mixed) window contributes little to the cov estimate while
    later (well-mixed) windows dominate. We mirror that: window
    ``i`` has weight ``ratio ** i``, normalised to sum to one and
    rounded to integer step counts.

    ``n_windows`` is automatically clamped so each window holds at
    least :data:`_MIN_STEPS_PER_WINDOW` (= 25) steps — Stan's default
    minimum. Short warmups (``num_warmup < 50``) collapse to a single
    phase: a fixed RGG-scaled identity proposal throughout with a
    one-shot Welford fit at the end.
    """
    if num_warmup <= 0:
        return []
    max_windows = max(1, num_warmup // _MIN_STEPS_PER_WINDOW)
    n = min(int(n_windows), max_windows)
    if n <= 1:
        return [num_warmup]
    weights = np.asarray([ratio ** i for i in range(n)], dtype=float)
    weights = weights / weights.sum()
    sizes = np.maximum(1, np.round(weights * num_warmup).astype(int))
    sizes[-1] += num_warmup - int(sizes.sum())
    return [int(s) for s in sizes]


# ---------------------------------------------------------------------------
# Adaptive warmup — fast (lax.scan) and eager (Python loop) variants
# ---------------------------------------------------------------------------


def _adaptive_warmup_fast(
    target_log_prob_fn: Callable[[Array], Array],
    init_state: Array,
    key: Array,
    num_warmup: int,
    *,
    n_windows: int,
) -> tuple[Any, Array, Array]:
    """Window-style adaptive warmup via ``lax.scan`` inside each window.

    Splits ``num_warmup`` into ``n_windows`` geometrically-growing
    windows. Each window samples with the current proposal Cholesky
    (initially ``2.38 / sqrt(d) * I``) while accumulating Welford
    statistics on positions; at the window boundary, the Cholesky is
    refreshed from the cumulative Welford state. The last window's
    estimate is returned as the production proposal cov.

    Welford state is *cumulative* across windows — the geometric
    schedule already downweights the early biased samples without
    needing to discard them.
    """
    d = init_state.shape[0]
    welf_init, welf_update, welf_final = welford_algorithm(is_diagonal_matrix=False)
    sizes = _window_sizes(num_warmup, n_windows)

    sigma = _initial_sigma(d)
    rw_state = blackjax.normal_random_walk(target_log_prob_fn, sigma=sigma).init(init_state)
    welf_state = welf_init(d)
    window_positions: list[Array] = []

    k = key
    for w_size in sizes:
        sampler = blackjax.normal_random_walk(target_log_prob_fn, sigma=sigma)
        # Re-init wraps the existing position into a state matched to
        # the new sampler's logdensity_fn closure (it would be the same
        # in our case, but staying consistent with BlackJAX's init/step
        # contract avoids edge cases if the kernel ever caches lp).
        rw_state = sampler.init(rw_state.position)

        def step(carry, step_key, _sampler=sampler):
            rw, welf = carry
            rw, _info = _sampler.step(step_key, rw)
            welf = welf_update(welf, rw.position)
            return (rw, welf), rw.position

        k, sub = jax.random.split(k)
        keys = jax.random.split(sub, w_size)
        (rw_state, welf_state), positions = jax.lax.scan(
            step, (rw_state, welf_state), keys,
        )
        window_positions.append(positions)

        cov, _, _ = welf_final(welf_state)
        sigma = _production_sigma(cov, d)

    if sizes:
        warmup_positions = jnp.concatenate(window_positions, axis=0)
        final_cov, _, _ = welf_final(welf_state)
    else:
        warmup_positions = jnp.empty((0, d), dtype=init_state.dtype)
        final_cov = jnp.eye(d)
    return rw_state, final_cov, warmup_positions


def _adaptive_warmup_eager(
    target_log_prob_fn: Callable[[Array], Array],
    init_state: Array,
    key: Array,
    num_warmup: int,
    *,
    n_windows: int,
) -> tuple[Any, Array, Array]:
    """Eager-path windowed warmup: same logic, Python ``for`` inside windows.

    BlackJAX primitives all work on concrete JAX arrays without
    tracing, so this path supports non-JAX-traceable log-densities.
    """
    d = init_state.shape[0]
    welf_init, welf_update, welf_final = welford_algorithm(is_diagonal_matrix=False)
    sizes = _window_sizes(num_warmup, n_windows)

    sigma = _initial_sigma(d)
    rw_state = blackjax.normal_random_walk(target_log_prob_fn, sigma=sigma).init(init_state)
    welf_state = welf_init(d)
    positions: list[Array] = []

    k = key
    for w_size in sizes:
        sampler = blackjax.normal_random_walk(target_log_prob_fn, sigma=sigma)
        rw_state = sampler.init(rw_state.position)
        for _ in range(w_size):
            k, sub = jax.random.split(k)
            rw_state, _info = sampler.step(sub, rw_state)
            welf_state = welf_update(welf_state, rw_state.position)
            positions.append(rw_state.position)
        cov, _, _ = welf_final(welf_state)
        sigma = _production_sigma(cov, d)

    if positions:
        warmup_positions = jnp.stack(positions)
        final_cov, _, _ = welf_final(welf_state)
    else:
        warmup_positions = jnp.empty((0, d), dtype=init_state.dtype)
        final_cov = jnp.eye(d)
    return rw_state, final_cov, warmup_positions


# ---------------------------------------------------------------------------
# Production sampling
# ---------------------------------------------------------------------------


def _sample_chain_fast(
    target_log_prob_fn: Callable[[Array], Array],
    init_position: Array,
    sigma: Array,
    num_results: int,
    key: Array,
) -> tuple[Array, dict[str, Array]]:
    """Run ``num_results`` BlackJAX RWMH steps under ``lax.scan``."""
    sampler = blackjax.normal_random_walk(target_log_prob_fn, sigma=sigma)
    state = sampler.init(init_position)

    def step(state, step_key):
        state, info = sampler.step(step_key, state)
        return state, (state.position, info.acceptance_rate, info.is_accepted)

    keys = jax.random.split(key, num_results)
    _, (positions, acc_rate, is_accepted) = jax.lax.scan(step, state, keys)
    return positions, {"acceptance_rate": acc_rate, "is_accepted": is_accepted}


def _sample_chain_eager(
    target_log_prob_fn: Callable[[Array], Array],
    init_position: Array,
    sigma: Array,
    num_results: int,
    key: Array,
) -> tuple[Array, dict[str, Array]]:
    """Run ``num_results`` BlackJAX RWMH steps in a Python loop."""
    sampler = blackjax.normal_random_walk(target_log_prob_fn, sigma=sigma)
    state = sampler.init(init_position)

    positions: list[Array] = []
    accept_rates: list[Array] = []
    accepts: list[Array] = []
    k = key
    for _ in range(num_results):
        k, sub = jax.random.split(k)
        state, info = sampler.step(sub, state)
        positions.append(state.position)
        accept_rates.append(info.acceptance_rate)
        accepts.append(info.is_accepted)
    return jnp.stack(positions), {
        "acceptance_rate": jnp.stack(accept_rates),
        "is_accepted": jnp.stack(accepts),
    }


# ---------------------------------------------------------------------------
# End-to-end runner (routes to fast or eager path)
# ---------------------------------------------------------------------------


def _run_one_chain(
    target_log_prob_fn: Callable[[Array], Array],
    init_state: Array,
    key: Array,
    *,
    num_results: int,
    num_warmup: int,
    adapt: bool,
    step_size: float,
    proposal_sigma_override: Array | None,
    n_windows: int,
    traceable: bool,
) -> tuple[Array, Array | None, dict[str, Array]]:
    """Drive one RWMH chain — warmup + production — under the chosen path.

    Returns ``(chain, warmup_positions_or_None, sample_stats)``.
    """
    d = init_state.shape[0]
    warmup_key, sample_key = jax.random.split(key)

    if proposal_sigma_override is not None:
        sigma = proposal_sigma_override
        warmup_positions = None
        if num_warmup > 0:
            _, _, warmup_positions = (
                _adaptive_warmup_fast if traceable else _adaptive_warmup_eager
            )(target_log_prob_fn, init_state, warmup_key, num_warmup,
              n_windows=n_windows)
            init_position = warmup_positions[-1]
        else:
            init_position = init_state
    elif adapt and num_warmup > 0:
        warmup_fn = _adaptive_warmup_fast if traceable else _adaptive_warmup_eager
        rw_state, cov, warmup_positions = warmup_fn(
            target_log_prob_fn, init_state, warmup_key, num_warmup,
            n_windows=n_windows,
        )
        sigma = _production_sigma(cov, d)
        init_position = rw_state.position
    else:
        sigma = jnp.eye(d) * step_size
        warmup_positions = None
        if num_warmup > 0:
            run_warmup = (
                _sample_chain_fast if traceable else _sample_chain_eager
            )
            warmup_positions, _ = run_warmup(
                target_log_prob_fn, init_state, sigma, num_warmup, warmup_key,
            )
            init_position = warmup_positions[-1]
        else:
            init_position = init_state

    sample_fn = _sample_chain_fast if traceable else _sample_chain_eager
    chain, sample_stats = sample_fn(
        target_log_prob_fn, init_position, sigma, num_results, sample_key,
    )
    return chain, warmup_positions, sample_stats


def _run_blackjax_rwmh(
    target_log_prob_fn: Callable[[Array], Array],
    init_state: Array,
    *,
    num_results: int,
    num_warmup: int,
    num_chains: int,
    adapt: bool,
    step_size: float,
    proposal_sigma_override: Array | None,
    n_windows: int,
    random_seed: int,
) -> tuple[list[Array], list[Array] | None, dict[str, np.ndarray], float]:
    """Run ``num_chains`` BlackJAX RWMH chains. Returns chains + diagnostics.

    Auto-routes to the fast path (``lax.scan`` + ``vmap``) when the
    target is JAX-traceable at ``init_state``; otherwise falls back to
    Python-loop execution (per chain).
    """
    traceable = is_jax_traceable(target_log_prob_fn, init_state)
    key = jax.random.PRNGKey(random_seed)
    chain_keys = jax.random.split(key, num_chains)

    if traceable:
        def run_one(chain_key):
            return _run_one_chain(
                target_log_prob_fn, init_state, chain_key,
                num_results=num_results, num_warmup=num_warmup,
                adapt=adapt, step_size=step_size,
                proposal_sigma_override=proposal_sigma_override,
                n_windows=n_windows,
                traceable=True,
            )
        chains_arr, warmups_arr, stats_arr = jax.vmap(run_one)(chain_keys)
        chains = [chains_arr[c] for c in range(num_chains)]
        warmups = (
            [warmups_arr[c] for c in range(num_chains)]
            if warmups_arr is not None and warmups_arr.shape[0] == num_chains
            else None
        )
        sample_stats = {
            k: np.asarray(v) for k, v in stats_arr.items()
        }
    else:
        chains_l: list[Array] = []
        warmups_l: list[Array] = []
        accept_l: list[np.ndarray] = []
        is_acc_l: list[np.ndarray] = []
        for chain_key in chain_keys:
            ch, wm, st = _run_one_chain(
                target_log_prob_fn, init_state, chain_key,
                num_results=num_results, num_warmup=num_warmup,
                adapt=adapt, step_size=step_size,
                proposal_sigma_override=proposal_sigma_override,
                n_windows=n_windows,
                traceable=False,
            )
            chains_l.append(ch)
            if wm is not None:
                warmups_l.append(wm)
            accept_l.append(np.asarray(st["acceptance_rate"]))
            is_acc_l.append(np.asarray(st["is_accepted"]))
        chains = chains_l
        warmups = warmups_l if len(warmups_l) == num_chains else None
        sample_stats = {
            "acceptance_rate": np.stack(accept_l),
            "is_accepted": np.stack(is_acc_l),
        }

    accept_rate = float(np.mean(sample_stats["is_accepted"]))
    return chains, warmups, sample_stats, accept_rate


# ---------------------------------------------------------------------------
# Workflow function
# ---------------------------------------------------------------------------


@workflow_function
def rwmh(
    dist: SupportsUnnormalizedLogProb,
    data: ArrayLike | None = None,
    *,
    log_prob_fn: Any | None = None,
    num_results: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 1,
    step_size: float = 0.1,
    adapt: bool = True,
    n_windows: int = 4,
    proposal_cov: ArrayLike | None = None,
    init: ArrayLike | None = None,
    random_seed: int = 0,
) -> ApproximateDistribution:
    """Gradient-free random-walk Metropolis-Hastings (BlackJAX-backed).

    Two execution paths share the same BlackJAX kernel:

    * a fast path using ``jax.lax.scan`` + ``jax.vmap`` across chains,
      when the target log-density is JAX-traceable at the initial
      state;
    * an eager Python-loop fallback otherwise (BridgeStan / scipy /
      external-simulator likelihoods).

    Parameters
    ----------
    dist
        Distribution providing ``_unnormalized_log_prob``. RWMH uses
        only the unnormalized density because the missing log
        normalizer cancels out of every accept/reject step.
    data
        Observed data forwarded to ``log_prob_fn`` when supplied.
    log_prob_fn
        ``log_prob_fn(params, data) -> float`` combined with
        ``dist._unnormalized_log_prob(params)`` to form the target.
    num_results, num_warmup, num_chains
        MCMC tuning parameters.
    step_size
        Diagonal proposal scale used when ``adapt=False`` and
        ``proposal_cov=None``.
    adapt
        When ``True`` (default), runs a window-style adaptive warmup —
        ``n_windows`` geometrically-growing windows that each sample
        with the current proposal Cholesky and accumulate Welford
        statistics on positions, refreshing the proposal at window
        boundaries. Production samples with
        ``proposal = chol(Sigma_hat) * 2.38 / sqrt(d)`` (the
        Roberts-Gelman-Gilks scaling). When ``False``, skips
        adaptation and uses ``sigma = step_size * I`` throughout.
    n_windows
        Number of geometric warmup windows when ``adapt=True``. The
        Stan-style window-adaptation pattern downweights the early
        biased samples by giving later windows more steps. Default
        ``4``; ``n_windows=1`` collapses to a single-phase warmup with
        a fixed RGG-scaled identity proposal throughout. Ignored when
        ``adapt=False``.
    proposal_cov
        Explicit ``(d, d)`` proposal Cholesky factor. Overrides both
        the adaptive fit and ``step_size``. Useful when the user has
        a precomputed covariance estimate from elsewhere.
    init
        Initial chain state. Resolved by
        :func:`~probpipe.inference._inference_utils.get_init_state`
        when ``None``.
    random_seed
        Seed for chain initialisation, warmup, and sampling RNG.

    Returns
    -------
    ApproximateDistribution
        Posterior samples with chain structure and an auxiliary
        ArviZ-shaped ``DataTree`` carrying per-step acceptance stats
        and warmup positions.
    """
    if not isinstance(dist, SupportsUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support log_prob "
            "(does not implement SupportsUnnormalizedLogProb)"
        )

    if log_prob_fn is not None and data is not None:
        def target_log_prob(params):
            return dist._unnormalized_log_prob(params) + log_prob_fn(params, data)
    else:
        def target_log_prob(params):
            return dist._unnormalized_log_prob(params)

    init_state = get_init_state(dist, init, random_seed=random_seed)
    proposal_sigma_override = (
        jnp.asarray(proposal_cov) if proposal_cov is not None else None
    )

    chains, warmups, sample_stats, accept_rate = _run_blackjax_rwmh(
        target_log_prob,
        init_state,
        num_results=num_results,
        num_warmup=num_warmup,
        num_chains=num_chains,
        adapt=adapt,
        step_size=step_size,
        proposal_sigma_override=proposal_sigma_override,
        n_windows=n_windows,
        random_seed=random_seed,
    )

    auxiliary = build_mcmc_datatree(chains, sample_stats, warmup_chains=warmups)
    record_template = extract_record_template(dist)
    return make_posterior(
        chains, parents=(dist,), algorithm="rwmh",
        auxiliary=auxiliary, record_template=record_template,
        num_results=num_results, num_warmup=num_warmup, num_chains=num_chains,
        step_size=step_size, accept_rate=accept_rate, adapt=adapt,
        n_windows=n_windows,
    )


# ---------------------------------------------------------------------------
# Registry method
# ---------------------------------------------------------------------------


class BlackJAXRWMHMethod(InferenceMethod):
    """Gradient-free RWMH on top of BlackJAX's ``normal_random_walk``.

    Tier 51-60 (slow per effective sample in high dimensions even when
    tuned). Priority 55. Auto-dispatched when no gradient-based method
    passes ``check()`` — e.g., for log-densities that are not
    JAX-traceable — or when the user pins ``method="blackjax_rwmh"``.
    """

    @property
    def name(self) -> str:
        return "blackjax_rwmh"

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    @property
    def priority(self) -> int:
        return 55

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> MethodInfo:
        prior = get_prior(dist)
        if not isinstance(prior, SupportsUnnormalizedLogProb):
            return MethodInfo(
                feasible=False, method_name=self.name,
                description="Requires SupportsUnnormalizedLogProb",
            )
        if observed is not None and isinstance(observed, dict):
            return MethodInfo(
                feasible=False, method_name=self.name,
                description="Does not support dict-based conditioning",
            )
        return MethodInfo(feasible=True, method_name=self.name)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        prior = get_prior(dist)
        log_prob_fn = None
        if is_simple_model(dist):
            lik = dist._likelihood
            log_prob_fn = lambda params, d: lik.log_likelihood(params=params, data=d)

        random_seed = kwargs.get("random_seed", 0)
        init = kwargs.get("init")
        if init is None:
            init = get_init_state(dist, None, random_seed=random_seed)

        return rwmh._func(
            prior, observed,
            log_prob_fn=log_prob_fn,
            num_results=kwargs.get("num_results", 1000),
            num_warmup=kwargs.get("num_warmup", 500),
            num_chains=kwargs.get("num_chains", 1),
            step_size=kwargs.get("step_size", 0.1),
            adapt=kwargs.get("adapt", True),
            n_windows=kwargs.get("n_windows", 4),
            proposal_cov=kwargs.get("proposal_cov"),
            init=init,
            random_seed=random_seed,
        )


class TFPRWMHMethod(BlackJAXRWMHMethod):
    """Deprecated alias for :class:`BlackJAXRWMHMethod`.

    Registered at ``priority=0`` (opt-in only) so auto-dispatch never
    selects it; explicit ``method="tfp_rwmh"`` keeps working and emits
    a :class:`DeprecationWarning`. To be removed one minor release out.
    """

    @property
    def name(self) -> str:
        return "tfp_rwmh"

    @property
    def priority(self) -> int:
        return 0

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        import warnings
        warnings.warn(
            'method="tfp_rwmh" is deprecated; use method="blackjax_rwmh" '
            "(the default for gradient-free MCMC) instead.",
            DeprecationWarning, stacklevel=2,
        )
        return super().execute(dist, observed, **kwargs)
