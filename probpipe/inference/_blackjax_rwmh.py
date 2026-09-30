"""BlackJAX-backed gradient-free MCMC: random-walk Metropolis-Hastings.

Two execution paths share the same BlackJAX kernel:

* **Fast path** — ``jax.lax.scan`` for the inner step loop, ``jax.vmap``
  across chains. Used when the target log-density is JAX-traceable.
* **Eager fallback** — a Python ``for`` loop over ``sampler.step``,
  used when the target is *not* JAX-traceable (BridgeStan / scipy /
  external-simulator likelihoods). BlackJAX's ``sampler.step`` accepts
  concrete arrays and runs the user's log-density host-side.

The default warmup is a Stan-style window adaptation. It splits into up
to ``n_windows`` geometrically growing windows of at least 25 steps each,
or runs as one window when it is too short to split. Each window samples
with the current proposal Cholesky and accumulates Welford statistics on
positions. At each window boundary the proposal covariance is refit as
the Welford estimate shrunk toward the covariance the proposal in use
assumes, so it stays positive definite even when the warmup positions
have no spread. Production samples with
``proposal = chol(Sigma) * 2.38 / sqrt(d)``, where ``Sigma`` is the last
refit covariance; the factor ``2.38 / sqrt(d)`` is the
Roberts-Gelman-Gilks scaling
([Roberts, Gelman & Gilks 1997](https://projecteuclid.org/journals/annals-of-applied-probability/volume-7/issue-1/Weak-convergence-and-optimal-scaling-of-random-walk-Metropolis-algorithms/10.1214/aoap/1034625254.full)).
The ``adapt=False`` path uses ``sigma = step_size * I`` throughout.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from blackjax.adaptation.mass_matrix import welford_algorithm

from ..core._dispatch import Feasibility
from ..core.protocols import SupportsUnnormalizedLogProb
from ..custom_types import Array, ArrayLike
from ..distributions._distribution import Distribution
from ._approximate_distribution import ApproximateDistribution, make_posterior
from ._inference_utils import (
    build_mcmc_datatree,
    extract_event_template,
    get_init_state,
    get_prior,
    is_jax_traceable,
    is_simple_model,
    parallel_chain_map,
    run_chain_scan,
)
from ._registry import InferenceMethod

logger = logging.getLogger(__name__)

__all__ = ["BlackJAXRWMHMethod", "rwmh"]


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
    """RGG-scaled Cholesky factor ``chol(cov) * 2.38 / sqrt(d)``.

    Returns a lower-triangular ``(d, d)`` matrix usable as the BlackJAX
    ``normal_random_walk`` ``sigma`` argument. JAX's ``cholesky`` returns
    non-finite entries rather than raising when ``cov`` is not
    numerically positive definite, so callers check the factor; see
    :func:`_refit_proposal`.
    """
    return jsl.cholesky(cov, lower=True) * _rgg_scale(d)


# Stan's covariance-regularization weight: a refit counts the proposal in
# use as this many pseudo-draws alongside the warmup positions.
_SHRINKAGE_COUNT = 5


def _refit_proposal(
    welford_cov: Array,
    count: int | Array,
    proposal_cov: Array,
    sigma: Array,
) -> tuple[Array, Array]:
    """Refit the proposal from warmup positions, keeping it positive definite.

    The refit covariance is
    ``(count * welford_cov + 5 * proposal_cov) / (count + 5)``, which
    shrinks the Welford estimate toward the covariance the proposal in use
    assumes. The refit proposal is its RGG-scaled Cholesky factor.

    Parameters
    ----------
    welford_cov : Array
        ``(d, d)`` Welford covariance of the warmup positions so far.
        Treated as zero when ``count < 2``, where Welford reports ``0 / 0``.
    count : int or Array
        Number of warmup positions behind ``welford_cov``.
    proposal_cov : Array
        ``(d, d)`` positive-definite covariance the proposal in use
        assumes, so that ``sigma == chol(proposal_cov) * 2.38 / sqrt(d)``.
    sigma : Array
        ``(d, d)`` lower-triangular Cholesky factor of the proposal in use.

    Returns
    -------
    tuple of Array
        ``(proposal_cov, sigma)`` for the next window, each ``(d, d)``.
        When the refit factor has a non-finite entry, the inputs are
        returned unchanged.

    Notes
    -----
    This is Stan's covariance regularization with the proposal in use,
    rather than ``1e-3 * I``, as the shrinkage target. The refit
    covariance is at least ``5 / (count + 5) * proposal_cov`` in the
    Loewner order, so it is positive definite even when ``welford_cov``
    is singular. That happens when every proposal is rejected, which
    leaves the positions without spread, and when fewer than ``d``
    proposals are accepted, which leaves them spanning fewer than ``d``
    directions. With no spread the refit shrinks the proposal covariance
    by the factor ``5 / (count + 5)``, so the window after a fully
    rejecting one proposes smaller steps. The shrinkage imposes no
    absolute scale: scaling ``welford_cov`` and ``proposal_cov`` by ``c``
    scales the refit covariance by ``c``.
    """
    d = proposal_cov.shape[0]
    welford_cov = jnp.where(count > 1, welford_cov, jnp.zeros_like(welford_cov))
    refit_cov = (count * welford_cov + _SHRINKAGE_COUNT * proposal_cov) / (count + _SHRINKAGE_COUNT)
    refit_sigma = _production_sigma(refit_cov, d)
    finite = jnp.all(jnp.isfinite(refit_sigma))
    return (
        jnp.where(finite, refit_cov, proposal_cov),
        jnp.where(finite, refit_sigma, sigma),
    )


# ---------------------------------------------------------------------------
# Window scheduling
# ---------------------------------------------------------------------------


_MIN_STEPS_PER_WINDOW = 25  # Stan's base adaptation window


def _window_sizes(num_warmup: int, n_windows: int, ratio: float = 2.0) -> list[int]:
    """Split ``num_warmup`` steps into geometrically growing adaptation windows.

    Window ``i`` gets weight ``ratio ** i``; the weights are normalised to
    sum to one and rounded to integer step counts, and the last window
    absorbs the rounding remainder. The schedule starts from a single
    window and adds windows, up to ``n_windows``, while every window of
    the next split still holds at least :data:`_MIN_STEPS_PER_WINDOW`
    (= 25) steps.

    Parameters
    ----------
    num_warmup : int
        Total number of warmup steps.
    n_windows : int
        Maximum number of windows.
    ratio : float
        Growth factor between consecutive window weights, at least 1.

    Returns
    -------
    list of int
        Window sizes in sampling order, summing to ``num_warmup``; empty
        when ``num_warmup <= 0``. The result is the single window
        ``[num_warmup]`` when ``n_windows <= 1`` or when the warmup is too
        short for two windows of 25 steps, which at ``ratio=2`` means
        fewer than 74 steps. That window holds fewer than 25 steps when
        ``num_warmup < 25``.

    Notes
    -----
    Stan-style window adaptation uses growing windows so the first,
    badly mixed window contributes little to the covariance estimate
    while later, well-mixed windows dominate.
    """
    if num_warmup <= 0:
        return []
    sizes = [num_warmup]
    # With ``ratio >= 1`` the first window is the smallest, and adding a
    # window only shrinks it, so the first split that breaks the minimum
    # ends the search.
    for n in range(2, int(n_windows) + 1):
        weights = ratio ** np.arange(n, dtype=float)
        split = np.round(weights / weights.sum() * num_warmup).astype(int)
        split[-1] += num_warmup - int(split.sum())
        if split.min() < _MIN_STEPS_PER_WINDOW:
            break
        sizes = [int(s) for s in split]
    return sizes


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

    Splits ``num_warmup`` into at most ``n_windows`` geometrically
    growing windows (see :func:`_window_sizes`). Each window samples with
    the current proposal Cholesky (initially ``2.38 / sqrt(d) * I``)
    while accumulating Welford statistics on positions; at the window
    boundary, :func:`_refit_proposal` refits the proposal from the
    cumulative Welford state. Returns ``(state, sigma, warmup_positions)``:
    the final random-walk state, the last refit's Cholesky factor (the
    production proposal), and the ``(num_warmup, d)`` warmup positions.

    Welford state is *cumulative* across windows — the geometric
    schedule already downweights the early biased samples without
    needing to discard them.
    """
    d = init_state.shape[0]
    welf_init, welf_update, welf_final = welford_algorithm(is_diagonal_matrix=False)
    sizes = _window_sizes(num_warmup, n_windows)

    proposal_cov = jnp.eye(d)
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
            step,
            (rw_state, welf_state),
            keys,
        )
        window_positions.append(positions)

        welford_cov, count, _ = welf_final(welf_state)
        proposal_cov, sigma = _refit_proposal(welford_cov, count, proposal_cov, sigma)

    if sizes:
        warmup_positions = jnp.concatenate(window_positions, axis=0)
    else:
        warmup_positions = jnp.empty((0, d), dtype=init_state.dtype)
    return rw_state, sigma, warmup_positions


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
    Returns ``(state, sigma, warmup_positions)`` as
    :func:`_adaptive_warmup_fast` does.
    """
    d = init_state.shape[0]
    welf_init, welf_update, welf_final = welford_algorithm(is_diagonal_matrix=False)
    sizes = _window_sizes(num_warmup, n_windows)

    proposal_cov = jnp.eye(d)
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
        welford_cov, count, _ = welf_final(welf_state)
        proposal_cov, sigma = _refit_proposal(welford_cov, count, proposal_cov, sigma)

    if positions:
        warmup_positions = jnp.stack(positions)
    else:
        warmup_positions = jnp.empty((0, d), dtype=init_state.dtype)
    return rw_state, sigma, warmup_positions


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
    positions, infos = run_chain_scan(sampler, state, num_results, key)
    return positions, {
        "acceptance_rate": infos.acceptance_rate,
        "is_accepted": infos.is_accepted,
    }


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


def _fixed_sigma_warmup(
    target_log_prob_fn: Callable[[Array], Array],
    init_state: Array,
    sigma: Array,
    num_warmup: int,
    key: Array,
    *,
    traceable: bool,
) -> tuple[Array, Array | None]:
    """Burn in under a fixed proposal ``sigma``; no adaptation.

    Shared by the ``proposal_cov``-override and ``adapt=False`` branches
    of :func:`_run_one_chain`, which both bypass adaptive warmup and
    simply sample under a fixed sigma. Returns
    ``(init_position, warmup_positions)`` — the last burn-in position to
    start production from, plus the burn-in trace (``None`` when
    ``num_warmup <= 0``, leaving production to start from ``init_state``).
    """
    if num_warmup <= 0:
        return init_state, None
    sample_fn = _sample_chain_fast if traceable else _sample_chain_eager
    warmup_positions, _ = sample_fn(
        target_log_prob_fn,
        init_state,
        sigma,
        num_warmup,
        key,
    )
    return warmup_positions[-1], warmup_positions


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
        # User-supplied sigma bypasses adaptation by design, so burn in
        # under it with the same fixed-sigma loop the ``adapt=False``
        # branch uses rather than running adaptive warmup.
        sigma = proposal_sigma_override
        init_position, warmup_positions = _fixed_sigma_warmup(
            target_log_prob_fn,
            init_state,
            sigma,
            num_warmup,
            warmup_key,
            traceable=traceable,
        )
    elif adapt and num_warmup > 0:
        warmup_fn = _adaptive_warmup_fast if traceable else _adaptive_warmup_eager
        rw_state, sigma, warmup_positions = warmup_fn(
            target_log_prob_fn,
            init_state,
            warmup_key,
            num_warmup,
            n_windows=n_windows,
        )
        init_position = rw_state.position
    else:
        sigma = jnp.eye(d) * step_size
        init_position, warmup_positions = _fixed_sigma_warmup(
            target_log_prob_fn,
            init_state,
            sigma,
            num_warmup,
            warmup_key,
            traceable=traceable,
        )

    sample_fn = _sample_chain_fast if traceable else _sample_chain_eager
    chain, sample_stats = sample_fn(
        target_log_prob_fn,
        init_position,
        sigma,
        num_results,
        sample_key,
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
                target_log_prob_fn,
                init_state,
                chain_key,
                num_results=num_results,
                num_warmup=num_warmup,
                adapt=adapt,
                step_size=step_size,
                proposal_sigma_override=proposal_sigma_override,
                n_windows=n_windows,
                traceable=True,
            )

        chains_arr, warmups_arr, stats_arr = parallel_chain_map(run_one, chain_keys)
        chains = [chains_arr[c] for c in range(num_chains)]
        warmups = (
            [warmups_arr[c] for c in range(num_chains)]
            if warmups_arr is not None and warmups_arr.shape[0] == num_chains
            else None
        )
        sample_stats = {k: np.asarray(v) for k, v in stats_arr.items()}
    else:
        chains_l: list[Array] = []
        warmups_l: list[Array] = []
        accept_l: list[np.ndarray] = []
        is_acc_l: list[np.ndarray] = []
        for chain_key in chain_keys:
            ch, wm, st = _run_one_chain(
                target_log_prob_fn,
                init_state,
                chain_key,
                num_results=num_results,
                num_warmup=num_warmup,
                adapt=adapt,
                step_size=step_size,
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
# Inference entry point
# ---------------------------------------------------------------------------


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
        Diagonal proposal scale used when ``proposal_cov=None`` and
        either ``adapt=False`` or ``num_warmup == 0``.
    adapt
        When ``True`` (default), runs a window-style adaptive warmup:
        geometrically growing windows that each sample with the current
        proposal Cholesky and accumulate Welford statistics on
        positions, refitting the proposal at every window boundary (see
        Notes). Production samples with
        ``proposal = chol(Sigma) * 2.38 / sqrt(d)``, which applies the
        Roberts-Gelman-Gilks scaling to the last refit covariance
        ``Sigma``. When ``False``, skips adaptation and uses
        ``sigma = step_size * I`` throughout.
    n_windows
        Maximum number of geometric warmup windows when ``adapt=True``.
        Windows are added only while each holds at least 25 steps, so a
        warmup shorter than 74 steps runs as a single window: a fixed
        RGG-scaled identity proposal throughout, refit once at the end.
        Default ``4``; ``n_windows <= 1`` always gives the single window.
        Ignored when ``adapt=False``.
    proposal_cov
        Explicit ``(d, d)`` proposal Cholesky factor, where ``d`` is the
        target dimension. Overrides both the adaptive fit and
        ``step_size``. Useful when the user has a precomputed covariance
        estimate from elsewhere. A wrong-shape matrix raises
        ``ValueError``.
    init
        Initial chain state. Resolved by
        :func:`~probpipe.inference._inference_utils.get_init_state`
        when ``None``.
    random_seed
        Seed for chain initialisation, warmup, and sampling RNG.

    Returns
    -------
    ApproximateDistribution
        Posterior samples with chain structure and an annotations
        ArviZ-shaped ``DataTree`` carrying per-step acceptance stats
        and warmup positions.

    Raises
    ------
    TypeError
        If ``dist`` does not implement ``SupportsUnnormalizedLogProb``.
    ValueError
        If ``proposal_cov`` is not a ``(d, d)`` matrix, or if ``init`` is
        ``None`` and no initial state can be derived from ``dist``.

    Warns
    -----
    UserWarning
        If ``adapt=True`` and ``num_warmup == 0`` without ``proposal_cov``.
        There are then no warmup positions to adapt on, so the proposal is
        ``sigma = step_size * I``.

    Notes
    -----
    At each window boundary the adaptive warmup refits the proposal
    covariance as ``(n * Sigma_hat + 5 * Sigma_prev) / (n + 5)``, where
    ``Sigma_hat`` is the Welford covariance of the ``n`` warmup positions
    so far and ``Sigma_prev`` is the covariance the proposal in use
    assumes, which is the identity before the first refit. This is Stan's
    covariance regularization with ``Sigma_prev`` in place of
    ``1e-3 * I``. The refit stays positive definite when the positions
    have no spread, as when every proposal in a window is rejected, and
    when they span fewer than ``d`` directions. The proposal therefore
    never collapses to zero, and the correction vanishes as ``n`` grows.
    The growing windows downweight the early, badly mixed positions by
    giving later windows more steps.
    """
    if not isinstance(dist, SupportsUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support log_prob "
            "(does not implement SupportsUnnormalizedLogProb)"
        )

    # Adaptation needs warmup samples to fit the proposal covariance.
    # With ``num_warmup == 0`` there is nothing to adapt on, so the
    # proposal silently falls back to ``step_size * I`` — warn rather
    # than degrade quietly, since the caller asked for adaptation.
    if adapt and num_warmup == 0 and proposal_cov is None:
        warnings.warn(
            "rwmh(adapt=True) with num_warmup=0 cannot fit a proposal "
            "covariance; falling back to sigma = step_size * I. Pass "
            "num_warmup > 0 to adapt, or adapt=False to silence this "
            "warning.",
            stacklevel=2,
        )

    if log_prob_fn is not None and data is not None:

        def target_log_prob(params):
            return dist._unnormalized_log_prob(params) + log_prob_fn(params, data)
    else:

        def target_log_prob(params):
            return dist._unnormalized_log_prob(params)

    init_state = get_init_state(dist, init, random_seed=random_seed)
    proposal_sigma_override = None
    if proposal_cov is not None:
        proposal_sigma_override = jnp.asarray(proposal_cov)
        d = init_state.shape[0]
        if proposal_sigma_override.shape != (d, d):
            raise ValueError(
                f"proposal_cov must be a square ({d}, {d}) matrix matching the "
                f"{d}-dimensional target; got shape "
                f"{tuple(proposal_sigma_override.shape)}."
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

    annotations = build_mcmc_datatree(chains, sample_stats, warmup_chains=warmups)
    event_template = extract_event_template(dist)
    return make_posterior(
        chains,
        parents=(dist,),
        algorithm="blackjax_rwmh",
        annotations=annotations,
        event_template=event_template,
        num_results=num_results,
        num_warmup=num_warmup,
        num_chains=num_chains,
        step_size=step_size,
        accept_rate=accept_rate,
        adapt=adapt,
        n_windows=n_windows,
    )


# ---------------------------------------------------------------------------
# Registry method
# ---------------------------------------------------------------------------


class BlackJAXRWMHMethod(InferenceMethod):
    """Gradient-free RWMH on top of BlackJAX's ``normal_random_walk``.

    Registered as ``blackjax_rwmh`` at priority 55. Applies to any target
    whose prior satisfies ``SupportsUnnormalizedLogProb``, JAX-traceable or
    not, so automatic selection reaches it when no gradient-based method
    passes ``check()``.

    Notes
    -----
    Slow per effective sample in high dimensions even when tuned, so it ranks
    below every gradient-based method.
    """

    @property
    def name(self) -> str:
        return "blackjax_rwmh"

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    @property
    def priority(self) -> int:
        return 55

    def check(self, dist: Any, observed: Any, **kwargs: Any) -> Feasibility:
        prior = get_prior(dist)
        if not isinstance(prior, SupportsUnnormalizedLogProb):
            return Feasibility(
                feasible=False,
                description="Requires SupportsUnnormalizedLogProb",
            )
        if observed is not None and isinstance(observed, dict):
            return Feasibility(
                feasible=False,
                description="Does not support dict-based conditioning",
            )
        return Feasibility(feasible=True)

    def execute(self, dist: Any, observed: Any, **kwargs: Any) -> ApproximateDistribution:
        prior = get_prior(dist)
        log_prob_fn = None
        if is_simple_model(dist):
            lik = dist._likelihood

            def log_prob_fn(params, d):
                return lik.log_likelihood(params=params, data=d)

        random_seed = kwargs.get("random_seed", 0)
        init = kwargs.get("init")
        if init is None:
            init = get_init_state(dist, None, random_seed=random_seed)

        return rwmh(
            prior,
            observed,
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
