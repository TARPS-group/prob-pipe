"""Simulation-based calibration (SBC) and interval coverage for method validation.

Where :mod:`probpipe.validation._comparison` scores an approximation against a
*reference*, these check an inference *method* for self-consistency with the
model that generated the data:

- :func:`simulation_based_calibration` (Talts et al. 2018) draws ``θ★ ~ prior``,
  ``y ~ p(·|θ★)``, and the posterior ``π(θ|y)``; the rank of each ``θ★``
  component among the posterior draws is Uniform on ``{0, …, L}`` iff the
  posterior is calibrated. Non-uniform ranks expose a biased / mis-tuned sampler.
- :func:`interval_coverage` is the companion frequentist check — does a central
  credible interval contain the truth at its nominal rate?

These orchestrate inference through a **Python loop over** :func:`condition_on`,
so they work for every backend (blackjax, Stan, PyMC, …); they are therefore not
themselves jit-compatible, though the per-fit MCMC inside the loop is
JAX-accelerated where the backend allows. The model must expose a sampleable
``prior``, a ``likelihood`` with ``generate_data``, and be conditionable — e.g. a
:class:`~probpipe.SimpleModel` built with a :class:`~probpipe.GLMLikelihood`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .._utils import _auto_key
from ..core.ops import condition_on
from ..custom_types import Array, ArrayLike, PRNGKey
from ._predictive_check import _supports_key_arg

__all__ = ["SBCResult", "interval_coverage", "simulation_based_calibration"]


# -- helpers ----------------------------------------------------------------


def _flatten_point(point: Any, fields: tuple[str, ...] | None) -> Array:
    """Flatten one parameter draw to the 1-D layout of ``flat_samples``.

    The single-draw analogue of :attr:`flat_samples`: ``prior._sample`` returns a
    bare array for a single-field prior (raveled directly) or a ``Record`` for a
    multi-field prior (fields raveled and concatenated in posterior field order).
    """
    if hasattr(point, "fields"):
        return jnp.concatenate([jnp.ravel(jnp.asarray(point[f])) for f in fields])
    return jnp.ravel(jnp.asarray(point))


def _ranks(draws: Array, point: Array) -> Array:
    """SBC rank of each component of *point*: ``#{draws < point}``, in ``{0, …, n}``.

    Uses strict ``<`` (ties, measure-zero for a continuous posterior, count as
    not-below). ``draws`` is ``(n, d)``; ``point`` is ``(d,)``; returns ``(d,)``.
    """
    return jnp.sum(draws < point[None, :], axis=0)


def _component_names(posterior: Any) -> tuple[str, ...] | None:
    """Per-flattened-component parameter names matching the ``flat_samples`` columns.

    A scalar field keeps its name; a field with ``k > 1`` flattened components
    becomes ``field[0] … field[k-1]`` (row-major), in posterior field order.
    Returns ``None`` if the posterior exposes neither ``fields`` nor
    ``event_shapes``.
    """
    fields = getattr(posterior, "fields", None)
    shapes = getattr(posterior, "event_shapes", None)
    if not fields or shapes is None:
        return None
    names: list[str] = []
    for f in fields:
        size = int(np.prod(shapes[f], dtype=int))
        names.extend([f] if size == 1 else [f"{f}[{i}]" for i in range(size)])
    return tuple(names)


def _kolmogorov_sf(d: float, n: int) -> float:
    """Asymptotic survival function of the one-sample KS statistic.

    ``P(D_n ≥ d)`` under the null, via the Kolmogorov limit
    ``Q_KS(λ) = 2 Σ_k (−1)^{k−1} e^{−2k²λ²}`` with Stephens' (1970) small-sample
    correction ``λ = (√n + 0.12 + 0.11/√n) d``. Dependency-free (no SciPy);
    accurate for the sample sizes SBC uses (``n ≳ 30``).
    """
    if d <= 0.0:
        return 1.0
    en = np.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * d
    k = np.arange(1, 101)  # series decays as e^{-2k²λ²}; 100 terms is far more than enough
    p = 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * (k * lam) ** 2))
    return float(min(max(p, 0.0), 1.0))


def _ks_uniform(ranks: np.ndarray, num_draws: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-parameter KS distance + p-value of the ranks vs ``Uniform{0..L}``.

    ``ranks`` is ``(num_simulations, num_params)`` in ``{0, …, num_draws}``; the
    ranks are mapped to ``(0, 1)`` via the midpoint ``(r + 0.5)/(L + 1)`` and
    compared to ``Uniform[0, 1]``.
    """
    u = (ranks + 0.5) / (num_draws + 1)
    v = np.sort(u, axis=0)
    s = v.shape[0]
    i_over_s = (np.arange(1, s + 1) / s)[:, None]
    im1_over_s = (np.arange(0, s) / s)[:, None]
    d = np.maximum(np.max(i_over_s - v, axis=0), np.max(v - im1_over_s, axis=0))
    pvals = np.array([_kolmogorov_sf(float(d[j]), s) for j in range(d.shape[0])])
    return d, pvals


# -- simulation-based calibration -------------------------------------------


@dataclass(frozen=True)
class SBCResult:
    """Result of :func:`simulation_based_calibration`.

    Attributes
    ----------
    ranks : np.ndarray
        Integer ranks of ``θ★`` among the posterior draws, shape
        ``(num_simulations, num_params)``, each in ``{0, …, num_posterior_draws}``.
        The rank counts draws strictly below ``θ★``; this assumes continuous
        draws (ties are measure-zero for a continuous posterior, but would bias
        ranks low for a discrete-valued parameter).
    num_posterior_draws : int
        ``L`` — the rank upper bound (the actual number of posterior draws used).
    param_names : tuple[str, ...] or None
        Per-flattened-component names aligned with the columns of ``ranks`` /
        ``ks_*`` (``field`` for a scalar field, ``field[i]`` otherwise), in
        posterior field order; ``None`` if the posterior exposes no field names.
    ks_statistic : np.ndarray
        Per-parameter KS distance of the normalized ranks from ``Uniform[0, 1]``.
    ks_pvalue : np.ndarray
        Per-parameter KS p-value against ``Uniform[0, 1]``; small ⇒ ranks are
        non-uniform ⇒ miscalibrated. This is a diagnostic — inspect it (and the
        rank histogram) and apply your own threshold, with a multiple-comparison
        correction across parameters, rather than reading off a pass/fail verdict.
    """

    ranks: np.ndarray
    num_posterior_draws: int
    param_names: tuple[str, ...] | None
    ks_statistic: np.ndarray
    ks_pvalue: np.ndarray

    def rank_histogram(self, num_bins: int = 20) -> np.ndarray:
        """Rank histogram per parameter, shape ``(num_params, num_bins)``.

        Bins the normalized ranks into ``num_bins`` equal-width bins over
        ``[0, 1]``; a calibrated method gives approximately flat histograms.
        """
        u = (self.ranks + 0.5) / (self.num_posterior_draws + 1)
        edges = np.linspace(0.0, 1.0, num_bins + 1)
        return np.stack([np.histogram(u[:, j], bins=edges)[0] for j in range(u.shape[1])])


def simulation_based_calibration(
    model: Any,
    *,
    num_simulations: int,
    num_posterior_draws: int,
    num_observations: int,
    method: str | None = None,
    key: PRNGKey | None = None,
    **infer_kwargs: Any,
) -> SBCResult:
    """Simulation-based calibration of an inference method (Talts et al. 2018).

    For each of ``num_simulations`` draws: sample ``θ★`` from ``model.prior``,
    generate ``y`` from ``model.likelihood.generate_data(θ★, num_observations)``,
    fit the posterior with :func:`condition_on`, and record the rank of each
    flattened ``θ★`` component among the ``num_posterior_draws`` posterior draws.
    Under correct calibration each rank is ``Uniform{0, …, L}``; the per-parameter
    KS distance from uniform and its p-value summarize the fit.

    Parameters
    ----------
    model
        A conditionable generative model — sampleable ``prior``, a ``likelihood``
        with ``generate_data``, and usable with :func:`condition_on` (e.g.
        ``SimpleModel(prior, GLMLikelihood(...))``).
    num_simulations
        Number of ``(θ★, y, posterior)`` replications.
    num_posterior_draws
        Posterior draws per fit (``num_results`` passed to :func:`condition_on`).
    num_observations
        Observations per generated dataset.
    method
        Inference method name for :func:`condition_on` (``None`` = auto-select).
    key
        JAX PRNG key (auto-generated if ``None``). Controls θ★, the data, and the
        per-fit MCMC seed, so a fixed key makes the whole run reproducible. Do not
        also pass ``random_seed`` in ``infer_kwargs``.
    **infer_kwargs
        Extra keyword arguments forwarded to :func:`condition_on` (e.g.
        ``num_warmup``, ``num_chains``).

    Returns
    -------
    SBCResult
    """
    if num_simulations < 1:
        raise ValueError(f"num_simulations must be >= 1, got {num_simulations}")
    if "random_seed" in infer_kwargs:
        raise ValueError(
            "simulation_based_calibration manages the per-fit random_seed; "
            "do not pass random_seed in infer_kwargs"
        )
    if key is None:
        key = _auto_key()
    prior = model.prior
    generate = model.likelihood.generate_data
    gen_takes_key = _supports_key_arg(model.likelihood)

    rank_rows: list[np.ndarray] = []
    fields: tuple[str, ...] | None = None
    component_names: tuple[str, ...] | None = None
    draws = None
    for _ in range(num_simulations):
        key, k_theta, k_data, k_mcmc = jax.random.split(key, 4)
        theta_star = prior._sample(k_theta, ())
        if gen_takes_key:
            y = generate(theta_star, num_observations, key=k_data)
        else:
            y = generate(theta_star, num_observations)
        seed = int(jax.random.randint(k_mcmc, (), 0, 2_000_000_000))
        posterior = condition_on(
            model,
            y,
            method=method,
            num_results=num_posterior_draws,
            random_seed=seed,
            **infer_kwargs,
        )
        draws = jnp.asarray(posterior.flat_samples)  # (L, p)
        if fields is None:
            fields = getattr(posterior, "fields", None)
            component_names = _component_names(posterior)
        theta_flat = _flatten_point(theta_star, fields)  # (p,)
        rank_rows.append(np.asarray(_ranks(draws, theta_flat)))

    ranks = np.stack(rank_rows).astype(int)  # (num_simulations, p)
    num_draws = int(draws.shape[0])  # actual total draws (handles multi-chain)
    ks_stat, ks_pvalue = _ks_uniform(ranks, num_draws)
    return SBCResult(
        ranks=ranks,
        num_posterior_draws=num_draws,
        param_names=component_names,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pvalue,
    )


# -- interval coverage ------------------------------------------------------


def interval_coverage(
    draws_or_dist: Any,
    truth: ArrayLike,
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> dict[float, Array]:
    """Central-credible-interval coverage of *truth*, per parameter.

    For each ``level``, checks whether each component of *truth* lies in the
    central ``level`` interval ``[q_{(1−level)/2}, q_{(1+level)/2}]`` of the
    per-parameter posterior. Returns ``{level: covered}`` with ``covered`` a
    boolean array over parameters. Averaging the indicators over many
    ``(posterior, truth)`` pairs gives the frequentist coverage, which matches the
    nominal level for a calibrated method.

    *draws_or_dist* is an ``(n, d)`` (or 1-D ``(n,)``) array of draws or a
    distribution exposing ``flat_samples`` (treated as equally weighted, as
    MCMC draws are); *truth* is the matching ``(d,)``.
    """
    draws = jnp.asarray(getattr(draws_or_dist, "flat_samples", draws_or_dist))
    if draws.ndim == 1:
        draws = draws[:, None]
    truth = jnp.atleast_1d(jnp.asarray(truth))
    if truth.shape[0] != draws.shape[1]:
        raise ValueError(f"truth dimension {truth.shape[0]} != draws dimension {draws.shape[1]}")
    out: dict[float, Array] = {}
    for level in levels:
        lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
        q = jnp.quantile(draws, jnp.array([lo, hi]), axis=0)  # (2, d)
        out[float(level)] = (truth >= q[0]) & (truth <= q[1])
    return out
