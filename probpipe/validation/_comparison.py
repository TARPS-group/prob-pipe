"""Posterior-vs-reference comparison metrics for method validation.

These score a posterior *approximation* against a trusted *reference* (analytic,
long-NUTS, or sandwich) — answering "does this method recover the right
posterior?", as opposed to per-fit convergence diagnostics (ESS, R-hat), which
are tracked separately in #193. They are plain, dependency-light JAX functions
used by the inference test suite and the ``probpipe-benchmark`` harness.

The approximation under test is consumed as draws (anything exposing
``flat_samples``, or a raw ``(n, d)`` array); the moment metrics use its sample
mean and covariance rather than any analytic moments it may expose.

Three metric families, by what the reference must carry (see :class:`Reference`):

- **moment** — ``standardized_mean_error`` and ``relative_cov_error`` need the
  reference's high-precision ``(mean, cov)``;
- **sample** — ``sliced_wasserstein`` and ``mmd`` need reference *draws*;
- **score** — ``ksd`` needs only the target score ``∇ log π`` (no reference
  draws).

These are deliberately plain functions rather than ``@workflow_function`` ops:
they return bare arrays (or, for :func:`score_posterior`, a ``dict``), which the
workflow-function output-wrapping would otherwise coerce into single-field
records.

All metrics return 0-d (or, for ``std_ratios``, 1-d) JAX arrays and are
jit-compatible. The moment metrics Cholesky-factor ``Σ_ref`` and so require it
to be positive definite; a non-PD reference covariance yields NaN rather than
raising.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..custom_types import Array, PRNGKey

__all__ = [
    "Reference",
    "ksd",
    "mmd",
    "relative_cov_error",
    "score_posterior",
    "sliced_wasserstein",
    "standardized_mean_error",
    "std_ratios",
]


# -- coercion + moment helpers ---------------------------------------------


def _as_draws(x: Any) -> Array:
    """Coerce an approximation / reference to an ``(n, d)`` draws matrix.

    Accepts an ``ApproximateDistribution`` / empirical (anything exposing
    ``flat_samples``) or a raw array; a 1-D array of ``n`` scalars becomes
    ``(n, 1)``.
    """
    arr = jnp.asarray(getattr(x, "flat_samples", x))
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"expected (n, d) draws, got shape {arr.shape}")
    return arr


def _sample_cov(d: Array) -> Array:
    """Unbiased sample covariance of ``(n, d)`` draws → ``(d, d)``."""
    centered = d - d.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / (d.shape[0] - 1)


class _MissingReference(ValueError):
    """A metric's required reference pieces are absent from the :class:`Reference`."""


@dataclass(frozen=True)
class Reference:
    """A reference posterior to score an approximation against.

    Carries whichever pieces are available so analytic, long-NUTS, and sandwich
    references share one interface: high-precision moments ``(mean, cov)`` (for
    the moment metrics), ``draws`` (for the sample-based distances), and a
    ``score_fn`` = ``∇ log π`` (for :func:`ksd`).

    Parameters
    ----------
    mean : Array or None
        Reference mean ``μ_ref``, shape ``(d,)``.
    cov : Array or None
        Reference covariance ``Σ_ref``, shape ``(d, d)``.
    draws : Array or None
        Reference draws, shape ``(n_ref, d)``.
    score_fn : callable or None
        ``θ ↦ ∇ log π(θ)`` for a ``(d,)`` input.
    """

    mean: Array | None = None
    cov: Array | None = None
    draws: Array | None = None
    score_fn: Callable[[Array], Array] | None = None

    @classmethod
    def from_draws(
        cls, draws: Any, *, score_fn: Callable[[Array], Array] | None = None
    ) -> Reference:
        """Build a reference from draws, deriving ``(mean, cov)`` from them."""
        d = _as_draws(draws)
        return cls(mean=d.mean(axis=0), cov=_sample_cov(d), draws=d, score_fn=score_fn)

    @classmethod
    def from_moments(
        cls,
        mean: Array,
        cov: Array,
        *,
        draws: Any | None = None,
        score_fn: Callable[[Array], Array] | None = None,
    ) -> Reference:
        """Build a reference from high-precision analytic/empirical moments."""
        return cls(
            mean=jnp.asarray(mean),
            cov=jnp.asarray(cov),
            draws=None if draws is None else _as_draws(draws),
            score_fn=score_fn,
        )


def _require(ref: Reference, *names: str) -> None:
    """Raise :class:`_MissingReference` if *ref* is missing any named piece."""
    missing = [n for n in names if getattr(ref, n) is None]
    if missing:
        present = [k for k in ("mean", "cov", "draws", "score_fn") if getattr(ref, k) is not None]
        raise _MissingReference(
            f"this metric needs reference {', '.join(missing)}; "
            f"the Reference carries only {present}"
        )


# -- moment metrics ---------------------------------------------------------


def standardized_mean_error(approx: Any, ref: Reference) -> Array:
    r"""Mean error in posterior-standard-deviation units: ``‖Σ_ref^{-1/2}(μ̂ − μ_ref)‖₂``.

    The Mahalanobis distance between the approximate mean ``μ̂`` and the
    reference mean ``μ_ref`` under ``Σ_ref`` — scale- and rotation-invariant,
    reported in units of reference posterior standard deviations. Computed as
    ``‖L⁻¹(μ̂ − μ_ref)‖₂`` with ``L = chol(Σ_ref)``.
    """
    _require(ref, "mean", "cov")
    mu_hat = _as_draws(approx).mean(axis=0)
    diff = mu_hat - ref.mean
    chol = jnp.linalg.cholesky(ref.cov)
    z = jax.scipy.linalg.solve_triangular(chol, diff, lower=True)
    return jnp.linalg.norm(z)


def relative_cov_error(approx: Any, ref: Reference) -> Array:
    r"""Operator-norm covariance error: ``‖I − Σ_ref⁻¹ Σ̂‖₂``.

    The spectral norm (largest singular value) of the whitened covariance
    deviation from identity — ``0`` iff ``Σ̂ = Σ_ref``, and it bounds the
    worst-direction variance-ratio error. ``Σ_ref⁻¹ Σ̂`` is formed by a Cholesky
    solve.
    """
    _require(ref, "cov")
    cov_hat = _sample_cov(_as_draws(approx))
    chol = jnp.linalg.cholesky(ref.cov)
    whitened = jax.scipy.linalg.cho_solve((chol, True), cov_hat)  # Σ_ref⁻¹ Σ̂
    deviation = jnp.eye(whitened.shape[0]) - whitened
    return jnp.linalg.norm(deviation, ord=2)


def std_ratios(approx: Any, ref: Reference) -> Array:
    r"""Per-coordinate standard-deviation ratios ``σ̂_d / σ_{ref,d}`` (a ``(d,)`` array).

    A coordinate-wise readout for reporting; the full-covariance
    :func:`relative_cov_error` subsumes it. A reference coordinate with zero
    variance yields ``inf`` for that ratio.
    """
    _require(ref, "cov")
    var_hat = jnp.diag(_sample_cov(_as_draws(approx)))
    var_ref = jnp.diag(ref.cov)
    return jnp.sqrt(var_hat / var_ref)


# -- sample-based distances -------------------------------------------------


def sliced_wasserstein(
    x: Any,
    y: Any,
    *,
    key: PRNGKey,
    n_projections: int = 128,
    n_quantiles: int = 256,
) -> Array:
    r"""Sliced 2-Wasserstein distance between two sets of draws.

    Projects ``x`` and ``y`` onto ``n_projections`` random unit directions,
    computes the 1-D ``W₂`` per projection via a shared quantile grid (so
    unequal sample sizes are handled), and returns
    ``(mean_θ W₂²(P_θ x, P_θ y))^{1/2}``.
    """
    x, y = _as_draws(x), _as_draws(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must share a dimension, got {x.shape[1]} and {y.shape[1]}")
    proj = jax.random.normal(key, (n_projections, x.shape[-1]))
    proj = proj / jnp.linalg.norm(proj, axis=-1, keepdims=True)
    xp, yp = x @ proj.T, y @ proj.T  # (n, P)
    qs = (jnp.arange(n_quantiles) + 0.5) / n_quantiles
    xq = jnp.quantile(xp, qs, axis=0)  # (Q, P)
    yq = jnp.quantile(yp, qs, axis=0)
    w2_sq = jnp.mean((xq - yq) ** 2, axis=0)  # (P,) 1-D W₂² per projection
    return jnp.sqrt(jnp.mean(w2_sq))


def _sq_dists(a: Array, b: Array) -> Array:
    """Pairwise squared Euclidean distances, ``(na, nb)``."""
    return jnp.sum(a**2, axis=1)[:, None] + jnp.sum(b**2, axis=1)[None, :] - 2.0 * a @ b.T


def mmd(x: Any, y: Any, *, bandwidth: float | str = "median") -> Array:
    r"""Unbiased squared MMD with an RBF kernel.

    ``MMD²_u = mean_{i≠j} k(xᵢ,xⱼ) + mean_{i≠j} k(yᵢ,yⱼ) − 2·mean_{i,j} k(xᵢ,yⱼ)``,
    ``k(a,b) = exp(−‖a−b‖² / ℓ²)``. With ``bandwidth="median"`` the length-scale
    ``ℓ²`` is the median cross-pair squared distance (a fixed-shape, jit-safe
    heuristic). The unbiased estimator can be slightly negative.
    """
    x, y = _as_draws(x), _as_draws(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must share a dimension, got {x.shape[1]} and {y.shape[1]}")
    m, n = x.shape[0], y.shape[0]
    if m < 2 or n < 2:
        raise ValueError(f"unbiased MMD needs >= 2 draws per sample, got {m} and {n}")
    dxx, dyy, dxy = _sq_dists(x, x), _sq_dists(y, y), _sq_dists(x, y)
    if bandwidth == "median":
        ell2 = jnp.median(dxy)
        ell2 = jnp.where(ell2 > 0, ell2, 1.0)
    else:
        ell2 = jnp.asarray(bandwidth) ** 2
    kxx, kyy, kxy = jnp.exp(-dxx / ell2), jnp.exp(-dyy / ell2), jnp.exp(-dxy / ell2)
    off_xx = (jnp.sum(kxx) - jnp.trace(kxx)) / (m * (m - 1))
    off_yy = (jnp.sum(kyy) - jnp.trace(kyy)) / (n * (n - 1))
    return off_xx + off_yy - 2.0 * jnp.mean(kxy)


def ksd(
    x: Any,
    score_fn: Callable[[Array], Array],
    *,
    bandwidth: float | str = "median",
) -> Array:
    r"""Kernel Stein discrepancy with the inverse-multiquadric (IMQ) kernel.

    The U-statistic ``KSD = (mean_{i≠j} u_p(xᵢ,xⱼ))^{1/2}`` for the Stein kernel
    ``u_p`` of the IMQ base kernel ``k(a,b) = (c² + ‖a−b‖²)^{-1/2}`` and score
    ``sₚ = ∇ log π``. With ``bandwidth="median"``, ``c²`` is the median
    off-diagonal squared distance. Needs no reference draws — a gradient-based
    goodness-of-fit to the target ``π``; ``→ 0`` as the draws come to match
    ``π``.
    """
    x = _as_draws(x)
    n, d = x.shape
    if n < 2:
        raise ValueError(f"KSD U-statistic needs >= 2 draws, got {n}")
    scores = jax.vmap(score_fn)(x)  # (n, d)
    diff = x[:, None, :] - x[None, :, :]  # (n, n, d)
    sq = jnp.sum(diff**2, axis=-1)  # (n, n)
    if bandwidth == "median":
        offdiag = sq + jnp.diag(jnp.full(n, jnp.nan))
        c2 = jnp.nanmedian(offdiag)
        c2 = jnp.where(c2 > 0, c2, 1.0)
    else:
        c2 = jnp.asarray(bandwidth) ** 2
    beta = -0.5
    base = c2 + sq  # (n, n)
    k_b = base**beta
    k_b1 = base ** (beta - 1.0)
    k_b2 = base ** (beta - 2.0)
    s_dot_s = scores @ scores.T  # (n, n)  sₓ·s_y
    sx_r = jnp.einsum("id,ijd->ij", scores, diff)  # sₓ·(xᵢ−xⱼ)
    sy_r = jnp.einsum("jd,ijd->ij", scores, diff)  # s_y·(xᵢ−xⱼ)
    grad_term = 2.0 * beta * k_b1 * (sy_r - sx_r)
    trace_term = -2.0 * beta * d * k_b1 - 4.0 * beta * (beta - 1.0) * k_b2 * sq
    u = s_dot_s * k_b + grad_term + trace_term  # (n, n)
    ksd_sq = (jnp.sum(u) - jnp.trace(u)) / (n * (n - 1))
    return jnp.sqrt(jnp.maximum(ksd_sq, 0.0))


# -- aggregator -------------------------------------------------------------

_DEFAULT_METRICS = (
    "standardized_mean_error",
    "relative_cov_error",
    "sliced_wasserstein",
    "mmd",
    "ksd",
)


def score_posterior(
    approx: Any,
    reference: Reference,
    *,
    metrics: Sequence[str] = _DEFAULT_METRICS,
    key: PRNGKey | None = None,
) -> dict[str, Array]:
    """Score *approx* against *reference* on the named metrics → a scorecard dict.

    Metrics whose required reference pieces are absent (e.g. ``ksd`` without a
    ``score_fn``, or the sample distances without ``draws``) are skipped rather
    than erroring, so one call serves analytic, long-NUTS, and sandwich
    references. Reused by the test suite and the ``probpipe-benchmark`` harness.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    out: dict[str, Array] = {}
    for name in metrics:
        # The moment metrics ``_require`` their reference pieces and raise
        # ``_MissingReference`` when absent; catch that to skip them. The
        # sample/score metrics take the relevant piece directly, so guard on it.
        try:
            if name == "standardized_mean_error":
                out[name] = standardized_mean_error(approx, reference)
            elif name == "relative_cov_error":
                out[name] = relative_cov_error(approx, reference)
            elif name == "std_ratios":
                out[name] = std_ratios(approx, reference)
            elif name == "sliced_wasserstein":
                if reference.draws is not None:
                    out[name] = sliced_wasserstein(approx, reference.draws, key=key)
            elif name == "mmd":
                if reference.draws is not None:
                    out[name] = mmd(approx, reference.draws)
            elif name == "ksd":
                if reference.score_fn is not None:
                    out[name] = ksd(approx, reference.score_fn)
            else:
                raise ValueError(f"unknown metric {name!r}")
        except _MissingReference:
            continue
    return out
