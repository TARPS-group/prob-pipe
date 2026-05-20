"""MCMC diagnostic workflow functions.

Usage::

    from probpipe.diagnostics import compute_rhat, mcmc_summary

    posterior = condition_on(model, data["y"])
    print(mcmc_summary(posterior))
"""
from __future__ import annotations

import warnings as _warnings

from ..core.distribution import Distribution
from ..core.node import WorkflowFunction
from ._arviz_bridge import (
    check_arviz_installed,
    extract_draws,
    to_arviz_dataset,
    build_warnings,
)

__all__ = [
    "compute_rhat",
    "compute_ess",
    "compute_mcse",
    "mcmc_summary",
    "plot_trace",
    "plot_rank",
    "plot_kde",
    "fit_kde",
]


def _axes_to_fig(axes):
    """Robustly extract a Figure from whatever az.plot_* returns.

    ArviZ plot functions return one of:
      - a 2-D ndarray of Axes  (plot_trace: shape (n_vars, 2))
      - a 1-D ndarray of Axes  (plot_rank, plot_posterior: shape (n_vars,))
      - a bare Axes object      (any of the above with n_vars == 1)
    """
    import matplotlib.axes
    if isinstance(axes, matplotlib.axes.Axes):
        return axes.get_figure()
    # ndarray — ravel to 1-D then take first element
    flat = axes.ravel()
    return flat[0].get_figure()


def _scalar(da) -> float:
    """Extract a Python float from an ArviZ diagnostic DataArray.

    ArviZ returns per-variable diagnostics as DataArrays. When the
    posterior variable was stored with a trailing event dimension (e.g.
    shape ``(chain, draw, 1)`` for a scalar parameter), the diagnostic
    result has shape ``(1,)`` rather than ``()``. Squeezing first
    handles both cases uniformly.
    """
    import numpy as np
    return float(np.asarray(da, dtype=float).squeeze().item())


# ── compute_rhat ──────────────────────────────────────────────────────────────

def _compute_rhat_impl(posterior: Distribution) -> dict:
    """Compute R-hat (Gelman-Rubin) convergence statistic.

    R-hat measures whether multiple MCMC chains have converged to the
    same distribution. Values close to 1.0 indicate good mixing;
    values above 1.01 indicate the chains have not yet converged.

    .. note::
        R-hat requires at least 2 chains for a meaningful result.
        ``condition_on`` defaults to a single chain — run with
        ``num_chains >= 2`` for reliable R-hat estimates.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.

    Returns
    -------
    dict
        ``{"rhat": {"intercept": 1.002, "slope": 1.001, ...}}``
    """
    check_arviz_installed()
    import arviz as az

    dataset  = to_arviz_dataset(posterior)
    n_chains = dataset.dims.get("chain", 1)

    if n_chains < 2:
        _warnings.warn(
            "R-hat requires at least 2 chains for a meaningful result. "
            "Run condition_on with num_chains >= 2.",
            stacklevel=4,
        )

    rhat_ds = az.rhat(dataset)
    return {
        "rhat": {
            var: _scalar(rhat_ds[var])
            for var in rhat_ds.data_vars
        }
    }


compute_rhat = WorkflowFunction(
    func=_compute_rhat_impl, name="compute_rhat"
)


# ── compute_ess ───────────────────────────────────────────────────────────────

def _compute_ess_impl(posterior: Distribution) -> dict:
    """Compute Effective Sample Size (bulk and tail).

    ESS estimates how many independent samples the chain is equivalent
    to, after accounting for autocorrelation. Low ESS (< 400) means
    the sampler is inefficient and estimates may be unreliable.

    - **Bulk ESS** — reliability of central tendency estimates.
    - **Tail ESS** — reliability of tail quantile estimates.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.

    Returns
    -------
    dict
        ``{"ess_bulk": {"intercept": 950.0, ...},``
        ``"ess_tail":  {"intercept": 910.0, ...}}``
    """
    check_arviz_installed()
    import arviz as az

    dataset  = to_arviz_dataset(posterior)
    ess_bulk = az.ess(dataset, method="bulk")
    ess_tail = az.ess(dataset, method="tail")

    return {
        "ess_bulk": {v: _scalar(ess_bulk[v]) for v in ess_bulk.data_vars},
        "ess_tail": {v: _scalar(ess_tail[v]) for v in ess_tail.data_vars},
    }


compute_ess = WorkflowFunction(
    func=_compute_ess_impl, name="compute_ess"
)


# ── compute_mcse ──────────────────────────────────────────────────────────────

def _compute_mcse_impl(posterior: Distribution) -> dict:
    """Compute Monte Carlo Standard Error (mean and sd).

    MCSE estimates the numerical accuracy of posterior summary
    statistics. A small MCSE relative to the posterior standard
    deviation indicates reliable estimates.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.

    Returns
    -------
    dict
        ``{"mcse_mean": {"intercept": 0.003, ...},``
        ``"mcse_sd":    {"intercept": 0.002, ...}}``
    """
    check_arviz_installed()
    import arviz as az

    dataset   = to_arviz_dataset(posterior)
    mcse_mean = az.mcse(dataset, method="mean")
    mcse_sd   = az.mcse(dataset, method="sd")

    return {
        "mcse_mean": {v: _scalar(mcse_mean[v]) for v in mcse_mean.data_vars},
        "mcse_sd":   {v: _scalar(mcse_sd[v])   for v in mcse_sd.data_vars},
    }


compute_mcse = WorkflowFunction(
    func=_compute_mcse_impl, name="compute_mcse"
)


# ── mcmc_summary ──────────────────────────────────────────────────────────────

def _mcmc_summary_impl(posterior: Distribution) -> dict:
    """Compute a full MCMC diagnostic summary in a single pass.

    Runs R-hat, ESS (bulk + tail), and MCSE together — sharing a
    single ArviZ dataset conversion — and attaches human-readable
    warnings for any values outside recommended thresholds.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.

    Returns
    -------
    dict
        ::

            {
                "rhat":      {"intercept": 1.002, "slope": 1.001},
                "ess_bulk":  {"intercept": 950.0, "slope": 987.0},
                "ess_tail":  {"intercept": 910.0, "slope": 942.0},
                "mcse_mean": {"intercept": 0.003, "slope": 0.002},
                "mcse_sd":   {"intercept": 0.002, "slope": 0.001},
                "warnings":  [],
            }

        An empty ``"warnings"`` list means all diagnostics passed.
    """
    check_arviz_installed()
    import arviz as az

    # Single dataset conversion — reused across all diagnostics
    dataset = to_arviz_dataset(posterior)

    rhat_ds   = az.rhat(dataset)
    ess_bulk  = az.ess(dataset, method="bulk")
    ess_tail  = az.ess(dataset, method="tail")
    mcse_mean = az.mcse(dataset, method="mean")
    mcse_sd   = az.mcse(dataset, method="sd")

    rhat   = {v: _scalar(rhat_ds[v])   for v in rhat_ds.data_vars}
    bulk   = {v: _scalar(ess_bulk[v])   for v in ess_bulk.data_vars}
    tail   = {v: _scalar(ess_tail[v])   for v in ess_tail.data_vars}
    m_mean = {v: _scalar(mcse_mean[v])  for v in mcse_mean.data_vars}
    m_sd   = {v: _scalar(mcse_sd[v])    for v in mcse_sd.data_vars}

    return {
        "rhat":      rhat,
        "ess_bulk":  bulk,
        "ess_tail":  tail,
        "mcse_mean": m_mean,
        "mcse_sd":   m_sd,
        "warnings":  build_warnings(rhat, bulk, tail),
    }


mcmc_summary = WorkflowFunction(
    func=_mcmc_summary_impl, name="mcmc_summary"
)


# ── plot_trace ────────────────────────────────────────────────────────────────

def _plot_trace_impl(
    posterior: Distribution,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
) -> dict:
    """Generate trace plots for MCMC chains.

    A trace plot shows sampled parameter values over iterations.
    Well-mixed chains look like horizontal fuzzy caterpillars with no
    visible trend. Trends, drift, or chains stuck in different regions
    all indicate convergence problems.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.
    figsize : tuple[int, int] or None
        Figure size passed to matplotlib. Defaults to
        ``(10, 2 * n_params)``.
    title : str or None
        Optional figure title.

    Returns
    -------
    dict
        ``{"fig": matplotlib.figure.Figure}``
    """
    check_arviz_installed()
    import arviz as az

    dataset  = to_arviz_dataset(posterior)
    n_params = len(list(dataset.data_vars))

    fs   = figsize or (10, 2 * n_params)
    axes = az.plot_trace(dataset, figsize=fs)

    fig = _axes_to_fig(axes)
    fig.suptitle(title or "Trace Plot", y=1.02, fontsize=12)
    fig.tight_layout()

    return {"fig": fig}


plot_trace = WorkflowFunction(
    func=_plot_trace_impl, name="plot_trace"
)


# ── plot_rank ─────────────────────────────────────────────────────────────────

def _plot_rank_impl(
    posterior: Distribution,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
) -> dict:
    """Generate rank plots for MCMC chains.

    A rank plot shows the distribution of ranks of each chain's
    samples relative to the pooled samples across all chains. Uniform
    rank distributions indicate good mixing. Rank plots are more
    sensitive than trace plots for detecting subtle convergence
    failures.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.
    figsize : tuple[int, int] or None
        Figure size passed to matplotlib. Defaults to
        ``(10, 2 * n_params)``.
    title : str or None
        Optional figure title.

    Returns
    -------
    dict
        ``{"fig": matplotlib.figure.Figure}``
    """
    check_arviz_installed()
    import arviz as az

    dataset  = to_arviz_dataset(posterior)
    n_params = len(list(dataset.data_vars))

    fs   = figsize or (10, 2 * n_params)
    axes = az.plot_rank(dataset, figsize=fs)

    fig = _axes_to_fig(axes)
    fig.suptitle(title or "Rank Plot", y=1.02, fontsize=12)
    fig.tight_layout()

    return {"fig": fig}


plot_rank = WorkflowFunction(
    func=_plot_rank_impl, name="plot_rank"
)


# ── fit_kde ───────────────────────────────────────────────────────────────────

def _fit_kde_impl(
    posterior: Distribution,
    *,
    bandwidth: float | None = None,
) -> dict:
    """Fit a :class:`~probpipe.distributions.KDEDistribution` per scalar parameter.

    Pools MCMC draws across all chains and fits one
    :class:`~probpipe.distributions.KDEDistribution` per scalar
    parameter. The returned distributions support ``log_prob``,
    ``sample``, and ``from_distribution`` — they are full ProbPipe
    distribution objects, not just plot helpers.

    Vector-valued parameters are skipped with a warning.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.
    bandwidth : float or None
        Scalar bandwidth forwarded to ``KDEDistribution`` as a
        per-dimension scale. ``None`` (default) applies Silverman's rule.

    Returns
    -------
    dict
        ``{"intercept": KDEDistribution, "slope": KDEDistribution, ...}``
        One entry per scalar parameter, keyed by parameter name.
    """
    import numpy as np
    from ..distributions.kde import KDEDistribution

    raw_draws = extract_draws(posterior)
    kde_distributions: dict = {}

    for var, arr in raw_draws.items():
        arr = np.asarray(arr)
        if arr.ndim > 2:
            _warnings.warn(
                f"fit_kde: skipping '{var}' — vector/matrix parameters are not "
                f"supported (shape {arr.shape[2:]}).",
                stacklevel=4,
            )
            continue
        pooled = arr.reshape(-1).astype(float)
        bw = np.array([bandwidth], dtype=float) if bandwidth is not None else None
        kde_distributions[var] = KDEDistribution(pooled, bandwidth=bw, name=var)

    return kde_distributions


fit_kde = WorkflowFunction(
    func=_fit_kde_impl, name="fit_kde"
)


# ── plot_kde ──────────────────────────────────────────────────────────────────

def _plot_kde_impl(
    posterior: Distribution,
    *,
    bandwidth: float | None = None,
    credible_interval: float = 0.94,
    point_estimate: str = "mean",
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
) -> dict:
    """Generate KDE marginal density plots using :class:`~probpipe.distributions.KDEDistribution`.

    For each scalar parameter, pools the MCMC draws across all chains,
    fits a :class:`~probpipe.distributions.KDEDistribution` (Gaussian
    mixture backed by TFP, with Silverman's rule bandwidth by default),
    and overlays its density curve on ArviZ's ``plot_posterior`` panels.
    The HDI shading and point estimate are computed by ArviZ from the
    raw pooled draws; the dashed curve is the ProbPipe KDE density.

    To obtain the fitted ``KDEDistribution`` objects for downstream use
    (log-density evaluation, sampling, ``from_distribution``), call
    :func:`fit_kde` separately — it returns a plain dict keyed by
    parameter name.

    Vector-valued parameters are skipped with a warning.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.
    bandwidth : float or None
        Scalar bandwidth forwarded to ``KDEDistribution``. ``None``
        (default) applies Silverman's rule.
    credible_interval : float
        HDI mass to shade, in (0, 1). Default is 0.94 (94% HDI).
    point_estimate : {"mean", "median", "mode"}
        Point estimate marker. Forwarded to ``az.plot_posterior``.
    figsize : tuple[int, int] or None
        Passed to ``az.plot_posterior``. Defaults to
        ``(4 * n_scalar_params, 3)``.
    title : str or None
        Figure-level suptitle. Defaults to ``"Posterior KDE"``.

    Returns
    -------
    dict
        ``{"fig": matplotlib.figure.Figure}``
    """
    check_arviz_installed()
    import numpy as np
    import jax.numpy as jnp
    import arviz as az
    import xarray as xr
    from ..distributions.kde import KDEDistribution

    raw_draws = extract_draws(posterior)

    scalar_vars: list[str] = []
    skipped: list[str] = []
    for var, arr in raw_draws.items():
        if np.asarray(arr).ndim <= 2:
            scalar_vars.append(var)
        else:
            skipped.append(var)

    if skipped:
        _warnings.warn(
            f"plot_kde: skipping {skipped} — vector/matrix parameters are not "
            f"supported. Use a pair-plot for multi-dimensional marginals.",
            stacklevel=4,
        )
    if not scalar_vars:
        raise ValueError(
            "plot_kde: no scalar parameters found. "
            "All parameters appear to be vector-valued."
        )

    # Fit KDEDistribution per parameter and build pooled-draw dataset for ArviZ
    kde_distributions: dict[str, KDEDistribution] = {}
    data_vars: dict[str, xr.DataArray] = {}
    for var in scalar_vars:
        pooled = np.asarray(raw_draws[var]).reshape(-1).astype(float)
        bw = np.array([bandwidth], dtype=float) if bandwidth is not None else None
        kde_distributions[var] = KDEDistribution(pooled, bandwidth=bw, name=var)
        data_vars[var] = xr.DataArray(pooled[np.newaxis, :], dims=["chain", "draw"])

    dataset  = xr.Dataset(data_vars)
    n_params = len(scalar_vars)
    fs       = figsize or (4 * n_params, 3)

    axes = az.plot_posterior(
        dataset,
        hdi_prob=credible_interval,
        point_estimate=point_estimate,
        figsize=fs,
    )

    # Overlay the ProbPipe KDE curve on each panel
    import matplotlib.axes as _maxes
    axes_list = (
        [axes] if isinstance(axes, _maxes.Axes) else list(axes.ravel())
    )
    for ax, var in zip(axes_list, scalar_vars):
        kde = kde_distributions[var]
        pooled = np.asarray(raw_draws[var]).reshape(-1).astype(float)
        margin = (pooled.max() - pooled.min()) * 0.15
        grid = np.linspace(pooled.min() - margin, pooled.max() + margin, 300)
        density = np.asarray(jnp.exp(kde._tfp_dist.log_prob(grid)))
        ax.plot(grid, density, color="C1", lw=1.5, ls="--",
                label=f"KDE (bw={float(kde._bandwidth[0]):.3f})")
        ax.legend(fontsize=7, loc="upper right")

    fig = _axes_to_fig(axes)
    fig.suptitle(title or "Posterior KDE", y=1.02, fontsize=12)
    fig.tight_layout()

    return {"fig": fig}


plot_kde = WorkflowFunction(
    func=_plot_kde_impl, name="plot_kde"
)
