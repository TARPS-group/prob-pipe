"""MCMC diagnostic workflow functions.

All diagnostic functions accept any posterior-like object:

- ProbPipe ``Distribution`` (result of ``condition_on``)
- ``EmpiricalDistribution``
- ArviZ ``InferenceData``
- Plain ``dict[str, array]``
- Bare array

The ``WorkflowFunction`` wrappers type-hint ``posterior`` as
``Distribution`` to prevent accidental broadcasting when a
Distribution is passed. For non-Distribution inputs (plain dict,
bare array), call the ``_*_impl`` functions directly or use
``extract_draws`` from ``_arviz_bridge``.

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
]


# ── compute_rhat ──────────────────────────────────────────────────────────────

def _compute_rhat_impl(posterior: Distribution) -> dict:
    """Compute R-hat (Gelman-Rubin) convergence statistic.

    R-hat measures whether multiple MCMC chains have converged to the
    same distribution. Values close to 1.0 indicate good mixing;
    values above 1.01 indicate the chains have not yet converged.

    .. note::
        R-hat requires at least 2 chains for a meaningful result.
        Multi-chain posteriors should store draws as shape
        ``(n_chains, n_draws)`` per parameter.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.
        Accepts any format supported by
        :func:`~probpipe.diagnostics._arviz_bridge.extract_draws`.

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
            "Multi-chain draws should have shape (n_chains, n_draws) "
            "per parameter.",
            stacklevel=2,
        )

    rhat_ds = az.rhat(dataset)
    return {
        "rhat": {
            var: float(rhat_ds[var].values)
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

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.

    Returns
    -------
    dict
        ::

            {
                "ess": {
                    "bulk": {"intercept": 950.0, "slope": 987.0},
                    "tail": {"intercept": 910.0, "slope": 942.0},
                }
            }
    """
    check_arviz_installed()
    import arviz as az

    dataset  = to_arviz_dataset(posterior)
    ess_bulk = az.ess(dataset, method="bulk")
    ess_tail = az.ess(dataset, method="tail")

    return {
        "ess": {
            "bulk": {v: float(ess_bulk[v].values) for v in ess_bulk.data_vars},
            "tail": {v: float(ess_tail[v].values) for v in ess_tail.data_vars},
        }
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
        ::

            {
                "mcse": {
                    "mean": {"intercept": 0.003, "slope": 0.002},
                    "sd":   {"intercept": 0.002, "slope": 0.001},
                }
            }
    """
    check_arviz_installed()
    import arviz as az

    dataset   = to_arviz_dataset(posterior)
    mcse_mean = az.mcse(dataset, method="mean")
    mcse_sd   = az.mcse(dataset, method="sd")

    return {
        "mcse": {
            "mean": {v: float(mcse_mean[v].values) for v in mcse_mean.data_vars},
            "sd":   {v: float(mcse_sd[v].values)   for v in mcse_sd.data_vars},
        }
    }


compute_mcse = WorkflowFunction(
    func=_compute_mcse_impl, name="compute_mcse"
)


# ── mcmc_summary ──────────────────────────────────────────────────────────────

def _mcmc_summary_impl(posterior: Distribution) -> dict:
    """Compute a full MCMC diagnostic summary in a single pass.

    Runs R-hat, ESS (bulk + tail), and MCSE together — sharing a
    single ArviZ dataset conversion — and attaches human-readable
    warnings for values outside recommended thresholds.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.

    Returns
    -------
    dict
        ::

            {
                "rhat": {"intercept": 1.002, "slope": 1.001},
                "ess": {
                    "bulk": {"intercept": 950.0, "slope": 987.0},
                    "tail": {"intercept": 910.0, "slope": 942.0},
                },
                "mcse": {
                    "mean": {"intercept": 0.003, "slope": 0.002},
                    "sd":   {"intercept": 0.002, "slope": 0.001},
                },
                "warnings": [],
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

    rhat = {v: float(rhat_ds[v].values)  for v in rhat_ds.data_vars}
    ess  = {
        "bulk": {v: float(ess_bulk[v].values) for v in ess_bulk.data_vars},
        "tail": {v: float(ess_tail[v].values) for v in ess_tail.data_vars},
    }
    mcse = {
        "mean": {v: float(mcse_mean[v].values) for v in mcse_mean.data_vars},
        "sd":   {v: float(mcse_sd[v].values)   for v in mcse_sd.data_vars},
    }

    return {
        "rhat":     rhat,
        "ess":      ess,
        "mcse":     mcse,
        "warnings": build_warnings(rhat, ess),
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
        Figure size. Defaults to ``(10, 2 * n_params)``.
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

    fig = axes.ravel()[0].get_figure()
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
    samples relative to pooled samples. Uniform rank distributions
    indicate good mixing. More sensitive than trace plots for
    detecting subtle convergence failures.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.
    figsize : tuple[int, int] or None
        Figure size. Defaults to ``(10, 2 * n_params)``.
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

    fig = axes.ravel()[0].get_figure()
    fig.suptitle(title or "Rank Plot", y=1.02, fontsize=12)
    fig.tight_layout()

    return {"fig": fig}


plot_rank = WorkflowFunction(
    func=_plot_rank_impl, name="plot_rank"
)