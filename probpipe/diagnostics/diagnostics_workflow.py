"""DiagnosticsModule — orchestrated MCMC diagnostic workflows.

Wraps the standalone functions from ``mcmc.py`` as ``@workflow_method``
steps so they participate in the ProbPipe dependency graph with
provenance tracking and optional Prefect orchestration.

Usage::

    from probpipe.diagnostics import DiagnosticsModule

    diag   = DiagnosticsModule()
    report = diag.summary(posterior)
    diag.trace_plot(posterior)["fig"]
    diag.rank_plot(posterior)["fig"]
    diag.kde_plot(posterior)["fig"]
"""
from __future__ import annotations

from ..core.distribution import Distribution
from ..core.node import Module, workflow_method
from .mcmc import (
    compute_rhat,
    compute_ess,
    compute_mcse,
    mcmc_summary,
    plot_trace,
    plot_rank,
    plot_kde,
)

__all__ = ["DiagnosticsModule"]


class DiagnosticsModule(Module):
    """Orchestrated MCMC diagnostics module.

    Provides the same results as the standalone functions in
    ``mcmc.py`` wrapped as ``@workflow_method`` steps for provenance
    tracking and optional Prefect orchestration.

    Examples
    --------
    ::

        from probpipe.diagnostics import DiagnosticsModule

        diag = DiagnosticsModule()

        # Numerical diagnostics
        report = diag.summary(posterior)
        print(report["warnings"])   # [] = all passed

        # Visual diagnostics
        diag.trace_plot(posterior)["fig"]
        diag.rank_plot(posterior)["fig"]
        diag.kde_plot(posterior)["fig"]

        # KDE distributions for downstream use
        result = diag.kde_plot(posterior)
        kde = result["kde_distributions"]["intercept"]

        # Everything at once
        plots = diag.plot_all(posterior)
    """

    # -- Numerical diagnostics -----------------------------------------

    @workflow_method
    def rhat(self, posterior: Distribution) -> dict:
        """R-hat convergence statistic.

        See :func:`compute_rhat` for full documentation.
        """
        return compute_rhat(posterior)

    @workflow_method
    def ess(self, posterior: Distribution) -> dict:
        """Effective Sample Size (bulk + tail).

        See :func:`compute_ess` for full documentation.
        """
        return compute_ess(posterior)

    @workflow_method
    def mcse(self, posterior: Distribution) -> dict:
        """Monte Carlo Standard Error (mean + sd).

        See :func:`compute_mcse` for full documentation.
        """
        return compute_mcse(posterior)

    @workflow_method
    def summary(self, posterior: Distribution) -> dict:
        """Full MCMC diagnostic summary (R-hat + ESS + MCSE + warnings).

        See :func:`mcmc_summary` for full documentation.
        """
        return mcmc_summary(posterior)

    # -- Visual diagnostics --------------------------------------------

    @workflow_method
    def trace_plot(
        self,
        posterior: Distribution,
        figsize: tuple[int, int] | None = None,
        title: str | None = None,
    ) -> dict:
        """Trace plot — sampled values over MCMC iterations.

        See :func:`plot_trace` for full documentation.

        Returns
        -------
        dict
            ``{"fig": matplotlib.figure.Figure}``
        """
        return plot_trace(posterior, figsize=figsize, title=title)

    @workflow_method
    def rank_plot(
        self,
        posterior: Distribution,
        figsize: tuple[int, int] | None = None,
        title: str | None = None,
    ) -> dict:
        """Rank plot — chain rank distributions relative to pooled samples.

        See :func:`plot_rank` for full documentation.

        Returns
        -------
        dict
            ``{"fig": matplotlib.figure.Figure}``
        """
        return plot_rank(posterior, figsize=figsize, title=title)

    @workflow_method
    def kde_plot(
        self,
        posterior: Distribution,
        *,
        bandwidth: float | None = None,
        credible_interval: float = 0.94,
        point_estimate: str = "mean",
        figsize: tuple[int, int] | None = None,
        title: str | None = None,
    ) -> dict:
        """KDE marginal density plot — one panel per scalar parameter.

        Fits a :class:`~probpipe.distributions.KDEDistribution` per
        parameter (Silverman's rule bandwidth by default) and overlays
        the density curve on ArviZ's ``plot_posterior`` panels.

        See :func:`plot_kde` for full documentation.

        Returns
        -------
        dict
            ``{"fig": matplotlib.figure.Figure,``
            ``"kde_distributions": dict[str, KDEDistribution]}``
        """
        return plot_kde(
            posterior,
            bandwidth=bandwidth,
            credible_interval=credible_interval,
            point_estimate=point_estimate,
            figsize=figsize,
            title=title,
        )

    @workflow_method
    def plot_all(self, posterior: Distribution) -> dict:
        """Trace, rank, and KDE plots in one call.

        Returns
        -------
        dict
            ``{"trace": fig, "rank": fig, "kde": fig}``
        """
        return {
            "trace": plot_trace(posterior, title="Trace Plot")["fig"],
            "rank":  plot_rank(posterior,  title="Rank Plot")["fig"],
            "kde":   plot_kde(posterior,   title="Posterior KDE")["fig"],
        }
