"""DiagnosticsModule — orchestrated MCMC diagnostic workflows.

Wraps the standalone functions from ``mcmc.py`` as ``@workflow_method``
steps so they participate in the ProbPipe dependency graph with
provenance tracking and optional Prefect orchestration.

Usage::

    from probpipe.diagnostics import DiagnosticsModule

    # Zero-config path
    record = DiagnosticsModule.default().run(posterior)
    record["mcmc"]["warnings"]

    # Extend with user diagnostics
    diag = DiagnosticsModule.default().with_diagnostic(
        sensitivity=my_sensitivity_fn,
    )
    record = diag.run(posterior)
    record["sensitivity"]

    # Individual methods still available
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

    Built-in diagnostics (R-hat, ESS, MCSE, trace/rank/KDE plots) are
    available as individual ``@workflow_method`` steps.  The
    :meth:`run` method collects all diagnostics — built-in and
    user-added — into a single nested dict.

    Examples
    --------
    ::

        from probpipe.diagnostics import DiagnosticsModule

        # Zero-config — all built-ins
        diag   = DiagnosticsModule.default()
        record = diag.run(posterior)
        print(record["mcmc"]["warnings"])   # [] = all passed

        # Extend with a custom diagnostic
        def my_loo(posterior):
            return {"elpd": -42.3}

        diag2  = diag.with_diagnostic(loo=my_loo)
        record = diag2.run(posterior)
        record["loo"]["elpd"]

        # Individual methods
        diag.trace_plot(posterior)["fig"]
        diag.rank_plot(posterior)["fig"]
        diag.kde_plot(posterior)["fig"]
    """

    def __init__(
        self,
        *,
        _extra_diagnostics: dict | None = None,
        **kwargs,
    ):
        # Store user-added diagnostics before super().__init__
        # so they never reach Node's child_nodes / inputs split.
        self._extra_diagnostics: dict = dict(_extra_diagnostics or {})
        super().__init__(**kwargs)

    # -- Factory methods -----------------------------------------------

    @classmethod
    def default(cls) -> "DiagnosticsModule":
        """Return a DiagnosticsModule with all built-in diagnostics.

        Convenience entry point for the zero-config path::

            record = DiagnosticsModule.default().run(posterior)

        Returns
        -------
        DiagnosticsModule
        """
        return cls()

    def with_diagnostic(self, **diagnostics) -> "DiagnosticsModule":
        """Return a new DiagnosticsModule with additional user diagnostics.

        Each keyword argument must be a callable that accepts a posterior
        ``Distribution`` and returns a ``dict``::

            def my_sensitivity(posterior):
                return {"khat": 0.43, "threshold": 0.7}

            diag = DiagnosticsModule.default().with_diagnostic(
                sensitivity=my_sensitivity,
            )
            record = diag.run(posterior)
            record["sensitivity"]["khat"]

        Raises on name collision with existing extra diagnostics — be
        explicit rather than silently overwriting.

        Parameters
        ----------
        **diagnostics : callable
            Named callables: ``{name: fn}`` where
            ``fn(posterior: Distribution) -> dict``.

        Returns
        -------
        DiagnosticsModule
            New instance — the original is unchanged (immutable pattern).

        Raises
        ------
        ValueError
            If any name already exists in the current extra diagnostics.
        """
        collisions = set(diagnostics) & set(self._extra_diagnostics)
        if collisions:
            raise ValueError(
                f"Diagnostic name(s) already registered: {collisions}. "
                f"Use a different name or create a fresh DiagnosticsModule."
            )
        merged = {**self._extra_diagnostics, **diagnostics}
        return DiagnosticsModule(_extra_diagnostics=merged)

    # -- Combined run --------------------------------------------------

    @workflow_method
    def run(self, posterior: Distribution) -> dict:
        """Run all diagnostics and return a combined nested result.

        Runs the built-in MCMC summary plus any user-added diagnostics
        registered via :meth:`with_diagnostic`. Failed diagnostics are
        caught and stored as ``{"error": str}`` so one failure does not
        abort the others.

        Parameters
        ----------
        posterior : Distribution
            Posterior from ``condition_on`` or ``EmpiricalDistribution``.

        Returns
        -------
        dict
            Nested result keyed by diagnostic name::

                {
                    "mcmc": {
                        "rhat":      {"intercept": 1.002, ...},
                        "ess_bulk":  {"intercept": 950.0, ...},
                        "ess_tail":  {"intercept": 910.0, ...},
                        "mcse_mean": {"intercept": 0.003, ...},
                        "mcse_sd":   {"intercept": 0.002, ...},
                        "warnings":  [],
                    },
                    # user-added examples:
                    "sensitivity": {"khat": 0.43},
                    "loo":         {"elpd": -42.3},
                }

            An empty ``"warnings"`` list means all MCMC diagnostics
            passed.
        """
        result: dict = {"mcmc": mcmc_summary(posterior)}

        for name, fn in self._extra_diagnostics.items():
            try:
                result[name] = fn(posterior)
            except Exception as exc:  # noqa: BLE001
                result[name] = {"error": str(exc)}

        return result

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

        Returns
        -------
        dict
            ``{"fig": matplotlib.figure.Figure}``
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