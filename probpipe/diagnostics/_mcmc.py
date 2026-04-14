"""MCMC diagnostics via ArviZ 1.0.

All functions accept raw samples (np.ndarray, dict, or
ArrayEmpiricalDistribution) and convert internally to xarray
before calling ArviZ 1.0 diagnostics.

Functions
---------
rhat            : R-hat convergence diagnostic
ess             : Effective Sample Size (bulk and tail)
mcse            : Monte Carlo Standard Error
mcmc_summary    : Combined summary of all diagnostics
trace_plot      : Trace plot (returns matplotlib axes)
autocorr_plot   : Autocorrelation plot (returns matplotlib axes)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import xarray as xr

from ..core.node import workflow_function
from ._utils import check_arviz_version, to_xarray_dataset

__all__ = [
    "rhat",
    "ess",
    "mcse",
    "mcmc_summary",
    "trace_plot",
    "autocorr_plot",
]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_dataset(samples: Any, distribution: Any = None) -> xr.Dataset:
    """Pass through if already xr.Dataset, otherwise convert."""
    if isinstance(samples, xr.Dataset):
        return samples
    return to_xarray_dataset(samples, distribution=distribution)


# ---------------------------------------------------------------------------
# R-hat
# ---------------------------------------------------------------------------

@workflow_function
def rhat(
    samples: Any,
    *,
    distribution: Any = None,
    method: str = "rank",
) -> dict:
    """Compute R-hat convergence diagnostic via ArviZ 1.0.

    R-hat measures convergence across MCMC chains.
    Values close to 1.0 indicate convergence.
    Rule of thumb: R-hat < 1.01 is acceptable.

    Parameters
    ----------
    samples : array-like, dict, or ArrayEmpiricalDistribution
        MCMC samples. Shape (n_chains, n_draws) or (n_draws,).
    distribution : Distribution or None
        Used to infer variable names from ``parameter_names``.
    method : str
        R-hat method passed to ArviZ. Default ``"rank"``.

    Returns
    -------
    dict
        - ``"rhat"``      : dict of ``{var_name: rhat_value}``
        - ``"converged"`` : bool — True if all R-hat < 1.01
        - ``"dataset"``   : raw ``xr.Dataset`` from ArviZ
    """
    check_arviz_version()
    import arviz as az

    dataset = _to_dataset(samples, distribution=distribution)
    rhat_ds = az.rhat(dataset, method=method)

    rhat_values = {
        var: float(np.asarray(rhat_ds[var]))
        for var in rhat_ds.data_vars
    }
    converged = all(v < 1.01 for v in rhat_values.values())

    return {
        "rhat": rhat_values,
        "converged": converged,
        "dataset": rhat_ds,
    }


# ---------------------------------------------------------------------------
# ESS
# ---------------------------------------------------------------------------

@workflow_function
def ess(
    samples: Any,
    *,
    distribution: Any = None,
    method: str = "bulk",
) -> dict:
    """Compute Effective Sample Size (ESS) via ArviZ 1.0.

    Parameters
    ----------
    samples : array-like, dict, or ArrayEmpiricalDistribution
        MCMC samples.
    distribution : Distribution or None
        Used to infer variable names.
    method : str
        ESS method: ``"bulk"`` (default) or ``"tail"``.

    Returns
    -------
    dict
        - ``"ess"``     : dict of ``{var_name: ess_value}``
        - ``"method"``  : method used
        - ``"dataset"`` : raw ``xr.Dataset`` from ArviZ
    """
    check_arviz_version()
    import arviz as az

    dataset = _to_dataset(samples, distribution=distribution)
    ess_ds = az.ess(dataset, method=method)

    ess_values = {
        var: float(np.asarray(ess_ds[var]))
        for var in ess_ds.data_vars
    }

    return {
        "ess": ess_values,
        "method": method,
        "dataset": ess_ds,
    }


# ---------------------------------------------------------------------------
# MCSE
# ---------------------------------------------------------------------------

@workflow_function
def mcse(
    samples: Any,
    *,
    distribution: Any = None,
    method: str = "mean",
) -> dict:
    """Compute Monte Carlo Standard Error (MCSE) via ArviZ 1.0.

    Parameters
    ----------
    samples : array-like, dict, or ArrayEmpiricalDistribution
        MCMC samples.
    distribution : Distribution or None
        Used to infer variable names.
    method : str
        MCSE method: ``"mean"`` (default), ``"sd"``, or ``"quantile"``.

    Returns
    -------
    dict
        - ``"mcse"``    : dict of ``{var_name: mcse_value}``
        - ``"method"``  : method used
        - ``"dataset"`` : raw ``xr.Dataset`` from ArviZ
    """
    check_arviz_version()
    import arviz as az

    dataset = _to_dataset(samples, distribution=distribution)
    mcse_ds = az.mcse(dataset, method=method)

    mcse_values = {
        var: float(np.asarray(mcse_ds[var]))
        for var in mcse_ds.data_vars
    }

    return {
        "mcse": mcse_values,
        "method": method,
        "dataset": mcse_ds,
    }


# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------

@workflow_function
def mcmc_summary(
    samples: Any,
    *,
    distribution: Any = None,
) -> dict:
    """Full MCMC summary: R-hat, ESS (bulk + tail), MCSE, warnings.

    Convenience wrapper that runs all diagnostics in one call
    and collects actionable warnings.

    Parameters
    ----------
    samples : array-like, dict, or ArrayEmpiricalDistribution
        MCMC samples.
    distribution : Distribution or None
        Used to infer variable names.

    Returns
    -------
    dict
        - ``"rhat"``     : dict of ``{var_name: rhat_value}``
        - ``"ess_bulk"`` : dict of ``{var_name: ess_bulk_value}``
        - ``"ess_tail"`` : dict of ``{var_name: ess_tail_value}``
        - ``"mcse"``     : dict of ``{var_name: mcse_value}``
        - ``"converged"``: bool — True if all R-hat < 1.01
        - ``"warnings"`` : list of human-readable warning strings
    """
    # Convert once, reuse for all diagnostics
    dataset = _to_dataset(samples, distribution=distribution)

    rhat_result  = rhat(dataset)
    ess_bulk     = ess(dataset, method="bulk")
    ess_tail     = ess(dataset, method="tail")
    mcse_result  = mcse(dataset)

    # Collect actionable warnings
    diagnostic_warnings: list[str] = []

    for var, val in rhat_result["rhat"].items():
        if val >= 1.01:
            diagnostic_warnings.append(
                f"R-hat for '{var}' is {val:.4f} (>= 1.01) — "
                "chains may not have converged."
            )

    for var, val in ess_bulk["ess"].items():
        if val < 400:
            diagnostic_warnings.append(
                f"ESS (bulk) for '{var}' is {int(val)} (< 400) — "
                "consider running more iterations."
            )

    for var, val in ess_tail["ess"].items():
        if val < 400:
            diagnostic_warnings.append(
                f"ESS (tail) for '{var}' is {int(val)} (< 400) — "
                "tail estimates may be unreliable."
            )

    return {
        "rhat":      rhat_result["rhat"],
        "ess_bulk":  ess_bulk["ess"],
        "ess_tail":  ess_tail["ess"],
        "mcse":      mcse_result["mcse"],
        "converged": rhat_result["converged"],
        "warnings":  diagnostic_warnings,
    }


# ---------------------------------------------------------------------------
# Plots — NOT decorated with @workflow_function (return matplotlib figures)
# ---------------------------------------------------------------------------

def trace_plot(
    samples: Any,
    *,
    distribution: Any = None,
    var_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Trace plot via ArviZ 1.0.

    Not decorated with ``@workflow_function`` — returns a matplotlib
    axes array rather than serialisable data.

    Parameters
    ----------
    samples : array-like, dict, or ArrayEmpiricalDistribution
        MCMC samples.
    distribution : Distribution or None
        Used to infer variable names.
    var_names : list of str or None
        Variables to plot. Defaults to all.
    **kwargs
        Extra keyword arguments forwarded to ``az.plot_trace``.

    Returns
    -------
    np.ndarray of matplotlib Axes
    """
    check_arviz_version()
    import arviz as az

    dataset = _to_dataset(samples, distribution=distribution)
    return az.plot_trace(dataset, var_names=var_names, **kwargs)


def autocorr_plot(
    samples: Any,
    *,
    distribution: Any = None,
    var_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Autocorrelation plot via ArviZ 1.0.

    Not decorated with ``@workflow_function`` — returns a matplotlib
    axes array rather than serialisable data.

    Parameters
    ----------
    samples : array-like, dict, or ArrayEmpiricalDistribution
        MCMC samples.
    distribution : Distribution or None
        Used to infer variable names.
    var_names : list of str or None
        Variables to plot. Defaults to all.
    **kwargs
        Extra keyword arguments forwarded to ``az.plot_autocorr``.

    Returns
    -------
    np.ndarray of matplotlib Axes
    """
    check_arviz_version()
    import arviz as az

    dataset = _to_dataset(samples, distribution=distribution)
    return az.plot_autocorr(dataset, var_names=var_names, **kwargs)