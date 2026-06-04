"""MCMC diagnostic functions for ProbPipe.

All functions mutate ``posterior._auxiliary`` in place and return ``None``.
They are plain Python — not ``@workflow_function`` — because diagnostics
are side-effectful operations on an already-fitted posterior and do not
belong in the probabilistic pipeline graph.

Usage::

    from probpipe.diagnostics import mcmc_diagnostics

    posterior = condition_on(model, data)
    mcmc_diagnostics(posterior)

    posterior.diagnostics.rhat      # {"intercept": 1.001, "slope": 1.002}
    posterior.diagnostics.warnings  # []
    posterior.diagnostics.runs      # []

Individual functions for targeted use::

    from probpipe.diagnostics import compute_rhat, compute_ess, compute_mcse

    compute_rhat(posterior, method="split")
    compute_ess(posterior)
    compute_mcse(posterior)

Provides:

- :func:`mcmc_diagnostics` -- convenience wrapper: rhat + ess + mcse
- :func:`compute_rhat`     -- R-hat (rank-normalized by default)
- :func:`compute_ess`      -- bulk and tail ESS
- :func:`compute_mcse`     -- MCSE of mean and sd
"""
from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..inference._approximate_distribution import ApproximateDistribution

__all__ = [
    "mcmc_diagnostics",
    "compute_rhat",
    "compute_ess",
    "compute_mcse",
]

_RHAT_THRESHOLD: float = 1.01
_ESS_THRESHOLD:  int   = 400


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_arviz() -> None:
    try:
        import arviz  # noqa: F401
    except ImportError:
        raise ImportError(
            "ArviZ is required for MCMC diagnostics. "
            "Install with: pip install probpipe[diagnostics]"
        )


def _rhat_warnings(values: dict, threshold: float) -> list[str]:
    from ._datatree import NotComputed
    msgs = []
    for param, val in values.items():
        if isinstance(val, NotComputed):
            continue
        if float(val) > threshold:
            msgs.append(
                f"R-hat > {threshold} for '{param}' ({val:.4f}) -- "
                f"chains may not have converged."
            )
    return msgs


def _ess_warnings(bulk: dict, tail: dict, threshold: int) -> list[str]:
    from ._datatree import NotComputed
    msgs = []
    for param, val in bulk.items():
        if isinstance(val, NotComputed):
            continue
        if float(val) < threshold:
            msgs.append(
                f"Low ESS (bulk) for '{param}' ({val:.0f}) -- "
                f"consider more iterations."
            )
    for param, val in tail.items():
        if isinstance(val, NotComputed):
            continue
        if float(val) < threshold:
            msgs.append(
                f"Low ESS (tail) for '{param}' ({val:.0f}) -- "
                f"tail estimates may be unreliable."
            )
    return msgs


def _scalar_from_da(da: Any) -> float:
    """Robustly extract a Python float from an ArviZ diagnostic DataArray."""
    import numpy as np
    return float(np.asarray(da, dtype=float).squeeze().item())


# ---------------------------------------------------------------------------
# compute_rhat
# ---------------------------------------------------------------------------

def compute_rhat(
    posterior: "ApproximateDistribution",
    *,
    method: str = "rank",
    threshold: float = _RHAT_THRESHOLD,
    force: bool = False,
) -> None:
    """Compute R-hat and attach to ``_auxiliary/diagnostics/mcmc/``.

    Skips computation if R-hat is already present unless ``force=True``.
    Emits a Python warning for any parameter with R-hat > ``threshold``.

    Parameters
    ----------
    posterior : ApproximateDistribution
        The fitted posterior. Mutated in place.
    method : str
        ArviZ R-hat variant: ``"rank"`` (default), ``"split"``,
        or ``"folded"``.
    threshold : float
        Warning threshold. Default ``1.01``.
    force : bool
        Recompute even if R-hat is already stored. Default ``False``.
    """
    from ._datatree import (
        NotComputed, _mcmc_has_field, _write_mcmc_field,
        to_named_posterior_dataset,
    )
    _check_arviz()
    import arviz as az

    if not force and _mcmc_has_field(posterior, "rhat"):
        return

    if posterior.num_chains < 2:
        values = {
            f: NotComputed("R-hat requires at least 2 chains")
            for f in posterior.fields
        }
        _write_mcmc_field(posterior, "rhat", values, attrs={
            "rhat_method":    method,
            "rhat_threshold": threshold,
            "rhat_warnings":  json.dumps([]),
        })
        return

    ds      = to_named_posterior_dataset(posterior)
    rhat_ds = az.rhat(ds, method=method)

    values = {p: _scalar_from_da(rhat_ds[p]) for p in rhat_ds.data_vars}
    warns  = _rhat_warnings(values, threshold)
    for w in warns:
        warnings.warn(w, stacklevel=2)

    _write_mcmc_field(posterior, "rhat", values, attrs={
        "rhat_method":    method,
        "rhat_threshold": threshold,
        "rhat_warnings":  json.dumps(warns),
    })


# ---------------------------------------------------------------------------
# compute_ess
# ---------------------------------------------------------------------------

def compute_ess(
    posterior: "ApproximateDistribution",
    *,
    threshold: int = _ESS_THRESHOLD,
    force: bool = False,
) -> None:
    """Compute bulk and tail ESS and attach to ``_auxiliary/diagnostics/mcmc/``.

    Skips computation if ESS is already present unless ``force=True``.
    Emits Python warnings for parameters below ``threshold``.

    Parameters
    ----------
    posterior : ApproximateDistribution
        The fitted posterior. Mutated in place.
    threshold : int
        Warning threshold. Default ``400``.
    force : bool
        Recompute even if ESS is already stored. Default ``False``.
    """
    from ._datatree import (
        _mcmc_has_field, _write_mcmc_field,
        to_named_posterior_dataset,
    )
    _check_arviz()
    import arviz as az

    if not force and _mcmc_has_field(posterior, "ess_bulk"):
        return

    ds      = to_named_posterior_dataset(posterior)
    bulk_ds = az.ess(ds, method="bulk")
    tail_ds = az.ess(ds, method="tail")

    bulk = {p: _scalar_from_da(bulk_ds[p]) for p in bulk_ds.data_vars}
    tail = {p: _scalar_from_da(tail_ds[p]) for p in tail_ds.data_vars}

    warns = _ess_warnings(bulk, tail, threshold)
    for w in warns:
        warnings.warn(w, stacklevel=2)

    _write_mcmc_field(posterior, "ess_bulk", bulk)
    _write_mcmc_field(posterior, "ess_tail", tail, attrs={
        "ess_threshold": threshold,
        "ess_warnings":  json.dumps(warns),
    })


# ---------------------------------------------------------------------------
# compute_mcse
# ---------------------------------------------------------------------------

def compute_mcse(
    posterior: "ApproximateDistribution",
    *,
    force: bool = False,
) -> None:
    """Compute MCSE (mean and sd) and attach to ``_auxiliary/diagnostics/mcmc/``.

    Skips computation if MCSE is already present unless ``force=True``.

    Parameters
    ----------
    posterior : ApproximateDistribution
        The fitted posterior. Mutated in place.
    force : bool
        Recompute even if MCSE is already stored. Default ``False``.
    """
    from ._datatree import (
        _mcmc_has_field, _write_mcmc_field,
        to_named_posterior_dataset,
    )
    _check_arviz()
    import arviz as az

    if not force and _mcmc_has_field(posterior, "mcse_mean"):
        return

    ds      = to_named_posterior_dataset(posterior)
    mean_ds = az.mcse(ds, method="mean")
    sd_ds   = az.mcse(ds, method="sd")

    _write_mcmc_field(posterior, "mcse_mean",
                      {p: _scalar_from_da(mean_ds[p]) for p in mean_ds.data_vars})
    _write_mcmc_field(posterior, "mcse_sd",
                      {p: _scalar_from_da(sd_ds[p])   for p in sd_ds.data_vars})


# ---------------------------------------------------------------------------
# mcmc_diagnostics
# ---------------------------------------------------------------------------

def mcmc_diagnostics(
    posterior: "ApproximateDistribution",
    *,
    metrics: list[str] | None = None,
    rhat_method: str = "rank",
    rhat_threshold: float = _RHAT_THRESHOLD,
    ess_threshold: int = _ESS_THRESHOLD,
    force: bool = False,
) -> None:
    """Compute MCMC diagnostics and attach to ``_auxiliary/diagnostics/mcmc/``.

    Computes R-hat, bulk ESS, tail ESS, and MCSE by default.
    Skips any metric already present unless ``force=True``.

    Parameters
    ----------
    posterior : ApproximateDistribution
        The fitted posterior. Mutated in place -- no return value.
    metrics : list of str or None
        Subset to compute. ``None`` computes all: ``["rhat", "ess", "mcse"]``.
    rhat_method : str
        ArviZ R-hat variant: ``"rank"`` (default), ``"split"``, ``"folded"``.
    rhat_threshold : float
        R-hat warning threshold. Default ``1.01``.
    ess_threshold : int
        ESS warning threshold. Default ``400``.
    force : bool
        Recompute even if metrics are already stored. Default ``False``.

    Examples
    --------
    ::

        posterior = condition_on(model, data)
        mcmc_diagnostics(posterior)

        posterior.diagnostics.rhat      # {"intercept": 1.001, "slope": 1.002}
        posterior.diagnostics.ess_bulk  # {"intercept": 962.0, "slope": 944.0}
        posterior.diagnostics.warnings  # []

        # Custom subset
        mcmc_diagnostics(posterior, metrics=["rhat"])

        # Force recompute with different method
        mcmc_diagnostics(posterior, rhat_method="split", force=True)
    """
    compute = set(metrics) if metrics is not None else {"rhat", "ess", "mcse"}

    if "rhat" in compute:
        compute_rhat(posterior, method=rhat_method,
                     threshold=rhat_threshold, force=force)
    if "ess" in compute:
        compute_ess(posterior, threshold=ess_threshold, force=force)
    if "mcse" in compute:
        compute_mcse(posterior, force=force)
