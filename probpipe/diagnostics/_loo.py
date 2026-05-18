"""LOO-PSIS diagnostics via ArviZ 1.0.

Leave-One-Out Cross-Validation using Pareto-Smoothed Importance
Sampling estimates predictive accuracy without re-fitting the model.

Pareto-k diagnostics flag influential observations:

    k < 0.5    good      — importance weights are reliable
    k < 0.7    ok        — slightly less reliable
    k < 1.0    bad       — results may be unreliable
    k >= 1.0   very bad  — LOO estimate not reliable

Functions
---------
loo         : compute LOO-ELPD and per-observation Pareto-k values
compare_loo : compare multiple models by LOO-ELPD
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.node import workflow_function
from ._arviz_bridge import check_arviz_installed

__all__ = ["loo", "compare_loo"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_log_likelihood_dataset(
    log_lik_array: Any,
    var_name: str = "obs",
) -> "xr.Dataset":
    """Convert a log likelihood array to an xarray Dataset for ArviZ 1.0.

    Accepts three shapes and promotes to ``(chain, draw, n_obs)``:

    - ``(n_obs,)``              → ``(1, 1, n_obs)``
    - ``(draw, n_obs)``         → ``(1, draw, n_obs)``
    - ``(chain, draw, n_obs)``  → unchanged

    Parameters
    ----------
    log_lik_array : array-like
        Pointwise log likelihood values.
    var_name : str
        Variable name in the returned Dataset.

    Returns
    -------
    xr.Dataset
        Dims ``(chain, draw, obs)``.
    """
    import xarray as xr

    arr = np.asarray(log_lik_array, dtype=float)

    if arr.ndim == 1:
        arr = arr[np.newaxis, np.newaxis, :]   # (1, 1, n_obs)
    elif arr.ndim == 2:
        arr = arr[np.newaxis, :]               # (1, draw, n_obs)
    elif arr.ndim != 3:
        raise ValueError(
            f"log_likelihood must be 1D (n_obs,), 2D (draw, n_obs), or "
            f"3D (chain, draw, n_obs). Got shape {arr.shape}."
        )

    return xr.Dataset({
        var_name: xr.DataArray(arr, dims=["chain", "draw", "obs"])
    })


def _pareto_k_summary(pareto_k: np.ndarray) -> dict:
    """Summarise Pareto-k values into threshold bucket counts.

    Returns
    -------
    dict
        ``{"good": int, "ok": int, "bad": int, "very_bad": int}``
    """
    return {
        "good":     int(np.sum(pareto_k < 0.5)),
        "ok":       int(np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))),
        "bad":      int(np.sum((pareto_k >= 0.7) & (pareto_k < 1.0))),
        "very_bad": int(np.sum(pareto_k >= 1.0)),
    }


def _build_loo_warnings(pareto_k: np.ndarray) -> list[str]:
    """Build human-readable warnings from Pareto-k values.

    Flags observations with k >= 0.7 as bad and k >= 1.0 as very bad.
    """
    msgs: list[str] = []

    very_bad_idx = np.where(pareto_k >= 1.0)[0].tolist()
    bad_idx      = np.where((pareto_k >= 0.7) & (pareto_k < 1.0))[0].tolist()

    if very_bad_idx:
        msgs.append(
            f"{len(very_bad_idx)} observation(s) with Pareto-k >= 1.0 "
            f"(very bad) at indices {very_bad_idx}. "
            f"LOO estimate is not reliable — consider model revision."
        )
    elif bad_idx:
        msgs.append(
            f"{len(bad_idx)} observation(s) with Pareto-k >= 0.7 "
            f"(bad) at indices {bad_idx}. "
            f"Results may be unreliable."
        )

    return msgs


# ---------------------------------------------------------------------------
# loo
# ---------------------------------------------------------------------------

@workflow_function
def loo(
    posterior: Any,
    log_likelihood: Any,
    *,
    var_name: str = "obs",
    pointwise: bool = False,
) -> dict:
    """Compute LOO-CV via PSIS (Pareto-Smoothed Importance Sampling).

    Estimates out-of-sample predictive accuracy using LOO-CV without
    re-fitting the model.  PSIS re-weights existing posterior draws to
    approximate leave-one-out posteriors.

    .. note::
        LOO requires pointwise log likelihood values.  These must be
        computed externally from the posterior draws and the observed
        data — see the example below.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.
        Used for provenance tracking only.
    log_likelihood : array-like
        Pointwise log likelihood values
        ``log p(y_i | theta^(s))`` for each draw ``s`` and observation
        ``i``.  Accepted shapes:

        - ``(n_obs,)``              — point estimate, no draws
        - ``(n_draw, n_obs)``       — single chain
        - ``(n_chain, n_draw, n_obs)`` — multiple chains
    var_name : str
        Variable name in the internal xarray Dataset. Default ``"obs"``.
    pointwise : bool
        If ``True``, include per-observation ELPD and Pareto-k arrays
        in the result. Default ``False``.

    Returns
    -------
    dict
        ::

            {
                "elpd_loo":  float,  # expected log pointwise predictive density
                "p_loo":     float,  # effective number of parameters
                "looic":     float,  # LOO information criterion = -2 * elpd_loo
                "se":        float,  # std error of elpd_loo estimate
                "pareto_k":  {       # Pareto-k observation counts
                    "good":     int, # k < 0.5
                    "ok":       int, # 0.5 <= k < 0.7
                    "bad":      int, # 0.7 <= k < 1.0
                    "very_bad": int, # k >= 1.0
                },
                "warnings":  list[str],
                # only when pointwise=True:
                "pointwise_elpd":    np.ndarray,  # shape (n_obs,)
                "pointwise_pareto_k": np.ndarray, # shape (n_obs,)
            }

        An empty ``"warnings"`` list means all Pareto-k values are
        below 0.7.

    Raises
    ------
    ValueError
        If ``log_likelihood`` has an unsupported shape.

    Examples
    --------
    ::

        import numpy as np
        from probpipe.diagnostics import loo

        # Compute pointwise log likelihood from posterior draws
        draws   = np.array([posterior.sample() for _ in range(1000)])
        log_lik = np.array([
            [likelihood.log_likelihood(params=d, data=yi)
             for yi in observed_data]
            for d in draws
        ])  # shape (n_draw, n_obs)

        result = loo(posterior, log_likelihood=log_lik)
        print(result["elpd_loo"])    # e.g. -142.3
        print(result["warnings"])   # [] = all Pareto-k < 0.7
    """
    check_arviz_installed()
    import arviz as az

    log_lik_ds = _to_log_likelihood_dataset(log_likelihood, var_name=var_name)
    loo_result = az.loo(log_lik_ds, var_name=var_name, pointwise=True)

    pareto_k_vals = np.asarray(loo_result.pareto_k)

    result: dict = {
        "elpd_loo": float(loo_result.elpd_loo),
        "p_loo":    float(loo_result.p_loo),
        "looic":    float(loo_result.looic),
        "se":       float(loo_result.se),
        "pareto_k": _pareto_k_summary(pareto_k_vals),
        "warnings": _build_loo_warnings(pareto_k_vals),
    }

    if pointwise:
        result["pointwise_elpd"]      = np.asarray(loo_result.loo_i)
        result["pointwise_pareto_k"]  = pareto_k_vals

    return result


# ---------------------------------------------------------------------------
# compare_loo
# ---------------------------------------------------------------------------

@workflow_function
def compare_loo(
    posteriors: dict[str, Any],
    log_likelihoods: dict[str, Any],
    *,
    var_name: str = "obs",
) -> dict:
    """Compare multiple models by LOO-ELPD.

    Runs :func:`loo` for each model and ranks them by ELPD (higher is
    better). The best model is returned as ``"best"``.

    Parameters
    ----------
    posteriors : dict[str, Distribution]
        Named posteriors — ``{model_name: posterior}``.
    log_likelihoods : dict[str, array-like]
        Named log likelihood arrays — ``{model_name: log_lik}``.
        Keys must match ``posteriors``.
    var_name : str
        Variable name forwarded to :func:`loo`. Default ``"obs"``.

    Returns
    -------
    dict
        ::

            {
                "ranking":  ["negbin", "poisson"],  # best first
                "best":     "negbin",
                "elpd_loo": {"negbin": -132.1, "poisson": -148.6},
                "looic":    {"negbin":  264.2, "poisson":  297.2},
                "se":       {"negbin":    8.1, "poisson":   10.3},
                "pareto_k": {"negbin": {...},  "poisson": {...}},
                "warnings": {"negbin": [],     "poisson": [...]},
            }

    Raises
    ------
    ValueError
        If ``posteriors`` and ``log_likelihoods`` have different keys.

    Examples
    --------
    ::

        from probpipe.diagnostics import compare_loo

        result = compare_loo(
            posteriors={
                "poisson": posterior_poisson,
                "negbin":  posterior_negbin,
            },
            log_likelihoods={
                "poisson": log_lik_poisson,
                "negbin":  log_lik_negbin,
            },
        )
        print(result["best"])      # "negbin"
        print(result["ranking"])   # ["negbin", "poisson"]
    """
    if set(posteriors) != set(log_likelihoods):
        raise ValueError(
            f"posteriors and log_likelihoods must have the same keys.\n"
            f"posteriors keys:       {sorted(posteriors)}\n"
            f"log_likelihoods keys:  {sorted(log_likelihoods)}"
        )

    elpd_loo: dict[str, float]      = {}
    looic:    dict[str, float]      = {}
    se:       dict[str, float]      = {}
    pareto_k: dict[str, dict]       = {}
    warnings: dict[str, list[str]]  = {}

    for name in posteriors:
        result = loo(
            posteriors[name],
            log_likelihoods[name],
            var_name=var_name,
        )
        elpd_loo[name] = result["elpd_loo"]
        looic[name]    = result["looic"]
        se[name]       = result["se"]
        pareto_k[name] = result["pareto_k"]
        warnings[name] = result["warnings"]

    # Rank highest ELPD first (best predictive accuracy)
    ranking = sorted(elpd_loo, key=lambda k: elpd_loo[k], reverse=True)

    return {
        "ranking":  ranking,
        "best":     ranking[0],
        "elpd_loo": elpd_loo,
        "looic":    looic,
        "se":       se,
        "pareto_k": pareto_k,
        "warnings": warnings,
    }