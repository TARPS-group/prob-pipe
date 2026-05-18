"""Sensitivity analysis for Bayesian models.

Two classes of sensitivity:

Local sensitivity
    Power-scaling multiplies the log prior or log likelihood by a
    scalar alpha and estimates the gradient of posterior summaries
    with respect to alpha at alpha=1.  High sensitivity means the
    posterior depends strongly on that component.

Global sensitivity
    Measures the effect of discrete, larger changes:
    - prior_sensitivity  : compare posteriors under different prior specs
    - data_sensitivity   : per-observation influence via LOO Pareto-k

Functions
---------
prior_sensitivity       : global — compare posteriors under named priors
data_sensitivity        : global — per-observation influence via LOO
power_scale_sensitivity : local  — power scaling over prior and likelihood
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.node import workflow_function
from ._arviz_bridge import (
    check_arviz_installed,
    extract_draws,
)
from ._loo import _to_log_likelihood_dataset, _build_loo_warnings

__all__ = [
    "prior_sensitivity",
    "data_sensitivity",
    "power_scale_sensitivity",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _posterior_summary(draws: dict[str, np.ndarray]) -> dict[str, dict]:
    """Compute mean, std, and 94% HDI per parameter from raw draws."""
    summary: dict[str, dict] = {}
    for var, arr in draws.items():
        pooled = np.asarray(arr, dtype=float).reshape(-1)
        summary[var] = {
            "mean":    float(np.mean(pooled)),
            "std":     float(np.std(pooled)),
            "hdi_3%":  float(np.percentile(pooled, 3)),
            "hdi_97%": float(np.percentile(pooled, 97)),
        }
    return summary


def _kl_divergence_gaussian(
    mean1: float, std1: float,
    mean2: float, std2: float,
) -> float:
    """KL(N(m1,s1) || N(m2,s2)) in closed form.

    Used to quantify how much two posterior marginals differ.
    Returns ``nan`` if either std is zero or negative.
    """
    if std1 <= 0 or std2 <= 0:
        return float("nan")
    return float(
        np.log(std2 / std1)
        + (std1 ** 2 + (mean1 - mean2) ** 2) / (2 * std2 ** 2)
        - 0.5
    )


def _compare_summaries(
    baseline: dict[str, dict],
    perturbed: dict[str, dict],
) -> dict[str, dict]:
    """Per-parameter mean shift, std ratio, and KL divergence."""
    result: dict[str, dict] = {}
    for var in baseline:
        if var not in perturbed:
            continue
        b = baseline[var]
        p = perturbed[var]
        result[var] = {
            "mean_shift":    float(abs(p["mean"] - b["mean"])),
            "std_ratio":     float(p["std"] / b["std"])
                             if b["std"] > 0 else float("nan"),
            "kl_divergence": _kl_divergence_gaussian(
                b["mean"], b["std"], p["mean"], p["std"]
            ),
        }
    return result


def _reweighted_summary(
    draws_stack: np.ndarray,
    param_names: list[str],
    log_weights: np.ndarray,
) -> dict[str, dict]:
    """Weighted posterior summary via normalised importance weights.

    Parameters
    ----------
    draws_stack : np.ndarray, shape (n_draw, n_params)
    param_names : list[str]
    log_weights : np.ndarray, shape (n_draw,)
    """
    lw = log_weights - np.max(log_weights)   # log-sum-exp stabilisation
    w  = np.exp(lw)
    w /= w.sum()

    summary: dict[str, dict] = {}
    for i, var in enumerate(param_names):
        vals   = draws_stack[:, i]
        w_mean = float(np.sum(w * vals))
        w_std  = float(np.sqrt(np.sum(w * (vals - w_mean) ** 2)))
        summary[var] = {"mean": w_mean, "std": w_std}
    return summary


def _finite_diff_gradient(arr: np.ndarray, idx: int, delta: float) -> float:
    """Estimate d(arr)/d(alpha) at index idx using finite differences."""
    if idx == 0:
        return float((arr[1] - arr[0]) / delta)
    if idx == len(arr) - 1:
        return float((arr[-1] - arr[-2]) / delta)
    return float((arr[idx + 1] - arr[idx - 1]) / (2 * delta))


# ---------------------------------------------------------------------------
# prior_sensitivity
# ---------------------------------------------------------------------------

@workflow_function
def prior_sensitivity(
    posteriors: dict[str, Any],
    *,
    baseline: str | None = None,
    kl_threshold: float = 0.1,
) -> dict:
    """Compare posterior summaries under different prior specifications.

    Global sensitivity analysis — compares pre-fitted posteriors to
    measure how much the choice of prior affects the posterior.  The
    caller is responsible for fitting the model under each prior.

    Parameters
    ----------
    posteriors : dict[str, Distribution]
        Named posteriors — e.g.::

            {
                "weakly_informative": posterior_wi,
                "regularizing":       posterior_reg,
                "flat":               posterior_flat,
            }

    baseline : str or None
        Name of the baseline posterior for pairwise comparison.
        Defaults to the first key in ``posteriors``.
    kl_threshold : float
        KL divergence threshold above which a warning is emitted.
        Default 0.1 (rule-of-thumb for meaningful prior sensitivity).

    Returns
    -------
    dict
        ::

            {
                "baseline":    "weakly_informative",
                "summaries": {
                    "weakly_informative": {
                        "mu": {"mean": ..., "std": ...,
                               "hdi_3%": ..., "hdi_97%": ...},
                    },
                    ...
                },
                "comparisons": {
                    "regularizing vs weakly_informative": {
                        "mu": {
                            "mean_shift":    0.12,
                            "std_ratio":     0.95,
                            "kl_divergence": 0.03,
                        },
                        ...
                    },
                    ...
                },
                "warnings": list[str],
            }

    Raises
    ------
    ValueError
        If fewer than 2 posteriors are provided, or if ``baseline``
        is not a key in ``posteriors``.

    Examples
    --------
    ::

        from probpipe.diagnostics import prior_sensitivity

        result = prior_sensitivity(
            posteriors={
                "weakly_informative": posterior_wi,
                "regularizing":       posterior_reg,
            },
            baseline="weakly_informative",
        )
        result["comparisons"]["regularizing vs weakly_informative"]
    """
    if len(posteriors) < 2:
        raise ValueError(
            "prior_sensitivity requires at least 2 posteriors. "
            f"Got {len(posteriors)}."
        )

    names     = list(posteriors)
    base_name = baseline if baseline is not None else names[0]

    if base_name not in posteriors:
        raise ValueError(
            f"Baseline '{base_name}' not found in posteriors. "
            f"Available: {names}"
        )

    # Compute per-posterior summaries
    summaries: dict[str, dict] = {
        name: _posterior_summary(extract_draws(posterior))
        for name, posterior in posteriors.items()
    }

    base_summary = summaries[base_name]

    # Pairwise comparisons against baseline
    comparisons:         dict[str, dict] = {}
    diagnostic_warnings: list[str]       = []

    for name in names:
        if name == base_name:
            continue
        label = f"{name} vs {base_name}"
        comparisons[label] = _compare_summaries(base_summary, summaries[name])

        for var, stats in comparisons[label].items():
            kl = stats["kl_divergence"]
            if not np.isnan(kl) and kl > kl_threshold:
                diagnostic_warnings.append(
                    f"'{var}' shows high prior sensitivity between "
                    f"'{name}' and '{base_name}' "
                    f"(KL divergence = {kl:.4f} > {kl_threshold})."
                )

    return {
        "baseline":    base_name,
        "summaries":   summaries,
        "comparisons": comparisons,
        "warnings":    diagnostic_warnings,
    }


# ---------------------------------------------------------------------------
# data_sensitivity
# ---------------------------------------------------------------------------

@workflow_function
def data_sensitivity(
    posterior: Any,
    log_likelihood: Any,
    *,
    var_name: str = "obs",
    threshold_bad: float = 0.7,
    threshold_very_bad: float = 1.0,
) -> dict:
    """Identify influential observations via LOO Pareto-k diagnostics.

    High Pareto-k values indicate observations that strongly influence
    the posterior — either outliers the model cannot accommodate, or
    observations that drive parameter estimates.  These are candidates
    for investigation.

    Pareto-k thresholds:

    - k < 0.5   good     — importance weights are reliable
    - k < 0.7   ok       — slightly less reliable
    - k < 1.0   bad      — results may be unreliable
    - k >= 1.0  very bad — LOO estimate not reliable

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.
    log_likelihood : array-like
        Pointwise log likelihood — shape ``(n_obs,)``,
        ``(n_draw, n_obs)``, or ``(n_chain, n_draw, n_obs)``.
    var_name : str
        Variable name in the internal xarray Dataset. Default ``"obs"``.
    threshold_bad : float
        Pareto-k threshold for "influential" flag. Default 0.7.
    threshold_very_bad : float
        Pareto-k threshold for "very bad" flag. Default 1.0.

    Returns
    -------
    dict
        ::

            {
                "pareto_k":            np.ndarray,  # shape (n_obs,)
                "influential_indices": list[int],   # k >= threshold_bad
                "very_bad_indices":    list[int],   # k >= threshold_very_bad
                "n_observations":      int,
                "n_influential":       int,
                "n_very_bad":          int,
                "warnings":            list[str],
                "recommendations":     list[str],
            }

    Examples
    --------
    ::

        from probpipe.diagnostics import data_sensitivity

        result = data_sensitivity(posterior, log_lik)
        print(result["influential_indices"])   # [3, 17, 42]
        print(result["recommendations"])
    """
    check_arviz_installed()
    import arviz as az

    log_lik_ds = _to_log_likelihood_dataset(log_likelihood, var_name=var_name)
    loo_result = az.loo(log_lik_ds, var_name=var_name, pointwise=True)
    pareto_k   = np.asarray(loo_result.pareto_k)
    n_obs      = len(pareto_k)

    influential_idx = np.where(pareto_k >= threshold_bad)[0].tolist()
    very_bad_idx    = np.where(pareto_k >= threshold_very_bad)[0].tolist()

    # Actionable recommendations
    recommendations: list[str] = []
    if very_bad_idx:
        recommendations.append(
            f"{len(very_bad_idx)} observation(s) with Pareto-k >= "
            f"{threshold_very_bad} at indices {very_bad_idx}. "
            f"Investigate for data entry errors or outliers. "
            f"Consider a heavier-tailed likelihood."
        )
    if influential_idx and not very_bad_idx:
        recommendations.append(
            f"{len(influential_idx)} observation(s) with Pareto-k >= "
            f"{threshold_bad} at indices {influential_idx}. "
            f"These drive the posterior — check whether the model "
            f"accommodates them well."
        )
    if not influential_idx:
        recommendations.append(
            "No highly influential observations detected. "
            "All Pareto-k values are below the threshold."
        )

    return {
        "pareto_k":            pareto_k,
        "influential_indices": influential_idx,
        "very_bad_indices":    very_bad_idx,
        "n_observations":      n_obs,
        "n_influential":       len(influential_idx),
        "n_very_bad":          len(very_bad_idx),
        "warnings":            _build_loo_warnings(pareto_k),
        "recommendations":     recommendations,
    }


# ---------------------------------------------------------------------------
# power_scale_sensitivity
# ---------------------------------------------------------------------------

@workflow_function
def power_scale_sensitivity(
    posterior: Any,
    log_likelihood: Any,
    log_prior: Any,
    *,
    lower: float = 0.5,
    upper: float = 2.0,
    n_steps: int = 10,
    high_sensitivity_threshold: float = 0.3,
) -> dict:
    """Local sensitivity via power scaling of prior and likelihood.

    Raises the log prior or log likelihood to a scalar alpha and
    estimates the gradient of posterior summaries at alpha=1 via
    importance reweighting.  No refitting is required.

    A high gradient (|d(mean)/d(alpha)| > ``high_sensitivity_threshold``)
    means the posterior is sensitive to that component — useful for
    detecting prior-data conflict and weakly identified parameters.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.
    log_likelihood : array-like
        Pointwise log likelihood — shape ``(n_draw, n_obs)`` or
        ``(n_chain, n_draw, n_obs)``.
    log_prior : array-like
        Per-draw log prior values — shape ``(n_draw,)`` or
        ``(n_chain, n_draw)``.
    lower : float
        Lower bound of alpha grid. Default 0.5.
    upper : float
        Upper bound of alpha grid. Default 2.0.
    n_steps : int
        Number of alpha values. Default 10.
    high_sensitivity_threshold : float
        |d(mean)/d(alpha)| above which a warning is emitted.
        Default 0.3.

    Returns
    -------
    dict
        ::

            {
                "alpha_grid": np.ndarray,  # shape (n_steps,)
                "prior_sensitivity": {
                    "mu": {
                        "means":           np.ndarray,  # (n_steps,)
                        "stds":            np.ndarray,  # (n_steps,)
                        "sensitivity":     float,       # d(mean)/d(alpha) at alpha=1
                        "high_sensitivity": bool,
                    },
                    ...
                },
                "likelihood_sensitivity": {
                    "mu": { ... },
                    ...
                },
                "warnings": list[str],
            }

    Raises
    ------
    ValueError
        If ``log_likelihood`` or ``log_prior`` have unsupported shapes.

    Examples
    --------
    ::

        from probpipe.diagnostics import power_scale_sensitivity

        result = power_scale_sensitivity(posterior, log_lik, log_prior)
        result["prior_sensitivity"]["mu"]["sensitivity"]
        result["warnings"]
    """
    draws_dict  = extract_draws(posterior)
    param_names = list(draws_dict.keys())

    # -- Normalise shapes ---------------------------------------------------
    ll_arr  = np.asarray(log_likelihood, dtype=float)
    lpr_arr = np.asarray(log_prior,      dtype=float)

    if ll_arr.ndim == 3:
        n_chain, n_draw, n_obs = ll_arr.shape
        ll_arr  = ll_arr.reshape(n_chain * n_draw, n_obs)
        lpr_arr = lpr_arr.reshape(n_chain * n_draw)
    elif ll_arr.ndim == 2:
        n_draw = ll_arr.shape[0]
    else:
        raise ValueError(
            f"log_likelihood must be 2D (n_draw, n_obs) or "
            f"3D (n_chain, n_draw, n_obs). Got shape {ll_arr.shape}."
        )

    n_draw = ll_arr.shape[0]

    if lpr_arr.ndim != 1 or lpr_arr.shape[0] != n_draw:
        raise ValueError(
            f"log_prior must be 1D with {n_draw} entries. "
            f"Got shape {lpr_arr.shape}."
        )

    # -- Stack draws --------------------------------------------------------
    draws_stack = np.column_stack([
        np.asarray(draws_dict[v], dtype=float).reshape(-1)[:n_draw]
        for v in param_names
    ])  # (n_draw, n_params)

    ll_sum     = ll_arr.sum(axis=1)   # (n_draw,)
    alpha_grid = np.linspace(lower, upper, n_steps)
    delta      = (upper - lower) / max(n_steps - 1, 1)
    alpha1_idx = int(np.argmin(np.abs(alpha_grid - 1.0)))

    # -- Sweep over alpha ---------------------------------------------------
    prior_means: dict[str, list] = {v: [] for v in param_names}
    prior_stds:  dict[str, list] = {v: [] for v in param_names}
    lik_means:   dict[str, list] = {v: [] for v in param_names}
    lik_stds:    dict[str, list] = {v: [] for v in param_names}

    for alpha in alpha_grid:
        # Prior power-scaled
        summ = _reweighted_summary(
            draws_stack, param_names,
            alpha * lpr_arr + ll_sum,
        )
        for var in param_names:
            prior_means[var].append(summ[var]["mean"])
            prior_stds[var].append(summ[var]["std"])

        # Likelihood power-scaled
        summ = _reweighted_summary(
            draws_stack, param_names,
            lpr_arr + alpha * ll_sum,
        )
        for var in param_names:
            lik_means[var].append(summ[var]["mean"])
            lik_stds[var].append(summ[var]["std"])

    # -- Compute gradients and assemble results -----------------------------
    prior_sens_result: dict[str, dict] = {}
    lik_sens_result:   dict[str, dict] = {}
    diagnostic_warnings: list[str]     = []

    for var in param_names:
        pm = np.array(prior_means[var])
        ps = np.array(prior_stds[var])
        lm = np.array(lik_means[var])
        ls = np.array(lik_stds[var])

        prior_grad = _finite_diff_gradient(pm, alpha1_idx, delta)
        lik_grad   = _finite_diff_gradient(lm, alpha1_idx, delta)
        high_prior = abs(prior_grad) > high_sensitivity_threshold
        high_lik   = abs(lik_grad)   > high_sensitivity_threshold

        prior_sens_result[var] = {
            "means":            pm,
            "stds":             ps,
            "sensitivity":      prior_grad,
            "high_sensitivity": high_prior,
        }
        lik_sens_result[var] = {
            "means":            lm,
            "stds":             ls,
            "sensitivity":      lik_grad,
            "high_sensitivity": high_lik,
        }

        if high_prior:
            diagnostic_warnings.append(
                f"'{var}' shows high prior sensitivity "
                f"(d(mean)/d(alpha) = {prior_grad:.3f}). "
                f"The posterior is sensitive to prior specification."
            )
        if high_lik:
            diagnostic_warnings.append(
                f"'{var}' shows high likelihood sensitivity "
                f"(d(mean)/d(alpha) = {lik_grad:.3f}). "
                f"The posterior is sensitive to likelihood scaling."
            )

    return {
        "alpha_grid":             alpha_grid,
        "prior_sensitivity":      prior_sens_result,
        "likelihood_sensitivity": lik_sens_result,
        "warnings":               diagnostic_warnings,
    }