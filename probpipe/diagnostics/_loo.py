"""LOO/PSIS diagnostics interface.

This module provides an in-place ProbPipe diagnostics wrapper around ArviZ's
LOO/PSIS functionality.

Design
------
Functions in this file mutate ``posterior._auxiliary`` in place and return
``None``. They are intentionally plain Python functions, not
``@workflow_function``s, because diagnostics are post-hoc annotations on an
already-fitted posterior.

ArviZ-compatible data are stored under::

    _auxiliary/arviz/

ProbPipe diagnostic summaries are stored under::

    _auxiliary/diagnostics/runs/loo

Main function
-------------
loo
    Compute PSIS-LOO using ArviZ and attach scalar summaries to
    ``posterior._auxiliary``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import xarray as xr

from ..core.distribution import Distribution
from ._datatree import _add_group

__all__ = ["loo", "run_loo"]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def _safe_float(value: Any) -> float:
    """Convert value to float, returning NaN if conversion fails."""
    if value is None:
        return float("nan")

    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr)
        return float(arr.ravel()[0])
    except Exception:
        return float("nan")


def _safe_int(value: Any) -> int:
    """Convert value to int, returning -1 if conversion fails."""
    if value is None:
        return -1

    try:
        return int(value)
    except Exception:
        try:
            return int(np.asarray(value).ravel()[0])
        except Exception:
            return -1


def _record_get(obj: Any, key: str, default: Any = None) -> Any:
    """Get a field from dict-like, pandas-Series-like, or attribute-like objects.

    ArviZ's ``az.loo`` returns an ELPDData object, which behaves somewhat like a
    pandas Series but also exposes attributes. This helper supports several
    access styles.
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    try:
        return obj[key]
    except Exception:
        pass

    try:
        return getattr(obj, key)
    except Exception:
        pass

    get = getattr(obj, "get", None)
    if callable(get):
        try:
            return get(key, default)
        except Exception:
            pass

    return default


def _as_numpy(obj: Any) -> np.ndarray | None:
    """Best-effort conversion to NumPy array."""
    if obj is None:
        return None

    if isinstance(obj, xr.DataArray):
        try:
            return np.asarray(obj.values)
        except Exception:
            return None

    if hasattr(obj, "values"):
        try:
            return np.asarray(obj.values)
        except Exception:
            pass

    if hasattr(obj, "samples"):
        try:
            return np.asarray(obj.samples)
        except Exception:
            pass

    try:
        return np.asarray(obj)
    except Exception:
        return None


def _json_dumps_safe(obj: Any) -> str:
    """JSON-dump helper for xarray attrs."""
    try:
        return json.dumps(obj)
    except TypeError:
        try:
            return json.dumps(str(obj))
        except Exception:
            return "{}"


# ---------------------------------------------------------------------
# ArviZ tree helpers
# ---------------------------------------------------------------------


def _get_arviz_tree(posterior: Distribution) -> Any:
    """Return the ArviZ-compatible subtree for a posterior.

    Preferred layout::

        posterior._auxiliary["arviz"]

    This function is defensive so that it also works during transition periods
    where ``posterior.inference_data`` may already return the ArviZ-compatible
    object.
    """
    aux = getattr(posterior, "_auxiliary", None)

    if aux is not None:
        try:
            return aux["arviz"]
        except Exception:
            pass

    try:
        idata = posterior.inference_data
        if idata is not None:
            # If inference_data is accidentally the full auxiliary tree,
            # prefer its /arviz subtree when present.
            try:
                return idata["arviz"]
            except Exception:
                return idata
    except Exception:
        pass

    return aux


def _has_group(tree: Any, path: str) -> bool:
    """Return True if a slash-separated path exists in a DataTree-like object."""
    if tree is None:
        return False

    node = tree
    for part in path.strip("/").split("/"):
        if not part:
            continue
        try:
            node = node[part]
        except Exception:
            return False

    return True


def _has_required_loo_pit_groups(arviz_tree: Any) -> bool:
    """Check whether groups needed for az.plot_loo_pit appear to exist.

    ArviZ LOO-PIT plotting generally needs observed data, posterior predictive
    samples, and log likelihood. Exact requirements may vary by ArviZ version.
    """
    return (
        _has_group(arviz_tree, "observed_data")
        and _has_group(arviz_tree, "posterior_predictive")
        and _has_group(arviz_tree, "log_likelihood")
    )


# ---------------------------------------------------------------------
# Log-likelihood conversion
# ---------------------------------------------------------------------


def _log_likelihood_to_dataset(
    log_likelihood: Any,
    *,
    var_name: str = "y",
) -> xr.Dataset:
    """Convert log likelihood input to an xarray Dataset.

    Common accepted shapes:

    - ``(draw, obs)``
    - ``(chain, draw, obs)``
    - ``(draw,)``
    - ``(chain, draw)``

    For PSIS-LOO, pointwise log likelihood is preferred, usually with shape
    ``(chain, draw, obs)`` or ``(draw, obs)``.
    """
    if isinstance(log_likelihood, xr.Dataset):
        return log_likelihood

    if isinstance(log_likelihood, xr.DataArray):
        name = log_likelihood.name or var_name
        return xr.Dataset({name: log_likelihood})

    arr = np.asarray(log_likelihood)

    if arr.shape == ():
        # Scalar log likelihood is not ideal for LOO, but store defensively.
        arr = arr.reshape(1, 1)
        dims = ["chain", "draw"]

    elif arr.ndim == 1:
        # Draws of a scalar total log likelihood.
        # Not ideal for pointwise LOO, but ArviZ may still reject it clearly.
        arr = arr[np.newaxis, :]
        dims = ["chain", "draw"]

    elif arr.ndim == 2:
        # Assume (draw, obs), add singleton chain dimension.
        arr = arr[np.newaxis, :, :]
        dims = ["chain", "draw", "obs"]

    elif arr.ndim == 3:
        # Assume (chain, draw, obs).
        dims = ["chain", "draw", "obs"]

    else:
        # Assume first two dimensions are chain/draw and the rest are obs dims.
        dims = ["chain", "draw"] + [f"obs_dim_{i}" for i in range(arr.ndim - 2)]

    return xr.Dataset({var_name: xr.DataArray(arr, dims=dims)})


# ---------------------------------------------------------------------
# LOO result extraction
# ---------------------------------------------------------------------


def _extract_pointwise_array(value: Any) -> np.ndarray | None:
    """Extract and flatten pointwise ArviZ arrays such as pareto_k or loo_i."""
    arr = _as_numpy(value)

    if arr is None:
        return None

    try:
        return np.asarray(arr, dtype=float).ravel()
    except Exception:
        return None


def _pareto_k_summary(pareto_k: Any, good_k: float | None = None) -> dict[str, float | int]:
    """Summarize Pareto-k values as scalar diagnostics."""
    arr = _extract_pointwise_array(pareto_k)

    if arr is None or arr.size == 0:
        return {
            "pareto_k_max": float("nan"),
            "pareto_k_mean": float("nan"),
            "pareto_k_bad_count": -1,
            "pareto_k_bad_fraction": float("nan"),
        }

    if good_k is None or np.isnan(good_k):
        threshold = 0.7
    else:
        threshold = float(good_k)

    bad = arr > threshold

    return {
        "pareto_k_max": float(np.nanmax(arr)),
        "pareto_k_mean": float(np.nanmean(arr)),
        "pareto_k_bad_count": int(np.nansum(bad)),
        "pareto_k_bad_fraction": float(np.nanmean(bad)),
    }


def _pointwise_dataarray(
    values: Any,
    *,
    name: str,
    dim: str = "obs",
) -> xr.DataArray | None:
    """Convert pointwise values to a 1D DataArray, if available."""
    arr = _extract_pointwise_array(values)

    if arr is None:
        return None

    return xr.DataArray(
        arr,
        dims=[dim],
        coords={dim: np.arange(arr.size)},
        name=name,
    )


# ---------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------


def loo(
    posterior: Distribution,
    *,
    log_likelihood: Any | None = None,
    var_name: str = "y",
    pointwise: bool = True,
    scale: str | None = None,
    reff: float | None = None,
    force: bool = False,
    store_pointwise: bool = True,
) -> None:
    """Compute PSIS-LOO and attach results to ``posterior._auxiliary``.

    This function mutates ``posterior._auxiliary`` in place and returns
    ``None``.

    Parameters
    ----------
    posterior : Distribution
        Posterior distribution whose ArviZ-compatible inference data are stored
        under ``posterior._auxiliary["arviz"]``.

    log_likelihood : optional
        Pointwise log likelihood. If provided, it is written to
        ``_auxiliary/arviz/log_likelihood`` before calling ArviZ.

    var_name : str
        Variable name to use when converting raw log-likelihood arrays into an
        xarray Dataset.

    pointwise : bool
        Passed to ``az.loo``. If True, ArviZ returns pointwise quantities such
        as ``loo_i`` and ``pareto_k`` when available.

    scale : str or None
        Optional ArviZ scale argument, e.g. ``"log"``, ``"negative_log"``, or
        ``"deviance"`` depending on ArviZ version.

    reff : float or None
        Optional relative MCMC efficiency passed to ``az.loo``.

    force : bool
        If False and ``diagnostics/runs/loo`` already exists, skip computation.
        If True, recompute and overwrite the existing LOO diagnostic node.

    store_pointwise : bool
        If True, store pointwise ``pareto_k`` and ``loo_i`` values as 1D arrays
        when available. These are stored in the diagnostic run dataset.

    Returns
    -------
    None
    """
    import arviz as az

    if not force and _has_group(getattr(posterior, "_auxiliary", None), "diagnostics/runs/loo"):
        return None

    # ------------------------------------------------------------------
    # Add/update log_likelihood group under /arviz/
    # ------------------------------------------------------------------

    if log_likelihood is not None:
        log_lik_ds = _log_likelihood_to_dataset(log_likelihood, var_name=var_name)
        _add_group(posterior, "arviz/log_likelihood", log_lik_ds)

    arviz_tree = _get_arviz_tree(posterior)

    if arviz_tree is None:
        raise ValueError(
            "No ArviZ-compatible inference data found. Expected "
            "posterior._auxiliary['arviz'] or posterior.inference_data."
        )

    if not _has_group(arviz_tree, "log_likelihood"):
        raise ValueError(
            "No log_likelihood group found. Provide log_likelihood=... or "
            "ensure posterior._auxiliary['arviz']['log_likelihood'] exists."
        )

    # ------------------------------------------------------------------
    # Run ArviZ LOO
    # ------------------------------------------------------------------

    loo_kwargs: dict[str, Any] = {
        "pointwise": pointwise,
    }

    if scale is not None:
        loo_kwargs["scale"] = scale

    if reff is not None:
        loo_kwargs["reff"] = reff

    loo_result = az.loo(arviz_tree, **loo_kwargs)

    # ------------------------------------------------------------------
    # Extract scalar summaries
    # ------------------------------------------------------------------

    elpd_loo = _safe_float(_record_get(loo_result, "elpd_loo"))
    se = _safe_float(_record_get(loo_result, "se"))
    p_loo = _safe_float(_record_get(loo_result, "p_loo"))

    # Some ArviZ versions expose looic; others only expose elpd_loo.
    looic_raw = _record_get(loo_result, "looic", None)
    looic = _safe_float(looic_raw)

    if np.isnan(looic) and not np.isnan(elpd_loo):
        looic = float(-2.0 * elpd_loo)

    n_samples = _safe_int(_record_get(loo_result, "n_samples"))
    n_data_points = _safe_int(_record_get(loo_result, "n_data_points"))

    warning_value = _record_get(loo_result, "warning", False)
    warning_numeric = 1.0 if bool(warning_value) else 0.0

    good_k = _safe_float(_record_get(loo_result, "good_k", np.nan))

    pareto_k = _record_get(loo_result, "pareto_k", None)
    loo_i = _record_get(loo_result, "loo_i", None)

    pk_summary = _pareto_k_summary(
        pareto_k,
        good_k=None if np.isnan(good_k) else good_k,
    )

    # ------------------------------------------------------------------
    # Build diagnostics dataset
    # ------------------------------------------------------------------

    data_vars: dict[str, xr.DataArray] = {
        "elpd_loo": xr.DataArray(elpd_loo),
        "se": xr.DataArray(se),
        "p_loo": xr.DataArray(p_loo),
        "looic": xr.DataArray(looic),
        "n_samples": xr.DataArray(float(n_samples)),
        "n_data_points": xr.DataArray(float(n_data_points)),
        "warning": xr.DataArray(warning_numeric),
        "good_k": xr.DataArray(good_k),
        "pareto_k_max": xr.DataArray(pk_summary["pareto_k_max"]),
        "pareto_k_mean": xr.DataArray(pk_summary["pareto_k_mean"]),
        "pareto_k_bad_count": xr.DataArray(float(pk_summary["pareto_k_bad_count"])),
        "pareto_k_bad_fraction": xr.DataArray(pk_summary["pareto_k_bad_fraction"]),
    }

    if store_pointwise:
        pareto_k_da = _pointwise_dataarray(pareto_k, name="pareto_k", dim="obs")
        if pareto_k_da is not None:
            data_vars["pareto_k"] = pareto_k_da

        loo_i_da = _pointwise_dataarray(loo_i, name="loo_i", dim="obs")
        if loo_i_da is not None:
            data_vars["loo_i"] = loo_i_da

    run_ds = xr.Dataset(data_vars)

    plot_ready = _has_required_loo_pit_groups(arviz_tree)

    run_ds.attrs = {
        "kind": "loo",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": "arviz",
        "pointwise": bool(pointwise),
        "scale": "" if scale is None else str(scale),
        "reff": float("nan") if reff is None else float(reff),
        "var_name": str(var_name),
        "plot_fn": "az.plot_loo_pit",
        "plot_groups": json.dumps(
            ["posterior", "log_likelihood", "posterior_predictive", "observed_data"]
        ),
        "plot_ready": bool(plot_ready),
        "compare_ready": True,
        "warning_bool": bool(warning_value),
        "loo_result_repr": repr(loo_result),
        "loo_result_json": _json_dumps_safe(
            {
                "elpd_loo": elpd_loo,
                "se": se,
                "p_loo": p_loo,
                "looic": looic,
                "n_samples": n_samples,
                "n_data_points": n_data_points,
                "warning": bool(warning_value),
                "good_k": good_k,
                **pk_summary,
            }
        ),
    }

    _add_group(posterior, "diagnostics/runs/loo", run_ds)

    return None


def run_loo(*args, **kwargs) -> None:
    """Alias for :func:`loo`.

    Provided for users who prefer verb-style diagnostic names.
    """
    return loo(*args, **kwargs)
