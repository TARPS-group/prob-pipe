"""Unified interface for Posterior/Prior/Sequential Predictive Checks.

Bridges the existing JAX-native engine in
``probpipe.validation._predictive_check`` with the diagnostics workflow.

All functions mutate the input distribution object's ``_auxiliary`` in place
and return ``None``.

ArviZ-compatible data are written under::

    _auxiliary/arviz/

ProbPipe diagnostic summaries and run metadata are written under::

    _auxiliary/diagnostics/

Functions
---------
run_ppc   : one or more test functions on a single posterior/prior
run_spc   : sequential predictive check across a list of posteriors
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import numpy as np
import xarray as xr

from ..core.distribution import Distribution
from ..custom_types import PRNGKey
from ..validation._predictive_check import predictive_check
from ._utils import _resolve_generative_likelihood
from ._datatree import _add_group

__all__ = ["run_ppc", "run_spc"]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def _record_get(obj: Any, key: str, default: Any = None) -> Any:
    """Get a field from dict-like, Record-like, or attribute-like objects.

    ``predictive_check`` may be decorated by ``@workflow_function``. In that
    case, its dictionary return value can be coerced into a ``Record``. This
    helper supports:

    - plain dictionaries: ``obj.get(key)``
    - Record-like objects: ``obj[key]``
    - attribute-style objects: ``obj.key``
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


def _safe_float(value: Any) -> float:
    """Convert value to float while preserving valid 0.0 values.

    Do not use ``value or np.nan`` because valid values like ``0.0`` are falsey.
    """
    if value is None:
        return float("nan")

    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr)
        return float(arr.ravel()[0])
    except Exception:
        return float("nan")


def _as_numpy(obj: Any) -> np.ndarray | None:
    """Best-effort conversion of arrays or Distribution-like objects to NumPy."""
    if obj is None:
        return None

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


def _observed_data_to_dataset(observed_data: Any, var_name: str = "y") -> xr.Dataset:
    """Convert observed data into an ArviZ-compatible observed_data Dataset."""
    arr = np.asarray(observed_data)

    if arr.shape == ():
        da = xr.DataArray(arr)
    elif arr.ndim == 1:
        da = xr.DataArray(arr, dims=["obs"])
    else:
        dims = [f"obs_dim_{i}" for i in range(arr.ndim)]
        da = xr.DataArray(arr, dims=dims)

    return xr.Dataset({var_name: da})


def _replicated_data_to_dataset(y_rep: Any, var_name: str = "y") -> xr.Dataset:
    """Convert actual replicated observations into posterior_predictive Dataset.

    Expected common shapes:

    - ``(draw,)``
    - ``(draw, obs)``
    - ``(chain, draw)``
    - ``(chain, draw, obs)``

    If no chain dimension is present, a singleton chain dimension is added.
    """
    arr = np.asarray(y_rep)

    if arr.shape == ():
        arr = arr.reshape(1, 1)
        dims = ["chain", "draw"]

    elif arr.ndim == 1:
        # Draws of a scalar replicated quantity.
        arr = arr[np.newaxis, :]
        dims = ["chain", "draw"]

    elif arr.ndim == 2:
        # Assume (draw, obs), add chain dimension.
        arr = arr[np.newaxis, :, :]
        dims = ["chain", "draw", "obs"]

    elif arr.ndim == 3:
        # Assume already (chain, draw, obs).
        dims = ["chain", "draw", "obs"]

    else:
        # Assume first two dimensions are chain/draw and the rest are obs dims.
        dims = ["chain", "draw"] + [f"obs_dim_{i}" for i in range(arr.ndim - 2)]

    return xr.Dataset({var_name: xr.DataArray(arr, dims=dims)})


def _extract_actual_replicated_data(check_result: Any) -> np.ndarray | None:
    """Extract actual replicated observations if predictive_check provides them.

    This intentionally does NOT use ``replicated_statistics`` as the first
    choice because replicated statistics are scalar diagnostic summaries, not
    necessarily posterior predictive observations.

    If the predictive-check engine later returns actual replicated data under
    one of these names, we can populate ``arviz/posterior_predictive``.
    """
    candidate_keys = [
        "replicated_data",
        "replicated_datasets",
        "replicated_observations",
        "posterior_predictive",
        "prior_predictive",
        "y_rep",
        "y_replications",
        "simulated_data",
        "simulations",
    ]

    for key in candidate_keys:
        value = _record_get(check_result, key, None)
        arr = _as_numpy(value)
        if arr is not None:
            return arr

    return None


def _extract_replicated_statistics(check_result: Any) -> np.ndarray | None:
    """Extract replicated test statistics from predictive_check output."""
    value = _record_get(check_result, "replicated_statistics", None)
    return _as_numpy(value)



def _latest_ppc_dataset(dist: Distribution) -> xr.Dataset | None:
    """Return the current PPC diagnostic dataset, if present.

    Expected layout::

        dist._auxiliary["diagnostics"]["runs"]["ppc"]

    This matches the current simple layout where each diagnostic kind has one
    node. Repeated calls to ``run_ppc`` overwrite the previous PPC node.
    """
    aux = getattr(dist, "_auxiliary", None)
    if aux is None:
        return None

    try:
        node = aux["diagnostics"]["runs"]["ppc"]
    except Exception:
        return None

    for attr in ("ds", "dataset"):
        ds = getattr(node, attr, None)
        if isinstance(ds, xr.Dataset):
            return ds

    if isinstance(node, xr.Dataset):
        return node

    return None


def _replicated_statistics_summary(
    replicated_stats_by_fn: dict[str, np.ndarray | None],
) -> dict[str, list[float]] | None:
    """Summarize replicated test statistics as scalar values per test function.

    Do not store the full replicated-statistic array in diagnostics/runs/ppc,
    because DiagnosticRunView.result expects scalar or 1D variables.
    """
    if not replicated_stats_by_fn:
        return None

    summary = {
        "replicated_stat_mean": [],
        "replicated_stat_sd": [],
        "replicated_stat_q05": [],
        "replicated_stat_q50": [],
        "replicated_stat_q95": [],
    }

    any_available = False

    for _, values in replicated_stats_by_fn.items():
        if values is None:
            summary["replicated_stat_mean"].append(float("nan"))
            summary["replicated_stat_sd"].append(float("nan"))
            summary["replicated_stat_q05"].append(float("nan"))
            summary["replicated_stat_q50"].append(float("nan"))
            summary["replicated_stat_q95"].append(float("nan"))
            continue

        arr = np.asarray(values, dtype=float).ravel()

        if arr.size == 0:
            summary["replicated_stat_mean"].append(float("nan"))
            summary["replicated_stat_sd"].append(float("nan"))
            summary["replicated_stat_q05"].append(float("nan"))
            summary["replicated_stat_q50"].append(float("nan"))
            summary["replicated_stat_q95"].append(float("nan"))
            continue

        any_available = True

        summary["replicated_stat_mean"].append(float(np.nanmean(arr)))
        summary["replicated_stat_sd"].append(float(np.nanstd(arr)))
        summary["replicated_stat_q05"].append(float(np.nanquantile(arr, 0.05)))
        summary["replicated_stat_q50"].append(float(np.nanquantile(arr, 0.50)))
        summary["replicated_stat_q95"].append(float(np.nanquantile(arr, 0.95)))

    if not any_available:
        return None

    return summary

# ---------------------------------------------------------------------
# PPC
# ---------------------------------------------------------------------


def run_ppc(
    posterior: Distribution,
    test_fns: Callable | Sequence[Callable],
    observed_data=None,
    *,
    generative_likelihood=None,
    n_samples: int | None = None,
    n_replications: int = 500,
    key: PRNGKey | None = None,
) -> None:
    """Run one or more posterior/prior predictive checks.

    This function mutates ``posterior._auxiliary`` in place and returns ``None``.

    Parameters
    ----------
    posterior : Distribution
        Prior or posterior to sample parameters from.

    test_fns : callable or sequence of callables
        One or more test statistics mapping data to a scalar.

    observed_data : optional
        If provided, this performs posterior predictive checking. If ``None``,
        this behaves like prior predictive checking.

    generative_likelihood : optional
        Generative likelihood. If not provided, this is resolved from the
        posterior when possible.

    n_samples : int or None
        Number of observations per replicated dataset.

    n_replications : int
        Number of replicated datasets.

    key : PRNGKey or None
        JAX PRNG key.

    Returns
    -------
    None
    """
    gl = _resolve_generative_likelihood(posterior, generative_likelihood)

    if callable(test_fns):
        test_fns = [test_fns]
    else:
        test_fns = list(test_fns)

    results: dict[str, dict[str, Any]] = {}
    replicated_stats_by_fn: dict[str, np.ndarray | None] = {}

    # Actual replicated observations, if predictive_check returns them.
    y_rep_data: np.ndarray | None = None

    for fn in test_fns:
        name = getattr(fn, "__name__", repr(fn))

        check_result = predictive_check(
            distribution=posterior,
            generative_likelihood=gl,
            test_fn=fn,
            observed_data=observed_data,
            n_samples=n_samples,
            n_replications=n_replications,
            key=key,
        )

        p_val = _record_get(check_result, "p_value")
        obs_val = _record_get(check_result, "observed_statistic")

        results[name] = {
            "p_value": p_val,
            "observed": obs_val,
        }

        replicated_stats_by_fn[name] = _extract_replicated_statistics(check_result)

        if y_rep_data is None:
            candidate = _extract_actual_replicated_data(check_result)
            if candidate is not None:
                y_rep_data = candidate

    # ------------------------------------------------------------------
    # Write ArviZ-compatible groups when available
    # ------------------------------------------------------------------

    wrote_posterior_predictive = False
    wrote_observed_data = False

    if y_rep_data is not None:
        pp_ds = _replicated_data_to_dataset(y_rep_data, var_name="y")
        _add_group(posterior, "arviz/posterior_predictive", pp_ds)
        wrote_posterior_predictive = True

    if observed_data is not None:
        obs_ds = _observed_data_to_dataset(observed_data, var_name="y")
        _add_group(posterior, "arviz/observed_data", obs_ds)
        wrote_observed_data = True

    plot_ready = bool(wrote_posterior_predictive and wrote_observed_data)

    # ------------------------------------------------------------------
    # Write ProbPipe diagnostic run node
    # ------------------------------------------------------------------

    fn_names = list(results.keys())

    p_values = [
        _safe_float(results[name].get("p_value"))
        for name in fn_names
    ]

    observed_values = [
        _safe_float(results[name].get("observed"))
        for name in fn_names
    ]

    data_vars: dict[str, xr.DataArray] = {
        "p_value": xr.DataArray(
            p_values,
            dims=["test_fn"],
            coords={"test_fn": fn_names},
        ),
        "observed": xr.DataArray(
            observed_values,
            dims=["test_fn"],
            coords={"test_fn": fn_names},
        ),
    }

    rep_stat_summary = _replicated_statistics_summary(replicated_stats_by_fn)

    if rep_stat_summary is not None:
        for key, values in rep_stat_summary.items():
            data_vars[key] = xr.DataArray(
                values,
                dims=["test_fn"],
                coords={"test_fn": fn_names},
            )
    run_ds = xr.Dataset(data_vars)

    run_ds.attrs = {
        "kind": "ppc",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_replications": int(n_replications),
        "n_samples": -1 if n_samples is None else int(n_samples),
        "has_observed_data": observed_data is not None,
        "wrote_arviz_posterior_predictive": wrote_posterior_predictive,
        "wrote_arviz_observed_data": wrote_observed_data,
        "plot_fn": "az.plot_ppc",
        "plot_groups": json.dumps(["posterior_predictive", "observed_data"]),
        "plot_ready": plot_ready,
        "results_json": _json_dumps_safe(results),
    }

    _add_group(posterior, "diagnostics/runs/ppc", run_ds)

    return None


# ---------------------------------------------------------------------
# SPC
# ---------------------------------------------------------------------


def run_spc(
    distributions: Sequence[Distribution],
    test_fns: Callable | Sequence[Callable],
    observed_data_sequence: Sequence,
    *,
    generative_likelihood=None,
    n_samples: int | None = None,
    n_replications: int = 500,
    key: PRNGKey | None = None,
) -> None:
    """Run sequential predictive checks across a sequence of distributions.

    Each distribution is mutated in place. This function returns ``None``.

    The function calls ``run_ppc`` for each pair::

        distributions[t], observed_data_sequence[t]

    It also writes a simple SPC summary to the final distribution under::

        diagnostics/runs/spc

    Parameters
    ----------
    distributions : sequence of Distribution
        Posteriors at each time step.

    test_fns : callable or sequence of callables
        One or more test statistics.

    observed_data_sequence : sequence
        Observed data at each time step.

    generative_likelihood : optional
        Generative likelihood. If not provided, this is resolved from the first
        distribution.

    n_samples : int or None
        Number of observations per replicated dataset.

    n_replications : int
        Number of replicated datasets per time step.

    key : PRNGKey or None
        JAX PRNG key.

    Returns
    -------
    None
    """
    distributions = list(distributions)
    observed_data_sequence = list(observed_data_sequence)

    if len(distributions) != len(observed_data_sequence):
        raise ValueError(
            "distributions and observed_data_sequence must have the same length: "
            f"got {len(distributions)} and {len(observed_data_sequence)}."
        )

    if len(distributions) == 0:
        raise ValueError("distributions must contain at least one distribution.")

    gl = _resolve_generative_likelihood(distributions[0], generative_likelihood)

    if callable(test_fns):
        test_fns = [test_fns]
    else:
        test_fns = list(test_fns)

    fn_names = [getattr(fn, "__name__", repr(fn)) for fn in test_fns]

    for dist, obs in zip(distributions, observed_data_sequence):
        run_ppc(
            posterior=dist,
            test_fns=test_fns,
            observed_data=obs,
            generative_likelihood=gl,
            n_samples=n_samples,
            n_replications=n_replications,
            key=key,
        )

    p_values_by_fn: dict[str, list[float]] = {name: [] for name in fn_names}

    for dist in distributions:
        ppc_ds = _latest_ppc_dataset(dist)

        for name in fn_names:
            if ppc_ds is None or "p_value" not in ppc_ds:
                p_values_by_fn[name].append(float("nan"))
                continue

            try:
                val = ppc_ds["p_value"].sel(test_fn=name).item()
                p_values_by_fn[name].append(_safe_float(val))
            except Exception:
                p_values_by_fn[name].append(float("nan"))

    final_dist = distributions[-1]

    p_value_matrix = np.asarray(
        [p_values_by_fn[name] for name in fn_names],
        dtype=float,
    )

    spc_ds = xr.Dataset(
        {
            "p_value": xr.DataArray(
                p_value_matrix,
                dims=["test_fn", "time"],
                coords={
                    "test_fn": fn_names,
                    "time": np.arange(len(distributions)),
                },
            )
        }
    )

    spc_ds.attrs = {
        "kind": "spc",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_steps": len(distributions),
        "n_replications": int(n_replications),
        "n_samples": -1 if n_samples is None else int(n_samples),
        "test_fns": json.dumps(fn_names),
        "plot_ready": False,
        "plot_fn": "",
        "plot_groups": json.dumps([]),
    }

    _add_group(final_dist, "diagnostics/runs/spc", spc_ds)

    return None
