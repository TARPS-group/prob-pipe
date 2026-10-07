"""Unified interface for posterior predictive checks.

Bridges the replications and test statistics of
``probpipe.validation._predictive_check`` with the diagnostics workflow.

Design
------
This module has two layers:

1. Private pure ops returning payload dicts:

   - ``_ppc_op``

   These compute diagnostics and return structured payload dicts.
   They do not mutate ``_annotations``. Not part of the public API.

2. In-place writer wrappers returning ``None``:

   - ``add_ppc``

   These call the pure ops, write results into ``distribution._annotations``,
   and return ``None``.

ArviZ-compatible data are written under::

    _annotations/arviz/

ProbPipe diagnostic summaries and run metadata are written under::

    _annotations/diagnostics/
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import xarray as xr

from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution
from ..functions import _broker, _context
from ..functions._broker import _PROBPIPE_DISTRIBUTION_PROVIDER_ABI
from ..validation._predictive_check import (
    _observed_event,
    _planned_statistics,
    _predictive_joint,
    _replicated_statistics,
)
from ..validation._workflow_rng import _validate_positive_int
from ._datatree import _add_group
from ._utils import (
    _json_dumps_safe,
    _safe_float,
)
from ._workflow_rng import _claim_ppc_key

__all__ = [
    "add_ppc",
]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------
# Shared conversion helpers are imported from ._utils — see imports above.


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
    - ``(chain, draw, obs)``

    If no chain dimension is present, a singleton chain dimension is added.

    Parameters
    ----------
    y_rep : array-like
        The replicated observations. A two-dimensional array is read as
        ``(draw, obs)``, and a three-dimensional one as ``(chain, draw, obs)``.
    var_name : str
        The name of the dataset's one variable.

    Returns
    -------
    xr.Dataset
        A dataset whose variable has the dims ``chain`` and ``draw``, followed
        by ``obs`` for a two- or three-dimensional *y_rep*, or by ``obs_dim_0``,
        ``obs_dim_1``, and so on for one of higher dimension.

    Notes
    -----
    This function should only be used for actual replicated observations, not
    replicated test statistics.
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


def _replicated_statistics_summary(
    replicated_stats_by_fn: dict[str, np.ndarray | None],
) -> dict[str, list[float]] | None:
    """Summarize replicated test statistics as scalar values per test function.

    Do not store the full replicated-statistic array in
    ``diagnostics/runs/ppc`` because ``DiagnosticRunView.result`` expects scalar
    or 1D variables.
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


def _dataset_from_payload(payload: Mapping[str, Any]) -> xr.Dataset:
    """Return the xarray Dataset stored in a diagnostic payload."""
    ds = payload["dataset"]
    if not isinstance(ds, xr.Dataset):
        raise TypeError(
            f"Expected payload['dataset'] to be an xarray.Dataset, got {type(ds).__name__}."
        )
    return ds


# ---------------------------------------------------------------------
# PPC pure op
# ---------------------------------------------------------------------


def _ppc_op(
    posterior: Distribution,
    test_fns: Callable | Sequence[Callable],
    observed_data: Any | None = None,
    *,
    kernel: ConditionalDistribution,
    n_replications: int = 500,
) -> dict[str, Any]:
    """Pure PPC operation returning a payload dict.

    This function computes one or more posterior/prior predictive checks and
    returns a structured payload dict. It does not mutate ``posterior._annotations``.
    The replications are draws of the kernel's event from ``kernel * posterior``,
    and every test statistic is computed on the same replications. They are one
    workflow-owned random event of the enclosing workflow scope.

    Parameters
    ----------
    posterior : Distribution
        Prior or posterior over the kernel's given slots.

    test_fns : callable or sequence of callables
        One or more test statistics mapping data to a scalar.

    observed_data : optional
        If provided, the statistics of the observed data and their p-values
        are computed. It takes the forms
        :func:`~probpipe.validation.predictive_check` takes.

    kernel : ConditionalDistribution
        The law of the observations given the parameters.

    n_replications : int
        Number of replicated datasets.

    Returns
    -------
    dict
        Diagnostic payload dict containing scalar results, xarray datasets, and
        plotting metadata.
    """
    _context._assert_workflow_admission()
    planned_tests = _planned_statistics(test_fns)
    n_replications = _validate_positive_int("n_replications", n_replications)
    joint = _predictive_joint(kernel, posterior, "add_ppc")
    observed = None if observed_data is None else _observed_event(kernel, observed_data)

    results: dict[str, dict[str, Any]] = {}
    replicated_stats_by_fn: dict[str, np.ndarray | None] = {}

    with _broker._managed_stochastic_scope():
        # One event draws the replications that every test statistic reads.
        effect_key = _claim_ppc_key(
            source_index=0,
            n_replications=n_replications,
            provider_abi=_PROBPIPE_DISTRIBUTION_PROVIDER_ABI,
        )
        stats_by_name = _replicated_statistics(
            joint, kernel, planned_tests, n_replications, effect_key
        )

    for name, fn in planned_tests:
        stats_array = stats_by_name[name]
        p_val = None
        obs_val = None
        if observed is not None:
            obs_val = float(fn(observed))
            p_val = float(np.mean(stats_array >= obs_val))

        results[name] = {
            "p_value": p_val,
            "observed": obs_val,
        }

        replicated_stats_by_fn[name] = stats_array

    # ------------------------------------------------------------------
    # Build optional ArviZ-compatible datasets
    # ------------------------------------------------------------------

    observed_data_dataset = None

    if observed is not None:
        observed_data_dataset = _observed_data_to_dataset(observed, var_name="y")

    wrote_observed_data = observed_data_dataset is not None
    plot_ready = False  # replicated observations are not captured here.
    plot_fn = ""
    plot_groups: list[str] = []

    # ------------------------------------------------------------------
    # Build diagnostic xarray Dataset
    # ------------------------------------------------------------------

    fn_names = list(results.keys())

    p_values = [_safe_float(results[name].get("p_value")) for name in fn_names]

    observed_values = [_safe_float(results[name].get("observed")) for name in fn_names]

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
        for key_, values in rep_stat_summary.items():
            data_vars[key_] = xr.DataArray(
                values,
                dims=["test_fn"],
                coords={"test_fn": fn_names},
            )

    run_ds = xr.Dataset(data_vars)

    attrs = {
        "kind": "ppc",
        "timestamp": datetime.now(UTC).isoformat(),
        "n_replications": int(n_replications),
        "has_observed_data": observed_data is not None,
        "wrote_arviz_observed_data": wrote_observed_data,
        "plot_fn": plot_fn,
        "plot_groups": json.dumps(plot_groups),
        "plot_ready": plot_ready,
        "results_json": _json_dumps_safe(results),
    }

    run_ds.attrs = attrs

    result_payload = {
        "p_value": {name: _safe_float(results[name].get("p_value")) for name in fn_names},
        "observed": {name: _safe_float(results[name].get("observed")) for name in fn_names},
        "replicated_summary": rep_stat_summary or {},
    }

    return {
        "kind": "ppc",
        "result": result_payload,
        "dataset": run_ds,
        "posterior_predictive_dataset": None,
        "observed_data_dataset": observed_data_dataset,
        "plot_fn": plot_fn,
        "plot_ready": plot_ready,
        "attrs": attrs,
    }


# ---------------------------------------------------------------------
# PPC writer wrapper
# ---------------------------------------------------------------------


def _write_ppc_payload(posterior: Distribution, payload: Mapping[str, Any]) -> None:
    """Write a PPC diagnostic payload into ``posterior._annotations``."""
    posterior_predictive_dataset = payload["posterior_predictive_dataset"]
    observed_data_dataset = payload["observed_data_dataset"]

    if posterior_predictive_dataset is not None:
        _add_group(
            posterior,
            "arviz/posterior_predictive",
            posterior_predictive_dataset,
        )

    if observed_data_dataset is not None:
        _add_group(
            posterior,
            "arviz/observed_data",
            observed_data_dataset,
        )

    _add_group(
        posterior,
        "diagnostics/runs/ppc",
        _dataset_from_payload(payload),
    )


def add_ppc(
    posterior: Distribution,
    test_fns: Callable | Sequence[Callable],
    observed_data: Any | None = None,
    *,
    kernel: ConditionalDistribution,
    n_replications: int = 500,
) -> None:
    """Compute a PPC and write results into ``posterior._annotations``.

    This is the in-place wrapper around :func:`_ppc_op`. Calls ``_ppc_op``
    and writes the resulting payload into ``posterior._annotations``,
    then returns ``None``. The replications are draws of the kernel's event
    from ``kernel * posterior``, as :func:`~probpipe.validation.predictive_check`
    draws them, and every test statistic is computed on the same replications.

    The replications are one workflow-owned random event of the enclosing
    workflow scope. A call inside ``workflow_run(seed=...)`` therefore
    reproduces them, and a call outside every scope draws fresh replications.

    Parameters
    ----------
    posterior : Distribution
        Prior or posterior over the kernel's given slots.
    test_fns : callable or sequence of callables
        One or more test statistics mapping data to a scalar.
    observed_data : optional
        If provided, the statistics of the observed data and their p-values
        are computed. It takes the forms
        :func:`~probpipe.validation.predictive_check` takes.
    kernel : ConditionalDistribution
        The law of the observations given the parameters, such as the
        likelihood of a model ``likelihood * prior``.
    n_replications : int
        Number of replicated datasets.

    Raises
    ------
    TypeError
        If *kernel* is not a ``ConditionalDistribution``, *posterior* does not
        sample, *test_fns* is neither a callable nor an iterable, a test
        function is not callable, or *n_replications* is not an integer.
    ValueError
        If *posterior* does not produce every given slot of *kernel*, naming
        the missing slots; if *posterior* produces a component that *kernel*
        produces; if *test_fns* is empty or holds two test functions of the
        same name; or if *n_replications* is not positive. Use distinct named
        functions instead of multiple lambdas so result keys cannot collide.
    """
    payload = _ppc_op(
        posterior,
        test_fns=test_fns,
        observed_data=observed_data,
        kernel=kernel,
        n_replications=n_replications,
    )

    _write_ppc_payload(posterior, payload)

    return None
