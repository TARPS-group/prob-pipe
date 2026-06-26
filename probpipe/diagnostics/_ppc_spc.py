"""Unified interface for posterior predictive checks.

Bridges the existing JAX-native engine in
``probpipe.validation._predictive_check`` with the diagnostics workflow.

Design
------
This module has two layers:

1. Private pure ops returning ``Record`` objects:

   - ``_ppc_op``

   These compute diagnostics and return structured ``Record`` results.
   They do not mutate ``_auxiliary``. Not part of the public API.

2. In-place writer wrappers returning ``None``:

   - ``add_ppc``

   These call the pure ops, write results into ``distribution._auxiliary``,
   and return ``None``.

ArviZ-compatible data are written under::

    _auxiliary/arviz/

ProbPipe diagnostic summaries and run metadata are written under::

    _auxiliary/diagnostics/
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import xarray as xr

from .._utils import _auto_key
from ..core.distribution import Distribution
from ..core.record import Record
from ..custom_types import PRNGKey
from ..validation._predictive_check import (
    _predictive_check_batched,
    _predictive_check_loop,
    _supports_key_arg,
)
from ._datatree import _add_group
from ._utils import (
    _json_dumps_safe,
    _resolve_generative_likelihood,
    _safe_float,
)

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
    - ``(chain, draw)``
    - ``(chain, draw, obs)``

    If no chain dimension is present, a singleton chain dimension is added.

    Important
    ---------
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


def _dataset_from_record(record: Record) -> xr.Dataset:
    """Return the xarray Dataset stored in a diagnostic Record."""
    ds = record["dataset"]
    if not isinstance(ds, xr.Dataset):
        raise TypeError(
            f"Expected record['dataset'] to be an xarray.Dataset, got {type(ds).__name__}."
        )
    return ds


# ---------------------------------------------------------------------
# PPC pure op
# ---------------------------------------------------------------------


def _ppc_op(
    posterior: Distribution,
    test_fns: Callable | Sequence[Callable],
    observed_data=None,
    *,
    num_observations: int | None = None,
    generative_likelihood=None,
    n_replications: int = 500,
    key: PRNGKey | None = None,
) -> Record:
    """Pure PPC operation returning a ``Record``.

    This function computes one or more posterior/prior predictive checks and
    returns a structured ``Record``. It does not mutate ``posterior._auxiliary``.

    Parameters
    ----------
    posterior : Distribution
        Prior or posterior to sample parameters from.

    test_fns : callable or sequence of callables
        One or more test statistics mapping data to a scalar.

    observed_data : optional
        If provided, this performs posterior predictive checking. If ``None``,
        this behaves like prior predictive checking.

    num_observations : int, optional
        Number of observations per replicated dataset. Required when
        ``observed_data`` is ``None``; otherwise defaults to ``len(observed_data)``.

    generative_likelihood : optional
        Generative likelihood. If not provided, this is resolved from the
        posterior when possible.

    n_replications : int
        Number of replicated datasets.

    key : PRNGKey or None
        JAX PRNG key.

    Returns
    -------
    Record
        Diagnostic Record containing scalar results, xarray datasets, and
        plotting metadata.
    """
    gl = _resolve_generative_likelihood(posterior, generative_likelihood)

    if callable(test_fns):
        test_fns = [test_fns]
    else:
        test_fns = list(test_fns)

    results: dict[str, dict[str, Any]] = {}
    replicated_stats_by_fn: dict[str, np.ndarray | None] = {}

    for fn in test_fns:
        name = getattr(fn, "__name__", repr(fn))

        if num_observations is None:
            if observed_data is None:
                raise ValueError("num_observations is required when observed_data is not provided.")
            _n_samples = len(observed_data)
        else:
            _n_samples = int(num_observations)

        if _n_samples < 1:
            raise ValueError("num_observations must be a positive integer.")

        _key = key if key is not None else _auto_key()

        if _supports_key_arg(gl):
            stats_array = _predictive_check_batched(
                posterior,
                gl,
                fn,
                _n_samples,
                n_replications,
                _key,
            )
        else:
            stats_array = _predictive_check_loop(
                posterior,
                gl,
                fn,
                _n_samples,
                n_replications,
                _key,
            )

        p_val = None
        obs_val = None
        if observed_data is not None:
            obs_val = float(fn(observed_data))
            p_val = float(np.mean(stats_array >= obs_val))

        results[name] = {
            "p_value": p_val,
            "observed": obs_val,
        }

        replicated_stats_by_fn[name] = np.asarray(stats_array, dtype=np.float64)

        # y_rep_data not available from direct helper calls — skip ArviZ
        # posterior_predictive population for now.

    # ------------------------------------------------------------------
    # Build optional ArviZ-compatible datasets
    # ------------------------------------------------------------------

    observed_data_dataset = None

    if observed_data is not None:
        observed_data_dataset = _observed_data_to_dataset(observed_data, var_name="y")

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

    result_record = Record(
        name="ppc_result",
        p_value={name: _safe_float(results[name].get("p_value")) for name in fn_names},
        observed={name: _safe_float(results[name].get("observed")) for name in fn_names},
        replicated_summary=rep_stat_summary or {},
    )

    return Record(
        name="ppc_diagnostic",
        kind="ppc",
        result=result_record,
        dataset=run_ds,
        posterior_predictive_dataset=None,
        observed_data_dataset=observed_data_dataset,
        plot_fn=plot_fn,
        plot_ready=plot_ready,
        attrs=attrs,
    )


# ---------------------------------------------------------------------
# PPC writer wrapper
# ---------------------------------------------------------------------


def _write_ppc_record(posterior: Distribution, record: Record) -> None:
    """Write a PPC diagnostic Record into ``posterior._auxiliary``."""
    posterior_predictive_dataset = record["posterior_predictive_dataset"]
    observed_data_dataset = record["observed_data_dataset"]

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
        _dataset_from_record(record),
    )


def add_ppc(
    posterior: Distribution,
    test_fns: Callable | Sequence[Callable],
    observed_data=None,
    *,
    num_observations: int | None = None,
    generative_likelihood=None,
    n_replications: int = 500,
    key: PRNGKey | None = None,
) -> None:
    """Compute a PPC and write results into ``posterior._auxiliary``.

    This is the in-place wrapper around :func:`_ppc_op`. Calls ``_ppc_op``
    and writes the resulting ``Record`` into ``posterior._auxiliary``,
    then returns ``None``.

    Parameters
    ----------
    posterior : Distribution
        Prior or posterior to sample parameters from.
    test_fns : callable or sequence of callables
        One or more test statistics mapping data to a scalar.
    observed_data : optional
        If provided, performs posterior predictive checking. If ``None``,
        behaves like prior predictive checking.
    num_observations : int, optional
        Number of observations per replicated dataset. Required when
        ``observed_data`` is ``None``; otherwise defaults to ``len(observed_data)``.
    generative_likelihood : optional
        Generative likelihood. Resolved from ``posterior`` if not provided.
    n_replications : int
        Number of replicated datasets.
    key : PRNGKey or None
        JAX PRNG key.
    """
    record = _ppc_op(
        posterior,
        test_fns=test_fns,
        observed_data=observed_data,
        num_observations=num_observations,
        generative_likelihood=generative_likelihood,
        n_replications=n_replications,
        key=key,
    )

    _write_ppc_record(posterior, record)

    return None
