"""Internal ArviZ/xarray conversion layer.

This module is private. Public diagnostic functions live in
``probpipe.diagnostics`` and write ProbPipe-computed results under
``posterior._auxiliary["diagnostics"]``.

The bridge owns conversion into ArviZ-compatible datasets and raw diagnostic
inputs stored under ``posterior._auxiliary["arviz"]``. Users should normally
interact with ``add_mcmc_diagnostics``, ``add_ppc``, ``add_loo``, and
``posterior.diagnostics`` instead of importing this module directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Absolute (not relative) so this file stays loadable standalone — the
# missing-xarray fallback test execs it outside the package.
from probpipe.diagnostics._utils import _leaf_keys

try:
    import xarray as xr
except ImportError:
    xr = None


# ── Installation check ────────────────────────────────────────────────────────


def check_arviz_installed() -> None:
    """Raise a clear ImportError if ArviZ or xarray is missing."""
    try:
        import arviz as az  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ArviZ is required for diagnostics. Install with: pip install arviz xarray"
        ) from exc
    if xr is None:
        raise ImportError("xarray is required for diagnostics. Install with: pip install xarray")


# ── Draw extraction ───────────────────────────────────────────────────────────


def extract_draws(posterior: Any) -> dict[str, np.ndarray]:
    """Extract named parameter draws from a posterior distribution.

    Handles two cases:

    1. **ApproximateDistribution** (from ``condition_on``) with
       ``.draws()`` returning a ``Record`` / ``NumericRecord`` or
       plain dict.
    2. **EmpiricalDistribution** with ``.samples`` as a
       ``Record`` / ``NumericRecord``.

    Parameters
    ----------
    posterior : Distribution
        Any posterior returned by ``condition_on`` or an
        ``EmpiricalDistribution``.

    Returns
    -------
    dict[str, np.ndarray]
        e.g. ``{"intercept": array([...]), "slope": array([...])}``.

    Raises
    ------
    TypeError
        If the posterior has neither ``.draws()`` nor ``.samples``.
    """
    # Case 1: ApproximateDistribution with .draws()
    if hasattr(posterior, "draws"):
        raw = posterior.draws()
        if hasattr(raw, "fields"):
            # One variable per leaf field, keyed by its full /-path (see
            # ``_leaf_keys`` for the nested-vs-duck-typed rule).
            return {k: np.asarray(raw[k]) for k in _leaf_keys(raw)}
        if isinstance(raw, dict):
            return {k: np.asarray(v) for k, v in raw.items()}

    # Case 2: EmpiricalDistribution with .samples
    if hasattr(posterior, "samples"):
        samples = posterior.samples
        if hasattr(samples, "fields"):
            return {k: np.asarray(samples[k]) for k in _leaf_keys(samples)}
        return {"x": np.asarray(samples)}

    raise TypeError(
        f"Cannot extract draws from {type(posterior).__name__}. "
        f"Expected a posterior with .draws() or .samples attribute."
    )


# ── Format conversion ─────────────────────────────────────────────────────────


def to_arviz_dataset(
    posterior: Any,
    *,
    var_names: list[str] | None = None,
) -> xr.Dataset:
    """Convert a posterior distribution to an xarray.Dataset for ArviZ 1.0.

    For ``ApproximateDistribution``, delegates to
    ``_datatree_store.to_named_posterior_dataset`` which builds variables with
    dims ``(chain, draw, *event_shape)``.

    Falls back to flat construction for plain ``EmpiricalDistribution``
    (no chain structure).

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.
    var_names : list[str] or None
        Subset of variables to include. ``None`` includes all.

    Returns
    -------
    xr.Dataset
        Dataset with dims ``(chain, draw, *event_shape)``.
    """
    if xr is None:
        raise ImportError("xarray is required. Install with: pip install xarray")

    # ── ApproximateDistribution: delegate to the canonical builder ────────────
    if hasattr(posterior, "chains") and hasattr(posterior, "fields"):
        from ._datatree_store import to_named_posterior_dataset

        ds = to_named_posterior_dataset(posterior)
        if var_names is not None:
            ds = ds[var_names]
        return ds

    # ── Fallback: flat EmpiricalDistribution — no chain structure ─────────────
    draws = extract_draws(posterior)
    if var_names is not None:
        draws = {k: v for k, v in draws.items() if k in var_names}

    data_vars = {}
    for name, arr in draws.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim >= 1:
            arr = arr.reshape((1, *arr.shape))
        event_dims = [f"dim_{i}" for i in range(arr.ndim - 2)]
        dims = ["chain", "draw", *event_dims]
        data_vars[name] = xr.DataArray(arr, dims=dims)

    return xr.Dataset(data_vars)
