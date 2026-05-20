"""Internal ArviZ conversion layer.

This module is *private* — users should never import from it directly.
All public diagnostic functions live in mcmc.py and are re-exported
via the package __init__.py.
"""
from __future__ import annotations                          

from typing import Any                                      

import numpy as np                                          

try:
    import xarray as xr
except ImportError:
    xr = None


# ── Installation check ────────────────────────────────────────────────────────

def check_arviz_installed() -> None:
    """Raise a clear ImportError if ArviZ or xarray is missing."""
    try:
        import arviz as az  # noqa: F401
    except ImportError:
        raise ImportError(
            "ArviZ is required for diagnostics. "
            "Install with: pip install probpipe[diagnostics]"
        )
    if xr is None:
        raise ImportError(
            "xarray is required for diagnostics. "
            "Install with: pip install probpipe[diagnostics]"
        )


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
            return {f: np.asarray(raw[f]) for f in raw.fields}
        if isinstance(raw, dict):
            return {k: np.asarray(v) for k, v in raw.items()}

    # Case 2: EmpiricalDistribution with .samples
    if hasattr(posterior, "samples"):
        samples = posterior.samples
        if hasattr(samples, "fields"):
            return {f: np.asarray(samples[f]) for f in samples.fields}
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
) -> "xr.Dataset":
    """Convert a posterior distribution to an xarray.Dataset for ArviZ 1.0.

    For ``ApproximateDistribution`` (returned by ``condition_on``), builds
    the dataset from ``posterior.draws(chain=i)`` per chain, producing one
    named variable per field with shape ``(chain, draw)`` each. This
    preserves chain structure and gives ArviZ the named per-parameter
    variables it needs to compute scalar diagnostics (R-hat, ESS, MCSE).

    The ``inference_data`` DataTree is intentionally not used here — it
    stores parameters as a flat ``params`` vector without named fields,
    which ArviZ cannot reduce to per-parameter scalars.

    Falls back to flat construction for plain ``EmpiricalDistribution``.

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
        raise ImportError(
            "xarray is required. pip install probpipe[diagnostics]"
        )

    # ── Fast path: ApproximateDistribution with named chain structure ────────────
    # The inference_data DataTree stores parameters as a flat "params" vector
    # (shape (chain, draw, n_params)) without named fields, so it cannot be
    # used directly. Instead, build the dataset from per-chain draws keyed by
    # field name — this preserves chain structure AND gives named variables
    # that ArviZ diagnostics (rhat, ess, mcse) reduce to per-parameter scalars.
    if hasattr(posterior, "chains") and hasattr(posterior, "fields"):
        # chains: list of (num_draws, *flat_event) arrays
        # fields: parameter names from the record_template
        data_vars = {}
        for field in posterior.fields:
            # Stack chains: shape (num_chains, num_draws, *event_shape)
            chain_arrays = []
            for i, chain_arr in enumerate(posterior.chains):
                draws_i = posterior.draws(chain=i)
                field_arr = np.asarray(draws_i[field], dtype=float)
                chain_arrays.append(field_arr)
            stacked = np.stack(chain_arrays, axis=0)  # (chain, draw, *event)
            event_dims = [f"dim_{j}" for j in range(stacked.ndim - 2)]
            dims = ["chain", "draw"] + event_dims
            if var_names is None or field in var_names:
                data_vars[field] = xr.DataArray(stacked, dims=dims)
        return xr.Dataset(data_vars)

    # ── Final fallback: flat EmpiricalDistribution — no chain structure ───────
    draws = extract_draws(posterior)
    if var_names is not None:
        draws = {k: v for k, v in draws.items() if k in var_names}

    data_vars = {}
    for name, arr in draws.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]   # (1, n_draws)
        event_dims = [f"dim_{i}" for i in range(arr.ndim - 2)]
        dims = ["chain", "draw"] + event_dims
        data_vars[name] = xr.DataArray(arr, dims=dims)

    return xr.Dataset(data_vars)


# ── Warning builder ───────────────────────────────────────────────────────────

def build_warnings(
    rhat: dict[str, float],
    ess_bulk: dict[str, float],
    ess_tail: dict[str, float],
) -> list[str]:
    """Generate human-readable warnings from diagnostic values.

    Thresholds follow ArviZ / Stan recommendations:

    - R-hat > 1.01       → convergence concern
    - ESS (bulk) < 400   → too few effective samples
    - ESS (tail) < 400   → tail estimates unreliable

    Parameters
    ----------
    rhat : dict[str, float]
        Per-variable R-hat values.
    ess_bulk : dict[str, float]
        Per-variable bulk ESS values.
    ess_tail : dict[str, float]
        Per-variable tail ESS values.

    Returns
    -------
    list[str]
        Warning messages. Empty list means all diagnostics passed.
    """
    messages = []

    for var, val in rhat.items():
        if val > 1.01:
            messages.append(
                f"R-hat > 1.01 for '{var}' ({val:.4f}) — "
                f"chains may not have converged."
            )
    for var, val in ess_bulk.items():
        if val < 400:
            messages.append(
                f"Low ESS (bulk) for '{var}' ({val:.0f}) — "
                f"consider more iterations."
            )
    for var, val in ess_tail.items():
        if val < 400:
            messages.append(
                f"Low ESS (tail) for '{var}' ({val:.0f}) — "
                f"tail estimates may be unreliable."
            )

    return messages