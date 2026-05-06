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

def _draws_from_raw(raw: Any) -> dict[str, np.ndarray] | None:
    """Try to extract a draws dict from a raw value.

    Used as a helper by :func:`extract_draws` for both
    ``.draws()`` return values and ``.samples`` attributes.

    Returns ``None`` when the value is not in a recognised format
    so the caller can try the next fallback.
    """
    # Record / NumericRecord (ProbPipe internal)
    if hasattr(raw, "fields"):
        return {f: np.asarray(raw[f]) for f in raw.fields}
    # Plain dict
    if isinstance(raw, dict):
        return {k: np.asarray(v) for k, v in raw.items()}
    # xarray Dataset (ArviZ InferenceData .posterior, PyMC trace)
    if xr is not None and isinstance(raw, xr.Dataset):
        return {str(v): np.asarray(raw[v].values) for v in raw.data_vars}
    # Bare numpy / jax array — single unnamed variable
    if hasattr(raw, "__array__"):
        return {"x": np.asarray(raw)}
    return None


def extract_draws(posterior: Any) -> dict[str, np.ndarray]:
    """Extract named parameter draws from any posterior-like object.

    Accepted input formats (checked in order):

    1. **ArviZ** ``InferenceData`` — uses ``.posterior`` xarray Dataset.
    2. **Any object with** ``.draws()`` — e.g. the result of
       ``condition_on`` (``ApproximateDistribution`` or similar).
       The return value may be a ``Record`` / ``NumericRecord``,
       a plain dict, an xarray Dataset, or a bare array.
    3. **Any object with** ``.samples`` — e.g. ``EmpiricalDistribution``.
       Same accepted types as ``.draws()``.
    4. **Plain** ``dict[str, array]`` — passed directly.
    5. **Bare array** — treated as a single unnamed variable ``"x"``.

    Arrays may have shape:

    - ``(n_draws,)``                 — single chain, single param.
    - ``(n_chains, n_draws)``        — multi-chain, single param.
    - ``(n_draws, *event_shape)``    — single chain, vector param.
      The ``event_ndim`` argument to :func:`to_arviz_dataset` is
      required to disambiguate this from multi-chain data.

    Parameters
    ----------
    posterior : Any
        Any posterior-like object described above.

    Returns
    -------
    dict[str, np.ndarray]
        e.g. ``{"intercept": array([...]), "slope": array([...])}``.

    Raises
    ------
    TypeError
        If the input does not match any recognised format.
    """
    # Case 1: ArviZ InferenceData — use .posterior directly
    try:
        import arviz as az
        if isinstance(posterior, az.InferenceData):
            if hasattr(posterior, "posterior"):
                return {
                    str(v): np.asarray(posterior.posterior[v].values)
                    for v in posterior.posterior.data_vars
                }
    except ImportError:
        pass

    # Case 2: .draws() method — ApproximateDistribution / any MCMC result
    if hasattr(posterior, "draws"):
        raw = posterior.draws()
        result = _draws_from_raw(raw)
        if result is not None:
            return result

    # Case 3: .samples attribute — EmpiricalDistribution
    if hasattr(posterior, "samples"):
        result = _draws_from_raw(posterior.samples)
        if result is not None:
            return result

    # Case 4: plain dict
    if isinstance(posterior, dict):
        return {k: np.asarray(v) for k, v in posterior.items()}

    # Case 5: bare array
    if hasattr(posterior, "__array__"):
        return {"x": np.asarray(posterior)}

    raise TypeError(
        f"Cannot extract draws from {type(posterior).__name__}. "
        f"Expected one of: ArviZ InferenceData, object with .draws() "
        f"or .samples, plain dict, or bare array."
    )


# ── Format conversion ─────────────────────────────────────────────────────────

def to_arviz_dataset(
    posterior: Any,
    *,
    var_names: list[str] | None = None,
    event_ndim: dict[str, int] | None = None,
) -> "xr.Dataset":
    """Convert any posterior-like object to an xarray.Dataset for ArviZ 1.0.

    ArviZ 1.0 expects shape ``(chain, draw, *event_shape)``.

    Shape handling rules per variable:

    - ``(n_draws,)``
      → ``(1, n_draws)`` — single chain, scalar event.
    - ``(n_chains, n_draws)``
      → pass through — multi-chain, scalar event.
    - ``(n_draws, *event_shape)``
      → ``(1, n_draws, *event_shape)`` — single chain, vector event.
      Requires ``event_ndim[var] >= 1`` to distinguish from the
      multi-chain case above.
    - ``(n_chains, n_draws, *event_shape)``
      → pass through — multi-chain, vector event.

    Parameters
    ----------
    posterior : Any
        Any object accepted by :func:`extract_draws`.
    var_names : list[str] or None
        Subset of variables to include. ``None`` includes all.
    event_ndim : dict[str, int] or None
        Number of event dimensions per variable. Only required when
        draws have shape ``(n_draws, *event_shape)`` to distinguish
        from ``(n_chains, n_draws)``. Defaults to ``{}`` (all scalar).

    Returns
    -------
    xr.Dataset
        Dataset with dims ``(chain, draw, *event_shape)``.
    """
    if xr is None:
        raise ImportError(
            "xarray is required. pip install probpipe[diagnostics]"
        )

    draws = extract_draws(posterior)

    if var_names is not None:
        draws = {k: v for k, v in draws.items() if k in var_names}

    event_ndim = event_ndim or {}
    data_vars  = {}

    for name, arr in draws.items():
        arr   = np.asarray(arr, dtype=float)
        e_ndim = event_ndim.get(name, 0)

        if arr.ndim == 1:
            # (n_draws,) → (1, n_draws): scalar event, single chain
            arr = arr[np.newaxis, :]

        elif arr.ndim == 2 and e_ndim == 0:
            # (n_chains, n_draws) or (n_draws, event_size)?
            # With e_ndim=0 we treat it as multi-chain → pass through.
            # Users with (n_draws, event_size) must pass event_ndim=1.
            pass

        elif arr.ndim >= 2 and e_ndim >= 1:
            # (n_draws, *event_shape) → (1, n_draws, *event_shape)
            arr = arr[np.newaxis, :]

        # (n_chains, n_draws, *event_shape) → pass through

        event_dims = [f"dim_{i}" for i in range(arr.ndim - 2)]
        dims = ["chain", "draw"] + event_dims
        data_vars[name] = xr.DataArray(arr, dims=dims)

    return xr.Dataset(data_vars)


# ── Warning builder ───────────────────────────────────────────────────────────

def build_warnings(
    rhat: dict[str, float],
    ess: dict[str, dict[str, float]],
) -> list[str]:
    """Generate human-readable warnings from diagnostic tree values.

    Thresholds follow ArviZ / Stan recommendations:

    - R-hat > 1.01      → convergence concern
    - ESS bulk < 400    → too few effective samples
    - ESS tail < 400    → tail estimates unreliable

    Parameters
    ----------
    rhat : dict[str, float]
        Per-variable R-hat values.
    ess : dict[str, dict[str, float]]
        Nested ESS tree: ``{"bulk": {...}, "tail": {...}}``.

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
    for var, val in ess.get("bulk", {}).items():
        if val < 400:
            messages.append(
                f"Low ESS (bulk) for '{var}' ({val:.0f}) — "
                f"consider more iterations."
            )
    for var, val in ess.get("tail", {}).items():
        if val < 400:
            messages.append(
                f"Low ESS (tail) for '{var}' ({val:.0f}) — "
                f"tail estimates may be unreliable."
            )

    return messages