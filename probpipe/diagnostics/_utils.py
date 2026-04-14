"""Shared utilities for probpipe.diagnostics.

Provides:
  - to_xarray_dataset()             : convert samples to ArviZ 1.0 compatible xarray
  - _resolve_generative_likelihood(): auto-detect generative likelihood from posterior
  - check_arviz_version()           : warn if ArviZ < 1.0
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "to_xarray_dataset",
    "_resolve_generative_likelihood",
    "check_arviz_version",
]


# ---------------------------------------------------------------------------
# ArviZ version check
# ---------------------------------------------------------------------------

def check_arviz_version() -> None:
    """Warn if ArviZ < 1.0 is installed.

    ArviZ 1.0 dropped InferenceData in favour of pure xarray/DataTree.
    All probpipe diagnostics target ArviZ >= 1.0.
    """
    try:
        import arviz as az
        from packaging.version import Version

        if Version(az.__version__) < Version("1.0"):
            import warnings
            warnings.warn(
                f"ArviZ {az.__version__} detected. "
                "probpipe diagnostics require ArviZ >= 1.0. "
                "Please upgrade: pip install 'arviz>=1.0'",
                UserWarning,
                stacklevel=2,
            )
    except ImportError:
        import warnings
        warnings.warn(
            "ArviZ is not installed. "
            "Install it with: pip install 'arviz>=1.0'",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# xarray conversion
# ---------------------------------------------------------------------------

def to_xarray_dataset(
    samples,
    var_name: str = "x",
    *,
    distribution=None,
) -> xr.Dataset:
    """Convert samples to an xarray Dataset compatible with ArviZ 1.0.

    Tries to infer variable names from the distribution's
    ``parameter_names`` when available (e.g. from
    :class:`~probpipe.modeling._simple.SimpleModel`).

    Parameters
    ----------
    samples : array-like or dict
        - ``np.ndarray`` / JAX array: shape ``(n_draws,)`` or
          ``(n_chains, n_draws)`` or ``(n_draws, n_params)``
        - ``dict``: ``{var_name: array}`` — passed through directly.
        - :class:`~probpipe.core.distribution.ArrayEmpiricalDistribution`:
          ``.samples`` attribute is extracted automatically.
    var_name : str
        Fallback variable name when ``distribution`` has no
        ``parameter_names`` and ``samples`` is not a dict.
    distribution : Distribution or None
        If provided, ``parameter_names`` is read to label variables.

    Returns
    -------
    xr.Dataset
        Dimensions: ``chain`` × ``draw`` for each variable.
    """
    # -- Unwrap ArrayEmpiricalDistribution ----------------------------------
    if hasattr(samples, "samples"):          # ArrayEmpiricalDistribution
        samples = samples.samples

    # -- dict path: already named -------------------------------------------
    if isinstance(samples, dict):
        return xr.Dataset({
            k: xr.DataArray(
                np.asarray(v).reshape(1, -1),   # (chain=1, draw)
                dims=["chain", "draw"],
            )
            for k, v in samples.items()
        })

    # -- array path ---------------------------------------------------------
    arr = np.asarray(samples, dtype=np.float64)

    # Resolve variable names from distribution.parameter_names if available
    if distribution is not None and hasattr(distribution, "parameter_names"):
        param_names = distribution.parameter_names   # e.g. ("parameters",)
    else:
        param_names = None

    # arr shape: (n_draws,) → single variable
    if arr.ndim == 1:
        name = param_names[0] if param_names else var_name
        return xr.Dataset({
            name: xr.DataArray(
                arr.reshape(1, -1),
                dims=["chain", "draw"],
            )
        })

    # arr shape: (n_chains, n_draws) → single variable, multiple chains
    if arr.ndim == 2:
        name = param_names[0] if param_names else var_name
        return xr.Dataset({
            name: xr.DataArray(arr, dims=["chain", "draw"])
        })

    # arr shape: (n_draws, n_params) → multiple variables
    if arr.ndim == 3:
        # (n_chains, n_draws, n_params)
        n_params = arr.shape[-1]
        names = (
            list(param_names)[:n_params]
            if param_names and len(param_names) >= n_params
            else [f"{var_name}_{i}" for i in range(n_params)]
        )
        return xr.Dataset({
            name: xr.DataArray(arr[..., i], dims=["chain", "draw"])
            for i, name in enumerate(names)
        })

    raise ValueError(
        f"Cannot convert samples with shape {arr.shape} to xarray Dataset. "
        "Expected 1D (n_draws,), 2D (n_chains, n_draws), "
        "or 3D (n_chains, n_draws, n_params)."
    )


# ---------------------------------------------------------------------------
# Auto-detect generative likelihood
# ---------------------------------------------------------------------------

def _resolve_generative_likelihood(
    distribution: Any,
    generative_likelihood: Any = None,
) -> Any:
    """Auto-detect generative likelihood from a posterior distribution.

    Resolution order:

    1. Explicitly passed ``generative_likelihood`` argument.
    2. ``distribution["data"]`` — works for
       :class:`~probpipe.modeling._simple_generative.SimpleGenerativeModel`
       where ``__getitem__("data")`` returns a
       :class:`~probpipe.modeling._likelihood.GenerativeLikelihood`.
    3. ``distribution._likelihood`` — direct attribute fallback.
    4. ``distribution.generative_likelihood`` — future-proofing.
    5. Raise a descriptive :class:`ValueError`.

    Parameters
    ----------
    distribution : Distribution
        Posterior or prior distribution — typically a
        ``SimpleGenerativeModel`` or a conditioned posterior.
    generative_likelihood : optional
        Explicitly supplied likelihood; returned as-is if not ``None``.

    Returns
    -------
    GenerativeLikelihood
        Object with a ``generate_data(params, n_samples, *, key)`` method.

    Raises
    ------
    ValueError
        If no generative likelihood can be found.
    """
    # 1. Explicit argument
    if generative_likelihood is not None:
        return generative_likelihood

    # 2. distribution["data"] — SimpleGenerativeModel path
    try:
        candidate = distribution["data"]
        if hasattr(candidate, "generate_data"):
            return candidate
    except (KeyError, TypeError):
        pass

    # 3. distribution._likelihood — direct attribute
    candidate = getattr(distribution, "_likelihood", None)
    if candidate is not None and hasattr(candidate, "generate_data"):
        return candidate

    # 4. distribution.generative_likelihood — future-proofing
    candidate = getattr(distribution, "generative_likelihood", None)
    if candidate is not None and hasattr(candidate, "generate_data"):
        return candidate

    # 5. Nothing found
    raise ValueError(
        "Could not auto-detect a generative likelihood from the distribution. "
        "Either:\n"
        "  (a) pass `generative_likelihood` explicitly, or\n"
        "  (b) use a SimpleGenerativeModel whose 'data' component "
        "has a `generate_data()` method."
    )