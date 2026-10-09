"""Internal ArviZ/xarray conversion layer.

This module is private. Public diagnostic functions live in
``probpipe.diagnostics`` and write ProbPipe-computed results under
``posterior._annotations["diagnostics"]``.

The bridge owns conversion into ArviZ-compatible datasets and raw diagnostic
inputs stored under ``posterior._annotations["arviz"]``. Users should normally
interact with ``add_mcmc_diagnostics``, ``add_ppc``, ``add_loo``, and
``posterior.diagnostics`` instead of importing this module directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Absolute (not relative) so this file stays loadable standalone — the
# missing-xarray fallback test execs it outside the package.
from probpipe._messages import unknown_names
from probpipe.core._shapes import NamesLike, _as_names
from probpipe.distributions._empirical import EmpiricalDistribution

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

    The posterior is an ``EmpiricalDistribution``, such as the result of
    ``condition_on``, whose atoms along one axis give one variable per leaf
    path, or one under the component of an array event.

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
        If the posterior is not an ``EmpiricalDistribution``.
    """
    if isinstance(posterior, EmpiricalDistribution):
        rows = posterior._rows
        if isinstance(rows, dict):
            return {path: np.asarray(column) for path, column in rows.items()}
        (component,) = posterior.event_spec.components
        return {component: np.asarray(rows)}

    raise TypeError(
        f"cannot extract draws: the posterior must be an EmpiricalDistribution; "
        f"got {type(posterior).__name__}"
    )


# ── Format conversion ─────────────────────────────────────────────────────────


def to_arviz_dataset(
    posterior: Any,
    *,
    var_names: NamesLike | None = None,
) -> xr.Dataset:
    """Convert a posterior distribution to an xarray.Dataset for ArviZ 1.0.

    For an inference result, whose atoms lie on the levels ``chain`` and
    ``draw``, it delegates to ``_datatree_store.to_named_posterior_dataset``,
    which builds variables with dims ``(chain, draw, *event_shape)``. Any other
    ``EmpiricalDistribution`` takes its atoms as one chain.

    Parameters
    ----------
    posterior : Distribution
        Posterior from ``condition_on`` or ``EmpiricalDistribution``.
    var_names : str, sequence of str, or None
        The variables to include, in the order given. A str names one
        variable, and ``None`` includes all.

    Returns
    -------
    xr.Dataset
        Dataset with dims ``(chain, draw, *event_shape)``.

    Raises
    ------
    ImportError
        If xarray is not installed.
    TypeError
        If *var_names* is not a str, a sequence of str, or ``None``.
    ValueError
        If *var_names* names a variable the posterior does not have.
    """
    if xr is None:
        raise ImportError("xarray is required. Install with: pip install xarray")
    names = None if var_names is None else _as_names(var_names, what="to_arviz_dataset var_names")

    from probpipe.inference._approximate_distribution import _has_chains

    # ── An inference result: delegate to the canonical builder ────────────────
    if _has_chains(posterior):
        from ._datatree_store import to_named_posterior_dataset

        ds = to_named_posterior_dataset(posterior)
        if names is not None:
            _check_variables(names, [str(name) for name in ds.data_vars])
            # A list, since xarray reads a tuple as one variable name.
            ds = ds[list(names)]
        return ds

    # ── Fallback: flat EmpiricalDistribution — no chain structure ─────────────
    draws = extract_draws(posterior)
    if names is not None:
        _check_variables(names, list(draws))
        draws = {name: draws[name] for name in names}

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


def _check_variables(names: tuple[str, ...], available: list[str]) -> None:
    """Raise ``ValueError`` if a name of *names* is not one of the *available* variables."""
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(unknown_names("variable", unknown, available))
