"""DataTree storage/write helpers for ProbPipe diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._view_base import NotComputed

if TYPE_CHECKING:
    import xarray as xr

    from ..distributions._empirical import EmpiricalDistribution


__all__ = [
    "_add_group",
    "_get_or_create_mcmc_ds",
    "_mcmc_has_field",
    "_write_mcmc_field",
    "to_named_posterior_dataset",
]


def _flatten_datatree(tree: xr.DataTree) -> dict[str, Any]:
    """Flatten a DataTree into a path -> Dataset dictionary."""
    out: dict[str, Any] = {}

    def _walk(node: xr.DataTree, prefix: str = "") -> None:
        try:
            ds = node.to_dataset()
        except Exception as exc:
            path = prefix or "/"
            raise RuntimeError(
                f"Could not export existing diagnostics DataTree node at {path!r}."
            ) from exc

        if len(ds.data_vars) > 0 or len(ds.coords) > 0 or len(ds.attrs) > 0:
            out[prefix or "/"] = ds

        children = getattr(node, "children", {}) or {}
        for child_name in children:
            child = node[child_name]
            child_path = f"{prefix}/{child_name}" if prefix else child_name
            _walk(child, child_path)

    _walk(tree)
    return out


def _add_group(
    posterior: EmpiricalDistribution,
    group_name: str,
    dataset: xr.Dataset,
) -> None:
    """Add or replace a group in ``posterior._annotations`` in place.

    Preserves the existing groups and the root's attributes, such as the
    ``method`` an inference result records, by flattening the tree to a
    path -> Dataset dictionary, replacing or inserting the target group, and
    rebuilding the DataTree.
    """
    import xarray as xr

    aux = getattr(posterior, "_annotations", None)

    dicto: dict[str, Any] = {}

    if aux is not None:
        dicto.update(_flatten_datatree(aux))

    dicto[group_name.lstrip("/")] = dataset

    object.__setattr__(
        posterior,
        "_annotations",
        xr.DataTree.from_dict(dicto),
    )


def _get_or_create_mcmc_ds(posterior: EmpiricalDistribution) -> xr.Dataset:
    """Return existing ``/diagnostics/mcmc`` dataset or an empty one."""
    import xarray as xr

    aux = getattr(posterior, "_annotations", None)

    if aux is not None:
        try:
            return aux["diagnostics"]["mcmc"].to_dataset().copy()
        except Exception:
            pass

    return xr.Dataset()


def _write_mcmc_field(
    posterior: EmpiricalDistribution,
    field_name: str,
    values: dict[str, float | NotComputed],
    *,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write one per-parameter metric into ``/diagnostics/mcmc``."""
    import xarray as xr

    params = list(values.keys())
    numeric: list[float] = []
    da_attrs: dict[str, str] = {}

    for p in params:
        v = values[p]

        if isinstance(v, NotComputed):
            numeric.append(float("nan"))
            da_attrs[f"not_computed_{p}"] = v.reason
        else:
            numeric.append(float(v))

    da = xr.DataArray(numeric, dims=["param"], coords={"param": params})
    da.attrs.update(da_attrs)

    ds = _get_or_create_mcmc_ds(posterior)
    ds[field_name] = da

    if attrs:
        ds.attrs.update(attrs)

    _add_group(posterior, "diagnostics/mcmc", ds)


def _mcmc_has_field(
    posterior: EmpiricalDistribution,
    field_name: str,
) -> bool:
    """Return True if ``field_name`` exists in ``/diagnostics/mcmc``."""
    aux = getattr(posterior, "_annotations", None)

    if aux is None:
        return False

    try:
        ds = aux["diagnostics"]["mcmc"].to_dataset()
    except Exception:
        return False

    return field_name in ds.data_vars


def to_named_posterior_dataset(
    posterior: EmpiricalDistribution,
) -> xr.Dataset:
    """Build a Dataset with one variable per parameter.

    Scalar parameters have dims ``(chain, draw)``. Vector or array-valued
    parameters preserve their event axes after ``draw``.

    Parameters
    ----------
    posterior : EmpiricalDistribution
        The inference result, whose atoms lie on the levels ``chain`` and
        ``draw``.

    Returns
    -------
    xr.Dataset
        The dataset, in which the event axes of a variable ``x`` are named
        ``x_dim_0``, ``x_dim_1``, and so on.

    Notes
    -----
    A *nested* posterior contributes one variable per leaf, named by the
    leaf's full ``/``-path (flat posteriors keep their plain field names).
    ``InferenceData.to_netcdf()`` rejects ``/`` in variable names, so rename
    path-named variables before persisting a nested posterior to netCDF.

    Raises
    ------
    ValueError
        If the posterior's atoms do not lie on the levels ``chain`` and ``draw``.
    """
    import xarray as xr

    from ..inference._approximate_distribution import _chain_columns

    data_vars: dict[str, xr.DataArray] = {}

    # One variable per leaf of the draws, keyed by its full /-path.
    for field, column in _chain_columns(posterior).items():
        stacked = np.asarray(column)
        event_dims = [f"{field}_dim_{i}" for i in range(max(stacked.ndim - 2, 0))]
        data_vars[field] = xr.DataArray(
            stacked,
            dims=["chain", "draw", *event_dims],
        )

    return xr.Dataset(data_vars)
