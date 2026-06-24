"""DataTree storage/write helpers for ProbPipe diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._view_base import NotComputed

if TYPE_CHECKING:
    import xarray as xr

    from ..inference._approximate_distribution import ApproximateDistribution


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
            if prefix:
                out[prefix] = ds

        children = getattr(node, "children", {}) or {}
        for child_name in children:
            child = node[child_name]
            child_path = f"{prefix}/{child_name}" if prefix else child_name
            _walk(child, child_path)

    _walk(tree)
    return out


def _add_group(
    posterior: ApproximateDistribution,
    group_name: str,
    dataset: xr.Dataset,
) -> None:
    """Add or replace a group in ``posterior._auxiliary`` in place.

    Preserves existing nested DataTree groups by flattening the tree to a
    path -> Dataset dictionary, replacing/inserting the target group, and
    rebuilding the DataTree.
    """
    import xarray as xr

    aux = getattr(posterior, "_auxiliary", None)

    dicto: dict[str, Any] = {}

    if aux is not None:
        dicto.update(_flatten_datatree(aux))

    dicto[group_name.lstrip("/")] = dataset

    object.__setattr__(
        posterior,
        "_auxiliary",
        xr.DataTree.from_dict(dicto),
    )


def _get_or_create_mcmc_ds(posterior: ApproximateDistribution) -> xr.Dataset:
    """Return existing ``/diagnostics/mcmc`` dataset or an empty one."""
    import xarray as xr

    aux = getattr(posterior, "_auxiliary", None)

    if aux is not None:
        try:
            return aux["diagnostics"]["mcmc"].to_dataset().copy()
        except Exception:
            pass

    return xr.Dataset()


def _write_mcmc_field(
    posterior: ApproximateDistribution,
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
    posterior: ApproximateDistribution,
    field_name: str,
) -> bool:
    """Return True if ``field_name`` exists in ``/diagnostics/mcmc``."""
    aux = getattr(posterior, "_auxiliary", None)

    if aux is None:
        return False

    try:
        ds = aux["diagnostics"]["mcmc"].to_dataset()
    except Exception:
        return False

    return field_name in ds.data_vars


def to_named_posterior_dataset(
    posterior: ApproximateDistribution,
) -> xr.Dataset:
    """Build a Dataset with one variable per parameter.

    Scalar parameters have dims ``(chain, draw)``. Vector or array-valued
    parameters preserve their event axes after ``draw``.
    """
    import xarray as xr

    data_vars: dict[str, xr.DataArray] = {}

    for field in posterior.fields:
        stacked = np.stack(
            [np.asarray(posterior.draws(chain=i)[field]) for i in range(posterior.num_chains)],
            axis=0,
        )
        event_dims = [f"{field}_dim_{i}" for i in range(max(stacked.ndim - 2, 0))]
        data_vars[field] = xr.DataArray(
            stacked,
            dims=["chain", "draw", *event_dims],
        )

    return xr.Dataset(data_vars)
