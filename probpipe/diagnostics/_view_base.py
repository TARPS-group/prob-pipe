"""Generic diagnostic view base classes.

This module contains generic, domain-independent helpers for reading
diagnostic information from an xarray DataTree.

It should not know about MCMC, PPC, LOO, ArviZ, or ProbPipe-specific
diagnostic names. Those concrete views live in ``_views.py``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None


__all__ = [
    "DataTreeView",
    "DatasetView",
    "DiagnosticRunView",
    "NotComputed",
    "read_indexed",
    "read_json_attr",
    "read_scalar",
]


# ---------------------------------------------------------------------------
# NotComputed sentinel
# ---------------------------------------------------------------------------


class NotComputed:
    """Sentinel for a diagnostic value that is unavailable or not computed.

    Parameters
    ----------
    reason : str
        Human-readable explanation.
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __repr__(self) -> str:
        return f"NotComputed({self.reason!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NotComputed) and other.reason == self.reason


# ---------------------------------------------------------------------------
# Generic scalar / indexed readers
# ---------------------------------------------------------------------------


def read_scalar(da: Any | None, *, label: str | None = None) -> float | NotComputed:
    """Extract one scalar value from an xarray DataArray-like object.

    Returns ``NotComputed`` instead of raising when the value is missing,
    non-scalar, NaN, or marked with a ``not_computed`` attr.
    """
    if da is None:
        return NotComputed("not available")

    nc_key = f"not_computed_{label}" if label else "not_computed"

    attrs = getattr(da, "attrs", {}) or {}
    if nc_key in attrs or "not_computed" in attrs:
        reason = attrs.get(nc_key) or attrs.get("not_computed", "unknown")
        return NotComputed(reason)

    try:
        arr = np.asarray(da, dtype=float).squeeze()
    except Exception:
        return NotComputed("could not convert value to float")

    if arr.size != 1:
        return NotComputed(f"expected scalar, got shape {arr.shape}")

    val = float(arr.item())

    if np.isnan(val):
        return NotComputed("value is NaN")

    return val


def read_indexed(
    ds: Any | None,
    field: str,
    *,
    dim: str,
) -> dict[str, float | NotComputed]:
    """Read a one-dimensional indexed variable from a dataset.

    Example
    -------
    For a variable like::

        p_value(test_fn=["var_mean_ratio", "zero_fraction"])

    this returns::

        {"var_mean_ratio": 0.43, "zero_fraction": 0.81}
    """
    if ds is None:
        return {}

    if field not in ds.data_vars:
        return {}

    da = ds[field]

    if dim not in da.dims:
        return {}

    coords = [str(x) for x in da.coords[dim].values]

    return {coord: read_scalar(da.sel({dim: coord}), label=coord) for coord in coords}


def read_json_attr(attrs: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read a JSON-encoded attribute safely."""
    if default is None:
        default = []

    raw = attrs.get(key)

    if raw is None:
        return default

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Generic DataTree-backed views
# ---------------------------------------------------------------------------


class DataTreeView:
    """Generic read-only view over an xarray DataTree node.

    This base class intentionally contains only generic DataTree navigation
    helpers. Concrete diagnostic semantics belong in subclasses.
    """

    __slots__ = ("_tree",)

    def __init__(self, tree: Any | None) -> None:
        self._tree = tree

    @property
    def exists(self) -> bool:
        """Whether this view points to an existing tree node."""
        return self._tree is not None

    @property
    def attrs(self) -> dict[str, Any]:
        """Node attributes, or an empty dict."""
        if self._tree is None:
            return {}
        return getattr(self._tree, "attrs", {}) or {}

    @property
    def children(self) -> dict[str, Any]:
        """Child nodes, or an empty dict."""
        if self._tree is None:
            return {}
        return getattr(self._tree, "children", {}) or {}

    def has_child(self, name: str) -> bool:
        """Return True if the node has a child with this name."""
        return name in self.children

    def child(self, name: str) -> Any | None:
        """Return child DataTree node or None."""
        if not self.has_child(name):
            return None
        try:
            return self._tree[name]
        except Exception:
            return None

    def dataset(self) -> Any | None:
        """Return this node's dataset, if available."""
        if self._tree is None:
            return None

        try:
            return self._tree.to_dataset()
        except Exception:
            pass

        for attr in ("ds", "dataset"):
            ds = getattr(self._tree, attr, None)
            if ds is not None:
                return ds

        return None

    def attr(self, key: str, default: Any = None) -> Any:
        """Read an attribute from the node."""
        return self.attrs.get(key, default)


class DatasetView(DataTreeView):
    """Generic view over a DataTree node that is expected to hold a Dataset."""

    def scalar(self, field: str) -> float | NotComputed:
        """Read a scalar variable from this node's dataset."""
        ds = self.dataset()
        if ds is None:
            return NotComputed("dataset not available")
        if field not in ds.data_vars:
            return NotComputed(f"{field!r} not recorded")
        return read_scalar(ds[field])

    def indexed(
        self,
        field: str,
        *,
        dim: str,
    ) -> dict[str, float | NotComputed]:
        """Read a one-dimensional indexed variable."""
        return read_indexed(self.dataset(), field, dim=dim)


class DiagnosticRunView(DatasetView):
    """Generic accessor for one ``/diagnostics/runs/<name>/`` node.

    This class is generic. It does not know whether the run is PPC, LOO, etc.
    """

    __slots__ = ("name",)

    def __init__(self, name: str, tree: Any | None) -> None:
        super().__init__(tree)
        self.name = name

    @property
    def result(self) -> dict[str, Any]:
        """Dataset variables as a plain Python dict.

        Scalar variables become scalar values. One-dimensional variables become
        dictionaries keyed by their coordinate values.
        """
        ds = self.dataset()
        if ds is None:
            return {}

        out: dict[str, Any] = {}

        for var in ds.data_vars:
            da = ds[var]

            if da.dims == ():
                out[var] = read_scalar(da)
                continue

            if len(da.dims) == 1:
                dim = da.dims[0]
                coords = [str(x) for x in da.coords[dim].values]
                out[var] = {
                    coord: read_scalar(da.sel({dim: coord}), label=coord) for coord in coords
                }
                continue

            # Keep non-scalar/multi-dimensional values out of scalar result.
            out[var] = NotComputed(f"non-scalar result with dims {da.dims}")

        return out

    @property
    def timestamp(self) -> str:
        return str(self.attr("timestamp", ""))

    @property
    def plot_fn(self) -> str:
        return str(self.attr("plot_fn", ""))

    @property
    def plot_ready(self) -> bool:
        return bool(self.attr("plot_ready", False))

    @property
    def plot_groups(self) -> list[str]:
        return read_json_attr(self.attrs, "plot_groups", default=[])

    def __repr__(self) -> str:
        return (
            f"DiagnosticRunView(name={self.name!r}, "
            f"plot_fn={self.plot_fn!r}, "
            f"plot_ready={self.plot_ready})"
        )
