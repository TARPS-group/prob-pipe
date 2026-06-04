"""DataTree helpers and accessor classes for ProbPipe diagnostics.

Private module — do not import directly.
Public symbols are re-exported via ``probpipe.diagnostics.__init__``.

Provides:

- :func:`_add_group` — add a group to ``_auxiliary`` in place
- :func:`_get_or_create_mcmc_ds` — read or initialise the ``/diagnostics/mcmc/`` dataset
- :func:`_write_mcmc_field` — write one metric into ``/diagnostics/mcmc/``
- :func:`_mcmc_has_field` — check whether a metric is already present
- :func:`to_named_posterior_dataset` — build ``(chain, draw)`` xr.Dataset per param
- :class:`DiagnosticsView` — structured accessor over ``/diagnostics/``
- :class:`DiagnosticRunView` — accessor for one ``/diagnostics/runs/<name>/`` node
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr
    from ..core.distribution import Distribution
    from ..inference._approximate_distribution import ApproximateDistribution

__all__ = [
    "DiagnosticsView",
    "DiagnosticRunView",
    "to_named_posterior_dataset",
]


# ---------------------------------------------------------------------------
# NotComputed sentinel
# ---------------------------------------------------------------------------


class NotComputed:
    """Explicit sentinel for a diagnostic metric unavailable for this backend.

    Stored in the DataTree as ``NaN`` with a ``not_computed`` attribute
    on the ``DataArray``.  Returned by accessor properties when the
    attribute is detected.

    Parameters
    ----------
    reason : str
        Human-readable explanation, e.g. ``"no chains for VI posterior"``.
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __repr__(self) -> str:
        return f"NotComputed({self.reason!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NotComputed) and other.reason == self.reason


# ---------------------------------------------------------------------------
# Internal DataTree helpers
# ---------------------------------------------------------------------------


def _add_group(
    posterior: "ApproximateDistribution",
    group_name: str,
    dataset: "xr.Dataset",
) -> None:
    """Add or replace a group in ``posterior._auxiliary`` in place.

    Follows the ArviZ DataTree pattern (section 2.3.4 of ArviZ docs):
    convert to dict, insert/replace, convert back.

    Parameters
    ----------
    posterior : ApproximateDistribution
        The posterior whose ``_auxiliary`` is updated in place.
    group_name : str
        DataTree path relative to root, e.g. ``"diagnostics/mcmc"``
        or ``"arviz/log_likelihood"``.
    dataset : xr.Dataset
        Dataset to store at that path.
    """
    import xarray as xr

    aux = getattr(posterior, "_auxiliary", None)
    dicto: dict[str, Any] = {k: v for k, v in aux.items()} if aux is not None else {}
    dicto[group_name] = dataset
    object.__setattr__(posterior, "_auxiliary", xr.DataTree.from_dict(dicto))


def _get_or_create_mcmc_ds(posterior: "ApproximateDistribution") -> "xr.Dataset":
    """Return the existing ``/diagnostics/mcmc/`` dataset or an empty one.

    Parameters
    ----------
    posterior : ApproximateDistribution

    Returns
    -------
    xr.Dataset
        A mutable copy of the existing dataset, or a fresh empty Dataset.
    """
    import xarray as xr

    aux = getattr(posterior, "_auxiliary", None)
    if aux is not None and hasattr(aux, "children"):
        if "diagnostics" in aux.children:
            diag = aux["diagnostics"]
            if hasattr(diag, "children") and "mcmc" in diag.children:
                return diag["mcmc"].to_dataset().copy()
    return xr.Dataset()


def _write_mcmc_field(
    posterior: "ApproximateDistribution",
    field_name: str,
    values: dict[str, float | NotComputed],
    *,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write one per-parameter metric into ``/diagnostics/mcmc/`` in place.

    Existing fields in ``/diagnostics/mcmc/`` are preserved — only
    ``field_name`` is added or overwritten.

    ``NotComputed`` values are stored as ``NaN`` with a
    ``not_computed_{param}`` attribute on the DataArray so they survive
    the xarray round-trip.

    Parameters
    ----------
    posterior : ApproximateDistribution
    field_name : str
        e.g. ``"rhat"``, ``"ess_bulk"``, ``"mcse_mean"``.
    values : dict[str, float | NotComputed]
        Per-parameter metric values.
    attrs : dict or None
        Extra attributes to merge into the ``/diagnostics/mcmc/`` dataset.
    """
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
    posterior: "ApproximateDistribution",
    field_name: str,
) -> bool:
    """Return ``True`` if ``field_name`` already exists in ``/diagnostics/mcmc/``.

    Used by compute functions to skip recomputation unless ``force=True``.
    """
    aux = getattr(posterior, "_auxiliary", None)
    if aux is None or not hasattr(aux, "children"):
        return False
    if "diagnostics" not in aux.children:
        return False
    diag = aux["diagnostics"]
    if not hasattr(diag, "children") or "mcmc" not in diag.children:
        return False
    ds = diag["mcmc"].to_dataset()
    return field_name in ds.data_vars


def to_named_posterior_dataset(
    posterior: "ApproximateDistribution",
) -> "xr.Dataset":
    """Build an ``(chain, draw)`` xr.Dataset with one variable per parameter.

    Required by ``az.rhat``, ``az.ess``, and ``az.mcse``, which expect a
    dataset where each variable has dims ``(chain, draw)``.

    The ``inference_data`` DataTree stored by the sampler is intentionally
    not used here — it stores all parameters as a single flat ``params``
    vector without named fields, which ArviZ cannot reduce to per-parameter
    scalars.

    Parameters
    ----------
    posterior : ApproximateDistribution

    Returns
    -------
    xr.Dataset
        Shape ``(num_chains, num_draws)`` per field.
    """
    import xarray as xr

    data_vars: dict[str, xr.DataArray] = {}
    for field in posterior.fields:
        stacked = np.stack(
            [np.asarray(posterior.draws(chain=i)[field])
             for i in range(posterior.num_chains)],
            axis=0,
        )  # (chain, draw)
        data_vars[field] = xr.DataArray(stacked, dims=["chain", "draw"])
    return xr.Dataset(data_vars)


# ---------------------------------------------------------------------------
# Internal scalar reader
# ---------------------------------------------------------------------------


def _read_scalar(da: "xr.DataArray | None", param: str | None = None) -> float | NotComputed:
    """Extract a Python float from a DataArray, or a NotComputed sentinel.

    Parameters
    ----------
    da : xr.DataArray or None
    param : str or None
        When ``da`` is a slice already selected to a single param, pass
        the param name to check for its ``not_computed_{param}`` attr.
    """
    if da is None:
        return NotComputed("not available")
    nc_key = f"not_computed_{param}" if param else "not_computed"
    if nc_key in da.attrs or "not_computed" in da.attrs:
        reason = da.attrs.get(nc_key) or da.attrs.get("not_computed", "unknown")
        return NotComputed(reason)
    import numpy as _np
    val = _np.asarray(da, dtype=float).squeeze()
    if _np.isnan(val):
        return NotComputed("value is NaN")
    return float(val)


def _read_param_dict(
    ds: "xr.Dataset",
    field: str,
) -> dict[str, float | NotComputed]:
    """Read a per-parameter metric from a dataset into a plain dict.

    Parameters
    ----------
    ds : xr.Dataset
        The ``/diagnostics/mcmc/`` dataset.
    field : str
        Variable name, e.g. ``"rhat"``.

    Returns
    -------
    dict[str, float | NotComputed]
        ``{"intercept": 1.001, "slope": NotComputed(...), ...}``
    """
    if field not in ds.data_vars:
        return {}
    da = ds[field]
    params = list(da.coords["param"].values)
    return {p: _read_scalar(da.sel(param=p), param=p) for p in params}


# ---------------------------------------------------------------------------
# DiagnosticRunView
# ---------------------------------------------------------------------------


class DiagnosticRunView:
    """Read-only accessor for a single ``/diagnostics/runs/<name>/`` node.

    Parameters
    ----------
    name : str
        Diagnostic name, e.g. ``"ppc"`` or ``"loo"``.
    tree : xr.DataTree
        The DataTree node at ``/diagnostics/runs/<name>/``.
    """

    __slots__ = ("name", "_tree")

    def __init__(self, name: str, tree: "xr.DataTree") -> None:
        self.name  = name
        self._tree = tree

    @property
    def result(self) -> dict[str, Any]:
        """Scalar results as a plain dict.

        For a PPC run::

            {"var_mean_ratio": {"p_value": 0.43, "observed": 3.2}, ...}

        For a LOO run::

            {"elpd_loo": -132.1, "looic": 264.2, "se": 8.1}
        """
        ds = self._tree.to_dataset()
        out: dict[str, Any] = {}
        for var in ds.data_vars:
            da = ds[var]
            if "param" in da.dims or da.dims == ():
                # scalar — LOO-style
                out[var] = _read_scalar(da)
            else:
                # coordinate-indexed — PPC-style (test_fn dimension)
                dim = da.dims[0]
                coords = list(da.coords[dim].values)
                out[var] = {
                    c: _read_scalar(da.sel({dim: c}))
                    for c in coords
                }
        return out

    @property
    def timestamp(self) -> str:
        """ISO timestamp of when this run was executed."""
        return self._tree.attrs.get("timestamp", "")

    @property
    def plot_fn(self) -> str:
        """ArviZ function to call for visualisation, e.g. ``"az.plot_ppc"``."""
        return self._tree.attrs.get("plot_fn", "")

    @property
    def plot_ready(self) -> bool:
        """``True`` when all required ArviZ groups are present in ``_auxiliary``."""
        return bool(self._tree.attrs.get("plot_ready", False))

    @property
    def plot_groups(self) -> list[str]:
        """ArviZ groups needed to render the plot."""
        raw = self._tree.attrs.get("plot_groups", "[]")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []

    def __repr__(self) -> str:
        return (
            f"DiagnosticRunView(name={self.name!r}, "
            f"plot_fn={self.plot_fn!r}, "
            f"plot_ready={self.plot_ready})"
        )


# ---------------------------------------------------------------------------
# DiagnosticsView
# ---------------------------------------------------------------------------


class DiagnosticsView:
    """Structured read-only accessor over the ``/diagnostics/`` DataTree subtree.

    Returned by ``posterior.diagnostics``. Do not instantiate directly.

    Parameters
    ----------
    tree : xr.DataTree
        The ``/diagnostics/`` subtree of ``_auxiliary``.
    """

    __slots__ = ("_tree",)

    def __init__(self, tree: "xr.DataTree") -> None:
        self._tree = tree

    # ── per-parameter MCMC metrics ────────────────────────────────────────

    def _mcmc_ds(self) -> "xr.Dataset | None":
        """Return the ``/diagnostics/mcmc/`` dataset, or ``None``."""
        if not hasattr(self._tree, "children") or "mcmc" not in self._tree.children:
            return None
        return self._tree["mcmc"].to_dataset()

    @property
    def rhat(self) -> dict[str, float | NotComputed]:
        """R-hat per parameter. Empty dict if not yet computed."""
        ds = self._mcmc_ds()
        return _read_param_dict(ds, "rhat") if ds is not None else {}

    @property
    def ess_bulk(self) -> dict[str, float | NotComputed]:
        """Bulk ESS per parameter. Empty dict if not yet computed."""
        ds = self._mcmc_ds()
        return _read_param_dict(ds, "ess_bulk") if ds is not None else {}

    @property
    def ess_tail(self) -> dict[str, float | NotComputed]:
        """Tail ESS per parameter. Empty dict if not yet computed."""
        ds = self._mcmc_ds()
        return _read_param_dict(ds, "ess_tail") if ds is not None else {}

    @property
    def mcse_mean(self) -> dict[str, float | NotComputed]:
        """MCSE of the mean per parameter. Empty dict if not yet computed."""
        ds = self._mcmc_ds()
        return _read_param_dict(ds, "mcse_mean") if ds is not None else {}

    @property
    def mcse_sd(self) -> dict[str, float | NotComputed]:
        """MCSE of the sd per parameter. Empty dict if not yet computed."""
        ds = self._mcmc_ds()
        return _read_param_dict(ds, "mcse_sd") if ds is not None else {}

    # ── model-level scalars ───────────────────────────────────────────────

    @property
    def n_divergences(self) -> int | NotComputed:
        """Total number of divergent transitions, or ``NotComputed``."""
        ds = self._mcmc_ds()
        if ds is None:
            return NotComputed("mcmc diagnostics not yet computed")
        raw = ds.attrs.get("n_divergences")
        if raw is None:
            return NotComputed("not recorded by this backend")
        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, dict) and "not_computed" in parsed:
                return NotComputed(parsed["not_computed"])
            return int(parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            return int(raw)

    # ── warnings ──────────────────────────────────────────────────────────

    @property
    def warnings(self) -> list[str]:
        """All diagnostic warnings accumulated across metrics."""
        ds = self._mcmc_ds()
        if ds is None:
            return []
        msgs: list[str] = []
        for key in ("rhat_warnings", "ess_warnings", "mcse_warnings"):
            raw = ds.attrs.get(key, "[]")
            try:
                msgs.extend(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                pass
        return msgs

    # ── on-demand runs ────────────────────────────────────────────────────

    @property
    def runs(self) -> list[DiagnosticRunView]:
        """On-demand diagnostic runs in the order they were executed.

        Each entry is a :class:`DiagnosticRunView` with ``name``,
        ``result``, ``plot_fn``, and ``plot_ready``.
        """
        if not hasattr(self._tree, "children") or "runs" not in self._tree.children:
            return []
        runs_node = self._tree["runs"]
        children = runs_node.children if hasattr(runs_node, "children") else {}
        return [DiagnosticRunView(name, runs_node[name]) for name in children]

    # ── summary ───────────────────────────────────────────────────────────

    def summary_table(self) -> str:
        """Pretty-printed table of MCMC metrics for notebook display."""
        rhat    = self.rhat
        ess_b   = self.ess_bulk
        ess_t   = self.ess_tail
        mcse_m  = self.mcse_mean

        params = list(rhat.keys()) or list(ess_b.keys())
        if not params:
            return "No MCMC diagnostics computed yet."

        def _fmt(d: dict, k: str, width: int) -> str:
            v = d.get(k, "—")
            if isinstance(v, NotComputed):
                s = "N/A"
            elif isinstance(v, float):
                s = f"{v:.4f}"
            else:
                s = str(v)
            return s.ljust(width)

        col_w = [14, 10, 12, 12, 12]
        header = (
            "Parameter".ljust(col_w[0])
            + "R-hat".ljust(col_w[1])
            + "ESS bulk".ljust(col_w[2])
            + "ESS tail".ljust(col_w[3])
            + "MCSE mean".ljust(col_w[4])
        )
        sep = "-" * len(header)
        rows = [header, sep]
        for p in params:
            rows.append(
                p.ljust(col_w[0])
                + _fmt(rhat,   p, col_w[1])
                + _fmt(ess_b,  p, col_w[2])
                + _fmt(ess_t,  p, col_w[3])
                + _fmt(mcse_m, p, col_w[4])
            )
        rows.append("")

        warns = self.warnings
        if warns:
            rows.extend(f"⚠  {w}" for w in warns)
        else:
            rows.append("✓  All MCMC diagnostics passed.")

        return "\n".join(rows)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary of all diagnostics.

        ``NotComputed`` values become ``{"not_computed": reason}``.
        """
        def _ser(v: Any) -> Any:
            if isinstance(v, NotComputed):
                return {"not_computed": v.reason}
            if isinstance(v, dict):
                return {k: _ser(vv) for k, vv in v.items()}
            return v

        return {
            "rhat":      _ser(self.rhat),
            "ess_bulk":  _ser(self.ess_bulk),
            "ess_tail":  _ser(self.ess_tail),
            "mcse_mean": _ser(self.mcse_mean),
            "mcse_sd":   _ser(self.mcse_sd),
            "n_divergences": _ser(self.n_divergences),
            "warnings":  self.warnings,
            "runs": [
                {
                    "name":       r.name,
                    "result":     _ser(r.result),
                    "plot_fn":    r.plot_fn,
                    "plot_ready": r.plot_ready,
                    "plot_groups": r.plot_groups,
                    "timestamp":  r.timestamp,
                }
                for r in self.runs
            ],
        }

    def __repr__(self) -> str:
        n_runs = len(self.runs)
        warns  = len(self.warnings)
        params = list(self.rhat.keys())
        return (
            f"DiagnosticsView("
            f"params={params}, "
            f"warnings={warns}, "
            f"runs={n_runs})"
        )
