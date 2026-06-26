"""MCMC diagnostic functions for ProbPipe.

This module has two layers:

1. Private pure ops returning ``Record`` objects:

   - ``_compute_rhat_op``
   - ``_compute_ess_op``
   - ``_compute_mcse_op``

   These compute diagnostics and return structured ``Record`` results.
   They do not mutate ``posterior._auxiliary``. Not part of the public API.

2. In-place writer wrappers returning ``None``:

   - ``add_rhat``
   - ``add_ess``
   - ``add_mcse``
   - ``add_mcmc_diagnostics``

   These call the pure ops, write results into ``posterior._auxiliary``,
   emit warnings, and return ``None``.

The in-place wrappers write MCMC summaries under::

    _auxiliary/diagnostics/mcmc/

The pure ops are useful for workflows where diagnostics are pure operations
returning ProbPipe ``Record`` objects.
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.record import Record
from ._utils import _dataset_values

if TYPE_CHECKING:
    import xarray as xr

    from ..inference._approximate_distribution import ApproximateDistribution

__all__ = [
    "add_ess",
    "add_mcmc_diagnostics",
    "add_mcse",
    "add_rhat",
]


_RHAT_THRESHOLD: float = 1.01
_ESS_THRESHOLD: int = 400


# ---------------------------------------------------------------------------
# ArviZ statistics bridge
# ---------------------------------------------------------------------------


def arviz_rhat(arviz_tree: Any, *, method: str = "rank") -> Any:
    _check_arviz_stats()
    try:
        import arviz_stats as azs

        return azs.rhat(arviz_tree, method=method)
    except ImportError:
        import arviz as az

        return az.rhat(arviz_tree, method=method)


def arviz_ess(arviz_tree: Any, *, method: str = "bulk") -> Any:
    _check_arviz_stats()
    try:
        import arviz_stats as azs

        return azs.ess(arviz_tree, method=method)
    except ImportError:
        import arviz as az

        return az.ess(arviz_tree, method=method)


def arviz_mcse(arviz_tree: Any, *, method: str = "mean") -> Any:
    _check_arviz_stats()
    try:
        import arviz_stats as azs

        return azs.mcse(arviz_tree, method=method)
    except ImportError:
        import arviz as az

        return az.mcse(arviz_tree, method=method)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_arviz_stats() -> None:
    """Raise a helpful error if ArviZ statistics support is unavailable."""
    try:
        import arviz_stats  # noqa: F401
    except ImportError as exc:
        try:
            import arviz  # noqa: F401
        except ImportError:
            raise ImportError(
                "ArviZ is required for MCMC diagnostics; arviz-stats is "
                "preferred for ArviZ 1.x statistics. "
                "Install with: pip install arviz arviz-stats"
            ) from exc


def _check_arviz() -> None:
    """Backward-compatible dependency check for MCMC diagnostics."""
    _check_arviz_stats()


def _scalar_from_da(da: Any) -> float:
    """Robustly extract a Python float from an ArviZ diagnostic DataArray."""
    return float(np.asarray(da, dtype=float).squeeze().item())


def _values_from_dataset(ds: xr.Dataset) -> dict[str, float]:
    """Flatten scalar or event-shaped diagnostic outputs into named values."""
    return _dataset_values(ds)


def _rhat_warnings(values: dict[str, Any], threshold: float) -> list[str]:
    """Generate R-hat warning messages."""
    from ._datatree import NotComputed

    msgs: list[str] = []

    for param, val in values.items():
        if isinstance(val, NotComputed):
            continue

        try:
            numeric_val = float(val)
        except Exception:
            continue

        if numeric_val > threshold:
            msgs.append(
                f"R-hat > {threshold} for '{param}' ({numeric_val:.4f}) -- "
                f"chains may not have converged."
            )

    return msgs


def _ess_warnings(
    bulk: dict[str, Any],
    tail: dict[str, Any],
    threshold: int,
) -> list[str]:
    """Generate ESS warning messages."""
    from ._datatree import NotComputed

    msgs: list[str] = []

    for param, val in bulk.items():
        if isinstance(val, NotComputed):
            continue

        try:
            numeric_val = float(val)
        except Exception:
            continue

        if numeric_val < threshold:
            msgs.append(
                f"Low ESS (bulk) for '{param}' ({numeric_val:.0f}) -- consider more iterations."
            )

    for param, val in tail.items():
        if isinstance(val, NotComputed):
            continue

        try:
            numeric_val = float(val)
        except Exception:
            continue

        if numeric_val < threshold:
            msgs.append(
                f"Low ESS (tail) for '{param}' ({numeric_val:.0f}) -- "
                f"tail estimates may be unreliable."
            )

    return msgs


def _emit_record_warnings(record: Record) -> None:
    """Emit warnings stored in a diagnostic Record."""
    try:
        warnings_list = record["warnings"]
    except Exception:
        return

    if warnings_list is None:
        return

    for msg in warnings_list:
        warnings.warn(str(msg), stacklevel=3)


def _json_warnings(warnings_list: list[str]) -> str:
    """JSON-encode warnings for xarray attrs."""
    return json.dumps(list(warnings_list))


def _record_kind(record: Record) -> str:
    """Return the diagnostic kind from a Record."""
    try:
        return str(record["kind"])
    except Exception as exc:
        raise ValueError("Diagnostic Record is missing required field 'kind'.") from exc


# ---------------------------------------------------------------------------
# Private pure op: _compute_rhat_op
# ---------------------------------------------------------------------------


def _compute_rhat_op(
    posterior: ApproximateDistribution,
    *,
    method: str = "rank",
    threshold: float = _RHAT_THRESHOLD,
) -> Record:
    """Pure R-hat diagnostic operation.

    Computes R-hat and returns a ``Record``. Does not mutate
    ``posterior._auxiliary`` and does not emit warnings.

    Parameters
    ----------
    posterior : ApproximateDistribution
        Fitted posterior.
    method : str
        ArviZ R-hat variant: ``"rank"`` by default.
    threshold : float
        Warning threshold.

    Returns
    -------
    Record
        Record with fields ``kind``, ``values``, ``warnings``, and ``attrs``.
    """
    from ._datatree import NotComputed, to_named_posterior_dataset

    if getattr(posterior, "num_chains", 1) < 2:
        values = {
            field: NotComputed("R-hat requires at least 2 chains")
            for field in getattr(posterior, "fields", ())
        }
        warns: list[str] = []

        attrs = {
            "rhat_method": method,
            "rhat_threshold": threshold,
            "rhat_warnings": _json_warnings(warns),
        }

        return Record(
            name="rhat_diagnostic",
            kind="rhat",
            values=values,
            warnings=warns,
            attrs=attrs,
        )

    ds = to_named_posterior_dataset(posterior)
    rhat_ds = arviz_rhat(ds, method=method)

    values = _values_from_dataset(rhat_ds)

    warns = _rhat_warnings(values, threshold)

    attrs = {
        "rhat_method": method,
        "rhat_threshold": threshold,
        "rhat_warnings": _json_warnings(warns),
    }

    return Record(
        name="rhat_diagnostic",
        kind="rhat",
        values=values,
        warnings=warns,
        attrs=attrs,
    )


# ---------------------------------------------------------------------------
# Private pure op: _compute_ess_op
# ---------------------------------------------------------------------------


def _compute_ess_op(
    posterior: ApproximateDistribution,
    *,
    threshold: int = _ESS_THRESHOLD,
) -> Record:
    """Pure ESS diagnostic operation.

    Computes bulk ESS and tail ESS and returns a ``Record``. Does not mutate
    ``posterior._auxiliary`` and does not emit warnings.

    Parameters
    ----------
    posterior : ApproximateDistribution
        Fitted posterior.
    threshold : int
        Warning threshold.

    Returns
    -------
    Record
        Record with fields ``kind``, ``bulk``, ``tail``, ``warnings``,
        ``bulk_attrs``, and ``tail_attrs``.
    """
    from ._datatree import to_named_posterior_dataset

    ds = to_named_posterior_dataset(posterior)

    bulk_ds = arviz_ess(ds, method="bulk")
    tail_ds = arviz_ess(ds, method="tail")

    bulk = _values_from_dataset(bulk_ds)
    tail = _values_from_dataset(tail_ds)

    warns = _ess_warnings(bulk, tail, threshold)

    bulk_attrs = {
        "ess_method": "bulk",
        "ess_threshold": threshold,
    }

    tail_attrs = {
        "ess_method": "tail",
        "ess_threshold": threshold,
        "ess_warnings": _json_warnings(warns),
    }

    return Record(
        name="ess_diagnostic",
        kind="ess",
        bulk=bulk,
        tail=tail,
        warnings=warns,
        bulk_attrs=bulk_attrs,
        tail_attrs=tail_attrs,
    )


# ---------------------------------------------------------------------------
# Private pure op: _compute_mcse_op
# ---------------------------------------------------------------------------


def _compute_mcse_op(
    posterior: ApproximateDistribution,
) -> Record:
    """Pure MCSE diagnostic operation.

    Computes MCSE of the mean and MCSE of the standard deviation and returns
    a ``Record``. Does not mutate ``posterior._auxiliary``.

    Parameters
    ----------
    posterior : ApproximateDistribution
        Fitted posterior.

    Returns
    -------
    Record
        Record with fields ``kind``, ``mean``, ``sd``, ``warnings``,
        ``mean_attrs``, and ``sd_attrs``.
    """
    from ._datatree import to_named_posterior_dataset

    ds = to_named_posterior_dataset(posterior)

    mean_ds = arviz_mcse(ds, method="mean")
    sd_ds = arviz_mcse(ds, method="sd")

    mean = _values_from_dataset(mean_ds)
    sd = _values_from_dataset(sd_ds)

    mean_attrs = {
        "mcse_method": "mean",
    }

    sd_attrs = {
        "mcse_method": "sd",
    }

    return Record(
        name="mcse_diagnostic",
        kind="mcse",
        mean=mean,
        sd=sd,
        warnings=[],
        mean_attrs=mean_attrs,
        sd_attrs=sd_attrs,
    )


# ---------------------------------------------------------------------------
# Writer helpers
# ---------------------------------------------------------------------------


def _write_mcmc_record(
    posterior: ApproximateDistribution,
    record: Record,
) -> None:
    """Write an MCMC diagnostic Record into ``posterior._auxiliary``."""
    from ._datatree import _write_mcmc_field

    kind = _record_kind(record)

    if kind == "rhat":
        _write_mcmc_field(
            posterior,
            "rhat",
            record["values"],
            attrs=record["attrs"],
        )
        return None

    if kind == "ess":
        _write_mcmc_field(
            posterior,
            "ess_bulk",
            record["bulk"],
            attrs=record["bulk_attrs"],
        )
        _write_mcmc_field(
            posterior,
            "ess_tail",
            record["tail"],
            attrs=record["tail_attrs"],
        )
        return None

    if kind == "mcse":
        _write_mcmc_field(
            posterior,
            "mcse_mean",
            record["mean"],
            attrs=record["mean_attrs"],
        )
        _write_mcmc_field(
            posterior,
            "mcse_sd",
            record["sd"],
            attrs=record["sd_attrs"],
        )
        return None

    if kind == "mcmc":
        records = record["records"]
        for child in records.values():
            _write_mcmc_record(posterior, child)
        return None

    raise ValueError(f"Unknown MCMC diagnostic Record kind: {kind!r}")


# ---------------------------------------------------------------------------
# In-place wrapper: add_rhat
# ---------------------------------------------------------------------------


def add_rhat(
    posterior: ApproximateDistribution,
    *,
    method: str = "rank",
    threshold: float = _RHAT_THRESHOLD,
    force: bool = False,
) -> None:
    """Compute R-hat and attach to ``_auxiliary/diagnostics/mcmc/``.

    This is the in-place wrapper around :func:`_compute_rhat_op`.

    Skips computation if R-hat is already present unless ``force=True``.
    Emits a Python warning for any parameter with R-hat > ``threshold``.
    """
    from ._datatree import _mcmc_has_field

    if not force and _mcmc_has_field(posterior, "rhat"):
        return None

    record = _compute_rhat_op(
        posterior,
        method=method,
        threshold=threshold,
    )

    _emit_record_warnings(record)
    _write_mcmc_record(posterior, record)

    return None


# ---------------------------------------------------------------------------
# In-place wrapper: add_ess
# ---------------------------------------------------------------------------


def add_ess(
    posterior: ApproximateDistribution,
    *,
    threshold: int = _ESS_THRESHOLD,
    force: bool = False,
) -> None:
    """Compute bulk and tail ESS and attach to ``_auxiliary/diagnostics/mcmc/``.

    This is the in-place wrapper around :func:`_compute_ess_op`.

    Skips computation if ESS is already present unless ``force=True``.
    Emits Python warnings for parameters below ``threshold``.
    """
    from ._datatree import _mcmc_has_field

    if not force and _mcmc_has_field(posterior, "ess_bulk"):
        return None

    record = _compute_ess_op(
        posterior,
        threshold=threshold,
    )

    _emit_record_warnings(record)
    _write_mcmc_record(posterior, record)

    return None


# ---------------------------------------------------------------------------
# In-place wrapper: add_mcse
# ---------------------------------------------------------------------------


def add_mcse(
    posterior: ApproximateDistribution,
    *,
    force: bool = False,
) -> None:
    """Compute MCSE and attach to ``_auxiliary/diagnostics/mcmc/``.

    This is the in-place wrapper around :func:`_compute_mcse_op`.

    Skips computation if MCSE is already present unless ``force=True``.
    """
    from ._datatree import _mcmc_has_field

    if not force and _mcmc_has_field(posterior, "mcse_mean"):
        return None

    record = _compute_mcse_op(posterior)

    _emit_record_warnings(record)
    _write_mcmc_record(posterior, record)

    return None


# ---------------------------------------------------------------------------
# In-place wrapper: add_mcmc_diagnostics
# ---------------------------------------------------------------------------


def add_mcmc_diagnostics(
    posterior: ApproximateDistribution,
    *,
    metrics: list[str] | None = None,
    rhat_method: str = "rank",
    rhat_threshold: float = _RHAT_THRESHOLD,
    ess_threshold: int = _ESS_THRESHOLD,
    force: bool = False,
) -> None:
    """Compute MCMC diagnostics and attach to ``_auxiliary/diagnostics/mcmc/``.

    Computes R-hat, bulk ESS, tail ESS, and MCSE by default.
    Skips any metric already present unless ``force=True``.

    Parameters
    ----------
    posterior : ApproximateDistribution
        The fitted posterior. Mutated in place.
    metrics : list of str or None
        Subset to compute. ``None`` computes all: ``["rhat", "ess", "mcse"]``.
    rhat_method : str
        ArviZ R-hat variant: ``"rank"`` by default.
    rhat_threshold : float
        R-hat warning threshold.
    ess_threshold : int
        ESS warning threshold.
    force : bool
        Recompute even if metrics are already stored.
    """
    compute = set(metrics) if metrics is not None else {"rhat", "ess", "mcse"}

    if "rhat" in compute:
        add_rhat(
            posterior,
            method=rhat_method,
            threshold=rhat_threshold,
            force=force,
        )

    if "ess" in compute:
        add_ess(
            posterior,
            threshold=ess_threshold,
            force=force,
        )

    if "mcse" in compute:
        add_mcse(
            posterior,
            force=force,
        )

    return None
