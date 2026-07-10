"""Prior/likelihood power-scaling sensitivity diagnostics.

Design
------
``add_sensitivity`` mutates ``posterior._auxiliary`` in place and returns
``None``. It is intentionally a plain Python function, not a
``@workflow_function``, matching ``add_loo`` and ``add_mcmc_diagnostics`` —
diagnostics are post-hoc annotations on an already-fitted posterior.

The power-scaling numerics (Pareto-smoothed-importance-sampling reweighting
and the sensitivity diagnostic) are delegated entirely to ArviZ's
``psense``/``psense_summary`` (``arviz_stats.psense``), which implements
Kallioinen et al. 2024 directly. This module does not reimplement PSIS or
any divergence metric — it only computes the ``log_prior`` input ArviZ
needs (there is no generic way to recover a distribution's prior from its
posterior alone) and adapts the result into ProbPipe's diagnostics layout.

ArviZ-compatible data are stored under::

    _auxiliary/arviz/

ProbPipe diagnostic summaries are stored under::

    _auxiliary/diagnostics/runs/sensitivity

Main function
-------------
add_sensitivity
    Compute prior/likelihood power-scaling sensitivity using ArviZ and
    attach per-parameter results to ``posterior._auxiliary``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import xarray as xr

from ..core.distribution import Distribution
from ..core.ops import log_prob
from ..core.record import Record
from ._datatree_store import _add_group
from ._loo import _get_arviz_tree, _has_group, _log_likelihood_to_dataset
from ._utils import _component_name, _json_dumps_safe, _leaf_keys

__all__ = ["add_sensitivity"]


# ---------------------------------------------------------------------
# log_prior computation
# ---------------------------------------------------------------------


def _draw_log_prior(
    prior: Distribution,
    leaf_keys: list[str],
    draws_chain: Any,
    j: int,
) -> float:
    """Evaluate the prior log-density at one posterior draw."""
    value = Record({field: np.asarray(draws_chain[field])[j] for field in leaf_keys})
    return float(log_prob(prior, value))


def _compute_log_prior(posterior: Distribution, prior: Distribution) -> np.ndarray:
    """Evaluate the prior log-density at every posterior draw.

    Returns an array shaped ``(n_chain, n_draw)``. Evaluated one draw at a
    time: reconstructing a ``Record`` from a flat parameter vector is not
    JAX-traceable, the same limitation ``_loo.py:_add_log_likelihood``
    documents for the analogous per-observation case.
    """
    leaf_keys = _leaf_keys(posterior)
    n_chain = posterior.num_chains

    rows: list[list[float]] = []
    for i in range(n_chain):
        draws_i = posterior.draws(chain=i)
        n_draw = len(np.asarray(draws_i[leaf_keys[0]]))
        rows.append([_draw_log_prior(prior, leaf_keys, draws_i, j) for j in range(n_draw)])

    return np.asarray(rows, dtype=float)


# ---------------------------------------------------------------------
# ArviZ result adaptation
# ---------------------------------------------------------------------


def _dataset_to_param_dict(ds: xr.Dataset) -> dict[str, float]:
    """Flatten an ``az.psense`` scalar/event-shaped result to {label: value}."""
    values: dict[str, float] = {}
    for name in ds.data_vars:
        arr = np.asarray(ds[name], dtype=float)
        if arr.shape == ():
            values[str(name)] = float(arr.item())
            continue
        for index in np.ndindex(arr.shape):
            values[_component_name(str(name), index)] = float(arr[index])
    return values


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def add_sensitivity(
    posterior: Distribution,
    prior: Distribution,
    *,
    log_likelihood: Any | None = None,
    var_names: Sequence[str] | None = None,
    alphas: tuple[float, float] = (0.99, 1.01),
    threshold: float = 0.05,
    force: bool = False,
) -> None:
    """Compute power-scaling sensitivity and attach it to ``posterior._auxiliary``.

    This function mutates ``posterior._auxiliary`` in place and returns
    ``None``.

    Power-scales the prior (and, when available, the likelihood) by a small
    exponent and estimates how much the posterior would change via
    Pareto-smoothed importance-sampling reweighting — no refitting required.
    A high sensitivity value indicates the posterior is sensitive to that
    component; comparing prior- and likelihood-side sensitivity flags
    prior-data conflict and weakly-identified parameters. See Kallioinen et
    al., *Detecting and diagnosing prior and likelihood sensitivity with
    power-scaling*, Stat Comput 34, 57 (2024).

    Parameters
    ----------
    posterior : Distribution
        Fitted posterior whose ArviZ-compatible xarray DataTree data are
        stored under ``posterior._auxiliary["arviz"]``.

    prior : Distribution
        The prior the posterior was conditioned from. Must support
        ``log_prob`` (``SupportsLogProb``). Required — there is no generic
        way to recover a distribution's prior from the posterior alone.

    log_likelihood : optional
        Pointwise or per-draw total log likelihood, accepted in exactly the
        shapes ``add_loo`` accepts (``(chain, draw, obs)``, ``(draw, obs)``,
        or ``(chain, draw)`` as a scalar total). If provided, it is written
        to ``_auxiliary/arviz/log_likelihood`` and likelihood-side
        sensitivity is computed alongside prior-side sensitivity. If
        omitted, likelihood-side sensitivity is skipped:
        ``add_sensitivity`` does not read a value left by an earlier
        ``add_loo`` call — each call is self-contained on this input.

    var_names : sequence of str, optional
        Parameters to include. Defaults to all.

    alphas : tuple of float
        Lower/upper power-scaling exponents passed to
        ``arviz.psense``/``psense_summary``. Defaults to ArviZ's own
        ``(0.99, 1.01)``.

    threshold : float
        Sensitivity threshold for the diagnosis column. Defaults to
        ArviZ's own ``0.05`` (Kallioinen et al. 2024).

    force : bool
        If False and ``diagnostics/runs/sensitivity`` already exists, skip
        computation. If True, recompute and overwrite the existing
        sensitivity diagnostic node.

    Returns
    -------
    None

    Examples
    --------
    ::

        add_sensitivity(posterior, prior)
        posterior.diagnostics.sensitivity.diagnosis

        # With likelihood-side sensitivity too:
        add_sensitivity(posterior, prior, log_likelihood=log_lik)
    """
    import arviz as az

    if not force and _has_group(
        getattr(posterior, "_auxiliary", None), "diagnostics/runs/sensitivity"
    ):
        return None

    arviz_tree = _get_arviz_tree(posterior)
    if arviz_tree is None or not _has_group(arviz_tree, "posterior"):
        raise ValueError(
            "No ArviZ-compatible DataTree data found. add_sensitivity needs "
            "posterior draws recorded under posterior._auxiliary['arviz']; "
            "use an inference backend/model path that records them."
        )

    # ------------------------------------------------------------------
    # log_prior — always computed; the required input.
    # ------------------------------------------------------------------

    log_prior_arr = _compute_log_prior(posterior, prior)
    log_prior_ds = xr.Dataset({"log_prior": xr.DataArray(log_prior_arr, dims=["chain", "draw"])})
    _add_group(posterior, "arviz/log_prior", log_prior_ds)

    # ------------------------------------------------------------------
    # log_likelihood — optional, explicit-pass-only (no fallback lookup).
    # ------------------------------------------------------------------

    has_likelihood = log_likelihood is not None
    if has_likelihood:
        log_lik_ds = _log_likelihood_to_dataset(log_likelihood)
        _add_group(posterior, "arviz/log_likelihood", log_lik_ds)

    arviz_tree = _get_arviz_tree(posterior)  # refresh after the writes above

    # ------------------------------------------------------------------
    # Delegate the numerics to ArviZ.
    # ------------------------------------------------------------------

    psense_kwargs: dict[str, Any] = {"alphas": alphas}
    if var_names is not None:
        psense_kwargs["var_names"] = list(var_names)

    if has_likelihood:
        df = az.psense_summary(arviz_tree, threshold=threshold, **psense_kwargs)
        params = [str(p) for p in df.index]
        prior_values = dict(zip(params, df["prior"].astype(float), strict=True))
        lik_values = dict(zip(params, df["likelihood"].astype(float), strict=True))
        diagnosis = dict(zip(params, df["diagnosis"].astype(str), strict=True))
    else:
        prior_ds = az.psense(arviz_tree, group="prior", **psense_kwargs)
        prior_values = _dataset_to_param_dict(prior_ds)
        params = list(prior_values)
        lik_values = {}
        diagnosis = {
            p: ("prior sensitive" if prior_values[p] >= threshold else "✓") for p in params
        }

    # ------------------------------------------------------------------
    # Build diagnostics dataset.
    # ------------------------------------------------------------------

    prior_da = xr.DataArray(
        [prior_values.get(p, float("nan")) for p in params],
        dims=["param"],
        coords={"param": params},
    )
    lik_da = xr.DataArray(
        [lik_values.get(p, float("nan")) for p in params],
        dims=["param"],
        coords={"param": params},
    )

    run_ds = xr.Dataset({"prior_sensitivity": prior_da, "likelihood_sensitivity": lik_da})
    run_ds.attrs = {
        "kind": "sensitivity",
        "timestamp": datetime.now(UTC).isoformat(),
        "backend": "arviz",
        "has_likelihood": bool(has_likelihood),
        "threshold": float(threshold),
        "alphas": _json_dumps_safe(list(alphas)),
        "diagnosis_json": _json_dumps_safe(diagnosis),
    }

    _add_group(posterior, "diagnostics/runs/sensitivity", run_ds)

    return None
