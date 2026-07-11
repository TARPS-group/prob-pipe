"""LOO/PSIS diagnostics interface.

This module provides an in-place ProbPipe diagnostics wrapper around ArviZ's
LOO/PSIS functionality.

Design
------
Functions in this file mutate ``posterior._annotations`` in place and return
``None``. They are intentionally plain Python functions, not
``@workflow_function``s, because diagnostics are post-hoc annotations on an
already-fitted posterior.

ArviZ-compatible data are stored under::

    _annotations/arviz/

ProbPipe diagnostic summaries are stored under::

    _annotations/diagnostics/runs/loo

Main function
-------------
add_loo
    Compute PSIS-LOO using ArviZ and attach scalar summaries to
    ``posterior._annotations``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import arviz as az
import numpy as np
import xarray as xr

from ..core.distribution import Distribution
from ..core.record import Record, _auto_record
from ._datatree import _add_group
from ._utils import _json_dumps_safe, _leaf_keys, _record_get, _safe_float

__all__ = ["add_loo"]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------
# _record_get, _safe_float, _json_dumps_safe imported from ._utils.


def _safe_int(value: Any) -> int:
    """Convert value to int, returning -1 if conversion fails."""
    if value is None:
        return -1

    try:
        return int(value)
    except Exception:
        try:
            return int(np.asarray(value).ravel()[0])
        except Exception:
            return -1


def _as_numpy(obj: Any) -> np.ndarray | None:
    """Best-effort conversion to NumPy array."""
    if obj is None:
        return None

    if isinstance(obj, xr.DataArray):
        try:
            return np.asarray(obj.values)
        except Exception:
            return None

    if hasattr(obj, "values"):
        try:
            return np.asarray(obj.values)
        except Exception:
            pass

    if hasattr(obj, "samples"):
        try:
            return np.asarray(obj.samples)
        except Exception:
            pass

    try:
        return np.asarray(obj)
    except Exception:
        return None


# ---------------------------------------------------------------------
# ArviZ tree helpers
# ---------------------------------------------------------------------


def _get_arviz_tree(posterior: Distribution) -> Any:
    """Return the ArviZ-compatible subtree for a posterior.

    Preferred layout::

        posterior._annotations["arviz"]

    This function is defensive so that it also works during transition periods
    where older posteriors expose only ``posterior.inference_data``.
    """
    aux = getattr(posterior, "_annotations", None)

    if aux is not None:
        try:
            return aux["arviz"]
        except Exception:
            pass

    for attr in ("arviz_data", "inference_data"):
        try:
            arviz_data = getattr(posterior, attr)
        except Exception:
            continue

        if arviz_data is not None:
            # If the accessor accidentally returns the full annotations tree,
            # prefer its /arviz subtree when present.
            try:
                return arviz_data["arviz"]
            except Exception:
                return arviz_data

    return aux


def _has_group(tree: Any, path: str) -> bool:
    """Return True if a slash-separated path exists in a DataTree-like object."""
    if tree is None:
        return False

    node = tree
    for part in path.strip("/").split("/"):
        if not part:
            continue
        try:
            node = node[part]
        except Exception:
            return False

    return True


def _has_required_loo_pit_groups(arviz_tree: Any) -> bool:
    """Check whether groups needed for az.plot_loo_pit appear to exist.

    ArviZ LOO-PIT plotting generally needs observed data, posterior predictive
    samples, and log likelihood. Exact requirements may vary by ArviZ version.
    """
    return (
        _has_group(arviz_tree, "observed_data")
        and _has_group(arviz_tree, "posterior_predictive")
        and _has_group(arviz_tree, "log_likelihood")
    )


# ---------------------------------------------------------------------
# Log-likelihood conversion
# ---------------------------------------------------------------------


def _log_likelihood_to_dataset(
    log_likelihood: Any,
    *,
    var_name: str = "y",
) -> xr.Dataset:
    """Convert log likelihood input to an xarray Dataset.

    Common accepted shapes:

    - ``(draw, obs)``
    - ``(chain, draw, obs)``
    - ``(chain, draw)`` as scalar total log likelihood per draw

    For PSIS-LOO, pointwise log likelihood is preferred, usually with shape
    ``(chain, draw, obs)`` or ``(draw, obs)``.
    """
    if isinstance(log_likelihood, xr.Dataset):
        _raise_for_1d_log_likelihood(log_likelihood)
        return log_likelihood

    if isinstance(log_likelihood, xr.DataArray):
        _raise_for_1d_log_likelihood(log_likelihood)
        name = log_likelihood.name or var_name
        return xr.Dataset({name: log_likelihood})

    arr = np.asarray(log_likelihood)

    if arr.shape == ():
        # Scalar log likelihood is not ideal for LOO, but store defensively.
        arr = arr.reshape(1, 1, 1)
        dims = ["chain", "draw", "obs"]

    elif arr.ndim == 1:
        raise ValueError(
            "1-D log_likelihood looks like one total log likelihood per draw. "
            "add_loo needs pointwise log likelihood values with shape "
            "(draw, obs) or (chain, draw, obs)."
        )

    elif arr.ndim == 2:
        # Assume (draw, obs), add singleton chain dimension.
        arr = arr[np.newaxis, :, :]
        dims = ["chain", "draw", "obs"]

    elif arr.ndim == 3:
        # Assume (chain, draw, obs).
        dims = ["chain", "draw", "obs"]

    else:
        # Assume first two dimensions are chain/draw and the rest are obs dims.
        dims = ["chain", "draw"] + [f"obs_dim_{i}" for i in range(arr.ndim - 2)]

    return xr.Dataset({var_name: xr.DataArray(arr, dims=dims)})


def _raise_for_1d_log_likelihood(log_likelihood: xr.Dataset | xr.DataArray) -> None:
    """Reject non-pointwise 1-D log-likelihood inputs."""
    if isinstance(log_likelihood, xr.DataArray):
        arrays = [log_likelihood]
    else:
        arrays = list(log_likelihood.data_vars.values())

    if any(da.ndim == 1 for da in arrays):
        raise ValueError(
            "1-D log_likelihood looks like one total log likelihood per draw. "
            "add_loo needs pointwise log likelihood values with shape "
            "(draw, obs) or (chain, draw, obs)."
        )


# ---------------------------------------------------------------------
# LOO result extraction
# ---------------------------------------------------------------------


def _extract_pointwise_array(value: Any) -> np.ndarray | None:
    """Extract and flatten pointwise ArviZ arrays such as pareto_k or loo_i."""
    arr = _as_numpy(value)

    if arr is None:
        return None

    try:
        return np.asarray(arr, dtype=float).ravel()
    except Exception:
        return None


def _pareto_k_summary(pareto_k: Any, good_k: float | None = None) -> dict[str, float | int]:
    """Summarize Pareto-k values as scalar diagnostics."""
    arr = _extract_pointwise_array(pareto_k)

    if arr is None or arr.size == 0:
        return {
            "pareto_k_max": float("nan"),
            "pareto_k_mean": float("nan"),
            "pareto_k_bad_count": -1,
            "pareto_k_bad_fraction": float("nan"),
        }

    if good_k is None or np.isnan(good_k):
        threshold = 0.7
    else:
        threshold = float(good_k)

    bad = arr > threshold

    return {
        "pareto_k_max": float(np.nanmax(arr)),
        "pareto_k_mean": float(np.nanmean(arr)),
        "pareto_k_bad_count": int(np.nansum(bad)),
        "pareto_k_bad_fraction": float(np.nanmean(bad)),
    }


def _pointwise_dataarray(
    values: Any,
    *,
    name: str,
    dim: str = "obs",
) -> xr.DataArray | None:
    """Convert pointwise values to a 1D DataArray, if available."""
    arr = _extract_pointwise_array(values)

    if arr is None:
        return None

    return xr.DataArray(
        arr,
        dims=[dim],
        coords={dim: np.arange(arr.size)},
        name=name,
    )


# ---------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------


def add_loo(
    posterior: Distribution,
    *,
    log_likelihood: Any | None = None,
    model: Any | None = None,
    data: Any | None = None,
    var_name: str = "y",
    pointwise: bool = True,
    scale: str | None = None,
    reff: float | None = None,
    force: bool = False,
    store_pointwise: bool = True,
) -> None:
    """Compute PSIS-LOO and attach results to ``posterior._annotations``.

    This function mutates ``posterior._annotations`` in place and returns
    ``None``.

    Parameters
    ----------
    posterior : Distribution
        Posterior distribution whose ArviZ-compatible xarray DataTree data are
        stored under ``posterior._annotations["arviz"]``.

    log_likelihood : optional
        Pointwise log likelihood. If provided, it is written to
        ``_annotations/arviz/log_likelihood`` before calling ArviZ. This is an
        advanced override; the normal user-facing workflow is
        ``add_loo(posterior)``.

    model, data : optional
        Inputs used to compute pointwise log likelihoods when
        ``_annotations/arviz/log_likelihood`` is missing. This keeps LOO owned by
        ``add_loo`` instead of requiring a separate public precomputation step.

    var_name : str
        Variable name to use when converting raw log-likelihood arrays into an
        xarray Dataset.

    pointwise : bool
        Passed to ``az.loo``. If True, ArviZ returns pointwise quantities such
        as ``loo_i`` and ``pareto_k`` when available.

    scale : str or None
        Optional ArviZ scale argument, e.g. ``"log"``, ``"negative_log"``, or
        ``"deviance"`` depending on ArviZ version.

    reff : float or None
        Optional relative MCMC efficiency passed to ``az.loo``.

    force : bool
        If False and ``diagnostics/runs/loo`` already exists, skip computation.
        If True, recompute and overwrite the existing LOO diagnostic node.

    store_pointwise : bool
        If True, store pointwise ``pareto_k`` and ``loo_i`` values as 1D arrays
        when available. These are stored in the diagnostic run dataset.

    Returns
    -------
    None
    """
    if not force and _has_group(getattr(posterior, "_annotations", None), "diagnostics/runs/loo"):
        return None

    # ------------------------------------------------------------------
    # Add/update log_likelihood group under /arviz/
    # ------------------------------------------------------------------

    if log_likelihood is not None:
        log_lik_ds = _log_likelihood_to_dataset(log_likelihood, var_name=var_name)
        _add_group(posterior, "arviz/log_likelihood", log_lik_ds)

    arviz_tree = _get_arviz_tree(posterior)

    if arviz_tree is None:
        if model is not None and data is not None:
            _add_log_likelihood(posterior, model, data, var_name=var_name)
            arviz_tree = _get_arviz_tree(posterior)

        if arviz_tree is None:
            raise ValueError(
                "No ArviZ-compatible DataTree data or pointwise "
                "log_likelihood found. add_loo needs pointwise log "
                "likelihoods with shape (chain, draw, obs). Either use an "
                "inference backend/model path that records pointwise log "
                "likelihoods, pass log_likelihood=... as an advanced override, "
                "or pass model=... and data=... when the model exposes a "
                "supported pointwise log-likelihood method."
            )

    if not _has_group(arviz_tree, "log_likelihood"):
        if model is not None and data is not None:
            _add_log_likelihood(posterior, model, data, var_name=var_name)
            arviz_tree = _get_arviz_tree(posterior)

        if arviz_tree is None or not _has_group(arviz_tree, "log_likelihood"):
            raise ValueError(
                "No pointwise log_likelihood group found under "
                "posterior._annotations['arviz']. add_loo needs pointwise log "
                "likelihoods with shape (chain, draw, obs). Either use an "
                "inference backend/model path that records pointwise log "
                "likelihoods, pass log_likelihood=... as an advanced override, "
                "or pass model=... and data=... when the model exposes a "
                "supported pointwise log-likelihood method."
            )

    # ------------------------------------------------------------------
    # Run ArviZ LOO
    # ------------------------------------------------------------------

    loo_kwargs: dict[str, Any] = {
        "pointwise": pointwise,
    }

    if scale is not None:
        loo_kwargs["scale"] = scale

    if reff is not None:
        loo_kwargs["reff"] = reff

    loo_result = az.loo(arviz_tree, **loo_kwargs)

    # ------------------------------------------------------------------
    # Extract scalar summaries
    # ------------------------------------------------------------------

    elpd_loo = _safe_float(_record_get(loo_result, "elpd_loo", _record_get(loo_result, "elpd")))
    se = _safe_float(_record_get(loo_result, "se"))
    p_loo = _safe_float(_record_get(loo_result, "p_loo", _record_get(loo_result, "p")))

    # Some ArviZ versions expose looic; others only expose elpd_loo.
    looic_raw = _record_get(loo_result, "looic", None)
    looic = _safe_float(looic_raw)

    if np.isnan(looic) and not np.isnan(elpd_loo):
        looic = float(-2.0 * elpd_loo)

    n_samples = _safe_int(_record_get(loo_result, "n_samples"))
    n_data_points = _safe_int(_record_get(loo_result, "n_data_points"))

    warning_value = _record_get(loo_result, "warning", False)
    warning_numeric = 1.0 if bool(warning_value) else 0.0

    good_k = _safe_float(_record_get(loo_result, "good_k", np.nan))

    pareto_k = _record_get(loo_result, "pareto_k", None)
    loo_i = _record_get(loo_result, "loo_i", None)

    pk_summary = _pareto_k_summary(
        pareto_k,
        good_k=None if np.isnan(good_k) else good_k,
    )

    # ------------------------------------------------------------------
    # Build diagnostics dataset
    # ------------------------------------------------------------------

    data_vars: dict[str, xr.DataArray] = {
        "elpd_loo": xr.DataArray(elpd_loo),
        "se": xr.DataArray(se),
        "p_loo": xr.DataArray(p_loo),
        "looic": xr.DataArray(looic),
        "n_samples": xr.DataArray(float(n_samples)),
        "n_data_points": xr.DataArray(float(n_data_points)),
        "warning": xr.DataArray(warning_numeric),
        "good_k": xr.DataArray(good_k),
        "pareto_k_max": xr.DataArray(pk_summary["pareto_k_max"]),
        "pareto_k_mean": xr.DataArray(pk_summary["pareto_k_mean"]),
        "pareto_k_bad_count": xr.DataArray(float(pk_summary["pareto_k_bad_count"])),
        "pareto_k_bad_fraction": xr.DataArray(pk_summary["pareto_k_bad_fraction"]),
    }

    if store_pointwise:
        pareto_k_da = _pointwise_dataarray(pareto_k, name="pareto_k", dim="obs")
        if pareto_k_da is not None:
            data_vars["pareto_k"] = pareto_k_da

        loo_i_da = _pointwise_dataarray(loo_i, name="loo_i", dim="obs")
        if loo_i_da is not None:
            data_vars["loo_i"] = loo_i_da

    run_ds = xr.Dataset(data_vars)

    plot_ready = _has_required_loo_pit_groups(arviz_tree)

    run_ds.attrs = {
        "kind": "loo",
        "timestamp": datetime.now(UTC).isoformat(),
        "backend": "arviz",
        "pointwise": bool(pointwise),
        "scale": "" if scale is None else str(scale),
        "reff": float("nan") if reff is None else float(reff),
        "var_name": str(var_name),
        "plot_fn": "az.plot_loo_pit",
        "plot_groups": json.dumps(
            ["posterior", "log_likelihood", "posterior_predictive", "observed_data"]
        ),
        "plot_ready": bool(plot_ready),
        "compare_ready": True,
        "warning_bool": bool(warning_value),
        "loo_result_repr": repr(loo_result),
        "loo_result_json": _json_dumps_safe(
            {
                "elpd_loo": elpd_loo,
                "se": se,
                "p_loo": p_loo,
                "looic": looic,
                "n_samples": n_samples,
                "n_data_points": n_data_points,
                "warning": bool(warning_value),
                "good_k": good_k,
                **pk_summary,
            }
        ),
    }

    _add_group(posterior, "diagnostics/runs/loo", run_ds)

    return None


# ---------------------------------------------------------------------
# Log-likelihood computation
# ---------------------------------------------------------------------


def _add_log_likelihood(
    posterior: Distribution,
    model: Any,
    data: Any,
    *,
    var_name: str = "y",
) -> None:
    """Compute pointwise log likelihoods for ``add_loo``.

    This is an internal helper. The public LOO workflow is ``add_loo``; callers
    should not need a separate manual log-likelihood precomputation step in the
    normal API.

    Parameters
    ----------
    posterior : ApproximateDistribution
        Fitted posterior.
    model : SimpleModel
        The model the posterior was conditioned from. Must expose
        ``_likelihood`` with a ``per_datum_log_likelihood(params, datum)``
        method — satisfied by ``GLMLikelihood`` and any other likelihood
        implementing ``ConditionallyIndependentLikelihood``.
    data : Record-like
        Observed data with fields matching the likelihood's
        ``data_template``. For ``GLMLikelihood``, needs ``X`` and ``y``.
    var_name : str
        Variable name written into the log-likelihood xarray Dataset.

    Returns
    -------
    None
        Mutates ``posterior._annotations`` in place.

    Notes
    -----
    Uses ``jax.vmap`` to vectorise over draws and observations in a single
    call per chain, giving a large speedup over a Python loop. Falls back
    to a Python loop if the likelihood or params are not JAX-traceable.

    Examples
    --------
    ::

        add_loo(posterior, model=model, data=data)
        print(posterior.diagnostics.loo.elpd_loo)
    """
    import jax
    import jax.numpy as jnp

    ll = model._likelihood
    n_chains = posterior.num_chains
    n_draws = posterior.num_draws

    y = jnp.asarray(data["y"])
    X = jnp.asarray(ll._x)  # (n_obs, n_features)
    n_obs = y.shape[0]

    # ------------------------------------------------------------------
    # Build field metadata from one reference draw for flat↔Record conversion
    # ------------------------------------------------------------------
    ref_draws = posterior.draws(chain=0)
    # Leaf fields keyed by full /-path (see ``_leaf_keys`` for the
    # nested-vs-duck-typed rule). _field_meta is the canonical leaf order for
    # the flat<->Record conversion.
    _field_meta = [(f, jnp.asarray(ref_draws[f][0]).shape) for f in _leaf_keys(ref_draws)]

    def _flat_to_record(flat: Any, field_meta: list) -> Record:
        """Reconstruct a named Record from a flat parameter array."""
        out = {}
        idx = 0
        for fname, shape in field_meta:
            size = int(np.prod(shape)) if shape else 1
            val = flat[idx : idx + size]
            out[fname] = val.reshape(shape) if shape else val[0]
            idx += size
        # Positional (path-keyed) construction so /-paths rebuild the nesting;
        # keyword construction would reject a key containing "/".
        return _auto_record(out)

    def _draws_to_flat(draws_c: Any) -> Any:
        """Stack all fields into a (n_draws, n_params) array."""
        parts = []
        for f, _shape in _field_meta:
            arr = jnp.asarray(draws_c[f])  # (n_draws, *shape)
            parts.append(arr.reshape(n_draws, -1) if arr.ndim > 1 else arr[:, None])
        return jnp.concatenate(parts, axis=1)  # (n_draws, n_params)

    # ------------------------------------------------------------------
    # Build a JAX-traceable per-(params, datum) log likelihood function.
    # params here is a flat 1D array of length n_params; datum is a
    # (x_i, y_i) pair. We avoid Record construction inside jax.vmap
    # because Records are Python objects, not JAX pytrees.
    # ------------------------------------------------------------------

    def _log_lik_single(params_flat: Any, x_i: Any, y_i: Any) -> Any:
        """Scalar log likelihood for one draw and one observation.

        Constructs the datum Record at the Python level via a closure so
        JAX only traces the numeric computation inside
        ``per_datum_log_likelihood``, not the Record construction.
        """
        datum = _auto_record({"X": x_i, "y": y_i})
        return ll.per_datum_log_likelihood(params_flat, datum)

    # Note: the per-datum record is constructed inside the vmapped
    # function. If the likelihood's per_datum_log_likelihood is not
    # JAX-traceable (e.g. uses Python control flow on datum fields),
    # the except branch below falls back to a Python loop.

    try:
        # vmap over observations (x_i, y_i) for a fixed draw
        _log_lik_obs = jax.vmap(_log_lik_single, in_axes=(None, 0, 0))

        # vmap over draws for a fixed chain
        _log_lik_draws = jax.vmap(_log_lik_obs, in_axes=(0, None, None))
    except Exception:
        _log_lik_draws = None

    log_lik = np.zeros((n_chains, n_draws, n_obs), dtype=np.float32)

    for c in range(n_chains):
        draws_c = posterior.draws(chain=c)
        params_flat = _draws_to_flat(draws_c)  # (n_draws, n_params)

        try:
            if _log_lik_draws is None:
                raise RuntimeError("JAX vmap is unavailable")

            # Fast path: vmap over (draws, obs) in one call
            chain_ll = _log_lik_draws(params_flat, X, y)  # (n_draws, n_obs)
            log_lik[c] = np.asarray(chain_ll, dtype=np.float32)

        except Exception:
            # Fallback: Python loop over draws (likelihood not JAX-traceable)
            for d in range(n_draws):
                param_record = _flat_to_record(params_flat[d], _field_meta)
                for i in range(n_obs):
                    datum = _auto_record({"X": X[i], "y": y[i]})
                    log_lik[c, d, i] = float(ll.per_datum_log_likelihood(param_record, datum))

    log_lik_ds = _log_likelihood_to_dataset(log_lik, var_name=var_name)
    _add_group(posterior, "arviz/log_likelihood", log_lik_ds)

    return None
