"""General utilities for probpipe.diagnostics.

Private module — do not import directly.
All public symbols are re-exported via probpipe.diagnostics.__init__.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "_component_name",
    "_dataset_values",
    "_resolve_generative_likelihood",
    "_record_get",
    "_safe_float",
    "_as_numpy",
    "_json_dumps_safe",
]

import json
import numpy as np


def _record_get(obj: Any, key: str, default: Any = None) -> Any:
    """Get a field from dict-like, Record-like, or attribute-like objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj[key]
    except Exception:
        pass
    try:
        return getattr(obj, key)
    except Exception:
        pass
    get = getattr(obj, "get", None)
    if callable(get):
        try:
            return get(key, default)
        except Exception:
            pass
    return default


def _safe_float(value: Any) -> float:
    """Convert value to float, returning NaN on failure."""
    if value is None:
        return float("nan")
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr)
        return float(arr.ravel()[0])
    except Exception:
        return float("nan")


def _as_numpy(obj: Any) -> "np.ndarray | None":
    """Best-effort conversion to NumPy array."""
    if obj is None:
        return None
    if hasattr(obj, "samples"):
        try:
            return np.asarray(obj.samples)
        except Exception:
            pass
    try:
        return np.asarray(obj)
    except Exception:
        return None


def _json_dumps_safe(obj: Any) -> str:
    """JSON-dump helper for xarray attrs."""
    try:
        return json.dumps(obj)
    except TypeError:
        try:
            return json.dumps(str(obj))
        except Exception:
            return "{}"


def _component_name(param: str, index: tuple[int, ...]) -> str:
    """Return a stable display name for an event component."""
    if not index:
        return param
    suffix = ", ".join(str(i) for i in index)
    return f"{param}[{suffix}]"


def _dataset_values(ds: Any) -> dict[str, float]:
    """Flatten scalar or event-shaped diagnostic outputs into named values."""
    values: dict[str, float] = {}

    for param in ds.data_vars:
        arr = np.asarray(ds[param], dtype=float)
        if arr.shape == ():
            values[param] = float(arr.item())
            continue

        for index in np.ndindex(arr.shape):
            values[_component_name(param, index)] = float(arr[index])

    return values


def _resolve_generative_likelihood(
    distribution: Any,
    generative_likelihood: Any = None,
) -> Any:
    """Auto-detect generative likelihood from a posterior distribution.

    Resolution order:

    1. Explicitly passed ``generative_likelihood`` argument.
    2. ``distribution["data"]`` — works for
       :class:`~probpipe.modeling.SimpleGenerativeModel` where
       ``__getitem__("data")`` returns a
       :class:`~probpipe.modeling.GenerativeLikelihood`.
    3. ``distribution._likelihood`` — direct attribute fallback.
    4. ``distribution.generative_likelihood`` — future-proofing.
    5. Raise a descriptive :class:`ValueError`.

    Parameters
    ----------
    distribution : Distribution
        Posterior or prior — typically a ``SimpleGenerativeModel``
        or a conditioned posterior.
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
    # 1. Explicit argument — highest priority
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
        "Could not auto-detect a generative likelihood from the distribution.\n"
        "Either:\n"
        "  (a) pass `generative_likelihood` explicitly, or\n"
        "  (b) use a SimpleGenerativeModel whose 'data' component "
        "has a `generate_data()` method."
    )
