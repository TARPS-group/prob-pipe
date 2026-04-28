"""Aux hooks for round-tripping Record leaves through `NumericRecord`.

ProbPipe's native array form is the JAX array. A :class:`Record`
permissively holds whatever the user passes (numpy / jax arrays,
``xarray.DataArray``, ``pandas.Series``/``DataFrame``, plain Python
scalars, opaque objects). Conversion to :class:`NumericRecord` calls
``jnp.asarray`` on every leaf, which discards backend-specific metadata
(xarray dims/coords/attrs, pandas index/columns/dtypes). The aux
registry restores that metadata on the reverse trip.

Each registered hook is an :class:`AuxHooks` pair:

* ``capture(leaf)`` — extract the metadata that ``jnp.asarray`` would
  drop. Called at ``NumericRecord`` construction.
* ``restore(jax_array, aux)`` — rebuild the original backend object
  from a JAX array plus the captured metadata. Called by
  :meth:`NumericRecord.to_native`.

Built-in registrations (gated on import availability):

* ``xarray.DataArray`` — captures ``(dims, coords, attrs)``.
* ``pandas.Series`` — captures ``(index, name, dtype)``.
* ``pandas.DataFrame`` — captures ``(index, columns, dtypes)``.

``numpy.ndarray`` and ``jax.Array`` are deliberately absent from the
registry — they have no metadata worth preserving and the
"no hook → no aux" branch in :meth:`NumericRecord.to_native` returns
them as plain JAX arrays.

The registry is a metadata side-channel, not a behavioural-dispatch
hierarchy. If you need backend-specific reductions or device
placement, convert to the appropriate type yourself; this module
only handles round-trip metadata.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

__all__ = [
    "AuxHooks",
    "register_aux",
    "aux_for",
]


@dataclass(frozen=True)
class AuxHooks:
    """A pair of ``(capture, restore)`` hooks for one backend type.

    Parameters
    ----------
    capture : callable
        ``capture(leaf) -> aux`` — extract backend-specific metadata
        that would otherwise be lost when ``jnp.asarray`` coerces the
        leaf to a JAX array. May return any pickle-friendly value.
    restore : callable
        ``restore(arr, aux) -> leaf`` — reconstruct an instance of the
        original backend type from a JAX array and the previously
        captured ``aux`` blob.
    """

    capture: Callable[[Any], Any]
    restore: Callable[[jax.Array, Any], Any]


# Registry keyed by leaf type. Walk via ``aux_for`` rather than direct
# ``__getitem__`` so subclasses pick up their parent's hooks.
aux_registry: dict[type, AuxHooks] = {}


def register_aux(
    leaf_type: type,
    *,
    capture: Callable[[Any], Any],
    restore: Callable[[jax.Array, Any], Any],
) -> None:
    """Register `(capture, restore)` hooks for a backend leaf type.

    Parameters
    ----------
    leaf_type : type
        The Python type of the leaves whose metadata should be
        preserved across a ``Record``/``NumericRecord`` round-trip.
    capture : callable
        ``capture(leaf) -> aux``.
    restore : callable
        ``restore(arr, aux) -> leaf``.

    Notes
    -----
    Re-registering an existing ``leaf_type`` overwrites the previous
    hook silently. Lookup uses :func:`aux_for` which walks the MRO of
    ``type(obj)``, so registering a base class also covers its
    subclasses.
    """
    aux_registry[leaf_type] = AuxHooks(capture=capture, restore=restore)


def aux_for(obj: Any) -> AuxHooks | None:
    """Return the registered hooks for ``obj``, or ``None`` if absent.

    Walks the MRO of ``type(obj)`` so subclass instances pick up
    base-class registrations.

    Notes
    -----
    Exact-type lookup is checked before the MRO walk so the common
    ``np.ndarray`` / ``jax.Array`` leaves on the
    ``NumericRecord.__init__`` / pytree-unflatten hot path skip a
    multi-step MRO traversal.
    """
    if not aux_registry:
        return None
    cls = type(obj)
    # Exact-type fast path — covers the common case where the
    # registered key is the leaf's concrete class.
    hooks = aux_registry.get(cls)
    if hooks is not None:
        return hooks
    for base in cls.__mro__[1:]:
        hooks = aux_registry.get(base)
        if hooks is not None:
            return hooks
    return None


# ---------------------------------------------------------------------------
# Built-in registrations (gated on import availability)
# ---------------------------------------------------------------------------


def _register_xarray() -> None:
    """Register the built-in ``xarray.DataArray`` aux hooks.

    No-op when xarray isn't importable, so probpipe stays usable
    without xarray installed.
    """
    try:
        import xarray as xr
    except ImportError:
        return

    def _capture(leaf: "xr.DataArray") -> dict[str, Any]:
        # Capture each coord's full ``(dims, values, attrs)`` triple so
        # multi-dim coords and per-coord attrs survive the round-trip
        # — not just the values aligned with the coord's own name.
        # ``v.values`` is already a ``np.ndarray``; no need to re-wrap.
        coords: dict[str, dict[str, Any]] = {}
        for k, v in leaf.coords.items():
            coords[k] = {
                "dims": tuple(v.dims),
                "values": v.values,
                "attrs": dict(v.attrs),
            }
        return {
            "dims": tuple(leaf.dims),
            "coords": coords,
            "attrs": dict(leaf.attrs),
            "name": leaf.name,
        }

    def _restore(arr: jax.Array, aux: dict[str, Any]) -> "xr.DataArray":
        # Rebuild each coord as a ``DataArray(values, dims=…, attrs=…)``
        # so xarray sees the original dims and attrs.
        coords = {
            k: xr.DataArray(
                v["values"], dims=v["dims"], attrs=v.get("attrs", {}),
            )
            for k, v in aux["coords"].items()
        }
        return xr.DataArray(
            np.asarray(arr),
            dims=aux["dims"],
            coords=coords,
            attrs=aux["attrs"],
            name=aux.get("name"),
        )

    register_aux(xr.DataArray, capture=_capture, restore=_restore)


def _register_pandas() -> None:
    """Register the built-in ``pandas.Series`` and ``pandas.DataFrame`` aux hooks.

    No-op when pandas isn't importable.

    Notes
    -----
    The captured aux blob keeps live ``pandas.Index`` / ``columns`` /
    ``dtype`` objects rather than copies. That's intentional for the
    in-memory round-trip semantics, but it means a ``NumericRecord.aux``
    dict containing pandas entries is not pickle-portable to a process
    that lacks pandas — unpickling will fail to reconstruct the
    ``Index`` type.
    """
    try:
        import pandas as pd
    except ImportError:
        return

    def _series_capture(leaf: "pd.Series") -> dict[str, Any]:
        return {
            "index": leaf.index,
            "name": leaf.name,
            "dtype": leaf.dtype,
        }

    def _series_restore(arr: jax.Array, aux: dict[str, Any]) -> "pd.Series":
        return pd.Series(
            np.asarray(arr),
            index=aux["index"],
            name=aux["name"],
            dtype=aux["dtype"],
        )

    def _frame_capture(leaf: "pd.DataFrame") -> dict[str, Any]:
        return {
            "index": leaf.index,
            "columns": leaf.columns,
            "dtypes": leaf.dtypes.to_dict(),
        }

    def _frame_restore(arr: jax.Array, aux: dict[str, Any]) -> "pd.DataFrame":
        df = pd.DataFrame(
            np.asarray(arr),
            index=aux["index"],
            columns=aux["columns"],
        )
        # Restore per-column dtypes (numpy materialisation may have
        # promoted everything to a common dtype).
        return df.astype(aux["dtypes"])

    register_aux(pd.Series, capture=_series_capture, restore=_series_restore)
    register_aux(
        pd.DataFrame, capture=_frame_capture, restore=_frame_restore
    )


_register_xarray()
_register_pandas()
