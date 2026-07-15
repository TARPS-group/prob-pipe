"""The array-backend registry: recognition and conversion for native leaves.

A :class:`~probpipe.NumericRecord` stores each leaf **in its native form**
(a ``jax`` / ``numpy`` array, an ``xarray.DataArray``, a ``pandas`` object,
...) and converts to ``jax.Array`` lazily, at the compute boundary. Most
numeric containers need no support code for that: anything exposing a numpy
``dtype`` and ``shape`` and speaking the numpy protocol is recognised by
duck-typing and converted with ``jnp.asarray``.

This registry exists for the containers the duck path cannot see or convert
correctly. A registered :class:`ArrayBackend` tells ProbPipe, for one leaf
type, how to answer the four questions every numeric gate asks:

* ``is_numeric(obj)`` — does *this instance* hold numeric data? (A frame
  type is registered as a whole; a particular frame may hold string
  columns.)
* ``event_shape(obj)`` — the leaf's event shape, read without touching
  values.
* ``numpy_dtype(obj)`` — the leaf's single numpy dtype, or ``None`` when it
  has no single dtype (a heterogeneous frame).
* ``to_jax(obj)`` / ``to_numpy(obj)`` — materialising conversions, used at
  the compute boundary and for content fingerprints.

Every consumer — template inference and ``ArraySpec.is_valid``, the
``Record`` → ``NumericRecord`` promotion probe, the ``NumericRecord``
conversion cache, batch stacking, and ``fingerprint()`` — resolves
**registry first, duck-typing second**, so registering one backend makes a
new array type recognised, validated, promoted, converted, and fingerprinted
everywhere at once.

Built-in registrations: ``xarray.DataArray`` and ``pandas.Series`` /
``pandas.DataFrame``. They are registered rather than left to the duck path
for two reasons. The first is to expose their identity-bearing metadata (an
``xarray`` array's dims / coords / attrs / name, a ``pandas`` object's index /
columns) through the ``metadata`` hook. The second, for pandas, is to handle
nullable / masked numeric columns (``Int64``, ``Float64``, ``boolean``). The
generic duck path cannot see those columns as numeric, but the pandas backend
accepts them, encoding each NA as ``NaN`` in the columns' common dense dtype.
Bare ``numpy`` / ``jax`` arrays stay unregistered — the duck path covers them and
they carry no metadata beyond their values.

Batch stacking hooks (a per-backend native ``stack`` / ``element``) are
expected to join this registry with the batch-axis rework; the keyword-only
field layout keeps that addition non-breaking for existing registrations.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "ArrayBackend",
    "array_backend_for",
    "register_array_backend",
]


def _default_to_jax(obj: Any) -> jax.Array:
    return jnp.asarray(np.asarray(obj))


def _default_to_numpy(obj: Any) -> np.ndarray:
    return np.asarray(obj)


def _safe_numpy_dtype(dtype: Any) -> np.dtype | None:
    """``np.dtype(dtype)`` or ``None`` when the dtype is not numpy-coercible.

    A pandas extension / masked dtype (``Int64Dtype`` etc.) reports a numpy
    ``kind`` but raises on ``np.dtype(...)``; such a container has no single
    dense numpy dtype, so this returns ``None`` rather than propagating.
    """
    try:
        return np.dtype(dtype)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ArrayBackend:
    """Recognition and conversion hooks for one native array-container type.

    All fields are keyword-only. ``event_shape`` is required; the rest have
    working defaults for containers that speak the numpy protocol.

    Parameters
    ----------
    event_shape : callable
        ``event_shape(obj) -> tuple[int, ...]`` — the leaf's event shape,
        read from container metadata without materialising values.
    numpy_dtype : callable, optional
        ``numpy_dtype(obj) -> np.dtype | None`` — the leaf's single numpy
        dtype, or ``None`` when the container has no single dtype (e.g. a
        heterogeneous frame). Defaults to reading ``obj.dtype``.
    is_numeric : callable, optional
        ``is_numeric(obj) -> bool`` — whether *this instance* holds numeric
        data. Defaults to ``True`` (the registered type is numeric by
        construction).
    to_jax : callable, optional
        ``to_jax(obj) -> jax.Array`` — the compute-boundary conversion.
        Materialises lazy/disk-backed values. Defaults to
        ``jnp.asarray(np.asarray(obj))``.
    to_numpy : callable, optional
        ``to_numpy(obj) -> np.ndarray`` — the fingerprint materialisation.
        Defaults to ``np.asarray(obj)``.
    metadata : callable, optional
        ``metadata(obj) -> Any`` — the container's **identity-bearing
        metadata beyond its values** (an ``xarray`` array's dims / coords /
        attrs / name, a ``pandas`` object's index / columns), as a
        fingerprint-able structure (nested tuples / dicts / arrays / scalars).
        Values are *not* included (they are covered by ``to_numpy``). This is
        what makes the metadata part of a record's identity: ``fingerprint``
        folds it into the digest and ``Record.__eq__`` compares it, so two
        containers with equal values but different coords are distinct.
        Defaults to ``None`` — a container with no identity-bearing metadata
        (a bare array, zarr, dask), whose identity is its values alone.
    """

    event_shape: Callable[[Any], tuple[int, ...]] = field(kw_only=True)
    numpy_dtype: Callable[[Any], np.dtype | None] = field(
        kw_only=True, default=lambda obj: _safe_numpy_dtype(obj.dtype)
    )
    is_numeric: Callable[[Any], bool] = field(kw_only=True, default=lambda obj: True)
    to_jax: Callable[[Any], jax.Array] = field(kw_only=True, default=_default_to_jax)
    to_numpy: Callable[[Any], np.ndarray] = field(kw_only=True, default=_default_to_numpy)
    metadata: Callable[[Any], Any] = field(kw_only=True, default=lambda obj: None)


# Registry keyed by leaf type. Walk via ``array_backend_for`` rather than
# direct ``__getitem__`` so subclasses pick up their parent's registration.
_backend_registry: dict[type, ArrayBackend] = {}


def register_array_backend(leaf_type: type, backend: ArrayBackend) -> None:
    """Register *backend* as the array-backend hooks for *leaf_type*.

    Registering a type makes its instances first-class numeric leaves
    everywhere at once: template inference and ``ArraySpec.is_valid``
    recognise them, all-numeric records holding them auto-promote to
    ``NumericRecord``, the compute boundary converts them through
    ``backend.to_jax``, batch stacking coerces them per element, and
    ``fingerprint()`` hashes their content through ``backend.to_numpy``.

    Parameters
    ----------
    leaf_type : type
        The container type to register. Lookup walks the MRO of a value's
        type, so registering a base class also covers its subclasses.
    backend : ArrayBackend
        The recognition/conversion hooks.

    Notes
    -----
    Re-registering an existing ``leaf_type`` overwrites the previous
    registration and issues a ``UserWarning``, so an accidental double
    registration (e.g. two libraries claiming the same type) is visible.
    """
    if leaf_type in _backend_registry:
        warnings.warn(
            f"register_array_backend: overwriting the existing ArrayBackend "
            f"registration for {leaf_type.__name__}.",
            stacklevel=2,
        )
    _backend_registry[leaf_type] = backend


def array_backend_for(obj: Any) -> ArrayBackend | None:
    """Return the registered backend for ``obj``'s type, or ``None``.

    Walks the MRO of ``type(obj)`` so subclass instances pick up
    base-class registrations. Consumers resolve registry-first and fall
    back to numpy-protocol duck-typing when this returns ``None``.

    Notes
    -----
    Exact-type lookup is checked before the MRO walk, so a registered type
    resolves in one dict probe and only a subclass of a registered type pays
    for the walk. Unregistered leaves (bare ``np.ndarray`` / ``jax.Array``,
    the common case) miss the exact probe and walk their MRO, but that is
    short and off the hot path — ``_child_field_as_jax`` fast-returns a
    ``jax.Array`` leaf before this is ever called.
    """
    if not _backend_registry:
        return None
    cls = type(obj)
    backend = _backend_registry.get(cls)
    if backend is not None:
        return backend
    for base in cls.__mro__[1:]:
        backend = _backend_registry.get(base)
        if backend is not None:
            return backend
    return None


def _to_jax_array(value: Any) -> jax.Array:
    """Convert a native numeric leaf to ``jax.Array`` — the compute-boundary step.

    Uses the registered backend's ``to_jax`` when *value*'s type is
    registered, else ``jnp.asarray`` (the numpy-protocol duck path). This is
    the single conversion every boundary — the ``NumericRecord`` cache,
    batch stacking — routes through, and the point where a lazy or
    disk-backed value materialises.
    """
    backend = array_backend_for(value)
    if backend is not None:
        return backend.to_jax(value)
    return jnp.asarray(value)


# ---------------------------------------------------------------------------
# Built-in registrations (gated on import availability)
# ---------------------------------------------------------------------------
#
# ``xarray`` and ``pandas`` containers are registered (not left to the
# numpy-protocol duck path) so their identity-bearing metadata — coords /
# dims / attrs / name, or index / columns — is available to ``fingerprint``
# and ``Record.__eq__`` via the ``metadata`` hook. The duck path remains for
# containers whose identity is their values alone (bare numpy / jax arrays,
# zarr, dask), which need no registration and carry no ``metadata``.


def _register_xarray() -> None:
    """Register the built-in ``xarray.DataArray`` backend. No-op without xarray."""
    try:
        import xarray as xr
    except ImportError:
        return

    from .event_template import _is_numeric_dtype

    def _metadata(da: xr.DataArray) -> Any:
        # Everything that distinguishes the container beyond its values: the
        # named dims, each coord's values, free-form attrs, and the name.
        return {
            "dims": tuple(da.dims),
            "coords": {str(k): np.asarray(v.values) for k, v in da.coords.items()},
            "attrs": dict(da.attrs),
            "name": da.name,
        }

    register_array_backend(
        xr.DataArray,
        ArrayBackend(
            event_shape=lambda da: tuple(da.shape),
            numpy_dtype=lambda da: _safe_numpy_dtype(da.dtype),
            is_numeric=lambda da: _is_numeric_dtype(da.dtype),
            to_jax=lambda da: jnp.asarray(da.values),
            to_numpy=lambda da: np.asarray(da.values),
            metadata=_metadata,
        ),
    )


def _register_pandas() -> None:
    """Register the built-in ``pandas.Series`` / ``pandas.DataFrame`` backends.

    A ``Series`` speaks the numpy protocol but carries an index and name; a
    ``DataFrame`` additionally has no single ``.dtype`` (per-column ``.dtypes``
    the duck path cannot read). Both answer shape / dtype / numericness from
    metadata without materialising values.

    Nullable / masked numeric columns (``Int64``, ``Float64``, ``boolean``,
    pyarrow numerics) are accepted as numeric even though the generic gate
    rejects their dtype: such a column has no dense numpy view, so the whole
    container converts through ``float64`` with each NA encoded as ``NaN`` —
    the only missing-value representation ``jax`` offers. The native leaf is
    stored verbatim, so the validity mask survives at rest; NA becomes NaN
    only at the compute boundary. Integer / boolean columns therefore promote
    to float. Categorical / string / datetime extension dtypes are not numeric
    and leave the container a plain ``Record``. No-op without pandas.
    """
    try:
        import pandas as pd
    except ImportError:
        return

    from .event_template import _is_numeric_dtype

    def _is_nullable_numeric(dtype: Any) -> bool:
        # A masked / extension *numeric* dtype: numeric, but with a validity
        # mask and no dense numpy view. Excludes categorical / string /
        # datetime, which are extension dtypes yet not numeric.
        is_ext = pd.api.types.is_extension_array_dtype(dtype)
        return is_ext and pd.api.types.is_numeric_dtype(dtype)

    def _column_numeric(dtype: Any) -> bool:
        # A column ProbPipe can turn into a dense numeric array: a plain numpy
        # numeric dtype (generic gate), or a nullable numeric one (NA -> NaN).
        return _is_numeric_dtype(dtype) or _is_nullable_numeric(dtype)

    def _dense_dtype(dtypes: Any) -> np.dtype | None:
        # The single dense numpy dtype a Series / DataFrame of these column
        # dtypes densifies to (via to_numpy / to_jax), or None when a column has
        # no dense numpy dtype. A masked numeric column has no dense numpy view
        # and must hold NA as NaN, so it contributes a float: a nullable float
        # keeps its own width (Float32 -> float32), a nullable integer / boolean
        # promotes to float64. Every other column contributes its own dtype. The
        # result is their common promotion, so a complex column yields complex128
        # rather than being truncated to float.
        contribs: list[np.dtype] = []
        for d in dtypes:
            if _is_nullable_numeric(d):
                nd = getattr(d, "numpy_dtype", None)
                if nd is not None and np.issubdtype(nd, np.floating):
                    contribs.append(np.dtype(nd))
                else:
                    contribs.append(np.dtype("float64"))
            else:
                nd = _safe_numpy_dtype(d)
                if nd is None:
                    return None
                contribs.append(nd)
        return np.result_type(*contribs) if contribs else None

    def _dense(obj: Any, dtypes: Any) -> np.ndarray:
        # A masked column has no dense numpy view; converting with an explicit
        # target dtype encodes each NA as NaN and keeps complex columns complex
        # (see _dense_dtype). Otherwise the native numpy view is used unchanged.
        if any(_is_nullable_numeric(d) for d in dtypes):
            return obj.to_numpy(dtype=_dense_dtype(dtypes), na_value=np.nan)
        return obj.to_numpy()

    def _series_metadata(s: pd.Series) -> Any:
        return {"index": np.asarray(s.index), "name": s.name}

    def _series_numpy_dtype(s: pd.Series) -> np.dtype | None:
        # Report the single dense dtype the series densifies to (see
        # _dense_dtype), so numpy_dtype matches what to_numpy / to_jax produce.
        return _dense_dtype((s.dtype,))

    register_array_backend(
        pd.Series,
        ArrayBackend(
            event_shape=lambda s: tuple(s.shape),
            numpy_dtype=_series_numpy_dtype,
            is_numeric=lambda s: _column_numeric(s.dtype),
            to_jax=lambda s: jnp.asarray(_dense(s, (s.dtype,))),
            to_numpy=lambda s: _dense(s, (s.dtype,)),
            metadata=_series_metadata,
        ),
    )

    def _frame_is_numeric(df: pd.DataFrame) -> bool:
        dtypes = df.dtypes
        return len(dtypes) > 0 and all(_column_numeric(d) for d in dtypes)

    def _frame_numpy_dtype(df: pd.DataFrame) -> np.dtype | None:
        # Report the single dense dtype the frame densifies to (see
        # _dense_dtype), so numpy_dtype matches what to_numpy / to_jax produce.
        return _dense_dtype(df.dtypes)

    def _frame_metadata(df: pd.DataFrame) -> Any:
        return {
            "index": np.asarray(df.index),
            "columns": np.asarray(df.columns),
            "dtypes": tuple(str(d) for d in df.dtypes),
        }

    register_array_backend(
        pd.DataFrame,
        ArrayBackend(
            event_shape=lambda df: tuple(df.shape),
            numpy_dtype=_frame_numpy_dtype,
            is_numeric=_frame_is_numeric,
            to_jax=lambda df: jnp.asarray(_dense(df, df.dtypes)),
            to_numpy=lambda df: _dense(df, df.dtypes),
            metadata=_frame_metadata,
        ),
    )


_register_xarray()
_register_pandas()
