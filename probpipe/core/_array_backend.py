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

Built-in registration: ``pandas.DataFrame`` (the ``.dtypes``-but-no-``.dtype``
shape the duck path misses). ``numpy`` / ``jax`` arrays, ``xarray.DataArray``,
and ``pandas.Series`` are deliberately unregistered — the duck path covers
them.

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
    """

    event_shape: Callable[[Any], tuple[int, ...]] = field(kw_only=True)
    numpy_dtype: Callable[[Any], np.dtype | None] = field(
        kw_only=True, default=lambda obj: np.dtype(obj.dtype)
    )
    is_numeric: Callable[[Any], bool] = field(kw_only=True, default=lambda obj: True)
    to_jax: Callable[[Any], jax.Array] = field(kw_only=True, default=_default_to_jax)
    to_numpy: Callable[[Any], np.ndarray] = field(kw_only=True, default=_default_to_numpy)


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
    Exact-type lookup is checked before the MRO walk so the common
    ``np.ndarray`` / ``jax.Array`` leaves on construction and
    pytree-flatten hot paths skip a multi-step MRO traversal.
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


def to_jax_array(value: Any) -> jax.Array:
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


def _register_pandas_frame() -> None:
    """Register the built-in ``pandas.DataFrame`` backend.

    A frame exposes per-column ``.dtypes`` but no single ``.dtype``, so the
    duck path cannot see it; the registration answers the shape / dtype /
    numericness questions from column metadata without materialising
    values. No-op when pandas isn't importable.
    """
    try:
        import pandas as pd
    except ImportError:
        return

    def _is_numeric(df: pd.DataFrame) -> bool:
        # Lazy import: this module is imported by event_template, which owns
        # the shared dtype predicate — resolve it at call time, post-load.
        from .event_template import _is_numeric_dtype

        dtypes = df.dtypes
        return len(dtypes) > 0 and all(_is_numeric_dtype(d) for d in dtypes)

    def _numpy_dtype(df: pd.DataFrame) -> np.dtype | None:
        dtypes = set(df.dtypes)
        if len(dtypes) == 1:
            return np.dtype(dtypes.pop())
        return None  # heterogeneous columns: no single dtype

    register_array_backend(
        pd.DataFrame,
        ArrayBackend(
            event_shape=lambda df: tuple(df.shape),
            numpy_dtype=_numpy_dtype,
            is_numeric=_is_numeric,
            to_jax=lambda df: jnp.asarray(df.to_numpy()),
            to_numpy=lambda df: df.to_numpy(),
        ),
    )


_register_pandas_frame()
