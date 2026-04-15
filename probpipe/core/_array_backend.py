"""Array backend abstraction for numpy/JAX interop.

Provides a small protocol and two implementations so that
``NumericRecord`` and ``NumericRecordArray`` can perform array
operations without hard-coding a specific library.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np

__all__ = ["ArrayBackend", "NumpyBackend", "JaxBackend", "detect_backend"]


@runtime_checkable
class ArrayBackend(Protocol):
    """Minimal array-operation protocol for NumericRecord/NumericRecordArray."""

    def stack(self, arrays: list, axis: int = 0) -> Any: ...
    def concatenate(self, arrays: list, axis: int = -1) -> Any: ...
    def zeros(self, shape: tuple[int, ...]) -> Any: ...
    def mean(self, array: Any, axis: int) -> Any: ...
    def var(self, array: Any, axis: int) -> Any: ...
    def reshape(self, array: Any, shape: tuple[int, ...]) -> Any: ...
    def ravel(self, array: Any) -> Any: ...


class NumpyBackend:
    """ArrayBackend backed by numpy."""

    __slots__ = ()

    def stack(self, arrays: list, axis: int = 0) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays: list, axis: int = -1) -> np.ndarray:
        return np.concatenate(arrays, axis=axis)

    def zeros(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape)

    def mean(self, array: np.ndarray, axis: int) -> np.ndarray:
        return np.mean(array, axis=axis)

    def var(self, array: np.ndarray, axis: int) -> np.ndarray:
        return np.var(array, axis=axis)

    def reshape(self, array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        return np.reshape(array, shape)

    def ravel(self, array: np.ndarray) -> np.ndarray:
        return np.ravel(array)


class JaxBackend:
    """ArrayBackend backed by jax.numpy."""

    __slots__ = ()

    def stack(self, arrays: list, axis: int = 0) -> jnp.ndarray:
        return jnp.stack(arrays, axis=axis)

    def concatenate(self, arrays: list, axis: int = -1) -> jnp.ndarray:
        return jnp.concatenate(arrays, axis=axis)

    def zeros(self, shape: tuple[int, ...]) -> jnp.ndarray:
        return jnp.zeros(shape)

    def mean(self, array: jnp.ndarray, axis: int) -> jnp.ndarray:
        return jnp.mean(array, axis=axis)

    def var(self, array: jnp.ndarray, axis: int) -> jnp.ndarray:
        return jnp.var(array, axis=axis)

    def reshape(self, array: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
        return jnp.reshape(array, shape)

    def ravel(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(array)


# Singletons
_NUMPY_BACKEND = NumpyBackend()
_JAX_BACKEND = JaxBackend()


def detect_backend(array: Any) -> ArrayBackend:
    """Detect the appropriate backend from an array instance.

    Returns ``JaxBackend`` for JAX arrays, ``NumpyBackend`` for numpy arrays.
    Defaults to ``JaxBackend`` for unknown types (since JAX accepts most inputs).
    """
    if isinstance(array, np.ndarray):
        return _NUMPY_BACKEND
    return _JAX_BACKEND
