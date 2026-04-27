"""dtype helpers for ProbPipe.

ProbPipe follows JAX's dtype rules: the default float dtype is whatever
``jnp.zeros(()).dtype`` returns (``float32`` normally, ``float64`` when
``jax.config.jax_enable_x64`` is set). Users opt into float64 either
globally via JAX's x64 flag, or per-distribution by passing parameters
that already carry float64 dtype.
"""

from __future__ import annotations

import jax.numpy as jnp

from .custom_types import Array, ArrayLike


def _default_float_dtype() -> jnp.dtype:
    """Return JAX's current default float dtype.

    Honors ``jax.config.jax_enable_x64`` automatically.
    """
    return jnp.zeros((), dtype=float).dtype


def _promote_floats(*xs: ArrayLike) -> tuple[jnp.dtype, list[Array]]:
    """Convert each input to an array, promote to a common float dtype.

    Returns the chosen dtype and the list of converted arrays. Inputs
    are first converted via ``jnp.asarray`` (preserving any explicit
    dtype the user provided), then promoted via ``jnp.result_type``.
    Integer-only inputs are promoted to JAX's default float dtype, so
    e.g. ``Normal(loc=0, scale=1)`` works as expected.
    """
    arrs = [jnp.asarray(x) for x in xs]
    dtype = jnp.result_type(*[a.dtype for a in arrs])
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.result_type(dtype, _default_float_dtype())
    return dtype, [a.astype(dtype) for a in arrs]


def _as_float_array(x: ArrayLike) -> Array:
    """Convert *x* to an array, promoting integer inputs to float.

    Single-input variant of :func:`_promote_floats`.
    """
    arr = jnp.asarray(x)
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        arr = arr.astype(_default_float_dtype())
    return arr
