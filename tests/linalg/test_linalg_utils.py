# tests/linalg/test_linalg_utils.py
import jax.numpy as jnp
import pytest

from probpipe.linalg import utils as U


def test_add_diag_jitter():
    M = jnp.eye(3)
    out = U.add_diag_jitter(M, jitter=1e-3, copy=True)
    assert jnp.allclose(jnp.diag(out), jnp.diag(M) + 1e-3)

    # vector jitter
    jitter_arr = jnp.array([1e-3, 2e-3, 3e-3])
    out2 = U.add_diag_jitter(M, jitter=jitter_arr, copy=True)
    assert jnp.allclose(jnp.diag(out2), jnp.array([1.0, 1.0, 1.0]) + jitter_arr)

    # JAX arrays are immutable, so copy=False still returns a new array
    # but the result should be correct
    M2 = jnp.eye(3)
    ret = U.add_diag_jitter(M2, jitter=1e-4, copy=False)
    assert jnp.allclose(jnp.diag(ret), jnp.array([1.0, 1.0, 1.0]) + 1e-4)

    # wrong-shaped jitter raises
    with pytest.raises(ValueError):
        U.add_diag_jitter(jnp.eye(3), jitter=jnp.array([1e-3, 2e-3]))
