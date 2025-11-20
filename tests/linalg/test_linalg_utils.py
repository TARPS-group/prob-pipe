# tests/linalg/test_utils.py
import numpy as np
import pytest

from probpipe.linalg import utils as U


def test_add_diag_jitter():
    M = np.eye(3)
    out = U.add_diag_jitter(M, jitter=1e-3, copy=True)
    assert out is not M
    assert np.allclose(np.diag(out), np.diag(M) + 1e-3)

    # vector jitter
    jitter_arr = np.array([1e-3, 2e-3, 3e-3])
    out2 = U.add_diag_jitter(M, jitter=jitter_arr, copy=True)
    assert np.allclose(np.diag(out2), np.array([1.0, 1.0, 1.0]) + jitter_arr)

    # in-place (copy=False) modifies same object when input is ndarray
    M2 = np.eye(3)
    ret = U.add_diag_jitter(M2, jitter=1e-4, copy=False)
    assert ret is M2
    assert np.allclose(np.diag(M2), np.array([1.0, 1.0, 1.0]) + 1e-4)

    # wrong-shaped jitter raises
    with pytest.raises(ValueError):
        U.add_diag_jitter(np.eye(3), jitter=np.array([1e-3, 2e-3]))
