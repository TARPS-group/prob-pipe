# tests/linalg/test_operations.py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from probpipe.linalg.linop import (
    DenseLinOp, DiagonalLinOp, TriangularLinOp,
    RootLinOp, CholeskyLinOp
)

from probpipe.linalg.operations import (
    mah_dist_squared
)


# -----------------------------------------------------------------------------
# Small helpers / baselines for comparison
# -----------------------------------------------------------------------------
def baseline_mah_dist_squared(x, A, y):
    """
    Reference implementation of squared Mahalanobis distance that uses dense 
    solves / numpy. Accepts x shape (n, d) or (d,) and y similarly (or None). 
    Returns array of length n (or scalar if input was single-vector).
    """
    d = A.shape[0]
    X = np.asarray(x).reshape(-1, d)
    if y is not None:
        Y = np.asarray(y).reshape(-1, d)
        if Y.shape[0] not in (1, X.shape[0]):
            # allow broadcasting of y if single row, otherwise require equal batch
            raise ValueError("y must have same batch dim as x or be length-1")
        # broadcast if necessary
        if Y.shape[0] == 1:
            Y = np.repeat(Y, X.shape[0], axis=0)
        X = X - Y

    # solve A z = x for z and compute x^T z
    # using np.linalg.solve for each row (A is dense)
    out = np.empty(X.shape[0], dtype=float)
    for i, xi in enumerate(X):
        zi = np.linalg.solve(A, xi)
        out[i] = float(np.dot(xi, zi))

    return out

# -----------------------------------------------------------------------------
# Fixtures: reproducible data and operator factories
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(26423)


@pytest.fixture(scope="module")
def spd_matrix_and_factors(rng):
    d = 7 # matrix dimension
    S = rng.normal(size=(d, d))
    A = S @ S.T  # symmetric positive definite
    L = np.linalg.cholesky(A, upper=False)
    return {"d": d, "S": S, "A": A, "L": L}


# Operator factory list: (id, factory_fn)
# factory_fn takes (S, A, L) and returns an operator instance
operator_factories = [
    ("root", lambda S, A, L: RootLinOp(S)),
    ("cholesky", lambda S, A, L: CholeskyLinOp(TriangularLinOp(L, lower=True))),
    ("dense", lambda S, A, L: DenseLinOp(A)),
]


# Parametrize over the operator factories so each test runs for each operator type.
@pytest.mark.parametrize("op_id,op_factory", operator_factories, ids=[t[0] for t in operator_factories])
@pytest.mark.parametrize("batch_mode", ["batch", "single"])
@pytest.mark.parametrize("with_y", [False, True])
def test_mah_dist_squared_against_baseline(op_id, op_factory, batch_mode, with_y, spd_matrix_and_factors, rng):
    """
    Parametrized test that compares mah_dist_squared(...) for different linear operator
    implementations to a dense baseline.
    We test:
      - batch inputs (n x d) and single-vector (d,) inputs,
      - with and without y (None or same-shape vector).
    """
    d = spd_matrix_and_factors["d"]
    S = spd_matrix_and_factors["S"]
    A = spd_matrix_and_factors["A"]
    L = spd_matrix_and_factors["L"]

    # Build operator under test
    op = op_factory(S, A, L)

    # Build x, y depending on the batch_mode
    if batch_mode == "batch":
        n = 20
        x = rng.normal(size=(n, d))
        y = rng.normal(size=(n, d)) if with_y else None
    else:
        # single vector case
        x = rng.normal(size=(d,))
        y = rng.normal(size=(d,)) if with_y else None

    # expected from dense baseline
    expected = baseline_mah_dist_squared(x, A, y)

    # compute using the implementation-under-test
    got = mah_dist_squared(x, op, y)

    # Compare. We return scalar for single-vector baseline; ensure shapes align.
    # convert both to arrays for consistent comparison
    expected_arr = np.atleast_1d(expected)
    got_arr = np.atleast_1d(got)

    assert_allclose(got_arr, expected_arr, rtol=0, atol=1e-10,
                    err_msg=f"Mismatch for operator '{op_id}', batch_mode={batch_mode}, with_y={with_y}")


# ----------------------
# Edge / additional tests
# ----------------------
def test_mah_dist_squared_broadcast_y(rng, spd_matrix_and_factors):
    """Test that a single y vector is broadcast across a batch of x (if this is supported)."""
    d = spd_matrix_and_factors["d"]
    S = spd_matrix_and_factors["S"]
    A = spd_matrix_and_factors["A"]
    L = spd_matrix_and_factors["L"]

    op = RootLinOp(S)
    n = 7
    x = rng.normal(size=(n, d))
    y_single = rng.normal(size=(d,))
    # baseline: pass y repeated
    expected = baseline_mah_dist_squared(x, A, np.repeat(y_single.reshape(1, -1), n, axis=0))
    got = mah_dist_squared(x, op, y_single)
    assert_allclose(np.atleast_1d(got), expected, rtol=0, atol=1e-10)


def test_mah_dist_squared_incorrect_shape_raises(rng, spd_matrix_and_factors):
    """
    Basic shape-mismatch test: if x has wrong second dimension, the function should raise.
    Adjust the expected exception type if your implementation raises a more specific error.
    """
    d = spd_matrix_and_factors["d"]
    S = spd_matrix_and_factors["S"]

    op = RootLinOp(S)

    # x with wrong second dimension
    x_bad = rng.normal(size=(5, d + 1))
    with pytest.raises((ValueError, np.linalg.LinAlgError, AssertionError, TypeError)):
        mah_dist_squared(x_bad, op, None)
