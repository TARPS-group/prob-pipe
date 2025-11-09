# tests/linalg/test_operations.py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from probpipe.linalg.linop import (
    DenseLinOp, DiagonalLinOp, TriangularLinOp,
    RootLinOp, CholeskyLinOp
)

from probpipe.linalg.operations import (
    logdet,
    mah_dist_squared
)

# The full set of operations being tested
ALL_OPERATIONS = {"logdet", "mah_dist_squared"}

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

# compare scalars/arrays consistently
def _compare_results(got, expected, *, rtol=0, atol=1e-10, msg=None):
    got_arr = np.atleast_1d(got)
    expected_arr = np.atleast_1d(expected)
    assert_allclose(got_arr, expected_arr, rtol=rtol, atol=atol, err_msg=msg)


# -----------------------------------------------------------------------------
# Baseline/reference functions to compare against
# -----------------------------------------------------------------------------

def baseline_logdet(A):
    """
    Baseline for log-determinant using numpy.linalg.slogdet.
    For pd matrices sign==1 and logabsdet is the log-determinant.
    """
    sign, logabsdet = np.linalg.slogdet(A)
    if sign <= 0:
        # If the implementation under test raises for non-positive det,
        # you can adjust expected behavior here. For these tests we use pd.
        raise np.linalg.LinAlgError("Non-positive determinant in baseline_logdet")
    return float(logabsdet)


def baseline_mah_dist_squared(x, A, y):
    """
    Baseline (reference) implementation that uses dense solves / numpy.
    Accepts x shape (n, d) or (d,) and y similarly (or None). Returns array
    of length n (or scalar if input was single-vector).
    """
    d = A.shape[0]
    X = np.asarray(x).reshape(-1, d)
    if y is not None:
        Y = np.asarray(y).reshape(-1, d)
        if Y.shape[0] not in (1, X.shape[0]):
            raise ValueError("y must have same batch dim as x or be length-1")
        if Y.shape[0] == 1:
            Y = np.repeat(Y, X.shape[0], axis=0)
        X = X - Y

    out = np.empty(X.shape[0], dtype=float)
    for i, xi in enumerate(X):
        zi = np.linalg.solve(A, xi)
        out[i] = float(np.dot(xi, zi))
    return out if np.asarray(x).ndim == 2 else out[0]


# -----------------------------------------------------------------------------
# Fixtures: reproducible data and operator factories
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(26423)

@pytest.fixture(scope="module")
def linop_resources(rng):
    """ Core components used to construct various types of linear operators """
    
    # Symmetric positive definite
    dim_pd = 7
    root_pd = rng.normal(size=(dim_pd, dim_pd))
    dense_pd = root_pd @ root_pd.T
    lower_chol_pd = np.linalg.cholesky(dense_pd, upper=False)
    upper_chol_pd = lower_chol_pd.T
    pd = {"dim": dim_pd, "root": root_pd, "dense": dense_pd, 
           "lower_chol": lower_chol_pd, "upper_chol": upper_chol_pd}

    # Diagonal matrix with positive diagonal (hence positive definite)
    dim_diag_pd = 10
    diagonal_diag_pd = rng.uniform(1.0, 10.0, size=(dim_diag_pd,))
    sqrt_diagonal_diag_pd = np.sqrt(diagonal_diag_pd)
    diag_pd = {"dim": dim_diag_pd,
               "diagonal": diagonal_diag_pd,
               "sqrt_diagonal": sqrt_diagonal_diag_pd,
               "dense": np.diag(diagonal_diag_pd),
               "dense_root": np.diag(sqrt_diagonal_diag_pd)}

    return {"pd": pd, "diag_pd": diag_pd}


# Factories of the form lambda linop_resources : (linop, densified linop, supported operations)
# The id naming scheme here is "<linop tag>_<resource tag>", where the 
# resource tag is the name of the dictionary element returned by `linop_resources()`
# Supported operations is a set of string function names of the supported operations (the set
# of operations well-defined for each linear operator).
linop_factories = [
    pytest.param("root_pd", 
                 lambda x: (RootLinOp(x["pd"]["root"]), 
                            x["pd"]["dense"],
                            ALL_OPERATIONS),
                 id="root_pd"),

    pytest.param("cholesky_pd", 
                 lambda x: (CholeskyLinOp(TriangularLinOp(x["pd"]["lower_chol"], lower=True)), 
                            x["pd"]["dense"],
                            ALL_OPERATIONS), 
                 id="cholesky_pd"),

    pytest.param("dense_pd", 
                 lambda x: (DenseLinOp(x["pd"]["dense"]), 
                            x["pd"]["dense"],
                            ALL_OPERATIONS), 
                 id="dense_pd"),

    pytest.param("diagonal_diag_pd", 
                 lambda x: (DiagonalLinOp(x["diag_pd"]["diagonal"]), 
                            x["diag_pd"]["dense"],
                            ALL_OPERATIONS), 
                 id="diagonal_diag_pd"),

    pytest.param("root_diag_pd", 
                 lambda x: (RootLinOp(x["diag_pd"]["dense_root"]), 
                            x["diag_pd"]["dense"],
                            ALL_OPERATIONS), 
                 id="root_diag_pd"),

    pytest.param("cholesky_diag_pd", 
                 lambda x: (CholeskyLinOp(TriangularLinOp(x["diag_pd"]["dense_root"], lower=True)), 
                            x["diag_pd"]["dense"],
                            ALL_OPERATIONS), 
                 id="cholesky_diag_pd"),

    pytest.param("dense_diag_pd", 
                 lambda x: (DenseLinOp(x["diag_pd"]["dense"]), 
                            x["diag_pd"]["dense"],
                            ALL_OPERATIONS), 
                 id="dense_diag_pd")
]


# -----------------------------------------------------------------------------
# Test Mahalanobis Distance 
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("factory_id,factory", linop_factories)
@pytest.mark.parametrize("batch_mode", ["batch", "single"])
@pytest.mark.parametrize("with_y", [False, True])
def test_mah_dist_squared_against_baseline(factory_id, factory, batch_mode, with_y, 
                                           linop_resources, rng):
    """
    Parametrized test that compares mah_dist_squared(...) for different linear operator
    implementations to a dense baseline.
    """
    op, dense, supports = factory(linop_resources)

    if "mah_dist_squared" not in supports:
        pytest.skip(f"factory '{factory_id}' does not support mah_dist_squared")

    dim = dense.shape[1]

    if batch_mode == "batch":
        n = 20
        x = rng.normal(size=(n, dim))
        y = rng.normal(size=(n, dim)) if with_y else None
    else:
        x = rng.normal(size=(dim,))
        y = rng.normal(size=(dim,)) if with_y else None

    expected = baseline_mah_dist_squared(x, dense, y)
    got = mah_dist_squared(x, op, y)

    _compare_results(got, expected, msg=f"Mismatch for operator '{factory_id}', batch_mode={batch_mode}, with_y={with_y}")


def test_mah_dist_squared_broadcast_y(rng, linop_resources):
    """Test that a single y vector is broadcast across a batch of x (diagonal/spd example)."""
    pd = linop_resources["pd"]
    dense = pd["dense"]
    root = pd["root"]

    op = RootLinOp(root)
    n = 7
    dim = dense.shape[0]
    x = rng.normal(size=(n, dim))
    y_single = rng.normal(size=(dim,))
    expected = baseline_mah_dist_squared(x, dense, np.repeat(y_single.reshape(1, -1), n, axis=0))
    got = mah_dist_squared(x, op, y_single)
    _compare_results(got, expected)


def test_mah_dist_squared_incorrect_shape_raises(rng, linop_resources):
    pd = linop_resources["pd"]
    dim = pd["dim"]
    root = pd["root"]
    op = RootLinOp(root)

    x_bad = rng.normal(size=(5, dim + 1))
    with pytest.raises((ValueError, np.linalg.LinAlgError, AssertionError, TypeError)):
        mah_dist_squared(x_bad, op, None)


# -----------------------------------------------------------------------------
# Test logdet
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("factory_id,factory", linop_factories)
def test_logdet_against_baseline(factory_id, factory, linop_resources):
    """
    Test that logdet(...) for different linear operator implementations matches
    the dense baseline log-determinant computed with numpy.
    """
    op, dense, supports = factory(linop_resources)

    if "logdet" not in supports:
        pytest.skip(f"factory '{factory_id}' does not support logdet")

    expected = baseline_logdet(dense)
    got = logdet(op)

    _compare_results(got, expected, msg=f"logdet mismatch for operator '{factory_id}'")


def test_logdet_matches_cholesky_formula(linop_resources):
    """
    For pd A with Cholesky L, logdet(A) == 2 * sum(log(diag(L)))
    Double checking the baseline reference function.
    """
    pd = linop_resources["pd"]
    dense = pd["dense"]
    lower_chol = pd["lower_chol"]

    expected = baseline_logdet(dense)
    logdet_from_chol = 2.0 * float(np.sum(np.log(np.diag(lower_chol))))
    _compare_results(logdet_from_chol, expected, msg=f"logdet reference baselines disagree")



 