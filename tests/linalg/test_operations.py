# tests/linalg/test_operations.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe.linalg.linear_operator import (
    DenseLinOp,
    DiagonalLinOp,
    TriangularLinOp,
    RootLinOp,
    CholeskyLinOp,
    LinAlgError,
)

from probpipe.linalg.operations import (
    logdet,
    mah_dist_squared,
)

# The full set of operations being tested
ALL_OPERATIONS = {"logdet", "mah_dist_squared"}

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

# compare scalars/arrays consistently
def _compare_results(got, expected, *, rtol=1e-4, atol=1e-4, msg=None):
    got_arr = jnp.atleast_1d(jnp.asarray(got))
    expected_arr = jnp.atleast_1d(jnp.asarray(expected))
    assert jnp.allclose(got_arr, expected_arr, rtol=rtol, atol=atol), (
        f"{msg or 'Mismatch'}: got={got_arr}, expected={expected_arr}"
    )


# -----------------------------------------------------------------------------
# Baseline/reference functions to compare against
# -----------------------------------------------------------------------------

def baseline_logdet(A):
    sign, logabsdet = jnp.linalg.slogdet(A)
    if sign <= 0:
        raise LinAlgError("Non-positive determinant in baseline_logdet")
    return float(logabsdet)


def baseline_mah_dist_squared(x, A, y):
    d = A.shape[0]
    X = jnp.asarray(x).reshape(-1, d)
    if y is not None:
        Y = jnp.asarray(y).reshape(-1, d)
        if Y.shape[0] == 1:
            Y = jnp.repeat(Y, X.shape[0], axis=0)
        X = X - Y

    # Solve and compute quadratic form per row
    Ainv_X = jnp.linalg.solve(A, X.T).T  # (n, d)
    out = jnp.sum(X * Ainv_X, axis=1)
    return out if jnp.asarray(x).ndim == 2 else out[0]


# -----------------------------------------------------------------------------
# Fixtures: reproducible data and operator factories
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rng_key():
    return jax.random.PRNGKey(26423)

@pytest.fixture(scope="module")
def linop_resources(rng_key):
    """Core components used to construct various types of linear operators."""
    k1, k2, k3 = jax.random.split(rng_key, 3)

    # Symmetric positive definite
    dim_pd = 7
    root_pd = jax.random.normal(k1, shape=(dim_pd, dim_pd))
    dense_pd = root_pd @ root_pd.T
    lower_chol_pd = jnp.linalg.cholesky(dense_pd)
    upper_chol_pd = lower_chol_pd.T
    pd = {"dim": dim_pd, "root": root_pd, "dense": dense_pd,
           "lower_chol": lower_chol_pd, "upper_chol": upper_chol_pd}

    # Diagonal matrix with positive diagonal (hence positive definite)
    dim_diag_pd = 10
    diagonal_diag_pd = jax.random.uniform(k2, shape=(dim_diag_pd,), minval=1.0, maxval=10.0)
    sqrt_diagonal_diag_pd = jnp.sqrt(diagonal_diag_pd)
    diag_pd = {"dim": dim_diag_pd,
               "diagonal": diagonal_diag_pd,
               "sqrt_diagonal": sqrt_diagonal_diag_pd,
               "dense": jnp.diag(diagonal_diag_pd),
               "dense_root": jnp.diag(sqrt_diagonal_diag_pd)}

    return {"pd": pd, "diag_pd": diag_pd}


# Factories of the form lambda linop_resources : (linop, densified linop, supported operations)
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
                                           linop_resources, rng_key):
    op, dense, supports = factory(linop_resources)

    if "mah_dist_squared" not in supports:
        pytest.skip(f"factory '{factory_id}' does not support mah_dist_squared")

    dim = dense.shape[1]
    k1, k2 = jax.random.split(rng_key)

    if batch_mode == "batch":
        n = 20
        x = jax.random.normal(k1, shape=(n, dim))
        y = jax.random.normal(k2, shape=(n, dim)) if with_y else None
    else:
        x = jax.random.normal(k1, shape=(dim,))
        y = jax.random.normal(k2, shape=(dim,)) if with_y else None

    expected = baseline_mah_dist_squared(x, dense, y)
    got = mah_dist_squared(x, op, y)

    _compare_results(got, expected, msg=f"Mismatch for operator '{factory_id}', batch_mode={batch_mode}, with_y={with_y}")


def test_mah_dist_squared_broadcast_y(rng_key, linop_resources):
    """Test that a single y vector is broadcast across a batch of x."""
    pd = linop_resources["pd"]
    dense = pd["dense"]
    root = pd["root"]

    op = RootLinOp(root)
    n = 7
    dim = dense.shape[0]
    k1, k2 = jax.random.split(rng_key)
    x = jax.random.normal(k1, shape=(n, dim))
    y_single = jax.random.normal(k2, shape=(dim,))
    expected = baseline_mah_dist_squared(x, dense, jnp.repeat(y_single.reshape(1, -1), n, axis=0))
    got = mah_dist_squared(x, op, y_single)
    _compare_results(got, expected)


def test_mah_dist_squared_incorrect_shape_raises(rng_key, linop_resources):
    pd = linop_resources["pd"]
    dim = pd["dim"]
    root = pd["root"]
    op = RootLinOp(root)

    x_bad = jax.random.normal(rng_key, shape=(5, dim + 1))
    with pytest.raises((ValueError, LinAlgError, AssertionError, TypeError)):
        mah_dist_squared(x_bad, op, None)


# -----------------------------------------------------------------------------
# Test logdet
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("factory_id,factory", linop_factories)
def test_logdet_against_baseline(factory_id, factory, linop_resources):
    op, dense, supports = factory(linop_resources)

    if "logdet" not in supports:
        pytest.skip(f"factory '{factory_id}' does not support logdet")

    expected = baseline_logdet(dense)
    got = logdet(op)

    _compare_results(got, expected, msg=f"logdet mismatch for operator '{factory_id}'")


def test_logdet_matches_cholesky_formula(linop_resources):
    pd = linop_resources["pd"]
    dense = pd["dense"]
    lower_chol = pd["lower_chol"]

    expected = baseline_logdet(dense)
    logdet_from_chol = 2.0 * float(jnp.sum(jnp.log(jnp.diag(lower_chol))))
    _compare_results(logdet_from_chol, expected, msg="logdet reference baselines disagree")
