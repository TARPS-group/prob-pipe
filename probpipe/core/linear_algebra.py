# linear_algebra.py

from typing import Optional, Tuple, Union

import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import cholesky, solve_triangular


Array = jnp.ndarray


def _as_matrix(x: Array) -> Array:
    """Ensure input is a 2D array, potentially via coercing.

    Follows behavior of `jax.numpy.atleast_2d` when called on a single array,
    except that an error is raised for input arrays of dimension greater
    than 2.
    """
    x = jnp.asarray(x)

    if x.ndim == 0:
        return x.reshape((1, 1))
    if x.ndim == 1:
        return x[jnp.newaxis, :]
    if x.ndim == 2:
        return x
    raise ValueError("x must be 0-D, 1-D, or 2-D array.")


def _as_flat_vector(x: Array) -> Array:
    """Ensure input is a 1d array, potentially via coercing."""
    x = jnp.asarray(x)

    if x.ndim == 0:
        return x.reshape((1,))
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[1] == 1:
        return x.flatten()

    raise ValueError("x must be 0-D, 1-D array, or a 2d array with a single column.")


def add_diag_jitter(matrix: Array, jitter: float|Array = 1e-6) -> Array:
    """Add constant to diagonal of square matrix.

    This function adds a scalar constant to the diagonal of a square matrix
    (2d array). The constant is typically a small "jitter" intended to
    promote numerical stability and ensure numerical positive definiteness
    for Cholesky factorizations and matrix inversion.

    Args:
        matrix: A 2d square array.
        jitter: The value to add to the diagonal. Defaults to 1e-6.

    Returns:
        The matrix with jitter added to the diagonal.

    Notes:
        This function was inspired by the `gpjax` function `add_jitter`.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got {matrix.ndim}D array")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected 2d square array, got shape {matrix.shape}")

    return matrix + jnp.eye(matrix.shape[0]) * jitter


def robust_cholesky(
    matrix: Array,
    *,
    lower: bool = True,
    jitter: float = 1e-6,
    symmetrize: bool = True,
) -> Array:
    """Return robust lower-triangular Cholesky factor of covariance matrix.

    This function is a wrapper around `jax.scipy.linalg.cholesky(matrix, lower=lower)`
    that optionally symmetrizes the matrix before attempting the Cholesky
    factorizion. If the initial Cholesky call fails, the jitter is added to
    the diagonal and the call is attempted again on the modified matrix.

    Args:
        matrix: square 2d array (d, d).
        lower: if True return lower-triangular factor L such that matrix = L @ L.T.
        jitter: initial jitter magnitude to add to diagonal if cholesky fails.
        symmetrize: if True, use (matrix + matrix.T)/2 before factorization.

    Returns:
        Lower- or upper-triangular Cholesky factor, depending on `lower`.

    Raises:
        LinAlgError if factorization fails after several jitter attempts.
    """
    C = jnp.asarray(matrix)
    if symmetrize:
        C = 0.5 * (C + C.T)

    try:
        return cholesky(C, lower=lower)
    except Exception:
        try:
            return cholesky(add_diag_jitter(C, jitter), lower=lower)
        except:
            raise ValueError(f"Cholesky factorization failed, even after adding jitter {jitter}")


@jit
def mahalanobis(
    x: Array|jnp.ndarray,
    mean: Array|jnp.ndarray,
    *,
    cov: Array|None = None,
    precision: Array|None = None,
    chol: Array|None = None,
) -> Array:
    """Compute Mahalanobis distance(s) for vector(s) x relative to a single mean and covariance.

    The Mahalanobis distance between vectors :math:`x` and :math:`m`
    under covariance matrix :math:`C` is defined as:

    .. math::

        D^2(x, m; C) = (x - m)^\\top C^{-1} (x - m)

    If multiple observations are provided as rows of a matrix :math:`X \\in \\mathbb{R}^{n \\times d}`,
    the function computes:

    .. math::

        D_i^2 = (x_i - m)^\\top C^{-1} (x_i - m), \\quad i = 1, \\ldots, n.

    Exactly one of `cov`, `precision`, or `chol` must be provided.

    Args:
        x: array-like, either a 1-D array of shape (d,) or a 2-D array of shape (n, d).
        mean: 1-D array of length d representing the mean vector.
        cov: optional covariance matrix C (d, d).
        precision: optional precision matrix P = C^{-1} (d, d).
        chol: optional lower-triangular Cholesky factor L of covariance C (C = L L^T).

    Returns:
        distances: if x was 1-D, returns a scalar 0-D array; if x was 2-D with shape (n, d),
                   returns array of shape (n,) with distances for each row.
    """
    _validate_inputs(cov, precision, chol)
    X = _as_matrix(x)  # shape (n, d)

    # Mean defaults to zero vector
    if mean is None:
        mean_arr = jnp.zeros(X.shape[1], dtype=X.dtype)
    else:
        mean_arr = _as_flat_vector(mean)

    Xc = X - mean_arr  # shape (n, d)

    if chol is not None:
        L = jnp.asarray(chol)
        distances = _mahalanobis_from_chol_lower(L, Xc)
    elif cov is not None:
        C = jnp.asarray(cov)
        L = _ensure_lower_cholesky_from_cov(C)
        distances = _mahalanobis_from_chol_lower(L, Xc)
    else:  # precision is not None
        P = jnp.asarray(precision)
        distances = _mahalanobis_from_precision(P, Xc)

    # If input was 1-D, return scalar (0-D array); otherwise return (n,)
    return distances[0] if x_arr.ndim == 1 else distances



def _validate_inputs(
    cov: Optional[Array],
    precision: Optional[Array],
    chol: Optional[Array],
) -> None:
    """Validate that exactly one of cov, precision, chol is provided and has correct ndim."""
    provided = sum(int(v is not None) for v in (cov, precision, chol))
    if provided != 1:
        raise ValueError("Exactly one of `cov`, `precision`, or `chol` must be provided.")
    for name, mat in (("cov", cov), ("precision", precision), ("chol", chol)):
        if mat is not None:
            if not isinstance(mat, jnp.ndarray):
                # accept array-like, convert to jnp array later
                pass
            if jnp.asarray(mat).ndim != 2:
                raise ValueError(f"{name} must be a 2-D square matrix.")


def _mahalanobis_from_chol_lower(chol_lower: Array, x_centered: Array) -> Array:
    """Compute Mahalanobis distances given lower-triangular Cholesky L of C (C = L L^T).

    Uses solve_triangular(L, x^T) for all rows at once.

    Args:
        chol_lower: Lower-triangular Cholesky factor L with shape (d, d).
        x_centered: array of shape (n, d) (n rows, each is x - mean).

    Returns:
        distances: array shape (n,) containing Mahalanobis distances.
    """
    # solve L z = (x - mu)^T for multiple RHS at once by passing (d, n) RHS
    # solve_triangular will return shape (d, n)
    rhs = x_centered.T  # shape (d, n)
    solved = solve_triangular(chol_lower, rhs, lower=True)  # shape (d, n)
    # squared norms of columns -> distances
    distances = jnp.sum(solved ** 2, axis=0)  # shape (n,)
    return distances


def _mahalanobis_from_precision(precision: Array, x_centered: Array) -> Array:
    """Compute Mahalanobis distances using the precision matrix P = C^{-1}.

    Computes for rows v: d = v^T P v, using matrix multiplication in a vectorized way.

    Args:
        precision: precision matrix shape (d, d).
        x_centered: array shape (n, d).

    Returns:
        distances: array shape (n,).
    """
    # X P -> shape (n, d). Then rowwise dot with X: sum(X * (X @ P), axis=1)
    xp = x_centered @ precision
    distances = jnp.sum(x_centered * xp, axis=1)
    return distances


def _ensure_lower_cholesky_from_cov(cov: Array) -> Array:
    """Return the lower-triangular Cholesky factor L of the covariance matrix C.

    Args:
        cov: covariance matrix (d, d), symmetric positive-definite.

    Returns:
        L: lower-triangular array such that cov = L @ L.T
    """
    return cholesky(cov, lower=True)


def _ensure_lower_cholesky_from_precision(precision: Array) -> Array:
    """Return lower-triangular Cholesky factor L_p of the precision matrix P.

    That is, precision = L_p @ L_p.T.

    Args:
        precision: precision matrix (d, d), symmetric positive-definite.

    Returns:
        L_p: lower-triangular array such that precision = L_p @ L_p.T
    """
    return cholesky(precision, lower=True)




















# Constants.
LOG_TWO_PI = math.log(2.0 * math.pi)

def mult_A_L(A, L):
    """ blas wrapper for matrix multiply A @ L, where L is lower triangular. """
    return dtrmm(side=1, a=L, b=A, alpha=1.0, trans_a=0, lower=1)

def mult_A_Lt(A, L):
    """ blas wrapper for matrix multiply A @ L.T, where L is lower triangular. """
    return dtrmm(side=1, a=L, b=A, alpha=1.0, trans_a=1, lower=1)

def mult_L_A(A, L):
    """ blas wrapper for matrix multiply L @ A, where L is lower triangular. """
    return dtrmm(side=0, a=L, b=A, alpha=1.0, trans_a=0, lower=1)

def squared_mah_dist(X, m=None, C=None, L=None):
    """ Computes squared Mahalanobis distance using Cholesky factor.
    Computes (x - m)^T C^{-1} (x - m) for each x in X. Returns array of length
    equal to the number of rows of X.
    """
    if L is None:
        L = cholesky(C, lower=True)
    if m is not None:
        X = X - m

    L_inv_X = solve_triangular(L, X.T, lower=True)

    return np.sum(L_inv_X ** 2, axis=0)


def log_det_tri(L):
    """
    Computes log[det(LL^T)], where L is lower triangular.
    """

    return 2 * np.log(np.diag(L)).sum()


def trace_Ainv_B(A_chol, B_chol):
    """
    A_chol, B_chol are lower Cholesky factors of A = A_chol @ A_chol.T,
    B = B_chol @ B_chol.T.

    Computes tr(A^{-1}B) using the Cholesky factors.
    """
    return np.sum(solve_triangular(A_chol, B_chol, lower=True) ** 2)
