# linalg.py

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from .types import Array

def as_matrix(x: Array) -> Array:
    """Ensure input is a 2D array, potentially via coercing.

    0d array is converted to shape (1,1), 1d to (n,), and 2d is unchanged.
    Higher dimensional array input results in error.
    """
    x = np.asarray(x)

    if x.ndim == 0:
        return x.reshape((1, 1))
    if x.ndim == 1:
        return x[jnp.newaxis, :]
    if x.ndim == 2:
        return x
    raise ValueError("x must be 0-D, 1-D, or 2-D array.")


def as_flat_vector(x: Array) -> Array:
    """Ensure input is a 1d array, potentially via coercing."""
    x = np.asarray(x)

    if x.ndim == 0:
        return x.reshape((1,))
    if x.ndim == 1:
        return x
    if x.ndim == 2 and (x.shape[0] == 1 or x.shape[1] == 1):
        return x.ravel()

    raise ValueError("x must be 0-D, 1-D array, or a 2d row/column vector.")


def robust_cholesky(
    matrix: Array,
    *,
    lower: bool = True,
    jitter: float = 1e-6,
    symmetrize: bool = False,
) -> Array:
    """Return robust lower-triangular Cholesky factor of covariance matrix.

    This function is a wrapper around `scipy.linalg.cholesky(matrix, lower=lower)`
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
    C = np.asarray(matrix)
    if symmetrize:
        C = 0.5 * (C + C.T)

    try:
        L = cholesky(C, lower=lower)
        return L
    except:
        try:
            L = cholesky(add_diag_jitter(C, jitter), lower=lower)
            return L
        except Exception as e:
            raise ValueError(
                f"Cholesky factorization failed, even after adding jitter {jitter}. Error: {e}"
            )


def mahalanobis(
    x: Array,
    mean: Array,
    *,
    cov: Array|None = None,
    chol: Array|None = None,
    precision: Array|None = None
) -> Array:
    """Compute squared Mahalanobis distance(s) for vector(s) x relative to a single mean and covariance.

    The squared Mahalanobis distance between vectors :math:`x` and :math:`m`
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
    _validate_covariance_arguments(cov, precision, chol)
    X = as_matrix(x)  # shape (n, d)

    # Mean defaults to zero vector
    if mean is None:
        mean_arr = np.zeros(X.shape[1], dtype=X.dtype)
    else:
        mean_arr = as_flat_vector(mean)

    Xc = X - mean_arr  # shape (n, d)

    if chol is not None:
        L = np.asarray(chol)
        return _mahalanobis_from_chol_lower(L, Xc)
    elif cov is not None:
        L = robust_cholesky(cov, lower=True)
        return _mahalanobis_from_chol_lower(L, Xc)
    else:  # precision is not None
        P = np.asarray(precision)
        return _mahalanobis_from_precision(P, Xc)


def log_det_tri(L):
    """ Compute log determinant of positive definite matrix using lower triangular square root.

    Given lower triangular matrix :math:`L \\in \\mathbb{R}^{d \\times d}`,
    this function computes the log of the determinant of the matrix :math`L L^\top`.
    """

    L = np.asarray(L)
    return 2 * np.log(np.diag(L)).sum()


def trace_Ainv_B(A_chol, B_chol):
    """ Computes trace of product of matrix inverse times another matrix using Cholesky factors.

    A_chol, B_chol are lower Cholesky factors of A = A_chol @ A_chol.T,
    B = B_chol @ B_chol.T. Computes tr(A^{-1}B) using the Cholesky factors.
    """
    return np.sum(solve_triangular(A_chol, B_chol, lower=True) ** 2)


def _validate_covariance_arguments(cov: Array|None,
                                   chol: Array|None,
                                   precision: Array|None) -> None:
    """Validate that exactly one of cov, precision, chol is provided and has correct ndim.

    TODO: should probably also validate that the provided arg is a square matrix.
    """

    num_args_provided = sum(int(v is not None) for v in (cov, chol, precision))

    if num_args_provided != 1:
        raise ValueError("Exactly one of `cov`, `chol`, or `precision` must be provided.")


def _mahalanobis_from_chol_lower(chol_lower: Array, x_centered: Array) -> Array:
    """ Compute Mahalanobis distances given Cholesky factor.

    This is a helper function to `mahalanobis` which computes the squared
    Mahalanobis distances using the lower Cholesky factor of the covariance
    matrix.

    Args:
        chol_lower: lower-triangular Cholesky factor L with shape (d, d).
        x_centered: array of shape (n, d) (n rows, each is x - mean).

    Returns:
        distances: array shape (n,) containing Mahalanobis distances.
    """
    # solve L z = x_centered^T for z
    L_inv_x = solve_triangular(chol_lower, x_centered.T, lower=True)  # shape (d, n)

    return jnp.sum(L_inv_x ** 2, axis=0)  # shape (n,)


def _mahalanobis_from_precision(precision: Array, x_centered: Array) -> Array:
    """Compute Mahalanobis distances given precision matrix.

    This is a helper function to `mahalanobis` which computes the squared
    Mahalanobis distances using the inverse covariance (precision) matrix.

    Args:
        precision: precision matrix, array of shape (d, d).
        x_centered: array of shape (n, d) (n rows, each is x - mean).

    Returns:
        distances: array shape (n,).
    """
    # X P -> shape (n, d). Then rowwise dot with X: sum(X * (X @ P), axis=1)
    xp = x_centered @ precision

    return np.sum(x_centered * xp, axis=1)
