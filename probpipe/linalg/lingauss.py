# lingauss.py

import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg.blas import dtrmm

from .types import Array
from .linalg import robust_cholesky, mahalanobis, log_det_tri, trace_Ainv_B

# Constants.
LOG_TWO_PI = math.log(2.0 * math.pi)


def kl_gauss(m0, m1, L0, L1):
    """ Compute Kullback-Leibler divergence between two Gaussians.

    Computes KL(N(m0,C0) || N(m1,C1)), where the Cholesky factors of the
    two covariances are given.

    # TODO: need to add argument checking here.
    """
    d = L0.shape[0]

    term1 = log_det_tri(L1) - log_det_tri(L0)
    term2 = mahalanobis(m0, m1, chol=L1)
    term3 = trace_Ainv_B(L1, L0)

    return 0.5 * (term1 + term2 + term3 - d)


def push_gaussian_affine_cov(m: Array, C: Array,
                             A: Array|None = None,
                             b: Array|None = None) -> Gaussian:

    # If no affine map is specified, defaults to identity map.
    if A is None and b is None:
        return m, C
    elif A is None: # Just shift mean.
        return m+b, C
    elif b is None:
        b = np.zeros(A.shape[0])

    pushforward_mean = b + A @ m
    pushforward_cov = A @ C @ A.T

    return pushforward_mean, pushforward_cov


def push_gaussian_affine_chol(m: Array, L: Array,
                              A: Array|None = None,
                              b: Array|None = None) -> Gaussian:

    # If no affine map is specified, defaults to identity map.
    if A is None and b is None:
        return m, L
    elif A is None: # Just shift mean.
        return m+b, L
    elif b is None:
        b = np.zeros(A.shape[0])

    pushforward_mean = b + A @ m

    # Computing Cholesky of ACA^T = (AL)(AL)^T. AL is a sqrt; turn into
    # Cholesky factor via QR decomposition.
    S = A @ L
    Q, R = qr(S.T, mode="economic")
    # enforce positive diagonal on returned lower-triangular cholesky
    diag = np.sign(np.diag(R))
    diag[diag == 0] = 1.0
    R = np.diag(diag) @ R

    pushforward_chol = R.T

    return pushforward_mean, pushforward_chol


def invert_affine_Gaussian(self, y: Array, A: Array,
                           b: Array|None = None, cov_noise: Array|None = None,
                           chol_noise: np.ndarray|None = None,
                           store: str = "chol") -> Gaussian:
    """
    Solves the inverse problem:
        y = Ax + b + e, e ~ N(0, cov_noise), x ~ N(m, C)
    where N(m, C) is the current Gaussian (self). That is, returns the
    posterior p(x|y), which is itself a Gaussian.
    """
    # TODO: currently performs "data space" update. Should update this
    # to choose whether to perform "data space" vs. "parameter space"
    # update based on dims and which Cholesky factors are provided.

    dim_out = y.shape[0]
    if A.shape[0] != dim_out:
        raise ValueError("`invert_affine_Gaussian()` dimension mismatch ",
                         "between `A` and `y`.")

    # Default to zero shift in the affine map, and identity noise covariance.
    if b is None:
        b = np.zeros(dim_out)
    if cov_noise is None and chol_noise is None:
        cov_noise = np.eye(dim_out)
    if cov_noise is None:
        cov_noise = chol_noise @ chol_noise.T

    # Chokesky factorize ACA^T + cov_noise
    L_prior = self.chol
    B = mult_L_A(mult_A_L(A, L_prior).T, L_prior)
    L_post = cholesky(A @ B + cov_noise, lower=True)

    # Posterior mean.
    r = y.flatten() - b.flatten() - (A @ self.mean).flatten()
    v = solve_triangular(L_post.T, solve_triangular(L_post, r, lower=True), lower=False)
    mean_post = self.mean + B @ v

    # Posterior covariance.
    C = solve_triangular(L_post, B.T, lower=True)
    cov_post = self.cov - C.T @ C

    return Gaussian(mean=mean_post, cov=cov_post, store=store, rng=self.rng)
