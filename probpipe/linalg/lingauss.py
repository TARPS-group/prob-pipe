import numpy as np
from scipy.linalg.blas import dtrmm
from scipy.linalg import cholesky, qr, solve_triangular

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


def kl_gauss(m0, m1, C0=None, C1=None, L0=None, L1=None):
    """
    Compute KL(N(m0,C0) || N(m1,C1))
    """
    if L0 is None:
        L0 = cholesky(C0, lower=True)
    if L1 is None:
        L1 = cholesky(C1, lower=True)

    d = L0.shape[0]

    term1 = log_det_tri(L1) - log_det_tri(L0)
    term2 = squared_mah_dist(m0, m1, L=L1)
    term3 = trace_Ainv_B(L1, L0)

    return 0.5 * (term1 + term2 + term3 - d)


class LinearGaussian(Module):

    def __init__(self):
        super().__init__()

    def apply_affine_map(x, A: np.ndarray|None = None, b: np.ndarray|None = None,
                         store: str = "chol") -> Gaussian:
        """
        Returns the Gaussian resulting by pushing x ~ N(m, C)
        through an affine map F(x) = Ax + b. The result is
        y := F(x) ~ N(Am + b, ACA^T).
        """
        # If no affine map is specified, defaults to identity map.
        if A is None and b is None:
            return copy.deepcopy(self)
        elif A is None: # Just shift mean.
            y = copy.deepcopy(self)
            y.mean += b
            return y
        elif b is None:
            b = np.zeros(A.shape[0])

        # Computing Cholesky of ACA^T = (AL)(AL)^T. AL is a sqrt; turn into
        # Cholesky factor via QR decomposition.
        S = mult_A_L(A, self.chol)
        if store == "chol":
            Q, R = qr(S.T, mode="economic")
            chol = R.T
            cov = None
        else:
            cov = S @ S.T
            chol = None

        return Gaussian(mean= A @ self.mean + b,
                        cov=cov, chol=chol, store=store, rng=self.rng)
