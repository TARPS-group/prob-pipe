from typing import Generic, TypeVar, Callable, Any, Optional, Union, Tuple
from numpy.typing import NDArray
import numpy as np
import scipy.stats as sp

from abc import ABC, abstractmethod

from probpipe.distributions.distributions import Distribution
from probpipe.distributions.dist_utils import _as_2d, _symmetrize_spd

T = TypeVar("T", bound=np.number)
Float_T = TypeVar("FloatDT", bound=np.floating)





class Multivariate(Distribution[Float_T], ABC):
    """
    Abstract base for multivariate, real-valued vector distributions with fixed dimension d.
    Event shape is assumed to be (d,). Subclasses should ensure consistency of shapes.
    """

    # ---- Core summary statistics ----
    @abstractmethod
    def mean(self) -> NDArray[Float_T]:
        """
        Return the mean vector μ with shape (d,).
        If the mean does not exist (e.g., Cauchy), raise NotImplementedError.
        """
        raise NotImplementedError

    @abstractmethod
    def cov(self) -> NDArray[np.floating]:
        """
        Return the covariance matrix Σ with shape (d, d).
        If covariance does not exist, raise NotImplementedError.
        """
        raise NotImplementedError

   
    @abstractmethod
    def cdf(self, x: NDArray[Float_T]) -> NDArray[np.floating]:
        """
        Joint CDF F(x) = P[X1 ≤ x1, ..., Xd ≤ xd].
        Accepts x with shape (..., d) and returns shape (...,).
        Implementations may use analytical formulas (rare), numerical integration,
        or library routines when available.
        """
        raise NotImplementedError

    @abstractmethod
    def inv_cdf(self, u: NDArray[np.floating]) -> NDArray[Float_T]:
        """
        Inverse CDF (Rosenblatt inverse) mapping u ∈ (0,1)^d to x ∈ R^d.
        Accepts u with shape (..., d) and returns x with shape (..., d).
        For elliptical families (e.g., MVN), a common implementation is:
          z = Φ^{-1}(u)  (componentwise univariate inverse CDF)
          x = μ + L z     (L is Cholesky factor of Σ)
        For general dependent structures, implement via conditional quantiles/copulas.
        """
        raise NotImplementedError

    # ---- Dimension helper ----
    @property
    def dimension(self) -> int:
        """
        Number of coordinates d. Default infers from mean(). Subclasses may override.
        """
        m = self.mean()
        if m.ndim != 1:
            raise ValueError("mean() must return a 1D array of shape (d,).")
        return int(m.shape[0])




class MvNormal(Multivariate[np.floating]):
    """
    Multivariate Normal N(mean, cov) using scipy.stats.multivariate_normal.

    Shape policy:
      - sample(n) -> (n, d)
      - density(values), log_density(values), cdf(x) -> ALWAYS return (n, 1)
        (scalar-per-sample outputs come back as column vectors)
      - inv_cdf(u) -> (n, d)  (event-shaped output)
    """

    def __init__(self, mean: NDArray[np.floating], cov: NDArray[np.floating],
                 *, rng: np.random.Generator | None = None):
        m = np.asarray(mean, dtype=float)
        C = _symmetrize_spd(np.asarray(cov, dtype=float))
        if m.ndim != 1:
            raise ValueError("mean must be shape (d,)")
        if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] != m.shape[0]:
            raise ValueError("cov must be (d,d) and match mean dimension")
        self._mean = m
        self._cov = C
        self._rng = rng or np.random.default_rng()

        self._mvn_cls = sp.multivariate_normal
        self._mvn = sp.multivariate_normal(mean=self._mean, cov=self._cov, allow_singular=False)

    # ------------------------ Distribution core ------------------------

    def sample(self, n_samples: int) -> NDArray[np.floating]:
        """
        Draw (n, d) samples. Uses Generator via random_state.
        """
        x = self._mvn.rvs(size=int(n_samples), random_state=self._rng)
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:  # when n_samples == 1 SciPy may return (d,)
            x = x.reshape(1, -1)
        return x  # (n, d)

    def density(self, values: NDArray) -> NDArray[np.floating]:
        """
        Return (n,1) column of pdf values for rows in `values`.
        Accepts (d,), (n,d).
        """
        X = _as_2d(values)                   # (n, d)
        p = self._mvn.pdf(X)                 # (n,) from SciPy
        return np.asarray(p, dtype=float).reshape(-1, 1)   # (n,1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """
        Return (n,1) column of log-pdf values for rows in `values`.
        Accepts (d,), (n,d).
        """
        X = _as_2d(values)                   # (n, d)
        lp = self._mvn.logpdf(X)             # (n,)
        return np.asarray(lp, dtype=float).reshape(-1, 1)  # (n,1)

    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray[np.floating]]) -> 'Multivariate':
        """
        Monte-Carlo CLT over f(X):
          - scalar f -> Normal1D(mean, se)
          - vector f -> MvNormal(mean, cov_of_mean)
        The function receives samples as (n_mc, d).
        """
        n_mc = 2048
        xs = self.sample(n_mc)                         # (n, d)
        ys = np.asarray(func(xs), dtype=float)

        # Treat (n,1) as scalar-valued
        if ys.ndim == 2 and ys.shape[1] == 1:
            ys = ys[:, 0]

        if ys.ndim == 1:
            m = float(ys.mean())
            s = float(ys.std(ddof=1)) / np.sqrt(n_mc)
            return Normal1D(m, max(s, 1e-12), rng=self._rng)
        elif ys.ndim == 2:
            m = ys.mean(axis=0)
            cov = np.cov(ys, rowvar=False, ddof=1) / n_mc
            cov = 0.5 * (cov + cov.T) + 1e-12 * np.eye(cov.shape[0])
            return MvNormal(mean=m, cov=cov, rng=self._rng)
        else:
            raise ValueError(f"func must return (n,), (n,1) or (n,k). Got {ys.shape!r}")

    @classmethod
    def from_distribution(cls, convert_from: 'Distribution', **fit_kwargs: Any) -> 'MvNormal':
        """
        Fit mean and covariance from samples drawn from another distribution-like object.
        """
        n = int(fit_kwargs.get("n", 4000))
        xs = np.asarray(convert_from.sample(n), dtype=float)

        X = _as_2d(xs)                                   # (n, d)
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False, ddof=1)
        cov = _symmetrize_spd(cov)
        return cls(mean=mean, cov=cov)


    def mean(self) -> NDArray[np.floating]:
        return self._mean  # (d,)

    def cov(self) -> NDArray[np.floating]:
        return self._cov   # (d,d)

    def cdf(self, x: NDArray) -> NDArray[np.floating]:
        """
        Exact MVN CDF via SciPy. Returns (n,1).
        Accepts (d,), (n,d).
        """
        X = _as_2d(x)                           # (n,d)
        # SciPy's frozen cdf doesn't vectorize over rows, so evaluate row-wise.
        # (For large n you may want to batch this.)
        vals = [float(self._mvn.cdf(row)) for row in X]
        return np.array(vals, dtype=float).reshape(-1, 1)   # (n,1)

    def inv_cdf(self, u: NDArray) -> NDArray[np.floating]:
        """
        Rosenblatt inverse using sequential conditionals and SciPy's univariate Φ^{-1}.
        Input u: (d,) or (n,d). Output: (d,) for 1 sample, or (n,d) for batch.
        """
    
        U_in = np.asarray(u, dtype=float)
        was_1d = (U_in.ndim == 1)
        U = U_in[None, :] if was_1d else U_in   # (n, d)

        n, d = U.shape
        if d != self.dimension:
            raise ValueError(f"u must have shape (..., {self.dimension})")

        out = np.empty((n, d), dtype=float)
        mu = self._mean
        Sigma = self._cov

        for b in range(n):
            x = np.empty(d, dtype=float)
            # 1) marginal of X1
            x[0] = mu[0] + np.sqrt(Sigma[0, 0]) * norm.ppf(U[b, 0])
            # 2..d) sequential conditionals
            for i in range(1, d):
                Sigma_AA = Sigma[:i, :i]
                Sigma_iA = Sigma[i, :i]
                Sigma_Ai = Sigma[:i, i]

                # Solve Σ_AA * w = (x_A - μ_A)
                w = np.linalg.solve(Sigma_AA, (x[:i] - mu[:i]))
                mu_cond = mu[i] + Sigma_iA @ w
                var_cond = Sigma[i, i] - Sigma_iA @ np.linalg.solve(Sigma_AA, Sigma_Ai)
                var_cond = float(max(var_cond, 1e-12))  # numeric guard
                x[i] = mu_cond + np.sqrt(var_cond) * norm.ppf(U[b, i])

            out[b] = x

        return out[0] if was_1d else out

    # ------------------------ helper ------------------------

    @property
    def dimension(self) -> int:
        return int(self._mean.shape[0])




