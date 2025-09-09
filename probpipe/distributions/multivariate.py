from typing import Generic, TypeVar, Callable, Any, Optional, Union, Tuple
from numpy.typing import NDArray
import numpy as np
import pymc as pm
from scipy.stats import norm, multivariate_normal
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scipy.stats import norm

from abc import ABC, abstractmethod


from probpipe.distributions.distributions import Distribution
from probpipe.distributions.continuous import Normal1D
from probpipe.distributions.dist_utils import _as_2d, _symmetrize_spd


T = TypeVar("T",bound=np.number)
#T=float, int, complex
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

        # Precompute consistent factors
        self._L = np.linalg.cholesky(self._cov)           # Σ = L L^T
        # Robust log|Σ|
        self._log_det = float(np.linalg.slogdet(self._cov)[1])

        # (Optional) you can keep precision if you want it elsewhere, but we won’t
        # use it for the quadratic to avoid inconsistencies:
        # self._prec = np.linalg.inv(self._cov)

    def sample(self, n_samples: int) -> np.ndarray:
    # Returns shape (n_samples, d).
        x = self._rng.multivariate_normal(
            mean=self._mean,
            cov=self._cov,
            size=int(n_samples)        # (n_samples, d)
        )
        return x.astype(float)

    def log_density(self, data: NDArray) -> NDArray[np.floating] | float:
        X_in = np.asarray(data, dtype=float)
        was_1d = (X_in.ndim == 1)
        X = X_in.reshape(1, -1) if was_1d else X_in # use your _as_2d if you prefer

        d = self.dimension
        diff = X - self._mean             # (n, d)

        # Mahalanobis via Cholesky solve: solve L y = diff^T  ⇒ quad = sum(y^2) per sample
        # y shape: (d, n)
        y = np.linalg.solve(self._L, diff.T)
        quad = (y * y).sum(axis=0)        # (n,)

        out = -0.5 * (d * np.log(2.0 * np.pi) + self._log_det + quad)
        if was_1d:
            return float(out[0])
        return out.astype(float)

    def density(self, data: NDArray) -> NDArray[np.floating] | float:
        logp = self.log_density(data)
        p = np.exp(logp)
        return float(p) if not isinstance(p, np.ndarray) else p.astype(float)

    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray]) -> 'Distribution':
        # Monte-Carlo CLT for vector/scalar functions of X in R^d
        n_mc = 2048
        xs = self.sample(n_mc)            # (n, d)
        ys = np.asarray(func(xs), dtype=float)
        if ys.ndim == 1:
            m = float(ys.mean())
            s = float(ys.std(ddof=1)) / np.sqrt(n_mc)
            s = max(s, 1e-12)
            return Normal1D(m, s, rng=self._rng)
        else:
            ys2 = _as_2d(ys)              # (n, k)
            m = ys2.mean(axis=0)
            cov = np.cov(ys2, rowvar=False, ddof=1) / n_mc
            cov = _symmetrize_spd(cov)
            return MvNormal(mean=m, cov=cov, rng=self._rng)

    @classmethod
    def from_distribution(cls, convert_from: 'Distribution', **fit_kwargs: Any) -> 'MvNormal':
        n = int(fit_kwargs.get("n", 4000))
        try:
            xs = np.asarray(convert_from.sample(n), dtype=float)
        except NotImplementedError:
            raise NotImplementedError("from_distribution requires convert_from.sample to be implemented")
        xs = _as_2d(xs)  # (n, d)
        mean = xs.mean(axis=0)
        cov = np.cov(xs, rowvar=False, ddof=1)
        cov = _symmetrize_spd(cov)
        return cls(mean=mean, cov=cov)

    # ----- Multivariate requirements -----

    def mean(self) -> NDArray[np.floating]:
        return self._mean

    def cov(self) -> NDArray[np.floating]:
        return self._cov

    def cdf(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Exact MVN CDF requires numerical integration.
        If SciPy is available, we use it; otherwise we raise NotImplementedError.
        """
    
        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            return np.array([multivariate_normal(mean=self._mean, cov=self._cov).cdf(X)], dtype=float)
        else:
            mvn = multivariate_normal(mean=self._mean, cov=self._cov)
            return np.array([mvn.cdf(row) for row in X], dtype=float)

    def inv_cdf(self, u):
        """
        Rosenblatt inverse using sequential univariate conditionals for MVN.
        Returns shape (d,) for 1-D input u (d,), and shape (n, d) for batched input (n, d).
        Requires SciPy for Φ^{-1}.
        """

        U_in = np.asarray(u, dtype=float)
        was_1d = (U_in.ndim == 1)
        U = U_in[None, :] if was_1d else U_in  # (n, d)

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
    

TINY = np.finfo(float).tiny
TAU = 2.0 * np.pi

class GaussianKDE(Multivariate[np.floating]):
    """
    Gaussian kernel density estimator with shared bandwidth matrix H.

    Parameters
    ----------
    samples : array-like, shape (n, d) or (n,)
        Points x_i where kernels are centered.
    weights : array-like, shape (n,), optional
        Nonnegative weights for each center; normalized to sum to 1.
    bandwidth : float | array(d,) | array(d,d) | None
        - float: scalar 'h' => H = h^2 * I
        - array(d,): per-dimension stds 'h_j' => H = diag(h_j^2)
        - array(d,d): full SPD matrix interpreted as H directly
        - None: use Scott/Silverman rule on the data covariance (diagonalized)
    rule : {'scott', 'silverman'}, default 'scott'
        Automatic rule when bandwidth is None.
    rng : np.random.Generator, optional
        RNG for sampling.
    cdf_mode : {'auto','mixture','mc'}, default 'auto'
        - 'mixture': sum_i w_i * MVN(μ=x_i, Σ=H).cdf(x) (requires SciPy)
        - 'mc': Monte-Carlo approximation via samples from KDE
        - 'auto': try 'mixture', fall back to 'mc' if SciPy missing
    cdf_mc_samples : int, default 20000
        MC budget for CDF approximation when using 'mc'.

    Notes
    -----
    - pdf/logpdf are exact for the Gaussian mixture with shared Σ=H.
    - mean = Σ_i w_i x_i ; cov = Cov_w(X) + H.
    - inv_cdf is not implemented (no simple Rosenblatt inverse for mixtures).
    """

    def __init__(
        self,
        samples: NDArray[np.floating],
        weights: Optional[NDArray[np.floating]] = None,
        *,
        bandwidth: float | NDArray[np.floating] | None = None,
        rule: str = "scott",
        rng: Optional[np.random.Generator] = None,
        cdf_mode: str = "auto",
        cdf_mc_samples: int = 20_000,
    ):
        X = _as_2d(samples)
        n, d = X.shape
        if n < 1:
            raise ValueError("GaussianKDE requires at least one sample.")
        # weights
        if weights is None:
            w = np.full(n, 1.0 / n, dtype=float)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] != n:
                raise ValueError("weights must have shape (n,).")
            if np.any(w < 0):
                raise ValueError("weights must be nonnegative.")
            s = w.sum()
            if s <= 0:
                raise ValueError("weights must sum to a positive value.")
            w = w / s

        self._X = X.astype(float)
        self._w = w.astype(float)
        self._n, self._d = n, d
        self._rng = rng or np.random.default_rng()
        self._cdf_mode = cdf_mode
        self._cdf_mc_samples = int(cdf_mc_samples)

        # data mean & (population) covariance under weights
        self._mean = (self._w[:, None] * self._X).sum(axis=0)
        diff = self._X - self._mean
        self._cov_x = diff.T @ (diff * self._w[:, None])  # (d, d)

        # bandwidth matrix H
        H = self._build_H(bandwidth, rule)
        self._H = _symmetrize_spd(H)
        self._L = np.linalg.cholesky(self._H)  # H = L L^T
        self._log_det_H = float(np.linalg.slogdet(self._H)[1])
        self._log_norm = -0.5 * (self._d * np.log(TAU) + self._log_det_H)
        self._inv_by_solve = True  # use solves via L instead of explicit H^{-1}

        # mixture cov = Cov(X) + H
        self._cov_mix = self._cov_x + self._H

    ...
    
