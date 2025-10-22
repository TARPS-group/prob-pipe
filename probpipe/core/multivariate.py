from typing import TypeVar, Callable, Any, Optional
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

import scipy.stats as sp
from scipy.stats import norm, multivariate_normal
from scipy.stats import multinomial as _multinomial
from scipy.stats import dirichlet as _dirichlet
from scipy.stats import binom as _sbinom
from scipy.stats import beta as _sps_beta

from ._utils import _as_2d, _symmetrize_spd, _clip_unit_interval, _to_1d_vector
from .distributions import Distribution

__all__ = [
    "Multivariate",
    "Normal1D",
    "MvNormal",
    "GaussianKDE",
    "Multinomial",
    "Dirichlet",
    "Binomial",
    "Beta",
]

T = TypeVar("T", bound=np.number)
Float_T = TypeVar("FloatDT", bound=np.floating)

class Multivariate(Distribution[Float_T], ABC):
    """
    Abstract base for multivariate, real-valued vector distributions with fixed dimension d.
    Event shape is assumed to be (d,). Subclasses should ensure consistency of shapes.
    """

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



class Normal1D(Multivariate[np.floating]):
    """
    Univariate Normal N(mu, sigma^2) as a Multivariate with event shape (1,).

    Shape policy:
      - sample(n) -> (n, 1)
      - density/log_density/cdf(values): ALWAYS return (n, 1), even though they are
        scalar-per-sample functions (reduced over the event axis).
      - inv_cdf(u): returns (n, 1).
    """

    def __init__(self, mu: float, sigma: float, *, rng: np.random.Generator | None = None):
        """Initialize a Normal1D distribution with mean 'mu' and std. dev. 'sigma'."""
        
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self._rng = rng or np.random.default_rng()

        self._norm = norm(loc=self.mu, scale=self.sigma)


    def sample(self, n_samples: int) -> NDArray[np.floating]:
        """Draw `n_samples` from N(mu, sigma²); returns array of shape (n, 1)."""
        
        xs = self._norm.rvs(size=(int(n_samples), 1), random_state=self._rng)
        return np.asarray(xs, dtype=float)  # (n, 1)


    def density(self, values: NDArray) -> NDArray[np.floating]:
        """Return PDF values at `values`, as column vector (n, 1)."""
        
        v = self._to_1d_vector(values)          # (n,)
        p = np.asarray(self._norm.pdf(v), dtype=float).reshape(-1, 1)
        return p                                 # (n, 1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """Return log-PDF values at `values`, as column vector (n, 1)."""
        
        v = _to_1d_vector(values)          # (n,)
        lp = np.asarray(self._norm.logpdf(v), dtype=float).reshape(-1, 1)
        return lp                                # (n, 1)

    def cdf(self, x: NDArray) -> NDArray[np.floating]:
        """Return CDF evaluated at `x`, as column vector (n, 1)."""
        
        v = self._to_1d_vector(x)               # (n,)
        c = np.asarray(self._norm.cdf(v), dtype=float).reshape(-1, 1)
        return c                                 # (n, 1)

    def inv_cdf(self, u: NDArray) -> NDArray[np.floating]:
        """Return quantiles (inverse CDF) for probabilities `u`; output shape (n, 1)."""
        
        U = np.asarray(u, dtype=float)
        if U.ndim == 0:
            q = float(self._norm.ppf(U))
            return np.array([[q]], dtype=float)
        if U.ndim == 1:
            q = self._norm.ppf(U)
            return np.asarray(q, dtype=float).reshape(-1, 1)
        if U.ndim == 2 and U.shape[1] == 1:
            q = self._norm.ppf(U[:, 0])
            return np.asarray(q, dtype=float).reshape(-1, 1)
        raise ValueError("u must be scalar, (n,) or (n,1).")


    def mean(self) -> NDArray[np.floating]:
        """Return the mean vector (1,)."""
        
        return np.array([self.mu], dtype=float)          # (1,)

    def cov(self) -> NDArray[np.floating]:
        """Return the covariance matrix (1×1) = [[sigma²]]."""
        
        return np.array([[self.sigma ** 2]], dtype=float)  # (1,1)


    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray[np.floating]]) -> 'Multivariate':
        """
        Monte-Carlo CLT over f(X):
          - scalar f -> Normal1D(mean, se)
          - vector f -> MvNormal(mean, cov_of_mean)
        func receives samples with shape (n_mc, 1).
        """
        
        n_mc = 2048
        xs = self.sample(n_mc)                   # (n_mc, 1)
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
    def from_distribution(cls, convert_from: 'Distribution', num_samples: int=1024, *, conversion_by_KDE: bool = False,  **fit_kwargs: Any) -> 'Normal1D': 
        """Fit a Normal1D to samples drawn from another Distribution (moment matching)."""
        
        #n = int(fit_kwargs.get("n", 2000))
        xs = np.asarray(convert_from.sample(num_samples), dtype=float)

        # Flatten (n,1) → (n,), accept (n,)
        if xs.ndim == 2 and xs.shape[1] == 1:
            xs = xs[:, 0]
        elif xs.ndim != 1:
            xs = xs.reshape(-1)

        mu = float(xs.mean())
        sigma = float(xs.std(ddof=1))
        return cls(mu, max(sigma, 1e-12))




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
         """Initialize a multivariate Normal with mean vector and covariance matrix."""
                     
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
    def from_distribution(cls, convert_from: 'Distribution', num_samples: int=1024, *, conversion_by_KDE: bool = False, **fit_kwargs: Any) -> 'MvNormal':
        """
        Fit mean and covariance from samples drawn from another distribution-like object.
        """
        #n = int(fit_kwargs.get("n", 4000))
        xs = np.asarray(convert_from.sample(num_samples), dtype=float)

        X = _as_2d(xs)                                   # (n, d)
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False, ddof=1)
        cov = _symmetrize_spd(cov)
        return cls(mean=mean, cov=cov)


    def mean(self) -> NDArray[np.floating]:
        """Return the mean vector (d,)."""
        
        return self._mean  # (d,)

    def cov(self) -> NDArray[np.floating]:
        """Return the covariance matrix (d, d)."""
        
        return self._cov   # (d,d)

    def cdf(self, x: NDArray) -> NDArray[np.floating]:
        """Evaluate the multivariate CDF at x; returns (n, 1)."""

        X = _as_2d(x)                           # (n,d)
        # SciPy's frozen cdf doesn't vectorize over rows, so evaluate row-wise.
        # (For large n you may want to batch this.)
        vals = [float(self._mvn.cdf(row)) for row in X]
        return np.array(vals, dtype=float).reshape(-1, 1)   # (n,1)

    def inv_cdf(self, u: NDArray) -> NDArray[np.floating]:
        """Compute Rosenblatt inverse transform for uniform samples u; returns (n, d)."""

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
        """Return the dimensionality (d) of the multivariate Normal."""
        return int(self._mean.shape[0])



class GaussianKDE(Multivariate[np.floating]):
    """
    Gaussian kernel density estimator with shared bandwidth matrix H.

    Each sample x_i defines a kernel N(x_i, H), and the overall density is a
    weighted mixture over these kernels.

    Shape policy
    -------------
    sample(n) → (n, d)
    density / log_density / cdf(x) → (n, 1)
    inv_cdf(u) → Not implemented
    """

    def __init__(
        self,
        samples: NDArray[np.floating],
        weights: Optional[NDArray[np.floating]] = None,
        *,
        bandwidth: float | NDArray[np.floating] | None = None,
        rule: str = "scott",                 # used when bandwidth is None
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize KDE with centers, weights, and optional bandwidth or rule."""

        X = _as_2d(samples)
        n, d = X.shape
        if n < 1:
            raise ValueError("GaussianKDE requires at least one sample.")

        if weights is None:
            w = np.full(n, 1.0 / n, dtype=float)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] != n:
                raise ValueError("weights must have shape (n,).")
            if np.any(w < 0):
                raise ValueError("weights must be nonnegative.")
            s = float(w.sum())
            if s <= 0:
                raise ValueError("weights must sum to a positive value.")
            w = w / s

        self._X = X.astype(float)     # (n, d) centers
        self._w = w.astype(float)     # (n,)
        self._n = n
        self._d = d
        self._rng = rng or np.random.default_rng()

        # Weighted mean & population covariance of centers
        self._mean = (self._w[:, None] * self._X).sum(axis=0)  # (d,)
        diff = self._X - self._mean
        self._cov_x = diff.T @ (diff * self._w[:, None])       # (d, d)

        # Build bandwidth matrix H
        self._H = self._build_H(bandwidth, rule)
        self._H = _symmetrize_spd(self._H)

        # Pre-create SciPy kernels for each center (mean = x_i, cov = H)
        self._kernel_cls = multivariate_normal
        self._kernels = [multivariate_normal(mean=self._X[i], cov=self._H, allow_singular=False)
                         for i in range(self._n)]

        # Precompute mixture covariance = Cov(X) + H
        self._cov_mix = self._cov_x + self._H

    # --------------------------- bandwidth helpers ---------------------------

    def _build_H(self, bandwidth: float | NDArray[np.floating] | None, rule: str) -> NDArray[np.floating]:
        """Construct the bandwidth matrix H from user input or from 'scott'/'silverman' rule."""
        
        d = self._d
        if bandwidth is not None:
            bw = np.asarray(bandwidth, dtype=float)
            if bw.ndim == 0:
                h = float(bw)
                if h <= 0:
                    raise ValueError("bandwidth scalar must be > 0.")
                return (h * h) * np.eye(d, dtype=float)
            elif bw.ndim == 1:
                if bw.shape[0] != d:
                    raise ValueError("bandwidth vector must have shape (d,).")
                if np.any(bw <= 0):
                    raise ValueError("bandwidth vector entries must be > 0.")
                return np.diag(bw * bw)
            elif bw.ndim == 2:
                if bw.shape != (d, d):
                    raise ValueError("bandwidth matrix must be (d, d).")
                return bw
            else:
                raise ValueError("Unsupported bandwidth shape.")

        # Automatic rule on weighted covariance (like scipy.stats.gaussian_kde)
        if rule.lower() == "scott":
            factor = self._n ** (-1.0 / (d + 4.0))
        elif rule.lower() == "silverman":
            factor = (self._n * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))
        else:
            raise ValueError("rule must be 'scott' or 'silverman'.")
        return (factor ** 2) * self._cov_x

    # --------------------------- Multivariate API ---------------------------

    def sample(self, n_samples: int) -> NDArray[np.floating]:
        """Draw samples from the KDE mixture by selecting centers and adding Gaussian noise."""

        n = int(n_samples)
        idx = self._rng.choice(self._n, size=n, replace=True, p=self._w)
        noise = multivariate_normal(mean=np.zeros(self._d, dtype=float), cov=self._H).rvs(
            size=n, random_state=self._rng
        )
        noise = np.asarray(noise, dtype=float)
        if noise.ndim == 1:  # n == 1
            noise = noise.reshape(1, -1)
        return self._X[idx] + noise  # (n, d)

    def density(self, values: NDArray) -> NDArray[np.floating]:
        """
        Mixture pdf: sum_i w_i * N(values | mean=x_i, cov=H).
        Returns (n, 1).
        """
        Xq = _as_2d(values)  # (n, d)
        out = np.empty(Xq.shape[0], dtype=float)
        for j, q in enumerate(Xq):
            # Weighted sum of kernel PDFs at q
            out[j] = float(np.dot(self._w, [k.pdf(q) for k in self._kernels]))
        return out.reshape(-1, 1)  # (n,1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """
        Stable log pdf via log-sum-exp over kernels: log ∑ w_i exp(log N_i(q)).
        Returns (n, 1).
        """
        Xq = _as_2d(values)  # (n, d)
        out = np.empty(Xq.shape[0], dtype=float)
        for j, q in enumerate(Xq):
            logs = np.array([np.log(self._w[i] + 1e-300) + k.logpdf(q) for i, k in enumerate(self._kernels)],
                            dtype=float)
            m = logs.max()
            out[j] = m + np.log(np.exp(logs - m).sum())
        return out.reshape(-1, 1)

    def mean(self) -> NDArray[np.floating]:
        """Mixture mean = weighted mean of centers."""
        return self._mean  # (d,)

    def cov(self) -> NDArray[np.floating]:
        """Mixture covariance = Cov_w(centers) + H."""
        return self._cov_mix  # (d,d)

    def cdf(self, values: NDArray) -> NDArray[np.floating]:
        """
        Mixture CDF via SciPy MVN CDF per-kernel:
          F(q) = ∑_i w_i * Φ_d(q; mean=x_i, cov=H).
        Returns (n, 1).
        """
        Xq = _as_2d(values)  # (n, d)
        out = np.empty(Xq.shape[0], dtype=float)
        for j, q in enumerate(Xq):
            out[j] = float(np.dot(self._w, [k.cdf(q) for k in self._kernels]))
        return out.reshape(-1, 1)

    def inv_cdf(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        No simple Rosenblatt inverse for Gaussian mixtures.
        """
        raise NotImplementedError("GaussianKDE.inv_cdf is not available for kernel mixtures.")

    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray]) -> 'Multivariate':
        """
        Monte-Carlo expectation under the KDE. Scalar f -> Normal1D over the mean,
        vector f -> MvNormal over the mean. `func` receives samples as (n_mc, d).
        """
        n_mc = 2048
        xs = self.sample(n_mc)                  # (n, d)
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
    def from_distribution(cls, convert_from: 'Distribution', num_samples: int=1024, *, conversion_by_KDE: bool = False, **fit_kwargs: Any) -> 'GaussianKDE':
        """
        Build KDE from another distribution-like object.

        kwargs:
          - n: number of samples to draw if only .sample(n) is available (default 2000)
          - weights: optional weights for the given/provided samples
          - bandwidth: scalar | (d,) | (d,d)  (optional)
          - rule: 'scott' or 'silverman' when bandwidth=None (default 'scott')
          - rng: np.random.Generator to store in the resulting KDE
        """
        # = int(fit_kwargs.get("n", 2000))
        bandwidth = fit_kwargs.get("bandwidth", None)
        rule = fit_kwargs.get("rule", "scott")
        rng = fit_kwargs.get("rng", None)
        weights = fit_kwargs.get("weights", None)

        # Try to use stored samples directly (Empirical-like)
        if hasattr(convert_from, "samples"):
            X = np.asarray(convert_from.samples, dtype=float)
            w = getattr(convert_from, "weights", None)
            if weights is not None:
                w = np.asarray(weights, dtype=float)
            return cls(samples=X, weights=w, bandwidth=bandwidth, rule=rule, rng=rng)

        # Otherwise, sample from the source distribution
        X = np.asarray(convert_from.sample(num_samples), dtype=float)

        return cls(samples=X, weights=weights, bandwidth=bandwidth, rule=rule, rng=rng)


class Multinomial(Multivariate[np.floating]):
    """
    Multinomial(n_trials, p) with event dim d = len(p).
    Backed by scipy.stats.multinomial.

    Shape policy:
      - sample(n) -> (n, d)  (counts per category)
      - density(values), log_density(values), cdf(values) -> (n, 1)
        (scalar-per-sample outputs come back as column vectors)
      - inv_cdf(u) -> NotImplementedError (no simple inverse for multinomial)

    Notes
    -----
    * Counts are integers but returned as float arrays to satisfy Multivariate[Float] typing.
      Cast to int if you need integer dtype.
    * `cdf` is estimated by Monte-Carlo (parameter `cdf_mc_samples`); exact CDF is combinatorial.
    """

    def __init__(
        self,
        n_trials: int,
        probs: NDArray[np.floating],
        *,
        rng: Optional[np.random.Generator] = None,
        cdf_mode: str = "mc",
        cdf_mc_samples: int = 20000,
    ):

        n = int(n_trials)
        if n < 0:
            raise ValueError("n_trials must be >= 0.")

        p = np.asarray(probs, dtype=float).reshape(-1)
        if p.ndim != 1 or p.size == 0:
            raise ValueError("probs must be a 1D non-empty array.")
        if np.any(p < 0):
            raise ValueError("probs must be nonnegative.")
        s = float(p.sum())
        if not np.isfinite(s) or s <= 0:
            raise ValueError("probs must sum to a positive finite value.")
        p = p / s  # normalize exactly

        self._n = n
        self._p = p
        self._d = int(p.size)
        self._rng = rng or np.random.default_rng()
        self._cdf_mode = cdf_mode
        self._cdf_mc_samples = int(cdf_mc_samples)
        
        self._mn = _multinomial(n=self._n, p=self._p)

        # Precompute mean & covariance (population)
        self._mean = self._n * self._p                                # (d,)
        self._cov = self._n * (np.diag(self._p) - np.outer(self._p, self._p))
        self._cov = _symmetrize_spd(self._cov)

    # ------------------------ Multivariate core ------------------------

    def sample(self, n_samples: int) -> NDArray[np.floating]:
        """
        Draw (n, d) samples of counts; cast to float for typing consistency.
        """
        X = self._mn.rvs(size=int(n_samples), random_state=self._rng)  # (n,d) or (d,) if n=1
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X  # (n, d) float

    def density(self, values: NDArray) -> NDArray[np.floating]:
        """
        Return (n,1) column of pmf at each row in `values`.
        Inputs may be float; they must represent integer counts that sum to n_trials.
        """
        X = _as_2d(values)  # (n,d)
        self._validate_counts_rows(X)
        # SciPy's logpmf/pmf can handle arrays row-wise via list comprehension
        pmfs = [self._mn.pmf(row.astype(int, copy=False)) for row in X]
        return np.asarray(pmfs, dtype=float).reshape(-1, 1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """
        Return (n,1) column of log-pmf at each row in `values`.
        """
        X = _as_2d(values)  # (n,d)
        self._validate_counts_rows(X)
        logpmfs = [self._mn.logpmf(row.astype(int, copy=False)) for row in X]
        return np.asarray(logpmfs, dtype=float).reshape(-1, 1)

    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray]) -> 'Multivariate':
        """
        Monte-Carlo CLT over f(X) with X ~ Multinomial.
          - scalar f -> Normal1D(mean, se)
          - vector f -> MvNormal(mean, cov_of_mean)
        The function receives samples with shape (n_mc, d).
        """
        n_mc = 2048
        xs = self.sample(n_mc)                    # (n, d)
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
    def from_distribution(cls, convert_from: 'Distribution', num_samples: int=1024, *, conversion_by_KDE: bool = False, **fit_kwargs: Any) -> 'Multinomial':
        """
        Fit (n_trials, p) from counts sampled from `convert_from`.
        Assumes each row of samples is a vector of counts whose sum is constant (n_trials).
        """
        #n_fit = int(fit_kwargs.get("n", 4000))
        
        X = np.asarray(convert_from.sample(num_samples), dtype=float)

        X = _as_2d(X)  # (n, d)
        # infer n_trials from row sums (must be constant up to rounding)
        row_sums = X.sum(axis=1)
        n_trials = int(np.round(np.median(row_sums)))
        if not np.allclose(row_sums, n_trials, atol=1e-8):
            raise ValueError("from_distribution expects count vectors with a constant row sum (n_trials).")
        mean_counts = X.mean(axis=0)
        p_hat = mean_counts / max(n_trials, 1)
        # cleanup numeric drift
        p_hat = np.clip(p_hat, 0.0, None)
        s = float(p_hat.sum())
        if s == 0:
            # degenerate (all zero), fall back to uniform
            p_hat = np.full(X.shape[1], 1.0 / X.shape[1], dtype=float)
        else:
            p_hat /= s
        return cls(n_trials=n_trials, probs=p_hat, rng=fit_kwargs.get("rng", None))


    def mean(self) -> NDArray[np.floating]:
        """Return the mean vector (d,)."""
        return self._mean  # (d,)

    def cov(self) -> NDArray[np.floating]:
         """Return the covariance matrix (d, d)."""
        return self._cov   # (d,d)

    def cdf(self, values: NDArray) -> NDArray[np.floating]:
        """
        Monte-Carlo approximation of P[X1<=x1, ..., Xd<=xd].
        Returns (n,1). For exact CDF (small n,d), implement a specialized enumerator.
        """
        if self._cdf_mode != "mc":
            raise NotImplementedError("Only Monte-Carlo CDF ('mc') is supported for Multinomial.")
        Xq = _as_2d(values)  # (n, d)
        # floor thresholds to nearest integer counts
        T = np.floor(Xq + 1e-12).astype(int)
        # MC draws
        m = self._cdf_mc_samples
        draws = self._mn.rvs(size=m, random_state=self._rng)  # (m,d) or (d,)
        draws = np.asarray(draws, dtype=int)
        if draws.ndim == 1:
            draws = draws.reshape(1, -1)
        out = np.empty(Xq.shape[0], dtype=float)
        for j, t in enumerate(T):
            out[j] = np.mean((draws <= t).all(axis=1))
        return out.reshape(-1, 1)

    def inv_cdf(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        No practical Rosenblatt inverse for a general Multinomial.
        """
        raise NotImplementedError("Multinomial.inv_cdf is not available.")

    # ------------------------ helper ------------------------

    def _validate_counts_rows(self, X: NDArray[np.floating]) -> None:
        """
        Ensure rows of X represent valid count vectors for this multinomial:
          - integer-valued (within tiny tolerance)
          - nonnegative
          - sum equals n_trials
        Raises ValueError on violation.
        """
        if X.shape[1] != self._d:
            raise ValueError(f"values must have shape (n, {self._d}) or ({self._d},).")
        # check (near-)integers
        if not np.allclose(X, np.round(X), atol=1e-8):
            raise ValueError("values must be integer counts for Multinomial.")
        Xi = np.round(X).astype(int)
        if np.any(Xi < 0):
            raise ValueError("counts must be nonnegative.")
        rs = Xi.sum(axis=1)
        if not np.all(rs == self._n):
            raise ValueError(f"each row must sum to n_trials={self._n}.")


class Dirichlet(Multivariate[np.floating]):
    """
    Dirichlet(α) on the probability simplex.

    SciPy-backed:
      - sampling via scipy.stats.dirichlet.rvs
      - pdf/logpdf via scipy.stats.dirichlet.{pdf,logpdf}

    Shape policy:
      - sample(n) -> (n, d)
      - density(values), log_density(values), cdf(values) -> (n, 1)
      - inv_cdf(u) -> NotImplementedError

    Parameters
    ----------
    alpha : array-like of shape (d,), α_i > 0
    rng : np.random.Generator, optional
    cdf_mode : {'mc'}, default 'mc'
        CDF is approximated by Monte Carlo under the Dirichlet law.
    cdf_mc_samples : int, default 20000
        Number of MC draws for CDF approximation.
    """

    def __init__(
        self,
        alpha: NDArray[np.floating] | NDArray[np.floating],
        *,
        rng: Optional[np.random.Generator] = None,
        cdf_mode: str = "mc",
        cdf_mc_samples: int = 20000,
    ):

        a = np.asarray(alpha, dtype=float).reshape(-1)
        if a.ndim != 1 or a.size == 0:
            raise ValueError("alpha must be a 1D, non-empty array.")
        if np.any(a <= 0):
            raise ValueError("alpha entries must be strictly positive.")

        self._alpha = a
        self._d = int(a.size)
        self._rng = rng or np.random.default_rng()
        self._cdf_mode = cdf_mode
        self._cdf_mc_samples = int(cdf_mc_samples)
        
        self._dir = _dirichlet(alpha=self._alpha)

        # Precompute mean & covariance (population)
        a0 = float(self._alpha.sum())
        self._mean = self._alpha / a0
        denom = (a0**2) * (a0 + 1.0)
        cov = -np.outer(self._alpha, self._alpha) / denom
        np.fill_diagonal(cov, self._alpha * (a0 - self._alpha) / denom)
        self._cov = 0.5 * (cov + cov.T)  # just in case of round-off


    def sample(self, n_samples: int) -> NDArray[np.floating]:
        """Draw `n_samples` from Dirichlet(alpha); returns array (n, d)."""
        
        X = self._dir.rvs(size=int(n_samples), random_state=self._rng)  # (n,d) or (d,) if n=1
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X  # (n, d)

    def density(self, values: NDArray) -> NDArray[np.floating]:
        """Return PDF values at `values` as column vector (n, 1)."""
        
        X = _as_2d(values)                 # (n, d)
        self._validate_simplex_rows(X)
        # Row-wise evaluation (SciPy supports vectorization, but this is explicit & safe)
        pmf = [self._dir.pdf(row) for row in X]
        return np.asarray(pmf, dtype=float).reshape(-1, 1)   # (n,1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """Return log-PDF values at `values` as column vector (n, 1)."""
        
        X = _as_2d(values)
        self._validate_simplex_rows(X)
        lpmf = [self._dir.logpdf(row) for row in X]
        return np.asarray(lpmf, dtype=float).reshape(-1, 1)  # (n,1)

    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray]) -> 'Multivariate':
        """
        Monte-Carlo CLT over f(X):
          - scalar f -> Normal1D(mean, se)
          - vector f -> MvNormal(mean, cov_of_mean)
        The function receives samples with shape (n_mc, d).
        """
        n_mc = 2048
        xs = self.sample(n_mc)                           # (n, d)
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
    def from_distribution(cls, convert_from: 'Distribution', num_samples: int=1024, *, conversion_by_KDE: bool = False, **fit_kwargs: Any) -> 'Dirichlet':
        """
        Moment-based fit of α from samples of a simplex-valued distribution.
        Steps:
          1) Draw samples X (n,d) from `convert_from`.
          2) Compute per-dimension sample means m and variances v.
          3) Estimate α0 ≈ mean_i( m_i(1-m_i)/v_i - 1 ), α_i = m_i * α0.
        Notes:
          - Rows are renormalized to sum to 1 (within tolerance).
          - Very small/zero variances are skipped in α0 averaging.
        """
        #n = int(fit_kwargs.get("n", 4000))
        X = np.asarray(convert_from.sample(num_samples), dtype=float)

        X = _as_2d(X)  # (n, d)

        # Coerce to simplex (guard for tiny numerical errors)
        X = np.clip(X, 0.0, None)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        X = X / row_sums

        m = X.mean(axis=0)                        # (d,)
        v = X.var(axis=0, ddof=1)                 # (d,)

        # Avoid divisions by zero; use only stable dimensions
        mask = (v > 1e-12) & (m > 1e-9) & (m < 1 - 1e-9)
        if not np.any(mask):
            # fallback: near-degenerate -> small uniform alpha
            d = X.shape[1]
            return cls(alpha=np.full(d, 1.0, dtype=float), rng=fit_kwargs.get("rng", None))

        alpha0_i = m[mask] * (1.0 - m[mask]) / v[mask] - 1.0
        alpha0 = float(np.maximum(alpha0_i.mean(), 1e-6))
        alpha = m * alpha0
        alpha = np.maximum(alpha, 1e-6)          # guard strictly positive

        return cls(alpha=alpha, rng=fit_kwargs.get("rng", None))


    def mean(self) -> NDArray[np.floating]:
        """Return the mean vector (d,)."""
        return self._mean  # (d,)

    def cov(self) -> NDArray[np.floating]:
         """Return the covariance matrix (d, d)."""
        return self._cov   # (d,d)

    def cdf(self, values: NDArray) -> NDArray[np.floating]:
        """
        Monte-Carlo approximation of P[X1<=x1, ..., Xd<=xd] for Dirichlet.
        Returns (n,1).
        """
        if self._cdf_mode != "mc":
            raise NotImplementedError("Only Monte-Carlo CDF ('mc') is supported for Dirichlet.")
        Xq = _as_2d(values)     # (n, d)
        self._validate_simplex_rows(Xq)
        m = self._cdf_mc_samples
        draws = self._dir.rvs(size=m, random_state=self._rng)  # (m,d) or (d,)
        draws = np.asarray(draws, dtype=float)
        if draws.ndim == 1:
            draws = draws.reshape(1, -1)
        out = np.empty(Xq.shape[0], dtype=float)
        for j, t in enumerate(Xq):
            out[j] = np.mean((draws <= t + 1e-12).all(axis=1))  # small tolerance
        return out.reshape(-1, 1)

    def inv_cdf(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        No practical Rosenblatt inverse for Dirichlet (without heavy numerics).
        """
        raise NotImplementedError("Dirichlet.inv_cdf is not available.")

    # ------------------------ helpers ------------------------

    def _validate_simplex_rows(self, X: NDArray[np.floating]) -> None:
        """
        Ensure rows lie on the simplex:
          - nonnegative (within tolerance),
          - each row sums to ~1.
        Raises ValueError on violation.
        """
        if X.shape[1] != self._d:
            raise ValueError(f"values must have shape (n, {self._d}) or ({self._d},).")
        if np.any(X < -1e-12):
            raise ValueError("values must be nonnegative (within numerical tolerance).")
        s = X.sum(axis=1)
        if not np.allclose(s, 1.0, atol=1e-6):
            raise ValueError("each row must sum to 1 (within tolerance).")


class Binomial(Multivariate[np.floating]):
    """
    Binomial(n_trials, p) with event dim = 1 (counts of successes).

    SciPy-backed:
      - sampling via scipy.stats.binom.rvs
      - pmf/logpmf via scipy.stats.binom.{pmf, logpmf}
      - cdf/ppf via scipy.stats.binom.{cdf, ppf}

    Shape policy:
      - sample(n) -> (n, 1)
      - density(values), log_density(values), cdf(values) -> (n, 1)
      - inv_cdf(u) -> (n, 1)
    """

    def __init__(
        self,
        n_trials: int,
        prob: float,
        *,
        rng: Optional[np.random.Generator] = None,
    ):

        n = int(n_trials)
        p = float(prob)
        if n < 0:
            raise ValueError("n_trials must be >= 0.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("prob must be in [0, 1].")

        self._n = n
        self._p = p
        self._rng = rng or np.random.default_rng()

        
        self._binom = _sbinom(n=self._n, p=self._p)

        # Precompute mean & covariance (population)
        mean = self._n * self._p
        var = self._n * self._p * (1.0 - self._p)
        self._mean = np.array([mean], dtype=float)          # (1,)
        self._cov = np.array([[var]], dtype=float)          # (1,1)


    def sample(self, n_samples: int) -> NDArray[np.floating]:
        """Draw (n, 1) samples of success counts (as float for typing consistency)."""
        x = self._binom.rvs(size=int(n_samples), random_state=self._rng)  # (n,) or scalar
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return x  # (n,1)

    def density(self, values: NDArray) -> NDArray[np.floating]:
        """Return (n,1) column of pmf at each provided count."""
        v = self._to_1d_counts(values)  # (n,), validated
        pmf = self._binom.pmf(v.astype(int, copy=False))    # (n,)
        return np.asarray(pmf, dtype=float).reshape(-1, 1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """Return (n,1) column of log-pmf at each provided count."""
        v = self._to_1d_counts(values)  # (n,)
        logpmf = self._binom.logpmf(v.astype(int, copy=False))  # (n,)
        return np.asarray(logpmf, dtype=float).reshape(-1, 1)

    def cdf(self, values: NDArray) -> NDArray[np.floating]:
        """Return (n,1) column of CDF values at each provided count."""
        v = self._to_1d_counts(values)  # (n,)
        c = self._binom.cdf(v.astype(int, copy=False))       # (n,)
        return np.asarray(c, dtype=float).reshape(-1, 1)

    def inv_cdf(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Quantile function (ppf). Accepts u as scalar, (n,) or (n,1). Returns (n,1).
        """
        U = np.asarray(u, dtype=float)
        if U.ndim == 0:
            q = self._binom.ppf(U)
            return np.asarray(q, dtype=float).reshape(1, 1)
        if U.ndim == 1:
            q = self._binom.ppf(U)
            return np.asarray(q, dtype=float).reshape(-1, 1)
        if U.ndim == 2 and U.shape[1] == 1:
            q = self._binom.ppf(U[:, 0])
            return np.asarray(q, dtype=float).reshape(-1, 1)
        raise ValueError("u must be scalar, (n,) or (n,1).")


    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray]) -> 'Multivariate':
        """
        Monte-Carlo CLT over f(X):
          - scalar f -> Normal1D(mean, se)
          - vector f -> MvNormal(mean, cov_of_mean)
        func receives samples as (n_mc, 1).
        """
        n_mc = 2048
        xs = self.sample(n_mc)                    # (n_mc, 1)
        ys = np.asarray(func(xs), dtype=float)

        # Treat (n,1) as scalar
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
    def from_distribution(cls, convert_from: 'Distribution', num_samples:int=1024, *, conversion_by_KDE: bool = False, **fit_kwargs: Any) -> 'Binomial':
   
        raise NotImplementedError


    def mean(self) -> NDArray[np.floating]:
        return self._mean  # (1,)

    def cov(self) -> NDArray[np.floating]:
        return self._cov   # (1,1)

    # ------------------------ helpers ------------------------

    def _to_1d_counts(self, values: NDArray) -> NDArray[np.floating]:
        """
        Normalize input to a 1-D vector of integer counts in [0, n_trials]:
          - scalar -> (1,)
          - (n,)   -> (n,)
          - (n,1)  -> (n,)
        """
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            v = arr.reshape(1)
        elif arr.ndim == 1:
            v = arr
        elif arr.ndim == 2 and arr.shape[1] == 1:
            v = arr[:, 0]
        else:
            raise ValueError("values must be scalar, (n,) or (n,1) for Binomial.")

        # integer-ish and within bounds
        if not np.allclose(v, np.round(v), atol=1e-8):
            raise ValueError("Binomial requires integer counts.")
        if np.any(v < -1e-12) or np.any(v > self._n + 1e-12):
            raise ValueError(f"Counts must lie in [0, n_trials={self._n}].")
        v = np.clip(v, 0, self._n)
        return v

    

class Beta(Multivariate[np.floating]):
    """
    Beta(alpha, β) distribution on [0, 1].

    SciPy-backed:
      - sampling via scipy.stats.beta.rvs
      - pdf/logpdf via scipy.stats.beta.{pdf, logpdf}
      - cdf/ppf via scipy.stats.beta.{cdf, ppf}

    Shape policy:
      - sample(n) -> (n, 1)
      - density(values), log_density(values), cdf(values) -> (n, 1)
      - inv_cdf(u) -> (n, 1)
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        *,
        rng: Optional[np.random.Generator] = None,
    ):

        a = float(alpha)
        b = float(beta)
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0 or b <= 0.0:
            raise ValueError("alpha and beta must be positive finite numbers.")

        self._a = a
        self._b = b
        self._rng = rng or np.random.default_rng()

        self._beta = _sps_beta(a=self._a, b=self._b)

        # Precompute mean & variance (population)
        a0 = self._a + self._b
        mean = self._a / a0
        var = (self._a * self._b) / (a0 * a0 * (a0 + 1.0))
        self._mean = np.array([mean], dtype=float)     # (1,)
        self._cov = np.array([[var]], dtype=float)     # (1,1)

    # ------------------------ Multivariate core ------------------------

    def sample(self, n_samples: int) -> NDArray[np.floating]:
        """
        Draw (n, 1) samples in [0,1].
        """
        x = self._beta.rvs(size=int(n_samples), random_state=self._rng)  # (n,) or scalar
        return np.asarray(x, dtype=float).reshape(-1, 1)

    def density(self, values: NDArray) -> NDArray[np.floating]:
        """
        Return (n,1) column of pdf at `values` ∈ [0,1].
        """
        v = _to_1d_vector(values)
        v = _clip_unit_interval(v)  # keep within [0,1]
        p = self._beta.pdf(v)       # (n,)
        return np.asarray(p, dtype=float).reshape(-1, 1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """
        Return (n,1) column of log-pdf at `values` ∈ [0,1].
        """
        v = _to_1d_vector(values)
        v = _clip_unit_interval(v)  # keep within [0,1]
        lp = self._beta.logpdf(v)   # (n,)
        return np.asarray(lp, dtype=float).reshape(-1, 1)

    def cdf(self, values: NDArray) -> NDArray[np.floating]:
        """
        Return (n,1) column of CDF values at `values` ∈ [0,1].
        """
        v = _to_1d_vector(values)
        v = _clip_unit_interval(v)
        c = self._beta.cdf(v)
        return np.asarray(c, dtype=float).reshape(-1, 1)

    def inv_cdf(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Quantile function (ppf). Accepts u as scalar, (n,) or (n,1). Returns (n,1).
        """
        U = np.asarray(u, dtype=float)
        if U.ndim == 0:
            q = self._beta.ppf(U)
            return np.asarray(q, dtype=float).reshape(1, 1)
        if U.ndim == 1:
            q = self._beta.ppf(_clip_unit_interval(U))
            return np.asarray(q, dtype=float).reshape(-1, 1)
        if U.ndim == 2 and U.shape[1] == 1:
            q = self._beta.ppf(_clip_unit_interval(U[:, 0]))
            return np.asarray(q, dtype=float).reshape(-1, 1)
        raise ValueError("u must be scalar, (n,), or (n,1).")

    # ------------------------ Expectations & converters ------------------------

    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray]) -> 'Multivariate':
        """
        Monte-Carlo CLT over f(X):
          - scalar f -> Normal1D(mean, se)
          - vector f -> MvNormal(mean, cov_of_mean)
        func receives samples as (n_mc, 1).
        """
        n_mc = 2048
        xs = self.sample(n_mc)                    # (n_mc, 1)
        ys = np.asarray(func(xs), dtype=float)

        # Treat (n,1) as scalar
        if ys.ndim == 2 and ys.shape[1] == 1:
            ys = ys[:, 0]

        if ys.ndim == 1:
            m = float(ys.mean())
            s = float(ys.std(ddof=1)) / np.sqrt(n_mc)
            from .distributions import Normal1D
            return Normal1D(m, max(s, 1e-12), rng=self._rng)
        elif ys.ndim == 2:
            m = ys.mean(axis=0)
            cov = np.cov(ys, rowvar=False, ddof=1) / n_mc
            cov = 0.5 * (cov + cov.T) + 1e-12 * np.eye(cov.shape[0])
            from .distributions import MvNormal
            return MvNormal(mean=m, cov=cov, rng=self._rng)
        else:
            raise ValueError(f"func must return (n,), (n,1) or (n,k). Got {ys.shape!r}")

    @classmethod
    def from_distribution(cls, convert_from: 'Distribution', num_samples: int=1024, *, conversion_by_KDE: bool = False, **fit_kwargs: Any) -> 'Beta':
        raise NotImplementedError


    def mean(self) -> NDArray[np.floating]:
        return self._mean  # (1,)

    def cov(self) -> NDArray[np.floating]:
        return self._cov   # (1,1)


