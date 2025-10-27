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
    """Abstract base class for multivariate, real-valued vector distributions.

    Represents probability distributions in ℝᵈ with a fixed dimension `d`.
    Subclasses define how to compute means, covariances, and cumulative
    distribution functions (CDFs). The event shape is assumed to be `(d,)`.

    Notes:
        - All subclasses should ensure shape consistency across methods.
        - Default dimension inference is based on the output of :meth:`mean`.

    Raises:
        NotImplementedError: If abstract methods are not implemented by a subclass.
    """

    @abstractmethod
    def mean(self) -> NDArray[Float_T]:
        """Computes the mean vector of the distribution.

        Returns:
            Mean vector μ with shape `(d,)`.

        Raises:
            NotImplementedError: If the mean does not exist (e.g., for Cauchy).
        """
        raise NotImplementedError

    @abstractmethod
    def cov(self) -> NDArray[np.floating]:
        """Computes the covariance matrix of the distribution.

        Returns:
            Covariance matrix Sigma with shape `(d, d)`.

        Raises:
            NotImplementedError: If the covariance does not exist.
        """
        raise NotImplementedError

   
    @abstractmethod
    def cdf(self, x: NDArray[Float_T]) -> NDArray[np.floating]:
        """Evaluates the joint cumulative distribution function (CDF).

        Computes ``F(x) = P[X_1 ≤ x_1, …, X_d ≤ x_d]`` for input vectors `x`.
        Implementations may use analytical formulas (rare), numerical integration,
        or library routines when available.

        Args:
            x: Points at which to evaluate the CDF, shape `(..., d)`.

        Returns:
            CDF values, shape `(...,)`.

        Raises:
            NotImplementedError: If not implemented by the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def inv_cdf(self, u: NDArray[np.floating]) -> NDArray[Float_T]:
        """Computes the inverse CDF (Rosenblatt inverse) mapping.

        Maps `u ∈ (0, 1)^d` to `x ∈ R^d` using the inverse of the joint CDF.
        For elliptical distributions (e.g., multivariate normal), a common
        implementation is:

        ```
        z = Phi^{-1}(u)      # Componentwise inverse of the univariate normal CDF
        x = mu + Lz      # L is the Cholesky factor of Sigma
        ```

        Args:
            u: Input array in the open unit hypercube `(0, 1)^d`
                with shape `(..., d)`.

        Returns:
            Output samples `x` in R^d with shape `(..., d)`.

        Raises:
            NotImplementedError: If not implemented by the subclass.
        """
        raise NotImplementedError

    # ---- Dimension helper ----
    @property
    def dimension(self) -> int:
        """Infers the dimensionality of the distribution.

        The default implementation infers `d` from the shape of :meth:`mean`.
        Subclasses may override this method for fixed or analytically known
        dimensions.

        Returns:
            Number of coordinates `d`.

        Raises:
            ValueError: If :meth:`mean` does not return a 1D array.
        """
        m = self.mean()
        if m.ndim != 1:
            raise ValueError("mean() must return a 1D array of shape (d,).")
        return int(m.shape[0])



class Normal1D(Multivariate[np.floating]):
    """Univariate Normal distribution N(μ, σ²) implemented as a 1D Multivariate.

    Represents a univariate Gaussian distribution with event shape (1,).
    Provides sampling, PDF/log-PDF evaluation, CDF/inverse-CDF methods,
    and Monte Carlo expectation computation.

    Shape policy:
        - ``sample(n)`` -> (n, 1)
        - ``density`` / ``log_density`` / ``cdf(values)`` always return (n, 1)
          even though they are scalar per sample.
        - ``inv_cdf(u)`` -> (n, 1)

    Attributes:
        mu: Mean of the distribution.
        sigma: Standard deviation (must be > 0).
        _rng: Random number generator used for sampling.
    """

    def __init__(self, mu: float, sigma: float, *, rng: np.random.Generator | None = None):
        """Initializes a Normal1D distribution.

        Args:
            mu: Mean of the distribution.
            sigma: Standard deviation (must be > 0).
            rng: Random number generator.
                If ``None``, a default generator is created.

        Raises:
            ValueError: If ``sigma`` is not positive.
        """
        
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self._rng = rng or np.random.default_rng()

        self._norm = norm(loc=self.mu, scale=self.sigma)


    def sample(self, n_samples: int) -> NDArray[np.floating]:
        """Draws random samples from the distribution.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Samples of shape (n_samples, 1).
        """
        
        xs = self._norm.rvs(size=(int(n_samples), 1), random_state=self._rng)
        return np.asarray(xs, dtype=float)  # (n, 1)


    def density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the probability density function (PDF) at given values.

        Args:
            values: Points at which to evaluate the PDF.

        Returns:
            PDF values of shape (n, 1).
        """
        
        v = self._to_1d_vector(values)          # (n,)
        p = np.asarray(self._norm.pdf(v), dtype=float).reshape(-1, 1)
        return p                                 # (n, 1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the log of the probability density function.

        Args:
            values: Points at which to evaluate the log-PDF.

        Returns:
            Log-PDF values of shape (n, 1).
        """
        
        v = _to_1d_vector(values)          # (n,)
        lp = np.asarray(self._norm.logpdf(v), dtype=float).reshape(-1, 1)
        return lp                                # (n, 1)

    def cdf(self, x: NDArray) -> NDArray[np.floating]:
        """Evaluates the cumulative distribution function (CDF).

        Args:
            x: Points at which to evaluate the CDF.

        Returns:
            CDF values of shape (n, 1).
        """
        
        v = self._to_1d_vector(x)               # (n,)
        c = np.asarray(self._norm.cdf(v), dtype=float).reshape(-1, 1)
        return c                                 # (n, 1)

    def inv_cdf(self, u: NDArray) -> NDArray[np.floating]:
        """Computes quantiles (inverse CDF) for given probabilities.

        Args:
            u: Probabilities in (0, 1) to transform.

        Returns:
            Quantiles of shape (n, 1).

        Raises:
            ValueError: If the input ``u`` has invalid shape.
        """
        
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
        """Returns the mean vector.

        Returns:
            Mean of shape (1,).
        """
        
        return np.array([self.mu], dtype=float)          # (1,)

    def cov(self) -> NDArray[np.floating]:
        """Returns the covariance matrix.

        Returns:
            Covariance matrix of shape (1, 1) = [[sigma^2]].
        """
        
        return np.array([[self.sigma ** 2]], dtype=float)  # (1,1)


    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray[np.floating]]) -> 'Multivariate':
        """Estimates the expectation of a function via Monte Carlo sampling.

        Draws samples from the distribution and computes the Central Limit
        Theorem approximation of E[f(X)].

        Behavior:
            - If ``f`` returns a scalar -> returns a :class:`Normal1D`
              summarizing the mean and standard error.
            - If ``f`` returns a vector -> returns a :class:`MvNormal`
              summarizing the mean vector and covariance of the mean.

        Args:
            func: Function mapping samples
                of shape (n, 1) to outputs.
        
        Returns:
            Normal1D or MvNormal summarizing the estimated expectation.

        Raises:
            ValueError: If ``func`` returns data of unsupported shape.
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
        """Fits a Normal1D to samples drawn from another distribution (moment matching).

        Draws samples from the given distribution and estimates mean and
        standard deviation to create a Normal1D approximation.

        Args:
            convert_from: Source distribution to sample from.
            num_samples: Number of samples to draw. Defaults to 1024.
            conversion_by_KDE: Placeholder for API compatibility; unused.
            **fit_kwargs: Optional keyword arguments (ignored).

        Returns:
            Fitted distribution instance.
        """
        
        #n = int(fit_kwargs.get("n", 2000))
        xs = np.asarray(convert_from.sample(num_samples), dtype=float)

        # Flatten (n,1) -> (n,), accept (n,)
        if xs.ndim == 2 and xs.shape[1] == 1:
            xs = xs[:, 0]
        elif xs.ndim != 1:
            xs = xs.reshape(-1)

        mu = float(xs.mean())
        sigma = float(xs.std(ddof=1))
        return cls(mu, max(sigma, 1e-12))




class MvNormal(Multivariate[np.floating]):
    """Multivariate Normal distribution N(mu, Sigma) using SciPy.

    Represents a d-dimensional Gaussian distribution implemented via
    ``scipy.stats.multivariate_normal``. Provides sampling, density/log-density,
    CDF, inverse-CDF, and Monte Carlo expectation functionalities.

    Shape policy:
        - ``sample(n)`` -> (n, d)
        - ``density`` / ``log_density`` / ``cdf(x)`` *-> (n, 1)
          (scalar-per-sample outputs returned as column vectors)
        - ``inv_cdf(u)`` -> (n, d)

    Attributes:
        _mean: Mean vector of shape (d,).
        _cov: Covariance matrix of shape (d, d).
        _rng: Random number generator for sampling.
    """

    def __init__(self, mean: NDArray[np.floating], cov: NDArray[np.floating],
                 *, rng: np.random.Generator | None = None):
         """Initializes a multivariate Normal distribution.

        Args:
            mean: Mean vector of shape (d,).
            cov: Covariance matrix of shape (d, d).
            rng: Random number generator.
                If ``None``, a default generator is created.

        Raises:
            ValueError: If ``mean`` or ``cov`` have incompatible shapes.
        """
                     
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
        """Generates random samples from the distribution.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            Samples of shape (n_samples, d).
        """
        x = self._mvn.rvs(size=int(n_samples), random_state=self._rng)
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:  # when n_samples == 1 SciPy may return (d,)
            x = x.reshape(1, -1)
        return x  # (n, d)

    def density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the probability density function (PDF) at given points.

        Args:
            values: Points at which to evaluate the PDF,
                of shape (d,) or (n, d).

        Returns:
            Column vector of PDF values, shape (n, 1).
        """
        X = _as_2d(values)                   # (n, d)
        p = self._mvn.pdf(X)                 # (n,) from SciPy
        return np.asarray(p, dtype=float).reshape(-1, 1)   # (n,1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the log of the probability density function.

        Args:
            values: Points at which to evaluate the log-PDF,
                of shape (d,) or (n, d).

        Returns:
            Column vector of log-PDF values, shape (n, 1).
        """
        X = _as_2d(values)                   # (n, d)
        lp = self._mvn.logpdf(X)             # (n,)
        return np.asarray(lp, dtype=float).reshape(-1, 1)  # (n,1)

    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray[np.floating]]) -> 'Multivariate':
        """Estimates the expectation of a function via Monte Carlo sampling.

        Draws samples from the distribution and applies ``func`` to each.
        Uses the Central Limit Theorem to estimate E[f(X)] and return a
        Normal1D or MvNormal summarizing the uncertainty.

        Behavior:
            - If ``func`` returns a scalar -> returns a :class:`Normal1D`
              summarizing the mean and standard error.
            - If ``func`` returns a vector -> returns a :class:`MvNormal`
              summarizing the mean vector and covariance of the mean.

        Args:
            func: Function mapping samples
                of shape (n, d) to numeric outputs.

        Returns:
            A :class:`Normal1D` or :class:`MvNormal` summarizing
            the Monte Carlo expectation.

        Raises:
            ValueError: If ``func`` returns an unsupported shape.
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
        """Fits a multivariate Normal to samples drawn from another distribution.

        Computes the sample mean and covariance matrix from draws of the
        provided distribution to fit a :class:`MvNormal`.

        Args:
            convert_from: Source distribution to sample from.
            num_samples: Number of samples to draw. Defaults to 1024.
            conversion_by_KDE: Placeholder for compatibility; unused.
            **fit_kwargs: Optional keyword arguments (ignored).

        Returns:
            Fitted multivariate Normal distribution.
        """
        #n = int(fit_kwargs.get("n", 4000))
        xs = np.asarray(convert_from.sample(num_samples), dtype=float)

        X = _as_2d(xs)                                   # (n, d)
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False, ddof=1)
        cov = _symmetrize_spd(cov)
        return cls(mean=mean, cov=cov)


    def mean(self) -> NDArray[np.floating]:
        """Returns the mean vector.

        Returns:
            Mean vector of shape (d,).
        """
        
        return self._mean  # (d,)

    def cov(self) -> NDArray[np.floating]:
        """Returns the covariance matrix.

        Returns:
            Covariance matrix of shape (d, d).
        """
        
        return self._cov   # (d,d)

    def cdf(self, x: NDArray) -> NDArray[np.floating]:
        """Evaluates the cumulative distribution function (CDF) at given points.

        SciPy’s frozen ``cdf`` does not vectorize over rows, so this method
        evaluates each sample row-wise.

        Args:
            x: Points of shape (d,) or (n, d).

        Returns:
            CDF values of shape (n, 1).
        """

        X = _as_2d(x)                           # (n,d)
        # SciPy's frozen cdf doesn't vectorize over rows, so evaluate row-wise.
        # (For large n you may want to batch this.)
        vals = [float(self._mvn.cdf(row)) for row in X]
        return np.array(vals, dtype=float).reshape(-1, 1)   # (n,1)

    def inv_cdf(self, u: NDArray) -> NDArray[np.floating]:
        """Computes the Rosenblatt inverse transform for uniform samples.

        Generates corresponding points ``x`` in R^d for each uniform sample ``u``
        in (0, 1)^d using sequential conditional Gaussian transforms.

        Args:
            u: Array of shape (d,), (n, d), or (..., d) with
                entries in (0, 1).

        Returns:
            Quantiles (inverse CDF) of shape (n, d) or (d,).

        Raises:
            ValueError: If the input shape does not match the distribution’s dimension.
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
        """Returns the dimensionality of the distribution.

        Returns:
           Number of coordinates ``d``.
        """
        return int(self._mean.shape[0])



class GaussianKDE(Multivariate[np.floating]):
    """Gaussian kernel density estimator (KDE) with shared bandwidth matrix H.

    Represents a weighted Gaussian mixture where each sample `x_i` defines
    a kernel N(x_i, H). The overall density is the mixture sum
    ``Sum_i w_i N(x | x_i, H)``.

    Shape policy:
        - ``sample(n)`` -> (n, d)
        - ``density`` / ``log_density`` / ``cdf(x)`` -> (n, 1)
        - ``inv_cdf(u)`` -> not implemented

    Attributes:
        _X: Sample centers, shape (n, d).
        _w: Normalized nonnegative weights, shape (n,).
        _n: Number of samples.
        _d: Dimensionality.
        _H: Bandwidth (covariance) matrix, shape (d, d).
        _rng: Random number generator.
        _kernels: Frozen SciPy multivariate normal kernels for each sample.
        _cov_mix: Total mixture covariance = Cov_w(X) + H.
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
        """Initializes the Gaussian KDE.

        Args:
            samples: Input samples of shape (n, d).
            weights: Optional nonnegative weights, shape (n,).
                Will be normalized to sum to 1. Defaults to uniform weights.
            bandwidth (float | NDArray | None): Bandwidth specification.
                Can be a scalar, vector of shape (d,), or matrix of shape (d, d).
                If ``None``, uses ``rule`` to compute bandwidth automatically.
            rule: Bandwidth selection rule ('scott' or 'silverman').
                Used only when ``bandwidth`` is ``None``.
            rng: Random generator instance.
                Defaults to a newly created generator.

        Raises:
            ValueError: If inputs are invalid (e.g., empty samples, negative weights,
                or mismatched dimensions).
        """

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
        """Constructs the bandwidth matrix H.

        Args:
            bandwidth (float | NDArray | None): Bandwidth definition.
                - Scalar: uses isotropic H = (h^2)I.
                - Vector: uses diagonal H = diag(h₁², …, h_d^2).
                - Matrix: directly uses user-provided H.
                - None: computes automatically via ``rule``.
            rule: 'scott' or 'silverman' rule for automatic bandwidth.

        Returns:
            Bandwidth matrix H, shape (d, d).

        Raises:
            ValueError: If the provided bandwidth or rule is invalid.
        """
        
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
        """Draws random samples from the KDE mixture.

        Each sample is drawn by selecting a center x_i (according to weights)
        and adding Gaussian noise from N(0, H).

        Args:
            n_samples: Number of samples to draw.

        Returns:
            Random samples of shape (n_samples, d).
        """

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
        """Evaluates the mixture probability density function (PDF).

        Computes:
            ``p(x) = Sum_i w_i N(x | x_i, H)``

        Args:
            values: Points of shape (n, d) or (d,).

        Returns:
            Column vector of PDF values, shape (n, 1).
        """
        Xq = _as_2d(values)  # (n, d)
        out = np.empty(Xq.shape[0], dtype=float)
        for j, q in enumerate(Xq):
            # Weighted sum of kernel PDFs at q
            out[j] = float(np.dot(self._w, [k.pdf(q) for k in self._kernels]))
        return out.reshape(-1, 1)  # (n,1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the log of the mixture density via a stable log-sum-exp.

        Computes:
            ``log p(x) = log Sum_i w_i exp(log N_i(x))``

        Args:
            values: Points of shape (n, d) or (d,).

        Returns:
            Column vector of log-PDF values, shape (n, 1).
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
        """Returns the mixture mean.

        Returns:
            Weighted mean of sample centers, shape (d,).
        """
        return self._mean  # (d,)

    def cov(self) -> NDArray[np.floating]:
        """Returns the mixture covariance.

        Returns:
            Covariance matrix of shape (d, d).
        """
        return self._cov_mix  # (d,d)

    def cdf(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the mixture cumulative distribution function (CDF).

        Approximates:
            ``F(x) = Sum_i w_i Phi_d(x; x_i, H)``

        Args:
            values: Evaluation points of shape (n, d) or (d,).

        Returns:
            CDF values, shape (n, 1).
        """
        Xq = _as_2d(values)  # (n, d)
        out = np.empty(Xq.shape[0], dtype=float)
        for j, q in enumerate(Xq):
            out[j] = float(np.dot(self._w, [k.cdf(q) for k in self._kernels]))
        return out.reshape(-1, 1)

    def inv_cdf(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        """Inverse CDF is not defined for Gaussian mixture models.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("GaussianKDE.inv_cdf is not available for kernel mixtures.")

    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray]) -> 'Multivariate':
        """Estimates the expectation of a function under the KDE via Monte Carlo.

        Draws samples from the KDE and applies ``func`` to estimate E[f(X)].

        Behavior:
            - If ``f`` returns a scalar -> returns a :class:`Normal1D`
              summarizing the mean and standard error.
            - If ``f`` returns a vector -> returns a :class:`MvNormal`
              summarizing the mean and covariance.

        Args:
            func: Function mapping samples
                of shape (n, d) to outputs.

        Returns:
            :class:`Normal1D` or :class:`MvNormal` summarizing
            the estimated expectation.

        Raises:
            ValueError: If the function output shape is unsupported.
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
        """Builds a Gaussian KDE from another distribution-like object.

        Args:
            convert_from: Source distribution to sample from.
            num_samples: Number of samples to draw. Defaults to 1024.
            conversion_by_KDE: Placeholder flag for compatibility. Unused.
            **fit_kwargs: Optional keyword arguments:
                - ``weights``: Optional sample weights.
                - ``bandwidth``: Bandwidth override.
                - ``rule``: Bandwidth rule ('scott' or 'silverman').
                - ``rng``: RNG for reproducibility.

        Returns:
            KDE fitted to the source distribution.
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
    """Multinomial distribution over discrete outcomes.

    Represents a multinomial distribution with a fixed number of trials ``n_trials``
    and category probabilities ``p``. Backed by ``scipy.stats.multinomial``.

    Shape policy:
        - ``sample(n)`` -> (n, d)  (counts per category)
        - ``density`` / ``log_density`` / ``cdf`` -> (n, 1)
          (scalar-per-sample outputs returned as column vectors)
        - ``inv_cdf(u)`` -> not implemented (no closed-form inverse)

    Notes:
        * Counts are represented as float arrays to satisfy ``Multivariate[float]`` typing.
          Cast to ``int`` if integer counts are needed.
        * The ``cdf`` is approximated via Monte Carlo simulation unless otherwise noted.
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
        """Initializes a multinomial distribution.

        Args:
            n_trials: Number of trials per draw. Must be nonnegative.
            probs: Category probabilities of shape (d,).
                Must be nonnegative and sum to a positive value.
            rng: Random number generator.
                Defaults to ``np.random.default_rng()``.
            cdf_mode: CDF computation mode. Only ``"mc"`` (Monte Carlo)
                is currently supported. Defaults to ``"mc"``.
            cdf_mc_samples: Number of Monte Carlo samples to approximate
                the CDF. Defaults to 20,000.

        Raises:
            ValueError: If probabilities are invalid or inconsistent with ``n_trials``.
        """

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
        """Draws samples of counts from the multinomial.

        Args:
            n_samples: Number of draws.

        Returns:
            Samples of shape (n_samples, d), representing category counts
            per trial batch, returned as float values.
        """
        X = self._mn.rvs(size=int(n_samples), random_state=self._rng)  # (n,d) or (d,) if n=1
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X  # (n, d) float

    def density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the probability mass function (PMF) for given count vectors.

        Args:
            values: Array of shape (n, d) or (d,), where each row
                represents integer-valued counts that sum to ``n_trials``.

        Returns:
            Column vector of PMF values, shape (n, 1).

        Raises:
            ValueError: If the input rows are invalid count vectors.
        """
        X = _as_2d(values)  # (n,d)
        self._validate_counts_rows(X)
        # SciPy's logpmf/pmf can handle arrays row-wise via list comprehension
        pmfs = [self._mn.pmf(row.astype(int, copy=False)) for row in X]
        return np.asarray(pmfs, dtype=float).reshape(-1, 1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the log of the probability mass function (log-PMF).

        Args:
            values: Array of shape (n, d) or (d,).

        Returns:
            Column vector of log-PMF values, shape (n, 1).
        """
        X = _as_2d(values)  # (n,d)
        self._validate_counts_rows(X)
        logpmfs = [self._mn.logpmf(row.astype(int, copy=False)) for row in X]
        return np.asarray(logpmfs, dtype=float).reshape(-1, 1)

    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray]) -> 'Multivariate':
        """Estimates the expectation of a function via Monte Carlo sampling.

        Draws multinomial samples and applies ``func`` to estimate E[f(X)].

        Behavior:
            - If ``f`` returns a scalar -> returns a :class:`Normal1D`
              summarizing the mean and standard error.
            - If ``f`` returns a vector -> returns a :class:`MvNormal`
              summarizing the mean vector and covariance.

        Args:
            func: Function mapping
                samples of shape (n, d) to outputs.

        Returns:
            :class:`Normal1D` or :class:`MvNormal` summarizing
            the estimated expectation.

        Raises:
            ValueError: If the function output shape is unsupported.
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
        """Fits a multinomial (n_trials, p) from sampled count vectors.

        Each row of the sampled data is assumed to be a valid count vector
        summing to a constant number of trials.

        Args:
            convert_from: Source distribution to sample from.
            num_samples: Number of samples to draw. Defaults to 1024.
            conversion_by_KDE: Placeholder for compatibility. Unused.
            **fit_kwargs: Optional keyword arguments; may include ``rng``.

        Returns:
            Estimated multinomial distribution.
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
        """Returns the mean vector.

        Returns:
            Mean vector of shape (d,).
        """
        return self._mean  # (d,)

    def cov(self) -> NDArray[np.floating]:
         """Returns the covariance matrix.

        Returns:
            Covariance matrix of shape (d, d).
        """
        return self._cov   # (d,d)

    def cdf(self, values: NDArray) -> NDArray[np.floating]:
        """Approximates the joint cumulative distribution function (CDF).

        Computes a Monte Carlo estimate of
        ``P[X_1 ≤ x_1, ..., X_d ≤ x_d]`` using random draws.

        Args:
            values: Array of shape (n, d) or (d,)
                representing upper thresholds for counts.

        Returns:
            Column vector of CDF values, shape (n, 1).

        Raises:
            NotImplementedError: If a non-Monte-Carlo mode is requested.
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
        """Inverse CDF (Rosenblatt transform) is not implemented."""
        raise NotImplementedError("Multinomial.inv_cdf is not available.")

    # ------------------------ helper ------------------------

    def _validate_counts_rows(self, X: NDArray[np.floating]) -> None:
        """Validates that rows in ``X`` represent valid count vectors.

        Args:
            X: Array of shape (n, d) or (d,).

        Raises:
            ValueError: If any row violates multinomial constraints:
                - Non-integer counts
                - Negative counts
                - Row sum != n_trials
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
    """Dirichlet(alpha) distribution over the probability simplex.

    Represents a continuous distribution supported on the simplex
    ``{x ∈ R^d : x_i >= 0, Sum x_i = 1}``, parameterized by positive concentration
    parameters α = (alpha_1, ..., alpha_d). Implements sampling, density evaluation,
    and Monte Carlo–based CDF estimation.

    SciPy-backed implementation:
        - Sampling: ``scipy.stats.dirichlet.rvs``
        - PDF / log-PDF: ``scipy.stats.dirichlet.{pdf, logpdf}``

    Shape policy:
        - ``sample(n)`` -> (n, d)
        - ``density`` / ``log_density`` / ``cdf(x)`` -> (n, 1)
        - ``inv_cdf(u)`` -> not implemented

    Attributes:
        _alpha: Concentration parameters of shape (d,).
        _d: Dimension of the simplex.
        _rng: Random number generator.
        _cdf_mode: CDF computation mode (currently 'mc').
        _cdf_mc_samples: Number of Monte Carlo draws used for CDF estimation.
        _mean: Mean vector of shape (d,).
        _cov: Covariance matrix of shape (d, d).
    """

    def __init__(
        self,
        alpha: NDArray[np.floating] | NDArray[np.floating],
        *,
        rng: Optional[np.random.Generator] = None,
        cdf_mode: str = "mc",
        cdf_mc_samples: int = 20000,
    ):
        """Initializes a Dirichlet distribution.

        Args:
            alpha: Positive concentration parameters, shape (d,).
            rng: Random generator. If ``None``,
                creates a new default generator.
            cdf_mode: CDF computation method ('mc' for Monte Carlo).
            cdf_mc_samples: Number of Monte Carlo samples for approximating
                the CDF when ``cdf_mode='mc'``.

        Raises:
            ValueError: If ``alpha`` is not 1D, empty, or contains nonpositive entries.
        """

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
        """Generates random samples from the Dirichlet distribution.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            Samples of shape (n_samples, d).
        """
        
        X = self._dir.rvs(size=int(n_samples), random_state=self._rng)  # (n,d) or (d,) if n=1
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X  # (n, d)

    def density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the probability density function (PDF).

        Args:
            values: Points on the simplex, shape (n, d) or (d,).

        Returns:
            Column vector of PDF values, shape (n, 1).

        Raises:
            ValueError: If inputs do not lie on the simplex.
        """
        
        X = _as_2d(values)                 # (n, d)
        self._validate_simplex_rows(X)
        # Row-wise evaluation (SciPy supports vectorization, but this is explicit & safe)
        pmf = [self._dir.pdf(row) for row in X]
        return np.asarray(pmf, dtype=float).reshape(-1, 1)   # (n,1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the log-probability density function (log-PDF).

        Args:
            values: Points on the simplex, shape (n, d) or (d,).

        Returns:
            Column vector of log-PDF values, shape (n, 1).

        Raises:
            ValueError: If inputs do not lie on the simplex.
        """
        
        X = _as_2d(values)
        self._validate_simplex_rows(X)
        lpmf = [self._dir.logpdf(row) for row in X]
        return np.asarray(lpmf, dtype=float).reshape(-1, 1)  # (n,1)

    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray]) -> 'Multivariate':
        """Estimates E[f(X)] via Monte Carlo sampling.

        Draws samples and applies ``func`` to approximate the expectation under
        the Dirichlet law.

        Behavior:
            - Scalar-valued ``f`` -> returns :class:`Normal1D`
            - Vector-valued ``f`` -> returns :class:`MvNormal`

        Args:
            func: Function mapping samples
                of shape (n, d) to numerical outputs.

        Returns:
            :class:`Normal1D` or :class:`MvNormal` summarizing
            the estimated mean and uncertainty.

        Raises:
            ValueError: If the output shape of ``func`` is unsupported.
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
        """Fits Dirichlet parameters via moment matching from another distribution.

        Estimates α parameters from samples drawn from ``convert_from``,
        assuming samples lie on the probability simplex.

        Steps:
            1. Draw samples X (n, d).
            2. Compute empirical means and variances.
            3. Estimate α₀ ≈ mean_i[m_i(1−m_i)/v_i − 1], then alpha_i = m_i alpha_e.

        Args:
            convert_from: Source distribution.
            num_samples: Number of samples to draw. Defaults to 1024.
            conversion_by_KDE: Placeholder, unused.
            **fit_kwargs: Additional keyword arguments, e.g.:
                - ``rng``: Custom RNG.

        Returns:
            Estimated Dirichlet distribution.

        Notes:
            - Samples are renormalized to ensure each row sums to 1.
            - If variances are too small, defaults to a small uniform alpha.
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
        """Returns the mean vector.

        Returns:
            Mean vector of shape (d,).
        """
        return self._mean  # (d,)

    def cov(self) -> NDArray[np.floating]:
         """Returns the covariance matrix.

        Returns:
            Covariance matrix of shape (d, d).
        """
        return self._cov   # (d,d)

    def cdf(self, values: NDArray) -> NDArray[np.floating]:
        """Approximates the CDF using Monte Carlo sampling.

        Computes:
            ``F(x) = P[X_1 ≤ x_1, ..., X_d ≤ x_d]``

        Args:
            values: Evaluation points on the simplex, shape (n, d).

        Returns:
           Column vector of CDF estimates, shape (n, 1).

        Raises:
            NotImplementedError: If ``cdf_mode`` is not 'mc'.
            ValueError: If inputs are invalid or off the simplex.
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
        """Inverse CDF is not available for Dirichlet distributions."""
        raise NotImplementedError("Dirichlet.inv_cdf is not available.")

    # ------------------------ helpers ------------------------

    def _validate_simplex_rows(self, X: NDArray[np.floating]) -> None:
        """Ensures each row lies on the probability simplex.

        Checks:
            - All entries are nonnegative (within tolerance).
            - Each row sums to approximately 1.

        Args:
            X: Array of shape (n, d).

        Raises:
            ValueError: If any row is invalid.
        """
        if X.shape[1] != self._d:
            raise ValueError(f"values must have shape (n, {self._d}) or ({self._d},).")
        if np.any(X < -1e-12):
            raise ValueError("values must be nonnegative (within numerical tolerance).")
        s = X.sum(axis=1)
        if not np.allclose(s, 1.0, atol=1e-6):
            raise ValueError("each row must sum to 1 (within tolerance).")


class Binomial(Multivariate[np.floating]):
    """Binomial(n_trials, p) distribution with event dimension 1 (counts of successes).

    Represents the discrete Binomial distribution parameterized by
    the number of trials ``n_trials`` and success probability ``p``.
    Provides methods for sampling, evaluating the PMF, log-PMF, CDF, and
    inverse CDF, along with Monte Carlo–based expectation estimation.

    SciPy-backed implementation:
        - Sampling: ``scipy.stats.binom.rvs``
        - PMF / log-PMF: ``scipy.stats.binom.{pmf, logpmf}``
        - CDF / quantile: ``scipy.stats.binom.{cdf, ppf}``

    Shape policy:
        - ``sample(n)`` -> (n, 1)
        - ``density`` / ``log_density`` / ``cdf`` -> (n, 1)
        - ``inv_cdf(u)`` -> (n, 1)

    Attributes:
        _n: Number of trials.
        _p: Success probability in [0, 1].
        _rng: Random number generator.
        _mean: Mean vector, shape (1,).
        _cov: Covariance matrix, shape (1, 1).
    """

    def __init__(
        self,
        n_trials: int,
        prob: float,
        *,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initializes a Binomial distribution.

        Args:
            n_trials: Number of independent trials (n >= 0).
            prob: Probability of success for each trial, in [0, 1].
            rng: Random generator. If ``None``,
                creates a new default generator.

        Raises:
            ValueError: If ``n_trials < 0`` or ``prob`` not in [0, 1].
        """

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
        """Draws samples of Binomial counts.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            Array of success counts, shape (n_samples, 1).
        """
        x = self._binom.rvs(size=int(n_samples), random_state=self._rng)  # (n,) or scalar
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return x  # (n,1)

    def density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the probability mass function (PMF).

        Args:
            values: Count values; scalar, (n,), or (n, 1).

        Returns:
            Column vector of PMF values, shape (n, 1).

        Raises:
            ValueError: If counts are non-integer or outside [0, n_trials].
        """
        v = self._to_1d_counts(values)  # (n,), validated
        pmf = self._binom.pmf(v.astype(int, copy=False))    # (n,)
        return np.asarray(pmf, dtype=float).reshape(-1, 1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the log-probability mass function (log-PMF).

        Args:
            values: Count values; scalar, (n,), or (n, 1).

        Returns:
            Column vector of log-PMF values, shape (n, 1).

        Raises:
            ValueError: If counts are non-integer or outside [0, n_trials].
        """
        v = self._to_1d_counts(values)  # (n,)
        logpmf = self._binom.logpmf(v.astype(int, copy=False))  # (n,)
        return np.asarray(logpmf, dtype=float).reshape(-1, 1)

    def cdf(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the cumulative distribution function (CDF).

        Args:
            values: Count values; scalar, (n,), or (n, 1).

        Returns:
            Column vector of CDF values, shape (n, 1).

        Raises:
            ValueError: If counts are non-integer or outside [0, n_trials].
        """
        v = self._to_1d_counts(values)  # (n,)
        c = self._binom.cdf(v.astype(int, copy=False))       # (n,)
        return np.asarray(c, dtype=float).reshape(-1, 1)

    def inv_cdf(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluates the quantile (inverse CDF) function.

        Args:
            u: Probabilities; scalar, (n,), or (n, 1).

        Returns:
            Quantiles (integer counts), shape (n, 1).

        Raises:
            ValueError: If input shape is invalid.
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
        """Estimates the expectation of a function under the Binomial distribution.

        Uses Monte Carlo sampling to compute E[f(X)] and summarize the
        uncertainty via the Central Limit Theorem.

        Behavior:
            - Scalar-valued ``f`` -> returns :class:`Normal1D`
            - Vector-valued ``f`` -> returns :class:`MvNormal`

        Args:
            func: Function mapping samples (n, 1) -> values.

        Returns:
            Normal1D or MvNormal summarizing the estimated mean
            and uncertainty.

        Raises:
            ValueError: If ``func`` returns outputs of unsupported shape.
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
        """Fit not implemented for Binomial distributions."""
   
        raise NotImplementedError


    def mean(self) -> NDArray[np.floating]:
        """Returns the mean vector (1,)."""
        return self._mean  # (1,)

    def cov(self) -> NDArray[np.floating]:
        """Returns the covariance matrix (1×1)."""
        return self._cov   # (1,1)

    # ------------------------ helpers ------------------------

    def _to_1d_counts(self, values: NDArray) -> NDArray[np.floating]:
        """Normalizes input to a 1D array of integer counts in [0, n_trials].

        Acceptable shapes:
            - scalar -> (1,)
            - (n,)   -> (n,)
            - (n, 1) -> (n,)

        Args:
            values: Array-like of count values.

        Returns:
            Flattened counts (n,).

        Raises:
            ValueError: If values are non-integer or outside [0, n_trials].
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
    """Beta(alpha, beta) distribution on the unit interval [0, 1].

    Provides standard Beta distribution functionality backed by
    :mod:`scipy.stats.beta`. Supports sampling, density evaluation,
    cumulative probabilities, quantiles, and Monte-Carlo expectation
    estimation.

    SciPy-backed implementation:
        - Sampling: :func:`scipy.stats.beta.rvs`
        - PDF / log-PDF: :func:`scipy.stats.beta.{pdf, logpdf}`
        - CDF / quantile: :func:`scipy.stats.beta.{cdf, ppf}`

    Shape policy:
        - ``sample(n)`` -> (n, 1)
        - ``density`` / ``log_density`` / ``cdf`` -> (n, 1)
        - ``inv_cdf(u)`` -> (n, 1)

    Attributes:
        _a : Alpha (shape) parameter, must be > 0.
        _b : Beta (shape) parameter, must be > 0.
        _rng: Random number generator.
        _mean: Mean vector, shape (1,).
        _cov: Covariance matrix, shape (1, 1).
    """


    def __init__(
        self,
        alpha: float,
        beta: float,
        *,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initializes a Beta distribution.

        Args:
            alpha: alpha > 0, the first shape parameter.
            beta: beta > 0, the second shape parameter.
            rng: Random generator. If ``None``,
                a default generator is created.

        Raises:
            ValueError: If ``alpha`` or ``beta`` is non-positive or non-finite.
        """

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
        """Draws random samples from the Beta distribution.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            Samples in [0, 1], shape (n_samples, 1).
        """
        x = self._beta.rvs(size=int(n_samples), random_state=self._rng)  # (n,) or scalar
        return np.asarray(x, dtype=float).reshape(-1, 1)

    def density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the probability density function (PDF).

        Args:
            values: Points in [0, 1] at which to evaluate.

        Returns:
            Column vector of PDF values, shape (n, 1).
        """
        v = _to_1d_vector(values)
        v = _clip_unit_interval(v)  # keep within [0,1]
        p = self._beta.pdf(v)       # (n,)
        return np.asarray(p, dtype=float).reshape(-1, 1)

    def log_density(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the log-PDF of the Beta distribution.

        Args:
            values: Points in [0, 1] at which to evaluate.

        Returns:
            Column vector of log-PDF values, shape (n, 1).
        """
        v = _to_1d_vector(values)
        v = _clip_unit_interval(v)  # keep within [0,1]
        lp = self._beta.logpdf(v)   # (n,)
        return np.asarray(lp, dtype=float).reshape(-1, 1)

    def cdf(self, values: NDArray) -> NDArray[np.floating]:
        """Evaluates the cumulative distribution function (CDF).

        Args:
            values: Points in [0, 1].

        Returns:
            Column vector of CDF values, shape (n, 1).
        """
        v = _to_1d_vector(values)
        v = _clip_unit_interval(v)
        c = self._beta.cdf(v)
        return np.asarray(c, dtype=float).reshape(-1, 1)

    def inv_cdf(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        """Computes the inverse CDF (quantile function).

        Args:
            u: Probabilities in [0, 1]; may be scalar, (n,), or (n, 1).

        Returns:
            Quantiles, shape (n, 1).

        Raises:
            ValueError: If input shape is invalid.
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
        """Estimates the expectation of a function under the Beta distribution.

        Uses Monte-Carlo sampling with Central-Limit approximation to
        summarize E[f(X)] as a Normal1D or MvNormal distribution.

        Behavior:
            - Scalar-valued ``f`` -> returns :class:`Normal1D`
            - Vector-valued ``f`` -> returns :class:`MvNormal`

        Args:
            func: Function mapping samples (n_mc, 1) -> outputs.

        Returns:
            Normal1D or MvNormal summarizing the mean and uncertainty.

        Raises:
            ValueError: If ``func`` output shape is unsupported.
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
        """Placeholder for fitting a Beta distribution from samples."""
        raise NotImplementedError


    def mean(self) -> NDArray[np.floating]:
        """Returns the mean vector (1,)."""
        return self._mean  # (1,)

    def cov(self) -> NDArray[np.floating]:
        """Returns the covariance matrix (1 × 1)."""
        return self._cov   # (1,1)


