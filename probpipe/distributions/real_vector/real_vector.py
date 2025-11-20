# distributions/real_vector.py
from __future__ import annotations

import numpy as np
import scipy.stats as sp
from typing import TypeVar, Any
from collections.abc import Callable
from abc import ABC, abstractmethod
from scipy.stats import(
    multivariate_normal,
    multinomial as _multinomial,
    dirichlet as _dirichlet,
    binom as _sbinom,
    beta as _sps_beta
)

from ...custom_types import Array, ArrayLike, Float, PRNG
from ...array_backend.utils import _ensure_matrix
from ...linalg.utils import symmetrize_pd
from ..distribution import Distribution


__all__ = [
    "RealVectorDistribution",
    "GaussianKDE",
    "Multinomial",
    "Dirichlet",
    "Binomial",
    "Beta",
]

T = TypeVar("T", bound=np.number)
FloatT = TypeVar("FloatT", bound=Float)


class RealVectorDistribution(Distribution[FloatT], ABC):
    """
    Abstract base for RealVectorDistribution, real-valued vector distributions with fixed dimension d.
    Event shape is assumed to be (d,). Subclasses should ensure consistency of shapes.
    """

    @abstractmethod
    def mean(self) -> Array[FloatT]:
        """
        Return the mean vector μ with shape (d,).
        If the mean does not exist (e.g., Cauchy), raise NotImplementedError.
        """
        raise NotImplementedError

    @abstractmethod
    def cov(self) -> Array[FloatT]:
        """
        Return the covariance matrix Σ with shape (d, d).
        If covariance does not exist, raise NotImplementedError.
        """
        raise NotImplementedError

    @abstractmethod
    def cdf(self, x: Array[FloatT]) -> Array[FloatT]:
        """
        Joint CDF F(x) = P[X1 ≤ x1, ..., Xd ≤ xd].
        Accepts x with shape (..., d) and returns shape (...,).
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
    def inv_cdf(self, u: Array[FloatT]) -> Array[FloatT]:
        """
        Inverse CDF (Rosenblatt inverse) mapping u ∈ (0,1)^d to x ∈ R^d.
        Accepts u with shape (..., d) and returns x with shape (..., d).
        For elliptical families (e.g., MVN), a common implementation is:
          z = Φ^{-1}(u)  (componentwise univariate inverse CDF)
          x = μ + L z     (L is Cholesky factor of Σ)
        For general dependent structures, implement via conditional quantiles/copulas.
        """
        raise NotImplementedError

    @property
    def dim(self) -> int:
        """
        Number of coordinates d. Default infers from mean(). Subclasses may override.
        """
        m = self.mean()
        if m.ndim != 1:
            raise ValueError("mean() must return a 1D array of shape (d,).")
        return int(m.shape[0])


class GaussianKDE(RealVectorDistribution[Float]):
    """
    Gaussian KDE with a shared bandwidth matrix H.

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
        samples: Array[Float],
        weights: Array[Float] | None = None,
        *,
        bandwidth: Float | Array[Float] | None = None,
        rule: str = "scott",
        rng: PRNG | None = None,
    ):
        """Initializes the Gaussian KDE.

        Args:
            samples: Input samples of shape (n, d).
            weights: Optional nonnegative weights, shape (n,).
                Will be normalized to sum to 1. Defaults to uniform weights.
            bandwidth (float | Array | None): Bandwidth specification.
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

        X = _ensure_matrix(samples, as_row_matrix=True)
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
        self._H = symmetrize_pd(self._H)

        # Pre-create SciPy kernels for each center (mean = x_i, cov = H)
        self._kernel_cls = multivariate_normal
        self._kernels = [multivariate_normal(mean=self._X[i], cov=self._H, allow_singular=False)
                         for i in range(self._n)]

        # Precompute mixture covariance = Cov(X) + H
        self._cov_mix = self._cov_x + self._H

    # --------------------------- bandwidth helpers ---------------------------

    def _build_H(self, bandwidth: Float | Array[Float] | None, rule: str) -> Array[Float]:
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


    def sample(self, n_samples: int = 1) -> Array[Float]:
        """
        Draw (n, d) samples from the KDE mixture:
          pick a center with prob w_i, then add N(0, H) noise.
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

    def density(self, x: ArrayLike) -> Array[Float]:
        """
        Mixture pdf: sum_i w_i * N(values | mean=x_i, cov=H).
        Returns (n, 1).
        """
        Xq = _ensure_matrix(x, as_row_matrix=True, num_cols=self.dim)  # (n, d)
        out = np.empty(Xq.shape[0], dtype=float)
        for j, q in enumerate(Xq):
            # Weighted sum of kernel PDFs at q
            out[j] = float(np.dot(self._w, [k.pdf(q) for k in self._kernels]))
        return out.reshape(-1, 1)  # (n,1)

    def log_density(self, x: ArrayLike) -> Array[Float]:
        """
        Stable log pdf via log-sum-exp over kernels: log ∑ w_i exp(log N_i(q)).
        Returns (n, 1).
        """
        Xq = _ensure_matrix(x, as_row_matrix=True, num_cols=self.dim)  # (n, d)
        out = np.empty(Xq.shape[0], dtype=float)
        for j, q in enumerate(Xq):
            logs = np.array([np.log(self._w[i] + 1e-300) + k.logpdf(q) for i, k in enumerate(self._kernels)],
                            dtype=float)
            m = logs.max()
            out[j] = m + np.log(np.exp(logs - m).sum())
        return out.reshape(-1, 1)

    def mean(self) -> Array[Float]:
        """Mixture mean = weighted mean of centers."""
        return self._mean  # (d,)

    def cov(self) -> Array[Float]:
        """Mixture covariance = Cov_w(centers) + H."""
        return self._cov_mix  # (d,d)

    def cdf(self, x: Array) -> Array[Float]:
        """
        Mixture CDF via SciPy MVN CDF per-kernel:
          F(q) = ∑_i w_i * Φ_d(q; mean=x_i, cov=H).
        Returns (n, 1).
        """
        Xq = _ensure_matrix(x, as_row_matrix=True, num_cols=self.dim) # (n, d)
        out = np.empty(Xq.shape[0], dtype=float)
        for j, q in enumerate(Xq):
            out[j] = float(np.dot(self._w, [k.cdf(q) for k in self._kernels]))
        return out.reshape(-1, 1)

    def inv_cdf(self, u: Array[Float]) -> Array[Float]:
        """
        No simple Rosenblatt inverse for Gaussian mixtures.
        """
        raise NotImplementedError("GaussianKDE.inv_cdf is not available for kernel mixtures.")

    def expectation(self, func: Callable[[Array[Float]], Array]) -> RealVectorDistribution:
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
    def from_distribution(cls, convert_from: Distribution, num_samples: int=1024, *, 
                          conversion_by_KDE: bool = False, **fit_kwargs: Any) -> GaussianKDE:
        """
        Build KDE from another distribution-like object.

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


class Multinomial(RealVectorDistribution[Float]):
    """
    Multinomial(n_trials, p) with event dim d = len(p).
    Backed by scipy.stats.multinomial.

    Shape policy:
        - ``sample(n)`` -> (n, d)  (counts per category)
        - ``density`` / ``log_density`` / ``cdf`` -> (n, 1)
          (scalar-per-sample outputs returned as column vectors)
        - ``inv_cdf(u)`` -> not implemented (no closed-form inverse)

    Notes
    -----
    * Counts are integers but returned as float arrays to satisfy RealVectorDistribution[Float] typing.
      Cast to int if you need integer dtype.
    * `cdf` is estimated by Monte-Carlo (parameter `cdf_mc_samples`); exact CDF is combinatorial.
    """

    def __init__(
        self,
        n_trials: int,
        probs: Array[Float],
        *,
        rng: PRNG | None = None,
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

    # ------------------------ RealVectorDistribution core ------------------------

    def sample(self, n_samples: int = 1) -> Array[Float]:
        """
        Draw (n, d) samples of counts; cast to float for typing consistency.
        """
        X = self._mn.rvs(size=int(n_samples), random_state=self._rng)  # (n,d) or (d,) if n=1
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X  # (n, d) float

    def density(self, x: Array) -> Array[Float]:
        """
        Return (n,1) column of pmf at each row in `values`.
        Inputs may be float; they must represent integer counts that sum to n_trials.
        """
        X = _ensure_matrix(x)  # (n,d)
        self._validate_counts_rows(X)
        # SciPy's logpmf/pmf can handle arrays row-wise via list comprehension
        pmfs = [self._mn.pmf(row.astype(int, copy=False)) for row in X]
        return np.asarray(pmfs, dtype=float).reshape(-1, 1)

    def log_density(self, x: Array) -> Array[Float]:
        """
        Return (n,1) column of log-pmf at each row in `values`.
        """
        X = _ensure_matrix(x)  # (n,d)
        self._validate_counts_rows(X)
        logpmfs = [self._mn.logpmf(row.astype(int, copy=False)) for row in X]
        return np.asarray(logpmfs, dtype=float).reshape(-1, 1)

    def expectation(self, func: Callable[[Array[Float]], Array]) -> RealVectorDistribution:
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
    def from_distribution(cls, convert_from: Distribution, num_samples: int=1024, *, 
                          conversion_by_KDE: bool = False, **fit_kwargs: Any) -> Multinomial:
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

        X = _ensure_matrix(X, as_row_matrix=True)  # (n, d)
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


    def mean(self) -> Array[Float]:
        return self._mean  # (d,)

    def cov(self) -> Array[Float]:
        return self._cov   # (d,d)

    def cdf(self, x: Array) -> Array[Float]:
        """
        Monte-Carlo approximation of P[X1<=x1, ..., Xd<=xd].
        Returns (n,1). For exact CDF (small n,d), implement a specialized enumerator.
        """
        if self._cdf_mode != "mc":
            raise NotImplementedError("Only Monte-Carlo CDF ('mc') is supported for Multinomial.")
        Xq = _ensure_matrix(x)  # (n, d)
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

    def inv_cdf(self, u: Array[Float]) -> Array[Float]:
        """
        No practical Rosenblatt inverse for a general Multinomial.
        """
        raise NotImplementedError("Multinomial.inv_cdf is not available.")

    # ------------------------ helper ------------------------

    def _validate_counts_rows(self, X: Array[Float]) -> None:
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


class Dirichlet(RealVectorDistribution[Float]):
    """
    Dirichlet(α) on the probability simplex.

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
        alpha: Array[Float] | Array[Float],
        *,
        rng: PRNG | None = None,
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


    def sample(self, n_samples: int = 1) -> Array[Float]:
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

    def density(self, values: Array) -> Array[Float]:
        """Evaluates the probability density function (PDF).

        Args:
            values: Points on the simplex, shape (n, d) or (d,).

        Returns:
            Column vector of PDF values, shape (n, 1).

        Raises:
            ValueError: If inputs do not lie on the simplex.
        """
        
        X = _ensure_matrix(values, as_row_matrix=True)                 # (n, d)
        self._validate_simplex_rows(X)
        # Row-wise evaluation (SciPy supports vectorization, but this is explicit & safe)
        pmf = [self._dir.pdf(row) for row in X]
        return np.asarray(pmf, dtype=float).reshape(-1, 1)   # (n,1)

    def log_density(self, values: Array) -> Array[Float]:
        """Evaluates the log-probability density function (log-PDF).

        Args:
            values: Points on the simplex, shape (n, d) or (d,).

        Returns:
            Column vector of log-PDF values, shape (n, 1).

        Raises:
            ValueError: If inputs do not lie on the simplex.
        """
        
        X = _ensure_matrix(values, as_row_matrix=True)
        self._validate_simplex_rows(X)
        lpmf = [self._dir.logpdf(row) for row in X]
        return np.asarray(lpmf, dtype=float).reshape(-1, 1)  # (n,1)

    def expectation(self, func: Callable[[Array[Float]], Array]) -> RealVectorDistribution:
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

        X = _ensure_matrix(X, as_row_matrix=True)  # (n, d)

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


    def mean(self) -> Array[Float]:
        """Returns the mean vector.

        Returns:
            Mean vector of shape (d,).
        """
        return self._mean  # (d,)

    def cov(self) -> Array[Float]:
        """Returns the covariance matrix.

        Returns:
            Covariance matrix of shape (d, d).
        """
        return self._cov   # (d,d)

    def cdf(self, values: Array) -> Array[Float]:
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
        Xq = _ensure_matrix(values, as_row_matrix=True)     # (n, d)
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

    def inv_cdf(self, u: Array[Float]) -> Array[Float]:
        """Inverse CDF is not available for Dirichlet distributions."""
        raise NotImplementedError("Dirichlet.inv_cdf is not available.")

    # ------------------------ helpers ------------------------

    def _validate_simplex_rows(self, X: Array[Float]) -> None:
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


class Binomial(RealVectorDistribution[Float]):
    """
    Binomial(n_trials, p) with event dim = 1 (counts of successes).

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
        rng: PRNG | None = None,
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

    # ------------------------ RealVectorDistribution core ------------------------

    def sample(self, n_samples: int) -> Array[Float]:
        """Draws samples of Binomial counts.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            Array of success counts, shape (n_samples, 1).
        """
        x = self._binom.rvs(size=int(n_samples), random_state=self._rng)  # (n,) or scalar
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return x  # (n,1)

    def density(self, values: Array) -> Array[Float]:
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

    def log_density(self, values: Array) -> Array[Float]:
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

    def cdf(self, values: Array) -> Array[Float]:
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

    def inv_cdf(self, u: Array[Float]) -> Array[Float]:
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


    def expectation(self, func: Callable[[Array[Float]], Array]) -> 'RealVectorDistribution':
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
        """Fit not implemented for Binomial distributions."""
   
        raise NotImplementedError


    def mean(self) -> Array[Float]:
        """Returns the mean vector (1,)."""
        return self._mean  # (1,)

    def cov(self) -> Array[Float]:
        """Returns the covariance matrix (1×1)."""
        return self._cov   # (1,1)

    # ------------------------ helpers ------------------------

    def _to_1d_counts(self, values: Array) -> Array[Float]:
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

    
def _clip_unit_interval(x: Array[Float], eps: float = 0.0) -> Array[Float]:
    """Clips values to the [0, 1] interval, optionally padding to an open range.

    Args:
        x: Values to clip.
        eps: If 0, clips to [0, 1]. If >0, clips to
            (eps, 1 − eps) using `np.nextafter` to avoid exact endpoints.
            Defaults to 0.0.

    Returns:
        Array with clipped values.
    """
    if eps <= 0.0:
        return np.clip(x, 0.0, 1.0)
    lo = np.nextafter(0.0 + eps, 1.0)
    hi = np.nextafter(1.0 - eps, 0.0)
    return np.clip(x, lo, hi)


class Beta(RealVectorDistribution[Float]):
    """
    Beta(alpha, β) distribution on [0, 1].

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
        rng: PRNG | None = None,
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

    # ------------------------ RealVectorDistribution core ------------------------

    def sample(self, n_samples: int) -> Array[Float]:
        """Draws random samples from the Beta distribution.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            Samples in [0, 1], shape (n_samples, 1).
        """
        x = self._beta.rvs(size=int(n_samples), random_state=self._rng)  # (n,) or scalar
        return np.asarray(x, dtype=float).reshape(-1, 1)

    def density(self, values: Array) -> Array[Float]:
        """Evaluates the probability density function (PDF).

        Args:
            values: Points in [0, 1] at which to evaluate.

        Returns:
            Column vector of PDF values, shape (n, 1).
        """
        v = _ensure_vector(values)
        v = _clip_unit_interval(v)  # keep within [0,1]
        p = self._beta.pdf(v)       # (n,)
        return np.asarray(p, dtype=float).reshape(-1, 1)

    def log_density(self, values: Array) -> Array[Float]:
        """Evaluates the log-PDF of the Beta distribution.

        Args:
            values: Points in [0, 1] at which to evaluate.

        Returns:
            Column vector of log-PDF values, shape (n, 1).
        """
        v = _ensure_vector(values)
        v = _clip_unit_interval(v)  # keep within [0,1]
        lp = self._beta.logpdf(v)   # (n,)
        return np.asarray(lp, dtype=float).reshape(-1, 1)

    def cdf(self, values: Array) -> Array[Float]:
        """Evaluates the cumulative distribution function (CDF).

        Args:
            values: Points in [0, 1].

        Returns:
            Column vector of CDF values, shape (n, 1).
        """
        v = _ensure_vector(values)
        v = _clip_unit_interval(v)
        c = self._beta.cdf(v)
        return np.asarray(c, dtype=float).reshape(-1, 1)

    def inv_cdf(self, u: Array[Float]) -> Array[Float]:
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

    def expectation(self, func: Callable[[Array[Float]], Array]) -> RealVectorDistribution:
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
    def from_distribution(cls, convert_from: Distribution, num_samples: int=1024, *, 
                          conversion_by_KDE: bool = False, **fit_kwargs: Any) -> Beta:
        """Placeholder for fitting a Beta distribution from samples."""
        raise NotImplementedError


    def mean(self) -> Array[Float]:
        """Returns the mean vector (1,)."""
        return self._mean  # (1,)

    def cov(self) -> Array[Float]:
        return self._cov   # (1,1)
