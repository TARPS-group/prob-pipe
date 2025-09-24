
from typing import Generic, TypeVar, Callable, Any, Optional, Union
from abc import ABC, abstractmethod
import pytensor.tensor as pt
import numpy as np
from numpy.typing import NDArray
from probpipe.core.distributions.dist_utils import _as_2d, _symmetrize_spd
#from probpipe.core.distributions.multivariate import MvNormal, Normal1D


T = TypeVar("T",bound=np.number)
#T=float, int, complex
Float_T = TypeVar("FloatDT", bound=np.floating)


# -------------------------- Abstract Classes ----------------------------


class Distribution(Generic[T], ABC):
    """
    Abstract base class for any distribution class.
    """

    #@abstractmethod
    def sample(self, n_samples: int) -> NDArray[T]:
        """
        Optional. If a subclass can’t sample, it may leave this unimplemented.

        Sample n_samples items from the distribution.
        Returns a ndarray of T.
        """
        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    
    #@abstractmethod
    def density(self, data: NDArray) -> NDArray[np.floating]:
        """
        Optional. If a subclass can’t sample, it may leave this unimplemented.

        Compute p(data) under this distribution.
        Returns a ndarray of prob values reduced over event dims
        (i.e., shape matching batch shape).
        """
        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    
    #@abstractmethod
    def log_density(self, data: NDArray) -> NDArray[np.floating]:
        """
        Optional. If a subclass can’t sample, it may leave this unimplemented.

        Compute log p(data) under this distribution.
        Returns a ndarray of log-prob values reduced over event dims
        (i.e., shape matching batch shape).
        """
        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    
    #func: Callable[[T], NDArray]
    #@abstractmethod
    def expectation(self, func: Callable[[NDArray[T]], NDArray]) -> 'Distribution':
        """
        Optional. If a subclass can’t sample, it may leave this unimplemented.

        Monte-Carlo sample from self, compute f(x) and return a Distribution over the mean of f.
        """

        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    

    @classmethod
    @abstractmethod
    def from_distribution(
        cls,
        convert_from: 'Distribution', #OR 'Distribution[T]'
        **fit_kwargs: Any,
    ) -> 'Distribution[T]':
        """
        Fit/convert from an empirical distribution to this parametric family.
        Typical implementations perform Gaussian KDE or Gaussian approx and return an instance of `cls`.
        """
        raise NotImplementedError("This method should be implemented by subclasses")


class EmpiricalDistribution:
    """
    Generic container for (weighted) empirical samples in R^d.
    Intended for storing MCMC draws (or any Monte Carlo samples).
    
    Parameters
    ----------
    samples : array-like, shape (n, d) or (n,)
        Stored draws.
    weights : array-like, shape (n,), optional
        Nonnegative weights; will be normalized to sum to 1. If None, uniform.
    rng : np.random.Generator, optional
        RNG used for resampling.

    Notes
    -----
    - This class does NOT inherit from your `Distribution`/`Multivariate` bases.
      Parametric classes use `from_distribution(empirical, ...)` to fit/convert.
    - Methods provided: sample (resample), mean, cov, var/std,
      expectation (numeric estimate and optionally a Normal1D/MvNormal over the MC mean).
    """

    def __init__(
        self,
        samples: NDArray[np.floating],
        weights: Optional[NDArray[np.floating]] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ):
        X = _as_2d(samples)
        n, d = X.shape
        if n < 1:
            raise ValueError("Empirical requires at least one sample.")

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
        self._n = int(n)
        self._d = int(d)
        self._rng = rng or np.random.default_rng()

        # Precompute weighted mean & population covariance (no ddof correction)
        self._mean = (self._w[:, None] * self._X).sum(axis=0)
        diff = self._X - self._mean
        self._cov = diff.T @ (diff * self._w[:, None])

        # cumulative weights for fast inverse-transform resampling
        self._cw = np.cumsum(self._w)

    # ------------------- basic properties -------------------

    @property
    def n(self) -> int:
        """Number of stored samples."""
        return self._n

    @property
    def d(self) -> int:
        """Dimensionality."""
        return self._d

    @property
    def samples(self) -> NDArray[np.floating]:
        """A view of the stored samples, shape (n, d)."""
        return self._X

    @property
    def weights(self) -> NDArray[np.floating]:
        """A view of normalized weights, shape (n,)."""
        return self._w

    # ------------------- summaries -------------------

    def mean(self) -> NDArray[np.floating]:
        """Weighted mean, shape (d,)."""
        return self._mean

    def cov(self) -> NDArray[np.floating]:
        """Weighted *population* covariance, shape (d, d)."""
        return self._cov

    def var(self) -> NDArray[np.floating]:
        """Weighted population variance per dimension, shape (d,)."""
        return np.diag(self._cov)

    def std(self) -> NDArray[np.floating]:
        """Weighted population standard deviation per dimension, shape (d,)."""
        return np.sqrt(np.maximum(self.var(), 0.0))


    # ------------------- resampling -------------------

    def sample(self, n_samples: int, *, replace: bool = True) -> NDArray[np.floating]:
        """
        Resample draws from the empirical distribution with (by default) replacement,
        using the stored weights. Returns shape (n_samples, d).
        """
        n_samples = int(n_samples)
        if not replace and n_samples > self._n:
            raise ValueError("Cannot sample more than n without replacement.")
        idx = self._rng.choice(self._n, size=n_samples, replace=replace, p=self._w)
        return self._X[idx]

    # alias
    rvs = sample

    # ------------------- expectation helpers -------------------

    def expectation(
        self,
        func: Callable[[NDArray[np.floating]], NDArray],
        *,
        n_mc: int = 2048,
    ) -> Union["Normal1D", "MvNormal"]:
        """
        Estimate E[f(X)] under the empirical law.

        scalar f: returns Normal1D(mean, std_error)
        vector f: returns MvNormal(mean, cov_of_mean)

        Notes:
          - We evaluate f on ALL stored samples once (vectorized), using the empirical
            weights to compute mean and (population) covariance of f(X).
          - The uncertainty reported corresponds to the mean of f over n_mc IID draws
            from the empirical distribution (CLT).
        """
        Y = np.asarray(func(self._X), dtype=float)

        if Y.ndim == 1:
            m = float((self._w * Y).sum())
            var = float((self._w * (Y - m) ** 2).sum())
            se = np.sqrt(max(var, 0.0)) / np.sqrt(n_mc)
            
            return Normal1D(m, max(se, 1e-12), rng=self._rng)
           
        else:
            Y = _as_2d(Y)  # (n, k)
            m = (self._w[:, None] * Y).sum(axis=0)  # (k,)
            diff = Y - m
            cov = diff.T @ (diff * self._w[:, None])  # (k, k) population cov of f(X)
            cov_mean = cov / float(n_mc)

            # small symmetrization + jitter for numerical stability
            cov_mean = 0.5 * (cov_mean + cov_mean.T) + 1e-12 * np.eye(cov_mean.shape[0])
            
            return MvNormal(mean=m, cov=cov_mean, rng=self._rng)
        


class BootstrapDistribution:
    """
    Container for bootstrap replicates in R^k (k = statistic dimension).

    Parameters
    ----------
    replicates : array-like, shape (B, k) or (B,)
        Bootstrapped statistic values (theta* draws).
    weights : array-like, shape (B,), optional
        Nonnegative replicate weights (rare for classic bootstrap). Will be normalized
        to sum to 1. If None, uniform 1/B.
    rng : np.random.Generator, optional
        RNG for resampling replicates via `sample()` / `rvs()`.

    Notes
    -----
    - This class is NOT a parametric distribution.
    - Mirrors your EmpiricalDistribution ergonomics: mean/cov/var/std, sample/rvs,
      and expectation() -> Normal1D / MvNormal over the Monte-Carlo mean of f(theta*).
    """

    # ------------------------------ init ------------------------------

    def __init__(
        self,
        replicates: NDArray[np.floating],
        weights: Optional[NDArray[np.floating]] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ):
        Theta = _as_2d(replicates)  # (B, k)
        B, k = Theta.shape
        if B < 1:
            raise ValueError("BootstrapDistribution requires at least one replicate.")

        if weights is None:
            w = np.full(B, 1.0 / B, dtype=float)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] != B:
                raise ValueError("weights must have shape (B,).")
            if np.any(w < 0):
                raise ValueError("weights must be nonnegative.")
            s = w.sum()
            if s <= 0:
                raise ValueError("weights must sum to a positive value.")
            w = w / s

        self._Theta = Theta.astype(float)   # (B, k)
        self._w = w.astype(float)           # (B,)
        self._B = int(B)
        self._k = int(k)
        self._rng = rng or np.random.default_rng()

        # Precompute weighted mean & population covariance on replicates
        self._mean = (self._w[:, None] * self._Theta).sum(axis=0)           # (k,)
        diff = self._Theta - self._mean
        self._cov = diff.T @ (diff * self._w[:, None])                      # (k, k)

        self._cw = np.cumsum(self._w)  # for inverse-transform resampling of replicates

    # ------------------------ basic properties ------------------------

    @property
    def n(self) -> int:
        """Number of bootstrap replicates (B)."""
        return self._B

    @property
    def d(self) -> int:
        """Dimensionality of statistic (k)."""
        return self._k

    @property
    def replicates(self) -> NDArray[np.floating]:
        """View of stored replicates, shape (B, k)."""
        return self._Theta

    @property
    def weights(self) -> NDArray[np.floating]:
        """View of normalized replicate weights, shape (B,)."""
        return self._w

    # --------------------------- summaries ----------------------------

    def mean(self) -> NDArray[np.floating]:
        """Weighted mean of replicates, shape (k,)."""
        return self._mean

    def cov(self) -> NDArray[np.floating]:
        """Weighted population covariance of replicates, shape (k, k)."""
        return self._cov

    def var(self) -> NDArray[np.floating]:
        """Weighted population variance of replicates, shape (k,)."""
        return np.diag(self._cov)

    def std(self) -> NDArray[np.floating]:
        """Weighted population standard deviation, shape (k,)."""
        return np.sqrt(np.maximum(self.var(), 0.0))

    # --------------------- resampling of replicates -------------------

    def sample(self, n_samples: int, *, replace: bool = True) -> NDArray[np.floating]:
        """
        Resample **replicates** (theta* values) with given weights.
        Returns shape (n_samples, k).
        """
        n_samples = int(n_samples)
        if not replace and n_samples > self._B:
            raise ValueError("Cannot sample more than B without replacement.")
        idx = self._rng.choice(self._B, size=n_samples, replace=replace, p=self._w)
        return self._Theta[idx]

    # alias
    rvs = sample

    # ------------------------- expectation ---------------------------

    def expectation(
        self,
        func: Callable[[NDArray[np.floating]], NDArray],
        *,
        n_mc: int = 2048,
    ) -> Union["Normal1D", "MvNormal"]:
        """
        Return a distribution over E[f(Theta*)] under the bootstrap law (on replicates).

        Scalar f -> Normal1D(mean, std_error)
        Vector f -> MvNormal(mean, cov_of_mean)

        where mean and (population) covariance are computed with replicate weights,
        and standard error / covariance-of-mean are scaled by 1/sqrt(n_mc) / 1/n_mc.
        """
        Y = np.asarray(func(self._Theta), dtype=float)

        if Y.ndim == 1:
            m = float((self._w * Y).sum())
            var = float((self._w * (Y - m) ** 2).sum())
            se = np.sqrt(max(var, 0.0)) / np.sqrt(n_mc)
            return Normal1D(m, max(se, 1e-12), rng=self._rng)
        else:
            Y = _as_2d(Y)  # (B, k2)
            m = (self._w[:, None] * Y).sum(axis=0)
            diff = Y - m
            cov = diff.T @ (diff * self._w[:, None])      # (k2, k2)
            cov_mean = 0.5 * (cov + cov.T) / float(n_mc)  # symmetrize & scale
            cov_mean += 1e-12 * np.eye(cov_mean.shape[0])
            return MvNormal(mean=m, cov=cov_mean, rng=self._rng)


    @classmethod
    def from_data(
        cls,
        data: NDArray[np.floating],
        stat_fn: Callable[[NDArray[np.floating]], NDArray],
        *,
        B: int = 1000,
        axis: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> "BootstrapDistribution":
        """
        Classic i.i.d. bootstrap for a statistic.

        Parameters
        ----------
        data : array-like
            Observations (samples along `axis`).
        stat_fn : callable
            Function mapping a resampled dataset (with samples on axis 0) to a
            statistic vector (shape (k,) or scalar).
            NOTE: we will pass the resampled array with **samples on axis 0**.
                  If your original data had samples on another axis, we move it here.
        B : int
            Number of bootstrap replicates.
        axis : int
            Axis of `data` that indexes samples; moved to 0 before calling `stat_fn`.
        rng : np.random.Generator, optional
            RNG for resampling indices.

        Returns
        -------
        BootstrapDistribution
            Container of `B` replicates of the statistic.
        """
        rng = rng or np.random.default_rng()
        X = np.asarray(data, dtype=float)
        X = np.moveaxis(X, axis, 0)  # samples now on axis 0
        n = X.shape[0]

        reps = []
        for _ in range(int(B)):
            idx = rng.integers(0, n, size=n)          # sample n rows with replacement
            Xb = X[idx]                                # (n, ...)
            theta = np.asarray(stat_fn(Xb), dtype=float).reshape(-1)
            reps.append(theta)

        Theta = np.vstack(reps)                        # (B, k)
        return cls(Theta, rng=rng)

    
