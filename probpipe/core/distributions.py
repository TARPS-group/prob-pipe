from typing import Generic, TypeVar, Callable, Any, Optional, Union
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ._utils import _as_2d

__all__ = [
    "Distribution",
    "EmpiricalDistribution",
    "BootstrapDistribution",
]

T = TypeVar("T",bound=np.number)
Float_T = TypeVar("FloatDT", bound=np.floating)


# -------------------------- Abstract Classes ----------------------------


class Distribution(Generic[T], ABC):
    """
    Abstract base class for any distribution class.
    """

    def sample(self, n_samples: int) -> NDArray[T]:
        """
        Optional. If a subclass can’t sample, it may leave this unimplemented.

        Sample n_samples items from the distribution.
        Returns a ndarray of T.
        """
        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    
    def density(self, data: NDArray) -> NDArray[np.floating]:
        """
        Optional. If a subclass can’t sample, it may leave this unimplemented.

        Compute p(data) under this distribution.
        Returns a ndarray of prob values reduced over event dims
        (i.e., shape matching batch shape).
        """
        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    
    def log_density(self, data: NDArray) -> NDArray[np.floating]:
        """
        Optional. If a subclass can’t sample, it may leave this unimplemented.

        Compute log p(data) under this distribution.
        Returns a ndarray of log-prob values reduced over event dims
        (i.e., shape matching batch shape).
        """
        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    

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
        convert_from: 'Distribution', 
        **fit_kwargs: Any,
    ) -> 'Distribution[T]':
        """
        Fit/convert from an empirical distribution to this parametric family.
        Typical implementations perform Gaussian KDE or Gaussian approx and return an instance of `cls`.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

from .multivariate import Normal1D, MvNormal

class EmpiricalDistribution(Distribution):
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
    Implements Distribution interface.
    """

    def __init__(
        self,
        samples: np.ndarray,
        weights: Optional[np.ndarray] = None,
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

    @property
    def n(self) -> int:
        """Number of stored samples."""
        return self._n

    @property
    def d(self) -> int:
        """Dimensionality."""
        return self._d

    @property
    def samples(self) -> np.ndarray:
        """A view of the stored samples, shape (n, d)."""
        return self._X

    @property
    def weights(self) -> np.ndarray:
        """A view of normalized weights, shape (n,)."""
        return self._w

    def mean(self) -> np.ndarray:
        """Weighted mean, shape (d,)."""
        return self._mean

    def cov(self) -> np.ndarray:
        """Weighted *population* covariance, shape (d, d)."""
        return self._cov

    def var(self) -> np.ndarray:
        """Weighted population variance per dimension, shape (d,)."""
        return np.diag(self._cov)

    def std(self) -> np.ndarray:
        """Weighted population standard deviation per dimension, shape (d,)."""
        return np.sqrt(np.maximum(self.var(), 0.0))

    def sample(self, n_samples: int, *, replace: bool = True) -> np.ndarray:
        """
        Resample draws from the empirical distribution with replacement,
        using the stored weights. Returns shape (n_samples, d).
        """
        n_samples = int(n_samples)
        if not replace and n_samples > self._n:
            raise ValueError("Cannot sample more than n without replacement.")
        idx = self._rng.choice(self._n, size=n_samples, replace=replace, p=self._w)
        return self._X[idx]

    rvs = sample

    def density(self, data: np.ndarray) -> np.ndarray:
        """
        Return the (unnormalized) sample density estimate at `data`.

        Notes
        -----
        Not implemented for empirical distributions without a kernel model.
        """
        raise NotImplementedError("Density not implemented for EmpiricalDistribution.")

    def log_density(self, data: np.ndarray) -> np.ndarray:
        """
        Return log-density estimate at `data`.

        Notes
        -----
        Not implemented for empirical distributions without a kernel model.
        """
        raise NotImplementedError("Log density not implemented for EmpiricalDistribution.")

    def expectation(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        *,
        n_mc: int = 2048,
    ) -> Union["Normal1D", "MvNormal"]:
        """
        Compute the weighted Monte Carlo expectation of a function under the
        empirical distribution.

        Parameters
        ----------
        func : callable
            Function mapping samples (array of shape (n, d)) to values.
        n_mc : int, default=2048
            Number of Monte Carlo draws used to estimate sampling error.

        Returns
        -------
        Normal1D or MvNormal
            Distribution summarizing the mean and Monte Carlo standard error
            (1D) or covariance (multivariate) of `func(X)`.
        """
        
        Y = np.asarray(func(self._X), dtype=float)

        if Y.ndim == 1:
            m = float((self._w * Y).sum())
            var = float((self._w * (Y - m) ** 2).sum())
            se = np.sqrt(max(var, 0.0)) / np.sqrt(n_mc)
            return Normal1D(m, max(se, 1e-12), rng=self._rng)
        else:
            Y = _as_2d(Y)
            m = (self._w[:, None] * Y).sum(axis=0)
            diff = Y - m
            cov = diff.T @ (diff * self._w[:, None])
            cov_mean = cov / float(n_mc)
            cov_mean = 0.5 * (cov_mean + cov_mean.T) + 1e-12 * np.eye(cov_mean.shape[0])
            return MvNormal(mean=m, cov=cov_mean, rng=self._rng)

    @classmethod
    def from_distribution(
        cls,
        convert_from: 'Distribution',
        **fit_kwargs: Any,
    ) -> 'EmpiricalDistribution':

        """
        Draw samples from another distribution and wrap them as an
        EmpiricalDistribution.

        Parameters
        ----------
        convert_from : Distribution
            Source distribution to sample from.
        **fit_kwargs
            Optional keyword arguments; may include ``num_samples`` (int)
            specifying the number of draws.

        Returns
        -------
        EmpiricalDistribution
            New empirical distribution constructed from sampled points.
        """

        samples = convert_from.sample(fit_kwargs.get("num_samples", 2048))
        return cls(samples)
        

