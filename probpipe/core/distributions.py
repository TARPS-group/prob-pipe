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
    Abstract base class for probability distributions.

    This class defines the general interface for any probabilistic
    distribution implementation used within probpipe. Subclasses are
    expected to implement methods for computing density, log-density,
    sampling, and expectations.

    Subclasses that cannot support a specific operation (e.g., sampling)
    may leave that method unimplemented.

    Type Variables:
        T: Numeric data type (e.g., float or np.floating).
    """

    def sample(self, n_samples: int) -> NDArray[T]:
        """
        Samples data points from the distribution.

        This method may be optionally implemented by subclasses that
        support random sampling. It returns `n_samples` draws from
        the distribution.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            NDArray[T]: An array containing `n_samples` draws from the distribution.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    
    def density(self, data: NDArray) -> NDArray[np.floating]:
        """
        Computes the probability density p(data) under this distribution.

        This method may be optionally implemented by subclasses that
        can evaluate densities. The returned array corresponds to the
        pointwise probability density values reduced over event
        dimensions (i.e., matching the batch shape).

        Args:
            data: Input array of observations for which to compute densities.

        Returns:
            NDArray[np.floating]: Probability density values for each input point.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    
    def log_density(self, data: NDArray) -> NDArray[np.floating]:
        """
        Computes the log-probability density log p(data).

        This method may be optionally implemented by subclasses that
        can evaluate log-densities. The returned array corresponds to
        the pointwise log-probability values reduced over event
        dimensions (i.e., matching the batch shape).

        Args:
            data: Input array of observations for which to compute log-densities.

        Returns:
            NDArray[np.floating]: Log-probability values for each input point.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    

    def expectation(self, func: Callable[[NDArray[T]], NDArray]) -> 'Distribution':
        """
        Computes the expectation of a function under this distribution.

        This method may be optionally implemented by subclasses that can
        perform Monte Carlo estimation. The result is typically a new
        `Distribution` instance representing the mean of the function
        applied to random samples drawn from this distribution.

        Args:
            func: A callable function f(x) to compute the expectation of.

        Returns:
            Distribution: A distribution representing E[f(X)].

        Raises:
            NotImplementedError: If the subclass does not implement this method.
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
        Constructs a new distribution by fitting or converting from another.

        This method defines how a subclass converts or fits itself from an
        existing `Distribution` instance — for example, transforming an empirical
        distribution into a parametric approximation (e.g., Gaussian KDE or
        Normal fit).

        Args:
            convert_from: The source distribution to fit or convert from.
            **fit_kwargs: Additional fitting parameters specific to the subclass.

        Returns:
            Distribution[T]: A new instance of `cls` fitted to the source distribution.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("This method should be implemented by subclasses")


from .multivariate import Normal1D, MvNormal

class EmpiricalDistribution(Distribution):
    """
    Container for weighted empirical samples in ℝᵈ.

    Represents a discrete empirical distribution defined by a set of
    weighted samples, typically used to store MCMC draws or Monte Carlo
    samples. Supports weighted summary statistics, resampling, and
    expectation estimation.

    Attributes:
        n (int): Number of stored samples.
        d (int): Dimensionality of the sample space.
        samples (NDArray): Stored draws of shape (n, d).
        weights (NDArray): Normalized nonnegative weights summing to 1.
        rng (np.random.Generator): Random number generator used for resampling.
    """

    def __init__(
        self,
        samples: NDArray,
        weights: Optional[NDArray] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initializes an EmpiricalDistribution from weighted samples.

        Args:
            samples (NDArray): Array of stored draws with shape (n, d) or (n,).
            weights (Optional[NDArray]): Optional array of nonnegative weights
                of shape (n,). If ``None``, uniform weights are assigned.
            rng (Optional[np.random.Generator]): Optional random number generator
                used for resampling. If ``None``, a default generator is created.

        Raises:
            ValueError: If the number of samples is less than one.
            ValueError: If ``weights`` has invalid shape or negative entries.
            ValueError: If weights sum to a nonpositive value.
        """
        
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
        """int: Number of stored samples."""
        return self._n

    @property
    def d(self) -> int:
        """int: Dimensionality of the stored samples."""
        return self._d

    @property
    def samples(self) -> NDArray:
        """NDArray: View of stored samples with shape (n, d)."""
        return self._X

    @property
    def weights(self) -> NDArray:
        """NDArray: Normalized sample weights with shape (n,)."""
        return self._w

    def mean(self) -> NDArray:
        """Computes the weighted mean of the samples.

        Returns:
            NDArray: Weighted mean vector of shape (d,).
        """
        return self._mean

    def cov(self) -> NDArray:
        """Computes the weighted population covariance.

        Returns:
            NDArray: Weighted covariance matrix of shape (d, d).
        """
        return self._cov

    def var(self) -> NDArray:
        """Computes the weighted population variance per dimension.

        Returns:
            NDArray: Variance vector of shape (d,).
        """
        return np.diag(self._cov)

    def std(self) -> NDArray:
        """Computes the weighted population standard deviation per dimension.

        Returns:
            NDArray: Standard deviation vector of shape (d,).
        """
        return np.sqrt(np.maximum(self.var(), 0.0))

    def sample(self, n_samples: int, *, replace: bool = True) -> NDArray:
        """Resamples draws from the empirical distribution.

        Performs weighted sampling (with or without replacement) from
        the stored empirical samples.

        Args:
            n_samples (int): Number of draws to generate.
            replace (bool): Whether to sample with replacement. Defaults to True.

        Returns:
            NDArray: Resampled points of shape (n_samples, d).

        Raises:
            ValueError: If ``replace=False`` and ``n_samples > n``.
        """
        n_samples = int(n_samples)
        if not replace and n_samples > self._n:
            raise ValueError("Cannot sample more than n without replacement.")
        idx = self._rng.choice(self._n, size=n_samples, replace=replace, p=self._w)
        return self._X[idx]

    rvs = sample

    def density(self, data: NDArray) -> NDArray:
        """Evaluates the sample-based density estimate at given points.

        Note:
            Density estimation is not implemented for purely empirical
            (discrete) samples without a kernel model.

        Args:
            data (NDArray): Input points at which to estimate density.

        Raises:
            NotImplementedError: Always raised, as density is not implemented.
        """
        raise NotImplementedError("Density not implemented for EmpiricalDistribution.")

    def log_density(self, data: NDArray) -> NDArray:
        """Evaluates the log-density estimate at given points.

        Note:
            Log-density estimation is not implemented for purely empirical
            (discrete) samples without a kernel model.

        Args:
            data (NDArray): Input points at which to estimate log-density.

        Raises:
            NotImplementedError: Always raised, as log-density is not implemented.
        """
        raise NotImplementedError("Log density not implemented for EmpiricalDistribution.")

    def expectation(
        self,
        func: Callable[[NDArray], NDArray],
        *,
        n_mc: int = 2048,
    ) -> Union["Normal1D", "MvNormal"]:
        """Computes the weighted Monte Carlo expectation of a function.

        Uses the stored samples and weights to estimate E[f(X)] and the
        associated Monte Carlo uncertainty. Returns a Normal1D or MvNormal
        summarizing the mean and variance (or covariance) of the estimate.

        Args:
            func (Callable[[NDArray], NDArray]): Function mapping samples
                to numeric outputs.
            n_mc (int): Number of Monte Carlo draws used to estimate sampling
                error. Defaults to 2048.

        Returns:
            Union[Normal1D, MvNormal]: Distribution summarizing the expectation
            and its uncertainty.
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

        """Constructs an empirical distribution from another distribution.

        Draws samples from the given source distribution and wraps them
        into an `EmpiricalDistribution` object.

        Args:
            convert_from (Distribution): Source distribution to sample from.
            **fit_kwargs: Optional keyword arguments, such as
                ``num_samples`` (int), specifying how many draws to take.

        Returns:
            EmpiricalDistribution: New instance containing sampled points.
        """

        samples = convert_from.sample(fit_kwargs.get("num_samples", 2048))
        return cls(samples)
        

