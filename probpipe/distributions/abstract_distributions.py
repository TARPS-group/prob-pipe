from typing import Generic, TypeVar, Callable, Any
from abc import ABC, abstractmethod
import pytensor.tensor as pt
import numpy as np
from numpy.typing import NDArray

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




    





    

    










