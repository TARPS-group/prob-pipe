
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
    
