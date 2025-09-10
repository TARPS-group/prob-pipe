from typing import Generic, TypeVar, Callable, Any, Optional, Union, Tuple
from numpy.typing import NDArray
import numpy as np
import scipy.stats as sp

from scipy.spatial.distance import cdist


from probpipe.distributions.dist_utils import _as_2d, _symmetrize_spd
from probpipe.distributions.distributions import Distribution
from probpipe.distributions.multivariate import MvNormal    


T = TypeVar("T",bound=np.number)




class Normal(MvNormal):
    """
    Univariate Normal N(μ, σ²) as an MvNormal of dimension 1.
    Everything (sample, pdf, cdf, inv_cdf, ...) will automatically
    return shape (n,1).
    """
    def __init__(self,
                 mu: float,
                 sigma: float,
                 *,
                 rng: np.random.Generator | None = None):
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        # build the 1‐dim mean and covariance
        mean = np.array([mu], dtype=float)        # shape (1,)
        cov  = np.array([[sigma*sigma]], float)   # shape (1,1)
        super().__init__(mean=mean, cov=cov, rng=rng)

    @property
    def mu(self) -> float:
        return float(self._mean[0])

    @property
    def sigma(self) -> float:
        return float(np.sqrt(self._cov[0,0]))

    @classmethod
    def from_distribution(cls,
                          convert_from: 'Distribution',
                          **fit_kwargs: Any) -> 'Normal':
        # delegate to MvNormal.from_distribution, then re-wrap
        mv = super().from_distribution(convert_from, **fit_kwargs)
        m = mv.mean()[0]
        s = np.sqrt(mv.cov()[0,0])
        return cls(mu=m, sigma=s, rng=mv._rng)

    def expectation(self, func):
        res = super().expectation(lambda x: func(x.reshape(-1,1)))
        # if we got back a multivariate of dim=1, convert to Normal
        if isinstance(res, MvNormal) and res.dimension == 1:
            m = res.mean()[0]
            s = np.sqrt(res.cov()[0,0])
            return Normal(mu=m, sigma=s, rng=self._rng)
        else:
            # either a true MvNormal(k>1) or a Normal1D from the parent
            return res