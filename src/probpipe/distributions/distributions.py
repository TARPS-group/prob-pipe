#This module implements Subclasses (not Mixins)

from typing import Generic, TypeVar, Callable, Any, Optional, Union, Tuple
from numpy.typing import NDArray
import numpy as np
import pymc as pm
from scipy.stats import norm, multivariate_normal
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scipy.stats import norm

from distributions.abstract_distributions import Distribution, Multivariate


T = TypeVar("T",bound=np.number)


# ----------------------------- Utilities -----------------------------

def _as_2d(x: NDArray) -> NDArray[np.floating]:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x

def _symmetrize_spd(cov: NDArray[np.floating], jitter: float = 1e-6) -> NDArray[np.floating]:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    # minimal jitter for numerical stability
    eigmin = np.linalg.eigvalsh(cov).min()
    if eigmin <= 0:
        cov = cov + (jitter - eigmin + 1e-12) * np.eye(cov.shape[0])
    else:
        cov = cov + jitter * np.eye(cov.shape[0])
    return cov

# ------------------- Empirical Distributions ------------------------

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
    

            
# ------------------- Distributions with Density ------------------------

#should inherit from multivariate
class Normal1D(Distribution[np.floating]):
    def __init__(self, mu: float, sigma: float, *, rng: np.random.Generator | None = None):
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self._rng = rng or np.random.default_rng()

    def sample(self, n_samples: int) -> NDArray[np.floating]:
        return self._rng.normal(loc=self.mu, scale=self.sigma, size=(n_samples,)).astype(float)

    #don't use the name data, instead rename it as values
    def density(self, data: NDArray) -> NDArray[np.floating]:
        x = np.asarray(data, dtype=float)
        z = (x - self.mu) / self.sigma
        return (np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * self.sigma)).astype(float)

    def log_density(self, data: NDArray) -> NDArray[np.floating]:
        x = np.asarray(data, dtype=float)
        z2 = ((x - self.mu) / self.sigma) ** 2
        return (-0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(self.sigma) + z2)).astype(float)

    def expectation(self, func: Callable[[NDArray[np.floating]], NDArray]) -> 'Distribution':
        # Monte-Carlo CLT: return Normal over mean of f(X)
        n_mc = 2048
        xs = self.sample(n_mc)  # shape (n,)
        ys = np.asarray(func(xs), dtype=float)
        if ys.ndim == 1:
            m = float(ys.mean())
            s = float(ys.std(ddof=1)) / np.sqrt(n_mc)
            s = max(s, 1e-12)  # guard against zero variance
            return Normal1D(m, s, rng=self._rng)
        else:
            # vector-valued f: return MVN over the mean
            ys2 = _as_2d(ys)  # (n, d)
            m = ys2.mean(axis=0)
            cov = np.cov(ys2, rowvar=False, ddof=1) / n_mc
            cov = _symmetrize_spd(cov)
            return MvNormal(mean=m, cov=cov, rng=self._rng)

    @classmethod
    def from_distribution(cls, convert_from: 'Distribution', **fit_kwargs: Any) -> 'Normal1D':
        # Fit mu, sigma from samples drawn from convert_from
        n = int(fit_kwargs.get("n", 2000))
        xs = np.asarray(convert_from.sample(n), dtype=float)
        if xs.ndim != 1:
            xs = xs.reshape(-1)
        mu = float(xs.mean())
        sigma = float(xs.std(ddof=1))
        sigma = max(sigma, 1e-9)
        return cls(mu, sigma)
    
def _symmetrize_spd(C: np.ndarray, jitter: float = 1e-9, eps: float = 1e-12) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    C = 0.5 * (C + C.T)
    eigmin = np.linalg.eigvalsh(C).min()
    # Add jitter only if needed
    if eigmin < eps:
        C = C + (max(jitter, eps - eigmin) + 1e-12) * np.eye(C.shape[0])
    return C
    

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
    
