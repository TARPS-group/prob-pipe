# distributions/distribution.py
from __future__ import annotations

from typing import Generic, TypeVar, Any, Sequence, Mapping
from collections.abc import Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import uuid

import numpy as np

from ..custom_types import Array, ArrayLike, PRNG, Float, Number
from ..array_backend.utils import _ensure_matrix, _ensure_vector
from ..linalg.linear_operator import LinOp, RootLinOp

__all__ = [
    "Distribution",
    "EmpiricalDistribution",
    "Provenance",
]

NumberT = TypeVar("NumberT", bound=Number)

# -------------------------- Provenance Tracking ----------------------------


@dataclass(frozen=True)
class Provenance:
    """
    Lightweight provenance tracking for probability distributions.

    Each distribution can carry structured metadata describing:
    - The operation that produced it (e.g., "prior", "transform", "approximate", "convert")
    - Parent distribution(s) it was derived from
    - Additional operation-specific details

    This transforms distributions from anonymous outputs into traceable probabilistic objects,
    forming a directed acyclic graph (DAG) of probabilistic operations.

    Attributes:
        op: The operation that created this distribution.
            Common values: "prior", "transform", "approximate", "convert", "condition", "marginalize"
        parents: Tuple of parent Provenance objects from which this was derived
        details: Additional operation-specific metadata (e.g., method, num_samples, parameters)
        uid: Unique identifier for this provenance node

    Example:
        >>> # Create a prior with provenance
        >>> prior = Gaussian(mean=0, cov=1)._with_source(op="prior", name="theta")
        >>> # Approximate it
        >>> emp = EmpiricalDistribution.from_distribution(prior, num_samples=1000)
        >>> # Check provenance
        >>> print(emp.source.op)  # "approximate"
        >>> print(emp.source.parents[0].op)  # "prior"
    """
    op: str
    parents: tuple[Provenance, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    uid: str = field(default_factory=lambda: uuid.uuid4().hex)

    def chain(self) -> Sequence[Provenance]:
        """
        Return a linearized view of the provenance graph for quick debugging.

        This performs a depth-first traversal of the provenance DAG, returning
        all ancestors in the order they are encountered. If the graph has multiple
        paths, the result may not be unique.

        Returns:
            List of Provenance objects from this node back to the roots
        """
        out = []
        stack = [self]
        seen = set()
        while stack:
            node = stack.pop()
            if node.uid in seen:
                continue
            seen.add(node.uid)
            out.append(node)
            stack.extend(node.parents)
        return out

    def tree_repr(self, indent: int = 0) -> str:
        """
        Return a tree representation of the provenance graph.

        Args:
            indent: Current indentation level (used for recursion)

        Returns:
            String representation of the provenance tree
        """
        prefix = "  " * indent
        details_str = ""
        if self.details:
            details_items = [f"{k}={v}" for k, v in self.details.items()]
            details_str = f" ({', '.join(details_items)})"

        result = f"{prefix}└─ {self.op}{details_str} [{self.uid[:8]}]\n"

        for parent in self.parents:
            result += parent.tree_repr(indent + 1)

        return result

    def __str__(self) -> str:
        """Concise string representation."""
        details_str = ""
        if self.details:
            details_items = [f"{k}={v}" for k, v in list(self.details.items())[:3]]
            details_str = f" ({', '.join(details_items)})"
        return f"Provenance(op='{self.op}'{details_str}, {len(self.parents)} parents)"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Provenance(op='{self.op}', parents={len(self.parents)}, details={dict(self.details)}, uid='{self.uid[:8]}...')"


# -------------------------- Abstract Classes ----------------------------


class Distribution(Generic[NumberT], ABC):
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

    Attributes:
        _source: Optional Provenance object tracking the origin of this distribution
    """

    def __init__(self, *, source: Provenance | None = None):
        """
        Initialize a distribution with optional provenance tracking.

        Args:
            source: Optional Provenance object describing how this distribution was created
        """
        self._source = source

    @property
    def source(self) -> Provenance | None:
        """
        Get the provenance metadata for this distribution.

        Returns:
            Provenance object if tracking is enabled, None otherwise
        """
        return self._source

    def _with_source(
        self,
        op: str,
        *,
        parents: list[Distribution] | None = None,
        **details
    ) -> Distribution:
        """
        Attach provenance metadata to this distribution.

        This method allows fluent chaining to add provenance information
        after construction. It modifies the distribution in-place and returns self.

        Args:
            op: Operation name (e.g., "prior", "transform", "approximate", "convert")
            parents: List of parent distributions (their provenance will be extracted)
            **details: Additional operation-specific metadata

        Returns:
            Self (for method chaining)

        Example:
            >>> dist = Gaussian(mean=0, cov=1)._with_source(
            ...     op="prior",
            ...     name="theta",
            ...     distribution_type="normal"
            ... )
        """
        parent_list = parents or []
        parent_provenances = tuple(
            d.source for d in parent_list if d.source is not None
        )
        self._source = Provenance(op=op, parents=parent_provenances, details=details)
        return self

    def sample(self, n_samples: int = 1) -> Array[NumberT]:
        """
        Samples data points from the distribution.

        This method may be optionally implemented by subclasses that
        support random sampling. It returns `n_samples` draws from
        the distribution.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            An array containing `n_samples` draws from the distribution.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    
    def density(self, x: Array[NumberT]) -> Array[Float]:
        """
        Computes the probability density p(data) under this distribution.

        This method may be optionally implemented by subclasses that
        can evaluate densities. The returned array corresponds to the
        pointwise probability density values reduced over event
        dimensions (i.e., matching the batch shape).

        Args:
            data: Input array of observations for which to compute densities.

        Returns:
            Probability density values for each input point.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    
    def log_density(self, x: Array[NumberT]) -> Array[Float]:
        """
        Computes the log-probability density log p(data).

        This method may be optionally implemented by subclasses that
        can evaluate log-densities. The returned array corresponds to
        the pointwise log-probability values reduced over event
        dimensions (i.e., matching the batch shape).

        Args:
            data: Input array of observations for which to compute log-densities.

        Returns:
            Log-probability values for each input point.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    

    def expectation(self, func: Callable[[Array[NumberT]], Array]) -> Distribution:
        """
        Computes the expectation of a function under this distribution.

        This method may be optionally implemented by subclasses that can
        perform Monte Carlo estimation. The result is typically a new
        `Distribution` instance representing the mean of the function
        applied to random samples drawn from this distribution.

        Args:
            func: A callable function f(x) to compute the expectation of.

        Returns:
            A distribution representing E[f(X)].

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """

        raise NotImplementedError("This method may be implemented by subclasses (optional)")
    

    @classmethod
    @abstractmethod
    def from_distribution(cls, other: Distribution, **fit_kwargs: Any) -> Distribution[NumberT]:
        """
        Convert the distribution `other` into a distribution of type `cls`. This will 
        typically be an approximation. Examples including moment matching and 
        kernel density estimation (KDE).
        """
        raise NotImplementedError("This method should be implemented by subclasses")


class EmpiricalDistribution(Distribution):
    """ Container for (weighted) empirical samples in R^d.

    The discrete distribution defined by a set of `n`, potentially weighed,
    samples.

    Args:
        x: array-like, shape (n, d) or (n,)
            The samples defining the empirical distribution.
        weights: array-like, shape (n,), optional
            Nonnegative weights; will be normalized to sum to 1. If None,
            uniform weights are assigned.
        rng: np.random.Generator, optional
            Random number generator for sampling.
        source: Provenance, optional
            Provenance metadata tracking the origin of this distribution.
    """

    def __init__(
        self,
        x: Array,
        weights: Array | None = None,
        *,
        rng: PRNG | None = None,
        source: Provenance | None = None,
    ):
        super().__init__(source=source)
        X = _ensure_matrix(x, as_row_matrix=True)
        n, d = X.shape
        if n < 1:
            raise ValueError("EmpiricalDistribution requires at least one sample.")

        if weights is None:
            w = np.full(n, 1.0 / n, dtype=X.dtype)
        else:
            w = _ensure_vector(weights, as_column=False, length=n)
            if np.any(w < 0):
                raise ValueError("weights must be nonnegative.")
            s = w.sum()
            if s <= 0:
                raise ValueError("weights cannot all be zero.")
            w = w / s

        self._X = X.astype(float)
        self._w = w.astype(float)
        self._n = int(n)
        self._d = int(d)
        self._rng = rng or np.random.default_rng()

        # Compute empirical mean and covariance.
        self._mean = (self._w[:, np.newaxis] * self._X).sum(axis=0)
        cov_root = ((self._X - self._mean) * np.sqrt(self._w)[:, np.newaxis]).T  # (d,n)
        self._cov = RootLinOp(cov_root)

        # cumulative weights for fast inverse-transform resampling
        self._cw = np.cumsum(self._w)

    @property
    def n(self) -> int:
        """Number of stored samples."""
        return self._n

    @property
    def dim(self) -> int:
        """Dimensionality."""
        return self._d

    @property
    def samples(self) -> Array:
        """A view of the stored samples, shape (n, d)."""
        return self._X

    @property
    def weights(self) -> Array:
        """A view of normalized weights, shape (n,)."""
        return self._w

    def mean(self) -> Array:
        """Weighted mean, shape (d,)."""
        return self._mean

    def cov(self) -> LinOp:
        """Weighted *population* covariance, shape (d, d)."""
        return self._cov

    def var(self) -> Array:
        """Weighted population variance per dimension, shape (d,)."""
        return self._cov.diag()

    def std(self) -> Array:
        """Weighted population standard deviation per dimension, shape (d,)."""
        return np.sqrt(np.maximum(self.var(), 0.0))

    def __str__(self) -> str:
        """String representation of the empirical distribution."""
        # Check if weights are uniform
        uniform_weights = np.allclose(self._w, 1.0 / self._n)
        weight_info = "uniform" if uniform_weights else "weighted"

        # Format mean and std for display
        if self._d == 1:
            mean_str = f"{self._mean[0]:.4g}"
            std_str = f"{self.std()[0]:.4g}"
        elif self._d <= 3:
            mean_str = "[" + ", ".join(f"{m:.4g}" for m in self._mean) + "]"
            std_str = "[" + ", ".join(f"{s:.4g}" for s in self.std()) + "]"
        else:
            mean_str = f"[{self._mean[0]:.4g}, ..., {self._mean[-1]:.4g}]"
            std_str = f"[{self.std()[0]:.4g}, ..., {self.std()[-1]:.4g}]"

        return (
            f"EmpiricalDistribution(n={self._n}, dim={self._d}, {weight_info}, "
            f"mean={mean_str}, std={std_str})"
        )

    def sample(self, n_samples: int = 1, *, replace: bool = True) -> Array:
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

    def density(self, x: Array) -> Array:
        """Approximate density using a Gaussian fit to the empirical samples. See log_density."""
        return np.exp(self.log_density(x))

    def log_density(self, x: Array) -> Array:
        """
        Approximate log density using a Gaussian fit to the empirical samples.
        """
        from .real_vector.gaussian import Gaussian
        return Gaussian(mean=self._mean, cov=self._cov.to_dense()).log_density(x)

    # TODO: come back to this:
    def expectation(
        self,
        func: Callable[[Array], Array],
        *,
        n_mc: int = 2048,
    ):
        raise NotImplemented

        """
        Y = np.asarray(func(self._X), dtype=float)

        if Y.ndim == 1:
            m = float((self._w * Y).sum())
            var = float((self._w * (Y - m) ** 2).sum())
            se = np.sqrt(max(var, 0.0)) / np.sqrt(n_mc)
            return Gaussian(m, max(se, 1e-12), rng=self._rng)
        else:
            Y = _ensure_matrix(Y, as_row_matrix=True)
            m = (self._w[:, None] * Y).sum(axis=0)
            diff = Y - m
            cov = diff.T @ (diff * self._w[:, None])
            cov_mean = cov / float(n_mc)
            cov_mean = 0.5 * (cov_mean + cov_mean.T) + 1e-12 * np.eye(cov_mean.shape[0])
            return Gaussian(mean=m, cov=cov_mean, rng=self._rng)
        """

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        **fit_kwargs: Any,
    ) -> EmpiricalDistribution:
        """
        Create an empirical distribution by sampling from another distribution.

        This is an approximation operation that converts any distribution into
        a sample-based representation.

        Args:
            other: The distribution to approximate
            **fit_kwargs: Additional keyword arguments
                - num_samples: Number of samples to draw (default: 2048)

        Returns:
            EmpiricalDistribution with provenance tracking the approximation
        """
        num_samples = fit_kwargs.get("num_samples", 2048)
        samples = other.sample(num_samples)

        # Create provenance metadata
        src = Provenance(
            op="approximate",
            parents=tuple([other.source] if other.source is not None else []),
            details={
                "method": "empirical",
                "num_samples": num_samples,
                "target_class": cls.__name__,
            },
        )

        return cls(samples, source=src)
        


class BootstrapDistribution(EmpiricalDistribution):
    """
    Empirical distribution over bootstrap replicates of a statistic.

    Semantics:
      - 'samples' (inherited) are the bootstrap replicates theta* (shape (B, k)).
      - 'replicates' is an alias to 'samples' for user-facing clarity.
      - All summaries (mean/cov/var/std), resampling (sample/rvs), and
        expectation() behavior are inherited from EmpiricalDistribution.

    Notes
    -----
    This class intentionally reuses EmpiricalDistribution's implementation to avoid
    duplication. The only additions are naming (replicates alias) and a convenience
    constructor 'from_data' to generate bootstrap replicates.
    """

    def __init__(
        self,
        replicates: Array,
        weights: Array | None = None,
        *,
        rng: PRNG | None = None,
        source: Provenance | None = None,
    ):
        # Just forward to EmpiricalDistribution
        super().__init__(replicates, weights, rng=rng, source=source)

    # --------- Aliases for semantics ---------

    @property
    def replicates(self) -> Array:
        """Alias for the stored bootstrap replicates, shape (B, k)."""
        return self.samples

    # --------- Convenience constructor ---------

    @classmethod
    def from_data(
        cls,
        data: Array,
        stat_fn: Callable[[Array], Array],
        *,
        B: int = 1000,
        axis: int = 0,
        rng: PRNG | None = None,
    ) -> BootstrapDistribution:
        """
        Classic i.i.d. bootstrap for a statistic.

        Parameters
        ----------
        data : array-like
            Observations (samples along `axis`).
        stat_fn : callable
            Function mapping a resampled dataset (with samples on axis 0) to a
            statistic vector (shape (k,) or scalar). We will pass the resampled
            array with **samples on axis 0**.
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
            idx = rng.integers(0, n, size=n)  # sample n rows with replacement
            Xb = X[idx]
            theta = np.asarray(stat_fn(Xb), dtype=float).reshape(-1)
            reps.append(theta)

        Theta = np.vstack(reps)  # (B, k)
        return cls(Theta, rng=rng)
