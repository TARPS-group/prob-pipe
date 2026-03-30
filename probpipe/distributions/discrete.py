"""
Discrete distributions backed by TFP.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ._tfp_base import TFPDistribution
from ..core.distribution import (
    ArrayDistribution,
    EmpiricalDistribution,
    _auto_key,
)
from ..core.provenance import Provenance
from ..core.constraints import (
    Constraint,
    boolean,
    non_negative_integer,
    integer_interval,
)
from ..custom_types import Array, ArrayLike, PRNGKey

__all__ = [
    "Bernoulli",
    "Binomial",
    "Poisson",
    "Categorical",
    "NegativeBinomial",
]


class Bernoulli(TFPDistribution):
    """
    Bernoulli distribution.

    Parameters
    ----------
    probs : array-like, optional
        Probability of a 1 outcome.  Exactly one of *probs* or *logits*
        must be provided.
    logits : array-like, optional
        Log-odds of a 1 outcome.
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        *,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        name: str | None = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be provided.")
        if probs is not None:
            self._probs = jnp.asarray(probs, dtype=jnp.float32)
            self._logits = None
            self._tfp_dist = tfd.Bernoulli(probs=self._probs)
        else:
            self._logits = jnp.asarray(logits, dtype=jnp.float32)
            self._probs = None
            self._tfp_dist = tfd.Bernoulli(logits=self._logits)
        self._name = name

    # -- convenient accessors -----------------------------------------------

    @property
    def probs(self) -> Array | None:
        return self._probs

    @property
    def logits(self) -> Array | None:
        return self._logits

    # -- support ------------------------------------------------------------

    @property
    def support(self) -> Constraint:
        return boolean

    @classmethod
    def _default_support(cls) -> Constraint:
        return boolean

    # -- expectation (exact over {0, 1}) ------------------------------------

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Array:
        """Exact expectation over the two-point support {0, 1}."""
        p = self._tfp_dist.probs_parameter()
        f0 = f(jnp.zeros(self.event_shape, dtype=self.dtype))
        f1 = f(jnp.ones(self.event_shape, dtype=self.dtype))
        return (1 - p) * f0 + p * f1



class Binomial(TFPDistribution):
    """
    Binomial distribution.

    Parameters
    ----------
    total_count : array-like
        Number of trials.
    probs : array-like, optional
        Probability of success per trial.  Exactly one of *probs* or
        *logits* must be provided.
    logits : array-like, optional
        Log-odds of success per trial.
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        total_count: ArrayLike,
        *,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        name: str | None = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be provided.")
        self._total_count = jnp.asarray(total_count, dtype=jnp.float32)
        if probs is not None:
            self._probs = jnp.asarray(probs, dtype=jnp.float32)
            self._logits = None
            self._tfp_dist = tfd.Binomial(
                total_count=self._total_count, probs=self._probs
            )
        else:
            self._logits = jnp.asarray(logits, dtype=jnp.float32)
            self._probs = None
            self._tfp_dist = tfd.Binomial(
                total_count=self._total_count, logits=self._logits
            )
        self._name = name

    # -- convenient accessors -----------------------------------------------

    @property
    def total_count(self) -> Array:
        return self._total_count

    @property
    def probs(self) -> Array | None:
        return self._probs

    @property
    def logits(self) -> Array | None:
        return self._logits

    # -- support ------------------------------------------------------------

    @property
    def support(self) -> Constraint:
        return integer_interval(0, int(self._total_count))

    @classmethod
    def _default_support(cls) -> Constraint:
        return non_negative_integer

    # -- expectation (exact over {0, ..., total_count}) ---------------------

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Array:
        """Exact expectation over the finite support {0, ..., total_count}."""
        n = int(self._total_count) + 1
        support = jnp.arange(n, dtype=self.dtype)
        probs = jnp.exp(self._tfp_dist.log_prob(support))
        f_vals = jax.vmap(f)(support)
        return jnp.einsum("n,n...->...", probs, f_vals)



class Poisson(TFPDistribution):
    """
    Poisson distribution.

    Parameters
    ----------
    rate : array-like
        Rate parameter (must be positive).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        rate: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._rate = jnp.asarray(rate, dtype=jnp.float32)
        self._tfp_dist = tfd.Poisson(rate=self._rate)
        self._name = name

    # -- convenient accessors -----------------------------------------------

    @property
    def rate(self) -> Array:
        return self._rate

    # -- support ------------------------------------------------------------

    @property
    def support(self) -> Constraint:
        return non_negative_integer

    @classmethod
    def _default_support(cls) -> Constraint:
        return non_negative_integer



class Categorical(TFPDistribution):
    """
    Categorical distribution over ``k`` classes.

    Parameters
    ----------
    probs : array-like, optional
        Probabilities for each category.  Exactly one of *probs* or
        *logits* must be provided.
    logits : array-like, optional
        Unnormalized log-probabilities for each category.
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        *,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        name: str | None = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be provided.")
        if probs is not None:
            self._probs = jnp.asarray(probs, dtype=jnp.float32)
            self._logits = None
            self._tfp_dist = tfd.Categorical(probs=self._probs)
        else:
            self._logits = jnp.asarray(logits, dtype=jnp.float32)
            self._probs = None
            self._tfp_dist = tfd.Categorical(logits=self._logits)
        self._name = name

    # -- convenient accessors -----------------------------------------------

    @property
    def probs(self) -> Array | None:
        return self._probs

    @property
    def logits(self) -> Array | None:
        return self._logits

    # -- support ------------------------------------------------------------

    @property
    def support(self) -> Constraint:
        return integer_interval(0, int(self._tfp_dist.num_categories) - 1)

    @classmethod
    def _default_support(cls) -> Constraint:
        return non_negative_integer

    # -- expectation (exact over {0, ..., k-1}) ------------------------------

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Array:
        """Exact expectation over the categorical support {0, ..., k-1}."""
        probs = self._tfp_dist.probs_parameter()
        k = probs.shape[-1]
        support = jnp.arange(k, dtype=self.dtype)
        f_vals = jax.vmap(f)(support)
        return jnp.einsum("n,n...->...", probs, f_vals)



class NegativeBinomial(TFPDistribution):
    """
    Negative binomial distribution.

    Parameters
    ----------
    total_count : array-like
        Number of successes before stopping.
    probs : array-like, optional
        Probability of success per trial.  Exactly one of *probs* or
        *logits* must be provided.
    logits : array-like, optional
        Log-odds of success per trial.
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        total_count: ArrayLike,
        *,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        name: str | None = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be provided.")
        self._total_count = jnp.asarray(total_count, dtype=jnp.float32)
        if probs is not None:
            self._probs = jnp.asarray(probs, dtype=jnp.float32)
            self._logits = None
            self._tfp_dist = tfd.NegativeBinomial(
                total_count=self._total_count, probs=self._probs
            )
        else:
            self._logits = jnp.asarray(logits, dtype=jnp.float32)
            self._probs = None
            self._tfp_dist = tfd.NegativeBinomial(
                total_count=self._total_count, logits=self._logits
            )
        self._name = name

    # -- convenient accessors -----------------------------------------------

    @property
    def total_count(self) -> Array:
        return self._total_count

    @property
    def probs(self) -> Array | None:
        return self._probs

    @property
    def logits(self) -> Array | None:
        return self._logits

    # -- support ------------------------------------------------------------

    @property
    def support(self) -> Constraint:
        return non_negative_integer

    @classmethod
    def _default_support(cls) -> Constraint:
        return non_negative_integer

